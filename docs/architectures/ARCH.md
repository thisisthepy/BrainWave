# 남은 꼬리의 모양 — 32 개 아키텍처를 재고, 세 op 을 넣고, GEMM 이 조용히 틀렸던 것을 찾은 기록

`docs/models/GPT2.md` §6 이 여섯 개를 재고 "꼬리는 아키텍처마다 다르다"고 적은 것을 이어받아,
**32 개 아키텍처를 실측**하고 `gelu` · `gather` · `zero_` 를 구현한 작업의 기록입니다.

**한 줄 결론.** 세 개를 넣자 **미구현 0 인 아키텍처가 8 개에서 17 개로** 늘었고, aten 레벨로
조립한 2 층 Gemma 와 2 층 BERT 가 상류와 같은 답을 냅니다. 그리고 그 과정에서 골든이
**한 번도 큰 행렬을 곱해보지 않았기 때문에 놓치고 있던 진짜 수치 오류**를 찾았습니다 —
**`float16` GEMM 을 상류는 `float32` 로 누적하는데 이 셰임은 `float16` 으로 누적하고 있었습니다.**

기준선 대비:

| | 이전 | 이후 |
|---|---|---|
| 골든 케이스 | 1781 | **1934** (실패 0, pending 0, exit 0) |
| `_aten_implemented()` | 82 | **85** |
| `_aten_all_implemented()` | 94 | **97** |
| 미구현 0 인 아키텍처 | 8 / 32 | **17 / 32** |
| 스키마 | 170/170 | 170/170 (변화 없음) |
| 스모크 | 65 ok | 65 ok |
| 3 타깃 | exit 0 | exit 0 |

---

## 0. 먼저 다시 쟀다 — 그리고 6 개가 아니라 32 개를 쟀다

GPT2.md §6 의 측정은 `_aten_implemented()` 가 82 개이던 시점의 것입니다. 그 목록을 믿지 않고
**82 개 시점에서 다시** 쟀고, 동시에 표본을 6 → 32 로 넓혔습니다. 꼬리의 *모양*을 알려면
표본이 여섯 개로는 부족하기 때문입니다.

- 방법: 상류 torch 2.13.0 (`/Volumes/macMini/caches/spike-venv/bin/python`, transformers 5.15.1)
  위에서 `TorchDispatchMode` 로 기록. 디코더는 `generate` 의 greedy ∪ do_sample 합집합,
  인코더는 순전파, seq2seq 는 greedy.
- 구현 목록은 **빌드한 산출물에서 직접** 읽습니다 (`_C._aten_all_implemented()`).
- 규모는 전부 2 층 · hidden 64 · heads 2 · intermediate 128 급.

### 82 개 시점의 결과 — 조율 세션의 예고와 정확히 일치

```
gemma  미구현 1   aten.gelu.default
bert   미구현 2   aten.gelu.default, aten.gather.default
llama · gpt2 · qwen2 · mistral   미구현 0
```

여기까지는 예고대로입니다. **넓힌 26 개가 준 정보가 이 절의 값어치입니다.**

### 32 개 전체 — `gelu` 하나가 14 개를 막고 있었다

빌드에 실패한 넷(`modernbert` · `phi3` · `glm` · `whisper`, 전부 `Padding_idx must be within
num_embeddings` — 이 작은 vocab 설정 탓이고 셰임과 무관)과 `t5`(`decoder_start_token_id` 미설정)를
빼고 **32 개**를 쟀습니다.

**미구현 op 을 "몇 개의 아키텍처를 여는가" 로 정렬한 것이 이 작업의 핵심 산출물입니다:**

```
aten.gelu.default              14   gemma gemma2 bert roberta electra distilbert
                                    deberta_v2 bart falcon gpt_neox gpt_bigcode
                                    starcoder2 mpt vit
aten.gather.default             5   bert roberta electra albert gptj
aten.permute.default            5   deberta_v2 falcon gptj bloom mpt
aten.le.Tensor                  4   falcon gptj bloom mpt
aten.scalar_tensor.default      4   falcon gptj bloom mpt
aten.where.self                 4   falcon gptj bloom mpt
aten.stack.default              4   gptj cohere helium mamba
aten.relu.default               3   opt nemotron persimmon
aten.index_put_.default         2   mixtral bart
aten.split_with_sizes.default   2   gpt_bigcode mamba
aten.convolution.default        2   mamba vit
(나머지 17 개는 전부 1)
```

**`gelu` 는 두 번째로 흔한 것의 세 배입니다.** GPT2.md 가 "Gemma 하나가 열린다"고 본 것은
표본이 여섯 개였기 때문이고, 32 개로 넓히자 그것은 **꼬리에서 압도적으로 가장 큰 항목**이었습니다.
`docs/kernels/GAP.md` 의 "꼬리는 고정 목록이 아니다"는 여전히 맞지만, **꼬리 안에 등급이 있다**는 것이
이번에 추가로 보입니다.

두 번째로 눈에 띄는 것은 **`falcon` · `gptj` · `bloom` · `mpt` 가 정확히 같은 네 개
(`le.Tensor` · `scalar_tensor` · `where.self` · `permute`)를 부른다**는 것입니다. 넷 다 구형
아키텍처이고, 넷 다 마스크를 `torch.where` 로 만드는 옛 관용구를 씁니다. **아키텍처 4 개를
한 묶음으로 여는 4-op 세트가 다음 작업의 가장 큰 덩어리입니다.**

### 세 번째 op 은 측정에 안 나온다 — 그리고 그것이 요점이다

`aten.zero_.default` 는 위 표 어디에도 없습니다. **순전파에서는 한 번도 안 불리기
때문입니다.** 생성자와 순전파를 따로 기록해서 확인했습니다:

```
nn.LayerNorm(8)  생성자 : empty.memory_format x2, fill_.Scalar x1, zero_.default x1
nn.LayerNorm(8)  순전파 : (없음)
nn.Linear(4,4)   생성자 : empty.memory_format x2, uniform_ x2   ← zero_ 안 부름
```

`fill_` 이 weight 를 1 로, `zero_` 가 bias 를 0 으로 놓습니다. 즉 `zero_` 는 **꼬리에 있는 것이
아니라 꼬리에 도달하기 전 길목에 있습니다** — 만들어지지 않는 모델은 op 을 세어볼 기회조차 없고,
그래서 `docs/models/GPT2.md` 가 `nn.LayerNorm` 이 두 겹으로 막힌다고 본 것입니다. `_C._get_cudnn_enabled`
를 답해줘도 그다음 `reset_parameters` 의 `TensorBase.zero_` 에서 죽습니다.

**측정 도구가 못 보는 자리가 있다는 것을 이 op 이 보여줍니다.** `TorchDispatchMode` 로 추론을
기록하는 방법은 정의상 "추론이 시작된 뒤"만 봅니다.

---

## 1. `gelu` — 이름 하나에 함수가 둘이고, 고르는 것이 반올림 문제가 아니다

`aten::gelu(Tensor self, *, str approximate="none") -> Tensor`

두 근사식이 얼마나 다른지부터 쟀습니다. `float32`, `[-3, 3]`:

```
x        [-3.0,   -1.0,     -0.5,     0.0,  0.5,     1.0,     3.0  ]
none     [-0.00405, -0.158655, -0.154269, 0.0, 0.345731, 0.841345, 2.995950]
tanh     [-0.00364, -0.158808, -0.154286, 0.0, 0.345714, 0.841192, 2.996363]
최대 |차|  4.1246e-04
```

**골든의 `float32` 허용오차(`1e-5`)의 41 배입니다.** 1 ulp 수준이 아니라, 잘못 고르면
"오차가 좀 크다"가 아니라 **다른 함수를 계산한 것**입니다.

### 누가 어느 쪽을 부르는가 — 추측하지 않고 kwargs 를 기록했다

```
gemma      approximate='tanh'   x4     ← 유일한 tanh 사용자 (gelu_pytorch_tanh)
gemma2     approximate='tanh'
bert       (기본값)              x2
gpt_neox   (기본값)              x4
bart       (기본값)              x4
falcon     (기본값)              x4
vit        (기본값)              x2
gpt2       gelu 호출 자체가 없음
```

두 가지가 나옵니다.

**(1) Gemma 만 tanh 다.** 나머지 13 개는 전부 기본값(=`none`, erf 형)입니다. 스키마의 기본값도
기억이 아니라 `torch.ops.aten.gelu.default._schema` 를 다시 읽어 확인했습니다.

**(2) GPT-2 는 tanh gelu 모델인데 이 op 을 안 부른다.** HF 가 `gelu_new` 를 파이썬으로 쓰기
때문에 `aten.tanh.default` 로 내려갑니다. 즉 **같은 수식이 두 철자로 존재하고 둘 다 맞아야
합니다.** 이것이 아래 구현에서 tanh 가지를 candle 에 위임하지 않은 이유입니다.

### 구현 — candle 의 `gelu`/`gelu_erf` 를 그대로 쓰지 않았고, 이유는 측정값이다

candle 에는 `Tensor::gelu`(tanh 근사)와 `Tensor::gelu_erf`(정확형)가 둘 다 있습니다. 그런데
**tanh 쪽의 결합 순서가 상류와 다릅니다:**

```
candle :  β · v · (1 + κ·v·v)
ATen   :  β · (v + κ·v³)
```

대수적으로 같고 `float32` 에서 같지 않습니다 — `[-3, 3]` 에서 **2.98e-08** 벌어집니다.
그래서 tanh 가지는 ATen 의 결합 순서 그대로 candle 연산으로 조립했고, 그 결과
**`float32`·`float64` 에서 상류와 비트가 같아졌습니다** (아래 표).

정확형(`none`)은 `gelu_erf` 에 위임합니다. candle 의 `erf` 는 `libm::erff` 이고 상류의 자체
커널과 다르므로 **비트까지 맞출 수 없습니다** — 조립 순서를 바꿔도 닫히지 않습니다. 실측
최대 절대차는 `float32` 에서 **1.79e-07** 이고, 어디서 나오는지는 §7 에 적었습니다.

`float16`/`bfloat16` 은 `float32` 로 올려 계산하고 한 번만 내립니다. **추측이 아니라 검증했습니다:**
`gelu(half x)` 는 `half(gelu(float(x)))` 와 **비트가 같습니다** — 두 근사식 모두, 모든 탐침 점에서.
(상류의 `at::opmath_type<Half> == float`.) 같은 규칙을 이 파일의 `silu_default` 가 이미 따르고
있었습니다.

**구현 후 상류와의 실측 차이:**

| dtype | `approximate` 미지정 / `'none'` | `'tanh'` |
|---|---|---|
| `float64` | 2.78e-17 | **비트 일치** |
| `float32` | 1.79e-07 | **비트 일치** |
| `float16` | **비트 일치** | **비트 일치** |
| `bfloat16` | **비트 일치** | **비트 일치** |

### 거부도 상류를 그대로 옮겼다

- 정수·불리언: `NotImplementedError: "GeluKernelImpl" not implemented for 'Long'`.
  이것은 `silu` 쪽 규칙이지 `tanh` 쪽이 아닙니다 — 승격하는 unary 헬퍼를 재사용했다면
  **상류가 거부하는 자리에서 계산했을 것**입니다.
- `approximate` 가 `'none'`/`'tanh'` 가 아니면 `RuntimeError: approximate argument must be
  either none or tanh.` 모르는 문자열에서 `'none'` 으로 폴백하면 **Gemma 의 오타를 BERT 의
  활성함수로 답하게 됩니다.** 문자열 검증은 산술보다 먼저 하므로 빈 텐서에서도 거부합니다.
- `approximate` 는 **키워드 전용**입니다(스키마의 `*`). `gelu(x, "tanh")` 는 상류에서 에러이고,
  받아들이면 "셰임에서는 tanh, 상류에서는 에러"라는 가장 시끄러운 방향의 불일치가 됩니다.
- `-inf` 는 `nan` 입니다(양쪽 근사식 모두). `0 · (-inf)` 의 부정형이고, "클램프 후 스케일"로
  쓴 커널이라면 `-0.0` 을 주면서 다른 모든 케이스를 통과합니다.

---

## 2. `gather` — `scatter.src` 를 거꾸로 읽은 것, 그런데 candle 것을 못 쓴다

`aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor`

candle 에 `Tensor::gather` 가 있는데 쓰지 않았습니다. **세 군데에서 torch 와 다르고, 셋 다
조용한 불일치가 됩니다:**

| | candle | torch |
|---|---|---|
| index dtype | `u8` · `u32` · `i64` | **`int32` · `int64` 만** |
| 음수 index | `as_usize()` 로 거대한 수가 되어 그 수를 에러에 찍음 | `index -1 is out of bounds for dimension 1 with size 3` |
| 비연속 입력 | `RequiresContiguous` 로 거부 | 그냥 계산 (실측: `arange(12).reshape(3,4).t()`) |

첫 줄이 가장 위험합니다. 이 셰임에서 `uint8` 텐서는 **마스크**인데, candle 은 그것을 위치로
읽습니다. `masked_fill` 이 `uint8` 마스크를 거부하는 것과 같은 이유로 여기서도 거부해야 합니다.

`gather` 에는 **음수 인덱스 관용이 없습니다** — `select`/`slice` 와 다릅니다(실측).
`index.Tensor` 의 처리와 맞추라는 지시가 있었는데, 실제로 재보니 `index.Tensor` 와도 다르고
`scatter.src` 의 규칙과 같았습니다: 범위를 벗어나면 그냥 거부합니다.

### 랭크 규칙 — "같아야 한다"가 아니다

상류는 양쪽에 `ensure_nonempty_dim`(= `max(rank, 1)`)을 적용합니다. 그래서:

```
0-d self + 1-d index  ->  허용, 결과 shape 은 index 의 것        gather(tensor(7.), 0, [0,0]) = [7., 7.]
1-d self + 0-d index  ->  허용, 결과 0-d                        gather([1.,2.], 0, tensor(1)) = 2.
0-d self + 2-d index  ->  거부
```

"랭크가 같아야 한다"고 짐작했다면 **상류가 답하는 호출 두 개를 거부**했을 것입니다.

출력 shape 은 `self` 가 아니라 **`index` 의 것**입니다. 축 밖에서는 index 가 `self` 보다
*작아도* 되고(남는 행은 안 읽힘), 축 위에서는 *길어도* 됩니다(값이 반복). 둘 다 실측.

`sparse_grad` 는 받고 무시합니다 — autograd 표현을 고르는 인자이고 여기엔 autograd 가 없습니다.
상류의 순전파 답도 값과 무관하게 같습니다(실측).

### BERT 는 이걸 어디서 부르는가

호출 지점을 스택으로 특정했습니다 — `modeling_bert.py:93`:

```python
buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)   # (1, 64)
buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)  # (1, 6)
```

전부 0 인 버퍼를 position_ids 로 gather 합니다. 즉 **축 위에서 index 가 self 보다 짧은** 모양이고,
위 골든 케이스가 그 모양을 직접 다룹니다.

---

## 3. `zero_` — `fill_(0)` 의 철자가 아니다

`aten::zero_(Tensor(a!) self) -> Tensor(a!)`

값은 같지만 상류에서 **별개의 오버로드**이고, 오버로드는 이 셰임의 키의 일부입니다
(`docs/design/TORCH_C.md` §1). `fill_inplace` 에 합치면 `_aten_implemented()` 가 둘을 요구받고 하나를
구현했다고 말하게 됩니다.

`fill_` 과 달리 `checked_convert` 가 없습니다 — **0 은 모든 dtype 에서 정확히 표현되므로
넘칠 값이 없습니다.** `bool` 은 `False` 가 되고 `nan`/`inf` 원소도 다른 것과 똑같이 덮어써집니다
(둘 다 실측, 양쪽 일치).

**파이썬 철자는 아직 없습니다.** `TensorBase.zero_` 를 `methods.json` 에 넣는 것은 이 작업의
범위 밖(다른 에이전트 소유)이라 한 줄도 안 고쳤습니다. 커널은 있고 `torch.ops.aten.zero_.default`
로 도달하지만, `nn.LayerNorm` 의 `reset_parameters` 가 실제로 이 커널에 닿으려면 그 표가
필요합니다. **다음 작업 항목입니다.**

---

## 4. 골든이 한 번도 큰 행렬을 안 곱해봤다 — 그리고 거기에 진짜 버그가 있었다

`docs/models/GPT2.md` §7 이 남긴 항목: *"큰 층의 오차는 §3.3 이 512×512 까지만 쟀다. 골든은 이 크기를
안 다룬다."* 이번에 다뤘습니다. **찾은 것은 예상과 달랐고, 예상보다 나빴습니다.**

### 4.1 `float32` 는 예상보다 훨씬 멀쩡했다

k 를 쓸어보며 상류와 대조했습니다 (LCG 백색잡음, 양쪽에 같은 리스트):

| shape | max abs | 실패 원소 / 전체 |
|---|---|---|
| (4,512)x(512,4) | **0** (비트 일치) | 0 / 16 |
| (64,512)x(512,64) | **0** | 0 / 4096 |
| (128,512)x(512,128) | **0** | 0 / 16384 |
| (256,256)x(256,256) | **0** | 0 / 65536 |
| (64,1024)x(1024,64) | 3.43e-05 | 4 / 4096 |
| (64,2048)x(2048,64) | 8.77e-05 | 52 / 4096 |
| addmm (512,512)x(512,512) | 2.10e-05 | 3 / 262144 |

**즉 §7 이 지목한 `k=512` 에서는 오차가 0 입니다** — 이 데이터에서 상류와 비트가 같습니다.
갈라지기 시작하는 것은 **k ≥ 1024** 이고, 그마저도 4096 개 중 4 개입니다. GPT2.md §3.3 의
`1.5e-05` 는 `nn.Linear` 의 기본 초기화 분포에서 나온 것이고, 이 실험의 분포에서는 (512,512,512)
addmm 에서 26 만 개 중 3 개가 걸립니다. §3.3 의 "정규화하면 3.5e-07, 평범한 float32 GEMM
반올림"이라는 판단은 맞았습니다.

### 4.2 진짜 문제는 `float16` 이었다 — 그리고 그건 반올림이 아니었다

같은 스윕을 `float16` 으로 돌리자 그림이 완전히 달랐습니다:

```
k       max|abs|    골든 허용오차(5e-3) 밖 원소 / 64
4       0.00049     0
8       0.00195     0
16      0.00391     0
32      0.00391     0
64      0.00586     1      ← 여기서 이미 갈라진다
128     0.01953     4
256     0.07031     9
512     0.07812     15
1024    0.09375     15
```

그리고 원인이 무엇인지 상류에 물어봤습니다:

```
mm(half a, half b)  ==  half(mm(float(a), float(b)))   ->  k=4,64,512 전부 True, max|d| = 0
bmm / addmm 도 동일
bfloat16 도 동일
```

**상류의 CPU GEMM 은 저장 dtype 과 무관하게 `float32` 로 누적합니다** (`at::opmath_type`).
candle 에는 그런 개념이 없어서 `float16` 텐서를 그대로 넘기면 `float16` 으로 누적합니다.
**반올림 순서 차이가 아니라 다른 함수입니다.** 그리고 깊이에 따라 자라므로 실제 모델은
층마다 어긋납니다.

**이것이 안 보였던 이유가 §7 의 지적 그대로입니다:** 골든의 GEMM 케이스가 전부 `float16`
누적으로도 무손실인 크기(k ≤ 5)였습니다.

### 4.3 고쳤다 — 때운 것이 아니라 상류가 하는 것을 한 것

`mm` · `bmm` · `addmm` · `matmul` 이 `float16`/`bfloat16` 을 `float32` 로 올려 곱하고 **끝에서
한 번만** 내립니다 (`gemm_accumulate_in`). `addmm` 은 `beta·self + alpha·(A@B)` 전체를
누적 dtype 에서 하고 한 번만 내립니다 — 곱을 먼저 내리고 bias 를 `float16` 으로 더하면
상류가 한 번 반올림하는 자리에서 두 번 반올림합니다.

정수 dtype 은 **일부러 안 올렸습니다.** candle 에 정수 matmul 이 없는 것은 진짜 갭이고,
`float32` 는 `int64` 곱을 정확히 담지 못합니다. 대신 세워두면 다른 질문에 답하는 것이 됩니다.

**부수 효과 하나: `bfloat16` 에 matmul 이 생겼습니다.** candle 에는 BF16 matmul 커널이 아예
없어서(`unsupported dtype BF16 for op matmul`) `mm`/`bmm`/`addmm` 이 이것을 갭으로 기록하고
있었는데, `float32` 로 누적하면 **candle 에게 BF16 matmul 을 요구할 일이 없어집니다** — 상류도
요구하지 않기 때문입니다. 갭을 덮은 것이 아니라 갭에 도달하지 않게 된 것이라, `cases.py` 의
`_MM_C_ERROR_DTYPES` 에서 `bfloat16` 을 빼고 `_MM_MATCH_DTYPES` 로 옮겼습니다. 골든이
"gap appears CLOSED" 로 먼저 알려줬고, 그 지시대로 승격했습니다.

수정 후 `float16`·`bfloat16` GEMM 은 상류와 비트가 같습니다.

### 4.4 평평한 허용오차는 GEMM 을 서술할 수 없다 — 그래서 케이스가 자기 판정자를 갖는다

`dtypes.py` 는 자기 docstring 에서 허용오차를 **"크기 1 에서 대략 1 ulp"** 로 잡았다고
말합니다. 길이 k 의 내적은 크기 1 짜리 입력에서 크기 1 짜리 답을 내지 않습니다 — 크기
`~sqrt(k)` 를 내고, 오차 한계는 **깊이와 출력 자체의 크기 양쪽에 비례**합니다. 그것을 상수와
비교하는 것은 범주 오류입니다: 원소가 작으면 상대오차 1e-3 짜리 텐서를 통과시키고, 원소가
크면 맞는 답을 떨어뜨립니다.

그래서 큰 크기 케이스들은 `Case.value_check` 로 자기 판정자를 답니다:

```
max|torch − c|  ≤  C · u(누적 dtype) · sqrt(k) · max|torch|  +  저장 dtype 의 1 ulp
```

세 가지가 다 이유가 있습니다.

- **`sqrt(k)`, `k` 가 아니다.** 교과서 한계가 k 에 선형인 것은 모든 반올림 오차의 부호가 같다고
  가정하기 때문입니다. 그렇지 않고, 선형 한계는 **틀린 커널도 통과시킬 만큼 헐겁습니다.**
- **`u(누적 dtype)`, `u(저장 dtype)` 이 아니다.** 이것이 §4.2 를 잡는 항입니다.
  처음 쓸 때는 `u(float16)` 을 썼고, 그러자 **틀린 커널이 20 배 여유로 통과**했습니다.
  `float32` 로 바꾸자 4.4 배 차이로 떨어졌습니다. **판정자를 먼저 틀려본 것이 이 절의 근거입니다.**
- **끝의 1 ulp 항.** 상류는 한 번만 반올림합니다. 같은 방법으로 누적한 셰임은 그 한 번 안에
  들어옵니다. 즉 통과 조건이 "운이 좋다"가 아니라 **"상류와 같은 방법을 쓴다"** 가 됩니다.

**허용오차를 늘려 숨긴 것이 아닙니다.** 잰 모든 크기에서 이 한계는 실제 오차보다 5~20 배
빡빡하고, 누적 dtype 이 어긋나면 실패합니다. 판정자는 **평평한 허용오차라면 뭐라고 했을지도
함께 계산해서 출력**하므로, `-v` 로 돌리면 두 판정이 나란히 찍힙니다. 이 문서를 믿을 필요가
없습니다.

수정 전 그 줄이 실제로 이렇게 찍혔습니다:

```
FAIL aten.mm.default :: mm(dtype=float16, (8,512)x(512,8)) [model-scale, k=512]
  -- GEMM error exceeds the scale-aware bound -- this is an accumulation difference,
     not rounding: k=512 n=64 max|d|=0.07812 |out|max=17.98 bound=0.01766
     ...; flat atol=0.005/rtol=0.005 would FAIL on 15/64 elements
```

마지막 절이 중요합니다: **평평한 허용오차도 이것을 거부했습니다.** 아무도 못 본 이유는
허용오차가 헐거워서가 아니라 **골든이 이 크기를 한 번도 안 돌렸기 때문**입니다. §7 의 지적이
정확히 맞았고, 다만 걸리는 dtype 이 `float32` 가 아니라 `float16` 이었습니다.

---

## 5. 진짜 판정 — Gemma 와 BERT 를 aten 레벨로 조립해 상류와 대조

`transformers` 는 셰임 위에서 아직 임포트되지 않으므로(`torch.distributed.Store` 벽)
`GemmaForCausalLM` 로는 판정할 수 없습니다. `docs/models/GPT2.md` §4 와 같은 방법을 쓰되,
**다리를 하나 더 놓았습니다:**

```
A. HF 모듈 (상류 torch)        vs  aten 전사 (상류 torch)     → 전사가 정말 그 아키텍처인가
B. aten 전사 (상류 torch)      vs  aten 전사 (셰임)           → 셰임이 같은 것을 계산하는가
```

A 가 없으면 B 는 "어떤 텐서 프로그램이 두 백엔드에서 같다"는 말밖에 안 됩니다. A 는 같은 LCG
가중치를 실제 HF 모듈에 `load_state_dict` 로 넣어서 확인합니다.

### Gemma — 2 층, GQA 2:1, `gelu_pytorch_tanh`

전사한 것: `sqrt(hidden)` 임베딩 스케일, `(1 + weight)` 형 `GemmaRMSNorm`, head_dim 32 ·
kv_head 1 의 GQA(`repeat_kv`), RoPE, causal `sdpa`, `gelu(approximate='tanh')` MLP, 묶인 `lm_head`.

```
A. 전사 충실도 : 토큰 일치       max|d| logits = 9.54e-07
B. 셰임 vs 상류: 토큰 일치       max|d| logits = 1.55e-06
   위치별 argmax  일치
   가중치 x3     greedy [84,56,80,16] 일치, 위치별 일치, max|d| logits = 9.88e-06
   가중치 x6     greedy [89,52,60,17] 일치, 위치별 일치, max|d| logits = 2.98e-05
```

가중치를 3 배·6 배로 키운 변형을 넣은 것은 `docs/models/GPT2.md` §4.1 의 경고 때문입니다 —
학습되지 않은 모델의 greedy 는 고정점으로 무너집니다(원래 크기에서 실제로 `[61,61,61,61]`
이었습니다). 키우면 greedy 가 실제로 움직이고, 그 상태에서도 일치합니다.

### BERT — 2 층 인코더 + pooler

전사한 것: `gather` 를 포함한 임베딩(§2), bias 있는 Q/K/V(`addmm`), 비causal `sdpa`,
post-LN, 정확형 `gelu` FFN, `tanh` pooler.

```
A. 전사 충실도 : max|d| hidden = 0        pooled = 0        ← HF 모듈과 비트 일치
B. 셰임 vs 상류: max|d| hidden = 1.43e-06  pooled = 9.39e-07
                 n=384, |hidden|max = 3.43
```

A 가 **정확히 0** 입니다. 전사가 BertModel 과 op 단위로 같다는 뜻이고, 따라서 B 의 1.43e-06 은
"BERT 를 셰임에서 돌린 오차"입니다.

### 5.1 greedy 토큰은 이 크기에서 `gelu` 를 판별하지 못한다 — 적어 둔다

대조군을 돌렸습니다: **같은 Gemma 를 `approximate='none'` 으로만 바꿔서** 셰임에서 실행.

```
scale 1: 올바른 gelu [61,61,61,61]  틀린 gelu [61,61,61,61]  토큰 SAME, 위치별 SAME, max|d| logits 5.87e-04
scale 3: 올바른 gelu [84,56,80,16]  틀린 gelu [84,56,80,16]  토큰 SAME, 위치별 SAME, max|d| logits 1.70e-03
scale 6: 올바른 gelu [89,52,60,17]  틀린 gelu [89,52,60,17]  토큰 SAME, 위치별 SAME, max|d| logits 1.84e-03
```

**세 설정 모두 토큰이 같습니다.** 즉 이 규모에서 토큰 일치는 필요조건이지 충분조건이 아니고,
§5 의 판정을 지탱하는 것은 **로짓 차이**입니다: 올바른 식이 1.55e-06, 틀린 식이 5.87e-04 —
**379 배**입니다. 로짓 대조 없이 토큰만 봤다면 이 작업의 핵심 결정(어느 근사식인가)을
검증하지 못한 채 통과했을 것입니다.

이것은 GPT2.md §4.1 이 "greedy 는 do_sample 보다 약한 증거"라고 적은 것의 더 강한 버전입니다:
**greedy 는 약한 증거인 정도가 아니라, 이 판별에 대해서는 증거가 아닙니다.**

---

## 6. 도달한 숫자 (전부 종료 코드와 함께)

```
골든                    1934/1934 통과, 실패 0, ops covered=85, pending 0      exit 0
  --inject-fault value                                                        exit 1
  --inject-fault shape                                                        exit 1
  --inject-fault dtype                                                        exit 1
스키마                  170/170 (overloads 72/72 + methods 98/98)              exit 0
스모크                  65 ok                                                  exit 0
호스트 빌드                                                                    exit 0
aarch64-linux-android                                                          exit 0
aarch64-apple-ios                                                              exit 0
```

스키마 170/170 은 **변하지 않았습니다.** `overloads.json`/`methods.json` 은 이 작업의 범위
밖이고 한 줄도 안 고쳤으므로 그것이 맞는 결과입니다 — 세 op 의 파이썬 철자가 아직 없다는
뜻이기도 합니다(§3, §7).

골든 케이스 증가분 1781 → 1934 (+153):

```
aten.gelu.default     82   4 dtype x 3 approximate x (5 시나리오 + nan/inf) = 72, 거부 10
aten.gather.default   32   dtype 9 + 축/모양 8 + 랭크 4 + 거부 10 + 비연속 1
aten.zero_.default    13   dtype 10 + 0-d/빈/nan 3
                     ---
                     127

mm    +9    큰 크기 6, bfloat16 승격 (c_error 1 -> match 4)
bmm   +3    큰 크기 2, bfloat16 승격 (c_error 1 -> match 2)
addmm +14   큰 크기 3, bfloat16 승격 (c_error 2 -> match 13)
                     ---
                     +26      큰 크기 11 개는 전부 scale-aware value_check

합계 127 + 26 = 153
```

---

## 7. 구현한 것 / 때운 것 / 못 한 것 / 모르는 것

### 구현한 것

- `aten.gelu.default` — 두 근사식, `float32`/`float64` tanh 가지는 상류와 비트 일치,
  `float16`/`bfloat16` 은 양쪽 다 비트 일치. 거부 4 종.
- `aten.gather.default` — `ensure_nonempty` 랭크 규칙 포함, candle 의 세 불일치를 손으로 우회.
- `aten.zero_.default` — 별도 오버로드로.
- **`mm`/`bmm`/`addmm`/`matmul` 의 누적 dtype** — 지시받은 범위 밖이지만 §4.2 에서 나온
  실제 수치 불일치이고, `aten.rs` 는 이 작업의 배타적 파일이라 여기서 고쳤습니다.
  `bfloat16` matmul 이 부수적으로 생겼습니다.
- 골든 큰 크기 케이스 + `sqrt(k)` 기반 scale-aware 판정자.

### 때운 것 (없음, 하나 빼고)

정확형 `gelu` 의 `float32` 1.79e-07 은 candle 의 `libm::erff` 와 상류 커널의 차이이고
조립으로 닫히지 않습니다. **닫지 않고 그대로 뒀습니다.**

그 최대값이 어디서 나오는지는 확인했습니다: `x = -3.0`, 즉 `x·(1 + erf(x/√2))` 가 상쇄되는
음의 로브입니다(출력 `-0.00405`). 절대오차 1.79e-07 은 골든 `float32` 허용오차의 1/56 이지만
**그 자리의 상대오차는 4.4e-05** 입니다 — 상쇄가 일어난 원소의 상대오차를 그대로 읽으면 안
된다는 `docs/models/GPT2.md` §3.3 의 경고가 여기에도 그대로 적용됩니다.

### 못 한 것

- **세 op 의 파이썬 철자.** `overloads.json`/`methods.json`/`bootstrap.py` 는 범위 밖이라
  한 줄도 안 고쳤습니다. 세 커널 다 `torch.ops.aten.*` 로는 도달하지만 `Tensor.gelu()` ·
  `Tensor.gather()` · `Tensor.zero_()` 로는 아직 못 갑니다. **`zero_` 는 이것이 특히 아픕니다** —
  §0 이 보인 대로 `nn.LayerNorm` 의 생성자가 그 철자로 부르기 때문에, 커널만으로는
  GPT2.md 가 보고한 벽이 그대로입니다.

  > **Correction (문서 감사, 2026-09):** 셋 중 둘이 닫혔습니다. `methods.json` 에 지금
  > `"gather"`/`"zero_"` 키가 있고, 실측: `torch.tensor([1,2,3]).gather(0, torch.tensor([0]))`
  > 와 `torch.tensor([1.0,2.0]).zero_()` 둘 다 오늘 성공합니다. **`nn.LayerNorm(4)` 생성도
  > 오늘 성공합니다** — GPT2.md 가 보고한 벽이 실제로 닫혔다는 뜻입니다. `Tensor.gelu()` 는
  > 여전히 안 닫혔습니다(`methods.json` 에 `"gelu"` 키 없음, `AttributeError` 로 실측 재확인 —
  > `NotImplementedError` 조차 아니고 애초에 메서드로 시도되지 않습니다).
  > <!-- DOCWATCH: json-key rust/torch_c/src/methods.json gather present -->
  > <!-- DOCWATCH: json-key rust/torch_c/src/methods.json zero_ present -->
- **§5 의 판정을 회귀 테스트로 못 박지 못했습니다.** `pytests/test_shim.py` 가 범위 밖입니다.
  `_E2EBackend` 옆에 Gemma·BERT 전사를 놓으면 그대로 테스트가 됩니다 — 다음 작업 항목.
- 다음 4-op 묶음(`le.Tensor` · `scalar_tensor` · `where.self` · `permute`)은 안 건드렸습니다.

### 모르는 것 / 확인하지 않은 것

- **`float32` GEMM 이 k ≤ 512 에서 왜 비트까지 같은지 모릅니다.** candle 의 `gemm` 크레이트와
  상류의 커널이 같은 순서로 누적한다는 뜻인데, 우연인지 구조적인지 안 봤습니다. k=1024 에서
  깨지는 것으로 보아 블로킹 크기의 문제일 가능성이 큽니다. **구조적이라고 가정하지 마십시오.**
- **`float64` 정확형 `gelu` 의 2.78e-17** 은 `libm::erf` 와 상류 `erf` 의 차이라고 보지만,
  상류가 어떤 구현을 쓰는지 확인하지 않았습니다.
- **시도한 37 개 중 5 개는 못 쟀습니다** (`modernbert` · `phi3` · `glm` · `whisper` 는 이 작은 vocab
  설정에서 `Padding_idx` 로 생성 실패, `t5` 는 `decoder_start_token_id` 미설정). 설정을 고치면
  꼬리가 더 나올 수 있습니다.
- **전부 2 층 · hidden 64 입니다.** sliding window, MoE 라우팅(mixtral 의 9 개가 그 증거),
  긴 컨텍스트를 켜면 더 나옵니다.
- **`bfloat16` GEMM 은 골든에서 k=512 까지만 봤습니다.** `float32` 가 k=1024 에서 갈라지므로
  `bfloat16` 도 어딘가에서 갈라질 것이나, 어디인지 안 쟀습니다.
- **기기(Android/iOS) 임포트는 이번에도 링크만 확인했습니다.**
