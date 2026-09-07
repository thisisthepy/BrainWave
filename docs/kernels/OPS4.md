# 네 아키텍처가 공유하던 4-op 묶음 — 그리고 그 넷 중 하나만 실제로 열렸다

`docs/architectures/ARCH.md` 가 32 개 아키텍처를 실측해 남긴 지형에서 **가장 큰 단일 덩어리**를 구현한 기록입니다.
`aten.le.Tensor` · `aten.scalar_tensor.default` · `aten.where.self` · `aten.permute.default` 넷과,
그다음 순위인 `aten.stack.default` · `aten.relu.default` 둘. 여섯 개.

**한 줄 결론이 두 개입니다.**

1. 여섯을 넣자 **미구현 0 인 아키텍처가 5 → 14** 로 늘었고, MPT 를 aten 레벨로 조립한 결과가
   상류 `MptForCausalLM` 과 **로짓 최대차 1.05e-05** 로 일치합니다(전사 충실도는 **0**).
2. **그런데 "넷을 넣으면 아키텍처 4 개가 한꺼번에 열린다"는 전제는 틀렸습니다.** 다시 재보니
   `mpt` 만 미구현 0 이 되고 `falcon` · `gptj` · `bloom` 은 각각 1~2 개가 남습니다. ARCH.md 는
   그런 주장을 한 적이 없고 — 그 문서의 "나머지 17 개는 전부 1" 이 바로 이 남은 것들입니다 —
   **작업 지시가 그 표를 한 칸 넓게 읽은 것**입니다. §3 에 실측을 적었습니다.

기준선 대비:

| | 이전 | 이후 |
|---|---|---|
| 골든 케이스 | 1934 | **2095** (실패 0, pending 0, exit 0) |
| `_aten_implemented()` | 85 | **91** |
| `_aten_all_implemented()` | 97 | **103** |
| 미구현 0 인 아키텍처 | 5 / 20 실측 | **14 / 20 실측** |
| 스키마 | 199/199 | 199/199 (변화 없음) |
| 3 타깃 | exit 0 | exit 0 |
| 스모크 | 65 ok | **63 ok, 2 FAIL** — §7, 의도된 적색 |
| `--self-test` | exit 0 | **exit 1** — §6, 의도된 적색 |

---

## 1. 먼저 네 아키텍처가 그 넷을 *어떻게* 부르는지 기록했다

ARCH.md 는 op 이름과 빈도만 남겼습니다. 구현 전에 **인자의 dtype 과 shape 까지** 기록했습니다.
`TorchDispatchMode`, 2 층 · hidden 64 · heads 2, greedy ∪ do_sample 합집합:

```
falcon   4  where.self(bool(1,1,6,6), float32(), float32())
        12  scalar_tensor(-inf,  dtype=float32,               device=cpu)
        12  scalar_tensor(0.0,   dtype=float32, layout=strided, device=cpu)
         2  le.Tensor(int64(1,1,1,6), int64(1,1,6,1))
        24  permute(float32(128,64), [1,0])          ← 모든 가중치가 여기를 지난다
         8  permute(float32(1,2,1,32), [0,2,1,3])
gptj    16  stack([float32(1,1,2,4), float32(1,1,2,4)], -1)
         6  scalar_tensor(-3.4028234663852886e+38, dtype=float32, layout=strided, device=cpu)
bloom    6  scalar_tensor(-3.4028234663852886e+38, ...)   (falcon/gptj/mpt 와 동일)
mpt      4  permute(float32(1,2,6,32), [0,2,1,3])
```

여기서 세 가지가 나오고, **셋 다 구현 결정을 바꿨습니다.**

**(1) `where.self` 의 두 분기는 항상 0-D 이고 항상 같은 dtype 이다.** 조건은 `(1,1,S,S)` bool 이고
분기 둘은 `scalar_tensor` 가 만든 0-D `float32` 입니다. 즉 이 네 아키텍처는 **타입 승격을 한 번도
요구하지 않습니다.** 승격은 `same_dtype` 으로 거부하고 그 자리를 골든의 `c_error` 로 남겨도
아무것도 막히지 않는다는 것이 이 측정의 결론입니다(§2).

**(2) `layout=torch.strided` 가 실제로 넘어온다.** `reject_unsupported` 는 `None` 이 아닌 모든
layout 을 거부합니다. 그대로 뒀다면 셰임이 갖고 있는 유일한 레이아웃을 이름으로 요청했다는
이유로 세 아키텍처가 막혔을 것입니다. `reject_layout` 을 따로 만들어 `strided` 만 통과시켰습니다
(`full`/`ones`/`empty` 는 **안 건드렸습니다** — 실측된 호출자가 없습니다).

**(3) `permute` 는 두 모양뿐이다.** 2-D `[1,0]` 과 4-D `[0,2,1,3]`. 둘 다 대합(involution)이라
**역치환 혼동을 모델 실행으로는 잡을 수 없습니다.** 골든에 `[2,0,1]` 과 `[1,2,0]` 을 서로의
역으로 넣은 것이 그 자리입니다.

---

## 2. `where.self` — 승격표를 다 재고, 그래도 승격하지 않기로 했다

`aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor`

### 2.1 조건의 dtype 규칙은 `masked_fill` 의 것이 아니다

```
where(bool  cond)  ->  계산
where(uint8 cond)  ->  계산 + "where received a uint8 condition tensor. This
                       behavior is deprecated and will be removed..."
where(int64 cond)  ->  RuntimeError: where expected condition to be a boolean
                       tensor, but got a tensor with dtype Long
where(int32/float32 cond) -> 같은 거부
```

`masked_fill` 은 `uint8` 마스크를 **거부**하고, `BOOL.md` §3 이 그 거부를 유지할 가치가 있는
가드레일로 적어 두었습니다. 여기서는 상류가 **받습니다.** 그 규칙을 그대로 옮겨왔다면
**상류가 답하는 호출을 거부**했을 것입니다. 반대로 여기 규칙을 `masked_fill` 로 가져갔다면
조용한 쪽 divergence 가 됐을 것입니다. 두 op 은 다른 op 이고 셰임에서도 다르게 뒀습니다.

`uint8` 조건은 비트가 아니라 **참/거짓**입니다. 값 2 와 5 를 넣어 확인했고 양쪽 다 첫 분기를
고릅니다(골든 케이스 있음).

### 2.2 결과 shape 은 조건의 것이 아니라 **셋의 조인**이다

```
where(tensor(True), ones(2,3), zeros(3))  ->  (2, 3)
```

조건이 0-D 인데 결과는 `(2,3)` 입니다. 조건 shape 으로 결과를 잡는 구현은 여기서 `()` 를 냅니다.
그래서 `masked_fill` 처럼 둘만 broadcast 하지 않고 셋을 함께 broadcast 합니다.

### 2.3 승격표 9×9 를 다 쟀고, 구현하지 않았다

전부 실측(torch 2.13.0). 추론으로 틀렸을 네 줄:

| self | other | 상류 결과 | 왜 추론으로 틀리나 |
|---|---|---|---|
| `float16` | `int64` | `float16` | **정수 분기는 실수 분기를 넓히지 않는다.** "더 넓은 쪽" 규칙이면 `int64` |
| `float16` | `bfloat16` | `float32` | 둘 다 16 비트인데 **밖으로** 승격한다 |
| `bfloat16` | `int32` | `bfloat16` | 위와 같은 이유 |
| `bool` | `uint8` | `uint8` | `bool` 이 격자의 바닥 |

**그런데 구현하지 않았습니다.** `same_dtype` 으로 거부하고 이름을 대게 했습니다. 근거 셋:

1. `same_dtype` 의 주석이 이 저장소의 규약을 적어두고 있습니다 — 두 텐서 사이의 조용한 승격은
   `DESIGN.md` §5 가 candle 의 주된 위험이라고 부르는 수치 표류이고, 거부는 작업 항목입니다.
   `add.Tensor` · `lt.Tensor` · `cat.default` 가 전부 이미 그렇게 합니다. `where` 만 예외로 두면
   **한 파일 안에 두 규약**이 생깁니다.
2. §1 이 잰 대로 **네 아키텍처 중 어느 것도 섞지 않습니다.** 구현해도 검증할 호출이 없습니다.
3. 대신 **표를 골든에 `c_error` 로 박았습니다.** 갭이 잊히지 않고, 상류가 승격을 바꾸거나
   셰임이 승격을 얻으면 그 케이스가 먼저 말합니다.

### 2.4 안 고른 분기는 값이 읽히지 않는다

`where(True, 1.0, nan)` 은 `1.0` 입니다(실측). 혼합이 아니라 선택이므로 `where_cond` 도 같습니다.
골든에 있습니다 — 두 분기를 곱해 더하는 구현이라면 여기서만 틀립니다.

---

## 3. 다시 쟀다 — 그리고 넷 중 하나만 열렸다

구현 후 같은 방법으로 다시 쟀습니다. `_aten_all_implemented()` 를 **빌드한 산출물에서 직접**
읽고, "이번 6 개를 빼면 어땠을까"를 같은 실행에서 함께 계산했습니다.

```
                 before  after
falcon    ops=54     6     2    남음: add_.Tensor, div_.Tensor
gptj      ops=57     6     1    남음: repeat.default
bloom     ops=53     5     1    남음: baddbmm.default
mpt       ops=45     4     0    ← 미구현 0
opt       ops=49     4     0    ← 미구현 0 (relu)
persimmon ops=53     4     0    ← 미구현 0 (relu)
cohere    ops=54     4     0    ← 미구현 0 (stack)
gemma · gpt2 · llama · gpt_neox · starcoder2   3 -> 0
bert · roberta · electra · distilbert · albert 0 -> 0  (이미 열려 있었음)
gpt_bigcode ops=50   4     1    남음: split_with_sizes.default
mamba     ops=53     6     5    남음: convolution, exp, softplus, split_with_sizes, zeros_like
mixtral   ops=65    12     9    남음: _grouped_mm, clamp_, div_, empty_like, floor_divide,
                                      ge.Scalar, histc, index_put_, masked_fill_

미구현 0 인 아키텍처: before=5  after=14  (실측 20 개 중)
```

> **정정 (§13):** 위 표의 `mamba`/`mixtral` 줄은 이 시점의 실측이고, 이후 낡았습니다.
> `split_with_sizes.default` 는 다음 회차에서 이미 구현됐고, §13 이 마저 다 구현해
> 둘 다 미구현 0 으로 만들었습니다(`mixtral` 은 `_grouped_mm.default` 하나만 범위 밖으로
> 남겼습니다). 권위 있는 값은 이 표가 아니라 `_aten_implemented()` 실측입니다.

### 3.1 전제가 틀렸다 — 그리고 어디서 틀렸는지 확인했다

작업 지시는 "넷을 넣으면 아키텍처 4 개가 한꺼번에 열린다" 였습니다. **열린 것은 `mpt` 하나입니다.**

구현 *전*에 찍어둔 트레이스를 다시 봤습니다. `falcon` 의 `_safe_softmax`/`add_.Tensor`,
`gptj` 의 `repeat.default`, `bloom` 의 `baddbmm.default` 는 **처음부터 거기 있었습니다.** 즉
이번 구현이 뭔가를 놓친 것이 아니라, 그 셋은 원래부터 그 넷 말고도 각자 하나씩을 더 필요로
했습니다.

그리고 ARCH.md 는 그 반대를 주장한 적이 없습니다. 그 문서의 목록은 **"이 op 이 몇 개의
아키텍처를 여는가"로 정렬한 것**이고, 끝에 `(나머지 17 개는 전부 1)` 이라고 적혀 있습니다.
`repeat`(gptj 만) · `baddbmm`(bloom 만) · `add_.Tensor`(falcon 만) 가 바로 그 17 개 안에 있습니다.
**"이 네 op 을 부르는 것이 정확히 이 네 아키텍처"** 와 **"이 네 아키텍처에 부족한 것이 이 네
op 뿐"** 은 다른 문장이고, 지시는 앞 문장을 뒤 문장으로 읽었습니다.

`falcon` 의 `div_.Tensor` 는 트레이스에서 attention 스케일링(`scores /= sqrt(d)`)의 in-place
형태로 나옵니다. `add_`/`div_` 는 값이 아니라 **in-place 계열이 통째로 없는 것**이고, 이것은
`mixtral` 의 `clamp_`/`div_`/`masked_fill_` 과 같은 항목입니다 — 다음 덩어리는 op 하나가
아니라 **in-place 오버로드 계열**로 보입니다. 이번 작업의 범위 밖이라 안 건드렸습니다.

### 3.2 못 잰 것

`nemotron` · `helium` · `deberta_v2` 는 이 작은 설정에서 생성이 실패합니다
(`unsupported operand type(s) for //: 'int' and 'NoneType'`,
`The size of tensor a (2) must match the size of tensor b (0)`,
`Unrecognized configuration class`). ARCH.md 가 다섯 개를 못 쟀다고 적은 것과 같은 종류이고,
**설정을 고치면 꼬리가 더 나올 수 있습니다.** `relu` 가 연다고 적힌 셋 중 `nemotron` 은 그래서
이번에도 확인 못 했습니다 — `opt` 와 `persimmon` 둘은 확인했습니다.

`vit` 은 `AutoModelForCausalLM`/`AutoModel` 의 텍스트 입력 경로가 아니라 빼뒀습니다.

---

## 4. 진짜 판정 — MPT 를 aten 레벨로 조립해 상류와 대조

`transformers` 는 셰임 위에서 아직 임포트되지 않으므로 `MptForCausalLM` 로는 판정할 수
없습니다. `docs/architectures/ARCH.md` §5 와 같은 2 단 방법을 씁니다.

```
A. HF 모듈 (상류 torch)       vs  aten 전사 (상류 torch)   → 전사가 정말 그 아키텍처인가
B. aten 전사 (상류 torch)     vs  aten 전사 (셰임)         → 셰임이 같은 것을 계산하는가
```

전사한 것: `wte` 임베딩, ALiBi 편향(`arange`·`pow`·`reciprocal`·`squeeze`·`slice`),
bias 없는 `LayerNorm`, `Wqkv` 한 장을 `split` 으로 셋으로, `matmul`+`softmax_scale`,
`+ position_bias`, `masked_fill`, `float32` softmax, `permute([0,2,1,3])`+`contiguous`+`view`,
정확형 `gelu` FFN, 묶인 `lm_head`.

인과 마스크는 네 아키텍처가 공유하는 관용구 그대로입니다:

```
allowed = le.Tensor(kv_idx, q_idx)
floated = where.self(allowed, scalar_tensor(0.0), scalar_tensor(finfo.min))
mask    = floated != 0                       # `.to(torch.bool)` 의 자리
```

**결과 (2 층 · hidden 64 · heads 2 · seq 6 · vocab 128):**

```
|logits|max = 11.1043
A  HF MptForCausalLM vs aten 전사 (torch) : max|d| logits = 0            ← 비트 일치
B  aten (torch)      vs aten (_C 셰임)    : max|d| logits = 1.04904e-05
   greedy 토큰  세 쪽 모두 [126, 71, 110, 79, 71, 23]
```

**A 가 정확히 0 입니다.** 전사가 `MptForCausalLM` 과 op 단위로 같다는 뜻이고, 따라서 B 의
`1.05e-05` 는 "MPT 를 셰임에서 돌린 오차" 입니다. 스케일 정규화하면 `9.4e-07`.

### 4.1 A 를 먼저 틀려봤다 — 그게 이 절의 근거다

처음 돌렸을 때 A 가 **14.5** 였고 B 는 그때도 `1e-05` 였습니다. 원인은 셰임이 아니라 전사였습니다:
**MPT 는 `lm_head` 를 `wte` 에 묶습니다**(`tie_word_embeddings=True` 이고 두 `Parameter` 가
같은 객체 — 실측). `lm_head` 에 따로 가중치를 준 것이 전부였습니다.

이것이 2 단 방법의 값어치입니다. **A 없이 B 만 봤다면 `1e-05` 를 보고 "MPT 가 통과했다"고
적었을 것이고, 실제로는 MPT 가 아닌 것을 돌리고 있었습니다.**

### 4.2 토큰 일치는 증거가 아니다 — 대조군 세 개

`docs/architectures/ARCH.md` §5.1 이 **틀린 gelu 로도 토큰이 똑같이 나왔다**고 적었습니다. 같은 질문을 이번
op 들에 했습니다. 셰임 쪽만 한 군데씩 일부러 틀리고 로짓 차이를 잰 것:

```
정상                                                    max|d| = 1.04904e-05   토큰 일치
le.Tensor -> lt.Tensor (인과 마스크 한 칸 밀림)          max|d| = 12.3254       토큰 불일치
where.self 의 두 분기를 뒤바꿈                           max|d| = 16.3423       토큰 불일치
permute([0,2,1,3]) -> 항등 치환                         max|d| = 15.1902       토큰 불일치
```

셋 다 **10⁶ 배**입니다. 마지막 것이 특히 중요합니다: 이 자리에서 항등 치환은 **shape 이 맞습니다**
(`1·2·6·32 == 1·6·64`). 즉 `dims` 를 무시하는 `permute` 는 뒤따르는 `view` 를 그대로 통과해
**예외 없이 틀린 수**를 답합니다.

**다만 `scalar_tensor` 는 이 모델이 판별하지 못합니다.** MPT 가 넘기는 값이 전부 float 이라
"정수도 float32 가 된다"는 §5.1 의 규칙이 여기서는 드러나지 않습니다. 그 규칙을 지키는 것은
골든 케이스이지 이 대조가 아닙니다.

---

## 5. `relu` 와 `stack` 은 MPT 가 안 부른다 — 따로 조립했다

같은 A/B 방법을 두 조각에 적용했습니다.

### 5.1 `OPTDecoderLayer` 통째 (relu)

```
|out|max = 21.8417
A  HF OPTDecoderLayer vs aten 전사 (torch) : max|d| = 3.8147e-06
B  aten (torch)       vs aten (_C 셰임)    : max|d| = 1.52588e-05
대조군 relu(x) -> -relu(-x) (반대쪽 반직선) : max|d| = 20.4177
```

### 5.2 GPT-J `apply_rotary_pos_emb` (stack)

`rotate_every_two` 가 `stack((-x2, x1), dim=-1)` 이고, `repeat_interleave(·, 2, -1)` 도
`stack([x, x], -1)` + reshape 으로 전사했습니다.

```
|out|max = 1.27288
A  HF apply_rotary_pos_emb vs aten 전사 (torch) : max|d| = 0     ← 비트 일치
B  aten (torch)            vs aten (_C 셰임)    : max|d| = 0     ← 비트 일치
대조군 stack(dim=-1) -> stack(dim=1) 후 reshape  : max|d| = 1.97051
```

B 가 **0** 입니다. `stack` 은 값을 계산하지 않고 옮기기만 하므로 이것이 맞는 결과이고,
반대로 `1e-07` 이라도 나왔다면 그게 조사할 일이었습니다.

---

## 6. `relu` 는 `max(x, 0)` 이 아니다

한 줄짜리 op 이고, 흥미로운 것 전부가 **어느 한 줄인가**에 있습니다.

```
relu([nan, inf, -inf, -0.0, 0.0])  ==  [nan, inf, 0.0, -0.0, 0.0]
signbit(relu(-0.0))                ==  True
```

`nan` 이 **살아남고** `-0.0` 이 **부호를 유지**합니다. 앞의 것은 비교 순서로 한쪽이 이기는
최댓값이 아니라는 뜻이고, 뒤의 것은 "클램프 후 정규화" 가 아니라는 뜻입니다. 둘 다
`x < 0 ? 0 : x` 에서 그냥 나옵니다 — `-0.0 < 0` 도 `nan < 0` 도 거짓이라 원소가 그대로 지나갑니다.

**`max` 모양의 구현은 위의 모든 케이스를 통과하고 정확히 이 두 입력에서만 틀립니다.**
골든이 둘 다 박아 뒀습니다(`float32` 와 `float64` 양쪽).

정수 dtype 은 **거부하지 않습니다.** `silu` 는 상류에 정수 CPU 커널이 없어 거부하지만 `relu` 는
있습니다. 한 함수 아래의 거부를 여기로 옮겼다면 상류가 답하는 호출을 거부했을 것입니다.
`bool` 만 거부하고 문구도 상류의 것입니다(`Boolean inputs not supported for relu`).
`uint8` 에서는 항등인데 이건 특례가 아니라 결과입니다 — 음수 원소가 없습니다.

---

## 7. `scalar_tensor` 의 dtype 추론은 `full` 의 것이 아니다

```
scalar_tensor(3)     -> float32        full([], 3)     -> int64
scalar_tensor(True)  -> float32        full([], True)  -> bool
scalar_tensor(1.5)   -> float32
```

**값의 범주를 통째로 무시하고 항상 기본 실수형입니다.** `full` 을 보고 유추했다면 상류가
`float32` 를 주는 자리에서 `int64` 를 주었을 것이고, 마스크 채움값이 조용히 잘렸을 것입니다.

넘침 규칙은 반대로 **`full` 의 것 그대로**이고, 그 안의 numel==1 구멍까지 같습니다:

```
scalar_tensor(1e6,   float16)  -> inf        (full([3], 1e6, float16) 은 거부)
scalar_tensor(1e300, bfloat16) -> inf
scalar_tensor(1e300, float32)  -> 거부       ← float32 는 구멍에 없다
scalar_tensor(2**40, int32)    -> 거부
scalar_tensor(-1,    uint8)    -> 255        (2 의 보수 감김, 크기가 맞으므로 허용)
scalar_tensor(300,   uint8)    -> 거부
scalar_tensor(-1.5,  int64)    -> -1         (0 쪽으로 절단, -2 아님)
scalar_tensor(nan,   int64)    -> 거부
```

즉 `checked_convert(..., numel = 1)` 이 정확히 이 표입니다. **재서 확인했지 유추하지
않았습니다** — `float16` 은 통과하고 `float32` 는 거부한다는 비대칭을 뒤집으면 절반이 틀립니다.

---

## 8. `permute` 는 별칭인가 — 상류는 그렇고, 이 셰임은 아니다

**질문에 답이 있습니다.** 추측이 아니라 양쪽을 다 돌렸습니다.

```
상류
  permute(x, [1,0]).data_ptr() == x.data_ptr()      True
  결과 contiguous                                    False, strides (1,3) ← base (3,1)
  뷰에 fill_(99)  -> base                            [99,99,99,99,99,99]
  뷰에 copy_(7)   -> base                            [7,7,7,7,7,7]

셰임
  뷰에 fill_.Scalar  -> base                         [0,1,2,3,4,5]   변화 없음
  뷰에 copy_.default -> base                         [0,1,2,3,4,5]   변화 없음
  t / transpose / slice 도 전부 동일                  [0,1,2,3,4,5]
```

**candle 쪽에서는 저장소가 공유됩니다** — `Tensor::permute` 는 `Arc<Storage>` 를 복제하고
layout 만 바꿉니다. 별칭이 관측되지 않는 이유는 `permute` 가 아니라 **이 파일의 in-place op 들이
저장소에 쓰지 않기 때문**입니다. `replace_with` 에 새 텐서를 넘기므로 뷰는 그 쓰기를 못 봅니다
(`aten.rs` 의 "In-place ops" 주석이 이미 그렇게 적어 두었습니다).

그러므로 **`permute` 는 `slice.Tensor`/`split.Tensor` 가 이미 갖고 있는 것과 같은 미해결 질문을
하나 더 얹은 것이 아니라, 같은 하나의 질문에 속합니다.** 셰임에는 op 별 별칭 규칙이 없고
**규칙이 하나** 있습니다: 어떤 뷰를 통해서도 쓰기가 원본에 닿지 않습니다. `docs/models/GPT2.md` §7 이
`split` 에 대해 "안 쟀다" 고 남긴 항목을, 이번에 **네 op 전부에 대해 쟀습니다.** 답은
"별칭 아님" 이고, 고치지는 않았습니다 — 고치는 것은 `replace_with` 의 설계를 바꾸는 일이고
이 작업의 범위 밖입니다.

---

## 9. 하니스가 스스로 신고한 인덱스 구멍 셋을 막았다

`docs/verification/HARNESS.md` §6 이 `--self-test` 로 찾아 `KNOWN_GAP` 에 박아둔 셋입니다. 셋 다
`(values, indices)` 쌍에서 **`indices` 쪽을 덜 보는** 같은 뿌리입니다.

| 비교기 + 모드 | 무엇을 놓쳤나 | 고친 것 |
|---|---|---|
| `_pair_result_check` + `dtype-last` | `indices` 의 dtype 미비교 | `dt.dtype_name()` 비교 1 줄 |
| `_topk_multiset_check` + `dtype-last` | 같은 구멍 | 같은 1 줄 |
| `_topk_multiset_check` + `shape-last` | `indices` 의 shape 미비교 (multiset 은 reshape 을 견딘다) | shape 비교 1 줄 |

`indices` dtype 이 중요한 이유는 HARNESS.md 가 적은 그대로입니다: 상류의 `max.dim`/`sort`/`topk`
는 `int64` 를 약속하고 그것이 `index_select`·`gather`·`embedding` 으로 그대로 들어갑니다.
값만 같고 dtype 이 다른 인덱스는 **하니스가 통과시킨 뒤 하류에서 터지는** 종류입니다.

수정 후 `--self-test` 표에서 세 칸이 전부 `GAP` → `CAUGHT` 로 바뀌었고,
`_pair_result_check` 는 9/11, `_topk_multiset_check` 는 8/11 을 잡습니다
(`permute-all` 만 의도된 무시로 남습니다).

### 9.1 그래서 `--self-test` 가 실패한다 — 그게 설계다

```
PROBLEM: _pair_result_check + dtype-last: UNEXPECTED -- KNOWN_GAP says this
         should not be caught; that entry is fixed, remove it
PROBLEM: _topk_multiset_check + shape-last: (같은 문구)
PROBLEM: _topk_multiset_check + dtype-last: (같은 문구)

SELF-TEST: FAIL -- 11 comparators x 11 fault modes, 3 problem(s),
           0 comparator(s) never exercised
```

`compare.py` 의 `KNOWN_GAP` 표를 지우는 것은 이 작업의 파일 범위 밖(다른 에이전트 소유)이라
**손대지 않았습니다.** HARNESS.md §3.3 이 "고치면 `--self-test` 가 낡은 표를 실패로 잡는다" 고
일부러 그렇게 만들었다고 적어 두었고, 지금 그 규칙이 정확히 발동한 것입니다.
**`compare.py` 에서 그 세 항목을 지우면 exit 0 으로 돌아갑니다.**
`0 comparator(s) never exercised` 는 그대로이므로 커버리지는 줄지 않았습니다.

### 9.2 스모크 테스트 둘도 같은 종류로 빨갛다

```
FAIL test_op_registry_routes_to_the_one_door : TypeError: aten.relu.default: missing required argument 'self'
FAIL test_unimplemented_op_names_itself      : TypeError: aten.relu.default: missing required argument 'self'
63 ok / 65
```

두 테스트 다 **`aten.relu.default` 를 "구현되지 않은 op" 의 표본으로** 쓰고 있었습니다.
`test_shim.py` 자신이 그 위험을 주석으로 적어 두었습니다 — *"`aten.embedding.default` used to
stand here and now has a kernel, which is the right failure mode for this test -- it goes red when
the op it samples stops being a sample."* 이번에 `relu` 가 그렇게 됐습니다.

`rust/torch_c/pytests/test_shim.py` 는 파일 범위 밖이라 **한 글자도 안 고쳤습니다.**
고치려면 표본을 다른 이름으로 바꾸면 됩니다. `_aten_all_implemented()` 103 개에 없고 §3 의
어느 아키텍처 꼬리에도 없는 이름이면 되고, `docs/design/TORCH_C.md` §1 이 예시로 `relu` 를 쓰고 있으므로
그 문서도 함께 봐야 합니다. **어떤 이름을 고를지는 조율 세션의 판단입니다** — 잘못 고르면
같은 일이 또 일어나고, 이번 실패가 그 증거입니다.

---

## 10. 도달한 숫자 (전부 종료 코드와 함께)

```
골든                    2095/2095 통과, 실패 0, ops covered=91, pending 0      exit 0
  --inject-fault 11 모드 전부                                                  exit 1
  --self-test                                                                  exit 1  ← §9.1, 의도된 적색
스키마                  199/199 (overloads 90/90 + methods 109/109)             exit 0
스모크                  63 ok / 65                                             ← §9.2, 의도된 적색
호스트 빌드                                                                     exit 0
aarch64-linux-android                                                           exit 0
aarch64-apple-ios                                                               exit 0
```

스키마 199/199 는 **변하지 않았습니다.** `overloads.json`/`methods.json` 은 범위 밖이고 한 줄도
안 고쳤으므로 그것이 맞는 결과입니다 — 여섯 op 의 파이썬 철자가 아직 없다는 뜻이기도 합니다(§11).

골든 케이스 증가분 1934 → 2095 (+161):

```
aten.scalar_tensor.default   50   추론 7 + dtype 9x3 + 넘침 12 + 인자 4
aten.stack.default           28   dtype 9 + dim 6 + 모양 7 + 거부 3 + 승격 3
aten.where.self              26   dtype 9 + 모양 6 + 조건 dtype 4 + 값 2 + 승격 4 + 관용구 1
aten.permute.default         23   dtype 9 + 모양 8 + 거부 6
aten.le.Tensor               19   dtype 8 x 시나리오 2 + nan/마스크/bool 3
aten.relu.default            15   dtype 8 + max 판별 2 + 모양 4 + bool 거부 1
                            ---
                            161
```

---

## 11. 구현한 것 / 때운 것 / 못 한 것 / 모르는 것

### 구현한 것

- `aten.le.Tensor` — `compare_tensor` 재사용. 별도 키, 같은 커널(`lt.Tensor`/`lt.Scalar` 와 동형).
- `aten.scalar_tensor.default` — `full` 과 **다른** dtype 추론, **같은** 넘침 규칙(§7).
- `aten.where.self` — 셋 broadcast, `uint8` 조건 허용, 승격은 거부(§2).
- `aten.permute.default` — 정확한 길이 규칙, 중복 거부, 음수 정규화(§8).
- `aten.stack.default` — `cat` 이 아닌 별도 커널, `dim == rank` 허용, 크기 완전 일치 요구.
- `aten.relu.default` — `x < 0 ? 0 : x`, `max` 아님(§6).
- `reject_layout` — `strided` 만 통과. `full`/`ones`/`empty` 는 안 건드렸습니다.
- 골든 161 케이스 + `_pair_result_check`/`_topk_multiset_check` 의 인덱스 구멍 셋(§9).

### 때운 것

없습니다. 승격을 구현하지 않은 것은 때운 것이 아니라 §2.3 의 규약대로 **거부하고 기록**한
것이고, 그 자리가 골든의 `c_error` 로 남아 있습니다.

### 못 한 것

- **여섯 op 의 파이썬 철자.** `overloads.json`/`methods.json`/`bootstrap.py` 는 범위 밖이라 한 줄도
  안 고쳤습니다. 여섯 다 `torch.ops.aten.*` 로는 도달하지만 `torch.where(...)` ·
  `torch.stack(...)` · `Tensor.permute(...)` · `torch.nn.functional.relu(...)` 로는 아직 못 갑니다.

  > **Correction (문서 감사, 2026-09):** 넷 다 지금은 닿습니다. 실측: `torch.where(...)`,
  > `torch.stack(...)`, `torch.ones(2,3).permute(1,0)`, `F.relu(...)` 전부 오늘 성공합니다.
  > `overloads.json` 에 `where`/`stack`/`relu` 키가, `methods.json` 에 `permute` 키가
  > 있습니다. `docs/models/GPT2.md`/`docs/models/SAMPLING.md`/`docs/devices/DEVICE.md`/`docs/architectures/ARCH.md` 감사(이
  > 라운드)에서 반복해서 발견한 것과 같은 패턴입니다 — 나머지 두 op 이름은 이 문단이 밝히지
  > 않아 확인하지 못했습니다.
  > <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json where present -->
  > <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json stack present -->
  > <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json relu present -->
  > <!-- DOCWATCH: json-key rust/torch_c/src/methods.json permute present -->
- **`compare.py` 의 `KNOWN_GAP` 세 항목 제거**(§9.1) 와 **`test_shim.py` 의 relu 표본
  교체**(§9.2). 둘 다 파일 범위 밖이고, 둘 다 지금 빨간 상태입니다.
- **§4·§5 의 판정을 회귀 테스트로 못 박지 못했습니다.** `pytests/test_shim.py` 가 범위 밖입니다.
  MPT · OPT 레이어 · GPT-J 로터리 세 전사는 그대로 옮기면 테스트가 됩니다.
- **`falcon`/`gptj`/`bloom` 을 열지 못했습니다.** 각각 `add_`+`div_` · `repeat` · `baddbmm` 이
  남습니다(§3.1). 앞의 것은 in-place 계열이라 op 하나가 아닙니다.

### 모르는 것 / 확인하지 않은 것

- **`falcon` 의 `_safe_softmax.default` 를 이번 측정에서 어떻게 처리해야 하는지 모릅니다.**
  구현 전 트레이스에도 있었고 지금도 미구현으로 잡히는데, `_aten_all_implemented()` 에도
  ARCH.md 의 명시 목록에도 없습니다. 상류가 어떤 조건에서 `_softmax` 대신 이것을 고르는지
  안 쟀습니다. `attn_implementation="sdpa"` 를 줘도 falcon 은 여전히 이것을 부르고,
  `gptj`/`bloom`/`mpt` 는 이 `transformers` 버전에서 sdpa 를 **지원하지 않습니다**
  (`does not support an attention implementation through torch...`). 즉 이 셋의 측정은
  eager 경로의 것이고, ARCH.md 가 어느 경로로 쟀는지는 그 문서에 적혀 있지 않습니다.
- **`where.self` 의 승격을 구현하면 무엇이 열리는지 모릅니다.** 실측한 20 개 중 승격을
  요구하는 호출은 한 건도 없었고, 더 큰 설정에서도 그런지는 안 쟀습니다.
- **`permute` 의 별칭을 셰임에서 성립시키면 무엇이 깨지는지 모릅니다.** §8 은 지금 성립하지
  않는다는 것을 쟀을 뿐, `replace_with` 를 저장소 쓰기로 바꿨을 때의 파급은 안 봤습니다.
- **전부 2 층 · hidden 64 · seq 6 입니다.** §4 의 `1.05e-05` 는 이 크기의 숫자이고, 층을 늘리면
  자랍니다. 얼마나 자라는지 안 쟀습니다.
- **`scalar_tensor` 를 모델 대조로 판별하지 못했습니다**(§4.2). dtype 규칙을 지키는 것은 골든
  케이스뿐입니다.
- **기기(Android/iOS) 는 이번에도 링크만 확인했습니다.**

---

## 12. 재현

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-ops
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
PY=/Volumes/macMini/caches/spike-venv/bin/python

sh vendor/vendor_torch.sh                       # 새 worktree 라면 먼저
(cd rust/torch_c && cargo build --release)

$PY tools/golden/compare.py             > /tmp/g.log 2>&1; echo "EXIT=$?"   # 0
$PY tools/golden/compare.py --self-test > /tmp/s.log 2>&1; echo "EXIT=$?"   # 1 (§9.1)
$PY rust/torch_c/pytests/verify_schemas.py > /tmp/sch.log 2>&1; echo "EXIT=$?"  # 0
PYTHON=$PY sh rust/torch_c/pytests/run.sh  > /tmp/run.log 2>&1; echo "EXIT=$?" # 1 (§9.2)
```

---

## 13. `mamba` 와 `mixtral` — 실측 20 개 중 마지막 둘

§3 의 표가 남긴 두 아키텍처를 마저 열었습니다. 시작 전에 §3 의 목록부터 다시 쟀는데,
**낡아 있었습니다**: `split_with_sizes.default` 는 다른 작업에서 이미 구현되어
`_aten_implemented()` 에 들어 있었고, 문서만 그 사실을 반영하지 못한 채 남아 있었습니다.
`aten.rs` 를 grep 해서 "미구현 분기 이름" 까지 걸리는 상위집합을 만드는 대신,
`_C._aten_implemented()` 를 빌드된 산출물에서 직접 읽어 진짜 남은 것을 확정했습니다.

```
재확인 (구현 전):
mamba     남음: convolution.default, exp.default, softplus.default, zeros_like.default   (5 -> 4, split_with_sizes 는 이미 구현됨)
mixtral   남음: _grouped_mm.default, clamp_.default, div_.Tensor, empty_like.default,
                floor_divide.default, ge.Scalar, histc.default, index_put_.default,
                masked_fill_.Scalar   (9, 낡지 않음)
```

### 13.1 실제 호출을 먼저 트레이스했다

§1 이 falcon/gptj/bloom/mpt 에 했던 것과 같은 방법 -- `TorchDispatchMode` 로 실제
`transformers` 5.15.1 모델(2 층 · hidden 64 · heads 2)을 한 번 돌려, dtype·shape·인자까지
기록한 뒤 구현했습니다. 커널 doc comment 를 베끼지 않고 상류 torch 2.13.0 으로 재확인하라는
지시를 따른 결과, 세 가지가 트레이스에서만 보였습니다:

- **`floor_divide.default` 의 `other` 가 텐서가 아니라 맨 파이썬 `int` 로 옵니다.**
  `perm // num_top_k` (mixtral 의 그룹드 MoE 라우팅, `transformers` 의
  `integrations/moe.py::grouped_mm_experts_forward`) 가 스키마상 `(Tensor, Tensor)` 인
  `.default` 오버로드를 그대로 호출하면서 스칼라 변환을 커널에 남겨 둡니다.
  `floor_divide.Scalar` 오버로드가 아니라 `.default` 가 선택된다는 것 자체가 스키마만 읽어서는
  못 봤을 사실입니다.
- **`aten.convolution.default` 는 depthwise causal 1-D conv 하나뿐입니다.**
  `groups == in_channels == out_channels == 128`, `padding == kernel_size - 1 == 3`
  (양쪽 다), `transposed=False`. `candle_core::Tensor::conv1d` 가 `groups` 를 그대로
  받아서, 재구현 없이 바로 맞았습니다.
- **`index_put_.default` 는 `inv_perm[perm] = torch.arange(...)` 하나뿐입니다.** 인덱스
  텐서 하나, `accumulate=False`, 전부 1-D -- 이미 있는 `scatter.src` 가 정확히 같은 산수이므로
  (`self[index[i]] = src[i]`, dim=0), 새로 쓰는 대신 그 호출을 조립해 재사용했습니다.

### 13.2 재확인해서 문서 초안과 달라진 지점 셋

1. **`clamp_(None, None)` 은 no-op 이 아니라 거부입니다.** "바꿀 게 없으니 자기 자신을
   반환"이 그럴듯한 추측이지만 틀렸습니다 -- 실측: `tensor.clamp_(None, None)` 은
   `"torch.clamp: At least one of 'min' or 'max' must not be None"` 을 던집니다. 골든
   하니스가 정확히 이 케이스에서 `SILENT DIVERGENCE` 로 잡아냈습니다(구현을 no-op 으로 먼저
   썼다가, 케이스를 쓰면서 재확인하지 않았다면 조용히 남았을 차이).
2. **`histc` 의 `min == max` 규칙은 "둘 다 0" 이 아니라 "값이 같으면"입니다.** `min=5,
   max=5` (0 이 아닌 값)도 데이터의 실제 min/max 로 대체됩니다, 실측. 그리고 데이터 자체가
   상수라 그 자동 범위마저 퇴화하면 (`min==max` 가 다시 성립하면) `[value-1, value+1]` 로
   한 번 더 대체됩니다 -- 같은 폴백을 두 번 적용한 것으로 재현했습니다.
3. **`clamp_` 는 `min > max` 를 거부하지 않습니다.** `min(max(x, min_val), max_val)` 공식을
   조건 없이 적용해서 `min=8, max=2` 가 상수 `2` 로 무너집니다(실측). `candle_core::Tensor::
   clamp` 가 이미 정확히 이 공식(`maximum(min).minimum(max)`)이라, 손으로 다시 구현하지
   않고 그대로 재사용했습니다.

### 13.3 거절한 것

**`_grouped_mm.default` 는 구현하지 않았습니다.** MoE 전용 그룹드 행렬곱으로, 다른 여덟
개와 성격이 다릅니다 -- 오프셋 기반 배치 GEMM 을 요구하고, 이 셰임에 배치 GEMM 커널이 아직
없습니다(`bmm`/`baddbmm` 은 있지만 오프셋 나눗셈은 다른 문제입니다). 지시대로 범위 밖으로
남겼습니다: mixtral 은 `_grouped_mm.default` 하나만 미구현으로 남고, 그 사실이
`_aten_implemented()` 로 바로 드러납니다 -- 숨겨지지 않았습니다.

### 13.4 결과

```
                 before  after
mamba     ops=53     5     0    ← 미구현 0 (split_with_sizes 는 이미 구현되어 있었음)
mixtral   ops=65     9     1    남음: _grouped_mm.default (범위 밖, §13.3)

미구현 0 인 아키텍처: 14 -> 15  (mamba, 실측 20 개 중)
```

| | 이전 | 이후 |
|---|---|---|
| 골든 케이스 | 2268 | **2383** (실패 0, pending 0, exit 0) |
| `_aten_implemented()` | 97 | **109** |
| `_aten_all_implemented()` | 108 | **120** |
| 스키마 | 233/233 | 233/233 (변화 없음) |
| `run.sh` | 113 ok | 113 ok |

**기능 추가:** `exp.default`, `softplus.default`, `convolution.default`, `zeros_like.default`,
`empty_like.default`, `ge.Scalar`, `floor_divide.default`, `histc.default`, `clamp_.default`,
`div_.Tensor`, `masked_fill_.Scalar`, `index_put_.default` -- op 12 개, 골든 케이스 115 개.
**결함 수정:** 없음(이번 작업의 구현 자체에서 났던 `clamp_(None,None)` 오분류는 착지 전에
잡아 고쳤으므로 회귀가 아닙니다). **문서 정정:** §3 의 `mamba`/`mixtral` 낡은 목록에 정정
각주 추가(위). **삭제:** 없음.

**못 한 것:** `_grouped_mm.default` (§13.3, 이유 기록하고 범위 밖으로 남김). `mixtral` 의
그룹드 MoE 를 aten 레벨로 조립해 상류와 대조하는 §4 급 판정(HF 모듈 vs aten 전사 vs 셰임)은
이번 작업에 포함되지 않았습니다 -- 골든 케이스가 op 단위 값 일치를 이미 확인하지만, `mpt`
처럼 끝까지 조립해 로짓을 비교하는 것은 별도 작업입니다.

`compare.py`/`verify_schemas.py` 는 `PYTHONPATH=$PWD/vendor` **없이** 돌립니다.
파이프로 종료 코드를 읽지 마십시오 — 파일로 리다이렉트한 뒤 `$?` 를 읽습니다.
