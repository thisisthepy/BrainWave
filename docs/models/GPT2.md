# GPT-2 — 두 번째 아키텍처, 그리고 남은 꼬리가 고정 목록이 아니라는 증거

`docs/kernels/GAP.md` 가 예고한 4 개(`addmm` · `native_layer_norm` · `split.Tensor` · `tanh`)를
**지금 시점에서 다시 재고**, 넷을 구현하고, 2 층 GPT-2 를 상류와 대조한 기록입니다.

**한 줄 결론.** 넷을 넣자 GPT-2 의 aten 갭이 **0** 이 됐고, 손으로 조립한 2 층 GPT-2 가
greedy · do_sample **19/19 시퀀스에서 상류와 같은 토큰**을 뱉습니다. 그리고 6 개 아키텍처를
같은 방법으로 재 보니 남은 꼬리는 정말로 **아키텍처마다 다른 모양**입니다 — Gemma 는 `gelu`
하나, BERT 는 `gelu`+`gather` 둘이 남고, Llama · Qwen2 · Mistral · GPT-2 는 0 입니다.

기준선 대비: 골든 **1616 → 1781**, ops covered **78 → 82**, 실패 0, pending 0.
스키마 154/154 그대로, 스모크 62 ok, 3 타깃 전부 exit 0.

---

## 0. 먼저 다시 쟀다 — GAP.md 의 4 개는 지금도 정확히 4 개다

`docs/kernels/GAP.md` §5 의 GPT-2 측정은 `_aten_implemented()` 가 **60 개**이던 시점의 것입니다.
그 뒤 Llama 쪽 작업으로 78 개가 됐으므로, 그 목록을 그대로 믿지 않고 **78 개 시점에서 다시**
쟀습니다.

- 방법: 상류 torch 2.13.0 (`/Volumes/macMini/caches/spike-venv/bin/python`, transformers 5.15.1)
  위에서 `TorchDispatchMode` 로 `GPT2LMHeadModel.generate` 를 기록.
- 구현 목록은 **빌드한 산출물에서 직접** 읽었습니다 (`_C._aten_implemented()` /
  `_C._aten_all_implemented()`) — 소스 파싱이 아니라 실제로 배포되는 값입니다.
- 모델: `vocab=100, n_positions=64, n_embd=64, n_layer=2, n_head=2, n_inner=128`,
  `attn_implementation="sdpa"`. GAP.md §5 와 같은 규모입니다.

```
implemented=78  all_implemented=90

GPT-2 greedy      : 42 distinct ops,  구현 안 된 것 4
GPT-2 do_sample   : 51 distinct ops,  구현 안 된 것 4

    aten.addmm.default              x32
    aten.native_layer_norm.default  x20
    aten.split.Tensor               x8
    aten.tanh.default               x8
```

**GAP.md 의 4 개와 정확히 일치합니다.** 더도 덜도 아니었고, greedy 와 do_sample 이 요구하는
집합도 같았습니다 — 샘플링 체인은 이미 `docs/models/SAMPLING.md` 작업이 다 메워 놨기 때문입니다.
(GAP.md 는 60 개 시점에 이미 "GPT-2 sample − (GPT-2 greedy ∪ Llama sample) = 공집합"이라고
적었고, 그 예측이 78 개 시점에서도 그대로 맞았습니다.)

호출 횟수가 정보를 하나 더 줍니다: `addmm` 이 32 회로 압도적입니다. 2 층 × (c_attn, c_proj,
c_fc, c_proj) = 8 회 × 4 스텝(prefill + 3) = 32. **GPT-2 의 모든 선형층이 bias 를 갖기**
때문이고, 그것이 §3 에서 `addmm` 을 다른 셋보다 무겁게 다루는 이유입니다.

---

## 1. 구현한 넷

전부 `_aten_dispatch` 의 `match` 한 자리씩입니다. 우회로는 만들지 않았습니다.

| op | 구현 위치 | 골든 케이스 |
|---|---|---|
| `aten.addmm.default` | `aten.rs` `addmm_default` | 60 |
| `aten.native_layer_norm.default` | `aten.rs` `native_layer_norm_default` | 50 |
| `aten.split.Tensor` | `aten.rs` `split_tensor` | 35 |
| `aten.tanh.default` | `aten.rs` `unary_float` 의 `Unary::Tanh` | 20 |

넷 다 `IMPLEMENTED` 에 올렸고 `IMPLEMENTED_AWAITING_GOLDEN` 은 쓰지 않았습니다.

### 1.1 추측했으면 틀렸을 것들 (전부 실측)

**`addmm` 의 `beta=0` · `alpha=0` 은 곱셈이 아니라 조기 반환이다.**

```
addmm(full(nan), m, n, beta=0)      -> 깨끗한 곱  [1, 2, 8, ...]      (0*nan 이면 nan)
addmm(b, m_with_inf, n, alpha=0)    -> 깨끗한 b   [1, 1, 1, ...]      (0*inf 면 nan)
```

두 케이스가 골든에 그대로 들어가 있습니다. 스칼라 배로 구현했으면 값은 대개 맞고 **비유한
값에서만** 갈렸을 것입니다 — 조용한 발산의 정확한 모양입니다.

**그런데 `self` 는 아무도 안 읽을 때도 검사한다.** `beta=0, alpha=0` 이라 어느 쪽도
`self` 를 안 쓰는데도 모양이 틀리면 거부합니다. 그래서 expand 검사는 무조건 돕니다.

**정수 `addmm` 에서 `beta`/`alpha` 는 결과 dtype 으로 잘린다.**

```
int64, alpha=1.9   == alpha=1      int64, alpha=-1.9  == alpha=-1
int64, beta=0.5    -> self 가 통째로 빠짐 (0.5 -> 0)
```

즉 "0 인가"를 판정하는 것은 파이썬 값이 아니라 **결과 dtype 으로 변환한 값**입니다.

**`native_layer_norm` 의 `mean`/`rstd` 는 평평하지 않다.** 입력의 랭크를 유지하고
정규화된 축만 1 로 바꿉니다 — `(2,3,4)` 에 `[4]` 면 `(2,3,1)`, `[3,4]` 면 `(2,1,1)`.
`(M,)` 로 만들었으면 GPT-2 는 돌았을 것이고(값이 안 쓰임) 골든만 빨개졌을 것입니다.

**`mean`/`rstd` 의 dtype 은 입력이 아니라 *파라미터* 를 따른다.**

| 입력 | weight/bias | out | mean/rstd |
|---|---|---|---|
| float16 | float16 또는 없음 | float16 | **float16** |
| float16 | float32 | float16 | **float32** |
| float32 | float16 | — | 거부 (`mixed dtype (CPU)`) |
| float64 | float32 | — | 거부 (다른 문장) |

셋째 줄과 둘째 줄이 대칭이 아닙니다. 상류의 autocast 경로라 **`float32` 파라미터 + 축소
정밀도 입력만** 허용됩니다. 값으로는 안 보이고 dtype 으로만 보이는 차이라, 이것만 보는
골든 케이스를 따로 뒀습니다.

**분산은 편향 추정(N 으로 나눔)이고, `eps` 는 표준편차가 아니라 분산에 더한다.**
상수 행이 이것을 못 박습니다 — 분산이 0 이므로 `rstd` 가 정확히 `1/sqrt(eps)` 
(`eps=1e-5` 에서 `316.2278`)이고, 순서를 바꾸면 이 값이 안 나옵니다.

**음수 `eps` 는 거부되지 않는다.** NaN 을 줍니다. 그래서 골든의 3 중 비교기는 양쪽이
NaN 이면 일치로 셉니다.

**`split.Tensor` 의 마지막 조각은 짧다, 패딩이 아니라.** `split(arange(10), 3)` 은
`3,3,3,1`. 그리고 `split_size` 가 축보다 크면 **에러가 아니라 조각 하나**입니다.
`split_size == 0` 은 축이 비었을 때만 합법이고, 거부 문장이 음수일 때와 다릅니다.
빈 축은 `split_size` 가 무엇이든 **빈 조각 하나**를 줍니다.

**`tanh` 는 `silu` 가 아니라 `cos`/`sin` 쪽 규칙이다.** `silu(int64)` 는 상류가 거부하는데
`tanh(int64)` 는 `float32` 로 승격합니다(`tanh(bool)` 도). 그래서 `Unary` 계열에 넣었지
`silu` 옆에 두지 않았습니다.

### 1.2 구현하지 않고 이름을 대며 거부한 것 하나

`native_layer_norm(normalized_shape=[0])` — 상류는 `mean=0` 과 `rstd=nan` 을 같이 줍니다.
**같은 축소에 대한 두 통계가 서로 모순**이고(0 개 원소의 평균이 한쪽은 0, 한쪽은 NaN),
관측 하나로 그 내부 불일치를 재현하는 것은 추측입니다. 이름을 대며 거부하고, 골든에
`expect="c_error"` 케이스로 박아 뒀습니다 — 나중에 누가 이유를 알아내면 `match` 로
승격하면 됩니다.

### 1.3 상속한 갭 하나 (새 갭이 아님)

`addmm` 의 곱셈은 candle 의 `matmul` 이고, candle 은 `int64`/`int32`/`int16`/`uint8`/`bfloat16`
에 matmul 커널이 없습니다. **`aten.mm.default` 가 이미 갖고 있는 바로 그 갭**이며
(`docs/design/TORCH_C.md` §2), 골든에 `c_error` 로 그대로 기록했습니다.

다만 **`alpha=0` 일 때는 곱셈이 아예 일어나지 않으므로 같은 dtype 에서도 양쪽이 일치**합니다.
이 대칭 파괴를 골든 케이스로 박아 뒀습니다 — "무엇이 없는가"가 op 이 아니라 **곱셈**이라는
것을 문서가 아니라 테스트가 말하게 하려는 것입니다.

---

## 2. 골든 — 도달한 숫자

```
기준선 (작업 전, 조율 세션 확인값과 일치)
    1616/1616 passed, 0 failed, ops covered=78, pending 0     exit 0

작업 후
    1781/1781 passed, 0 failed, ops covered=82, pending 0     exit 0
```

**+165 케이스, +4 op.** 내역: `addmm` 60, `native_layer_norm` 50, `split.Tensor` 35, `tanh` 20.

| 검사 | 결과 |
|---|---|
| `compare.py` | **exit 0**, 1781/1781, ops covered=82, pending 0 |
| `--inject-fault value` | **exit 1** |
| `--inject-fault shape` | **exit 1** |
| `--inject-fault dtype` | **exit 1** |
| `verify_schemas.py` | **exit 0**, **154/154** (변화 없음) |
| 스모크 (`pytests/run.sh`) | **exit 0**, 62 ok |
| 호스트 빌드 | **exit 0** |
| Android (`aarch64-linux-android`) | **exit 0** |
| iOS (`aarch64-apple-ios`) | **exit 0** |

**스키마가 154 에서 안 움직인 것이 맞습니다.** `overloads.json` 과 `methods.json` 은 이번
작업의 범위가 아니었고 한 줄도 안 고쳤습니다 — 새 op 의 파이썬 스펠링이 아직 없다는 뜻이며,
§5 가 그 목록입니다.

### 2.1 `--inject-fault` 가 못 닿는 곳을 따로 자가검사했다

`compare.py` 의 결함 주입은 **`value_check` 가 붙은 케이스를 일부러 건너뜁니다**
(`_corrupt` 가 `.tolist()` 를 가진 단일 텐서를 가정하므로). 그런데 이번에 추가한 두 비교기는
정확히 그 `value_check` 들입니다 — 즉 **주입 자가검사가 새 비교기를 검증하지 않습니다.**

그래서 두 비교기를 직접 자가검사했습니다 (저장소 밖 스크립트, 가짜 결과 객체 주입):

```
_triple_result_check   값 틀림 / 모양 틀림 / dtype 틀림 / mean 틀림 / rstd dtype 틀림
                       / NaN 한쪽만 / 원소 수 부족          -> 전부 거부
                       양쪽 NaN, 완전 일치                  -> 통과
_chunk_list_check      조각 수 틀림 / 마지막 조각 패딩 / 값 틀림 / dtype 틀림
                                                            -> 전부 거부
```

특히 **"마지막 조각을 짧게 자르지 않고 패딩한" 구현**이 거부되는 것을 확인했습니다. 그것이
`split` 에서 가장 그럴듯한 오구현이고, 원소 단위 비교만 하면 통과해 버립니다.

---

## 3. `addmm` 전환 — 일어났다. 그런데 오차는 안 변했다

`docs/bindings/NN_SURFACE.md` §5 는 bias 있는 `nn.Linear` 를 `matmul` + `add.Tensor` 로 때우면서,
그 분기가 `_aten_all_implemented()` 를 읽으니 **`addmm` 커널이 들어오는 날 저절로 상류
경로로 갈아탄다**고 적었습니다.

### 3.1 전환은 실제로 일어난다

`bootstrap.py` 를 한 글자도 안 고치고 확인했습니다.

```
_HAS_ADDMM = True

Linear(3,4)  bias=True  2-D : ['aten.t.default', 'aten.addmm.default']
Linear(4,6)  bias=True  3-D : ['aten.t.default', 'aten.view.default', 'aten.addmm.default', 'aten.view.default']
Linear(4,6)  bias=True  4-D : ['aten.t.default', 'aten.view.default', 'aten.addmm.default', 'aten.view.default']
Linear(3,4)  bias=False 2-D : ['aten.t.default', 'aten.matmul.default']
```

상류가 같은 입력에서 부르는 것과 대조:

```
상류 2-D bias : ['aten.t.default', 'aten.addmm.default']
상류 N-D bias : ['aten.view.default', 'aten.t.default', 'aten.addmm.default', 'aten.view.default']
```

**같은 op 열입니다** (`t` 와 `view` 의 순서만 다른데, 그것은 bootstrap 이 `wt` 를 먼저 만드는
철자 차이이고 결과에 영향이 없습니다). 즉 §5 의 때움은 은퇴했습니다.

> 측정 방법 주의: 밖에서 `torch._C._aten_dispatch` 를 감싸면 **아무것도 안 잡힙니다.**
> `bootstrap.py` 가 `import torch` 시점에 그 함수 객체를 클로저에 붙잡아 두므로, 모듈 속성을
> 바꿔도 도는 것은 옛 객체입니다. 클로저 셀(`linear.__closure__`)을 바꿔야 잡힙니다.
> 같은 방법으로 `_HAS_ADDMM` 셀을 뒤집어 **한 산출물에서 두 경로를 A/B** 했습니다.

### 3.2 그런데 숫자는 한 비트도 안 변했다

같은 산출물, 같은 입력, `_HAS_ADDMM` 만 뒤집어 상류와 대조했습니다.

| 케이스 | `addmm` 경로 최대 상대오차 | 때움(`matmul`+`add`) 경로 |
|---|---|---|
| `Linear(3,4)` 2-D | 6.498e-08 | 6.498e-08 |
| `Linear(4,6)` 3-D | 2.069e-07 | 2.069e-07 |
| `Linear(4,6)` 4-D | 1.129e-06 | 1.129e-06 |
| `Linear(64,192)` 3-D (GPT-2 `c_attn` 크기) | 5.585e-06 | 5.585e-06 |
| `Linear(256,64)` 3-D | 3.276e-06 | 3.276e-06 |
| `Linear(512,512)` 2-D | 1.650e-04 | 1.650e-04 |

**두 경로의 출력이 원소 단위로 완전히 동일합니다** (`got == prev` → `True`).

이유는 단순합니다. **이 셰임의 `addmm` 도 내부적으로는 candle `matmul` + 별도 브로드캐스트
덧셈**입니다 — candle 에 융합 GEMM 이 없기 때문입니다. 그러니 NN_SURFACE §5 가 지목한
"융합 GEMM 과 누적이 다르다"는 문제는 **사라진 것이 아니라 파이썬 층에서 커널 안으로
옮겨간 것**입니다.

**즉 NN_SURFACE §5 의 기대를 반만 충족합니다.** 디스패치 키는 상류와 같아졌고(그래서
`_aten_implemented()` 가 정직해졌고 골든이 걸립니다), 누적 순서는 여전히 상류와 다릅니다.
이걸 진짜로 닫으려면 candle 의 GEMM 자리에 융합 커널을 넣어야 하고, 그건 이 작업의 범위가
아닙니다.

### 3.3 `1.65e-04` 는 겁먹을 숫자가 아니다 — 다만 새 숫자다

`Linear(512,512)` 의 `1.650e-04` 는 **상대**오차이고, 값이 `4.967e-03` 인 원소에서 나옵니다
(절대오차 `1.526e-05`). 텐서의 최대 크기 `43.46` 로 정규화하면 **`3.5e-07`** — 평범한
float32 GEMM 반올림입니다. 상쇄가 일어난 원소의 상대오차를 그대로 읽으면 안 됩니다.

다만 NN_SURFACE §7 이 잰 `1.4e-07` 은 **3×4 짜리**였고, §10 이 "큰 행렬에서 얼마나
벌어지는지는 안 쟀다"고 남긴 항목입니다. 그 빈칸을 이 표가 채웁니다:
**축소 차원이 512 로 늘면 절대오차는 `1.5e-05` 까지 간다.** float32 골든 허용오차(`1e-5`)를
넘는 크기이므로, 이 크기의 층을 골든으로 잡으려면 허용오차를 크기에 따라 정해야 합니다.
지금 골든은 이 크기를 안 다룹니다.

---

## 4. 진짜 판정 — 2 층 GPT-2 가 상류와 같은 토큰을 뱉는가

`transformers` 는 셰임 위에서 아직 임포트되지 않으므로(`torch.distributed.Store`),
`docs/models/SAMPLING.md` §3 과 같은 방법을 썼습니다: **같은 산술을 한 파일에 써서 상류와 셰임에서
각각 돌립니다.** 아키텍처는 HF `GPT2LMHeadModel` 을 op 단위로 옮긴 것입니다 —
`Conv1D`(=`addmm`), `LayerNorm`(=`native_layer_norm`), QKV 하나를 셋으로 쪼개는
`split.Tensor`, `gelu_new` 의 tanh 근사, causal `sdpa`, 묶인 `lm_head`. 생성 쪽은
transformers 5.15.1 의 `TemperatureLogitsWarper` → `TopKLogitsWarper` → `TopPLogitsWarper`
→ `multinomial` 을 그대로 옮겼습니다. 가중치는 결정적 LCG 로 채워 양쪽이 같은 비트를 받습니다.

규모: `vocab=100, d=64, head=2, layer=2, n_inner=256, seq=6`.

```
순전파 로짓 (n=600)                    최대 절대오차 4.098e-08
"크게 부풀린" 모델 순전파 (n=600)       최대 절대오차 3.338e-06
```

**토큰:**

```
greedy 6 토큰                     torch=[67,67,67,67,67,67]              shim=동일
greedy 8 토큰 (부풀린 모델)        torch=[14,14,14,14,49,14,66,49]        shim=동일
모든 위치의 argmax                 torch=[67,67,14,67,67,67]              shim=동일
모든 위치의 argmax (부풀린 모델)    torch=[49,49,49,66,14,14]              shim=동일

do_sample (매 스텝 재시딩, 5 구성) ─ 전부 동일
do_sample (한 번만 시딩하고 흘려보냄, 7 구성) ─ 전부 동일
    T=1.0 k=50  p=0.95 seed=0     [41, 15, 51, 76,  2,  4]
    T=1.0 k=50  p=0.95 seed=1     [14, 89, 71, 30, 49, 10]
    T=1.0 k=50  p=0.95 seed=1234  [11, 59, 44, 25, 14, 79]
    T=0.7 k=20  p=0.90 seed=0     [76, 24, 71, 76, 99, 99]
    T=0.7 k=20  p=0.90 seed=1234  [52, 23,  9, 75, 14, 35]
    T=1.3 k=100 p=1.00 seed=0     [58, 68, 51, 76, 36,  4]
    T=1.3 k=100 p=1.00 seed=1234  [25, 59, 44,  7, 14, 79]
    부풀린 모델 seed=0             [52, 24, 57, 76, 66,  4, 49, 12]
    부풀린 모델 seed=7             [14, 23, 66, 92, 41, 66, 76, 66]
```

**19/19 시퀀스 일치, 불일치 0.** "한 번만 시딩하고 흘려보내는" 구성이 더 강한 판정입니다 —
매 스텝의 난수 **소비량**까지 같아야 6~8 스텝 뒤가 맞습니다.

### 4.1 greedy 증거의 한계를 적어 둔다

처음 쓴 가중치 생성기(sin/cos)는 `wte` 의 모든 행을 거의 평행하게 만들어, **모델이 모든
위치에서 같은 토큰을 argmax 로 내는 고정점**이 됐습니다. 양쪽이 똑같이 그랬으므로 일치는
일치지만, "6 개 토큰이 전부 같은 값"인 일치는 증거로 약합니다. 그래서 백색잡음(LCG)으로
바꿔 greedy 가 실제로 움직이는 케이스(`[14,14,14,14,49,14,66,49]`)를 만들었습니다.
그래도 **greedy 가 do_sample 보다 약한 증거라는 점은 그대로입니다** — 학습되지 않은 모델의
argmax 는 소수의 토큰에 몰립니다. 강한 판정은 샘플링 쪽과 로짓 오차 쪽입니다.

---

## 5. 커널은 있는데 파이썬 스펠링이 없다 — 다른 작업으로 넘길 목록

`overloads.json` · `methods.json` · `bootstrap.py` 는 **이 작업의 범위 밖**이라(메인 트리
에이전트 소유) 한 줄도 안 고쳤습니다. 그래서 넷 다 `torch.ops.aten.*` 로는 도달하지만
자연스러운 파이썬 철자로는 아직 못 갑니다. §4 의 판정을 aten 레벨로 한 이유가 이것입니다.

셰임 위에서 실측:

| 철자 | 결과 | 막는 것 |
|---|---|---|
| `torch.ops.aten.addmm.default` | **OK** | — |
| `torch.ops.aten.native_layer_norm.default` | **OK** | — |
| `torch.ops.aten.split.Tensor` | **OK** | — |
| `torch.ops.aten.tanh.default` | **OK** | — |
| `nn.Linear(bias=True)` | **OK** | — (`_C._nn.linear` 가 §3 의 경로로 내려감) |
| `torch.addmm(...)` | ~~실패~~ **정정: OK** | ~~`overloads.json` 에 `addmm` 없음~~ 지금은 있음 |
| `torch.tanh(x)` | ~~실패~~ **정정: OK** | ~~`overloads.json` 에 `tanh` 없음~~ 지금은 있음 |
| `x.tanh()` | ~~실패~~ **정정: OK** | ~~`methods.json` 에 `tanh` 없음~~ 지금은 있음 |
| `torch.split(x, n, d)` / `x.split(n, d)` | ~~실패~~ **정정: OK** | ~~`overloads.json` 에 `split` 없음~~ 지금은 있음 |
| `F.layer_norm(...)` | ~~실패~~ **정정: OK** | ~~**`_C._get_cudnn_enabled`**~~ `bootstrap.py:7071` 이 답을 답함(위 §5 텍스트와 별개로 이미 구현돼 있었던 것으로 보임 — 실측만 갱신) |
| `nn.LayerNorm(...)` | ~~실패~~ **정정: OK** | ~~**`TensorBase.zero_`**~~ `methods.json` 에 `zero_` 가 생겨 닫힘 — `docs/architectures/ARCH.md` 감사(이 라운드)가 같은 것을 발견 |
| `torch.layer_norm` / `torch.native_layer_norm` | ~~실패~~ **정정: OK** | `torch.layer_norm` 은 `overloads.json` 항목이 아니라 `bootstrap.py:5863` 의 파이썬 컴포지트(`aten.native_layer_norm.default` 를 직접 호출)로 배선됨 |

> **Correction (문서 감사, 2026-09):** 위 표의 "실패" 여섯 줄 전부 오늘은 성공합니다(실측
> 재확인, `nn.LayerNorm(4)` 생성 포함). `overloads.json`/`methods.json` 이 이 문서를 쓴 뒤
> 다른 라운드들에서 계속 채워진 결과로 보입니다 — `docs/models/SAMPLING.md`/`docs/devices/DEVICE.md`/
> `docs/architectures/ARCH.md` 감사(이 라운드)에서 이미 반복해서 발견한 것과 같은 패턴입니다.
> <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json addmm present -->
> <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json tanh present -->
> <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json split present -->
> <!-- DOCWATCH: json-key rust/torch_c/src/methods.json tanh present -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py layer_norm present -->

**이번 작업이 만든 새 빚이 아닙니다.** 같은 상태인 것이 이미 여럿 있었습니다(이 문서를 쓴
시점 기준) — `F.softmax` / `x.softmax` (`TensorBase.softmax` 없음), `torch.multinomial`,
`torch.topk`, `torch.sort`, `torch.cumsum`, `x.masked_fill` (`TensorBase.__gt__` 없음) 이 전부
커널은 있는데 철자가 없는 상태로 실측됐습니다. `docs/models/SAMPLING.md` 가 판정을 aten 레벨로 한
것도 같은 이유였습니다 — 다만 `docs/models/SAMPLING.md` 자신의 감사(이 라운드)가 이미 확인했듯
`multinomial`/`topk`/`sort` 도 지금은 파이썬 철자로 닿습니다.

**`nn.LayerNorm` 은 두 겹이었습니다.** `_get_cudnn_enabled` 를 답해 줘도 `TensorBase.zero_` 에서
먼저 죽었습니다(`reset_parameters` 가 `init.zeros_(self.bias)` 를 부름). NN_SURFACE §10 이
"`_get_cudnn_enabled` 는 답하기 싼 설정 게터로 보인다"고 남긴 항목은 맞았지만, 그것만으로는
`nn.LayerNorm` 이 열리지 않았습니다 — 지금은 둘 다 닫혀 있습니다.

---

## 6. 남은 꼬리는 고정 목록이 아니다 — 6 개 아키텍처 실측

이 작업의 전제("남은 것이 아키텍처마다 다르다")를 GPT-2 하나로 확인하고 끝내지 않고,
같은 방법으로 여섯 개를 쟀습니다. 전부 2 층, hidden 64 급, greedy ∪ do_sample 합집합.

```
all_implemented = 94   (IMPLEMENTED 82 + AWAITING_GOLDEN 12)

llama    56 ops   미구현 0
gpt2     52 ops   미구현 0
qwen2    51 ops   미구현 0
mistral  56 ops   미구현 0
gemma    56 ops   미구현 1   aten.gelu.default
bert     14 ops   미구현 2   aten.gelu.default, aten.gather.default
```

(`bert` 는 `generate` 가 없어 순전파만이라 op 수가 작습니다.)

**어떤 아키텍처에만 있는 op:**

```
gpt2 에만 : aten.split.Tensor
bert 에만 : aten.gather.default
나머지 넷 : 없음
```

**넷을 누가 부르는가:**

| op | 부르는 아키텍처 |
|---|---|
| `aten.addmm.default` | gpt2, **qwen2**, bert |
| `aten.native_layer_norm.default` | gpt2, bert |
| `aten.split.Tensor` | gpt2 |
| `aten.tanh.default` | gpt2, bert |
| `aten.gelu.default` (미구현) | gemma, bert |
| `aten.gather.default` (미구현) | bert |

여기서 두 가지가 나옵니다.

**(1) GAP.md 의 주장이 확인됐다.** 꼬리는 고정 목록이 아닙니다. GPT-2 를 열려면 `split.Tensor`
가 필요한데 다른 다섯은 아무도 안 부르고, 반대로 Gemma 를 열려면 `gelu` 가 필요한데 GPT-2 는
안 부릅니다.

**(2) 그런데 겹침이 예상보다 크다.** `addmm` 하나가 GPT-2 · Qwen2 · BERT 셋을 동시에
건드립니다. 측정한 op 집합에서 역산하면 **이번 작업 전 Qwen2 는 `addmm` 하나만 부족한
상태**였고(다른 셋은 안 부름), 그래서 이 작업이 GPT-2 뿐 아니라 **Qwen2 도 같이 열었습니다.**
"아키텍처마다 다르다"는 "매번 처음부터"라는 뜻은 아닙니다 — bias 있는 선형층처럼 널리
공유되는 것이 있고, `split` 처럼 한 집안 것이 있습니다.

**다음 작업 후보는 `aten.gelu.default` 하나입니다.** 그것 하나로 Gemma 가 열리고 BERT 는
`gather` 하나만 남습니다.

---

## 7. 확인하지 않은 것 / 모르는 것

- **`float16` layer norm 의 `rstd` 가 왜 `f32` 계산의 반올림과 다른지 모릅니다.**
  측정값 `6.26171875` 는 `f32` 로 계산해 narrow 한 `6.2578125` 가 아닙니다(상대차 6.2e-4).
  상류의 벡터화 커널이 어떤 순서로 누적하는지 재지 않았고, `float16` 골든 허용오차 `5e-3`
  안이라 통과합니다. **허용오차를 조이면 이 케이스가 먼저 빨개집니다.**
- **`split` 의 aliasing 을 재현하지 않았습니다.** 상류의 조각은 뷰라서 조각에 쓰면 원본이
  바뀝니다(실측). candle 의 `narrow` 도 뷰지만, 이 셰임의 `TensorBase` 를 통한 쓰기가
  원본에 닿는지는 **안 쟀습니다** — `aten.slice.Tensor` 가 이미 갖고 있는 같은 미해결
  질문이고, 이번 작업이 답하지 않았습니다.
- **`native_layer_norm(normalized_shape=[0])`** 은 §1.2 대로 거부합니다. 상류가 왜
  `mean=0` 과 `rstd=nan` 을 같이 주는지 모릅니다.
- **`addmm` 의 `beta`/`alpha` 에 0-d 텐서를 넣으면** 상류는 파서 단계에서 거부하는데
  (`Cannot cast tensor(2.) to number`) 이 셰임의 `scalar_arg` 는 받습니다. 실제 경로는
  `bootstrap.py` 의 오버로드 해석기를 먼저 지나므로 모델에서는 안 걸리고, 골든은
  `_aten_dispatch` 를 직접 부르므로 이 차이를 안 봅니다. **고치지 않았습니다.**
- **`uint8` 에 음수 리터럴을 넣으면** `torch.tensor` 는 255 로 감고
  `_C._tensor_from_flat` 은 0 으로 포화합니다. `tanh` 와 무관한 **생성자 차이**이며
  (`tanh([0,1,255,2])` 는 양쪽 일치), 이번에 발견했지만 범위 밖이라 안 고치고 골든의
  `uint8` 입력을 음이 아닌 값으로 두었습니다.
- **큰 층의 오차**는 §3.3 이 `512×512` 까지만 쟀습니다. 실제 모델 크기(GPT-2 small 의
  `768×2304`)는 안 쟀습니다.
- **기기(Android/iOS) 에서의 임포트**는 이번에도 **링크만** 확인했습니다.
- 여섯 아키텍처는 전부 **2 층 · hidden 64** 입니다. 층을 늘리거나 다른 설정
  (sliding window, MoE, GQA 비율)을 켜면 더 나올 수 있습니다.
