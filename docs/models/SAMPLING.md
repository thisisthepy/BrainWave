# SAMPLING — `do_sample=True` 를 막던 8 개, 그리고 같은 시드에서 같은 토큰이 나오는가

`docs/kernels/GAP.md` §4 가 열 개를 예측했고, 조율 세션이 실제 transformers Llama 를 `TorchDispatchMode`
로 다시 재어 `_aten_implemented()` 대비 **여덟 개**가 남았다고 확정했습니다. 이 문서는 그 여덟 개를
구현하면서 **측정한 것**과, 마지막에 남는 질문 하나에 대한 답입니다:

> 상류와 **같은 시드**로 `do_sample=True` 샘플링을 돌리면 **같은 토큰**이 나오는가?

**나옵니다.** 15 개 구성 × 6 스텝 = **90 개 토큰이 전부 일치**했고, 그중 6 개 구성은 시드를 한 번만
주고 스트림을 끝까지 흘려보낸 것입니다 — 즉 두 쪽이 **소비하는 난수 워드 수까지 같습니다**(§3).

여덟 개 중 일곱은 평범한 커널이고, 하나(`multinomial`)만 난수를 뽑습니다. 그 하나가 이 작업의
전부였습니다. 나머지 일곱에서도 **추측했으면 틀렸을 것 세 가지**가 나왔고(§4), 맞추지 못한 것
하나를 명시적으로 남깁니다(§4.2).

측정 환경: `/Volumes/macMini/caches/spike-venv/bin/python` (torch 2.13.0, CPython 3.13.0,
Apple Silicon / darwin 25.5.0, aarch64). 실험 스크립트는 `/Volumes/macMini/caches/bw-sample-probe/`
에 있습니다(커밋 대상 아님).

---

## 0. 도달한 숫자 — 전부 종료 코드와 함께

```
골든 대조        1616/1616 통과, 0 실패, ops covered=78, pending case builders=0   EXIT=0
inject-fault     value / shape / dtype                                             각각 EXIT=1
스키마 대조      127/127 (overloads 47 + methods 80)                                EXIT=0
스모크           rust/torch_c/pytests/test_shim.py 전체                             EXIT=0
호스트 빌드      aarch64-apple-darwin                                               EXIT=0
Android 빌드     aarch64-linux-android  lib_C.so   3,960,224 B                      EXIT=0
iOS 빌드         aarch64-apple-ios      lib_C.dylib 3,112,632 B                      EXIT=0
```

`ops covered` 는 70 → **78**, `pending case builders` 는 0 → **0**. 스키마 숫자는 바뀌지 않았습니다 —
이 작업은 `overloads.json` 과 `methods.json` 을 건드리지 않았습니다(소유자가 다릅니다). 그 결과
새 여덟 개는 `_aten_dispatch` 로는 닿지만 `torch.<op>` 철자로는 아직 해석되지 않습니다(§6).

케이스 수가 크게 는 것(1616)은 여덟 개 때문만이 아닙니다. `uniform_`/`normal_` 의 검사를 승격하면서
`normal_` 의 경로 A/B 경계(n = 15/16/17/20/32)와 `uniform_` 의 비-2의거듭제곱 범위를 케이스로
깔았기 때문입니다(§5).

---

## 1. 여덟 개가 무엇이었나

| op | 어디서 부르는가 | 성격 |
|---|---|---|
| `aten._softmax.default` | 로짓 → 확률, 그리고 `TopPLogitsWarper` 안에서 한 번 더 | 커널 |
| `aten.sort.default` | `TopPLogitsWarper` — 어휘 전체를 정렬 | 커널 (쌍 반환) |
| `aten.topk.default` | `TopKLogitsWarper` — k 번째 로짓을 임계값으로 | 커널 (쌍 반환) |
| `aten.le.Scalar` | `TopPLogitsWarper` — `cumulative_probs <= (1 - top_p)` | 커널 |
| `aten.scatter.src` | `TopPLogitsWarper` — 정렬 순서의 마스크를 원래 순서로 되돌림 | 커널 |
| `aten.squeeze.dim` | `_sample` — `next_tokens.squeeze(1)` | 뷰 |
| `aten.fill_.Tensor` | 생성 루프의 슬라이스 대입 | 제자리 |
| `aten.multinomial.default` | `_sample` — 유일하게 **뽑는** 것 | 난수 |

`aten.fill_.Tensor` 는 사실 이미 구현되어 있었고 `IMPLEMENTED_AWAITING_GOLDEN` 에 주차되어
있었습니다. 케이스 빌더 하나와 줄 하나를 옮기는 것이 전부였습니다 — 그 상수의 주석이 예고한 그대로입니다.

---

## 2. `multinomial` — 상류를 읽지 않았으면 전부 틀렸을 것

### 2.1 알고리즘이 둘이고, 분기가 인자 이름이 시사하는 곳이 아니다

`torch/include` 에는 `MultinomialKernel.cpp` 가 없습니다(헤더만 배포됩니다). 그래서 **소스를 읽는
대신 스트림을 셌습니다.** 방법: `multinomial` 을 부른 뒤 float64 균일난수를 하나 뽑고, 같은 시드에서
float32 균일난수 `k` 개를 소비한 뒤 뽑은 값과 일치하는 `k` 를 찾습니다. `uniform_` 의 소비량이
dtype 당 1 워드/2 워드로 확정되어 있으므로(`docs/numerics/RNG.md` §1.2) 이것은 **정확한 워드 카운터**입니다.

```
n_cat=  5  n_sample=1  replacement=False  ->  10 워드
n_cat=  5  n_sample=1  replacement=True   ->  10 워드      <-- replacement 와 무관
n_cat=  5  n_sample=3  replacement=False  ->  10 워드      <-- n_sample 과 무관
n_cat=  5  n_sample=3  replacement=True   ->   6 워드      <-- 여기서만 다르다
n_cat=100  n_sample=1  replacement=False  -> 200 워드
n_cat=100  n_sample=1  replacement=True   -> 200 워드
n_cat=100  n_sample=3  replacement=True   ->   6 워드
2D (3,7)   n_sample=1  replacement=True   ->  42 워드  = 2 × numel
2D (3,7)   n_sample=2  replacement=True   ->  12 워드  = 2 × n_dist × n_sample
```

읽는 법: `2 × numel` 은 **텐서 전체에 대해 f64 균일난수를 하나씩** 뽑았다는 뜻이고,
`2 × n_dist × n_sample` 은 **표본마다 하나씩** 뽑았다는 뜻입니다. 즉 알고리즘이 둘입니다.

그리고 분기는 `replacement` 가 아닙니다. **`!replacement || n_sample == 1`** 입니다 —
`replacement=True, n_sample=1` 이 10 워드를 쓰는 것이 그 직접 증거입니다.

**이것이 이 작업에서 가장 중요한 사실입니다.** `GenerationMixin._sample` 이 부르는 것은
`torch.multinomial(probs, num_samples=1)` 이고, `replacement` 는 기본값 False 입니다. 즉 실제
샘플링은 **누적합 + 이분탐색을 한 번도 지나가지 않습니다.** `multinomial` 이라는 이름과 문서가
설명하는 알고리즘을 그대로 구현했다면, 이 문서의 마지막 절은 "토큰이 다릅니다" 로 끝났을 것입니다.

### 2.2 빠른 경로는 Gumbel 이다

워드 카운트가 가리키는 것을 순수 파이썬 MT19937 포팅(`rng.rs` 와 같은 엔진)으로 재현했습니다:

```
q = empty_like(self).exponential_(1)     # 원소당 f64 균일난수 1 개 = 2 워드
q = self / q                              # at::div_out — self 의 dtype 으로 저장
n_sample == 1 : argmax(q, dim=-1, keepdim=True)
n_sample >  1 : topk(q, n_sample).indices
```

`exponential_` 은 `cpu_serial_kernel` 이고 dtype 과 무관하게 `exponential_distribution<double>` 을
인스턴스화합니다(`DistributionTemplates.h:336-343`) — `normal_` 의 경로 B 와 같은 구조입니다.
변환은 `TransformationHelper.h:129-145` 이고, **CPU 갈래는 `-1/lambda * log1p(-val)`** 입니다.
CUDA 갈래의 클램프를 옮겨오거나 `log(1-val)` 로 쓰면 15 자리까지 같고 마지막 비트에서 갈립니다 —
두 범주가 가까울 때 토큰을 바꾸기에 충분한 양입니다.

느린 경로(`replacement=True && n_sample > 1`)는 문서가 설명하는 그것입니다: 누적합, 마지막 버킷을
정확히 1 로 덮어쓰기, 표본당 f64 균일난수 하나, 좌편향 이분탐색.

**대조**: `(3, 11)` 분포, 시드 6 개, 두 경로 모두 — **12/12 인덱스 목록 완전 일치**, 허용오차 없음.

### 2.3 중간값은 전부 입력 dtype 으로 좁혀진다 — **단, 느린 경로는 아니다**

빠른 경로에서 상류는 `q = empty_like(self)` 와 `at::div_out(q, self, q)` 를 씁니다. 둘 다 **입력의
dtype 을 가진 진짜 텐서**이므로, 지수난수도 나눗셈 결과도 저장 시점에 좁혀집니다. f64 로 끝까지
계산하고 마지막에 한 번 좁히면 **두 범주가 1 ulp 안에 있을 때 argmax 가 달라집니다** — 샘플러의
선택이 결정되는 바로 그 자리입니다.

느린 경로는 반대였고, 이것이 **추측이 틀린 두 번째 지점**입니다. 주변 코드의 결에 따라
`scalar_t sum = 0; sum += val;` 을 그대로 옮겨 bf16 누적으로 구현했더니 상류와 갈렸습니다 —
측정한 bf16 추첨 140 회 중 2 회에서 다른 버킷.

**측정 방법** (이것이 근거입니다): 알려진 MT 스트림에서 20,000 회 추첨하고, 각 추첨의 균일난수
`u` 와 상류가 고른 버킷 `k` 로부터 상류의 열한 개 경계 전부를 협착합니다 (`pick <= k` 면
`cum[k] >= u`, 아니면 `cum[k] < u`). 협착 구간의 폭은 약 `2.25e-4` 로, 그 자리 `bfloat16` 간격의
약 1/5 입니다:

```
 k      lo(torch)      hi(torch)   bf16 누적판   float 누적판(=정확값)
 4   0.4464305668   0.4468499724  0.4472656250  0.4468085106   <- bf16 은 구간 밖
 5   0.5104044083   0.5107562153  0.5117187500  0.5106382979   <- 구간 밖
 8   0.8292939420   0.8302481975  0.8281250000  0.8297872340   <- 구간 밖
```

`bfloat16` 누적판은 열한 개 중 여섯 개가 구간 밖으로 나가고, `float` 누적판은 `float16`·`bfloat16`
양쪽에서 **열한 개 전부 구간 안**에 들어갑니다. 즉 느린 경로의 누적 타입은
`at::acc_type<scalar_t, false>` — f16/bf16/f32 는 float, f64 는 double — 이고 **어디서도 좁히지
않습니다.** 나머지 커널들이 `opmath_type` 을 쓰는 것과 같고, 빠른 경로가 예외인 이유는 그쪽이
진짜 텐서를 만들기 때문입니다.

두 사실은 **반대 방향**이라 하나만 맞추면 다른 하나가 틀립니다. 그래서 둘 다 적어 둡니다.

---

## 3. 진짜 판정 — 같은 시드에서 같은 토큰이 나오는가

`transformers` 는 아직 임포트되지 않으므로(`docs/kernels/OPS8.md` §5-3, `torch.distributed`),
`docs/kernels/OPS8.md` §4.1 이 greedy 에 한 것과 같은 방식으로 **같은 산술을 `torch.ops.aten.*` 만으로 손으로
써서** 양쪽에서 돌렸습니다. 2 층, hidden 64, head 2, intermediate 128, vocab 100, seq 4 —
RMSNorm · RoPE · SwiGLU · sdpa · `x @ w.t()`. 생성 쪽은 transformers 5.15.1 의
`TopKLogitsWarper`·`TopPLogitsWarper`·`_sample` 을 op 단위로 옮겼으므로, 실제 `generate` 가 내는
op 열과 같습니다.

**시드를 매 추첨 직전에 다시 주는 구성** (9 개):

```
T=1.0 top_k=50  top_p=0.95 seed=0     torch=[58, 63, 41, 78, 83, 78]  shim=동일   max|logit diff|=2.503e-06
T=1.0 top_k=50  top_p=0.95 seed=1     torch=[63, 41, 85, 13, 47, 91]  shim=동일   max|logit diff|=2.384e-06
T=1.0 top_k=50  top_p=0.95 seed=1234  torch=[ 1, 43, 50, 33,  9, 50]  shim=동일   max|logit diff|=3.695e-06
T=0.7 top_k=20  top_p=0.90 seed=0     torch=[58, 63, 41, 78, 83, 47]  shim=동일
T=0.7 top_k=20  top_p=0.90 seed=1     torch=[63, 41, 85, 13, 47, 91]  shim=동일
T=0.7 top_k=20  top_p=0.90 seed=1234  torch=[ 1, 43, 50, 33, 48, 91]  shim=동일
T=1.3 top_k=100 top_p=1.00 seed=0     torch=[58, 80, 84, 78, 83, 47]  shim=동일
T=1.3 top_k=100 top_p=1.00 seed=1     torch=[80, 84, 78, 13, 78, 91]  shim=동일
T=1.3 top_k=100 top_p=1.00 seed=1234  torch=[25, 12, 50, 32, 70, 91]  shim=동일
```

**시드를 한 번만 주고 스트림을 끝까지 흘려보낸 구성** (6 개) — 이쪽이 더 강한 판정입니다.
매 스텝 재시딩하면 "같은 시작점에서 같은 값" 만 보이지만, 흘려보내면 **매 스텝의 소비량까지 같아야**
6 스텝 뒤가 일치합니다:

```
T=1.0 top_k=50 top_p=0.95 seed=0     torch=[58, 68, 51,  4,  2, 69]  shim=동일
T=1.0 top_k=50 top_p=0.95 seed=1     torch=[63, 89, 24, 30, 44, 10]  shim=동일
T=1.0 top_k=50 top_p=0.95 seed=1234  torch=[ 1, 25, 98, 48, 14, 33]  shim=동일
T=0.7 top_k=20 top_p=0.90 seed=0     torch=[58, 63, 30, 63, 63, 24]  shim=동일
T=0.7 top_k=20 top_p=0.90 seed=1     torch=[63, 45, 24, 90, 25, 48]  shim=동일
T=0.7 top_k=20 top_p=0.90 seed=1234  torch=[ 1, 25, 48, 91, 29, 95]  shim=동일
```

**15/15 구성, 90/90 토큰 일치.** 로짓의 최대 절대오차는 `3.7e-06` 으로, 2 층 12 회 행렬곱을
누적한 뒤의 값이고 `float32` 골든 허용오차(`1e-5`) 안입니다.

그리고 워드 소비량을 shim 쪽에서도 같은 방법으로 직접 재어 대조했습니다:

```
n_cat  n_sample  replacement   torch words   shim words   match
    5         1        False            10           10    YES
    5         3         True             6            6    YES
    8         1         True            16           16    YES
  100         1        False           200          200    YES
  100         3         True             6            6    YES
                                             (15/15 조합 일치)
```

**따라서 `multinomial` 은 상류와 같은 생성기에서 같은 수열을 같은 양만큼 소비합니다.**

---

## 4. 나머지 일곱에서 나온 것

### 4.1 추측했으면 틀렸을 것 셋

**(a) `squeeze.dim` 은 크기가 1 이 아닌 축에 대해 no-op 이지 에러가 아니다.** `(1,3,1,2)` 에
`squeeze(1)` 은 `(1,3,1,2)` 를 그대로 돌려줍니다(실측). 거부하도록 만들었다면 배치가 1 행일 때
생성 루프가 그 자리에서 멈춥니다.

**(b) `_softmax` 의 `half_to_float=True` 는 CPU 에서 dtype 과 무관하게 전부 거부된다.**
`float16`·`bfloat16`·`float32` 셋 다 `RuntimeError: softmax with half to float conversion is not
supported on CPU` 입니다 — CUDA 전용 융합입니다. 조용히 들어주면 상류가 예외를 내는 자리에서
`float32` 를 반환하게 됩니다. 정수 입력은 `RuntimeError` 가 아니라 **`NotImplementedError`** 이고
커널 이름을 답니다.

**(c) `scatter.src` 의 모양 규칙은 candle 의 것보다 느슨하다.** torch 는 `index` 가 `src` 보다,
그리고 (스캐터 축을 뺀 곳에서) `self` 보다 **크지만 않으면** 됩니다. candle 의 `Tensor::scatter`
는 `index.dims() == src.dims()` 와 `self.dims() == src.dims()` 를 요구합니다. `TopPLogitsWarper`
가 쓰는 모양이 정확히 candle 이 거부하는 그것이라, candle 의 검사를 빌려 왔다면 정작 중요한
호출이 막힙니다. 그래서 손으로 썼습니다. 부수적으로 `index` 의 dtype 은 `int32` 도 받습니다
(`masked_fill` 의 마스크가 정확히 `bool` 이어야 하는 것과 대조적입니다).

덤: **`sort` 의 NaN 은 "가장 큰 값"** 입니다. 오름차순에서 맨 뒤, 내림차순에서 맨 앞이고
`topk(largest=True)` 는 `+inf` 보다 먼저 집습니다. IEEE 는 NaN 과의 모든 비교가 거짓이라고 하므로
이것은 torch 의 **선택**이지 상속되는 성질이 아닙니다. `partial_cmp().unwrap()` 으로 썼다면 첫
NaN 에서 패닉합니다.

### 4.2 못 한 것 — `topk` 의 동점 순서

**`sort` 는 안정적이고, 그것은 측정입니다.** `[3,1,3,1,2,3]` 을 내림차순 정렬하면 인덱스가
`[0,2,5,4,1,3]` 입니다 — 오름차순의 역순이 아니라 동점끼리 원래 순서를 지킨 것입니다. 80 개가
두 값으로만 이루어진 텐서는 `0..79` 로 돌아옵니다. 그래서 동점을 **정확히** 대조합니다.

**`topk` 는 다릅니다.** 같은 입력에서 `k=3` 은 안정 정렬과 일치하지만(`[0,2,5]`, 이 shim 도 일치),
`k=6` 은 상류가 `[0,2,5,4,3,1]` 을 냅니다 — 동점인 두 1.0 이 뒤집힙니다. 이 shim 은
`[0,2,5,4,1,3]` 을 냅니다.

`sorted=False` 는 같은 상황이 한 단계 더 간 것입니다. 8 원소 텐서의 `k=3` 에서 상류는
`sorted=True` 의 `[6,7,0]` 대신 `[7,6,0]` 을 냅니다 — 분할(partition)의 잔여물입니다.

**억지로 맞추지 않았습니다.** `torch.topk` 는 동점 순서에 대해서도 `sorted=False` 의 순서에 대해서도
아무것도 약속하지 않고, 재현하려면 분할 알고리즘을 통째로 옮겨야 합니다. 대신:

- 이 shim 은 **항상 정렬된 순서**를 냅니다 (그 약속 안에 있는 답입니다).
- 골든 하네스에서 **인덱스를 대조하는 케이스는 전부 동점 없는 입력**을 씁니다.
- 동점 케이스와 `sorted=False` 케이스는 `_topk_multiset_check` 로 **(값, 인덱스) 쌍의 다중집합**을
  대조합니다 — 순서는 안 보고 **고른 원소는 봅니다.** 순서를 고정하면 상류가 약속하지 않은 것을
  고정하는 것이고, 검사를 없애면 **틀린 원소**를 골라도 통과합니다.

**측정된 영향 범위: 없음.** `TopKLogitsWarper` 는 `values[..., -1]` 만 읽고 값은 동점에서도 같습니다.
`multinomial` 의 무복원 경로가 `topk` 에 넘기는 것은 연속 난수의 비율이라 동점이 나오지 않습니다.
§3 의 90 토큰이 그것을 보여 줍니다. 그래도 **모른다고 적어 둡니다** — 로짓이 정확히 같은 값을
갖는 모델에서 `top_k` 의 컷이 갈릴 수 있고, 그 경우는 확인하지 않았습니다.

### 4.3 남은 수치 차이 하나 — 축소 dtype 의 비-마지막 축 `_softmax`

`float16`/`bfloat16` 입력에 대해 **마지막 축이 아닌** 축으로 softmax 를 걸면 1~2 ulp 갈립니다:

```
float16  dim=0   torch 0.1192626953125    shim 0.11920166015625   (정확값 0.11920292)
bfloat16 dim=0   torch 0.267578125        shim 0.26953125
```

골든 허용오차(`float16` 5e-3, `bfloat16` 6e-2) 안이라 케이스는 통과합니다.

> **정정 (조율 세션, 실측).** 이 절이 이어서 적었던 **"`float32`·`float64` 는 갈리지 않고,
> 마지막 축은 어떤 dtype 에서도 갈리지 않는다" 는 틀렸습니다. 정확히 반대입니다** —
> `float32`·`float64` 는 **마지막 축에서만** 갈리고, `dim=0` 에서는 비트까지 일치합니다.
>
> ```
> float32, n=8, 첫 원소
>   상류 dim=-1   0.0037626323755830526
>   상류 dim=0    0.0037626326084136963
>   shim  dim=-1  0.0037626326084136963
>   shim  dim=0   0.0037626326084136963
> ```
>
> **상류의 두 값이 서로 다른 것이 열쇠입니다.** 상류는 마지막 축에 전용 벡터화 커널
> (`softmax_lastdim_kernel`)을 쓰고 그 외에는 일반 경로를 씁니다. 우리 shim 은 한 경로만 쓰며,
> 그것이 상류의 **일반 경로**와 일치합니다. 그래서 상류가 특수 커널을 쓰는 자리에서만 갈립니다.
> 차이는 float32 기준 약 1 ulp (상대 6e-8) 입니다.
>
> 초안의 "벡터화 `exp` 대 libm `expf`" 라는 설명은 방향은 맞았지만 **축을 반대로 짚었습니다** —
> 벡터화되는 쪽이 마지막 축입니다.
>
> **이것이 §7 의 안전 논증을 약화시킵니다.** 아래 "샘플링 경로에는 물지 않는다 — 로짓은
> `float32` 이고 softmax 는 마지막 축이다" 는 **바로 그 갈리는 설정**을 안전의 근거로 들고
> 있습니다. 토큰 90/90 일치라는 실측 결과는 그대로 유효하지만, **그것이 성립한 이유는 이
> 문장이 말하는 이유가 아닙니다.** 1 ulp 차이가 버킷 경계에서 선택을 뒤집을 수 있다는 것은
> 이 문서 자신이 §4.2 에서 bf16 누적으로 보인 바입니다. 안전은 측정으로 확인된 것이지
> 구조적으로 보장된 것이 아닙니다.

**원인은 확인하지 못했습니다.** 유력한 설명은 상류의 비-마지막-축 커널이 `Vectorized<float>::exp()`
(다항식 근사)를 쓰고 이 shim 은 libm `expf` 를 부른다는 것입니다 — 마지막 축의 짧은 경로가
스칼라 `std::exp` 로 떨어져 일치하는 것과 앞뒤가 맞습니다. **다만 이것은 추론이지 측정이 아닙니다.**
`float16` 에서 이 shim 의 값이 정확값에 더 가깝다는 것만 확실합니다.

샘플링 경로에는 물지 않습니다 — 로짓은 `float32` 이고 softmax 는 마지막 축입니다.

---

## 5. `uniform_` / `normal_` 의 검사 승격 (작업 2)

`docs/numerics/RNG.md` §5 항목 3 이 승격 기준을 정해 두었고, 그대로 따랐습니다. 두 케이스 빌더는 이제
**양쪽 생성기를 같은 시드로 맞춘 뒤 뽑은 값을 원소 단위로 대조**합니다 (`_rng_stream_check`).

| op | 대상 | 대조 강도 |
|---|---|---|
| `aten.uniform_.default` | **모든 플랫폼** | 비트 단위 + `[from, to)` 범위 |
| `aten.normal_.default` | aarch64 (`platform.machine()`) | 비트 단위 |
| `aten.normal_.default` | 그 외 | dtype 허용오차, **그리고 그 사실을 메시지에 찍는다** |
| `aten.randint.low` | 모든 플랫폼 | `_range_check` 유지 — RNG.md §6 이 알고리즘 미확인으로 남김 |

`normal_` 의 플랫폼 분기는 `RNG.md` §3.3 이 정한 그대로입니다: 비트 일치는 aarch64 에서 측정된
것이고, `NormalFill16` 의 AVX2/VSX 특수화가 사는 호스트에서는 상류의 `sincos256_ps` 가 libm 과
같은 값을 내는지 **아무도 재지 않았습니다.** 재지 않은 플랫폼에 비트 일치를 요구하지 않습니다.
허용오차 갈래는 같은 주장의 약한 버전이 아니라 **다른 주장**이므로, 어느 쪽을 했는지 메시지에
남깁니다.

승격만으로는 부족해서 케이스도 늘렸습니다:

- **`normal_` 의 n = 6, 15, 16, 17, 20, 32.** `RNG.md` §5 항목 2 가 "반드시 케이스로 두라"고 한
  다섯에 기존 6 을 더한 것입니다. 승격 전 이 빌더는 `(2,3)` = 6 원소뿐이었고 — **경로 B 만**
  건드렸습니다. 경로 A 와 꼬리 재계산은 검사된 적이 없었습니다.
- **`uniform_` 의 `(2.0, 7.5)` 범위.** 처음 시도하는 범위 `(0,1)`·`(-1,1)`·`(-0.5,0.5)` 는 전부
  폭이 2 의 거듭제곱이라 아핀 단계의 곱이 정확해지고, `x*(to-from)+from` 과 clang 이 실제로
  컴파일하는 융합 곱셈-덧셈이 구별되지 않습니다. `rng.rs` 주석이 `mul_add` 이전에 추첨의 약 9.5%
  가 1 ulp 낮게 나왔다고 적어 둔 그 차이입니다. 폭이 2 의 거듭제곱이 아닌 범위만 그것을 봅니다.
- **`uniform_` 의 범위 검사는 유지.** 비트 대조로 갈음하지 않았습니다. 반개구간 보장은 **좁히는
  캐스트 뒤에** 걸리는 클램프가 지키고, `float16` 에 `to=1.0` 이면 약 4096 추첨에 한 번 발동합니다.
  클램프가 빠진 shim 은 나머지 전부에서 스트림과 일치합니다.
- **양쪽의 거부 케이스.** `normal_(std<0)`, `uniform_(from>to)`.

`cases.py` 모듈 주석의 "두 RNG 는 시드로 맞출 수 없다" 서술도 갱신했습니다 — **독립적인** 두
생성기에 대해서는 맞지만, 포팅 후의 이 코드베이스를 설명하지 않습니다.

부수적으로 `_pair_result_check` 를 고쳤습니다. `math.isclose(nan, nan)` 은 거짓이므로, `sort` 가
NaN 을 돌려주는 케이스를 처음 넣었을 때 **양쪽이 NaN 으로 일치하는데 불일치로 보고**됐습니다.
`max.dim` 은 NaN 케이스가 없어 드러나지 않았던 구멍입니다.

---

## 6. 못 한 것 · 모르는 것 — 명시

- ~~**`torch.<op>` 철자로는 아직 안 닿습니다.**~~ **정정 (문서 감사, 2026-09): 지금은 닿습니다.**
  이 작업은 `overloads.json`·`methods.json`·`bootstrap.py` 를 건드리지 않았지만(소유자가
  다름, 이 문서를 쓸 때는 사실), 다른 라운드(§0 표의 8 개가 landed 된 뒤)가 표 항목을
  채운 것으로 보입니다 — 실측: `torch.multinomial(torch.tensor([0.1,0.2,0.7]), 1)`,
  `torch.sort(...)`, `torch.topk(...)` 전부 오늘 성공합니다. `overloads.json` 에
  `"multinomial"`/`"sort"`/`"topk"` 키가 있습니다. 스키마 대조 숫자(127/127)가 이 문서를 쓴
  시점 이후 바뀌었는지는 확인하지 않았습니다 — 이 항목의 요점(파이썬 철자로 안 닿는다)만
  재확인했습니다.
  <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json multinomial present -->
  <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json sort present -->
  <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json topk present -->
- **`topk` 의 동점 순서 (§4.2).** 재현하지 않았고, 재현하지 않기로 한 근거를 적었습니다.
  로짓이 정확히 같은 값을 갖는 모델에서 `top_k` 의 컷이 갈릴 수 있는지는 **확인하지 않았습니다.**
- **축소 dtype 의 비-마지막-축 `_softmax` (§4.3).** 차이는 재었고 원인은 추론입니다.
- **`aten.exponential_` 은 구현하지 않았습니다.** `multinomial` 이 내부적으로 쓰는 것을
  `rng.rs::exponential_serial` 로 옮겼을 뿐이고, op 으로 광고하지 않습니다 — 아무 측정도 그것을
  직접 부르지 않았고, 요구된 적 없는 op 을 광고하는 것은 `docs/design/TORCH_C.md` §1 이 거부하는 방향입니다.
- **`sort.stable` · `topk` 의 `out=` 변형 · `scatter.value` · `scatter.reduce` 는 범위 밖입니다.**
  측정된 여덟 개에 없습니다.
- **비연속 텐서에 대한 `normal_` 경로 B.** `RNG.md` §1.3 이 `is_contiguous()` 가 관측 가능한 값을
  가른다고 실측했지만, 이 shim 의 슬라이스가 저장소를 공유하지 않아(`docs/bindings/TENSORBASE.md`) 골든
  케이스로 만들 방법이 없습니다. 커널의 분기는 구현되어 있고 **대조되지 않았습니다.**
- **GPT-2 계열의 샘플링.** `GAP.md` §5 가 별도로 요구하는 op 들은 재지 않았습니다.
- **기기에서의 임포트.** Android·iOS 는 **링크만** 확인했습니다 — `docs/design/TORCH_C.md` §6 이 이미
  미확인으로 적어 둔 항목이고, 이 작업이 바꾸지 않았습니다.

---

## 7. 재현 방법

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-sample
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
PY=/Volumes/macMini/caches/spike-venv/bin/python
DIST=/Volumes/macMini/caches/target-python

# 벤더 트리는 git 에 없다 — 새 worktree 에서는 이것부터
./vendor/vendor_torch.sh

cd rust/torch_c            # cd 필수 — .cargo/config.toml 은 cwd 기준
PYTHON=$PY ./pytests/run.sh > /tmp/smoke.log 2>&1; echo "EXIT=$?"

# 골든 · 스키마 — PYTHONPATH=vendor 를 붙이지 않는다.
cd ../..
$PY tools/golden/compare.py > /tmp/golden.log 2>&1; echo "EXIT=$?"
$PY rust/torch_c/pytests/verify_schemas.py > /tmp/schemas.log 2>&1; echo "EXIT=$?"
for m in value shape dtype; do
  $PY tools/golden/compare.py --inject-fault $m > /tmp/fault-$m.log 2>&1; echo "$m EXIT=$?"
done

# 3 타깃 — docs/design/TORCH_C.md §7 과 동일
```

샘플링 대조 스크립트(`e2e_sample.py`)와 워드 카운터·경계 협착 프로브는
`/Volumes/macMini/caches/bw-sample-probe/` 에 있습니다 — 파일 범위 규정을 지키기 위해 커밋 대상에
넣지 않았습니다.

---

## 8. 이 작업이 건드린 파일

```
rust/torch_c/src/aten.rs     여덟 개의 커널, IMPLEMENTED 70 -> 78,
                             IMPLEMENTED_AWAITING_GOLDEN 에서 fill_.Tensor 제거
rust/torch_c/src/rng.rs      exponential_serial / uniform_sample_f64 추가
tools/golden/cases.py        여덟 개의 케이스 빌더, uniform_/normal_ 승격,
                             _pair_result_check 의 NaN 처리
docs/models/SAMPLING.md             이 문서
```

`rust/torch_c/src/bootstrap.py`·`overloads.json`·`methods.json` 은 건드리지 않았습니다.
