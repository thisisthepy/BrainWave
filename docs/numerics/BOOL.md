# `torch.bool` — candle 에 없는 dtype 을 어떻게 할 것인가

TORCH_C.md §5-3 이 남긴 항목입니다. **결론부터: `torch.bool` 을 candle 의 `U8` 로 별칭하지
않습니다. `_C` 가 소유하는 dtype 태그로 두고, 저장은 `U8` 로 하되 불리언 연산을 명시적으로
구현합니다** (아래 §5 의 선택지 B).

근거는 전부 실측입니다 — candle 은 소스를 읽었고(파일·행 번호 표기), torch 는
`/Volumes/macMini/caches/spike-venv` 의 torch 2.13.0 으로 돌렸습니다.

---

## 0. 한눈에

| 질문 | 답 | 근거 |
|---|---|---|
| candle 의 비교 연산은 무엇을 돌려주는가 | **`U8`, 값은 0/1** | `tensor.rs:1124` 주석 + `cpu_backend/mod.rs:62-83` + 실행 |
| candle 의 `where_cond` 는 무엇을 받는가 | **정수 dtype 전부** (`U8`·`U32`·`I16`·`I32`·`I64`), `!= 0` 을 참으로 | `cpu_backend/mod.rs:2735-2751`, `dtype.rs:239-268` |
| candle 에 bitwise 연산이 있는가 | **하나도 없다** (100 개 `.rs` 전체에 `bitwise` 0 회) | grep |
| `torch.bool` 의 element_size | **1 바이트**, `uint8` 과 같음 | 실측 |
| 그러면 같은 것인가 | **아니다.** 연산 의미론이 다르다 | §2 |
| 이 모델이 불리언을 지나는 op 수 | **15 개** (TORCH_C §5-3 이 센 9 개가 아니다) | §4, 재계측 |
| `bool → uint8` 별칭이 틀리는 방식 | **조용히** — 그리고 **torch 자신의 방어막 6 개를 지운다** | §3, §7 |
| 권고 | **선택지 B**: `_C` 소유 dtype 태그 + `U8` 저장 + 불리언 연산 명시 구현 | §6 |

---

## 1. candle 이 불리언을 실제로 어떻게 다루는가

candle 은 **불리언 dtype 없이, `U8` 을 관례적으로 불리언처럼 쓰는** 설계입니다. 관례라는 것이
핵심입니다 — 타입이 아니라 규약이므로 강제되지 않습니다.

### 1.1 `DType` 에 불리언이 없다

`candle-core-0.11.0/src/dtype.rs:9-38` 의 14 개 variant:

```
U8  U32  I16  I32  I64  BF16  F16  F32  F64  F8E4M3  F6E2M3  F6E3M2  F4  F8E8M0
```

`Bool` 도 `I8` 도 없습니다. `size_in_bytes()` (`dtype.rs:93-110`) 에서 `U8` 은 1 바이트,
`is_int()` (`dtype.rs:113`) 에서 `U8` 은 정수로 분류됩니다.

### 1.2 비교 연산은 `U8` 을 돌려준다 — 값은 0/1 보장

`src/tensor.rs:1121-1173`. `cmp` 하나가 `eq`·`ne`·`lt`·`gt`·`ge`·`le` 전부를 뒷받침하고,
주석이 명시합니다 (`tensor.rs:1124`):

> The returned tensor has the same shape as the original tensors and uses `u8` elements.

실제 커널은 `src/cpu_backend/mod.rs:62-83` 이고, `Map2U8` 을 구현하며 `u8::from(x == y)` 를
씁니다 — **출력은 반드시 0 또는 1** 입니다.

실행 확인 (`/Volumes/macMini/caches/bool-probe/candle-probe`):

```
lt dtype          = U8
lt values         = [1, 0, 0]
eq dtype          = U8  values = [0, 1, 0]
lt scalar         = [1, 0, 0]
```

**즉 candle 이 *만드는* 불리언은 항상 정규(0/1)입니다.** 이것이 뒤에 나올 권고의 전제가 됩니다.

### 1.3 `where_cond` 는 정수면 무엇이든 받고, `!= 0` 을 참으로 본다

`src/cpu_backend/mod.rs:2735-2751`:

```rust
match self {
    Self::U8(pred)  => WCond(pred, layout).map(t, t_l, f, f_l),
    Self::U32(pred) => ...
    Self::I16(pred) | Self::I32(pred) | Self::I64(pred) => ...
    _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "where-cond")),
}
```

참/거짓 판정은 `IntDType::is_true` (`src/dtype.rs:239-268`) 이고 전부 `*self != 0` 입니다.
`WCond::f` (`cpu_backend/mod.rs:87-118`) 가 이것으로 분기합니다.

실행 확인:

```
where(u8 cond)    = [10.0, -2.0, -3.0]
where(u8 = 2,3,0) = [10.0, 20.0, -3.0]     <- 2 도 3 도 참
where(i64 cond)   = [10.0, -2.0, 30.0]     <- i64 조건도 받는다
where(f32 cond)   !! unsupported dtype F32 for op where-cond
```

**candle 은 조건 텐서의 dtype 을 검사하지 않습니다.** `I64` 인덱스 텐서를 실수로 마스크 자리에
넣어도 통과합니다. torch 는 `masked_fill` 에서 `bool` 이 아니면 거부합니다(§2.4) — 이 방어막이
candle 쪽에는 아예 없습니다.

또 `where_cond` 는 **브로드캐스팅하지 않습니다** (`tensor.rs:1565-1567` 이
`same_shape_binary_op` 을 두 번 호출). `broadcast_binary_op!` 목록(`tensor.rs:618-627`)에
`broadcast_where_cond` 는 없습니다. torch 의 `masked_fill` 은 마스크를 브로드캐스팅하므로,
shim 이 `broadcast_as` 를 직접 걸어야 합니다. 확인:

```
where (1,3)/(2,3) !! shape mismatch in where_cond, lhs: [1, 3], rhs: [2, 3]
broadcast_as 후    = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
```

### 1.4 `U8` 리덕션은 `U8` 로 누산한다 — 256 에서 감긴다

`src/cpu_backend/mod.rs:231-306`. `ReduceSum::fold_impl<T>` 는 `T::zero()` 로 시작해
`dst[dst_index] += src` (287·294 행) 로 누산합니다. `T` 는 입력 dtype 그대로입니다
(`Map1 for ReduceSum` — `mod.rs:301-306`). 빠른 경로인 `vec_reduce_sum`
(`src/cpu/kernels.rs:41-46`) 도 `*res += *xs.add(i)` 로 같은 타입에 누산합니다.

**릴리스 빌드에서 `u8` 덧셈은 오버플로 검사가 없으므로 조용히 감깁니다.** 측정:

```
u8 ones(300).sum  = 44      <- torch 의 bool.sum() 은 300
i64 ones(300).sum = 300
```

`44 == 300 - 256`. 예외도 경고도 없습니다.

### 1.5 정수 dtype 에는 단항 연산이 아예 없다 — 그리고 `Err` 가 아니라 **패닉**

`src/op.rs:464-503` 의 `unary_op!` 매크로가 정수 타입 분기를 `todo!()` 로 채웁니다:

```rust
fn u8(_: u8) -> u8 { todo!("no unary function for u8") }   // op.rs:488
```

`unary_op!(Neg, "neg", v, -v)` (`op.rs:589`) 이므로 `u8` 텐서에 `neg()` 를 부르면:

```
thread 'main' panicked at candle-core-0.11.0/src/op.rs:589:1:
not yet implemented: no unary function for u8
```

`catch_unwind` 로 확인했고 실제로 `Err` 가 아닌 **패닉**입니다. PyO3 는 패닉을
`pyo3_runtime.PanicException` 으로 바꾸므로 파이썬에서 잡히기는 합니다만, `Result` 경로가
아니라는 점은 shim 의 오류 처리 설계에 영향을 줍니다.

### 1.6 bitwise 연산이 하나도 없다

```
$ grep -rin "bitwise" --include="*.rs" candle-core-0.11.0/   →  0 건 (100 개 파일)
```

`bitwise_and`·`bitwise_or`·`bitwise_not`·`bitwise_xor` 전부 없습니다. **이것은 bool 결정과
독립적인 사실입니다** — 어느 선택지를 고르든 이 세 op 은 손으로 구현해야 합니다. 다만 값 집합이
{0,1} 로 닫혀 있으면 기존 op 으로 합성할 수 있습니다(§6.2).

### 1.7 dtype 승격이 없다

```
f32 + f64         !! dtype mismatch in add, lhs: F32, rhs: F64
```

candle 은 승격하지 않고 거부합니다. TORCH_C §2 가 `add.Tensor` 에서 이미 부딪힌 벽이고
(§5-2 로 남아 있음), **이 사실이 bool 결정과 강하게 얽힙니다** — §3.5 참조.

---

## 2. `torch.bool` 이 실제로 무엇인가 (torch 2.13.0 실측)

### 2.1 저장 표현은 `uint8` 과 같다

```
bool element_size            => 1
uint8 element_size           => 1
bool untyped_storage nbytes  => 3        (원소 3 개)
bool storage bytes           => [1, 0, 1]
bool.view(uint8)             => [1, 0, 1]
uint8.view(bool)             => [True, False, True]
```

**바이트 수준에서는 구별되지 않습니다.** `view` 로 무비용 상호 변환됩니다. 그러므로 차이는
저장이 아니라 **연산 의미론**에 있습니다.

메타데이터도 `itemsize=1`, `is_signed=False`, `is_floating_point=False` 로 `uint8` 과 같습니다.
다른 것은 정체성뿐입니다 — `torch.bool == torch.uint8` 은 `False`.

### 2.2 산술이 다르다

| 식 | `bool` | `uint8` |
|---|---|---|
| `x + x` (`[1,0,1]`) | `[True, False, True]` — **논리합** | `[2, 0, 2]` — **산술합** |
| `x + x` 의 dtype | `torch.bool` | `torch.uint8` |
| `x * x` 의 dtype | `torch.bool` | `torch.uint8` |
| `x - x` | **`RuntimeError`** ("use `^` or `logical_xor()`") | `[0, 0, 0]` |
| `-x` | **`RuntimeError`** ("use `~` or `logical_not()`") | `[255, 0, 255]` |
| `~x` | `[False, True, False]` — **논리부정** | `[254, 255, 254]` — **비트반전** |
| `x.sum()` | `2`, dtype `int64` | `2`, dtype `int64` |
| `x.mean()` | **`RuntimeError`** | `1.0`(f32 승격 후) |
| `x.cumsum(0)` | `[1, 1, 2]` (int64) | 같음 |

`~` 가 가장 노골적입니다. `bool` 의 `~` 는 논리부정이고 `uint8` 의 `~` 는 비트반전이며,
**비트반전 결과는 전부 0 이 아니므로 전부 참**입니다. 인과 마스크 반전이 정확히 이 경로입니다(§3.1).

### 2.3 승격 격자에서 `bool` 은 바닥, `uint8` 은 아니다

```
promote_types(bool,  X) -> X          (모든 X 에 대해)
promote_types(uint8, bool)  -> uint8
promote_types(uint8, int8)  -> int16   <- bool 이면 int8
promote_types(uint8, uint8) -> uint8
```

`bool` 은 승격 격자의 **바닥 원소**입니다 — 어떤 것과 만나도 상대를 그대로 돌려줍니다.
`uint8` 은 그렇지 않습니다. 이것이 TORCH_C §5-2 (dtype 승격표)와 직접 충돌합니다(§3.5).

### 2.4 마스킹 API 가 dtype 으로 방어된다

```
masked_fill(bool)   => [-1.0, 1.0, -1.0]
masked_fill(uint8)  !! RuntimeError: masked_fill_ only supports boolean masks,
                                    but got mask with dtype unsigned char
x[bool]             => [0.0, 2.0]
x[uint8]            => [0.0, 2.0]   + UserWarning: indexing with dtype torch.uint8 is now deprecated
torch.where(bool)   => [0.0, -1.0, 2.0]
torch.where(uint8)  => [0.0, -1.0, 2.0]  + UserWarning: where received a uint8 condition tensor.
                                           This behavior is deprecated ...
```

**torch 는 두 타입을 섞지 못하게 의도적으로 막아 두었습니다.** 하드 에러 4 개(`masked_fill`,
`bool - bool`, `-bool`, `bool.mean()`)와 폐기 경고 2 개(`x[uint8]`, `where(uint8)`).
이 방어막은 전부 **dtype 태그**에 걸려 있습니다 — 태그를 지우면 방어막도 함께 사라집니다(§7).

### 2.5 `any` 의 반환 dtype 이 다르다

```
bool.any().dtype   => torch.bool
uint8.any().dtype  => torch.uint8      <- bool 이 아님
uint8.any() value  => tensor(1, dtype=torch.uint8)
```

CORE_ATEN 목록의 `aten.any.default` · `aten.any.dim` 이 여기 걸립니다. 별칭하면 반환 dtype 이
`uint8` 이 되고, 그 결과를 다시 `masked_fill` 에 넣는 호출자는 torch 라면 `RuntimeError` 를
봤을 자리에서 조용히 통과합니다.

### 2.6 `bool` 은 읽을 때 정규화한다, `uint8` 은 안 한다

바이트가 0/1 이 아닌 텐서를 만들어 두 dtype 으로 보면:

```
raw = uint8([2, 3, 0])
raw.to(float32)              => [2.0, 3.0, 0.0]
raw.view(bool).to(float32)   => [1.0, 1.0, 0.0]      <- 정규화
raw.sum()                    => 5
raw.view(bool).sum()         => 2                    <- 참의 개수
raw.view(bool) + 자기자신     => 바이트 [1, 1, 0]     <- 정규 출력
~raw.view(bool)              => 바이트 [0, 0, 1]
```

**`torch.bool` 은 "바이트가 0/1 이다" 를 보장하지 않고, "연산이 `!= 0` 으로 읽고 0/1 로 쓴다"
를 보장합니다.** candle 의 `U8` 에는 그런 계약이 없습니다 — `to_dtype(F32)` 로 확인:

```
candle (2,3,0).to(f32) = [2.0, 3.0, 0.0]     <- torch.bool 은 [1.0, 1.0, 0.0]
candle (2,3,0).to(i64).sum = 5               <- torch.bool 은 2
```

### 2.7 팩토리

```
torch.full((2,), True).dtype  => torch.bool
torch.full((2,), 1).dtype     => torch.int64
torch.tensor([True]).dtype    => torch.bool
```

TORCH_C §2 가 `full.default` 에서 맞출 수 없다고 적은 항목이 이것입니다. 파이썬 `bool` 이
`int` 의 서브클래스라 정수 분기로 떨어지지만 torch 는 `torch.bool` 을 줍니다.

---

## 3. 어긋나면 어디서 어떻게 드러나는가 (재현 가능)

전부 `/Volumes/macMini/caches/bool-probe/{repro,silent}.py` 로 재현됩니다.

### 3.1 인과 마스크 반전 — 출력 전체가 NaN

Llama 어텐션의 실제 경로입니다. `~mask` 로 하삼각 마스크를 뒤집습니다.

```python
mask_bool = torch.ones(S, S, dtype=torch.bool).tril()
ref   = softmax(scores.masked_fill(~mask_bool, -inf)) @ v

mask_u8 = torch.ones(S, S, dtype=torch.uint8).tril()
alias = softmax(scores.masked_fill(torch.bitwise_not(mask_u8).to(torch.bool), -inf)) @ v
```

```
bool  경로 출력[0][:3] = [1.6459, -1.3602, 0.3446]
uint8 경로 출력[0][:3] = [nan, nan, nan]
NaN 개수 (bool / uint8) = 0 / 48

~bool  첫 행 : [0, 1, 1, 1, 1, 1]
~uint8 첫 행 : [254, 255, 255, 255, 255, 255]   <- 전부 truthy
```

`~` 가 비트반전이라 마스크가 "전부 가림" 이 되고, 한 행이 통째로 `-inf` 가 되어 softmax 가
NaN 을 냅니다. **이것은 그나마 시끄러운 축입니다** — NaN 이 보이니까요. 다만 예외는 나지
않고, 그리디 디코딩의 `argmax(NaN...)` 은 조용히 인덱스를 돌려줍니다.

### 3.2 패딩 마스크의 토큰 수 — 조용히 틀린다

`aten.sum.default` 가 실제로 부르는 것이 이것입니다(§4 에서 `(1, seq_len)` bool 로 관측).

```
bool.sum()                = 300
U8 로 누산 시              = 44        <- candle 이 하는 것 (§1.4)
```

예외 없음, NaN 없음. **프롬프트가 256 토큰을 넘는 순간부터 감깁니다.**

### 3.3 조용히 틀리는 종합 사례 — 마스크 평균 풀링

```python
def meanpool(mask, sum_dtype):
    w   = mask.to(torch.float32).unsqueeze(-1)
    num = (h * w).sum(dim=1)
    den = mask.sum(dim=1, dtype=sum_dtype).to(torch.float32).unsqueeze(-1)
    return num / den
```

`B=2, S=300, H=4`, 유효 길이 `[300, 260]`:

```
bool  : [0.0017, 0.0294, 0.0230, 0.0160]
uint8 : [0.0117, 0.2005, 0.1569, 0.1089]
NaN? False   Inf? False
상대오차 최대: 64.0
```

**예외도 NaN 도 없고 값의 크기도 그럴듯합니다.** 64 배 틀린 값이 정상적으로 흘러갑니다.
DESIGN.md §5 가 A 의 주 리스크로 지목한 "수치 불일치가 조용히 번짐" 의 교과서적 형태입니다.

### 3.4 마스크 논리 결합 — 조용히 2 배

```
bool  (m1+m2).to(f32) * x = [1.0, 2.0, 3.0]
uint8 (m1+m2).to(f32) * x = [2.0, 2.0, 3.0]     <- 첫 원소만 2 배
```

`bool + bool` 은 논리합이지만 `uint8 + uint8` 은 산술합이라 겹치는 자리에 2 가 생깁니다.
마스크로 쓸 때는 2 도 참이라 무해하지만, **float 로 캐스팅해 가중치로 쓰는 순간 틀립니다.**
§4 의 실측에서 `aten.mul.Tensor` 가 `in=bool,float32,int64` 로 관측되므로 이 경로는 가설이
아닙니다.

### 3.5 승격표(TORCH_C §5-2)를 구현 불가능하게 만든다

가장 구조적인 문제입니다. `_C` 는 candle 이 안 하는 dtype 승격을 직접 구현해야 하고
(TORCH_C §5-2), 그 승격표의 격자에서 **`bool` 은 바닥, `uint8` 은 바닥이 아닙니다**(§2.3).

`bool` 을 `uint8` 로 별칭하면 격자에서 두 원소가 같은 자리를 차지하므로,
`promote(bool, X) = X` 와 `promote(uint8, X)` 를 동시에 만족시키는 표를 쓸 수 없습니다.
`promote(bool, bool) = bool` 대 `promote(uint8, uint8) = uint8` 이 이미 다른 결과를 요구합니다.

즉 **§5-2 와 §5-3 은 독립 항목이 아니라 하나입니다.** 별칭을 택하면 승격표는 나중에 고칠 수
있는 것이 아니라, 애초에 옳게 쓸 수 없습니다.

---

## 4. 이 모델이 실제로 불리언을 지나는 곳 — 9 개가 아니라 15 개

TORCH_C §5-3 은 9 개를 셌습니다. **op 이름이 아니라 실제 흐르는 dtype 을 기준으로 다시 재면
15 개입니다.** `TorchDispatchMode` 로 CORE_ATEN §2 와 같은 구성(hidden=64, layers=2, heads=2,
intermediate=128, vocab=100, `generate(max_new_tokens=4, do_sample=False)`)을 돌려
입출력 텐서의 dtype 을 함께 기록했습니다
(`/Volumes/macMini/caches/bool-probe/trace_bool.py`, 총 47 개 op 관측).

| op | 입력 dtype | 출력 dtype | §5-3 목록에 있었나 |
|---|---|---|---|
| `aten._local_scalar_dense.default` | `bool` | (파이썬 스칼라) | **없었음** |
| `aten._to_copy.default` | `bool,float32,int64` | `bool,float32,int64` | **없었음** |
| `aten.any.default` | `bool` | `bool` | 있음 |
| `aten.any.dim` | `bool` | `bool` | 있음 |
| `aten.bitwise_and.Tensor` | `bool,int64` | `int64` | 있음 |
| `aten.bitwise_not.default` | `bool` | `bool` | 있음 |
| `aten.bitwise_or.Tensor` | `bool` | `bool` | 있음 |
| `aten.eq.Scalar` | `int64` | `bool` | 있음 |
| `aten.full.default` | — | `bool` | **없었음** |
| `aten.isin.Tensor_Tensor` | `int64` | `bool` | **없었음** |
| `aten.lt.Scalar` | `int64` | `bool` | 있음 |
| `aten.masked_fill.Scalar` | `bool,int64` | `int64` | 있음 |
| `aten.mul.Tensor` | `bool,float32,int64` | `bool,float32,int64` | **없었음** |
| `aten.ne.Tensor` | `int64` | `bool` | 있음 |
| `aten.sum.default` | `bool` | `int64` | **없었음** |

새로 드러난 여섯 개가 각각 다른 것을 말합니다.

- **`aten._local_scalar_dense.default` (입력 `bool`, 스칼라 `()` 모양).** 불리언 텐서가 파이썬
  `bool` 로 내려와 **제어 흐름을 결정합니다** (`if mask.any(): ...`). 여기서 틀리면 값이 아니라
  **실행 경로가 갈라집니다.** 별칭하면 `uint8` 스칼라 `2` 가 `bool(2) == True` 로 통과하므로
  대부분은 우연히 맞겠지만, 그 "우연히" 를 근거로 삼을 수 없습니다.
- **`aten.sum.default` (입력 `bool` → 출력 `int64`).** §3.2 의 오버플로가 걸리는 정확한 지점.
  관측된 모양이 `(1, 8) (1, 9) (1, 10) (1, 11)` — **시퀀스 길이에 비례**합니다. 장난감
  모델이라 작을 뿐이고, 실제 프롬프트가 256 토큰을 넘으면 `U8` 누산은 감깁니다.
- **`aten.mul.Tensor` (입력에 `bool` 포함, 출력에도 `bool` 포함).** §3.4 의 경로.
- **`aten.full.default` (출력 `bool`)** — TORCH_C §2 의 미해결 항목이 실제로 밟힙니다.
- **`aten.isin.Tensor_Tensor` (`int64` → `bool`)** — 불리언 *생산자*가 하나 더 있습니다.
- **`aten._to_copy.default` (`bool` ↔ `bool`)** — dtype 캐스팅 경로가 `bool` 을 알아야 합니다.

**요약: 불리언은 이 모델의 주변부가 아니라 47 개 op 중 15 개, 약 3 분의 1 을 지납니다.**
"불리언 텐서를 아예 안 만든다" 는 선택지(§5-D)의 실현 가능성이 여기서 판정됩니다.

---

## 5. 선택지

### A. candle 의 `U8` 에 별칭 — `torch.bool → DType::U8`

`_C` 가 `torch.bool` 이라는 이름을 `DType::U8` 에 붙입니다. 지금 `rust/torch_c/src/dtype.rs:20-22`
의 `PyDtype { inner: DType }` 구조를 그대로 두고 등록 목록에 한 줄 더하면 되는, 가장 싼 변경입니다.

| | |
|---|---|
| 장점 | 변경량이 사실상 0. 15 개 op 이 즉시 "동작" 한다 |
| 단점 1 | **`torch.bool == torch.uint8` 이 참이 된다.** `PyDtype::__eq__` 가 `inner` 를 비교하므로(`dtype.rs:67-72`) 두 이름이 같은 객체가 된다. 벤더링한 파이썬 트리가 `dtype == torch.bool` 로 분기하는 모든 지점이 오작동한다 |
| 단점 2 | §2.2 의 산술 차이 전부 (`+`, `~`, `-`, `mean`) |
| 단점 3 | §3.2·§3.3 의 `sum` 오버플로 — 조용함 |
| 단점 4 | §3.5 — 승격표(§5-2)를 옳게 쓸 수 없게 된다 |
| 단점 5 | **§7 — torch 자신의 방어막 6 개를 지운다** |
| 실패 방식 | **조용함.** §3.3 이 실증 (예외·NaN 없이 64 배 오차) |

### B. `_C` 가 dtype 태그를 소유하고, 저장만 `U8` 로 한다 ← **권고**

`torch.bool` 을 candle 의 dtype 이 아니라 **`_C` 가 소유하는 별개의 dtype** 으로 둡니다.
저장 표현은 `DType::U8` 이지만 태그는 다르고, 불리언 연산은 태그를 보고 명시적으로 구현합니다.

이것은 **이 저장소가 `device` 에서 이미 내린 결정과 같은 형태**입니다 — TORCH_C §1 의
"`device` 는 candle 의 `Device` 를 감싸지 않는다. 라벨이고, 쓸 때 `resolve()` 한다."
torch 레벨 개념을 shim 이 라벨로 소유하고, candle 은 그 아래에서 저장·커널만 담당합니다.

| | |
|---|---|
| 장점 1 | `torch.bool != torch.uint8` 이 성립한다. 파이썬 트리의 dtype 분기가 산다 |
| 장점 2 | **미구현이 기존 `NotImplementedError` 깔때기로 떨어진다** (TORCH_C §1). 불리언 규칙이 없는 op 은 조용히 통과하는 대신 이름을 대고 터진다 |
| 장점 3 | 승격 격자에 `bool` 을 바닥으로 넣을 수 있다 → §5-2 와 정합 |
| 장점 4 | candle 포크 불필요. 상류 일정에 묶이지 않는다 |
| 단점 1 | **`bool` 로 태그된 텐서의 바이트가 0/1 이라는 불변식을 shim 이 지켜야 한다.** 이 불변식이 깨지면 조용히 틀린다 (§6.3 에서 다룸) |
| 단점 2 | 불리언 op 을 하나씩 손으로 쓴다 (§6.2 — 다행히 전부 기존 candle op 의 합성) |
| 단점 3 | `bool.sum()` 에 `to_dtype(I64)` 물질화가 한 번 든다 (마스크가 `(1, seq)` 라 무시 가능, §4 실측) |
| 단점 4 | **`int8` 로는 일반화되지 않는다.** bool 이 되는 이유는 값 집합이 2 원소라 연산 규칙이 유한하기 때문이고, `int8` 은 그렇지 않다 |
| 실패 방식 | **시끄러움** — 규칙이 없는 조합은 `NotImplementedError`. 단 불변식 위반만은 조용하므로 §6.3 의 검사가 필요 |

### C. candle 에 dtype 추가 (포크 또는 상류 기여)

| | |
|---|---|
| 장점 | 커널 레벨에서 옳아진다. `where_cond` 가 dtype 을 검사할 수 있게 된다 |
| 단점 1 | **표면이 크다.** `candle-core/src/` 39,494 줄에 `DType::` 이 **1,529 회** 등장하고, 백엔드가 CPU·CUDA·Metal 셋이다 |
| 단점 2 | 포크 유지 부담. 다만 TORCH_C §5-1 (`tokenizers` 비선택 의존)이 이미 `[patch.crates.io]` 포크를 검토 중이므로 **한계 비용은 생각보다 낮을 수 있다** |
| 단점 3 | 상류 PR 은 일정이 우리 손에 없다 |
| 단점 4 | dtype 을 추가해도 §1.5 의 정수 단항 `todo!()` 패닉과 §1.6 의 bitwise 부재는 그대로다 |
| 실패 방식 | 시끄러움 (컴파일 에러 / `UnsupportedDTypeForOp`) |

### D. 불리언 텐서를 아예 안 만들도록 op 을 다르게 구현

| | |
|---|---|
| 판정 | **단독으로는 불가능.** §4 가 실측으로 닫는다 |
| 이유 | `aten._local_scalar_dense.default(bool)` 은 불리언 텐서를 파이썬 `bool` 로 내리는 호출이고, `aten._to_copy.default` 는 `bool → bool` 캐스팅이며, `aten.full.default` 는 `bool` 텐서를 **만든다.** 이 셋은 shim 이 아니라 **벤더링한 파이썬 트리**가 부른다. 우리가 op 구현을 바꿔도 호출자가 불리언 텐서를 요구한다 |
| 부분적 유용성 | 있다. `masked_fill` 을 불리언 중간값 없이 `where_cond` 합성으로 바로 내리는 것 같은 최적화는 B 위에서 하면 된다 |

### E. (추가) `I64` 를 불리언 저장으로 쓰기

`U8` 대신 `I64` 를 불리언의 저장 dtype 으로 삼으면 §1.4 의 오버플로가 사실상 사라집니다
(`where_cond` 는 `I64` 조건을 받습니다 — §1.3 에서 확인).

| | |
|---|---|
| 장점 | `sum` 오버플로 소멸. `to_dtype` 물질화 불필요 |
| 단점 1 | **메모리 8 배.** 어텐션 마스크는 `(B, H, S, S)` 로 커질 수 있어 무시 못 한다 |
| 단점 2 | `element_size` 를 torch 와 맞추려면(1) 거짓말을 하거나, 노출 시 변환해야 한다 |
| 단점 3 | `bool ↔ uint8` 의 무비용 `view`(§2.1)가 불가능해진다 |
| 판정 | **B 의 하위 변형으로 유효.** B 를 택하면 저장 dtype 은 나중에 바꿀 수 있는 내부 결정이 된다 — 이것 자체가 B 의 장점이다 |

---

## 6. 권고 — **B**

### 6.1 왜 B 인가

세 문장으로:

1. **A 는 §3.5 때문에 "나중에 고칠 수 있는 임시방편" 이 아닙니다.** 승격표(TORCH_C §5-2)는
   로드맵 2 번이고 bool 은 3 번인데, 격자에서 `bool` 이 바닥이라는 사실 때문에 **3 번을 A 로
   정하면 2 번을 옳게 쓸 수 없습니다.** 순서상 A 를 먼저 넣으면 바로 다음 항목이 막힙니다.
2. **A 의 실패는 조용하고, 그것도 torch 가 일부러 만들어 둔 방어막을 지우면서 조용해집니다**(§7).
   DESIGN.md §5 가 A 경로(candle)의 주 리스크로 지목한 것이 정확히 이 형태입니다.
3. **B 는 새 개념이 아니라 이 저장소가 `device` 에서 이미 쓴 패턴입니다**(TORCH_C §1).
   "torch 레벨 라벨을 shim 이 소유하고 candle 은 그 아래" — 일관성이 있고, 실패 지점이
   torch 와 같은 자리에 놓입니다.

C 는 지금 하기에는 표면이 너무 큽니다(1,529 개 `DType::` 사이트, 3 백엔드). 다만 §5-1 의
`tokenizers` 문제로 포크가 어차피 생긴다면 그때 **B 를 C 로 승격**하는 것은 자연스럽습니다 —
B 는 저장 dtype 을 shim 내부 결정으로 만들어 두므로, `U8` → `DType::Bool` 교체가 국소 변경이
됩니다. **B 는 C 를 막지 않고, A 는 막습니다.**

D 는 §4 가 닫았습니다.

### 6.2 B 로 갔을 때 각 연산을 무엇으로 구현하는가 (실행 확인 완료)

값이 0/1 로 정규화돼 있다는 전제 아래, **전부 기존 candle op 의 합성으로 됩니다.**
`/Volumes/macMini/caches/bool-probe/candle-probe` 실행 결과:

| torch op | candle 구현 | 확인된 출력 |
|---|---|---|
| `bitwise_or` / `logical_or` / `bool + bool` | `a.maximum(&b)` | `[1,1,1,0]` |
| `bitwise_and` / `logical_and` / `bool * bool` | `a.minimum(&b)` | `[1,0,0,0]` |
| `bitwise_not` / `logical_not` / `~` | `a.ones_like()? - a` | `[0,0,1,1]` |
| `bitwise_xor` / `logical_xor` / `bool - bool` | `a.ne(&b)` (결과가 `U8` 0/1) | `[0,1,1,0]` |
| `sum` (→ `int64`) | `a.to_dtype(I64)?.sum_all()` | `2` |
| `any` | `a.max_all()? != 0` | `1` |
| `all` | `a.min_all()? != 0` | `0` |
| `masked_fill` | `mask.broadcast_as(shape)?.where_cond(&fill, &self)` | §1.3 확인 |
| `eq`/`ne`/`lt` 등 (생산자) | candle `cmp` 그대로 — **이미 0/1 보장** (§1.2) | — |
| `_to_copy` bool→float | `to_dtype` — **정규화된 입력에서만 옳다** | §6.3 |
| `mean` | 구현하지 않고 `NotImplementedError` (torch 도 거부, §2.2) | — |
| `neg` | 구현하지 않고 `NotImplementedError` (torch 도 거부, §2.2) | — |

`maximum`/`minimum`/`ne`/`to_dtype`/`max_all`/`min_all` 은 전부 `U8` 에서 정상 동작합니다
(§1.5 의 `todo!()` 패닉은 **단항** op 에만 있고, 이 목록에는 단항이 없습니다).

`bitwise_*` 의 **비-불리언** 오버로드는 별개 문제입니다 — §4 에서 `bitwise_and.Tensor` 가
`in=bool,int64 → out=int64` 로 관측됐고, candle 에는 정수 bitwise 가 아예 없습니다(§1.6).
승격 후 진짜 비트 연산이 필요하므로 커스텀 커널이 듭니다. **bool 결정과 무관하게 드는 비용입니다.**

### 6.3 B 의 유일한 조용한 실패 지점과 그 대책

B 의 불변식은 하나입니다.

> **`torch.bool` 로 태그된 텐서의 `U8` 바이트는 `0` 또는 `1` 이다.**

깨지면 조용히 틀립니다 — 측정된 대로 `(2,3,0)` 에 §6.2 의 규칙을 적용하면:

```
ones - (2,3,0)        = [255, 254, 1]    <- torch.bool 은 [0, 0, 1]   (언더플로)
maximum((2,3,0), 1)   = [2, 3, 1]        <- torch.bool 은 [1, 1, 1]
(2,3,0).to(i64).sum   = 5                <- torch.bool 은 2
```

다행히 **불변식을 깰 수 있는 입구가 적고, 전부 우리 코드입니다:**

| 입구 | 상태 |
|---|---|
| candle `cmp` (`eq`/`ne`/`lt`/…) | **안전** — `u8::from(x==y)` 로 0/1 보장 (§1.2) |
| §6.2 의 합성 규칙들 | **안전** — 0/1 입력에 닫혀 있음 (위 표에서 확인) |
| `full.default` 의 bool fill | shim 이 씀 — 0/1 로 쓰면 됨 |
| `_to_copy` (`uint8 → bool`) | **여기가 위험.** torch 는 `!= 0` 으로 정규화한다(§2.6). shim 도 정규화해야 하며 `U8` 을 그대로 재태그하면 안 된다 |
| `_tensor_from_flat` (TORCH_C §2 의 뒷문) | **여기가 위험.** 임의 값이 들어온다. 삭제 대상이지만 그전까지는 bool 태그를 붙이지 말 것 |

권고: **두 가지를 둡니다.**

1. bool 태그 텐서를 만드는 **단일 생성자**를 두고, 그 안에서만 태그를 붙입니다
   (`device` 의 `resolve()` 와 같은 "한 지점" 원칙 — TORCH_C §1).
2. 환경 변수로 켜는 **불변식 검사**를 둡니다 (예: `BRAINWAVE_CHECK_BOOL=1` 이면 bool 태그
   텐서 생성 시 `max_all() <= 1` 을 확인, 아니면 패닉이 아니라 명시적 오류).
   `max_all` 은 `U8` 에서 동작하는 것을 확인했으므로(§6.2) O(n) 한 번입니다.

이 두 가지로 **B 의 조용한 실패 지점이 하나로 줄고, 그 하나는 켤 수 있는 검사로 시끄러워집니다.**

### 6.4 코드에 무엇이 바뀌는가 (구현하지 않음, 형태만)

지금 `rust/torch_c/src/dtype.rs:20-22` 는 candle 의 dtype 을 그대로 감쌉니다:

```rust
pub struct PyDtype { inner: DType }        // DType = candle_core::DType
```

B 는 이 한 줄이 바뀐다는 뜻입니다 — `inner` 가 shim 소유 열거형이 되고, 거기에
`storage_dtype() -> candle_core::DType` 이 붙습니다. `torch_name` (`dtype.rs:35-49`),
`__eq__` (`:67`), `__hash__` (`:74`), `is_signed` (`:85`), `itemsize` (`:89`), `register`
(`:96-113`) 가 그 열거형 위에서 다시 쓰입니다. `device.rs` 가 이미 하는 것과 같은 모양입니다.

**이 문서는 그 변경을 하지 않습니다.** `rust/torch_c/` 는 다른 작업이 동시에 쓰고 있습니다.

---

## 7. 틀렸을 때 어떻게 드러나는가 — 이 판단의 핵심

| 선택지 | 실패 방식 | 근거 |
|---|---|---|
| **A (별칭)** | **조용함** | §3.3: 예외·NaN 없이 64 배 오차. §3.2: `sum` 이 300 대신 44. §3.4: 마스크 가중치 2 배 |
| **B (태그)** | **시끄러움** — 규칙 없는 조합은 `NotImplementedError` 로 이름을 댄다 | TORCH_C §1 의 단일 관문 |
| B 의 예외 | 불변식 위반만 조용함 | §6.3 — 입구 5 개 중 2 개, 환경 변수 검사로 시끄럽게 만들 수 있음 |
| **C (candle 포크)** | **시끄러움** — 컴파일 에러 또는 `UnsupportedDTypeForOp` | §1.3 |
| **D** | 해당 없음 (§4 가 닫음) | |

### A 의 진짜 문제 — torch 의 방어막을 지운다

이 판단에서 가장 중요한 관찰입니다. **`bool` 과 `uint8` 이 갈라지는 지점마다 torch 가 이미
방어막을 세워 두었고, 그 방어막은 전부 dtype 태그에 걸려 있습니다.**

| torch 가 막는 것 | 어떻게 | 별칭하면 |
|---|---|---|
| `masked_fill(uint8 mask)` | **`RuntimeError`** | 통과. 조용히 다른 답 |
| `bool - bool` | **`RuntimeError`** ("use `^`") | `uint8` 뺄셈. `1-0=1`, `0-1=255` |
| `-bool` | **`RuntimeError`** ("use `~`") | `[255, 0, 255]` |
| `bool.mean()` | **`RuntimeError`** | f32 승격 후 계산 |
| `x[uint8]` | `UserWarning` (폐기 예정) | 조용 |
| `torch.where(uint8)` | `UserWarning` (폐기 예정) | 조용 |

**별칭은 위험을 새로 만드는 것이 아니라, 이미 있던 여섯 개의 경보를 끄는 것입니다.**
그러므로 "일단 별칭하고 문제가 생기면 고친다" 는 성립하지 않습니다 — 문제가 생겼다는 신호가
그 별칭 때문에 사라지기 때문입니다.

### 반대로 B 를 잘못 골랐을 때의 비용

B 가 틀린 선택이었다면 드러나는 방식은 **작업량**입니다 — 불리언 op 규칙을 하나씩 쓰다가
"이것은 candle 에 dtype 을 넣는 게 싸다" 는 결론이 나는 것. 그 전환은 §6.1 대로 국소적이고,
그동안 만든 테스트는 그대로 남습니다. **비용이 검증 시간이지 깨진 수치가 아닙니다.**

---

## 8. 미확인 항목

| 항목 | 상태 |
|---|---|
| Metal / CUDA 백엔드의 `where_cond` dtype 허용 범위 | **미확인** — CPU (`cpu_backend/mod.rs:2735-2751`) 만 읽었다. `metal_backend/mod.rs:846-886` 에 별도 매치가 있고 `crate::bail!("Metal where_cond {left:?} {right:?} not implemented")` 로 끝나는 분기가 보이나 확인하지 않았다. 이 프로젝트는 CPU 만 쓰므로 지금은 무관 |
| `U8` 덧셈이 디버그 빌드에서 패닉하는지 | **미확인** — 릴리스 빌드에서 감기는 것만 확인(300→44). 디버그면 Rust 규칙상 패닉해야 하나 돌려보지 않았다. 배포는 릴리스이므로 판단에 영향 없음 |
| 실제 프롬프트 길이가 256 을 넘는지 | **미확인** — §4 의 장난감 모델에서는 `(1, 11)` 이 최대. 오버플로가 *실제로* 밟히는지는 실제 모델·프롬프트로 재야 한다. 다만 **밟히지 않는다는 보장이 없다는 것**이 논거이고, 밟히면 조용하다 |
| `torch.bool` 의 C++ 레벨 정규화 규약 | **부분 확인** — 파이썬에서 관측한 동작(§2.6)만 근거. `c10::load<bool>` 의 소스는 읽지 않았다 |
| B 로 갔을 때 15 개 op 전부가 §6.2 의 규칙으로 닫히는지 | **미확인** — `isin.Tensor_Tensor` 와 `_local_scalar_dense(bool)` 의 구현 형태는 검토하지 않았다 |
| `bitwise_and.Tensor` 의 `bool,int64 → int64` 조합 | **미해결** — candle 에 정수 bitwise 가 없다(§1.6). bool 결정과 무관하게 커스텀 커널이 필요하다 |
| candle 상류에 bool dtype PR/이슈가 있는지 | **미확인** — 웹 조회를 하지 않았다 |
| `_scaled_dot_product_flash_attention_for_cpu` 의 마스크 인자 | **관측되지 않음** — §4 트레이스에서 입출력 모두 `float32` 였다. 마스크를 안 받은 것인지 이 구성에서만 그런 것인지 확인하지 않았다 |

---

## 9. 재현 방법

실험 코드는 전부 `/Volumes/macMini/caches/bool-probe/` 에 있습니다 (저장소 밖).

```bash
VENV=/Volumes/macMini/caches/spike-venv/bin/python
P=/Volumes/macMini/caches/bool-probe

# §2  torch.bool 대 uint8 전수 비교
$VENV $P/probe.py

# §3.1 §3.2 §3.4  어긋남 재현 (인과 마스크 NaN, sum 오버플로, 마스크 결합)
$VENV $P/repro.py

# §3.3  조용히 틀리는 사례 (평균 풀링, 상대오차 64 배)
$VENV $P/silent.py

# §4  이 모델이 불리언을 지나는 15 개 op — dtype 까지 기록
$VENV $P/trace_bool.py
$VENV $P/trace_shapes.py      # 불리언 텐서의 실제 모양

# §1 §6.2  candle 쪽 확인 (비교 연산 dtype, where_cond, U8 누산, 구현 후보)
export PATH="$HOME/.cargo/bin:$PATH"
cd $P/candle-probe && CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target cargo run --release
```

candle 소스:
`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/candle-core-0.11.0/`

| 참조 | 파일:행 |
|---|---|
| `DType` 열거형 (bool 없음) | `src/dtype.rs:9-38` |
| 비교 연산 = `U8` 반환 | `src/tensor.rs:1121-1173` (주석 `:1124`) |
| 비교 커널 = `u8::from(...)` 0/1 | `src/cpu_backend/mod.rs:62-83` |
| `where_cond` 의 dtype 허용 목록 | `src/cpu_backend/mod.rs:2735-2751` |
| `is_true()` = `!= 0` | `src/dtype.rs:239-268` |
| `where_cond` 브로드캐스팅 없음 | `src/tensor.rs:1565-1567`, `:618-627` |
| `U8` 로 누산하는 `sum` | `src/cpu_backend/mod.rs:231-306`, `src/cpu/kernels.rs:41-46` |
| 정수 단항 op = `todo!()` 패닉 | `src/op.rs:464-503`, `:589` |
| bitwise 부재 | `grep -rin bitwise --include="*.rs" .` → 0 건 |
