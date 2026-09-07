# 텐서 메서드 — `x * y` 와 `x.sum()` 을 여는 문

`docs/bindings/OVERLOAD.md` 는 `torch.<op>` 13 개를 뚫었고, 마지막 절(§9 항목 8)에 이렇게 적었습니다:

> `TensorBase` 50 개 — **손대지 않았습니다.** `x + y` · `x.sum()` 같은 메서드 경로는 여전히 이름을 댑니다.

```
x * y      ->  NotImplementedError: TensorBase.__mul__      (막혀 있던 것)
x.sum()    ->  NotImplementedError: TensorBase.sum
x.item()   ->  NotImplementedError: TensorBase.item
```

이 문서는 그 50 개를 여는 작업의 기록입니다. **덤으로, `from_config` 을 향해 다섯 개의 벽을 더
밟았고 여섯 번째(RNG)에서 멈췄습니다** — 지시받은 대로.

판정은 전부 종료 코드입니다. 출력 grep 은 쓰지 않았습니다.

---

## 0. 한눈에

| | 이전 (OVERLOAD.md §0) | 지금 |
|---|---|---|
| `C_SURFACE.md` §4 의 50 멤버 | 7 (`shape`·`dtype`·`device`·`dim`·`ndim`·`numel`·`size`, Rust 네이티브) | **47 / 49** (§1) |
| 구현된 aten op | 20 | **73** (`_aten_implemented()` 60 + 대기 13) |
| 골든 하네스 | 490/490, ops covered=**19** | **1027/1027, ops covered=60** |
| 골든 케이스 빌더 대기 | 0 | **2** (`normal_` · `uniform_` — RNG, §7) |
| 골든 자가검사 (`--inject-fault`) | 1/1/1 | **1/1/1** (그대로) |
| 호스트 스모크 | 34/34 | **54/54** |
| 스키마 검증 | 45/45 (`overloads.json`) | **127/127** (두 표 모두) |
| 사용자 API 대조 (상류 torch vs shim) | 49/49 | **107/108** (§6 에 나머지 1) |
| 엄격 `import torch` · `transformers` | 0 · 0 | **0 · 0** |
| 세 타깃 빌드 | 0 | **0** |
| `from_config` | `_C._dynamo` 에서 정지 | **`nn.Linear` 안의 `uniform_` 까지 전진** (§7) |

> **Correction (문서 감사, 2026-09):** "`nn.Linear` 안의 `uniform_` 까지 전진"도 이제
> 낡았습니다 — §7 정정 참고. `nn.Linear(4, 3)` 이 지금은 그냥 성공합니다.

`__class__` 는 `C_SURFACE.md` §4 가 이미 "잡음" 이라고 표시한 상속 던더이므로 분모는 49 입니다.

---

## 1. 50 개 — 무엇을 어떻게 했나

네 갈래로 나뉘고, **그 갈래가 곧 "어떻게 했나" 입니다.**

| 갈래 | 49 개 중 | 어디에 |
|---|---|---|
| A. 표 기반 오버로드 해석 | **34** | `rust/torch_c/src/methods.json` + `bootstrap.py::_tensor_method` |
| B. 파이썬 레벨로 직접 작성 | **5** (`to` · `float` · `long` · `item`/`__bool__` · `__getitem__`) | `bootstrap.py::_install_tensor_{conversions,scalars,indexing}` |
| C. Rust 네이티브 (아이덴티티) | **7** (`shape` · `dtype` · `device` · `dim` · `ndim` · `numel` · `size`) | `tensor.rs` — 이전부터 있던 것 |
| D. autograd 모양 (**때움**) | **3** (`requires_grad` · `requires_grad_` · `grad_fn`) | `bootstrap.py::_install_autograd_shape` — §5 |

`methods.json` 자체는 이름 **48 개 · 스키마 80 개**입니다 — 50 목록 밖의 이름(`mul` · `div` ·
`bitwise_and` 처럼 던더와 짝을 이루는 메서드 철자, `rsqrt`, `matmul`)도 함께 열었기 때문입니다.
같은 커널에 닿더라도 **철자마다 바인딩이 따로**이므로 각각 표에 있어야 합니다.

실측 (진짜 `import torch` 위에서 49 개를 전부 호출):

```
WORKING 47/49
  BLOCKED normal_  -> aten op not implemented in torch._C shim: aten.normal_.default
  BLOCKED uniform_ -> aten op not implemented in torch._C shim: aten.uniform_.default
```

**막힌 둘은 정확히 RNG 쌍입니다.** 지시대로 여기서 멈췄습니다 (§7).

### A. 표 기반 — 오버로드 기계를 그대로 재사용했다

`methods.json` 은 `overloads.json` 의 형제입니다. 같은 알고리즘(`PythonArgParser::raw_parse`),
같은 `_TypeChecker`, 같은 "순서가 곧 의미론" 규칙. 차이는 **하나뿐**입니다:

> 메서드의 수신자는 어떤 시그니처를 보기도 전에 이미 정해져 있다.

그래서 `_Overloads(self_bound=True)` 로 만들고, `args[0]` 에 수신자를 넣은 뒤
**위치 인자를 세는 부분만 하나 건너뜁니다.**

```python
key, bound = entry.resolve((self,) + args, kwargs)
return dispatch(key, **bound)      # <- 그 문. 다른 경로 없음.
```

**`_aten_dispatch` 단일 진입 원칙은 깨지지 않았습니다.** 메서드는 *어떤 키인가*만 정하고,
계산하지 않으며, 다른 경로로 커널에 닿지 않습니다. `TensorBase` 타입 자체에는 여전히 산술이
없습니다 — `aten.rs` 의 파일 머리 주석이 요구하는 그대로입니다.

### 왜 메서드에 별도의 표가 필요한가

`torch.mul(a, b)` 와 `a.mul(b)` 는 같은 커널에 닿지만 **상류에서 다른 바인딩**입니다 —
`THPVariable_mul`(`TensorBase` 의 메서드)과 `_VariableFunctions` 항목은 별개의 C 함수이고
각자의 시그니처 목록을 갖습니다. `C_SURFACE.md` 가 두 표면을 따로 센 것(50 대 13)이 바로 그
이유이고, `overloads.json` 에 이름을 넣어도 메서드는 열리지 않습니다 — 아무도 거기를 보지
않습니다.

### 표에 필요했던 규칙 하나 — 크기 있는 int 리스트

`_TypeChecker` 에 규칙을 하나 더해야 했습니다. `sum.dim_IntList(Tensor self, int[1]? dim, ...)`
의 `int[1]` 은 **맨 int 하나도 받습니다** — `FunctionParameter::check` 의

```cpp
// if a size is specified (e.g. IntArrayRef[2]) we also allow passing a single int
return size > 0 && THPUtils_checkLong(obj);
```

이 없으면 `x.sum(0)` 이 **"매칭되는 오버로드 없음"** 으로 떨어집니다 — 맞는 답의 모양을 한 틀린
답, `OVERLOAD.md` §4 가 지목한 바로 그 실패 양식입니다. `_decompose_type` 이 대괄호 안의 숫자를
돌려주게 하고, 바인딩할 때 한 원소짜리 튜플로 정규화합니다(커널이 스키마대로 리스트를 받게).

기존 `overloads.json` 에는 `int[N]` 철자가 없으므로 **이 변경으로 기존 동작은 하나도 바뀌지
않습니다** (골든 490 → 1027 로 늘어나기만 했고 회귀 0).

### 가변 인자 규칙은 `self` 를 센 뒤에 적용된다

`x.view(2, 3)` 은 `view([2, 3])` 이어야 합니다. torch 의 전제는 "시그니처의 위치 인자가 **하나**"
인데, `view` 가 그 조건을 만족하는 것은 **`self` 를 셈에서 뺀 뒤**입니다. 회귀 테스트가 있습니다
(`test_varargs_rule_applies_after_self_is_bound`).

### 표의 순서는 어디서 왔나 — 읽지 않고 관측했다

`OVERLOAD.md` §3 과 같은 방법입니다. 벤더링 트리 자신의 `torch/_C/__init__.pyi`(`class
TensorBase`)에 선언된 `@overload` 순서를 기준으로 삼고, **torch 2.13.0 에 `TorchDispatchMode`
로거를 걸어 실제 호출이 어느 키로 가는지 대조**했습니다. 둘이 어긋난 자리가 §4 입니다.

### B. 파이썬 레벨 — 상류도 오버로드 집합이 아닌 것들

`torch.tensor` 가 `OVERLOAD.md` §6.7 에서 유일한 예외였는데, 여기서 다섯 개가 더 나왔습니다.
**전부 "상류의 바인딩 자체가 평범한 오버로드 집합이 아니다" 라는 같은 이유입니다.**

| 이름 | 상류가 무엇을 하는가 | 여기서 |
|---|---|---|
| `to` · `float` · `long` · `bool` … | `THPVariable_to` 가 인자를 읽고 여러 aten 호출 중 하나를 고름 | `_to_copy.default` 하나로 보냄. 바뀔 게 없으면 **`self` 를 그대로 돌려줌** |
| `item` · `__bool__` | `_local_scalar_dense` 로 내려가고 numel 검사는 그 위에 있음 | 같은 분할. 메시지는 torch 의 것 |
| `__getitem__` | `THPVariable_getitem` 이 인덱스를 걸어가며 **여러 개**의 aten 호출을 냄 | 같은 걸음 (§3) |

**`to` 가 표로 안 되는 이유는 실측입니다.** torch 의 파서 시그니처는
`to(Device device=None, ScalarType dtype=None, ...)` 인데 aten 쪽 `to.device` 는 `device` 와
`dtype` 을 **둘 다 필수**로 받습니다. 그래서 `x.to('cpu')` 는 파서 시그니처에는 바인딩되고 어떤
aten 스키마에도 바인딩되지 않습니다 — 표로 만들면 상류가 받는 호출에 "매칭 없음" 을 답하게
됩니다. `OVERLOAD.md` §9 항목 7 이 "미확인" 으로 남겨둔 가정("파이썬 시그니처가 aten 스키마와
어긋나는 op 이 더 있는가")의 **첫 실제 사례**입니다.

### C. Rust 네이티브 — 디스패처를 지나지 않는 아홉

`shape` · `dtype` · `device` · `dim` · `ndim` · `numel` · `size` 는 `TensorImpl` 의 메타데이터를
읽을 뿐 aten 을 지나지 않습니다. `tools/golden/cases.py` 의 모듈 주석도 같은 것을 독립적으로
관측했습니다("Nine of the 50 names never reach the ATen dispatcher at all"). 이 일곱에
`grad_fn` · `requires_grad` 를 더한 아홉이 그 목록이고, 뒤의 둘은 §5 입니다.

---

## 2. aten 커널 — 53 개를 새로 넣었고 41 개가 상류와 대조된다

`_aten_implemented()` 가 19 → **60** 이 되었고, 골든 하네스는 `490/490, ops covered=19` 에서
**`1027/1027, ops covered=60`** 이 되었습니다.

**하네스에 케이스 빌더가 이미 준비되어 있었습니다.** 첫 실행에서

```
PENDING: 43 case builder(s) registered for ops not yet in _aten_implemented()
         -- waiting, not failing: [...]
```

이 나왔습니다 — 동시에 도는 다른 작업이 이 43 개를 **미리 심어 두었습니다.** 제가 구현한 것과
39 개가 겹쳤고, 나머지 넷 중 `bitwise_and.Scalar` · `bitwise_or.Scalar` 는 빌더가 있는 것을 보고
**추가로 구현**했습니다(남은 둘이 §7 의 RNG 쌍입니다). 그래서 이 41 개를
`IMPLEMENTED_AWAITING_GOLDEN` 에 파킹하는 대신 **`IMPLEMENTED` 로 올려 실제로 상류와 대조**
시켰습니다. `OVERLOAD.md` §7.1 이 "고치는 방법은 케이스 빌더 하나와 한 줄 이동" 이라고 적은 그
이동입니다.

새 커널은 모두 **53 개**입니다: 골든이 대조하는 41 개(`IMPLEMENTED` 19 → 60)와, 하네스에 아직
빌더가 없어 `IMPLEMENTED_AWAITING_GOLDEN` 에 남은 12 개
(`add.Scalar` · `any.dims` · `contiguous.default` · `div.Scalar` · `fill_.Tensor` ·
`masked_fill.Tensor` · `matmul.default` · `max.other` · `mul.Scalar` · `reshape.default` ·
`sub.Scalar` · `zeros.default`). 뒤의 12 개는 **동작하지만 커버리지 숫자에서 빠집니다** —
`OVERLOAD.md` §7.1 이 정한 대로 과장이 아니라 축소 방향입니다.

### 새 커널이 대조에서 잡힌 것 — 네 가지, 전부 고쳤다

하네스를 켜자마자 **20 건이 빨갛게** 났습니다. 전부 실제 결함이었습니다.

| # | 증상 | 원인 | 조치 |
|---|---|---|---|
| 1 | `masked_fill` · `index.Tensor` 의 bool 케이스 10 건이 케이스 구성 단계에서 죽음 | `_tensor_from_flat` 이 `torch.bool` 을 **거부**하고 있었고(BOOL.md §6.3), 하네스는 모든 피연산자를 이 함수로 만듦 | **정규화하도록 바꿈** — §2.1 |
| 2 | `cumsum(int64/bfloat16)` 4 건이 `unsupported dtype I64 for op matmul` | candle 의 `cumsum` 은 **삼각행렬 matmul** 이라 gemm 이 있는 dtype 에만 존재 | 직접 누산 — §2.2 |
| 3 | `fill_(float16, 1e6)` → `inf`, `fill_(int32, 2**31)` → 감김. torch 는 거부 | `checked_convert` 를 `fill_` 에 안 걸었음 | 걺. `fill_` 이 사실 상류의 numel==1 구멍이 **사는** 자리 |
| 4 | (사용자 API 대조에서) `len(x)` 가 죽음 | `torch/_tensor.py::__len__` 이 `_C._get_tracing_state()` 를 부름 | `None` 을 답하게 함 (§4) |

**이 넷은 전부 "혼자 돌리면 안 보이는" 것들입니다** — 값이 그럴듯하게 나오거나(1·3), 특정
dtype 에서만 터지거나(2), 파이썬 계층을 한 겹 지나야 닿습니다(4).

### 2.1 `_tensor_from_flat` 이 `torch.bool` 을 받게 됐다 — 규칙 완화가 아니라 이전

`BOOL.md` §6.3 은 이 함수를 "`torch.bool` 불변식이 조용히 깨질 수 있는 두 경로 중 하나" 로
지목했고, 그래서 태그를 **거부**하고 있었습니다. 이제 **정규화**합니다.

```rust
let bytes: Vec<u8> = values.iter().map(|v| u8::from(*v != 0.0)).collect();
PyTensorBase::boolean(Tensor::from_vec(bytes, shape, &device)?)
```

**불변식이 약해진 것이 아니라, 지켜지는 장소가 옮겨진 것입니다.** `OVERLOAD.md` §6.7 이
`_tensor_new_from_data` 를 추가할 때 쓴 논증과 같습니다 — 모든 바이트가 `!= 0` 을 거쳐
`PyTensorBase::boolean` 이라는 **단 하나의 태깅 생성자**로 들어가므로 0/1 은 **구성으로**
성립합니다. `!= 0` 은 또한 torch 가 bool 텐서의 *읽기*에 대해 보장하는 것이기도 합니다
(BOOL.md §2.6).

거부는 반대 방향으로 하중을 받고 있었습니다: 하네스가 마스크를 만들 수 없어서 **`masked_fill`
과 `index.Tensor` 를 상류와 대조할 수 없었습니다.** 골든 대조가 불가능한 op 이 정규화하는
생성자보다 나쁩니다.

> `tools/golden/cases.py` 의 주석이 이 거부를 우회하는 방법을 길게 설명하고 있습니다(빌더를
> 람다 안으로 미루기). **그 우회는 이제 필요 없습니다** — 다만 하네스 파일은 다른 작업이
> 쥐고 있어 한 줄도 고치지 않았고, 우회한 채로도 전부 통과합니다.

### 2.2 `cumsum` 은 candle 의 것을 쓸 수 없다

```rust
// candle_core::Tensor::cumsum -- 삼각행렬과의 matmul
```

`int64` · `bfloat16` 에는 candle gemm 이 없습니다. 그래서 직접 누산합니다. **부동소수는 `f64`
로 누산**하는데, torch 의 CPU 커널은 축소 정밀도 float 을 `float`(`acc_type<BFloat16>`)로
누산하고 마지막에 한 번 좁힙니다 — 같은 모양의 계산에 **더 넓은 누산기**이므로 긴 `bfloat16`
누적에서 마지막 비트가 어긋날 수 있습니다. **더 정확한 방향**입니다. §6 에 남깁니다.

### 2.3 승격은 여전히 하지 않는다

두 텐서의 dtype 이 다르면 `same_dtype` 이 **양쪽 이름을 대고 거부**합니다. torch 는 승격합니다.
`DESIGN.md` §5 가 candle 의 주된 위험으로 꼽은 "조용한 수치 드리프트" 를 만들지 않기 위한
기존 규칙이고, 새 커널 전부가 그대로 따릅니다.

**반면 파이썬 스칼라는 승격 규칙을 그대로 재현합니다** — torch 의 "wrapped number" 규칙이고,
실측했습니다.

```
float32_t * 2    -> float32     (파이썬 정수는 같은 카테고리의 텐서를 넓히지 않는다)
int64_t   * 3    -> int64
int64_t   * 3.0  -> float32
int64_t   / int64_t -> float32  (참 나눗셈은 예외 없이 float 이다)
```

---

## 3. `__getitem__` — 하나의 op 이 아니라 걸음이다

상류의 인덱싱은 aten 호출 **하나가 아닙니다.** 인덱스를 걸어가며 정수에는 `select.int`,
슬라이스에는 `slice.Tensor`, `None` 에는 `unsqueeze` 를 내고, 텐서 인덱스가 있으면 마지막에
`index.Tensor` 를 한 번 냅니다. torch 2.13.0 에서 실측:

```
f[0]        -> [select.int]
f[0, 1]     -> [select.int, select.int]
f[:, 1]     -> [select.int]          (전체 슬라이스는 아무것도 안 낸다)
f[0:1]      -> [slice.Tensor]
f[None]     -> [unsqueeze]
f[bool_t]   -> [index.Tensor]
```

그래서 `getitem` 이라는 op 을 발명하지 않고 **그 걸음을 재현**했습니다. 각 걸음은 전부
`_aten_dispatch` 를 지납니다.

**하지 않은 것: 기본 인덱싱과 고급 인덱싱의 혼합.** `x[mask, 0:1]` 처럼 텐서 인덱스와 자명하지
않은 슬라이스가 섞이면 **이름을 대고 거부**합니다 — 상류는 기본 인덱싱을 먼저 적용한 뒤
`index.Tensor` 를 거는데, 그 합성을 재지 않았습니다. 인덱스 텐서가 **둘 이상**인 경우
(`x[i, j]`)도 같은 이유로 거부합니다(상류는 인덱스끼리 브로드캐스트합니다).

여기서 함정이 하나 있었습니다. **인덱스 튜플 안에서 `==` 를 쓰면 안 됩니다** — `TensorBase.__eq__`
가 이제 마스크를 돌려주므로 `tuple.index(Ellipsis)` 나 `item == slice(None)` 이 원소마다 비교하며
지나가다 텐서에 걸립니다. 전부 `is` 와 필드 검사로 씁니다.

---

## 4. `TensorBase.__eq__` 를 켰다 — `UNSAFE_DUNDERS` 규칙의 재해석

`bootstrap.py` 의 `UNSAFE_DUNDERS` 는 `__eq__` · `__ne__` 를 **부재**로 남겨 두고 있었습니다.
그 규칙의 근거는 "**터지는** 대역이 dict/set 사용을 깨뜨린다" 입니다. 마스크를 돌려주는
**동작하는** 구현은 그것과 다른 것이고, 상류가 가진 바로 그것입니다. 그래서 켰습니다.

**대신 연산자 던더는 바인딩에 실패하면 `NotImplemented` 를 답합니다.** 이건 예의가 아니라
하중입니다 — 벤더링 트리는 여러 곳에서 텐서를 문자열이나 `None` 과 비교하고, `__eq__` 가
사양해야만 파이썬이 자신의 아이덴티티 비교로 폴백합니다. `TypeError` 를 냈으면
`x == "cpu"` 가 상류에서는 `False` 인데 여기서는 크래시였을 것입니다. 회귀 테스트가 있습니다.

---

## 5. 구현 / off-switch / 때움 — 구분

지시대로 나눕니다. **아래 두 항목 외에는 전부 구현입니다** — §1 · §2 에서 "구현" 이라고 쓴 것은
요구된 동작을 실제로 합니다.

### 5.1 때움 — `requires_grad` 는 저장되고, 아무것도 읽지 않는다 ⚠

`from_config` 은 다른 어떤 흥미로운 것보다 먼저 `TensorBase.requires_grad_` 에 닿습니다
(FROM_CONFIG.md §2.2, 2 회). 선택지는 **거기서 멈추거나, 무력한 플래그를 지니거나** 였습니다.

플래그를 지니되, **경계를 보이는 자리에 그었습니다.**

| | |
|---|---|
| `requires_grad` | 설정한 값을 저장하고 그대로 보고한다. **아무것도 읽지 않는다** |
| `grad_fn` · `grad` | 항상 `None` — 이건 **참**이다. 그래프 노드가 만들어진 적이 없다 |
| `backward()` | **터지는 스텁 그대로.** 플래그가 뭔가를 뜻한다고 믿는 코드는 이름을 대고 실패한다 |

이것은 `OVERLOAD.md` §4 가 팩토리의 `requires_grad=True` 를 거부한 것과 긴장 관계에 있습니다.
그 거부는 **그대로 유지**됩니다 (`torch.ones(2, requires_grad=True)` 는 여전히 거부, 스모크
테스트 `test_requires_grad_is_refused_not_ignored` 가 지킵니다). 차이는 팩토리에서는 거부해도
길이 막히지 않지만 `requires_grad_` 에서는 막힌다는 것이고, **그래서 이것이 때움이라고 적습니다.**

### 5.2 때움 — grad 모드 플래그 ⚠

`_set_grad_enabled` 는 `from_config` 트레이스에서 **84 회**로 가장 많이 불립니다 —
`@torch.no_grad()` 로 감싼 모든 초기화가 진입/탈출마다 한 번씩 뒤집습니다. 거부한 채로 둘 수
없습니다.

**구현된 것은 플래그이지 플래그의 의미가 아닙니다.** 상태는 정확히 왕복합니다(`no_grad()` 가
이전 값을 읽고, `False` 로 두고, 되돌리고, 저장한 것을 되받습니다) — `torch/autograd/grad_mode.py`
가 이 계층에 요구하는 것의 전부입니다. 없는 것은 그 플래그가 다스릴 대상입니다.
`_is_multithreading_enabled` · `_is_grad_layout_enforcement_enabled` 도 같은 모양입니다.

### 5.3 구현이되 상류와 다른 것 — in-place 는 저장소에 쓰지 않는다 ⚠

`fill_` · `copy_` 는 래퍼의 텐서를 **교체**하고, 저장소에 쓰지 않습니다
(`PyTensorBase::replace_with`). 관측 가능한 차이가 있습니다.

```
같은 파이썬 객체를 통한 변이   p.data.fill_(0)      상류와 같다 (`.data` 가 self 를 돌려주므로)
detach()/뷰가 만든 별칭        y = x.detach()
                              y.fill_(0)            상류는 x 도 바뀐다. 여기서는 안 바뀐다
```

계측된 `from_config` 경로는 항상 같은 객체를 통해 변이합니다. **다른 경로가 별칭 쓰기에
의존하면 여기서 조용히 갈립니다** — §6 에 미확인으로 남깁니다.

### 5.4 off-switch — 없음

이번 작업에서 이름을 **빼는** 방식(VENDOR.md 벽 11)은 하나도 쓰지 않았습니다.

### 5.5 존재만 하면 되는 것 — `_DISCOVERED_RETURNS` 에 12 개 추가

지시 항목 3 의 "동작 없이 존재만 하면 되는 것" 입니다. 전부 **참인 답**이지 대역이 아닙니다.

| 이름 | 답 | 왜 참인가 |
|---|---|---|
| `_log_api_usage_once` · `_log_api_usage_metadata` | `None` | 상류는 사용량 카운터를 올리고 아무것도 안 돌려준다. 카운터가 없다 |
| `_has_torch_function{,_unary,_variadic}` | `False` | "어떤 인자의 타입이 `__torch_function__` 를 오버라이드하는가" — 이 트리의 추론 경로에는 없다 |
| `_is_torch_function_{enabled,mode_enabled,all_disabled}` · `_len_torch_{function,dispatch}_stack` | `False` / `0` | 모드가 푸시된 적이 없으므로 스택이 비어 있다 |
| `_get_tracing_state` | `None` | TorchScript 트레이서가 없다. `len(tensor)` 가 이 경로에 있다 |

`_log_api_usage_once` 는 `nn.Module.__init__` 의 **첫 줄**이라 모든 모듈 생성이 부릅니다 —
`import torch` 다음 벽이 바로 이것이었습니다.

---

## 6. 상류와 다른 지점 · 미확인

| # | 항목 | 상태 |
|---|---|---|
| 1 | **`cos` 의 마지막 비트** | 사용자 API 대조 108 건 중 **유일하게 어긋난 1 건**. `cos(1.0)` 이 `0.5403022766` 대 torch 의 `0.5403023362` — float32 1 ULP. candle 의 libm 대 torch 의 SLEEF 차이. 골든 하네스는 허용오차 안이라 통과 |
| 2 | **`__matmul__` 이 대는 키** | `aten.matmul.default`. 상류는 파서 수준에서 같은 것을 고르지만 디스패처에는 `mm.default`(2-D) 또는 `expand/view/bmm/_unsafe_view`(배치) 가 기록된다 — `matmul` 이 composite 이기 때문. **답은 같고 트레이스가 다르다** (§6.5 의 `is_floating_point` 과 같은 종류) |
| 3 | **`mul.Scalar` 대 `mul.Tensor`** | `x * 2` 는 여기서 `mul.Scalar` 로 해석된다. 상류의 *파서*도 그렇지만 그 커널이 스칼라를 감싸 `mul.Tensor` 로 재디스패치하므로 `TorchDispatchMode` 에는 `mul.Tensor` 가 찍힌다. **비교 연산자는 반대** — `x == 2` 는 양쪽 다 `eq.Scalar` (실측) |
| 4 | **`reshape` 와 `view` 가 같은 커널** | 상류의 `view` 는 기존 stride 가 허용하지 않으면 **거부**하고 `reshape` 는 복사로 폴백한다. 여기서는 둘 다 복사하므로 **상류가 거부할 `view` 가 성공한다.** 안전한 방향이지만 차이 |
| 5 | **in-place 가 별칭에 안 보인다** | §5.3 |
| 6 | **`.data` 가 `self` 를 돌려준다** | 상류는 detach 된 별칭. 쓰기 관통은 같고, `requires_grad` 보고가 다르다 |
| 7 | **`max.dim` 의 반환 타입** | 상류는 `torch.return_types.max`(structseq). 여기서는 같은 필드 이름의 `collections.namedtuple` — 인덱스 접근과 `.values`/`.indices` 는 되고 타입은 다르다 |
| 8 | **`cumsum` 의 누산기 폭** | §2.2. `f64` 대 torch 의 `float` |
| 9 | **`__getitem__` 의 혼합·다중 인덱스** | §3. 거부한다. 상류의 합성/브로드캐스트 규칙 **미측정** |
| 10 | **`bitwise_*` 가 원소별 i64 경유** | 정확하지만 느리다. candle 에 비트 커널이 없다. 마스크 결합용이라 감수 |
| 11 | **`normal_` · `uniform_`** | ~~**미구현.** §7~~ **정정 (문서 감사, 2026-09): 이제 구현됨** — §7 정정 참고 |
| 12 | **`_to_copy` 의 device 인자** | `cpu` 만. 다른 device 는 `device_arg` 가 이름을 댄다 |
| 13 | **`import torch` 비용이 1.15 s → 0.35 s 로 줄었다** | OVERLOAD.md §10 의 1.15 s 와 **다른 조건에서 잰 것일 수 있습니다.** 이번 작업이 임포트를 빠르게 할 이유가 없으므로 이 숫자는 **개선의 근거로 쓰지 마십시오** — 미조사 |
| 14 | **기기 (Android · iOS)** | 이번에도 링크만 |
| 15 | **`promote` 의 비용** | op 결과마다 파이썬 타입 호출이 한 번 는다(§7.2). 측정하지 않았다 |

---

## 7. `from_config` 은 어디까지 갔나

**다섯 개의 벽을 밟았고 여섯 번째에서 멈췄습니다.** `OVERLOAD.md` §8 이 "다음 임계 경로는 op 이
아니라 `_C._dynamo` 다" 라고 적었는데, 그 벽은 **이번에 그냥 사라졌습니다** —
`import transformers` 가 이미 통과하고 있고(`probe.py --mode strict --target transformers` = 0),
`nn.Module` 생성 경로는 dynamo 를 지나지 않습니다.

| # | 벽 | 성격 | 처리 |
|---|---|---|---|
| 1 | `torch._C._log_api_usage_once` — `nn/modules/module.py:530`, `nn.Module.__init__` 첫 줄 | 존재만 하면 됨 | `None` (§5.5) |
| 2 | `nn/parameter.py:69` — `t._is_param = True` 가 `TensorBase` 에 `__dict__` 가 없다고 실패 | **`_C` 가 무엇을 돌려주는가** | §7.2 |
| 3 | `torch._C._has_torch_function_variadic` — `nn/init.py:597` | 존재+답 | `False` (§5.5) |
| 4 | `torch.is_grad_enabled` / `torch._C._set_grad_enabled` — `autograd/grad_mode.py:82` | 상태 플래그 | §5.2 |
| 5 | `torch._C._get_tracing_state` — `_tensor.py::__len__` | 존재+답 | `None` (§5.5) |
| 6 | **`tensor.uniform_(-bound, bound, generator=...)`** — `nn/init.py:616`, `kaiming_uniform_` 안 | **RNG** | **여기서 멈췄습니다** |

현재 도달점:

```
>>> import torch.nn as nn
>>> nn.Linear(4, 3)
  File ".../torch/nn/init.py", line 616, in kaiming_uniform_
    return tensor.uniform_(-bound, bound, generator=generator)
NotImplementedError: aten op not implemented in torch._C shim: aten.uniform_.default
```

**메서드 해석은 제대로 동작했습니다** — 이름이 아니라 **정확한 오버로드 키**를 댑니다.
`generator=` 키워드까지 스키마에 바인딩된 뒤 커널이 없어서 멈춘 것입니다.

지시대로 **여기서 억지로 아무 난수도 쓰지 않았습니다.** `FROM_CONFIG.md` §4.3 이
`normal_` 17 회 · `uniform_` 15 회로 요구를 계측해 두었고, RNG 는 다른 작업이 조사 중입니다.
골든 하네스에도 두 op 의 케이스 빌더가 이미 준비되어 대기 중입니다:

```
PENDING: 2 case builder(s) registered for ops not yet in _aten_implemented()
         -- waiting, not failing: ['aten.normal_.default', 'aten.uniform_.default']
```

**이 둘이 들어오는 순간 `nn.Linear` 가 만들어집니다.** 그 뒤의 벽은 아직 모릅니다.

> **Correction (문서 감사, 2026-09):** 이 둘이 들어왔습니다. `git log -S'"aten.uniform_.default"'
> -- rust/torch_c/src/aten.rs` 가 찾는 커밋은 `2d3663f` ("Feat: Port torch's CPU generator, and
> give `_C._dynamo` the two names that do work") — 커밋 메시지가 그대로 말하듯, candle CPU
> 백엔드가 시딩을 거부해서 상류 CPU generator(MT19937)를 이식한 결과입니다.
> `uniform_inplace`/`normal_inplace` 가 `aten.rs:10274`/`10350` 에 있고, 둘 다 현재
> `_aten_implemented()` 목록에 있습니다. 실측: `torch.nn.Linear(4, 3)` 이 지금은 그냥
> 성공합니다(§0 표의 "정지" 도 같이 낡았습니다). `docs/numerics/RNG.md` 가 이 이식을 자세히 다루는
> 문서로 보입니다(같은 커밋).
> <!-- DOCWATCH: op-implemented aten.uniform_.default -->
> <!-- DOCWATCH: op-implemented aten.normal_.default -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs uniform_inplace present -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs normal_inplace present -->

### 7.2 벽 2 — `_C` 는 `TensorBase` 가 아니라 `torch.Tensor` 를 돌려줘야 한다

이번 작업에서 **설계 전제를 건드린 유일한 변경**이라 따로 적습니다.

`torch/nn/parameter.py:54`:

```python
if type(data) is torch.Tensor or type(data) is Parameter:
    return torch.Tensor._make_subclass(cls, data, requires_grad)
# 아니면: 커스텀 텐서 경로 -- Parameter 가 아닌 것을 돌려주고,
# nn.Module.__setattr__ 이 그것을 평범한 속성으로 분류한다.
```

`torch.empty(...)` 가 `TensorBase` 를 돌려주면 이 분기가 **아래로** 갑니다. 결과는 조용합니다:
**모델은 만들어지고 파라미터가 하나도 없습니다.** `FROM_CONFIG.md` §2.3 이 "`TensorBase`·
`Parameter`·`Module` 이 전부 진짜 타입이어야 이 분류가 성립한다" 고 적은 것의 나머지 절반입니다.

상류의 방식을 그대로 따랐습니다: `THPVariable_Wrap` 이 `THPVariableClass`(= `torch._tensor.Tensor`)
를 인스턴스화합니다. 여기서는

- `_C._set_tensor_class(cls)` — `_initExtension` 이 부릅니다. **상류와 같은 순간**입니다
  (`torch/__init__.py:1931` 이 `Tensor` 를 임포트하고 `:2189` 가 `_initExtension` 을 부릅니다).
- `tensor::promote` — 디스패처의 **단 하나의 출구**에서, 정확히 `TensorBase` 인 결과만 감쌉니다.
  이미 서브클래스인 것은 건드리지 않으므로 in-place op 은 변이한 객체를 그대로 돌려줍니다.
- `TensorBase` 에 `#[new]` — PyO3 의 `tp_new` 는 **호출된 서브타입**으로 할당하므로
  `Tensor(base)` 는 `Tensor` 를, `Parameter(base)` 는 `Parameter` 를 만듭니다.
- `TensorBase._make_subclass` — 세 줄짜리 파이썬 함수. **`staticmethod` 여야** 하고
  (`torch.Tensor._make_subclass(cls, data, rg)` 로 불림), 안에서 `cls(data)` 가 아니라
  `TensorBase.__new__(cls, data)` 를 써야 합니다 — `Parameter.__new__` 가 호출자이므로
  `cls(data)` 는 재진입합니다(실측: `RecursionError`).

**골든 하네스는 영향을 받지 않습니다.** `tools/golden/loader.py` 는 `_C` 를 `torch` 패키지 없이
단독으로 로드하므로 클래스가 등록되지 않고 `promote` 는 항등입니다 — 1027/1027 이 그 증거입니다.

---

## 8. 검증

**전부 종료 코드입니다.**

| | 명령 | 결과 |
|---|---|---|
| 골든 하네스 | `tools/golden/compare.py` | **0** — **1027/1027, ops covered=60**, 대기 2 (이전 490/490, 19) |
| 골든 자가검사 | `--inject-fault value/shape/dtype` | **1 / 1 / 1** (의도대로) |
| 호스트 스모크 | `rust/torch_c/pytests/run.sh` | **0** — 54/54 (이전 34) |
| 스키마 검증 | `rust/torch_c/pytests/verify_schemas.py` | **0** — 127/127 (두 표) |
| 사용자 API 대조 108 케이스 | 상류 torch 와 shim 을 각각 돌려 diff | **107/108 동일** (§6-1) |
| 엄격 `import torch` | `probe.py --mode strict --target torch` | **0** |
| 기록 `import torch` | `probe.py --mode record --target torch` | **0** |
| 엄격 `import transformers` | `probe.py --mode strict --target transformers` | **0** |
| `aarch64-apple-darwin` | `cargo build --release` | **0** |
| `aarch64-linux-android` | `cargo ndk -t arm64-v8a` | **0** |
| `aarch64-apple-ios` | `cargo build --target aarch64-apple-ios` | **0** |

`probe.py` 는 **`TORCH_USE_RTLD_GLOBAL=1` 이 있어야 합니다** (VENDOR.md 벽 1). 없이 돌리면
`libtorch_global_deps.dylib` 를 못 찾아 실패하는데, 이건 회귀가 아니라 환경 변수 누락입니다 —
한 번 그렇게 오진할 뻔했습니다.

### 사용자 API 대조 — 하네스가 못 보는 것을 덮는다

골든 하네스는 **aten 키 수준**에서 대조합니다. 그것만으로는 "`x.sum(0)` 이 올바른 키로
해석되는가" 를 알 수 없습니다. 그래서 108 개의 같은 표현식을 상류 torch 와 shim 에서 각각
돌려 `(shape, dtype, values)` 를 diff 했습니다. `OVERLOAD.md` §5 의 49 케이스와 같은 장치입니다.

```
SUMMARY: 107/108 identical, 1 different
```

### 크기

| 타깃 | 이전 (OVERLOAD §10) | 지금 | 차이 |
|---|---|---|---|
| 호스트 | 2,092,512 | 2,769,472 | +676,960 |
| Android | 2,993,624 | 3,688,064 | +694,440 |
| iOS | 2,177,984 | 2,850,080 | +672,096 |

`methods.json` 은 8 KB 이고 파이썬 추가분도 수 KB 이므로, 증가분의 대부분은 **새로 참조된
candle 커널**(`broadcast_{mul,div,sub,eq,ne,lt,maximum}` · `where_cond` · `narrow` ·
`sum_keepdim` · `mean_keepdim` · `max_keepdim` …)이 dtype 마다 단형화된 것으로 보입니다. 셋이
같은 방향으로 비슷하게 움직인 것이 그 방증입니다. 분해는 **미측정**(§6-15 와 같은 종류).

iOS 는 여전히 `@rpath/Python.framework/Python` 로 링크됩니다 (올바른 모양).

### 임포트 비용

```
import _C  (단독)   20.5 ms      (이전 17.6~18.1 ms — 표가 하나 늘고 .so 가 커짐)
import torch        0.35 s       (§6-13 참고 -- 이 숫자를 개선으로 읽지 마십시오)
```

---

## 9. 재현

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target
cd /Volumes/macMini/thisisthepy/torchnative
PY=/Volumes/macMini/caches/spike-venv/bin/python

./vendor/install_shim.sh                                # 빌드 + 구멍에 넣기
$PY tools/golden/compare.py;                       echo "EXIT=$?"
$PY rust/torch_c/pytests/verify_schemas.py;        echo "EXIT=$?"
(cd rust/torch_c && ./pytests/run.sh);             echo "EXIT=$?"

TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/vendor $PY -c \
  "import torch
x = torch.full((2,2), 2.0)
print((x*x).tolist(), x.sum().item(), x[0].tolist(), (x==2.0).tolist())"

# 벽 6 -- RNG. 여기서 멈춥니다.
TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/vendor $PY -c "import torch.nn as nn; nn.Linear(4,3)"
```

iOS 빌드는 `TORCHNATIVE_PYTHON_FRAMEWORK_DIR` 도 필요합니다 (`build.rs` 가 없으면 그 자리에서
메시지를 내고 멈춥니다):

```bash
TORCHNATIVE_PYTHON_FRAMEWORK_DIR=/Volumes/macMini/caches/target-python/arm64-iphoneos \
PYO3_CONFIG_FILE=<suppress_build_script_link_lines=true 인 설정> \
PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
PYO3_CROSS_LIB_DIR=/Volumes/macMini/caches/target-python/arm64-iphoneos/lib \
cargo build --release --target aarch64-apple-ios
```

해석기가 무엇을 할 수 있는지는 물어보면 됩니다.

```python
>>> torch._C._shim_methods["sum"]
['aten.sum.default', 'aten.sum.dim_IntList']
>>> torch._C._shim_methods["__mul__"]
['aten.mul.Tensor', 'aten.mul.Scalar']
>>> len(torch._C._aten_implemented()), len(torch._C._aten_implemented_awaiting_golden())
(60, 13)
>>> torch._C._shim_grad_state
{'grad': True, 'multithreading': True, 'layout_enforcement': False}
```

---

## 10. 이번에 만진 것

| 파일 | 변경 |
|---|---|
| `rust/torch_c/src/methods.json` | **신규** — 메서드 오버로드 표, 48 이름 · 80 스키마 |
| `rust/torch_c/src/aten.rs` | aten op 19 → 73. 산술 · 비교 · 비트 · 축소 · 형태 · 인덱싱 · in-place |
| `rust/torch_c/src/bootstrap.py` | 메서드 설치, `self_bound` 해석, `int[N]` 규칙, `__getitem__`, `to`/`item`/`__bool__`, grad 모드, `_make_subclass`, `_DISCOVERED_RETURNS` 12 항목 |
| `rust/torch_c/src/tensor.rs` | `requires_grad` 필드, `replace_with`, `#[new]`, `_set_tensor_class` · `promote` |
| `rust/torch_c/src/lib.rs` | `methods.json` 삽입, `_tensor_from_flat` 이 bool 을 정규화 |
| `rust/torch_c/src/overloads.json` | `zeros` 추가 |
| `rust/torch_c/pytests/verify_schemas.py` | 두 표를 모두 검증. op 이름을 **스키마에서** 유도 |
| `rust/torch_c/pytests/test_shim.py` | 34 → 54. 메서드 해석 · 인덱싱 · in-place · grad 모드 · `_make_subclass` · RNG 벽 |
| `docs/bindings/TENSORBASE.md` | 이 문서 |

벤더링 트리의 파이썬 소스는 한 줄도 고치지 않았습니다. **`tools/golden/` 과 `docs/` 의 기존
파일도 건드리지 않았습니다** — `git status --short` 로 확인했습니다.

> `tools/golden/cases.py` 가 수정된 상태로 보이지만 **이번 작업의 것이 아닙니다.** 동시에 도는
> 다른 작업이 43 개의 케이스 빌더를 먼저 심어 둔 것이고(§2), 이번 구현이 그중 39 개를 켰습니다.
