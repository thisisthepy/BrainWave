# 오버로드 해석 — 사용자 API 를 여는 문

IMPORT_TORCH.md 는 `import torch` 를 통과시켰고, C_SURFACE.md 는 **실제로 호출되는 것이
`torch.<op>` 13 개와 `TensorBase` 멤버 50 개뿐**이라고 셌습니다. 그 13 개가 전부 막혀 있었습니다.

```
torch.ops.aten.full.default([2], True).dtype   ->  torch.bool          (되던 것)
torch.full((2,), True)                         ->  NotImplementedError (막혀 있던 것)
    "overload resolution is not implemented; call torch.ops.aten.full.<overload>"
```

**transformers 는 두 번째 철자를 부릅니다.** 이 문서는 그 벽을 뚫은 기록입니다.

판정은 전부 종료 코드입니다. 출력 grep 은 쓰지 않았습니다.

---

## 0. 한눈에

| | 이전 | 지금 |
|---|---|---|
| `torch.<op>(...)` | **전부 `NotImplementedError`** | 표에 있는 **14 개**가 해석됨 |
| 구현된 aten op | 3 | **20** (`_aten_implemented()` 19 + §7.1 의 1) |
| C_SURFACE 의 13 개 | 1 (`full`, aten 철자로만) | **13/13** |
| 골든 하네스 | 188/188, ops covered=3 | **490/490, ops covered=19** |
| 골든 자가검사 (`--inject-fault`) | 1/1/1 | **1/1/1** (그대로) |
| 호스트 스모크 | 27/27 | **34/34** |
| 엄격 `import torch` · `transformers` | 0 · 0 | **0 · 0** |
| 세 타깃 빌드 | 0 | **0** |
| 사용자 API 대조 (49 케이스, 상류 torch vs shim) | — | **49/49 동일** |

`from_config` 는 여전히 실패합니다. **다만 막히는 이유가 op 이 아닙니다** — §8.

> **Correction (문서 감사, 2026-09):** 아래 §8 의 정정과 같은 자리 — `from_config` 는 이제
> 성공합니다. `2d3663f` 가 §8 이 지목한 바로 그 이름을 실제로 넣었습니다.

---

## 1. 무엇이 문제였나 — `torch.full` 은 aten op 이 아니다

상류에서 `torch.full` 은 **C 바인딩**입니다. 실제 인자를 읽고, 여러 네이티브 함수 중 하나를
고르고, 그것을 호출합니다. **aten 오버로드 이름은 "고른 네이티브 함수"의 성질이지 사용자가 친
이름의 성질이 아닙니다.**

그래서 `torch.ops.aten.<op>.<overload>` 까지만 열려 있던 shim 은 사용자 표면이 통째로 꺼져
있었습니다. 벤더링 트리도 transformers 도 그 철자를 쓰지 않습니다.

재현한 것은 `PythonArgParser::raw_parse` 입니다. 이름마다 시그니처 목록을 두고, **순서대로
시도해 처음 바인딩되는 것을 고릅니다.** 특수 케이스 표가 아니라 같은 알고리즘을 쓴 이유는
**순서 자체가 의미론**이기 때문입니다 (§3).

### 단일 진입 원칙은 깨지지 않았다

해석기가 정하는 것은 **어떤 키인가**뿐입니다. 계산하지 않고, 다른 경로로 커널에 닿지 않습니다.

```
torch.full((2,), True)
  -> bootstrap.py::_Overloads.resolve  ->  ("aten.full.default", {size, fill_value})
  -> _C._aten_dispatch("aten.full.default", size=..., fill_value=True)      <- 그 문
  -> aten.rs::full_default
```

해석은 됐는데 커널이 없으면 **전과 똑같이** `aten op not implemented in torch._C shim: <key>`
가 납니다. 오히려 계측기가 좋아졌습니다 — 전에는 `torch.full(...)` 이 op 집합 전체를 대고
거부했는데, 이제 **호출자가 실제로 필요했던 오버로드 하나**를 댑니다.

```
>>> torch.arange(5, out=t)
NotImplementedError: aten op not implemented in torch._C shim: aten.arange.out
```

표에 없는 이름은 예전 거부를 유지합니다. `.default` 로 찍어보내지 않습니다 — 그러면 대부분
존재하지 않는 키를 대게 되어 작업 큐가 오염됩니다.

---

## 2. 스키마는 어디서 오는가 — 트리에서 **못** 가져옵니다

작업 지시가 "`surface.json` 이나 벤더 트리의 `.pyi` 에서 얻을 수 있는지 보라" 였습니다.
**확인했고, 답은 아니오입니다.** 그 이유를 남깁니다. 이것이 이 작업에서 가장 갚을 것이 큰 부채입니다.

트리는 **두 조각을 따로** 갖고 있고, 둘을 잇는 것을 갖고 있지 않습니다.

| 트리가 가진 것 | 어디에 | 무엇이 없나 |
|---|---|---|
| **aten 오버로드 이름** | `_decomp` · `_meta_registrations` · `_refs` 가 `aten.arange.start_step` 처럼 **문자 그대로** 참조. 트리 전체에서 **983 개** 수확됨 | 인자 타입 · 기본값 · 순서 |
| **파이썬 레벨 시그니처** | `torch/_C/_VariableFunctions.pyi` (30,899 행). `full` 3 개 · `arange` 6 개 오버로드 | **어느 aten 오버로드로 내려가는지** |

`.pyi` 의 `@overload def full(size, fill_value, *, out=None, ...)` 는 그것이 `full.out` 인지
`full.default` 인지 말하지 않습니다. 그 대응은 `native_functions.yaml` 에 있고, **벤더링 트리에
`native_functions.yaml` 도 `torchgen` 도 없습니다** (확인함 — 트리의 유일한 `.yaml` 은
`torch/_export/serde/schema.yaml` 로 무관).

C_SURFACE 의 13 개 중 **11 개**는 오버로드 *이름*이 트리에 있습니다
(`is_floating_point` · `isin` 만 없음). 이름만으로는 바인딩을 못 합니다.

### 그래서 어떻게 했나 — 옮겨 적고, 검증기를 함께 둔다

`rust/torch_c/src/overloads.json` 에 **실제 aten 스키마 문자열 45 개**를 적었습니다.
출처는 `str(torch.ops.aten.<op>.<ov>._schema)`, torch 2.13.0.

**IMPORT_TORCH.md §1 이 `surface.json` 에 금지한 것과 다릅니다.** 거기서 금지한 것은
*빌드가* 상류 `.so` 를 요구하게 되는 것이었습니다. 여기서는 표가 **아티팩트에 컴파일되어**
들어가므로 `cargo build` 는 torch 를 요구하지 않습니다. 상류를 쓰는 것은 **생성과 검증 시점**
뿐이고, 그것은 `tools/golden/compare.py` 와 같은 종류의 의존입니다.

옮겨 적은 것에는 검사가 필요하므로, `rust/torch_c/pytests/verify_schemas.py` 를 두었습니다.
상류에서 다시 뽑아 표와 대조합니다.

```
$ /Volumes/macMini/caches/spike-venv/bin/python rust/torch_c/pytests/verify_schemas.py
torch 2.13.0
SUMMARY: 45/45 table entries matched upstream, 0 failed        EXIT=0
```

반대 방향(상류에만 있는 오버로드)은 **일부러 오류가 아닙니다.** 표는 "torch 의 파이썬
바인딩이 노출하는 오버로드"이지 "그 이름의 모든 aten 오버로드"가 아닙니다. `aten::pow` 는
오버로드가 15 개인데 그중 11 개(`pow.int` · `pow.float_complex` · `pow.Scalar_Scalar` …)는
TorchScript 전용 수치 빌트인이고, 그것들을 표에 넣으면 **`torch.pow(2, 3)` 이 float 을 돌려주게
됩니다 — 상류는 TypeError 를 냅니다.**

---

## 3. 순서가 알고리즘이다 — 추론했으면 두 번 틀렸을 자리

순서를 **읽어서 정하지 않았습니다.** 진짜 torch 에 `TorchDispatchMode` 로거를 걸어 어떤 키가
나오는지 관측했습니다.

```python
class Log(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print("->", func); return func(*args, **(kwargs or {}))
```

결과가 두 군데에서 직관과 어긋났습니다.

**(1) `arange(0, 5)` 는 `arange.start` 이지 `start_step` 이 아니다.**

```
torch.arange(5)      -> aten.arange.default
torch.arange(0,5)    -> aten.arange.start
torch.arange(0,5,2)  -> aten.arange.start_step
```

`arange.start_step(Scalar start, Scalar end, Scalar step=1, ...)` 은 `step` 에 기본값이 있어서
2 인자 호출도 **바인딩됩니다.** 표에서 `.start` 를 먼저 두지 않았으면 세 줄 중 가운데가 조용히
`start_step` 으로 갔을 것이고, 작업 큐가 불린 적 없는 op 을 보고했을 것입니다.

**(2) `.out` 변형이 먼저 와야 한다.** `out` 은 기본값 없는 kwarg-only 이므로 **`out=` 을 준
호출에만** 바인딩됩니다. 뒤에 두면 비-out 스키마가 먼저 삼켜서 **출력 텐서를 조용히 버립니다.**

이 두 근거는 `overloads.json` 안의 `_README` 키에 그대로 들어 있습니다 — 표를 고치는 사람이
문서를 찾아가지 않아도 되게 했습니다.

---

## 4. 타입 검사 규칙 — 셋은 torch 의 것이고 직관과 다르다

`_TypeChecker` 는 `python_arg_parser.cpp` 의 `FunctionParameter::check` 입니다.

| 규칙 | 왜 |
|---|---|
| **`bool` 은 `int` 를 만족하지 **않는다**** | 파이썬에서 `bool` 은 `int` 의 서브클래스지만 torch 의 `THPUtils_checkLong` 이 명시적으로 뺍니다. 빼야만 `torch.full((2,), True)` 의 `True` 가 `Scalar` 로 들어가 결과가 `int64` 가 아닌 `torch.bool` 이 됩니다 |
| **`int` 은 `float` 을 만족한다** (한 방향) | |
| **0-dim 텐서는 `Scalar` 를 만족한다** | `DOUBLE`/`SCALAR` 분기가 0-dim `THPVariable` 로 폴스루합니다. 그래서 `torch.pow(t, 0dim_t)` 는 (먼저 선언된) `pow.Tensor_Tensor` 로 갑니다 |

**가변 인자 int 리스트 규칙**도 torch 의 전제 그대로입니다 — `max_pos_args == 1` 이고 그 인자가
int 리스트일 때만 적용됩니다.

```
torch.ones(2, 3)   -> ones([2, 3])     (ones 는 위치 인자가 하나)
torch.full(2, 3)   -> TypeError        (full 은 둘 — full([2], 3) 이 되지 않는다)
```

`torch.full(2, 3)` 이 에러로 남는 것이 중요합니다. 일반화된 규칙을 썼으면 조용히 통과했을 것입니다.

### 표가 모르는 타입 철자는 **설치 시점에** 터진다

`_TypeChecker` 가 모르는 철자가 표에 생기면, 그 오버로드는 **어떤 호출에도 매칭되지 않게**
됩니다 — 그러면 "매칭되는 오버로드 없음" 이라는, 맞는 답의 모양을 한 틀린 답이 나옵니다.
그래서 `install` 이 표를 파싱하면서 모든 철자를 검사하고, 모르는 것이 있으면 **`import _C` 가
그 자리에서 멈춥니다.**

### 파이썬 전용 kwarg

- `requires_grad=False` → 버립니다. `True` → **이름을 대고 거부합니다.** autograd 가 없는데
  아무것도 기록하지 않는 텐서를 돌려주는 것이 거부보다 나쁩니다.
- `out=None` → 뺍니다. 두면 `.out` 에도(필수인데 None) 비-out 에도(인자가 아님) 안 붙어서
  "매칭 없음"이 됩니다.
- **스키마 기본값과 같은 값은 버립니다.** `torch.ones(2, pin_memory=False)` 가 "非 None
  `pin_memory` 는 거부" 커널에 닿지 않게 하는 것이 이 규칙입니다. 커널은 **호출자가 실제로
  요구한 인자만** 봅니다.

---

## 5. 13 개 — 무엇이 되고, 근거는 무엇인가

**전부 됩니다.** 근거는 두 겹입니다.

1. **골든 하네스** — aten 키 수준에서 상류와 dtype · shape · 값을 대조. `490/490`.
2. **사용자 API 대조 49 케이스** — 같은 표현식을 상류 torch 와 shim 에서 각각 돌려 결과를 diff.
   `49/49 동일`. 하네스가 못 보는 것(어느 키로 해석되는가, `torch.tensor`)을 덮습니다.

| # | `torch.<op>` | 해석되는 키 | 커널 | 비고 |
|---|---|---|---|---|
| 1 | `arange` | `.default` · `.start` · `.start_step` | **구현** | §6.1 |
| 2 | `argmax` | `.default` | **구현** | `dim=None, keepdim=True` → shape `[1]` (실측). 결과 int64 (candle 은 u32) |
| 3 | `cat` | `.default` | **구현** | dtype 불일치는 승격하지 않고 이름을 댐 |
| 4 | `embedding` | `.default` | **구현** | candle 은 rank-1 인덱스만 받음 → 평탄화 후 복원. 뒤 인자 셋은 §6.4 |
| 5 | `empty` | `.memory_format` | **구현** | **0 으로 채웁니다 — §6.2 의 차이** |
| 6 | `full` | `.default` | 이미 있음 | 이제 사용자 철자로 닿음 |
| 7 | `is_floating_point` | `.default` | **구현** | §6.5 |
| 8 | `isin` | `.Tensor_Tensor` | **구현** | `Tensor_Scalar` · `Scalar_Tensor` 는 표에 있고 커널 없음 |
| 9 | `ones` | `.default` | **구현** | |
| 10 | `pow` | `.Tensor_Tensor` · `.Tensor_Scalar` · `.Scalar` | **구현** | **candle 의 `pow` 를 쓰지 않았습니다 — §6.3** |
| 11 | `randint` | `.low` · `.default` | **구현** | 난수는 candle 것. §6.6 |
| 12 | `rsqrt` | `.default` | **구현** | 정수 입력 → float32, float16 입력 → float16 (양쪽 실측) |
| 13 | `tensor` | (aten 오버로드가 아님) | **구현** | §6.7 |

<!-- DOCWATCH: op-implemented aten.arange.default -->
<!-- DOCWATCH: op-implemented aten.embedding.default -->
<!-- DOCWATCH: op-implemented aten.is_floating_point.default -->
<!-- DOCWATCH: op-implemented aten.isin.Tensor_Tensor -->
<!-- DOCWATCH: op-implemented aten.pow.Tensor_Tensor -->
<!-- DOCWATCH: op-implemented aten.randint.low -->
곁가지로 `mm` · `add` 도 표에 넣었습니다 — 이미 커널이 있고, 표가 어떤 모양의 오버로드
집합까지 다루는지 보여주는 대조군입니다 (`add.Scalar` 는 커널이 없어 이름을 댑니다).

---

## 6. 각 op 에서 실제로 갈린 지점

### 6.1 `arange` — 상류가 **거부**하는 것을 일부러 거부한다

값은 `start + i*step` 으로 계산합니다. candle 의 `arange_step` 은 누산기(`current += step`)라
float 에서 드리프트합니다. 길이는 정수면 정수 연산, 아니면 `ceil((end-start)/step)`.

**골든 하네스가 잡은 것이 하나 있습니다.**

```
FAIL aten.arange.default :: arange(end=20, dtype=uint32)
     SILENT DIVERGENCE: torch raised NotImplementedError('"arange_cpu" not implemented
     for 'UInt32'') but c computed a value: [0, 1, ..., 19]
```

**우리 쪽이 torch 보다 유능해서 어긋난 경우입니다.** IMPORT_TORCH.md §7 이 `full` 의
numel==1 구멍에서 내린 것과 같은 판단을 했습니다 — 하네스는 torch 와 대조하는 장치이므로
**더 유능한 shim 도 반대 방향으로 똑같이 어긋납니다.**

그래서 상류가 `arange_cpu` 커널을 갖지 않는 dtype 을 **재현해서 거부합니다.** 저장 가능한 모든
dtype 에 대해 실측했습니다: `uint16` · `uint32` · `uint64` · `bool` · float8 계열을 거부하고
나머지는 받습니다. 메시지도 torch 의 것입니다(`"arange_cpu" not implemented for 'UInt32'`).

> 곁가지로 **같은 집합의 세 번째 이름 표기**가 필요해졌습니다. `TorchDType::name()` 은
> `uint32`, `c10_name` 은 `uint32_t`, 그리고 이 메시지가 쓰는 `toString(ScalarType)` 은
> `UInt32` 입니다. 셋 다 유도 불가라서 각각 실제 torch 오류에서 읽었습니다.

### 6.2 `empty` 는 0 을 돌려준다 — 명시적 차이

torch 의 `empty` 는 할당에 있던 것을 그대로 돌려주고 **아무 약속도 하지 않습니다.** 0 은 그
계약을 만족하지만 **같은 바이트가 아닙니다.** 초기화되지 않은 텐서를 읽어서 torch 와 비교하는
테스트는 잡음을 비교하는 것입니다. 우리는 torch 가 결정적이지 않은 곳에서 결정적입니다 —
안전한 방향이지만 차이입니다.

### 6.3 `pow` — candle 의 `pow` 는 쓸 수 없다

```rust
// candle_core::Tensor::pow
pub fn pow(&self, rhs: &Tensor) -> Result<Self> { rhs.mul(&self.log()?)?.exp() }
```

`exp(exponent * log(base))` 는 **음수 밑에서 전부 NaN** 입니다. 그리고 `torch.pow(x, 2)` 는
RMSNorm 경로이고 hidden state 는 음수를 갖습니다 — 첫 모델에서 바로 틀렸을 것입니다.
(`powf(e)` 쪽은 진짜 `f64::powf` 를 타서 멀쩡한데, 정수 dtype 을 거부합니다.)

그래서 원소별로 직접 계산합니다. 부동소수 결과는 f64 로, 정수 결과는 i64 로 — `int64` 의 큰
값이 f64 를 경유해 정밀도를 잃지 않게 하기 위해서입니다.

dtype 규칙은 torch 의 "wrapped number" 승격을 **실측**한 것입니다.

```
pow(int64_t, 2)     -> int64     (파이썬 정수 스칼라는 텐서를 넓히지 않는다)
pow(int64_t, 2.0)   -> float32
pow(float32_t, 2)   -> float32
```

정수 결과에 음수 지수는 torch 의 문구 그대로 거부하고
(`Integers to negative integer powers are not allowed.`), 오버플로는 torch 의 정수 커널처럼
**감습니다** — 거부하면 반대 방향으로 어긋납니다.

`torch.bool` 피연산자는 **거부합니다.** torch 의 결과 카테고리를 재지 않았고, 재지 않은 것을
찍는 것이 이 shim 이 막으려는 바로 그것입니다.

### 6.4 `embedding` 의 뒤 인자 셋

`padding_idx` · `scale_grad_by_freq` · `sparse` 는 torch 에서 **역전파 전용**이고 순전파
결과는 어느 것에도 의존하지 않습니다. `padding_idx` 는 받아서 무시합니다(상류 순전파가 하는
것과 같음). 나머지 둘은 **참이면 거부합니다** — 역전파가 없는 shim 에서 기울기 동작을 켜는
것은 뭔가를 주장하는 것이기 때문입니다.

### 6.5 `is_floating_point` — 상류는 디스패처를 지나지 않는다

`TorchDispatchMode` 로 재보면 `torch.is_floating_point(t)` 는 **aten 기록을 하나도 남기지
않습니다.** 그럼에도 우리는 `_aten_dispatch("aten.is_floating_point.default", t)` 로 보냅니다 —
표면이 "문 하나 + 지름길 하나"가 되는 것보다 문 하나인 편이 낫다는 판단입니다.
**상류와 다른 지점이므로 §9 에 남깁니다.**

### 6.6 `randint` — 난수는 재현되지 않는다

`dtype=4` 는 `ScalarType::Long` 이라 이 팩토리만 기본이 int64 입니다.

**생성기는 candle 것이므로 시드를 고정한 torch 실행과 값이 같지 않습니다.** 시드 배선이 없고,
torch 의 Philox 스트림을 재현한다고 주장하는 것은 테스트가 못 잡는 거짓말입니다. 재현한 것은
**범위 · shape · dtype** 입니다.

### 6.7 `torch.tensor` — 유일하게 오버로드 집합이 아니다

`aten::tensor` 는 존재하지만 TorchScript 빌트인이고, `torch.tensor([1, 2])` 는 그리로 가지
않습니다. 실측하면 **aten 기록이 정확히 하나, `aten.lift_fresh.default` 뿐**입니다. 데이터는
`THPVariable_tensor` → `internal_new_from_data` 라는 `_C` 함수가 만들고 디스패처는 그것을 보지
않습니다.

그래서 상류의 분할을 그대로 따랐습니다.

```
torch.tensor(data) = _aten_dispatch("aten.lift_fresh.default",
                                    _C._tensor_new_from_data(data, dtype, device))
```

`_tensor_new_from_data` 는 중첩 시퀀스를 걸어 shape · 값 · dtype 을 정합니다. dtype 추론은
카테고리 순서(전부 bool → `torch.bool`, 전부 int → `int64`, float 이 하나라도 → 기본 float).
**bool 을 먼저 보는 것이 전부입니다** — `torch.tensor([True])` 는 마스크입니다.

- **`_tensor_from_flat` 과 별개로 두었습니다.** 그쪽은 BOOL.md §6.3 이 지목한 자리라
  `torch.bool` 을 거부하는 채로 남습니다(삭제 예정 스캐폴딩). 새 함수는 bool 을 받아야 하고,
  받되 **`PyTensorBase::boolean` 이라는 단 하나의 태깅 생성자**를 지나므로 0/1 불변식이
  구성으로 성립합니다.
- 들쭉날쭉한 입력 검사에 함정이 있었습니다. **원소 수만 세면 `[[1], 2]` 가 통과합니다** —
  shape `[2, 1]` 로 걸어가고 그 shape 이 요구하는 두 값을 정확히 내놓습니다. 그래서 **모든 잎이
  같은 깊이에 있는지**를 따로 확인합니다. 회귀 테스트가 있습니다.

---

## 7. 때운 것 · 끈 것 · 구현한 것

**구현이 아닌 것은 아래 둘뿐입니다.** §5 · §6 에서 "구현"이라고 쓴 것은 요구된 동작을 실제로 합니다.

### 7.1 `_aten_implemented()` 가 op 하나를 **덜** 보고한다 (보고의 문제, 능력의 문제가 아님)

골든 하네스는 "`_aten_implemented()` 에 있는데 `tools/golden/cases.py::CASE_BUILDERS` 에 없는
op" 을 **일부러 하드 실패로** 다룹니다. 상류와 대조하지 않은 op 이 몰래 들어오지 못하게 하는
옳은 규칙입니다. 이번 작업은 하네스를 고칠 수 없었고, 하네스에 케이스 빌더가 없는 op 이 하나
나왔습니다.

```rust
pub const IMPLEMENTED_AWAITING_GOLDEN: &[&str] = &["aten.randint.default"];
```

`_aten_dispatch` 는 다른 op 과 똑같이 닿습니다 — `torch.randint(10, (2,))` 는 **동작합니다.**
다만 커버리지 숫자가 **적게** 나옵니다(과장이 아니라 축소 방향).
`_C._aten_implemented_awaiting_golden()` 으로 읽을 수 있고, 스모크 테스트가 두 목록이
서로소인지 확인합니다. **고치는 방법은 케이스 빌더 하나와 한 줄 이동입니다.**

> 하네스 쪽(다른 작업)이 이미 16 개 op 의 케이스 빌더를 먼저 넣어 두었고
> (`PENDING: ... waiting, not failing`), 이번 구현이 그 16 개를 전부 켰습니다.
> `randint.default` 만 그 목록에 없었습니다.

### 7.2 accelerator 존재 질문 둘 (§8 을 재기 위해 필요했던 것)

`_xpu_getDeviceCount` → `0`, `_mtia_isBuilt` → `False`.

**엄밀히는 때움이 아닙니다** — XPU 장치가 없는 빌드의 개수는 0 이고, MTIA 지원이 컴파일되지
않은 빌드의 답은 False 입니다. 참인 답입니다. 여기 적는 이유는 이것이 op 작업이 아니라 표면
작업이고, `_DISCOVERED_RETURNS` 를 만지는 다른 작업과 겹칠 수 있기 때문입니다.

이름을 빼는 off-switch 를 쓸 수 없습니다: `torch.xpu` · `torch.mtia` 는 **항상 임포트되는 평범한
파이썬 패키지**라 `hasattr` 이 질문이 될 기회가 없습니다.

---

## 8. 다음 벽은 op 이 아니다

`from_config` 는 여전히 실패하고, 여전히 `transformers` 의 `GenerationMixin` 지연 임포트에서
멈춥니다(IMPORT_TORCH.md §11 항목 3 과 같은 자리). **그런데 막는 것이 op 이 아닙니다.**

한 겹씩 벗겨 관측한 순서입니다.

| # | 벽 | 성격 | 처리 |
|---|---|---|---|
| 1 | `torch._C._xpu_getDeviceCount` — `transformers/masking_utils.py:39` → `torch.xpu.is_available()` | accelerator 존재 질문 | `0` (§7.2) |
| 2 | `torch._C._mtia_isBuilt` — `torch/_dynamo/device_interface.py:297`, 모듈 스코프 | 같음 | `False` (§7.2) |
| 3 | **`torch._C._dynamo.eval_frame.set_guard_error_hook`** — `torch/_dynamo/guards.py:5457`, 모듈 스코프 | **`_C._dynamo` 서브모듈 표면** | **미구현 — 여기서 멈췄습니다** |

3 번은 op 이 아니라 **dynamo 의 C 표면 전체**입니다. C_SURFACE.md §5 가 이미 지목한 것과 같은
결의 발견입니다 — `transformers.masking_utils` 가 `torch._dynamo` 를 끌어오므로
**`torch.compile` 을 한 번도 쓰지 않아도 dynamo 가 임포트 그래프에 딸려 옵니다.**

**그러므로 다음 임계 경로는 op 이 아니라 `_C._dynamo` 입니다.** 이 작업의 범위 밖이라 2 번에서
멈추고 기록합니다.

> **Correction (문서 감사, 2026-09):** 3 번은 더 이상 미구현이 아닙니다 — `RUN THE CHECK: 이 절이
> "여기서 멈췄습니다" 라고 이름 댄 바로 그 심볼이 이제 존재합니다.** `git log -S"set_guard_error_hook"
> -- rust/torch_c/src/bootstrap.py` 가 찾는 커밋은 `2d3663f` ("Feat: Port torch's CPU generator,
> and give _C._dynamo the two names that do work") 이고, 그 커밋 메시지가 이름 댄 "두 개" 중
> 하나가 정확히 이 심볼입니다(다른 하나는 `set_eval_frame_isolate_recompiles_id`). 실측:
> `hasattr(torch._C._dynamo.eval_frame, 'set_guard_error_hook')` → `True`; 더 결정적으로,
> **§8 이 실패한다고 기록한 바로 그 호출이 이제 성공합니다** —
> `AutoModelForCausalLM.from_config(LlamaConfig(...))` 를 이 문서가 쓴 것과 같은 작은 설정으로
> 직접 호출해 `LlamaForCausalLM` 인스턴스를 얻었습니다(2026-09, 이 셰임에서). §0 표의
> "`from_config` 는 여전히 실패합니다" 도 같은 이유로 낡았습니다.
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py set_guard_error_hook present -->

---

## 9. 미확인 · 상류와 다른 지점

| # | 항목 | 상태 |
|---|---|---|
| 1 | `torch.is_floating_point` 이 디스패처를 지난다 | 상류는 안 지납니다(§6.5). 답은 같고 트레이스가 다릅니다 |
| 2 | `torch.empty` 가 0 을 돌려준다 | §6.2. 계약은 지키지만 바이트가 다릅니다 |
| 3 | `torch.randint` 값이 재현되지 않는다 | §6.6. 범위 · shape · dtype 만 일치 |
| 4 | `pow` 의 `torch.bool` 결과 카테고리 | **미측정**. 거부합니다 |
| 5 | `arange` 가 거부하는 dtype 집합 | torch **2.13.0 에서** 실측. 다른 판본에서 바뀌면 어긋납니다 |
| 6 | `overloads.json` 이 14 개 op 만 덮는다 | `_VariableFunctions` 는 609 개. 나머지는 "표 항목 없음" 으로 거부 |
| 7 | 파이썬 시그니처가 aten 스키마와 어긋나는 op | torch 의 바인딩에는 파이썬 레벨에만 있는 인자(`names=` 등)가 있는 이름이 더 있습니다. 이번 14 개에서는 `requires_grad` · `out` 밖에 나오지 않았고, **더 넓은 표에서 이 가정이 유지되는지 미확인** |
| 8 | `TensorBase` 50 개 | **손대지 않았습니다.** C_SURFACE §4 의 목록 그대로 남아 있고, `x + y` · `x.sum()` 같은 메서드 경로는 여전히 이름을 댑니다 |
| 9 | 표를 자동 생성할 수 있는가 | §2 의 두 조각을 잇는 것이 있으면 됩니다. `native_functions.yaml` 을 벤더링하거나, `.pyi` 오버로드 ↔ aten 오버로드 대응을 트리에서 유도할 방법을 찾는 것 — **둘 다 미조사** |
| 10 | 기기(Android · iOS) | 이번에도 링크만 |
| 11 | 크기 증가분 +352 KB 의 내역 | candle 커널이 새로 참조되어 단형화된 것으로 **추정**. 분해 미측정 |

---

## 10. 검증

**전부 종료 코드입니다.**

| | 명령 | 결과 |
|---|---|---|
| 호스트 스모크 | `rust/torch_c/pytests/run.sh` | **0** — 34/34 (이전 27) |
| 골든 하네스 | `tools/golden/compare.py` | **0** — **490/490, ops covered=19** (이전 188/188, 3) |
| 골든 자가검사 | `--inject-fault value/shape/dtype` | **1 / 1 / 1** (의도대로) |
| 스키마 검증 | `rust/torch_c/pytests/verify_schemas.py` | **0** — 45/45 |
| 사용자 API 대조 49 케이스 | 상류 torch 와 shim 을 각각 돌려 diff | **49/49 동일** |
| 엄격 `import torch` | `probe.py --mode strict --target torch` | **0** |
| 기록 `import torch` | `probe.py --mode record --target torch` | **0** |
| 엄격 `import transformers` | `probe.py --mode strict --target transformers` | **0** |
| `aarch64-apple-darwin` | `cargo build --release` | **0** |
| `aarch64-linux-android` | `cargo ndk -t arm64-v8a` | **0** |
| `aarch64-apple-ios` | `cargo build --target aarch64-apple-ios` | **0** |

**`tools/golden/` 과 `docs/` 의 기존 파일은 한 줄도 고치지 않았습니다.** `git status --short` 로
확인한 변경 범위는 `rust/torch_c/` 아래 6 개 파일과 이 문서뿐입니다.

### 크기

| 타깃 | 이전 (IMPORT_TORCH §10) | 지금 | 차이 |
|---|---|---|---|
| 호스트 | 1,740,064 | 2,092,512 | +352,448 |
| Android | 2,618,312 | 2,993,624 | +375,312 |
| iOS | 1,811,208 | 2,177,984 | +366,776 |

`overloads.json` 은 7,358 B 이고 파이썬 추가분도 수 KB 이므로, 증가분의 대부분은 **새로 참조된
candle 커널**(`cat` · `index_select` · `floor` · `clamp` · `rand` · `argmax` …)이 dtype 마다
단형화된 것으로 보입니다. 셋이 같은 방향으로 비슷하게 움직인 것이 그 방증입니다. 분해는
미측정(§9-11).

iOS 는 여전히 `@rpath/Python.framework/Python` 로 링크됩니다 (올바른 모양).

### 임포트 비용

```
import _C  (단독)   17.6 ~ 18.1 ms   (이전 15.2 ms — 표 45 개 파싱과 커진 .so)
import torch        1.15 s           (이전 1.15 s — 변화 없음)
```

`import _C` 첫 회는 24 ms, 이후 17.6~18.1 ms 로 안정.

---

## 11. 재현

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target
cd /Volumes/macMini/thisisthepy/torchnative
PY=/Volumes/macMini/caches/spike-venv/bin/python

./vendor/install_shim.sh                       # 빌드 + 구멍에 넣기
$PY tools/golden/compare.py;                        echo "EXIT=$?"
$PY rust/torch_c/pytests/verify_schemas.py;         echo "EXIT=$?"
(cd rust/torch_c && ./pytests/run.sh);              echo "EXIT=$?"

TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/vendor $PY -c \
  "import torch; print(torch.full((2,), True).dtype, torch.arange(0,5,2).tolist())"
```

`TORCH_USE_RTLD_GLOBAL=1` 이 없으면 `import torch` 자체가 안 됩니다 (VENDOR.md 벽 1).

해석기가 무엇을 할 수 있는지는 물어보면 됩니다 — 표를 아티팩트에서 읽어낼 필요가 없습니다.

```python
>>> torch._C._shim_overloads["arange"]
['aten.arange.start_out', 'aten.arange.out', 'aten.arange.start',
 'aten.arange.start_step', 'aten.arange.default']
>>> torch._C._aten_implemented_awaiting_golden()
['aten.randint.default']
```

세 타깃 명령은 `docs/platform/RUST_CROSSBUILD.md` §0.5 그대로이고, 이번에 달라진 것은 없습니다.

---

## 12. 이번에 만진 것

| 파일 | 변경 |
|---|---|
| `rust/torch_c/src/overloads.json` | **신규** — 스키마 표 45 개 (14 op) |
| `rust/torch_c/pytests/verify_schemas.py` | **신규** — 표를 상류와 대조하는 검증기 |
| `rust/torch_c/src/bootstrap.py` | 오버로드 해석기(`_TypeChecker` · `_Overloads`), `torch.tensor` 팩토리, `_shim_overloads`, `_DISCOVERED_RETURNS` 2 항목 |
| `rust/torch_c/src/aten.rs` | aten op 17 개 추가(3 → 20), `Scalar` 인자 처리, `scalar_type_name` · `arange_has_cpu_kernel` |
| `rust/torch_c/src/lib.rs` | `overloads.json` 을 `include_str!` 로 삽입, `_tensor_new_from_data` |
| `rust/torch_c/pytests/test_shim.py` | 27 → 34 개. 해석기 · `torch.tensor` · 두 목록의 서로소성 |

벤더링 트리의 파이썬 소스는 한 줄도 고치지 않았습니다. `tools/golden/` 과 `docs/` 의 기존
파일도 건드리지 않았습니다.
