# `import torch` 완주 — 벽 하나씩

VENDOR.md 는 "목표는 되게 하는 것이 아니라 어디서 깨지는지 아는 것" 이라고 적고, 미확인 항목 1 번에
**`import torch` 완주** 를 남겼습니다. 이 문서는 그 항목의 기록입니다.

**엄격 모드 `import torch` 가 종료 코드 0 으로 끝납니다.** 허수아비 없이, 벤더링 트리를 한 줄도
고치지 않고, `torch/__init__.py` 3087 행을 끝까지 지나갑니다.

판정은 전부 종료 코드입니다. 출력 grep 은 쓰지 않았습니다.

---

## 0. 한눈에

| | 이전 (VENDOR.md) | 지금 |
|---|---|---|
| **엄격 모드 `import torch`** | `torch/__init__.py:1050` 정지 | **완주. EXIT=0** |
| 기록 모드 `import torch` | 2885 행 (93%) 정지 | **완주. EXIT=0** |
| `import transformers` (엄격) | 0 | 0 |
| 우리 `_C` 의 `dir()` | 17 | **1207** (상류 989) |
| `TensorBase` 멤버 | ~3 | **694** (상류 694) |
| `_VariableFunctions` 멤버 | 없음 | **1006** (상류 985) |
| `sys.modules` 의 `torch._C.*` | 0 | **35** (상류 32) |
| `torch` 에 주입된 enum 인스턴스 | 11 | **59** (상류 74 — §8 항목 1) |
| 임포트 중 해소된 `torch.ops.aten.*` | 3 | **889** |
| 등록된 분해(decomposition) | — | **973** |
| dtype | 10 (candle 것) | **33 + 별칭 9** (`_C` 가 소유) |
| 골든 하네스 | 183/188 (5 실패) | **188/188** |
| 세 타깃 빌드 | 0 | **0** |

`from_config` 는 여전히 실패합니다. 다만 **실패 지점이 `import torch` 밖으로 나갔습니다** — 이제
transformers 쪽에서 멈춥니다 (§8 항목 3).

> **정정 (문서 감사, 2026-09):** §11 항목 3 도 같이 본다 — `from_config` 는 오늘 shim 위에서
> 직접 성공한다. `docs/design/REGISTRATIONS.md`/`docs/models/FROM_CONFIG.md` 감사(이 라운드)가 각각 독립적으로
> 재확인했다: `torch._C._log_api_usage_once` 가 생겼고, 14개 초기화 op 이 전부 구현됐다. 아래
> §0/§11 원문은 이 문서가 쓰인 시점 그대로 남긴다.
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _log_api_usage_once present -->

---

## 1. 구조 — `_C` 는 여전히 `.so` 하나다

임포트를 통과시키는 데 필요한 것의 대부분은 **계산이 아니라 이름**입니다. 타입 172 개, 서브모듈
27 개, `TensorBase` 멤버 694 개, `_VariableFunctions` 976 개. 이것들은 전부 임포트 시점에 한 번
동적으로 만들어지는 파이썬 객체입니다.

그래서 두 조각을 넣었습니다.

```
rust/torch_c/src/surface.json    벤더링 트리의 .pyi 에서 뽑은 이름표 (120 KB)
rust/torch_c/src/bootstrap.py    그 표로 표면을 짓는 코드 (1297 행)
```

둘 다 `include_str!` 로 `.so` 안에 들어가고, `lib.rs::run_bootstrap` 이 `#[pymodule]` 초기화
중에 실행합니다. **`_C` 는 네이티브 확장 모듈 하나 그대로입니다** — 파이썬 패키지로 바꾸지
않았고, 런타임에 디스크에서 읽는 것도 없습니다.

### 왜 파이썬으로 짓는가

기계적인 이유입니다. 지어야 하는 것이 힙 타입 생성, 속성 설정, `sys.modules` 삽입인데, Rust 로
쓰면 같은 연산을 `Bound<'_, PyAny>` 로 몇 배 길게 쓰는 것이고 실행 시점도 똑같습니다. 타입 안전성이
느는 것도 아닙니다.

**행동이 있는 것은 전부 Rust 에 남습니다** — dtype, device, TensorBase, 그리고 aten 디스패처.
`bootstrap.py` 는 아무것도 계산하지 않습니다. 만드는 호출 가능 객체는 전부 (a) `_aten_dispatch`
로 내려가거나 (b) 자기 이름을 대고 `NotImplementedError` 를 냅니다.

### 이름은 어디서 오는가 — 상류 `.so` 가 아니라 벤더링 트리의 `.pyi`

`vendor/probe.py --dump-surface` 는 **설치된 상류 `_C.so`** 에서 이름을 읽습니다. 계측기로는
옳지만(구멍을 재는 것이니까) shim 의 입력으로는 틀립니다 — 우리가 교체하려는 바로 그 바이너리에서
빌려오는 것이고, 빌드가 진짜 torch 설치를 요구하게 됩니다.

`vendor/gen_surface.py` 는 대신 `torchnative/src/main/torch/_C/*.pyi` 를 읽습니다. **벤더링한 BSD 트리의 일부**
이고, 상류가 "다른 도구가 `_C` 를 로드하지 않고도 인터페이스를 알 수 있도록" 넣어 둔 파일이며,
무엇보다 **트리 자신이 무엇을 기대하는지에 대한 진술**입니다.

`ast` 로 파싱합니다. 정규식이 아닌 이유는 중요한 구분이 정확히 AST 노드 구분이기 때문입니다 —
`def name(...)` 은 메서드, `name: T` 는 getset 디스크립터(VENDOR.md 벽 10 이 "디스크립터 객체
자체" 를 요구한 그것).

`.pyi` 만으로는 부족한 자리가 두 곳 있어서 트리 본문도 훑습니다.

- **`from torch._C import ...` 와 `torch._C.<name>`** — 스텁이 선언하지 않는 이름이 있습니다.
  `_ScalingType` · `_SwizzleType` 은 `torch/nn/functional.py:12` 가 임포트하는데 스텁에 없습니다.
- **`add_docstr_all("<method>")`** — `torch/_tensor_docs.py` 의 547 개 호출이 각각
  `getattr(torch._C.TensorBase, method)` 를 합니다. 스텁의 627 개로는 모자랍니다.

### 세 가지 규칙

1. **`_C` 모듈에는 catch-all `__getattr__` 을 두지 않는다.** 트리는
   `hasattr(torch._C, "_c10d_init")` 로 서브시스템을 끄고 켭니다(VENDOR.md 벽 11). catch-all 은
   그 질문 전부에 "예" 라고 답해서 distributed · RPC · CUDA · XPU · MTIA · MPS 를 한꺼번에 켭니다.
2. **타입과 서브모듈에는 catch-all 을 둔다.** 스텁이 불완전하고(`_special` · `_fft` · `_linalg` 는
   스텁 자체가 없음), 타입의 *멤버* 를 게이트로 쓰는 곳은 없기 때문입니다.
3. **미구현은 자기 이름을 댄다.** DESIGN.md §6.

### off-switch 는 손으로 적지 않고 트리에서 뽑는다

`gen_surface.py` 가 `hasattr(torch._C, "...")` · `getattr(torch._C, "...", ...)` 를 스캔해
**32 개**를 뽑습니다. 손으로 적었던 판본은 이미 `_cuda_isInBadFork` · `_mps_is_in_bad_fork` ·
`_xpu_isInBadFork` 를 빠뜨리고 있었습니다.

`_C._shim_off_switches` 로 읽을 수 있습니다 — **없는 것을 역추적하지 않고 물어볼 수 있게** 두었습니다.

> **이름을 빼는 것과 미구현은 다릅니다.** 안 만든 이름은 *존재하고* 쓰면 터집니다. off-switch 에
> 있는 이름은 *존재하면 안 됩니다* — 그 부재가 트리가 던진 질문에 대한 답이기 때문입니다.

---

## 2. VENDOR.md 의 벽 19 개 — 현재 상태

| # | 벽 | 처리 | 무엇을 했나 |
|---|---|---|---|
| 1 | `libtorch_global_deps` | off-switch | `TORCH_USE_RTLD_GLOBAL` (그대로) |
| 2 | `_initExtension` 없음 | **구현** | dtype·layout·memory_format·qscheme 를 `torch` 에 주입 |
| 3 | `TensorBase` 694 멤버 | **구현** | PyO3 타입에 `setattr` — §3 |
| 4 | `torch_shm_manager` 존재 | 때움 | 0 바이트 표식 (그대로) |
| 5 | `_VariableFunctions` 수확 | **구현** | 비바인딩 인스턴스 속성 1006 개 |
| 6 | PyO3 가 `__all__` 을 만듦 | **구현** | `__all__` 을 직접 세팅 |
| 7 | `_C` 가 `torch` 에 이름을 씀 | **구현** | `_initExtension` 이 59 개 주입 |
| 8 | `_C` 는 패키지여야 함 | **구현** | `sys.modules` 35 개 + meta_path 파인더 |
| 9 | `type(TensorBase) is _TensorMeta` | **부분** | `_TensorMeta = type` — §3 의 단서 |
| 10 | getset 디스크립터 | **구현** | `property` 객체로 생성 |
| 11 | 이름을 빼는 것이 끄는 방법 | off-switch | 32 개, 트리에서 자동 추출 |
| 12 | 메타타입이 타입마다 다름 | **구현** | 기본 `_ShimMeta`, 예외는 실행으로 발견 — §3 |
| 13 | TorchScript 소스 파서 | **구현** | 인자 받는 생성자 + `ErrorReport.call_stack` |
| 14 | `DispatchKey` 타입 검사 | **구현** | 진짜 `enum.Enum` + `DispatchKeySet` |
| 15 | `_autograd_init()` | **구현** | `True` 반환 |
| 16 | `_jit_init()` | **구현** | `True` 반환 |
| 17 | `_multiprocessing_init` | **구현** | 패키지에 이름 주입 |
| 18 | op 레지스트리 | **구현** | 세 함수 전부 `_aten_dispatch` 로 — §4 |
| 19 | `Generator` 메타클래스 | **구현 + 정정** | 아래 |

### 벽 19 는 양방향 결합이 아니라 상류가 준 훅이다

VENDOR.md 는 이렇게 적었습니다.

> C 확장 모듈이 자기 초기화 중에 파이썬 패키지에서 메타클래스를 가져와 자기 타입에 붙여야 합니다.
> 지금까지의 방향 — 파이썬이 `_C` 에 의존 — 과 **반대 방향의 결합**이고, 이번 조사에서 처음
> 나온 종류입니다.

**정정합니다.** `torch/_prims/rng_prims.py:411` 에 주석이 그대로 붙어 있습니다.

```python
# Late-bind OpaqueBaseMeta as Generator's metaclass. This is done here
# rather than in THPGenerator_init (C++) to avoid making torch._C depend
# on torch._opaque_base at init time.
from torch._opaque_base import OpaqueBaseMeta
torch._C._set_generator_metaclass(OpaqueBaseMeta)
```

즉 상류는 정확히 그 결합을 **피하려고** 세터를 만들어 두었고, `_C` 는 그것을 노출하기만 하면
됩니다. 뒤집을 것이 없습니다. 기록 모드가 거기서 죽은 것은 허수아비 세터가 아무것도 안 했기
때문입니다.

우리 구현은 `module.Generator.__class__ = meta` 한 줄입니다. **`Generator` 를 `_ShimMeta`(힙 타입)
로 만들어 두었기 때문에 성립합니다.** PyO3 타입이었다면 안 됩니다 — 그쪽 메타타입은 정적 `type`
이고 `__class__` 대입이 거부됩니다(실측).

---

## 3. `TensorBase` 와 메타타입 — 측정으로 갈린 설계

### PyO3 타입은 `setattr` 을 받는다

벽 3 은 `TensorBase` 에 694 개 멤버가 필요하다는 것이고, 처음 세운 계획은 파이썬 서브클래스를
만들어 거기에 얹는 것이었습니다. **그러면 `isinstance(op_result, TensorBase)` 가 깨집니다** —
연산 결과는 Rust 쪽 타입으로 나오니까요.

먼저 재 봤습니다.

```
_C.TensorBase.zzz = 1        -> OK          (PyO3 타입은 힙 타입, IMMUTABLETYPE 아님)
_C.TensorBase.__class__ = M  -> TypeError   ("only supported for mutable types")
```

그래서 **`TensorBase` 는 네이티브 타입 그대로 두고 멤버만 얹었습니다.** 서브클래스가 없으므로
정체성 문제도 없습니다.

### 그 대가가 `_TensorMeta = type` 이다

`__class__` 대입이 거부되므로 `type(TensorBase)` 는 `type` 이고, `_C._TensorMeta` 도 `type` 입니다.
`torch/nn/parameter.py:19` 의 `class _ParameterMeta(torch._C._TensorMeta)` 와
`class Parameter(torch.Tensor, metaclass=_ParameterMeta)` 는 양쪽 다 성립합니다.

**차이는 남습니다: `isinstance(X, torch._C._TensorMeta)` 가 모든 클래스에 대해 참입니다.**
벤더링 트리에서 그렇게 쓰는 곳은 없지만(grep 으로 확인: `_TensorMeta` 참조는 `nn/parameter.py`
두 곳뿐), 상류와 다른 지점이므로 §8 에 남깁니다.

### 메타타입 예외는 실행으로만 찾을 수 있다

VENDOR.md 벽 12 는 "일괄 규칙으로는 둘 중 하나가 반드시 깨진다" 고 했고, 그대로였습니다.
기본값은 `_ShimMeta`(`_Await` 가 `type` 이면 `duplicate base class`)이고, `type` 이어야 하는
예외를 **클래스 문이 실행될 때 나는 `metaclass conflict` 로** 하나씩 찾았습니다.

```
_LegacyVariableBase    torch/autograd/variable.py:14
_FunctionBase          torch/autograd/function.py:365
```

두 개입니다. 상류 분포(pybind11_type 135 · type 51)와 다르지만, **실제로 걸리는 것은 이 둘뿐**
이라는 것이 이번 측정입니다.

### 안전하지 않은 던더는 주지 않는다

`torch/_tensor.py:1115` 는 `__itruediv__ = _C.TensorBase.__idiv__` 이므로 연산자 던더는 이름으로
요구됩니다. 그러나 일부 던더는 대역품을 주면 **터지는 대신 객체가 망가집니다** —
`__getattribute__` 는 전부를 가로채고, `__del__` 은 수집 중에 던지고, `__repr__` 은 모든
트레이스백을 못 쓰게 만듭니다. 24 개를 명시적으로 제외했습니다(`bootstrap.py::UNSAFE_DUNDERS`).

---

## 4. op 레지스트리 — 문은 하나로 유지된다

`torch.ops.aten.<op>.<overload>` 는 세 함수로 만들어집니다.

| 함수 | 반환 | 우리 구현 |
|---|---|---|
| `_jit_get_operation(qualname)` | `(op, overload_names)` | `op` 는 `_aten_dispatch(f"{ns}.{name}.{overload}")` 래퍼 |
| `_get_operation_overload(qualname, overload)` | `(op, op_dk, tags)` | 같음 |
| `_get_schema(qualname, overload)` | 스키마 객체 | `_Schema` (§5) |

**즉 벤더링 트리가 만든 모든 op 호출이 `_aten_dispatch` 로 들어옵니다.** 실측:

```python
>>> a = torch.ops.aten.full.default([2, 2], 3.0)
>>> torch.ops.aten.add.Tensor(a, torch.ops.aten.full.default([2,2], 4.0)).tolist()
[[7.0, 7.0], [7.0, 7.0]]
>>> torch.ops.aten.relu.default(a)
NotImplementedError: aten op not implemented in torch._C shim: aten.relu.default
```

두 번째 줄이 §6 의 계측기가 살아 있다는 증거입니다 — 벤더링 트리를 통과한 호출이 이름을 대고
터집니다.

### 던더 이름 — aten 에는 진짜로 있다

`_OpNamespace.__getattr__` 은 파이썬이 던지는 질문(`__origin__`, `__deepcopy__`, `__wrapped__`)도
받으므로 던더를 전부 거부하는 규칙을 먼저 넣었는데, **aten 에는 던더 이름 op 이 실제로 10 개
있습니다.** `torch/_decomp/decompositions.py:6239` 가 `register_inplace(aten.__iand__, aten.__and__)`
를 임포트 중에 합니다. 트리에서 뽑은 10 개만 허용합니다.

### 때운 것: 모르는 op 이름에도 답한다

`_jit_get_operation` 은 **이름이 op 인지 아닌지 모릅니다.** 스키마 데이터베이스가 없으므로,
던더가 아닌 모든 이름에 호출 가능 객체를 돌려줍니다.

- **결과:** `hasattr(torch.ops.aten, "무엇이든")` 이 항상 참입니다. 상류는 없는 op 에
  `AttributeError` 를 냅니다.
- **위험:** op 존재 여부로 기능을 탐지하는 코드가 잘못된 가지를 탑니다. 임포트 중에는 걸리지
  않았지만(EXIT=0), 실행 중에는 걸릴 수 있습니다.
- **고치는 방법:** 실제 op 이름표를 갖는 것. 벤더링 트리에 `native_functions.yaml` 은 없지만
  `_decomp` · `_meta_registrations` · `_refs` 가 참조하는 이름을 추출하면 상당 부분 덮입니다.

---

## 5. 스키마는 스텁이 아니라 실제 파서다

`torch._C.parse_schema` 는 임포트 중에 일곱 군데에서 불립니다(`torch/library.py:80`·`:151`,
`torch/_prims/__init__.py:322`, `torch/_library/custom_ops.py:742`, …). 그리고 호출자가 결과로
하는 일이 **인자를 세고, `kwarg_only` 텐서를 찾고, `alias_info.is_write` 를 보는 것**입니다.

빈 인자 목록을 돌려주는 스텁은 "가변 인자가 없다" 고 **모든 질문에 답합니다.** 그것은 없는 답이
아니라 틀린 답입니다. 그래서 문법을 실제로 읽습니다.

```
ns::name.overload(Type(alias) name=default, *, ...) -> (R1, R2)
```

`is_mutable()` · `_is_view_op()` 가 그 위에서 진짜로 계산됩니다.

> 이 파서를 쓰다가 조용한 버그를 하나 잡았습니다. 닫는 괄호를 `rindex(")")` 로 찾으면
> `aten::add_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)` 에서 **반환 타입의 별칭 표기**
> 를 집어, 인자가 두 개가 되고 두 번째 이름이 `Tensor(a!` 가 됩니다. 예외는 나지 않습니다.
> 깊이를 세어 짝을 찾도록 고쳤고, 회귀 테스트를 두었습니다
> (`test_parsed_schema_really_reads_the_schema`).

---

## 6. dtype — BOOL.md 의 권고 B 를 넣었고, 그것이 임계 경로였다

작업 지시에서 bool 은 "여유가 있으면" 항목이었습니다. **여유의 문제가 아니라 필수였습니다.**

`torch/_prims_common`, `torch/_tensor_str.py`, `torch/_refs` 가 `torch.bool` · `torch.int8` ·
`torch.complex64` · 양자화 dtype 위에 표를 만드는데, 그 코드가 `import torch` 중에 돕니다.
dtype 이 10 개인 shim 으로는 임포트를 끝낼 수 없습니다.

그래서 BOOL.md 의 선택지 B 를 **모든 dtype 으로 일반화**했습니다.

```rust
pub enum TorchDType { Float32, ..., Bool, ..., Bits16 }   // 33 개
fn storage(self) -> Option<candle_core::DType>            // 10 개만 Some
```

- **`_C` 가 태그를 소유하고 candle 이 저장을 소유합니다.** `device` 가 이미 쓰는 패턴과 같습니다.
- **`from_storage` 는 `storage` 의 역함수가 아닙니다.** `U8` 은 `uint8` 로만 돌아옵니다. bool
  태그는 저장에서 복원되지 않고 텐서에 함께 실려야 하므로, `PyTensorBase` 에 `tag` 필드를
  두었습니다.
- **bool 태그를 붙이는 입구는 `PyTensorBase::boolean` 하나입니다**(BOOL.md §6.3 항목 1).
  `BRAINWAVE_CHECK_BOOL=1` 이면 그 자리에서 바이트가 0/1 인지 확인합니다(항목 2).
- **`_tensor_from_flat` 은 bool 태그를 거부합니다.** BOOL.md §6.3 이 불변식을 조용히 깰 수 있는
  두 입구 중 하나로 지목한 자리이고, 삭제 예정인 임시 함수가 shim 에게 거짓말을 가르칠 이유가
  없습니다.

측정 가능한 결과:

```
torch.bool != torch.uint8            True
torch.float is torch.float32         True     (별칭은 같은 객체)
torch.full((2,), True).dtype         torch.bool   (TORCH_C.md §2 의 미해결 항목)
tolist()                             [True, True]  (0/1 아님)
bool + bool                          NotImplementedError: ... logical or, not arithmetic
```

마지막 줄이 B 의 요점입니다. candle 의 `broadcast_add` 는 2 를 내놓고 2 는 truthy 라서 조용히
지나갑니다. 태그가 있으니 이름을 대고 멈춥니다.

### 곁가지로 필요해진 것

- **`.abbr`** — `torch/utils/_dtype_abbrs.py:5` 가 `{dt: dt.abbr for dt in _get_all_dtypes()}` 를
  임포트 중에 만듭니다. 이름에서 유도할 수 없습니다(`bits8` 은 `b8x1`, `bool` 은 `b8`).
  torch 2.13.0 에서 읽었습니다.
- **`_get_all_dtypes()`** — 양자화 5 개를 뺀 27 개. torch 가 그렇습니다(실측).
- **`finfo` / `iinfo`** (`src/info.rs`) — `torch/ao/quantization/observer.py:238` 이
  `eps=torch.finfo(torch.float32).eps` 를 **클래스 본문 기본값**으로 씁니다. 숫자는 전부
  torch 에서 읽었습니다. 두 개는 형식에서 유도하면 틀립니다: `bfloat16.tiny` 는 bfloat16 의
  최소 정규수가 아니라 **float32 의 것**이고, `resolution` 은 표현 가능한 값이 아니라 반올림한
  십진수입니다.
- `torch.iinfo(torch.bool)` 은 torch 가 거부합니다. 우리도 거부합니다 — BOOL.md §7 이 센
  여섯 개 방어막 중 하나입니다.

---

## 7. 골든 하네스가 잡은 버그 2 종

```
FAIL aten.full.default :: full(shape=[3], fill=1e6, dtype=float16)
     SILENT DIVERGENCE: torch raised RuntimeError(...c10::Half without overflow)
     but c computed a value: [inf, inf, inf]
FAIL aten.full.default :: full(shape=[], fill=2147483648, dtype=int32)
     ... but c computed a value: -2147483648
```

`c10::checked_convert` 를 옮겨 왔습니다. **규칙은 C++ 을 읽은 것이 아니라 torch 2.13.0 을 돌려
잰 것**이고, 두 가지는 추측했으면 틀렸을 것입니다.

### (1) 음수는 부호 없는 dtype 으로 감기는 것이 허용된다

`torch.full((3,), -1, dtype=torch.uint8)` 은 **255 로 성공합니다.** `-300` 은 실패합니다.
c10 의 표현으로 "allow for negative numbers to wrap using two's complement arithmetic" 입니다.
하네스에 이미 통과하고 있던 케이스라, 일괄 거부로 갔으면 회귀였습니다.

### (2) 원소 1 개짜리 축소 정밀도 float 은 검사를 건너뛴다 — 상류의 구멍

```
full([],  1e6, float16)  -> inf          (검사 안 함)
full([3], 1e6, float16)  -> RuntimeError
full([1], 1e6, float16)  -> inf
full([0], 1e6, float16)  -> RuntimeError
full([],  1e300, float32) -> RuntimeError  (float32 는 1 개여도 검사)
```

`fill_` 이 CPU numel==1 에서 변환이 검사되지 않는 빠른 경로를 타고, 그 경로가 Half · BFloat16 ·
Float8 에서만 무검사입니다.

**상류의 비일관성을 일부러 재현했습니다.** 근거는 두 가지입니다. 골든 하네스는 torch 와 대조하는
장치이므로 항상 거부하는 shim 은 **반대 방향으로** 어긋납니다. 그리고 이 프로젝트가 막으려는 것은
"torch 와 다른 답" 이지 "torch 가 못생긴 것" 이 아닙니다. 경계는 테스트로 박아 두었습니다
(`test_full_reproduces_torchs_one_element_hole`).

### 결과

```
$ /Volumes/macMini/caches/spike-venv/bin/python tools/golden/compare.py
SUMMARY: 188/188 cases passed, 0 failed, ops covered=3      EXIT=0

$ ... --inject-fault {value,shape,dtype}                     EXIT=1 / 1 / 1
```

`--inject-fault` 세 가지가 전부 여전히 1 로 떨어집니다 — 비교기가 고무도장이 되지 않았다는 확인입니다.

**`tools/golden/` 은 한 줄도 고치지 않았습니다.** 실패 5 건은 전부 `rust/torch_c` 쪽 변경으로
닫혔습니다.

---

## 8. 새로 밟은 벽 — 순서대로, 그리고 무엇을 했는지

`[구]` 구현 · `[스]` off-switch · `[때]` 때움.

| # | 벽 (파일:행) | | 처리 |
|---|---|---|---|
| 20 | `_dlpack_exchange_api()` — `torch/_tensor.py:105`, 클래스 본문 | `[구]` | `None` 반환. DLPack 브리지 없는 빌드가 갖는 값 |
| 21 | `TensorBase.__idiv__` — `torch/_tensor.py:1115` | `[구]` | 연산자 던더 생성, 위험 던더 24 개 제외 |
| 22 | `_ScalingType` · `_SwizzleType` — `torch/nn/functional.py:12` | `[구]` | 스텁에 없음 → 트리 스캔으로 보강 |
| 23 | `ErrorReport.call_stack()` — `torch/_sources.py:122` | `[구]` | `""`. TorchScript 컴파일 스택이 없으므로 빈 문자열이 정답 |
| 24 | 애노테이션이 진짜 타입이어야 함 — `nn/functional.py:7170` | `[구]` | 대문자 이름은 진짜 클래스로 생성 (`X \| list[X]`) |
| 25 | `_get_custom_class_python_wrapper` | `[때]` | no-op. 커스텀 클래스가 없으니 확인할 것도 없음 |
| 26 | `_register_opaque_type` | `[때]` | no-op. 스키마 레지스트리가 없음 |
| 27 | `_dispatch_keyset_full()` — `torch/_ops.py:304` | `[구]` | 진짜 `DispatchKeySet` (frozenset 위) |
| 28 | `TransformType.Jvp` 가 property — `torch/_ops.py:139` | `[구]` | 생성기 버그: 서브모듈 타입의 `bases` 를 버리고 있었음 |
| 29 | `_FunctionBase` 메타클래스 충돌 | `[구]` | 메타타입 예외 목록에 추가 |
| 30 | `torch._VF.stft` — `torch/jit/_builtins.py:117` | `[구]` | `_VariableFunctions` 에 catch-all (976 vs 상류 985) |
| 31 | `_tracer_warn_use_python()` | `[때]` | no-op. 트레이서가 없음 |
| 32 | `_get_all_dtypes()` + `.abbr` | `[구]` | Rust — §6 |
| 33 | `TensorBase.mtia` — `torch/_tensor_docs.py:10` | `[구]` | `add_docstr_all` 547 개를 스캔해 보강 |
| 34 | `torch.finfo(...).eps` — `ao/quantization/observer.py:238` | `[구]` | `src/info.rs` — §6 |
| 35 | `_additional_keys_to_prop_for_wrapper_tensors` | `[구]` | 빈 `DispatchKeySet` (값이지 함수가 아님) |
| 36 | `_dispatch_library(...)` — `torch/library.py:244` | **`[때]`** | 기록기. §9 의 최대 항목 |
| 37 | `IntType.get()` — `_higher_order_ops/schema.py:56` | `[구]` | JitType 싱글턴 14 개, 캐시해 동일성 유지 |
| 38 | `parse_schema` — 일곱 군데 | `[구]` | 진짜 파서 — §5 |
| 39 | `parse_schema` 가 바운드 메서드 — `torch/__init__.py:1091` | `[구]` | 평범한 함수로. 바운드 메서드는 `__module__` 대입 불가 |
| 40 | `_Schema._is_view_op()` — `_library/custom_ops.py:794` | `[구]` | `MathBitsFallback.h` 규칙 이식 |
| 41 | `register_ad_inplace_or_view_fallback` | `[때]` | 레지스트리 객체에 catch-all 기록기 |
| 42 | `_parse_dispatch_key` — `torch/library.py:915` | `[구]` | 스텁 enum 조회 |
| 43 | `_dispatch_has_kernel` — `torch/_decomp/__init__.py:90` | **`[때]`** | `True`. §9 |
| 44 | `aten.__iand__` — `_decomp/decompositions.py:6239` | `[구]` | aten 던더 op 10 개 허용 — §4 |
| 45 | `_dispatch_get_computed_kernel_for_dispatch_key` — `torch/_native/registry.py:894` | `[때]` | `None`. 앞선 커널이 없음 |

45 번을 넘기면 `import torch` 가 끝납니다.

---

## 9. 때운 것 — 나중에 무엇을 물게 되는가

**여기 있는 것만 때운 것입니다.** §8 에서 `[구]` 로 표시한 것은 요구된 동작을 실제로 하고,
`[스]` 는 상류가 제공하는 스위치입니다.

### 9.1 `_dispatch_library` — 등록이 무효다 (가장 큼)

`torch/library.py:244` 가 `Library(...)` 마다 하나씩 만들고, `torch/_meta_registrations.py` ·
`torch/_decomp/` · `torch/_refs` 가 **임포트 중에** `define`/`impl` 을 부릅니다.

상류의 것은 C++ 디스패처에 씁니다. 우리 것은 **기록하고 버립니다.** 구체적으로:

> `torch.library.impl(...)` 이 성공한 것처럼 보이고 아무 효과가 없습니다.

`import torch` 한 번에 **1549 건**이 그렇게 버려집니다. `_C._shim_registrations` 로 읽을 수
있게 두었습니다 — **그 목록의 길이가 구멍의 크기**입니다.

`_aten_dispatch` 만이 호출에 답하므로 단일 관문은 유지되지만, 파이썬에서 등록한 커널이 잡히지
않는다는 뜻입니다. `torch.library` 로 op 을 확장할 계획이 있다면 여기가 시작점입니다.

### 9.2 `_dispatch_has_kernel` → 항상 `True`

`torch/_decomp/__init__.py:90` 이 TorchScript 의 쓰레기 오버로드(`aten.add.float_int`)를 분해
레지스트리에서 걸러내는 데 씁니다.

**방향을 골랐습니다.** `False` 면 분해 표가 통째로 비고, 그것은 DESIGN.md §2 가 "Core ATen 밖
롱테일이 자동 분해된다" 며 의존하는 바로 그 기제입니다. `True` 면 표(973 항목)가 살고 쓰레기
오버로드 몇 개가 섞여 들어와 그냥 놀고 있습니다.

**거짓말의 값은 레지스트리 항목 몇 개고, 진실의 값은 기능 전체입니다.**

### 9.3 op 이름 조회가 실패하지 않는다

§4 참조. `hasattr(torch.ops.aten, <아무거나>)` 가 항상 참입니다.

### 9.4 나머지 no-op 넷

`_get_custom_class_python_wrapper` · `_register_opaque_type` · `_tracer_warn_use_python` ·
`_dispatch_get_computed_kernel_for_dispatch_key`. 전부 반환값을 쓰지 않거나 없는 서브시스템의
전역을 건드리는 호출입니다. 커스텀 클래스 · opaque 타입 스키마 · TorchScript 트레이싱 ·
`torch._native` 커널 오버라이드를 실제로 쓰게 되면 각각이 다시 나타납니다.

### 9.5 `torch_shm_manager` 0 바이트 표식

VENDOR.md 벽 4 그대로. 존재만 검사하므로 유효하고, 표식으로 남겨 요구 사항이 보이게 둡니다.

---

## 10. 검증

**전부 종료 코드입니다.**

| | 명령 | 결과 |
|---|---|---|
| 호스트 스모크 | `rust/torch_c/pytests/run.sh` | **0** — 27/27 |
| 엄격 `import torch` | `probe.py --mode strict --target torch` | **0** |
| 기록 `import torch` | `probe.py --mode record --target torch` | **0** |
| 엄격 `import transformers` | `probe.py --mode strict --target transformers` | **0** |
| 골든 하네스 | `tools/golden/compare.py` | **0** — 188/188 |
| 골든 자가검사 | `--inject-fault value/shape/dtype` | **1 / 1 / 1** (의도대로) |
| `aarch64-apple-darwin` | `cargo build --release` | **0** |
| `aarch64-linux-android` | `cargo ndk -t arm64-v8a` | **0** |
| `aarch64-apple-ios` | `cargo build --target aarch64-apple-ios` | **0** |

### 크기 — 표면을 넣은 값

| 타깃 | 이전 (VENDOR.md §1) | 지금 | 차이 |
|---|---|---|---|
| 호스트 | 1,406,592 | 1,740,064 | +333,472 |
| Android | 2,274,800 | 2,618,312 | +343,512 |
| iOS | 1,496,544 | 1,811,208 | +314,664 |

증가분은 대체로 `surface.json` 120,686 B + `bootstrap.py` 약 46 KB 이고, 셋이 같은 방향으로
같은 크기만큼 움직였습니다 — abi3 때와 달리 원인이 분명합니다.

iOS 는 여전히 `@rpath/Python.framework/Python` 로 링크되고 `_Py*` 미정의 심볼 98 개가 남습니다
(올바른 모양).

### 임포트 비용

```
import _C     (단독)   15.2 ms   (자기 몫 11.9 ms — 표면을 짓는 시간)
import torch          1.15 s
```

기기에서의 첫 임포트 비용은 미측정입니다(§11).

---

## 11. 남은 벽과 미확인

| # | 항목 | 상태 |
|---|---|---|
| 1 | **`torch` 네임스페이스 59 vs 상류 74** | 차이는 하위 바이트 dtype 15 개 (`bit`, `int1`–`int7`, `uint1`–`uint7`). 벤더링 `.pyi` 가 선언하지 않아 넣지 않았습니다. 임포트는 통과하고, 실행 중 요구되는지는 **미측정** |
| 2 | `isinstance(X, _C._TensorMeta)` 가 항상 참 | §3. 트리는 이렇게 쓰지 않지만 상류와 다름 |
| 3 | **`from_config` 는 여전히 실패** | 다만 `import torch` 밖에서 — `transformers` 의 `GenerationMixin` 지연 임포트에서 멈춥니다. IMPORT_WALLS 가 지목한 `@auto_docstring`(범주 6) 영역에 **처음 도달**했습니다 |
| 4 | `torch.library` 등록이 무효 | §9.1. 1549 건 |
| 5 | op 이름 조회가 실패하지 않음 | §9.3 |
| 6 | 순전파 · `generate` | 미도달. 3 에 막힘 |
| 7 | **기기(Android · iOS)에서의 임포트** | 이번에도 링크만. 표면을 짓는 11.9 ms 와 45 KB 파이썬 소스 실행이 기기에서 어떤지 **미확인** |
| 8 | `TORCH_USE_RTLD_GLOBAL` 의 기기 영향 | VENDOR.md §7 항목 4 그대로 |
| 9 | `aarch64-apple-ios-sim` | 이번에도 빌드하지 않음 |
| 10 | `float8_e4m3fn` 텐서 생성이 멈추는 현상 | 골든 하네스가 양쪽 독립적으로 관측한 그대로. **원인 미조사**. **정정 (문서 감사, 2026-09):** 오늘은 멈추지 않는다 — `torch.full((3,), 1.0, dtype=torch.float8_e4m3fn)` 가 5초 알람 안에서 정상 반환된다. 원인은 여전히 추적하지 않았다(무엇이 고쳤는지는 미확인), 증상만 재확인 |
| 11 | dtype 승격표 | 여전히 미구현. `add.Tensor` 가 이름을 대고 거부. **확인 (문서 감사, 2026-09):** 오늘도 여전히 그렇다 — `docs/design/TORCH_C.md` 감사(라운드 2)가 같은 것을 확인했고, `docs/bindings/TENSORBASE.md` §2.3 이 이것을 **의도적** 결정으로 명명한다(자리표시자가 아니라) |
| 12 | bool 연산 (BOOL.md §6.2 의 표) | 태그와 단일 생성자·불변식 검사는 들어갔지만, `bitwise_*`·`any`·`masked_fill` 커널은 **미구현**. **정정 (문서 감사, 2026-09):** 전부 닫혔다. `aten.bitwise_and.Tensor`/`aten.bitwise_and.Scalar`/`aten.bitwise_or.Tensor`/`aten.bitwise_or.Scalar`/`aten.bitwise_not.default`/`aten.any.default`/`aten.any.dim`/`aten.masked_fill.Scalar`/`aten.masked_fill_.Scalar` 가 전부 오늘 168-op `_aten_implemented()` 목록에 있다. 실측: `x & y`, `t.any()`, `x.masked_fill(mask, 0.0)` 전부 오늘 shim 위에서 계산된다 |
| 13 | 재벤더링 시 표면이 얼마나 흔들리는지 | `surface.json` 은 2.13.0 에서 생성. 다른 판본에서 재생성했을 때의 차이 **미측정** |
| 14 | `_shim_registrations` 1549 건 중 실제로 필요한 비율 | 미측정 |

<!-- DOCWATCH: op-implemented aten.bitwise_and.Tensor -->
<!-- DOCWATCH: op-implemented aten.any.default -->
<!-- DOCWATCH: op-implemented aten.masked_fill.Scalar -->

---

## 12. 재현

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target
cd /Volumes/macMini/thisisthepy/torchnative

./vendor/vendor_torch.sh          # 없으면
./vendor/gen_surface.py           # .pyi -> rust/torch_c/src/surface.json
./vendor/install_shim.sh          # 빌드 + 구멍에 넣기

PY=/Volumes/macMini/caches/spike-venv/bin/python
PYTHONDONTWRITEBYTECODE=1 TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/vendor \
  $PY vendor/probe.py --mode strict --target torch; echo "EXIT=$?"

$PY tools/golden/compare.py; echo "EXIT=$?"
(cd rust/torch_c && ./pytests/run.sh); echo "EXIT=$?"
```

**`gen_surface.py` 는 벤더링 트리를 요구합니다.** 생성된 `surface.json` 은 크레이트에 있으므로
빌드 자체는 요구하지 않습니다 — 트리 없이도 `cargo build` 는 됩니다.

세 타깃 명령은 `docs/platform/RUST_CROSSBUILD.md` §0.5 그대로이고, 이번에 달라진 것은 없습니다.

---

## 13. 이번에 만진 것

| 파일 | 변경 |
|---|---|
| `rust/torch_c/src/bootstrap.py` | **신규** — 표면 빌더 (1297 행) |
| `rust/torch_c/src/surface.json` | **신규(생성물)** — `.pyi` 에서 뽑은 이름표 |
| `rust/torch_c/src/info.rs` | **신규** — `finfo` / `iinfo` |
| `rust/torch_c/src/dtype.rs` | candle `DType` 래퍼 → `_C` 소유 33 dtype + 별칭 · `abbr` · `_get_all_dtypes` |
| `rust/torch_c/src/tensor.rs` | dtype 태그 필드, bool 단일 생성자와 불변식 검사, bool `tolist` |
| `rust/torch_c/src/aten.rs` | `checked_convert`(골든 버그 2 종), bool 팩토리, 태그 기준 dtype 비교 |
| `rust/torch_c/src/lib.rs` | 부트스트랩 실행, `_tensor_from_flat` 의 bool 거부 |
| `rust/torch_c/pytests/test_shim.py` | 13 → 27 개. 승격 메시지 단언을 torch 철자로 |
| `vendor/gen_surface.py` | **신규** — `.pyi` + 트리 스캔 → `surface.json` |

**벤더링 트리의 파이썬 소스는 여전히 한 줄도 고치지 않았습니다.** `docs/` 의 다른 파일도
건드리지 않았습니다.

### 테스트 하나를 고쳤다 — 이유

`test_add_refuses_to_guess_a_promotion` 이 승격 거부 메시지에서 `"f32"` 와 `"f64"` 를 찾고
있었습니다. 메시지가 이제 torch 철자(`float32` / `float64`)를 씁니다.

**candle 의 철자로는 `torch.bool` 과 `torch.uint8` 을 구별할 수 없습니다** — 아래에서는 둘 다
`u8` 입니다. 그 구별을 못 하는 메시지는 이 구별이 존재하는 이유인 버그를 보고할 수 없습니다.
동작 변경이 의도된 것이므로 단언을 따라 옮겼습니다.


> **정정 (`docs/design/REGISTRATIONS.md`).** 이 절이 `_dispatch_has_kernel` 을 `True` 로 고른 이유로
> "`False` 로 하면 973 항목짜리 분해 테이블이 비어버린다" 를 들었는데, **그 숫자와 그 인과가 모두
> 틀렸습니다.**
>
> 실측하면 진짜 torch 의 `decomposition_table` 은 **1097**(`core_aten_decompositions()` 는 940)
> 이고, 우리 shim 은 **592** 입니다. 즉 `True` 로 골랐는데도 테이블이 절반가량입니다 —
> **그 선택이 목적을 달성하지 못했습니다.**
>
> 부족분의 원인은 `_dispatch_has_kernel` 이 아니라 **`_jit_get_operation` 이 항상
> `overload_names=["default"]` 를 돌려주는 것**입니다. `OpOverloadPacket.op_overloads()` 가 op 마다
> 오버로드를 하나씩만 지어내서, 진짜 오버로드 715 개(`.out` · `.Scalar` · in-place)가 빠지고
> 대응물 없는 유령 `.default` 210 개가 생깁니다.
>
> 그리고 `_dispatch_has_kernel=True` 는 **혼자 설 수 없는 거짓말**입니다. 진짜 torch 에 같은 값을
> 강제하면 테이블이 1097 → 1348 로 늘고, torch 자신의 `activate_meta()` 가 그 251 개를 검증하다
> **`RuntimeError` 로 죽습니다.** 우리 shim 이 살아남는 것은 `_dispatch_has_kernel_for_dispatch_key`
> 도 이름을 검증하지 않고 항상 `False` 를 돌려주기 때문입니다 — **두 플래그가 짝입니다.**
