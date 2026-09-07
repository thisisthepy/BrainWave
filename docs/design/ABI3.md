# `torch._C` 를 abi3 로 만들 것인가

`rust/torch_c` 를 CPython Limited API(abi3) 로 빌드할지, 특정 버전(3.13)에 고정할지에 대한
판단 근거입니다. `docs/platform/RUST_CROSSBUILD.md` §1 이 **미결**로 남긴 항목입니다.

현재 상태: `rust/torch_c/Cargo.toml:13` 이 `features = ["extension-module"]` — **non-abi3(버전 고정)**
이고, 이 상태로 세 타깃이 빌드됩니다.

> **Correction (문서 감사, 2026-09):** 이 문서의 권고("abi3 를 켜라, floor 는 `abi3-py313`")가
> 채택됐습니다. `rust/torch_c/Cargo.toml` 의 `pyo3` 의존성이 지금
> `features = ["extension-module", "abi3-py313"]` 입니다(줄 번호는 이동했을 수 있음, 실측
> 재확인 2026-09). "현재 상태" 줄은 이 문서를 쓴 시점의 상태로 남겨 두고 여기서 정정합니다.
> <!-- DOCWATCH: symbol-in-file rust/torch_c/Cargo.toml abi3-py313 present -->

---

## 결론부터

**abi3 를 켜라. floor 는 `abi3-py313`.**

가장 무거운 근거는 성능도 기능도 아니라 **되돌리는 비용의 비대칭**입니다.

| 방향 | 비용 |
|---|---|
| abi3 → 버전 고정 | **싸다.** Limited API 는 전체 API 의 부분집합이므로, abi3 로 쓴 코드는 버전 고정 빌드에서 그대로 컴파일된다. feature 하나 빼고 세 타깃 재빌드. |
| 버전 고정 → abi3 | **비싸다.** 그동안 쓴 사설·구조체 API 를 전부 찾아 대체해야 하고, 일부는 대체가 없다. 그리고 실패가 조용하다(아래 §5). |

그 위에 실측 근거가 얹힙니다: abi3 로 잃는 기능은 **이 프로젝트에서 하나도 확인되지 않았고**(§2),
비용은 경계 호출당 약 1 ns(§4), 크로스 빌드 배선에는 **아무 영향이 없습니다**(§3).

그리고 버전 고정 쪽 전제 하나가 이미 틀렸습니다 — **PythonMultiplatform 은 3.13 이 아니라
3.14.7 입니다**(§5).

---

## 0. 이 문서의 실험 환경

세 종류의 근거를 씁니다. 출처를 구분해 표시합니다.

| 표시 | 뜻 |
|---|---|
| **[실측]** | 이 조사에서 직접 빌드하거나 실행해 얻은 것 |
| **[출처]** | 파일 경로 · 행 번호, 또는 공식 문서 URL |
| **[추론]** | 이름과 API 성격으로 분류한 것. 측정하지 않음 |

실험은 저장소 밖 `/Volumes/macMini/caches/abi3-probe` 에서 `rust/torch_c` 를 복사해 진행했습니다.
**저장소의 어떤 파일도 수정하지 않았습니다.**

- 호스트: macOS 26 (Darwin 25.5.0), aarch64, 8 코어 / 16 GB
- Rust: `aarch64-apple-darwin`, `aarch64-apple-ios`, `aarch64-apple-ios-sim`, `aarch64-linux-android` 설치됨
- PyO3 **0.29.2**, pyo3-ffi 0.29.2, pyo3-build-config 0.29.2
- 호스트 CPython **3.13.0**, 비교용 **3.11.10** (`/opt/homebrew/bin/python3.11`)
- 타깃 CPython: `/Volumes/macMini/caches/target-python/{aarch64-linux-android,arm64-iphoneos}`
- 참조 torch: `/Volumes/macMini/caches/spike-venv` 의 **torch 2.13.0** (`torch/version.py`)
- Stable ABI 정의: `python/cpython` `3.13` 브랜치의 `Misc/stable_abi.toml` (2525 행, 항목 1121 개)

> 조사 중(15:55) 다른 작업이 `target-python/arm64-iphoneos/linkstub` 을 `linkstub.disabled` 로
> 이름을 바꿨습니다. iOS 빌드는 그 새 경로를 `-L native=` 로 가리켜 재현했습니다. 그 작업의
> 결과에 따라 iOS 링크 배선은 달라질 수 있고, 이 문서의 결론은 그 배선과 무관합니다(§3).

---

## 1. Limited API 로 못 하는 것

### 1a. CPython 쪽 정의

[출처] <https://docs.python.org/3/c-api/stable.html>

`Py_LIMITED_API` 를 정의하면 `Python.h` 가 노출하는 것이 부분집합으로 줄어듭니다. 제외되는 것:

- **구조체 내부** — 버전마다 배치가 달라지는 필드 직접 접근
- **빠른 매크로 변형** — `PyList_GET_ITEM()` 은 빠지고 `PyList_GetItem()` 만 남는다
- **static inline 함수와 매크로** — 인라이닝이 꺼진다
- **사설 API** — 밑줄로 시작하는 것. 패치 릴리스에서도 예고 없이 바뀔 수 있다

같은 문서가 두 가지를 경고합니다. 둘 다 이 결정에 직접 걸립니다.

> "Python does **not** verify that such extensions actually conform to the Stable ABI."
> — `abi3` 태그가 붙은 `.so` 라도 인터프리터는 검사하지 않는다.

> "Compiling with `Py_LIMITED_API` is not a complete guarantee of Limited API/Stable ABI
> conformance" — 정의만 덮을 뿐 의미(semantics)는 덮지 않는다.

### 1b. PyO3 쪽 제약 — floor 를 313 으로 잡으면 **전부 사라진다**

[출처] <https://pyo3.rs/latest/building-and-distribution.html>

PyO3 가 abi3 에서 안 된다고 명시하는 것은 다섯 개입니다.

| 제약 | 해소되는 버전 | `abi3-py313` 에서 |
|---|---|---|
| `#[pyo3(text_signature)]` on classes | 3.10+ | **해당 없음** |
| `dict` / `weakref` on classes | 3.9+ | **해당 없음** |
| 버퍼 API | 3.11+ | **해당 없음** |
| 네이티브 타입 상속 (`PyException` 등) | 3.12+ | **해당 없음** |
| "컴파일 대상 파이썬 버전을 정확히 아는 데 의존하는 최적화" | — | **남는다** (§4 에서 측정) |

**목록의 넷이 전부 "3.X 이상에서는 된다" 형태이고 그 X 가 모두 3.12 이하입니다.**
floor 를 3.13 으로 잡으면 실질적으로 남는 abi3 제약은 마지막 하나 — 성능 — 뿐입니다.

`pyo3 0.29.2` 의 feature 목록: `abi3-py38` ~ `abi3-py315`, 그리고 free-threaded 용 `abi3t` /
`abi3t-py315`. [출처] `~/.cargo/registry/src/index.crates.io-*/pyo3-0.29.2/Cargo.toml:62-98`

### 1c. [실측] torch 모양의 확장을 abi3 로 실제로 만들어 봤다

`torch._C.TensorBase` 가 요구하는 성질을 모아 하나의 `#[pyclass]` 로 만들고 `abi3-py313` 으로
빌드해 3.13 에서 실행했습니다. 확인한 것:

| 요구 성질 | 왜 필요한가 | abi3-py313 |
|---|---|---|
| `#[pyclass(subclass)]` + 파이썬 서브클래싱 | `class Parameter(torch.Tensor)` | **동작** |
| `dict` (인스턴스 `__dict__`) | `p.requires_grad = True` | **동작** |
| `weakref` | autograd 훅 · 캐시 | **동작** |
| 버퍼 프로토콜 (`__getbuffer__` / `PyBuffer_FillInfo`) | `memoryview(t)`, numpy 상호운용 | **동작** (32 바이트, 쓰기 반영 확인) |
| `PyCapsule` 왕복 | **DLPack** — 텐서 무복사 교환의 실제 경로 | **동작** |
| `create_exception!` + 파이썬 서브클래싱 | `torch.TorchError` 계층 | **동작** |
| `#[classmethod]`, `__len__`, `__repr__` | `torch.Tensor` 의 던더들 | **동작** |

동일 소스가 **세 타깃 모두 빌드**됩니다 — `aarch64-apple-darwin`, `aarch64-apple-ios`,
`aarch64-linux-android` (ios-sim 은 링크 배선 문제로 미시도, §3 참조).

그리고 생성된 산출물을 기계적으로 감사했습니다.

```
iOS     abi3 .dylib : 미해결 CPython 심볼 89 개 → stable ABI 밖 0 개
Android abi3 .so    : 미해결 CPython 심볼 89 개 → stable ABI 밖 0 개
```

`nm -u` 결과를 `Misc/stable_abi.toml` 항목 1121 개와 대조한 것입니다. **위반 0.**

---

## 2. `torch._C` 가 실제로 요구하는 것

### 2a. 상류 torch 는 Limited API 밖을 많이 쓴다 — 하지만 그것은 우리 얘기가 아니다

[실측] torch 2.13.0 의 `libtorch_python.dylib` (29.9 MB) 가 참조하는 CPython 심볼 **341 개** 중
**54 개(16%)** 가 stable ABI 밖입니다.

이것만 보면 "Limited API 로는 `torch._C` 를 못 만든다"로 읽히지만, **54 개의 내역을 보면
그렇지 않습니다.** [추론] API 성격으로 분류하면:

| 분류 | 개수 | 예 | 추론 경로에 필요한가 |
|---|---|---|---|
| **프레임 평가 후킹 · 코드 객체 조사** | 21 | `_PyEval_EvalFrameDefault`, `_PyInterpreterState_SetEvalFrameFunc`, `PyUnstable_Code_{Get,Set}Extra`, `PyFrame_*`, `PyFunction_*`, `PyCode_*` | **아니오** — TorchDynamo (PEP 523) |
| **dict 워처** | 3 | `PyDict_Watch`, `PyDict_AddWatcher`, `PyDict_Unwatch` | **아니오** — dynamo 가드 |
| **프로파일러 · 스레드 순회** | 6 | `PyEval_SetProfileAllThreads`, `PyThreadState_Next`, `PyGILState_Check` | **아니오** — `torch.profiler` |
| **타입/속성 조회 최적화** | 5 | `_PyType_Lookup`, `_PyObject_GetDictPtr`, `PyType_GetDict`, `PyObject_{Clear,Visit}ManagedDict` | **여기가 유일한 실질 위험** (아래) |
| **대체 있는 편의 함수** | 12 | `PyObject_CallOneArg`→`PyObject_Vectorcall`(3.12), `PyUnicode_AsUTF8`→`AsUTF8AndSize`(3.10), `PyStructSequence_InitType`→`NewType` | 대체로 해결 |
| **구조체 값 전달** | 2 | `PyComplex_{As,From}CComplex` | 복소 텐서를 안 다루면 무관 |
| **사설 문자 분류** | 4 | `_PyUnicode_IsDigit` 등 | 무관 |
| **아레나 할당자** | 1 | `PyObject_GetArenaAllocator` | 무관 |

**IMPORT_WALLS.md 5 차의 결과가 이 분류를 그대로 뒷받침합니다.** [출처] `docs/models/IMPORT_WALLS.md:243-296`
— 임포트되는 torch 모듈 1084 개 중 추론 중 파이썬이 실행되는 것은 **14 개(1.3%)**, 실질은 10 개.
그리고 그 10 개 안에 dynamo · 프로파일러 · 코드 객체 조사가 **하나도 없습니다.** 실행 목록의
`torch.compiler` · `torch.jit._trace` 는 IMPORT_WALLS 가 짚은 대로 "일하는 것이 아니라 질문받는
것" (`is_compiling()` · `is_tracing()`) 이고, 그 답은 우리 `_C` 에서 그냥 `False` 입니다.

즉 **54 개 중 30 개(프레임·워처·프로파일러)는 우리가 구현하지 않을 기능의 것**입니다.

### 2b. [실측] 실행되는 10 개가 `_C` 에 무엇을 요구하나

IMPORT_WALLS 가 지목한 10 개 파이썬 모듈에서 `_C.<이름>` 접근을 전부 뽑았습니다.
(`site-packages/torch` 안에서 `grep -oE '_C\.[A-Za-z_][A-Za-z0-9_.]*'`)

| 모듈 | `_C` 표면 |
|---|---|
| `nn/functional.py` | `_C._nn.*` **68 개** — `linear`, `gelu`, `silu`, `scaled_dot_product_attention`, `softplus`, `cross_entropy_loss`, `pad`, 각종 pooling / upsample |
| `_tensor.py` | `TensorBase` 와 그 메서드, `_add_docstr`, **`_to_dlpack` · `_to_dlpack_versioned` · `_dlpack_exchange_api`**, `DisableTorchFunctionSubclass`, `_VariableFunctions.rsub` |
| `autograd/grad_mode.py` | `_set_grad_enabled`, `_InferenceMode`, `_is_multithreading_enabled`, `_is_view_replay_enabled`, `_autograd._unsafe_set_version_counter` |
| `nn/modules/module.py` | `_get_tracing_state`, `_log_api_usage_once`, `_nn._parse_to`, `ScriptMethod` |
| `_utils.py` | `_get_tensor_metadata`, `_set_dispatch_mode`, `_TorchDispatchModeKey.FUNCTIONAL`, `_nn.{,un}flatten_dense_tensors` |
| `_jit_internal.py` | `ScriptFunction`, `TensorType.getInferred`, `_jit_{get,set}_emit_hooks` |
| `nn/modules/{linear,sparse,container}.py`, `utils/_contextlib.py` | 거의 없음 (`_get_privateuse1_backend_name` 하나) |

**이 표면 전체가 Limited API 안에서 표현됩니다.**

- 68 개의 `_C._nn.*` 는 전부 **평범한 모듈 수준 함수** — `PyModule_AddObjectRef` + `PyCFunction`.
- `TensorBase` 는 **`PyType_FromSpec` 으로 만드는 힙 타입**이면 되고, §1c 에서 상속·`dict`·
  `weakref`·버퍼가 전부 동작함을 확인했습니다.
- **DLPack 이 결정적입니다.** 텐서 무복사 교환의 실제 경로가 `PyCapsule` 인데,
  `PyCapsule_New` 는 **3.2 부터 stable ABI** 입니다. 즉 데이터를 옮기는 가장 중요한 길이
  Limited API 안에 통째로 들어 있습니다.
  [실측] `stable_abi.toml` — `PyCapsule_New ('function', '3.2')`

### 2c. 진짜 위험 하나 — `__torch_function__` / `__torch_dispatch__`

상류가 `_PyType_Lookup` 과 `_PyObject_GetDictPtr` 을 쓰는 이유는 **모든 텐서 연산마다** "이 인자에
`__torch_function__` 이 있나" 를 물어야 하는데, 그것을 `PyObject_GetAttr` 로 하면 MRO 를 돌고
디스크립터 프로토콜을 태우고 실패 시 예외까지 만들기 때문입니다.

Limited API 에는 이 사설 함수가 없습니다. 대체는 있습니다 —
**`PyObject_GetOptionalAttr` (3.13 에 stable ABI 로 추가)** 는 없을 때 예외를 만들지 않고 `0` 을
돌려줍니다. `PyDict_GetItemRef` · `PyList_GetItemRef` 도 3.13 stable ABI 입니다.
[실측] `stable_abi.toml` — 셋 다 `added = '3.13'`

**이것이 floor 를 3.13 으로 잡아야 하는 이유입니다.** 3.12 이하로 내리면 이 대체가 없어져
`PyObject_GetAttr` + `PyErr_Clear` 라는 진짜 느린 길밖에 남지 않습니다.

**[미확인]** 대체 경로의 실제 비용은 재지 않았습니다. 우리 `_C` 가 `__torch_function__` 을 어떤
구조로 구현할지 아직 정해지지 않았으므로, 이것은 설계가 정해진 뒤에 측정할 항목입니다.

### 2d. [미확인] 남은 것

- **candle 쪽 요구사항** — `_C` 뒤에 무엇이 들어가는지에 따라 파이썬 C API 요구가 달라질 수 있음.
  §5(A/B) 가 정해지기 전에는 판단 불가.
- **양자화(`torch.ao`, 75 모듈)** 경로 — IMPORT_WALLS 5 차는 양자화 없이 측정했습니다.
- **`generate` 이후** — `online()` · 학습 경로는 이 조사 범위 밖.

---

## 3. [실측] 크로스 빌드 배선에는 영향이 **없다**

"abi3 를 켜면 타깃 파이썬을 안 찾아도 되니 크로스 빌드가 쉬워진다"는 기대가 있을 수 있어
대조 실험을 했습니다. **성립하지 않습니다.**

```
실험 A  abi3-py313,  PYO3_CROSS=1 만                          → ld: library 'python3.13' not found
실험 B  non-abi3,    PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 → ld: library 'python3.13' not found
```

**둘의 링크 실패가 완전히 같습니다.** 이유는 PyO3 소스에 있습니다.

[출처] `pyo3-build-config-0.29.2/src/impl_.rs:1587-1596`

```rust
pub fn is_linking_libpython_for_target(target: &Triple) -> bool {
    target.operating_system == OperatingSystem::Windows
        || ...
        || target.environment == Environment::Android
        || matches!(target.operating_system, OperatingSystem::IOS(_))
        || !is_extension_module()
}
```

**iOS 와 Android 는 `extension-module` 이든 아니든 무조건 libpython 을 링크합니다.**
그리고 이름을 고르는 함수:

[출처] `pyo3-build-config-0.29.2/src/impl_.rs:2445` `default_lib_name_unix`

```rust
PythonAbiKind::Stable(StableAbi::Abi3) if use_stable_abi_lib => Ok("python3".to_string()),
```

`use_stable_abi_lib` 는 **유닉스 호출부에서 전부 `cygwin` 플래그에 묶여** 있습니다. 즉
**유닉스에서 abi3 를 켜도 링크 이름은 `python3` 이 되지 않고 `python3.13` 그대로입니다.**
`python3` 로 바뀌는 것은 Windows 뿐입니다.

abi3 가 실제로 덜어주는 것은 하나뿐입니다 — **`PYO3_CROSS_PYTHON_VERSION` 을 안 줘도 됩니다**
(floor 에서 유도). 두 실험 모두 `PYO3_CROSS_LIB_DIR` 없이 링크 단계까지 도달했으므로,
sysconfigdata 탐색은 abi3 와 무관하게 이미 생략 가능합니다.

**그러므로 `docs/platform/RUST_CROSSBUILD.md` 가 기록한 링크 배선(링크스텁, 프레임워크 링크, NDK 툴체인)은
abi3 를 켜든 안 켜든 그대로 필요합니다.** 이 항목은 결정에서 빼야 합니다.

세 타깃 모두 abi3 로 링크가 성사되는 것은 확인했습니다.

```
aarch64-apple-darwin   abi3-py313  OK
aarch64-apple-ios      abi3-py313  OK  (linkstub.disabled 를 -L native 로 지정)
aarch64-linux-android  abi3-py313  OK  (NDK 27.1.12297006, API 24, PYO3_CROSS_LIB_DIR=prefix/lib)
```

**[미확인]** `aarch64-apple-ios-sim` 은 링크 배선이 조사 중에 바뀌어(§0) 성사시키지 못했습니다.
abi3 고유의 문제가 아니며 — 실패 메시지가 `ld: library 'python3.13' not found` 로 §3 의 대조 실험과
같습니다 — 배선이 정리되면 함께 통과할 것으로 봅니다. 실기기 로드는 **확인하지 않았습니다.**

---

## 4. [실측] abi3 의 비용은 얼마인가

abi3 를 켜면 인라인이던 것이 함수 호출이 됩니다. PyO3 소스에서 확인되는 지점:

| 무엇이 | 어디서 | 버전 고정 | abi3 |
|---|---|---|---|
| `Py_INCREF` | `pyo3-ffi-0.29.2/src/refcount.rs:179` | `ob_refcnt` 직접 증가 | **`_Py_IncRef()` 호출** |
| `Py_DECREF` | `pyo3-ffi-0.29.2/src/refcount.rs:255` | `ob_refcnt` 직접 감소 | **`_Py_DecRef()` 호출** |
| `Py_None()` | `pyo3-ffi-0.29.2/src/object.rs:661-663` | `&_Py_NoneStruct` 주소 | **`Py_GetConstantBorrowed()` 호출** |
| `PyType_HasFeature` | `pyo3-ffi-0.29.2/src/object.rs:722` | `(*ty).tp_flags` 읽기 | **`PyType_GetFlags()` 호출** |
| `Py_TYPE` | `pyo3-ffi-0.29.2/src/object.rs:209-222` | `(*ob).ob_type` | floor 3.13 에선 **인라인 유지**, floor **3.14 부터 호출** |

이것은 PyO3 의 선택이 아니라 CPython 의 선택입니다 — PEP 683(불멸 객체)과 PEP 703(NO GIL)을
위해 3.12 부터 Limited API 에서 `Py_INCREF`/`Py_DECREF` 를 불투명 함수 호출로 바꿨습니다.
`Py_SET_REFCNT` 는 3.13 부터, `Py_TYPE`/`Py_REFCNT` 는 **3.14 부터**.
[출처] <https://github.com/python/cpython/issues/105387>,
<https://discuss.python.org/t/py-type-and-py-refcnt-are-opaque-function-calls-in-limited-c-api-3-14/64887>

**산출물에서 그대로 확인됩니다.** 같은 소스를 두 방식으로 빌드해 `nm -u` 로 비교:

```
abi3       : _Py_IncRef  _Py_DecRef  Py_GetConstantBorrowed  PyType_GetFlags  PyTuple_Size  PyTuple_SetItem  ...
버전 고정  : _Py_Dealloc  _Py_NoneStruct  PyType_Modified  ...      (앞의 셋은 아예 없음)
```

버전 고정 빌드가 참조하는 `_Py_Dealloc` 과 `_Py_NoneStruct` 는 **둘 다 stable ABI 밖**입니다.
(크기: abi3 542,720 B / 버전 고정 562,080 B)

### 4a. 마이크로벤치

호스트 CPython 3.13.0, `timeit` `repeat=9` 의 최솟값, 3 회 반복.

| 케이스 | 버전 고정 (ns/op) | abi3-py313 (ns/op) | 차이 |
|---|---|---|---|
| `noop()` — None 반환 | 22.0 / 22.1 / 22.6 | 23.2 / 23.3 / 23.4 | **+1.2 (+5%)** |
| `roundtrip(o)` — 인자 그대로 반환 | 29.8 / 29.8 / 29.9 | 30.6 / 31.2 / 31.2 | **+1.1 (+4%)** |
| `make_list(64)` — 리스트 64 개 append | 602 / 605 / 613 | 622 / 623 / 633 | **+20 (+3%)** |
| `sum_tuple(64)` — 튜플 원소 64 개 읽기 | 332 / 333 / 333 | 477 / 477 / 477 | **+144 (+43%)** |
| `cell.bump()` — `#[pymethods]` 호출 | 26.6 / 26.7 / 26.7 | 26.6 / 26.6 / 26.7 | **0** |
| `cell.v` — `#[pyo3(get)]` | 24.1 / 24.2 / 24.2 | 23.5 / 23.7 / 23.9 | -0.5 (노이즈) |

**읽는 법.**

- **경계 하나당 약 1 ns.** `noop` · `roundtrip` 의 +1.1~1.2 ns 가 그 값입니다.
- **컨테이너 원소 접근당 약 2.25 ns.** `sum_tuple(64)` 의 +144 ns ÷ 64. `PyTuple_GET_ITEM`
  (인라인 로드) → `PyTuple_GetItem` (함수 호출 + 경계 검사 + 오류 경로) 의 차이입니다.
  **abi3 의 비용이 몰려 있는 곳은 여기 하나입니다.**
- **메서드 호출과 속성 접근은 차이가 없습니다.** vectorcall 경로가 양쪽 다 같기 때문입니다
  (`PyObject_Vectorcall` 은 3.12 부터 stable ABI).

### 4b. 이 비용이 실제 추론에서 얼마인가

[출처] `docs/models/IMPORT_WALLS.md:255` — 순전파 1 회에 파이썬 호출 474 건, 그중 torch 186 건.

경계 크로싱을 순전파당 수백~수천 건으로 잡으면 abi3 의 추가 비용은 **순전파당 마이크로초 단위**
입니다. 실제 연산은 C++/candle 커널에서 마이크로초~밀리초가 걸리므로 **측정 잡음에 묻힙니다.**

**단, `sum_tuple` 이 보여준 함정은 피해야 합니다.** 파이썬 컨테이너의 원소를 하나씩 도는 코드를
`_C` 안에 쓰면 원소당 2.25 ns 가 붙습니다. 텐서 데이터는 **DLPack 캡슐이나 버퍼 프로토콜로
한 번에** 넘기고, 리스트/튜플을 원소 단위로 순회하지 않는다는 규칙이면 이 비용은 발생하지
않습니다. 둘 다 Limited API 안에 있습니다(§2b, §1c).

이것은 PythonMultiplatform 의 ROADMAP 이 이미 같은 결론에 도달한 지점이기도 합니다.
[출처] `PythonMultiplatform/ROADMAP.md:564-567` — `PySequence_Fast_ITEMS` 는 `PyListObject->ob_item`
을 읽는 매크로라 Stable ABI 에서 쓸 수 없고, "한 번의 호출로 리스트 원소를 뽑는 지원되는 방법은
없다".

### 4c. 측정의 한계 — 반드시 함께 읽을 것

**측정 당시 기계 부하가 높았습니다** (`load average 10.79 → 12.53`, 8 코어).
`PythonMultiplatform/CLAUDE.md` 의 "측정 작업은 단독으로 돌린다" 규정을 어긴 조건입니다.

그럼에도 이 숫자를 쓰는 이유:

- `repeat=9` 의 **최솟값**을 취했고, 3 회 반복에서 편차가 **2% 미만**이었습니다.
- 두 빌드를 **교대로** 돌려 드리프트를 상쇄했습니다.
- **순위가 한 번도 뒤집히지 않았습니다.**

그래도 절대값은 신뢰하지 마십시오. **신뢰할 것은 차이의 부호와 대략의 크기입니다.**
정확한 값이 필요하면 부하 없는 상태에서 다시 재야 합니다.

### 4d. floor 를 올리면 느려진다

[실측] `abi3-py311` 로도 빌드해 3.13 에서 같은 벤치를 돌렸습니다:
`noop 22.7 / roundtrip 31.0 / make_list64 625.8 / sum_tuple64 456.4 / cell_bump 26.7 / cell_get 24.4`.
`abi3-py313` 과 유의미한 차이가 없습니다(오히려 `sum_tuple` 이 조금 낮은데, 코드 생성 우연으로
보고 과해석하지 않습니다).

**중요한 것은 반대 방향입니다.** §4 표의 마지막 줄 — floor 를 **3.14 로 올리면 `Py_TYPE` 이 함수
호출이 됩니다.** 즉 **floor 는 필요 없이 올리면 안 되는 값**입니다. 3.13 이 적절합니다.

---

## 5. 버전 고정의 실제 비용

### 5a. 전제가 이미 틀렸다 — 플랫폼은 3.13 이 아니다

이 결정의 원래 서술은 "PythonMultiplatform 의 `binary/` 에 3.13 배포본이 있다" 였습니다.
아카이브는 그렇지만 **살아 있는 설정은 다릅니다.**

```
binary/*.tar.zst, *.zip          cpython-3.13.0+20241008   (2026-08-10 이후 갱신 없음)
gradle.properties:26             pythonVersion=3.14.7      ← 실제로 빌드되는 것
gradle.properties:28             pythonBuildStandaloneRelease=20260807
```

[출처] `PythonMultiplatform/gradle.properties:26`

게다가 3.15 까지 이미 시험됐습니다.
[출처] `PythonMultiplatform/ROADMAP.md:1241-1270` — `-PpythonVersion=3.15.0rc1` 로 데스크톱
**208 테스트 0 실패**.

**즉 `torch._C` 를 3.13 에 고정하면, 첫 줄을 쓰기도 전에 이미 한 버전 뒤처집니다.**

### 5b. 버전 하나 올릴 때 다시 해야 하는 것

버전 고정을 택했을 때 `pythonVersion` 이 3.14 → 3.15 로 갈 때 드는 일:

1. `binary/` 의 배포본 교체 (6 개 아카이브, 약 500 MB)
2. `target-python/` 재추출, 헤더·`_sysconfigdata*` 경로 재확인
3. iOS 링크스텁 / 프레임워크 재생성 (`libpython3.15.dylib` 로)
4. `PYO3_CROSS_PYTHON_VERSION` · `.cargo/config.toml` 갱신
5. **세 타깃 재빌드 + 실기기 재검증**
6. 제거된 C API 가 있으면 소스 수정

6 번의 크기는 실측치가 있습니다. [출처] `PythonMultiplatform/ROADMAP.md:1252-1268` —
3.14 → 3.15 에서 제거된 것은 **함수 3 개**(`PySys_ResetWarnOptions`,
`PyImport_ImportModuleNoBlock`, `PyWeakref_GetObject`), 고칠 곳 **약 20 군데**, 동작 변화 없음.
**소스 수정 자체는 작습니다.**

**그러므로 버전 고정의 비용은 코드가 아니라 1~5 번, 즉 배포와 재검증입니다.**
abi3 를 켜면 1~3 은 여전히 필요하지만(파이썬 자체는 실어야 하므로) **4~5 가 사라집니다** —
`_C` 바이너리는 그대로 다시 씁니다.

### 5c. PythonMultiplatform 이 Stable ABI 를 쓴다는 말의 정확한 의미

이 결정의 근거로 제시된 것 — "이 생태계 전체가 Stable ABI 를 전제로 설계되어 있다,
`EmbedAPI.kt` 에 `expect` 선언 314 개" — 는 **절반만 맞습니다.**

[실측] `python-multiplatform/src/commonMain/kotlin/python/native/ffi/EmbedAPI.kt` 는 4161 행,
`expect` 선언 **314 개**. 각 함수에 `*Part of the Stable ABI.*` 주석이 붙어 있습니다.

그런데 ROADMAP 이 명시적으로 정정합니다.
[출처] `PythonMultiplatform/ROADMAP.md:1236-1239`

> `Py_LIMITED_API` **is never defined anywhere in this build** — desktop binds symbols by name at
> runtime through Panama, and the native targets cinterop against the full headers. "abi3" in this
> codebase means **a self-imposed rule about *which* functions to call, not a compilation mode.**

**즉 PythonMultiplatform 은 abi3 로 컴파일되지 않습니다.** 규율일 뿐입니다.
그러므로 **"플랫폼이 abi3 니까 `torch._C` 도 abi3 여야 한다"는 논증은 성립하지 않습니다.**
두 결정은 독립입니다.

**그리고 방향이 오히려 반대입니다.** PythonMultiplatform 은 Panama 로 **이름으로** 심볼을 찾기
때문에, 헤더에서 사라진 함수도 라이브러리에 남아 있으면 계속 동작합니다
(ROADMAP 이 `nm` 으로 확인). **컴파일된 Rust cdylib 에는 그 도피처가 없습니다.**
같은 저장소 안에서 Kotlin 쪽이 버전 이동에 강한 이유가 우리에게는 적용되지 않으므로,
`torch._C` 는 **자기 힘으로** 버전 이동에 견뎌야 합니다.

### 5d. [실측] 버전이 어긋나도 조용히 로드된다 — 이것이 가장 나쁘다

빌드 산출물의 파일명은 `_C.so` 입니다 (`rust/torch_c/src/lib.rs:1` 주석 — "renamed on install").
**여기에는 ABI 태그가 없습니다.** macOS 의 `EXTENSION_SUFFIXES` 는
`['.cpython-313-darwin.so', '.abi3.so', '.so']` 이므로 맨 `.so` 는 **어느 인터프리터에서든
로드 후보**입니다.

실제로 확인했습니다. **3.13 으로 빌드한 버전 고정 모듈을 3.11.10 에 넣었더니 그냥 돌아갔습니다.**

```
$ python3.11 -c "import _C; ..."     # 3.13 으로 빌드한 non-abi3 모듈
version torch._C spike
noop None
roundtrip 1234
sum_tuple 45
make_list [0, 1, 2, 3, 4]
cell 3 4
EXIT=0
```

**이것을 "괜찮다"로 읽으면 안 됩니다.** 이 스파이크가 작아서 우연히 살아난 것입니다. 같은 모듈이
`_Py_Dealloc` 과 `_Py_NoneStruct` 를 참조하고, 인라인된 `Py_DECREF` 는 3.13 의 불멸 객체 로직으로
3.11 의 `ob_refcnt` 를 직접 조작합니다. CPython 문서가 이미 못을 박아 놨습니다 —
"Python does not verify that such extensions actually conform to the Stable ABI."

**버전 고정의 진짜 위험은 "안 돌아간다"가 아니라 "돌아가다가 어긋난 곳에서 깨진다"입니다.**
그리고 그 깨짐은 참조 카운트 손상 — 즉 임의의 시점에 나는 크래시 — 으로 나타납니다.

비교. `abi3-py311` 로 빌드한 모듈은 **3.11.10 과 3.13.0 양쪽에서 모두 정상 동작**합니다.
[실측]

```
3.11.10  torch._C spike  45
3.13.0   torch._C spike  45
```

**[실측, 해소]** 이후 조사에서 `uv python install` 로 3.14.7 과 3.15.0rc1 을 확보해 확인했습니다.
작업 디렉터리는 저장소 밖 `/Volumes/macMini/caches/abi3-314`(스크립트만) —
**빌드는 새로 하지 않고**, §1c 를 만든 바로 그 `abi3-py313` 산출물
(`/Volumes/macMini/caches/abi3-probe/m-feat/_C.so`, 3.13.0 에서 컴파일)을 그대로 복사해
3.14.7 · 3.15.0rc1 인터프리터로 실행했습니다. **재컴파일 없이 같은 바이너리를 다른 인터프리터에
물린 것**이므로, 이것이 정확히 abi3 가 약속하는 전방 호환 그 자체를 시험합니다.

§1c 의 표면 전체를 다시 확인했습니다 — 서브클래싱, `dict`, `weakref`, 버퍼 프로토콜
(`memoryview` 로 32 바이트 읽고 `add_()` 뒤 값 변화가 반영되는지 `struct.unpack` 으로 확인),
`PyCapsule` 왕복, `create_exception!` + 파이썬 서브클래싱. 판정은 그레이들 규정과 동일하게
**종료 코드로만** 했습니다(출력 grep 안 함 — 이 프로젝트에서 트레이스백이 성공 마커와 겹친 전례가
있어서).

```
cpython-3.14.7-macos-aarch64-none (uv, python3.14)   exit=0, 8/8 통과
cpython-3.15.0rc1-macos-aarch64-none (uv, python3.15) exit=0, 8/8 통과
```

대조군으로 같은 스크립트를 호스트 3.13.0 에도 돌렸습니다 (`exit=0`, 8/8) — 스크립트 자체가
버전 무관하게 같은 판정을 내리는지 확인하기 위해서입니다. (초안 스크립트는 3.13.0 에서도
동일하게 두 항목이 실패했는데, 원인은 abi3/버전과 무관한 스크립트 버그 둘 — ①
`TensorBase` 가 `__new__` 만 정의하고 `__init__` 은 없어 서브클래스에서 `super().__init__(n)`
을 인자와 함께 부르면 `object.__init__` 이 거부함, ② `PyBuffer_FillInfo` 가 항상 포맷 `'B'`
(unsigned byte) 로 채우는데 자체 제작 검증 함수가 `PyBuffer<f32>` 로 읽으려 해 포맷 불일치로
거부됨 — 이었습니다. 스크립트를 고친 뒤 3.13.0 도 8/8 통과로 확인했습니다.)

**남는 것.** 이번 확인은 darwin 호스트뿐입니다. iOS·Android 기기에서 abi3 모듈을 3.14.7/3.15
로 로드하는 것은 여전히 **[미확인]**입니다(§9 항목 2, 원래부터 별도 항목). 그리고 3.14.7/3.15
용으로 abi3-py313 을 **새로 컴파일**해서 로드하는 경로는 시험하지 않았습니다 — 이번 시험은
"3.13 에서 컴파일한 abi3 바이너리가 3.14/3.15 에서 도는가"이고, 실제 배포에서는 어느 CPython
으로 컴파일하든(abi3 floor 만 3.13 이면) 무방하다는 것이 stable ABI 의 정의이므로 이 구분은
결정에 영향을 주지 않습니다.

**결론: §7 의 1 번 선결 조건이 해소됐습니다. 플랫폼의 3.14.7 에서 abi3-py313 모듈이 로드되고
전체 기능 표면이 정상 동작함을 확인했습니다.**

---

## 6. abi3 를 켜서 잃는 것 (반대편 근거의 정리)

정직하게 남는 것들입니다.

### 6a. free-threaded 빌드에서 못 쓴다

`abi3t` 는 PEP 803 으로 **3.15 부터** Final 이고, **3.14 free-threaded 에는 Limited API 자체가
없습니다.** [출처] `PythonMultiplatform/ROADMAP.md:1235`

즉 abi3 를 켜면 그동안은 **free-threaded 인터프리터에 `torch._C` 를 실을 수 없습니다.**

다만 같은 ROADMAP 이 free-threading 의 실제 가용성을 조사해 놨습니다.
[출처] `PythonMultiplatform/ROADMAP.md:1224-1232`

| 타깃 | free-threaded 프리빌트 |
|---|---|
| 데스크톱 | 있음 |
| **Android** | **없음** — python.org 아카이브에 `libpython3.14.so` 만 |
| **iOS** | **없음** — BeeWare · python.org XCframework 어느 쪽도 `Py_GIL_DISABLED` 를 정의하지 않음 |

> "free-threading is a desktop-only capability for as long as that holds, and the flag should stay
> off by default."

**torchnative 는 기기 추론이 목표이므로 이 손실은 현재 목표와 겹치지 않습니다.**
데스크톱에서 free-threaded 추론을 하고 싶어지면 그때 재검토 항목입니다.

### 6b. 벌크 접근의 지름길이 없다

§4b 에서 다뤘습니다. `PySequence_Fast_ITEMS` 급의 매크로가 없으므로 파이썬 컨테이너를 통째로
꺼내는 지원되는 방법이 없습니다. **DLPack 캡슐과 버퍼 프로토콜로 우회 가능하고, 둘 다
Limited API 안입니다.**

### 6c. `__torch_function__` 조회가 느려질 수 있다

§2c. 대체(`PyObject_GetOptionalAttr`)는 있으나 비용은 **[미확인]**.

### 6d. 규율이 강제되지 않는다

abi3 를 켜도 CPython 은 검사하지 않습니다(§1a). PyO3 가 `pyo3-ffi` 수준에서 `cfg(Py_LIMITED_API)`
로 막아주지만, `pyo3::ffi` 를 직접 부르거나 C 를 섞으면 뚫립니다.

**대응이 있습니다.** §1c 에서 한 감사를 CI 게이트로 만드십시오 — 명령 두 개입니다.

```bash
nm -u <artifact> | sed 's/^_//' | grep -E '^(Py|_Py)' | sort > /tmp/used.txt
# Misc/stable_abi.toml 의 [function.*] / [data.*] / [const.*] 항목과 대조, 차집합이 비어야 한다
```

세 타깃 산출물 모두 지금 **차집합이 비어 있습니다.** 그 상태를 회귀 테스트로 고정하면
"의도"였던 규율이 "불변식"이 됩니다.

---

## 7. 권고

**`rust/torch_c/Cargo.toml` 을 `features = ["extension-module", "abi3-py313"]` 로 바꾼다.**

근거를 무게순으로:

1. **되돌리는 방향이 비대칭이다.** abi3 → 버전 고정은 싸고, 그 반대는 비싸다(§8).
   지금 정보가 불완전하므로 **되돌리기 싼 쪽**에서 시작해야 한다.
2. **버전 고정의 전제가 이미 깨져 있다.** 플랫폼은 3.14.7 이고 3.15 를 시험 중이다(§5a).
3. **버전 불일치의 실패가 조용하다.** `_C.so` 는 태그가 없어 잘못된 인터프리터에도
   로드되고 그대로 돈다(§5d). abi3 는 이 실패 종류를 없앤다.
4. **잃는 기능이 확인되지 않았다.** torch 모양의 `#[pyclass]` 가 요구하는 것 — 상속 · `dict` ·
   `weakref` · 버퍼 · 캡슐 · 예외 — 이 전부 `abi3-py313` 에서 동작한다(§1c). 세 타깃 산출물의
   CPython 심볼 89 개가 전부 stable ABI 안이다.
5. **비용이 작다.** 경계당 ~1 ns, 원소당 ~2.25 ns. 추론 1 회에 마이크로초 단위(§4).
6. **크로스 빌드는 어느 쪽이든 같다.** 이 항목은 판단에서 빼야 한다(§3).
7. **상류가 Limited API 밖에서 쓰는 54 개 중 30 개가 우리가 구현하지 않을 기능의 것이다**
   — dynamo · 프로파일러 · 프레임 조사(§2a). IMPORT_WALLS 5 차가 그 배제를 뒷받침한다.

**floor 는 정확히 3.13.** 낮추면 `PyObject_GetOptionalAttr` 등 3.13 stable ABI 대체를 잃고(§2c),
올리면 `Py_TYPE` 이 함수 호출이 된다(§4d).

### 함께 해야 할 것

| # | 할 일 | 이유 |
|---|---|---|
| 1 | ~~3.14.7 인터프리터에서 abi3 모듈 로드를 확인~~ **완료** | §5d 에서 [실측, 해소] — 3.14.7 · 3.15.0rc1 양쪽에서 §1c 표면 전체 exit=0 |
| 2 | 산출물 이름을 `_C.abi3.so` 로 | 태그가 진실을 말하게. 지금 `_C.so` 는 아무 인터프리터에나 붙는다 |
| 3 | `nm -u` × `stable_abi.toml` 감사를 CI 게이트로 | §6d. 규율을 불변식으로 |
| 4 | `_C` 안에서 파이썬 컨테이너 원소 순회 금지 규칙 | §4b. abi3 비용이 몰린 유일한 곳 |
| 5 | `.cargo/config.toml` 에서 `PYO3_CROSS_PYTHON_VERSION` 제거 | abi3 floor 가 대신한다. 두 곳에 버전을 두지 않기 |

### 재검토 방아쇠

아래가 발생하면 이 결정을 다시 연다.

- **free-threaded 기기 파이썬이 나온다** — Android 나 iOS 에 `Py_GIL_DISABLED` 프리빌트가 생기고,
  거기서 추론을 돌리고 싶어질 때(§6a)
- **`__torch_function__` 조회가 프로파일에서 보인다**(§2c)
- **§5 의 A/B 가 B(selective libtorch)로 정해진다** — 그러면 `torch._C` 를 우리가 쓰지 않으므로
  이 결정 자체가 사라진다. [출처] `docs/models/IMPORT_WALLS.md:233-241`

---

## 8. 되돌리는 비용

### abi3 → 버전 고정 (싸다)

1. `Cargo.toml` 에서 `abi3-py313` 제거
2. `PYO3_CROSS_PYTHON_VERSION` 복원
3. 세 타깃 재빌드 + 실기기 재검증
4. 산출물 이름 되돌리기

**소스는 한 줄도 고칠 필요가 없습니다.** Limited API 는 전체 API 의 부분집합이므로 abi3 로 쓴
코드는 버전 고정 빌드에서 그대로 컴파일되고, 오히려 **더 빨라집니다**(§4 의 인라인이 돌아옴).
잃는 것은 다중 버전 호환뿐입니다.

**추정: 반나절. 위험 낮음.**

### 버전 고정 → abi3 (비싸다)

1. `abi3-py313` 추가 후 컴파일 — `cfg(Py_LIMITED_API)` 로 사라진 항목마다 에러
2. **각 에러를 대체 API 로 재작성.** 대체가 없는 것은 설계를 바꿔야 함
   (예: `__torch_function__` 조회를 `_PyType_Lookup` 위에 지었다면 §2c 로 다시 감)
3. 구조체 필드를 직접 읽는 코드는 전부 접근자 호출로
4. 성능 회귀 재측정 — 인라인이 사라지므로 벌크 경로가 §4a 의 `sum_tuple` 처럼 될 수 있음
5. `nm -u` 감사로 잔여 위반 확인
6. 세 타깃 재빌드 + 실기기 재검증

**추정: 코드 규모에 비례. 스파이크 크기면 하루, 실물 `_C` 면 미지수.**

**그리고 시간이 갈수록 비싸집니다** — 사설 API 는 쓰기 편하고 빠르므로, 막아두지 않으면
자연스럽게 늘어납니다. 그래서 §7 의 3 번(CI 게이트)이 결정 자체만큼 중요합니다.

### 두 방향의 비대칭이 이 결정의 핵심

정보가 불완전한 상태에서는 **틀렸을 때 싸게 되돌아올 수 있는 쪽**을 골라야 합니다.
abi3 가 그쪽입니다.

---

## 9. 확인하지 못한 것 (추측으로 채우지 않음)

| # | 항목 | 왜 미확인 |
|---|---|---|
| 1 | ~~abi3 모듈이 3.14.7 / 3.15 에서 로드되는가~~ **해소 (§5d)** | `uv python install` 로 3.14.7 · 3.15.0rc1 확보 후 §1c 표면 전체를 재확인, 둘 다 exit=0. darwin 호스트만 — 기기 로드는 항목 2 로 남음 |
| 2 | **실기기 로드** | iOS · Android 는 **링크만** 확인. 기기에서 `import _C` 는 하지 않음 |
| 3 | `aarch64-apple-ios-sim` abi3 링크 | 조사 중 링크 배선이 바뀌어(§0) 미성사. abi3 고유 문제 아님 |
| 4 | **`__torch_function__` 대체 경로의 비용** | 우리 `_C` 의 구현 구조가 미정 |
| 5 | **candle 쪽 C API 요구** | §5 의 A/B 미결 |
| 6 | 54 개 심볼의 기능별 귀속 | 이름으로 분류한 **[추론]**. 실제 호출 지점을 추적하지 않음 (휠에 C++ 소스 없음) |
| 7 | 양자화(`torch.ao`) · `generate` 이후 경로 | IMPORT_WALLS 5 차의 한계 그대로 |
| 8 | 마이크로벤치의 절대값 | 부하 `load avg 12.5` 에서 측정(§4c). 부호와 크기만 신뢰 |
| 9 | Windows · Linux 데스크톱 타깃 | 이 조사는 darwin · ios · android 만 |

---

## 10. 재현

`/Volumes/macMini/caches/abi3-probe` 에 실험 트리가 남아 있습니다 (594 MB, 대부분 cargo 산출물).

```
torch_c/          rust/torch_c 의 복사본. src/lib.rs 를 torch 모양 pyclass 로 교체
t-abi3/  t-ver/   §4a 마이크로벤치용 darwin 빌드 (abi3-py313 / non-abi3)
t-abi311/         §5d 다중 버전 로드 확인용
t-feat/ t-feat3/  §1c 기능 확인용 darwin / ios · android
m-*/              `_C.so` 로 이름 바꾼 로드용 디렉터리
bench.py          §4a 벤치 스크립트
```

핵심 명령:

```bash
# 산출물 감사 (§1c, §6d)
nm -u <artifact> | sed 's/^_//' | grep -E '^(Py|_Py)' | sort
curl -sL https://raw.githubusercontent.com/python/cpython/3.13/Misc/stable_abi.toml

# 크로스 빌드 대조 (§3)
PYO3_CROSS=1 cargo build --release --target aarch64-apple-ios                              # abi3
PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 cargo build --release --target aarch64-apple-ios  # 대조
```
