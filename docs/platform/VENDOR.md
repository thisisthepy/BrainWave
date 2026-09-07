# 상류 torch 파이썬 트리 벤더링 — §2 의 베팅을 처음 시험한 기록

DESIGN.md §2 는 이 프로젝트의 핵심 베팅을 한 문장으로 적어 두었습니다.

> "PyTorch 를 다시 만든다" 가 아니라 **"파이썬 계층은 벤더링하고 `_C` 만 교체한다"** 입니다.

**지금까지 이 문장은 한 번도 시험된 적이 없습니다.** IMPORT_WALLS 의 1~5 차는 전부 (a) 가짜 torch
스텁이거나 (b) **진짜 torch 2.13.0** 을 깔아 놓고 잰 것이었습니다. 상류 파이썬 트리를 그대로 두고
**`_C` 자리에만 우리 것을 넣은** 상태는 아무도 만들어 본 적이 없습니다.

이 문서는 그 상태를 실제로 만들고, 어디서 깨지는지를 순서대로 기록한 것입니다.

> **Correction (문서 감사, 2026-09):** 이 문서는 genesis 단계의 벽 목록입니다 (`_C` 표면
> 17 개 vs 상류 989 개, §0). 이 감사가 이미 확인한 나중 라운드들(`docs/bindings/OVERLOAD.md`,
> `docs/bindings/TENSORBASE.md`, `docs/distributed/DISTRIBUTED.md`, `docs/models/COMPAT.md`, `docs/design/DESIGN.md` §11.1 감사
> 등, round 1·2)이 이 문서가 이름 붙인 벽 다수를 이미 닫았습니다 — `AutoModelForCausalLM.
> from_config`/`from_pretrained` 둘 다 오늘 성공하고(round 1 DESIGN.md 감사, round 2
> DISTRIBUTED.md/CKPT2.md/GENERATE.md 감사), `import transformers` 도 통과합니다. 이 문서
> 자체의 개별 항목(§1-§8)을 하나씩 다시 검증하지는 않았습니다 — "목표는 되게 하는 것이
> 아니라 어디서 깨지는지 아는 것" 이라는 이 문서의 진단은 역사적 기록으로 여전히 유효하고,
> 그 진단이 가리킨 벽들이 이후 라운드에서 닫혔다는 사실이 그 진단을 무효화하지 않습니다.

**목표는 "되게 하는 것" 이 아니라 "어디서 깨지는지 아는 것" 입니다.** 그래서 아래의 어떤 항목도
"해결했다" 고 적지 않았습니다.

---

## 0. 한눈에

| | |
|---|---|
| abi3 | **켬.** `abi3-py313`, 세 타깃 전부 종료 코드 0, 호스트 스모크 13/13 |
| 산출물 이름 | `_C.so` → **`_C.abi3.so`** (ABI3.md §7 항목 2) |
| 벤더링한 것 | torch **2.13.0** 파이썬 트리, `.py` **2285 개**, 53 MB, 네이티브 산출물 **0 개** |
| 저장소 배치 | `vendor/` 안, **`.gitignore` 로 제외**. 재현은 `vendor/vendor_torch.sh` |
| **엄격 모드 `import torch`** | **`torch/__init__.py:1050` 에서 정지.** `_initExtension` 없음 |
| 기록 모드 `import torch` | `torch/__init__.py:2885` (전체 3087 행의 **93%**) 까지. `torch._decomp` 안에서 정지 |
| `import transformers` | **성공(종료 코드 0), 그리고 `is_torch_available() == True`** — 이것이 함정입니다 (§5) |
| `AutoModelForCausalLM.from_config` | **미도달.** `import torch` 를 넘지 못함 |
| 우리 `_C` 의 표면 | **17 개 이름**. 상류는 **989 개** |

---

## 1. abi3 를 켰다

ABI3.md §7 의 권고를 그대로 적용했습니다.

```toml
pyo3 = { version = "0.29.2", features = ["extension-module", "abi3-py313"] }
```

### 세 타깃 결과 — 판정은 전부 종료 코드

| 타깃 | 종료 코드 | 산출물 크기 | 비고 |
|---|---|---|---|
| `aarch64-apple-darwin` | **0** | 1,406,592 B | `./pytests/run.sh` **13/13 통과** |
| `aarch64-linux-android` | **0** | 2,274,800 B | |
| `aarch64-apple-ios` | **0** | 1,496,544 B | |

abi3 이전(TORCH_C.md §4)과 비교하면 호스트 −19,360 B, Android −6,168 B, iOS **+1,480 B** 입니다.
**세 타깃이 서로 다른 방향으로 움직였으므로 abi3 의 크기 효과라고 말할 수 없습니다.** 인라인이
사라진 만큼과 심볼 테이블이 줄어든 만큼이 상쇄되는 정도로만 읽어야 합니다.

### 함께 확인된 것

- **`PYO3_CROSS_PYTHON_VERSION` 이 더는 필요 없습니다** (ABI3.md §7 항목 5). Android 를
  이 변수 **없이** 빌드해 종료 코드 0 을 받았습니다. abi3 floor 가 버전을 대신합니다.
- **미정의 CPython 심볼은 세 타깃 모두 94 개로 같습니다.** 그중 사설처럼 보이는 것은
  `_Py_IncRef` · `_Py_DecRef` 둘뿐이고, 이 둘은 Limited API 에서 `Py_INCREF` 가 함수 호출로
  바뀐 결과이므로 stable ABI 안입니다.
- **iOS 는 `PYO3_CONFIG_FILE` 이 여전히 필요합니다.** 없이 빌드하면 PyO3 가 `-lpython3.13` 을
  계속 방출해 `ld: library 'python3.13' not found` 로 실패합니다(종료 코드 101 로 확인).
  abi3 를 켜도 이 항목은 달라지지 않습니다 — ABI3.md §3 의 "크로스 빌드 배선에는 영향이 없다" 와
  일치합니다.

### 하지 않은 것

ABI3.md §7 의 항목 3(`nm -u` × `stable_abi.toml` 감사를 CI 게이트로)과 항목 4(`_C` 안에서 파이썬
컨테이너 원소 순회 금지 규칙)는 **이번에 하지 않았습니다.** 감사 자체는 위 수치로 한 번 돌렸지만
불변식으로 박아 두지는 않았습니다.

---

## 2. 벤더링을 어떻게 했는가

### 저장소 안, 그러나 커밋하지 않음

`vendor/` 를 저장소 안에 두되 트리는 `.gitignore` 로 제외하고, **재현 스크립트만 커밋 대상**으로
남겼습니다. 근거는 두 가지입니다.

- 파이썬 트리만 53 MB · 2285 파일입니다. 저장소에 넣으면 이후 모든 clone·diff·리뷰가 그 무게를
  집니다.
- 반대로 저장소 **밖**에 두면 경로가 기계마다 달라집니다. TORCH_C.md §3 이 걷어낸
  하드코딩된 iOS `-F` 경로와 같은 함정을 다른 자리에 다시 파는 셈입니다.

```
vendor/
├─ vendor_torch.sh     상류 트리를 가져온다        (커밋됨)
├─ install_shim.sh     우리 `_C` 를 구멍에 넣는다  (커밋됨)
├─ probe.py            벽을 찾는 계측기            (커밋됨)
├─ torch/              벤더링된 트리               (.gitignore)
└─ torch-2.13.0.dist-info/                        (.gitignore)
```

### 무엇을 빼는가가 정의다

`vendor_torch.sh` 는 **버리는 것**으로 정의됩니다.

```
torch/lib/       353 MB   libtorch·libc10 등 네이티브 런타임
torch/include/    61 MB   C++ 헤더
torch/bin/         7 MB   호스트 도구
torch/test/                상류 자체 테스트
*.so *.dylib *.a           트리 안 모든 컴파일 산출물
```

**마지막 줄이 파이썬 트리에서 실제로 지우는 파일은 정확히 하나입니다 — `_C.cpython-313-darwin.so`.**
이것은 가정이 아니라 측정입니다. `torch/` 아래에서 `lib/`·`bin/`·`include/`·`test/` 를 뺀 나머지에
네이티브 파일은 그 하나뿐이고, 벤더링 후 `.stamp` 의 `native_left=0` 이 그것을 확인합니다.

즉 **§2 의 "네이티브인 것은 `torch._C` 하나뿐" 은 파일 수준에서는 참입니다.** 참이 아닌 것은
그 하나가 얼마나 큰 인터페이스인가입니다 (§4).

### 재현

```bash
# 1. 벤더링 (기본 소스는 spike-venv 의 torch 2.13.0)
./vendor/vendor_torch.sh                       # TORCHNATIVE_TORCH_SRC 로 소스 변경 가능

# 2. 우리 _C 를 넣기
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target
./vendor/install_shim.sh

# 3. 상류 _C 의 이름 표면을 뜬다 (계측기 입력. 진짜 torch 가 있는 인터프리터에서)
/Volumes/macMini/caches/spike-venv/bin/python vendor/probe.py \
    --dump-surface /tmp/bw_surface.json

# 4. 시험
PY=/Volumes/macMini/caches/spike-venv/bin/python
PYTHONDONTWRITEBYTECODE=1 TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/vendor \
  $PY vendor/probe.py --mode strict --target torch; echo "EXIT=$?"
PYTHONDONTWRITEBYTECODE=1 TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/vendor \
  $PY vendor/probe.py --mode record --surface /tmp/bw_surface.json \
     --target torch --report /tmp/rec.json; echo "EXIT=$?"
```

인터프리터는 **spike-venv 의 3.13.0** 을 씁니다. torch 의 서드파티 의존(sympy·networkx·filelock·
fsspec·jinja2·typing_extensions)이 이미 거기 있고, `PYTHONPATH` 가 `site-packages` 보다 앞서므로
**벤더링 트리가 설치된 torch 를 가립니다.** `torch.__file__` 이 `vendor/` 아래를 가리키는 것으로
확인했습니다. 이렇게 하지 않으면 나오는 벽이 "우리 `_C` 의 부족" 이 아니라 "의존 패키지 없음" 이
되어 측정이 무의미해집니다.

### 판정은 종료 코드로

`vendor/probe.py` 는 성공/실패를 **종료 코드로만** 냅니다. IMPORT_WALLS 2 차가 `grep -q MODEL_OK`
로 판정하다 트레이스백이 소스 줄을 그대로 출력하는 바람에 한 회차를 통째로 버린 전례가 있어,
이 계측기에는 성공 마커 문자열 자체를 두지 않았습니다.

### 계측기의 두 모드 — 이 구분이 이 문서의 전부다

| 모드 | 하는 일 | 무엇을 말해 주는가 |
|---|---|---|
| **strict** | 우리 `_C` 를 `torch._C` 로 올리고 그대로 실패시킨다 | **얼마나 갔는가에 대한 정직한 답.** 아무것도 가짜가 아니므로 과장할 수 없다 |
| **record** | 없는 이름마다 기록하고 허수아비를 돌려준다 | **얼마나 많은 표면이 요구되는가.** strict 는 한 번에 이름 하나만 알려준다 |

**record 모드가 더 멀리 간 것은 진전이 아닙니다.** 허수아비 `torch.float32` 는 dtype 이 아닙니다.
record 가 사는 것은 오직 하나 — **"이름이 없어서 막힌 벽(폭)" 과 "이름은 있는데 동작이 필요해서
막힌 벽(깊이)" 을 구분하는 능력**이고, §2 의 베팅이 걸린 곳이 정확히 그 구분입니다.

record 모드는 상류 `_C` 의 **이름과 종류만** 읽습니다(`--dump-surface`). 값도 코드도 가져오지
않습니다. 구멍의 크기를 재는 것이지 상류에서 빌려오는 것이 아닙니다.

---

## 3. 부딪힌 벽 — 순서대로

번호는 만난 순서입니다. `[S]` 는 엄격 모드에서도 나오는 벽, `[R]` 은 계측기가 그 앞의 벽을 지나야
비로소 보이는 벽입니다.

### 1 `[S]` `torch/__init__.py:444` — 파이썬 트리가 네이티브 라이브러리를 먼저 찾는다

```
OSError: dlopen(torchnative/src/main/torch/lib/libtorch_global_deps.dylib): no such file
```

`_load_global_deps()` 가 `ctypes.CDLL(torch/lib/libtorch_global_deps.dylib, RTLD_GLOBAL)` 을
호출합니다. **`_C` 가 아닙니다** — 파이썬 트리 자체가 별도의 네이티브 산출물을 요구합니다.

**우회는 상류가 제공하는 것으로 했습니다.** `torch/__init__.py:406` 의 분기가
`os.getenv("TORCH_USE_RTLD_GLOBAL")` 을 보고, 참이면 `_load_global_deps()` 를 건너뛰고 곧장
`from torch._C import *` 로 갑니다. 벤더링 트리를 **한 줄도 고치지 않았습니다.**

> 이 벽은 사라진 것이 아니라 옆으로 비켜선 것입니다. 상류가 이 우회로를 제공하는 이유는
> "libtorch_global_deps 가 없는 빌드 환경(fbcode)" 을 위해서이고, 우리가 정확히 그 환경입니다.
> 다만 `TORCH_USE_RTLD_GLOBAL` 분기는 `sys.setdlopenflags(RTLD_GLOBAL|RTLD_LAZY)` 도 함께
> 켭니다. **기기에서 그 플래그가 어떤 영향을 주는지는 미확인입니다.**

### 2 `[S]` `torch/__init__.py:1050` — **엄격 모드는 여기서 끝난다**

```
ImportError: cannot import name '_initExtension' from 'torch._C'
    (torchnative/src/main/torch/_C.abi3.so)
```

**이것이 정직한 답입니다.** 우리 `_C` 로 `import torch` 는 `torch/__init__.py` 3087 행 중
1050 행까지 갑니다. 그 앞의 `from torch._C import *` (445 행) 는 통과합니다 — 별표 임포트는
없는 이름에 대해 불평하지 않기 때문이고, 통과했다는 사실 자체는 아무 의미가 없습니다.

우리 `_C` 가 내보내는 이름은 **17 개**이고, 그중 상류 `torch._C` 에도 있는 이름은
**`dtype` · `device` · `TensorBase` 셋뿐**입니다. 나머지 14 개(dtype 인스턴스 11 개와
`_aten_dispatch` 등 4 개)는 상류 `_C` 에 없는 이름입니다. 상류 `dir(torch._C)` 는 **989 개**입니다.

### 3 `[R]` `torch/_tensor.py:799` — `TensorBase` 는 이름이 아니라 표면이다

```
AttributeError: type object 'torch._C.TensorBase' has no attribute 'detach'
```

`class Tensor(torch._C.TensorBase)` 의 **클래스 본문**이 `_C._add_docstr(_C.TensorBase.<이름>, ...)`
를 수백 번 합니다. 클래스 정의가 실행되는 시점에 전부 해소되므로, **`import torch` 가 끝나기 전에
그 집합 전체가 요구됩니다.**

- 상류 `dir(TensorBase)`: **694 개**
- 이번 실행에서 실제로 요구된 것: **543 개** (그중 던더 2 개)
- 우리 `TensorBase` 가 가진 것: `shape` · `dtype` · `device` 수준

TORCH_C.md 가 `TensorBase` 라는 **이름**을 미리 맞춰 둔 판단은 옳았습니다. 다만 맞춰야 할 것이
이름만이 아니라는 것이 이번에 드러났습니다.

### 4 `[R]` `torch/__init__.py:2179` — `torch/bin/torch_shm_manager` 가 **존재**해야 한다

```
RuntimeError: Unable to find torch_shm_manager at torchnative/src/main/torch/bin/torch_shm_manager
```

`_manager_path()` 가 Windows 가 아닌 모든 플랫폼에서 **무조건** 이 파일의 존재를 확인하고, 없으면
임포트를 거부합니다. 프로세스 간 텐서 스토리지 공유용 헬퍼 실행 파일입니다 — 휴대폰에는 공유할
상대가 없고, 우리는 이것을 절대 싣지 않을 것입니다.

확인하는 것은 **존재뿐**이고 경로는 `_initExtension` 에 넘겨질 뿐 `torch.multiprocessing` 을 쓰기
전에는 실행되지 않으므로, `install_shim.sh` 가 **0 바이트 표식**을 놓습니다. 조용히 고치지 않고
표식으로 남긴 이유는 요구 사항이 계속 보이게 하기 위해서입니다.

### 5 `[R]` `torch/__init__.py:2212` — `torch.<op>` 네임스페이스 전체가 `_C` 객체 하나에서 수확된다

```python
for __name in dir(_C._VariableFunctions):
    __obj = getattr(_C._VariableFunctions, __name)
    __obj.__module__ = __name__          # "torch"
    globals()[__name] = __obj
```

**`torch.add` · `torch.mm` · `torch.full` 은 파이썬 트리 어디에도 쓰여 있지 않습니다.** 임포트
시점에 `_C._VariableFunctions` 에서 긁어옵니다.

- 상류 `dir(_C._VariableFunctions)`: **985 개**, 그중 공개 **625 개**
- 각 항목은 `dir()` 로 열거 가능해야 하고, **`__module__` 대입이 가능해야** 합니다 —
  즉 바인딩되는 메서드여서는 안 됩니다. 상류는 `builtin_function_or_method` 라 바인딩되지
  않습니다. 우리가 Rust 로 만들 것도 같은 성질을 가져야 합니다.

**이것은 `_aten_dispatch` 단일 관문과 충돌하지 않습니다.** 625 개를 전부 `_aten_dispatch` 로
내려보내는 얇은 래퍼로 만들면 문이 하나로 유지됩니다. 다만 **625 개의 이름이 열거 가능해야
한다**는 요구는 그대로 남습니다.

### 6 `[R]` PyO3 가 `__all__` 을 만든다

`#[pymodule]` 로 만든 모듈에는 PyO3 0.29 가 `__all__` 을 붙입니다. 그러므로
`from torch._C import *` 는 **크레이트가 등록한 이름만** 복사합니다. 파이썬 쪽에서 속성을 얹어도
별표 임포트에는 보이지 않습니다.

`torch` 네임스페이스의 상당 부분이 그 별표 임포트로 만들어지므로, **재export 하고 싶은 것은
"닿을 수 있는" 것으로 부족하고 "등록된" 것이어야 합니다.**

### 7 `[R]` `_initExtension` 이 **`torch` 모듈에** 이름을 써 넣는다

`torch.float32` · `torch.strided` · `torch.contiguous_format` 은
**`torch._C` 의 속성이 아닙니다.** 상류의 `initializeDtypes()` 가 C 에서 `torch` 모듈을 임포트해
`PyModule_AddObject` 로 직접 써 넣습니다.

| | |
|---|---|
| `torch` 에 주입되는 enum 인스턴스 | **74 개** (`dtype` 56 · `layout` 8 · `memory_format` 5 · `qscheme` 5) |
| 그중 `dir(torch._C)` 에도 있는 것 | **0 개** |

**벤더링한 소스를 아무리 읽어도 이 74 개가 존재해야 한다는 사실이 나오지 않습니다.** grep 으로
찾을 수 없고, 트리를 실제로 돌려야만 드러납니다.

우리 `_C` 는 dtype 11 개를 `__all__` 로 내보내 별표 임포트를 태우고 있어서 결과적으로
`torch.float32` 는 생깁니다 — **자리는 다르지만 결과는 같은 경우**입니다. 나머지 63 개는 없습니다.

### 8 `[R]` `torch/_sources.py:9` — `_C` 는 패키지여야 한다

```
ModuleNotFoundError: No module named 'torch._C._jit_tree_views'; 'torch._C' is not a package
```

`from torch._C._jit_tree_views import SourceRangeFactory` 는 **임포트 문**이므로 속성으로는
만족되지 않습니다. 상류 `_C` 는 C 에서 서브모듈 **32 개**를 `sys.modules` 에 등록합니다.
그리고 그 서브모듈들은 **자기도 패키지**입니다 — `torch/utils/_python_dispatch.py:22` 가
`torch._C._dynamo.guards` 를 임포트합니다.

이번 실행에서 서브모듈 멤버로 요구된 것이 **270 개**이고, 분포는:

```
_special 56   _jit_tree_views 46   _linalg 42   _functorch 39
_autograd 26  _fft 22              _profiler 16 _nn 12
```

### 9 `[R]` `torch/nn/parameter.py:26` — `type(TensorBase) is _C._TensorMeta` 여야 한다

`class _ParameterMeta(torch._C._TensorMeta)` 다음에
`class Parameter(torch.Tensor, metaclass=_ParameterMeta)` 가 옵니다. `_TensorMeta` 가 실제로
`TensorBase` 의 메타클래스가 아니면 `metaclass conflict` 입니다.

### 10 `[R]` `torch/_prims_common/__init__.py:90` — getset 디스크립터

`torch.Tensor.is_sparse.__get__` 를 꺼내 보관합니다. `TensorBase` 표면의 상당 부분이 메서드가
아니라 getset 디스크립터(`is_sparse` · `grad` · `shape` · `data`)이고, 파이썬 트리는 그 **디스크립터
객체 자체**를 꺼내 씁니다. 호출 가능한 것으로는 부족합니다.

### 11 `[R]` `torch/distributed/__init__.py:28` — **없는 것이 끄는 방법이다**

```python
def is_available() -> bool:
    return hasattr(torch._C, "_c10d_init")
```

파이썬 트리는 `_C` 에 이름이 **있는지**를 물어 서브시스템을 켜고 끕니다. 우리 `_C` 가
`_c10d_init` 을 내보내지 않으면 `torch.distributed` 는 조용히 비활성이 됩니다.

**이것은 IMPORT_WALLS 4 차에 대한 부분적인 정정입니다.** 4 차는 "349 개 모듈을 끊을 수 없다" 고
결론지었는데, 그 실험은 **상류 `_C` 를 그대로 둔 채 모듈 단위로** 끊으려 했습니다. `_C` 를 우리가
소유하면 **이름을 내보내지 않는 것**이라는 상류가 지원하는 스위치가 생깁니다. 모듈은 여전히
임포트되지만 내용이 죽습니다.

다만 **끄는 것으로 끝나지 않습니다.** 같은 파일 41~45 행이 `is_available()` 과 무관하게
`torch._C._DistError` · `_DistBackendError` · `_DistNetworkError` · `_DistStoreError` 를 요구합니다.

### 12 `[R]` 메타타입은 타입마다 다르고, 양쪽으로 걸린다

- `torch/_awaits/__init__.py:12` — `class _PyAwaitMeta(type(torch._C._Await), type(Generic))` 은
  `type(_Await)` 이 `type` 이면 **`duplicate base class`** 로 실패합니다.
- `torch/autograd/variable.py:14` — `class Variable(_C._LegacyVariableBase, metaclass=VariableMeta)`
  는 `type(_LegacyVariableBase)` 가 `type` 이 **아니면** `metaclass conflict` 로 실패합니다.

상류 189 개 `_C` 타입의 메타타입 분포:

| 메타타입 | 개수 |
|---|---|
| `pybind11_type` | 135 |
| `type` | 51 |
| `_TensorMeta` | 2 |
| `OpaqueBaseMeta` | 1 |

**타입마다 맞춰야 하고, 일괄 규칙으로는 둘 중 하나가 반드시 깨집니다.**

### 13 `[R]` `torch/nn/functional.py:4808` — TorchScript 소스 파서가 임포트 시점에 돈다

`@_overload` 데코레이터가 `_check_overload_body(func)` → `parse_def(func)` →
`SourceContext(SourceRangeFactory)` 로 내려갑니다. `torch/_sources.py:87` 의
`class SourceContext(SourceRangeFactory)` 가 `super().__init__(source, filename, file_lineno,
leading_whitespace_len)` 을 호출하므로, `_C._jit_tree_views.SourceRangeFactory` 는
**존재하는 것으로 부족하고 4 인자로 인스턴스화 가능해야** 합니다.
`_check_overload_body` 는 `OSError` 만 잡습니다.

**이것은 IMPORT_WALLS 2 차의 `@auto_docstring` 과 같은 종류입니다** — introspection 이 진짜 객체를
요구하는 벽. 그리고 그 벽이 **`torch.nn.functional` 에 있습니다.** IMPORT_WALLS 5 차가 추론 중
실제로 실행되는 14 개 모듈 중 하나로 지목한 바로 그 모듈입니다.

### 14 `[R]` `torch/_ops.py:139` — **4 차와 같은 벽에, 반대 방향에서 도달했다**

```
AssertionError: expected DispatchKey, got <class 'Placeholder'>
```

IMPORT_WALLS 4 차가 **끝난 지점과 같은 행**입니다. 4 차는 상류 `_C` 를 두고 모듈을 끊다가
`torch/_higher_order_ops/auto_functionalize.py` 경로로 여기 도달했습니다. 이번에는 모듈을 전부
두고 `_C` 를 바꾼 채 `torch/_functorch/autograd_function.py:106` 경로로 도달했습니다.

**양쪽에서 같은 벽이 나온다는 것은 이것이 우회의 문제가 아니라 구조라는 뜻입니다.**
임포트 시점 연산자 등록이 `DispatchKey` · `TransformType` 이라는 C 타입으로 타입 검사됩니다.

### 15 `[R]` `torch/autograd/__init__.py:653` — autograd 는 선택이 아니다

```python
if not torch._C._autograd_init():
    raise RuntimeError("autograd initialization failed")
```

`hasattr` 게이트가 없습니다. 거짓을 돌려주면 즉시 실패합니다.

**이것은 이 프로젝트의 살아 있는 가정 하나에 걸립니다.** DESIGN.md §3 의 단계 0 은 "기기에서
backward 없음" 을 전제하는데, **그 전제는 "autograd 를 임포트하지 않아도 된다" 로 확장되지
않습니다.**

### 16 `[R]` `torch/jit/__init__.py:315` — jit 도 마찬가지

`_jit_init()` 도 같은 모양이고, `torch/__init__.py:2298` 이 임포트하는 `torch.distributions` 가
`torch.jit` 을 끌어옵니다.

### 17 `[R]` `torch/multiprocessing/__init__.py:37` — C 가 파이썬 **패키지에** 이름을 써 넣는다 (두 번째)

`torch._C._multiprocessing_init()` 이 `_prctl_pr_set_pdeathsig` 를 `torch.multiprocessing`
패키지에 주입하고, 두 줄 뒤 `spawn.py:14` 가 그것을 임포트합니다. 벤더링 소스 어디에도 그 이름을
정의하는 곳이 없습니다. **§7 과 같은 패턴이고, 이것으로 두 건입니다.**

### 18 `[R]` `torch/fx/node.py:102` — `torch.ops` 의 모양이 확정됐다

TORCH_C.md §5-4 는 이렇게 적었습니다.

> `torch._C._jit_get_operation(name)` 이 호출 가능한 객체를 돌려주는 모양을 기대합니다. …
> **§11 의 1 단계를 실제로 해 보기 전에는 어떤 모양이 요구되는지 확정할 수 없습니다.**

이번이 그 1 단계이고, 모양은 이렇습니다.

| 함수 | 반환 | 출처 |
|---|---|---|
| `_jit_get_operation(qualname)` | **쌍** `(op, overload_names)` | `torch/_ops.py:1415` |
| `_get_operation_overload(qualname, overload)` | **삼중** `(op, op_dk, tags)` | `torch/_ops.py:1238` |
| `_get_schema(qualname, overload)` | 스키마 객체 | `torch/_ops.py:1247` |

`op` 은 호출 가능해야 하고 `__module__` 대입이 가능해야 하며, `_ops.py` 가 그것을
`torch.jit._builtins._register_builtin(op, qualname)` 에 그대로 넘깁니다.

**그리고 이것은 지연 API 가 아닙니다.** `torch/fx/node.py:102` 가 `torch.ops.aten._assert_async.msg`
를 **`import torch` 가 아직 도는 중에** 해소합니다. 이번 실행에서:

| | |
|---|---|
| `import torch` 중 해소된 정규화 op 이름 | **327 개** |
| 그중 `aten::` | **313 개** |
| 해소된 오버로드 | **147 개** |
| 우리 `_C` 가 구현한 aten op | **3 개** |

나머지 네임스페이스는 `quantized` 6 · `profiler` 3 · `inductor` 2 · `prim` · `export` ·
`debug_mode_ops` 각 1 입니다.

> **주의: 327 은 "구현해야 할 op 수" 가 아닙니다.** 임포트 시점에는 이름과 스키마를 조회할 뿐
> 커널을 부르지 않습니다. 그러나 **조회가 실패하면 임포트가 실패합니다.** 즉 이 숫자는
> 커널 작업량이 아니라 **op 레지스트리 작업량**입니다. IMPORT_WALLS 3 차가 "모듈 벤더링 문제와
> op 커버리지 문제는 분리되어 있다" 고 적었는데, 그 사이에 **세 번째 항목**이 있습니다 —
> 실행되지 않지만 등록은 되어 있어야 하는 op 스키마.

### 19 `[R]` `torch/_prims/rng_prims.py:419` — **`_C` 가 파이썬 트리의 메타클래스를 써야 한다**

```
TypeError: Opaque type <class 'torch._C.Generator'> must subclass
    torch._opaque_base.OpaqueBase or 'metaclass=torch._opaque_base.OpaqueBaseMeta'
```

확인: 상류에서 `type(torch._C.Generator).__mro__` 는
`(torch._opaque_base.OpaqueBaseMeta, type, object)` 입니다.

**`torch._opaque_base` 는 벤더링한 파이썬 트리의 파일입니다.** 즉 C 확장 모듈이 자기 초기화 중에
파이썬 패키지에서 메타클래스를 가져와 자기 타입에 붙여야 합니다. 지금까지의 방향 — 파이썬이 `_C`
에 의존 — 과 **반대 방향의 결합**이고, 이번 조사에서 처음 나온 종류입니다.

**기록 모드는 여기서 멈춥니다.** 도달 지점은 `torch/__init__.py:2885` — 전체 3087 행의 93% 이고,
그 아래는 `torch._decomp` 입니다. **DESIGN.md §2 가 "Core ATen 밖 롱테일이 자동 분해된다" 며
벤더링 대상으로 명시한 바로 그 디렉터리**입니다.

---

## 4. 숫자로 본 표면

| | 상류 `torch._C` | 우리 `_C` |
|---|---|---|
| `dir()` 이름 | **989** | **17** |
| 타입 | 189 | 3 |
| C 서브모듈 | 32 | 0 |
| 호출 가능 | 738 | 5 |
| 값 | 30 | 11 (dtype 인스턴스) |
| `TensorBase` 멤버 | 694 | ~3 |
| `_VariableFunctions` 멤버 | 985 (공개 625) | 없음 |
| `torch` 네임스페이스 주입 | 74 | 11 (다른 자리에서) |
| 임포트 중 해소되는 op 이름 | 327 | 3 (구현된 aten op) |

기록 모드 1 회 실행에서 **실제로 요구된** 항목 (허수아비가 답한 횟수 ≥ 1):

```
TensorBase 멤버        543
torch.ops 이름 조회    327
torch._C 서브모듈 멤버 270
torch.ops 오버로드     147
torch 네임스페이스 주입  64
모듈 수준 이름          20
타입 멤버                5
```

> **이 표의 한계.** 별표 임포트로 미리 심어 둔 972 개는 "요구됨" 으로 계수되지 않습니다 —
> 존재하므로 기록 훅을 거치지 않기 때문입니다. 따라서 위 숫자는 **하한**이고, 972 개 중 실제로
> 쓰인 것이 몇 개인지는 **미측정**입니다.

---

## 5. `import transformers` 는 통과한다 — 그리고 그것이 함정이다

| 대상 | 모드 | 종료 코드 |
|---|---|---|
| `import transformers` | strict | **0** |
| `import transformers` | record | **0** |
| `AutoModelForCausalLM.from_config` | strict | 1 (`import torch` 에서) |
| `AutoModelForCausalLM.from_config` | record | 1 (§3 의 벽 19) |

엄격 모드에서 `import transformers` 는 **`torch._C` 이름을 하나도 요구하지 않고**(요구 0 건)
종료 코드 0 으로 끝납니다. IMPORT_WALLS 1 차·3 차의 결과가 벤더링 트리에서도 그대로 성립합니다.

**그런데 `is_torch_available()` 이 `True` 를 돌려줍니다.** 벤더링한 `torch-2.13.0.dist-info` 가
`importlib.metadata.version("torch") >= 2.5.0` 을 만족시키기 때문입니다.

**이 조합은 torch 가 아예 없는 것보다 나쁩니다.** torch 가 없으면 transformers 는
`AutoModelForCausalLM requires the PyTorch library but it was not found` 로 깨끗하게 거부합니다.
지금은 관문을 통과한 뒤 **훨씬 깊은 곳에서** 깨집니다. 벤더링 트리를 배포할 때 이 상태를 중간
단계로 두면, 실패 지점이 원인에서 멀어집니다.

`from_config` 는 두 모드 모두 **`import torch` 를 넘지 못해서** 실패합니다. IMPORT_WALLS 가
지목한 `@auto_docstring`(범주 6)에는 **도달조차 못 했습니다.**

---

## 6. §2 의 베팅은 성립하는가 — 판단

**반증되지 않았습니다. 그러나 §2 의 서술은 비용을 크게 과소평가합니다.**

### 성립하는 부분

- **"네이티브인 것은 `torch._C` 하나뿐" 은 파일 수준에서 참입니다.** 파이썬 트리 2285 개 파일
  중 네이티브는 정확히 하나였고, 그것을 우리 것으로 갈아 끼운 트리가 실제로 `torch/__init__.py`
  1050 행까지 갑니다(엄격 모드). 트리를 **한 줄도 수정하지 않았습니다.**
- **`_aten_dispatch` 단일 관문은 무사합니다.** 요구되는 진입로(`_VariableFunctions` 625 개,
  `_jit_get_operation`, `_get_operation_overload`)는 전부 **조회 층**이고, 그 아래를 하나의
  디스패처로 모으는 데 구조적 장애가 없습니다. TORCH_C.md §5-4 가 미룬 판단은 이제 내릴 수
  있습니다 — **얇은 조회 층을 얹는 것으로 충분하고, 문은 하나로 유지됩니다.**
- **`_C` 에서 이름을 빼는 것으로 서브시스템을 끌 수 있습니다** (벽 11). IMPORT_WALLS 4 차가
  "쳐낼 수 없다" 고 한 것은 상류 `_C` 를 유지한 상태의 결론이었고, `_C` 를 소유하면 상류가
  지원하는 스위치가 생깁니다.

### 성립을 어렵게 하는 부분

1. **`_C` 는 "모듈 하나" 가 아니라 989 개 이름 · 32 개 서브모듈(그 자체가 패키지) ·
   694 멤버 타입 · 985 멤버 객체 · 74 개 네임스페이스 주입입니다.** §2 의 그림에서 `torch/_C` 는
   한 줄이지만, 그 한 줄의 인터페이스는 파이썬 트리보다 촘촘합니다.

2. **파이썬 트리는 `_C` 가 자기 네임스페이스에 이름을 써 넣기를 기대합니다** (벽 7 · 17, 두 건).
   이것은 소스를 읽어서 알 수 없고 실행해야만 드러납니다. **정적 분석으로 목록을 만들 수 없는
   요구 사항이 존재한다**는 뜻이고, 이 프로젝트가 §6 에서 채택한 "발견은 shim 이 스스로 한다" 를
   파이썬 계층에도 반드시 적용해야 하는 근거입니다.

3. **결합이 양방향입니다** (벽 19). `_C.Generator` 의 메타클래스가 `torch/_opaque_base.py` 에서
   와야 합니다. "파이썬이 `_C` 위에 얹힌다" 는 그림이 여기서 깨집니다 — C 모듈이 초기화 중에
   벤더링한 파이썬 패키지를 임포트해야 합니다.

4. **임포트 시점 op 레지스트리가 실행 경로와 분리된 세 번째 작업 항목입니다.** `import torch`
   하나가 327 개 정규화 op 이름과 147 개 오버로드를 조회합니다. 커널은 필요 없지만 **스키마는
   있어야** 합니다. IMPORT_WALLS 3 차의 "모듈 문제 ≠ op 커버리지 문제" 이분법에 항목이 하나
   빠져 있었습니다.

5. **끌 수 없는 것이 있습니다.** autograd(벽 15)와 jit(벽 16)는 `hasattr` 게이트가 없어
   무조건 성공해야 합니다. "기기에서는 추론만" 이라는 전제가 **임포트 단계에는 적용되지
   않습니다.**

6. **IMPORT_WALLS 4 차의 벽이 반대 방향에서도 나옵니다** (벽 14). `torch/_ops.py:139` 의
   `DispatchKey` 타입 검사는 우회로 피할 수 있는 것이 아니라 구조입니다.

### 이것이 A/B 판단에 주는 것

IMPORT_WALLS 4 차는 A(candle + shim) 에 "파이썬 트리 prune 이 싸지 않다" 는 비용 항목을
붙였습니다. **이번 조사는 그 항목을 정정하고 하나를 추가합니다.**

- **정정:** prune 은 4 차가 생각한 것보다 **덜** 나쁩니다. `_C` 를 소유하면 이름 생략으로 서브트리를
  죽일 수 있습니다(벽 11). 4 차가 그 수단을 못 쓴 것은 상류 `_C` 를 유지했기 때문입니다.
- **추가:** 대신 **`_C` 표면 자체가 A 의 주 비용**입니다. 989 개 이름 · 694 멤버 `TensorBase` ·
  625 개 `torch.<op>` · 327 개 op 스키마. 그리고 이 중 **압도적 다수는 한 번도 실행되지
  않습니다** — IMPORT_WALLS 5 차가 "추론 중 실행되는 파이썬은 14 개, 실질 10 개" 라고 잰 것의
  `_C` 판 대응물입니다.

> **5 차의 문장을 이번 결과로 다시 쓰면 이렇습니다.**
> A 의 비용은 "도는 것을 구현하는 일" 이 아니라 **"돌지 않는 것을 임포트되게 만드는 일"** 이고,
> 5 차는 그 비용을 파이썬 모듈 1070 개로 셌습니다. 이번 조사는 같은 비용을 `_C` 쪽에서 셌고,
> **그쪽이 더 큽니다** — 파이썬 모듈은 벤더링으로 공짜지만, `_C` 표면은 전부 우리가 씁니다.

### 그래서 다음에 확인해야 할 것

이 조사는 **판단을 내리지 않고 판단 재료를 하나 더 놓습니다.** 다음 회차가 답해야 할 질문은
"989 개를 다 만들어야 하는가" 가 아니라 **"임포트를 통과시키는 데 필요한 최소 표면은 몇 개인가"**
입니다. 이번 실행이 그 하한을 이미 일부 보여 줍니다(요구된 것 1376 항목) — 남은 것은 972 개
허수아비 중 실제로 쓰인 비율을 재는 것이고, 그것은 허수아비를 하나씩 빼며 다시 돌리면 나옵니다.

---

## 7. 미확인 (추측으로 채우지 않음)

| # | 항목 | 왜 미확인 |
|---|---|---|
| 1 | **`import torch` 완주** | 어느 모드에서도 못 했습니다. 기록 모드 93% 가 최대 |
| 2 | `from_config` · 순전파 · `generate` | 1 에 막혀 미도달 |
| 3 | **972 개 허수아비 중 실제 사용 비율** | 별표 임포트로 심은 것은 기록 훅을 거치지 않음 (§4 의 한계) |
| 4 | `TORCH_USE_RTLD_GLOBAL` 의 기기 영향 | 벽 1 의 우회는 `RTLD_GLOBAL\|RTLD_LAZY` 도 켭니다. 데스크톱에서만 확인 |
| 5 | **327 개 op 중 스키마만으로 족한 것과 커널이 필요한 것의 구분** | 임포트만 봤고 실행을 못 봄 |
| 6 | 벽 19 를 넘은 뒤의 지형 | 여기서 멈춤. 그 아래가 `torch._decomp` 라는 것만 앎 |
| 7 | **기기(Android · iOS)에서의 abi3 로드** | ABI3.md §9 항목 2 그대로. 이번에도 링크만 확인 |
| 8 | `aarch64-apple-ios-sim` | 이번에 빌드하지 않음 |
| 9 | abi3 의 크기 효과 | 세 타깃이 서로 다른 방향으로 움직여 결론 못 냄 (§1) |
| 10 | 벤더링 트리의 라이선스 고지 요건 | BSD 라 벤더링 자체는 문제없으나, 배포 시 고지 방법은 정하지 않음 |
| 11 | 상류 버전 추종 비용 | 2.13.0 한 판본만 벤더링. 재벤더링 시 무엇이 깨지는지 미측정 |

---

## 8. 이번에 만진 것

| 파일 | 변경 |
|---|---|
| `rust/torch_c/Cargo.toml` | `abi3-py313` 추가 |
| `rust/torch_c/pytests/run.sh` | 산출물 이름 `_C.so` → `_C.abi3.so` |
| `.gitignore` | `/torchnative/src/main/torch/` · `/torchnative/src/main/torch-*.dist-info/` · `/vendor/.stamp` |
| `vendor/vendor_torch.sh` | 신규 |
| `vendor/install_shim.sh` | 신규 |
| `vendor/probe.py` | 신규 |
| `docs/platform/VENDOR.md` | 이 문서 |

**벤더링한 트리의 파이썬 소스는 한 줄도 수정하지 않았습니다.** 벽 1 은 상류가 제공하는 환경
변수로 비켰고, 나머지 벽은 전부 계측기 쪽에서 처리했습니다. 상류와 대조한 결과 차이는 **정확히
두 가지**입니다.

```
$ diff -rq --exclude=__pycache__ <상류>/torch torchnative/src/main/torch | grep -v '^Only in <상류>'
Only in torchnative/src/main/torch: _C.abi3.so                     # 우리 것 (§2 의 구멍)
Files .../bin/torch_shm_manager and ... differ       # 벽 4 의 0 바이트 표식
```

`Only in <상류>` 로만 나오는 것들은 제외 규칙(`lib/` · `include/` · `bin/` 의 나머지 · `test/` ·
컴파일 산출물)에 걸린 것입니다.
