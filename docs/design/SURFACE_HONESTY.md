# 표면이 스스로에 대해 하는 말이 사실인가

두 가지를 봤습니다. 뿌리는 하나입니다 — **우리 `_C` 가 "이 이름은 있다" 고 말하는데 그게 사실이
아닌 자리.** 하나는 고쳤고, 하나는 **고치지 않고 판단을 미룹니다.** 미룬 이유와, 무엇을 알아야
정할 수 있는지를 §2.6 에 적었습니다.

측정 환경: `/Volumes/macMini/caches/spike-venv/bin/python`, torch 2.13.0, transformers 5.15.1.
worktree `/Volumes/macMini/worktrees/bw-surface`, 브랜치 `work/surface-honesty`.

---

## 0. 한눈에

| | 상태 |
|---|---|
| §1 `_Unimplemented` 가 truthy 라 플래그가 거짓말을 하던 것 | **고침.** `__bool__` 이 거부하고, 스텁이 `_bool` 로 선언한 14 개는 전부 진짜 값으로 답합니다 |
| §1 이 실제로 바꾼 동작 | `torch.backends.cudnn.benchmark_limit` 이 자리표 쌍 → `None`. 상류와 일치 |
| §2 `torch.distributed.Store` 재수출 | **결정됨 (§2.6).** 벤더 트리는 한 줄도 안 고칩니다 — `torch.distributed` 를 `world_size=1` 부터 실체로 구현하고, 그 부수 효과로 벽이 열립니다. 세 갈래 실측은 §2.6.1 에 남겨둡니다 |
| §2 의 핵심 발견 | **상류 torch 자신도 같은 자리에서 같은 예외로 죽습니다.** 우리 결함이 아닙니다 (§2.2) |
| `from_config` 진행 | **변화 없음.** 같은 벽(`fake_pg.py:7`)에서 멈춥니다. **정정 (문서 감사, 2026-09): 닫혔다 — §2.7 정정 참조** |
| 판정 기준 | 골든 1043/1043 covered=62 · 스키마 127/127 · 스모크 62/62 · strict probe torch·transformers · 3 타깃 — **전부 exit 0** |

---

## 1. `_Unimplemented` 의 진리값

### 1.1 조율 세션이 준 10 개는 한 덩어리가 아니었다

지시받은 목록은 10 개였습니다. **뭉뚱그리지 말라는 주의가 정확했습니다** — 상류 torch 에 직접
물어보니 셋으로 갈렸습니다.

```python
import torch._C as C
for n in names:
    v = getattr(C, n); print(n, type(v).__name__, bool(v))
```

| 갈래 | 개수 | 이름 | 판정 |
|---|---|---|---|
| 상류에서 진짜 `bool` | **5** | `_has_cudnn` F · `_has_cusparselt` F · `_has_kleidiai` **T** · `_has_magma` F · `_has_mkldnn_acl` F | **진짜 거짓말.** 고쳐야 함 |
| 상류에서 `builtin_function_or_method` | **4** | `_has_tensorexpr_cpp_tests` · `_is_alias_of` · `_is_cached_tensor` · `_is_cow_tensor` | 함수는 원래 truthy. **문제 아님** |
| 상류에 아예 없음 | **1** | `_is_batched` | 벤더링 트리의 유일한 언급이 `common_dtensor.py:1005` 의 **주석 처리된 줄**입니다. 산 호출자 없음 |

`_has_tensorexpr_cpp_tests` 가 이름 모양(`_has_`)과 달리 함수라는 것이, **이름 모양은 증거가
아니라는** 것을 보여줍니다. 이 넷의 벤더링 트리 내 등장은 전부 `_dynamo/trace_rules.py` 의
**문자열 리스트** 이거나 결과에 대한 호출이지, 이름 자체에 대한 진리 검사가 아닙니다.

### 1.2 그런데 10 개가 전부가 아니었다

`dir(_C)` 의 `_Unimplemented` 를 **전부** 세고, 각각을 상류 torch 의 실제 타입과 대조했습니다.

```
_C 위의 _Unimplemented 총계         177
  상류에서 callable                 147   ← truthy 가 정답
  상류에서 non-callable (진짜 값)    14   ← 이 중 9 개가 bool
  상류에 없음                        16   ← 텍스트 스캐너의 의도된 과수집
```

non-callable 14 개의 내역:

| | 이름 | 상류 값 |
|---|---|---|
| bool **9** | `_has_cudnn` `_has_cusparselt` `_has_kleidiai` `_has_magma` `_has_mkldnn_acl` **`has_lapack` `has_mkl` `has_openmp` `has_spectral`** | F F **T** F F / T F T T |
| `DispatchKeySet` 3 | `_after_ADInplaceOrView_keyset` · `_after_autograd_keyset` · `_dispatch_autogradother_backends` | 비어 있지 않은 집합 |
| module 2 | `_return_types` · `_te` | 진짜 서브모듈 |

**조율 세션의 스캔이 놓친 것이 넷 더 있었습니다** — `has_lapack` · `has_mkl` · `has_openmp` ·
`has_spectral`. 앞에 밑줄이 없어서 `_has_` 패턴에 안 걸립니다. 그리고 다섯 번째가 더 있는데,
이건 자리표조차 아니었습니다: **`_GLIBCXX_USE_CXX11_ABI` 는 *클래스* 로 답하고 있었습니다.**
설치 루프가 앞머리 대문자를 타입 이름으로 읽는데(`bootstrap.py`, `name.lstrip("_")[:1].isupper()`),
`_GLIBCXX...` 가 거기 걸립니다. 클래스도 truthy 이므로 같은 병입니다.

### 1.3 실제로 아픈 자리는 한 곳이었다

목록을 세는 것과 **그게 실행되느냐** 는 다른 질문입니다. `_Unimplemented.__bool__` 을 *답을
바꾸지 않고 기록만 하는* 것으로 갈아끼우고(즉 현상 유지를 관찰), `import torch` 와 `from_config`
경로를 돌렸습니다.

```
--- truth tests on _Unimplemented: 1 distinct (name, site) ---
  torch._C._has_cudnn   at torch/backends/cudnn/__init__.py:231   x1
```

**딱 하나입니다.** 그리고 그 하나는 진짜입니다 — `:231` 은 `CudnnModule` 의 **클래스 몸체**라
`import torch` 중에 실행됩니다.

```python
benchmark_limit = None
if is_available():                      # -> return torch._C._has_cudnn
    benchmark_limit = ContextProp(
        torch._C._cuda_get_cudnn_benchmark_limit,   # 둘 다 _Unimplemented
        torch._C._cuda_set_cudnn_benchmark_limit,
    )
```

truthy 자리표가 "cuDNN 이 있다" 가지를 타서, 상류가 `None` 을 두는 자리에 **읽으면 터지는 프로퍼티**
가 놓여 있었습니다.

### 1.4 무엇을 골랐나 — (a) 와 (b) 둘 다, 그리고 그게 배타적이지 않은 이유

지시는 (a) `__bool__` 이 예외를 던지게 / (b) 아는 답은 진짜 값으로, 중 **최소 하나**를 고르라는
것이었습니다. §1.3 의 측정이 **둘이 서로 배타적이지 않다는 것**을 보여줍니다.

- **(b) 를 적용하면 경로 위의 진리 검사가 0 이 됩니다.** 그러면 **(a) 는 오늘 아무것도 깨지
  않습니다** — 비용 0 에 다음번을 막습니다. 실측으로 확인했습니다 (§1.6).
- **(b) 만 하면 손으로 적은 목록입니다.** 지금 `_BUILD_FLAGS` 의 주석이 그렇게 말합니다 — 이
  목록은 *벽에 부딪혀서* 만들어졌습니다(`_has_mps` 가 `torch.manual_seed(0)` 를 잡아먹은 건).
  다음에 상류가 `_bool` 을 하나 추가하면 같은 방식으로 또 벽에 부딪힙니다.
- **(a) 만 하면** `_has_cudnn` 이 조용한 오답에서 시끄러운 예외가 될 뿐입니다. `import torch` 가
  깨집니다. 답을 아는데 던지는 것은 게으름입니다.

그래서 **셋을 했습니다.**

1. `_Unimplemented.__bool__` 이 이름을 대며 거부합니다.
2. `_BUILD_FLAGS` 가 스텁이 `_bool` 로 선언한 **14 개 전부**를 답합니다.
3. **그 둘을 손 목록이 아니라 불변식으로 묶었습니다** — `gen_surface.py` 가 `.pyi` 의 `_bool`
   주석을 `surface.json` 에 `"bool"` 이라는 별도 kind 로 남기고, `install` 은 그 kind 의 이름이
   `_BUILD_FLAGS` 에 없으면 **`_C` 임포트 자체를 거부**합니다.

3 번이 핵심입니다. 빌드 플래그는 **답이 둘뿐이고 둘 다 동작을 바꿉니다.** 자리표 모양의 세 번째
답이 없으므로, 답을 안 정한 채로 표면을 만드는 것 자체를 막는 것이 맞습니다.

### 1.5 왜 `__bool__` 이 함수 이름에까지 던져도 되는가

`_is_alias_of` 는 상류에서 함수이고 따라서 truthy 입니다. 우리는 `bool()` 에 예외를 던지므로
**상류와 다릅니다.** 그래도 맞다고 판단한 근거:

`_Unimplemented` 가 되는 이름은 **스텁이 그 이름에 대해 아무 말도 하지 않는** 것들입니다.
스텁이 `def` 로 선언한 이름은 `_make_function` 을 거쳐 **진짜 파이썬 함수**가 되고, 그건 상류의
builtin 과 똑같이 truthy 합니다. 그러므로 `_Unimplemented` 에게 "너 있니?" 는 **우리 자료로는
어느 쪽으로도 답할 근거가 없는 질문**이고, 없다고 말하는 것이 정직합니다.

`surface.json` 이 이 넷을 `"value"` 로 적고 있는 것도 주장이 아닙니다 — `gen_surface.py` 의
`module.setdefault(name, "value")` 는 트리 텍스트 스캔으로 주운 이름의 **기본값**이지, 모양에 대한
진술이 아닙니다.

값은 전부 `False` 이고, 이건 기본값이 아니라 이 빌드에 대한 사실입니다. 둘만 따로 적습니다.

- **`_has_kleidiai`** — 상류는 이 호스트에서 **`True`** 입니다(KleidiAI 는 ARM 커널 라이브러리이고
  arm64 mac 빌드가 집어갑니다). 그건 **상류 빌드에 대한 사실이지 API 에 대한 사실이 아닙니다.**
  베껴 오는 것은 truthy 자리표와 같은 잘못을 점잖게 저지르는 것입니다. 우리는 KleidiAI 를 링크하지
  않으므로 `False`.
- **`_GLIBCXX_USE_CXX11_ABI`** — **질문 자체가 성립하지 않는 유일한 항목입니다.** 이 shim 아래에는
  libstdc++ 이 아예 없으므로 어느 답도 사실을 기술하지 않습니다. `False` 를 고른 이유는 그것이
  **호출자가 GNU C++ ABI 가 있다고 가정하게 만들지 않는 쪽**이기 때문입니다. 유일한 독자는
  `torch/__init__.py:2354` (`compiled_with_cxx11_abi`) 이고 `import torch` 경로에서 호출되지 않습니다.

### 1.6 깨진 것 — 없습니다. 바뀐 것은 하나이고, 원래 잘못이던 자리입니다

| | 이전 | 이후 | 상류 |
|---|---|---|---|
| `torch.backends.cudnn.is_available()` | `<shim: _has_cudnn>` (truthy) | `False` | `False` |
| `CudnnModule.benchmark_limit` | `ContextProp(자리표, 자리표)` | `None` | `None` |
| `torch.cuda.has_magma` | `<shim: _has_magma>` | `False` | `False` |
| `torch.backends.mkl.is_available()` | `<shim: has_mkl>` | `False` | `False` |

**전부 상류와 일치합니다.** 그리고 `torch.cuda.has_magma` 는 `torch/cuda/__init__.py:155` 의
**모듈 스코프 대입**이라, 이전에는 자리표가 그 이름으로 *발행되고* 있었습니다.

(b) 적용 후 §1.3 의 계측기를 다시 돌렸습니다.

```
--- truth tests on _Unimplemented: 0 distinct (name, site) ---   (import torch)
--- truth tests on _Unimplemented: 0 distinct (name, site) ---   (from_config)
```

**0 입니다.** 그래서 `__bool__` 이 던지는 것의 오늘 비용은 0 입니다.

`has_mkl` 이 `False` 가 되면서 사라진 동작이 하나 있는데, 사라지는 게 맞습니다 —
`_meta_registrations.py:2864` 가 `if torch._C.has_mkl:` 아래에서 `torch.ops.mkl._mkl_linear` 의
meta 커널을 등록하고 있었습니다. 이 빌드에 그런 op 은 없습니다.

### 1.7 §1 에서 고치지 않은 것

`_after_ADInplaceOrView_keyset` · `_after_autograd_keyset` ·
`_dispatch_autogradother_backends` (상류는 `DispatchKeySet`) 와 `_return_types` · `_te`
(상류는 모듈) 는 **타입이 틀렸지만 진리값은 양쪽 다 참**이라 이번 주제가 아닙니다. 경로 위에서
아무도 진리 검사를 하지 않는 것을 확인했습니다(§1.6 의 0). **미해결로 남깁니다.**

---

## 2. `torch.distributed.Store` — 고치지 않았습니다

### 2.1 재현

```
torch/_dynamo/variables/functions.py:102   from torch.distributed.fsdp._fully_shard import _fsdp_param_group
torch/distributed/fsdp/_flat_param.py:31   from torch.testing._internal.distributed.fake_pg import FakeProcessGroup
torch/testing/_internal/distributed/fake_pg.py:7   class FakeStore(dist.Store):
AttributeError: module 'torch.distributed' has no attribute 'Store'
```

조율 세션의 보고와 일치합니다. **타입이 없는 것이 아닌 것도 맞습니다** — `surface.json` 의
`_distributed_c10d` 는 타입 43 개를 선언하고 있고 `Store` · `FileStore` · `HashStore` ·
`PrefixStore` · `TCPStore` · `ProcessGroup` · `Backend` · `Work` 가 전부 그 안에 있습니다.
끊긴 것은 **파이썬 레벨 재수출**입니다.

### 2.2 이건 우리 결함이 아닙니다 — 상류에서 그대로 재현됩니다

`torch/distributed/__init__.py` 의 `else:` 가지(= `USE_DISTRIBUTED=0`)를 읽으면 상류가 스스로
적어둔 말이 있습니다.

```python
else:
    # This stub is sufficient to get
    #   python test/test_public_bindings.py -k test_correct_module_names
    # working even when USE_DISTRIBUTED=0.  Feel free to add more
    # stubs as necessary.
    class _Stub: pass
    sys.modules["torch.distributed"].GroupName = _Stub
    sys.modules["torch.distributed"].ProcessGroup = _Stub
```

**`Store` 는 없습니다.** 그 가지는 "테스트 하나가 돌 만큼" 이라고 스스로 밝힙니다.

**진짜 상류 torch 로 확인했습니다.** shim 은 관여하지 않습니다 — 상류 자신의 스위치인
`_c10d_init` 을 숨기고 `torch.distributed` 를 다시 임포트했습니다.

```
baseline                is_available: True   Store: True
USE_DISTRIBUTED=0 sim   is_available: False  Store: False
fsdp FAILED: AttributeError: module 'torch.distributed' has no attribute 'Store'
    torch/distributed/fsdp/__init__.py:1
    torch/distributed/fsdp/_flat_param.py:31
    torch/testing/_internal/distributed/fake_pg.py:7 | class FakeStore(dist.Store):
```

**같은 줄, 같은 예외.** 즉 `import torch.distributed.fsdp` 는 torch 2.13.0 의 어떤
`USE_DISTRIBUTED=0` 빌드에서도 깨집니다. 우리 shim 은 상류가 **출시는 하지만 이 경로로 시험하지는
않는** 구성을 충실히 재현하고 있을 뿐입니다. (PyPI 의 mac wheel 은 `USE_DISTRIBUTED=1` 로
나옵니다 — 그래서 아무도 안 밟았습니다.)

상류 가드가 `except ModuleNotFoundError` 인 것도 이 각도에서 보면 다릅니다. `functions.py:103` 은
실패하면 `_fsdp_param_group = None` 으로 둡니다 — **상류는 fsdp 가 없는 것을 지원할 의사가
있습니다.** 우리가 막히는 이유는 잡는 예외 종류가 한 칸 좁아서일 뿐입니다.

### 2.3 갈래 (a) — 스위치는 끄고 `Store` 만 바인딩: **구현 수단이 없습니다**

`Store` 는 `if is_available():` **안에서만** 바인딩됩니다. 그 블록 밖에서 그 이름을 만들려면
둘 중 하나가 필요합니다.

1. **벤더링 트리를 고친다** — DESIGN.md §2 의 "파이썬 계층은 벤더링하고 `_C` 만 교체한다" 와
   IMPORT_TORCH.md 서두의 "벤더링 트리를 한 줄도 고치지 않고" 라는 기록된 성질을 깹니다.
2. **`torch.distributed` 네임스페이스에 밖에서 써넣는다** — DESIGN.md §1 이 금지한 파사드입니다.
   그리고 **수단도 없습니다**: `_C` 가 파이썬 모듈 네임스페이스에 쓰는 자리는 `_initExtension` 과
   `_multiprocessing_init` 인데, 둘 다 `torch.distributed` 임포트 시점에 걸려 있지 않습니다.
   그 시점에 걸려 있는 훅은 `_c10d_init` 하나이고, 그건 지금 꺼져 있는 바로 그 스위치입니다.

덧붙여 이 갈래는 **자기모순**입니다 — `is_available()` 은 False 인데 `Store` 는 있는 상태입니다.
조율 세션이 지적한 그대로입니다.

### 2.4 갈래 (b) — 스위치를 켠다: **실측 결과 회귀입니다**

켜 봤습니다(커밋하지 않은 실험 스크립트, `_c10d_init = lambda: True` 한 줄).

```
STOP at [import torch]: AttributeError: '_Unimplemented' object has no attribute 'UNDEFINED'
     torch/utils/data/__init__.py:1        from torch.utils.data.dataloader import (
     torch/utils/data/dataloader.py:26     import torch.distributed as dist
     torch/distributed/__init__.py:146     from .device_mesh import DeviceMesh, init_device_mesh
     torch/distributed/distributed_c10d.py:270  class Backend(str):
     torch/distributed/distributed_c10d.py:321      UNDEFINED: ProcessGroup.BackendType.UNDEFINED,
```

**지금 통과하는 `import torch` 가 깨집니다.** `torch.utils.data` 가 `torch.distributed` 를 끌고
오므로 이건 지연 경로가 아니라 `import torch` 본체입니다. 벽이 늦은 좁은 경로에서 이른 넓은
경로로 **옮겨옵니다.**

"이름만 채우면 되나" 를 재기 위해 `_distributed_c10d` 를 `probe.py` 의 **카멜레온**(모든 질문에
답하는 계측기) 으로 갈아끼우고 다시 돌렸습니다.

```
STOP at [import torch]: AttributeError: __members__
     torch/distributed/distributed_c10d.py:560   reduce_op = _reduce_op()
     torch/distributed/distributed_c10d.py:547   for k, v in ReduceOp.RedOpType.__members__.items():

_distributed_c10d top-level names asked: 50
names asked *below* a member (structure, not presence): 7
  ProcessGroup.BackendType.{UNDEFINED,GLOO,NCCL,UCC,MPI,XCCL,CUSTOM}
```

**모든 질문에 답하는 카멜레온조차 `import torch` 를 끝내지 못합니다.** 요구되는 것이 이름이
아니라 **구조**이기 때문입니다 — 진짜 `enum` 이어야 하고(`__members__` 를 순회함), 중첩되어
있어야 합니다(`ProcessGroup.BackendType`). 최상위 이름 50 개는 **하한**이지 상한이 아닙니다
(카멜레온이 거기서 멈췄으므로 그 너머는 못 셌습니다).

즉 (b) 는 "표면이 넓어진다" 가 아니라 **서브시스템 하나를 구현하는 일**이고, 그 대가로 지금
동작하는 `import torch` 를 담보로 겁니다.

### 2.5 갈래 (c) — 지시에 없던 것: 실패 종류를 바꾼다: **실측 결과 막힘**

§2.2 에서 나온 관찰을 밀어봤습니다. `functions.py:102` 는 `ModuleNotFoundError` 를 잡습니다.
`torch/testing/_internal` 이 **벤더링 대상에서 빠지면** `_flat_param.py:31` 이 바로 그
`ModuleNotFoundError` 를 던지고, 상류가 이미 지원하는 가지로 들어갑니다. 그리고 이건 트리를
고치는 것이 아니라 **벤더링 정책**입니다 — `vendor_torch.sh` 는 이미 `torch/lib/` ·
`torch/include/` · `torch/bin/` · `torch/test/` 를 그렇게 떨굽니다.

실제로 떼고 돌렸습니다(측정 후 복원, `probe --target torch` 재확인 exit 0).

```
STOP at [import torch]: ModuleNotFoundError: No module named 'torch.testing._internal'
     torch/_higher_order_ops/flex_attention.py:33  from torch.utils.checkpoint import _CachedTorchDispatchMode, ...
     torch/utils/checkpoint.py:19   from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
```

**막힙니다.** torch 2.13.0 의 **코어**인 `torch/utils/checkpoint.py` 가 모듈 스코프에서
`torch/testing/_internal/` 을 임포트하고, 그게 `import torch` 중에 실행됩니다. 테스트 지원
트리가 `import torch` 에 대해 load-bearing 이라는 뜻이고, 상류도 이걸 못 뗍니다.

(`_dynamo/repro/after_aot.py:537` 에도 같은 임포트가 보이지만 **`textwrap.dedent` 안의 문자열
리터럴**입니다. 산 임포트가 아닙니다 — 죽은 코드를 산 것으로 읽지 않으려고 확인했습니다.)

### 2.6 결론 — 정해졌습니다 (2026-08-24)

> **패치 세트는 두지 않습니다. 벤더링 트리는 한 줄도 고치지 않습니다.**
> `import transformers` 는 **분산 표면이 실제로 선 뒤에** 진행합니다.

아래 §2.6.1 이 이 결정이 나오기 전의 분석이고, 그대로 둡니다 — 세 갈래가 왜 막혔는지는
여전히 유효한 측정이기 때문입니다. **다만 결론 부분("지금 정하지 않는다")은 위 결정으로
대체되었습니다.**

**이 결정이 (b) 를 고른 것은 아닙니다.** (b) 는 "오프스위치를 켜서 상류 서브시스템을
떠안는다" 였고, 실제로 갈 길은 **`torch.distributed` 를 우리가 실체로 구현하는 것**입니다 —
`world_size = 1` 이라는 참말부터 시작해서요. 온디바이스 추론 프로세스는 진짜로 랭크가
하나이므로 그것은 축소판이 아니라 정상 구성입니다. `Store` 가 실제로 바인딩되면서
`functions.py:102` 의 벽은 **부수 효과로** 열립니다.

그리고 이것은 우회가 아니라 **범위 안의 작업**입니다. torchnative 의 목표에 FL 이 처음부터
들어 있고, 연합 학습은 집합 통신 위에 세워집니다 — `all_reduce` · `broadcast` · `gather` 가
곧 FedAvg 입니다. 지금까지 "온디바이스에서 쓸 일 없다" 며 꺼둔 서브시스템이 사실 **FL 축의
기반**이었습니다.

**계획한 스택** (위가 아래에 의존):

```
brainwave.federated       라운드 · 클라이언트 선택 · 집계 · 이탈 처리
  └ torch.distributed     ProcessGroup · 집합 통신 (전송 추상)
      └ 백엔드             register_backend 로 우리 것
          └ 장치 추상       CPU · Metal · Vulkan · NPU
```

세 가지 주의:

1. **FL 층의 이름공간은 이미 정해져 있습니다 — 이 항목은 저장소를 안 보고 쓴 것이라 정정합니다.**
   구현은 `torchnative.nn.federated` 에 있고, `torch/nn/federated.py` 는 그것을 `torch`
   이름공간에 얹는 **한 줄 add-hook** (`from torchnative.nn.federated import *`) 입니다.
   `torch.federated` 를 새로 만드는 것이 아니라 `torch.nn` 아래이고, 구현이 아니라 재수출입니다.

   **이것은 위의 "패치를 두지 않는다" 결정과 충돌하지 않습니다.** 패치는 상류에 이미 있는 파일을
   고치는 것이라 재벤더링마다 다시 붙여야 하고 버전마다 어긋납니다. add-hook 은 상류에 **없는
   파일을 더하는 것**이라 `vendor_torch.sh` 의 rsync 와 겹칠 경로가 없습니다. 더하기와 고치기는
   비용이 다릅니다.

   `DESIGN.md` §2 가 이미 두 가지를 못 박아뒀습니다 — **주입 지점을 하나로 모을 것**, 그리고
   **add-hook 은 편의이지 의존이 아닐 것**(데스크톱에서 상류 torch 위에서도 동작해야 하므로).
   `torchnative/src/main/torch/README.md` 는 그 합치는 방법이 아직 미정이라고 적어두었고,
   그것은 여전히 열린 항목입니다.
2. **`ProcessGroup` 의 가정은 cross-device FL 과 안 맞습니다.** 그것은 고정 세계 크기 ·
   전원 참석 · 동기 · 신뢰를 전제하는데, FL 은 부분 참여 · 이탈 · 비동기가 정상입니다.
   기기 안 이기종(랭크 = 장치)과 cross-silo FL 에는 맞습니다. 해법은 `ProcessGroup` 을
   **전송 추상으로만** 쓰고, 라운드마다 그때 도착한 참가자로 **임시 그룹을 구성**하는 것입니다.
3. **공통 기반은 분산 표면이 아니라 그 아래 장치 추상입니다.** 랭크가 장치를 가리키려면
   `torch.device` 와 장치별 디스패치가 먼저 있어야 하고, 가속기(Metal · Vulkan · NPU)도
   전부 그 위에 얹힙니다. 분산을 먼저 세워도 랭크가 가리킬 것이 없으면 껍데기입니다.

**그때까지 검증은 손으로 옮겨 적은 모델로 계속합니다.** `from_pretrained` 와 실제 체크포인트
경로는 이 벽 뒤에 있으므로, 그 둘은 분산 표면이 선 뒤에야 진짜로 시험됩니다.

---

### 2.6.1 (기록) 결정 전의 분석 — 무엇을 알아야 정할 수 있었는가

세 갈래가 각각 다른 이유로 막혔고, **셋 다 shim 바깥의 전제에 걸립니다.**

| 갈래 | 막힌 지점 | 성격 |
|---|---|---|
| (a) `Store` 만 바인딩 | 수단이 없음. 만들면 파사드 | 설계 전제 |
| (b) 스위치 ON | `import torch` 회귀 + 서브시스템 구현 | 작업량 · 회귀 |
| (c) `testing/_internal` 미벤더링 | `torch/utils/checkpoint.py:19` | 상류 구조 |

**정하려면 알아야 하는 것 — 하나입니다.**

> **벤더링 트리에 대한 패치 세트를 이 프로젝트가 감당할 의사가 있는가?**

이 전제가 **어느 문서에도 기록되어 있지 않습니다.** `docs/` 전체에서 찾은 것은 "한 줄도 고치지
않았다" 는 *성취 기록*(IMPORT_TORCH.md:6, DYNAMO.md:245) 뿐이고, **금지 규정이 아닙니다.**
DESIGN.md §1 이 금지하는 것은 파사드 — transformers 모양을 흉내내는 층 — 이지, 벤더링한 상류
소스에 대한 패치가 아닙니다. 둘은 다릅니다.

이 답에 따라 갈립니다.

- **감당한다면** — 한 줄입니다. `functions.py:102` 의
  `except ModuleNotFoundError` → `except (ModuleNotFoundError, AttributeError)`.
  상류의 의도(`_fsdp_param_group = None`)를 그대로 실행할 뿐이고, §2.2 가 이게 **상류 버그**임을
  보였으므로 업스트림에 그대로 올릴 수 있는 모양입니다.
- **감당하지 않는다면** — (b) 뿐이고, `_C._distributed_c10d` 를 진짜 열거형과 중첩 구조까지
  포함해 구현하는 별도 작업입니다. 그 전에 `import torch` 회귀를 감수할 수 없으므로 **작업이
  끝날 때까지 스위치를 켤 수 없습니다** — 즉 한 번에 착지시켜야 하는 큰 덩어리입니다.

**미확인으로 남기는 것들:**

| # | 항목 | 상태 |
|---|---|---|
| 1 | 패치 세트 허용 여부 | **미기록.** 위 질문. 이 결정의 유일한 갈림길 |
| 2 | (b) 를 끝까지 갔을 때의 실제 크기 | **미측정.** 카멜레온이 `import torch` 를 못 끝내 50 개 너머를 못 셌음 |
| 3 | `Store` 벽을 넘은 뒤의 다음 벽 | **미확인.** 넘지 못했으므로 셋 중 무엇도 그 너머를 보지 못했습니다 |
| 4 | `torch.distributed` 를 `import torch` 에서 떼는 경로가 있는지 (`torch.utils.data.dataloader:26` 이 무조건 끌고 옴) | **미조사** |

### 2.7 `from_config` 진행 상황

**진전 없습니다.** §1 의 수정 전후로 같은 자리에서 멈춥니다.

```
fake_pg.py:7  class FakeStore(dist.Store):
AttributeError: module 'torch.distributed' has no attribute 'Store'
```

§1 은 이 벽보다 **앞쪽**(`import torch` 중 `torch/backends/cudnn`)을 고친 것이라 도달 거리를
바꾸지 않습니다. 다음 벽이 무엇인지는 **모릅니다** — 위 §2.6 항목 3.

> **정정 (문서 감사, 2026-09):** 이 벽은 이 문서의 바로 다음날 닫혔다. `docs/distributed/DISTRIBUTED.md`
> (착지 커밋 `99fec1b`, "Feat: Stand up torch.distributed, and import transformers for the first
> time", 2026-08-25 06:52 — 이 문서의 마지막 커밋 `eae2a42` 은 전날 22:38)가 §2.6 이 여기서
> 결정한 바로 그 계획("`torch.distributed` 를 `world_size=1` 부터 실체로 구현")을 실행했다.
> `torch.distributed.Store` 가 오늘 존재하고(`hasattr(torch.distributed, 'Store')` → `True`),
> `from_config` 는 오늘 이 정확한 시나리오로 성공한다(실측, 파라미터 수 95,040개 —
> `docs/models/FROM_CONFIG.md` 감사(이 라운드)가 실물 torch 로 잰 것과 정확히 같음). §2.6 이 스스로
> "그때까지 검증은 손으로 옮겨 적은 모델로 계속합니다" 라고 적어 둔 "그때" 가 왔다.
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_distributed_c10d present -->

이제 위 §0 요약표의 "`from_config` 진행: 변화 없음" 줄도 같은 이유로 낡았다 — 아래에서 다시
쓰지 않고 이 정정을 가리킨다.

---

## 3. 변경 범위와 검증

```
 M rust/torch_c/src/bootstrap.py        __bool__ · _BUILD_FLAGS 14 개 · install 의 불변식 검사
 M rust/torch_c/pytests/test_shim.py    테스트 2 개 (구현 전 둘 다 적색 확인)
 M vendor/gen_surface.py                `_bool` 주석을 "bool" kind 로 보존
 M rust/torch_c/src/surface.json        재생성. 14 개 이름의 kind 만 바뀜, 그 외 바이트 동일
```

`surface.json` 은 변경 **전에** 한 번 재생성해 `git` 판본과 완전히 동일함을 확인한 뒤 바꿨습니다.
그래서 diff 가 의도한 14 개뿐임이 보장됩니다.

TDD 순서를 지켰습니다. 구현 전 실행:

```
FAIL test_a_placeholder_refuses_a_truth_test: AssertionError: a placeholder must not answer a truth test
FAIL test_every_build_flag_the_stubs_declare_answers_with_a_real_bool: AssertionError: []
```

최종:

| 검증 | 결과 |
|---|---|
| 스모크 (`pytests/run.sh`) | **exit 0** — ok 62 / FAIL 0 |
| 골든 (`tools/golden/compare.py`) | **exit 0** — 1043/1043, ops covered=**62** |
| 스키마 (`pytests/verify_schemas.py`) | **exit 0** — 127/127 |
| strict probe `--target torch` | **exit 0** |
| strict probe `--target transformers` | **exit 0** |
| 호스트 `cargo build --release` | **exit 0** |
| Android `cargo ndk -t arm64-v8a` | **exit 0** |
| iOS `--target aarch64-apple-ios` | **exit 0** |

`compare.py` 와 `verify_schemas.py` 는 지시대로 `PYTHONPATH=$PWD/vendor` **없이** 돌렸습니다.
probe 는 `TORCH_USE_RTLD_GLOBAL=1` 이 필요합니다 (VENDOR.md:181 — `libtorch_global_deps` 부재).

### 재현

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-surface
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
PY=/Volumes/macMini/caches/spike-venv/bin/python

vendor/vendor_torch.sh && vendor/install_shim.sh
PYTHON=$PY sh rust/torch_c/pytests/run.sh
$PY tools/golden/compare.py
$PY rust/torch_c/pytests/verify_schemas.py
TORCH_USE_RTLD_GLOBAL=1 $PY vendor/probe.py --mode strict --target torch
```

§1.3 의 진리 검사 계측과 §2.4 의 카멜레온 깊이 측정은 `/tmp` 의 임시 스크립트로 돌렸습니다 —
저장소에 남기지 않았습니다. 방법은 §1.3 · §2.4 에 적은 것이 전부입니다(자리표 클래스의
`__bool__` 을 기록기로 교체 / `sys.modules["torch._C._distributed_c10d"]` 를 기록하는 카멜레온
모듈로 교체).
