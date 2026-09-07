# `torch.distributed` — world_size 1 부터 세운 기록

`DESIGN.md` §11.1 이 1 단계가 멈춘 자리로 지목한 벽 — `torch.distributed.Store` 재수출이 끊겨
`from transformers import AutoModelForCausalLM` 이 실패하던 것 — 을 여는 작업의 기록입니다.

측정일 2026-08-25. 호스트 `darwin/arm64`, CPython 3.13.0, 상류 torch 2.13.0 · transformers 5.15.1
(`/Volumes/macMini/caches/spike-venv`). **벤더링 트리는 한 줄도 고치지 않았습니다** — 바뀐 것은
`rust/torch_c/src/bootstrap.py`, `rust/torch_c/pytests/{test_shim.py,verify_schemas.py}`,
그리고 새로 만든 `torchnative/src/main/torchnative/distributed/__init__.py` 뿐입니다.

---

## 0. 한눈에

| | 전 | 후 |
|---|---|---|
| `import torch` | 통과 | 통과 |
| `torch.distributed.is_available()` | **False** | **True** |
| `torch.distributed.Store` | **없음** | 있음 |
| `import torch.distributed.fsdp` | `AttributeError: ... has no attribute 'Store'` | 통과 |
| `import transformers` | 통과 (원래 torch 를 건드리지 않음) | 통과 |
| `from transformers import AutoModelForCausalLM` | **실패** | **통과** |
| `LlamaForCausalLM` 임포트 | 실패 | 통과 |
| `AutoModelForCausalLM.from_config(...)` | 실패 | **통과 — 진짜 transformers 모델 객체** |
| 순전파 | 실패 | 실패 — **`torch._C.is_autocast_enabled`** (§7) |
| `init_process_group(world_size=1)` | 실패 | 통과 (`backend="local"`) |
| `pytests/run.sh` | 113 통과 | **129 통과** |
| `tools/golden/compare.py` | 2268/2268 | 2268/2268 (변화 없음) |
| `verify_schemas.py` | 233/233 | **255/255** |

**판정: `import transformers` 는 됩니다.** 그리고 그 너머로 `from_config` 까지 갑니다.
다음 벽은 분산이 아니라 **autocast** 이고, 이름은 §7 에 있습니다.

---

## 1. 무엇이 없어서 막혔는지 — 측정으로 만든 목록

지시대로 추측하지 않고, `_c10d_init` 스위치를 켠 뒤 나오는 예외를 하나씩 따라가며 목록을
만들었습니다. 각 줄은 실제로 만난 예외입니다. **이 목록이 이 작업의 사양이었습니다.**

| # | 막힌 지점 | 성격 |
|---|---|---|
| 1 | `torch/distributed/__init__.py:28` — `hasattr(torch._C, "_c10d_init")` | 스위치. `bootstrap.py` 의 "Deliberate omissions" 에 들어 있었음 |
| 2 | `distributed_c10d.py:321` — `ProcessGroup.BackendType.UNDEFINED` | **구조**. 중첩 `enum`, 멤버 7 개 |
| 3 | `distributed_c10d.py:547` — `ReduceOp.RedOpType.__members__` 순회 | **구조**. 진짜 `enum.Enum` |
| 4 | `_shard/sharded_tensor/metadata.py:20` — `torch.get_default_dtype()` | 아예 없었음. aten op 이 아니라 오버로드 표에 있을 리가 없었음 |
| 5 | `_functional_collectives.py:637` — `register_autograd("_c10d_functional::wait_tensor", ...)` | 스키마가 비어 있었음 (§3) |
| 6 | `torch/_library/utils.py:104` — `if schema.is_mutable:` | **결함**. 메서드였음 → 항상 참 (§3.1) |
| 7 | `device_mesh.py:1679` — `register_opaque_type(ProcessGroup, ...)` | 메타클래스가 `OpaqueBaseMeta` 여야 함 |
| 8 | `_dynamo/polyfills/tensor.py:8` — `TensorBase._make_subclass` 시그니처 대조 | **결함** (§3.2) |
| 9 | `c10d_logger.py:96` — `_WaitCounter(...).guard()` | 컨텍스트 매니저가 필요 |
| 10 | `distributed_c10d.py:1448` — `_set_global_rank` | 프로세스 전역 부기 |
| 11 | `transformers/integrations/moe.py:253` — `register_autograd("transformers::grouped_mm_fallback", ...)` | 트리가 `define()` 한 스키마를 shim 이 버리고 있었음 (§3) |
| 12 | `_prims/__init__.py:368` — `type_.isSubtypeOf(...)` | 11 을 고치자 드러남 (§3.3) |

`SURFACE_HONESTY.md` §2.4 가 "카멜레온조차 `import torch` 를 못 끝낸다" 고 적은 것이 정확했습니다 —
2·3 번은 **이름이 아니라 구조**를 요구하고, 모든 질문에 답하는 계측기로는 넘을 수 없습니다.

---

## 2. 갈림길은 어떻게 정해졌나

`SURFACE_HONESTY.md` §2.6.1 은 세 갈래 (a) `Store` 만 바인딩 (b) 스위치 ON
(c) `testing/_internal` 미벤더링 을 놓고 **"벤더링 트리 패치를 감당할 의사가 있는가"** 를
유일한 갈림길로 지목하고 미기록으로 남겼습니다.

이번 작업의 지시가 그 질문에 답했습니다 — **벤더링 트리는 고치지 않는다.**
그래서 갈래는 **(b) 하나**였고, 그 문서가 (b) 에 붙여 둔 대가("지금 통과하는 `import torch` 를
담보로 건다")를 그대로 지불했습니다. 실제로 `torch/utils/data/dataloader.py:26` 이
`import torch.distributed` 를 무조건 하므로, 이 서브시스템은 **`import torch` 본체 안에서**
완성되어 있어야 합니다. 그것이 이 작업이 한 번에 착지해야 했던 이유입니다.

---

## 3. 분산이 아니었던 결함 넷

목록의 6 · 8 · 11 · 12 는 분산 코드가 아닙니다. **분산을 켜자 드러난 것**이고, 넷 다
"없어서 막힌 것" 이 아니라 **틀린 답을 조용히 하고 있던 것**입니다.

### 3.1 `FunctionSchema.is_mutable` 이 메서드였다 — 항상 참인 술어

상류에서는 **property** 입니다 (2.13.0 에서 직접 확인: `torch.parse_schema(...).is_mutable` 이
`bool`, 디스크립터가 `property object`). shim 에서는 메서드였고, 벤더링 트리는 **16 곳에서
속성으로 읽습니다** (`torch/_library/utils.py:104` 외 15).

바운드 메서드는 truthy 이므로 **모든 스키마가 "mutable" 로 답했고**, `is_functional_schema` 가
어디서나 False 였고, `torch.library.register_autograd` 가 받는 op 마다 거절했습니다.

**왜 여태 안 걸렸는가**: 이것을 덮던 유일한 테스트가 shim 자신의 철자인 `is_mutable()` 를
썼습니다. 괄호를 붙이면 메서드든 property 든 (property 면 `False()` 로 터지지만, 당시엔
메서드였으므로) 통과합니다. **CLAUDE.md §5.5 가 말하는 "실패할 수 없는 검증"** 의 표본입니다.

`_is_view_op` 는 메서드로 둡니다 — `torch/distributed/tensor/_dispatch.py:569` 가 괄호를 붙입니다.

### 3.2 `TensorBase._make_subclass` 가 시그니처를 광고하고 있었다

`torch/_dynamo/decorators.py:966` 은 `inspect.signature(original_fn)` 을
`except ValueError: pass` 로 감쌉니다. **상류의 `_make_subclass` 는 C 빌트인이라 시그니처를
못 읽으므로 상류는 이 비교를 아예 안 합니다.** shim 것은 파이썬 함수라 비교가 돌았고,
폴리필이 `requires_grad` 로 쓰는 이름을 shim 은 `require_grad` 로 쓰고 있어 거절당했습니다.

**어느 쪽이 맞는가를 실측했습니다.** 상류 2.13.0 에서:

```
torch.Tensor._make_subclass(torch.Tensor, t, requires_grad=True)  -> TypeError (unexpected keyword)
torch.Tensor._make_subclass(torch.Tensor, t, require_grad=True)   -> OK
```

즉 **shim 의 철자가 맞고 폴리필의 철자가 틀렸습니다** (`torch/_C/__init__.pyi:2389` 도
`require_grad`). 폴리필에 맞추면 상류가 거절하는 키워드를 받고 상류가 받는 키워드를 거절하게
되므로, **시그니처를 감췄습니다** — `__signature__` 를 `(*args, **kwargs)` 로 둡니다. 이는
같은 함수 안의 `sig_ident(...) != sig_ident(wildcard_sig)` 가 dynamo 자신이 마련해 둔 예외이고,
상류가 주는 것과 **같은 양의 정보**(= 없음)입니다. 무엇을 받는지는 `def` 와 스텁이 계속 말합니다.

### 3.3 `define()` 한 스키마를 버리고 있었다 — 그리고 그것을 지키자 `isSubtypeOf` 가 필요해졌다

`_DispatchLibrary.define(schema)` 는 스키마 **텍스트를 받아서 기록만 하고 버렸고**,
나중에 `torch.ops.<ns>.<op>._schema` 를 읽으면 빈 스키마가 나왔습니다. 빈 스키마는 반환값이
없으므로 절대 functional 이 아니고, `@torch.library.custom_op` 로 정의된 op 은 전부
`register_autograd` 에서 거절당했습니다 (`transformers/integrations/moe.py:253`).

고침: `define()` 이 텍스트를 파싱해 표에 넣습니다. **이것은 갭을 좁히지 않습니다** — op 의
스키마를 아는 것과 op 을 돌릴 줄 아는 것은 다르고, `_aten_dispatch` 는 여전히 이 네임스페이스에
커널이 없습니다.

**그 결과 `_prims/__init__.py:368` 이 새로 도달했습니다.** 전에는 `any(...)` 가 빈 인자 목록 위를
돌아 False 였는데, 스키마가 실물이 되자 질문이 실제로 도착해
`'_SchemaType' object has no attribute 'isSubtypeOf'` 가 났습니다. `isSubtypeOf` 와
`containedTypes` 를 구현했습니다 — **필요한 두 관계만**: `T <: T`, `T <: Any`.
`T?` 와 `T[]` 는 `T` 의 서브타입이 **아니고**(상류 규칙), 그래서 `contains_tensor_types` 가
`containedTypes()` 로 재귀합니다. 그 이상은 타입 격자이고, 만들면 틀린 답이 조용히 지나갑니다.
읽을 수 없는 상대(`ListType.ofInts()` 등)는 **이름을 대고 거절**합니다.

### 3.4 `torch.get_default_dtype()` 이 없었다

aten op 이 아닙니다 — 상류가 `THPModule_getDefaultDtype` 을 `_C` 에 직접 붙입니다
(`torch/_C/__init__.pyi:1399`). 그런데 `_VariableFunctions` 목록에도 있어서 오버로드 해석을
거쳤고, **존재하지 않는 `torch.ops.aten.get_default_dtype.<overload>` 를 가리키는 거절**을
냈습니다 — 아무도 닫을 수 없는 작업 항목.

값은 자유가 아닙니다: `aten.rs` 의 `DEFAULT_FLOAT` 와 같아야 하고, 테스트가 상수를 단언하는
대신 디스패처에 물어서 확인합니다. `set_default_dtype` 은 **이름을 대고 거절**합니다 —
Rust 상수에 닿아야 하는데 닿을 수 없으므로, 받아놓고 아무것도 안 하는 것보다 거절이 맞습니다.

> **거절이 닫혔습니다 (2026-08-28).** `transformers` 의 `modeling_utils.py:239` 이
> `from_pretrained` 로 들어가는 길에 `torch.set_default_dtype(dtype)` 을 무조건 부르므로,
> 상수를 프로세스 전역으로 바꿨습니다 — `dtype.rs` 의 `DEFAULT_FLOAT_CHOICE`(`AtomicU8`)
> 와 `default_float()`. 위 문단의 `aten.rs` `DEFAULT_FLOAT` 는 더 이상 없습니다.
>
> **핵심은 그 전역이 실제로 읽히는가입니다.** `ones`·`zeros`·`empty`·`arange`·
> `torch.tensor`·`full`·`scalar_tensor`·`rsqrt`·`cos`·`pow`·`mul.Scalar`·`div.Scalar`·
> `torch.finfo()` 가 전부 따라가는 것을 단언합니다
> (`test_set_default_dtype_moves_every_rule_that_reads_the_default`). 그중 `full` 과
> `finfo()` 는 상수를 직접 적고 있었으므로 같이 고쳤습니다 — 안 고쳤다면 그 둘만 float32 에
> 남는, **아무것도 안 하는 세터보다 나쁜** 상태가 됐을 것입니다.
>
> 받고 거절하는 것은 상류를 재서 그대로 옮겼습니다: float32·float64·float16·bfloat16 만
> 받고, 부동소수가 아닌 태그는 `TypeError: only floating-point types are supported as the
> default type`, float8/float4 여섯은 상류의 저장소 클래스 탐색을 그대로 재현해
> `couldn't find storage object <X>Storage`, dtype 이 아닌 객체는 `invalid dtype object: ...`
> 입니다. `set_default_tensor_type` 은 다른 이름이고, 걸음이 요구하지 않아 그대로 둡니다.

---

## 4. `_distributed_c10d` — 무엇이 실물이고 무엇이 거절인가

### 4.1 실물

| | 왜 실물인가 |
|---|---|
| `Store` · `HashStore` · `PrefixStore` | **world_size 1 에서 "분산 저장소" 와 "로컬 dict" 는 같은 객체입니다.** 이 프로세스가 유일한 기록자이자 유일한 독자이므로 대역이 아니라 구현입니다. `PrefixStore` 는 진짜 뷰라 두 그룹이 한 저장소를 겹치지 않고 씁니다 |
| `Work` | 항상 완료. 단일 랭크가 수행하는 집합 통신은 반환 시점에 이미 끝나 있습니다 — 기다릴 상대가 없다는 것이 사실입니다 |
| `ReduceOp` / `RedOpType` · `ProcessGroup.BackendType` · `DebugLevel` · `ErrorType` | 진짜 `enum.Enum`. `__members__` 를 순회당하고 중첩되어 있어야 합니다 |
| `Backend` · `Backend.Options` · `ProcessGroup` | 진짜 타입. `class _IllegalWork(Work)` 상속, `_export_c_types()` 의 `__module__` 대입, `C10dBackend.Options \| None` 평가를 받습니다 |
| `_register_process_group` / `_resolve_process_group` / `_set_global_rank` … | 프로세스 전역 부기. 상대가 필요 없으므로 거절할 이유가 없습니다 |

### 4.2 이름을 대고 거절하는 것

| | 거절 이유 |
|---|---|
| `TCPStore` | 소켓 반대편에 아무도 없습니다. 로컬 dict 처럼 굴면서 랑데부 지점이라고 주장하는 것이 `CKPT.md` 가 기록한 사고입니다 |
| `Store.wait(absent_key)` | world_size 1 에서 없는 키는 "아직" 이 아니라 **"영원히"** 입니다. 조용히 반환하면 호출자는 상대가 답한 줄 압니다 |
| `send` · `recv` · `recv_anysource` | 지역 작업으로 참이 될 수 없습니다. 조용한 무연산이면 호출자가 **쓰이지 않은 버퍼를 읽고** 이유를 영영 모릅니다 |
| `all_reduce(op=PREMUL_SUM)` | `sum(factor * x_i)` 는 world 1 에서 `factor * x` 이지 `x` 가 **아닙니다.** 다른 리덕션과 같이 항등 취급하면 맞는 모양의 틀린 답입니다 |
| `rootRank != 0` | 크기 1 인 세계에 랭크 3 을 루트로 지정하는 것은 호출자의 버그입니다. 0 으로 깎으면 그것을 감춥니다 |
| `Work.get_future` · `Work.boxed` | `torch.futures.Future`/ScriptObject 가 이 빌드에 없습니다 |

### 4.3 **부재**로 두는 것 — 스텁이 아니라 없음

`ProcessGroupGloo` · `ProcessGroupNCCL` · `ProcessGroupUCC` · `ProcessGroupXCCL` ·
`ProcessGroupMPI` · `_ProcessGroupWrapper`.

`distributed_c10d.py:204-242` 는 각각을 자기 `try/except ImportError` 로 임포트하고
`_GLOO_AVAILABLE` 같은 플래그를 세웁니다. **부재가 트리가 읽도록 쓰여 있는 답**입니다.

> **이 작업에서 두 번 틀렸고, 둘 다 테스트가 잡았습니다.**
>
> 1. 모듈의 catch-all `__getattr__` 이 대문자 이름을 타입으로 합성했습니다 → `_GLOO_AVAILABLE = True`.
> 2. 1 을 고쳐 `AttributeError` 를 내게 했더니, `from X import Y` 는 속성 다음에 **임포트**를
>    시도하고 `_SubmoduleFinder` 가 같은 이름을 **빈 모듈로** 돌려줬습니다 → 다시 True.
>
> 두 번째가 특히 그렇습니다 — 속성 조회는 올바르게 거절했는데 두 번째 경로가 뒤집었습니다.
> `_SubmoduleFinder` 에 `closed` 를 추가해 `_distributed_c10d` 아래로는 모듈로 답하지 않게
> 했습니다. 지금은 트리 자신의 문장이 나옵니다:
> `RuntimeError: Distributed package doesn't have Gloo built in`.
>
> **이것을 발견한 것은 코드 리뷰가 아니라 측정입니다.** 처음에 쓴 docstring 은
> "ProcessGroupGloo 는 부재로 둔다" 였는데, 코드는 그렇지 않았습니다.

### 4.4 스텁으로 두는 것 — 선언된 멤버까지

`surface.json` 이 `_distributed_c10d` 에 선언한 타입 43 개 · 함수 29 개 중 위에서 구현하지 않은
것들은 **다른 모든 `_C` 서브모듈과 같은 방식**으로 만듭니다 (`_build_type`) — 선언된 멤버가
전부 `_Unimplemented` 로 붙습니다.

이것도 한 번 틀렸습니다. 처음에는 모듈 catch-all 이 대문자 이름에 빈 타입을 돌려줬고,
그러면 `Reducer()` 는 만들어지는데 `Reducer().prepare_for_backward(...)` 가
**`AttributeError`** 였습니다 — "그런 것 없다" 는 답인데 진실은 "안 만들었다" 입니다. 다른
질문에 대한 다른 답입니다. 지금은:

```
NotImplementedError: not implemented in torch._C shim: Reducer.prepare_for_backward
```

---

## 5. `local` 백엔드 — 스택의 세 번째 칸

`DESIGN.md` §11.1 의 스택대로 배치했습니다.

```
torchnative.nn.federated                      (아직 비어 있음)
  └ torch.distributed                         상류 벤더링 트리, 손대지 않음
      └ backends
          ├ 집합 통신 구현 : torch._C._distributed_c10d.ProcessGroupLocal
          └ 등록          : torchnative/distributed/__init__.py
              └ devices   CPU 만 (Metal · Vulkan · NPU 는 없음)
```

**구현이 `_C` 에 있고 등록이 `torchnative` 에 있는 이유**: 상류는 백엔드를 C++ 로 만들고
이 프로젝트가 그 절반을 대체합니다. 반면 `Backend.register_backend` 는
`distributed_c10d` 의 API 라 `_C` 임포트 시점에는 아직 존재하지 않습니다. 트리를 고치지 않고
쓸 수 있는 것이 상류가 이 목적으로 공개한 이 확장점입니다.

`fake` 가 아닌 이유: 상류 자신의 docstring 이 `FakeProcessGroup` 을
*"would produce wrong results for every collective"* 라고 적습니다. **틀린 결과는 거절보다
나쁩니다.**

```python
import torchnative.distributed          # 등록만 한다
torch.distributed.init_process_group(backend="local", rank=0, world_size=1,
                                     store=torch.distributed.HashStore())
```

`world_size != 1` 은 `ProcessGroupLocal.__init__` 이 **이름을 대고 거절**합니다 — 전송이
없으므로 그 이상은 구현되지 않았다고 말합니다.

---

## 6. 검증

### 6.1 상류 gloo 와의 대조 — 값이 전부 일치

**단언한 값이 아니라 잰 값입니다.** 같은 스크립트를 두 번 돌렸습니다 — 한 번은 벤더 트리 +
`backend="local"`, 한 번은 spike-venv 의 상류 torch 2.13.0 + `backend="gloo"`,
둘 다 `world_size=1`. 입력은 `[1.0, -2.0, 3.5]`.

**값을 내는 집합 통신 16 개가 전부 바이트 단위로 같았습니다.** 다른 것은 셋뿐이고,
셋 다 **양쪽이 실패하는 자리**입니다:

| | 상류 gloo | 이 빌드 |
|---|---|---|
| `all_reduce(PREMUL_SUM)` | `TypeError: incompatible function arguments` (pybind 인자 오류) | `NotImplementedError: ...allreduce with ReduceOp.PREMUL_SUM: ...` |
| `send(dst=1)` | `IndexError: vector` | `NotImplementedError: ...send: no rank 1 exists in a world of size 1` |
| `recv(src=1)` | `IndexError: vector` | `NotImplementedError: ...recv: ...` |

즉 상류도 못 하고 우리도 못 하는데, **우리 쪽이 무엇이 없는지 말합니다.**

| 집합 통신 | world 1 결과 | 성격 |
|---|---|---|
| `all_reduce` SUM · AVG · PRODUCT · MIN · MAX | `[1.0, -2.0, 3.5]` | 항등 |
| `broadcast` · `reduce` | `[1.0, -2.0, 3.5]` | 무연산 |
| `barrier` | `None` | 무연산 (참가자 1 인 배리어는 도달 즉시 만족) |
| `all_gather` · `gather` | `[[1.0, -2.0, 3.5]]` | **복사** |
| `all_gather_single` · `scatter` · `reduce_scatter` · `reduce_scatter_single` · `all_to_all_single` | `[1.0, -2.0, 3.5]` | **복사** |

마지막 두 줄이 중요합니다 — 이것들은 **0 으로 채운 버퍼**로 복사해 넣습니다. 아무것도 안 하는
구현이라면 리덕션은 같은 답을 내지만 이쪽은 `[0, 0, 0]` 이 됩니다. 그래서 목록에 넣었습니다.

### 6.2 게이트

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   exit 0   129 통과 (전 113, +16)
$PY tools/golden/compare.py                 exit 0   2268/2268, ops=97 (변화 없음)
$PY rust/torch_c/pytests/verify_schemas.py  exit 0   255/255 (전 233, +22)
```

`verify_schemas.py` 에 `_NON_ATEN_SCHEMA_TEXT` 대조를 붙였습니다 — 22 개 스키마를 상류 레지스트리
(`torch._C._jit_get_all_schemas()`)에서 다시 유도해 대조하고, **양방향**입니다: 표에 있는데 상류에
없어도, 상류에 있는데 표에 없어도 실패합니다 (aten 표는 의도적으로 부분집합이지만 이 표는 트리가
전부 요구하므로). `bootstrap.py` 를 임포트하지 않고 `ast` 로 읽습니다 — 이 스크립트는 상류 torch 로
돌아야 하고, 한 인터프리터에 둘을 넣는 것이 골든 하네스가 별도 프로세스를 쓰는 이유입니다.

### 6.3 실패할 수 있는지 확인했다

`CLAUDE.md` §5.5. 새 테스트가 진짜로 도는지 **일부러 깨서** 확인했습니다:

```
기대값을 [9,9,9] 로 바꿈   → FAIL test_world_size_one_collectives_agree_with_upstream_gloo
"OK" 를 "NOPE" 로 바꿈     → FAIL test_the_road_reaches_transformers
스키마 표에 인자 하나 추가 → FAIL c10d schemas _c10d_functional::wait_tensor
스키마 표에서 한 줄 삭제   → FAIL c10d schemas _c10d_functional::broadcast (역방향)
```

---

## 7. 다음 벽 — 분산이 아닙니다

> **열렸습니다 (2026-08-28). `docs/models/E2E_REAL.md` §4 를 보십시오.** `from_config` 로 만든
> `LlamaForCausalLM` 이 순전파를 돌고, 로짓이 상류와 2.24e-08 안에서 일치합니다. 이 이름
> 외에 실제로 필요했던 것은 `torch._C._is_tracing` 하나와 `cat` 의 legacy-empty 규칙뿐이었고,
> 다음 벽은 `from_pretrained` 의 `torch._C._set_default_dtype` — 즉 §3.4 가 남겨 둔 그
> 항목입니다.

```
torch._C.is_autocast_enabled
```

`transformers/utils/generic.py:250` 의 `maybe_autocast` 가 부릅니다. 도달 경로:

```
AutoModelForCausalLM.from_config(LlamaConfig(...))   OK  ← 진짜 transformers 모델이 만들어집니다
model(input_ids)
  └ modeling_llama.py:121   with maybe_autocast(device_type=..., enabled=False):
      └ generic.py:250      if torch.is_autocast_enabled(device_type) or enabled:
          NotImplementedError: not implemented in torch._C shim: torch._C.is_autocast_enabled
```

**`from_config` 가 통과한다는 것이 이 작업의 실제 성과입니다.** `DESIGN.md` §11.1 이
*"`from_pretrained` 와 실제 체크포인트 경로는 아직 한 번도 실행되지 않았습니다 — 그 둘은 이 벽
뒤에 있습니다"* 라고 적은 그 벽이 열렸고, 지금까지 **손으로 옮겨 적은 모델**로만 하던 골든 대조를
진짜 transformers 모델로 할 수 있는 자리까지 왔습니다. 다만 **순전파는 아직입니다** — autocast
표면이 남았고, 그것이 다음 작업 항목입니다.

---

## 8. 미확인 — 숨기지 않는 것

| # | 항목 | 상태 |
|---|---|---|
| 1 | `AutoModelForCausalLM.from_config` 로 만든 모델의 **순전파** | ~~**미통과.** §7 의 autocast 벽~~ **정정 (문서 감사, 2026-09): §7 자신의 correction 과 어긋남 — 통과함, `docs/models/E2E_REAL.md` §4.** 실측 재확인: `AutoModelForCausalLM.from_config(...)` 로 만든 `LlamaForCausalLM` 의 순전파가 오늘도 통과합니다 |
| 2 | `from_pretrained` / 실제 체크포인트 경로 | ~~**미시도.** 1 번 뒤에 있습니다~~ **정정 (문서 감사, 2026-09): 됩니다** — `torch._C._set_default_dtype` 도 지금 존재합니다(실측), `docs/design/DESIGN.md` §11.1 감사(round 1)가 실제 Hub 체크포인트 적재까지 확인했습니다 |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py is_autocast_enabled present -->
| 3 | `world_size >= 2` | **구현하지 않음.** `ProcessGroupLocal` 이 이름을 대고 거절합니다. 전송 계층이 없습니다 |
| 4 | `torchnative.nn.federated` (스택의 맨 위 칸) | **비어 있습니다.** 이번 작업은 그 아래 두 칸만 세웠습니다 |
| 5 | 장치 추상의 가속기 칸 (Metal · Vulkan · NPU) | **없음.** `local` 백엔드는 `devices=["cpu"]` 로만 등록합니다 |
| 6 | DDP 기계 (`Reducer` · `Logger` · `GradBucket` · `_broadcast_coalesced` …) | **거절.** 이름은 있고 부르면 실패합니다. 두 번째 랭크가 있어야 의미가 생깁니다 |
| 7 | `_c10d_functional` op 을 **호출**했을 때 | **커널 없음.** 스키마만 압니다 — `_aten_dispatch` 에 이 네임스페이스 커널이 없어 이름을 대고 거절합니다. 스키마를 아는 것과 돌릴 줄 아는 것은 다릅니다 |
| 8 | DTensor · FSDP 를 **쓰는** 것 | **미시도.** 임포트가 통과할 뿐입니다. 그 둘은 world_size >= 2 를 전제합니다 |
| 9 | `Tensor.dtype` 이 인터닝되지 않음 (`x.dtype is torch.float32` 가 상류는 True, 여기는 False) | **이번 작업 밖.** §3.4 테스트에서 마주쳐 `==` 로 적고 기록만 했습니다 |
| 10 | 안드로이드 · iOS 에서의 동작 | **미측정.** 호스트(darwin/arm64)에서만 돌렸습니다 |
| 11 | `_c10d_init` 을 켠 것이 `import torch` 시간에 주는 비용 | **미측정.** `torch.distributed` 145 모듈이 이제 무조건 올라옵니다 |
| 12 | **`is_mutable` 이 이번엔 반대 방향으로 항상 같은 답을 낸다** | **해결됨 (2026-08-28).** `docs/bindings/SCHEMA.md` — 아래 §8.1 은 기록으로 남긴다 |

### 8.1 `is_mutable` — 항상 참을 고쳤더니 항상 거짓이 됐다

> **해결됐습니다 (2026-08-28).** 원인은 이 절이 지목한 그대로였습니다 — 규칙이 아니라 입력이
> 없었던 것. `_get_schema` 가 벤더링된 `torchgen/.../native_functions.yaml` 을 읽어 스키마
> 텍스트를 실제로 채웁니다(상류 torch 를 런타임에 요구하지 않으므로 휠에서도 성립합니다).
>
> 조율 세션이 착지 검증에서 **구현된 117 개를 전부 상류와 대조**했습니다:
>
> ```
> is_mutable 불일치   0 / 117
> 자리표시자          0 / 117
> str(schema) 차이    0 / 117
> mutable 로 판정     12 개  (add_ clamp_ copy_ div_ fill_ ×2 index_put_
>                            masked_fill_ normal_ relu_ uniform_ zero_)
> ```
>
> **아래 본문이 일곱 개라고 적은 것은 그때 op 이 97 개였기 때문입니다.** 117 개가 된 지금은
> 열둘이고, 늘어난 다섯이 바로 이 절의 요지입니다 — 틀리는 방향이 "변경하지 않는다" 였으므로
> **op 을 늘릴수록 거짓말이 조용히 함께 자랐습니다.**
>
> 아래 본문은 지우지 않고 둡니다. 무엇이 왜 틀렸는지의 기록이 결론보다 오래 쓰입니다.

§3 이 `FunctionSchema.is_mutable` 을 메서드에서 프로퍼티로 고쳤습니다. **타입은 맞아졌는데
값이 여전히 상수입니다.** 조율 세션이 착지 검증 중 상류와 대조해서 찾았습니다:

```
상류   aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
       add.Tensor  is_mutable = False      add_.Tensor is_mutable = True
우리   aten::add_.Tensor(...) -> ...
       add.Tensor  is_mutable = False      add_.Tensor is_mutable = False   ← 틀림
```

구현은 옳습니다 — `any(a.alias_info.is_write for a in self.arguments)` 는 상류 규칙 그대로입니다.
**입력이 없습니다.** `_aten_implemented()` 의 97 개를 전부 훑으면 **자리표시자 스키마 97 개,
진짜 스키마 0 개**입니다. `(...)` 에는 인자가 없으니 `arguments` 가 비고, `any([])` 는 거짓입니다.

즉 술어가 **항상 거짓**입니다. 고치기 전의 **항상 참**과 같은 종류의 결함이고 방향만 반대입니다.

**영향 범위는 좁지만 방향이 나쁩니다.** 연산자 대부분은 함수형이라 거짓이 정답이고, 틀린 답을
받는 것은 자리표시자를 가진 in-place 일곱 개뿐입니다:

    add_  copy_  fill_  normal_  relu_  uniform_  zero_

그런데 **틀리는 방향이 "변경하지 않는다"** 입니다. `register_autograd` 가 이전에는 전부 거절했고
이제는 **거절해야 할 in-place 까지 받습니다.** 조용히 통과하는 쪽이라 다음에 발견하기 더 어렵습니다.

**고치려면 스키마 텍스트가 있어야 합니다.** `verify_schemas.py` 의 255/255 는 이것을 잡지 못합니다 —
파이썬 표면의 시그니처를 대조하지 `_schema` 텍스트를 대조하지 않기 때문입니다. 별도 작업입니다.

---

## 9. 재현

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cd /Volumes/macMini/worktrees/bw-dist
bash vendor/vendor_torch.sh
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-dist
PY=/Volumes/macMini/caches/spike-venv/bin/python
bash vendor/install_shim.sh
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib

PYTHON=$PY sh rust/torch_c/pytests/run.sh          # 129
$PY tools/golden/compare.py                        # 2268/2268 ops=97
$PY rust/torch_c/pytests/verify_schemas.py         # 255/255

# 이 문서의 판정
PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY -c \
  "from transformers import AutoModelForCausalLM; print('OK')"
```

상류 대조(§6.1)는 같은 스크립트를 `PROBE_BACKEND=local` + 벤더 트리로 한 번,
`PROBE_BACKEND=gloo` + 상류 torch 로 한 번 돌려 전사를 `diff` 합니다.
**`compare.py` 와 `verify_schemas.py` 는 벤더 트리를 `PYTHONPATH` 에 넣지 않고 돌립니다** —
넣으면 상류 torch 를 가려 기준선이 사라집니다.
