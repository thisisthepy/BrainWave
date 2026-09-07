# `_dispatch_library` 1549건 — 무엇이 걸려 있는가

IMPORT_TORCH.md §9.1 이 "때우고 넘어간 것 중 가장 큰 것"으로 지목한 항목의 크기를 잰 문서입니다.
`torch.library.impl(...)` 이 성공한 척하고 아무 일도 하지 않는 등록이 임포트 한 번에 1549건이라는
것은 이미 알려져 있었고, 여기서는 **그 1549건의 정체**와 **실제로 쓰이는 비율**, 그리고 §9.2 의
`_dispatch_has_kernel=True` 가 진짜 torch 앞에서 어떤 대가를 치르는지를 계측했습니다.

숫자는 전부 직접 센 것입니다. 셋을 썼습니다.

- **shim** — `TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/vendor /Volumes/macMini/caches/spike-venv/bin/python`
  (벤더링 트리 + `rust/torch_c` shim)
- **진짜 torch** — `/Volumes/macMini/caches/spike-venv/bin/python` (PYTHONPATH 없이. pip 로 설치된
  진짜 torch 2.13.0 + transformers 5.15.1)
- 계측 스크립트는 전부 `/tmp` 에 두었고, 이 저장소에는 아무것도 추가하지 않았습니다.

---

## 0. 답부터

| 질문 | 답 |
|---|---|
| 1549건의 정체 | `impl` 1416 + `define` 131 + 기타 catch-all 2. `fallback` 은 **0건** (임포트 중 한 번도 안 불림) |
| 어느 라이브러리가 많은가 | `aten` 883, `prims` 630 이 전체의 98% |
| eager 순전파+`generate` 중 실제로 디스패치되는 것 | **0 / 1904** (진짜 torch, 같은 성격의 등록을 같은 방식으로 세었을 때) |
| `_dispatch_has_kernel=True` 의 실측 대가 | 진짜 torch 에서 강제로 켜보면 분해 테이블에 **TorchScript 잔재 251개**가 추가로 섞여 들어오고, torch 자신의 `activate_meta()` 가 그것들을 검증하다가 **`RuntimeError: operator ... does not exist` 로 크래시**한다 (측정) |
| shim 이 그 크래시를 피하는 이유 | `_dispatch_has_kernel_for_dispatch_key` 를 **항상 False, 절대 안 던지게** 짝지어 놓았기 때문 — 이 둘은 세트다 |
| DESIGN §2 가 의존한다는 "973항목 분해 테이블" | **지금 재현되지 않는다.** `core_aten_decompositions()` 는 shim 에서 다른 이유로 크래시하고, 임포트 직후의 `decomposition_table` 은 592건으로, 973 과도 진짜 torch 의 1097 과도 다르다 |
| 지금 고쳐야 하는가 | 아니오 — §5 참고. 이 1549건은 **eager 순전파에서는 진짜 torch 도 안 쓴다.** 지금 급한 것은 다른 벽(§5, `_log_api_usage_once`)이다 |

> **Correction (문서 감사, 2026-09):** both "미룰 수 없는 것" items in §5 and the crash in §4's
> last row are closed, and `from_config` no longer stops at `_log_api_usage_once`. **How checked:**
> `hasattr(torch._C, '_log_api_usage_once')` and `hasattr(torch._C, '_dispatch_get_registrations_for_dispatch_key')`
> are both `True` against today's shim; `core_aten_decompositions()` no longer crashes and returns
> 417 entries (real torch: 940 — a real gap, but a count, not a crash);
> `decomposition_table` is 1008 today (was 592 when this document measured it; real torch is
> unchanged at 1097, confirmed live). `docs/graph/DECOMP.md` §3 (round 2 of this audit) documents the
> `_jit_get_operation`/`overload_names` fix directly, citing this document by name in its own code
> comment (`bootstrap.py:1223`, "the reason it exists is docs/graph/DECOMP.md §3"). `from_config`
> succeeds end-to-end today (independently confirmed while auditing `docs/models/FROM_CONFIG.md`, this
> round). "973" itself is still not reproduced or explained — that part of §0's row is unresolved,
> not fixed, and is left as reported.
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _log_api_usage_once present -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _dispatch_get_registrations_for_dispatch_key present -->

---

## 1. 1549건의 분류

`torch._C._shim_registrations` 는 튜플의 리스트입니다. 첫 필드가 종류(`kind`), 둘째가 네임스페이스,
셋째가 op 이름, 넷째가 종류별로 다른 부가 정보(`define` 은 스키마 문자열, `impl` 은 dispatch key)
입니다.

### 종류별

| kind | 건수 |
|---|---|
| `impl` | 1416 |
| `define` | 131 |
| `register_ad_inplace_or_view_fallback` (다섯 함수 밖의 catch-all) | 2 |
| `fallback` | **0** |
| `impl_with_aoti_compile` | **0** |

**`fallback` 은 이번 `import torch` 에서 단 한 번도 불리지 않았습니다.** `torch/library.py:244`
의 `Library.fallback` 은 벤더링 트리 안에서 임포트 시점에는 아무도 쓰지 않는다는 뜻입니다(런타임에
쓰일 수는 있습니다 — 미확인, §7).

### 라이브러리(네임스페이스)별

| 라이브러리 | 건수 | 비율 | 출처 |
|---|---:|---:|---|
| `aten` | 883 | 57.0% | `torch/_meta_registrations.py` (거의 전부), `_native/registry.py`, `_refs/linalg` (MPS 1건), CUDA flash-attention 모듈 (1건) |
| `prims` | 630 | 40.7% | `torch/_prims/__init__.py` — PrimTorch 원시 연산자 정의+구현 |
| `onednn` | 11 | 0.7% | `_meta_registrations.py` |
| `debug_mode_ops` | 6 | 0.4% | `torch.utils._debug_mode` 계열 |
| `rngprims` | 6 | 0.4% | `_prims/rng_prims.py` |
| `export` | 3 | 0.2% | `torch/export/custom_ops.py` |
| `_native` | 3 | 0.2% | `torch/_native/registry.py` |
| `debugprims` | 2 | 0.1% | `_meta_registrations.py` |
| `mkldnn` | 2 | 0.1% | `_meta_registrations.py` |
| `quantized` | 2 | 0.1% | `_meta_registrations.py` |
| `mkl` | 1 | 0.1% | `_meta_registrations.py` |
| **합계** | **1549** | 100% | |

`aten` + `prims` 둘이 1513건(97.7%)입니다. 나머지 9개 라이브러리는 전부 합쳐도 36건입니다.

### dispatch key 별 (`impl` 1416건 안에서)

| (라이브러리, key) | 건수 |
|---|---:|
| `aten` / `Meta` | 881 |
| `prims` / `CompositeExplicitAutograd` | 126 |
| `prims` / `Autograd` | 126 |
| `prims` / `Meta` | 126 |
| `prims` / `BackendSelect` | 126 |
| `onednn` / `Meta` | 11 |
| `mkldnn` / `Meta` | 2 |
| `quantized` / `Meta` | 2 |
| 나머지 (export/debug_mode_ops/rngprims/debugprims/`_native`/`mkl`/`aten`-MPS/`aten`-CUDA) | 각 1~2건 |

**`aten` 은 883건이 전부 `Meta` 키(881) 아니면 MPS(1)·CUDA(1) 뿐이고, `CPU` 키 등록은 하나도
없습니다.** 이 저장소가 타깃으로 삼는 CPU 추론 경로에서 실제로 쓰이는 백엔드 키(`CPU`)로 파이썬이
등록하는 것이 원래 없다는 뜻입니다 — CPU 커널은 C++ 로 컴파일되어 있고 `torch.library` 를 거치지
않습니다. `aten` 라이브러리 등록은 전부 "이 op 을 메타 텐서(shape-only)로 어떻게 다루는가"를
답하는 것이지, "이 op 을 CPU 에서 어떻게 계산하는가"가 아닙니다.

**`prims` 는 정의된 126개 원시 연산자마다 4개 dispatch key 를 전부 구현하는 형태입니다**
(126 × 4 = 504, `define` 126 + `impl` 504 = 630). PrimTorch 는 자기 op 을 Python 으로 직접
완전히 구현하는 설계이므로 `CompositeExplicitAutograd`(실제 계산) 까지 포함합니다.

---

## 2. 실제로 디스패치되는가 — eager 순전파+`generate` 에서는 0건

### 측정 방법

`torch.library.Library.impl` (shim 이 대신하는 `torch._C._dispatch_library` 의 한 단계 위,
벤더링 트리가 그대로 쓰는 진짜 파이썬 API)을 파이썬 임포트 후크로 가로채, 등록되는 함수마다
호출 횟수를 세는 래퍼로 감쌌습니다. `torch.library` 모듈이 실행되는 시점에 `Library.impl` 을
패치하므로, 그 뒤에 `_meta_registrations.py`·`_decomp/`·`_prims/`·`_refs/` 가 등록하는 모든
콜백이 계측됩니다. **진짜 torch 2.13.0 에서** 돌렸습니다(shim 이 아닙니다 — shim 은 `from_config`
조차 도달하지 못합니다, §5).

```
소형 Llama: hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=128, vocab_size=100  (CORE_ATEN.md 와 동일 스펙)
model.eval(); with torch.no_grad(): model(input_ids); model.generate(max_new_tokens=4, do_sample=False)
```

### 결과

```
등록 가로챈 횟수 (Library.impl 호출 수):     1904
순전파+generate 중 실제로 불린 등록 함수 수:   0
```

**한 번도 안 불립니다.** 계측이 죽은 코드가 아니라는 것은 직접 확인했습니다 — 같은 모델에
`torch.export.export(model, (input_ids,))` 를 걸어보면(transformers 의 `DynamicCache` 가
pytree 미등록이라 끝까지 가지는 못하지만, 실패 지점까지는 shape/meta 추론이 실제로 돕니다)
`prims.broadcast_in_dim`·`prims.transpose`·`prims.cat`·`aten.arange(Meta)` 등 **18종 72회**가
호출됩니다. 즉 계측기는 살아 있고, "0" 은 진짜 0입니다.

### 왜 0인가

`aten` 라이브러리의 883개 등록은 전부 `Meta`(shape-only) 아니면 CUDA/MPS 키이고, `prims` 의 630개는
`torch.ops.prims.*` 를 직접 부르는 코드(`torch._refs`, 몇몇 백워드 공식, 익스포트/컴파일 트레이싱)
에서만 걸립니다. **평범한 CPU eager 순전파는 둘 중 어느 것도 거치지 않습니다** — 실제 수치 계산은
빌드 시점에 링크된 C++ 커널이 담당하고, 그 커널은 `Library.impl` 을 거치지 않습니다. `Meta` 키는
FakeTensor·`torch.export`·`torch.compile` 처럼 "실제 텐서 없이 shape/dtype 만 알고 싶다"는 요청에만
응답합니다.

**즉 이 저장소가 지금 하려는 것(eager CPU 순전파, autograd 없음, compile 없음)의 범위 안에서는,
진짜 torch 도 이 1549건에 상응하는 등록을 쓰지 않습니다.** shim 이 그것들을 기록만 하고 버리는 것이
지금 당장 계산 결과를 틀리게 만들지는 않습니다.

---

## 3. `_dispatch_has_kernel=True` 의 실측 대가

### 3.1 shim 자신의 결정: 대가는 "몇 개 항목", 정말인가

`bootstrap.py:1060` 은 `_dispatch_has_kernel` 을 항상 `True` 로 고정하면서 주석에 "거짓의 값은
레지스트리 항목 몇 개" 라고 적었습니다. **더 좁은 질문(`_dispatch_has_kernel_for_dispatch_key`,
`_dispatch_has_kernel_for_any_dispatch_key`)은 반대로 항상 `False`(그리고 절대 예외를 던지지
않음)로 고정되어 있습니다(`bootstrap.py:1063-1064`).** 이 둘은 세트로 읽어야 합니다 — 아래가 그
이유입니다.

### 3.2 진짜 torch 에 강제로 같은 거짓말을 시켜본다

`torch._decomp/__init__.py:90` 의 `_add_op_to_registry` 가 `_dispatch_has_kernel` 로 TorchScript
잔재 오버로드(`aten.add.float_int` 류)를 거릅니다. **진짜 torch 를 import 하기 직전에
`torch._C._dispatch_has_kernel` 을 shim 과 동일하게 "항상 True" 로 패치하고 나머지는 그대로
둔** 상태로 다시 임포트시켜 봤습니다(방법: `torch._decomp.decompositions` 모듈이 실행되기 직전을
가로채는 `importlib` 파인더).

```
정상 (진짜 has_kernel):        decomposition_table = 1097
강제 True (has_kernel 만):     decomposition_table = 1348   (+251)
```

**251개가 추가로 들어옵니다.** 이름을 보면 정확히 그 "TorchScript 잔재"입니다.

```
aten::ldexp                              (오버로드가 아니라 이름 전체가 없음)
aten::_unsafe_index.Tensor_hacked_twin
aten::_unsafe_index_put.hacked_twin
aten::index_put_.hacked_twin
aten::acos.int / .float / .complex / .Scalar     (Scalar 오버로드 4종)
aten::acosh.int / .float / .complex / .Scalar
aten::asin.int / .float / .complex / .Scalar
aten::asinh.int / .float / .complex / .Scalar
... (그리고 같은 패턴의 다른 삼각함수·기타 op)
```

**대가가 "몇 개 항목"이라는 shim 의 주석은 정확합니다 — 딱 251개, 만들어 낸 숫자가 아니라 실측입니다.**

### 3.3 하지만 그 251개는 진짜 torch 앞에서 무해하지 않다

`torch/_meta_registrations.py::activate_meta()` 는 `decomposition_table` 전체를 순회하며
`op_overload.py_impl(DispatchKey.Meta)(fn)` 을 걸고, 그 뒤 **자기 자신이**
`torch._C._dispatch_has_kernel_for_dispatch_key(op_overload.name(), "CompositeImplicitAutograd")`
로 그 op 이 진짜 디스패처에 있는지 재확인합니다. 이 재확인용 C++ 함수는 shim 의 것과 달리
**모르는 op 이름에 `False` 를 돌려주지 않고 `RuntimeError` 를 던집니다.**

`has_kernel` 만 강제로 True 로 켠 상태로 `activate_meta()` 까지 실행시키면:

```
RuntimeError: operator aten::ldexp does not exist
```

**바로 이 지점에서 죽습니다.** 예외를 잡아 세어보면 251개 잔재 전부가 이 재확인을 통과하지
못합니다(502회 감지 — `activate_meta()` 내부 순회 구조상 항목마다 두 번씩 걸림).

### 3.4 shim 이 살아남는 이유는 우연이 아니라 짝을 맞췄기 때문이다

shim 의 `_dispatch_has_kernel_for_dispatch_key` 는 진짜 이름 검증을 전혀 하지 않는 상수 함수라서
`aten::ldexp` 같은 이름에도 그냥 `False` 를 돌려줄 뿐, **진짜 torch 처럼 "그런 이름은 존재하지
않는다"고 판단해서 던지는 것이 아닙니다.** 그래서 §3.3 의 크래시가 shim 에서는 일어나지 않습니다.

**결론: `_dispatch_has_kernel=True` 라는 선택 하나만 놓고 보면 실측 대가는 251개의 무해한
잔재입니다. 하지만 그 무해함은 `_dispatch_has_kernel_for_dispatch_key` 가 절대 이름을 검증하지
않는다는 별개의 결정에 얹혀 있는 것이고, 진짜 torch 코드(`activate_meta`)가 그 조합을 실제로
테스트하면 깨집니다.** 나중에 "더 사실적으로" 만들겠다고 `_dispatch_has_kernel_for_dispatch_key`
쪽만 손대면(예: 실제 오버로드 이름표와 대조하도록), `_dispatch_has_kernel` 이 만들어 둔 251개의
가짜 항목이 §3.3 과 같은 방식으로 터질 잠재 지점이 됩니다.

---

## 4. 분해 테이블과 "973" — 지금은 재현되지 않는다

IMPORT_TORCH.md §0 은 "등록된 분해: 973" 이라고 적었습니다. 이번에 직접 셌더니 다릅니다.

| 측정 | 값 |
|---|---:|
| 진짜 torch, `import torch` 직후 `len(torch._decomp.decomposition_table)` | **1097** |
| 진짜 torch, `core_aten_decompositions()` | **940** (CORE_ATEN.md 와 일치) |
| shim, `import torch` 직후 `len(torch._decomp.decomposition_table)` | **592** |
| shim, `core_aten_decompositions()` 호출 | **크래시** — `NotImplementedError: torch._C._dispatch_get_registrations_for_dispatch_key` |

**"973" 은 이번 측정으로 재현되지 않았습니다.** 어디서 나온 숫자인지 다른 문서에서도 찾지
못했습니다(`grep` 으로 IMPORT_TORCH.md 바깥에는 등장하지 않습니다) — 커밋 사이에 코드가 바뀌어
값이 이동했거나, 다른 산출 경로로 얻은 숫자로 보이지만 **재구성하지 못했으므로 미확인으로 남깁니다.**

### 592 는 왜 1097 보다 훨씬 작은가 — `_dispatch_has_kernel` 이 원인이 아니다

`_dispatch_has_kernel=True` 는 항목을 **더 통과시키는** 방향입니다. 그런데 shim 의 표는 진짜
torch 보다 작습니다. 두 표의 op 이름 집합을 직접 대조했습니다.

```
진짜 torch 에는 있고 shim 에는 없는 것:   715개  (대부분 .out / .Scalar / 인플레이스 오버로드)
shim 에만 있고 진짜 torch 에는 없는 것:   210개  (전부 실재하지 않는 가짜 ".default")
겹치는 것:                                 382개
382 + 210 = 592  (shim 합계와 일치)
```

**원인은 `_dispatch_has_kernel` 이 아니라 `bootstrap.py` 의 `_jit_get_operation` 이 모든 op 에
대해 `overload_names` 를 무조건 `["default"]` 하나로만 돌려주는 것입니다**
(`rust/torch_c/src/bootstrap.py:1247-1253`). `register_decomposition(aten.foo)` 처럼 op
패킷 전체를 등록하는 코드는 `OpOverloadPacket.op_overloads()` 를 호출해 실제 오버로드 수만큼
등록하는데, 이 shim 아래에서는 그 함수가 항상 정확히 1개(그리고 그 이름이 실재하는지도 확인 안
된 `"default"`)만 돌려줍니다. 그 결과:

- 진짜로 `.out`·`.Scalar`·인플레이스(`_`) 오버로드를 갖는 715개 op 이 등록되지 않고,
- `aten.abs_`·`aten.__iand__` 처럼 진짜로는 `.default` 오버로드가 없는 210개 op 에 **존재하지
  않는 `.default` 항목이 생겨** 등록됩니다.

**§9.3("op 이름 조회가 실패하지 않는다")과 결이 같은 문제이지만 다른 증상입니다.** §9.3 은
"모르는 이름을 물어도 참을 돌려준다"였고, 여기서는 한 걸음 더 나아가 "모든 op 이 오버로드를
정확히 하나만 가진 척한다"가 분해 테이블 크기 자체를 절반 가까이 깎아 먹습니다.

### `core_aten_decompositions()` 자체가 crash 하는 것은 완전히 다른 문제

`torch/_export/utils.py:1362` 의 `_collect_all_valid_cia_ops_for_aten_namespace()` 가
`torch._C._dispatch_get_registrations_for_dispatch_key("CompositeImplicitAutograd")` 를
부르는데, 이 이름은 §8 표(45개 벽)에도, `_shim_registrations` 목록에도 없는 **아예 구현되지 않은
함수**입니다. `_dispatch_has_kernel` 과는 무관한, 마흔여섯 번째 벽입니다.

---

## 5. 판단 — 지금 고쳐야 하는가

**아니오, 지금은 아닙니다.** 근거는 세 가지입니다.

1. **§2 가 보여주듯, 이 프로젝트가 지금 노리는 워크로드(eager CPU 순전파, autograd 없음, compile
   없음)에서는 진짜 torch 자신도 이 1549건에 상응하는 등록을 쓰지 않습니다.** 등록을 버리는
   것이 지금 계산 결과를 틀리게 만든다는 증거가 없습니다.
2. **`from_config` 조차 아직 도달하지 못합니다.** 오늘 다시 재현해 보니 실패 지점이 이전
   기록(transformers `GenerationMixin` 지연 임포트)보다도 더 앞으로 옮겨져 있었습니다 —
   `LlamaConfig` 임포트 체인이 `nn.Module.__init__` 을 거치는 순간
   `torch._C._log_api_usage_once` 가 없다며 `NotImplementedError` 로 죽습니다. **순전파에
   도달하기 전에 넘어야 할 벽이 이미 하나 더 있고, 그것은 `_dispatch_library` 와 무관합니다.**
   이 벽을 먼저 넘지 못하면 §2 의 "0건" 조차 shim 위에서는 검증할 수 없습니다(진짜 torch 로
   대신 잰 이유입니다).
   > **정정 (문서 감사, 2026-09):** 닫혔다 — `torch._C._log_api_usage_once` 는 오늘 존재하고
   > (`bootstrap.py:4777`), `from_config` 는 shim 위에서 직접 성공한다(§0 위 정정, 그리고
   > `docs/models/FROM_CONFIG.md` 감사에서 독립적으로 재확인).
3. **§3 이 보여주듯, `_dispatch_has_kernel=True` 홀로는 실제로 값싼 거짓말입니다** — 진짜
   torch 도 그 251개를 "무해한 잔재"로 취급하도록 설계돼 있습니다(`activate_meta()` 가 그것을
   걸러내려고 존재). 문제는 그 짝인 `_dispatch_has_kernel_for_dispatch_key` 를 나중에 따로
   손대는 경우입니다.

**미룰 수 없는 것은 따로 있습니다.**

- **`_jit_get_operation` 의 `overload_names=["default"]` 고정 (§4).** 이것은 `_dispatch_library`
  와 별개 항목이면서, `torch._decomp.core_aten_decompositions()` 를 아예 못 쓰게 만드는 더 앞선
  장애물입니다. DESIGN §2 가 "Core ATen 밖 롱테일은 자동 분해로 처리한다"고 적었는데, **그 자동
  분해를 얻는 표준 API 자체가 지금은 크래시합니다.** `torch.export`/`torch.compile` 경로를 열
  계획이 있다면, `_dispatch_library` 보다 이쪽이 먼저 막힙니다.
- **`_dispatch_get_registrations_for_dispatch_key` 미구현.** `core_aten_decompositions()` 가
  기본으로 타는 경로이므로, "분해 테이블을 실제로 얻어 쓴다"가 다음 목표가 되는 순간 바로
  마주칩니다.

**요약하면:** `_dispatch_library` 를 기록만 하고 버리는 결정 자체는, 지금 이 프로젝트가 재는
워크로드 안에서는 대가가 없다는 것이 이번 계측의 결론입니다. 대신 순전파에 이르는 길 위에 있는
더 앞선 두 벽(`_log_api_usage_once`, 오버로드 이름 고정)이 실제 병목입니다. `_dispatch_library`
는 **`torch.export`/`torch.compile`/backward 를 실제로 켜는 시점**에 다시 열어보면 됩니다 —
그 전까지는 `_C._shim_registrations` 로 크기만 지켜보는 것으로 충분합니다.

---

## 6. 재현

```bash
cd /Volumes/macMini/thisisthepy/torchnative
PY=/Volumes/macMini/caches/spike-venv/bin/python

# 1549건 자체 (shim)
TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/vendor $PY -c \
  "import torch; print(len(torch._C._shim_registrations))"

# 진짜 torch, 분해 테이블 베이스라인
$PY -c "import torch; from torch._decomp import decomposition_table, core_aten_decompositions; \
  print(len(decomposition_table), len(core_aten_decompositions()))"

# shim, 분해 테이블 (core_aten_decompositions 는 여기서 크래시함)
TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/vendor $PY -c \
  "import torch; from torch._decomp import decomposition_table; print(len(decomposition_table))"
```

§2·§3 의 계측 스크립트는 `/tmp/probe_dispatch_usage2.py`(디스패치 사용 여부),
`/tmp/probe_haskernel.py`(`_dispatch_has_kernel` 강제 True 실험)에 남겨 두었습니다. 둘 다
`importlib.abc.MetaPathFinder` 로 특정 모듈이 실행되기 직전에 `torch._C`/`torch.library` 의
함수를 몇 줄 감싸는 방식이라, **진짜 torch 의 site-packages 파일 자체는 건드리지 않습니다**
(한 번 직접 편집을 시도했다가 권한 문제로 되돌리고 이 방식으로 바꿨습니다).

---

## 7. 미확인 항목

| 항목 | 상태 |
|---|---|
| "973" 이 어디서 나온 숫자인지 | 미확인 — 이번 측정(1097 진짜 / 592 shim)과도, 다른 문서(CORE_ATEN.md 의 940)와도 안 맞음 |
| `Autograd` 키 등록(126건, `prims`)이 실제 backward 를 켰을 때 쓰이는지 | 미측정 — DESIGN §3 stage 0 이 backward 를 범위 밖에 두고 있어 아직 켤 수 없음 |
| `torch.compile`(dynamo+inductor) 경로에서의 사용 비율 | 미측정 — 순전파 자체가 아직 안 되므로 그 앞 단계를 잴 수 없음 |
| shim 의 `Library.impl` 진입점 수(1549 중 impl 1416)와 진짜 torch 를 같은 방식으로 잰 1904 의 정확한 대응 | 미해결 — 계측 계층이 다름(shim 은 `_dispatch_library` 저수준, 이번 §2 계측은 `torch.library.Library.impl` 고수준). 같은 것을 세었다는 보장이 없어 직접 비교하지 않았음 |
| `fallback()` 이 런타임(임포트 이후)에 불리는 경로가 있는지 | 미측정 — 이번 계측은 `import torch` 시점만 봄 |
| 양자화·분산·oneDNN 융합처럼 이 모델에 없는 경로가 실제로 이 등록들에 의존하는지 | 미측정 — 소형 Llama 에는 해당 op 이 등장하지 않음 |
| `_log_api_usage_once` 벽을 넘으면 그 다음에 바로 순전파에 닿는지, 아니면 그 사이에 또 다른 벽이 있는지 | 미확인 — 오늘 처음 마주친 벽이라 그 너머는 보지 못함 |
