# `_C._dynamo` — 다음 임계 경로의 실제 크기

OVERLOAD.md §8 이 멈춘 자리(`torch._C._dynamo.eval_frame.set_guard_error_hook`,
`torch/_dynamo/guards.py:5457`, 모듈 스코프)를 이어서 잰 기록입니다. **코드는 고치지 않았습니다.**
`git status --short`로 확인 가능한 변경 범위는 이 문서 하나뿐입니다.

판정은 두 가지로 나눕니다 — 표면 크기는 `dir()` 로 직접 센 것, 실제 요구사항은 계측기가 잡은
것. 계측기가 진짜로 접근을 잡는지 자체 검증(self-check)을 먼저 통과시킨 뒤에 측정했습니다.

환경: `/Volumes/macMini/caches/spike-venv/bin/python`, torch 2.13.0, transformers 5.15.1.

---

## 0. 한눈에

| | 값 |
|---|---|
| `_C._dynamo` 전체 표면 (직접 셈, `dir()` 재귀) | **137개** — 서브모듈 4개(`eval_frame` 37 · `guards` 86 · `utils` 1 · `compiled_autograd` 5) + 최상위 직속 8개 |
| 기존 `surface.json`이 잡고 있는 것 | **8개** (최상위만, 서브모듈 미탐지 — §5) |
| `GenerationMixin` 임포트가 실제로 접근하는 이름 | **52개** (137개 중 38%) |
| 그중 실제로 **호출**되는 이름 | **2개** — `set_guard_error_hook`, `set_code_exec_strategy` |
| 모델 생성 + forward + `generate()` 단계에서 추가로 접근되는 이름 | **0개** (import 시점에서 전부 끝남) |
| `.pyi` 스텁에 이미 선언되어 있는가 (접근된 52개 기준) | **52/52** (완전) |
| off-switch(이름 부재로 끄는 방법) 가능성 | **없음** — 전부 무조건 임포트문, `try/except` 없음 |

**결론을 먼저 적습니다.** "`_C._dynamo` 서브시스템 전체를 구현해야 한다"는 처음 우려보다 범위가
훨씬 좁습니다. `GenerationMixin` 을 통과시키는 데 필요한 것은 137개 중 52개이고, 그중 **동작까지
필요한 것은 2개뿐**입니다. 나머지 50개는 임포트 문이 성공하기만 하면 되는 자리표입니다
(§3). `torch.compile` 을 쓰지 않는다는 전제가 성립하는 한 이 결론은 안정적입니다 — 실제로
forward·`generate()` 를 돌려도 추가 접근이 0이었습니다(§4).

---

## 1. 표면 — `dir()` 로 직접 센 것

```
>>> torch._C._dynamo
서브모듈 4개: eval_frame(37) · guards(86) · utils(1) · compiled_autograd(5)
최상위 직속 8개: PyMappingSlots · PyNumberSlots · PySequenceSlots · PyTypeSlots ·
                get_type_slots · has_slot · is_valid_var_name · strip_function_call
합계: 4 + 8 + 37 + 86 + 1 + 5 = 137 (서브모듈 자체 4개를 이름으로 셀지는 관점 차이 —
     "속성"으로 보면 141, "말단 심볼"로 보면 137. 이 문서는 후자를 씁니다)
```

`eval_frame` 은 프레임 평가 후킹(`set_eval_frame`), `guards` 는 가드 트리 빌드/평가 C++ 타입
(`GuardManager`, `RootGuardManager`, 각종 `*GuardAccessor`), `utils` 는 이름 하나
(`is_instancemethod`), `compiled_autograd` 는 컴파일된 backward 관련 5개입니다. `eval_frame`
말고도 서브시스템 표면이 크다는 배경 설명(OVERLOAD.md §8)이 정확했습니다 — 오히려 `guards` 가
`eval_frame` 보다 2배 이상 큽니다.

---

## 2. 계측 방법과 작동 증거

### 2.1 왜 검증부터 했는가

이 프로젝트에서 계측 방법 자체가 틀렸던 전례가 넷 있다고 지시받았습니다(C_SURFACE.md §5의
"접근 99.7%" 오염, IMPORT_WALLS.md의 `grep -q MODEL_OK` 오탐 등). 그래서 **측정 전에 계측기가
실제로 접근을 잡는지부터 자체 검증**했습니다.

### 2.2 방법

`torch._C._dynamo` 의 4개 서브모듈과 최상위 모듈 자체를, 각각 `types.ModuleType` 프록시로
교체했습니다. 설치 지점을 **둘 다** 바꿨습니다 — `sys.modules["torch._C._dynamo.eval_frame"]`
(→ `from torch._C._dynamo.eval_frame import X` 스타일을 잡음)과 부모 모듈의 속성
`torch._C._dynamo.eval_frame = proxy`(→ `torch._C._dynamo.eval_frame.X` 스타일을 잡음) 둘 다.
프록시의 `__getattr__` 에서 이름과 스테이지를 기록하고, 값이 콜러블이면 호출까지 감싸는
wrapper 를 돌려줘 실제 호출도 별도로 기록합니다.

### 2.3 자체 검증 — 실행 로그

```
$ /Volumes/macMini/caches/spike-venv/bin/python /tmp/dynamo_probe2.py
SELF-CHECK PASSED (top-level + submodule, both getattr styles, call tracking, negative control).
```

자체 검증이 확인한 것 4가지:

1. `from torch._C._dynamo.eval_frame import set_guard_error_hook` 스타일(대부분의 실제 코드가
   쓰는 형태)이 잡히는가 — 잡힘.
2. `torch._C._dynamo.eval_frame.set_guard_error_hook` 속성 접근 스타일이 **별개로** 잡히는가 —
   잡힘 (import 시 1회, 이후 속성 접근 시 카운트가 늘어남을 직접 확인).
3. 실제 호출(`sgeh(dummy)`)이 `CALLS` 에 기록되는가 — 기록됨.
4. **음성 대조군**: 건드리지 않은 이름(`has_slot`, `profile_guard_manager`)이 0으로 남는가 —
   남음. (이게 없으면 "전부 접근됨"처럼 보이는 §5의 함정에 다시 빠집니다.)

같은 프로세스에서 `import torch` 는 이미 끝난 상태(즉 `_C._dynamo` 서브모듈들은 이미 존재)에서
프록시를 설치했습니다 — 이는 §5에서 확인하듯 `import torch` 자체는 `torch._dynamo` 파이썬
패키지를 아직 끌어오지 않으므로, 이후 `GenerationMixin` 경로에서 벌어지는 최초의 전체 임포트를
그대로 잡을 수 있습니다.

---

## 3. `GenerationMixin` 이 실제로 요구하는 것

### 3.1 측정

```python
from transformers.generation.utils import GenerationMixin
```

이 한 줄로 `torch._dynamo` 파이썬 패키지(93개 모듈, IMPORT_WALLS.md 3차)가 최초로 통째로
임포트됩니다 (`masking_utils.py:42` 의 `from torch._dynamo._trace_wrapped_higher_order_op import
...`, `torch >= 2.6` 이면 무조건 실행 — §6). 그 안에서 `_C._dynamo` 의 이름 **52개**가
`getattr` 됩니다.

- `_C._dynamo` 최상위: `PyMappingSlots` · `PyNumberSlots` · `PySequenceSlots` · `PyTypeSlots` ·
  `get_type_slots` · `has_slot` · `strip_function_call` (7개) — 출처
  `torch/_dynamo/variables/object_protocol.py:20` 의 일괄 임포트,
  `torch/_dynamo/guards.py:798` 의 `strip_function_call = torch._C._dynamo.strip_function_call`.
  (`is_valid_var_name` 은 표면에 있지만 **이 경로에서는 접근되지 않았습니다** — 사용처가
  `guards.py:2073` 인데 메서드 본문 안이라 실제 컴파일이 일어날 때만 실행됩니다.)
- `eval_frame` 17개: `torch/_dynamo/eval_frame.py:59-70` 의 임포트 블록(11개) +
  `torch/_dynamo/guards.py:49` 의 `code_framelocals_names`(1개) +
  `torch/_dynamo/bytecode_debugger.py:74` 의 `NULL_STACK_VALUE` 등 산발적 임포트(5개).
- `guards` 28개: `torch/_dynamo/guards.py:50-77` 의 단일 임포트 블록에서 26개 +
  `torch/_dynamo/convert_frame.py:59` 의 `GlobalStateGuard`(1개) +
  `compute_overlapping_tensors`(1개, 사용처 미추적).
- `utils` · `compiled_autograd` — **0개.** 이 경로에서 전혀 건드리지 않습니다.

### 3.2 그중 실제로 호출되는 것 — 2개뿐

```
[A_import_GenerationMixin] eval_frame.set_code_exec_strategy(
    args=(<code object nonrecursive_disable_wrapper ...>,), kwargs={}) -> None
[A_import_GenerationMixin] eval_frame.set_guard_error_hook(
    args=(<function guard_error_hook at 0x...>,), kwargs={}) -> None
```

두 호출 다 **모듈 스코프의 부기(bookkeeping) 등록**이지, 결과를 쓰는 계산이 아닙니다.

| 이름 | 호출 지점 | 인자 | 반환값 사용 | 성격 |
|---|---|---|---|---|
| `set_guard_error_hook` | `torch/_dynamo/guards.py:5457`, 모듈 스코프 | 콜백 함수 1개 | **아니오** (대입 없는 문장) | 가드 평가 중 예외가 났을 때 부를 훅을 등록. 훅은 **eval-frame 후킹이 실제로 설치돼 있을 때만** 트리거됨 — `torch.compile` 을 안 쓰면 절대 안 불림 |
| `set_code_exec_strategy` | `torch/_dynamo/decorators.py:125`, 모듈 스코프 (`skip_code()` 경유) | 코드 객체 1개 + `FrameExecStrategy` | **아니오** | 특정 함수의 바이트코드에 "추적하지 말고 건너뛰라"는 표식을 다는 것. 이 표식도 **eval-frame 후킹이 실제로 도는 동안에만** 참조됨 |

**둘 다 "이름이 있으면 되는" 것과 "동작해야 하는" 것의 중간입니다** — `hasattr` 로 존재만
확인하고 넘어가는 자리(off-switch)가 아니라 **무조건 호출되지만**, 그 호출의 효과는
`torch.compile` 을 쓰지 않는 한 아무도 관측하지 않습니다. 그러므로 시그니처(인자 개수·타입)만
맞는 아무 동작 없는 no-op 이면 충분합니다 — 예외만 던지지 않으면 됩니다.

`.pyi` 스텁(`torchnative/src/main/torch/_C/_dynamo/eval_frame.pyi:12-18`)의 실제 시그니처:

```python
def set_guard_error_hook(hook: DynamoGuardHook) -> None: ...
def set_code_exec_strategy(code: types.CodeType, strategy: _FrameExecStrategy) -> None: ...
```

둘 다 반환형이 `None` 으로 선언돼 있습니다 — 상류 자신도 반환값을 안 쓰는 API 라는 뜻입니다.

### 3.3 나머지 50개 — 존재만 필요

`from X import (...)` 문에 이름이 나열돼 있으면, 파이썬은 그 문장이 실행되는 시점에
**그 이름들이 존재하는지만** 확인합니다. 클래스(`GuardManager`, `RootGuardManager`,
`LeafGuard`, 각종 `*GuardAccessor` 등 19개)는 `isinstance`/서브클래싱에 쓰이지만, **그 사용은
전부 함수·메서드 본문 안**이라 그 함수가 실제로 호출될 때만 실행됩니다 — 그리고 그 함수들은
가드 트리를 빌드/순회하는 코드라서 **`torch.compile` 이 실제로 컴파일을 시도할 때만** 실행됩니다.

직접 확인한 예:

```
torch/_dynamo/convert_frame.py:332  (함수 _fn 본문 안)
    guards = GlobalStateGuard()
```

`GlobalStateGuard()` 인스턴스화는 `_fn` 이 실제로 호출될 때만 일어나고, `_fn` 은
`torch._dynamo.eval_frame` 이 프레임 평가 후킹으로 설치하는 콜백입니다 — 후킹 자체가
`torch.compile()` 호출 없이는 설치되지 않습니다.

`GlobalStateGuard` 타입 힌트로도 한 번 더 나오는데(`guards.py:4392`,
`self.global_state: torch._C._dynamo.guards.GlobalStateGuard | None = None`), `guards.py:18` 에
`from __future__ import annotations` 가 있어 **이 어노테이션은 런타임에 평가되지 않습니다**
(문자열로만 저장). 그래서 이 자리는 접근 카운트에 안 잡혔습니다 — 이것도 자체 검증이 맞았다는
방증입니다(잡을 게 없는데 안 잡힘).

---

## 4. forward + `generate()` — 추가 접근 0

Llama 2층 hidden 32 모델을 만들어 `model(input_ids)` 와 `model.generate(input_ids,
max_new_tokens=3)` 까지 같은 계측기로 돌렸습니다.

```
[A_import_GenerationMixin] OK
[B_build_and_forward] OK logits shape=(1, 5, 64)
[C_generate] OK gen shape=(1, 8)

Total distinct names accessed: 61   (52개 실이름 + __file__/__path__ 등 더더 9개)
Total distinct names CALLED: 2
```

B·C 단계에서 **`_C._dynamo` 이름이 단 하나도 추가로 접근되지 않았습니다.** import 시점에
전부 끝난다는 뜻입니다 — `torch.compile` 을 쓰지 않는 한 실행 경로는 이 서브시스템을 다시
건드리지 않습니다. 이것이 "얼마나 들어가야 하는가"라는 질문에 대한 가장 중요한 답입니다:
**런타임 동작이 아니라 순수하게 import-time 부기입니다.**

---

## 5. 부수 발견 — 기존 `surface.json` 이 이 표면을 놓치고 있었다

`vendor/gen_surface.py::submodule_stubs()` 는 `torch/_C/` 아래에서 `.pyi` 파일 또는
`__init__.pyi` 가 있는 디렉터리를 **한 항목**으로 등록합니다. `_dynamo/` 는 디렉터리이고
`__init__.pyi` 가 있으므로 **"_dynamo" 라는 이름 하나**로 등록되고, 그 안의
`eval_frame.pyi`(105줄) · `guards.pyi`(509줄) · `compiled_autograd.pyi`(13줄) 는
**재귀적으로 파싱되지 않습니다.**

```
$ python3 -c "import json; d=json.load(open('rust/torch_c/src/surface.json'));
              print(d['submodules']['_dynamo'])"
{'functions': ['strip_function_call', 'is_valid_var_name', 'get_type_slots', 'has_slot'],
 'types': {'PyMappingSlots': ..., 'PyNumberSlots': ..., 'PySequenceSlots': ..., 'PyTypeSlots': ...},
 'values': []}
```

**8개.** 실제 137개의 6%입니다. `_export/` 도 같은 구조(중첩 `.pyi` 1개)를 갖고 있어 같은 버그의
영향을 받을 가능성이 있습니다(확인은 `_dynamo` 만 했습니다 — `_export` 는 이 작업 범위 밖).

> **Correction (커밋 `70f288b` "Fix: Recurse into directory stubs, which the surface scanner
> walked only one level deep" — 이 문서 바로 다음 커밋): 이 버그는 고쳐졌습니다.** `gen_surface.py`
> 가 이제 디렉터리형 서브모듈을 재귀적으로 파싱하고, `surface.json` 의 `_dynamo` 항목은 지금
> `submodules` 키 아래 `eval_frame`(35) · `guards`(34) · `compiled_autograd`(5) 를 온전히 담고
> 있습니다(현재 트리에서 직접 읽어 확인). `_export/` 도 같은 커밋에서 함께 고쳐졌습니다
> (`submodules.pt2_archive_constants` 로 재귀됨) — §5 가 "이 작업 범위 밖"으로 남겨둔 그 우려가
> 맞았고, 그대로 고쳐졌습니다.

**좋은 소식은 `.pyi` 데이터 자체는 이미 있다는 것입니다.** §3.1에서 접근된 52개 전부
(`eval_frame.pyi` · `guards.pyi` 기준) 스텁에 **선언돼 있습니다** — 시그니처까지 포함해서.
빠진 건 `gen_surface.py` 가 그 파일들을 찾아 읽는 로직이지, 원본 데이터가 아닙니다.

`utils.pyi` 는 벤더링 트리에 **아예 없습니다**(런타임에는 `is_instancemethod` 하나가 있는데도).
이 경로에서 안 쓰이므로 우선순위는 낮지만, `_C._dynamo` 를 패키지로 완성하려면 언젠가 채워야
합니다 — **미확인.**

---

## 6. off-switch 가능성 — 없음

IMPORT_TORCH.md 가 찾은 32개 off-switch(`hasattr(torch._C, "...")` 패턴)는 **`torch._C` 최상위에
대한 질문**입니다. `_dynamo` 내부의 임포트문들은 전부 무조건 실행되는 `from X import (...)` 이지,
`try/except ImportError` 로 감싸여 있지 않습니다 — `torch/_dynamo/eval_frame.py:59`,
`torch/_dynamo/guards.py:49-77` 둘 다 확인했습니다.

한 단계 위, `transformers.masking_utils` 자체에도 조건이 있긴 합니다.

```python
# masking_utils.py:38-42
_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)
if _is_torch_greater_or_equal_than_2_6:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
```

이론적으로는 "torch 버전을 2.6 미만으로 자칭하면 이 임포트를 피할 수 있다"는 레버가
존재하지만, **DESIGN.md §1 의 전제(파사드 금지, 벤더링한 소스는 한 줄도 안 고침)와
IMPORT_TORCH.md 의 관문 조건(`transformers` 가 요구하는 `torch >= 2.5.0`, 그리고 최신 API 표면
전반이 2.6+ 를 가정)에 부딪힙니다.** 버전을 낮춰 자칭하는 것은 이 우회로가 막는 것보다 더 많은
것을 깰 가능성이 높습니다 — **실측하지 않았고, 이 문서의 판단으로는 시도할 가치가 낮습니다
(미확인으로 남깁니다).**

**결론: off-switch 없음.** `_C._dynamo` 표면 자체를 채우는 것 외에 다른 경로가 안 보입니다.

---

## 7. 최소 구현 제안

`torch.compile` 을 절대 쓰지 않는다는 전제 위에서:

1. **`_C._dynamo` 를 패키지로 등록**하고 그 아래 4개 서브모듈(`eval_frame` · `guards` ·
   `utils` · `compiled_autograd`)도 패키지로 등록합니다 — VENDOR.md 벽 8과 같은 메커니즘
   (`_SubmoduleFinder`, `rust/torch_c/src/bootstrap.py:260` 부근)을 그대로 재사용하면 됩니다.
   이미 있는 인프라입니다.
2. **52개 이름을 채웁니다** — 소스는 이미 벤더링된 `.pyi` 3개
   (`torchnative/src/main/torch/_C/_dynamo/{eval_frame,guards,compiled_autograd}.pyi`)와 §3.1의 접근 목록의
   교집합입니다. 클래스(19개)는 빈 몸체의 자리표 타입으로, 함수(29개, 호출 2개 제외)는
   시그니처만 맞춘 no-op 으로 충분합니다 — **호출되지 않으므로 동작을 구현할 필요가 없습니다.**
3. **`set_guard_error_hook` · `set_code_exec_strategy` 2개만 신경 씁니다** — 인자 개수와
   타입(콜러블 1개 / 코드객체+전략객체)만 받아들이고 `None` 을 돌려주는 no-op. 예외만 던지지
   않으면 정확성에 영향이 없습니다(§3.2 — 훅이 실제로 트리거되려면 eval-frame 후킹이 설치돼
   있어야 하는데, `torch.compile` 을 안 쓰므로 설치되지 않습니다).
4. **`utils`(`is_instancemethod` 1개) · `compiled_autograd`(5개)는 이번 경로에서 필요 없습니다**
   — 빈 패키지로만 존재하면 됩니다. 나중에 다른 임포트 경로가 건드릴 수 있으니 완전히
   생략하지는 말되, 이번 최소 구현의 우선순위에서는 뒤로 둡니다.
5. 기존 `overloads.json`/aten 작업과 같은 패턴으로, **표를 상류 `.pyi` 와 대조하는 검증기**
   (`verify_schemas.py` 류)를 하나 두는 것을 권합니다 — §5에서 `.pyi` 자체는 신뢰할 수 있음을
   확인했으니, 표를 손으로 옮겨 적을 때 생기는 오타/누락만 잡으면 됩니다.

**작업량 추정(코드 작성 없이, 이미 있는 인프라와 비교한 상대적 크기):** `TensorBase` 694개
멤버를 채운 작업(C_SURFACE.md §3, 이미 완료됨)에 비하면 **1/13 규모**(52 vs 694)이고, 그마저
50개는 no-op 자리표, 실제 동작이 필요한 건 2개뿐입니다. "서브시스템 표면"이라는 처음 우려보다
훨씬 작은 작업입니다.

---

## 8. 미확인

| # | 항목 | 상태 |
|---|---|---|
| 1 | `utils.pyi` 가 벤더링 트리에 없는 이유, 필요해지는 경로가 있는지 | 미확인. 이번 경로에선 0회 접근 |
| 2 | `_export/` 도 `gen_surface.py` 의 같은 미탐지 버그 영향을 받는지 | 존재는 확인(§5), 내용은 미조사 — 이 작업 범위 밖 |
| 3 | `transformers.masking_utils` 의 `torch >= 2.6` 조건을 낮춰 dynamo 임포트를 우회하는 것이 실제로 더 적은 피해로 가능한지 | 미실측. DESIGN.md §1 전제와 충돌 가능성이 커서 시도하지 않았음 |
| 4 | `torch._dynamo` **파이썬 패키지**(93개 `.py`) 쪽에서 벤더링한 소스 자체가 추가로 요구하는 것(순수 파이썬 의존성, 예: `sympy` — `eval_frame.py:9` 에서 확인됨)이 전부 충족 가능한지 | 이번 측정 환경(spike-venv)엔 이미 있어 막히지 않았음. torchnative 최종 벤더링 tree 기준으로는 미확인 |
| 5 | `GenerationMixin` 이외의 다른 transformers 진입점(비-생성형 모델, 트레이닝 경로)이 `_C._dynamo` 의 다른 이름을 추가로 요구하는지 | 미측정. 이 문서는 `from_config`/`generate` 경로만 봄(OVERLOAD.md §8과 동일 범위) |
| 6 | `torch.compile` 을 실수로라도 트리거하는 transformers 내부 경로가 있는지(예: 일부 최신 모델의 `attn_implementation="sdpa"` 선택 로직이 조건부로 `torch.compile` 래핑을 시도하는 경우) | 미확인. 이번 측정은 `do_sample=False`, 기본 config 만 사용 |

---

## 9. 재현

```bash
PY=/Volumes/macMini/caches/spike-venv/bin/python
$PY /tmp/dynamo_probe2.py
```

핵심 부분만 발췌:

```python
proxy = types.ModuleType(qualname)
def tracked_getattr(name, ...):
    ACCESS.setdefault(key, []).append(STAGE)          # 접근 기록
    val = getattr(real_mod, name)
    if callable(val) and not isinstance(val, type):
        def wrapper(*a, **k):
            ret = val(*a, **k)
            CALLS.setdefault(key, []).append((...))    # 호출 기록
            return ret
        return wrapper
    return val
proxy.__getattr__ = tracked_getattr
sys.modules[f"torch._C._dynamo.{sub}"] = proxy   # style 1: from-import
setattr(dynamo_c, sub, proxy)                    # style 2: attribute access
```

자체 검증(양성 3종 + 음성 대조군 1종)을 통과한 뒤에만 실측값을 신뢰했습니다(§2.3).

---

# 파트 B — `torch.compile()` 을 실제로 불러봤다

§0~9 는 **`torch.compile` 을 한 번도 부르지 않는** 경로(`GenerationMixin` 임포트, `forward`,
`generate()`)가 `_C._dynamo` 에 요구하는 것을 쟀습니다. 결론은 "부기 2개만 no-op 이면 된다"였고,
그 결론은 지금도 유효합니다 — **`torch.compile` 을 안 쓰면**.

이 파트는 다른 질문입니다: **`torch.compile(f)(x)` 를 실제로 불렀을 때 무엇이 막는가.**
README 의 "Dynamo 안에서 멈춘다"는 한 줄을 대체하는 것이 목적이고, 아래가 그 진단입니다.

**요약 먼저:**

| | 값 |
|---|---|
| 막힌 지점 (`backend='eager'`, 실제 호출 시) | `torch._C._dynamo.eval_frame.set_eval_frame` — CPython 의 PEP 523 프레임 평가 후킹 |
| 그 심볼이 요구하는 CPython 기능 | `_PyInterpreterState_SetEvalFrameFunc`, `_PyInterpreterFrame` — 전부 `Py_BUILD_CORE` 로 가드된 내부 헤더 (§15, 실측) |
| Limited API 에서 닿는가 | **아니오 — 구조적으로 닿지 않습니다.** 근거는 상류 C 소스 (§15) |
| 막힌 지점 (기본 백엔드 `inductor`, 호출 이전 `compile()` 시점) | `torch._C._export.pt2_archive_constants` 부재 → 때우면 `torch.jit.script_method` → TorchScript 프론트엔드(`_jit_tree_views`) 요구 — **다른 벽, eval-frame 과 무관** (§12) — **이 행은 §12 끝의 correction 을 보라: `docs/graph/TORCHSCRIPT.md` 이후 기본값에서는 재현되지 않는다** |
| 부기를 전부 no-op 으로 때웠을 때 실제로 컴파일이 일어나는가 | **아니오 — 실측으로 확인.** 함수가 매 호출마다 파이썬 레벨에서 다시 실행되고 dynamo 카운터가 전부 0 (§13) |
| 캡처(`_aten_dispatch` 단일 관문) 가 대신 주는 것 | 구간 기록 + Core ATen 분해, NPU 델리게이트가 필요로 하는 서브그래프 모양까지 (§16, 기존 CAPTURE.md/DECOMP.md) |

---

## 10. 실행 환경과 정직성 노트

산출물이 낡아 있었습니다 — `$CARGO_TARGET_DIR/release/lib_C.dylib` 의 mtime 이 8/24 였고, 이번
조사는 8/30 부터입니다. **측정 전에 `vendor/install_shim.sh` 로 다시 빌드하고 설치했습니다**
(경고 1개, `dtype.rs::by_name` dead-code, 무해). 이 문서의 모든 트레이스백은 그 재빌드 이후
산출물로 낸 것입니다.

```
PY=/Volumes/macMini/caches/spike-venv/bin/python
export TORCH_USE_RTLD_GLOBAL=1
export PYTHONPATH=<worktree>/torchnative/src/main
```

이 파트에서 "됐다"고 적은 모든 것은 **스텁을 얹은 뒤의 결과**입니다. 스텁을 얹지 않은 원래
상류 소스는 벤더 트리 그대로이고 한 줄도 고치지 않았습니다(`git status --short` 로 확인
가능 — `docs/graph/DYNAMO.md` 하나만 바뀝니다). 스텁은 전부 `/tmp/dynamo_probe_stub*.py` 에 있고
저장소 밖입니다. 어디서부터 스텁인지는 §12·§13 표에 전부 표시했습니다.

---

## 11. 세 가지 모델, 같은 벽

지시대로 trivial 함수 → `nn.Linear` → 작은 트랜스포머 블록 순으로 돌렸습니다
(`backend='eager'`, §12 의 import-time 벽은 `pt2_archive_constants` 스텁으로 우회한 상태).

```python
def f(x): return x + 1
cf = torch.compile(f, backend='eager'); cf(torch.ones(3))

m = nn.Linear(4, 8)
cm = torch.compile(m, backend='eager'); cm(torch.randn(2, 4))

layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True)
cm = torch.compile(layer, backend='eager'); cm(torch.randn(2, 5, 16))
```

**세 경우 다 정확히 같은 곳에서 죽습니다:**

```
  File ".../torch/_dynamo/eval_frame.py", line 1084, in compile_wrapper
    prior = set_eval_frame(None)
  File "torch_c_bootstrap.py", line 204, in __call__
NotImplementedError: not implemented in torch._C shim: torch._C._dynamo.eval_frame.set_eval_frame
```

**차이가 없다는 것 자체가 정보입니다.** `compile_wrapper` 의 서두(`set_eval_frame(None)`)는 `fn`
이 무엇인지 보기 *전에* 실행되는 부기라서, 모델의 크기나 모양은 이 벽에 전혀 관여하지 않습니다.
막는 것은 op 커버리지도, `nn.Module` 지원도 아니고 **호출 진입점 자체**입니다. 이 벽은
`torch.compile` 을 부르는 어떤 모델에도 모델 이전에 옵니다.

---

## 12. 기본 백엔드(`inductor`)로는 더 일찍, 다른 이유로 막힌다

**정정(자체 발견).** 초안에서는 "`get_compiler_fn` 의 무조건 import 가 `backend='eager'` 에도
똑같이 적용된다"고 썼는데, 이건 절반만 맞았습니다. 실제로 갈라 재보니 두 개의 **서로 다른**
임포트 사슬이 있고, 하나만 백엔드와 무관합니다. 아래는 세 가지 조합을 각각 스텁 없이/스텁 하나로
실측해 나온 정정된 그림입니다.

| | `backend` 인자 | 어디서 죽나 | 벽 |
|---|---|---|---|
| A | 아무거나(`'eager'` 포함) — `get_compiler_fn` 의 무조건 import | `torch.compile(f, backend=...)` **구성 시점** | `pt2_archive_constants` (아래) |
| B | 기본값(`'inductor'`) 전용 — `_TorchCompileInductorWrapper.get_compiler_config()` → `compile_fx` | 같은 구성 시점, A 를 넘은 뒤 | `SourceRangeFactory.make_range` (mkldnn/TorchScript, 아래) |

**A — 스텁 없이, `backend='eager'` 로도 재현됨(구성 시점, 호출 전):**

```
File ".../torch/_functorch/aot_autograd.py", line 29, in <module>
    from torch._inductor.codecache import resolve_pre_grad_pass_timing
File ".../torch/export/pt2_archive/constants.py", line 5, in <module>
    AOTINDUCTOR_DIR: str = pt2_archive_constants.AOTINDUCTOR_DIR
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: '_Unimplemented' object has no attribute 'AOTINDUCTOR_DIR'
```

이건 `get_compiler_fn`(`torch/_dynamo/eval_frame.py:1512`, `from .repro.after_dynamo import
wrap_backend_debug`)의 무조건 import 사슬이 맞고, **여기까지는 백엔드와 무관하다는 것을 직접
확인했습니다** — `backend='eager'`, 스텁 전혀 없이 돌려서 똑같이 재현됨(§19 재현 스크립트).

`torch._C._export.pt2_archive_constants` 는 `.pt2` 아카이브 안의 경로 상수 27개(`AOTINDUCTOR_DIR`,
`ARCHIVE_FORMAT_VALUE` 등, 상류 값은 §19 재현 스크립트에 그대로 있음) — **값이지 동작이 아닙니다.**
상류 값을 그대로 채운 스텁 모듈을 `sys.modules` 에 얹으면(스캐폴드일 뿐, `bootstrap.py` 는
안 건드림) — **`backend='eager'` 는 여기서 곧장 §11 의 `set_eval_frame` 벽(호출 시점)으로
넘어갑니다.** 즉 B 는 `backend='eager'` 에서는 **일어나지 않습니다** — 아래는 기본값
(`backend` 인자를 안 주는 경우, `'inductor'`)로 다시 스텁을 얹고 구성까지 밀었을 때만 나옵니다.

**B — `backend` 를 안 주고(기본값 `'inductor'`) A 를 스텁으로 넘긴 뒤, 여전히 구성 시점:**

```
File ".../torch/utils/mkldnn.py", line 19, in MkldnnLinear
    @torch.jit.script_method
File ".../torch/jit/_script.py", line 384, in script_method
    ast = get_jit_def(fn, fn.__name__, self_name="ScriptModule")
File ".../torch/jit/frontend.py", line 428, in build_def
    r = ctx.make_range(...)
NotImplementedError: not implemented in torch._C shim: SourceRangeFactory.make_range
```

**이건 eval-frame 과 무관한 완전히 다른 벽입니다.** `torch._inductor.compile_fx` →
`fx_passes.pre_grad` → `torch.fx.experimental.optimization` → `torch.utils.mkldnn` 의 임포트
사슬에서, `MkldnnLinear` 라는 클래스가 **모듈 스코프에서** `@torch.jit.script_method` 로 자기
메서드를 TorchScript 로 컴파일합니다 — `MkldnnLinear` 를 쓸 생각이 없어도, `torch.compile` 을
기본 백엔드로 한 번이라도 부르면 이 임포트가 무조건 실행됩니다.

`SourceRangeFactory.make_range` 는 `torch._C._jit_tree_views` 의 멤버이고, **TorchScript 소스를
파이썬 AST 에서 자체 tree-view IR 로 옮기는 파서의 진입점**입니다(`torch/csrc/jit/python/python_tree_views.cpp`,
상류 409줄 — `Py_BUILD_CORE`/`internal/` 헤더 **없음**, 확인함). `IMPORT_TORCH.md` 표 13번이
"TorchScript 소스 파서 — 구현" 이라고 적은 것은 **이 자리가 아닙니다** — 거기서 구현한 건
`SourceContext` 생성자와 `ErrorReport.call_stack()` 이 빈 문자열을 돌려주는 것(임포트 시점에
필요한 만큼)이지, 실제로 함수를 파싱하는 `make_range` 는 처음부터 `_Unimplemented` 자리였습니다.

`make_range` 를 상류 시그니처에 맞게 스텁하면 한 단계 더 들어갑니다:

```
File ".../torch/jit/frontend.py", line 900, in build_Attribute
    source = ctx.source.encode("utf-8")
AttributeError: 'function' object has no attribute 'encode'
```

`build_def` → `build_stmts` → `build_stmt`(`Return`) → `build_expr`(`Tuple`, `Call`, `Attribute`)
— 파이썬 함수 하나의 AST 를 옮기는 데만 `Ident`·`Def`·`Return`·`Tuple`·`Call`·`Attribute` 같은
tree-view 노드 타입이 줄줄이 필요합니다. 이건 **막힌 심볼 하나가 아니라 두 번째 컴파일러
전체**입니다 — 상류 `torch/csrc/jit/` 은 21만 3천 줄입니다(`wc -l`, 실측). 여기서 멈췄습니다.
더 들어가는 것은 "TorchScript 프론트엔드를 새로 구현한다"는 별개 프로젝트이지, `torch.compile`
벽의 크기를 아는 데 필요한 정보를 더 주지 않습니다.

**이 벽은 abi3 문제가 아닙니다.** `python_tree_views.cpp` 는 평범한 pybind11 바인딩입니다
(§15 이 대조하는 `Py_BUILD_CORE` 가 여기 없습니다). 이 벽의 성격은 "missing symbol" 도
"abi3 밖" 도 아니고 **"컴파일러 백엔드가 없다"** 입니다 — TorchInductor 가 자기 코드 생성
경로에서(무관해 보이는 헬퍼 클래스를 통해서까지) TorchScript 를 끌어오고, 그 전체를 다시
구현하지 않는 한 기본 백엔드는 열리지 않습니다. `DESIGN.md` §5 가 이미 "TorchScript 는 JIT
컴파일러가 아니라 바이트코드 인터프리터라 iOS 에서도 된다"고 정정해 둔 것과 이 벽은 같은
서브시스템(TorchScript)이지만 **다른 질문**입니다 — 거기는 "TorchScript 아티팩트를 실행할 수
있는가"(된다)이고 여기는 "TorchScript 로 컴파일할 수 있는가"(안 됨, 프론트엔드가 없음).

> **Correction (문서 감사, 재측정): 이 절의 "Wall B" 는 이제 기본값에서 재현되지 않습니다.**
> `docs/graph/TORCHSCRIPT.md` (이 문서보다 나중, `02758e8`, 2026-08-31) 가
> `os.environ.setdefault("PYTORCH_JIT", "0")` 를 `bootstrap.py` 에 추가해 상류의 스크립팅
> 끄기 모드를 기본값으로 만들었습니다. 그 결과 `@torch.jit.script_method` 는
> (`torch/jit/_script.py` 의 `if not _enabled: return fn`) 함수를 그대로 돌려주고,
> `MkldnnLinear` 는 더 이상 여기서 `SourceRangeFactory.make_range` 를 부르지 않습니다.
> 재측정 확인: `pt2_archive_constants` 만 스텁하고 `PYTORCH_JIT` 를 건드리지 않은 채
> (오늘의 기본값) `torch.compile(f)()` 를 부르면 임포트가 이 벽을 그냥 통과해
> `torch._C._dynamo.eval_frame.set_skip_guard_eval_unsafe` — §11/§13/§15 가 이미 abi3 밖으로
> 판정한 것과 **같은 eval-frame 계열의 심볼** — 에서 막힙니다. `PYTORCH_JIT=1` 을 명시적으로
> 강제하면 (이 절이 측정할 당시의 조건과 같음) 이 절이 적은 그대로 `make_range` 에서 막히는
> 것도 재확인했습니다 — 즉 원래 측정 자체는 틀리지 않았고, 이 절이 전제한 환경(스크립팅 기본
> 켜짐)이 나중 라운드에서 바뀌었습니다. "다른 벽, eval-frame 과 무관" 이라는 요약표(파트 B 상단)
> 항목과 아래 §12 의 결론은 **오늘의 기본 경로에는 더 이상 적용되지 않습니다** — 기본 백엔드도
> 결국 §15 가 abi3 밖으로 판정한 것과 같은 종류의 벽에 부딪힙니다. §18 의 권고("캡처가 먼저다")는
> 영향받지 않습니다 — 도달하는 벽이 바뀌었을 뿐 그 벽이 abi3 로 못 넘는다는 결론은 §15 그대로
> 성립합니다.

---

## 13. `backend='eager'` 로 eval-frame 벽만 남기고, 전부 no-op 으로 때워봤다

§11 의 벽(`set_eval_frame`)이 진짜 하드 스톱인지, 아니면 "이름만 없는" 자리인지 가르는 방법은
하나뿐입니다 — **때워보고 그 뒤에 뭐가 나오는지 보는 것.** 한 번에 여러 개를 때우면 어느 것이
진짜 막았는지 알 수 없으므로, 하나씩 순서대로 추가했습니다. 전부 `/tmp/dynamo_probe_stub2.py`
에서 재현됩니다.

| # | 심볼 | 성격 | 때운 것 |
|---|---|---|---|
| 1 | `eval_frame.set_eval_frame` | **진짜 후킹** — PEP 523 콜백 설치/해제 | "이전 콜백"을 흉내내는 딕셔너리 하나. **실제 프레임 평가 후킹은 설치하지 않음** |
| 2 | `eval_frame.set_skip_guard_eval_unsafe` | 부기 플래그 | `lambda v: False` |
| 3 | `eval_frame.set_eval_frame_isolate_recompiles_id` | 부기 | `lambda v: -1` |
| 4 | `eval_frame.set_fullgraph_error_on_nested_compile` | 부기 | `lambda v: False` |
| 5 | `eval_frame.set_fullgraph_compiled_frame_count` | 부기 | `lambda v: -1` |
| 6 | `torch._C._dispatch_tls_local_include_set` | 디스패치 키 TLS 조회 | 빈 `DispatchKeySet()` 리턴 |
| 7 | `torch._C._dispatch_tls_local_exclude_set` | 〃 | 〃 |
| 8 | `torch._C._ForceDispatchKeyGuard` | 컨텍스트 매니저 | **다른 종류의 실패** — `TypeError: '...' object does not support the context manager protocol`. 합성된 클래스는 생성은 되지만(`_permissive_init`) `__enter__`/`__exit__` 는 안 생깁니다 — 이 셰임의 모듈 catch-all(`_attach_module_catchall`, `rust/torch_c/src/bootstrap.py:360`)이 던더는 일부러 걸러내기 때문(`if attr.startswith("__") ...: raise AttributeError`). 진짜 `__enter__`/`__exit__` 를 가진 클래스로 교체 |
| — | `torch._C._functorch.get_dynamic_layer_stack_depth` | — | **이미 진짜로 구현돼 있었습니다.** 스텁 없이 통과 — functorch/vmap 지원의 부산물로 보임(미확인, 이 조사 범위 밖) |
| 9 | `torch._C._functorch.pop_dynamic_layer_stack_and_undo_to_depth` | 부기 | `lambda d: None` |

**9개(+ 1개는 이미 구현됨)를 전부 때우면 `cf(x)` 가 "성공"합니다** — 예외 없이 값을 돌려줍니다.
하지만 이것이 "컴파일이 됐다"는 뜻인지는 별개로 확인해야 합니다(다음 절).

---

## 14. 그 "성공"이 진짜인지 확인했다 — 아니었다

**"실패할 수 없는 검사는 검사가 아니다"** — §13 의 성공이 컴파일을 증명하는지, 아니면 그냥
예외가 안 났다는 것만 증명하는지 갈라야 합니다. `set_eval_frame` 을 흉내낸 스텁은 **실제
PEP 523 후킹을 전혀 설치하지 않으므로**, `fn(*args, **kwargs)` 가 평범한 파이썬 함수 호출로
실행될 뿐이라는 것이 제 예상이었습니다. 직접 쟀습니다:

```python
calls = {'n': 0}
def f(x):
    calls['n'] += 1
    return x + 1

cf = torch.compile(f, backend='eager')
for _ in range(3):
    out = cf(x)

print('python-level calls to f:', calls['n'])
print('dynamo frame counters:', dict(dynamo_utils.counters['frames']))
print('dynamo stats counters:', dict(dynamo_utils.counters['stats']))
```

```
python-level calls to f: 3
dynamo frame counters: {}
dynamo stats counters: {}
```

**`f` 가 세 번 다 파이썬 레벨에서 재실행됐고, dynamo 의 프레임/컴파일 카운터가 전부 비어
있습니다.** 진짜 컴파일이 한 번이라도 일어났다면 최소한 첫 호출에서 프레임이 하나 잡혀야
합니다. 즉 **§13 의 "성공"은 eager 로의 조용한 폴백입니다** — dynamo 는 아무것도 추적하지
않았고, 그래프도 안 만들었고, 가드도 안 세웠습니다. `torch.compile` 이라는 이름이 붙은 껍데기가
`f` 를 그냥 다시 부른 것과 관측적으로 구별되지 않습니다.

**이것이 §15 의 결론을 실행으로 뒷받침합니다.** 부기를 전부 채워도 컴파일이 안 되는 이유는
"아직 하나 더 빠졌다" 가 아니라, **`set_eval_frame` 자체가 실제 후킹을 설치할 능력이 없기
때문**입니다 — 파이썬에서 흉내낼 수 있는 것은 "이전 콜백을 기억하고 돌려주는 부기"까지이고,
CPython 인터프리터가 매 바이트코드 프레임마다 우리 콜백으로 되돌아오게 만드는 것은 그 아래
C 계층의 일이며, 그 계층이 abi3 밖입니다(다음 절).

---

## 15. abi3 판정 — 상류 C 소스로 확인했다

**질문**: Dynamo 의 eval-frame 후킹이 Limited API 에서 원리적으로 닿는가?

**방법**: `/Volumes/macMini/caches/pytorch-spike/pytorch` (torch 2.13.0 소스 트리, 커밋
`16ee93b8a9e7`, 2026-08-23)의 `torch/csrc/dynamo/` 를 직접 읽었습니다. 이 저장소의 벤더 트리는
파이썬 소스만 담고 있어(C++ 은 없음), abi3 여부는 **상류가 그 C 구현을 무엇으로 빌드하는지**로만
답할 수 있습니다.

**실측 1 — 이 서브시스템 전체가 `Py_BUILD_CORE` 를 켭니다:**

```
$ grep -rn "Py_BUILD_CORE" torch/csrc/dynamo/*.c torch/csrc/dynamo/*.cpp torch/csrc/dynamo/*.h
torch/csrc/dynamo/cpython_defs.c:8:#define Py_BUILD_CORE
torch/csrc/dynamo/cpython_includes.h:13:#define Py_BUILD_CORE
torch/csrc/dynamo/framelocals_mapping.cpp:6:#define Py_BUILD_CORE
torch/csrc/dynamo/eval_frame.c:14:#define Py_BUILD_CORE
torch/csrc/dynamo/stackref_bridge.c:13:#define Py_BUILD_CORE
torch/csrc/dynamo/guards.cpp:101:#define Py_BUILD_CORE
```

`Py_BUILD_CORE` 는 CPython 자신을 빌드할 때 쓰는 매크로로, 이걸 켜면 `internal/pycore_*.h` —
CPython 인터프리터의 사설 구현 헤더 — 에 접근할 수 있습니다. **`Py_LIMITED_API`(abi3 가 서는
토대)와 `Py_BUILD_CORE` 는 같은 확장 모듈에서 양립하지 않습니다** — 전자는 "버전이 바뀌어도
안정된 표면만 본다"는 약속이고 후자는 정확히 그 반대, "인터프리터 내부 구조체를 직접 본다"입니다.

**실측 2 — 정확히 어떤 심볼:**

```
torch/csrc/dynamo/eval_frame.c:252:    _PyInterpreterState_SetEvalFrameFunc(
torch/csrc/dynamo/eval_frame.c:261:    _PyInterpreterState_SetEvalFrameFunc(tstate->interp, previous_eval_frame);
torch/csrc/dynamo/eval_frame.h:18:#define THP_EVAL_API_FRAME_OBJECT _PyInterpreterFrame
torch/csrc/dynamo/eval_frame_cpp.cpp:109:    _PyInterpreterFrame* iframe = frame->f_frame;
```

`set_eval_frame` — §11~14 가 실측으로 하드 스톱임을 확인한 바로 그 심볼 — 은 `_PyInterpreterState_SetEvalFrameFunc`
(PEP 523 의 진입점, 밑줄 접두사가 이미 "사설" 이라고 말하고 있음)와 `_PyInterpreterFrame`(프레임
객체의 내부 표현)에 직접 의존합니다.

**실측 3 — 그 내부 구조체는 CPython 마이너 버전마다 모양이 다릅니다** — abi3 가 "하나의 바이너리가
3.13 이상 전부에서 로드된다"고 약속하는 바로 그 지점에서 깨집니다:

```c
// cpython_includes.h
#if IS_PYTHON_3_14_PLUS && !defined(_WIN32)
#include <internal/pycore_code.h>
#include <internal/pycore_genobject.h>
#include <internal/pycore_interpframe.h>
#include <internal/pycore_stackref.h>
#elif IS_PYTHON_3_14_PLUS && defined(_WIN32)
#include <internal/pycore_interpframe_structs.h> // _PyInterpreterFrame
#endif
```

같은 `cpython_includes.h` 안에 3.11 · 3.12 · 3.13 · 3.14 조건부 매크로가 더 있고 (`F_CODE`,
`PREV_INSTR`, `FUNC` 매크로가 버전마다 다른 필드를 읽습니다 — `eval_frame.c`·`eval_frame_cpp.cpp`
가 이 매크로를 통해 그 필드를 씁니다), **abi3 로 3.13 기준 한 번
빌드해도 3.14 가 나오면 그 구조체 레이아웃이 바뀔 수 있다는 뜻입니다.** 이건 우리 shim 이 아직
안 채운 구멍이 아니라, **상류 자신도 "고정된 ABI" 가 아니라 "CPython 버전마다 다시 컴파일"로
이 문제를 풀고 있다는 증거**입니다.

**판정: Limited API 에서 원리적으로 닿지 않습니다.** "아직 못 만든 심볼"이 아니라 "abi3 로 빌드된
확장 모듈 하나가 CPython 내부 구조체에 접근하는 것 자체가 금지된 조합"입니다. `docs/design/ABI3.md`
§2a 는 상류 `libtorch_python.dylib` 의 미해결 심볼 54개 중 21개를 "프레임 평가 후킹 · 코드 객체
조사"로 분류하고 "추론 경로에 필요한가 — 아니오, TorchDynamo(PEP 523)" 라고 표를 통해 이미
짚어 두었습니다. 이 파트는 그 표 항목 하나를 **실제로 불러서** 실행 시점 확인으로 뒷받침합니다 —
그 표는 정적 심볼 분류였고, 여기는 그 심볼이 정말 호출 경로에서 하드 스톱인지, 그리고 정말
대체 불가능한지(§14, 부기를 전부 채워도 안 됨)까지 실측했습니다.

**따로 열어둔 것**: 이 abi3 확장과 별도로, dynamo 의 eval-frame 후킹만을 위한 **두 번째, 비-abi3,
CPython 마이너 버전별 보조 확장**을 만들어 조건부로 로드하는 경로는 이론적으로 있습니다 — 하지만
그건 "심볼을 채운다"가 아니라 **"하나의 바이너리" 라는 배포 모델 자체를 버리는 결정**입니다(현재
5개 배포 wheel 이 "3.13 이상 전부에서 로드"를 전제합니다). 이 조사는 그 트레이드오프를 재지
않았습니다 — 결정할 사람의 몫으로 남깁니다(§17 미확인 1).

---

## 16. 캡처가 대신 주는 것 — 그리고 안 주는 것

`torch.compile` 이 막힌 이유가 "이 shim 이 아직 op 을 다 못 채워서"가 아니라 **eval-frame 후킹이
구조적으로 안 되기 때문**이라는 것이 §15 의 결론입니다. 그런데 `DESIGN.md` §11.1 이 NPU 를 위해
필요하다고 짚은 것도 결국 "코드 한 구간을 그래프로 받는 것"이고, 그건 이미 다른 문으로 있습니다.

`docs/graph/CAPTURE.md`·`docs/graph/DECOMP.md` 가 이미 세운 것 — **`_aten_dispatch` 단일 관문**에서 구간을
기록하는 경로 — 을 이 질문("`torch.compile` 대신 무엇을 주는가")의 틀로 다시 요약합니다.

| | Dynamo (`torch.compile`) | 이 저장소의 캡처 |
|---|---|---|
| 어디서 가로채나 | CPython 바이트코드 프레임 (PEP 523) | 텐서 연산 디스패치 (`_aten_dispatch` 끝 한 줄) |
| Limited API 에서 닿는가 | **아니오** (§15) | **예 — 이미 동작**(rust 함수 훅 하나, CPython 헤더 무관) |
| 구간을 어떻게 정하나 | 자동 — 바이트코드를 보고 그래프가 끊기는 지점(graph break)을 스스로 찾음 | **수동** — `_capture_begin`/`_capture_end` 를 호출자가 부름. 자동 구간 선택은 없음(CAPTURE.md §9) |
| 제어 흐름 · 동적 형태 | 부분 지원 (graph break 로 폴백) | **거절** — 이름을 대고 eager 로 폴백(CAPTURE.md §4) |
| 얻는 것 | 파이썬 함수를 감싸는 투명한 최적화 (사용자는 `torch.compile(f)` 만 씀) | `CaptureTrace` 객체 — FX/`ExportedProgram` 과 1:1 대응(CAPTURE.md §5), Core ATen 분해까지 감(DECOMP.md, 37개 중 9개 낮춤) |
| NPU 델리게이트가 받는 것 | (구조적으로 도달 못 함) | **서브그래프** — ANE·NNAPI·QNN 이 필요로 하는 바로 그 모양 |

**어디까지 가고 어디서 멈추는가를 정직하게 적으면:**

캡처는 **"사람이 시작·끝을 부르면, 그 구간을 op 하나 안 놓치고 재생 가능한 그래프로 기록한다"**
까지 이미 증명됐습니다(CAPTURE.md §3, 비트 단위 일치). 사람들이 보통 "`torch.compile` 이 된다"고
할 때 기대하는 것 — **아무 상류 코드에 손 안 대고 데코레이터 하나로 자동으로 빨라진다** — 은
캡처가 주지 않습니다. 캡처에는:

- **자동 구간 선택이 없습니다.** `model.generate()` 를 통째로 감싸면 안에서 `_local_scalar_dense`
  (샘플링의 `item()` 호출 등)를 만나 캡처가 통째로 무효화됩니다 — Dynamo 라면 그 지점에서
  graph break 를 내고 앞뒤는 여전히 컴파일하는데, 캡처는 지금 "전부 아니면 전무"입니다.
- **가드 캐시가 없습니다.** 같은 형태가 다시 들어와도 재사용은 호출자가 `replay()` 를 직접
  불러야 합니다.
- **백엔드가 없습니다.** 재생은 여전히 같은 `_aten_dispatch` 문으로 돌아갑니다(CAPTURE.md §3) —
  그래프를 기록할 수 있다는 것과 그 그래프를 다른 곳에서 실행한다는 것은 다른 문제이고, 후자는
  아직 아무것도 없습니다.

**그래도 이것이 DESIGN.md §11.1 이 원래 물었던 질문 — "NPU 에 op 이 아니라 그래프를 넘기려면
무엇이 있어야 하는가" — 에는 답이 됩니다.** ExecuTorch 스타일 델리게이트가 필요로 하는 것은
"아무 파이썬 함수나 자동으로 빨라지는 것"이 아니라 **"이 구간은 이런 모양의 서브그래프다"라는
사실**이고, 캡처는 그 사실을 이미 만들어 냅니다. `torch.compile` 의 사용자 경험(자동·투명)과
캡처가 주는 것(수동으로 표시한 구간의 정확한 기록)은 **다른 제품**입니다 — 후자가 전자의 부분
집합이 아니라, 후자는 "NPU 델리게이트 인프라"이고 전자는 "일반 사용자를 위한 자동 가속기"라서
목적이 다릅니다.

---

## 17. 미확인

| # | 항목 | 상태 |
|---|---|---|
| 1 | eval-frame 후킹만을 위한 별도(비-abi3) 보조 확장을 조건부로 로드하는 경로의 실제 비용 | **미실측.** "하나의 바이너리" 배포 모델을 버리는 결정이라 이 조사 범위 밖에 남겨둠(§15) |
| 2 | `torch.export` (Dynamo 와 다른 경로 — FakeTensor 기반, eval-frame 후킹을 안 씀)가 이 shim 에서 어디까지 가는지 | **미측정.** eval-frame 벽과 무관할 가능성이 있고, 캡처 경로와 겹칠 수 있음 — 별도 조사가 필요 |
| 3 | 기본 백엔드(`inductor`) 경로의 남은 크기 — TorchScript 프론트엔드를 넘어서도 Triton 코드 생성 · C++ 컴파일러 호출까지 있음 | **미측정.** §12 가 멈춘 지점(TorchScript 프론트엔드 전체가 없음)이 이미 구조적으로 커서, 더 들어가도 "`torch.compile` 벽의 크기"라는 질문에 새 정보를 주지 않는다고 판단해 멈춤 |
| 4 | `transformers` 내부 경로 중 사용자가 `torch.compile` 을 명시적으로 안 써도 우연히 트리거하는 곳이 있는지 | **미확인** — 파트 A §8 항목 6 이 이미 같은 미확인을 남겨 둠. 여전히 안 풀림 |
| 5 | Android/iOS 실제 기기에서 PEP 523 후킹 시도가 다른 실패 모드를 내는지 (예: 코드 서명·W^X 가 abi3 문제보다 먼저 막는지) | **미측정.** 데스크톱(spike-venv, macOS)에서만 확인. `DESIGN.md` §5 는 iOS 의 W^X 가 **런타임 코드 생성**(TorchInductor)을 막는다고 이미 정리했는데, 이번에 막힌 것(eval-frame 후킹 자체)은 코드 생성이 아니라 CPython 내부 구조체 접근이라 W^X 와는 다른 제약 — 기기에서 실제로 어느 쪽이 먼저 걸리는지는 안 재봤음 |
| 6 | `_ForceDispatchKeyGuard`·`_dispatch_tls_local_include_set` 류를 실제로 구현하면(§13 항목 6-8) 캡처·다른 경로에 부작용이 있는지 | **미측정** — 이번엔 순수 스캐폴드였고 저장소에 반영 안 함 |

---

## 18. 권고

**Dynamo 경로를 더 파지 말고, 캡처 경로를 밀어야 합니다.** 이유: `torch.compile` 이 실제로
호출됐을 때 멈추는 첫 지점(`set_eval_frame`)은 "아직 못 채운 심볼"이 아니라 CPython 이
`Py_BUILD_CORE` 로만 여는 내부 구조체(`_PyInterpreterFrame`)에 대한 요구이고, 그 구조체는
마이너 버전마다 모양이 바뀝니다(§15) — 이건 이 shim 이 abi3(`abi3-py313`, 5개 배포 wheel 이
전제하는 "하나의 바이너리")를 포기하지 않는 한 채울 수 없는 종류의 구멍이고, 실제로 부기를
전부 no-op 으로 채워도 컴파일은 일어나지 않고 조용히 eager 로 폴백한다는 것까지 실측으로
확인했습니다(§14). 반대로 `_aten_dispatch` 캡처는 이미 abi3 안에서 동작하고(CAPTURE.md), NPU
델리게이트가 실제로 필요로 하는 것 — 정확한 서브그래프 — 을 이미 만들어 내고 있습니다(§16).
남은 일(자동 구간 선택, 가드 캐시, 직렬화, 첫 델리게이트 — CAPTURE.md §10)은 전부 "무엇을
만들지 아는" 상태의 유한한 엔지니어링이고, Dynamo 쪽은 "abi3 를 버릴 것인가"라는 이 shim의
정체성 질문이 먼저 풀려야 하는 다른 종류의 결정입니다. 그 결정을 지금 내릴 근거는 이 문서에
없습니다 — 그래서 위 권고는 "Dynamo 를 절대 하지 마라"가 아니라 **"캡처가 먼저다, Dynamo 는
abi3 트레이드오프에 대한 별도 결정이 나기 전까지는 다음 순번이 아니다"** 입니다.

---

## 19. 재현

```sh
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-dynamo
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh          # 산출물이 낡아 있을 수 있음 — 반드시 재빌드
PY=/Volumes/macMini/caches/spike-venv/bin/python
export TORCH_USE_RTLD_GLOBAL=1
export PYTHONPATH=$PWD/torchnative/src/main

# §11 — 세 모델, 같은 벽 (backend='eager', pt2_archive_constants 만 스텁)
$PY -c "import sys; sys.path.insert(0,'/tmp'); import dynamo_probe_stub as s; s.install_all_known_stubs(); \
        import torch; cf = torch.compile(lambda x: x+1, backend='eager'); print(cf(torch.ones(3)))"

# §12 — 기본 백엔드(inductor), pt2_archive_constants 만 스텁 (SourceRangeFactory.make_range 에서 멈춤)
$PY /tmp/dynamo_probe_stub.py

# §13-14 — eval-frame 벽 너머를 전부 no-op 으로 때우고, 진짜 컴파일이 일어났는지 카운터로 확인
$PY /tmp/dynamo_probe_stub2.py

# §15 — abi3 판정의 근거 (상류 C 소스, 이 저장소 밖 캐시)
grep -rn "Py_BUILD_CORE" /Volumes/macMini/caches/pytorch-spike/pytorch/torch/csrc/dynamo/*.{c,cpp,h}
grep -n "_PyInterpreterState_SetEvalFrameFunc\|_PyInterpreterFrame" \
    /Volumes/macMini/caches/pytorch-spike/pytorch/torch/csrc/dynamo/eval_frame.c
```

스텁 스크립트(`/tmp/dynamo_probe_stub.py`, `/tmp/dynamo_probe_stub2.py`)는 저장소 밖이고,
어떤 심볼을 어떤 순서로 무엇으로 때웠는지 코드에 그대로 남아 있습니다. `rust/torch_c/src/bootstrap.py`
·`aten.rs`·`overloads.json`·`tools/golden/`·`tools/wheel/` 은 이번 조사에서 전혀 건드리지
않았습니다 — `git status --short` 로 확인 가능한 변경 범위는 이 문서 하나뿐입니다.
