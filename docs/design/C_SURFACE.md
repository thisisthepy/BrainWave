# `torch._C` 이름 수준 계측 — 989 개 중 실제로 쓰이는 것은 몇 개인가

VENDOR.md 는 상류 `torch._C` 의 크기(이름 989 개·서브모듈 32 개·`TensorBase` 694 개·
`_VariableFunctions` 985 개)를 재고, 그중 **"임포트가 요구하는 것"** 을 다른 방법(빈 shim +
기록 모드)으로 쟀습니다. 이 문서는 같은 크기의 표면에 대해 **다른 질문**을 묻습니다 —
**진짜 torch 2.13.0 을 그대로 돌리면서, 989 개 중 어느 이름이 실제로 접근/호출되는가.**

**요약 한 줄:** `torch._C` 의 dir() 표면(989 개)은 `import torch` 만으로 이미 99% 가
"존재를 확인당합니다." 그러나 실제로 **호출**까지 되는 것은 훨씬 좁습니다 — 작은 Llama 로
순전파 + `generate(4 토큰)` 까지 돌렸을 때 `TensorBase` 멤버는 694 개 중 **50 개(7.2%)**,
`_VariableFunctions` 로 호이스팅된 연산자는 609 개 중 **13 개(2.1%)** 만 실제로 호출됐습니다.

---

## 0. 한눈에

| 표면 | 상류 크기 | "접근됨" (getattr 성공) | "실제 호출됨" | 비고 |
|---|---|---|---|---|
| `dir(torch._C)` | **989** | **979 (99.0%)**, 이미 `import torch` 단독으로 도달 | 미측정 (아래 §7-1) | 10 개는 4 단계 내내 한 번도 안 닿음 |
| `dir(TensorBase)` (via `torch.Tensor` 인스턴스) | **694** | **50 (7.2%)**, from_config + forward + generate 로만 도달 | = 접근됨(성질상 같음, §2-3 참고) | import 시점 클래스 구성 요구치(543, VENDOR §3)와는 **다른 것을 잰다** |
| `dir(_C._VariableFunctions)` | **985** (공개 625, `torch.*` 로 호이스팅 609) | **607/609 (99.7%)**, `import transformers` 만으로 도달 — **오염됨** (§3) | **13/609 (2.1%)** | "접근" 수치는 `torch._dynamo` 규칙 테이블 구축이 만든 잡음. 진짜 신호는 호출 수치 |

**이 표의 두 번째·세 번째 열이 이 문서의 핵심입니다.** "접근됨" 은 존재해야 한다는 뜻이고,
"호출됨" 은 제대로 동작해야 한다는 뜻입니다. `_VariableFunctions` 에서 이 둘의 차이가
99.7% 대 2.1% 로 극단적으로 벌어진다는 것이 §3 의 발견입니다.

---

## 1. 계측 방법

### 1-1. 왜 PEP 562 `__getattr__` 로는 안 되는가

`vendor/probe.py` (다른 작업 셋의 것, VENDOR.md 가 씀)는 **빈 shim** 위에서 모듈 레벨
`__getattr__` 로 "없는 이름" 접근을 잡습니다. 이번 계측은 **진짜 torch** 위에서 돕니다 —
`torch._C` 의 989 개 이름이 전부 이미 존재하므로, PEP 562 `__getattr__` 은 아예 호출되지
않습니다(그 훅은 "없는" 속성에만 걸립니다). **이미 있는 속성에 대한 접근도 잡아야 하므로
다른 방법이 필요합니다.**

### 1-2. 방법 — 모듈은 `__class__` 를 바꿔치기, 클래스는 무리별로 다른 전략

세 개의 실제 객체를 각기 다른 방식으로 계측했습니다. 셋 다 **진짜 객체를 그대로 두고
얇은 관찰 계층만 얹는** 방식입니다 — 값도 동작도 대체하지 않습니다.

**(a) `torch._C` (모듈)** — 커스텀 메타패스 파인더로 `'torch._C'` 서브모듈의 로더를 감싸,
실제 확장 모듈이 `exec_module` 을 마친 직후 `module.__class__` 를 `ModuleType` 의 서브클래스로
바꿔치기합니다. 모듈 객체는 원래 mutable 이므로 `__class__` 교체가 됩니다. 이렇게 하면
이미 채워진 속성에 대한 접근도 오버라이드한 `__getattribute__` 로 전부 걸립니다.

```python
class TracingLoader(importlib.abc.Loader):
    def exec_module(self, module):
        self.real_loader.exec_module(module)      # 진짜 PyInit__C 실행, 실제 속성 채움
        wrap_module_class(module, self.qualname)   # 그 다음에야 클래스 교체

class TracingFinder(importlib.abc.MetaPathFinder):
    TARGETS = {"torch._C"}
    def find_spec(self, fullname, path, target=None):
        if fullname not in self.TARGETS:
            return None
        for finder in sys.meta_path:                 # 진짜 파인더에게 위임
            ...
            spec.loader = TracingLoader(spec.loader, fullname)
            return spec
```

`sys.meta_path` 에 이 파인더를 **`import torch` 이전에** 꽂아 두면, `torch/__init__.py:445`
의 `from torch._C import *` 가 트리거하는 최초 로드 시점부터 계측이 걸립니다. 즉 `torch/
__init__.py` 의 나머지 실행(약 87% 구간)부터 이후 모든 단계까지 놓치지 않습니다.

**`torch` 패키지 모듈 자체**도 (a) 와 같은 기법으로 `import torch` 완료 직후 계측을 겁니다.
(모듈 로드가 이미 끝난 뒤라 메타패스 트릭이 필요 없고, `torch.__class__` 를 직접 바꿉니다.)

**중요한 버그와 그 수정 — 이것도 방법이 실제로 걸린다는 증거입니다.** 처음에는
`__getattribute__` 안에서 `object.__getattribute__(self, name)` 로 위임했는데, **진짜
`import transformers` 가 이 상태에서 깨졌습니다**:

```
AttributeError: '_Tracer' object has no attribute '_inductor'
  File ".../torch/_inductor/config.py", line 300, in <module>
    post_grad_custom_pre_pass: torch._inductor.custom_graph_pass.CustomGraphPassType = None
```

원인: `object.__getattribute__` 는 `ModuleType` 의 실제 `tp_getattro`(`module_getattro`,
PEP 562 의 모듈 레벨 `__getattr__` 폴백을 구현하는 C 함수)와 **다릅니다.** 상류 torch 는
`torch/_inductor/__init__.py` 가 아직 실행 중인 상태에서(자기 패키지가 `torch` 의 속성으로
아직 안 붙은 시점에) `torch._inductor.*` 를 참조하는 순환 자기참조 패턴을 쓰고, 이게
정상 동작하는 것은 `module_getattro` 의 폴백 경로 덕분입니다. `object.__getattribute__` 로
바꿔치면 이 폴백이 사라져 **실제 torch import 가 깨집니다.** `super().__getattribute__(name)`
로 고치자(이러면 `ModuleType` 의 진짜 구현을 타므로) 정상화됐습니다. **이 버그 자체가
계측이 무언가를 건드리지 않는 게 아니라 실제로 개입하고 있다는 실측 증거**입니다 — 아무
일도 안 했다면 이런 식으로 깨질 수 없습니다.

**(b) `torch.Tensor` (클래스, `TensorBase` 의 실사용 경로)** — 진짜 `TensorBase` 는
`Py_TPFLAGS_IMMUTABLETYPE` 가 걸린 면역 타입이라 `__getattribute__` 를 못 바꿉니다(실측):

```python
>>> torch._C.TensorBase.__getattribute__ = f
TypeError: cannot set '__getattribute__' attribute of immutable type 'torch._C.TensorBase'
>>> type(torch._C.TensorBase).__getattribute__ = f   # 메타클래스도 마찬가지
TypeError: cannot set '__getattribute__' attribute of immutable type 'torch._C._TensorMeta'
```

반면 `torch.Tensor` (`torch/_tensor.py` 의 평범한 파이썬 `class Tensor(TensorBase)`) 는
mutable 입니다. 실사용은 전부 `torch.Tensor` 인스턴스를 거치므로,
`torch.Tensor.__getattribute__` 를 원본을 감싸는 함수로 바꿔 관찰합니다.

```python
_orig = torch.Tensor.__getattribute__
def _traced(self, name):
    record("torch.Tensor", name)
    return _orig(self, name)
torch.Tensor.__getattribute__ = _traced
```

**주의: 이걸로도 절반만 잡힙니다.** `x + y` 나 `x @ w` 같은 연산자는 CPython 이 암시적
특수 메서드 디스패치를 할 때 **인스턴스의 `__getattribute__` 를 거치지 않고**
`type(obj).__mro__` 에서 슬롯을 직접 찾습니다. 그래서 `+`·`@`·`==`·`[]` 등은 위 후킹에
안 걸립니다. `torch.Tensor` 는 mutable 이므로, `__add__`·`__matmul__`·`__getitem__` 등
45 개 연산자 던더에 개별로 같은 방식의 얇은 래퍼를 추가로 얹었습니다(원본은
`TensorBase.__dict__` 에서 읽기만 하므로 면역 타입이어도 읽는 것은 허용됩니다).

**(c) `_C._VariableFunctions`** — 이것도 면역 타입의 인스턴스(`_VariableFunctionsClass`)라
직접 후킹이 안 됩니다. 대신 상류 구조를 역이용합니다: `torch/__init__.py` 의 한 루프가
임포트 시점에 `torch.<opname> = _C._VariableFunctions.<opname>` 로 **전량 호이스팅**합니다
(VENDOR.md 벽 5). 어떤 이름이 호이스팅됐는지는 `getattr(torch, name) is getattr(VF, name)`
로 사후 판별할 수 있고, 그 이름이 **실제로 호출**됐는지는 `torch.<name>` 자리에 얇은 카운팅
wrapper 를 심어(609 개 전부, 실패 0) CALL 시점만 기록하면 됩니다. — 이게 필요했던 이유는
§3 에서 설명합니다(단순 getattr 로그는 이 표면에서 오염돼 못 씁니다).

### 1-3. 검증 — 알려진 접근이 실제로 잡히는지 확인

작업 지시대로, 본 측정을 시작하기 전에 **알려진 접근**(`torch._C._get_tracing_state`)이
계측에 걸리는지 먼저 확인했습니다.

```
$ /Volumes/macMini/caches/spike-venv/bin/python /tmp/bw_validate_hook.py
torch._C class after import: <class '__main__.wrap_module_class.<locals>._Tracer'>
known access already logged for torch._C: 995
was _get_tracing_state already seen (should be False or True, just info): True
VALIDATION _get_tracing_state captured after explicit call: True
VALIDATION PASSED
```

`_get_tracing_state` 는 **강제로 부르기도 전에 이미 로그에 있었습니다** — 즉 `import
torch` 자체가 이 이름에 내부적으로 접근합니다. 그 뒤 명시적으로 `torch._C.
_get_tracing_state()` 를 불러도 여전히 잡히는 것으로, 계측이 "우연히 이미 있던 로그"가
아니라 **매 접근마다 실제로 개입**한다는 것을 재확인했습니다. 본 측정 스크립트에도 같은
assert 를 넣어 매 실행마다 자동 재검증하도록 했습니다(`[validate] ... OK` 로그).

**부가 확인 (약한 증거, §7-5 참고):** 실측 중 CUDA·NCCL·XPU 전용 이름(`_cuda_init`,
`_nccl_all_reduce` 등) 102 개가 `dir(torch._C)` 에는 없는데도 로그에 나타났습니다. 이
머신은 CPU 전용 빌드라 이 이름들은 **존재하지 않고**, 그런데도 로그에 찍혔다는 것은
`hasattr()` 류의 **실패하는 조회 시도**까지 잡힌다는 뜻입니다(우리 `__getattribute__` 는
`record()` 를 먼저 부르고 나서 위임하므로, 위임이 `AttributeError` 로 끝나도 기록은 이미
남습니다). 이건 목적한 검증은 아니지만 방법의 포괄성을 보강하는 관찰입니다.

---

## 2. 단계 정의와 파이프라인

하나의 프로세스 안에서 순서대로 실행하고, 매 단계 경계마다 누적 스냅샷을 남겨 델타를
계산했습니다. 환경은 `/Volumes/macMini/caches/spike-venv/bin/python` (torch 2.13.0,
transformers 5.15.1, CPython 3.13.0).

| 단계 | 내용 |
|---|---|
| 1 | `import torch` |
| 2 | `import transformers` + `from transformers import AutoModelForCausalLM, LlamaConfig` |
| 3 | `LlamaConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, intermediate_size=128, vocab_size=100)` 로 `AutoModelForCausalLM.from_config(cfg)` |
| 4 | `model(input_ids)` (순전파) + `model.generate(input_ids, max_new_tokens=4, do_sample=False)` |

`torch.Tensor`/연산자 던더 후킹과 `_VariableFunctions` 호출 카운터는 각각 **1 단계 완료
직후**, **2 단계 완료 직후**부터 설치했습니다 — 따라서 이 두 표면에 대해서는 1 단계(그리고
`_VariableFunctions` 는 2 단계)의 데이터가 없습니다. 이유와 영향은 §7 에 명시했습니다.

측정은 **단독 실행**으로 돌렸습니다(`uptime` load average 1.0~1.9, 8 코어 기준 여유 있음).
모델은 랜덤 초기화(시드 고정 안 함)지만 `do_sample=False` 그리디 디코딩이라 **접근되는
이름 집합** 자체는 가중치 값에 의존하지 않습니다 — 다만 반복 실행으로 안정성을 재확인하지는
않았습니다(§7-6).

---

## 3. `torch._C` — dir() 표면

| 단계 | 누적 접근(원시 로그) | dir() 과 교집합 | 실패 프로브 잡음 |
|---|---|---|---|
| 1 import torch | 995 | **979 / 989 (99.0%)** | 16 |
| 2 import transformers | 1081 | 979 / 989 (변화 없음) | 102 |
| 3 from_config | 1081 | 979 / 989 (변화 없음) | 102 |
| 4 forward+generate | 1081 | 979 / 989 (변화 없음) | 102 |

**핵심 발견: `torch._C` 의 dir() 표면 989 개 중 979 개(99.0%)가 `import torch` 단독으로
이미 다 접근됩니다.** 그 뒤 transformers·모델 생성·순전파·generate 를 다 거쳐도 **단 하나도
늘지 않았습니다.** 즉 이 표면에 관한 한, 병목은 "추론이 무엇을 요구하는가" 가 아니라
"`import torch` 자체가 무엇을 요구하는가" 입니다 — VENDOR.md 가 기록 모드로 잰 "임포트가
93% 까지 간다" 는 것과 같은 결의 결과이지만, **이번엔 진짜 torch 라서 100% 임포트가
실제로 끝난 뒤의 그림**이라는 점이 다릅니다.

**"접근" 잡음의 정체 (102 개, dir() 에 없는데 로그에 있는 이름):** 전부
`_cuda_*`·`_nccl_*`·`_xpu_*`·`_cudnn_*`·`_cusparselt_*`·`_itt`·`_nvtx`·`_cudart` 류입니다.
이 머신은 CPU 전용이라 이 이름들은 `torch._C` 에 **존재하지 않고**, torch 내부의
`hasattr(torch._C, "_cuda_init")` 류 가속기 감지 코드가 조회를 **시도**했다가 실패한
흔적입니다. 989 라는 분모에 넣으면 안 되는 것들이라 표에서 분리했습니다.

**4 단계 내내 한 번도 안 닿은 10 개** (`dir(torch._C)` 에는 있지만 로그에 전혀 없음):

```
_DisableAutocast  _DisableFuncTorch  _EnableTorchFunction  _InferenceMode
_RestorePythonTLSSnapshot  _autograd  _distributed_autograd
_xpu_beginAllocateCurrentThreadToPool  _xpu_endAllocateToPool  _xpu_releasePool
```

앞의 다섯은 컨텍스트 매니저류(autocast·functorch·torch_function 비활성화 등, 이번 그리디
추론 경로에서 안 씀), 뒤의 세 개는 XPU(Intel GPU) 전용입니다. **이번 실행 한 번의 결과이지
"영원히 불필요" 라는 뜻은 아닙니다** — 다른 모델·다른 경로(예: `torch.inference_mode()` 를
쓰는 코드)는 이 중 일부를 요구할 수 있습니다.

**미확인:** 979 개 중 실제로 **호출**(콜러블인 것이 실행)까지 되는 것은 몇 개인지는
이번 측정에서 재지 않았습니다 — `_VariableFunctions` 처럼 개별 call-wrapper 를 씌우지
않았기 때문입니다(§7-1).

---

## 4. `TensorBase` (실사용은 `torch.Tensor` 인스턴스를 통해)

| 단계 | 누적(교집합, `dir(TensorBase)` 대비) | 이번 단계에 새로 늘어난 이름 |
|---|---|---|
| 1 import torch | 0 (후킹 미설치) | — |
| 2 import transformers | 0 (후킹은 1 단계 직후 설치했지만, transformers 임포트 자체가 텐서 인스턴스를 안 건드림) | — |
| 3 from_config | 18 | `__class__`\*, `__mul__`, `__truediv__`, `clone`, `copy_`, `detach`, `device`, `dim`, `fill_`, `grad_fn`, `normal_`, `reciprocal`, `requires_grad`, `requires_grad_`, `shape`, `size`, `to`, `uniform_` |
| 4 forward+generate | **50 / 694 (7.2%)** | `__add__`, `__and__`, `__bool__`, `__eq__`, `__getitem__`, `__invert__`, `__lt__`, `__matmul__`, `__or__`, `__sub__`, `any`, `contiguous`, `cos`, `cumsum`, `dtype`, `expand`, `float`, `long`, `masked_fill`, `max`, `mean`, `ndim`, `ne`, `new_ones`, `numel`, `pow`, `reshape`, `sin`, `sum`, `transpose`, `unsqueeze`, `view` |

\* `__class__` 는 `dir(TensorBase)` 에 `object` 로부터 상속되어 잡힌 던더로, 내부
isinstance/repr 기계장치가 만드는 잡음입니다 — 진짜 API 표면으로 세면 49 개입니다.

**이 목록은 그럴듯합니다** — RoPE 용 `cos`/`sin`, RMSNorm 용 `pow`+`mean`+`reciprocal`
(rsqrt 는 `_VariableFunctions` 쪽, §5), 어텐션 마스킹용 `masked_fill`/`ne`/`any`, 텐서
초기화용 `normal_`/`uniform_`/`fill_`, 형태 조작용 `view`/`reshape`/`transpose`/`expand`/
`unsqueeze`, 그리고 연산자 던더로 잡힌 `__matmul__`(어텐션 행렬곱)·`__add__`(잔차 연결)까지
— 실제 Llama 순전파가 쓸 법한 것들과 정확히 일치합니다. 이 정합성 자체가 계측이 잡음이
아니라 진짜 실행 경로를 보고 있다는 방증입니다.

**VENDOR.md §3 의 543/694 와 헷갈리지 마십시오 — 다른 것을 잰 겁니다.** VENDOR 의 543 은
"**빈 shim** 으로 `import torch` 를 시도했을 때, `torch/_tensor.py` 의 `class Tensor
(TensorBase)` **클래스 본문**이 (아직 인스턴스 하나 없이) `_C._add_docstr(_C.TensorBase.
<name>, ...)` 패턴으로 요구한 이름의 수"입니다. 클래스 본문 실행은 `TensorBase` 의
**메타클래스** 를 거치는데, 그 메타클래스도 면역 타입이라 이번 방법으로는 못 잡습니다
(§7-2). 이번 50 은 "**진짜 torch** 로 실제 추론을 돌렸을 때, 텐서 **인스턴스**가 실행
중에 실제로 쓴 이름의 수"입니다. 두 숫자를 더하거나 비교해서 결론 내리면 안 됩니다 —
**543 은 "존재해야 클래스 정의가 성공한다", 50 은 "그중 이번 추론이 실제로 실행한다"**로,
서로 다른 질문에 대한 답입니다.

---

## 5. `_VariableFunctions` — "접근" 과 "호출" 이 극단적으로 갈리는 경우

| 단계 | getattr 로 접근됨 (오염, 아래 참고) | **실제로 호출됨** (누적) | 이번 단계에 처음 호출된 이름 |
|---|---|---|---|
| 1 import torch | 0 (후킹 미설치) | — | — |
| 2 import transformers | **607 / 609 (99.7%)** | 0 (후킹은 2 단계 직후 설치) | — |
| 3 from_config | 609 / 609 (100%) | **5 / 609** | `arange`, `empty`, `ones`, `pow`, `tensor` |
| 4 forward+generate | 609 / 609 (변화 없음) | **13 / 609 (2.1%)** | `argmax`, `cat`, `embedding`, `full`, `is_floating_point`, `isin`, `randint`, `rsqrt` |

(4 단계에서 이번 단계에 실제로 불린 이름 전체 — 재호출 포함: `arange`, `argmax`, `cat`,
`embedding`, `full`, `is_floating_point`, `isin`, `ones`, `randint`, `rsqrt`, `tensor`.)

### 왜 "접근" 수치(607/609, 99.7%)를 그대로 믿으면 안 되는가

처음에는 `torch` 모듈 자체의 getattr 로그와 호이스팅된 이름 집합의 교집합으로 이 수치를
냈는데, **`import transformers` 하나만으로 609 개 중 607 개가 찍혔습니다.** 원인을
스택트레이스로 추적했습니다:

```
transformers/generation/utils.py -> from ..masking_utils import create_masks_for_generate
  masking_utils.py -> from torch._dynamo._trace_wrapped_higher_order_op import ...
    torch/_dynamo/__init__.py -> from .polyfills import loader
      .../polyfills/loader.py -> importlib.import_module(f".{submodule}", ...)
        .../polyfills/_collections.py -> @substitute_in_graph(...)
          torch/_dynamo/decorators.py -> get_torch_obj_rule_map()
            torch/_dynamo/trace_rules.py:3059 -> load_object(k)
              trace_rules.py:3079 -> getattr(importlib.import_module(module), obj_name)
```

`torch._dynamo.trace_rules.get_torch_obj_rule_map()` 이 `torch.compile` 을 위한
**그래프 트레이싱 규칙 테이블**을 임포트 시점에 구축하면서, 공개 `torch.*` API 를 알파벳
순으로(`abs`, `abs_`, `absolute`, `acos`, ... — 실측 로그가 정확히 이 순서였습니다) 거의
전부 `getattr` 로 훑습니다. **이건 실제 사용이 아니라 dynamo 내부 부기입니다.** 그리고
이건 `transformers.masking_utils` 가 `torch._dynamo` 를 끌어오기 때문에 벌어지는 일이라,
**`torch.compile` 을 한 번도 쓰지 않아도** 순전히 `from transformers import
AutoModelForCausalLM` 만으로 발생합니다 — VENDOR.md §3 벽 15·16 이 지적한 "autograd·jit 은
꺼지지 않는다" 는 것과 같은 결의 발견입니다: **dynamo 도 안 씁니다 라고 선언할 수 없고,
임포트 그래프에 딸려 옵니다.**

### 그래서 CALL 만 세는 별도 계측을 추가했다

`torch.<name>` 자리(609 개 전부, 실패 0)에 원본을 감싸는 얇은 카운팅 wrapper 를 심어
**진짜 호출된 순간만** 기록했습니다:

```python
def _make_call_wrapper(name, real_fn):
    def _wrapper(*args, **kwargs):
        CALLED.setdefault(CURRENT_STAGE, set()).add(name)
        return real_fn(*args, **kwargs)
    return _wrapper
for name in hoisted:
    setattr(torch, name, _make_call_wrapper(name, getattr(torch, name)))
```

결과가 **13/609 (2.1%)** — dynamo 스윕이 만든 99.7% 와는 완전히 다른 이야기를 합니다.
`argmax` 는 그리디 디코딩, `embedding` 은 토큰 임베딩 조회, `cat`/`full`/`arange` 는
KV 캐시·포지션·마스크 구성, `rsqrt` 는 RMSNorm, `isin`/`is_floating_point` 는 `generate()`
내부 방어 체크로 보입니다 — 역시 그럴듯합니다.

**이 13 개도 하한입니다.** `x.sum()` 처럼 텐서 **메서드**로 호출되는 연산(`sum`·`mean`·
`max`·`pow`·`cos`·`sin` 등, §4 의 50 개 목록에 이미 있음)은 `_VariableFunctions` 를 거치지
않고 `TensorBase` 경로로 잡힙니다 — 같은 ATen 커널이라도 파이썬에서 **어느 문으로
들어오는지**에 따라 이 표에는 안 잡힙니다. 그리고 이 wrapper 는 **`torch.<name>` 심볼을
거치는 파이썬 레벨 호출만** 잡습니다 — C++ 디스패치나 텐서 연산자(`+`, `@`)를 통해 내부적으로
같은 ATen 커널이 불려도 이 카운터에는 안 잡힙니다(그건 §4 의 연산자 던더 계측이 별도로
잡습니다).

---

## 6. 989 · 694 · 985(625/609) 대비 정리

| 표면 | 분모 | 접근됨(존재 확인) | 호출됨(실제 실행) | 호출됨 비율 |
|---|---|---|---|---|
| `torch._C` dir() | 989 | 979 (99.0%) | 미측정 | 미측정 |
| `TensorBase` (via Tensor 인스턴스) | 694 | — (접근=사용, 아래 참고) | **50** | **7.2%** |
| `_VariableFunctions` (호이스팅된 것) | 609 (전체 985 중) | 609 (100%, 오염) | **13** | **2.1%** |

`TensorBase` 는 getset 디스크립터·메서드가 대부분이라 "읽음" 이 곧 "그 값/동작이 필요함"과
거의 같습니다(속성을 읽었는데 그 결과를 안 쓰는 경우는 드뭅니다) — 그래서 "접근"과 "호출"을
따로 안 나눴습니다. 반면 `_VariableFunctions` 는 §5 에서 본 것처럼 "읽기"(dynamo 부기)와
"호출"(실행)이 크게 갈리므로 반드시 나눠 봐야 합니다.

**한 문장 요약:** `_C` 의 존재 표면(989 개 이름·32 서브모듈)은 거의 전부(99%) `import
torch` 하나로 이미 요구되지만, **그 표면이 실제로 "일" 을 하는 부분은 훨씬 좁습니다** —
텐서 연산은 7%, 함수형 연산자는 2% 남짓입니다. 이건 IMPORT_WALLS 5 차가 파이썬 모듈
수준에서 낸 "추론 중 실행되는 모듈은 1084 개 중 14 개(1.3%)" 와 **같은 모양의 격차**가
`_C` 표면 안에도 있다는 뜻입니다 — 다만 이번엔 "존재 요구" 자체가 이미 99%로 훨씬 세다는
점이 다릅니다.

---

## 7. 구현 우선순위 제안

측정 결과를 그대로 뒤집으면 우선순위가 나옵니다 — **"존재만 하면 되는 것" 과 "제대로
동작해야 하는 것" 을 분리**하는 것이 핵심입니다.

### 1 순위 — 이번 실행에서 실제로 호출된 것 (동작까지 필요)

- **`TensorBase` 50 개** (§4 목록). 산술 연산자 던더 10 개(`__add__`·`__matmul__` 등)와
  형태 조작·통계·활성 함수류가 섞여 있습니다. **작은 Llama 순전파+그리디 생성 하나를 그대로
  돌리려면 이 50 개가 정확히 동작해야 합니다.**
- **`_VariableFunctions` 13 개** (§5 목록). `embedding`·`rsqrt`·`argmax` 등, 파이썬 레벨
  `torch.<op>()` 호출로 들어오는 것들.
- 이 둘의 합집합(중복 없이, 개념적으로 겹치는 커널이 있어도 진입 경로가 다르므로 각각
  구현 필요)이 **"이번 데모가 실제로 실행되기 위한 최소 동작 표면"** 입니다.

### 2 순위 — 존재는 확인되지만 아직 호출되지 않은 것

- `torch._C` dir() 의 나머지 (979 − 위 표면과 겹치는 것). **"존재 확인용 스텁"**(호출되면
  타입 오류를 내도 되는 더미 값/타입)으로 충분할 가능성이 높습니다 — `hasattr`/`isinstance`/
  `getattr` 을 만족시키는 것이 목적이지 실제 계산이 아니기 때문입니다. 다만 VENDOR.md 벽
  8·9·12 (서브모듈이 패키지여야 함, 메타타입이 타입마다 달라야 함)는 "존재"의 기준이
  단순 값이 아니라 **타입 모양**이라는 것을 이미 보였으므로, 스텁도 그 제약은 지켜야 합니다.
- `_VariableFunctions` 의 나머지(596 개, 609 − 13). dynamo 규칙 테이블 구축 때문에
  **"존재해야" 하는 건 맞지만**, 그 자체는 getattr 만 성공하면 되므로 우선순위는 낮습니다.

### 3 순위 — 이번 실행에서 아예 닿지 않은 것

- `torch._C` 의 10 개(§3) — autocast/functorch/torch_function 비활성화 컨텍스트, XPU
  풀 관리. **다른 모델·다른 추론 경로가 필요로 할 수 있으므로 "영구 미구현" 결정은 이르지만,
  이번 데모 통과에는 불필요.**

### 제안이 기대는 전제와 그 한계

이 우선순위는 **"작은 Llama, 그리디 디코딩, eval 모드, 배치 1, 4 토큰"** 이라는 매우 좁은
경로 하나에서 나왔습니다. 다른 아키텍처(어텐션 변형·양자화)·학습 모드(`requires_grad=True`
+ `backward()`)·배치 처리·긴 시퀀스는 다른 이름을 요구할 것이 거의 확실합니다. **이 50 +
13 은 "충분 집합"이 아니라 "이 경로의 필요 최소 집합"입니다.** 실제 구현 우선순위를 정할
때는 torchnative 가 지원하려는 모델·경로 각각에 대해 같은 계측을 반복해 합집합을 넓혀가야
합니다.

---

## 8. 미확인 (추측으로 채우지 않음)

| # | 항목 | 왜 미확인인가 |
|---|---|---|
| 1 | **`torch._C` 979 개 중 실제로 "호출"(콜러블 실행)되는 것의 수** | `_VariableFunctions` 처럼 개별 call-wrapper 를 씌우지 않았습니다. `_C` 표면은 타입·서브모듈·상수·함수가 섞여 있어 "호출" 개념이 균일하게 적용되지 않는다는 점도 있습니다(타입은 "쓰임"이 곧 isinstance/서브클래싱이지 호출이 아닙니다). |
| 2 | **`_C.TensorBase.<name>` 클래스 레벨 접근** (`torch/_tensor.py` 의 클래스 본문이 `_add_docstr` 로 읽는 것) | `TensorBase` 의 메타클래스(`_TensorMeta`)도 면역 타입이라 이 경로는 이번 방법으로 못 잡습니다. VENDOR.md §3(543/694)이 **다른 방법**(빈 shim + record 모드)으로 잰 것을 그대로 참조해야 하며, 이번 50 과 합산하면 안 됩니다(§4 참고). |
| 3 | **`_C._VariableFunctions.<name>` 자체에 대한 직접 접근** (호이스팅을 거치지 않는 경로가 있는지) | `_VariableFunctionsClass` 도 면역 타입이라 직접 후킹 불가. `torch.<name>` 호이스팅 집합과의 교집합으로 **간접 추론**만 했습니다. |
| 4 | **다른 모델/경로**: 양자화, 배치>1, 긴 시퀀스, KV 캐시 재사용을 포함한 다회 `generate()`, 학습 모드(backward 포함) | 이번 실행은 위 §7 "제안이 기대는 전제" 그대로 하나의 좁은 경로만 돕니다. |
| 5 | **`hasattr` 류 실패 조회가 "전부" 잡히는지** | §1-3 의 CUDA/NCCL/XPU 102 개는 이 방법이 실패하는 조회도 잡는다는 근거지만, 목적을 갖고 검증한 것은 아니라 "약한 증거"로만 씁니다. |
| 6 | **반복 실행 안정성** | 1 회만 실행했습니다. `do_sample=False` 라 접근되는 **이름 집합**은 랜덤 시드에 안 의존할 것으로 보이지만(가중치 값이 아니라 어떤 연산이 실행되는지가 이름 집합을 결정), 반복 실행으로 재확인하지 않았습니다. |
| 7 | **진짜 torch 위에서 측정한 이 필요 집합이, 우리 shim 으로 실제 대체했을 때도 그대로 필요조건인지** | 이번 측정은 진짜 torch 의 코드 경로를 관찰한 것입니다. 우리 shim 이 어떤 기능을 빠뜨리면 상류 파이썬 트리가 **다른 분기**(예: `hasattr` 로 켜고 끄는 서브시스템, VENDOR.md 벽 11)를 타면서 요구 이름 집합 자체가 달라질 수 있습니다. |

---

## 9. 재현

측정 스크립트는 `/tmp/bw_measure_c_surface.py` (검증 스크립트는 `/tmp/bw_validate_hook.py`)
에 있습니다 — 지시대로 저장소에는 커밋하지 않았고 `/tmp` 는 휘발성이므로, 이 문서의 §1 코드
스니펫이 재현에 필요한 핵심 로직 전부입니다. 실행:

```bash
/Volumes/macMini/caches/spike-venv/bin/python /tmp/bw_measure_c_surface.py
```

결과 JSON은 `/tmp/bw_c_surface_report.json` 에 씁니다(마찬가지로 휘발성). 표의 숫자는 전부
이 리포트를 직접 세어 옮긴 것이며, 어림값은 없습니다.
