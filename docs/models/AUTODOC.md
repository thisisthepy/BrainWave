# `@auto_docstring` 이 클래스 정의 시점에 정말 무엇을 요구하는가

IMPORT_WALLS.md **2 차**가 "범주 6 — 런타임 코드 생성" 으로 기록한 벽의 사전 분석입니다.
그 벽은 스텁 `torch` 로 `modeling_llama.py:347` 의 `@auto_docstring class LlamaModel(...)` 을
정의하려다 `TypeError: function() argument 'code' must be code, not str` 로 죽었고, 문서는
"이것은 스텁을 정교하게 만들어 넘을 수 있는 벽이 아니다" 라고 결론짓고 넘어갔습니다. **그 결론은
검증이 아니라 추론이었습니다.** 이 문서는 그것을 실측으로 검증합니다.

환경: `/Volumes/macMini/caches/spike-venv/bin/python` (CPython 3.13.0), transformers **5.15.1**
(IMPORT_WALLS 1 차와 동일 버전 — 버전 차이로 인한 재현 실패가 아님을 먼저 배제합니다),
torch **2.13.0** 실물.

---

## 결론 먼저

**이 벽은 우리 경로에서 실제 문제가 아닙니다 — 현재 확인 기준으로는.**

`@auto_docstring` 의 클래스 데코레이션 경로(`auto_class_docstring`/`auto_method_docstring`,
`transformers/utils/auto_docstring.py`)는 **함수나 클래스를 다시 만들지 않습니다.** `__doc__`
속성을 대입할 뿐입니다. `types.FunctionType` 을 호출하는 코드는 이 경로 안에 전혀 없습니다.

2 차가 만난 `TypeError` 의 유일한 서식지인 `transformers.utils.doc.copy_func` 는 **`auto_docstring`
과 무관한 다른 아홉 곳**(`push_to_hub` 재바인딩, `AutoModelForXxx.from_config`/`from_pretrained`
재바인딩)에서만 불리고, 그 아홉 곳은 전부 transformers 자신이 손으로 쓴 함수를 복사하는 것이지
**torch 객체를 복사하지 않습니다.** 실측으로 확인했습니다(아래 §3).

그리고 category 1~5 요구사항(가용성 게이트 · 서브모듈 존재 · 어노테이션에 쓰이는 실제 타입 ·
상속 가능한 클래스 · 호출 가능한 데코레이터)만 만족하는 **일반적인 이름-흉내 스텁**을 새로 만들어
`modeling_llama.py` 전체(`LlamaModel` · `LlamaForCausalLM` · `LlamaForSequenceClassification` ·
`LlamaForQuestionAnswering` · `LlamaForTokenClassification`)를 임포트했더니, **실물 torch 로 돌린
결과와 `__doc__` 길이가 한 글자도 다르지 않게 통과했습니다** (아래 §4). 즉 category 6 은 category
1~5 를 넘어서는 추가 요구를 만들지 않습니다.

2 차가 실제로 본 `TypeError` 가 어디서 왔는지는 원본 스크립트가 남아있지 않아 **확정할 수 없습니다.**
다만 유력한 메커니즘은 짚을 수 있습니다(§5).

---

## 1. `auto_docstring` 이 정확히 무엇을 하는가

진입점은 `transformers/utils/auto_docstring.py:4517`:

```python
def auto_docstring(obj=None, *, custom_intro=None, custom_args=None, checkpoint=None):
    def auto_docstring_decorator(obj):
        if len(obj.__qualname__.split(".")) > 1:
            return auto_method_docstring(...)      # 메서드에 붙은 경우
        else:
            return auto_class_docstring(...)        # 클래스에 붙은 경우 (LlamaModel 이 이쪽)
    ...
```

(`:4664-4670`)

`LlamaModel` 은 클래스이므로 `auto_class_docstring(cls, ...)` (`auto_docstring.py:4281`) 을 탑니다.
그 함수가 하는 일은 다음 세 단계뿐입니다.

1. **`cls.__mro__` 를 문자열로 훑어 종류를 분류**합니다 (`"PreTrainedModel" in (x.__name__ for x in
   cls.__mro__)` 같은 이름 비교, `:4295`). `LlamaModel` 은 `PreTrainedModel` 계열이므로 이 분기를
   탑니다.
2. **`auto_method_docstring(cls.__init__, parent_class=cls, ...)` 을 호출**해 `__init__` 의 문서
   조각을 얻습니다 (`:4296-4298`). `auto_method_docstring` (`auto_docstring.py:4207`) 안에서 유일한
   "진짜 introspection" 은 `sig = inspect.signature(func)` (`:4221`) 뿐이고, 함수 끝에서
   `func.__doc__ = docstring; return func` (`:4277-4278`) — **입력받은 함수 객체를 그대로 반환**
   합니다. 복사도, 재구성도 없습니다.
3. **문자열을 조립해 `cls.__doc__ = docstring` 대입 후 `return cls`** (`:4512, 4514`) — 역시 클래스
   객체를 그대로 반환합니다.

**어디에도 `copy_func`, `types.FunctionType`, `functools.update_wrapper` 호출이 없습니다.**
`auto_docstring.py` 전체(4675 행)를 grep 해도 나오지 않습니다. `inspect.signature` 는 함수를
**읽기만** 합니다 — 새 함수를 만들지 않습니다.

## 2. `copy_func`/`types.FunctionType` 은 어디 있고, 무엇을 복사하는가

`types.FunctionType` 을 호출하는 곳은 transformers 전체에서 **딱 한 곳**입니다
(`grep -rn "types.FunctionType\|FunctionType(" transformers/` — torch 자신을 제외하면 다른 곳은
없습니다).

```python
# transformers/utils/doc.py:1085-1090
def copy_func(f):
    """Returns a copy of a function f."""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                            argdefs=f.__defaults__, closure=f.__closure__)
    g = cast(types.FunctionType, functools.update_wrapper(g, f))
    ...
```

이걸 부르는 곳은 아홉 군데이고, **전부 transformers 자신이 정의한 메서드를 복사하는 모듈 최상위
코드**입니다. `LlamaModel` 의 클래스 본문이나 `@auto_docstring` 과는 무관합니다.

| 파일:행 | 무엇을 복사하는가 |
|---|---|
| `configuration_utils.py:1475` | `PreTrainedConfig.push_to_hub` |
| `tokenization_utils_base.py:3649` | `PreTrainedTokenizerBase.push_to_hub` |
| `image_processing_base.py:490` | `ImageProcessingMixin.push_to_hub` |
| `feature_extraction_utils.py:668` | `FeatureExtractionMixin.push_to_hub` |
| `processing_utils.py:2356` | `ProcessorMixin.push_to_hub` |
| `video_processing_utils.py:767` | `BaseVideoProcessor.push_to_hub` |
| `modeling_utils.py:4933` | `PreTrainedModel.push_to_hub` |
| `pipelines/base.py:1303` | `Pipeline.push_to_hub` |
| `auto_factory.py:489, 498` | `_BaseAutoModelClass.from_config` / `.from_pretrained` (Auto* 클래스마다 등록 시) |

이 아홉 개는 전부 `PushToHubMixin`(파일 유틸)이나 `auto_factory.py` 의 손으로 쓴 파이썬 함수입니다.
정의부가 torch 를 참조하는지 여부와 무관하게, **컴파일된 `__code__` 는 항상 transformers 소스에서
나온 진짜 code object 입니다.**

## 3. 실측 — `LlamaModel` 을 임포트하는 동안 `copy_func` 는 몇 번, 무엇에 불리는가

`transformers.utils.doc.copy_func` 를 트레이싱 래퍼로 감싼 뒤(원본은 그대로 위임),
실물 torch 2.13.0 환경에서 `from transformers.models.llama.modeling_llama import LlamaModel` 을
실행했습니다(스크립트는 `/tmp/autodoc_probe/trace_functype2.py`, 커밋 대상 아님).

```
== import transformers (should be lazy for model submodules) ==
transformers imported. modeling_utils in sys.modules? False
modeling_llama in sys.modules? False
== now importing modeling_llama (triggers modeling_utils etc.) ==
OK: LlamaModel imported successfully, __doc__ len= 883
== copy_func call count: 87
```

87 회 전부의 호출 스택을 확인했고, 예외 없이 위 표의 아홉 자리(`push_to_hub` 재바인딩,
`_BaseAutoModelClass.from_config`/`from_pretrained` 를 `models/auto/modeling_auto.py` 의
`auto_class_update` 루프가 Auto* 클래스마다 반복 호출하는 것) 중 하나였습니다. **`LlamaModel` 의
클래스 본문이나 `@auto_docstring` 데코레이터 실행 프레임에서 발생한 호출은 0 건입니다.**

## 4. 재현 — 일반적인 이름-흉내 스텁으로 `modeling_llama.py` 전체를 통과시켜 봤다

2 차가 쓴 원본 프로브 스크립트는 남아있지 않습니다(`vendor/probe.py` 는 3~4 차가 **실물 torch +
우리 `_C` 셔플**을 재는 별개의 도구이고, 스텁-torch 방식이 아닙니다). 그래서 category 1~5 가 정리한
요구사항만 따라 새로 하나 만들어 재현을 시도했습니다. 코드는 `/tmp/stub_env/`(커밋 대상 아님).

구성:

- `sys.meta_path` 에 파인더를 꽂아 `torch` 와 `torch.*` 를 **전부** 자동 생성 모듈로 만듭니다
  (category 3 — "서브모듈은 파일이 존재해야 함" 을 파일 대신 meta path 로 만족).
- 모듈의 없는 속성에 접근하면, 이름이 대문자로 시작하면 **진짜 `type()` 클래스**를 동적으로 만들어
  돌려주고(caching), 아니면 **호출하면 자기 자신을 돌려주는 진짜 함수**(`def _probe(*a, **kw): return
  _probe`)를 돌려줍니다.
  - 전자가 category 2(`eos_token_id: int | list[int] | torch.Tensor` 같은 어노테이션에 쓰이는 진짜
    타입)와 category 4(`class _AllReduceBackward(torch.autograd.Function)` 같은 상속)를 만족시킵니다
    — `type` 인스턴스는 `|` 로 유니온을 만들 수 있고 상속도 됩니다.
  - 후자가 category 5(`@torch.no_grad()` — 호출한 결과가 다시 호출 가능해야 함)를 만족시킵니다.
- `importlib.metadata.version("torch")` 를 몽키패치해 `"2.9.0"` 을 돌려줘 category 1(가용성 게이트)
  을 만족시킵니다.

이 스텁으로:

```
is_torch_available: True
OK: LlamaModel defined with stub torch. doc len= 883
```

전체 모듈(`import transformers.models.llama.modeling_llama`)까지 확장해도 동일합니다.

| 클래스 | 실물 torch 2.13.0 | 이번 제너릭 스텁 |
|---|---|---|
| `LlamaModel` | 883 | 883 |
| `LlamaForCausalLM` | 850 | 850 |
| `LlamaForSequenceClassification` | 0 | 0 |
| `LlamaForQuestionAnswering` | 0 | 0 |
| `LlamaForTokenClassification` | 0 | 0 |

`__doc__` 길이가 **바이트 단위로 일치**합니다. (0 인 세 개는 실물에서도 0 입니다 — 그 클래스들은
`ClassDocstring` 레지스트리에 등록되어 있지 않아 `auto_class_docstring` 이 빈 문자열을 넣고 지나가는
경로이고, 실물/스텁 둘 다 같은 코드를 타므로 당연히 같습니다.)

**이것이 이 문서의 핵심 실측입니다.** category 1~5 만 만족하는 스텁이 `@auto_docstring` 이 걸린
클래스 정의를 실물과 구분 불가능하게 통과시켰습니다. category 6 이 별도의 요구를 추가한다는 증거를
찾지 못했습니다.

## 5. 그럼 2 차가 본 `TypeError` 는 뭐였나 — 미확인, 그러나 유력한 메커니즘은 있다

메시지를 직접 재현해 확인했습니다.

```python
>>> import types
>>> types.FunctionType('not-a-code-object', {})
TypeError: function() argument 'code' must be code, not str
```

이 메시지는 `types.FunctionType(...)` 의 **첫 인자가 code object 가 아닐 때만** 나옵니다. 이
프로세스 전체(transformers, torch 제외 site-packages)를 뒤져도 `types.FunctionType` 을 부르는 곳은
`copy_func` 하나뿐이므로, 2 차의 실패는 **`copy_func(f)` 가 `f.__code__` 로 문자열을 가진 어떤 객체를
받았다**는 뜻입니다.

이번 재현에서 그 조건이 만들어지지 않은 이유는 구조적입니다 — 제 스텁의 "함수처럼 구는 것"은
**진짜 `def` 로 만든 진짜 클로저**(`_probe`)입니다. 진짜 함수는 `__code__` 가 항상 진짜 code
object이므로, 설령 이 `_probe` 가 어쩌다 `copy_func` 에 흘러 들어가도 깨지지 않습니다.

2 차의 프로브가 깨진 이유는 아마 반대 방향입니다 — **함수가 아닌 객체에 `__code__` 같은 던더를
문자열로 흉내만 냈을 가능성**이 높습니다. `record` 모드가 "이름이 없어서 막힌 벽" 을 최대한 밀어서
보려는 설계였다는 점(VENDOR.md §2 의 strict/record 구분과 같은 발상)을 생각하면, 없는 속성마다
`repr` 이 그럴듯하게 보이도록 `MagicMock` 류의 객체나 `SimpleNamespace` 로 `__code__ = "<probe
...>"` 식의 문자열을 채워 넣는 구현이었을 개연성이 있습니다. 그런 객체가 (transformers 자신이 아니라)
**프로브 자신의 기록/직렬화 계층**(리포트를 만들기 위해 캡처한 콜러블을 복사하거나 재구성하는 코드)
을 통과하다 `types.FunctionType` 류의 재구성 로직에 걸렸을 가능성도 배제할 수 없습니다 — 이 경로는
transformers 코드가 아니라 **프로브 자신의 코드**일 수 있다는 뜻입니다.

**이것은 미확인입니다.** 2 차의 원본 스크립트가 남아있지 않아 정확한 호출 스택을 대조할 수 없습니다.
확실한 것은 (a) 메시지가 `types.FunctionType` 호출부에서만 나올 수 있다는 것, (b) transformers 안에
그 호출부가 `copy_func` 하나뿐이고 `auto_docstring` 경로와 무관하다는 것, (c) 그 경로에 진짜 함수를
공급하는 스텁으로는 재현되지 않는다는 것 — 이 셋뿐입니다.

## 6. 앞으로 스텁/셔플을 다시 만들 일이 있다면

이번 재현에서 나온 실용적 규칙 하나: **함수 흉내를 낼 때는 반드시 진짜 `def`(또는 `compile()` 로
만든 진짜 code object)로 만들 것, dunder 를 문자열로 눈속임하지 말 것.** 이것만 지키면
`types.FunctionType` 재구성 계열의 크래시는 애초에 발생 여지가 없습니다. category 1~5 를
만족시키는 스텁이면 `@auto_docstring` 이 걸린 모델 클래스 정의는 통과합니다.

다만 이것이 "스텁만으로 충분하다" 는 뜻은 아닙니다. **모델을 실제로 인스턴스화하고 순전파를 돌리는
단계에서는** `nn.Module` 의 진짜 동작(파라미터 등록, `__setattr__` 후킹, buffer 관리)과 진짜 텐서
연산이 필요합니다 — 그건 IMPORT_WALLS 3~5 차와 VENDOR.md 가 이미 다루는 영역이고 이 문서의
범위 밖입니다. 이 문서가 좁힌 것은 딱 하나, **"클래스 정의 시점" 이 스텁으로 못 넘는 벽인가** 라는
질문이고, 답은 "아니다, 최소한 지금 재현한 조건에서는" 입니다.

## 7. 벤더링 경로(§2 의 실제 계획)에서는 애초에 질문 자체가 성립하지 않는다

VENDOR.md 가 이미 만들어 둔 상태 — 실물 torch 의 파이썬 트리를 그대로 두고 `_C` 만 우리 것으로
바꾼 상태 — 에서는 `torch.Tensor`, `torch.nn.Module`, `torch.autograd.Function` 이 전부 **진짜
객체**입니다. category 2/4 가 "진짜 타입/클래스" 를 요구한다는 것 자체가 이미 공짜로 만족됩니다.
VENDOR.md 의 벽 목록(`_load_global_deps` · `_initExtension` · `TensorBase` 543 개 표면 ·
`torch_shm_manager` · `_VariableFunctions` 625 개 · `_C` 서브패키지 32 개)에 `auto_docstring` 이나
`modeling_llama.py:347` 이 전혀 등장하지 않는 것도 이 결론과 일치합니다 — **아직 `import torch`
자체를 못 넘었기 때문에 도달하지 못한 것이지, 도달했을 때 새 벽이 될 이유가 보이지 않습니다.**
`import torch` 를 뚫는 작업이 실제로 `_initExtension` 이후까지 도달하면, 이 판단을 다시 실측으로
확인할 가치는 있습니다(§2 는 여기서 실물 torch 로 확인했지만 **벤더링된 트리 + 우리 `_C`** 조합
에서는 아직 확인되지 않았습니다 — 이 문서가 다루지 않은 유일한 남은 변수입니다).

> **정정 (문서 감사, 2026-09):** 확인됐다. `from transformers.models.llama.modeling_llama import
> LlamaModel, LlamaForCausalLM` 이 오늘 벤더링된 트리 + 우리 `_C` 조합 위에서 직접 성공하고,
> `__doc__` 길이가 이 문서가 실물 torch 로 잰 것과 정확히 같다 (`LlamaModel` 883, `LlamaForCausalLM`
> 850). 판단은 실측으로 재확인됐다 — `@auto_docstring` 은 벤더링 경로에서도 새 벽이 아니다.

---

## 요약

| 질문 | 답 | 근거 |
|---|---|---|
| `auto_docstring` 이 함수/클래스를 재구성하는가 | **아니다** — `__doc__` 대입만 한다 | `auto_docstring.py:4277-4278, 4512-4514` |
| `copy_func`/`types.FunctionType` 이 이 경로에서 불리는가 | **아니다** | grep 전수 + 87 회 호출 트레이스, 전부 무관한 9 개 지점 |
| torch 의 무언가에 의존하는가 | **의존하지 않는 것으로 확인** — 제너릭 스텁으로 실물과 바이트 동일한 결과 | `/tmp/stub_env/run_stub2.py` vs `/tmp/stub_env/run_real.py` |
| 2 차가 본 TypeError 의 정확한 원인 | **미확인** — 원본 스크립트 소실. 함수 아닌 객체가 `__code__` 를 문자열로 흉내냈을 가능성이 유력 | 메시지 재현(`types.FunctionType('x', {})`)과 소거법 |
| 벤더링 경로(§2)에서 이 벽이 문제가 되는가 | **문제가 될 이유를 찾지 못함**. 다만 벤더링 트리 + 우리 `_C` 조합에서 `import torch` 를 실제로 넘긴 뒤 재확인은 안 함. **정정 (문서 감사, 2026-09): 재확인됨, 문제 없음** — `LlamaModel`/`LlamaForCausalLM` 이 오늘 벤더링 트리 + 우리 `_C` 위에서 직접 임포트되고 `__doc__` 길이가 실물 torch 측정과 바이트 단위로 같다 | VENDOR.md 의 벽 목록에 부재 + §7 정정 |
