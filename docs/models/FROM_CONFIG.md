# `AutoModelForCausalLM.from_config` 가 요구하는 것 — 사전 계측

`docs/models/IMPORT_TORCH.md` 는 `import torch` 를 완주시켰고, 남은 벽은 `from_config` — 그런데 이제
**import 밖, `GenerationMixin` 안**이라고 적었습니다(§11 항목 3). 이 문서는 그 다음 벽을 미리
계측한 기록입니다. **우리 shim 은 아직 여기 도달하지 못했으므로, 진짜 torch 2.13.0 +
transformers 5.15.1 실물로 계측해서 목록을 만들었습니다.**

> **Correction (문서 감사, 2026-09):** 이 문서는 사전 계측 시점(당시 `IMPLEMENTED` 3 개:
> `aten.add.Tensor`/`aten.full.default`/`aten.mm.default`) 을 기록한 것이고, §2.1 이 요구한
> 14 개 op 은 오늘 전부 구현되어 있다 (`grep`으로 14 개 전부 현재 168-op `_aten_implemented()`
> 목록에 있음을 확인). §6 이 "다음 검증 단계"로 남긴 것 — 벤더링 트리 + 우리 `_C` 조합에서
> 같은 스크립트를 다시 돌려 대조하는 것 — 도 실제로 일어났다: 우리 shim 위에서
> `AutoModelForCausalLM.from_config(cfg)` (이 문서와 동일한 llama config) 을 직접 호출하면
> 성공하고, 파라미터 수까지 이 문서가 실물 torch 로 잰 것과 정확히 같다(95,040개). §4.3 이
> 미확인으로 남긴 "candle RNG 가 torch RNG 값과 같은가"도 `docs/numerics/RNG.md`/`docs/bindings/TENSORBASE.md`의
> 포트로 답이 났다(`rust/torch_c/src/rng.rs`에 MT19937 엔진과 `torch.manual_seed` 리매핑이
> 있다). 아래 §2.1/§5/§6 원문은 계측 당시 그대로 남긴다.
> <!-- DOCWATCH: op-implemented aten.normal_.default -->
> <!-- DOCWATCH: op-implemented aten.uniform_.default -->
> <!-- DOCWATCH: op-implemented aten.empty.memory_format -->
> <!-- DOCWATCH: op-implemented aten.arange.start_step -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/rng.rs manual_seed present -->

환경: `/Volumes/macMini/caches/spike-venv/bin/python` (torch 2.13.0, transformers 5.15.1, 둘 다
실물 설치). 대상 모델:

```python
cfg = AutoConfig.for_model("llama", hidden_size=64, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=128, vocab_size=100)
model = AutoModelForCausalLM.from_config(cfg)
```

스크립트는 전부 `/tmp/from_config_probe/`(커밋 대상 아님)에 있습니다. `docs/design/C_SURFACE.md` 는
열지 않았습니다 — 방법은 이 문서에서 독립적으로 설계하고 검증했습니다.

---

## 1. 계측 방법과 작동 증거

### 1.1 방법

두 층을 따로 잽니다. **하나만으로는 답이 안 나옵니다** — 아래 §1.2 가 그 이유를 실측으로 보여줍니다.

| 층 | 도구 | 잡는 것 |
|---|---|---|
| ATen 디스패치 층 | `torch.utils._python_dispatch.TorchDispatchMode` | `torch.ops.aten.<op>.<overload>` 호출. **우리 `_aten_dispatch` 가 답해야 하는 것과 정확히 같은 층** |
| 파이썬 API 층 | `torch.overrides.TorchFunctionMode` | `torch.zeros`, `TensorBase.uniform_` 같은 이름. **이름이 해석 가능해야 하는 것**(bootstrap.py 의 표면 문제) |
| Generator/RNG | 서브클래싱 (아래 설명) | `torch.Generator()` 명시적 생성·시딩. 위 두 층에는 잡히지 않음 |
| `nn.Module` 생명주기 | 메서드 몽키패치 | `__setattr__`, `apply` 호출 횟수 |

두 모드 다 `with RecordFunction(), RecordDispatch(): model = AutoModelForCausalLM.from_config(cfg)`
로 호출부를 감싸는 컨텍스트 매니저이고, 실제 연산은 원본 `func(*args, **kwargs)` 를 그대로 위임
호출하므로 **모델이 실제로 만들어집니다** — 통과/실패 판정과 계측이 동시에 됩니다.

### 1.2 검증 — 알려진 호출을 일부러 일으켜 걸리는지 먼저 확인

```python
with RecordFunction(), RecordDispatch():
    x = torch.zeros(3, 3); x.uniform_(-1, 1)
    y = torch.empty(3, 3); torch.nn.init.kaiming_uniform_(y)
    g = torch.Generator(); g.manual_seed(42)
    z = torch.normal(0.0, 1.0, size=(3,), generator=g)
```

결과(`/tmp/from_config_probe/verify_dispatch_mode.py`, 전문 실행 로그 보존):

```
=== dispatch (ATen op) calls seen ===
   2  aten.uniform_.default
   1  aten.zeros.default
   1  aten.empty.memory_format
   1  aten.normal.float_float

=== torch-function (python API) calls seen ===
   1  _VariableFunctionsClass.zeros
   1  TensorBase.uniform_
   1  _VariableFunctionsClass.empty
   1  kaiming_uniform_
   1  _VariableFunctionsClass.normal
```

**`torch.Generator()` 생성과 `g.manual_seed(42)` 는 둘 중 어느 카운터에도 찍히지 않았습니다.**
`Generator` 는 텐서가 아니므로 두 모드 다 통과할 이유가 없고, 이것이 §1.1 표에 "Generator/RNG" 를
별도 행으로 둔 이유입니다 — **하나의 계측기만 썼으면 이 구멍을 놓쳤을 것**입니다.

`torch._C.Generator` 는 정적(비-힙, `IMMUTABLETYPE`) pybind11 타입이라 메서드를 직접 몽키패치할
수 없습니다(`TypeError: cannot set 'manual_seed' attribute of immutable type`, 인스턴스 속성 대입도
`AttributeError: ... object attribute ... is read-only` 로 거부됨 — 둘 다 인터랙티브로 실측).
**서브클래싱은 허용됩니다**(`class MyGen(torch.Generator): pass` 는 성공, 실측). 그래서
`torch.Generator` 모듈 이름을 로깅 서브클래스로 바꿔치기하는 방식을 썼습니다. 이 방식은 **명시적
`torch.Generator()` 생성만 잡고, 연산이 암묵적으로 쓰는 전역 기본 제너레이터(`torch.default_generator`)
사용은 잡지 못합니다** — 그건 §4 에서 별도로 다룹니다.

**결론: 계측 방법은 알려진 호출 5 종 중 5 종(ATen)·5 종(함수) 을 정확히 잡았고, 놓친 것(Generator)도
이유를 알고 놓쳤습니다.** 이 방법으로 `from_config` 를 재도 신뢰할 수 있다고 판단했습니다.

---

## 2. `from_config` 가 실제로 호출하는 것 — 목록과 개수

전체 스크립트: `/tmp/from_config_probe/run_from_config.py`. 실행 결과, `LlamaForCausalLM` 생성 성공
(파라미터 95,040 개).

### 2.1 ATen 디스패치 층 — 우리 `_aten_dispatch` 가 답해야 하는 것

```
   17  aten.normal_.default
   16  aten.empty.memory_format
   15  aten.uniform_.default
    5  aten.ones.default
    5  aten.fill_.Scalar
    2  aten.arange.start_step
    2  aten.div.Tensor
    2  aten.pow.Scalar
    2  aten.reciprocal.default
    2  aten.mul.Tensor
    2  aten.detach.default
    2  aten.lift_fresh.default
    2  aten.copy_.default
    1  aten.clone.default
TOTAL_DISPATCH_CALLS=75  UNIQUE_OPS=14
```

**서로 다른 op 14 개, 총 호출 75 회.** 이 숫자는 두 개의 서로 다른 소스로 갈립니다.

- **가중치 생성/초기화** (`empty`·`uniform_`·`normal_`·`ones`·`fill_`) — Linear/Embedding/RMSNorm
  파라미터를 만들고 채우는 호출입니다. §4 에서 자세히 다룹니다.
- **RoPE `inv_freq` 계산** (`arange.start_step`·`pow.Scalar`·`div.Tensor`·`reciprocal.default`·
  `mul.Tensor`·`copy_.default`·`detach.default`·`lift_fresh.default`·`clone.default`) —
  `LlamaRotaryEmbedding` 이 `1.0 / (base ** (arange(0, dim, 2) / dim))` 를 계산해 버퍼로 등록하는
  경로입니다. `_init_weights` 의 `"RotaryEmbedding" in module.__class__.__name__` 분기(아래 §3.3)가
  이 계산을 **한 번 더** 반복해 `inv_freq` 와 `original_inv_freq` 양쪽에 `copy_` 하므로 관련 op 이
  전부 짝수(2)로 나옵니다.

### 2.2 파이썬 API 층 — 이름이 해석 가능해야 하는 것

```
   84  _set_grad_enabled
   25  getset_descriptor.__get__
   17  normal_
   16  _VariableFunctionsClass.empty
   15  kaiming_uniform_
    6  device
    5  _VariableFunctionsClass.ones
    5  TensorBase.fill_
    2  _VariableFunctionsClass.arange
    2  TensorBase.div
    2  Tensor.__rpow__
    2  Tensor.__rdiv__
    2  TensorBase.to
    2  TensorBase.detach
    2  TensorBase.requires_grad_
    2  _VariableFunctionsClass.tensor
    2  TensorBase.copy_
    1  TensorBase.clone
TOTAL_FUNCTION_CALLS=192  UNIQUE_NAMES=18
```

**`_set_grad_enabled` 가 84 회로 가장 많습니다** — `@torch.no_grad()` 로 감싼 모든 초기화 호출이
진입/탈출마다 한 번씩 켜고 끕니다. `TensorBase.fill_` 이 5 회인데 ATen 층의 `aten.fill_.Scalar` 도
5 회로 정확히 대응합니다 — 두 층이 서로 다른 것을 재지만, 같은 이벤트를 양쪽에서 본 것이므로
숫자가 맞아떨어지는 것이 계측이 일관됨을 보여주는 교차검증입니다.

**흥미로운 점: `kaiming_uniform_` 이 15 회 불립니다 — HF 의 `_init_weights` 는 Linear 가중치를
`init.normal_` 로 채운다고 적혀 있는데(§3), 정작 그보다 먼저 `kaiming_uniform_` 이 같은 횟수(15,
Linear 레이어 수와 일치)로 불립니다.** 이유는 §4 에서 설명합니다 — `nn.Linear.__init__` 자체가
`reset_parameters()` 를 호출해 먼저 `kaiming_uniform_` 로 채우고, 그 뒤 HF 의 `_init_weights` 가
`normal_` 로 **덮어씁니다.** 즉 **한 텐서가 초기화되는 데 실제로는 두 가지 분포가 순서대로 관여**
합니다 — 첫 결과는 버려지지만, 그 호출 자체는 실제로 일어나고 우리 shim 이 답해야 합니다.

### 2.3 `nn.Module` 생명주기

```
  196  Module.__setattr__
    0  Module.apply   (호출 없음)
```

`Module.apply` 가 0 인 것은 주목할 만합니다 — transformers 5.x 의 `initialize_weights()`(§3.2) 는
표준 `torch.nn.Module.apply` 대신 **자체 재귀 함수(`smart_apply`)** 를 씁니다. `torch.nn.Module.apply`
자체를 구현하는 것은 이 모델 생성 경로에서는 급하지 않다는 뜻입니다 — 다만 다른 코드 경로(사용자
코드, 다른 모델)가 표준 `apply` 를 쓸 가능성은 남습니다.

`Module.__setattr__` 196 회는 벤더링한 순정 `torch/nn/modules/module.py` 코드가 그대로 도는 것이고,
`isinstance(value, Parameter)` / `isinstance(value, Module)` / `isinstance(value, Tensor)` 로 대상을
분류해 `_parameters`/`_modules`/버퍼 딕셔너리에 넣는 로직입니다. `TensorBase`·`Parameter`·`Module`
이 전부 진짜 타입이어야 이 분류가 성립하는데, `IMPORT_TORCH.md` §3 이 이미 확보해 둔 상태입니다 —
**새 벽이 아니라 기존에 확보한 것의 재확인입니다.**

---

## 3. `GenerationMixin` 이 실제로 요구하는 것

`transformers/generation/utils.py` (4089 행)를 읽었습니다.

### 3.1 import 시점 — `torch.distributed` 는 실제 서브모듈이어야 한다

```python
import torch
import torch.distributed as dist
from torch import nn
```

(`generation/utils.py:25-27`). `IMPORT_WALLS.md` 1 차의 category 3("실제 서브모듈 파일 — 모듈
수준 `__getattr__` 로는 만족되지 않음")과 정확히 같은 요구이고, `torch.distributed` 가 이미 그
15 개 서브모듈 후보 목록에 있었으므로 **새 발견은 아닙니다.** `dist.all_reduce` / `dist.ReduceOp.SUM`
호출은 `generate()` 본문(`:2690`)에만 있고 `from_config`/`GenerationMixin` 클래스 정의 시점에는
쓰이지 않습니다 — **런타임(멀티프로세스 `generate` 호출) 요구이지 `from_config` 요구가 아닙니다.**

### 3.2 클래스 정의 시점 — `GenerationMixin` 자체엔 메타클래스도 데코레이터도 없다

```python
class GenerationMixin(ContinuousMixin):
    output_modalities = ("text",)
    def adjust_generation_fn(self, ...): ...
    ...
```

(`:359`). 클래스 데코레이터 없음, 커스텀 메타클래스 없음 — 평범한 클래스 문입니다. **부모
`ContinuousMixin`** (`generation/continuous_batching/continuous_api.py:1083`) 도 마찬가지입니다.

### 3.3 진짜 요구는 두 가지 — `torch.no_grad()` 데코레이터와 `torch.*Tensor` 애노테이션

**(a) `@torch.no_grad()` 가 클래스 본문에서 실제로 데코레이터로 쓰인다.**
`generation/utils.py` 에 1 회, `continuous_api.py`(`ContinuousMixin`, `GenerationMixin` 의 부모) 에
**6 회.** `IMPORT_WALLS.md` 1 차의 category 5("데코레이터로 호출되는 것")가 "미확인 — 여기서
멈췄다" 고 적어 둔 항목인데, **여기서 정확한 요구 시점과 프로토콜을 확인했습니다.**

- **시점**: 클래스 본문이 실행되는 순간 — 즉 `import`(모듈이 처음 로드되는 시점, `from_config`
  이 `LlamaForCausalLM` 을 지연 임포트하는 순간에 해당)입니다. `generate()` 를 실제로 호출할 때가
  아닙니다.
- **프로토콜**: `torch.no_grad` 는 (1) 인자 없이 호출 가능해야 하고, (2) 그 호출 결과가 **함수를
  받아 함수를 돌려주는 또 다른 호출 가능 객체**여야 합니다 — 즉 컨텍스트 매니저(`__enter__`/
  `__exit__`)이면서 동시에 데코레이터(`__call__`)로도 동작해야 합니다. 상류 구현은
  `torch.autograd.grad_mode._DecoratorContextManager.__call__` 이 `functools.wraps` 로 감싼 래퍼를
  돌려주는 방식입니다. **이 이중 프로토콜이 스텁이 흉내내야 하는 것의 전부**이고, 실제로 안에서
  그래디언트를 껐다 켰다 하는 동작 자체는 클래스 정의 시점에는 실행되지 않습니다(함수가 호출될
  때만 실행) — §2.2 의 `_set_grad_enabled` 84 회는 **모델 생성이 실제로 실행될 때** 나온 숫자이지,
  클래스 정의 시점의 숫자가 아닙니다.

**(b) `torch.*Tensor` 타입 애노테이션이 클래스 본문 실행 시점에 즉시 평가된다.**
`generation/utils.py` 는 `sequences: torch.LongTensor`, `scores: tuple[torch.FloatTensor] | None`
같은 `@dataclass` 필드를 118 줄에 걸쳐 씁니다(`GenerateDecoderOnlyOutput` 등 4 개 dataclass,
`:168`·`:204`·`:252`·`:296`). 파일에 `from __future__ import annotations` 가 **없으므로**(확인함),
CPython 3.13 은 이 애노테이션을 **클래스 본문 실행 시점에 즉시 평가**합니다 — `IMPORT_WALLS.md`
category 2 와 같은 요구이지만, **모델 클래스가 아니라 `GenerationMixin` 자신의 모듈이 이미 이
요구를 걸고 있다**는 것이 새로 확인된 지점입니다. `torch.LongTensor`·`torch.FloatTensor` 가
진짜 클래스가 아니면 `generation/utils.py` 자체의 **import** 가 여기서 멈춥니다 — `from_config`
호출까지 갈 필요도 없습니다.

### 3.4 `from_config` 경로에서 실제로 실행되는 `GenerationMixin` 코드는 거의 없다

계측된 `dispatch_calls`/`function_calls` 에 `generate`·`_sample`·`_beam_search` 류의 이름은 전혀
없습니다 — `from_config` 는 모델 **객체**를 만들 뿐 `generate()` 를 호출하지 않으므로 당연합니다.
`GenerationMixin` 이 `from_config` 단계에 실제로 거는 부담은 **§3.1·§3.3 의 import/클래스-정의
시점 요구뿐**이고, `generate()` 본문의 방대한 로직(캐시 관리·로짓 프로세서·빔서치 등)은 이번
측정 범위 밖입니다(§6 참조).

---

## 4. 가중치 초기화 경로 — 무엇이 어떤 순서로 불리는가

### 4.1 이중 초기화 — `nn.Linear.__init__` 이 먼저, HF 의 `_init_weights` 가 나중

`transformers/modeling_utils.py:2374` 의 `PreTrainedModel._init_weights`:

```python
@torch.no_grad()
def _init_weights(self, module):
    ...
    if isinstance(module, (nn.Linear, nn.Conv1d, ...)):
        if getattr(module, "weight", None) is not None:
            init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            init.zeros_(module.bias)
    ...
    elif isinstance(module, nn.Embedding):
        init.normal_(module.weight, mean=0.0, std=std)
        if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
            init.zeros_(module.weight[module.padding_idx])
    ...
    elif (isinstance(module, (nn.GroupNorm, ...)) or "RMSNorm" in module.__class__.__name__):
        if getattr(module, "weight", None) is not None:
            init.ones_(module.weight)
        ...
    elif "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
        rope_fn = ROPE_INIT_FUNCTIONS[module.rope_type] if module.rope_type != "default" else module.compute_default_rope_parameters
        buffer_value, _ = rope_fn(module.config)
        init.copy_(module.inv_freq, buffer_value)
        init.copy_(module.original_inv_freq, buffer_value)
```

이것이 `_initialize_weights` → `initialize_weights`(자체 `smart_apply` 재귀, §2.3) 를 통해 모든
서브모듈에 대해 실행됩니다. **그러나 이 함수가 도는 시점에는 `nn.Linear`·`nn.Embedding` 이 이미
자기 자신의 `reset_parameters()` 로 한 번 채워진 뒤입니다** — `nn.Linear.__init__` 이 끝에서
`self.reset_parameters()` 를 부르고, 그 기본 구현은 `init.kaiming_uniform_(self.weight, a=sqrt(5))`
입니다(벤더링한 순정 `torch/nn/modules/linear.py`, 변경 없음). §2.2 에서 `kaiming_uniform_` 이
Linear 레이어 수(15)와 정확히 같은 횟수로 불린 것이 이 경로입니다. **즉 각 Linear 가중치는 생성
과정에서 (a) `kaiming_uniform_` 로 한 번, (b) `_init_weights` 의 `normal_` 로 한 번 더 — 두 번
채워지고 첫 결과는 버려집니다.** 값 자체는 버려지지만 **호출은 실제로 일어나므로 우리
`_aten_dispatch` 는 두 경로(`aten.uniform_.default`, `aten.normal_.default`) 를 전부 구현해야
합니다.**

### 4.2 `transformers.initialization` — torch.nn.init 을 가로채는 자체 계층

`transformers/initialization.py`(신설 모듈, transformers 5.x)를 읽었습니다. 이것은 **IMPORT_WALLS
문서들이 아직 보지 못한, transformers 5.x 에서 새로 생긴 계층**입니다.

- **`TORCH_INIT_FUNCTIONS`** — 모듈 로드 시점에 `torch.nn.init.uniform_`·`normal_`·`constant_`·
  `ones_`·`zeros_`·`eye_`·`dirac_`·`xavier_uniform_`·`xavier_normal_`·`kaiming_uniform_`·
  `kaiming_normal_`·`trunc_normal_`·`orthogonal_`·`sparse_` **14 개를 딕셔너리에 캡처**합니다
  (`initialization.py:24-39`). 이후 자기 자신의 `uniform_`/`normal_`/... 래퍼가 `_is_hf_initialized`
  플래그를 검사한 뒤 캡처해 둔 원본을 호출합니다.
- **`guard_torch_init_functions()`** — `PreTrainedModel.initialize_weights()` 를 감싸는
  컨텍스트 매니저(`@init.guard_torch_init_functions()`, `modeling_utils.py:3176`)로, `torch.nn.init`
  뿐 아니라 `torch.nn.modules.{activation,transformer,linear,loss,batchnorm,conv,normalization,
  rnn,sparse}` **9 개 서브모듈까지 순회하며** 거기 바인딩된 이름(`from torch.nn.init import
  xavier_uniform_` 식으로 모듈 로드 시점에 이름이 복사된 경우 대비)까지 자기 래퍼로 **일시
  교체**합니다. 이유는 주석에 명시: `torch.nn.modules.activation` 의 `MultiheadAttention` 이
  `xavier_uniform_` 을 이런 식으로 임포트해 두기 때문입니다.

**이것이 우리 shim 에 갖는 의미는 두 가지입니다.**

1. `torch.nn.init.*` 14 개와 위 9 개 서브모듈은 **벤더링한 순정 파이썬 트리 그대로**이므로(§1의
   전제) `setattr`/`getattr` 자체는 우리가 손대지 않아도 이미 동작합니다 — 이것은 **파이썬 계층의
   완전히 평범한 모듈 속성 재할당**이라 `_C` 의 어떤 특수 취급도 요구하지 않습니다. `_C` 쪽에서
   봐야 할 새 벽은 없습니다.
2. **다만 이것이 성립하려면 그 14 개 함수 각각이 실제로 실행 가능해야 하고, 실행되면 결국 §2.1 의
   ATen op(`uniform_`·`normal_`·`constant_`(`fill_`)·`ones_`·`zeros_`·`eye_`·`dirac_`·
   `xavier_uniform_`·`xavier_normal_`·`kaiming_uniform_`·`kaiming_normal_`·`trunc_normal_`·
   `orthogonal_`·`sparse_`)로 내려갑니다.** 이번 계측(작은 Llama)은 이 중 `uniform_`/`normal_`/
   `ones_`/`fill_`(=constant_/zeros_ 경로) 만 실제로 밟았습니다 — **`xavier_*`·`kaiming_normal_`·
   `trunc_normal_`·`orthogonal_`·`dirac_`·`eye_`·`sparse_` 는 이번 모델 구성에서는 불리지 않았고,
   다른 아키텍처(예: 합성곱·LSTM·비전 모델)를 계측하면 추가로 나올 것으로 예상됩니다 — 미확인.**

### 4.3 RNG — `torch._C.Generator`, 그리고 우리 shim 이 지금 가진 것

**이번 계측에서 `torch.Generator()` 가 명시적으로 생성되거나 시딩되는 일은 없었습니다**
(§2 의 rng_calls 카운터가 비어 있음, `/tmp/from_config_probe/out1.txt` 참조). `_init_weights` 의
`normal_`/`uniform_`/`kaiming_uniform_` 호출은 전부 `generator=None`(기본값)으로 이뤄지고, 이
경우 torch 는 **전역 기본 제너레이터**(`torch.default_generator`, 프로세스마다 하나인 C 소유
싱글턴)를 암묵적으로 씁니다. 이 사용은 파이썬 레벨에서는 안 보이고, ATen op 내부(`aten.uniform_.
default`·`aten.normal_.default` 자체의 C++ 구현)에 **감춰져 있습니다** — 그래서 §2.1 의 목록이
사실상 RNG 요구사항의 전부입니다: **`aten.uniform_.default` 와 `aten.normal_.default` 를 구현하는
것 자체가 곧 "기본 제너레이터로 난수를 뽑는 것"을 구현하는 일입니다.**

`torch.default_generator` **읽기 자체**(모듈의 평범한 속성이라 후킹하지 않고는 셀 수 없음)와
`torch.Generator()` **명시적 생성**은 이번 계측에서 0 회였지만, 이것이 "필요 없다"는 뜻은
아닙니다 — 다른 모델·다른 초기화 경로(`generator=` 를 명시로 넘기는 코드, `torch.manual_seed(...)`
로 재현성을 고정하는 사용자 코드)에서는 요구될 수 있습니다. **미확인.**

**우리 shim 이 지금 가진 것: 아무것도 없습니다.**

- `rust/torch_c/src/aten.rs` 의 `IMPLEMENTED` 상수는 `["aten.add.Tensor", "aten.full.default",
  "aten.mm.default"]` **세 개뿐**입니다. §2.1 이 요구하는 14 개 중 **0 개가 구현되어 있습니다.**
- `Generator` 는 `bootstrap.py` 안에서 **이름만** 존재합니다(`IMPORT_TORCH.md` 벽 19 — 메타클래스
  훅만 연결). `manual_seed`·`seed`·`get_state`·`set_state`·실제 난수 상태 — 전부 **미구현**이고,
  `grep -rn "manual_seed\|default_generator\|seed" rust/torch_c/src/*.rs` 는 아무것도 찾지
  못했습니다.
- **candle 의 RNG 가 torch 의 CPU RNG 알고리즘과 값이 같은지는 미확인입니다.** `aten.uniform_.
  default`/`aten.normal.*` 을 candle 의 난수 생성으로 구현하면 **구조적으로는** 동작하겠지만
  (모델이 만들어지고 순전파가 돎), torch 가 실제로 생성하는 **정확한 바이트열과 같은 값**을
  내는지는 별개 문제입니다 — torch CPU 는 Mersenne Twister(`at::CPUGeneratorImpl`) 기반 알고리즘을
  쓰고, candle 이 같은 알고리즘을 구현했다는 근거는 찾지 못했습니다. **골든 하네스가 지금까지
  다룬 op(`add`·`full`·`mm`) 는 전부 결정적이라 이 문제를 아직 만나지 않았습니다** — 난수 op 을
  추가하는 순간 "같은 시드로 torch 와 값까지 같아야 하는가" 라는 새로운 종류의 질문이 열립니다.
  이 문서는 그 답을 내지 않고 **미확인으로 남깁니다.**

---

## 5. 우리 shim 이 지금 가진 것과의 차이 — 요약

| 요구 | `from_config` 가 요구하는 양 | 우리 shim 현재 | 격차 |
|---|---|---|---|
| ATen op 구현 | 14 종, 75 회 호출 | 3 종(`add.Tensor`·`full.default`·`mm.default`), **14 종 중 0 종** | **14 개 신규 구현 필요** |
| `torch.Generator`/RNG | 명시적 사용 0 회(이번 계측), 그러나 `uniform_`/`normal_` 내부에서 암묵적 사용 필수 | 이름만 있고 상태·메서드 없음(`grep` 확인) | `uniform_`/`normal_` 구현에 반드시 동반되어야 함. 값의 torch 일치 여부는 별개 미확인 사안 |
| `torch.nn.init.*` 14 개 함수 | `transformers.initialization` 이 import 시점에 캡처, 이번 계측은 그중 4 개(`uniform_`·`normal_`·`ones_`·`zeros_`류) 를 실제로 밟음 | 벤더링 파이썬 트리 그대로라 **함수 자체는 이미 있음** — 실행하면 위 ATen 격차로 귀결 | 새 파이썬 계층 벽 없음. 순수 ATen 문제로 환원됨 |
| `torch.no_grad()` 데코레이터 프로토콜 | `GenerationMixin`/`ContinuousMixin` 클래스 본문에서 7 회, **모듈 import 시점**에 이중 프로토콜(컨텍스트 매니저 + 데코레이터) 요구 | `_C` 쪶 확인 필요 — `no_grad` 가 벤더링 파이썬 트리(`torch/autograd/grad_mode.py`) 것이라면 함수 자체는 있으나, `_DecoratorContextManager.__call__` 이 의존하는 하위 C 훅(그래디언트 on/off 스위치)이 우리 shim 에 있는지는 **미확인** | 확인 필요 |
| `torch.*Tensor` 타입 애노테이션 | `GenerationMixin` 자신의 모듈(생성 클래스가 아니라!) 이 import 시점에 4 개 dataclass, 118 줄에 걸쳐 요구 | `IMPORT_TORCH.md` §1 category 2 로 이미 해결(진짜 클래스 생성) — `LongTensor`/`FloatTensor` 가 표면에 있는지만 확인하면 됨 | 새 벽 아님, 표면 커버리지 확인만 필요 |
| `torch.distributed` 실제 서브모듈 | `generation/utils.py` import 시점 1 회(`import torch.distributed as dist`), 실사용은 `generate()` 런타임(`dist.all_reduce`)에 한정 | `IMPORT_WALLS.md` 1 차가 이미 서브모듈 15 개 후보에 포함, `IMPORT_TORCH.md` 벽 1 이 off-switch 로 처리 | `from_config` 자체는 새 요구 추가 없음. `generate()` 를 실제로 멀티프로세스로 돌릴 때만 재부상 |
| `nn.Module.apply` | 0 회(자체 `smart_apply` 재귀 사용) | 미확인 | `from_config` 경로에서는 급하지 않음 |
| `Module.__setattr__` 타입 분류 | 196 회 | `IMPORT_TORCH.md` §3 이 이미 확보(`TensorBase`/`Parameter`/`Module` 진짜 타입) | 새 벽 아님 |

**한 줄 요약: `from_config` 벽의 실체는 새로운 종류의 벽이 아니라, `import torch` 가 이미 통과시킨
"이름의 문제"에서 처음으로 "값을 만드는 문제"로 넘어가는 지점입니다.** `import torch` 는 계산 없이
1207 개의 이름만 옳으면 통과했지만(`IMPORT_TORCH.md` §1), `from_config` 는 그 이름들 중 최소
14 개(§2.1) 를 **실제로 실행해서 올바른 shape·dtype 의 텐서를 만들어내야** 통과합니다. 이 14 개는
전부 `rust/torch_c/src/aten.rs` 의 `IMPLEMENTED` 목록에 없습니다.

---

## 6. 이 문서가 다루지 않은 것 — 명시적 미확인

- **`generate()` 본문** — 이번 계측은 `from_config` 까지만입니다. 캐시 관리·로짓 프로세서·빔서치·
  `torch.distributed` 의 `generate()` 내부 실사용(§3.1)은 범위 밖입니다.
- **candle RNG 와 torch RNG 의 값 일치 여부** — §4.3. 구조적 통과와 값 일치는 다른 질문입니다.
- **`xavier_*`·`kaiming_normal_`·`trunc_normal_`·`orthogonal_`·`dirac_`·`eye_`·`sparse_` 를 요구하는
  모델 구성** — 이번 Llama 계측에서는 밟지 않았습니다. 합성곱·LSTM·비전 계열 모델을 계측하면
  추가로 나올 가능성이 있습니다.
- **`torch.no_grad()` 데코레이터가 실제로 의존하는 하위 C 훅**(그래디언트 활성/비활성 전역 상태)이
  우리 `_C` 셔플에 있는지 — 이 문서는 벤더링 트리에 함수 자체가 존재한다는 것만 확인했고, 그 함수가
  호출하는 C 레벨 기능까지는 추적하지 않았습니다.
- **`torch.default_generator` 읽기 횟수** — 모듈의 평범한 속성이라 `torch` 모듈 자체를 감싸지
  않고는 셀 수 없어 계측하지 않았습니다(§1.1).
- **벤더링된 트리 + 우리 `_C` 조합에서 재확인** — 이 문서의 모든 수치는 **실물 torch 2.13.0** 으로
  잰 것입니다. `import torch` 가 우리 shim 위에서 통과한 뒤, 이 문서의 14 개 op 을 구현하고 나서
  같은 스크립트(`/tmp/from_config_probe/run_from_config.py`)를 우리 `_C` 위에서 다시 돌려
  대조하는 것이 다음 검증 단계입니다.
