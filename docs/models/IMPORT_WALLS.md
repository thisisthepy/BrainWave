# `import transformers` 가 torch 에 요구하는 것

DESIGN.md §11 의 1 단계 — "벤더링한 torch 파이썬 트리 + 빈 `_C` 스텁으로 `import transformers` 를
시도하고 나오는 벽을 기록한다" — 의 결과입니다.

환경: transformers **5.15.1**, CPython 3.13.0, torch 미설치.

---

## 가장 큰 발견 — `import transformers` 는 torch 없이도 성공한다

벽이 import 에 있을 것으로 보고 계획을 세웠는데, **없습니다.**

```
$ python -c "import transformers; print(transformers.__version__)"
[transformers] PyTorch was not found. Models won't be available and only
tokenizers, configuration and file/data utilities can be used.
5.15.1
```

`from transformers import AutoModelForCausalLM` 도 통과합니다 — 클래스 객체가 지연 생성이라
존재만 합니다. **실제로 쓸 때** 비로소 막힙니다.

```
AutoModelForCausalLM requires the PyTorch library but it was not found
```

덧붙여 **transformers v5 에서도 torch 는 하드 의존성이 아닙니다.** `uv pip install transformers`
가 torch 없이 설치를 끝냅니다(설치된 것은 tokenizers · huggingface-hub · safetensors 등).

**그러므로 §11 의 1 단계는 "벽의 개수를 센다" 가 아니라 "관문 하나를 통과시킨 뒤 그 너머를 센다"
입니다.** 계획의 전제를 하나 고쳐야 합니다.

## 관문은 `is_torch_available()` 하나다

`transformers/utils/import_utils.py:179`:

```python
is_available, torch_version = _is_package_available("torch", return_version=True)
return is_available and version.parse(torch_version) >= version.parse("2.5.0")
```

`_is_package_available` 은 `importlib.util.find_spec("torch")` 로 존재를, `importlib.metadata.
version(...)` 으로 버전을 봅니다(메타데이터가 없으면 직접 import 해 `__version__` 을 보는 폴백이
있습니다).

**즉 임포트 가능한 것만으로는 부족하고 배포 메타데이터가 필요하며, 버전이 2.5.0 이상이어야
합니다.** 벤더링한 트리는 어느 상류 버전에서 가져왔는지를 그대로 선언하면 됩니다.

확인: `torch/__init__.py` 에 `__version__` 을 두고 `torch-2.9.0.dist-info/METADATA` 를 놓는 것만으로
`is_torch_available() == True` 가 됩니다.

## 관문 너머 — 요구되는 것은 목록이 아니라 다섯 범주다

기록하는 스텁 `torch` 로 `AutoModelForCausalLM.from_config` 를 반복 실행하며 수집했습니다.
**중요한 것은 이름의 목록이 아니라 각 이름이 어떤 *종류* 여야 하는가입니다** — 그것이 벤더링에서
무엇을 진짜로 가져와야 하는지를 정합니다.

| # | 범주 | 근거 | 스텁으로 되는가 |
|---|---|---|---|
| 1 | **배포 메타데이터** | `importlib.metadata.version("torch") >= 2.5.0` | 예 |
| 2 | **어노테이션에 쓰이는 진짜 타입** | `generation/logits_process.py:142` 의 `eos_token_id: int \| list[int] \| torch.Tensor` — 어노테이션이 클래스 정의 시점에 평가되므로 `\|` 가 성립해야 함 | 예 (진짜 클래스면) |
| 3 | **실제 서브모듈 파일** | `generation/utils.py:26` 의 `import torch.distributed as dist` — **모듈 수준 `__getattr__` 로는 만족되지 않음** | 예 (파일이 있으면) |
| 4 | **상속 가능한 클래스** | `class _AllReduceBackward(torch.autograd.Function)` | 예 (진짜 클래스면) |
| 5 | **데코레이터로 호출되는 것** | `continuous_batching/input_outputs.py:244` 의 `@torch.no_grad()` — 호출 결과가 다시 호출 가능해야 함 | 미확인 (여기서 멈춤). **정정 (문서 감사, 2026-09):** `docs/models/FROM_CONFIG.md` §3.3 이 이 범주를 이어받아 정확한 요구 시점(클래스 본문 실행 시점)과 이중 프로토콜(컨텍스트 매니저 + 데코레이터)을 확인했다 — 자기 문서 안에서 이 문서를 이름으로 지목한다 ("IMPORT_WALLS.md 1 차의 category 5"). `from_config` 는 오늘 shim 위에서 직접 성공한다(이 라운드에서 독립적으로 재확인) |

범주 2 와 4 는 **속성마다 동적으로 진짜 클래스를 돌려주면** 한꺼번에 풀립니다. 범주 3 은 클래스로
풀리지 않습니다 — 파이썬의 `import a.b` 는 실제 모듈을 찾으므로 파일이 있어야 합니다.

## 지금까지 확인된 서브모듈 (11 개)

발견 순서입니다. 하나를 만들면 다음이 드러나는 식으로 수집했습니다.

```
torch.distributed
torch.distributed._composable
torch.distributed._composable.fsdp
torch.distributed.fsdp
torch.distributed.device_mesh
torch.nn
torch.nn.functional
torch.nn.attention
torch.nn.attention.flex_attention
torch._dynamo
torch._dynamo._trace_wrapped_higher_order_op
```

**두 가지가 눈에 띕니다.**

- **사설 모듈이 섞여 있습니다** — `torch._dynamo`, `torch.distributed._composable`. 벤더링 대상이
  공개 API 로 끝나지 않습니다.
- **분산 학습 모듈이 다수입니다** — FSDP · device_mesh · `_composable`. 기기에서 쓸 일이 없는데도
  **import 경로에 있다는 이유만으로** 존재해야 합니다. 비어 있는 스텁으로 충분한지, 아니면 실제
  구현이 필요한지는 아직 모릅니다.

`torch._dynamo` 가 요구되는 것은 §5 의 판단과도 맞물립니다 — `torch.compile` 은 기기에서 쓸 수
없지만 **그 모듈이 존재하기는 해야** 합니다.

## 2 차 — 정정과 한계

> **정정.** 이 문서의 이전 판본은 "모델 생성 성공" 으로 끝났습니다. **틀렸습니다.**
> 반복 스크립트의 성공 판정이 `grep -q MODEL_OK` 였는데, 실패할 때 트레이스백이 **소스 줄을 그대로
> 출력**하므로 그 문자열이 오류 출력에도 걸렸습니다. 종료 코드로 바꿔 다시 돌리니 세 번 모두
> 실패합니다. 스텁으로 모델을 만드는 데는 아직 도달하지 못했습니다.

종료 코드로 판정하도록 고친 뒤 서브모듈이 넷 더 나왔습니다. **총 15 개.**

```
torch.utils            torch.utils._pytree      torch.utils.checkpoint
torch.distributions
```

### 범주 6 — 런타임 코드 생성 **[정정됨 — 아래 참조]**

> **정정.** 이 절의 결론 — "`@auto_docstring` 이 런타임 코드 생성을 하므로 스텁으로 못 넘는다,
> 따라서 여기가 벤더링이 필요해지는 최소 지점" — **은 틀렸습니다.** 근거는 `docs/models/AUTODOC.md`,
> 요약은 이렇습니다.
>
> **`types.FunctionType(` 은 transformers 전체에서 딱 한 곳에만 있습니다** — `utils/doc.py:1088`
> 의 `copy_func`. 그리고 그 호출처는 전부 `X.push_to_hub = copy_func(X.push_to_hub)` 형태로
> **transformers 자기 함수**를 대상으로 하지 torch 객체를 건드리지 않습니다.
> `auto_docstring.py` 는 `inspect.signature` 로 읽고 `__doc__` 을 대입할 뿐이고,
> `func.__code__.co_filename` 을 **읽기만** 합니다 — 재구성하지 않습니다.
> 실제로 계측했을 때 `LlamaModel` 임포트 중 `copy_func` 이 87 회 불리는데 **전부 그 9 개
> 호출처에서 나오고, `@auto_docstring` 프레임에서는 0 회**입니다.
>
> **즉 그 벽은 transformers 의 것이 아니라 우리 프로브의 것이었습니다.** 당시 프로브가
> 함수 비슷한 객체의 `__code__` 를 문자열로 흉내냈고, 그것이 `copy_func` 에 들어가 터진 것으로
> 보입니다. 더 충실한 스텁으로 다시 만들면 `modeling_llama.py` 가 **끝까지 임포트되고**,
> 각 클래스의 `__doc__` 길이가 진짜 torch 와 바이트 단위로 같습니다.
>
> **따라서 "벤더링이 필요해지는 최소 지점 = 모델 클래스 정의" 라는 결론도 근거를 잃습니다.**
> 벤더링이 필요한 이유는 여전히 있지만(§3 의 1084 개 모듈, `VENDOR.md` 의 989 개 이름),
> **이 벽은 그 이유가 아니었습니다.**
>
> 남은 교훈 하나: 함수를 흉내내는 프로브는 **진짜 `def` 로 만든 클로저**여야 합니다. 던더를
> 문자열로 가짜로 채우면 이런 종류의 오진을 만듭니다.
>
> 아직 확인 안 된 것: 벤더링한 트리 + 우리 `_C` 조합에서 재확인한 것은 아닙니다. `import torch`
> 가 아직 안 통과해서 거기까지 못 갔습니다.

원래 기록은 아래에 남깁니다.

`modeling_llama.py:347` 의 `@auto_docstring` 이 클래스 정의 시점에 함수를 **다시 만듭니다.**

```
TypeError: function() argument 'code' must be code, not str
```

프로브가 돌려주는 것은 이름을 흉내낸 객체이지 코드 객체가 아니므로 여기서 멈춥니다. **이것은
스텁을 더 정교하게 만들어 넘을 수 있는 벽이 아닙니다** — introspection 이 진짜 객체를 요구합니다.

**그러므로 스텁 실험은 여기까지입니다.** 이 지점 너머는 `nn.Module` 을 비롯한 실물이 있어야 하고,
그것이 곧 §2 의 "상류 파이썬 트리를 벤더링한다" 입니다. 바꿔 말하면 **이 실험이 벤더링이 필요한
최소 지점을 찾아준 것**이고, 그 답은 "모델 클래스가 정의되는 순간" 입니다.

### 그래서 계획에 주는 정보

- **관문(`is_torch_available`)까지는 메타데이터만으로 넘습니다** — 벤더링 없이.
- **서브모듈 15 개는 파일이 존재하기만 하면 됩니다** — 대부분 분산 학습 · dynamo 계열이라 기기에서
  쓰지 않지만 import 경로에 있습니다. 빈 스텁으로 충분해 보이나, 모델 정의를 통과한 뒤에야 확정할
  수 있습니다.
- **모델 클래스 정의부터는 실물이 필요합니다.** `@auto_docstring` 이 그 경계입니다.

## 3 차 — 진짜 torch 를 놓고 잰 벤더링 규모

스텁으로 더 파는 것은 소득이 없다고 판단해(범주 6), **실물 torch 2.13.0 을 설치하고 어떤 모듈이
실제로 올라오는지 셌습니다.** 이것이 §2 "파이썬 트리를 벤더링한다" 의 실제 크기입니다.

측정: Llama 2 층 · hidden 64 로 모델을 만들고 순전파와 `generate` 까지 돌린 뒤 `sys.modules` 를 셈.

| 단계 | 누적 torch 모듈 | 신규 |
|---|---|---|
| `import transformers` | **0** | 0 |
| 모델 클래스 · config 접근 | **1084** | **+1084** |
| 모델 생성 | 1084 | 0 |
| 순전파 | 1084 | 0 |
| `generate` | 1084 | 0 |

**세 가지가 확정됩니다.**

1. **`import transformers` 는 torch 를 전혀 건드리지 않습니다** (1 차 결과의 정량 확인 — 0 개).
2. **비용은 torch 를 처음 만지는 순간 한 번에 전부 지불됩니다.**
3. **순전파와 `generate` 는 모듈을 하나도 더 올리지 않습니다.** 즉 **모듈 벤더링 문제와 op 커버리지
   문제는 분리되어 있습니다** — 전자는 임포트 그래프, 후자는 실행 경로입니다.

### 1084 개의 내역

맨 `import torch` 만으로 **765 개**가 올라오고, transformers 경로가 **319 개**를 더 올립니다.
추가분은 거의 전부 `torch._dynamo` 계열입니다.

| 최상위 | 모듈 수 | 기기에서 쓰나 |
|---|---|---|
| `torch.distributed` | 145 | **아니오** |
| `torch.nn` | 101 | 예 |
| `torch._dynamo` | 93 | **아니오** (§5 — iOS 에서 `torch.compile` 불가) |
| `torch.utils` | 89 | 일부 |
| `torch.ao` | 75 | 아마도 (양자화) |
| `torch.fx` | 63 | **아니오** |
| `torch.distributions` | 48 | 아마도 아니오 |
| `torch._C` | 38 | — (우리가 교체) |
| `torch._higher_order_ops` | 30 | 아마도 아니오 |
| `torch.onnx` | 27 | **아니오** |
| `torch._functorch` | 25 | 아마도 아니오 |
| `torch.cuda` | 21 | **아니오** |

### 이것이 계획에 주는 것

**§2 의 "파이썬 트리를 벤더링한다" 는 생각보다 큰 일입니다.** 사소한 Llama 순전파 하나에 1084 개
모듈이 필요하고, 그중 가장 큰 덩어리들이 기기에서 절대 쓰이지 않는 것들입니다 — distributed(145) ·
dynamo(93) · fx(63) · onnx(27) · cuda(21) 만 **349 개, 전체의 32%** 입니다.

**그러므로 이 경로의 진짜 작업은 op 을 구현하는 것이 아니라 임포트 그래프를 쳐내는 것입니다.**
§5 에서 "`torch.compile` 은 못 쓰지만 모듈이 존재하기는 해야 한다" 고 적었는데, 그 비용이 이제
정량화됐습니다 — **모듈 하나가 아니라 93 개**이고, 그것들을 전부 실어야 하는지 빈 스텁으로 끊을 수
있는지가 다음 질문입니다.

1 차에서 찾은 서브모듈 15 개는 **끊는 지점의 후보 목록**이었던 셈입니다. 그 15 개가 파일 존재만
요구했다는 사실은 **저 349 개를 빈 스텁으로 대체할 수 있을 가능성**을 시사하지만, 실물 torch 로
확인한 것은 아닙니다.

## 4 차 — 349 개를 끊을 수 있는가: **없습니다**

3 차에서 "기기에서 안 쓰는 349 개(32%)를 쳐내는 것이 이 경로의 진짜 작업" 이라고 적었습니다.
**시험해 보니 그 방식으로는 안 됩니다.**

방법: meta path finder 로 `torch.distributed` · `torch._dynamo` · `torch.fx` · `torch.onnx` ·
`torch.cuda` 를 가로채 빈 스텁을 돌려주고, 실물 torch 2.13.0 으로 모델 생성 · 순전파 · `generate`
가 여전히 통과하는지 확인. 판정은 종료 코드.

세 개의 벽을 차례로 만났고, **세 번째에서 멈춥니다.**

| # | 실패 | 성격 |
|---|---|---|
| 1 | `torch/nn/parallel/distributed.py:412` — `class _DDPJoinHook(JoinHook)` | `torch.nn` 이 차단된 `torch.distributed` 에서 **상속**합니다. 스텁이 클래스를 돌려주게 하여 통과 |
| 2 | `... in schema_to_signature_cache` | 컨테이너 프로토콜 요구. 던더를 추가해 통과 |
| 3 | `torch/_ops.py:139` — `AssertionError: expected DispatchKey, got _Meta` | **여기서 끝** |

### 왜 끝인가

세 번째는 스텁을 더 다듬어 넘을 수 있는 종류가 아닙니다. 경로가 이렇습니다.

```
torch/export/decomp_utils.py → torch/_export/__init__.py → wrappers.py
  → torch/_higher_order_ops/__init__.py → _invoke_quant.py → base_hop.py
  → auto_functionalize.py:995 → torch/_ops.py:139
```

`torch._higher_order_ops` 가 **import 시점에 연산자를 등록**하는데, 그 dispatch key 를 차단된
서브트리에서 가져옵니다. 그리고 등록은 **C++ 디스패처가 타입을 검사**합니다 — 파이썬 스텁이
`DispatchKey` 행세를 할 수 없습니다.

**즉 torch 의 서브트리들은 import 시점 연산자 등록으로 서로 엮여 있습니다.** 디렉터리 단위로
분리되지 않습니다.

### 계획에 주는 정정

**3 차의 결론을 정정합니다.** "임포트 그래프를 쳐내면 된다" 가 아니라, 쳐내려면 **코드와 그 등록을
함께 일관되게 제거**해야 합니다 — 즉 소스 수준의 진짜 prune 이고, import 가로채기로 흉내낼 수 있는
것이 아닙니다. 그 작업의 크기는 아직 모릅니다.

**그리고 이것은 §5 의 A/B 판단을 움직입니다.**

- **A(candle + shim)** 는 벤더링한 파이썬 트리를 우리가 관리하므로, 1084 개를 그대로 지고 가든지
  소스 prune 을 직접 하든지 해야 합니다. **이번 결과는 후자가 싸지 않다는 것을 보여줍니다.**
- **B(selective libtorch)** 에는 이 문제가 **아예 없습니다.** 파이썬 트리는 상류 것을 그대로 쓰고
  줄이는 것은 C++ 쪽 op 이므로, 등록 일관성이 저절로 유지됩니다.

§5 는 A/B 를 "빌드 스파이크 한 번으로 판정" 하기로 했는데, **A 쪽에 새 비용 항목이 하나 생겼습니다.**
결정을 §11 의 4 단계에서 내리는 것은 그대로 두되, 이 항목을 판단 재료에 넣어야 합니다.

## 5 차 — 임포트되는 것과 실행되는 것의 비율

4 차에서 A 에 붙은 비용 항목을 정량화하기 위해, **1084 개 중 추론 중에 실제로 파이썬이 실행되는
모듈**을 셌습니다. 모델 생성까지는 추적을 끄고 순전파 · `generate` 구간에만 `sys.settrace` 를
걸었습니다.

| | 개수 |
|---|---|
| 임포트된 torch 모듈 | **1084** |
| 추론 중 파이썬이 실행된 것 | **14** |
| 비율 | **1.3%** |

추적이 살아 있었음을 따로 확인했습니다 — 순전파 1 회에 파이썬 호출 474 건, 그중 torch 186 건.

### 실행된 14 개

```
torch.nn.modules.module      torch.nn.modules.linear
torch.nn.modules.sparse      torch.nn.modules.container
torch.nn.functional
torch._tensor                torch._utils              torch._jit_internal
torch.autograd.grad_mode     torch.utils._contextlib
torch.compiler               torch.jit._trace
torch.cuda.graphs            torch.distributed.distributed_c10d
```

**마지막 넷은 일하는 것이 아니라 질문받는 것들입니다.** `compiler` · `jit._trace` ·
`cuda.graphs` · `distributed_c10d` 가 실행 목록에 있는 이유는 추론 코드가 "너 지금 활성이냐" 를
묻기 때문입니다 — `is_compiling()` · `is_tracing()` · `is_current_stream_capturing()` ·
`is_initialized()`. **쓰이는 것이 아니라 조회되는 것입니다.**

그러면 실제로 일하는 파이썬은 열 개 남짓 — `nn.modules` 넷과 `nn.functional`, 텐서/유틸 셋,
`no_grad` 기계 둘입니다.

### 이것이 A 의 비용을 정의한다

**A 의 비용은 "도는 것을 구현하는 일" 이 아니라 "돌지 않는 것을 임포트되게 만드는 일" 입니다.**

- 추론 중 실행되는 파이썬은 **14 개, 그중 실질은 10 개**. 이것을 우리 `_C` 위에서 동작시키는 것은
  작은 일입니다.
- 그러나 나머지 **1070 개는 임포트만이라도 성공해야** 하고, 4 차가 보여준 대로 **그것을 잘라내는
  것이 싸지 않습니다** — import 시점 연산자 등록이 C++ 디스패처에 걸려 있기 때문입니다.

즉 A 에서 지불하는 것의 대부분은 **한 번도 실행되지 않는 코드를 임포트 가능한 상태로 유지하는 비용**
입니다. B 는 그 코드를 상류 그대로 두므로 이 비용이 0 입니다.

### 한계

- `sys.settrace` 는 **파이썬 프레임만** 봅니다. torch 작업의 대부분은 C++ 이고 여기서는 보이지
  않습니다 — 그것이 바로 `_C` 가 제공해야 할 몫이므로, 이 측정은 "파이썬 계층이 얼마나 얇은가" 를
  말하는 것이지 "일이 얼마나 적은가" 를 말하는 것이 아닙니다.
- 2 층 Llama, 양자화 없음, 기본 어텐션 경로, `do_sample=False`. 더 큰 모델이나 다른 경로는 더
  건드릴 수 있습니다.

## 아직 답하지 않은 것

- 범주 5(`@torch.no_grad()`) 너머는 미탐색입니다. 여기서 멈췄습니다.
  > **정정 (문서 감사, 2026-09):** 위 §"관문 너머" 표의 인라인 정정을 참조 — `docs/models/FROM_CONFIG.md`
  > §3.3 이 이 지점을 이어받아 답했다.
- 위 서브모듈들이 **빈 스텁으로 충분한지** — 즉 transformers 가 그 안의 무엇을 실제로 호출하는지는
  다음 회차의 일입니다.
  > **정정 (문서 감사, 2026-09):** 이 질문은 답해지지 않고 **무의미해졌다** — `docs/design/DESIGN.md`
  > §11(살아 있고, 그 스스로 이 문서의 4·5 차를 근거로 인용한다)이 "A(candle 위 `torch._C`)로
  > 간다"고 최종 결정했고, A 는 스텁이 아니라 **벤더링한 실물 파이썬 트리 전체**를 지고 간다.
  > 그러므로 "빈 스텁으로 충분한가"는 이 프로젝트가 실제로 선택한 경로에서 다시 묻지 않는
  > 질문이 되었다 — 답이 아니라 전제가 사라졌다.
- 이 실험은 **모델 생성(`from_config`)** 까지만 봅니다. 순전파 · `generate` · `online()` 은 그
  너머입니다.
  > **정정 (문서 감사, 2026-09):** 순전파와 `generate` 는 오늘 shim 위에서 실제로 동작한다
  > (`docs/models/CKPT2.md`/`docs/models/GENERATE.md`, 라운드 2 감사 — SmolLM2-135M 실물 체크포인트로
  > 재확인됨).

## 방법에 대한 메모

이 수집은 §6 의 "발견은 shim 이 스스로 한다" 를 파이썬 계층에 적용한 것입니다. 목록을 미리 만들지
않고, 스텁이 실패하면 그 실패가 다음에 필요한 것을 지목하게 했습니다. **요구 순서가 곧 우선순위
순서**라는 성질이 여기서도 성립합니다 — `torch.distributed` 가 먼저 나온 것은 그것이 중요해서가
아니라 import 경로에서 먼저 만나기 때문이고, 그 사실 자체가 "빈 스텁으로 족할 가능성이 높다" 는
신호입니다.
