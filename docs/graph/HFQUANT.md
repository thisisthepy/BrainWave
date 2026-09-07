# `from_pretrained(quantization_config=...)` — 밀집 모델을 한 번도 만들지 않고 양자화 모델을 얻는다

측정일 2026-09-02. 브랜치 `work/hfq`. 호스트 Apple M1 (P 4 + E 4, macOS darwin 25.5.0),
CPython 3.13.0, transformers 5.15.1, candle-core 0.11.0.
**벤더링 트리와 설치된 `transformers` 는 한 줄도 고치지 않았습니다.** 기기는 쓰지 않았습니다.

`docs/graph/QUANT2.md` 가 `quantize_(model, format=...)` 로 닫아둔 자리에서 이어집니다. 그 함수의
한계는 정확도도 형식도 아니라 **시점**입니다 — `from_pretrained` 가 반환한 뒤에 도는 함수라,
어떤 형식을 고르든 **최대 상주 메모리는 밀집 모델**입니다. 135M 에서는 견딜 수 있고 7B 에서는
목적을 스스로 무너뜨리며, 온디바이스가 이 저장소의 전제입니다(`docs/design/DESIGN.md` §1).

> **결론 먼저.**
>
> 1. **SmolLM2-135M 최대 RSS: 밀집 1171.8 MB, `quantize_` 사후 1270.2 MB, 플러그인 874.4 MB.**
>    플러그인이 사후 경로보다 **395.8 MB (31.2%) 낮고**, 밀집 적재보다도 **297.4 MB 낮습니다.**
>    이 회차가 정당화되는 지점이 이것입니다(§2).
> 2. **먼저 하는 것이 같은 것을 합니다.** 플러그인이 만든 블롭이 `quantize_` 가 만든 블롭과
>    **바이트 단위로 동일**하고(210/210 층), 두 모델의 로짓이 **비트 단위로 동일**합니다(§4).
> 3. **`lm_head` 를 조용히 정하지 않습니다.** 기본값은 transformers 자신의 규약
>    (`get_keys_to_not_convert`)이고, 그것을 썼다는 사실이 `model.torchnative_quantization`
>    에 기록됩니다. 명시 리스트를 주면 기본값을 **대체**합니다(§5).
> 4. **담을 수 없는 조합은 거절합니다.** `q4_k` 는 576 폭을 담지 못하므로(§5.2 of QUANT2)
>    SmolLM2 에서 **적재 자체가 거절**됩니다 — 조용히 밀집으로 두지 않습니다(§6).
> 5. **등록은 이름 하나이고, 요청 전에는 아무 일도 하지 않습니다.** `import torchnative.quant`
>    는 `transformers` 를 불러오지 않고 등록도 하지 않습니다(§7).

---

## 0. 이 문서의 숫자로 하면 안 되는 것

- **최대 RSS 의 절대값을 다른 문서와 비교하지 마십시오.** 인터프리터, 벤더링 트리, 체크포인트의
  mmap 이 전부 들어 있고 이 회차는 그중 무엇도 바꾸지 않았습니다. 결론에 쓰는 것은 **같은 기계
  같은 시각에 번갈아 잰 세 프로세스의 차이**입니다. 측정 전후 load average 는 2.70 ~ 2.75 였고
  다른 에이전트를 붙이지 않았습니다.
- **`ru_maxrss` 는 최고 수위선입니다.** 그래서 세 경로를 각각 **별도 프로세스**에서 쟀습니다.
  한 프로세스에서 두 번 적재하면 둘 다 큰 쪽을 보고합니다.
- **속도를 재지 않았습니다.** 이 회차는 시점과 메모리에 대한 것입니다. 층 단위·모델 단위 시간은
  `docs/graph/QUANT2.md` §5.5·§6.3 이고, 그쪽은 이 변경으로 움직이지 않습니다 — 적재가 끝난 뒤의
  모델이 **비트 단위로 같기** 때문입니다(§4).
- **기기 측정이 없습니다.** 전부 호스트 M1 입니다.
- **perplexity 를 재지 않았습니다.** §3 의 정확도는 프롬프트 하나입니다. `docs/graph/QUANT2.md` §5.3
  의 경고가 그대로 적용됩니다.

---

## 1. 왜 `dtype=torch.int8` 이 아닌가 — 직접 확인했습니다

사용자가 원한 철자는 이것입니다:

```python
AutoModelForCausalLM.from_pretrained(name, dtype=torch.int8)
```

**그 철자는 닫혀 있고, 닫은 것은 우리가 아닙니다.** 이 저장소의 코드가 한 줄도 돌기 전에
transformers 가 거절합니다 (실측, 시임 위, transformers 5.15.1):

```
ValueError: LlamaForCausalLM cannot be instantiated under `dtype=torch.int8`
            as it's not a floating-point dtype
```

거절 지점은 `transformers/modeling_utils.py` 의 **`local_torch_dtype`** 입니다.
**이 줄은 원래 `_get_dtype` 이라고 적혀 있었고, 그것은 틀렸습니다** — `docs/numerics/INT8B.md` §1.2 가
고칩니다. `_get_dtype` 은 `torch.int8` 을 그대로 통과시키고 그 안에서
`hf_quantizer.update_dtype(dtype)` 을 부릅니다. 즉 **`quantization_config` 이 함께 주어지면
`update_dtype` 이 먼저 도는 자리가 있고**, 그 자리에서 `torch.int8` 을 받는 것이
`docs/numerics/INT8B.md` 가 한 일입니다.

`quantization_config` 없이 `dtype=torch.int8` 만 주는 철자는 여전히 닫혀 있습니다.
`get_hf_quantizer` 는 `dtype` 을 인자로 받지 않으므로(그 앞에서 이미 `hf_quantizer = None` 이
확정됩니다) 이 저장소의 코드가 닿을 수 있는 훅이 없습니다. 열려면 `transformers` 를 고쳐야
하는데, **이 프로젝트가 하지 않기로 한 유일한 것이 그것입니다**(`docs/design/DESIGN.md` §1 — 파사드를
만드는 순간 임베디드 CPython 의 존재 이유가 사라집니다).

문 자체가 없는 것도 아닙니다. `docs/graph/QUANT.md` §2.1 이 이미 독립적으로 닫아두었습니다 —
candle-core 0.11 의 `DType` 에 `I8` 이 없으므로 **`torch.int8` 텐서가 이 스택에 존재하지
않습니다.** 즉 저 철자는 두 겹으로 닫혀 있습니다.

transformers 가 실제로 내주는 자리는 `quantization_config` 이고, 그것은 **공개 플러그인
API** 입니다 (`transformers.quantizers.auto.register_quantizer`,
`register_quantization_config`, `AUTO_QUANTIZER_MAPPING`). 이 회차가 채운 것이 그 자리입니다.

```python
from torchnative.quant import TorchnativeConfig
m = AutoModelForCausalLM.from_pretrained(name, quantization_config=TorchnativeConfig("q8_0"))
```

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/quant/hf.py TorchnativeConfig present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/quant/hf.py TorchnativeHfQuantizer present -->

---

## 2. 이 회차를 정당화하는 숫자 — 최대 RSS

`HuggingFaceTB/SmolLM2-135M`, `dtype=torch.float32` 를 세 경로 모두에 **명시**해서 활성
dtype 을 맞췄습니다(체크포인트의 config 는 `bfloat16` 을 요청하는데, 그것은 §8 이 다룹니다).
세 경로 모두 **별도 프로세스**, `resource.getrusage(RUSAGE_SELF).ru_maxrss`.

| 경로 | 최대 RSS | 사후 경로 대비 | 모델의 가중치 바이트 |
|---|---:|---:|---:|
| 밀집 `float32` | 1171.8 MB | −98.4 MB | 538.1 MB |
| 밀집 적재 후 `quantize_` (lm_head 제외) | **1270.2 MB** | — | 226.2 MB |
| 밀집 적재 후 `quantize_` (전체) | 1270.5 MB | +0.3 MB | 256.3 MB |
| **플러그인 `TorchnativeConfig("q8_0")`** | **874.4 MB** | **−395.8 MB (−31.2%)** | 226.2 MB |

`import` 직후의 기준선이 세 프로세스 모두 220.0 ~ 220.2 MB 였으므로, 적재가 만든 증가분만
보면 **951.8 / 1050.2 / 654.4 MB** 입니다 — 그쪽으로 보면 감소폭은 **37.7%** 입니다.

**읽어야 할 것 세 가지.**

1. **사후 경로는 밀집 적재보다 비쌉니다** (1270.2 대 1171.8). 당연합니다 — 밀집 모델이 이미
   다 있는 상태에서 양자화 사본을 만들기 시작하고, 교체가 끝나야 밀집 가중치가 풀립니다.
   `quantize_` 로는 **최대 메모리가 개선될 수 없습니다.** 개선되는 것은 적재 이후의 상주량이고,
   그것이 이 회차가 고친 것이 아닙니다.
2. **플러그인은 밀집 적재보다도 쌉니다** (874.4 대 1171.8). 이것이 "해체했다" 가 아니라
   **"조립한 적이 없다"** 는 뜻입니다. 밀집 모델은 어느 시점에도 존재하지 않았습니다.
3. **남은 874 MB 의 대부분은 이 회차의 것이 아닙니다.** 적재 후 모델의 가중치는 226.2 MB 이고
   (임베딩 113.4 + 양자화 112.8), 나머지는 인터프리터·벤더링 트리·체크포인트 mmap 입니다.

### 2.1 무엇이 그 차이를 만드는가

`transformers` 는 `from_pretrained` 안에서 모델을 **meta 장치**로 짓고
(`modeling_utils.get_init_context` → `torch.device("meta")`), 그다음에 체크포인트를 흘려
넣습니다. `HfQuantizer` 의 훅은 그 사이에 있습니다:

```
cls(config)                             meta 스켈레톤  — 저장소 없음
  hf_quantizer.preprocess_model(...)      <- _process_model_before_weight_loading
                                             nn.Linear -> QuantizedLinear (placeholder)
_load_pretrained_model(...)             가중치가 하나씩 디스크에서 온다
  param_needs_quantization(...)           <- 이 키를 우리가 처리한다고 답한다
  get_quantize_ops().convert(...)         <- 여기서 양자화하고 밀집 텐서를 놓는다
```

밀집 텐서의 유일한 참조가 그 `convert` 의 지역 변수이므로, 반환과 동시에 풀립니다. 즉
**한 번에 살아 있는 밀집 가중치는 한 층분**(SmolLM2 에서 최대 1536×576×4 = 3.5 MB)입니다.

부수적으로 `core_model_loading.py` 는 `hf_quantizer` 가 있고 `pre_quantized` 가 거짓이면
**스레드 풀을 끕니다** (`has_on_the_fly_quantization`) — 상류가 같은 이유로 이미 넣어둔
장치이고, 워커가 메인 스레드보다 빨리 밀집 텐서를 쌓는 것을 막습니다.

### 2.2 회귀로부터 지켜집니다

`rust/torch_c/pytests/test_shim.py` 의
`test_the_quantizer_plugin_replaces_the_leaves_before_the_weights_land` 가 68 MB 짜리
로컬 체크포인트로 같은 세 프로세스를 돌리고, **사후 경로 대비 절감이 밀집 가중치의 40% 를
넘을 것**을 요구합니다(실측 절감은 78%).

**이 단언이 실패할 수 있는지 확인했습니다.** `_process_model_before_weight_loading` 을 비우고
`_process_model_after_weight_loading` 에서 `quantize_` 를 부르도록 — 즉 **플러그인의 옷을 입은
사후 교체**로 — 고쳐서 돌렸습니다:

```
FAILED  plugin peak 386.2 MB vs post-hoc 386.2 MB: saved 0.0 MB of 68.0 MB of dense weight.
        The pre-load swap is not saving memory.
FAILED  a 256-block format was accepted on a 64-wide model
```

같은 개조에서 §4 의 동일성 테스트는 **통과합니다** — 사후 교체도 같은 모델을 만들기 때문이고,
그것이 그 테스트가 판정하는 것이 아니기 때문입니다. 두 테스트가 서로 다른 것을 잡습니다.

기계가 시끄러워 RSS 로 판정할 수 없는 날을 위해 두 번째 축이 있습니다: 훅이 스스로
**교체한 모든 잎의 가중치가 저장소 없는 상태였는지**를 기록하고
(`report.swapped_before_weights`), 테스트가 그것을 요구합니다. 사후 교체는 이 값을 참으로
만들 수 없습니다.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_quantizer_plugin_replaces_the_leaves_before_the_weights_land present -->

---

## 3. 플러그인 경로의 결과 — 모듈, 크기, 정확도

`AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M",
dtype=torch.float32, quantization_config=TorchnativeConfig("q8_0"))`:

```
LlamaForCausalLM 1   LlamaModel 1   Embedding 1   ModuleList 1
LlamaDecoderLayer 30   LlamaAttention 30   LlamaMLP 30   SiLUActivation 30
LlamaRMSNorm 61   LlamaRotaryEmbedding 1
QuantizedLinear 210        <- 30 층 x 7
Linear 1                   <- lm_head (기본 규약이 제외)
```

```
storage_bytes  밀집 113.4 MB (임베딩) + 양자화 112.8 MB = 226.2 MB
밀집 모델      538.1 MB (유니크 파라미터, 묶인 head 를 한 번만 셈)
비              2.379x
```

**정확도.** 프롬프트 `[1, 4380, 314, 260, 3387, 286]` (docs/graph/QUANT2.md §5 과 같은 것),
같은 입력에 대한 `float32` 모델과의 로짓 비교:

| | 값 |
|---|---:|
| 로짓 최대 절대차 | 6.634447 |
| 기준 로짓의 최대 절댓값 | 29.981180 |
| 최대 절대차 / 최대 로짓 | 0.221287 |
| **로짓 상대 RMS** | **5.2702%** |
| 원소별 최대 상대 오차 | 111.87 |
| 6 자리 argmax 일치 | 5/6 |

**원소별 최대 상대 오차 111.87 은 정확도 지표가 아닙니다.** 0 에 가까운 로짓 하나에서 나온
값이고 — 분모가 작으면 무엇이든 큽니다 — 여기 적는 이유는 요청받았기 때문입니다. 의미가 있는
것은 최대 로짓 대비 0.221 과 상대 RMS 5.27% 입니다.

**이 숫자들은 `docs/graph/QUANT2.md` §5.3 의 "q8_0 (lm_head 제외)" 열과 소수점까지 같습니다**
(6.634 / 0.2213 / 5.27%). 그 회차는 사후 `quantize_` 로 만든 모델을 쟀고 이 회차는
플러그인으로 만든 모델을 쟀는데, §4 가 보이듯 **두 모델은 비트 단위로 같은 모델**입니다.

---

## 4. 먼저 하는 것이 같은 것을 하는가 — 두 축으로 정확히 대조

더 일찍 하는 것은 **같은 것을 할 때만** 개선입니다. 전치된 텐서나 반쯤 실체화된 텐서를
양자화하는 경로는 싸면서 틀리고, 이 저장소는 그 두 모양을 실제로 만들어낸 적이 있습니다.
그래서 층의 양쪽에서 정확히 비교합니다.

```
층 1  가중치   _C._quantized_blob(플러그인) vs _C._quantized_blob(quantize_)   bytes 의 ==
층 2  출력     플러그인 모델의 로짓 vs quantize_ 모델의 로짓                    비트 동일
```

SmolLM2-135M 실측: **블롭 동일 210/210**, **로짓 최대차 0**, 비트 동일 참.

`test_the_quantizer_plugin_and_quantize_produce_the_same_model` 이 같은 두 축을 로컬 픽스처
모델(14 층)로 회귀에서 지킵니다. **음성 대조가 붙어 있습니다** — 같은 비교를 `q4_0` 모델에
대해서도 돌려 **달라야 한다**고 요구합니다. 달라질 수 없는 두 값의 비트 일치는 아무것도
판정하지 않기 때문입니다.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_quantizer_plugin_and_quantize_produce_the_same_model present -->

---

## 5. `lm_head` — 조용히 정하지 않습니다

`quantize_` 는 `lm_head` 를 특별 취급하지 않고 `predicate` 로 넘깁니다. 이유가 그 독스트링에
있습니다: SmolLM2-135M 에서 **파라미터의 63%** 이고 동시에 **오차가 로짓에 바로 얹히는**
층이며, 둘 다 참이고 반대 방향으로 당깁니다.

플러그인에서 그것에 대응하는 것이 transformers 의 철자인 `modules_to_not_convert` 입니다.

| 준 값 | 무엇이 되는가 |
|---|---|
| `None` (기본) | **transformers 자신의 규약** — `HfQuantizer.get_modules_to_not_convert` → `get_keys_to_not_convert`: 출력 임베딩, 마지막 파라미터, 모든 묶인 가중치. SmolLM2 에서는 `['lm_head', 'model.embed_tokens']` |
| `[]` | 전부 변환. 묶이지 않은 모델에서는 `lm_head` 도 포함됩니다 |
| `[...]` | 준 리스트가 기본값을 **대체**합니다 (더하지 않습니다) |

**어느 쪽이 돌았는지가 기록됩니다.** `model.torchnative_quantization` 이 해석된 리스트와
그것이 기본값이었는지를 함께 출력합니다:

```
format=q8_0 converted=210 left dense=1
  modules_to_not_convert=['lm_head', 'model.embed_tokens'] -- transformers' default
    (get_keys_to_not_convert: output embedding, last parameter, tied weights)
  every replaced leaf was swapped before its weight had storage
  left dense (1): modules_to_not_convert
    e.g. lm_head
```

**기본값을 transformers 의 규약으로 둔 것은 근거가 있습니다.** SmolLM2 는
`tie_word_embeddings=True` 라 `lm_head.weight is embed_tokens.weight` 이고, 그 head 를
교체하면 **묶임이 끊어져 메모리가 늘어납니다** — 전체 적용 2.10 배 대 head 제외 2.38 배
(`docs/graph/QUANT2.md` §5.4). 최대 메모리가 이 회차의 목적이므로, 기본값이 그것을 거스르면 안
됩니다. 속도로는 반대 부호이고(§6.3 of QUANT2), 그래서 **선택지가 남아 있습니다.**

---

## 6. 담을 수 없는 조합 — 거절합니다

### 6.1 블록 크기

`q4_k` 를 SmolLM2 에 걸면 **적재가 거절**됩니다:

```
format='q4_k' cannot be applied to 180 layer(s), so nothing was loaded:
  180 layer(s), e.g. model.layers.0.self_attn.q_proj:
    torch._C._quantize: q4_k stores 256 elements per block, so it must have their last dim
    divisible by block size -- got 576, which is not a multiple of 256 (shape [1, 576])
    Either pick a format whose block size divides this width, or name these layers in
    TorchnativeConfig(modules_to_not_convert=[...]) to load them dense on purpose.
```

**이것이 `quantize_` 와 의도적으로 다른 지점이고, 다른 이유는 판단이 아니라 통로입니다.**
`quantize_` 는 `_Report` 를 반환하므로 스킵을 이유별로 묶어 **보여줄 수 있습니다.**
`from_pretrained` 는 모델만 반환합니다. 같은 "건너뛰고 로그를 남긴다" 를 하면
**양자화된 것처럼 보이는데 실은 전부 밀집인 모델**이 손에 남고 — `q4_k` + 576 조합에서는
정확히 *전부* 입니다 — 그것이 이 저장소가 반복해서 대가를 치른 "성공처럼 읽히는 실패" 입니다.

**규칙을 두 번 쓰지 않았습니다.** 블록 크기는 `rust/torch_c/src/quant.rs` 에만 있습니다.
플러그인은 후보 폭의 1×N 텐서를 **실제로 양자화해 보고** 같은 거절을 받습니다
(`_probe_shape`). 그래서 벽이 움직이면 검사도 함께 움직이고, 이 저장소가 모르는 블록 크기를
가진 형식이 나중에 들어와도 편집이 필요 없습니다.

### 6.2 묶인 head

`modules_to_not_convert=[]` 로 묶인 `lm_head` 를 달라고 하면 거절합니다. 이유가 §5 의
메모리 논거만이 아닙니다 — **체크포인트에 그 텐서가 없습니다.** 묶여 있으므로 안전텐서
파일에는 `model.embed_tokens.weight` 하나뿐이고, 적재 전 훅에는 양자화할 것이 도착하지
않습니다. 거절 메시지가 그 사실과 우회로(`quantize_`)를 함께 말합니다.

### 6.3 그 밖의 거절

| 무엇 | 언제 | 무엇이라고 |
|---|---|---|
| 알 수 없는 형식 | `TorchnativeConfig("q9_9")` **구성 시점** | `unknown quantisation format 'q9_9'. This build has: ...` |
| `modules_to_not_convert="lm_head"` | 구성 시점 | 문자열이 아니라 리스트라고 말하고 `['lm_head']` 를 제시 |
| 시임이 아닌 `torch` | 구성 시점 | `torch._C has no _quantize` + 실제로 열린 `torch.__file__` |
| `bfloat16` 활성 | `update_dtype` | §8 |
| `float64` 등 | `update_dtype` | candle `QMatMul` 이 f32/f16 만 받는다고 이름을 대고 거절 |
| 이미 양자화된 체크포인트 | `validate_environment` | 그런 체크포인트는 존재할 수 없다 (`state_dict` 에 `qweight` 가 없음, QUANT2 §4) |
| 아무것도 변환되지 않음 | 적재 전 훅 끝 | 밀집으로 조용히 통과시키지 않고 거절 |
| placeholder 가 가중치를 못 받음 | 적재 후 훅 | 첫 forward 가 아니라 여기서 |

형식을 구성 시점에 검사하는 이유: 오타 하나가 **수 GB 를 내려받은 뒤에** 드러나면 안 됩니다.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_quantizer_plugin_refuses_the_combinations_that_cannot_work present -->

---

## 7. 등록은 이름 하나이고, 요청 전에는 아무 일도 하지 않습니다

두 가지가 각각 지켜집니다.

1. **`import torchnative.quant` 는 `transformers` 를 부르지 않습니다.** `transformers` 는
   경성 의존이 아니므로, 그것이 없는 기계에서도 `quantize_` 는 그대로 써야 합니다.
   `TorchnativeConfig` 는 PEP 562 `__getattr__` 로 지연 반입되고, 그때에야 `hf.py` 가 로드되며
   그때에야 등록이 일어납니다.
2. **등록해도 기본값이 바뀌지 않습니다.** `AutoHfQuantizer.from_config` 는 넘겨받은
   `quantization_config.quant_method` 로 분기하므로, `TorchnativeConfig` 를 주지 않은 적재는
   이 이름이 표에 있든 없든 **완전히 같은 경로**를 탑니다.

실측(테스트가 이 순서 그대로 확인합니다):

```
import torchnative.quant           -> "transformers" in sys.modules == False
import transformers                -> "torchnative" in AUTO_QUANTIZER_MAPPING == False
from torchnative.quant import TorchnativeConfig
                                   -> "torchnative" in AUTO_QUANTIZER_MAPPING == True
                                      AUTO_QUANTIZATION_CONFIG_MAPPING["torchnative"] is TorchnativeConfig
_hf._register()  (두 번째)          -> 예외 없음
```

`register_quantizer` 는 같은 이름을 두 번 받으면 예외를 던집니다 — 레지스트리로서는 옳고,
경로가 둘일 수 있는 모듈에게는 곤란합니다. 그래서 `_register()` 는 모듈 수준 플래그가 아니라
**표 자체**를 보고 건너뜁니다.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_quantizer_registers_a_name_and_changes_nothing_else present -->

---

## 8. `bfloat16` — 넓히고, 넓혔다고 말합니다

SmolLM2-135M 의 `config.json` 은 `bfloat16` 을 요청합니다. candle 의 `QMatMul::forward` 는
`f32` 와 `f16` 만 받습니다(`docs/graph/QUANT2.md` §7, 벽 4). 즉 **기본 경로가 곧 그 벽입니다.**

`update_dtype` 이 `float32` 로 넓히고 경고를 냅니다. 상류의
`Bnb8BitHfQuantizer.update_dtype` 이 같은 이유로 하는 것과 같은 조치입니다.

**경고가 장식이 아닌 이유:** 넓히면 교체되지 *않은* 전부가 두 배가 됩니다. 묶인 모델에서
그것은 임베딩이고, 남아 있는 것 중 가장 큽니다(113.4 MB → `bfloat16` 이었다면 56.7 MB).
`bfloat16` 밀집 적재와 이 문서의 숫자를 비교하는 독자는 **활성 dtype 이 다르다**는 것을
알아야 합니다. `dtype=torch.float16` 을 주면 넓히지 않습니다.

§2 의 세 측정은 이 효과를 없애려고 **세 경로 모두에 `dtype=torch.float32` 를 명시**했습니다.

---

## 9. 무엇을 만들었는가

| 층 | 무엇 |
|---|---|
| `torchnative/quant/hf.py` | **새 파일.** `TorchnativeConfig` · `TorchnativeHfQuantizer` · `_QuantizeOnLoad` · `_LoadReport` · `_register` |
| `torchnative/quant/__init__.py` | `QuantizedLinear.pending_from_linear` · `QuantizedLinear.adopt` · `forward` 의 미착 가중치 거절 · PEP 562 `__getattr__` |
| `rust/torch_c/pytests/test_shim.py` | 4 개 테스트 (329 → 333) |
| `rust/torch_c/src/` | **변경 없음.** Rust 는 한 줄도 고치지 않았습니다 |

### 9.1 `QuantizedLinear` 를 다시 만들지 않았습니다

`QuantizedLinear`·`quantize_`·`storage_bytes`·`FORMATS` 는 이미 있던 것이고 플러그인은 그것을
씁니다. 추가된 것은 **적재용 placeholder 두 메서드**뿐입니다.

placeholder 가 필요한 이유는 transformers 의 키 대조입니다. 적재기는 체크포인트 키를
`model.state_dict()` 와 맞추는데, `QuantizedLinear` 는 설계상 `weight` 를 `state_dict` 에
두지 않습니다(그 독스트링 — `.to()` 가 밀집 커널을 부르게 되기 때문). 그러면
`...q_proj.weight` 가 **unexpected key** 가 되어 텐서가 아예 읽히지 않고 층은 조용히 빈 채로
남습니다. 그래서 교체 시점에는 원래 `nn.Linear` 의 **meta `Parameter` 를 그대로 물려받아**
등록해 두고, `adopt` 가 실제 텐서를 받아 양자화한 뒤 그 placeholder 를 지웁니다. 적재가 끝난
모델의 `state_dict` 에는 그 키가 없습니다(§4 의 `dense_weight_keys_left == []`).

`_QuantizeOnLoad.convert` 가 **빈 dict 를 반환하는 것**도 같은 계약 때문입니다. 상류의 기본
경로는 op 의 반환값을 `set_param_for_module` 로 보내고, 거기서 평범한 텐서는
`torch.nn.Parameter` 로 감싸집니다 — 양자화 텐서가 `Parameter` 가 되면 안 되는 이유가
`QuantizedLinear` 독스트링에 있습니다. 그래서 모듈에 직접 넣고 빈 것을 돌려주며,
`missing_keys` 에서 그 키를 지워 적재기가 "안 채워졌다" 고 보고하지 않게 합니다.

---

## 10. 확인하지 않은 것

| 항목 | 상태 |
|---|---|
| 7B 급 모델 | **없음.** §2 의 논거는 135M 에서 잰 것이고, 절감의 *비율*이 모델 크기에 어떻게 가는지는 재지 않았습니다 |
| 기기 (안드로이드 · iOS) | **없음** |
| 속도 | **없음.** 적재 후 모델이 §4 로 같으므로 QUANT2 §6.3 이 그대로 적용된다고 봅니다만, 다시 재지는 않았습니다 |
| `device_map` 을 쓴 적재 | **거절합니다** (`cpu` 외). 오프로딩 경로는 시험하지 않았습니다 |
| 샤딩된 체크포인트 | **시험 안 함.** SmolLM2 는 단일 파일입니다 |
| `save_pretrained` | **불가.** `is_serializable()` 이 거짓입니다. 시임에서는 `TensorBase.untyped_storage` 도 없어 밀집 모델조차 쓰지 못합니다(그래서 테스트 픽스처를 상류 쪽 프로세스가 씁니다) |
| perplexity · 여러 프롬프트 | **없음** (§3) |
| `transformers` 가 없는 기계 | 부분적. `import torchnative.quant` 가 `transformers` 를 부르지 않음은 확인했습니다(§7). `transformers` 를 **제거한** 환경에서 돌려보지는 않았습니다 |
| 다른 아키텍처 (Qwen, Mistral, ...) | **없음.** Llama 계열만 |

---

## 11. 검증 기준선

이 작업 전후 모두, 전부 exit 0:

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh     329 -> 333  (+4),  DOCWATCH 241 -> 248
$PY tools/golden/compare.py                   7751/7751, ops=168   (변화 없음)
```

**골든이 한 비트도 안 움직였습니다.** 이 회차는 Rust 를 건드리지 않았습니다.

<!-- DOCWATCH: count smoke_ok ge 333 -->

### 11.1 재현

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
PY=/Volumes/macMini/caches/spike-venv/bin/python
cd /Volumes/macMini/worktrees/bw-hfq
bash vendor/vendor_torch.sh && bash vendor/install_shim.sh

PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY <스크립트>
```

측정 스크립트는 저장소 밖 `/Volumes/macMini/caches/hfq-scratch/` 에 있습니다:

| 파일 | 무엇 | 절 |
|---|---|---|
| `probe_int8.py` | `dtype=torch.int8` 의 거절 | §1 |
| `rss.py` | 최대 RSS, 경로당 1 프로세스 (`dense`/`posthoc`/`posthoc_all`/`plugin`) | §2 |
| `accuracy.py` | 로짓 정확도 + 블롭·로짓 동일성 | §3, §4 |
| `behave.py` | 거절과 명시 리스트 | §5, §6 |

---

## 12. 보고 분류

`CLAUDE.md` §5.3.

| 종류 | 무엇 |
|---|---|
| **기능 추가** | `torchnative.quant.hf` 전체 (transformers 플러그인 등록, 적재 전 교체, 적재 중 양자화, 적재 후 보고) · `QuantizedLinear.pending_from_linear` / `adopt` |
| **결함 수정** | 없음. **골든 7751 개가 변화 없음** |
| **테스트 추가** | 4 개 (329 → 333). 그중 하나는 **최대 RSS 를 프로세스 간 차이로 판정**하고(§2.2 에서 개조 주입으로 실패 가능성 확인), 하나는 **블롭 바이트 + 로짓 비트 동일성**을 음성 대조와 함께 봅니다 |
| **측정** | 최대 RSS 세 경로(§2) · 플러그인 모델의 정확도와 크기(§3) · 사후 경로와의 정확 대조(§4) · `dtype=torch.int8` 거절의 위치와 문구(§1) |
| **문서 정정** | 없음. `docs/graph/QUANT2.md` §5.3 의 "q8_0 (lm_head 제외)" 열이 이 경로에서 **재현**됩니다(§3) |
| **삭제** | 없음 |
