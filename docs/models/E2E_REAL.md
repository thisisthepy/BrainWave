# 진짜 모델을 끝까지 — `print(tensor)` 와 순전파

> **§6.2 와 §7 의 1·2 번, §9 의 3 번은 뒤에 나온 측정으로 바뀌었습니다 — `docs/models/CKPT2.md`.**
> `UntypedStorage.from_file` 이 구현되어 **`from_pretrained` 가 가중치를 읽습니다.**
> 상류가 쓴 체크포인트를 네 경로(safetensors mmap/bytes, `.bin` mmap/no-mmap) 전부로 읽어
> 가중치가 **비트 단위로 일치**하고, 순전파 로짓이 상류와 **2.235e-08** 입니다.
> 허브의 진짜 사전학습 모델(SmolLM2-135M, 273 텐서)도 **가중치는 전부 읽습니다**;
> ~~그 모델의 순전파는 커널 둘이 남아 아직 안 됩니다(CKPT2.md §7.1)~~ **정정 (문서 감사,
> 2026-09): 그 둘도 닫혔습니다** — `CKPT2.md` §7.1 자신의 정정을 보십시오. 순전파와
> `generate()` 모두 오늘 재확인했습니다(SmolLM2-135M, 실측).
> **§7 의 미측정 항목 셋** — shared tensor · sharded index · `_metadata` — 도
> 전부 측정되어 통과했습니다(CKPT2.md §6).
>
> 아래 §6.2 의 한 문장은 그 자리에서 **틀렸습니다**: `disable_mmap=True` 가 우회하지
> 못하는 것은 `.bin` 경로뿐이고, safetensors 쪽은 그때 이미 21/21 텐서를 읽고 있었으며
> 막고 있던 것은 `torch.empty_like` 였습니다. CKPT2.md §3.1 에 그 회차별 진행이 있습니다.

`docs/distributed/DISTRIBUTED.md` 가 열어 둔 자리에서 이어지는 작업의 기록입니다. 그 회차는
`AutoModelForCausalLM.from_config(...)` 로 **진짜 `LlamaForCausalLM` 을 만드는 데까지** 갔고,
두 가지에서 멈췄습니다 — 순전파가 `torch._C.is_autocast_enabled` 에서 죽었고,
`docs/platform/WHEEL.md` §5 가 기록한 대로 **텐서를 찍을 수가 없었습니다.**

측정일 2026-08-28. 호스트 `darwin/arm64`, CPython 3.13.0, 상류 torch 2.13.0 ·
transformers 5.15.1 (`/Volumes/macMini/caches/spike-venv`). **벤더링 트리는 한 줄도 고치지
않았습니다.**

---

## 0. 한눈에

| | 전 | 후 |
|---|---|---|
| `print(tensor)` | **실패** — `_functorch.is_functorch_wrapped_tensor` | **통과, 상류와 문자열까지 동일** (§2) |
| `repr` 대조 케이스 | 없음 | **17/17 이 상류 2.13.0 과 `==`** |
| `from_config` 모델의 **순전파** | **실패** — `is_autocast_enabled` | **통과.** 로짓 최대 오차 **2.24e-08** (§4) |
| 그 로짓의 argmax | — | 상류와 **완전 일치** (`[14, 20, 28, 27]`) |
| `from_pretrained` — **모델 생성** | 미시도 | **통과** (§6.1) |
| `from_pretrained` — **가중치 적재** | 미시도 | **미통과.** `UntypedStorage.from_file` (§6.2) |
| `torch.set_default_dtype` | 이름을 대고 거절 | **실물 전역.** 읽는 자리 12곳이 전부 따라갑니다 (§6.1) |
| `pytests/run.sh` | 129 통과 | **142 통과** |
| `tools/golden/compare.py` | 2383/2383, ops=109 | **2486/2486, ops=116** |
| `verify_schemas.py` | 255/255 | **270/270** |

**이번 회차에 실제로 열린 것 셋**: 텐서를 찍을 수 있게 되었고, **손으로 옮겨 적지 않은**
`transformers` 모델이 순전파를 돌며, `from_pretrained` 가 모델을 **만드는 데까지** 갑니다.
이 저장소의 모델 대조는 지금까지 전부 `test_shim.py` 가 op 단위로 전사한 디코더였습니다
(`_e2e_build`/`_e2e_forward`) — 그것은 커널을 증명하지 트리를 증명하지 않습니다.

**아직 안 되는 것**: `from_pretrained` 의 **가중치 적재**. §6.2 에 이름이 있습니다.
디스크의 체크포인트를 읽는 길 자체는 서 있습니다 — `torch.load(..., mmap=False)` 는 21/21
텐서를 읽습니다. 막힌 것은 **mmap 진입점 하나**이고, safetensors 경로와 `.bin` 경로가
둘 다 그 이름으로 모입니다.

---

## 1. 방법 — 추측하지 않고 계측기를 상류에 붙였다

직전 회차와 같은 방법을 썼습니다: 막힌 것을 하나씩 따라가 목록을 만들고, **그 목록을 사양으로
삼습니다.** 다만 `repr` 에는 더 나은 계측기가 하나 더 있었습니다.

### 1.1 상류의 `repr` 이 실제로 무슨 op 을 부르는지 잰다

`torch/_tensor_str.py` 의 `_str` 은 첫 줄이 이렇습니다:

```python
def _str(self, *, tensor_contents=None):
    with torch.no_grad(), torch.utils._python_dispatch._disable_current_modes():
```

그래서 **밖에서 `TorchDispatchMode` 를 걸면 아무것도 기록되지 않습니다.** 실제로 처음 시도했을
때 14개 텐서 전부에 대해 빈 목록이 나왔습니다. `_str_intern` 을 그 가드 *안쪽에서* 직접 부르게
바꿔 다시 재니 답이 나왔습니다.

`_str_intern` 의 모든 분기(빈 텐서 · 0차원 · 정수 · bool · 요약 · 과학표기 · 전부 0 ·
inf/nan · requires_grad)에 닿는 텐서 14개에 대해, **상류의 `repr` 은 정확히 21개 aten op 을
디스패치합니다.** 그중 15개는 이미 커널이 있었고 6개가 없었습니다.

    aten.abs.default   aten.ceil.default   aten.gt.Scalar
    aten.min.default   aten.unbind.int     aten.masked_select.default

**`torch.isfinite` 는 이 목록에 없고, 그 사실이 측정의 값어치입니다.** `_tensor_str.py:155`
가 `torch.isfinite(...)` 를 부르므로 shim 에서는 그 이름이 거절했지만, 상류에서는
`CompositeImplicitAutograd` 라 로거에는 `eq.Tensor`/`abs`/`ne.Scalar`/`mul.Tensor` 로 보입니다.
**`aten.isfinite.default` 는 한 번도 발화하지 않습니다.** 그것을 커널로 이름 붙였다면 상류에도
없는 작업 항목을 하나 만들어 놓는 셈이었고, 이는 `overloads.json` 이 `layer_norm` 에 대해
적어 둔 것과 같은 실수입니다.

`docs/platform/WHEEL.md` §5 가 "최소 8개" 라고 적은 목록과 이번 목록이 다른 이유도 여기 있습니다.
그 목록은 막힌 이름을 `False` 로 갈아끼우며 얻은 것이라 **거절한 이름**의 목록이지
**필요한 셈**의 목록이 아니었습니다. 실제로 그중 `torch.get_default_dtype` 은 직전 회차가
이미 구현했고(`docs/distributed/DISTRIBUTED.md` §3.4), 대신 그 목록에 없던 6개의 커널이 필요했습니다.

### 1.2 순전파는 예외를 하나씩 따라갔다

여기에는 상류 계측기를 쓸 수 없습니다 — 상류는 그냥 돌기 때문에 "무엇이 없는가" 를 알려주지
않습니다. 그래서 실패한 이름에 프로브 값을 꽂고 다시 돌리기를 반복했고, 마지막에 **실제로
호출된 프로브만 세었습니다.** 미리 넣어둔 8개 중 실제로 쓰인 것은 둘뿐입니다:

```
      3  torch._C._is_tracing        ARGS=()
      1  torch._C.is_autocast_enabled ARGS=('cpu',)
```

`get_autocast_dtype` · `set_autocast_enabled` · `clear_autocast_cache` · `_increment_version`
등은 **한 번도 불리지 않았습니다.** 넣어 두었더라면 "필요해서 넣었다" 고 쓸 뻔했습니다.

---

## 2. `print(tensor)` — 되는 것과 맞는 것

```
>>> print(torch.ops.aten.mm.default(torch.ones(3, 4), torch.ones(4, 2)))
tensor([[4., 4.],
        [4., 4.],
        [4., 4.]])
```

**되는 것으로는 끝이 아닙니다.** 텐서 서식화는 값이 맞으면서 열 너비 · 정밀도 · `...` 축약 ·
`dtype=` 접미사 · 줄바꿈이 틀릴 수 있는 자리이고, 그 전부를 사용자가 읽습니다. 그래서
`test_repr_matches_upstream_character_for_character` 는 **문자열을 `==` 로 비교합니다** —
기대값은 이 인터프리터의 상류 torch 2.13.0, 실제값은 서브프로세스의 벤더 트리.

대조 케이스 17개는 §1.1 의 디스패치 트레이스를 뜬 그 목록이고, 각각이 `_str_intern` 의 다른
분기에 닿습니다. **17/17 이 문자 단위로 같습니다.**

### 2.1 술어 11개 — 왜 `False` 가 답이라고 말할 수 있는가

`repr` 이 숫자를 찍기 전에 묻는 술어가 11개입니다. 전부 `False` 를 돌려주고, 그것은
**CLAUDE.md §5.5 가 의심하라고 한 모양**입니다. 근거를 셋으로 나눕니다.

**(a) 장치에서 유도된 것 — 애초에 상수가 아니다.**
`is_mps` · `is_xpu` · `is_maia` 는 `device.type == ...` 이고, `tensor.rs` 의
`is_cpu`/`is_cuda`/`is_meta` 가 이미 쓰던 그 유도입니다. `torch.zeros(2, device="meta").is_meta`
는 이 빌드에서 **`True`** 이므로 이 계열은 상수가 아닙니다. 테스트는 값이 아니라
**`.device` 와의 일치**를 단언합니다. `mps` 는 특히 — `torch.device("mps")` 는 이 shim 에서
만들어지는 라벨이므로, 이 술어는 백엔드 하나 거리에서 스스로 `True` 를 낼 수 있습니다.

**(b) 표현에서 유도된 것 — 망라적 `match`.**
`is_nested` · `is_sparse` · `is_quantized` · `_is_zerotensor` · `is_neg` · `layout` 은
`Repr` 열거형에 대한 **망라적 `match`** 로 씁니다. 이 빌드의 표현은 정확히 둘입니다 —
candle 의 조밀 strided 버퍼와, 모양·dtype 만 있는 `Meta`. 팔을 하나 더 붙이면 **컴파일러가
이 여섯에 대해 답을 요구합니다.** 맨 `false` 였다면 조용히 상속됐을 것이고, 그것이
`docs/distributed/DISTRIBUTED.md` §8.1 의 `is_mutable` 사고가 난 모양 그대로입니다.

**(c) 그리고 그 근거를 반증 가능하게 만드는 테스트.**
`test_the_alternative_representations_have_no_constructors` 는 이 표현들로 들어가는 **모든
입구가 이름을 대고 거절하는지**를 단언합니다 — 12개 전부:

```
torch.sparse_coo_tensor    torch.sparse_csr_tensor    torch._efficientzerotensor
torch._neg_view            torch._to_functional_tensor torch.quantize_per_tensor
torch.nested.nested_tensor torch._nested_tensor_from_tensor_list
torch.Tensor.to_sparse     _functorch._wrap_for_grad
_functorch._add_batch_dim  _functorch._vmap_increment_nesting
```

즉 `False` 는 **생성자 집합에서 유도되는 것**이지 적어 놓은 값이 아닙니다. 그중 하나라도
착지하면 이 테스트가 깨지고, 조용히 `False` 라고 답하던 술어가 **누군가 알아차린 거짓말**이
됩니다. 이것이 불변식과 가정의 차이입니다.

> **일부러 깨서 확인했습니다.** `_efficientzerotensor` 를 만들어지게 바꾸니
> `FAIL ... ('torch._efficientzerotensor', 'MADE')`.

**(d) functorch 는 진짜로 비어 있는 스택을 읽는다.**
`is_functorch_wrapped_tensor` 는 상수가 아니라 `maybe_get_level(t) != -1` 이고,
`maybe_get_level` 은 **동적 레이어 스택**을 읽습니다. 상류에서 실측한 대로 — `vmap` 밖에서
`-1`, 안에서 `1`. 이 shim 의 스택은 진짜 리스트이고 깊이가 0인데, 거기에 밀어 넣는 모든 것이
거절하기 때문입니다. functorch 가 언젠가 착지하면 미는 쪽이 바뀌고 이 셋은 **건드리지 않아도**
따라옵니다.

### 2.2 `bool * bool` — 두 연산이 겹치는 자리

`torch.isfinite` 의 상류 본문은 `(self == self) * (self.abs() != inf)` 이고, 이는 bool 텐서
둘의 곱입니다. shim 은 이것을 거절하고 있었습니다 — `BOOL.md` §2.2 가 "bool 산술은 논리 연산" 이라
적었고 `arith_tag` 가 전부 막았기 때문입니다.

**`mul.Tensor` 만 예외로 열었고, 편해서가 아니라 두 연산이 겹치기 때문입니다.**
`BOOL.md` §6.3 이 `bool` 태그에 붙인 불변식 — 바이트는 0 또는 1, `boolean()` 만이 그 태그를
붙일 수 있음 — 아래에서 산술 곱은 **논리곱 그 자체**입니다: 1·1=1, 1·0=0, 0·0=0, 다른 쌍은
존재하지 않습니다. `+` 는 그렇지 않고(2가 나와 클램프가 필요합니다), `-` 와 `/` 는 상류가
직접 거절합니다.

**스칼라 오버로드는 계속 거절합니다.** 이것은 신중함이 아니라 사실의 차이입니다 —
`bool_tensor * 2` 는 상류에서 `int64` 로, `* 1.5` 는 `float32` 로 승격되므로 스칼라 형태는
애초에 논리곱이 아니고, 필요한 승격 규칙은 여기 없습니다.

---

## 3. 이번에 드러난 결함 둘 — 없어서 막힌 것이 아니라 틀린 답을 하고 있던 것

### 3.1 `max.default` 가 NaN 을 삼키고 있었다

```
상류 2.13.0   torch.tensor([3., nan, 1.]).max()  ->  tensor(nan)
이 빌드(전)                                       ->  3.0
```

candle 의 리덕션은 NaN 을 **건너뜁니다.** torch 의 규칙은 IEEE *maximum* 이지 `fmax` 가
아닙니다 — NaN 을 이길 수 있는 순서가 없으므로 입력에 NaN 이 하나라도 있으면 그것이 답입니다.

**왜 여태 안 걸렸는가**: `max_default_cases` 에 NaN 케이스가 없었습니다. 같은 하네스의
`_pair_result_check` 는 `sort`/`topk` 를 위해 NaN 을 명시적으로 다루므로, **이 하네스는 처음부터
이것을 잡을 수 있었고 그저 물어본 적이 없었습니다.** `CLAUDE.md` §5.5 의 "실패할 수 없는 검증"
과 같은 계열입니다 — 이쪽은 검증이 실패할 수 없었던 게 아니라 아예 존재하지 않았습니다.

`min.default` 를 새로 만들면서 `max` 와 한 함수로 합치고 둘 다 고쳤습니다. NaN 검사는
`x != x` 를 합치는 벡터 연산 한 번이고, 부동소수 dtype 에서만 돕니다. 골든 케이스는 양쪽에
있습니다.

> **일부러 깨서 확인했습니다.** NaN 분기를 무력화하니
> `FAIL aten.max.default :: ... torch=nan c=3.0` 과 `FAIL aten.min.default :: ... torch=nan c=1.0`.

### 3.2 `cat` 이 "legacy empty" 를 몰랐다

`transformers` 의 KV 캐시는 매 레이어를 `torch.tensor([])` 로 시작하고
`torch.cat([self.keys, key_states], dim=-2)` 로 키웁니다 (`cache_utils.py:144`). 즉 **모든 모델의
첫 어텐션 레이어가 1차원 빈 텐서와 4차원 텐서를 이어붙입니다.** shim 은 여기서
`IndexError: Dimension out of range` 를 냈고 순전파가 거기서 멈췄습니다.

상류 규칙은 좁고, 실측했습니다:

- **모양이 정확히 `(0,)` 인 것만** 건너뜁니다. `torch.ones(0, 5)` 는 비어 있어도 건너뛰지
  **않고** `Tensors must have same number of dimensions: got 2 and 4` 를 냅니다. 즉 판정 기준은
  "비었는가" 가 아니라 "`(0,)` 인가" 입니다. **"빈 텐서를 건너뛴다" 는 그럴듯한 과일반화이고
  틀립니다.**
- 건너뛴 항목은 rank 검사 · `dim` 검사 · extent 검사 어디에도 참여하지 않습니다.
- 전부 건너뛰면 결과는 `dim` 이 무엇이든 `(0,)` 입니다 — `cat([e, e], dim=5)` 도 에러가 아닙니다.
- **dtype 에는 참여합니다.** `cat([int64 (0,), int32 (2,3)])` 은 `int64` 입니다. 이 shim 은
  혼합 dtype 을 전처럼 거절합니다 — 승격은 별도의 갭이고 자기 거절이 있으며, 모양은 건너뛰면서
  dtype 은 조용히 버리는 것은 둘 중 어느 쪽도 아닌 세 번째 동작이 됩니다.

---

## 4. 순전파 — 손으로 옮겨 적지 않은 모델

```python
model = AutoModelForCausalLM.from_config(LlamaConfig(
    vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
    tie_word_embeddings=False))
model.eval()
model.load_state_dict(deterministic_weights)      # RNG 없이, 양쪽이 같은 절차로
logits = model(torch.tensor([[3, 7, 1, 19]])).logits
```

| | |
|---|---|
| 모델 클래스 | `LlamaForCausalLM` (transformers 5.15.1) |
| 로짓 모양 | `(1, 4, 32)` `float32` |
| argmax | `[14, 20, 28, 27]` — **상류와 동일** |
| 로짓 최대 오차 | **2.24e-08** |

양쪽 인터프리터가 **같은 transformers** 를 쓰고 밑에 깔린 `torch` 만 다릅니다. 가중치는
RNG 없는 생성기에서 `load_state_dict` 로 밀어 넣으므로 어느 쪽도 상대의 난수열에 의존하지
않고, `state_dict` 키 순서와 모양은 이 파일이 아니라 `transformers` 가 정합니다.

### 4.1 이 순전파가 실제로 부르는 op — 147 노드, 26 종

shim 자신의 캡처(`_capture_begin`/`_capture_end`)로 잰 것입니다:

```
  16 matmul.default      20 mul.Tensor        15 t.default         9 cat.default
   9 slice.Tensor         9 transpose.int      8 add.Tensor         8 unsqueeze.default
   6 add.Scalar           6 view.default       5 mean.dim           5 pow.Tensor_Scalar
   5 rsqrt.default        4 contiguous.default 4 lift_fresh.default 4 neg.default
   2 _sdpa_flash_cpu      2 mul.Scalar         2 reshape.default    2 silu.default
   1 _to_copy.default     1 arange.default     1 cos.default        1 embedding.default
   1 expand.default       1 sin.default
```

### 4.2 이 테스트의 허용오차는 파일 기본값을 쓰면 **실패할 수 없다**

`_E2E_LOGIT_ATOL` 은 1e-5 **절대값**이고, 이 파일의 다른 비교가 쓰는 64폭·100어휘 디코더에
맞춰진 값입니다. 이 모델은 16폭·32어휘라 로짓이 0.05 근처이므로 1e-5 는 **로짓의 약 20%** 입니다.

숫자를 고르는 대신 쟀습니다 — shim 쪽에서만 `aten.silu.default` 출력에 배율을 넣고 같은 비교를
다시 돌렸습니다:

```
정상                         2.24e-08
silu x 1.0001 (0.01% 높게)   1.42e-07     1e-5 미만 — 못 잡음
silu x 1.001  (0.1%  높게)   1.44e-06     1e-5 미만 — 못 잡음
```

그래서 이 테스트의 경계는 **5e-7** 입니다. 정상값의 22배 위이고, 파일 기본값이 통과시키는
0.1% 오차를 **잡습니다.** 0.01% 는 **못 잡고, 그것을 숨기지 않고 적습니다** — 이 크기의 모델은
활성 함수 한 곳의 오차를 희석하며, 그 답은 더 촘촘한 숫자가 아니라 더 큰 모델입니다.

> 이 절 전체가 §3.1 과 같은 교훈입니다. 처음에 `_E2E_LOGIT_ATOL` 을 그대로 쓴 테스트는
> 초록이었고, **`silu` 를 0.1% 틀리게 만들어도 계속 초록이었습니다.** 그때 다른 두 테스트
> (전사한 디코더 쪽)는 빨갛게 변했는데, 그 대비가 아니었으면 알아채지 못했을 것입니다.

### 4.3 autocast — 읽기가 상수가 아닌 이유는 옆의 거절이다

`is_autocast_enabled` 는 **진짜 플래그를 읽고**, `set_autocast_enabled(dev, True)` 는
**이름을 대고 거절합니다.** autocast 는 op 의 입력을 낮은 정밀도로 캐스팅하는 디스패치 키인데
이 shim 에는 그 키가 없고 캐스팅하는 커널도 없습니다. `True` 를 돌려주면 블록 안의 모든 op 이
`float32` 로 돌면서 호출자는 `bfloat16` 으로 돌았다고 믿게 됩니다 — 모양이 맞는 틀린 숫자입니다.

`False` 로 되돌리는 것은 받습니다. 이미 참이고, `torch.autocast.__exit__` 가 하는 일이 그것이라
거절하면 안 됩니다. 즉 플래그는 **결코 올라갈 수 없고**, 그래서 읽기가 유도된 것이 됩니다.
`_install_repr_surface` 의 functorch 스택과 같은 모양입니다.

`get_autocast_dtype` 은 상류의 장치별 **기본값**을 답합니다 — autocast 블록이 없어도
`torch._C.get_autocast_dtype("cpu")` 는 상류에서 `torch.bfloat16` 입니다(실측). 플래그가 내려가
있는 동안 아무도 이 값을 읽지 않으므로 무엇도 켜지지 않고, 대신 `float32` 라고 답했다면
그것은 커널이 하는 일을 우연히 묘사하는 지어낸 숫자였을 것입니다.

`_is_tracing` 은 `_get_tracing_state` 에서 유도합니다. 상류의 `_is_tracing()` 은 "추적 상태가
있는가" 이고 이 프로세스에 그 답은 하나뿐입니다. 상수 둘은 서로 어긋날 수 있지만 이것은 그럴 수
없습니다.

---

## 5. 검증

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh    exit 0   142 통과 (전 129, +13)
$PY tools/golden/compare.py                  exit 0   2486/2486, ops=116 (전 2383/109)
$PY rust/torch_c/pytests/verify_schemas.py   exit 0   270/270 (전 255)
```

테스트 +13 은 §2~§4 가 9개, §6.1 이 4개입니다.
골든 케이스 +103: 새 커널 7개에 대한 102개(다른 세션이 `tools/golden/cases.py` 에 작성)와,
§3.1 의 `max` NaN 1개. §6.1 의 메타 커널 넷은 `ops covered` 를 늘리지 않습니다 — 이미 세어지던
키의 메타 경로이기 때문입니다.

**보고를 종류별로 나눕니다** (`CLAUDE.md` §5.3):

| 종류 | 무엇 |
|---|---|
| 기능 추가 | 커널 7개(`abs`·`ceil`·`gt.Scalar`·`gt.Tensor`·`masked_select`·`min.default`·`unbind.int`) + 메타 커널 4개 · `torch.isfinite` 합성 · `repr` 술어 11개 · autocast 3개 · `_is_tracing` · `set_default_dtype` 전역 |
| 결함 수정 | `max.default` 의 NaN(§3.1) · `cat` 의 legacy-empty(§3.2) · `full` 과 `torch.finfo()` 가 기본 dtype 을 직접 적어 두던 것(§6.1) |
| 테스트 추가 | 13개, 전부 상류와의 대조이거나 거절의 단언 |
| 문서 정정 | `WHEEL.md` §5 와 `DISTRIBUTED.md` §7·§3.4 에 앞을 가리키는 배너 |
| 삭제 | 없음 |

### 5.1 실패할 수 있는지 확인했다

`CLAUDE.md` §5.5. 새 테스트를 **일부러 깨서** 확인했습니다 — 이 세션에서 넷, §6.1 작업에서
여섯, 전부 git 이 아니라 **직접 되돌리는 국소 편집**으로.

| 무엇을 깼나 | 무엇이 빨개졌나 |
|---|---|
| `max`/`min` 의 NaN 분기 무력화 | `compare.py`: `aten.max.default`, `aten.min.default` 2건 |
| `ceil` 을 항등으로 | `test_repr_matches...` 17개 중 2개 (`tensor(2.)` vs `tensor(1.5000)`) |
| `_efficientzerotensor` 를 만들어지게 | `test_the_alternative_representations_have_no_constructors` |
| `silu` 에 배율 1.001 | 5e-7 경계에서 `test_a_real_transformers_llama_forward...` (§4.2) |
| 세터가 아무것도 저장하지 않게 | `get_default_dtype` 단언 |
| `full` 을 다시 리터럴 float32 로 | `{'full': 'torch.float32'}` 를 이름 대고 보고 |
| 부동소수 게이트 제거 · 메타 `div`/`pow`/`reciprocal` 규칙 뒤집기 | 각각의 해당 테스트 |

두 번째가 이 형태의 테스트가 왜 필요한지 보여줍니다 — `ceil` 이 항등이 되면 값은 여전히
맞지만 정수 모드 판정이 뒤집혀 **서식이** 틀립니다. `.tolist()` 로 판정하는 테스트는 통과합니다.

### 5.2 이 회차에 저지른 사고 하나 — 기록해 둡니다

커널 검증을 되돌리면서 `git checkout -- rust/torch_c/src/aten.rs` 를 썼습니다.
**커밋되지 않은 그 파일의 작업 전체가 사라졌습니다** — CLAUDE.md 가
"에이전트 작업을 되돌려 볼 때는 stash 를 쓴다" 로 정확히 경고한 그 실수입니다. 대화에 남은
편집 기록으로 재구성해 복구했고, 이후의 탬퍼 검증은 전부 **직접 되돌리는 국소 편집**으로
했습니다. 그 되돌리기도 한 번 틀렸습니다 — `ceil_default` 안에 같은 문자열이 두 번 있어
엉뚱한 줄을 되돌렸고, `grep -c` 가 1을 세어 주는 바람에 **복구된 것처럼 보였습니다.**
테스트가 잡았습니다.

---

## 6. `from_pretrained`

> 이 절은 두 층입니다. 아래 본문은 **처음 만난 벽**과 그때의 판단이고, §6.1 이 그것을 연 기록,
> §6.2 가 **지금 남아 있는 벽**입니다. 순서를 남겨 둔 것은 §6.1 이 §3.1·§3.2 와 같은 종류의
> 결함(상수를 직접 적어 둔 자리 둘)을 드러냈고, 그 발견이 "왜 전역이어야 하는가" 라는 이
> 문단의 질문에서 나왔기 때문입니다.

```
transformers/modeling_utils.py:4304   with ContextManagers(model_init_context):
NotImplementedError: not implemented in torch._C shim: torch._C._set_default_dtype
```

상류가 저장한 로컬 체크포인트(`save_pretrained`, `config.json` + `model.safetensors`)로
시도했습니다 — **네트워크는 쓰지 않았습니다.** `from_config` 로 가는 길과 달리
`from_pretrained` 는 모델을 만들기 전에 기본 dtype 을 바꾸는 컨텍스트 매니저를 엽니다.

`docs/distributed/DISTRIBUTED.md` §3.4 가 이 이름을 왜 거절하는지 이미 적어 두었습니다:
`torch.get_default_dtype()` 이 읽는 값은 `aten.rs` 의 `DEFAULT_FLOAT` 상수이고, 세터는
그 Rust 상수에 닿아야 하는데 닿을 수 없습니다. **이것을 진짜 전역으로 만드는 것이 다음 작업
항목**이고, 그 전역은 반드시 추론 규칙이 실제로 읽는 것이어야 합니다 —
`set_default_dtype(float64)` 를 받아 놓고 `torch.ones(3).dtype` 이 `float32` 로 남는 세터는
지금의 거절보다 나쁩니다.

읽는 자리는 10곳입니다 (`lib.rs:227`, `aten.rs` 9곳).

### 6.1 이 벽은 열렸습니다 (2026-08-28, 같은 날 뒤이어)

`_set_default_dtype` 을 진짜 전역으로 만들었고(`docs/distributed/DISTRIBUTED.md` §3.4 의 추가 문단),
**읽는 자리는 10곳이 아니라 12곳이었습니다** — 위 문단이 센 것은 `DEFAULT_FLOAT` 라는
*식별자*를 쓰는 곳이고, `full`(`aten.rs`)과 `torch.finfo()`(`info.rs`)는 같은 규칙을
`TorchDType::Float32` 라고 **직접 적고** 있었습니다. 식별자로 세면 안 잡히는 두 곳이고,
그대로 뒀다면 그 둘만 float32 에 남았을 것입니다.

그 뒤 `from_pretrained` 는 벽 넷을 더 냈고 전부 **meta 커널**이었습니다. `init_empty_weights`
아래에서 `LlamaRotaryEmbedding.__init__` 이 값을 계산하기 때문입니다
(`transformers/models/llama/modeling_llama.py:108`):

```python
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
#                              div.Scalar ^      ^ pow.Scalar
#           reciprocal + mul.Scalar ^
```

`div.Scalar` · `pow.Scalar` · `reciprocal.default` · `mul.Scalar` 넷을 추가하니 **모델
구성이 전부 통과**합니다. 넷 다 dtype 규칙을 조밀 커널과 **공유**합니다(`arith_tag`,
`pow_result_tag`) — meta 커널이 조밀 커널과 다른 dtype 을 약속하면, 그 shape 과 dtype 으로
할당한 뒤에 계산이 거절되기 때문입니다.

### 6.2 그다음 벽 — `torch.UntypedStorage.from_file`

```
transformers/modeling_utils.py:4470   file_pointer = safe_open(file, framework="pt", ...)
torch/storage.py:212                  def from_file(cls, filename, shared, nbytes): raise NotImplementedError
```

**경로 둘이 같은 이름에서 만납니다.** safetensors 0.8.0 의 `safe_open` 이 그것을 부르고,
`.bin` 체크포인트도 `torch/serialization.py:1594` 에서 같은 것을 부릅니다 —
`transformers` 가 zipfile 체크포인트에 `mmap=True` 를 **박아** 넣기 때문이고
(`modeling_utils.py:363`), `from_pretrained(..., disable_mmap=True)` 로도 우회되지 않습니다.

`_StorageBase.from_file` 은 **메시지 없는** `raise NotImplementedError` 입니다. 상류 트리의
추상 스텁이고 상류는 `_C.StorageBase` 가 MRO 앞에서 가리므로, 여기서는 이름을 대지 않는
거절이 그대로 새어 나옵니다 (DESIGN.md §6 위반이지만 **벤더링 트리는 고치지 않습니다**).

그 이름 뒤에 무엇이 있는지 파이썬 대역으로 재 봤습니다 — 구현이 아니라 측정입니다:

| # | 이름 | 부르는 곳 |
|---|---|---|
| 1 | `UntypedStorage.from_file(filename, shared, nbytes)` | `safe_open`, `serialization.py:1594` |
| 2 | `UntypedStorage.__getitem__(slice)` | safetensors 슬라이스, `serialization.py:2115` |
| 3 | `torch.asarray` | safetensors 의 pt 텐서 구성 |

1 번만 파이썬으로 대신 채워 주면 `.bin` 경로는 **21개 텐서를 끝까지 읽습니다**
(`torch.load(..., mmap=False)` 는 지금도 통과합니다 — 즉 나머지 적재 경로는 이미 서 있고
막고 있는 것은 mmap 진입점 하나입니다). safetensors 경로는 1·2 를 채운 뒤 3 에서 멈춥니다.

상류 `from_file` 을 재 둔 것 (torch 2.13.0):

```
from_file(p, False, n)            n 바이트 UntypedStorage, .filename 은 None
from_file(p, False, 16)           16 바이트
from_file(p, False, 0)            0 바이트
nbytes > 파일 크기                RuntimeError: file <p> size <N> is smaller than
                                  the required mapping size <M>
없는 파일                          RuntimeError: unable to open file <p> in read-only
                                  mode: No such file or directory (2)
s[0:8]                            8 바이트 UntypedStorage
s[0]                              int
```

`shared=True` 는 이 shim 에서 **재현할 수 없습니다** — 여기 저장소는 `Vec<u8>` 사본이고
MAP_SHARED 는 쓰기가 파일로 돌아가야 합니다. 이름을 대고 거절할 자리이지 조용히 사본을 줄
자리가 아닙니다 (`storage.rs` 모듈 독스트링의 `filled` 불변식과 같은 종류의 문제).

---

## 7. 미확인 — 숨기지 않는 것

| # | 항목 | 상태 |
|---|---|---|
| 1 | `from_pretrained` 의 **가중치 적재** | **미통과.** `UntypedStorage.from_file` — §6.2. 모델 *생성*은 통과합니다 |
| 1b | `from_file(shared=True)` | **재현 불가.** 이 shim 의 저장소는 `Vec<u8>` 복사본이고 `MAP_SHARED` 는 쓰기가 파일에 닿기를 요구합니다. 이름을 대고 거절할 자리이지 조용히 복사본을 건넬 자리가 아닙니다 (§6.2) |
| 2 | 실제 사전학습 체크포인트(허브에서 받은 것) | **미시도.** 1번 뒤에 있고, 네트워크가 필요합니다 |
| 3 | `generate()` / 다단계 디코딩 — **진짜 모델로** | **미시도.** 전사한 디코더로는 이미 통과합니다(`test_do_sample_...`) |
| 4 | 순전파 대조의 **모델 크기** | 16폭·32어휘·2층·4토큰 하나뿐. §4.2 가 이 크기의 민감도 한계를 잽니다 |
| 5 | Llama 외 아키텍처의 **진짜 `transformers` 순전파** | **미시도.** `docs/kernels/OPS4.md` 의 20개는 전사 기준입니다 |
| 6 | `repr` 의 미대조 분기 | 희소 · 양자화 · nested · functorch 래퍼 · `_to_functional_tensor` 접두사. 만들 수 없으므로 그 서식 코드는 이 빌드에서 **도달 불가**이고, 대조 케이스도 없습니다 |
| 7 | `layout` 이 `strided` 외의 값을 답하는 경우 | **없음.** `_layout_name` 이 문자열을 돌려주고 `bootstrap.py` 가 모르는 문자열에 이름을 대고 거절합니다 |
| 8 | `bool + bool`(논리합) · `bool - bool` | **거절.** §2.2 는 `mul.Tensor` 만 열었습니다 |
| 9 | `cat` 의 dtype 승격 | **거절.** §3.2. legacy-empty 규칙은 모양에만 적용됩니다 |
| 10 | `min.dim` · `min.other` | **커널 없음.** `methods.json` 이 해석하고 `aten.rs` 가 이름을 대고 거절합니다 — `max` 와 대칭이 아닙니다 |
| 11 | autocast 를 **켰을 때** | **거절.** §4.3 |
| 12 | 안드로이드 · iOS | **미측정.** 호스트(darwin/arm64)에서만 돌렸습니다 |
| 13 | `is_conj` | **거절인 채.** `repr` 이 도달하지 않았습니다(복소 저장이 없어 `conj` 를 만들 수 없습니다). §2.1 (b) 의 다른 다섯과 달리 구현하지 않았습니다 |
| 14 | §4.2 의 0.01% 오차 | **못 잡습니다.** 더 큰 모델이 필요합니다 |

---

## 8. 재현

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cd /Volumes/macMini/worktrees/bw-e2e
bash vendor/vendor_torch.sh
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-e2e
PY=/Volumes/macMini/caches/spike-venv/bin/python
bash vendor/install_shim.sh
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib

PYTHON=$PY sh rust/torch_c/pytests/run.sh          # 142
$PY tools/golden/compare.py                        # 2486/2486 ops=116
$PY rust/torch_c/pytests/verify_schemas.py         # 270/270

# 이 문서의 두 판정
PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY -c \
  "import torch; print(torch.ops.aten.mm.default(torch.ones(3,4), torch.ones(4,2)))"
PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY -c \
"
import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
m = AutoModelForCausalLM.from_config(LlamaConfig(
    vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
    tie_word_embeddings=False))
m.eval()
print(m(torch.tensor([[3,7,1,19]])).logits.shape)
"
```

**`compare.py` 와 `verify_schemas.py` 는 벤더 트리를 `PYTHONPATH` 에 넣지 않고 돌립니다** —
넣으면 상류 torch 를 가려 기준선이 사라집니다. `compare.py` 는 `TORCH_C_ARTEFACT` 가 필요합니다;
없으면 조용히 낡은 산출물을 잽니다(이번에 한 번 그렇게 재고 "ops 가 안 늘었다" 고 잘못 읽었습니다).

§1.1 의 상류 디스패치 트레이스는 `_tensor_str._str` 을 `_disable_current_modes()` 를 빼고
다시 정의한 뒤 `TorchDispatchMode` 를 안쪽에 걸어 뜹니다. 밖에 걸면 빈 목록이 나옵니다.

---

## 9. 세 목표에 대한 정직한 채점

작업 지시는 셋이었습니다.

| # | 목표 | 결과 |
|---|---|---|
| 1 | **`print(tensor)` 가 되게 하라** | **달성.** 되는 것에 그치지 않고 상류 2.13.0 과 17/17 문자 단위 일치 (§2) |
| 2 | **순전파를 열어라 — 로짓이 나올 때까지** | **달성.** `from_config` 로 만든 진짜 `LlamaForCausalLM` 이 로짓을 내고, 상류와 2.24e-08 안에서 일치하며 argmax 가 같습니다 (§4) |
| 3 | **`from_pretrained` 로 실제 체크포인트를 읽어 돌려라 — 되는 데까지** | **부분.** 모델 *생성*은 통과, *가중치 적재*는 미통과 (§6). 네트워크는 쓰지 않고 상류가 `save_pretrained` 로 쓴 로컬 체크포인트를 썼습니다 |

3번을 "부분" 이라고 적는 이유를 분명히 합니다. `from_pretrained(...)` 은 여전히 예외로
끝납니다. 그 안에서 지나온 것은 config 파싱 · 모델 클래스 결정 · 기본 dtype 컨텍스트 ·
`init_empty_weights` 아래의 전체 모듈 트리 구성이고, 멈추는 곳은 체크포인트 파일을 여는
첫 줄입니다. **"`from_pretrained` 가 된다" 고 쓸 수 있는 상태가 아닙니다.**

§7 의 표가 남은 것 전부입니다. 그중 이 회차가 **새로 만든** 미확인은 1b(`shared=True`)와
14(§4.2 의 민감도 한계) 둘뿐이고, 나머지는 이전부터 있던 것을 이름 붙여 옮겨 적은 것입니다.
