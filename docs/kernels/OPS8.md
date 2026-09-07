# 8 개를 메웠고, greedy Llama 순전파가 돌았습니다

`docs/kernels/GAP.md` §3 이 "greedy 2 층 Llama 가 부르는 op 중 8 개가 없어서 순전파가 지금 멈춘다" 고
적은 그 8 개를 구현하고 골든 케이스를 붙인 기록입니다.

**결론 먼저.** 8 개 전부 구현했습니다. 그리고 상류 torch 위에서 `TorchDispatchMode` 로 다시 재 보니
**greedy 2 층 Llama 의 `generate()` 가 부르는 aten op 46 개 중 미구현이 0 개**입니다 (이 변경 전에는
7 개였습니다). 순전파만 보면 25 개 중 0 개입니다. 벤더링 트리 위에서 aten 층으로 손으로 쓴 같은
모델의 순전파를 실제로 돌렸고, 상류와 로짓이 **max|diff| = 2.3e-09** 로 일치했습니다.

**하지만 `transformers` 의 `LlamaForCausalLM` 은 아직 못 돕니다.** 남은 벽은 aten 층이 아니라 그
위층입니다 — `torch._C._nn.linear`, `torch._C._nn.scaled_dot_product_attention`, 그리고
`torch.distributed` 의 임포트 벽. 셋 다 `bootstrap.py` 와 벤더링 트리 쪽 문제이고 이 작업의 파일
범위 밖이라 손대지 않았습니다. §5 에 이름과 위치를 정확히 적었습니다.

---

## 0. 도달한 숫자 — 전부 종료 코드와 함께

| 검증 | 이전 | 이후 | 종료 코드 |
|---|---|---|---|
| 골든 (`tools/golden/compare.py`) | 1043/1043, ops covered=**62** | **1212/1212**, ops covered=**70** | **0** |
| 골든 실패 / pending | 0 / 0 | **0 / 0** | — |
| 스키마 (`pytests/verify_schemas.py`) | 127/127 | **127/127** | **0** |
| 스모크 (`pytests/run.sh`) | 60 ok | **60 ok** | **0** |
| `--inject-fault value` | — | 첫 `match` 케이스에서 잡힘 | **1** |
| `--inject-fault shape` | — | 잡힘 | **1** |
| `--inject-fault dtype` | — | 잡힘 | **1** |
| 호스트 빌드 | — | — | **0** |
| Android (`cargo ndk -t arm64-v8a`) | — | — | **0** |
| iOS (`aarch64-apple-ios`) | — | — | **0** |

**스키마 수는 바뀌지 않았습니다 (127/127).** `verify_schemas.py` 가 검사하는 것은
`src/overloads.json` 과 `src/methods.json` — `torch.<op>` 와 `TensorBase.<method>` 의 오버로드
해석표입니다. 이번에 추가한 8 개는 `torch.ops.aten.<op>.<overload>` 로 도달하는 경로만 쓰므로 그
두 표에 항목이 필요 없고, 따라서 숫자가 움직일 이유가 없습니다. (§5-2 가 이 사실의 다른 면 —
`torch.bmm(...)` 같은 파이썬 철자는 아직 해석되지 않는다는 것 — 을 적습니다.)

새 골든 케이스는 **169 개**입니다.

| op | 케이스 수 |
|---|---|
| `aten.t.default` | 29 |
| `aten._unsafe_view.default` | 28 |
| `aten._scaled_dot_product_flash_attention_for_cpu.default` | 22 |
| `aten.rsub.Scalar` | 22 |
| `aten.alias.default` | 21 |
| `aten.neg.default` | 18 |
| `aten.bmm.default` | 15 |
| `aten.silu.default` | 14 |

---

## 1. `bmm` 가설 — 절반만 맞았습니다

조율 세션의 가설: "`bmm` 은 커널이 없는 게 아니라 라우팅 누락이다. `matmul_default` 가 이미
candle 의 `broadcast_matmul` 로 배치 행렬곱을 하니, 디스패치 한 줄이면 끝난다."

**커널 쪽은 맞았습니다. 라우팅 한 줄은 틀렸습니다.**

`matmul_default` 가 배치를 처리하는 방식은 **브로드캐스트**이고, `bmm` 은 브로드캐스트하지
않습니다. 상류에서 실측했습니다:

```
bmm((1,3,4), (2,4,5))  ->  RuntimeError: Expected size for first two dimensions of
                           batch2 tensor to be: [1, 4] but got: [2, 4].
```

`broadcast_matmul` 은 이 입력을 **계산합니다**. 그러니 `"aten.bmm.default" => matmul_default(...)`
로 한 줄 배선했다면 **상류가 거부하는 자리에서 값을 만들어내는** shim 이 됐을 것입니다 —
DESIGN.md §5 가 candle 위에 얹을 때의 주된 위험으로 지목한, 조용한 발산의 정확한 형태입니다.

그래서 `bmm_default` 는 별도 함수로 뒀습니다. 곱셈은 같고, **계약이 다릅니다**: 양쪽 모두 정확히
3-D, 배치 크기 일치, 브로드캐스트 없음. 상류의 거부 문구 셋(`batch1 must be a 3D tensor`,
`batch2 must be a 3D tensor`, 위의 배치 불일치 메시지)을 그대로 재현했습니다.

`bmm(1,3,4)×(2,4,5)` 를 양쪽이 거부하는지 확인하는 골든 케이스를 넣어 뒀습니다. 나중에 누군가
"어차피 같은 커널인데" 하고 합치면 그 케이스가 걸립니다.

부수적으로 확인된 것: **`bmm` 은 `mm` 의 dtype 구멍을 그대로 물려받습니다.** candle 의 matmul 은
`int64`·`int32`·`int16`·`uint8`·`bfloat16` 커널이 없어서 상류가 계산하는 자리에서 거부합니다
(`_MM_C_ERROR_DTYPES` 와 같은 목록). `expect="c_error"` 로 각각 케이스를 뒀습니다 — 한쪽에서
구멍이 메워지면 다른 쪽에서도 메워졌는지 따로 보이도록 op 별로 나눠 뒀습니다.

---

## 2. 구현한 것 / 때운 것 / 못 한 것

### 구현한 것 — 8/8

| op | 어떻게 | 상류와 다른 점 |
|---|---|---|
| `aten.bmm.default` | 전용 커널. rank·배치 검사 후 candle `matmul` | 없음 (dtype 구멍은 `mm` 과 동일, §1) |
| `aten._unsafe_view.default` | `reshape_like` 재사용, 키만 분리 | 없음 — autograd 가 없으므로 `view` 와 값이 같다 |
| `aten.alias.default` | 값 그대로 재포장 | **저장소 공유를 재현하지 않음** (§4) |
| `aten.neg.default` | 부동소수는 candle `neg`, 정수는 `i64` 왕복 | 없음 |
| `aten.rsub.Scalar` | `arith_tag`/`apply_arith` 재사용, 피연산자 반전 | 없음 |
| `aten.silu.default` | `f16`/`bf16` 은 `f32` 로 올려 계산 후 축소 | 없음 |
| `aten.t.default` | rank 로 분기 (0/1-D 그대로, 2-D 전치, 3-D+ 거부) | 없음 |
| `aten._scaled_dot_product_flash_attention_for_cpu.default` | 융합 커널을 직접 작성 (§3) | 없음 |

전부 상류와 dtype·shape·값까지 대조했고, 거부하는 자리까지 맞췄습니다.

**측정해서 알게 된 것 — 추론했으면 틀렸을 것들:**

- **`neg` 를 `unary_float` 로 재사용하면 틀립니다.** `cos`/`sin`/`reciprocal` 은 정수 입력을 기본
  부동소수로 승격하지만 `neg` 는 입력 dtype 을 유지합니다 (`int64` in, `int64` out).
- **candle 의 `neg` 를 정수에 부르면 에러가 아니라 `todo!()` 로 패닉합니다.** `unary_op!` 매크로의
  정수 갈래가 전부 `todo!()` 입니다 — 인터프리터가 통째로 죽습니다. 그래서 정수는 `bitwise_not`
  이 이미 쓰던 `i64` 왕복 경로로 돌렸습니다. `uint8` 에서 `to_dtype` 이 잘라내는 것이 상류의 답과
  같습니다 (`neg(uint8 [1,2,0]) == [255,254,0]`, 실측).
- **`neg` 는 `uint32` 에서 상류가 거부합니다** (`"neg_cpu" not implemented for 'UInt32'`). `uint8`
  은 되는데 그보다 넓은 부호 없는 타입은 안 됩니다. 그대로 따라 거부합니다.
- **`silu` 는 정수를 승격하지 않고 거부합니다** (`"silu_cpu" not implemented for 'Long'`). 이것이
  `silu` 를 `Unary` enum 에 넣지 않은 이유입니다.
- **`t()` 는 `transpose(-2,-1)` 이 아닙니다.** 0-D 와 1-D 는 **그대로** 돌아오고, 3-D 이상은
  에러입니다. `transpose(-2,-1)` 로 읽었다면 상류가 거부하는 3-D 입력에서 값을 만들었을 것입니다.
- **`rsub` 의 `alpha` 는 스칼라가 아니라 `self` 를 곱합니다** (`other - alpha*self`). 부호만
  뒤집은 `sub.Scalar` 가 아닙니다.

### 때운 것 — 없음

반쯤 만든 것, 값만 맞추고 dtype 을 흘린 것, `expect` 를 느슨하게 열어 둔 것은 없습니다.
새 케이스 169 개 중 `c_error` 는 `bmm` 의 candle dtype 구멍 5 개뿐이고, 그것은 이 변경이 만든
구멍이 아니라 `mm` 이 이미 문서화해 둔 같은 구멍을 `bmm` 쪽에서도 추적하는 것입니다.

### 못 한 것

- **`transformers` 의 `LlamaForCausalLM` 을 우리 shim 위에서 못 돌렸습니다.** 막힌 곳이 aten 층이
  아니어서 이 작업으로는 못 넘습니다 — §5.
- **`aten.alias.default` 이 실제로 필요한지 재현하지 못했습니다.** 구현하고 골든까지 붙였지만,
  아래 §4 에서 재 본 어떤 구성에서도 호출되지 않았습니다.

---

## 3. `_scaled_dot_product_flash_attention_for_cpu` — 이름이 말하지 않는 것

나머지 7 개와 성격이 다릅니다. 텐서 **두 개**를 돌려주고, 관측된 계약이 이름에서 짐작되는 것과
다릅니다. 전부 상류 torch 2.13.0 으로 먼저 재고 구현했습니다.

| 항목 | 실측 결과 |
|---|---|
| 반환 | `(output, logsumexp)` **평범한 튜플** (namedtuple 아님) |
| `logsumexp` 모양 | `(B, H, T)` — 마스크·스케일이 **적용된 뒤**의 점수에 대한 logsumexp |
| `is_causal` 정렬 | **좌상단 정렬**. 행 `t` 는 키 `0..=t` 를 본다 — 키가 쿼리보다 길어도 그렇다 |
| `is_causal` + `attn_mask` | **둘 다 적용된다.** `F.scaled_dot_product_attention` 은 둘 다 주면 거부하지만 이 aten op 은 더한다 |
| `float16`/`bfloat16` 입력 | 출력은 입력 dtype, **`logsumexp` 는 `float32`** |
| `dropout_p > 0` | 상류가 거부 (`Currently do not support dropout > 0`) |
| 4-D 아닌 입력 | 상류가 거부 (`Accept only 4 dims inputs shape of {B, H, T, K}`) |
| `attn_mask` dtype | 쿼리와 같아야 함 (bool 마스크 거부) |

**`float16` 입력에서 `logsumexp` 만 `float32` 로 나오는 것이 가장 중요한 단서입니다.** 그것이
"내부 누산이 float 에서 일어난다" 를 상류가 밖으로 드러내는 유일한 자리이고, 그래서 이 구현은
`f16`/`bf16` 입력을 본문 전체에서 `f32` 로 올려 계산하고 출력만 다시 좁힙니다. 골든의
`_sdpa_pair_check` 는 두 반쪽의 dtype 을 **따로** 검사합니다 — 하나로 묶어 비교하면 전부
`float16` 으로 계산한 shim 이 통과해 버립니다.

softmax 는 손으로 썼습니다. `candle-core` 에는 softmax 가 없고(그것은 `candle-nn` 에 있으며
DESIGN.md §4 는 그것을 끌어오지 않습니다), 행 최댓값을 먼저 빼는 것은 최적화가 아니라
**정확성**입니다 — 빼지 않으면 마스크된 자리의 `exp(-inf)` 와 큰 로짓의 `exp(big)` 이 똑같이
NaN 이 됩니다. `-inf` 열이 들어간 마스크 케이스가 그것을 지킵니다.

---

## 4. Llama 순전파가 어디까지 갔는가

### 4.1 aten 층 — 끝까지 갔고, 로짓이 상류와 일치합니다

`transformers` 가 못 임포트되므로(§5-3), 같은 산술을 `torch.ops.aten.*` 만으로 손으로 써서
양쪽에서 돌렸습니다. 2 층, hidden 64, head 2, intermediate 128, vocab 100, seq 4 — `GAP.md` 와
`FROM_CONFIG.md` 가 쓴 것과 같은 크기입니다. RMSNorm · RoPE(`neg` 로 `rotate_half`) ·
SwiGLU(`silu`) · sdpa 어텐션 · `nn.Linear` 의 `x @ w.t()`(`t`) · 전치 뒤의
`reshape`(`_unsafe_view`) 를 전부 포함합니다.

```
상류 torch 2.13.0          greedy tokens [14, 95, 22, 17]
벤더링 트리 + 우리 shim     greedy tokens [14, 95, 22, 17]

n=400  max|diff| = 2.328e-09   mean|diff| = 2.697e-10   argmax 일치
```

`float32` 의 골든 허용오차가 `atol=rtol=1e-5` 이므로, 2 층 · 12 회 행렬곱을 누적한 뒤의
2.3e-09 는 누산 순서 차이 이상의 것이 아닙니다.

### 4.2 상류에서 다시 잰 op 커버리지 — greedy 는 0 개 부족

`TorchDispatchMode` 로 상류 torch 위에서 진짜 `transformers` Llama 를 돌리고, 그 op 집합을
`_aten_implemented() ∪ _aten_implemented_awaiting_golden()` 과 대조했습니다.

| 시나리오 | 부르는 op | 이 변경 **전** 미구현 | 이 변경 **후** 미구현 |
|---|---|---|---|
| 순전파만 (`model(ids)`) | 25 | — | **0** |
| greedy `generate(max_new_tokens=4)`, MHA | 46 | 7 | **0** |
| greedy `generate`, GQA (4 heads, 2 kv) | 46 | 7 | **0** |
| greedy `generate`, **eager** 어텐션 | 49 | — | **4** |
| `do_sample=True` (top_k=50, top_p=0.95) | 55 | — | **7** |

**greedy 경로의 aten 격차는 닫혔습니다.**

### 4.3 `alias.default` — 구현했지만 필요하다는 것을 재현하지 못했습니다

위 세 greedy 구성(MHA · GQA · eager) 어디에서도 `aten.alias.default` 이 호출되지 않았습니다.
`GAP.md` §3 의 8 개 목록은 `CORE_ATEN.md` §2 의 원본 48 개에서 왔는데, **`GAP.md` §0 자신이 그
재현에서 47/48 만 일치했고 빠진 하나가 정확히 `alias.default` 이라고 적어 두었습니다.** 이번
측정이 그 각주를 확인해 줍니다 — transformers 5.15.1 + torch 2.13.0 조합의 Llama 는 이 op 을
부르지 않습니다.

구현은 해 뒀습니다(싸고, 목록에 있었고, 골든 대조도 걸려 있습니다). 다만 **"이 순전파가 필요로
한다" 는 근거는 없습니다.** 원본 48 개 목록이 어떤 버전에서 나왔는지는 모릅니다.

`alias` 는 이 shim 에서 **저장소를 공유하지 않습니다.** `detach` 가 그렇지 않은 것과 같은
이유이고(`PyTensorBase::replace_with` 는 래퍼의 텐서를 갈아끼울 뿐 저장소에 쓰지 않습니다),
같은 결과를 낳습니다 — 둘 중 하나에 in-place 로 쓰면 다른 하나가 못 봅니다. 읽기만 하는
경로에서는 물지 않고, KV 캐시 쓰기에서는 뭅니다. 감추지 않고 적어 둡니다.

---

## 5. 다음 벽 — 셋, 전부 aten 층 밖

### 5-1. `torch._C._nn.linear` 과 `torch._C._nn.scaled_dot_product_attention`

`nn.Linear(x)` 와 `F.scaled_dot_product_attention(...)` 이 각각 여기로 내려갑니다.

```
NotImplementedError: not implemented in torch._C shim: torch._C._nn.linear
NotImplementedError: not implemented in torch._C shim: torch._C._nn.scaled_dot_product_attention
```

**커널은 이제 전부 있습니다** — `linear` 은 `t` + `matmul`(또는 `addmm`)이고, sdpa 는 §3 의 융합
커널입니다. 없는 것은 그 파이썬 이름에서 aten 키로 내려가는 배선이고, 그것은
`rust/torch_c/src/bootstrap.py` 에 있습니다. 이 작업의 파일 범위 밖이라 손대지 않았습니다.

같은 상태인 `_C._nn` 이름을 몇 개 더 확인했습니다: `gelu`, `silu`, `softmax`, `layer_norm`,
`pad`, `_parse_to` — 전부 같은 문구로 거부합니다. `_C._nn` 표면 전체가 아직 비어 있는 것으로
보입니다.

> **Correction (문서 감사, 2026-09):** 배선이 갈렸습니다. `_C._nn.linear` 와
> `_C._nn.scaled_dot_product_attention` (이 절이 위에서 예로 든 둘) 은 오늘 성공합니다.
> `_C._nn.gelu`/`_C._nn.silu`/`_C._nn.pad` 도 오늘 성공합니다. 다만 `_C._nn.softmax` 와
> `_C._nn.layer_norm` 은 **여전히 거부합니다** — `torch.layer_norm`/`F.layer_norm` 은 이제
> 되지만(`docs/models/GPT2.md` 감사가 이 라운드에서 이미 확인) `bootstrap.py:5863` 의 별도 파이썬
> 컴포지트를 경유하는 것이지 `_C._nn.layer_norm` 자체가 채워진 것이 아닙니다. **"`_C._nn`
> 표면 전체가 비어 있다"는 더 이상 맞지 않지만, "완전히 채워졌다"도 아닙니다** — 이름마다
> 다릅니다.
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py linear present -->

### 5-2. `torch.bmm(...)` 같은 파이썬 철자는 아직 해석되지 않습니다

`torch.ops.aten.bmm.default(...)` 은 됩니다. `torch.bmm(...)` 은 `src/overloads.json` 에 항목이
없어 오버로드 해석이 거부합니다. `TensorBase.t()`/`.neg()`/`.bmm()` 도 `src/methods.json` 기준으로
같습니다. 두 표는 `verify_schemas.py` 가 검사하는 대상이고 이 작업의 범위 밖이라 건드리지
않았습니다 — 그래서 §0 의 127/127 이 그대로입니다. 항목을 넣으면 그 숫자가 올라갑니다.

> **Correction (문서 감사, 2026-09):** 항목이 들어갔습니다. 실측: `torch.bmm(...)`,
> `t.t()`, `t.neg()`, `t.bmm(...)` 전부 오늘 성공합니다. `overloads.json`/`methods.json` 에
> `bmm`/`t`/`neg` 키가 있습니다 (지금 스키마 총계는 §0 의 127/127 이 아닙니다 —
> `docs/verification/AUDIT.md` 의 이 라운드 기준선은 4475/4475).
> <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json bmm present -->
> <!-- DOCWATCH: json-key rust/torch_c/src/methods.json bmm present -->

### 5-3. `torch.distributed` — 임포트 벽, aten 과 무관

`import transformers` 자체가 실패합니다. 사슬은 이렇습니다:

```
transformers.generation.utils
  -> torch._dynamo
    -> torch._dynamo/variables/functions.py:102
       from torch.distributed.fsdp._fully_shard import _fsdp_param_group
      -> torch/distributed/fsdp/_flat_param.py:31
        -> torch/testing/_internal/distributed/fake_pg.py:7
           class FakeStore(dist.Store):
           AttributeError: module 'torch.distributed' has no attribute 'Store'
```

`torch.distributed.is_available()` 는 `False` 인데, **`_dynamo` 는 그 값을 보지 않고 무조건
`fsdp` 를 임포트합니다.** VENDOR.md 벽 11 의 "없는 것이 끄는 방법이다" 가 이 경로에서는
성립하지 않는다는 뜻입니다.

`Store` 를 임시로 채워 넣고 더 가 봤더니 다음 벽이 나왔습니다:

```
torch/testing/_internal/distributed/fake_pg.py:30
  AttributeError: type object 'Backend' has no attribute 'register_backend'
```

즉 이름 하나가 아니라 **`torch.distributed` 의 실질적 표면**이 요구됩니다. 이것도 벤더링 트리와
`bootstrap.py` 쪽 작업입니다.

### 5-4. eager 어텐션과 `do_sample=True` 가 추가로 요구하는 것

측정값 그대로입니다.

```
eager 어텐션 (+4):   aten._softmax.default   aten.le.Tensor
                      aten.scalar_tensor.default   aten.where.self

do_sample=True (+7): aten._softmax.default   aten.le.Scalar
                      aten.multinomial.default   aten.scatter.src
                      aten.sort.default   aten.squeeze.dim   aten.topk.default
```

`do_sample` 쪽 7 개는 `GAP.md` §4 가 예측한 것과 정확히 같습니다. eager 쪽 4 개는 `GAP.md` 가
재지 않은 것이고, `_softmax` 만 겹칩니다 — 즉 **`_softmax.default` 하나가 두 경로 모두의
전제조건**입니다.

---

## 6. 모르는 것 · 명시적 미확인

- **`aten.alias.default` 이 어떤 구성에서 불리는지 모릅니다.** §4.3.
- **기기에서는 확인하지 않았습니다.** Android · iOS 는 **링크만** 확인했습니다(종료 코드 0).
  기기에서 이 8 개가 도는지는 미확인이며, `docs/design/TORCH_C.md` §6 의 같은 항목과 상태가 같습니다.
- **표본은 여전히 Llama 2 층 하나입니다.** GPT-2 는 재지 않았습니다 — `GAP.md` §5 가 이미
  `addmm`·`native_layer_norm`·`split.Tensor`·`tanh` 4 개가 더 필요하다고 적었고, 이번 작업은
  그 목록을 건드리지 않았습니다.
- **`_scaled_dot_product_flash_attention_for_cpu` 에서 한 행이 통째로 마스크된 경우**(`is_causal`
  인데 키가 쿼리보다 **짧은** 경우 등)의 동작은 양쪽 다 재지 않았습니다. 행 최댓값이 `-inf` 가
  되어 `NaN` 이 나올 것으로 보이지만 상류와 대조하지 않았습니다.
- **`bmm` 의 candle dtype 구멍을 메우지 않았습니다.** `mm` 과 같은 구멍이고, 메우려면
  candle 밖에서 정수 행렬곱을 쓰거나 candle 을 고쳐야 합니다. 범위 밖으로 뒀습니다.
- **`torch.no_grad()` 아래에서 돌렸지만** 그 컨텍스트가 이 shim 에서 실제로 무엇을 끄는지는
  확인하지 않았습니다 — autograd 자체가 없으므로 무해할 것으로 보이나 측정하지 않았습니다.

---

## 7. 재현 방법

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target
PY=/Volumes/macMini/caches/spike-venv/bin/python
DIST=/Volumes/macMini/caches/target-python
cd rust/torch_c            # cd 필수 — .cargo/config.toml 은 cwd 기준

# 호스트 빌드 + 스모크
PYTHON=$PY ./pytests/run.sh > /tmp/smoke.log 2>&1; echo "EXIT=$?"

# 골든 · 스키마 — PYTHONPATH=vendor 를 **붙이지 않는다**.
# 붙이면 벤더링 트리가 상류 torch 를 가려서 비교의 양쪽이 같은 것이 되고 가짜 실패가 난다.
cd ../..
$PY tools/golden/compare.py > /tmp/golden.log 2>&1; echo "EXIT=$?"
$PY rust/torch_c/pytests/verify_schemas.py > /tmp/schemas.log 2>&1; echo "EXIT=$?"
for m in value shape dtype; do
  $PY tools/golden/compare.py --inject-fault $m > /tmp/fault-$m.log 2>&1; echo "$m EXIT=$?"
done

# Llama 순전파 — 벤더링 트리에 새 산출물을 먼저 넣어야 한다.
# 이것을 빼먹으면 낡은 _C.abi3.so 를 재게 되고, 방금 구현한 op 이 "미구현" 으로 나온다.
./vendor/install_shim.sh > /tmp/install.log 2>&1; echo "EXIT=$?"
PYTHONDONTWRITEBYTECODE=1 TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/vendor \
  $PY <순전파 스크립트> ...

# 3 타깃 — docs/design/TORCH_C.md §7 과 동일
```

순전파 스크립트와 op 커버리지 측정 스크립트는 `/tmp/ops8/` 에 있습니다(저장소 밖 — 파일 범위
규정을 지키기 위해 커밋 대상에 넣지 않았습니다).

---

## 8. 이 작업이 건드린 파일

```
rust/torch_c/src/aten.rs      8 개 커널 + 디스패치 + IMPLEMENTED (62 -> 70)
tools/golden/cases.py         8 개 케이스 빌더 (169 케이스) + CASE_BUILDERS 등록
docs/kernels/OPS8.md                  이 문서
```

`rust/torch_c/src/bootstrap.py` 는 **건드리지 않았습니다** (지시대로). §5-1/§5-2 가 그 파일에서
해야 할 일을 적어 둔 것입니다.
