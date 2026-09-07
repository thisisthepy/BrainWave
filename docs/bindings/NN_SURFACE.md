# `_C._nn` 과 파이썬 스펠링 — 재고, 배선하고, 남은 것

커널은 있는데 배선이 없다던 벽(`docs/kernels/OPS8.md`)을 실측하고 닫은 기록입니다.

**한 줄 결론.** `_C._nn` 표면 70개는 전부 비어 있던 것이 맞지만, **모델 경로가 부르는 것은 3개뿐**
입니다. 그 3개와 파이썬 스펠링 11개를 배선한 결과, `torch.nn` 으로 직접 조립한 2층 Llama 가
**상류와 같은 토큰을 뱉습니다** (§7).

기준선 대비 회귀 없음: 골든 1212/1212 ops covered=70, 스모크 exit 0, 3 타깃 exit 0,
스키마 127 → **154** (늘어난 것, §8).

---

## 1. OPS8.md 의 보고를 실측으로 교정

OPS8.md 는 벽을 이렇게 적었습니다.

> `_C._nn` 표면 전체가 비어 보인다 (`gelu`, `softmax`, `layer_norm`, `pad` 가 똑같이 거부).

절반은 맞고 절반은 틀렸습니다. 넷 중 **둘은 `_C._nn` 의 이름이 아닙니다.**

| 이름 | `_C._nn` 멤버인가 | 실제로 무엇이 막는가 |
|---|---|---|
| `gelu` | **예** | `_C._nn.gelu` 스텁. Llama 는 `silu` 를 쓰므로 경로에 없음 |
| `pad` | **예** | `_C._nn.pad` 스텁. 단, `F.pad` 는 그 전에 `_C._get_deterministic_algorithms` 에서 먼저 죽음 |
| `softmax` | **아니오** | `F.softmax` 는 `input.softmax(dim)` — **TensorBase 메서드**입니다 (§6) |
| `layer_norm` | **아니오** | `F.layer_norm` 은 `_C._nn` 을 거치지 않고, **`_C._get_cudnn_enabled`** 에서 죽습니다 |

마지막 두 줄이 중요합니다. `_C._nn` 을 다 채워도 `F.softmax` 와 `F.layer_norm` 은 안 됩니다 —
막는 것이 다른 곳에 있기 때문입니다. 반대로 `_get_cudnn_enabled` / `_get_deterministic_algorithms`
는 설정값 게터라 답하기 싼데, **둘 다 Llama 경로에 없어서** 이번에 건드리지 않았습니다.

> **정정 (문서 감사, 2026-09):** 둘 다 오늘은 막히지 않는다. `torch._C._get_cudnn_enabled` 가
> 존재하고 `True` 를 답한다(오늘 shim 위에서 확인). 실측: `F.layer_norm(x, (4,))` 와
> `F.pad(x, (1,1))` 둘 다 오늘 shim 위에서 성공한다 — `F.layer_norm` 은 `docs/models/GPT2.md` 감사(라운드
> 2)가 이미 확인한 것과 같은 발견으로, `_C._nn` 을 거치지 않는 별도 파이썬 합성
> (`bootstrap.py:5863`)이 답한다.

---

## 2. `_C._nn` 의 실제 범위 (재본 숫자)

**우리 쪽.** 셰임의 `_C._nn` 은 `surface.json` 의 70개 이름을 갖고, **70개 전부** 호출 시
`NotImplementedError` 였습니다. OPS8 의 "통째로 비어 있다"는 정확했습니다.

```
dir(_C._nn) = 70,  전부 function,  호출 결과 Counter({'NotImplementedError': 70})
```

**상류 쪽.** `torch._C._nn` 의 호출 가능 멤버는 **96개**입니다 (스텁이 선언하는 70개보다 많음 —
이 26개 차이는 이번 경로에서 하나도 불리지 않아 그대로 둡니다).

**모델 경로가 부르는 것.** 96개 각각을 기록 래퍼로 감싸고(중첩 호출이 `F.*` 파이썬 래퍼에
가려지지 않도록 `TorchFunctionMode` 대신 직접 래핑), 2층 Llama 의 순전파 + greedy `generate` 를
`sdpa` 와 `eager` 두 어텐션 구현 모두로 돌렸습니다.

```
   150  _C._nn.linear
    10  _C._nn.scaled_dot_product_attention
    20  _C._nn.silu
        ── 96개 중 3개. 나머지 93개는 0회.
```

**즉 채워야 할 것은 표면 전체가 아니라 셋입니다.** `gelu` 가 안 불리는 이유는 Llama 가 `silu` 를
쓰기 때문이고, `layer_norm` 이 안 불리는 이유는 RMSNorm 을 쓰기 때문이며, `softmax` 가 안 불리는
이유는 `sdpa` 가 그것을 커널 안으로 삼키기 때문입니다.

---

## 3. `torch.*` / `Tensor.*` 중 모델이 부르는데 없던 것

같은 방식으로 `TorchFunctionMode` 로 파이썬 레벨 스펠링을 전부 기록했습니다 (`sdpa`+`eager`,
순전파+`generate` 합산).

**`torch.<name>` — 부르는 14개 중 없던 것 2개:**

| 스펠링 | 호출 수 | 이전 상태 | 지금 |
|---|---|---|---|
| `torch.matmul` | 20 | 표 항목 없음 | **배선** → `aten.matmul.default` |
| `torch.where` | 5 | 표 항목 없음 | **배선** (커널 없음, §9) |

나머지 12개(`arange` `argmax` `cat` `embedding` `empty` `full` `is_floating_point` `isin` `ones`
`pow` `rsqrt` `tensor`)는 이미 있었습니다.

**`Tensor.<method>` — 부르는 38개 중 없던 것 3개:**

| 스펠링 | 호출 수 | 지금 |
|---|---|---|
| `Tensor.neg` / `__neg__` | 40 | **배선** → `aten.neg.default` |
| `Tensor.__rsub__` (`1 - x`) | 8 | **배선** → `overloads.json` 의 `rsub` (§4) |
| `Tensor.le` / `__le__` | 5 | **배선** (커널 없음, §9) |

**`F.dropout` → `_VF.dropout`** 이 10회 불립니다. `torch.dropout` 도 없었고, 지금은 §5 의
합성으로 답합니다.

`torch.bmm` 은 이 경로에서 불리지 않지만 지시에 명시되어 함께 넣었습니다. `torch.t` / `torch.neg`
는 대응하는 메서드가 측정되었고 커널이 이미 있어 대칭으로 넣었습니다 — **측정된 것이 아니라는
점을 여기 적어 둡니다.**

---

## 4. `Tensor.__rsub__` 가 `methods.json` 이 아니라 `overloads.json` 인 이유

`1 - x` 는 메서드처럼 보이지만, 벤더링된 트리는 이렇게 씁니다.

```python
# torchnative/src/main/torch/_tensor.py:1108
def __rsub__(self, other):
    return _C._VariableFunctions.rsub(self, other)
```

그러니까 실제로 도는 경로는 `TensorBase.__rsub__` 가 아니라 **`torch._C._VariableFunctions.rsub`**
입니다. `overloads.json` 에 `rsub` 를 넣는 것이 그 경로를 고치는 방법이고, 그렇게 했습니다.
`methods.json` 의 `__rsub__` 도 함께 넣었는데 — 상류 `TensorBase` 도 그 멤버를 갖기 때문에
(벤더링 트리 없이 `TensorBase` 만 쓸 때 일관되도록) — **도는 것은 앞쪽입니다.**

---

## 5. `linear` 과 `dropout` 을 커널이 아니라 파이썬 합성으로 만든 이유

지시는 "`_C._nn.linear` 는 자기 커널을 갖지 말고 이미 있는 aten 커널로 내려가야 한다" 였고,
**상류가 하는 것이 정확히 그것입니다.**

`aten::linear` 와 `aten::dropout` 은 상류에서 `CompositeImplicitAutograd` 입니다 — 커널이
아예 없고, 분해가 곧 구현입니다. `TorchDispatchMode` 로 `F.linear(...)` 밑을 보면
`aten.linear.default` 는 **한 번도 나오지 않습니다.**

`at::native::linear` 의 분기를 케이스별로 재서 옮겼습니다.

| 입력 | bias | 상류가 부르는 aten |
|---|---|---|
| 2-D | 있음 | `t`, `addmm` |
| N-D (contiguous) | 있음 | `view`, `t`, `addmm`, `view` |
| 아무 rank (non-contiguous) | 있음 | `t`, …, `matmul`, …, `add.Tensor` |
| 2-D | 없음 | `t`, `mm` |
| N-D | 없음 | `t`, `view`, `mm`, `_unsafe_view` — 즉 그냥 `matmul(input, weight.t())` |

**bias 없는 경우는 모든 rank 에서 `matmul(input, weight.t())` 이고**, 그것이 지금 배선된 것입니다.
Llama 는 `attention_bias=False` / `mlp_bias=False` 라 **실제 모델 경로는 전부 이 무-bias 경로**를
탑니다 (측정된 150회 전부 `t`+`mm`+`view`+`_unsafe_view`, `addmm` 0회).

### 때운 것 — bias 경로

`aten.addmm.default` 커널이 없습니다. 그래서 bias 가 있으면 `matmul` + `add.Tensor` 로 갑니다.

- **값은 맞습니다** — `nn.Linear` bias=True 순전파에서 상류 대비 최대 상대오차 **1.4e-07** (§7).
- **하지만 상류와 다른 aten 을 부릅니다.** `addmm` 은 융합 GEMM 이고 이것은 GEMM + 별도 브로드
  캐스트 덧셈이라, 누적 순서가 다릅니다. 성능도 갈립니다.
- **자동으로 은퇴하게 만들었습니다.** 분기가 `_aten_all_implemented()` 를 보고 있어서,
  `addmm` 커널이 들어오는 날 `bootstrap.py` 를 고치지 않아도 상류 경로로 갈아탑니다.

`dropout` 은 때운 것이 아닙니다. `aten::dropout` 의 본문이
`if (p == 0 || !train || numel == 0) return input;` 로 디스패처에 닿기 전에 끊어지고 —
`F.dropout(x, 0.0, False)` 가 상류에서 **aten 기록을 하나도 남기지 않는 것으로 확인** — 추론
모드 모델은 커널이 필요 없습니다. `train=True` 는 그대로 `aten.dropout.default` 를 부르고
한 문에서 거부하며, 없는 커널 이름을 정확히 말합니다.

---

## 6. `scaled_dot_product_attention` — 분해가 아니라 선택

상류의 `sdpa` 는 백엔드를 고릅니다. 무엇을 고르는지 **추론하지 않고 쟀습니다.**

| 입력 | 상류가 가는 곳 |
|---|---|
| 4-D, float32/float64/float16/bfloat16, `dropout_p == 0` | `aten._scaled_dot_product_flash_attention_for_cpu` |
| 마스크 있음 / `is_causal` / **둘 다** | 같은 곳 (이 aten 은 둘을 더합니다) |
| `enable_gqa=True` (H≠H_kv) | 같은 곳 — 커널이 내부에서 브로드캐스트 |
| **3-D 입력** | math 백엔드 (`mul.Scalar`, `expand`, `view`, `bmm`, `_safe_softmax`, …) |
| **`dropout_p > 0`** | math 백엔드 + `bernoulli_`, `div_` |
| **bool 마스크** | `scalar_tensor` + `where.self` 로 float 마스크 변환 후 flash |

flash 경로만 배선했습니다. 그 커널은 이미 있고(`aten.rs`), `(output, logsumexp)` 쌍을 돌려주므로
상류처럼 첫 번째만 취합니다. **나머지는 근사하지 않고 이름을 대며 거부합니다** —
`aten._safe_softmax.default` 를 평범한 softmax 로 대체하면 전부 마스킹된 행에서 정확히 갈리는데,
그것이 `_safe_softmax` 가 존재하는 이유이기 때문입니다.

`enable_gqa=True` 는 우리 커널이 헤드 브로드캐스트를 안 하므로 거부합니다. transformers 의
Llama 는 `repeat_kv` 를 스스로 하고 `False` 로 넘기므로 경로에 없습니다.

**`F.softmax` 에 대해.** 표에 넣지 **않았습니다.** 파서 레벨 키는 `aten.softmax.int` 인데
디스패처가 실제로 보는 키는 `aten._softmax.default` 입니다(측정). `softmax.int` 를 적으면 이
셰임이 절대 구현하지 않을 작업 항목을 이름 붙이는 셈이라, 아무것도 안 적는 편이 낫습니다.
(`to.dtype` 대신 `_to_copy` 를 택한 기존 판단과 같은 근거입니다.)

---

## 7. 판정 — 실제로 도는가

`vendor/install_shim.sh` 로 벤더 트리에 넣고, **같은 스크립트를 상류 torch 와 벤더 트리에서 각각
돌려** 숫자를 대조했습니다. 가중치는 결정적 공식으로 채워 양쪽이 같은 수를 받습니다.

```
상류  /Volumes/macMini/caches/spike-venv/.../torch/__init__.py
셰임  /Volumes/macMini/thisisthepy/torchnative/torchnative/src/main/torch/__init__.py
```

| 케이스 | n | 최대 상대오차 |
|---|---|---|
| `nn.Linear(3,4)` bias=True | 15 | 1.43e-07 |
| `nn.Linear(3,4)` bias=False | 15 | 1.23e-07 |
| `nn.Linear(2,3,4)` bias=True | 30 | 6.38e-08 |
| `nn.Linear(2,3,4)` bias=False | 30 | 5.34e-08 |
| `nn.Linear(2,3,6,4)` bias=True | 180 | 6.38e-08 |
| `nn.Linear(2,3,6,4)` bias=False | 180 | 1.07e-07 |
| `F.scaled_dot_product_attention` (plain) | 64 | 1.74e-07 |
| `F.scaled_dot_product_attention` (`is_causal`) | 64 | 5.98e-08 |
| `F.scaled_dot_product_attention` (마스크) | 64 | 1.16e-07 |
| `F.scaled_dot_product_attention` (마스크+scale) | 64 | 1.79e-07 |
| **2층 Llama 순전파** (`torch.nn` 직접 조립) | 128 | **5.79e-07** |

전부 float32 반올림 범위입니다. 그리고 greedy:

```
2층 Llama, greedy 4 토큰
  상류  [1, 2, 3, 4, 29, 30, 31, 1]
  셰임  [1, 2, 3, 4, 29, 30, 31, 1]      ← 동일
```

조립에 쓴 것: `nn.Embedding`, `nn.Linear`, `nn.Parameter`, `nn.ModuleList`, RMSNorm,
RoPE(`cos`/`sin`/`cat`/`-x`), `F.scaled_dot_product_attention(is_causal=True)`, `F.silu`,
`torch.argmax`, `torch.cat`. **`from_config` 은 지시대로 판정에 쓰지 않았습니다** —
`torch.distributed.Store` 벽은 그대로입니다.

---

## 8. 회귀 확인

| 검사 | 결과 |
|---|---|
| 골든 | **1212/1212, ops covered=70, pending 0, exit 0** (기준선과 동일) |
| `--inject-fault value` | exit 1 |
| `--inject-fault shape` | exit 1 |
| `--inject-fault dtype` | exit 1 |
| 스모크 (`pytests/run.sh`) | exit 0, 62 ok |
| 스키마 (`verify_schemas.py`) | exit 0, **154/154** |
| 호스트 빌드 | exit 0 |
| Android (`aarch64-linux-android`) | exit 0 |
| iOS (`aarch64-apple-ios`) | exit 0 |

**스키마가 127 → 154 로 늘어난 이유** (줄면 회귀, 늘면 정상):

```
overloads.json  47 → 64   (+17)   bmm 4, matmul 2, neg 2, t 1, rsub 2, where 6
methods.json    80 → 90   (+10)   __rsub__ 2, neg 1, __neg__ 1, bmm 1, t 1, le 2, __le__ 2
```

**골든의 `ops covered=70` 이 안 변한 것이 맞습니다.** 이번 작업은 커널을 하나도 추가하지 않았고,
있는 커널에 파이썬 스펠링을 붙였을 뿐이기 때문입니다.

---

## 9. 커널이 없어서 못 한 것 — 다른 작업으로 넘길 목록

전부 **배선은 끝났고 커널만 없는** 상태입니다. 즉 호출하면 `aten op not implemented in torch._C
shim: <정확한 키>` 로 거부하며, 필요한 것의 이름을 스스로 말합니다.

| aten 키 | 무엇이 막히나 | 모델 경로 |
|---|---|---|
| `aten.le.Tensor`, `aten.le.Scalar` | `Tensor.le`, `x <= y` | **예** (5회, 측정됨) |
| `aten.where.self` | `torch.where(cond, a, b)` | **예** (5회, 측정됨) |
| `aten.scalar_tensor.default` | `torch.where(cond, t, 스칼라)`, sdpa 의 bool 마스크 변환 | **예** |
| `aten.addmm.default` | `nn.Linear(bias=True)` 가 상류와 같은 aten 을 부르게 함 | 아니오 (Llama 는 bias 없음). §5 의 때움을 은퇴시킴 |
| `aten._softmax.default` | `F.softmax`, `Tensor.softmax` | eager 어텐션에서만 |
| `aten._safe_softmax.default` | sdpa 의 math 백엔드 (3-D 입력, `dropout_p>0`) | 아니오 |
| `aten.dropout.default` | `F.dropout(train=True, p>0)` | 아니오 (추론은 §5 로 답함) |
| `aten.matmul.default` 의 1-D 피연산자 | `nn.Linear` 에 1-D 입력 | 아니오 |

> **정정 (문서 감사, 2026-09):** 표의 여덟 줄 중 일곱은 오늘 커널이 생겼다 — `aten.le.Tensor`/
> `aten.le.Scalar`/`aten.where.self`/`aten.scalar_tensor.default`/`aten.addmm.default`/
> `aten._softmax.default`/`aten._safe_softmax.default` 가 전부 오늘 168-op `_aten_implemented()`
> 목록에 있다. 실측: `x <= y`, `torch.where(cond, a, b)`, `nn.Linear(3, 4, bias=True)(x)`,
> `F.softmax(x, dim=0)` 전부 오늘 shim 위에서 계산된다. **`aten.dropout.default` 만 여전히
> 미구현** — 실측: `torch.ops.aten.dropout.default(x, 0.5, True)` 가 오늘도 이름을 대고 거부한다
> (§5 가 이미 "때운 것이 아니다" 라고 적어 둔 그대로 — 추론 경로는 필요 없으므로 낮은 우선순위가
> 맞다). `aten.matmul.default` 의 1-D 피연산자 거부는 별개 항목(커널은 있고 특정 입력만 거부)이라
> 아래에서 재확인하지 않는다.
> <!-- DOCWATCH: op-implemented aten.le.Tensor -->
> <!-- DOCWATCH: op-implemented aten.where.self -->
> <!-- DOCWATCH: op-implemented aten.addmm.default -->
> <!-- DOCWATCH: op-implemented aten._softmax.default -->
> <!-- DOCWATCH: op-not-implemented aten.dropout.default -->

마지막 줄은 커널이 있는데 거부하는 경우입니다: `aten.matmul.default` 가
`matmul with a 1-D operand (1D x 2D) is not implemented ... torch's vector rules were not
measured` 로 스스로 거부합니다. 모델 경로는 항상 2-D 이상이라 막히지 않습니다.

---

## 10. 확인하지 않은 것 / 모르는 것

- **`_C._nn` 의 나머지 93개**는 이 경로에서 0회라는 것만 확인했습니다. 다른 모델(예: GELU 를 쓰는
  BERT 계열, `LayerNorm` 을 쓰는 것)이 무엇을 부르는지는 **재지 않았습니다.**
- 상류 `_C._nn` 이 96개인데 `surface.json` 은 70개입니다. **이 26개 차이가 무엇인지 확인하지
  않았습니다** — 이번 경로에서 하나도 불리지 않아 그대로 두었습니다.
- `F.layer_norm` / `F.pad` 를 막는 `_C._get_cudnn_enabled` / `_C._get_deterministic_algorithms`
  는 답하기 싼 설정 게터로 보이지만, **모델 경로에 없어 건드리지 않았고 검증도 하지 않았습니다.**
- `enable_gqa=True` 일 때 상류 flash 커널이 헤드를 **정확히 어떤 규칙으로** 브로드캐스트하는지는
  재지 않았습니다. 거부만 합니다.
- bias 경로의 `matmul`+`add` 때움이 **큰 행렬에서** 상류와 얼마나 벌어지는지는 재지 않았습니다.
  잰 것은 §7 의 크기까지입니다.
- 기기(Android/iOS) 에서의 임포트는 이번에도 **링크만** 확인했습니다.
