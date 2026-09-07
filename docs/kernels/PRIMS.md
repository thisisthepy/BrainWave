# PRIMS — `prims.*` 열세 개, 그리고 그것이 실제로 무엇을 열었는가

docs/graph/DECOMP.md §12.4 가 "NNAPI 를 막는 것은 분해 24 개가 아니라 `prims.*` 13 개" 라고 셌습니다.
이 회차는 그 13 개를 구현했습니다. **열렸고, 그리고 §12.4 가 예상하지 못한 것이 하나 드러났습니다.**

## 1. 무엇을 구현했는가

| op | 커널 | 비고 |
|---|---|---|
| `prims.cos` `sin` `erf` `tanh` `sqrt` `reciprocal` | `unary_float` | aten 커널을 키만 바꿔 재사용 |
| `prims.rsqrt` | `rsqrt_default(op=...)` | 위와 같음. 거절 메시지가 호출된 이름을 대도록 키를 인자로 받게 바꿈 |
| `prims.neg` | `neg_default(op=...)` | 위와 같음 |
| `prims.clone` | `clone_default(op=...)` | 위와 같음 |
| `prims.view_of` | `prims_view_of` | `aten.alias` 와 같은 커널. 인자 이름이 `self` 가 아니라 `a` 라서 따로 씀 |
| `prims.transpose` | `prims_transpose` | **별도 커널** |
| `prims.split_dim` | `prims_split_dim` | **별도 커널** |
| `prims.broadcast_in_dim` | `prims_broadcast_in_dim` | **별도 커널** |

**아홉 개는 aten 커널의 다른 이름입니다** — 상류의 `impl_aten` 자체가 `torch.cos` / `torch.clone`
입니다. 그래도 자기 골든 케이스를 갖습니다: `_aten_implemented()` 는 "커널이 있고 골든이 상류와
대조한다" 는 뜻이고, 아무것도 대조하지 않는 키는 옆 키가 커버해 주지 않습니다.

**네 개는 별도 커널이고, 그중 셋은 같은 이름의 aten op 과 의미가 다릅니다.** 아래는 전부 상류
2.13.0 을 *호출해서* 읽은 것이지 참조 소스에서 유도한 것이 아닙니다.

- `prims.transpose(a, permutation)` 는 전체 순열입니다 — `aten.transpose.int` 처럼 두 축을
  바꾸는 것이 아닙니다. 그리고 `aten.permute` **보다 엄격합니다**:
  `prims.transpose(t, [-1, 0])` 는 `ValueError: Received an invalid permutation, [-1, 0]!` 이고
  `aten.permute(t, [-1, 0])` 는 계산합니다. aten 커널로 넘겼다면 상류가 거절하는 순열을
  받아들였을 것입니다.
- `prims.split_dim(a, dim, outer_length)` 는 **음수 `dim` 을 거절합니다** (`utils.validate_idx`).
  이 셰임의 모든 aten op 이 받아들이는 것을 이것만 거절합니다.
- `prims.broadcast_in_dim(a, shape, broadcast_dimensions)` 는 XLA 의 브로드캐스트입니다:
  입력의 `i` 번째 축이 결과의 *어느* 축이 되는지를 호출자가 말하므로 축을 가운데에 끼워 넣을 수
  있습니다. `broadcast_in_dim(ones(3), [3, 2], [0])` 에는 `aten.expand` 스펠링이 **없습니다** —
  expand 는 오른쪽 정렬이라 3 을 2 에 브로드캐스트하려다 거절합니다 (테스트의 대조군).
- `prims.view_of(a)` 는 `aten.alias` 입니다.

예외 *타입*이 op 마다 다른 것(`ValueError` · `AssertionError` · `RuntimeError` ·
`ZeroDivisionError`)은 상류가 그렇기 때문이고, 정돈하지 않고 그대로 옮겼습니다.

## 2. 열린 것 — 상류의 분해가 끝까지 돕니다

`rust/torch_c/pytests/nnapi_sizing.py`, 같은 세 그래프:

| | 이전 (docs/graph/DECOMP.md §12.5) | 지금 |
|---|---|---|
| union 표 LOWERED | 11 | **24** |
| 막고 있는 `prims.*` 커널 | 13 | **0** |
| union 표 REFUSED | 24 | 24 |

`_refs.erf` · `_refs.t` · `_refs.native_batch_norm` 같은 규칙이 이제 중간에 거절되지 않고 끝까지
돕니다. 값은 전부 맞습니다 — `whole_module` · `gelu` · `gelu_tanh` · `t` 가 전부
`max_abs_diff == 0.0` 입니다.

## 3. 드러난 것 — 개수는 내려가지 않았고, 이유가 바뀌었다

`Linear→GELU→Linear` 그래프에서 NNAPI 밖 op 은 여전히 **2 개**입니다. 다만 이름이 바뀌었습니다:

```
이전   aten.permute.default   aten.erf.default      (분해가 prims 에서 거절되어 멈춤)
지금   prims.transpose.default  prims.erf.default   (분해가 끝까지 돌아 prims 에 도달)
```

**그리고 두 번째가 더 어렵습니다.** `torch/backends/_nnapi/serializer.py` 의 `ADDER_MAP` 은
TorchScript 노드 종류(`aten::add`)로 키가 잡혀 있으므로, `prims.*` 노드는 무엇을 계산하든
**그 직렬화기가 절대 받을 수 없습니다.** 그리고 prims 는 참조 원시 연산이므로 **어느 상류 표도
그것을 더 분해하지 않습니다** — 그래서 `lower_to` 의 거절 문구가
"`prims.erf.default` 를 구현하라" 에서 "`prims.erf.default` 에 규칙이 없다" 로 바뀌었습니다.
그것은 커널을 추가해서 닫는 구멍이 아니라, 원시 연산이 원래 그런 것입니다.

모델별로도 같은 결이 나옵니다 (union 표, NNAPI 밖 op 수):

| 모델 | 이전 | 지금 |
|---|---|---|
| `smollm2_llama` | 17→15 | 17→15 |
| `vit` | 11→10 | 11→**11** |
| `mobilenet_v2` | 3→6 | 3→6 |

`vit` 은 **한 개 나빠졌습니다.** 이것을 숨기지 않는 것이 이 문서의 절반입니다.

> **다음 사람이 할 일은 prims 커널을 더 만드는 것이 아닙니다.** NNAPI 로 가려면 `prims.*` 노드를
> 다시 aten 으로 접는 단계(prims → aten 재작성)가 목적지 쪽에 필요하거나, `lower_to` 가
> 목적지의 네임스페이스를 알고 그 아래로 내려가지 않아야 합니다. 어느 쪽이든
> `torchnative/export/` 의 일이고 이 회차의 영역이 아니었습니다.

## 4. 이 회차가 함께 고친 것

세 곳이 `aten` 을 문자열로 박아 두고 있었고, 구현된 집합이 두 네임스페이스가 되자 드러났습니다.
셋 다 **틀린 답을 조용히 내는 종류**였습니다 — `torch.ops.aten.clone` 은 존재하므로,
`prims.clone.default` 키를 aten 으로 조회하면 *다른 연산자의* 스키마와 대조하고 통과합니다.

- `test_shim.py` 의 schema-road 스크립트 · `verify_schemas.py:check_shim_schemas`
  — `getattr(torch.ops, namespace)` 로 바꿈
- `test_every_implemented_op_has_schema_text` — `startswith("aten::")` 를 키가 나르는
  네임스페이스로 바꿈
- `tools/golden/reach.py` — `aten` 밖 키의 스펠링은 `torch.ops.<ns>.<op>.<overload>` 입니다
  (`torch.broadcast_in_dim` 은 없고, 있어서도 안 됩니다). 면제가 아니라 shape 3 과 같은 코퍼스
  검사로 넣었으므로, 아무 테스트도 부르지 않는 prims 커널은 여전히 여기서 빨개집니다.
  `test_the_thirteen_prims_ops_are_callable_by_their_own_key` 가 열세 개를 전부 그 이름으로
  부릅니다. 그 결과로 `expand` 와 `permute` 가 shape 3 허용목록에서 빠졌습니다.

## 5. 남은 구멍

`OpOverload.tags` 가 열세 개 전부에서 `[]` 이고 상류는 `['pt2_compliant_tag']` 입니다
(`verify_schemas.py`: 4685/4698, 13 failed). 태그는 `native_functions.yaml` 에서 읽는데 그 파일은
aten 만 담고 있고, prims 의 태그는 `torch/_prims/__init__.py` 의 `Library.define(..., tags=...)`
에 있습니다. 고칠 자리는 `bootstrap.py:_scan_aten_tags` 이며 이 회차의 영역 밖이었습니다.
`pt2_compliant_tag` 는 분해 경로가 읽는 태그(`core`, `maybe_aliasing_or_mutating`)가 아니므로
지금 무엇을 막고 있지는 않습니다. docwatch 의 `schema_entries_matched ge 4458` 은 통과합니다.

## 6. `rwkv` — docs/architectures/DEMAND8.md 의 다섯 번째 모델이 닫혔습니다

docs/architectures/DEMAND8.md §2.6 이 남긴 하나입니다. `TensorBase.new_empty` 뒤에 벽이 **하나 더** 있었고,
첫 번째를 닫기 전에는 아무도 그것을 볼 수 없었습니다.

| 벽 | 무엇 | 어떻게 닫았는가 |
|---|---|---|
| `TensorBase.new_empty` | 스펠링 + 커널 | `aten.new_empty.default` (= `new_zeros` 의 스키마 그대로, 0 채움) + `methods.json` |
| `torch.maximum` | 스펠링 + 키 | `aten.maximum.default` (= `max.other` 의 커널, NaN 규칙 포함) + `overloads.json` · `methods.json` 양쪽 |

**결과:** `rwkv` 가 forward 하고 상류와 일치합니다 — 256 개 원소, **max abs diff 7.15e-07**
(scale 2.17). toy config(`hidden_size=32`, 2 layer), `RwkvModel`, eval 모드.

> **가중치는 상류에서 만들어 양쪽에 실었습니다.** 다른 네 모델과 같은 주장이 아니므로 적어 둡니다:
> `RwkvPreTrainedModel._init_weights` 는 `nn.init.orthogonal_` 을 부르고 그것은
> `torch.linalg.qr` 인데, 이 셰임에는 QR 이 없습니다(`torch._C._linalg.linalg_qr`). 따라서 두 쪽을
> **같은 시드로 같게 초기화할 수는 없습니다.** 같은 가중치를 싣는 것은 `new_empty` 가 막고 있던
> **forward 를** 분리해서 재는 방법이고, 이 회차가 닫은 것도 forward 입니다.
> 초기화 경로의 `linalg_qr` 은 열려 있는 다음 벽입니다.

`aten.maximum` 은 `max.other` 와 **같은 함수이고 다른 스키마**입니다. 그래서 커널은 공유하되
키는 따로 두었고, 골든도 `max.other` 의 전체 스윕을 그대로 다시 돌립니다 — 상류 쪽
`torch_call` 이 `torch.ops.aten.maximum.default` 를 풀기 때문에, 새 키가 `min` 쪽에 잘못 배선되거나
NaN 보정을 잃는 것을 그 스윕만이 잡습니다.
