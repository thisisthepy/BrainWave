# Demand Round 4

## 1. What Landed

Five gaps from DEMAND3.md were implemented and closed:
- `torch.max_pool2d` (`aten.max_pool2d.default`)
- `aten.where.default`
- `torch._C._nn.hardtanh`
- `torch.adaptive_avg_pool1d` (`aten.adaptive_avg_pool1d.default`)
- `torch._C._nn.one_hot`

These implementations provided the missing operators for five models.

## 2. New Walls

As predicted, closing the above gaps revealed the next wall for each respective model:
- `resnet`: Hits `torch._C._nn.adaptive_avg_pool2d` (inside pooler).
- `mobilenet_v2`: Hits `torch._C._nn.adaptive_avg_pool2d` (inside pooler).
- `swin`: Hits `torch.roll` (`overload resolution has no table entry for this op`).
- `switch_transformers`: Hits `torch.greater` (inside expert routing).
- `whisper` (`.generate()`): Hits `aten.where.ScalarSelf` (`aten op not implemented in torch._C shim`).

## 3. Re-Ranking Open Items

Closed items have been removed from the open list. The remaining items from DEMAND3.md and the new walls discovered are ranked below based on reach and implementation cost.

| rank | gap | model that hit it | kind | why this position |
|---|---|---|---|---|
| 1 | `aten.squeeze.default` | `mbart` | same as DEMAND3.md rank 1 | Promoted to the top by attrition. Plausibly a dispatch-arm wiring gap rather than new arithmetic. |
| 2 | `torch._C._nn.adaptive_avg_pool2d` | `resnet`, `mobilenet_v2` | new — missing kernel | Two models hit this immediately after their previous gaps were closed. High reach in vision models (pooler layer). |
| 3 | legacy `torch.Tensor(ndarray)` constructor | `pegasus` | same as DEMAND3.md rank 4 | A structural gap. Remains highly relevant across multiple models but is a non-trivial structural change. |
| 4 | `aten.where.ScalarSelf` | `whisper` (`.generate()`) | new — missing overload, generation-path | Hit deep inside HF's `generate()` loop immediately after `where.default`. |
| 5 | `torch.linspace` | `convnext` | same as DEMAND3.md rank 7 | Unmoved. Common in vision backbones for stochastic-depth drop-path schedule. |
| 6 | `aten.linalg_vector_norm.default` | `sentence_embed` | same as DEMAND3.md rank 8 | Unmoved. Implementation-ready but narrow reach. |
| 7 | `torch.roll` | `swin` | new — missing overload | Essential for Swin transformer's cyclic shift. Reach currently limited to Swin-like architectures. |
| 8 | `torch.greater` | `switch_transformers` | new — missing overload | Narrow reach, used in MoE expert routing logic. |

## 4. Gates

```text
370 ok
ops covered: 198
DOCWATCH: PASS -- 306/306 evaluated marker(s) hold
SUMMARY: 4618/4618 table entries matched upstream, 0 failed
```

<!-- DOCWATCH: op-implemented aten.where.default -->
<!-- DOCWATCH: op-implemented aten.max_pool2d.default -->
<!-- DOCWATCH: op-implemented aten.hardtanh.default -->
<!-- DOCWATCH: op-implemented aten.adaptive_avg_pool1d.default -->
<!-- DOCWATCH: op-implemented aten.one_hot.default -->
