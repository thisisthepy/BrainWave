# Demand Round 5

## 1. What Landed

Four gaps from DEMAND4.md were implemented and closed:
- `torch._C._nn.adaptive_avg_pool2d` (`aten.adaptive_avg_pool2d.default`)
- `torch.roll` (`aten.roll.default`)
- `torch.greater` (`aten.greater.Tensor` and `aten.greater.Scalar`)
- `aten.where.ScalarSelf`

These implementations provided the missing operators for five model paths.

## 2. New Walls

Closing the above gaps successfully ran two models to completion, and revealed the next wall for the others:
- `resnet`: SUCCESS
- `mobilenet_v2`: SUCCESS
- `swin`: Hits `torch.floor` (`overload resolution has no table entry for this op`).
- `switch_transformers`: Hits `TensorBase.nonzero` (`not implemented in torch._C shim`).
- `whisper` (`.generate()`): Hits `TensorBase.nonzero` (`not implemented in torch._C shim`).

## 3. Re-Ranking Open Items

Closed items have been removed from the open list. The remaining items from DEMAND4.md and the new walls discovered are ranked below based on reach and implementation cost.

| rank | gap | model that hit it | kind | why this position |
|---|---|---|---|---|
| 1 | `aten.squeeze.default` | `mbart` | same as DEMAND4.md rank 1 | Promoted to the top by attrition. Plausibly a dispatch-arm wiring gap rather than new arithmetic. |
| 2 | `TensorBase.nonzero` | `switch_transformers`, `whisper` | new — missing method/kernel | Hit by two models (`switch_transformers` routing and `whisper` generation). Requires allocating a dynamic-size output tensor based on count. |
| 3 | legacy `torch.Tensor(ndarray)` constructor | `pegasus` | same as DEMAND4.md rank 3 | A structural gap. Remains highly relevant across multiple models but is a non-trivial structural change. |
| 4 | `torch.floor` | `swin` | new — missing overload/kernel | Next step for Swin transformer. Straightforward elementwise op. |
| 5 | `torch.linspace` | `convnext` | same as DEMAND4.md rank 5 | Unmoved. Common in vision backbones for stochastic-depth drop-path schedule. |
| 6 | `aten.linalg_vector_norm.default` | `sentence_embed` | same as DEMAND4.md rank 6 | Unmoved. Implementation-ready but narrow reach. |

## 4. Gates

```text
380 ok
ops covered: 202
DOCWATCH: PASS -- 311/311 evaluated marker(s) hold
SUMMARY: 4638/4638 table entries matched upstream, 0 failed
```

<!-- DOCWATCH: op-implemented aten.where.ScalarSelf -->
<!-- DOCWATCH: op-implemented aten.greater.Tensor -->
<!-- DOCWATCH: op-implemented aten.greater.Scalar -->
<!-- DOCWATCH: op-implemented aten.adaptive_avg_pool2d.default -->
<!-- DOCWATCH: op-implemented aten.roll.default -->
