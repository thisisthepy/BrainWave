# DEMAND7 — Re-measuring the 19 and widening the set

This round re-measures the 19 architectures from docs/architectures/DEMAND.md and docs/architectures/DEMAND6.md, and then widens the set with 10 new architectures and tasks (including a decoder-only MoE, vision transformers with detection/segmentation, a multimodal model, state-space/recurrent models, and `.generate()` across seq2seq).

**Nothing in this round changed source** — `rust/torch_c/`, `bootstrap.py`, `tools/golden/cases.py`, `torchnative/src/main/torch/` (vendored) are all untouched. `git status --short` is empty throughout.

## 1. The existing nineteen

All 19 models were re-measured using the same toy `AutoConfig`s and hand-built tensors as before. 

| model (task) | loads | forwards | matches upstream |
|---|---|---|---|
| `bert` | yes | yes | **yes** — 44/44 weights identical, output max abs diff 1.49e-07 |
| `roberta` | yes | yes | **yes** — 44/44 weights identical, output max abs diff 1.19e-07 |
| `albert` | yes | yes | **yes** — 30/30 weights identical, output max abs diff 2.68e-07 |
| `vit` | yes | yes | **yes** — 40/40 weights identical, output max abs diff 5.22e-08 |
| `clip` | yes | yes | **yes** — 78/78 weights identical, output max abs diff 4.77e-07 |
| `wav2vec2` | yes | yes | **yes** — 53/53 weights identical, output max abs diff 8.05e-07 |
| `qwen2_moe` | yes | yes | **yes** — 35/35 weights identical, output max abs diff 1.34e-07 |
| `whisper` (forward) | yes | yes | **yes** — 90/90 weights identical, output max abs diff 3.02e-07 |
| `whisper` (`.generate()`) | yes | yes | **yes** — 90/90 weights identical, output max abs diff 0.00e+00 |
| `t5` | yes | yes | **yes** — 50/50 weights identical, output max abs diff 2.26e-06 |
| `switch_transformers` | yes | **no** | n/a |
| `bart` | yes | yes | **yes** — 95/95 weights identical, output max abs diff 1.19e-07 |
| `mbart` | yes | yes | **yes** — 99/99 weights identical, output max abs diff 8.94e-08 |
| `pegasus` | yes | yes | **yes** — 95/95 weights identical, output max abs diff 1.19e-07 |
| `resnet` | yes | yes | **yes** — 92/92 weights identical, output max abs diff 4.47e-07 |
| `mobilenet_v2` | yes | yes | **NO — forwards, but output max abs diff is 9.40e-04** |
| `convnext` | yes | yes | **yes** — 92/92 weights identical, output max abs diff 1.79e-07 |
| `swin` | yes | **no** | n/a |
| `sentence_embed` | yes | yes | **yes** — 39/39 weights identical, output max abs diff 5.96e-08 |

**17 of the 19 existing targets now forward.** 
- 16 of them match upstream exactly (within floating-point accumulation noise).
- **`mobilenet_v2` forwards but does NOT match upstream.** This is a critical divergence. The output differs by 9.40e-04, significantly above the noise floor. 
- `whisper.generate()` (which failed on `where.ScalarSelf` in DEMAND6) now completes and matches upstream perfectly (0.00e+00 diff for generated token IDs).
- `resnet` (which failed on `adaptive_avg_pool2d` in DEMAND6) now completes and matches perfectly.
- `swin` and `switch_transformers` still refuse, but have progressed to new walls (see §3).

## 2. The widened set

10 new architectures and tasks were added to widen the scope and find new coverage gaps:

| model (task) | loads | forwards | matches upstream | exact refusal |
|---|---|---|---|---|
| `mixtral` (decoder-only MoE) | yes | yes | **yes** — 21/21 weights identical, output max abs diff 8.94e-08 | — |
| `yolos` (DETR-shaped vision-detection) | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch._C._nn.upsample_bicubic2d` |
| `segformer` (vision-segmentation) | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch.floor(...) -- overload resolution has no table entry for this op` |
| `blip` (multimodal vision-language) | yes | yes | **yes** — 93/93 weights identical, output max abs diff 1.34e-07 | — |
| `mamba` (state-space) | yes | yes | **yes** — 23/23 weights identical, output max abs diff 9.54e-07 | — |
| `rwkv` (recurrent) | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: TensorBase.ndimension` |
| `t5` (`.generate()`) | yes | yes | **yes** — output max abs diff 0.00e+00 | — |
| `bart` (`.generate()`) | yes | yes | **yes** — output max abs diff 0.00e+00 | — |
| `mbart` (`.generate()`) | yes | yes | **yes** — output max abs diff 0.00e+00 | — |
| `pegasus` (`.generate()`) | yes | yes | **yes** — output max abs diff 0.00e+00 | — |

**6 of 10 new targets forward and match upstream.** All four `.generate()` tests (T5, BART, MBART, Pegasus) worked on the first try. Mamba and BLIP also passed perfectly.

## 3. Ranked list of what is missing

Ranked by **how many distinct models** hit each wall as their first blocker in this combined round.

| rank | gap | models wanting it | kind | notes |
|---|---|---|---|---|
| 1 | `torch.floor` | `swin`, `segformer` (2) | missing spelling / kernel | Present in `overloads.json` but missing a table entry in the shim's overload resolution. This is the last wall standing before these two vision models reach their next block (or pass). |
| 2 | numeric divergence | `mobilenet_v2` (1) | correctness bug | The model successfully runs end-to-end but produces outputs differing by ~1e-03 from upstream. This is more severe than a refusal. Could be an issue in `hardtanh`, `adaptive_avg_pool2d`, or a similar newly-implemented kernel. |
| 3 | `torch._C._nn.upsample_bicubic2d` | `yolos` (1) | missing kernel | New leaf op required for the YOLO/DETR-shaped detection head. |
| 4 | `TensorBase.index_add_` | `switch_transformers` (1) | missing spelling / member | Previously blocked on `greater`, MoE expert routing has moved on to this in-place mutation. |
| 5 | `TensorBase.ndimension` | `rwkv` (1) | missing unbound member | Simple structural access needed by the recurrent model's routing. |

*(Note: `llava` was attempted but failed locally during configuration with a Hub-related import. A multimodal model `blip` was substituted and successfully forwarded and matched upstream.)*

## 4. Gates (unchanged)

```
380 ok
DOCWATCH: PASS -- 329/329
SUMMARY: 8476/8476 cases passed, 0 failed, ops covered=203, pending case builders=0
```

Run once at the end. All gates perfectly match their previous baseline. No source files were touched.
