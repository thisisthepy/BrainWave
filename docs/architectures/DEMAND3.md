# DEMAND3 — re-sweeping the ten stalled models plus `whisper.generate()`

Measurement round only. **Nothing in this round changed source** — `rust/torch_c/{aten.rs,tensor.rs,dtype.rs,flash.rs}`,
`bootstrap.py`, `tools/golden/cases.py`, `torchnative/src/main/torch/` (vendored) are all untouched.
`git status --short` in the worktree is empty throughout.

docs/architectures/DEMAND.md ran 18 architectures; 7 forwarded and matched upstream, 1 (`whisper`) forwarded but
its `.generate()` did not, and 10 did not forward at all. docs/architectures/DEMAND1.md then closed five gaps
(`native_batch_norm`, `full_like`, `new_zeros`, `as_tensor`, `meshgrid`). This round re-runs the
eleven stalled targets (the 10 non-forwarders plus `whisper.generate()`) against the closed-gap
build to see which moved, using the **same scope** as DEMAND.md §3: self-built toy `AutoConfig`s,
hand-built tensors, no Hub checkpoints, no real image/audio preprocessing.

## 0. A method correction found this round, before any model result

**`torch.randint` does not reproduce upstream bit-for-bit after an identical `torch.manual_seed`,
even though `torch.randn` does.** Measured directly, no model involved:

```
torch.manual_seed(1); torch.randn(2,6)     ->  identical on both sides, all 12 values
torch.manual_seed(1); torch.randint(0,2,(4,))  ->  shim [1,1,1,0]   upstream [1,1,0,0]
torch.manual_seed(1); torch.randint(2,100,(2,6)) -> shim and upstream disagree on every entry
```

This was first noticed on `t5`: state_dict weights compared bit-identical (confirming
`torch.manual_seed(0)` reproduces upstream's init correctly, same as DEMAND.md §1's method),
but the forward output was off by several units — not float32-eps noise — until the token ids
themselves were checked and turned out to differ between the two sides despite identical
`torch.manual_seed(1)` immediately before `torch.randint`. **Every text-model input this round
uses a hand-built fixed token-id list instead of `torch.randint`.** `torch.randn`-built
image/audio tensors are unaffected and used as before. This is a per-op RNG-algorithm divergence
in the shim (a different sampling routine from upstream's, for integers specifically), not a
per-model finding — flagged here once rather than in every row, and worth a line in DEMAND4 as a
candidate gap even though nothing in this round's eleven targets calls `torch.randint` on the
model-under-test's own path (only the *harness* used it, to build inputs).

---

## 1. Method

Same as docs/architectures/DEMAND.md §1: `transformers` 5.15.1, `torch` 2.13.0 upstream, toy `AutoConfig`s
(small hidden size, 1-2 layers, few heads, tiny vocab). Every script prints
`"shim" if hasattr(torch._C, "_aten_implemented") else "upstream"` as its first line, both sides,
every run — transcribed literally below. Every run wrapped in a 120s `SIGALRM` (see
`/tmp/sweep-resweep/common.py`). Each side its own subprocess
(`PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1` for the shim;
`env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL` for upstream). Built via
`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-resweep`,
`TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib`, `bash vendor/install_shim.sh`.
Scratch scripts under `/tmp/sweep-resweep/` (not committed).

"Matches upstream" method as DEMAND.md §1: `torch.manual_seed(0)` immediately before model
construction on both sides, then a forward on identical hand-built input tensors (fixed token-id
lists for text models per §0 above; `torch.manual_seed(1)`-seeded `torch.randn` for vision/audio,
matching DEMAND.md), comparing every `state_dict` tensor and available output tensors by
`.tolist()`.

## 2. Results table — the eleven stalled targets

| model | DEMAND.md result | this round | forwards now? | matches upstream? | refusal (if still stopped) |
|---|---|---|---|---|---|
| `t5` (`T5ForConditionalGeneration`) | did not forward — `torch.full_like`, `modeling_t5.py:258 _relative_position_bucket` | **forwards** | **yes** | **yes** — 50/50 state_dict keys bit-identical, `encoder_last_hidden_state` max abs diff 7.2e-07, `logits` max abs diff 1.2e-06 (float32-eps scale) | — closed by DEMAND1's `full_like` |
| `switch_transformers` (MoE-T5) | did not forward — same `torch.full_like` as `t5` | **moved, still does not forward** | no | n/a | **NEW WALL**: `NotImplementedError: not implemented in torch._C shim: torch._C._nn.one_hot` — MoE router's expert-dispatch one-hot (not reached before, because `full_like` fired first in the shared `_relative_position_bucket` code both models call). Confirms `full_like`'s closure moved this model past the shared wall onto a wall specific to the MoE routing path. |
| `bart` (`BartForConditionalGeneration`) | did not forward — `TensorBase.new_zeros`, `modeling_bart.py:62 shift_tokens_right` | **forwards** | **yes** | **yes** — 95/95 state_dict keys bit-identical, `encoder_last_hidden_state` max abs diff 4.8e-07, `logits` max abs diff 3.0e-08, `loss` max abs diff 4.8e-07 (all float32-eps scale) | — closed by DEMAND1's `new_zeros` |
| `mbart` (`MBartForConditionalGeneration`) | did not forward — `aten.squeeze.default`, `modeling_mbart.py:77 shift_tokens_right` | **unchanged** | no | n/a | **same wall**: `NotImplementedError: aten op not implemented in torch._C shim: aten.squeeze.default`. Consistent with DEMAND.md §0.1 rank 2 — `squeeze.default` was not one of the five closed this round, so this is the expected non-move. |
| `pegasus` (construction, `PegasusForConditionalGeneration`) | did not construct — legacy `torch.Tensor(ndarray)`, `modeling_pegasus.py:91 create_weight` | **unchanged** | n/a (construction fails) | n/a | **same wall**: `NotImplementedError: torch._C shim: TensorBase(...) ... building from data, torch.Tensor(ndarray), is a third form and is not implemented`. Matches DEMAND.md §0.1 rank 1 and DEMAND1.md's explicit note that this constructor is out of scope and untouched this round. |
| `resnet` (`ResNetForImageClassification`) | did not forward — `torch.batch_norm` at the stem `BatchNorm2d`, after a successful `conv2d` | **moved past batch_norm, still does not forward** | no | n/a | **NEW WALL**: `NotImplementedError: not implemented in torch._C shim: torch.max_pool2d(...) -- overload resolution has no table entry for this op`, with the suggested workaround `torch.ops.aten.max_pool2d.<overload>` printed in the refusal itself. Reached from the embedder's `MaxPool2d` right after the stem conv+batch_norm, which is exactly what was hidden behind the `batch_norm` wall in DEMAND.md. |
| `mobilenet_v2` (`MobileNetV2ForImageClassification`) | did not forward — same `torch.batch_norm` as `resnet` | **moved past batch_norm, still does not forward** | no | n/a | **NEW WALL, different from resnet's**: `NotImplementedError: not implemented in torch._C shim: torch._C._nn.hardtanh` — MobileNetV2's ReLU6 activation (`hardtanh(0, 6)`), previously hidden behind `batch_norm`. Confirms the two models, which shared one wall in DEMAND.md, diverge onto architecture-specific walls once that shared wall is gone. |
| `convnext` (`ConvNextForImageClassification`) | did not forward (construction-time) — `torch.linspace`, `modeling_convnext.py:219` stochastic-depth schedule | **unchanged** | no | n/a | **same wall**: `NotImplementedError: not implemented in torch._C shim: torch.linspace(...) -- overload resolution has no table entry for this op`. Consistent with DEMAND.md §0.1 rank 4 — `linspace` was not one of the five closed this round. |
| `swin` (`SwinForImageClassification`) | did not construct — `torch.meshgrid`, `modeling_swin.py:354 _create_relative_position_index` | **construction now succeeds, forward moved further, still does not forward** | no | n/a | **NEW WALL**: `NotImplementedError: not implemented in torch._C shim: torch.adaptive_avg_pool1d(...) -- overload resolution has no table entry for this op`. This is the biggest jump of the round — `meshgrid` was a construction-time blocker (the model never got to run at all), and closing it took `swin` all the way through construction and most of the forward pass (patch embed, all window-attention blocks) to the final classification head's pooling. |
| `sentence_embed` (`bert` + mean-pool + `F.normalize`) | forward yes, normalize no — `torch._C._linalg.linalg_vector_norm`, `torch/nn/functional.py:6100 normalize` | **unchanged** | forward: yes (unchanged from DEMAND.md, architecturally identical to `bert`); normalize: no | n/a for normalize | **same wall**: `NotImplementedError: not implemented in torch._C shim: torch._C._linalg.linalg_vector_norm`. Consistent with DEMAND.md §0.1 rank 3 — `linalg_vector_norm` was not one of the five closed this round. |
| `whisper` `.generate()` | forward yes, generate no — `torch.as_tensor(...)`, `generation_whisper.py:1608 _retrieve_init_tokens` | **moved past `as_tensor`, generate still does not complete** | n/a (forward already established in DEMAND.md) | n/a | **NEW WALL**: `NotImplementedError: aten op not implemented in torch._C shim: aten.where.default`, reached inside the actual autoregressive decode loop (init-token construction, the previous wall, now succeeds). Checked `rust/torch_c/src/aten.rs`: `where.self` and `where.ScalarOther` are implemented, but not `where.default` — the **single-argument** form `torch.where(condition)` (upstream: equivalent to `nonzero(condition, as_tuple=True)`), a genuinely different overload from the three-argument `where` already covered, not a case the existing code should already handle. Not investigated further, per this round's implement-nothing rule. |

## 3. Headline: closing those five moved most of the queue, not nothing

**Seven of the eleven stalled targets moved.** Two closed completely and match upstream at
float32-eps scale (`t5`, `bart` — both were stopped by ops this round's predecessor round closed,
`full_like` and `new_zeros` respectively). Five moved onto **new** walls further into their own
forward/construction/generation path (`switch_transformers`, `resnet`, `mobilenet_v2`, `swin`,
`whisper.generate()`). Four are unchanged, stopped by the same wall as before, because that wall
was not one of the five DEMAND1 closed (`mbart`/`squeeze.default`, `pegasus`/legacy `Tensor`
constructor, `convnext`/`linspace`, `sentence_embed`/`linalg_vector_norm`) — this is the expected,
correct non-move, not a failure to move.

`swin` is the single largest jump: it did not even **construct** in DEMAND.md (`torch.meshgrid`
fired inside `_create_relative_position_index`, before any forward call existed to make). With
`meshgrid` closed, `swin` now constructs, runs its patch embedding, and clears every window-attention
block in both stages before stopping at the classification head's `adaptive_avg_pool1d`.

Updated combined denominator (18 architectures, this round's numbers only — ARCH26's 26 are
untouched by this round): **9 of 18 forward and match upstream** (was 7 — `t5` and `bart` join
`bert`, `roberta`, `albert`, `vit`, `clip`, `wav2vec2`, `qwen2_moe`); **2 forward but a follow-on
call does not complete** (`whisper.generate()`, `sentence_embed`'s `normalize`, unchanged count but
`whisper` moved its stopping point further in); **7 do not forward at all** (was 10 —
`switch_transformers`, `mbart`, `pegasus`, `resnet`, `mobilenet_v2`, `convnext`, `swin`).

## 4. Re-ranked open list

Ranking method is unchanged from DEMAND.md §0: by how many distinct models hit each wall as their
first blocker, ties broken by real-world reach. **Every remaining gap in this round has exactly one
vote** — the multi-vote walls (`batch_norm`, `full_like`, `new_zeros`, `meshgrid`, `as_tensor`) are
gone, and none of the five new walls found this round were hit by more than the one model that
found them (nothing else in the eleven-model set reaches MoE routing, `max_pool2d`, `hardtanh`,
`adaptive_avg_pool1d`, or the decode-loop `where.default` this round). So this round's ranking is
tie-broken by reach almost entirely — **read it as judgment, not as a vote count**, and treat it as
a starting order for the next round to confirm or overturn once more models are tried against it.

| rank | gap | model that hit it | kind | why this position |
|---|---|---|---|---|
| 1 | `aten.squeeze.default` | `mbart` | same as DEMAND.md §0.1 rank 2 | Unmoved from last round's #2. Per DEMAND1.md this is the GOLDEN.md blind-spot shape — declared in both `overloads.json` and `methods.json`, kernel logic for `squeeze()` almost certainly exists already (the `.dim` overload is implemented), so the fix is plausibly a dispatch-arm wiring gap rather than new arithmetic. `.squeeze()` with no args is also a common enough idiom that reach likely extends past `mbart` once more models are tried. |
| 4 | legacy `torch.Tensor(ndarray)` constructor | `pegasus` | same as DEMAND.md §0.1 rank 1 | Still explicitly out of scope per DEMAND1.md (`tensor.rs`'s `#[new]` slot, structural). Two distinct call shapes confirmed blocked across two rounds now (`sew_d`'s `torch.Tensor(int)`, `pegasus`'s `torch.Tensor(ndarray)`). Kept below the mechanical/spelling items above because DEMAND1 already characterized it as the one structural (not kernel-shaped) gap on the list — cost and risk are qualitatively different from the others here even though nothing here directly measures relative reach. |
| 7 | `torch.linspace` | `convnext` | same as DEMAND.md §0.1 rank 4 | Unmoved. Construction-time stochastic-depth (`drop_path_rate`) schedule, the same "N vision backbones with a drop-path schedule" pattern DEMAND.md named. |
| 8 | `aten.linalg_vector_norm.default` | `sentence_embed` | same as DEMAND.md §0.1 rank 3 | Unmoved. DEMAND1.md already measured this op's full spec in detail (six-arm `ord` family, empty-reduction split, dtype promotion) — implementation-ready, but reach is narrow (only `F.normalize`-shaped code hits it in this set). |

## Closed in DEMAND4

| rank | gap | model that hit it | kind | why this position |
|---|---|---|---|---|
| 2 | `torch.max_pool2d` (overload table entry) | `resnet` | new — construction/forward, mechanical (table entry, not kernel) | `nn.MaxPool2d` is one of the most common building blocks in classic CNNs (not just `resnet` — `alexnet`, `vgg`, `densenet`, `googlenet` families all use it), so reach is plausibly wide even on a 1-vote sample. The refusal text itself names the escape hatch (`torch.ops.aten.max_pool2d.<overload>`), suggesting the underlying kernel already exists and this is a spelling/overload-table gap like `squeeze`, not a missing kernel. |
| 3 | `aten.where.default` (single-arg `torch.where(condition)`) | `whisper` (`.generate()`) | new — missing overload, generation-path | Hit deep inside HF's shared `generate()` machinery, not whisper-specific code. Four other targets in this same eleven (`t5`, `switch_transformers`, `bart`, `mbart`, `pegasus` — all seq2seq-LM classes) expose a `.generate()` method that was never called this round because forward itself was the target; closing this is likely to matter the moment any of them is asked to generate rather than just forward, which is the more realistic use of an encoder-decoder LM. Ranked above the still-open-since-last-round items on that basis, acknowledging it is a projection, not a count. |
| 5 | `torch._C._nn.hardtanh` | `mobilenet_v2` | new — missing kernel, elementwise activation | ReLU6 (`hardtanh(0,6)`) is standard in the MobileNet family and other quantization-friendly / edge nets (ShuffleNet, EfficientNet-lite-style variants), narrower than `max_pool2d`'s reach but broader than a single model. |
| 6 | `torch.adaptive_avg_pool1d` | `swin` | new — missing overload, classification-head pooling | Reached at `SwinForImageClassification`'s final global-average-pool-over-tokens step. Plausible reach into other hierarchical/windowed vision transformers with a similar classification head; not confirmed against any other model this round. |
| 9 | `torch._C._nn.one_hot` | `switch_transformers` | new — missing kernel, MoE router dispatch | Narrowest reach of the nine: only fires inside MoE top-1 routing (Switch/Mixtral-shaped models), and none of this round's other ten targets are MoE architectures. |

**What moved position and why.** `squeeze.default` rose from #2 to #1 not because anything about it
changed, but because DEMAND.md's #1 (the legacy `Tensor` constructor) had exactly this
"promoted by attrition" property last round too, and this round it happens again one level down:
`squeeze` is now first because nothing above it has more reach *and* it is still open, the same
mechanism DEMAND.md §0.1 flagged for its own rank 1. The legacy constructor itself drops from #1 to
#4 — not because it shrank, but because it is structural while five newly-discovered items are
kernel/spelling-shaped and plausibly cheaper, so reach-and-cost judgment now outranks it rather than
default-by-elimination. `linspace` and `linalg_vector_norm` hold their relative order from
DEMAND.md's #4 and #3 (now #7 and #8) since nothing about them changed and five new items were
judged to have comparable or greater reach.

## 5. Same scope as DEMAND.md §3 — kept for comparability, not re-litigated

Unchanged from DEMAND.md: self-built toy `AutoConfig`s (small hidden size, 1-2 layers, few heads,
tiny vocab), hand-built tensors in place of real image/audio preprocessing, no Hub checkpoints, no
`sentence-transformers` package, no DETR-shaped object detection, no `float8_e4m3fn`/exotic dtypes.
One addition this round, not a scope change: text-model inputs are fixed hand-built token-id lists
rather than `torch.randint` output, per §0's finding that `randint` itself diverges from upstream
after an identical seed — the same "hand-built tensors" principle DEMAND.md already applied to
vision/audio, extended to text once the reason to extend it was found.

## 6. Gates — tree unchanged, pasted in full

```
$ PYTHON=$PY sh rust/torch_c/pytests/run.sh
... smoke_ok = 348 (claim: ge 339)
DOCWATCH: PASS -- 274/274 evaluated marker(s) hold

$ $PY tools/golden/compare.py
SUMMARY: 8126/8126 cases passed, 0 failed, ops covered=185, pending case builders=1

$ $PY tools/golden/compare.py --self-test
SELF-TEST: PASS -- 21 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised

$ $PY rust/torch_c/pytests/verify_schemas.py
SUMMARY: 4574/4574 table entries matched upstream, 0 failed
```

All four numbers match the round's expected baseline exactly (348 ok, DOCWATCH 274/274, 8126/8126
ops=185). `git status --short` was empty before, during, and after this round except for this file
itself — nothing in `rust/torch_c/src/`, `bootstrap.py`, `tools/golden/cases.py`, or the vendored
tree moved.

---

## The method finding, promoted — `randint` is a silent divergence

Found while debugging this sweep's own first result, and it is worth more than any
row in the table above. `t5`'s weights were bit-identical to upstream and its
output was off by whole units; the cause was the harness's own randomly drawn
token ids differing between the two sides.

Re-measured directly, no model involved, `manual_seed(1234)` before each:

```
                upstream                        shim
randn(4)        [0.04613, 0.402403, …]          identical
rand(4)         [0.028979, 0.401899, …]         identical
randint(0,100)  [75, 71, 6, 65, 16, 64]         [96, 17, 58, 14, 11, 20]
randperm(6)     [3, 2, 4, 5, 1, 0]              not implemented
```

**`randn` and `rand` reproduce upstream and `randint` does not**, with no error and
no warning. That is the shape this project treats as worst: a wrong answer that
announces nothing. It also means every seeded comparison that draws integers --
token ids, indices, masks -- has been comparing two different inputs, and any
that passed did so because the values did not reach the output.

Not fixed here; this round implements nothing. It outranks the table above,
because the table's own credibility depends on it.

<!-- DOCWATCH: op-implemented aten.randint.low -->
