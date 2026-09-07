# AGREE — the caveat, measured: 284 of 285 architectures that forward also match

Worktree `work/agree` on develop `b33e2ee` (vendored tree assembled fresh). torch 2.13.0 upstream
(`/Volumes/macMini/caches/spike-venv/bin/python`) is the oracle throughout. No Rust,
`bootstrap.py` or `aten.rs` was changed in this round: it adds
`rust/torch_c/pytests/agree_sweep.py`, `rust/torch_c/pytests/test_agree.py` and this document.
Golden stays at **11336/11336, ops=299** — exactly unmoved, which is the correct result for a
round that changed no kernel.

**This is a snapshot of this checkout, not of develop tonight.** Three other agents were landing
operators in parallel worktrees while this ran. Every number below is the tree as checked out, at
one point in time.

---

## 1. The headline

`docs/architectures/ARCH100.md` and `docs/architectures/ARCH200.md` both end on the same sentence, and neither acted on it:

> **A forward is not a match** — this sweep measures reachability, not numerics.

290 of 297 upstream-clean architectures forward. Until now **26** of them had ever been checked
for agreement — the twenty of `docs/architectures/ARCH20.md` plus the six of `docs/architectures/ARCH26.md`. This round ran
all 290 with the same weights and the same inputs on both sides and compared element-wise.

```text
architectures attempted                       528   (every model type AutoModel can build)
  forward on UPSTREAM torch (the denominator) 297   <- identical to ARCH100 and ARCH200
  replayed under the SHIM on those weights    290   <- 98%, matching the reachability figure

of the 290, after excluding what cannot be judged:
  exact  (bit-identical)                        0
  agree  (within 1.19e-06 relative)           258
  agree_within_float32                         26   <- outside it, but no further from the
                                                       float64 truth than upstream itself is
  DIVERGE                                       1
  degenerate    (output underflowed)            3   } not counted either way
  nondeterministic (upstream re-samples)        2   }

  284 of 285 judgeable architectures AGREE  (99.6%)
```

The 297 landing exactly on ARCH200's 297 is the control that this is the same population, not a
different one: upstream was not touched, so it cannot report a different set of buildable,
forwardable architectures. The 290 replaying is the same 290 the reachability figure names.

**And the attribution result, which is the part that surprised:** *no divergence in this
population attributes to an operator.* §4 replays every leaf module of the seven largest on
upstream's own recorded input, so nothing accumulates, and the worst single-op error found
anywhere is **8.4 ulp** (a `Conv2d` in `hgnet_v2`); most are 2 ulp. Every large end-to-end number
in §3 is float32 accumulation over depth, exactly as `docs/architectures/DEMAND8.md` §1.5 concluded for
`mobilenet_v2` alone — which this round reproduces independently, at 1.9 ulp worst-op and
`last_hidden_state` scale 6.0, from a different harness.

## 2. The tolerance, and why it is not a choice

A tolerance picked by eye is not a result, so this one is read off **upstream's own float32 error
distribution**. Every architecture was additionally run upstream in **float64** and both float32
answers scored against it — `docs/architectures/DEMAND8.md` §1.4's control, generalised from one model to 263.

```text
upstream's OWN float32-vs-float64 relative error, 263 architectures with a working oracle:
    median 2.74e-07    p90 1.19e-06    p99 6.07e-06    max 6.08e-04
float32 eps                                                        1.192e-07
TOLERANCE USED (relative, against the tensor's own scale)          1.186e-06  = 10 ulp
```

The number is the **p90 of that distribution, floored at 8 ulp**. Its defence is that anything
tighter fails upstream against float64 on a tenth of the same set: a threshold that would have to
call upstream wrong is not measuring the shim. The floor exists so that a population which
happened to be numerically easy could not drive the tolerance below a few ulp, where LayerNorm and
GELU differ for reasons `docs/devices/MPSFWD.md` already recorded.

Above the tolerance there is a second, per-architecture rule rather than a bigger constant: a
difference is still not a defect if it is **within 4× upstream's own distance from the float64
truth for that same output**. `docs/architectures/DEMAND8.md` measured that ratio at ~1.2 for `mobilenet_v2`
(1.28e-04 apart, upstream itself 1.06e-04 from the exact answer); 4 leaves room for it and refuses
an order of magnitude. That factor and its derivation are pinned by
`test_the_oracle_factor_is_stated_and_is_not_a_free_parameter`, so widening it to make a number
look better is a failing test.

Two outcomes are **refusals rather than passes**, and they are excluded from the denominator:

* `degenerate` — the output underflowed (`efficientnet` 8.0e-29, `sam_vision_model` and
  `sam_hq_vision_model` 8.0e-21). Both sides agree on noise; that is not agreement.
* `nondeterministic` — **upstream does not reproduce itself.** Every architecture was forwarded
  twice upstream; `vit_mae` re-draws its patch mask (self-repeat 1.36) and `vits` samples a
  duration (self-repeat `inf`, different output length). There is no fixed answer to match, so
  any shim-vs-upstream number about them is measuring a sampler. 288 of 290 were bit-identical to
  themselves.

22 architectures have **no float64 oracle**, so only the fixed tolerance applies to them. The
reason is upstream's own, not ours, and is worth recording: the MoE families refuse `Double` at
their grouped matmul (`Expected mat_a to be Float32, BFloat16 or Float16 matrix, got Double` —
`qwen3_moe`, `qwen2_moe`, `glm4_moe`, `nemotron_h`, `zaya`, …), `xglm` overflows converting its
mask sentinel, `mra` mixes `Float` and `Double`. All 22 land in `agree`, none near the tolerance.

## 3. The ranked list — the deliverable

`rel` is `max|shim − upstream| / max|upstream|`. `ratio` is `rel` divided by upstream's own
float32-vs-float64 error on that same output: **1.0 means the shim is exactly as far from the
truth as upstream is**, which is not a defect but a restatement of what float32 costs.

```text
        rel         abs      scale  up f32-f64  ratio  verdict     architecture [output]
  7.625e-04   9.769e-02  1.281e+02    6.08e-04   1.25  agree_f32   hgnet_v2 [feature_maps.0]
  1.587e-04   9.522e-04  6.000e+00    9.85e-05   1.61  agree_f32   mobilenet_v2 [last_hidden_state]
  3.735e-05   2.241e-04  6.000e+00    4.28e-05   0.87  agree_f32   mobilenet_v1 [last_hidden_state]
  5.378e-06   1.378e-06  2.563e-01    2.72e-06   1.98  agree_f32   yolos [pooler_output]
  4.657e-06   1.270e-05  2.726e+00    4.11e-06   1.13  agree_f32   wav2vec2-conformer
  4.319e-06   1.514e-05  3.505e+00    6.07e-06   0.71  agree_f32   data2vec-audio
  3.543e-06   9.738e-06  2.748e+00    2.90e-06   1.22  agree_f32   gemma3n_text
  3.255e-06   3.815e-06  1.172e+00    2.03e-06   1.60  agree_f32   resnet [pooler_output]
  3.135e-06   1.451e-06  4.628e-01    2.45e-06   1.28  agree_f32   groupvit [image_embeds]
  2.873e-06   1.401e-06  4.876e-01    6.00e-07   4.79  DIVERGE     chinese_clip [logits_per_image]
  2.354e-06   6.378e-06  2.709e+00    2.75e-06   0.86  agree_f32   unispeech / unispeech-sat / wav2vec2
  2.293e-06   6.199e-06  2.704e+00    3.31e-06   0.69  agree_f32   wavlm
  2.261e-06   7.033e-06  3.111e+00    1.87e-06   1.21  agree_f32   hubert
```

The shape of that column is the finding. Of the 27 architectures above the tolerance, the `ratio`
sits between 0.43 and 1.98 for 26 of them — the shim is sometimes closer to the truth than
upstream and sometimes further, by less than a factor of two either way, which is what two
independent float32 truncation paths look like. Nine of the 26 have a ratio **below 1**: on those
outputs the shim's float32 answer is *nearer the float64 one* than upstream's is.

### 3.1 The one `diverge`, and why it should not be read as a defect

`chinese_clip` [`logits_per_image`] is the single architecture outside both rules, at ratio 4.79.
It is on the list because the rule is mechanical and it must be, but three measurements say what
it is:

* the **absolute** difference is **1.4e-06** on a tensor of scale 0.49 — twelve ulp;
* the per-module replay (§4) finds **no operator above 2.5 ulp** in the whole model, the largest
  being a `GELUActivation` at 2.93e-07 and the patch-embedding `Conv2d` at 2.89e-07;
* it crossed the 4× rule because **upstream's own oracle error on that output is unusually
  small** (6.00e-07, five ulp) — `logits_per_image` is a dot product of two L2-normalised
  embeddings times a learned scale, and normalisation removes most of the accumulated magnitude
  error from the numerator and the denominator alike. A small denominator makes the ratio large;
  it does not make the numerator large.

**Verdict: not attributed to any operator, and no defect found.** The honest statement is that
the mechanical rule flags one architecture out of 285 and the flag does not survive inspection —
recorded here rather than quietly reclassified, because a rule that is edited whenever it fires
is not a rule.

### 3.2 Two architectures whose numbers are large and whose cause is depth

`hgnet_v2` at 7.6e-04 and `mobilenet_v2` at 1.6e-04 are the two biggest numbers in the sweep, and
both are deep convnets whose per-op errors are 8.4 and 1.9 ulp respectively. `mobilenet_v2` is the
model `docs/architectures/DEMAND7.md` §3 ranked as a "correctness bug" and `docs/architectures/DEMAND8.md` struck; this round
reproduces DEMAND8's result from a different harness and different random weights —
`last_hidden_state` scale 6.0, absolute difference 9.5e-04, worst leaf module a `BatchNorm2d` at
2.24e-07. `hgnet_v2`'s `feature_maps.0` has scale 128, so its 9.8e-02 absolute is the same
relative story at a larger scale.

## 4. Attribution: replaying every leaf module on upstream's own input

`docs/architectures/DEMAND8.md` §1.3's technique, reused. Upstream records the input and output of every leaf
module; the shim then re-runs **each module in isolation on upstream's recorded input**, so
nothing accumulates and any error that appears belongs to that module.

```text
architecture   leaves  worst single module                                    rel        ulp
hgnet_v2          324  Conv2d   encoder.stages.1.blocks.0.layers.2.conv    1.001e-06    8.4
groupvit          110  BatchNorm1d  text_projection.1                      6.393e-07    5.4
resnet             41  Conv2d   embedder.embedder.convolution              4.407e-07    3.7
chinese_clip       51  GELUActivation  text_model...intermediate_act_fn    2.933e-07    2.5
yolos              27  GELUActivation  encoder.layer.0...intermediate      2.524e-07    2.1
mobilenet_v1       82  Conv2d   layer.12.convolution                       2.323e-07    1.9
mobilenet_v2      140  BatchNorm2d  layer.7.conv_3x3.normalization         2.241e-07    1.9
```

Seven of the largest end-to-end differences, and **not one has an operator behind it.** The worst
single kernel invocation anywhere in 775 replayed leaf modules is 8.4 ulp; the median top-row is
2 ulp. `Conv2d`, `BatchNorm2d`, `BatchNorm1d`, `LayerNorm` and `GELUActivation` are what fills
these tables, and they are the same ops `docs/architectures/DEMAND8.md` §1.3 found at 2.4 ulp and
`docs/architectures/DEMAND1.md` predicts for a differently-associated but equally valid arrangement of the same
formula.

**So the ranked list in §3 is a ranking of depth, not of defects.** That is a real answer to the
question the sweep was asked — but it is a *negative* result about operators, and the technique
that would have found a positive one is the same technique, so its silence is informative.

**What §4 could not have caught**, since CLAUDE.md §5.4 asks for it: an operator that is wrong in
a way both the composed and the isolated run share (the replay uses the shim's own kernel in
both), a defect that only fires at shapes outside these configs, and — the big one — an operator
that is wrong only in a *fused* or *composed* path that no leaf module isolates.

## 5. Same weights, and how the seeding was done

The two sides must run the same numbers or the experiment is about initialisation. Two mechanisms,
and the weaker one is deliberately not relied on:

* **Weights and inputs travel as bytes.** The upstream side serialises its `state_dict` and its
  input dict to `.npz`; the shim side loads them. `state_dict` transferred cleanly for **290 of
  290** — no missing and no unexpected key anywhere, which is also a check that the same
  `transformers` builds the same module tree on both sides. The transport is `tolist()` out and
  `torch.as_tensor()` in, because `Tensor.numpy` and `torch.from_numpy` are both absent on the
  shim; `test_agree.py` round-trips every dtype through it, including a 0-dim and an empty tensor,
  whose shapes `tolist()` cannot reconstruct on its own. `int8` cannot be transported at all —
  candle will not store it — and is asserted as the only such dtype. No architecture carried one.

* **The RNG claim, verified rather than assumed.** `agree_sweep.py --rng-check` run on both sides
  at seed 0 gives **byte-identical** values across `randn`, `rand`, `randint`, `randperm`,
  `normal_` and `uniform_`. So the claim holds on this checkout. Upstream's values are frozen into
  `test_agree.py` so that a future drift fails a test rather than silently turning some other
  round's same-weights experiment into a comparison of two initialisations.

Everything is seeded at **`torch.manual_seed(0)` before `AutoConfig`/`AutoModel`**, on both sides,
and the calibration batch of §6 at 1234.

## 6. Two harness artefacts that were found and removed, because both looked like defects

Recorded at length because each produced a large, plausible, entirely false number, and because
§5.5 of CLAUDE.md is about verification that cannot fail.

**BatchNorm running statistics.** Fresh `BatchNorm` has `running_var=1`, and `docs/architectures/DEMAND8.md`
§1.1 measured that this drives a deep convnet's activations to ~1e-23. It does here too:
`mobilenet_v2`'s `last_hidden_state` came out at scale **3.9e-22** before calibration. So the
producer runs one forward in **train mode at `momentum=1.0`**, exactly as DEMAND8 did — and
because running stats live in the `state_dict`, the shim receives them by loading the same bytes.
*No calibration is repeated on the shim side, so no calibration can diverge.* `mobilenet_v2`'s
scale went to **6.0**, the same figure DEMAND8 reports.

**The calibration batch's rows must differ, and it took two rounds to get right.** A batch of one
is refused by `BatchNorm` in train mode (`mobilenet_v2`'s last block is `(1, 576, 1, 1)`), so the
input is repeated four times. Repeating it *verbatim* gives a batch variance of exactly zero:

* on the float side `rsqrt(0 + eps)` turns every BatchNorm into a ~1/sqrt(eps) amplifier, and
  `mobilenet_v2`'s **upstream-vs-float64** error read **4.5e-01** — upstream disagreeing with
  itself, which is the tell;
* on an integer-fed branch the layer sees `x == mean` and collapses to its bias. `groupvit`'s
  `text_projection` BatchNorm1d emitted scale **5.3e-05** from a 3.9e-01 input, the L2
  normalisation downstream then amplified the direction of a near-zero vector, and `groupvit`
  reported an end-to-end divergence of **5.8e-01**. It is **3.1e-06** once the rows differ —
  a factor of 185,000, all of it the harness.

Float rows now get fresh noise and integer rows are resampled inside their observed range; boolean
masks are still repeated, since a random mask is a different sequence length rather than more
variance. `calibration_batch` is a separate function for exactly this reason and
`test_the_calibration_batch_rows_differ_on_both_float_and_integer_inputs` asserts the batch
variance is non-zero on both branches.

**The module capture must clone.** The bisection bundle is serialised after the forward finishes,
so a residual `+=` or an in-place activation rewrites a referenced tensor before it is written —
and the replay then runs on an input upstream never saw. Measured: without the clone, `groupvit`'s
`vision_model.encoder.stages.1.downsample.assign.proj`, a plain `nn.Linear`, reported a relative
error of **1.0**, and `visual_projection.1` likewise. Both fall to ≤6.4e-07 with the clone. No
value assertion can catch this from outside the harness, so it is pinned at the source by
`test_the_module_capture_clones_rather_than_referencing`.

## 7. Limits — `docs/architectures/ARCH200.md` §5's, restated, plus this round's own

Inherited, and they apply here in full:

* **`AutoModel` bodies only.** No task heads, no generation loop, no KV-cache reuse, no training
  step. An architecture that agrees here has not been shown to agree under `generate`.
* **Shrunk configs and random weights.** Random weights reach the same *operators*
  (`arch_sweep.py --verify-random-weights` is the check for that claim) but they do **not** reach
  data-*dependent* branches. A model that takes a different path on trained weights was not
  measured on that path.
* **One checkout, one moment.** Three other agents were landing operators while this ran.

New to this round:

* **Single-output-tensor summary.** An architecture's score is the worst relative error across
  every tensor in its output, and one number per architecture is what the ranking sorts. A model
  agreeing on `last_hidden_state` and disagreeing on a small auxiliary head is reported at the
  larger number, which is right, but the ranking does not say how much of the output disagreed.
* **The oracle is upstream's float64, not exact arithmetic.** Where upstream's own float64 path
  shares an error with its float32 path, this cannot see it. It is a control against *rounding*,
  not against a wrong formula both of upstream's dtypes implement.
* **`degenerate` and `nondeterministic` are 5 architectures that remain unmeasured**, not
  architectures shown to agree. `vits` in particular is a real gap: it is a generative audio model
  and the sampler is the interesting part.
* **The 7 that forward upstream and not under the shim are unchanged and unmeasured here**:
  `fastspeech2_conformer` (`repeat_interleave` with a tensor `repeats`), `led` and `longformer`
  (`new_zeros`), `nystromformer` and `sam3_lite_text_text_model` (asymmetric convolution padding),
  `univnet` (`TensorBase.unfold`), `vilt` (`multinomial`).

## 8. How to take this round again

Not wired into `run.sh` and must not be: it needs both interpreters and takes hours. The upstream
side runs **first** — it is the producer of weights, inputs and the float64 oracle.

```text
PY=/Volumes/macMini/caches/spike-venv/bin/python
D=/Volumes/macMini/tmp-agree                 # NOT /tmp: ~8 GB of bundles, and /tmp is internal
cd rust/torch_c/pytests

env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL $PY agree_sweep.py --produce --dir $D
PYTHONPATH=$REPO/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
                                           $PY agree_sweep.py --replay  --dir $D
$PY agree_sweep.py --report --dir $D

# attribute one architecture to an operator: upstream captures, the shim replays
env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL $PY agree_sweep.py --capture ARCH --dir $D
PYTHONPATH=... TORCH_USE_RTLD_GLOBAL=1     $PY agree_sweep.py --bisect  ARCH --dir $D
```

Both drivers are **resumable**: each architecture's record is written the moment it finishes and a
re-run skips what is already on disk, so `--deadline 500` can be issued repeatedly in the
foreground rather than backgrounded and lost. That is not a convenience — CLAUDE.md records three
rounds lost to backgrounding a sweep, and a driver that must start over is a driver that gets
backgrounded.

## 9. Gates

```text
rust/torch_c/pytests/run.sh    884 ok, 0 FAIL, exit 0        (868 before; +16 from test_agree.py)
DOCWATCH                       PASS -- 785/785 evaluated marker(s) hold
tools/golden/compare.py        11336/11336 cases passed, 0 failed, ops covered=299, pending=0
```

Golden is **exactly unmoved** from this worktree's starting point, which is the correct result:
this round changed no Rust. All three measured on the freshly built `lib_C.dylib` in this
worktree, with `TORCH_C_ARTEFACT` set explicitly — an unset one falls back to the shared cache
binary and reports a plausible but wrong number (`docs/architectures/ARCH100.md` §8).

`HF_HOME=/tmp/hf-agree` was set and removed afterward; no checkpoint was downloaded, since this
sweep builds every model from a config.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/agree_sweep.py calibration_batch present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/agree_sweep.py diff_stats present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/agree_sweep.py verdict present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/agree_sweep.py bisect_one present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_agree.py test_the_shim_reproduces_upstreams_seeded_random_numbers present -->
<!-- DOCWATCH: count golden_cases_passed ge 11336 -->
<!-- DOCWATCH: count golden_ops_covered ge 299 -->
<!-- DOCWATCH: count golden_pending eq 0 -->
