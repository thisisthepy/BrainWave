# W13: convolution's backward, and a model that trains through it

`docs/platform/RELEASE_0_0_13a0.md` §5 states two of this project's own gaps in one sentence:

> **No transformer trains through `loss.backward()` yet.** ... There is no convolution backward
> rule, so vision models stop.

They are two different facts and only one of them was true. This document is both halves measured,
and the rules that closed the true one.

Environment: worktree `work/train` on `develop` `da9e3bf`, torch 2.13.0,
`/Volumes/macMini/caches/spike-venv/bin/python`,
`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-train`. Every shim reading printed `shim`;
every upstream reading was taken with `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`. Four agents
were on the machine, so **no time is reported anywhere below**.

---

## 0. Answers, before the evidence

| question | answer |
|---|---|
| **Which rules were actually missing?** | Four, and the release note names one of them. `aten.convolution.default`, `aten.avg_pool2d.default`, `aten.adaptive_avg_pool2d.default` and `aten.max_pool2d.default` — found by running the models rather than by reading `RULE_OPS`, which is how the fourth one turned up. §1 |
| **Which landed?** | The first three. **`max_pool2d` did not**, and it is refused by name with its size in §5 — it needs a forward that returns indices, which this shim does not have. |
| **Does a vision model train end to end?** | **Yes.** A depthwise-separable convolutional network — strided conv, depthwise conv, pointwise conv, three training-mode batch norms, global average pool, linear head — five steps of `backward` / `step` / `zero_grad` with a real `torch.optim.SGD`. Against upstream running the identical program: losses **2.98e-08**, first-step gradients **1.08e-07** over 414 floats, final parameters **2.98e-08**, batch-norm running buffers **7.45e-09**. §3 |
| **Does a transformer train?** | **Yes, and it did before this round started.** The release note's first clause was already false when it was written. A tiny attention LM (embedding, pre-norm block, causal SDPA, GELU MLP, `cross_entropy`, SGD) trains for five steps at **8.94e-08** against upstream over 872 parameter floats, using **no rule this round added**. §4 |
| **What did the finite-difference oracle miss?** | **The weight gradient, entirely.** `test_shim.py`'s float64 central differences differentiate one input, so exchanging `stride` and `dilation` in the weight-gradient convolution leaves them **green**. Only the upstream comparison goes red. §6, nullification N2 |
| **What did the vision loop miss?** | **The zero tail.** Nullification N6 leaves the whole training loop green and is caught only by a gradient case with a stride that does not divide its input. §6 |

---

## 1. Which rules were missing, and how that was established

Not by reading `RULE_OPS`. By running eight programs and reading the refusals — which is the only
method that finds a rule nobody thought to look for, and it found two:

```text
conv2d bare / stride 2 / groups / conv1d   no derivative rule for aten.convolution.default
maxpool                                    no derivative rule for aten.max_pool2d.default
avgpool                                    no derivative rule for aten.avg_pool2d.default
adaptive avg pool                          no derivative rule for aten.adaptive_avg_pool2d.default
flatten + linear                           OK
embedding + layer norm                     OK
a transformer block (SDPA, GELU, 2 norms)  OK, 16/16 parameters
a tiny transformer LM + cross entropy      OK, 19/19 parameters
```

The last two are §4. The point of running them was that the release note's sentence bundles them
with convolution, and a round that assumed the sentence was one fact would have implemented
convolution and then *claimed* the transformer as a consequence of it.

`docs/training/BACKWARD8.md` §1 had already met this from the side: its `Conv2d + BatchNorm2d` model hit "a
**third** wall — there is no `aten.convolution.default` derivative rule either" and was reduced to a
`Linear + BatchNorm1d` stack to get past it. That reduction is what this round undoes.

## 2. The convolution rule: three gradients, no kernel

Upstream's `convolution_backward` is three convolutions, and this shim already has the forward one,
its transposed sibling, and `groups`. So the rule is a composition through the same door
(`aten_dispatch`) the forward used, which is the property `tape.rs`'s header buys and the reason a
derivative here cannot disagree with the forward about what a convolution *is*.

| gradient | spelled as |
|---|---|
| **input** | a **transposed** convolution of the output gradient with the same weight, one group at a time |
| **weight** | a forward convolution of the input against the output gradient with **batch and channel axes swapped on both**, and with **`stride` and `dilation` exchanged** |
| **bias** | the sum over every axis but the channel one |

Three things in that table are not obvious, and each is a defect this round actually made and
measured before fixing.

### 2.1 `padding` is not passed to the transposed convolution — it is a crop

A transposed convolution's `padding` argument crops its scatter by `p` on *both* sides. The far side
of that crop is not always outside the input: with `stride > 1` the last input position can be read
by the last window and still fall past the cropped result, so it comes back **missing rather than
zero**.

Measured, on the first version of this rule, which passed `padding` through:

```text
2d_groups2_stride2   gx   relative 1.000e+00      <-- a whole row zero where upstream had values
1d_stride3_pad2_dil2 gx   relative 6.131e-01
every other case     gx   relative <= 2.9e-07
```

Every `padding = 0` case and every `stride = 1` case in the same table agreed. **A test suite whose
convolution cases did not have one with both would have shipped this**, and the shapes would all
have been right.

The fix is not `output_padding` either, though `output_padding` is exactly the right size
(`(H + 2p - d(k-1) - 1) mod s`, always `< stride`). It cannot be used here because the two spatial
axes can need *different* amounts and `aten.rs` refuses an asymmetric `output_padding`. So the
scatter is taken whole — `padding = 0` — and the window `[p, p + extent)` is **sliced** out of it,
which is what `padding` means on this side and works per axis. What falls past the end of the
scatter is padded with zeros, and those zeros are a fact rather than a truncation: no window reaches
those positions.

### 2.2 The weight gradient exchanges `stride` and `dilation`

```text
gw[co,ci,u,v] = sum_{n,y,x} gout[n,co,y,x] * x[n,ci, y*s - p + u*d, x*s - p + v*d]
```

As a convolution over `u`, the step through `x` is **`dilation`** and the step through `gout` is
**`stride`**. Every case with `stride == dilation` — which is every case anybody writes first —
passes with them either way round.

### 2.3 `groups` is decomposed, never passed

Both gradients are formed one group at a time and concatenated. `groups` never reaches a kernel from
this rule, for two independent reasons: `aten.rs`'s 2-D transposed convolution takes no `groups`
argument at all (candle's `conv_transpose2d` has no field for one), and the weight-gradient trick's
grouping is *not* the forward's — its channel axis is the batch. A rule that forwarded `groups`
would be refused by one call and silently wrong in the other.

### 2.4 What it refuses, by name

* **a transposed convolution** (`transposed=True`). Its gradient is a *forward* convolution and the
  roles of `padding` and `output_padding` swap with it. `test_a_transposed_convolution_is_refused_by_name_rather_than_differentiated`
  is a live `nn.ConvTranspose2d` that refuses, not a source grep.
* **an asymmetric `padding`.** The forward lowers one by pre-padding its input; there is no input to
  pre-pad on this side. (Asymmetric `stride` and `dilation` cannot reach here — the forward refuses
  them.) `nystromformer`'s `padding=[k//2, 0]` is the caller that would want it.
* **rank other than 3 or 4**, which is the forward's own bound.

### 2.5 The measurement

Fourteen configurations, each of the three gradients and the forward, against upstream running the
identical program. Worst **4.56e-07** relative across all of it, with the forward in the same table
at **2.14e-07** — so the backward loses nothing the forward has not already lost.

```text
2d_plain  2d_stride2  2d_stride2_ragged  2d_pad1  2d_dil2  2d_stride2_pad2_dil2
2d_even_kernel  2d_groups2  2d_depthwise  2d_groups2_stride2  2d_nobias
1d_plain  1d_stride3_pad2_dil2  1d_depthwise_causal
```

`2d_even_kernel` is there because `docs/kernels/RNN.md` measured that `lasr`'s configuration takes an
**even** kernel where its own comment says odd. `1d_depthwise_causal` is `mamba`'s shape.
`2d_stride2_ragged` has a stride that does not divide its input, and §6's N6 is the nullification
only it catches.

## 3. Average pooling, for the windows that tile

`avg_pool2d` and `adaptive_avg_pool2d` are one rule. For a **non-overlapping** window the gradient
is a scatter with no arithmetic in it: every input position belongs to exactly one window and
receives that window's gradient divided by the window's area. Spelled in `reshape`, `expand`,
`reshape` and one `div.Scalar` — so it agrees with upstream **exactly, 0.0**, on every tiling case,
and the test asserts `== 0.0` rather than a bound. A bound there would stop noticing if the rule
grew arithmetic it does not need.

What refuses, by name: an **overlapping** window (`stride < kernel`), which makes the scatter an
accumulation and is a different rule rather than this one with a different constant; a padded one,
whose divisor depends on `count_include_pad` and differs per cell; `ceil_mode`; a
`divisor_override`; and a window that does not divide the input — which for `adaptive_avg_pool2d` is
the ragged case where upstream's own windows have different sizes and overlap.

`test_the_pooling_windows_that_do_not_tile_are_refused_by_name` pins both directions: that they
refuse here **and** that upstream computes them. Without the second half the refusal might not be a
gap at all.

## 4. The transformer half of the release note was already false

The sentence "no transformer trains through `loss.backward()` yet" was written beside the
convolution gap, and only the convolution half was true. A tiny attention LM —
`nn.Embedding`, a pre-norm block with **causal** `scaled_dot_product_attention`, a GELU MLP, a
final layer norm, a linear head, `F.cross_entropy`, real `torch.optim.SGD`, five steps — trains
here with **no rule this round added** anywhere in its path. Every rule it needs
(`_scaled_dot_product_flash_attention_for_cpu`, `native_layer_norm`, `embedding`, `gelu`,
`nll_loss_forward`, `_log_softmax`) was already in `RULE_OPS`.

| quantity | elements | worst relative vs upstream |
|---|---:|---|
| loss trajectory (2.862 → 2.425, monotone) | 5 | `8.33e-08` |
| every parameter gradient, step 0 | 872 | `3.35e-08` |
| every parameter, after five `optimizer.step()`s | 872 | `8.94e-08` |

This is a test rather than a correction in prose because "a transformer trains" is exactly the shape
of claim `docs/verification/AUDIT.md` found going stale — and it had gone stale in the *conservative* direction,
which is the direction nobody re-checks.

**What is still not established** is a transformer with pretrained weights at real width.
`docs/training/BACKWARD.md` §6 is why that is a separate problem and not a bigger version of this one: on a
real model the finite-difference oracle disagrees with **upstream's own autograd** by a factor of
600, so the oracle that would scale to it does not exist yet. The central oracle used here does
scale — it is two interpreters running one program — and running SmolLM2-135M through five SGD steps
in both is the obvious next measurement. It is not in this round.

## 5. What is refused, sized

**`aten.max_pool2d.default`.** A max pool's gradient routes each output's gradient to the
**argmax** of its window, so the rule needs the indices. Upstream's forward returns them —
`max_pool2d_with_indices` — and this shim's does not: `_aten_implemented()` carries
`aten.max_pool2d.default` and not `aten.max_pool2d_with_indices.default`, which
`test_max_pool2d_backward_is_still_refused_by_name` asserts rather than describes, so this paragraph
cannot go stale.

That makes it two pieces of work: a forward that also returns indices (`aten.rs`), and a scatter
rule over them (`tape.rs`). Recomputing the argmax inside the rule would be a **third**
implementation of the window geometry, which is the shape `docs/verification/AUDIT.md` keeps finding go stale, so
it is not the way in. Sized at one round, mostly in the forward.

Also refused, and listed here so the total is honest: transposed convolution's gradient, asymmetric
convolution padding, overlapping average pooling, padded average pooling, `ceil_mode`,
`divisor_override`, ragged adaptive pooling.

## 6. Nullification

Six faults, each built into the rule, rebuilt, and run. The column that matters is the last one:
**two of the six leave the finite-difference oracle completely green**, and one leaves the entire
training loop green.

| | fault | test_train.py | float64 central differences |
|---|---|---|---|
| **N1** | `padding` passed to the transposed convolution | conv table (`gx`, relative **1.0**), vision loop | **red** |
| **N2** | `stride`/`dilation` left in the forward order in the weight gradient | conv table, `..._needs_stride_and_dilation_exchanged`, vision loop | **GREEN** |
| **N3** | bias gradient reduces the batch axis instead of keeping the channel one | conv table, vision loop, accumulation test | **GREEN** |
| **N4** | `groups` not decomposed (one pass over every channel) | conv table, vision loop, accumulation test | red |
| **N5** | average pool's gradient not divided by the window area | pooling table, vision loop | red |
| **N6** | the zero tail past the last window not restored | **conv table only** | **GREEN** |

**N2 is the round's sharpest result.** The finite-difference oracle is the one `docs/training/BACKWARD.md`
built and it is a good oracle — but its cases differentiate *one input*, so it never sees a weight
gradient at all. A rule with `stride` and `dilation` the wrong way round is invisible to it and
visible only to upstream. That is `docs/training/BACKWARD8.md`'s finding with the sides swapped: there the FD
oracle could not see a training-mode batch norm because capture refuses that mode; here it cannot
see two of three gradients because the oracle's shape has one input.

**N6 is the second.** The training loop stayed green — every layer in the vision model has a stride
that divides its input — and only the `2d_stride2_ragged` and `2d_stride2` cases in the conv table
turned red. A round that had measured only "does a model train" would have shipped it.

**Accumulation.** `docs/training/BACKWARD9.md`'s finding is repeated rather than assumed:
`zero_grad(set_to_none=True)` drops `.grad` every step, so a loop test cannot observe accumulation
and a `+=` degraded to `=` stays green through it. `test_the_training_loop_actually_accumulates_into_dot_grad`
takes two backwards of the same loss with no `zero_grad` between them and asserts every gradient is
**exactly twice** the first — and asserts that at least 50 of them were large enough for the ratio
to mean anything, because a test whose ratios were all `None` would pass against a rule that never
accumulated.

## 7. The forward did not move

`docs/training/BACKWARD7.md` §9.1's three SmolLM2-135M logits digests, reproduced rather than copied: the
script was written from `docs/training/BACKWARD6.md` §7's recipe — sha256 over
`struct.pack("<%df" % len(row), *row)` for each row of `logits.tolist()[0]`, real
`HuggingFaceTB/SmolLM2-135M`, float32, greedy prefill of `torch.arange(S).unsqueeze(0)` — and run
against this build.

| S | sha256, this build | |
|---:|---|:--:|
| 6 | `606e00d1be05fccce84b23b02dae8f473d490ad0f706b9b612036dd55cee531b` | ✅ |
| 32 | `62d923807cce47cee35a5e7b6091c9544e180c6c9ff771bb4cad3f478f4cbb79` | ✅ |
| 128 | `4750428e94f7383c42853b0d547192a7946fdde9e086d3b16cebdeb2ca68e37d` | ✅ |

All three equal `docs/training/BACKWARD7.md` §9.1's, `docs/training/BACKWARD6.md` §7's, `docs/training/BACKWARD5.md` §6.5.1's
and `docs/training/BACKWARD4.md` §6.2's. A derivative rule adds no forward op, and the golden harness holds
the same line from the other side — **11385/11385, ops 300**, unmoved by this round.

## 8. Gates

| | |
|---|---|
| suite | **939 ok**, 0 FAIL, EXIT=0 (929 before; +10, all of them `pytests/test_train.py`) |
| DOCWATCH | **PASS — 837/837** (830 before; +7, all in this document) |
| golden | **11385/11385**, ops covered 300, 0 pending — unmoved |

Counted per CLAUDE.md §5.3, since "+10 tests" is four different things otherwise:

| | |
|---|---|
| **features added** | 3 derivative rules (`convolution`, `avg_pool2d`, `adaptive_avg_pool2d`) |
| **defects fixed** | none — nothing here was previously wrong; it was absent |
| **tests added** | 10 in `test_train.py`, 3 finite-difference cases in `test_shim.py` |
| **documentation corrected** | `docs/platform/RELEASE_0_0_13a0.md` §5's transformer clause is now measurably false (§4) — **not edited by this round**, because that file is not this round's territory; it is named here so the next round has the measurement |
| **deleted** | nothing |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs convolution_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs avg_pool_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_train.py test_a_convolutional_model_trains_end_to_end_and_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_train.py test_a_tiny_transformer_language_model_trains_and_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_train.py test_the_weight_gradient_needs_stride_and_dilation_exchanged present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_train.py test_max_pool2d_backward_is_still_refused_by_name present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_train.py test_the_training_loop_actually_accumulates_into_dot_grad present -->
