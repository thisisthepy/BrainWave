# VOICE4 — a voicestudio model, end to end, and the wall a coverage list cannot see

Worktree `work/voice4` on develop `976a01b`. Upstream torch 2.13.0
(`/Volumes/macMini/caches/spike-venv/bin/python`) is the oracle throughout, in its own process
with `PYTHONPATH` stripped.

Seven rounds of `docs/architectures/VOICE.md` and `docs/architectures/VOICE3.md` closed operators that speech models *tripped
over* — `hann_window`, `sinc`, `kaiser_window`, `i0`, `var`, `std`, `col2im`, `upsample_nearest1d`
— one wall at a time. **None of them ever ran a voicestudio model.** This round did.

---

## 0. Answers first

| question | answer | where |
|---|---|---|
| Which model, and why that one? | **BigVGAN v2 24 kHz 100-band 256x**, on NVIDIA's published weights (112,414,512 params). It is the only thing in voicestudio's zoo that is *both* meaningful and numerically comparable end to end. | §1 |
| Did it reach the end? | **Yes.** 1.0 s of real speech → a 23,808-sample waveform, on the shim, from `from_pretrained`. | §5 |
| Does the output agree? | **5.69e-05 relative**, against a tolerance of **2.71e-04** derived from upstream's own float32-vs-float64 error. Pearson r = 0.9999999997. | §5 |
| Is the shim worse than upstream? | **No — it is closer to the truth.** Ratio to upstream's own oracle error is **0.61**. | §5 |
| How many operators did it need? | **Zero new operators. Two new meta kernels.** | §4 |
| What was the wall, by name? | `aten.sum.default` and `aten.view.default`, **both already in `_aten_implemented()`**, neither with a meta kernel. | §4 |
| What nullification did the tests miss? | **Widening the tolerance 100×.** Every test stayed green. | §6 |

---

## 1. Which model, and what was rejected

voicestudio 5.16.0 (`pip download voicestudio --no-binary :all:`) ships **20 model folders** in
four groups: voice cloning, voice design, voice editing, and "vocoders and codecs". The package
describes itself as "a unified toolkit for voice cloning, designing and editing", and every model
is an ordinary `transformers` model.

**Chosen: BigVGAN.** Four reasons, in the order they decided it:

1. **It is what "voice" means in this package that the shim can actually be scored on.** BigVGAN
   is the module that turns features into *audio*. The README lists it as a submodel that F5-TTS
   and the CosyVoice family hold directly, so it is not a corner of the zoo — it is on the path of
   several headline models.
2. **Its output is a waveform, so "compare the output numerically" is a real question.** A relative
   error on 23,808 audio samples is a meaningful number in a way that a logits tensor from a
   sampling decoder is not.
3. **It is deterministic.** No `generate`, no sampler, no temperature. `docs/numerics/AGREE.md` had to
   *exclude* `vits` and `vit_mae` from its population because upstream does not reproduce itself
   on them; BigVGAN reproduces itself bit-for-bit, so any difference measured is the shim's.
4. **Real published weights load without a research-only dependency chain.** 112M parameters
   through voicestudio's own `weight_conversion`, which is a documented, supported path.

**Rejected, and why — this is the part a round is tempted to skip:**

| rejected | why not |
|---|---|
| **Vocos** | The obvious alternative vocoder, and `docs/architectures/VOICE.md` §2.1 already ran its construction on real weights. Its forward is an **iSTFT**: `torch.polar`, `view_as_complex`, `fft.irfft`. VOICE.md §3 measured that as a **candle-level** gap — `candle_core::DType` has no complex variant at all. Not reachable by adding kernels, so it cannot answer "did it reach the end". |
| **Spark-TTS BiCodec** | The other codec. Blocked in `__init__` at `floor_divide` with broadcasting (VOICE.md rank 20), and its output is *codes*, not audio — the numerical comparison would be over integers, where "agrees" is a much weaker statement than a waveform tolerance. |
| **F5-TTS** | Meaningful and headline, but it is a **flow-matching sampler**: the forward runs an ODE solver over many steps, so an end-to-end number confounds operator error with step-count divergence. It also needs `mish`, `rms_norm` and `chunk` (VOICE.md ranks 4, 16, 17), and it *contains* BigVGAN — so BigVGAN is the honest first half of it. |
| **Higgs TTS 2/3, Qwen3-TTS, Dia2, Chroma** | 1.7B–4B parameters and autoregressive with sampling. The download alone is tens of gigabytes, and `generate` is nondeterministic, which `AGREE.md` §2 excludes from its denominator for exactly this reason. Picking one of these would have produced a number that could not be defended. |
| **Parler-TTS** | Still blocked at `aten.detach_.default`, which `docs/kernels/INPLACE.md` refuses **on purpose**. Running it would mean reversing a deliberate design refusal, not closing a gap. |
| **PromptTTS++, VoxInstruct** | Weights only inside a HF Space or a mirror repo; the conversion path is the least supported in the package. Fragile provenance for a round whose deliverable is a measurement. |

The one thing BigVGAN does **not** exercise is the mel front end, which is `torch.stft` and
therefore the same complex-tensor wall Vocos hit. That is stated rather than hidden: the mel is
computed **once on upstream** and the *identical* tensor is fed to both sides, so the comparison
is of the generator, which is the 112M parameters that matter.

---

## 2. The setup, and the one thing that is not upstream's

```
model        voicestudio.models.bigvgan.BigVGANModel      (voicestudio 5.16.0, PyPI sdist)
weights      nvidia/bigvgan_v2_24khz_100band_256x         112,414,512 parameters, 449 tensors
             through voicestudio's own weight_conversion.build_model_files
audio in     Narsil/asr_dummy mlk.flac -> afconvert -> 24 kHz mono WAV, 1.0 s from t=1.0 s
mel          the model's OWN mel_spectrogram(): n_fft 1024, hop 256, win 1024, 100 bands,
             Slaney scale and Slaney norm, uncentered  ->  (1, 100, 93)
waveform out (1, 23808)   = 93 frames x 256 hop
```

Two substitutions were needed and both are recorded, because a substitution that is not recorded
is a place where the measurement is not of what it says:

* **`torchaudio` is not installed** in the spike venv, and voicestudio's feature extractor imports
  it at module scope. The generator never calls it; the only entry point that is reached is
  `torchaudio.functional.melscale_fbanks`, which is a **closed-form filterbank**, so torchaudio's
  own Slaney implementation was transcribed rather than installed. It affects only the *input*,
  and the input is byte-identical on both sides, so it cannot flatter the comparison.
* **`remove_weight_norm()` is not called.** The converted checkpoint already carries folded
  weights and has no parametrization to remove — upstream raises `ValueError` if asked.

---

## 3. What the model asks for — measured, not inventoried

`TorchDispatchMode` over the real forward and over `__init__`, on upstream. Checked in as
`rust/torch_c/pytests/voice4_bigvgan_ops.json`; `voice4_capture.py` regenerates it.

```
forward        14 distinct ops, 2168 calls
construction   15 distinct ops, 3529 calls
```

| forward | calls | | construction | calls |
|---|---:|---|---|---:|
| `mul.Tensor` | 436 | | `mul.Tensor` | 654 |
| `convolution.default` | 334 | | `empty.memory_format` | 449 |
| `add.Tensor` | 290 | | `zero_.default` | 333 |
| `exp.default` | 218 | | `uniform_.default` | 231 |
| `replication_pad1d.default` | 218 | | `add.Tensor` | 218 |
| `view.default` | 218 | | `arange.start` | 218 |
| `expand`, `pow`, `reciprocal`, `sin`, `slice` | 109 each | | `div.Tensor`, `kaiser_window.beta`, `sinc.default`, `sum.default`, `view.default` | 218 each |
| `div.Tensor` | 6 | | `normal_.default` | 116 |
| `clamp.default`, `squeeze.dim` | 1 each | | `copy_`, `detach` | 109 each |

**All 29 were already in `_aten_implemented()`.** `replication_pad1d` (VOICE.md rank 13),
`kaiser_window` (rank 7) and `sinc` (rank 6) had each been closed by an earlier round, and
`convolution.default` covered the 1-D and transposed cases without anything new — which VOICE.md
§1 had already noted was not the expectation going in.

So the round's first answer was: **zero operators needed.**

---

## 4. That answer was wrong, and the method is why

Running the model stopped it twice. Both walls, by name:

```
aten.sum.default    NotImplementedError: torch._C shim has no meta kernel for aten.sum.default
aten.view.default   NotImplementedError: torch._C shim has no meta kernel for aten.view.default
```

Both at `modeling_bigvgan.py:89`, one line:

```python
def build_anti_alias_filter(cutoff, half_width, kernel_size):
    ...
    return (taps / taps.sum()).view(1, 1, kernel_size)
```

**`from_pretrained` constructs under `init_empty_weights`, i.e. on the `meta` device**, and
BigVGAN builds 218 anti-alias resampling filters in `__init__` — each of which normalises by its
own sum and reshapes. A meta tensor has shape and dtype and no storage, so both calls have to
answer without computing.

### 4.1 The finding, which is worth more than the two kernels

`_aten_implemented()` **contained both ops the whole time.** They have had dense kernels and
golden cases for months. The list means "has a kernel and `tools/golden/cases.py` compares it
against upstream" — and a meta tensor has no values to compare, so meta support is invisible to
it by construction.

This is `docs/architectures/ARCH100.md`'s lesson with a different variable. That round measured missing *names*
outnumbering missing *kernels* 49 to 22, and the rule it produced — look for the other spelling
first — was followed here and found nothing, because **the missing thing was not a name or a
kernel but a device**. An operator can be on every coverage list this repository keeps and still
be a wall.

It is also `CLAUDE.md` §5.4 exactly: the criterion ("is the op in `_aten_implemented()`?") decided
the answer, and the answer agreed with the hypothesis, which is when to suspect it hardest. What
broke the loop was not a better list. It was running the model.

`docs/devices/META.md` §7.4 had in fact already named both — its table of what meta still cannot reach
lists `sum` under 축소 (22 ops) and `view` under 뷰·모양 (13 ops). The information was on disk and
the round did not consult it, because the coverage question had already been answered.

### 4.2 The two kernels

Both follow META.md §7.1's stated convention: **the dtype rule is called, not restated**, so a
meta kernel cannot promise a dtype the dense kernel would refuse to produce.

| kernel | shape | dtype |
|---|---|---|
| `aten.sum.default` | **always rank 0**, whatever went in — it does not consult numel, so `sum(empty(0))` is rank 0 too | `sum_natural_tag`, extracted from `sum_or_mean`'s body: floating keeps its own, **everything else widens to `int64`, `bool` included**. An explicit `dtype=` wins. |
| `aten.view.default` | `resolve_shape`, the dense path's own helper, which was already storage-free | the input's, unchanged — a view never converts |

`sum_natural_tag` is new only as a *name*: the rule was inline in `sum_or_mean` and is now called
from both paths, which is the same move META.md §7.1 records for `unary_float_tag`,
`clamp_result_tag` and `expand_target`. The dense behaviour is unchanged.

**`view` needs one check the dense path gets for free.** `resolve_shape` only consults `numel`
when it has a `-1` to fill, so with no wildcard it cannot notice that the requested shape holds a
different number of elements. On the dense side candle's `reshape` catches that; on meta there is
no candle call afterwards, so a meta `view(4, 2)` of a 6-element tensor would silently answer
`[4, 2]`. The check is written here and the wording is upstream's own
(`shape '[4, 2]' is invalid for input of size 6`). This is the same asymmetry META.md §7.2 already
records for `expand`, which likewise cannot share candle's extent check.

### 4.3 Aliases, looked for first

Per the standing rule, each was looked for under another spelling before a kernel was written.
Neither is an alias:

| candidate | looked for under | verdict |
|---|---|---|
| `sum.default` meta | `sum.dim_IntList`, `mean.dim` | **kernel.** Neither has a meta kernel either; there was nothing to borrow. `sum.default` is a distinct overload with no `dim` and no `keepdim`, and it is the one BigVGAN calls. |
| `view.default` meta | `reshape.default`, `expand.default`, `_unsafe_view` | **kernel.** `expand` has a meta kernel but its rule is broadcasting, not re-layout. `reshape` deliberately stays refused: it may *copy*, which `view` never does, so answering it from `view`'s rule would promise a view where upstream might return a copy. |

### 4.4 A third wall, which is the harness's and not the model's

The first thing the replay hit was **`torch.from_numpy`**, which the shim does not implement.
That is the test harness loading a `.npy` file, not BigVGAN — so it was routed around (the mel
travels as JSON) rather than counted, and it is named here because a round that quietly fixed it
would be reporting a harness change as a model result.

---

## 5. The measurement

Same weights, same mel, both sides, `float32`. Upstream additionally at `float64` to supply the
oracle.

```
waveform length                    23808 samples, both sides
scale (max |upstream f32|)         0.472636

shim  vs  upstream f32             5.6904e-05      <- the result
upstream f32  vs  upstream f64     6.7785e-05      <- the tolerance source
shim  vs  upstream f64             4.1260e-05

TOLERANCE = 4 x oracle             2.7114e-04      PASS
ratio (shim's error / upstream's)  0.609
pearson r                          0.9999999996857
bit-identical samples              74 / 23808
rms   shim 0.10529363   upstream 0.10529330
```

### 5.1 Where the tolerance comes from

Not chosen. `docs/numerics/AGREE.md` §2's method, applied to this model rather than to its population:
upstream is run **in float64 on the same input**, and its own float32 answer is scored against
that. The rule is `max(floor, 4 × oracle)`, where the floor is AGREE.md's 1.186e-06 and the factor
of 4 is AGREE.md's own, carried over with its citation rather than re-derived.

**The per-model derivation is load-bearing here, and this is the number that says so.** AGREE.md's
population tolerance is 1.186e-06, the p90 of 263 architectures. BigVGAN's own oracle error is
**6.78e-05 — about 57× that**. A round that had applied AGREE.md's constant would have reported a
divergence of 48× the tolerance and gone hunting for a defect that does not exist; the constant
would have been calling *upstream* wrong. A 112M-parameter convolutional stack with 334
convolutions in series accumulates far more float32 error than the median architecture in that
sweep, and the only honest tolerance is the one that model's own oracle sets.

### 5.2 The ratio below 1

`ratio = 0.609` means **the shim's float32 waveform is nearer the float64 truth than upstream's
float32 waveform is.** This is not a claim of superiority — it is what two independent float32
truncation paths look like, and AGREE.md §3 found the same shape across its population (nine of
26 architectures above tolerance had a ratio below 1). It is reported because the opposite result,
a ratio of 4 or 40, is what a real defect would look like, and the number has to be able to say
that.

74 samples of 23,808 are bit-identical, which is the expected count for two independent
accumulation orders over a deep stack — not evidence of anything, and stated so it is not later
read as such.

---

## 6. Nullification — including the one that was not caught

Every kernel and every claim was broken deliberately and the suite re-run.

| # | nullification | caught by | verdict |
|---|---|---|---|
| 1 | meta `sum` returns the input dtype instead of widening to `int64` | `test_the_sum_meta_kernel_answers_what_upstream_answers` (`sum_bool`: shim `torch.bool` vs upstream `torch.int64`) | **red** |
| 2 | meta `view` drops the numel check | `test_the_view_meta_kernel_answers_what_upstream_answers` (`view_bad_numel`: shim answered `[4, 2]`, upstream raised) | **red** |
| 3 | meta `view` kernel deleted entirely | the kernel test **and** the end-to-end replay | **red, both** |
| 4 | `_ORACLE_FACTOR` widened from 4.0 to 400.0 | **nothing** | **GREEN — not caught** |

### 6.1 The one that was not caught, which is the finding

Nullification 4 left **every test in the file green**, the end-to-end replay included. The reason
is that the assertion guarding the tolerance was

```python
tol = max(_AGREE_FLOOR, _ORACLE_FACTOR * oracle)
assert tol == _ORACLE_FACTOR * oracle      # <- true for ANY factor
```

a **tautology**. It restates the definition of `tol` and therefore cannot fail. This is
`CLAUDE.md` §5.5's category — a verification that cannot fail is not a verification — and it
arrived the way that category always does: the test was written to document the derivation, and
documenting a derivation reads exactly like checking it.

The replay does not catch it either, and could not: a 100× wider tolerance still accepts a
*correct* answer. Every number this round measured passed, so nothing had ever demonstrated that
the comparison is capable of saying no.

Both halves are now closed:

* `_ORACLE_FACTOR` is pinned to `4.0` with AGREE.md's citation, so widening it is an edit to a
  named assertion rather than to a constant nobody reads;
* `test_the_tolerance_would_actually_reject_a_wrong_waveform` drives the comparison with a
  waveform wrong by 10× the tolerance and asserts rejection, and with one wrong by a tenth and
  asserts acceptance — so the tolerance is shown to be **calibrated**, not merely large.

### 6.2 What nullification 1 also revealed

The end-to-end replay stayed **green** under nullification 1. BigVGAN sums a `float32` tensor, so
the `int64` widening rule is never exercised by the model. That is the standing argument for
keeping kernel tests alongside end-to-end ones: the model reaches one point in the dtype lattice,
and a kernel is the whole lattice. `docs/architectures/VOICE3.md` §2 made the same argument about `i0` needing a
sweep rather than a spot check.

---

## 7. What is not closed

* **`torch.from_numpy`** (§4.4). Harness-level, deliberately not fixed in this round.
* **`sum.dim_IntList`, `mean.dim`, `reshape`, `slice`, `t`, `permute`, `cat`, `mm`, `bmm`** still
  have no meta kernel. META.md §7.4's table stands apart from the two entries this round removed;
  this round closed **two members of two families, not the families.**
* **The mel front end under the shim.** BigVGAN's own `mel_spectrogram` needs `torch.stft` and
  complex tensors, which is VOICE.md §3's candle-level gap. The mel was computed upstream and
  handed to both sides. So this round proves the **generator**, not the feature extractor, and
  `forward(labels=...)` — whose mel reconstruction loss goes through the same STFT — was not run.
* **The other 19 voicestudio models.** One model is one model. §1 says which were rejected and why,
  and none of those rejections became easier because this one passed.
* **GraalVM native image.** Not attempted; out of this round's scope.

---

## 8. Numbers

```
test_voice4.py                12 tests
new meta kernels               2   (aten.sum.default, aten.view.default)
new operators                  0
golden cases                   unchanged -- a meta kernel has no values to compare,
                               and both ops were already golden-covered as dense kernels
```

<!-- DOCWATCH: op-implemented aten.sum.default -->
<!-- DOCWATCH: op-implemented aten.view.default -->
<!-- DOCWATCH: op-implemented aten.replication_pad1d.default -->
<!-- DOCWATCH: op-implemented aten.kaiser_window.beta -->
<!-- DOCWATCH: op-implemented aten.sinc.default -->
<!-- DOCWATCH: op-implemented aten.convolution.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs sum_natural_tag present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs resolve_shape present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_voice4.py test_the_sum_meta_kernel_answers_what_upstream_answers present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_voice4.py test_the_view_meta_kernel_answers_what_upstream_answers present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_voice4.py test_the_tolerance_would_actually_reject_a_wrong_waveform present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_voice4.py test_the_two_meta_kernels_are_the_meta_half_of_ops_already_implemented present -->
<!-- DOCWATCH: json-key rust/torch_c/pytests/voice4_bigvgan_ops.json forward_ops present -->
<!-- DOCWATCH: json-key rust/torch_c/pytests/voice4_bigvgan_ops.json construction_ops present -->
<!-- DOCWATCH: count golden_ops_covered ge 301 -->
