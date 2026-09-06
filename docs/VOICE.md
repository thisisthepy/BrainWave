# VOICE — five speech models, and the wall they all share is complex numbers

Worktree `work/voice` on develop `2d58d27`. The consumer is real: **`voicestudio`**
(LatentForge, PyPI 5.16.0) declares `torchnative; extra == "native"`, and its models are speech.
Every architecture this repository had verified before today was an LLM or a vision model, so
this is the first round where STFT, complex tensors, `conv1d`, `ConvTranspose1d`, mel
filterbanks and resampling are in scope at all.

Upstream torch 2.13.0 (`/Volumes/macMini/caches/spike-venv/bin/python`) is the oracle
throughout. **Every operator below was run on both sides.** An operator that fails on upstream
too is not this project's gap and is not on the list — the first draft of the probe had one
(`torch.kaiser_window(12, beta=9.0)` is a `TypeError` upstream, because `periodic` is
positional), and it was fixed rather than reported.

---

## 0. Answers first

| question | answer | where |
|---|---|---|
| Is `torch.stft` reachable? | **No, and not by adding a kernel.** It returns a complex tensor. | §3 |
| Are complex tensors a candle-level gap? | **Yes — the same shape as `int8`.** `candle_core::DType` (0.11.0) has *no* complex variant, and candle has no FFT of any kind. `TorchDType` already has all three complex tags, so the shim side is not the problem. | §3 |
| How many distinct operators do the five need that this build lacked? | **21** at the start of the round, **18** now. | §1 |
| What landed? | 5 ops: `hann_window` (`.default` and `.periodic`), `sinc`, `clip`, `cumprod`. | §4 |
| What moved? | Vocos went from *blocked at construction* to *blocked in the forward*, on real published weights. Spark-TTS moved one wall further into `__init__`. | §2 |
| Which models ran on real weights? | **Vocos only**, and it needed a conversion step. The other four are `from_config`. | §2.1 |
| What was sized rather than built? | Complex dtypes + FFT (§3), and `mish`, whose kernel was written, verified against upstream, and then **removed** (§4.4). | §3, §4.4 |

---

## 1. The ranked list

Ranked by how many of the five models need the operator, then by whether it blocks
*construction* (nothing about the model can be exercised) or the *forward* (the architecture is
at least built). That distinction is docs/DEMAND8.md's and it is load-bearing here: three of the
five were construction-blocked at the start of the round, which means nothing had ever been
learned about their forwards.

`aten` names are given where the shim would need one; several of these are `torch._C._nn.*`
composites and are named as upstream spells them.

| # | operator | models | blocks | status |
|---|---|---|---|---|
| 1 | `torch.hann_window` | vocos, bigvgan, f5, spark | construction | **landed** (§4.1) |
| 2 | `torch.stft` / `torch.istft` | vocos, bigvgan, f5, spark | forward | **candle-level** (§3) |
| 3 | `torch.clip` | vocos | forward | **landed** (§4.3) |
| 4 | `torch._C._nn.mish` | f5 | forward | kernel written, **not landed** (§4.4) |
| 5 | `torch.cumprod` | spark | construction | **landed** (§4.2) |
| 6 | `torch.sinc` | bigvgan | construction | **landed** (§4.1) |
| 7 | `torch.kaiser_window` | bigvgan | construction | open (§5.1) |
| 8 | `torch.polar` | vocos | forward | **candle-level** (§3) |
| 9 | `torch.view_as_real` / `torch.view_as_complex` | vocos, bigvgan, f5 | forward | **candle-level** (§3) |
| 10 | `complex64` storage (`zeros`/`empty`) | vocos | forward | **candle-level** (§3) |
| 11 | `torch.fft.rfft` | vocos | forward | **candle-level** (§3) |
| 12 | `F.pad(mode="reflect")` | bigvgan, f5, parler, vocos | forward | open (§5.2) |
| 13 | `F.pad(mode="replicate")` | bigvgan | forward | open (§5.2) |
| 14 | `torch._C._nn.upsample_nearest1d` | bigvgan | forward | open (§5.3) |
| 15 | `torch._C._nn.col2im` / `im2col` (`F.fold`/`F.unfold`) | vocos | forward | open (§5.3) |
| 16 | `torch.rms_norm` (`nn.RMSNorm`) | f5 | forward | open |
| 17 | `torch.chunk` | f5 | forward | open |
| 18 | `torch.var` | spark | forward | open |
| 19 | `aten.detach_.default` | parler | construction | **deliberately refused** (docs/INPLACE.md) — §5.4 |
| 20 | `aten.floor_divide` with broadcasting | spark | construction | open (§5.5) |

Ranks 1–18 are the probe's own findings; 19 and 20 came out of running the models and are
different in kind — one is a standing refusal this repository made on purpose, the other is a
limitation of an op that *is* implemented.

**What is conspicuously already there.** `conv1d`, `ConvTranspose1d`, `nn.Conv1d`,
`nn.ConvTranspose1d`, `avg_pool1d`, `BatchNorm1d`, `one_hot`, `F.normalize`, `leaky_relu`,
`sdpa`, `nonzero`, `isin`, `triu`/`tril`, `cumsum`, `repeat_interleave`, `outer`, `linspace`,
`maximum`, `norm`, `rand_like`, `split` all pass. `aten.convolution.default` covers the 1-D and
transposed cases without anything new — the round did not have to touch convolution at all,
which was not the expectation going in.

---

## 2. The five models, before and after

`from_config` with a small config, forward with random inputs. Both columns are the *first*
failure, which is what makes "construction" and "forward" mean something.

| model | upstream | shim, before | shim, after |
|---|---|---|---|
| Vocos | forward OK | **construction** — `hann_window` | **forward** — `polar` |
| BigVGAN | forward OK | **construction** — `kaiser_window` | construction — `kaiser_window` |
| Parler-TTS | forward OK | **construction** — `detach_` | construction — `detach_` |
| F5-TTS | forward OK | forward — `mish` | forward — `mish` |
| Spark-TTS BiCodec | forward OK | **construction** — `cumprod` | construction — `floor_divide` broadcast |

All five forward on upstream, so none of these is an architecture that simply does not work.

The per-operator probe moved from **23 pass / 21 fail** to **26 pass / 18 fail**; upstream is
44 / 0 on the same probe.

### 2.1 Real weights versus a config — which claim is which

**Vocos ran on its published weights.** That is one model, not five, and getting there needed a
step worth recording: `AutoModel.from_pretrained("charactr/vocos-mel-24khz")` **cannot** work,
because the Hub checkpoint predates the transformers port and its `config.yaml` has no
`model_type`. voicestudio ships `models/vocos/weight_conversion.py`; `converted_checkpoint("mel")`
downloads the original and rewrites it into the port's layout. Through that path:

```text
upstream   constructed 13,531,650 params from the converted checkpoint, forward OK
shim       constructed 13,531,650 params from the same checkpoint,
           forward BLOCKED at torch.polar
```

The construction half is a real result on real weights — 80 tensors loaded, every parameter
shape agreed — and the forward wall is the *same* one the small config found, which is the
cross-check that says the config was not lying about where the wall is.

**BigVGAN, Parler-TTS, F5-TTS and Spark-TTS BiCodec were run from a config only.** For BigVGAN
and Spark the wall is inside `__init__` and a checkpoint cannot move it. For Parler-TTS and
F5-TTS a checkpoint could in principle reach further into the forward, and that is a claim this
round does not make.

### 2.2 A harness note that is not a detail

The models import `torchaudio` at module scope. Installing it would have pulled upstream's
`torch>=2.8` onto the path *behind* the shim, and every measurement after that point would have
been measuring the wrong torch. A stub package supplying only
`torchaudio.functional.melscale_fbanks` (which is stored as a buffer and never differentiated)
was used instead, and every script asserts `hasattr(torch._C, "_aten_implemented")` and prints
`shim` or `upstream` before doing anything.

`voicestudio` also requires `transformers>=5.16`, and the shared venv at
`/Volumes/macMini/caches/spike-venv` is on 5.15.1 with every gate in this repository pointing at
it. Nothing was installed into it: a separate venv shadows `transformers` and `tokenizers` and
reaches the shared one through a `.pth`. Re-checked after every install —
`spike-venv` still reports **transformers 5.15.1 / tokenizers 0.22.2**.

---

## 3. `torch.stft` and complex tensors: a candle-level gap, sized not built

This is the headline and the answer is negative, so it is worth being exact about *where* the
gap is.

### 3.1 The shim side is already done

`TorchDType` has `Complex32`, `Complex64` and `Complex128`, `is_complex()`, `to_real()`,
`to_complex()`, the `chalf`/`cfloat`/`cdouble` spellings, the element sizes, and `finfo`'s rule
that a complex dtype reports its *component* float. None of that is missing.

### 3.2 candle has no complex dtype at all

`candle_core::DType` (0.11.0, the pinned version) is exactly
`U8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3, F6E2M3, F6E3M2, F4…` — **no complex
variant**. The only two occurrences of the word "complex" anywhere in `candle-core/src` are a
prose sentence in `lib.rs` and a commented-out line in `npy.rs`'s dtype table. There is no FFT
either: the only hits for "fft" are `conv.rs` (a comment about FFT-based convolution not being
used) and a CUDA cudnn path.

So the failure is not "the shim declines `stft`". It is that `stft` returns a tensor this
backend cannot store, and the error says so in the right words:

```text
torch.zeros(4, dtype=torch.complex64)
  -> NotImplementedError: aten.zeros.default: dtype not storable by the candle backend
```

### 3.3 What that costs, and why it is not half-buildable

Six of the eighteen remaining gaps are downstream of this one dtype: `stft`, `istft`,
`view_as_real`, `view_as_complex`, `polar`, `fft.rfft`. They are not independent items on a work
list — implementing any one of them requires the storage first.

This is **the same shape as docs/INT8.md**, and that document is the model for the write-up:
`I8` was also absent from `candle_core::DType`, also already present as a `TorchDType` tag, and
was sized (+252 net lines across 11 files, CPU backend only) rather than half-landed. Complex is
strictly larger than `I8` was, for a reason that is structural rather than a matter of line
count: `I8` is one more scalar in an existing numeric tower, while a complex dtype is a
*pair* — every kernel that dispatches on dtype has to decide whether it means the pair or the
component, `is_floating_point()` must answer `False` for it while arithmetic still works, and
comparison, `max`/`min` and the ordering predicates have no meaning at all and must refuse.
Upstream encodes exactly that with a separate `AT_DISPATCH_..._AND_COMPLEX` family, which is
evidence that the split cannot be avoided rather than a stylistic choice.

An `stft` that returned a *real* `(…, 2)` tensor instead of a complex one would be a different
operator wearing the name, and it would break at the first `view_as_complex` — which is exactly
what F5-TTS and BigVGAN call next. **This round therefore builds none of it.**

### 3.4 What this section could not establish

Whether a later `candle-core` release or `main` has added complex support. docs/INT8.md §1.1
answered the equivalent question for `I8` by checking crates.io and candle's `main`, and that
check is what turned "we could bump the version" into a closed question. It was not repeated
here. **The claim above is about the pinned 0.11.0 only**, and re-doing §1.1's check is the
first thing anyone picking this up should do.

---

## 4. What landed

Five ops, each with golden cases compared against upstream and a test that calls the *spelling*
(docs/REACH.md shape 3 — a kernel nothing spells is reachable only by `_aten_dispatch`).

```text
golden   9235/9235, ops covered=229, pending case builders=0
suite    479 ok, 0 failed
docwatch PASS -- 493/493
```

493 rather than the 475 this round started from: eighteen of those markers are this document's
own, at the bottom of it. Five assert the new ops are in `_aten_implemented()`, five assert the
ones §3 and §5 call unreachable are *not* — so the day `stft` or `mish` lands, this document
fails rather than quietly becoming wrong.

### 4.1 `hann_window` (both overloads) and `sinc`

`hann_window` is rank 1 and the reason the round has a "before and after" at all. Two things had
to be measured rather than reasoned:

* **`periodic` changes the divisor, not the length.** Upstream adds one to `window_length`,
  builds the symmetric window of *that* length, and narrows the last sample off — so the divisor
  is the original `window_length`. `hann_window(5)` and `hann_window(5, False)` are both five
  long and agree only on element 0. `window_length == 1` is an early return of `[1.0]`, taken
  before any division; a first draft without it answered `[0.0]`.
* **The arithmetic is upstream's four in-place ops in the output dtype, not a closed form in
  `f64`.** Computing `0.5 - 0.5*cos(2*pi*i/N)` in `f64` and narrowing once is the plausible
  shortcut and it is wrong: at `float16`, element 1 of `hann_window(5)` is upstream's
  `0.345703125` where the correctly rounded value of the exact answer is `0.345458984375` — a
  relative 7e-4, far outside any comparator. Reproducing the per-op narrowing
  (`float_narrower`, the device `linspace_default` already uses) makes `float64`, `float16` and
  `bfloat16` **bit-identical** to upstream across both periodic modes.

`float32` cannot be held to bit-equality and the reason is upstream's *build*, not this kernel:
`cos_()` on a `float32` tensor is vectorised (Sleef), up to half an ULP off the correctly
rounded cosine, and `0.5 - 0.5*c` then cancels that error up into the result — which is why
upstream's own `hann_window(5)` is not symmetric. Measured over lengths 0/1/2/3/5/7/16/64 in
both modes, the worst disagreement is **|d| = 5.96e-08** (one ULP at 0.736, index 21 of
`hann_window(64)`) and **3.10e-06 relative**. The golden cases use `_bounded_divergence(2e-7,
1e-5)` for `float32` — a ceiling with headroom, so a kernel that agreed exactly still passes and
one that drifted further does not — and `_bit_exact` for the other three dtypes, which is where
the `f64` shortcut is actually caught.

`sinc` has the same "compute in the output precision" argument from the other side:
`torch.sinc(1.0)` at `float32` is `-2.78e-08`, not `0`, because it is the residue of pi not
being representable. An `f64` accumulator gives `0.0` and is therefore *less* faithful.

### 4.2 `cumprod`

Spark-TTS BiCodec's factorised quantiser derives its per-level strides with this during
`__init__`. The dtype rule was re-measured rather than carried over from `cumsum` (`cumprod` on
a `bool` input is `int64`, not `bool`).

The first draft copied `cumsum_default`'s written-out loop, which pulls the buffer to the host —
and **the suite caught it twice**: once as an unclassified host readback in `aten.rs`, once as an
op missing from `device.rs`'s mps readback list. Rewriting it as `n - 1` `narrow`/`mul`/`cat`
tensor operations means it never leaves the device and needs no such declaration. That is a
better kernel than the one it replaced, and it exists because a gate that could fail did.

### 4.3 `clip`

A true alias of `clamp`, and that was measured rather than assumed: the promotion ladder, the
bool-bound refusal (which names `clamp_scalar_cpu`, not a `clip_` kernel) and the "both bounds
absent" wording (`torch.clamp: At least one of 'min' or 'max' must not be None`) are all
`clamp`'s, byte for byte. So the kernel is shared, not copied.

One asymmetry is recorded rather than hidden: `aten::clip.Tensor` is **not** declared, though
`clamp.Tensor` beside it is. A dead overload key counts against `reach_allow.json`'s
`shape1_dead_overload_keys_ceiling`, which is a ratchet that may not grow. The cost is that
`torch.clip(x, min=some_tensor)` refuses by "no matching overload" where `torch.clamp(...)`
refuses by naming the overload it needed.

### 4.4 `mish` — written, verified, and deliberately not landed

> **Still not landed, and the reason has changed.** `bootstrap.py` is no longer the blocker —
> the round that owned it looked at this and could not pay it, because the *kernel* is what is
> missing: `aten.mish.default` is not in `_aten_implemented()`, so a `_nn.mish` composite would
> be a door onto nothing. It is now a one-round job rather than two. `docs/BINDINGS.md` §5.

`aten.mish.default` was implemented (`x * tanh(log1p(exp(x)))`, `f32` accumulator for the
reduced floats, upstream's `"mish_cpu" not implemented` refusal for integral inputs) and matched
upstream to within one ULP across `float64`/`float32`/`float16`/`bfloat16`, including the
saturating tail where `exp` overflows and the answer is `x`.

It was then **removed**, because upstream's only spelling is `torch._C._nn.mish` — there is no
`torch.mish` and no `Tensor.mish` — and that spelling is installed by `bootstrap.py`, which was
outside this round's territory. The suite says so directly:

```text
shape 2: aten.mish.default has a kernel and is compared by golden, but no
         `torch.<name>` / `Tensor.<name>` spelling and no composite in
         bootstrap.py reaches it -- it is invisible from Python.
```

Landing a kernel nothing can call would have raised the op count without moving F5-TTS one
instruction. **This is the cheapest item on the whole list for whoever owns `bootstrap.py`**: two
lines beside `silu`'s, in the shape `def mish(input): return dispatch("aten.mish.default", input)`,
plus the name in the `_nn` surface list. The kernel body is in this document's git history at the
commit that carries it.

---

## 5. The open items, with what each actually needs

### 5.1 `kaiser_window` — BigVGAN's whole construction

The one construction wall left that is a plain missing kernel. It needs a modified Bessel
function of the first kind (`i0`), which candle does not have and which upstream implements as a
Cephes `chbevl` polynomial — so it is transcription work rather than composition, and getting it
half right produces a window that looks plausible and is wrong. `periodic` follows
`hann_window`'s rule (`kaiser_window(5)` is the six-point symmetric window with the last sample
dropped), so that half is already understood.

### 5.2 `reflect` / `replicate` padding — four of the five models

`torch._C._nn.pad` is wired for `constant` only. This one is worth more than its rank suggests:
it is what `torch.stft(center=True)` calls, so it hides `stft`'s own refusal (the first probe
run reported `stft` as failing on *padding*, which is why the probe has a `center=False` row
beside it). It is also the only gap on this list shared with Parler-TTS.

### 5.3 `upsample_nearest1d`, `col2im` / `im2col`

`upsample_bilinear2d` and `upsample_bicubic2d` already exist, so the 1-D nearest case has
neighbours to follow. `col2im`/`im2col` are `F.fold`/`F.unfold`, which Vocos uses for its
overlap-add — and which it only reaches *after* `istft`, so it is behind §3 in practice.

### 5.4 `detach_` — Parler-TTS, and it is a refusal not a gap

`aten.detach_.default` is refused by name on purpose (docs/INPLACE.md). Parler-TTS hits it in
`__init__`, where upstream is setting `requires_grad=False` on a leaf. This is the one item on
the list where the right first step is reading that document rather than writing a kernel.

### 5.5 `floor_divide` broadcasting — Spark-TTS

Now that `cumprod` is in, Spark's quantiser gets one step further and stops at
`aten.floor_divide.default: broadcasting other than a scalar or an exact shape match is not
implemented`. The op exists; its broadcast path does not. That is a narrower and better-defined
task than it was this morning, which is the argument for having landed `cumprod` at all.

---

## 6. What this round did not do

* **No real weights for four of the five models** (§2.1). BigVGAN's and Spark's walls are inside
  `__init__` so a checkpoint cannot move them, but Parler-TTS's and F5-TTS's could.
* **No check of candle `main` for complex** (§3.4).
* **No timings.** Another agent was running throughout; docs/PERF.md's rule is that a measured
  number from a loaded machine is worse than no number.
* **`voicestudio`'s other fifteen models were not looked at.** The five here were chosen for
  disjoint operator demand, not coverage — CosyVoice, Higgs, Dia, Qwen3-TTS and the rest may
  well need things nothing above names.

---

<!-- DOCWATCH: op-implemented aten.hann_window.default -->
<!-- DOCWATCH: op-implemented aten.hann_window.periodic -->
<!-- DOCWATCH: op-implemented aten.sinc.default -->
<!-- DOCWATCH: op-implemented aten.clip.default -->
<!-- DOCWATCH: op-implemented aten.cumprod.default -->
<!-- DOCWATCH: op-not-implemented aten.mish.default -->
<!-- DOCWATCH: op-not-implemented aten.kaiser_window.beta -->
<!-- docs/FFT.md landed `aten.stft.default` and `aten.stft.center`. The marker
     above was `op-not-implemented aten.stft.default`; it is inverted rather than
     deleted, so this section still fails if the op ever leaves again. What §3
     says about `torch.stft` being unreachable is superseded by docs/FFT.md for
     `center=False`; `center=True` still needs `bootstrap.py`'s `F.pad(reflect)`
     branch (docs/PAD.md §5). -->
<!-- DOCWATCH: op-implemented aten.stft.default -->
<!-- DOCWATCH: op-implemented aten.stft.center -->
<!-- DOCWATCH: op-not-implemented aten.polar.default -->
<!-- DOCWATCH: op-not-implemented aten.view_as_complex.default -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json hann_window present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json sinc present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json clip present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json cumprod present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json cumprod present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs hann_window_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs sinc_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs cumprod_default present -->
