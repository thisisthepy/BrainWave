# Six more architectures behind the same wall GPT-BigCode was behind, and one of them is not

docs/architectures/ARCH20.md and the `d1198de` follow-up got GPT-BigCode to 20/20 by taking upstream's own
`PYTORCH_JIT=0` path, which turns every `@torch.jit.script` decorator in the model file into a
no-op — the scripted function *is* the plain Python function under it. That same environment
variable is set unconditionally at shim import (`bootstrap.py`), so it silently affects every
model that has a module-scope `@torch.jit.script`, not just GPT-BigCode. Six more of
`transformers`' architectures carry one and were never in the twenty: `deberta`, `deberta_v2`,
`vits`, `zoedepth`, `sew_d`, `sam3_video`.

**Written incrementally, one model at a time, so a kill mid-task leaves something behind** (a
prior attempt at this exact brief did not).

Method follows docs/architectures/ARCH20.md: toy `AutoConfig` (small hidden size, one layer or few, few heads,
tiny vocab — coverage, not model quality), `TORCH_USE_RTLD_GLOBAL=1`,
`PYTHONPATH=torchnative/src/main`, transformers 5.15.1 in `spike-venv`. Scratch scripts live under
`/tmp/arch26/`, not committed, the same as ARCH20's `/tmp/arch7/sweep.py`.

---

## 1. `deberta` and `deberta_v2` — blocked before any weight ever multiplies

**Verdict: neither forwards. Both stop on the same missing kernel, `aten.sqrt.default`, before
a numeric comparison is possible at all.** This is a stronger finding than "scripted vs eager
drift" — there is no forward to compare yet.

### 1.1 What actually blocks them

`deberta`'s `DebertaLayerNorm.forward` (`modeling_deberta.py:52`) computes its own layer norm by
hand — `(hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)` — instead of going
through `nn.LayerNorm`/`aten.native_layer_norm.default`, which is why none of the twenty
architectures (all of which use `nn.LayerNorm`) ever exercised this path. It fires in the
embeddings stage, before the model reaches an attention block, and it fires whether or not
`relative_attention` is on:

```
NotImplementedError: not implemented in torch._C shim: torch.sqrt(...) -- overload resolution
has no table entry for this op (rust/torch_c/src/overloads.json)
```

`deberta_v2` uses real `nn.LayerNorm`, so it gets past embeddings, but its attention block calls
`scaled_size_sqrt(query_layer, scale_factor)` **unconditionally, before the
`if self.relative_attention:` branch** (`modeling_deberta_v2.py:242`) — every forward computes an
attention temperature via `torch.sqrt`, relative-position or not. Same wall, different line.

### 1.2 Why this is a kernel gap, not a spelling gap

Checked before touching anything, because §9 of ARCH20.md has a whole inventory of names that
*look* missing but resolve to an existing kernel:

```
$ python -c "import torch; print(torch._C._dispatch_has_kernel_for_dispatch_key('aten::sqrt', 'CompositeImplicitAutograd'))"
False
```

and a `TorchDispatchMode` trace of `torch.sqrt(x)` on upstream fires exactly one op,
`aten.sqrt.default` — a leaf kernel, not a composite that decomposes into ops this shim already
has (the way `torch.square` decomposes into `pow.Tensor_Scalar`, ARCH20.md §3). Grepping
`rust/torch_c/src/aten.rs`, `overloads.json`, and `methods.json` for `sqrt` finds only
`rsqrt`/`clamp`-adjacent entries — no `sqrt` kernel exists to wire a name to. Composing `sqrt` out
of `pow(x, 0.5)` in `bootstrap.py` was considered and rejected: that would be inventing a
computation path upstream does not take, in a round whose whole point is not doing that silently
(the brief's example of exactly this mistake pattern). `aten.rs` is out of my territory this
round regardless.

**Finding, by name: `aten.sqrt.default` is a missing kernel.** Not a name-table gap — a genuine
absence in `aten.rs`.

### 1.3 The fix that *is* in territory, found on the way

Before reaching `torch.sqrt`, both models hit a second, earlier wall that **is** a missing name:
`torch._C._dynamo.eval_frame.set_eval_frame`.

`torch/_dynamo/__init__.py:133` rebinds `torch.manual_seed = torch._disable_dynamo(torch.manual_seed)`
**unconditionally at `torch._dynamo` import time** — and `torch._dynamo` is itself an
unconditional import once `transformers.masking_utils` is reached (docs/graph/DYNAMO.md §6, already
established for the other twenty). So merely importing `transformers` mutates `torch.manual_seed`
process-wide; the next call to it — which a numeric-comparison harness needs, to get identical
initial weights — routes through `torch/_dynamo/eval_frame.py`'s `_fn`, which calls
`prior = set_eval_frame(None)` before the wrapped call and `set_eval_frame(prior)` after. Only two
`_dynamo.eval_frame` names were ever *called* by any of the twenty (`set_guard_error_hook`,
`set_code_exec_strategy`, both hook registrations that discard their argument) — `deberta` is the
first architecture in this project's history to call `torch.manual_seed` after `transformers` has
already pulled in `torch._dynamo`, because none of the twenty needed reproducible init to forward.

Unlike the two existing no-ops, `set_eval_frame` is a **get-and-set** — the caller uses the return
value to restore the prior state on the way out, so an unconditional `None` return (which is what
the existing no-op shape would have produced) would have been a real behavioral bug for any nested
`disable()` context, not merely an unreachable stub. Fixed as a state cell — get returns what was
last set, set stores and returns the prior — in `rust/torch_c/src/bootstrap.py`, alongside a
sibling cell for `set_eval_frame_isolate_recompiles_id` (same call shape, same file, a few lines
away, not yet observed to be called by anything but added for the same reason: an unconditional
`None` there would be a landmine the moment something does call it while nested).

Verified fixed: `import torch; from transformers import DebertaModel; torch.manual_seed(0)` no
longer raises, on the rebuilt artefact (marker-checked via `strings _C.abi3.so | grep -c
set_eval_frame_isolate_recompiles_id` before and after, per the `bootstrap.py` staleness trap).

### 1.4 What was checked anyway: TorchScript vs eager on the live functions themselves

The brief's sharper concern — upstream runs these helpers **scripted**, we would run them
**eager** under `PYTORCH_JIT=0`, and TorchScript's semantics can differ from eager's (integer
division, type promotion) with no error at all — is real and worth checking independent of the
sqrt blocker, because the functions that would diverge don't need a forward to exercise; they can
be called directly.

Checked entirely on **upstream torch** (no shim): every `@torch.jit.script` helper that is live in
`deberta`/`deberta_v2`'s forward path (`build_relative_position`, `c2p_dynamic_expand`,
`p2c_dynamic_expand`, `pos_dynamic_expand`, `scaled_size_sqrt`, `build_rpos`,
`compute_attention_span`, `uneven_size_corrected` for `deberta`; `make_log_bucket_position`,
`build_relative_position`, `c2p_dynamic_expand`, `p2c_dynamic_expand`, `pos_dynamic_expand`,
`scaled_size_sqrt`, `build_rpos` for `deberta_v2`, which is where `//` integer division
(`bucket_size // 2`) and `torch.log`/`torch.ceil`/`torch.sign`/`torch.where` composition actually
live) was called once as the real `ScriptFunction` (`PYTORCH_JIT=1`, upstream's default) and once
as the plain undecorated function (`PYTORCH_JIT=0`, what this shim's forward would drive), on
identical hand-built input tensors covering the branches each function has (`query_layer.size(-2)
== key_layer.size(-2)` and `!=` for `build_rpos`; positive and negative `relative_pos` either side
of `mid` for `make_log_bucket_position`'s bucketing, which is the one with integer division and a
`torch.where` predicate that could plausibly round differently under scripting).

**Every value matched exactly, printed at 10 digits of precision, no diff.**
`/tmp/arch26/scripted_funcs_probe.py`, run as
`PYTORCH_JIT=1 python probe.py deberta > scripted.txt` and
`PYTORCH_JIT=0 python probe.py deberta > eager.txt`, `diff` empty for both `deberta` and
`deberta_v2`.

**What this does and does not establish.** It rules out TorchScript-vs-eager drift for exactly the
inputs tried, on exactly these functions, run standalone. It is not the acceptance-grade check the
brief asked for (argmax-per-position and max-logit-diff on a real forward through the shim against
identical upstream weights) — that check needs a full model forward, which needs `aten.sqrt`, which
does not exist yet. **The real answer to "do the numbers drift" is: cannot be measured until
`aten.sqrt.default` lands.** That is the finding to carry forward, not the standalone check above,
which is corroborating evidence at best.

> **MEASURED, in docs/kernels/KERNELS26.md §2.4.** `sqrt` and `repeat` landed and both
> architectures forward. Same toy config, `torch.manual_seed(0)`, a 6-token
> forward, upstream against the shim, with the full `state_dict` diffed as well
> as the output: **all 37 (`deberta`) and 45 (`deberta_v2`) weight tensors are
> bit-identical**, and the outputs agree to a maximum relative difference of
> **1.61e-07** and **1.21e-07** — float32 epsilon is 1.19e-07. **There is no
> TorchScript-versus-eager drift in either architecture.**

### 1.5 Summary

| | deberta | deberta_v2 |
|---|---|---|
| imports | yes | yes |
| constructs (`from_config`, toy) | yes | yes |
| forwards through the shim | **no** — `torch.sqrt`, embeddings stage (`DebertaLayerNorm`), config-independent | **no** — `torch.sqrt`, first attention block (`scaled_size_sqrt`), config-independent |
| blocking kind | **missing kernel**, `aten.sqrt.default` | **missing kernel**, `aten.sqrt.default` |
| numeric comparison against upstream | **not possible yet** | **not possible yet** |
| scripted-vs-eager check on the live helper functions alone (upstream only, no shim) | identical, all functions, all branches tried | identical, all functions, all branches tried |
| names fixed on the way | `_dynamo.eval_frame.set_eval_frame`, `set_eval_frame_isolate_recompiles_id` | (shared fix) |

---

## 2. `vits` — blocked at construction, on `weight_norm`

**Verdict: does not construct. `VitsModel(cfg)` raises before any forward is possible.**

Toy config (tiny hidden/window/flow sizes, one WaveNet layer, `use_stochastic_duration_prediction`
disabled to keep the toy small) constructs and forwards cleanly on **upstream** torch — waveform
out, `(1, 176)`, sane sum. Under the shim, construction itself fails:

```
VitsModel.__init__ -> VitsResidualCouplingBlock -> VitsResidualCouplingLayer -> VitsWaveNet.__init__
  in_layer = weight_norm(in_layer, name="weight")            modeling_vits.py:334
  torch/nn/utils/parametrizations.py:380 weight_norm(...)
  torch/nn/utils/parametrize.py:645 register_parametrization(...)
  torch/nn/utils/parametrize.py:205 ParametrizationList.__init__ -> _maybe_set(original, new)
  torch/nn/utils/parametrize.py:92  dest.set_(src)
NotImplementedError: TensorBase.set_: expected a torch.UntypedStorage, got Parameter -- the
no-argument and tensor-argument spellings of set_ are not implemented in this shim
```

**This is a missing kernel, confirmed by trace, not a name that needs wiring.** `tensor.rs::set_`
(`rust/torch_c/src/tensor.rs:1244`) implements only the storage-argument overload
(`aten.set_.source_Storage_storage_offset`-shaped: copies out of an already-filled
`torch.UntypedStorage`) and explicitly refuses anything else by name. A `TorchDispatchMode` trace
of `a.set_(b)` for tensor `b` on upstream fires exactly one op:

```
aten.set_.source_Tensor
```

which is a distinct overload from the one implemented, and `tensor.rs` is forbidden territory this
round (`rust/torch_c/src/{aten.rs,tensor.rs,dtype.rs,flash.rs}`).

**Finding, by name: `aten.set_.source_Tensor` is a missing kernel/overload.** It is reached through
`torch.nn.utils.parametrizations.weight_norm` — a fairly generic utility (any architecture using
weight-normalized convolutions, not just VITS's WaveNet stack, would hit the same wall) — so this
is plausibly not a one-architecture-only gap.

`vits` gets no further: the model never finishes constructing, so there is no forward to attempt
and nothing else behind this wall has been explored.

---

## 3. `zoedepth` — two names fixed, then a real 2-D convolution kernel gap

**Verdict: constructs and gets one step into its backbone's forward, then stops on a kernel
capability gap (2-D convolution), after two genuine name-level fixes landed on the way.**

`ZoeDepthForDepthEstimation` is not a causal LM, so `AutoModel` does not apply cleanly here —
`AutoBackbone`/`ZoeDepthForDepthEstimation` is the model-specific class this architecture actually
exposes, and the input is an image, `pixel_values` of shape `(batch, channels, H, W)`. The toy
config recipe is transcribed from `transformers`' own `ZoeDepthModelTester`
(`tests/models/zoedepth/test_modeling_zoedepth.py`, fetched from the `transformers` GitHub tree
since the installed wheel does not ship its test suite) rather than hand-built, because ZoeDepth's
`backbone_config`/`neck_hidden_sizes`/`num_out_features`/`num_attractors` have to agree in length
with each other in ways not obvious from the config docstring alone — a hand-built attempt hit
three different `IndexError`/`ValueError`s before this was found. Backbone: `Dinov2Config`, 2
layers, hidden size 4, `out_features=["stage1","stage2"]`. **Constructs and forwards cleanly on
upstream torch**: `(2, 32, 32)` predicted depth, non-degenerate sum.

### 3.1 Two names fixed on the way, both in territory

**`TensorBase.new_tensor`.** `Dinov2`'s `_init_weights` calls `init.trunc_normal_` on every linear
and conv weight (`transformers/initialization.py`), whose CPU bounds path
(`torch/nn/init.py::_no_grad_trunc_normal_`) reads `tensor.new_tensor(a, device="cpu").item()`.
None of the twenty in ARCH20.md exercises `trunc_normal_`'s truncated-bounds branch during
construction, so this member was never called before. Measured with a `TorchDispatchMode` logger:
`x.new_tensor(5.0, device="cpu")` fires exactly `aten.lift_fresh.default` — the same single op
`torch.tensor(...)` makes. Not a new kernel: `_install_tensor_conversions` (`bootstrap.py`) now
builds it from the same two primitives `_tensor_factory`'s `torch.tensor` uses
(`module._tensor_new_from_data` + `dispatch("aten.lift_fresh.default", ...)`), defaulting
`dtype`/`device` from the receiver rather than from nothing, and refusing `requires_grad=True`/
`pin_memory=True` the same way `torch.tensor` does. (First attempt called `module.tensor(...)`
directly and failed — `module` here is `_C`, and `torch.tensor` lives on `varfns`, harvested onto
the top-level `torch` module only after `_C` finishes loading, so `_C` itself has no `.tensor`.
Fixed by inlining the two primitives instead of forwarding to a name that does not exist yet at
that point in bootstrap.)

**`torch.conv2d`.** `Dinov2`'s patch embedding is an ordinary `nn.Conv2d`, and `F.conv2d` binds
straight to `torch.conv2d` the same way `F.conv1d` binds to `torch.conv1d` (ARCH20.md §4, `mamba`'s
wall). Same shape of gap, same fix shape: `aten::conv2d` is `CompositeImplicitAutograd`, and a
`TorchDispatchMode` trace on 2.13.0 shows it firing exactly one record —
`conv2d(x, w, b, stride=2, padding=1, dilation=1, groups=1)` becomes
`convolution(x, w, b, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)` — the kernel
(`aten.convolution.default`) was already implemented and golden-compared; only the spelling was
missing. Added beside `conv1d` in `bootstrap.py`, same structure, widening a scalar
`stride`/`padding`/`dilation` to **two** elements instead of one (measured in the trace above) and
refusing `padding="same"` component-wise wherever `dilation[i]*(kernel[i]-1)` is odd on either
axis, for the same reason `conv1d` refuses it.

### 3.2 The wall that is not in territory

With both of the above fixed, the composite correctly reaches `aten.convolution.default` with a
4-D input — and the kernel itself refuses it:

```
NotImplementedError: aten.convolution.default: only 1-D convolution (3-D input, (batch, channels,
length)) is implemented in torch._C shim, got 4-D
```

**Finding, by name: `aten.convolution.default` only handles the 1-D case (3-D input).** 2-D
convolution (4-D input, the ordinary image case) is not implemented. This is a capability gap
inside an existing kernel, in `aten.rs`, forbidden territory this round. `zoedepth` — and any other
architecture with a real `nn.Conv2d` in its forward, which patch-embedding vision backbones
generically have — stops here regardless of anything wirable in `bootstrap.py`.

`zoedepth` gets no further: nothing behind 2-D convolution in the backbone (or the neck/head that
consumes its output) has been reached or explored.

---

## 4. `sew_d` — blocked at construction, on the legacy `torch.Tensor(int)` constructor

**Verdict: does not construct.** `SEWDModel` is `Wav2Vec2`-shaped audio input
(`input_values`, a raw waveform, `(batch, samples)`) through a conv feature extractor into a
DeBERTa-v2-style disentangled-attention encoder — `modeling_sew_d.py`'s
`DisentangledSelfAttention` is line-for-line the same code as `deberta_v2`'s, including the
unconditional `torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)`
at `modeling_sew_d.py:731` (§1 above). Toy config (tiny hidden size, two encoder layers, two short
conv-feature-extractor stages) **constructs and forwards on upstream torch**: `(1, 39, 32)` hidden
states, non-degenerate sum.

Under the shim, it never reaches that `sqrt` wall — a different, earlier one stops construction
outright:

```
self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())
NotImplementedError: torch._C shim: TensorBase(...) takes an existing tensor to re-wrap; upstream's
legacy `torch.Tensor(int)` storage constructor is not implemented
```

This parameter is created unconditionally in `SEWDModel.__init__`, whether or not
`apply_spec_augment` is set — turning it off (as this toy config does, to keep construction small)
skips *using* it during forward, not creating it. So no toy config on this architecture avoids the
wall.

**Not fixable in territory.** `torch.Tensor(n)` — upstream's legacy "allocate n uninitialized
elements" constructor, distinct from `TensorBase(existing_tensor)` (which re-wraps a tensor that
already exists) — is refused **in Rust**, at `#[new] fn py_new` in
`rust/torch_c/src/tensor.rs:920-929`, the PyO3-generated `__new__` for the native `TensorBase`
type. `torch.Tensor` itself (the subclass users actually construct) is `class Tensor(TensorBase)`
in the *vendored* `torch/_tensor.py`, which is out of bounds for a different reason
(`torchnative/src/main/torch` is the vendored tree, absolutely off-limits this round). Both places
that could plausibly grow a `__new__` override are therefore forbidden territory for this brief —
`tensor.rs` explicitly, the vendored tree by the brief's own rule. `bootstrap.py` has no hook into
`TensorBase.__new__` the way it has hooks into ordinary members (`setattr(tensorbase, name, fn)`
adds *methods*, not the type's constructor slot).

**Finding, by name: `torch.Tensor(int)` / `TensorBase.__new__` does not support the legacy
uninitialized-storage constructor.** Semantically this would be `torch.empty(n)` under the hood
(measured: upstream's `torch.Tensor(3)` and `torch.empty(3)` are both uninitialized float32 storage
of the same shape, and `torch.Tensor(3, 4)` and `torch.empty(3, 4)` agree the same way) — so the fix
is small in principle, but it is a change to `#[new]` on `PyTensorBase` in `tensor.rs`, which is
this round's forbidden file, not a `bootstrap.py`-reachable name.

`sew_d` gets no further: nothing behind this wall (including the `sqrt` wall it shares with
`deberta_v2`, and whatever comes after that) has been reached or explored for this architecture.

> **Closed, and one sentence above is wrong — corrected 2026-09-02, docs/bindings/CTOR.md.**
> `sew_d` constructs (49 weights, `masked_spec_embed` `(32,)` float32). The size form was
> implemented in `tensor.rs::py_new`; this section's "forbidden file" was forbidden to *that*
> round's brief, not permanently. The data forms are in `bootstrap.py`.
>
> The wrong sentence is *"Both places that could plausibly grow a `__new__` override are
> therefore forbidden territory"*, and with it the reasoning that `torch.Tensor` is unreachable
> because `torch/_tensor.py` is vendored. **Measured: `torch.Tensor` inherits `__new__` from
> `TensorBase`, defines none of its own, and is a settable heap type** — so it takes a `__new__`
> by `setattr` without the vendored file being touched, which is what `bootstrap.py` now does at
> the `_initExtension` hook. Being vendored blocks *editing*, not *patching*.
>
> The sentence after it — *"`bootstrap.py` has no hook into `TensorBase.__new__` the way it has
> hooks into ordinary members"* — is **true**, and beside the point: the override does not go on
> `TensorBase`, and docs/bindings/CTOR.md §1.1 measures why it must not. Putting it there makes the native
> allocator permanently unreachable from Python and takes `Parameter` with it.
>
> Recorded here rather than edited in place, per this file's convention and docs/verification/AUDIT.md's
> finding: the recurring failure is a later commit closing a gap while the document that named
> it goes on saying it is shut.

---

## 5. `sam3_video` — a different shape of architecture, and a missing arithmetic kernel

**Verdict: does not construct. Blocked before the detector submodule finishes building, on a
missing kernel (`aten.remainder`).**

This one is not causal-LM-shaped, is not a single model in the `AutoModel` sense, and — unlike the
other 25 — **has no toy `ModelTester` anywhere in `transformers`' own test suite**: `sam3_video`'s
only tests are `@slow` integration tests against the real `facebook/sam3` checkpoint
(`tests/models/sam3_video/test_modeling_sam3_video.py`, fetched from the `transformers` GitHub tree
since the installed wheel ships no tests). Neither does its tracker component,
`sam3_tracker_video`. This is itself a finding about the architecture, not a shortcut taken here:
`Sam3VideoConfig` composes a full `Sam3Config` (a SAM3 detector: ViT vision backbone, CLIP-shaped
text encoder, DETR encoder/decoder, geometry encoder, mask decoder) and a full
`Sam3TrackerVideoConfig` (a SAM2-shaped memory-conditioned video tracker), and the top-level
`forward` does not take a tensor — it takes a stateful `Sam3VideoInferenceSession` built by
`Sam3VideoProcessor.init_video_session` from a *video*, and each call advances that session by one
frame.

A toy config was still built, because a `ModelTester` exists for each of the two constituent
models even though not for their composite: `Sam3ModelTester`
(`tests/models/sam3/test_modeling_sam3.py`) for `detector_config`, `Sam3TrackerModelTester`
(`tests/models/sam3_tracker/test_modeling_sam3_tracker.py`, the non-video tracker, whose config
shape `sam3_tracker_video`'s mostly mirrors field-for-field) for `tracker_config`. Assembled, this
constructs in under 2 seconds on upstream torch (589K params, vs. ~860M and 18 seconds for the
default-sized config tried first) — confirming the recipe is structurally valid before trying it
under the shim.

**Under the shim, construction fails before `detector_model` finishes building** — inside
`Sam3ViT`'s rotary position embedding, which needs `x_positions = (flattened_indices % end_x) * scale`:

```
Sam3VideoModel.__init__ -> AutoModel.from_config(detector_config) -> Sam3Model.__init__
  -> Sam3VisionModel.__init__ -> AutoModel.from_config(backbone_config) -> Sam3ViTModel
  -> Sam3ViTLayer -> Sam3ViTRotaryEmbedding.__init__          modeling_sam3.py:428
NotImplementedError: not implemented in torch._C shim: TensorBase.__mod__
```

**A missing kernel, not a missing name.** A `TorchDispatchMode` trace of `x % 3` and
`x % torch.tensor(3)` on upstream fires `aten.remainder.Scalar` and `aten.remainder.Tensor`
respectively — and grepping `aten.rs`, `overloads.json`, and `methods.json` for `remainder` finds
nothing: no kernel exists to wire `__mod__` to. (`fmod` is absent the same way, for the same
operation on the signed-remainder convention rather than Python's.) `tensor.rs`/`aten.rs` are
forbidden territory this round.

**Finding, by name: `aten.remainder.Scalar` and `aten.remainder.Tensor` are missing kernels.**
`__mod__`/`__rmod__` would be a `methods.json` one-liner once either exists — there is simply
nothing to bind them to yet.

**What this does and does not cover.** The detector's own forward (`Sam3Model`, called directly as
`m.detector_model(pixel_values=..., input_ids=..., attention_mask=...)`) is a plain tensor-in,
tensor-out call and **does forward correctly on upstream torch** with this toy config (checked
separately, before the shim run above, since it seemed the closer of the two components to a
"straightforward forward"). It was not reached under the shim because construction itself dies
first, in that same vision backbone, on `__mod__`. The tracker side was not explored as deeply:
`Sam3TrackerVideoModel.forward` (and everything else at every level of the tracker stack) also
takes a session-shaped argument, not a plain tensor, mirroring the top-level model rather than
offering a simpler path in.

`sam3_video` gets no further: nothing behind `__mod__` in the vision backbone, and nothing in the
detector's DETR/geometry stages, the tracker, or the video-session orchestration, has been reached.

---

## 6. Operator coverage, kept separate from "actually forwards"

README already separates two claims — "operator coverage" (traced on upstream, diffed against
`_aten_implemented()`) and "actually forward" (a real run through the built shim) — and says
conflating them was a real mistake in an earlier round of this project. Kept separate here too:
everything in §1-5 above is the second claim, forward-attempted-on-the-shim. This section is the
first, run only on **upstream** torch (`docs/architectures/ARCH20.md`'s method 1, §0), for the same six toy
configs used in §1-5, so the new denominator carries an operator number as well as a forward
number.

| architecture | ops dispatched | missing from `_aten_implemented()` (139 ops) |
|---|---:|---|
| `deberta` | 43 | `aten.repeat.default`, `aten.sqrt.default`, `aten.zeros.default` |
| `deberta_v2` | 38 | `aten.repeat.default`, `aten.sqrt.default`, `aten.zeros.default` |
| `vits` | 50 | `aten._weight_norm_interface.default`, `aten.clamp_min.default`, `aten.flip.default`, `aten.leaky_relu.default`, `aten.ones_like.default`, `aten.randn_like.default`, `aten.sigmoid.default` |
| `zoedepth` | 35 | `aten.add.Scalar`, `aten.upsample_bilinear2d.default` |
| `sew_d` | 47 | `aten._weight_norm_interface.default`, `aten.avg_pool2d.default`, `aten.erf.default`, `aten.masked_fill.Tensor`, `aten.native_group_norm.default`, `aten.repeat.default`, `aten.sign.default`, `aten.sqrt.default` |
| `sam3_video` (`detector_model` forward only) | 54 | `aten.all.default`, `aten.div.Tensor_mode`, `aten.log2.default`, `aten.repeat.default`, `aten.sigmoid.default`, `aten.sign.default` |

**None of the six reach zero missing operators**, unlike the twenty in ARCH20.md, which all do. So
this widened sweep's "operator coverage" claim, honestly stated, is **20 of 26 at zero missing
operators**, not 26 of 26.

**This table demonstrates ARCH20.md §0.3's blind spot directly, on `sam3_video`.** The actual first
wall the shim hits (§5) is `TensorBase.__mod__`/`aten.remainder`, inside
`Sam3ViTRotaryEmbedding.__init__` — construction, not forward. The trace above wraps only
`m.detector_model(...)`, the forward call, exactly as ARCH20.md's method does, and it does not see
`aten.remainder` at all — the same way it never saw `bert`'s `_get_deterministic_algorithms` wall,
which also fired during construction (ARCH20.md §2). A reader who only ran this table's kind of
sweep on `sam3_video` would report a *different* first blocker (`aten.all.default` or whichever
forward-time op sorts first) than the one that actually stops it.

`aten.sqrt.default` and `aten.repeat.default` recur across unrelated architectures (`sqrt` in every
DeBERTa-family model; `repeat` in `deberta`, `deberta_v2`, `sew_d`, `sam3_video`) — consistent with
§1's read that `sqrt` is a foundational gap, not a one-model quirk. `_weight_norm_interface`
appearing in both `vits` and `sew_d`'s traces, alongside `aten.set_.source_Tensor` from §2's
construction-time wall, means `weight_norm` costs **two** separate kernels here, not one: the
parametrize machinery needs `set_.source_Tensor` at registration time, and every forward of a
weight-normalized layer needs `_weight_norm_interface.default` to recompute `g * v/||v||` — fixing
only one would still leave the other blocking.

---

## 7. Names added, in territory

| name | file | what it does |
|---|---|---|
| `torch._C._dynamo.eval_frame.set_eval_frame` | `bootstrap.py` | get-and-set state cell (was un-set, raised `NotImplementedError`) — unblocked `torch.manual_seed` after `transformers` imports `torch._dynamo` |
| `torch._C._dynamo.eval_frame.set_eval_frame_isolate_recompiles_id` | `bootstrap.py` | same shape, added alongside on the same reasoning, not yet observed called by anything |
| `TensorBase.new_tensor` | `bootstrap.py` | Python composite over the same two primitives `torch.tensor` uses (`_tensor_new_from_data` + `lift_fresh`), defaulting `dtype`/`device` from the receiver |
| `torch.conv2d` | `bootstrap.py` | Python composite over the existing `aten.convolution.default` kernel, mirroring `torch.conv1d`'s existing composite with 2-D-widened scalar arguments |

All four verified live via the `strings _C.abi3.so | grep -c <marker>` check (this brief's
staleness trap) after every rebuild, and via direct call (`torch.manual_seed(0)` after importing
`transformers`; `x.new_tensor(...)` and `torch.conv2d(...)` reached in the `zoedepth` toy run)
rather than only by re-running the aggregate gates, per the brief's own note that golden compares
by dispatch key and cannot see a missing spelling.

## 8. Every missing kernel found, by name

Not written — `aten.rs` is forbidden territory this round.

| kernel | blocks | how it was confirmed missing |
|---|---|---|
| `aten.sqrt.default` | `deberta`, `deberta_v2`, `sew_d` (§1, §4) | not `CompositeImplicitAutograd` upstream (`_dispatch_has_kernel_for_dispatch_key` false); grepped absent from `aten.rs`/`overloads.json`/`methods.json` |
| `aten.set_.source_Tensor` | `vits` (§2) | `tensor.rs::set_` implements only the storage-argument overload and refuses this one by name; `TorchDispatchMode` trace confirms `a.set_(b)` fires exactly this overload upstream |
| `aten.convolution.default`, 2-D case | `zoedepth` (§3) | kernel exists (1-D only, confirmed by its own refusal text: "only 1-D convolution ... is implemented"); 4-D input refused |
| `TensorBase.__new__` / legacy `torch.Tensor(int)` | `sew_d` (§4) | hardcoded refusal in `tensor.rs::py_new`; the class that could grow a Python-level override (`torch.Tensor`) is in the vendored tree, also forbidden |
| `aten.remainder.Scalar`, `aten.remainder.Tensor` | `sam3_video` (§5) | `TorchDispatchMode` trace of `x % 3` / `x % torch.tensor(3)` upstream; grepped absent from `aten.rs`/`overloads.json`/`methods.json` |
| `aten._weight_norm_interface.default` | `vits`, `sew_d` (§6) | absent from `_aten_implemented()`; a second, independent weight-norm gap from `set_.source_Tensor` above — this one is needed at forward time, not registration time |
| `aten.norm.ScalarOpt_dim` | `vits`, `sew_d` — **added by docs/kernels/KERNELS26.md §5.4** | **This table missed it, and §6's own blind spot is why.** `torch.norm_except_dim` is a *composite* (so no op name appears in the source) and it is called from `register_parametrization` at **construction** time, while the trace that produced the §6 table ran on a forward. So `weight_norm` costs **three** kernels here, not two — and `ParametrizationList.__init__` swallows the `NotImplementedError` with its own `except NotImplementedError: pass`, so the failure surfaces 200 frames away as a `TypeError` naming no kernel at all |
| `aten.repeat.default` | `deberta`, `deberta_v2`, `sew_d`, `sam3_video` (§6) | absent from `_aten_implemented()`, recurs across four of the six |
| `aten.zeros.default`, `aten.clamp_min.default`, `aten.flip.default`, `aten.leaky_relu.default`, `aten.ones_like.default`, `aten.randn_like.default`, `aten.sigmoid.default`, `aten.add.Scalar`, `aten.upsample_bilinear2d.default`, `aten.avg_pool2d.default`, `aten.erf.default`, `aten.masked_fill.Tensor`, `aten.native_group_norm.default`, `aten.sign.default`, `aten.all.default`, `aten.div.Tensor_mode`, `aten.log2.default` | one or more of the six (§6 table) | absent from `_aten_implemented()`, each confirmed by the operator-coverage trace in §6 rather than assumed |

**Six of these were taken in docs/kernels/KERNELS26.md** — `sqrt`, `repeat`,
`remainder.{Scalar,Tensor}`, `set_.source_Tensor`, the legacy `TensorBase(int)`
constructor and `convolution.default`'s 2-D case — taking the sweep from 20/26
to **22/26** (`deberta` and `deberta_v2` forward). The rest of this table stands.

> **Correction (docs/DOCWATCH round, 2026-09):** "the rest of this table stands" is itself now
> stale. Re-checking every op the §8 table named as missing against live `_aten_implemented()`
> today finds only **four** still absent: `aten.set_.source_Tensor` (the KERNELS26 credit above
> was for `remainder`/`sqrt`/etc., not this overload — it is still unresolved),
> `aten.zeros.default`, `aten.randn_like.default`, `aten.masked_fill.Tensor`. Every other op this
> table named as missing is implemented now — `aten.remainder.Scalar`, `aten.remainder.Tensor`,
> `aten._weight_norm_interface.default`, `aten.clamp_min.default`, `aten.flip.default`,
> `aten.leaky_relu.default`, `aten.sigmoid.default`, `aten.ones_like.default`, `aten.add.Scalar`,
> `aten.upsample_bilinear2d.default`, `aten.avg_pool2d.default`, `aten.erf.default`,
> `aten.native_group_norm.default`, `aten.sign.default`, `aten.all.default`,
> `aten.div.Tensor_mode`, `aten.log2.default`. Standing check (docs/verification/DOCWATCH.md):
> <!-- DOCWATCH: op-not-implemented aten.set_.source_Tensor -->
> <!-- DOCWATCH: op-not-implemented aten.zeros.default -->
> <!-- DOCWATCH: op-not-implemented aten.randn_like.default -->
> <!-- DOCWATCH: op-not-implemented aten.masked_fill.Tensor -->
> <!-- DOCWATCH: op-implemented aten.remainder.Scalar -->
> <!-- DOCWATCH: op-implemented aten._weight_norm_interface.default -->
> <!-- DOCWATCH: op-implemented aten.clamp_min.default -->
> <!-- DOCWATCH: op-implemented aten.leaky_relu.default -->
> <!-- DOCWATCH: op-implemented aten.native_group_norm.default -->
> <!-- DOCWATCH: op-implemented aten.div.Tensor_mode -->

**None of these are attempted here, including the ones that look like a one-line composite
(`aten.sqrt.default` over `pow(x, 0.5)` was considered and rejected in §1.2 — the same reasoning
applies to every entry above: a Python-level composite that computes a value upstream reaches
through a leaf kernel is inventing a computation path, exactly the mistake this brief's opening
paragraph is about).**

---

## 9. Verification

All of the following were re-run against the artefact rebuilt after every fix above, not only at
the end:

```
bash vendor/install_shim.sh                       exit 0
PYTHON=$PY sh rust/torch_c/pytests/run.sh          261 ok, 0 FAIL          exit 0
$PY tools/golden/compare.py                        4284/4284, ops=139, pending=1   exit 0
$PY tools/golden/compare.py --self-test             13 x 11, 0 problems    exit 0
$PY rust/torch_c/pytests/verify_schemas.py          4353/4353              exit 0
```

All five numbers are unchanged from the pre-existing baseline measured at the start of this round
(before any edit) — expected, since every fix in §7 is a Python-level spelling/member/surface name,
not a new kernel, and none of them is exercised by `overloads.json`/`methods.json`-driven schema
counting (the same reason `conv1d` did not move these numbers in ARCH20.md).

**The existing twenty still all forward**, re-run through `/tmp/arch7/sweep.py` (the same script
ARCH20.md's own measurement used, still on disk from that round) against the rebuilt artefact:

```
llama gpt2 qwen2 mistral gemma gpt_neox opt mpt starcoder2 stablelm olmo phi mixtral
bert bloom cohere falcon gpt_bigcode mamba persimmon
TOTAL 20/20
```

`git status --short` in the worktree shows exactly two changes: `rust/torch_c/src/bootstrap.py`
(modified) and `docs/architectures/ARCH26.md` (new) — nothing in `aten.rs`, `tensor.rs`, `dtype.rs`, `flash.rs`,
`tools/golden/cases.py`, `tools/wheel/`, or the vendored tree.

## 10. The new sweep denominator

**26** (the 20 in ARCH20.md + `deberta`, `deberta_v2`, `vits`, `zoedepth`, `sew_d`, `sam3_video`).

| | of 20 (ARCH20.md) | of 26 (this document) | after docs/kernels/KERNELS26.md |
|---|---|---|---|
| operator coverage (zero missing, traced on upstream) | 20/20 | **20/26** | 20/26 |
| actually forwards through the shim | 20/20 | **20/26** — none of the six new architectures forward yet | **22/26** — `deberta` and `deberta_v2` forward |

Both numbers move together here only because all six new architectures happen to fail both
measures; §6 above is the reminder that they are not the same measurement and do not have to agree
(`sam3_video`'s wall is invisible to the operator-coverage number entirely).

