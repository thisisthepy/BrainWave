# DEMAND — regenerating the work queue by running real models

Measurement round only. **Nothing in this round changed source** — `rust/torch_c/{aten.rs,tensor.rs,dtype.rs,flash.rs}`,
`bootstrap.py`, `tools/golden/cases.py`, `torchnative/src/main/torch/` (vendored) are all untouched.
`git status --short` in the worktree is empty throughout. Every finding below is a *candidate*
for the next round, not something applied here.

docs/architectures/ARCH26.md (26 architectures, `transformers` 5.15.1, toy `AutoConfig`s) is the existing
denominator. This round goes **wider**: 18 new architectures/tasks the 26 do not cover, plus one
feature path (`TorchnativeConfig` quantised load) explicitly asked for, run against **upstream
torch as the oracle**, not against each other.

---

## 0. The ranked list — build these next

Ranked by **how many distinct models hit each wall as their first blocker** in this round. Ties
broken by real-world reach (how many architecture *families*, not just the one model tried, use
the same code path).

> **Re-ranked 2026-09-02 (docs/architectures/DEMAND1.md).** Ranks 1, 3, 4 and 5 of the original list are
> **closed**, along with one of the honourable mentions. §0.1 below is what remains, renumbered;
> §0.2 records what closed and how. The original ranking is preserved in §0.2 rather than edited
> in place, because "which wall did each model hit" is the measurement this file exists to carry
> and rewriting it would lose the evidence.
>
> **Re-ranked again 2026-09-02 (docs/bindings/CTOR.md).** The renumbered rank 1 — the legacy
> `torch.Tensor(...)` constructor — is **closed**, and how it closed is worth reading, because
> the word this table used for it was wrong. It is recorded in §0.2 with the rest, and the
> three below move up one.

> **Re-ranked again 2026-09-02 (docs/architectures/DEMAND2.md).** Ranks 2, 3 and 4 of the list above are now
> **closed** too — everything the first re-ranking left open except rank 1. §0.1 is, once again,
> renumbered to what remains; §0.2 gets three more rows. Same reason as before for not editing
> the closed ranks' own rows in place: the vote counts and "why this is the cheapest/closest"
> reasoning are the measurement, not decoration.

### 0.1 What is still open, renumbered

| rank | gap | models that hit it | kind |
|---|---|---|---|
**Nothing on this document's list is open.** Its rank 1 (the legacy
`torch.Tensor(...)` constructor) closed in docs/bindings/CTOR.md, and ranks 2 to 4
(`squeeze.default`, `linalg_vector_norm.default`, `linspace`) closed in
docs/architectures/DEMAND2.md — two rounds that ran at the same time, each removing what it
had closed, which is why this section merged to empty.

That does **not** mean nothing is open. This file ranks the *first* sweep of
eighteen architectures; re-running them after five closures moved seven models
onto new walls, and that list lives in [`docs/architectures/DEMAND3.md`](../architectures/DEMAND3.md). Read
that one for what to build next — `max_pool2d`, `where.default` (the
single-argument overload), `hardtanh`, `adaptive_avg_pool1d`, `one_hot` — and
the `randint` divergence recorded at the end of it, which outranks all of them.


### 0.2 Closed, and by what

| was | gap | closed by |
|---|---|---|
| 0.1-1 | legacy `torch.Tensor(...)` constructor | **docs/bindings/CTOR.md.** One function in `bootstrap.py`, installed on `torch.Tensor` at the existing `_initExtension` hook. **Not structural, and that is the finding**: this table said the only class that could carry a Python-level `__new__` is in the vendored tree, and concluded the gap was therefore unreachable. `torch.Tensor` *is* that vendored class — `class Tensor(torch._C.TensorBase)`, MRO `(Tensor, TensorBase, object)` — but it **inherits** `__new__` and is a settable heap type, so it can be given one without editing the file that declares it. Being vendored blocks *editing*, not *patching*, and `bootstrap.py` patches classes it does not own everywhere. The real obstacle was ordering (`_C` imports before `torch.Tensor` exists) and the hook for it already existed and already ran on this exact class. Shapes implemented: `Tensor()`, `Tensor(2, 3)`, `Tensor(torch.Size(...))`, `Tensor(existing)`, `Tensor(sequence)`, `Tensor(ndarray)`, `device=`; refusals transcribed from upstream by wording. `torch.Tensor(int)` was already native by `3c53d16`. **Two divergences recorded rather than closed** (CTOR.md §3.3): the ndarray path copies where upstream aliases (the same one §0.2's `as_tensor` row records), and the size form's bytes are zeros where upstream's are uninitialised. **`ops covered` is unchanged at 185** — no kernel; every form routes to `lift_fresh` or to the existing native size constructor, so the golden harness is structurally blind to it and six vendored-tree road tests are what cover it. `pegasus` and `sew_d` both construct and match upstream bit for bit. |
| 1 | `torch.batch_norm` / `aten.native_batch_norm.default` | **docs/architectures/DEMAND1.md §1.** Kernel in `aten.rs` (`native_batch_norm_default`), spelling as a `bootstrap.py` composite beside `layer_norm`/`group_norm` — not an `overloads.json` entry, because `aten::batch_norm` is `CompositeImplicitAutograd` and never fires. The train/eval split, the in-place running-statistic update, the biased-for-output / **unbiased**-for-running-variance divergence, the `(0,)`-shaped `save_mean`/`save_invstd` in eval, and upstream's **fused** `x*alpha + beta` affine (which the golden harness caught, on the constant-input case) are all measured and golden-compared. `capture.rs` grew an argument-aware guard: this op mutates and its name does not say so. `nn.BatchNorm2d`, `1d` and `3d` all forward. |
| 2 | `aten.squeeze.default` (and `aten.squeeze.dims`, found to have the identical hole while landing `.default`) | **docs/architectures/DEMAND2.md.** `squeeze` was *declared* in both `overloads.json` and `methods.json` with three overloads and `aten.rs`'s dispatch `match` had an arm only for `.dim`; `.default` and `.dims` are now both wired. `.default` removes every axis of size 1 (not just one); `.dims` removes each *named* axis that is size 1 and leaves the rest — a named axis that is not size 1 is a no-op, the same rule `.dim` carries, and an **empty `dim` list on `.dims` is a no-op**, not "every axis" the way `norm.ScalarOpt_dim`/`linalg_vector_norm`'s empty `dim` is. `.dims`' repeated-dim refusal (`dim 0 appears multiple times in the list of dims`) is checked **before** any size is looked at, measured: `squeeze(x, dim=(1,1))` refuses even though axis 1 is not size 1. `mbart`'s `shift_tokens_right` calls plain `squeeze()`; no measured model called `.dims`. |
| 3 | `aten.linalg_vector_norm.default` | **docs/architectures/DEMAND2.md**, building on the full upstream measurement already recorded in docs/architectures/DEMAND1.md §5. A **distinct** leaf from `aten.norm.ScalarOpt_dim`, and the kernel **shares** that op's six-arm accumulate walk (`norm_pow_walk` in `aten.rs`) rather than re-deriving it, per docs/architectures/DEMAND1.md §7's stated reason not to diverge. What stays this op's own: the `dim=None`/`dim=[]` "every axis" rule (identical to `norm.ScalarOpt_dim`'s), the dtype-mismatch wording (`"linalg.vector_norm: Expected a floating point or complex tensor as input. Got {name}"`, no trailing "instead." the way `norm()`'s has), `dtype=` promotion with a narrowing refusal (`Half`/`BFloat16` cannot convert to each other despite being the same width), and the empty-reduction split — `ord=2` still answers `0.0`, but `ord=±inf` now **refuses**, with one message naming the empty axis (`"...on the dimension {d}because..."`, upstream's own missing space, transcribed) when `dim` was a non-empty list, and another (`"...on an empty tensor because..."`) when it was not. `F.normalize`'s four ops (`linalg_vector_norm.default, clamp_min.default, expand.default, div.Tensor`) all have kernels now; `sentence_embed` forwards and matches upstream end to end. |
| 4 | `torch.linspace` | **docs/architectures/DEMAND2.md.** Leaf kernel, fetched and read from upstream's `RangeFactoriesKernel.cpp::linspace_kernel` rather than guessed. `steps=0` answers empty (not an error); `steps<0` refuses (`"number of steps must be non-negative"`); `steps=1` answers `[start]` without reading `end`. The default dtype is always the default float dtype, never `int64` the way `arange`'s all-integral rule would suggest, and an explicit integer dtype **truncates toward zero**. The endpoint is exact by construction — upstream fills forward from `start` and *backward* from `end`, split at `steps // 2`, and the `Float`/`Double` arms fold each half's multiply-add into one hardware FMA (this shim's own `f32`/`f64::mul_add`) to match the compiler's contraction of upstream's literal `a + b*c`; `Half`/`BFloat16` do not fuse and are narrowed after every individual operator call instead, matching `at::Half`/`at::BFloat16` arithmetic. `ConvNextModel.__init__`'s stochastic-depth rate schedule is the caller. |
| 5 | `torch.full_like` / `aten.full_like.default` | **docs/architectures/DEMAND1.md §2.** Kernel reusing `full`'s own `filled_block` rather than a second fill; `overloads.json` entry. The rule that separates it from `full` — dtype from the *reference tensor*, not inferred from the fill value — is what the golden cases are built around. |
| 6 | `torch.as_tensor` | **docs/architectures/DEMAND1.md §4.** A spelling, as predicted: a `bootstrap.py` composite branching between `lift_fresh` (new data) and `_to_copy` (a tensor needing a cast), with the identity case — `torch.as_tensor(t) is t` — returning the receiver. **One difference is recorded rather than closed**: upstream's ndarray path shares memory with the array and this one copies, because this shim's tensors do not wrap foreign buffers. |
| 7 | `TensorBase.new_zeros` / `aten.new_zeros.default` | **docs/architectures/DEMAND1.md §3.** One function in `aten.rs` now serves `new_ones` and `new_zeros`; `methods.json` entry. |
| HM | `torch.meshgrid` | **docs/architectures/DEMAND1.md §6.** A spelling, as predicted — a `bootstrap.py` composite over `view` + `expand`, both long implemented. `"xy"` is done by swapping the first two inputs and swapping the outputs back, which is what upstream's op trace shows and what a transpose-based implementation would have got wrong invisibly (the shapes and values agree; only the trace differs). |

Closing the first five (native_batch_norm, full_like, as_tensor, new_zeros, meshgrid) moved
`ops covered` from 168 to 171 (three kernels; `as_tensor` and `meshgrid` added none, being
spellings over ops that were already there — which is exactly why they needed smoke coverage
through the vendored tree instead, GOLDEN.md's blind spot). Closing ranks 2, 3 and 4 above moved
it from 185 to 189: `squeeze.default`, `squeeze.dims`, `linalg_vector_norm.default` and
`linspace.default` are each a real `_aten_implemented()` entry, one kernel apiece (`squeeze`'s
two new overloads share their existing `.dim` kernel's candle call, `linalg_vector_norm` shares
`norm.ScalarOpt_dim`'s walk) — four entries, four kernels, +4.

**Honorable mentions still open, one vote each** (`linalg_vector_norm`, `squeeze.default` and
`linspace` were promoted into §0.1 above and are no longer listed here). The original text of the
two that closed is kept in §0.2. For reference, the mentions as first written were:
`aten.linalg_vector_norm.default` (`sentence_embed`'s `F.normalize`, missing kernel — a **distinct**
leaf from the already-implemented `aten.norm.ScalarOpt_dim`, confirmed by
`_dispatch_has_kernel_for_dispatch_key` returning `False` for it too); `torch.meshgrid` (`swin`'s
relative-position-index construction, missing **spelling** — `CompositeImplicitAutograd` upstream,
decomposing purely into `aten.view.default` + `aten.expand.default`, both already implemented, so
this is a same-shape fix to ARCH26 §7's `conv2d`/`new_tensor` composites); `aten.squeeze.default`
(`mbart`'s `shift_tokens_right`, **the GOLDEN.md blind-spot shape**: `squeeze` is *declared* in both
`overloads.json` and `methods.json` with three overloads — `squeeze()`, `squeeze.dim`,
`squeeze.dims` — but `aten.rs`'s dispatch `match` only has an arm for `squeeze.dim`; the no-arg
overload looks present in the name tables and is not reachable); `torch.linspace` (`convnext`'s
stochastic-depth rate schedule, missing kernel, leaf upstream — the generic "N vision backbones
with a `drop_path_rate` schedule" pattern, same family as `swin`'s missing `meshgrid` in that both
are vision-transformer/CNN construction-time utilities rather than forward-path arithmetic).

---

## 1. Method

- `transformers` 5.15.1, `torch` 2.13.0 (upstream, the vendored/shim build's origin), toy
  `AutoConfig`s (small hidden size, 1–2 layers, few heads, tiny vocab) — coverage, not model
  quality, same recipe as ARCH20.md/ARCH26.md.
- Every script printed `"shim" if hasattr(torch._C, "_aten_implemented") else "upstream"` as its
  first line, both sides, every run. Transcribed literally in every log; no run of the 18
  architectures × 2 sides (36 runs) plus 8 numeric dumps × 2 sides (16 runs) plus the quant-load
  run (2 runs) printed the wrong label.
- Every run wrapped in a 120s `SIGALRM` per model. **No hang observed** — every run (upstream and
  shim, forward and construction failures alike) returned in well under a second, toy configs
  being tiny. `float8_e4m3fn` specifically (the brief's named hang case) was not exercised this
  round — no target below asked for a non-default dtype.
- Each side run as its own subprocess (`PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1`
  for the shim; `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL` for upstream) — never both `torch`
  variants in one interpreter.
- Built via the instructed pipeline: `cargo build --release` in `rust/torch_c` with
  `CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-sweep`,
  `TORCH_C_ARTEFACT=.../release/lib_C.dylib` exported before every gate run (docs/verification/GOLDEN.md §6's
  own trap — a build without the matching artefact env var silently measures a stale binary).
  `bash vendor/install_shim.sh` after.
- Scratch scripts under `/tmp/sweep/` (not committed, matching ARCH26/KERNELS26 convention):
  `check_common.py` (18-architecture pass/fail sweep, shared by both sides via `PYTHONPATH`),
  `dump_model.py` + `compare_dumps.py` (numeric comparison for the 8 that forward), `quant_build_ckpt.py`
  + `quant_load_test.py` (the `TorchnativeConfig` path).
- **"Matches upstream" method**: `torch.manual_seed(0)` immediately before model construction on
  both sides (KERNELS26.md §2.4's method — the shim's RNG must reproduce upstream's init bit for
  bit for this to mean anything, and it does, see §2 below), then a forward on identical
  hand-built input tensors (`torch.manual_seed(1)` before any randomly-generated input, so both
  sides see the same pixels/waveform), comparing every `state_dict` tensor and the full output by
  `.tolist()`.

## 2. Model table

"loads" = model class constructs from a toy `AutoConfig`. "forwards" = a real forward call
returns. "matches upstream" = numeric comparison done in §1's method; "n/a" where forward itself
did not succeed, so there is nothing to diff yet — GOLDEN.md's whole point, restated by the
brief: a model that runs and disagrees is worse than one that refuses, so this column is never
guessed.

| model (task) | loads | forwards | matches upstream | refusal (exact) |
|---|---|---|---|---|
| `bert` (`BertForMaskedLM`, `AutoModelForMaskedLM`) | yes | yes | **yes** — 28/28 weights identical, max weight diff 1.9e-06 (accumulation noise), output max abs diff 8.9e-08 (float32 eps 1.19e-07) | — |
| `roberta` (`RobertaForMaskedLM`) | yes | yes | **yes** — 28/28 weights identical, max abs diff 7.5e-08 | — |
| `albert` (`AlbertForMaskedLM`) | yes | yes | **yes** — 30/30 weights identical, max abs diff 7.5e-08 | — |
| `vit` (`ViTForImageClassification`) | yes | yes | **yes** — 24/24 weights identical, max abs diff 3.7e-08 | — |
| `clip` (`CLIPModel`, text+vision) | yes | yes | **yes** — 46/46 weights identical, max abs diff 4.5e-08 | — |
| `wav2vec2` (`Wav2Vec2ForCTC`) | yes | yes | **yes, with a caveat** — 32/32 weights identical, max abs diff 2.3e-07 (still float32-eps scale); the *relative*-diff figure (2.3e-02) looked alarming in isolation and is a near-zero-denominator artefact, not drift — flagged rather than silently dropped, per the brief's own "a model that runs and produces different numbers is worse than one that refuses" |
| `qwen2_moe` (`Qwen2MoeForCausalLM`, small MoE) | yes | yes | **yes** — 19/19 weights identical, max abs diff 4.5e-08 | — |
| `whisper` (`WhisperForConditionalGeneration`, forward only) | yes | yes | **yes** — 51/51 weights identical, max abs diff 1.2e-07 | — |
| `whisper` (`.generate()`) | yes | forward yes, **generate no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch.as_tensor(...)` — `generation_whisper.py:1608`, `_retrieve_init_tokens` |
| `t5` (`T5ForConditionalGeneration`) | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch.full_like(...)` — `modeling_t5.py:258`, `_relative_position_bucket` |
| `switch_transformers` (MoE-T5, `AutoModelForSeq2SeqLM`) | yes | **no** | n/a | same as `t5`: `torch.full_like`, same shared code |
| `bart` (`BartForConditionalGeneration`) | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: TensorBase.new_zeros` — `modeling_bart.py:62`, `shift_tokens_right` |
| `mbart` (`MBartForConditionalGeneration`) | yes | **no** | n/a | `NotImplementedError: aten op not implemented in torch._C shim: aten.squeeze.default` — `modeling_mbart.py:77`, `shift_tokens_right` (different shift-right code than `bart`'s) |
| `pegasus` (`PegasusForConditionalGeneration`) | **no** (construction) | n/a | n/a | `NotImplementedError: torch._C shim: TensorBase(...) ... building from data, torch.Tensor(ndarray), is a third form and is not implemented` — `modeling_pegasus.py:91`, `PegasusSinusoidalPositionalEmbedding.create_weight` |
| `resnet` (`ResNetForImageClassification`) | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch.batch_norm(...)` — stem `BatchNorm2d` after a **successful** 2-D `conv2d` |
| `mobilenet_v2` (`MobileNetV2ForImageClassification`) | yes | **no** | n/a | same as `resnet`: `torch.batch_norm` |
| `convnext` (`ConvNextForImageClassification`) | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch.linspace(...)` — `modeling_convnext.py:219`, stochastic-depth rate schedule (construction time, not forward) |
| `swin` (`SwinForImageClassification`) | **no** (construction) | n/a | n/a | `NotImplementedError: not implemented in torch._C shim: torch.meshgrid(...)` — `modeling_swin.py:354`, `_create_relative_position_index` |
| `sentence_embed` (`bert` + mean-pool + `F.normalize`, the sentence-transformers shape) | yes | forward yes, **normalize no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch._C._linalg.linalg_vector_norm` — `torch/nn/functional.py:6100`, `normalize` |
| `TorchnativeConfig("q8_0")` load (toy 2-layer `llama`, self-built checkpoint) | yes | yes | not re-diffed this round (docs/graph/HFQUANT.md already established bit-identical vs. `quantize_`) | — dense and quantised loads both forward under the shim |

**7 of 18 architectures forward and match upstream** (`bert`, `roberta`, `albert`, `vit`, `clip`,
`wav2vec2`, `qwen2_moe`); **1 forwards but its `.generate()` does not** (`whisper`); **10 do not
forward** (`t5`, `switch_transformers`, `bart`, `mbart`, `pegasus`, `resnet`, `mobilenet_v2`,
`convnext`, `swin`, `sentence_embed`'s normalize step). The quantised-load feature path works.

**Combined with ARCH26's 26** (of which 22 forward per docs/kernels/KERNELS26.md), the widened denominator
this project has actually run is **44 architectures/tasks**, of which **29 forward** — narrower
than "26/26" or any round-specific number implies on its own, which is exactly why this round
exists.

## 3. What was not run, and why

- **Real checkpoints and real preprocessing.** Every vision/audio target used hand-built tensors
  (`torch.zeros`/`torch.randn` of the right shape) in place of `AutoImageProcessor`/
  `AutoFeatureExtractor` output. `pillow`, `sentencepiece`, `protobuf` were installed (small, pure
  dependencies, none of which touch `torch`) to keep tokenizer/config paths open, but no `soundfile`,
  `librosa`, `torchvision`, `torchaudio`, or `accelerate` was installed — real audio decoding and
  real image loading were never exercised, only the tensor-in/tensor-out model forward. A model
  that needs a working `WhisperFeatureExtractor` (log-mel spectrogram from a raw waveform, which
  wants `numpy`+`scipy`, present, but conventionally routes through `torchaudio` for resampling) was
  not checked end-to-end from an audio file.
- **`sentence-transformers` the package.** Not installed — it pulls `scikit-learn`/`scipy` and,
  more importantly, its own `torch` dependency pin, which is exactly the "transformers can replace
  torch" trap this round's brief calls out by name. `sentence_embed` above approximates the
  library's actual `SentenceTransformer.encode()` (mean-pooling + `F.normalize`, the standard
  recipe for `all-MiniLM`-class models) using plain `AutoModel`, which is architecturally identical
  to `bert` — so the *forward* half of this target adds no new evidence beyond `bert`'s own row; only
  the `normalize` tail is new information.
- **Object detection (DETR-shaped models).** Considered (would add more `batch_norm`/`conv2d`
  votes through a ResNet backbone plus new box-regression ops) and skipped for time — the toy-config
  recipe for DETR's `num_queries`/`d_model`/backbone composition is not as mechanical as the
  `AutoConfig.for_model` targets used here and would have cost disproportionate setup time for one
  more `batch_norm` vote already established by two other models.
- **`float8_e4m3fn` / other exotic dtypes.** Not attempted. The brief names this as a known hang
  case; none of this round's 19 runs touched a non-default dtype, so the per-run 120s alarm was
  never tested against a real hang, only against ordinary failures (which all resolved in
  sub-second time).
- **A real HF Hub checkpoint end-to-end** (tokenizer + real weights + generation against a known
  reference transcript/translation) — network access to `huggingface.co` was confirmed working
  (`curl`, and Python via `certifi`'s CA bundle — the venv's default `ssl` context could not verify
  TLS without it, worth flagging for anyone else scripting downloads here), and `HF_HOME` was
  pointed at `/Volumes/macMini/caches/hf` per the brief, but no download was actually made: every
  target that reached construction used a **self-built toy config**, and the one real-weights test
  (`TorchnativeConfig`) used a **self-built local checkpoint** (`save_pretrained` on upstream, 2-layer
  toy `llama`, ~100 KB) rather than a Hub download, to keep the round finishing quickly and to avoid
  a licence-gated or multi-hundred-MB checkpoint eating the external disk's 80 GB headroom
  unnecessarily. No checkpoint was skipped for being too large or gated — none was attempted.

## 4. Gates — tree unchanged, pasted in full

```
$ PYTHON=$PY sh rust/torch_c/pytests/run.sh
343 ok
SELF-TEST: PASS -- 20 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised
DOCWATCH: PASS -- 257/257 evaluated marker(s) hold

$ $PY tools/golden/compare.py
SUMMARY: 7763/7763 cases passed, 0 failed, ops covered=168, pending case builders=1

$ $PY tools/golden/compare.py --self-test
SELF-TEST: PASS -- 20 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised

$ $PY rust/torch_c/pytests/verify_schemas.py
SUMMARY: 4487/4487 table entries matched upstream, 0 failed
```

All four numbers match the brief's expected baseline (343 ok, DOCWATCH 257/257, 7763/7763
ops=168) exactly. `git status --short` was empty before, during (checked between models), and
after this round — nothing in `rust/torch_c/src/`, `bootstrap.py`, `tools/golden/cases.py`, or the
vendored tree moved.

---

## The ranking, as checks

Every op named above is asserted **absent**, so the day one of them lands this document
fails and has to be re-ranked rather than quietly describing a closed gap. That is the
failure mode this file exists to break: the last demand list went stale and the queue
was fed by ad-hoc probes instead, one of which misclassified fifteen names.

**It worked, on 2026-09-02, twice.** Three of the five markers below failed on the round that
implemented them, and failing is what forced §0 to be re-ranked instead of being left
describing gaps that had closed. The three that closed are now asserted **present** — the
same mechanism pointed the other way, so that a regression (a kernel removed, or removed
from `_aten_implemented()`) fails here too.

**It worked a second time the same day (docs/architectures/DEMAND2.md).** The two markers that were still
open — `linalg_vector_norm.default` and `squeeze.default` — both failed the moment their
kernels landed, exactly as designed, and are flipped below. `squeeze.dims` and
`torch.linspace` never had markers of their own (the previous round did not add one for
`linspace`, and `squeeze.dims` was not yet distinguished from `.default`); both get one now,
so a regression in either fails here too rather than only in `_aten_implemented()`'s own count.

Closed — asserted present now:

<!-- DOCWATCH: op-implemented aten.native_batch_norm.default -->
<!-- DOCWATCH: op-implemented aten.full_like.default -->
<!-- DOCWATCH: op-implemented aten.new_zeros.default -->
<!-- DOCWATCH: op-implemented aten.linalg_vector_norm.default -->
<!-- DOCWATCH: op-implemented aten.squeeze.default -->
<!-- DOCWATCH: op-implemented aten.squeeze.dims -->
<!-- DOCWATCH: op-implemented aten.linspace.default -->

The two spellings that closed have no kernel of their own to assert, so they are pinned by
name against upstream instead — the same check `hasattr gelu false` makes elsewhere, in the
other direction:

<!-- DOCWATCH: hasattr as_tensor true -->
<!-- DOCWATCH: hasattr meshgrid true -->

Nothing is still open under this mechanism: rank 1 (§0.1) is a structural refusal inside
PyO3's `#[new]` slot, not a missing aten kernel, so there is no `_aten_implemented()` name for
a marker to watch yet.

### Found while closing rank 1, and not ranked because no model asked for it yet

`Tensor.shape` returned a plain `tuple`, not `torch.Size`, for every tensor —
pre-existing, not from any round in this pair. `torch.Size` is a tuple subclass,
so indexing and unpacking were unaffected and nothing in the eighteen models
noticed; what would notice is `isinstance(x.shape, torch.Size)` or a call to
`Size.numel()`. Recorded here rather than ranked, because this file ranks by how
many models want a thing and the answer was none.

**Closed anyway — docs/numerics/SCALAR2.md §5.** It was ranked at nothing and cost
almost nothing, which turned out to be the same fact: the `Size` class already
existed in `bootstrap.py` with its `numel()`, and the only missing piece was
that `TensorBase.shape` never constructed it. A registration (`_set_size_class`,
the same shape as `_set_tensor_class` beside it) rather than a new type. Four
things that were *not* obvious came out of measuring the rest of the surface:
`repr` is `torch.Size([2, 3])` and not the tuple's own spelling, the type is
closed under slicing / `+` / reflected `+` / `*`, `__qualname__` had to be set
or the class was unpicklable, and `stride()` is a plain tuple upstream and had
to be left one. `ops covered` is unchanged at 203 — no kernel.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_autograd_shape present -->
