# GAPS — what refuses, why, and which of it is actually missing

Measured 2026-09-12 on `work/gaps`, host `darwin/arm64`, CPython 3.13, the
vendored tree rebuilt from `vendor/vendor_torch.sh` + `vendor/install_shim.sh`
against upstream torch 2.13.0 in `/Volumes/macMini/caches/spike-venv`.

This document exists because counting refusals is not a measurement. A count of
`raise NotImplementedError` per subpackage gives

    nn/federated 10    adapt 5    delta 1    quant 0    distributed 0

and reads like "adaptation and federated learning are stubs" — the two things
`torchnative/__init__.py`'s first line names as the package's purpose. Both
numbers are almost entirely **abstract bases and deliberate refusals of
degenerate cases**. Of the fifteen, **one** was a gap a user could walk into, and
it is not in either subpackage's arithmetic: it was `adapt`'s stage-0 road,
which nothing supplied. **It is closed** (§3.4, 2026-09-13). The gaps that matter were found somewhere else
entirely — the *public surface*, where a name that exists and does nothing
costs more than a name that is absent.

House style is `docs/devices/QNNOPS.md` §5: every classification carries its
evidence, and what could not be settled is named in §5 rather than guessed at.

---

## 1. Abstract bases — a subclass in this repo supplies it. Not a gap.

| site | method | who supplies it |
|---|---|---|
| `adapt/__init__.py:79` | `Method.select` | `Tent.select` (`adapt/__init__.py:147`), which walks `named_modules()` through `_is_normalisation` |
| `adapt/__init__.py:88` | `Method.objective` | `Tent.objective` (`adapt/__init__.py:158`), mean prediction entropy from `softmax`/`log_softmax` |
| `device/__init__.py:303` | `Device.availability` | five subclasses: `availability` is defined again at `device/__init__.py` lines 369, 388, 439, 467 and 661 (`cpu`, `mps`, `vulkan`, `cuda`, `npu`) |

**Evidence.** `grep -n "def availability"` on `device/__init__.py` returns the
base plus five overrides; `Tent` is the only `Method` subclass in the tree and
defines both declared methods. Nothing here is reachable without a user
subclass that chose not to implement its own contract.

Three more sites are *exception classes*, not refusals: `RefoldRefused`,
`DecompositionRefused`, `FusionRefused`, `IntelNPUUnsupported` and
`UnsupportedArgument` all subclass `NotImplementedError` so that a caller's
sane response — fall back and run eagerly — can be spelled as one `except`.
They are raised with a reason from data-dependent walks, and each is the
*mechanism* of a refusal rather than a missing capability.

## 2. Deliberate refusals — a document says why, and the reason still holds

### 2.1 Degenerate cases, refused rather than served

Nine of the ten in `nn/federated` are this, and they are one idea: **FedAvg
over one delta is that delta**, so a world (or cohort, or survivor set) too
small to aggregate would return its input and report success — and a test of it
would pass with no aggregation at all.

| site | refuses | cited |
|---|---|---|
| `nn/federated/__init__.py:174` | `world_size == 1` | `docs/distributed/TRANSPORT.md` |
| `:187` | `world_size < minimum` (3 for a proper subset / survivor set) | `docs/distributed/FEDERATED4.md` |
| `:652` | an integer parameter table | — (a delta holds floating parameters) |
| `:961` | a cohort of fewer than two | `docs/distributed/FEDERATED4.md` §5 |
| `:1244` | `on_missing='average_arrived'` in a world below three | `docs/distributed/FEDERATED3.md` §4.1, which **measured** the identity to 6e-8 |
| `delta/__init__.py:306` | a non-floating dtype in a delta | — |

**Checked, not inherited.** `FEDERATED3.md` §4.1's 6e-8 measurement is what
these rest on and it is still the right number; what had gone stale is the
*conclusion* drawn from it — see §4.1 below.

### 2.2 Named, reasoned, and large

| site | refuses | reason given, and whether it still holds |
|---|---|---|
| `nn/federated/__init__.py:1143` | `secure_aggregation=` | Needs pairwise secrets → point-to-point `send`/`recv`, which `ProcessGroupLocal` refuses (`docs/distributed/TRANSPORT.md` §3), plus key agreement, threshold sharing and an unmasking round. **Holds** — `send`/`recv` still refuse (README, `docs/distributed/COLLECT2.md`). |
| `:1157` | `differential_privacy=` | Needs per-example gradients; this backward is per-batch. **Holds.** |
| `:1087` | `allow_missing=` | An alias redirect to `on_missing=`, which is implemented. Not a capability refusal. |
| `:288` | a process group that is not this shim's | `allreduce_partial` has no upstream spelling. **Holds.** |
| `:511` | a tensor above `_WIRE_SAFE_BYTES` | States outright that it is a refusal and not a measured wall, and says what would raise it. **Holds** as written. |
| `export/target.py:158` | `coreml_ops()` | coremltools is not installed and there is no CoreML op set in the tree; a hand-written list would decide a measurement by writing it (`docs/graph/DECOMP.md` §12). **Holds** — no coremltools in the spike venv. |
| `device/_module_to.py:201` | `to(device.npu)` on the `coreml`/`qnn` backends | Returning `self` would be the silent partial offload `docs/graph/NPU2.md` §1 is about. **Holds.** |
| `transformers/__init__.py:113` | `export=` | See §4.3 — the refusal stands, its cross-reference had gone stale and is corrected. |
| `transformers/__init__.py:125` | `load_in_4bit=` | candle-core 0.11's `DType` has no `I8` (`docs/graph/QUANT.md` §2.1). **Holds** — `Cargo.toml` still pins candle-core 0.11.0. |
| `adapt/__init__.py:260` | a stage-2 (full autograd) method | `DESIGN.md` §3 excludes stage 2 from device targets permanently. The refusal carries its own falsifier: `torch.ones(1, requires_grad=True).sum().backward()`. **That now returns** rather than refusing (README's Training row, `docs/training/TRAIN2.md`) — so the *check* in the message has expired even though the *decision* has not. Listed in §5 as undecided rather than silently kept or silently dropped. |

### 2.3 Diagnostics, not refusals

`adapt/__init__.py:436` raises `NotImplementedError` naming the ops on the
gradient path that have no derivative rule, computed by
`trace.differentiable()` on the first step. It is data-dependent: what is
missing is a tape rule for some op in *the caller's* model, and the message
names the whole list rather than whichever the walk reached first. The gap it
reports is the tape rule inventory, which `docs/training/BACKWARD9.md` tracks.

---

## 3. Real gaps

### 3.1 CLOSED — the method spelling of five working functions

**What a user calls:** `a.equal(b)`, `a.allclose(b)`, `a.diff()`, `a.fmod(2)`,
`a.multiply(b)`.

**What they got:** `NotImplementedError: not implemented in torch._C shim:
TensorBase.equal` — from the *raising stub* that `surface.json` installs for
every upstream `TensorBase` member this shim does not implement. So
`hasattr(torch.Tensor, "equal")` was `True` and `dir()` listed it; only a call
told the truth.

**Why nothing saw it.** `methods.json`'s own `_README` says it: "A method is not
reachable by putting its name in `overloads.json`; nothing looks there."
`test_equal_allclose.py` measures both ops against upstream across dtypes and
edges and every call in it goes through `torch.<name>`.
`docs/platform/RELEASE_0_1_0b2.md` named `torch.allclose`/`torch.equal` as the
release's missing operators, they were added to `overloads.json`, and the
member half was never wired.

**Measurement.** A diff of the two JSON files names nine candidates and is
wrong about four: `chunk`, `where` and `is_floating_point` are installed by
hand (`_install_tensor_chunk`, `_install_tensor_where`,
`_install_tensor_predicates`) and do work, and `stft` is rebound in Python by
the vendored `torch/_tensor.py`, so its stub is unreachable from a caller.
Calling each is what separates them.

**Closed** by adding the same schemas to `methods.json`.
`rust/torch_c/pytests/test_methodspell.py` holds it, and its load-bearing test
is the general one: it walks `_C._shim_overloads`, keeps the names upstream
carries on `torch._C.TensorBase` and has *not* replaced on `torch.Tensor`, and
calls each — so a future `overloads.json` entry that forgets its sibling goes
red without anyone remembering the file exists. 127 names in the walk today.
Nullified by removing `fmod` from `methods.json` and rebuilding: red, naming
`fmod`.

### 3.2 CLOSED — seven of nine subpackages were unreachable through the package

**What a user calls:** `import torchnative; torchnative.adapt.wrap(...)`.

**What they got:** `AttributeError: module 'torchnative' has no attribute
'adapt'` — and the same for `api`, `delta`, `distributed`, `export`,
`kernels`, `nn` and `quant`. Only `device` and `transformers` resolved.

`__init__.py` resolves submodules through PEP 562 and its `__getattr__` refuses
anything not in `__all__`; `__all__` listed two of the ten. So the package
whose first docstring line is "on-device test-time learning and federated
learning" could not reach either of them by attribute.

**Why nothing saw it.** `from torchnative import adapt` works regardless — the
import system binds the submodule onto the parent as a side effect — and every
docstring and document in this tree spells it that way. The hole is only on the
attribute road, and only before anything has imported the subpackage.

**Closed** by listing every subpackage. Laziness is preserved and is itself
tested: `import torchnative` still pulls in neither `torch` nor
`transformers`. `rust/torch_c/pytests/test_tnnamespace.py` holds it, reading
the name list off the disk and probing each in a **fresh subprocess** — done
in-process, one earlier `from torchnative import x` would bind the attribute
and the test would pass against unfixed code. Nullified by dropping `adapt`
from `__all__`: red, naming `adapt`.

### 3.3 CLOSED — `torch.backends.mps.is_available()` was `False` on a host computing on Metal

**What a user calls:** `torch.backends.mps.is_available()`, which is the one
line every piece of third-party code uses to decide whether to use the GPU.

**What they got:** `False` (previously). `bootstrap.py` installed
`_mps_is_available` as `_constant_function(..., False)`, justified by "candle's
`metal` feature is off in `Cargo.toml`, so there is no Metal backend linked in,
which is why `PyDevice::resolve` refuses an `mps` label".

**Both halves of that justification were false**, measured on this host
against the shim built from this worktree:

    torch._C._mps_is_available()            True
    torch.backends.mps.is_available()       True
    torch.backends.mps.is_built()           True
    torch.empty(2, 2, device="mps")         ok      (m + m).device -> mps:0

and `rust/torch_c/Cargo.toml:170` reads
`candle-core = { ..., features = ["metal"] }`.

**Closed 2026-09-12.** `torch.backends.mps.is_available()` and `is_built()` now
both answer honestly through `torch._C._mps_probe()`. `is_available()` returns
`True` when a Metal device can be resolved on the host, and `is_built()` returns
`cfg!(target_vendor = "apple")` (`True` on Apple targets). See
`docs/numerics/DTYPEDEV.md` section 2 and `docs/devices/DEVICE_NS.md` section 6.


### 3.4 CLOSED — `adapt`'s stage 0 had no implementation and no method

**What a user called:** a `Method` subclass declaring `stage =
STAGE_FORWARD_ONLY`, wrapped by `adapt.wrap`.

**What they got:** `NotImplementedError` from `adapt/__init__.py:252`: "A
stage-0 method updates statistics inside the forward and needs no step at all;
nothing here provides that path yet."

**Why it was a gap rather than a refusal.** `DESIGN.md` §3's survey table puts
normalisation calibration on *both* sides of the differentiation line, and the
module docstring makes that the reason the stage is declared per method rather
than by directory. Stage 1 was built (`Tent`). The row above it — recompute the
statistics, no backward — was named as in scope by the design and absent, and
it is the cheap half of the two on a device.

**The design question it was blocked on, and the answer.** §5 recorded
*"whether a delta covers buffers"* as undecided, because running statistics are
buffers and `Delta` is keyed on `named_parameters()`. **Answered: a delta covers
parameters, not buffers**, and stage 0 is built on a second type rather than on
a widened `Delta`. The argument is written out in `DESIGN.md` §3 under
"델타는 버퍼를 덮지 않는다"; in short, all three of `Delta`'s operations mean
something different for a statistic than for a parameter — `record` stores an
*additive offset an optimiser produced* where a statistic is a *re-estimate*,
`apply`'s `base + value` can drive `running_var` negative, and `persist` and
`publish` already refuse a non-floating table (`delta:306`,
`federated:652`) while `num_batches_tracked` is `int64`. Widening the key would
make those two refusals fire after a round had been arranged, or force an
exception list — and an exception list is the evidence that two kinds of thing
are wearing one name.

What stage 0 needed was the **first** of `DESIGN.md` §3's three lifetime
questions and neither of the other two: snapshot the base, and restore it.

**Closed** with three additions, and no change to `Delta` at all:

| | |
|---|---|
| `delta.BufferSnapshot` | `over` / `covers` / `nbytes` / `drift` / `revert`. **No** `record`, `apply`, `persist`, `publish` — absent rather than raising |
| `adapt.StatisticsMethod` | the stage-0 protocol. Declares `select_modules(model)` — *module* names, not parameter names, because reusing `Method.select` would put two meanings behind one name |
| `adapt.BatchNormStats` | the method. Selects by buffer (`running_mean`/`running_var` present), not by class name |

`Adapted` branches on the stage: a stage-0 `online()` opens a `BufferSnapshot`
and **no optimiser**, and `step` takes no capture and no tape — it runs the
forward for the prediction, then runs it again with the selected modules in
training mode so their statistics take the batch in, and restores the mode it
found. It returns `(outputs, None)`, not `(outputs, 0.0)`: a stage-0 method
descends nothing, and a zero in `history` would draw a flat curve that looked
like convergence.

**Measured at grade `agrees`.** `rust/torch_c/pytests/test_stage0.py` runs the
same `nn.BatchNorm1d` model source on both sides — this stack in a vendored-tree
subprocess, upstream torch 2.13 from the spike venv in-process — over three
batches drawn from three different distributions, and compares the buffers
element by element:

    running_mean   7.45e-09      running_var   0 (exact)
    num_batches_tracked  3 == 3  predict-then-adapt logits  2.98e-08

with a non-vacuity check on the same numbers: the statistics moved 6.92e-02
over the run, so "agrees" is not two untouched buffers.

**`revert` was the specific trap, and is tested directly.** A stage-0 run
mutates buffers; `Adapted.revert` reverting only `self._delta` would have
returned, reported success, and left the model carrying the test distribution's
statistics. It now reverts whichever state is open, and the test asserts
`running_mean`, `running_var` **and** `num_batches_tracked` are back at base —
the counter too, because `momentum=None` makes the running estimate a cumulative
average over it, so a restore that missed it would leave the next EMA weighting
wrong on a model that reported itself reverted.

Nullified four ways, each made and watched go red: the step not entering
training mode (agreement + revert + prediction tests red), `Adapted.revert`
reverting only the delta (revert test red), `BufferSnapshot.revert` restoring
nothing (revert test red), and `select_modules` returning every module
(selection test red, naming `['', 'fc', 'norm', 'head']`).

**`Tent`'s documented half-implementation is NOT made whole, and that is
deliberate.** `Tent`'s docstring says that on a BatchNorm model it "is
implementing half of Tent", because it moves the affine parameters and never
puts the layer into batch-statistic mode. The *other* half now exists — it is
`BatchNormStats` — but it is a second method and not part of `Tent`, so a caller
writing `adapt.wrap(m, method=Tent())` still gets exactly what that docstring
says they get. Folding it in would change the numbers every existing BatchNorm
caller gets from an unchanged call, including `nn.federated.Engine`'s local
step, and it would put a `BufferSnapshot` inside a stage-1 wrapper — which is a
decision about whether a *method* may span two stages, and that question is not
this round's. The docstring is therefore left exactly as written. Listed in §5
as the next thing this makes answerable.

### 3.5 OPEN (small) — `torchnative.api.TorchNativeAPI.deploy` accepts a model and does nothing

`api/__init__.py:19` defines `TorchNativeAPI` with `__init__` and
`deploy(self, model)`, both `pass`. A caller who writes
`TorchNativeAPI().deploy(model)` gets `None` and no indication that nothing
happened — which is the shape this repo's own CLAUDE.md §6 puts *below* a
refusal.

`DESIGN.md:89` and :919 give the intended role (the device-side counterpart of
a weight-distribution client). `docs/verification/AUDIT.md:596` describes this
file as "docstring-only with imports deferred", which is not what it contains —
that line was written about the `SyntaxError` that used to be on line 4 and did
not look past it.

**Not closed here** because both honest options change the public surface: make
`deploy` refuse by name, or remove the class. Either is a withdrawal, and
withdrawals in this repo are announced (`IntelNPUWithdrawn` is the pattern).
Recorded for the user to choose.

---

## 4. Documentation corrected in this round

Each of these was checked against the code before being changed.

1. **`README.md`, Accelerators row** — said the `mps` refused set is **85** ops
   while the Metal row twelve hundred lines down said **87**.
   `_C._shim_mps_host_readback_ops()` returns **87**. Corrected to 87, with the
   disagreement named rather than quietly resolved.
2. **`torchnative/distributed/__init__.py` module docstring** — opened with
   "the `local` backend — `torch.distributed` with a world of one" and
   described reductions as the identity and `broadcast`/`barrier` as no-ops.
   `ProcessGroupLocal` has run a real world of three or more over loopback TCP
   with eleven collectives agreeing with upstream gloo since
   `docs/distributed/COLLECT2.md`. Rewritten, keeping the world-of-one
   behaviour as a case rather than as the whole story, and correcting
   `send`/`recv`'s stated reason (no route between two non-hub ranks, not "no
   local work makes them mean anything"). The `devices=["cpu"]` comment
   justified itself with "Metal, Vulkan, NPU is not built"; two of the three
   are, so the restriction now stands on the ground that no collective has been
   run on a non-CPU tensor.
3. **`nn/federated/__init__.py`, `ON_MISSING`** — the comment read "Only the
   first is implemented." `on_missing='average_arrived'` has been built since
   `docs/distributed/FEDERATED4.md` §6, is served with `min_participants=k`,
   refuses only below a world of three, and has tests
   (`test_average_arrived_divides_by_the_survivors_and_refuses_below_the_floor`).
   Corrected.
4. **`docs/distributed/FEDERATED3.md`** — a round record of 2026-09-06 whose §4.1
   concluded `on_missing='average_arrived'` "cannot be honestly served here even
   as an experiment" and whose §8 scoped out both it and proper-subset cohorts.
   Both were built by `FEDERATED4.md`. The round record is kept as written and a
   supersession note added at the head and inline at each of the three claims,
   rather than rewriting history.
5. **`docs/api/TRANSFORMERS.md` §5 and the `export=` refusal message** — both said
   the missing step is "the same wall `model.to(torchnative.device.npu)`
   reports". `to(npu)` on the `openvino` backend lowers eligible `nn.Linear`
   leaves and returns the same `nn.Module`; only `coreml` and `qnn` still
   refuse. The refusal itself is unchanged and correct — a captured graph still
   cannot become a module leaf — but it now points at what *does* work instead
   of at a wall that moved.

Nothing in `README.md`'s Status table was found overstating. Its rows carry
DOCWATCH markers and the one disagreement found (85/87) was internal to the
README and understated the count.

---

## 5. UNDECIDED — what this round could not settle, and what would settle it

* ~~**Whether `adapt`'s stage-0 road should cover buffers.**~~ **SETTLED
  2026-09-13: it does not.** A delta covers parameters; stage 0 is built on
  `delta.BufferSnapshot`, a second and smaller type. The argument — that
  `record`, `apply`, `persist` and `publish` each mean something different for
  a re-estimated statistic than for an optimiser-produced offset, and that
  `num_batches_tracked` being `int64` would make two existing refusals fire
  late — is written out in `DESIGN.md` §3 ("델타는 버퍼를 덮지 않는다") and
  summarised in §3.4 above. No existing caller changes: `Delta`'s signature,
  behaviour and coverage are untouched.
* **Whether a single method may span two stages** — specifically, whether `Tent`
  should fold `BatchNormStats` in and stop being half of the paper on a
  BatchNorm model. Newly answerable: until 2026-09-13 the other half did not
  exist, and now it does. Folding it in would change the numbers an unchanged
  `adapt.wrap(m, method=Tent())` returns for every existing BatchNorm caller
  (including `nn.federated.Engine`'s local step), and would put a
  `BufferSnapshot` inside a stage-1 wrapper — so it is a decision about the
  stage axis, not a fix. `Tent`'s docstring is left stating the hole, which is
  still the truth for that call. **Settled by:** a decision on whether `stage`
  stays a single value per method.
* **Whether `adapt`'s stage-2 refusal should keep its falsifier.** The message
  tells the reader to run
  `torch.ones(1, requires_grad=True).sum().backward()` and says an autograd
  exists that the refusal predates if it returns. It *does* return now. The
  decision the refusal encodes (stage 2 is excluded from device targets
  permanently, `DESIGN.md` §3) is a scope decision that a working desktop
  autograd does not overturn — but the check as written now fires on a healthy
  tree. Left alone rather than edited, because deciding whether stage 2 is
  reachable-but-unsupported or genuinely impossible is above this round.
  **Settled by:** a stage-2 method attempted against the current backward.
* ~~**What `_has_mps` and `_mps_is_available` are each supposed to claim.**~~ **SETTLED
  2026-09-12: §3.3 is closed.** `_has_mps` is `is_built()` (`cfg!(target_vendor = "apple")`)
  and `_mps_is_available` is `is_available()` via `torch._C._mps_probe()`. See
  `docs/numerics/DTYPEDEV.md` section 2.
* **Whether `torchnative.api.TorchNativeAPI` should refuse or be withdrawn.**
  §3.5. Both are public-surface changes. **Settled by:** the user choosing.
* **`torchnative/nn/__init__.py` is a zero-byte file.** `nn/federated` is
  reached through it and works, so it is functional as a namespace anchor, but
  every other subpackage in this tree carries a docstring saying what it is.
  Whether `nn` is intended to grow siblings to `federated` is not recorded
  anywhere this round could find. **Settled by:** `DESIGN.md` saying so.
* **Whether any `torch.<name>` function is missing beyond the member door.**
  §3.1 measures the *method* spelling against the *function* table. It does not
  measure the function table against upstream's `torch` module — a name
  upstream has and this shim does not would not appear in `_shim_overloads` at
  all, so the walk cannot see it. **Settled by:** a walk over
  `dir(upstream.torch)` filtered to what this tree claims to cover, which is a
  different and much larger question than this round was scoped for.
* **`export/`, `quant/` and `kernels/`** were read for refusals (§1, §2) but
  their surfaces were not walked name by name against their documents the way
  `torchnative/__init__.py` was in §3.2. `kernels/__init__.py` is a docstring
  and no code at all, which may be intended (DESIGN.md §8 describes a
  resolution *contract*, not a module) or may be the same shape as §3.5.
  **Settled by:** the same treatment §3.2 got, applied per subpackage.

## 6. Entry count

**21** `raise NotImplementedError` statements in
`torchnative/src/main/torchnative/`, not 15 — the count that produced the table
at the head of this document covered only `nn/federated` and `adapt`. All 21
are classified above:

| | |
|---|---|
| abstract base (§1) | **3** — `adapt:79`, `adapt:88`, `device:303` |
| deliberate refusal (§2) | **16** — `federated` ×10, `delta:306`, `_module_to:201`, `target:158`, `transformers` ×2, `adapt:260` |
| diagnostic, data-dependent (§2.3) | **1** — `adapt:436` |
| **real gap** (§3.4) | **1** — `adapt:252`, the stage-0 road. **Closed 2026-09-13**; the site still raises, but now it refuses a stage-0 method that supplies a *stage-1* contract (no `select_modules`) and names `StatisticsMethod`, so it has moved from §3 to §1's shape — an abstract protocol a subclass in this repo supplies (`BatchNormStats`) |

**Of the fifteen** the original count named (`nn/federated` 10 + `adapt` 5):
2 abstract bases, 11 deliberate refusals, 1 diagnostic, **1 real gap**.

That one is now closed too (§3.4, 2026-09-13). Three further real gaps were found
outside that count entirely and are closed (§3.1, §3.2, §3.3); one more is open by
decision (§3.5). Five exception
*classes* deriving from `NotImplementedError` are counted nowhere here — they
are the mechanism of a refusal, listed at the end of §1.
