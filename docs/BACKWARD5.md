# W10, W9, W8: what mutation actually costs, measured — and the recommendation

`docs/BACKWARD2.md` §1.5 closed the structural group with a claim that this round had to test before
writing anything:

> **W10 — mutation, and this is the one that cannot be bought cheaply.** … an eager tape must either
> refuse in-place ops on any tensor it has recorded — which stops `torch.optim` — or track a version
> per tensor and invalidate, and the second is `c10::VariableVersion` plus the `ADInplaceOrView` key,
> i.e. precisely the layer `docs/AUTOGRAD.md` §6 chose the tape *in order to avoid*.

The claim was made from the outside. `docs/BACKWARD2.md` §8 row 4 says so itself — *"the two options
in §1.5 are the two that were thought of"*. This round let one in-place op through and looked.

**It is wrong about which failure happens, and therefore wrong about which field is needed.** A
replay tape *functionalises* an in-place op for free, and gets a gradient upstream refuses to give at
all. What breaks instead is **aliasing** and **constant freshness** — and the second of those is a
live defect in the shipped tape today, reachable with no in-place op recorded at all.

Environment: worktree `work/bw3` on `develop` `61cf4f0`, torch 2.13.0, transformers 5.15.1,
`/Volumes/macMini/caches/spike-venv/bin/python`, `CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-bw3`.
Every shim reading printed `shim`; every upstream reading was taken with
`env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`.

### Answers, before the evidence

| question | answer |
|---|---|
| **Q1.** What breaks if an in-place op is *recorded* rather than refused? | **Not the thing BACKWARD2 names.** Single assignment comes back for free — the recorder renames by object identity and the replay is functional, so the two programs upstream refuses with a version-counter error (`exp` then `mul_`; a saved operand then `add_`) the tape gets **right**. What breaks is a **second Python object over one storage**: with a view in the region the replayed forward is a *different function*, and the gradient was **100× wrong** against upstream, element-wise. §1 |
| **Q1b.** Is that "unsound replay" or "silently wrong gradient"? | **Both, by one mechanism, and the forward moves first.** In every case measured the replayed *loss* diverged before the gradient did — 1.79 against 179.0. That is a cheaper thing to detect than a wrong gradient. §1.2 |
| **Q2.** How much of `VariableVersion` is needed? | **Two independent mechanisms, and a tape needs only part of one.** Upstream's leaf/view in-place refusal is structural (no counter). The counter + a per-saved-value snapshot catches three classes, and **a replay tape moots all three** — it never reads a saved value. What it does *not* moot is the constants, which are held live. §2 |
| **Q3.** What does W9 cost? | **18.9 MiB at S=8, 302.6 MiB at S=128**, for SmolLM2-135M — 1862 nodes, 1893 tensor result slots, of which 702 are view results whose storage is already held. A naive count over result *shapes* says 538 MiB and is wrong for exactly the reason §1 is about. Lifetime is **one iteration**. §3 |
| **Q4.** Does a shape exist that reuses the tape rather than building a second one? | **Yes, and it is smaller than BACKWARD2 §3 could see.** `tape::backward` reads five things off a trace, and `Recorder` already has four of them. The fifth is `Env`, and **`Env` already exists during capture** — it is `Recorder::keepalive`, dropped at `_capture_end`. §4 |
| **The live defect this turned up** | The tape reads its constants **live at `backward()` time**. Mutate a parameter between `_capture_end` and `trace.backward()` and the tape silently differentiates at the new weights. No in-place op is recorded; this is today's shipped artefact. §1.3 |
| **The recommendation** | **Pay for W10a (constant freshness) and do not pay for W10b (aliasing) yet.** W10a is one integer per tensor, closes a live defect, and is worth landing on its own merits whether or not `.backward()` is ever built. W10b is upstream's whole aliasing layer and buys nothing until a recorder exists. §6 |
| What landed? | **The measurement, and one test that pins §1.3's defect as a known divergence.** No recorder, no engine, no version counter. §5 |

---

## 1. Q1 — what actually breaks, measured

The probe: `capture::refusal_for` was gated on an environment variable so that `is_mutating` could be
stepped over, and `capture::record` was made to record the in-place op **under its functional name**
— `aten.mul_.Scalar` recorded as `aten.mul.Scalar`. That second half is not a cheat; it is the model
of what a naive eager recorder does. It records the *mathematics* and forgets the *aliasing*, which
is the hypothesis worth testing. The probe was applied to a `cp` backup, built, measured, and
**removed**; `git status --short` is clean.

The first attempt found something else first and it is worth recording:

```
NotImplementedError: torch._C tape: no derivative rule for aten.mul_.Scalar --
  a gradient reached it, and the tape refuses to guess.
```

**Recording an in-place op without renaming it does not produce a wrong gradient — it produces a
loud refusal by name**, because none of `RULE_OPS`'s 60 entries is an in-place op. That is a real
part of the answer: the tape's existing rule table is a second line of defence that BACKWARD2's
framing does not mention, and it fires before any aliasing question is asked.

### 1.1 The four cases, against upstream

`x = [0.3, 0.7, 1.1]`. Shim column is the probe build; upstream column is real torch 2.13.0.

| | program | tape: replayed loss | tape: grad wrt x | upstream loss | upstream grad |
|---|---|---|---|---|---|
| **A1** | `y = x.exp(); y.mul_(2); y.sum()` | 12.735556 = eager ✅ | `[2.699718, 4.027505, 6.008332]` **= 2·exp(x)** ✅ | — | **`RuntimeError`** — *"…modified by an inplace operation … which is output 0 of Exp"* |
| **A2** | `a = x*1; b = a*a; a.add_(10); b.sum()` | 1.79 = eager ✅ | `[0.6, 1.4, 2.2]` **= 2x** ✅ | — | **`RuntimeError`** — *"…which is output 0 of Add"* |
| **A5** | `a = x*1; v = a.view(3); a.mul_(10); (v*v).sum()` | **1.79**, eager was **179.0** ❌ | `[0.6, 1.4, 2.2]` | 179.0 | **`[60.0, 140.0, 220.0]`** |
| **A6** | `a = x*1; v = a.view(3); v.mul_(10); (a*a).sum()` | **1.79**, eager was **179.0** ❌ | `[0.6, 1.4, 2.2]` | 179.0 | **`[60.0, 140.0, 220.0]`** |

**A1 and A2 are the cases BACKWARD2's argument is about, and the tape gets them right where upstream
refuses.** The reason is mechanical rather than lucky. `capture::record` keys `known` on the
*object address* of a result, and an in-place op returns its receiver, so `known[addr]` is
**rebound** to the new node — which is single-static-assignment renaming, done for free by identity.
And `tape::backward` reads its operand values out of `Env`, which `PyCaptureTrace::run` builds by
**re-executing** the record; under a functional name the re-execution allocates a new tensor instead
of writing into the old one, so the value an earlier rule reads is the un-mutated one. A replay tape
is a functionaliser. Upstream's version counter exists precisely because upstream's engine reads a
*saved live tensor* and therefore can be lied to; this tape cannot be, because it saves nothing.

**A5 and A6 are where it breaks, and it is not single assignment.** A view is a *second Python
object over one storage*. `known` is keyed on address, so the view's address still points at the
node that produced it, and the functionalised replay computes that node from the *un-mutated* base.
Eager wrote through the alias; the replay did not. The two programs are different functions and the
tape differentiates the wrong one.

`(v*v)` rather than `v.sum()` is deliberate. An earlier version of this measurement used `v.sum()`
and both arms returned `[1.0, 1.0, 1.0]` — **the gradient agreed with upstream while the forward was
15× wrong**, because `d/dx` of `sum(x)` and of `sum(x+10)` are the same. That is `docs/CAPTURE.md`
§9-1's trap in miniature, met inside the measurement designed to look for it, and it is the reason
the table above is stated on a program whose gradient *cannot* agree by coincidence. It does not:
`[0.6, 1.4, 2.2]` against `[60.0, 140.0, 220.0]`, **a factor of 100, silently.**

### 1.2 "Unsound replay" and "silently wrong gradient" are the same failure here, and the forward moves first

The brief asked which of the two it is, on the grounds that they have different costs. **The
measurement says the distinction does not survive: every case where the gradient was wrong was a
case where the replayed loss was wrong first**, and by a larger factor. That is good news for cost,
because the forward is the cheaper thing to check — `replay()` already exists, already returns the
recorded outputs, and comparing it against the eager value the region produced is one subtraction.

Two things follow that are worth stating separately from the plan:

* The failure is not detectable by looking at the gradient, and it **is** detectable by looking at
  the loss. Any W10 work should put its check on the forward.
* The tape's existing eager-versus-replay comparison (`docs/CAPTURE.md` §3) is the shape of that
  check and it is not run at `backward()` time.

### 1.3 The live defect: the tape's constants are read at `backward()` time, not at capture time

This one needs **no in-place op recorded**. The probe was off; this is the shipped artefact.

```
<CaptureTrace 2 nodes, 1 inputs, 1 constants, 1 outputs>   eager loss at capture: [4.2]
backward BEFORE step:  d/dx = [2.0, 2.0, 2.0]   d/dw = [0.3, 0.7, 1.1]
w after step:          [3.0, 3.0, 3.0]                      <- with torch.no_grad(): w.add_(1.0)
backward AFTER  step:  d/dx = [3.0, 3.0, 3.0]   d/dw = [0.3, 0.7, 1.1]
replay after step:     [6.3]                                 (the loss at capture was 4.2)
VERDICT: STALE -- the tape silently moved its point of differentiation
```

`PyCaptureTrace` holds `const_objects: Vec<Py<PyAny>>` — **strong references to the caller's live
parameter tensors** — and `run()` copies those references into `Env` at replay time. So a trace
differentiates whatever its constants hold *when `backward()` is called*, which for a training loop
is whatever `optimizer.step()` last wrote. Upstream refuses the same shape by name
(`…is at version 1; expected version 0`), measured, in the same script.

**`torchnative.adapt` is safe from this by ordering and by nothing else.** `adapt/__init__.py:399`
calls `trace.backward(...)` and `adapt/__init__.py:421` calls `self._optimizer.step()`, in that
order, and drops the trace at the end of the call. Nothing enforces that order, nothing tests it,
and a caller who keeps a trace — which the class does not do and the `CaptureTrace` API fully
permits — gets a gradient at the wrong point with no message. That is a check that cannot fail
because the situation it guards against is currently unreachable by accident, which is
`CLAUDE.md` §5.5's shape, and §5 pins it.

---

## 2. Q2 — how much of `VariableVersion` is needed

Established from what upstream's checks actually catch, by reproducing each class and reading
`torch/include/torch/csrc/autograd/saved_variable.h` and `VariableTypeUtils.h` from the installed
2.13.0.

| # | class | upstream | mechanism upstream uses | does a **replay** tape need it? |
|---|---|---|---|---|
| 1 | in-place on a leaf requiring grad | **raises** | `check_inplace` — pure `is_leaf() && requires_grad()`. **No counter.** | no. `docs/BACKWARD4.md` §4.2 already declined to mark leafness, deliberately |
| 2 | a saved operand mutated before backward (A1/A2) | **raises** | per-storage counter + `SavedVariable::saved_version_` snapshot, compared at `unpack()` | **no — mooted.** §1.1: the tape saves nothing and recomputes, so there is no snapshot to invalidate |
| 3 | in-place on a *view* of a leaf requiring grad | **raises** | structural `view_of_leaf` check. **No counter.** | no, same as 1 |
| 4 | in-place on a value nothing saved (`y=x.clone(); y.add_(1)`) | **succeeds, correct** | the counter correctly staying silent | n/a — the tape is also correct here (E3, E8) |
| 5 | `optimizer.step()`'s `add_` on a leaf | **exempt** | `if (!GradMode::is_enabled()) return;` in the generated in-place wrapper — **not** `AccumulateGrad`, not `_foreach_`. With grad mode **on** the identical call raises class 1 | n/a |
| 6a | `.data` mutation | **silent, and upstream's own gradient is wrong** (12 where 2 is correct) | none — `.data` decouples the counter deliberately | n/a; upstream does not guarantee this either |
| 6b | `.detach()` mutation | **raises** | the counter **is** shared with `detach()`'s output | mooted with 2 |
| 7 | mutation between two `retain_graph=True` backwards | **raises** | the same counter + snapshot, at the second `unpack()` | mooted with 2 |
| **8** | **a burned-in constant mutated between record and backward (§1.3)** | **raises** (it is class 2 wearing a parameter) | the same counter + snapshot | **YES. This is the only one that survives.** |

So `VariableVersion`'s two jobs split cleanly, and a replay tape needs **neither of the two upstream
built it for and one it did not**:

* the **leaf/view structural half** is not a counter at all and this project already decided against
  it (`docs/BACKWARD4.md` §4.2);
* the **saved-value half** — the counter's actual reason to exist — is *structurally* mooted by
  replay, which is a stronger statement than "not needed yet";
* what is left is **freshness of the values the trace holds by reference**, and upstream gets that
  from the same counter as a side effect. It is 333 tensors for SmolLM2, not every intermediate.

**Judgement: the field a replay tape needs is not `c10::VariableVersion`. It is one monotonic
`u64` per tensor, bumped at the door on any in-place op, snapshotted per constant at
`_capture_end`, and compared in `run()`.** That is the *snapshot* half of upstream's design
applied to 333 values instead of to every saved variable, and it does not need the
`ADInplaceOrView` dispatch key, alias sets, view metadata, or rebasing — the four things
`docs/AUTOGRAD.md` §6 chose the tape to avoid. §6 calls this **W10a**.

**What it does not catch, and this is the honest half:** class 6a's shape — a mutation that reaches
the storage without going through the door. Upstream does not catch that either. And it does not
catch A5/A6, because a *view* of a mutated tensor is a different object and would carry a different
counter unless the counter lives on the storage. That is **W10b**, and it is where upstream's
`intrusive_ptr<VersionCounter>` sharing starts to be load-bearing.

---

## 3. Q3 — what W9 costs, in the unit `docs/BACKWARD4.md` used for W8

`docs/BACKWARD4.md` gave W8's door a unit — 1862 recorded nodes for one `S=8` forward. The same
capture, real `HuggingFaceTB/SmolLM2-135M`, `float32`, measured over the trace's own recorded
result metadata:

```
nodes                 1862
tensor result slots   1893
all result bytes      538.1 MiB
```

**538 MiB is the wrong number and it is wrong for §1's reason.** 702 of the 1893 slots are results
of view ops — `aten.t.default` alone accounts for a 108 MiB slot, which is the transposed
`(576, 49152)` LM head, a tensor with no storage of its own. Counting result *shapes* counts one
storage many times, which is exactly the alias-blindness A5 is about. Subtracting slots produced by
view ops:

| S | nodes | all result bytes | of which view results | **new storage** |
|---:|---:|---:|---:|---:|
| 8 | 1862 | 538.1 MiB | 519.2 MiB | **18.9 MiB** |
| 32 | 1862 | 613.5 MiB | 537.8 MiB | **75.7 MiB** |
| 128 | 1862 | 914.9 MiB | 612.3 MiB | **302.6 MiB** |

Node count does not move with `S` — the graph is the same program — and the bytes grow roughly
linearly, as activations do. For reference the constants are 333 tensors and 513.1 MiB, which is the
model itself and is alive anyway.

**So W9's cost is: an eager tape for a 135M model holds 18.9 MiB at `S=8` and 302.6 MiB at
`S=128`, plus 1893 `Py<PyAny>` handles, from the first op of the forward until `backward()`
returns.** Two things follow:

1. **This is not an exotic cost — it is the cost upstream already pays**, because upstream's
   `SavedVariable`s hold the same activations over the same interval. An eager tape is not more
   expensive than autograd; it is *as* expensive, and the replay tape's 0 MiB is what is unusual.
   The replay tape buys that by paying one extra forward per backward, which
   `PyCaptureTrace::backward`'s doc comment already names.
2. **The lifetime question is not bytes, it is who drops it**, and `docs/BACKWARD2.md` §1.5's
   option (a) — *"free at `backward()` and refuse a second one"* — is not a lesser refusal. It is
   **upstream's default**: `retain_graph=False` frees the graph and a second `.backward()` raises.
   BACKWARD2 called it *"at least a named refusal"*, which undersells it. Matching upstream's
   default is the whole of W9 for the single-backward case, and it is a `Vec` drop.

---

## 4. Q4 — the shape that reuses the tape, and it is smaller than BACKWARD2 could see

`docs/BACKWARD3.md` established that `tape.rs`'s 1673 lines of derivative rules never touch a trace
type. That still holds — `derivative(py, node, env, gouts, outs)` takes `&Node` and `&Env`, and
neither is a trace. This round asks the next question: **what exactly would an eager recorder have
to produce?**

`tape::backward`, `wrt_set`, `reachable` and `wanted` read **five** things off a `PyCaptureTrace`
and nothing else:

| read | `Recorder` already has it? |
|---|---|
| `trace.nodes: Vec<Node>` | ✅ `Recorder::nodes` — the identical type |
| `trace.inputs: Vec<TensorMeta>` | ✅ `Recorder::inputs` |
| `trace.consts: Vec<TensorMeta>` | ✅ `Recorder::consts` |
| `trace.outputs: Vec<Ref>` | ⚠️ set at `_capture_end`; eagerly it is *the tensor `.backward()` was called on*, which is `known[addr]` — one lookup |
| `trace.run(py, inputs) -> Env` | ❌ — and this is the whole difference |

And `Env` is

```rust
pub(crate) struct Env { inputs: Vec<Py<PyAny>>, consts: Vec<Py<PyAny>>, nodes: Vec<Vec<Py<PyAny>>> }
```

**which is `Recorder::keepalive`, `Recorder::const_objects` and the recorded input objects,
reshaped.** `record()` already pushes every tensor result into `keepalive` in node-output order, for
the identity reason (`capture.rs:176` — *"an address can never be reused under us while the
recording is open"*), and `_capture_end` **drops it**.

> **`docs/BACKWARD2.md` §1.5 asked, of W8, "what keeps the intermediates alive". The answer is that
> the thing that keeps them alive already exists and is thrown away.** An eager `Env` is `keepalive`
> not dropped. That is why §3's 18.9 MiB is a *retention* cost and not an *allocation* cost — the
> tape already holds those bytes during the forward.

The reshape is not free of judgement: `keepalive` is a flat `Vec` and skips `Slot::Other` results
(the `METADATA_ONLY` allowlist), so it is not 1:1 with `nodes[i].outputs[j]` and would have to become
`Vec<Vec<Py<PyAny>>>` to be indexed the way `Ref` indexes. That is a field change in `Recorder`, not
a new subsystem.

**So the answer to Q4 is yes, and the recorder is `Recorder` with three changes** — none of which is
in `tape.rs`:

1. `CAPTURING` is on outside a region (W8);
2. `keepalive` becomes the `Env` shape and is **not** dropped, and is dropped at `backward()` instead (W9);
3. `outputs` is set at `.backward()` from `known[addr]` rather than at `_capture_end`.

`derivative()` and all 60 rules are reused verbatim. `reachable()` is the one function that would be
replaced rather than reused — its own doc comment says why, and `docs/BACKWARD4.md`'s propagated
`requires_grad` flag is now the other of the two sources it names.

**Still not established:** nobody has built a `Node` outside `capture::record` and fed it to
`derivative()`. `docs/BACKWARD2.md` §8 row 3 left that open and this round did not close it — what it
adds is that the *producer* need not be written, because `Recorder` is already it.

---

## 5. What landed

Split the way `CLAUDE.md` §5.3 asks for.

| | |
|---|---|
| **feature added** | — **nothing.** No recorder, no engine, no version counter, no kernel |
| **defect fixed** | — |
| **test added** | 1 — `test_a_traces_constants_are_read_at_backward_time_not_at_capture_time`, which pins §1.3's staleness as a **known** divergence with upstream's own refusal quoted beside it |
| **documentation corrected** | `docs/BACKWARD2.md` §1.5's W10 framing, in this document rather than in place — §1 |
| **deleted** | the probe (`cp` backup restored; `git status --short` clean) |

The test is the only code. It exists because §1.3 is currently unreachable by accident and therefore
invisible: `adapt.step()` orders its calls correctly and nothing says it must. The test asserts the
staleness **as it is**, so that a round which fixes it has to come here and invert the assertion —
which is how the two tests in `docs/BACKWARD4.md` §5.1 got stronger.

**No engine was built and `_ImperativeEngine.run_backward` still refuses by name.** This round
produced **no new gradient**: the only gradients computed were inside a throwaway probe build that
has been removed, and every one of them was checked element-wise against upstream 2.13.0 in §1.1.

---

## 6. The recommendation, with the numbers behind it

The brief asked for *"W10 costs X and here is why the project should or should not pay it"*.
**W10 is two items and the project should pay one of them.**

### W10a — constant freshness. Pay it. It is worth landing with or without `.backward()`

| | |
|---|---|
| **what** | one monotonic `u64` on `PyTensorBase`, bumped at the door on any op `is_mutating` recognises; `PyCaptureTrace` snapshots it per constant at `_capture_end`; `run()` compares and refuses by name |
| **size** | one field, one bump site (the door already branches on grad mode since BACKWARD4 — this is the same branch), one `Vec<u64>` of length 333, one loop in `run()`. Plus a test |
| **what it buys** | §1.3, which is a **live silent-wrong-gradient path in the shipped tape**, closed with upstream's own error class |
| **what it costs** | one integer increment on in-place ops only. The SmolLM2 prefill in §3 issues none |
| **why now** | it is independent of W8/W9/W10b and of `.backward()` entirely. It is a defect fix for `CaptureTrace`, and the fact that it is also a prerequisite for an eager tape is a bonus rather than the reason |
| **what it does not buy** | anything about A5/A6. A counter on the tensor does not see a write through an alias |

### W10b — aliasing. Do not pay it yet

| | |
|---|---|
| **what** | the counter moves onto the *storage* and is shared by views, which is `c10::VariableVersion`'s `intrusive_ptr<VersionCounter>`, plus enough view metadata to know what shares with what |
| **size** | unknown, and §7 says so. It is the layer `docs/AUTOGRAD.md` §6 chose the tape to avoid, and this round found no evidence that the avoidance was wrong |
| **why not now** | it buys **nothing at all** until an eager recorder exists, because `is_mutating`'s refusal makes A5/A6 unreachable. Landing it first would be a subsystem with no consumer and no way to fail |
| **and the cheaper alternative that §1.2 found** | the failure shows up in the **forward** before the gradient, by a larger factor (1.79 against 179.0), and `replay()` already computes that forward. A recorder could compare its replayed output against the value the region actually produced and refuse — one subtraction against a whole aliasing layer. **Not costed**, and §7 lists it as unknown |

### `docs/BACKWARD2.md` §8 row 4's uncosted third option

*"recording a copy of any operand an in-place op is about to touch, i.e. paying memory instead of a
version counter"*. §3 now gives it a unit: the copy would be bounded by the same 18.9 MiB / 302.6 MiB
an eager `Env` holds, and for the training-loop case it would be a copy of the **parameters** —
513.1 MiB for a 135M model, per iteration. **That is the answer: for `optimizer.step()` specifically
the copy costs the whole model and the counter costs 333 integers, so the third option loses to
W10a on the one case W10 exists for.** It remains competitive for the A5/A6 aliasing case, where
W10a does not work at all.

### The dependency-ordered plan

`docs/BACKWARD2.md` §2.1 ordered these **W10 → W9 → W8 → W5**. W5 landed out of order in
`docs/BACKWARD4.md` and did not need the rest, so the remaining order is **W10a → W9 → W8**, with
W10b deferred and not on the path.

| # | piece | size | depends on | status of the estimate |
|---|---|---|---|---|
| **0** | **W10a**, constant freshness | one `u64` field, one bump site, one `Vec<u64>`, one comparison loop, one test. **Self-contained; lands alone; fixes a live defect** | nothing | **sized.** The bump site is the door's existing grad-mode branch |
| **1** | **W9**, lifetime | `Recorder::keepalive` becomes `Vec<Vec<Py<PyAny>>>`; it is not dropped at `_capture_end`; it is dropped at `backward()` and a second `backward()` refuses, which is upstream's `retain_graph=False` default | W10a (a tape that outlives its capture is exactly what makes §1.3 reachable on purpose) | **sized in memory** (§3) and **not in time** |
| **2** | **W8**, the eager recorder | `CAPTURING` on outside a region; `outputs` from `known[addr]` at `.backward()`; the grad-mode gate `docs/BACKWARD4.md` §4.3 already installed becomes the suppression scope `docs/BACKWARD2.md` §3.2 requires around the backward itself | W9 | **not sized.** The per-dispatch cost of an always-on recorder is the number nobody has |
| **3** | wire `_ImperativeEngine.run_backward` | reuses `tape::backward` with `reachable()` replaced by BACKWARD4's propagated flag | W8 | **not sized** |
| — | **W10b**, aliasing | storage-shared counters, view metadata | W8 (no consumer before it) | **unknown, deliberately.** §7 |

**The one-line recommendation: land W10a now as a `CaptureTrace` defect fix, and treat W8/W9 as a
separate decision that W10a does not commit the project to.** The reason to separate them is that
this round found W10's difficulty is not where it was thought to be: the expensive half of upstream's
version machinery is structurally unnecessary to a replay tape, and the half that *is* necessary is
small enough to be worth having on its own.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs is_mutating present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs const_objects present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs keepalive present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs derivative present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_a_traces_constants_are_read_at_backward_time_not_at_capture_time present -->

---

## 6.5 Gates

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh
    381 ok, 0 FAIL          (380 before; +1 test, 0 inverted, 0 removed, 0 weakened)
    SELF-TEST: PASS -- 19 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised
    DOCWATCH: PASS -- 334/334 evaluated marker(s) hold
    EXIT=0

$PY tools/golden/compare.py
    SUMMARY: 8476/8476 cases passed, 0 failed, ops covered=203, pending case builders=0
    EXIT=0
```

`ops=203` is unchanged **on purpose** — no kernel landed, so nothing here could have moved it.

### 6.5.1 The forward did not move, and the control is stronger than a digest

`docs/BACKWARD4.md` §7 built its control by stashing and rebuilding rather than trusting a digest
whose recipe is not in the tree. **This round can do better than either, and it is worth saying why
rather than quoting numbers.** The round's only edits are `docs/BACKWARD5.md` and one test in
`rust/torch_c/pytests/test_shim.py`. Neither is a compile input: `bootstrap.py` is the only Python
`include_str!`'d into the crate and it was not touched, and `git status --short` at the end of the
round is exactly those two paths. **Every source the artefact is built from is byte-identical to
`develop` `61cf4f0`**, so the forward cannot have moved — that is a stronger statement than a digest
comparison, which can only fail to detect a difference.

The probe of §1 *was* a compile input while it existed. It was applied to a `cp` backup, built,
measured, and restored from that backup; `git status --short` is clean of `capture.rs` and the
gates above were run on the rebuilt artefact after the restore, not before it.

Measured anyway, on the final artefact, real SmolLM2-135M `float32`, logits sha256 over the
little-endian bytes of the flattened `[1,S,49152]`:

| S | sha256 | |
|---:|---|:--:|
| 6 | `606e00d1be05fccce84b23b02dae8f473d490ad0f706b9b612036dd55cee531b` | ✅ |
| 32 | `62d923807cce47cee35a5e7b6091c9544e180c6c9ff771bb4cad3f478f4cbb79` | ✅ |
| 128 | `4750428e94f7383c42853b0d547192a7946fdde9e086d3b16cebdeb2ca68e37d` | ✅ |

All three equal `docs/BACKWARD4.md` §6.2's. And the tape, same capture, `S=8`:

```
<CaptureTrace 1862 nodes, 1 inputs, 333 constants, 1 outputs>   loss 15.292611122131348
differentiable()                      nodes 1862, on a gradient path 1853, missing {}
differentiable(wrt_constants=params)  nodes 1862, on a gradient path 1723
parameter constants: 272 of 272        parameters with a gradient: 272 of 272
_tape_rules(): 60 rules
```

`1862`, `333`, `1853`, `1723`, `272 of 272`, `60` and the loss to every digit are
`docs/BACKWARD4.md` §6.2's. **`docs/BACKWARD4.md`'s `grad sum digest` is deliberately not quoted
here**: the recipe behind `77d9cf34…` is not in the tree, so a value computed from a
reconstructed recipe that disagreed would say nothing about the tape. That is the same objection
`docs/BACKWARD4.md` §6.2 raises against `docs/SEQLEN.md` §1.3's digests, applied to its own.

---

## 7. What this document does not establish

| # | not established | why |
|---|---|---|
| 1 | **That A5/A6's shape occurs in a real training loop.** | §1.1 constructed it. `optimizer.step()` writes to parameters, which are leaves and not views, so the shipped path does not obviously reach it — but `nn.utils.clip_grad_norm_`, the `_foreach_` fused optimisers, and any model that writes through a slice do, and **none of them was measured.** Until that is known, W10b's priority is a guess |
| 2 | **The per-dispatch cost of an always-on recorder (W8).** | No timing was taken. `CLAUDE.md` forbids reporting one from a loaded machine and other agents were running. §3's numbers are byte counts and node counts, which contamination cannot move |
| 3 | **Whether the forward-comparison alternative to W10b works.** | §6 proposes it from §1.2's observation that the loss diverges before the gradient does. It was **not built and not costed**, and there is an obvious hole: a mutation that changes the forward by less than float32 noise would pass |
| 4 | **That a `Node` built outside `capture::record` feeds `derivative()`.** | Inherited unopen from `docs/BACKWARD2.md` §8 row 3. §4 argues the *producer* need not be written because `Recorder` is it, which is a weaker claim than having done it |
| 5 | **That §3's view-op list is the right one.** | The 18.9/75.7/302.6 MiB figures subtract slots produced by ops on a hand-written list of view spellings. A view op missing from that list inflates the number; a non-view op wrongly on it deflates it. The list was not checked against `native_functions.yaml`'s alias annotations |
| 6 | **Anything about `torch.autograd.Function`, hooks that fire, `create_graph`, or double backward.** | Inherited unchanged from `docs/BACKWARD4.md` §7 row 4 |
| 7 | **Whether W10a's counter should live on the tensor or the storage.** | §6 chooses the tensor because it is what §1.3 needs and it is one field. §2's last paragraph is the argument that the storage is where it eventually has to go. The migration cost between the two was not estimated |
