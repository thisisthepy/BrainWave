# W10a: the tape's constants stop being read at `backward()` time

`docs/BACKWARD5.md` §1.3 found a **silent wrong gradient in the shipped artefact** and §6 sized the
fix. This round landed the fix. The defect, in the program BACKWARD5 wrote it in:

```python
w = torch.tensor([2.0, 2.0, 2.0]); x = torch.tensor([1.0, 1.0, 1.0])
torch._C._capture_begin([x]); y = (x * w).sum(); tr = torch._C._capture_end(y)
tr.backward([x])   # [2.0, 2.0, 2.0]
w.add_(1.0)        # what optimizer.step() does
tr.backward([x])   # [3.0, 3.0, 3.0]   <- no error
```

**Before**, measured on `develop` `cf287d5`:

```
backward #1: [2.0, 2.0, 2.0]
w now: [3.0, 3.0, 3.0]
backward #2: [3.0, 3.0, 3.0]
```

**After**:

```
backward #1: [2.0, 2.0, 2.0]
w now: [3.0, 3.0, 3.0]
backward #2 RAISED RuntimeError: torch._C capture: constant 0 of this trace
  (torch.float32[3] on cpu) has been modified by an in-place operation since the trace
  was captured -- it is at version 1; expected version 0. This trace was captured
  *before* that tensor moved and holds it by reference, so replaying or differentiating
  it now would silently answer at the new value instead of the one the region ran on.
  Capture the region again after the update -- e.g. call trace.backward() before
  optimizer.step(), not after (docs/BACKWARD6.md)
```

Upstream 2.13.0 refuses the same program, measured in the same session:

```
RuntimeError: one of the variables needed for gradient computation has been modified by
an inplace operation: [torch.FloatTensor [3]] is at version 1; expected version 0 instead.
```

The `is at version N; expected version M` clause is kept verbatim for that reason -- upstream
refuses this by that phrase, and anything matching on it should keep matching.

Environment: worktree `work/w10a` on `develop` `cf287d5`, torch 2.13.0, transformers 5.15.1,
`/Volumes/macMini/caches/spike-venv/bin/python`, `CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-w10a`.
Every shim reading printed `shim`; every upstream reading was taken with
`env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`.

---

## 1. What landed

Split the way `CLAUDE.md` §5.3 asks for.

| | |
|---|---|
| **defect fixed** | 1 -- `docs/BACKWARD5.md` §1.3, the silent wrong gradient above. One `u64` per storage, one bump site, one `Vec<Stamp>` per trace, one comparison in `run()` |
| **feature added** | 1, and it is small: `CaptureTrace.constant_versions`, so that "the counter never moved" is distinguishable from "the counter moved and the check let it through" |
| **test added** | 1 -- `test_the_constant_version_check_sees_a_write_through_a_view` |
| **test inverted** | 1 -- `test_a_traces_constants_are_read_at_backward_time_not_at_capture_time` became `test_a_trace_refuses_to_differentiate_at_constants_that_moved_since_capture`, and asserts four clauses of the message, the unmoved case, the `replay()` case and the remedy, where it used to assert one number |
| **documentation corrected** | `docs/BACKWARD5.md` §6's sizing of the bump site, in this document rather than in place -- §3 |
| **deleted** | nothing |

W10b was **not** built, and `docs/BACKWARD5.md` §6's reasoning for deferring it is unchanged by
anything measured here. §5 is the one place this round can say something new about it.

## 2. The shape, and what it is deliberately not

| | |
|---|---|
| the field | `STORAGE_VERSIONS`, one monotonic `u64` per candle `Storage` address, in `capture.rs` |
| the bump | `capture::note_mutation`, called once per dispatch from the door in `aten.rs`, next to the capture hook and `mark_from_op` |
| the snapshot | `PyCaptureTrace::const_stamps: Vec<Option<(usize, u64)>>`, taken at `_capture_end` over `const_objects` |
| the comparison | `PyCaptureTrace::check_constants_are_fresh`, called at the top of `run()` -- so `replay()`, `backward()` and anything else built on `run()` are covered by one check |

It is not `c10::VariableVersion`. There is no `ADInplaceOrView` dispatch key, no alias set, no view
metadata and no rebasing -- the four things `docs/AUTOGRAD.md` §6 chose the tape in order to avoid.
`docs/BACKWARD5.md` §2 is the argument that a replay tape needs only the *snapshot* half of
upstream's design, applied to the 333 values a trace holds rather than to every saved variable, and
this round did not find that argument wrong.

**Only the receiver is bumped.** Bumping every tensor argument of an in-place op would have been a
line shorter and would fire on `w.add_(x)` for `x` as well -- a refusal for a constant that was
merely *read*, which is precisely the "check that fires when it should not" this round was told not
to introduce. The receiver is found the way `aten.rs`'s `tensor_receiver` finds it: positional 0, or
the keyword `self`, because `bootstrap.py` binds every argument by keyword.

**The one op that mutates without saying so in its name** -- `aten.native_batch_norm.default`, and
the judgement is per call -- is handled by reusing `mutates_this_call` and bumping arguments 3 and 4
(`running_mean`, `running_var`) rather than the receiver, which is what that kernel actually writes.

## 3. `docs/BACKWARD5.md` §6 sized the bump site wrong, by one branch

BACKWARD5 §6 says the bump goes on *"the door's existing grad-mode branch -- the dispatch hot path
already has that branch, so nothing new is tested per op"*, and the brief for this round repeated it.
**It is wrong, and putting the bump there would have produced a fix that does not fix anything.**

`tensor::mark_from_op` opens with

```rust
if !GRAD_ENABLED.load(Ordering::Relaxed) { return; }
```

and **`optimizer.step()` runs under `no_grad`** -- which `docs/BACKWARD5.md` §2 row 5 establishes in
its own table, quoting upstream's `if (!GradMode::is_enabled()) return;` in the generated in-place
wrapper. So a bump behind that branch would have missed the one call the whole round exists for: the
`add_` on a parameter after a capture. The defect would have survived the fix, and the inverted test
would have failed, which is the only reason the mis-sizing cost nothing.

The bump is therefore its own branch at the door, beside `mark_from_op` rather than inside it. That
is one extra call per dispatch and not zero, so §6 measures it instead of asserting it.

## 4. Per **storage**, not per object -- and what that does and does not buy

`docs/BACKWARD5.md` §7 row 7 left the choice open and chose the tensor "because it is what §1.3
needs and it is one field". This round chose the **storage**, at the same size, for one reason: in
this shim in-place ops go through `tensor::write_into`, which writes into the buffer the wrapper
already points at (`docs/VIEWS.md` §6), so a base and its views share one candle `Storage` and
therefore one counter for free. A per-object counter would pass every assertion about the program at
the top of this document and would not see

```python
alias = w[0:2]; alias.add_(5.0)      # w is a burned-in constant of the trace
```

which is now refused, and `test_the_constant_version_check_sees_a_write_through_a_view` is that case.

**This is not W10b and does not become it.** W10b is alias *sets* -- knowing which values in a trace
are views of which others, so that A5/A6's "the replayed forward is a different function" can be
detected. Nothing here models that; what is bought is one property that falls out of the storage
model already in the tree. `docs/BACKWARD5.md` §6's reason to defer W10b (it has no consumer until
an eager recorder exists, because `is_mutating` makes A5/A6 unreachable) is untouched, and this round
found no evidence against it.

A stamp is the pair `(storage address, version)` and not the version alone. That is what covers
`replace_with` -- `set_` and `tensor.data = ...` -- which *rebinds* rather than writes: the address
changes, no version moves, and the mismatch is caught anyway, with its own clause in the message.

**Address reuse cannot produce a false refusal for a trace constant**, and the reason is structural
rather than lucky: a trace holds a strong reference to every constant, so the storage a stamp names
cannot be freed while the stamp exists.

## 5. `torchnative.adapt`'s ordering dependency is now enforced

`docs/BACKWARD5.md` §1.3 recorded that `adapt` is correct *"by ordering and by nothing else"* --
`adapt/__init__.py:399` calls `trace.backward(...)` and `:421` calls `self._optimizer.step()`, and
nothing enforced that order.

**It is now enforced, in the only sense that matters: violating it raises by name instead of
answering.** Measured with a real `torch.optim.SGD` on a real `nn.Linear`, not with a hand-written
`add_`:

```
backward before step: ok, [(3, 4)]
after step RAISED: torch._C capture: constant 0 of this trace (torch.float32[3, 4] on cpu)
  has been modified by an in-place operation since the trace was captu...
```

Stated precisely, because "enforced" can be read too strongly: the guard does not *impose* an
ordering on a caller who has not captured anything, and it cannot make a wrong order right. What it
does is remove the failure mode -- a trace differentiated after its parameters moved no longer
returns a plausible number. `adapt` itself never reaches the check, because it captures a fresh
trace inside each `step()`; the ten-step adaptation tests
(`test_tent_adapts_an_nn_layer_norm_model_and_the_wrong_sign_does_not` and the training-mode one)
pass unchanged, which is the evidence that the guard does not fire where it should not.

## 6. Cost

`docs/BACKWARD5.md` §6 predicted "one integer increment on in-place ops only". Measured rather than
asserted, with a control built the way `docs/BACKWARD4.md` §7 built its own: the two edited source
files were reverted to their pre-round contents **in the same target directory**, rebuilt, installed
and re-measured, then restored and rebuilt. No digest from another document is trusted for the
before column.

20 000 calls of `_aten_dispatch(op, a, b)` on `float32[3]`, 9 repeats, median and min of the
per-repeat mean:

| op | before | after | delta |
|---|---:|---:|---:|
| `aten.add.Tensor` (not mutating -- the hot path) | 889.4 ns (min 883.6) | 885.6 ns (min 881.4) | **−3.8 ns, i.e. under the noise floor** |
| `aten.add_.Tensor` (mutating -- pays the bump) | 654.0 ns (min 647.9) | 680.9 ns (min 677.2) | **+26.9 ns (+4.1%)** |

The direction is the same in the medians and the minima, and the sign is the one the design
predicts: an op that does not mutate pays `is_mutating` -- one `rsplit_once` on a `&str` already in a
register -- and a branch that is not taken, which does not show above the noise; an op that mutates
pays a mutex, a `BTreeMap` entry and a read lock on candle's storage `RwLock`, which is the ~27 ns.
The SmolLM2 prefill in `docs/BACKWARD5.md` §3 issues no in-place op at all.

**The caveat, and it is the one `CLAUDE.md` requires.** `uptime` reported load averages between 1.97
and 3.24 on 8 cores across the two runs, with other agents active. That is not a quiet machine, so
**these numbers are indicative and not a measurement in the sense `docs/BACKWARD5.md` §3's byte
counts are.** What survives the contamination is the comparison, because before and after were taken
minutes apart under the same load with the same script, and the two columns disagree by 27 ns on one
op and by nothing on the other -- a load spike moves both columns of a row together. A quiet-machine
re-measurement would be worth having and this round did not have a quiet machine.

## 7. The forward did not move, and the recipe is written down

`docs/BACKWARD5.md` §6.5.1 could argue the forward was unmoved *without measuring*, because that
round's only edits were documentation and a test. This round edits the dispatch hot path, so that
argument is not available and the measurement is the evidence.

Real `HuggingFaceTB/SmolLM2-135M`, `float32`, greedy prefill of `torch.arange(S).unsqueeze(0)`.
**The recipe, because a digest whose recipe is not in the tree proves nothing** (the objection
`docs/BACKWARD4.md` §6.2 raises against `docs/SEQLEN.md` §1.3, and against its own `grad sum
digest`): sha256 over `struct.pack("<%df" % len(row), *row)` for each row of `logits.tolist()[0]` in
order -- i.e. the little-endian float32 bytes of the flattened `[1, S, 49152]`.

| S | before (reverted sources, rebuilt) | after | |
|---:|---|---|:--:|
| 6 | `606e00d1be05fccce84b23b02dae8f473d490ad0f706b9b612036dd55cee531b` | identical | ✅ |
| 32 | `62d923807cce47cee35a5e7b6091c9544e180c6c9ff771bb4cad3f478f4cbb79` | identical | ✅ |
| 128 | `4750428e94f7383c42853b0d547192a7946fdde9e086d3b16cebdeb2ca68e37d` | identical | ✅ |

All three also equal `docs/BACKWARD5.md` §6.5.1's and `docs/BACKWARD4.md` §6.2's -- which is a
stronger result than it looks, because the recipe above was written for this round from the
description in those documents rather than copied from a script, so the agreement is a
reproduction of the digest and not a re-run of it.

## 8. Every gradient this round produced, checked element-wise against upstream

`docs/CAPTURE.md` §9-1 records this project's other instance of a plausible-but-wrong gradient, and
`docs/BACKWARD5.md` §1.1 records a measurement that nearly missed a factor of 100 by choosing a
program whose gradient could agree by coincidence. So the check here is on a program whose gradient
**cannot** agree by coincidence -- `loss = ((x @ w.T + b).tanh() ** 2).sum()`, which is nonlinear in
every argument -- and it is element-wise, not a digest.

| quantity | elements | max abs difference from torch 2.13.0 |
|---|---:|---|
| `loss` | 1 | `0.000e+00` |
| `d(loss)/dx` | 8 | `1.192e-07` |
| `d(loss)/dw` | 12 | `8.941e-08` |
| `d(loss)/db` | 3 | `8.941e-08` |
| `loss` at the moved weights | 1 | `4.768e-07` |
| `d(loss)/dw` at the moved weights | 12 | `1.192e-07` |

37 elements, all at float32 rounding. The last two rows are the **remedy** the refusal names: after
`w.add_(0.5)`, the stale trace refuses, and a trace captured again over the moved weights gives the
gradient at the new point, which upstream agrees with. A refusal that could not be worked around
would be a different and worse answer than the defect.

## 9. Gates

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh
    382 ok, 0 FAIL          (381 before; +1 added, 1 inverted, 0 removed, 0 weakened)
    SELF-TEST: PASS -- 19 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised
    DOCWATCH: PASS -- 340/340 evaluated marker(s) hold   (334 before; +6, all in this document)
    EXIT=0

TORCH_C_ARTEFACT=... $PY tools/golden/compare.py
    SUMMARY: 8476/8476 cases passed, 0 failed, ops covered=203, pending case builders=0
    EXIT=0
```

`ops=203` is unchanged **on purpose** -- no kernel landed, so nothing here could have moved it.

**Both gates must be run with `TORCH_C_ARTEFACT` set, and the golden one silently is not.**
`tools/golden/loader.py:33` falls back to `/Volumes/macMini/caches/cargo-target/release/lib_C.dylib`
-- the *shared* target directory -- when the variable is unset, so a run from a worktree with its own
`CARGO_TARGET_DIR` measures **another agent's binary** and says nothing about the tree it was
launched in. This round hit it: a bare `compare.py` reported `8470/8476` with six `float8_e4m3fn`
in-place cases announcing *"gap appears CLOSED"*, which is a real and interesting result about
somebody else's build and was not reproducible in this worktree by any direct call. The six are not
this round's, in either direction. With the variable set, this worktree's artefact is 8476/8476.

## 10. What this document does not establish

| # | not established | why |
|---|---|---|
| 1 | **A quiet-machine cost.** | §6 was taken at load 2.0-3.2 on 8 cores with other agents running. The before/after comparison survives that; the absolute ns/call does not |
| 2 | **That the table cannot grow without bound.** | `STORAGE_VERSIONS` gains an entry per distinct storage ever written in place and never loses one. For a training loop that is the parameters, re-entered at the same addresses each step, and for an inference forward it is nothing. A program that mutates a long stream of temporaries would grow it, and **that was not measured**. The entry is 24 bytes; the fix, if it is ever needed, is to key on something the tensor owns rather than on an address |
| 3 | **Anything about A5/A6.** | W10b was not built. `docs/BACKWARD5.md` §6's reasoning is unchanged, and §4 above is careful that a storage-keyed counter is not a down payment on it |
| 4 | **That a mutation which bypasses the door is caught.** | It is not, and neither is it upstream (`docs/BACKWARD5.md` §2 row 6a, where upstream's own gradient is wrong). Everything that reaches candle's buffer without going through `_aten_dispatch` is invisible here |
| 5 | **That the check is free for a trace with many constants.** | `run()` now walks 333 constants and takes 333 mutex acquisitions before replaying. That is a per-*replay* cost, not a per-op cost, and against a 1862-node forward it should vanish -- **should**, because it was not separated out of §6's measurement |
| 6 | **Whether the six `float8_e4m3fn` in-place cases really are a closed gap.** | §9's note: they came from the shared target directory's artefact, not from this worktree's. Somebody's build fills F8 in place now. Chasing it here would have been reporting on another agent's tree |
| 7 | **W9, W8, or the eager recorder.** | Not started. `docs/BACKWARD5.md` §6's order was W10a → W9 → W8; this is the first of those and, as that section insisted, it does not commit the project to the rest |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs note_mutation present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs check_constants_are_fresh present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs const_stamps present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs STORAGE_VERSIONS present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_a_trace_refuses_to_differentiate_at_constants_that_moved_since_capture present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_constant_version_check_sees_a_write_through_a_view present -->
