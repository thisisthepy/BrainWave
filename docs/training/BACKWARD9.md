# W12: `loss.backward()`

`docs/training/BACKWARD7.md` §6 is a list of seven things standing between an eager recorder that works and
the spelling every tutorial uses. `docs/training/BACKWARD8.md` closed the two on it that turned out to be
defects. This document is the rest of the list, and the sentence it exists to make true:

```python
loss = criterion(model(x), y)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

runs, six steps of it, on a real `nn.Sequential` with a real `torch.optim.SGD`, and every loss and
every final parameter agrees with upstream torch 2.13.0 running the identical program to **2.98e-08**.

Environment: worktree `work/backward` on `develop` `6d016f0`, torch 2.13.0, transformers 5.15.1,
`/Volumes/macMini/caches/spike-venv/bin/python`,
`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-backward`. Every shim reading printed `shim`;
every upstream reading was taken with `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`. Six agents were
on the machine, so **no time is reported anywhere below**.

---

## 0. Answers, before the evidence

| question | answer |
|---|---|
| **Does `loss.backward()` work?** | **Yes**, through upstream's own `torch/_tensor.py` → `torch/autograd/__init__.py` → `_engine_run_backward` → `_ImperativeEngine.run_backward`. Nothing in the test program calls a shim-specific name. §1 |
| **Does a training loop close?** | **Yes.** Six steps of `backward` / `step` / `zero_grad` on `nn.Sequential(Linear(4,3), Tanh, Linear(3,2))` with `torch.optim.SGD(lr=0.1)`. Losses and final parameters compared element-wise to upstream: **52 floats, worst 2.98e-08**, which is one float32 ulp at that magnitude. §1 |
| **What was the piece with design content?** | `.grad` accumulation, and it had **two** defects that only a measurement finds. The tape returns *the same tensor object* for both operands of an `add`, so storing it would make two leaves share one gradient; and `sum()`'s gradient is *expanded*, `stride() == (0,0)`, so a plain `clone()` gives a `.grad` whose first in-place write refuses. Both are real programs, not hypotheses. §2 |
| **`retain_graph`?** | **Built.** `Recorder::duplicate_for_retained_backward` -- the tape survives its own backward by being duplicated, because the backward moves every field into a trace and an `Env` and there is nothing left to put back. §3 |
| **`allow_unused`?** | **Closed, and closed in the place that makes both defaults reachable.** `docs/training/BACKWARD7.md` §6 called ours an "opposite default"; it is upstream's now. Where the fix went is the content -- §4 |
| **`create_graph`, `autograd.Function`, hooks, `retain_grad` on non-leaves?** | **Refused by name and sized.** §6. `create_graph=True` would be silently wrong rather than slow, which is why it raises rather than degrading |
| **Did the forward move?** | **No.** The three `docs/training/BACKWARD7.md` §7 digests reproduced from the recipe in the tree. §7 |

---

## 1. The training loop, against upstream

The program is one file, run twice in two interpreters -- once with the vendored tree on
`PYTHONPATH` and once with it removed -- and the two outputs compared field by field. That is
`docs/distributed/FEDERATED3.md`'s central-oracle pattern, and it is why nothing below is a literal: a hardcoded
trajectory is a claim about 2.13.0 that nothing re-checks, and `docs/verification/AUDIT.md` found that shape of
claim stale six times out of eleven.

`nn.Sequential(Linear(4,3), Tanh(), Linear(3,2))`, `SGD(lr=0.1)`,
`criterion = lambda o, t: ((o - t) ** 2).mean()`, six steps.

| quantity | elements | max abs difference | `max\|upstream\|` |
|---|---:|---|---:|
| loss trajectory | 6 | `2.980e-08` | 0.2832 |
| every parameter gradient, step 0 | 23 | `2.980e-08` | 0.5607 |
| every parameter, after six `optimizer.step()`s | 23 | `2.980e-08` | 0.4925 |

`2.98e-08` is `2**-25`: a single float32 ulp at that magnitude, in a quantity that has been through
six multiply-accumulate rounds. The losses fall monotonically `0.2832 → 0.1001` in both
interpreters.

Three things about the program are deliberate:

* **The parameters are initialised arithmetically, not from a seed.** The two interpreters do not
  share an RNG stream, so seeded initialisation would be comparing two different programs.
  `docs/training/BACKWARD8.md` §3 met the same wall from the other side, where a mask had to be *shared*
  rather than avoided.
* **The criterion is spelled out rather than `nn.MSELoss`.** `nn.MSELoss` reaches
  `torch.broadcast_tensors`, which has no overload-table entry in this shim and refuses by name.
  That is a gap in the op surface and not in the engine -- the loop still calls a criterion, and
  `.backward()` does not know the difference -- but it is named here rather than left as a silent
  simplification.
* **`.grad`'s whole protocol is asserted beside the numbers**: `None` before the first backward, a
  leaf (`grad_fn is None`, `requires_grad False`) after it, `None` again after `zero_grad()`, and
  zeroed **in place** after `zero_grad(set_to_none=False)`. A loop that never populated `.grad`
  would still produce a self-consistent flat "trajectory", so the numbers alone do not say the
  gradient arrived.

`test_a_real_training_loop_runs_through_loss_backward_and_agrees_with_upstream`.

### 1.1 What `run_backward` is, and what it deliberately is not

It is a **translation**, in `bootstrap.py`, and it holds no derivative rule, no traversal and no
lifetime rule. Everything it does is turn upstream's call

    run_backward(tensors, grad_tensors, keep_graph, create_graph, inputs,
                 allow_unreachable=False, accumulate_grad=False)

into `torch._C._eager_backward(output, grad_output, wrt, retain_graph)` and turn the answer back.
Putting any of the three back into it would be a second place for the eager tape and the capture
tape to disagree, which is the property `docs/training/BACKWARD7.md` §2 bought by splitting `tape::backward`
in two rather than writing a second one.

The two entry points differ by exactly two flags, which is why one function serves both:
`torch.autograd.backward` passes `allow_unreachable=True, accumulate_grad=True` and ignores the
return; `torch.autograd.grad` passes **`allow_unused` in the `allow_unreachable` slot**,
`accumulate_grad=False`, and returns the tuple. `allow_unused` is not a separate parameter anywhere
in torch, and it is not one here.

## 2. `.grad` accumulation, and the two defects a measurement found

`docs/training/BACKWARD7.md` §6 called this "the piece with real design content" and said the interaction
with the storage guard had not been designed. The interaction turned out to be the easy half, and
two things nobody had named turned out to be the hard half.

**The guard half, first, because it is short.** Accumulation writes in place, and
`docs/training/BACKWARD8.md`'s guard poisons a tape when a write lands on a storage it holds. It cannot fire
here, and for a structural reason rather than a lucky one: `_eager_backward` **takes** the tape at
entry, so by the time anything is written to `.grad` there is no tape to poison. The write is also
performed with grad mode off, so it cannot start a new one. `forgive_own_write` and this are
independent, and neither had to learn about the other.

### 2.1 Two leaves were given one gradient tensor

`z = x + y` has the same derivative for both operands, and the tape hands back **the same object**
for both:

```
same object: True
same storage: True
```

An engine that stored what the tape returned would make `x.grad` and `y.grad` one tensor, and the
*next* step's in-place `+=` would double the other leaf's gradient. Nothing about that is an error a
caller could see; it is a wrong number, arriving one step later than the cause.

`_accumulate_into_grad` therefore copies on the first store.
`test_two_leaves_of_one_add_do_not_share_one_gradient_tensor` performs the write and reads the other
leaf afterwards, because identity alone would pass for a copy that still aliased.

### 2.2 ... and the obvious copy is not writable

The first version copied with `clone()`, and the very next program refused:

```
RuntimeError: unsupported operation: more than one element of the written-to tensor refers to a
single memory location. Please clone() the tensor before performing the operation.
```

`sum()`'s gradient is the seed **expanded** to the operand's shape -- `stride() == (0, 0)`, one
element of storage read from every position -- and `clone()` preserves that layout. Stored as
`.grad`, it is a tensor whose first in-place write refuses. So `p.grad.zero_()`,
`zero_grad(set_to_none=False)` and every `foreach` optimizer would fail on the **second step of any
loop whose loss ends in `.sum()`**, which is the most ordinary loss there is.

`_dense_copy_of` is `contiguous()` followed by `clone()` **only when `contiguous()` returned the
same storage**. That is exactly one materialisation in both cases rather than two in one, and the
`data_ptr` comparison is what makes it so.

Neither defect is reachable by reading the code. Both were found by running a program and reading
the failure.

### 2.3 The semantics, each row against upstream

`+=` and not `=`; identity stable across accumulation; `None` before the first backward; `None`
after `zero_grad()`; zeroed in place after `zero_grad(set_to_none=False)`. The identity row is the
one an implementation gets wrong in the *other* direction -- `leaf.grad = leaf.grad + g` accumulates
correctly and breaks every `foreach` optimizer and `zero_grad(set_to_none=False)`, both of which
hold `p.grad` across steps.

Measured in both interpreters: two backwards without a `zero_grad()` give exactly twice the
gradient, and `b - 2a` is **0.000e+00** on all 23 elements in both.

## 3. `retain_graph`

W9's lifetime rule is upstream's `retain_graph=False` **default**: the backward that walks the graph
frees it. `retain_graph=True` is the caller saying they will walk it again -- and `eager_backward`
*takes* the tape and moves every field into a `PyCaptureTrace` and an `Env`, so there is nothing
left to put back.

`Recorder::duplicate_for_retained_backward` is the answer, and duplicating rather than borrowing is
the load-bearing choice: the backward runs Python, Python runs kernels, and a kernel that reached
`record_into` would try to borrow the same `RefCell` this call is holding. Duplicating releases the
borrow before any of that. It is not a copy of the tensors -- every `Py` in it is a reference count
-- so what it costs is the node list, which is the honest price of asking for the graph twice.

Two smaller decisions inside it:

* **`const_stamps` is duplicated as it stands, not re-read.** A retained backward asks the same
  freshness question twice, so a write between the two must still be refused by the second.
* **`Arg::duplicate` is hand-written rather than derived.** `Py<PyAny>` is deliberately not `Clone`
  in this pyo3 build; `clone_ref` takes the `Python` token, so the duplication is explicit about
  being a refcount rather than a copy of a value.

Measured: `s.backward(retain_graph=True)` twice accumulates to exactly twice the gradient in both
interpreters, the tape is the same size afterwards, and a third backward without the flag frees it
and the fourth refuses with upstream's *"Trying to backward through the graph a second time"*.

## 4. `allow_unused`, and where the fix went

`docs/training/BACKWARD7.md` §6 recorded that `_eager_backward` returns `None` for a leaf no gradient reached
where upstream **raises** unless `allow_unused=True`, and called them "two different defaults".

They are one default now, and **where the fix went is the whole content**. The obvious wiring passes
`inputs` down as `_eager_backward`'s `wrt`, and `_eager_backward` *raises* for a tensor no recorded
op read. That makes `allow_unused=True` unreachable rather than merely wrong -- there is no argument
that gets a `None` out of it. So `run_backward` asks the tape for **every** leaf (`wrt=None`) and
selects in Python, which is what makes both defaults expressible.

One divergence remains and is named rather than fixed: **upstream checks `inputs` before it
traverses and this checks after.** Both raise; upstream raises without doing the work. A program
that asks for an unused input on a freed graph therefore gets *"backward through the graph a second
time"* here and *"appears to not have been used"* upstream. It is an ordering difference in two
refusals, and it cost a test's first draft (§8 row 3).

## 5. What was checked by being switched off

`CLAUDE.md` §5.5. Six nullifications, each a one-line change to a shipped source, rebuilt and
re-run. **Every one turns something red, and they name different things.**

| # | nullified | red | green (control) |
|---|---|---|---|
| 1 | `.grad` stores the tape's tensor -- no copy at all | `two_leaves_...`: *both leaves were given one gradient tensor*; `..._dense_enough_...`: stride `(0,0)` | `..._accumulates_rather_than_assigning`, the training loop |
| 2 | `_dense_copy_of` → `gradient.clone()` (a copy, but not a dense one) | `..._dense_enough_...`: stride `(0,0)`; `two_leaves_...`: *more than one element of the written-to tensor refers to a single memory location* | the training loop |
| 3 | `+=` → assign | `..._accumulates_rather_than_assigning`: *the second backward assigned instead of accumulating* | the training loop, `two_leaves_...` |
| 4 | `retain_graph` ignored (tape taken either way) | `retain_graph_...`: *retain_graph=True freed the tape anyway, 0 != 2* | `..._engine_answers_...`, the training loop |
| 5 | `allow_unreachable` never raises | `allow_unused_...`: *an unused input returned instead of raising* | `..._engine_answers_...`, the training loop |
| 6 | `accumulate_grad` ignored -- `.grad` never written | the training loop: subprocess exits 1 on `p.grad.reshape` of `None`; `..._engine_answers_...`: *run_backward answered but wrote no .grad* | `allow_unused_...` |

**Two of those rows are worth more than a tick.**

* **Row 3's control is the interesting one.** The training loop **passes** when accumulation is
  replaced by assignment, and it should: `zero_grad(set_to_none=True)` drops `.grad` every step, so
  the loop never accumulates twice into anything. A round that had only built the loop would have
  shipped `=` for `+=` and been green. That is the argument for the unit test existing beside the
  road test, made by measurement rather than by taste.
* **Row 2 did not produce the failure it was written for.** `two_leaves_...` was expected to stay
  green under it (a clone is still a distinct object) and instead went red *on the layout*, from the
  in-place write it performs to prove non-aliasing. The two defects of §2 are entangled in that
  test; row 1 is where they separate, and row 1 is the one that reports them under two different
  names.

## 6. What still refuses, by name

Each of these raises where a caller meets it. None degrades quietly.

| refused | why it is a refusal and not an approximation | size |
|---|---|---|
| **`create_graph=True`** / double backward | The eager backward runs under `NoGradGuard`, so its own ops are **not recorded**. A `create_graph=True` that behaved like `False` would find an *empty* graph on the second backward rather than fail -- a gradient penalty would carry the wrong number all the way to a loss | The backward would have to run *on* the tape: recording its own ops, which means lifting the `NoGradGuard` and making `backward_in`'s operands tape values. Not small |
| **more than one root tensor** | The eager tape has one output. Seeding several and summing into one traversal is a different traversal | Small: seed each root's `Ref` and merge before `reachable()`. Not done because nothing needed it |
| **`GradientEdge` inputs** | The tape is addressed by tensor identity and has no name for an edge | Needs a public name for a `Ref`, which is a surface decision |
| **`torch.autograd.Function`** | Not looked at, and **deliberately not half-built.** A custom `Function` needs `ctx`, `save_for_backward`, a node the tape can hold whose backward is Python, and `once_differentiable` | The largest of these by far. It is the first thing here that makes the tape hold a *callback* rather than a recorded op, and it wants its own round |
| **hooks** (`register_hook`, `post_accumulate_grad_hook`) | Not fired, not enumerated | Needs a per-node and per-leaf hook list and a firing point in `backward_in`. Medium, and cheaper after `Function` than before it |
| **`retain_grad` on non-leaves** | `retains_grad` reports `True` and `.grad` stays `None`. The engine accumulates onto the tape's *constants*, and a non-leaf is a node result. The tape holds the value; nothing surfaces it | Small: `backward_in` already computes it. It needs a way to name the node from Python, which is the `GradientEdge` problem again |
| **`nn.MSELoss` and anything through `torch.broadcast_tensors`** | No overload-table entry. Not an engine gap; named here because §1's criterion is spelled out because of it | An `overloads.json` entry, owned elsewhere |

## 7. The forward did not move

This round edits `record_into`'s caller and adds a parameter to `eager_backward`, so the measurement
is the evidence rather than an argument about compile inputs. `docs/training/BACKWARD7.md` §9.1's recipe is in
the tree -- sha256 over `struct.pack("<%df" % len(row), *row)` for each row of `logits.tolist()[0]`,
real `HuggingFaceTB/SmolLM2-135M`, float32, greedy prefill of `torch.arange(S).unsqueeze(0)` -- and
the script was written from it rather than copied.

| S | sha256, this build | |
|---:|---|:--:|
| 6 | `606e00d1be05fccce84b23b02dae8f473d490ad0f706b9b612036dd55cee531b` | ✅ |
| 32 | `62d923807cce47cee35a5e7b6091c9544e180c6c9ff771bb4cad3f478f4cbb79` | ✅ |
| 128 | `4750428e94f7383c42853b0d547192a7946fdde9e086d3b16cebdeb2ca68e37d` | ✅ |

All three equal `docs/training/BACKWARD8.md` §5's and through it `docs/training/BACKWARD7.md` §9.1's,
`docs/training/BACKWARD6.md` §7's, `docs/training/BACKWARD5.md` §6.5.1's and `docs/training/BACKWARD4.md` §6.2's -- a six-round
chain, one link of which was taken against a deliberately reverted build.

The reason it *can* be unmoved: nothing on the dispatch path changed. `retain_graph` is a parameter
of `eager_backward`, which a forward never calls, and `run_backward` is Python that a forward never
reaches.

## 8. Two tests were inverted, not deleted

Both carried an instruction to invert, and both got a *stronger* assertion rather than a removed one.

1. **`test_tensor_backward_still_refuses_even_though_an_eager_graph_exists`** →
   `test_the_engine_answers_now_that_an_eager_graph_exists`. Its docstring said "Invert this test
   when the engine lands; do not delete it." What replaces the refusal is four rows, and the fourth
   is the control: `accumulate_grad=False` must **return** and not write, because an engine that
   wrote `.grad` unconditionally would pass the first three and be wrong about
   `torch.autograd.grad`. The `docs/training/BACKWARD7.md` DOCWATCH marker follows the test to its new name
   so the lineage stays greppable.
2. **`test_the_autograd_boundary_is_where_autograd_md_says_it_is`**, wall 3. It had already been
   inverted once, for a wall that turned out not to be autograd (`_stash_obj_in_tls`). This time the
   wall was the right one and it opened: the assertion is now that the engine differentiates
   `y = x·x` at `x = 2` to `4` and **writes it into `x.grad`**, which is the second of the two
   boundaries that docstring keeps apart and the one that had never moved.
3. **`test_the_backward_seed_is_absent_and_nothing_guesses_a_one`** kept its shape and changed
   sides. Its pair was "the seed is right AND nothing consumes it"; it is now "the seed is right AND
   what consumes it produces upstream's number", checked against the oracle rather than against the
   literals beside it. Its first draft was wrong in a way worth recording: it asked
   `torch.autograd.grad` for a gradient over a graph `.backward()` had already freed, and got
   *"backward a second time"* where upstream got *"appears to not have been used"* -- §4's ordering
   difference, found by a test rather than by reading.

## 9. Reported by kind, not by count

`CLAUDE.md` §5.3.

| kind | what |
|---|---|
| **feature added** | `_ImperativeEngine.run_backward` -- `Tensor.backward()` and `torch.autograd.grad()` both work |
| **feature added** | `.grad` accumulation onto leaves: dense first store, in-place `+=` after, stable identity |
| **feature added** | `retain_graph=True` (`Recorder::duplicate_for_retained_backward`, `_eager_backward`'s fourth argument) |
| **feature added** | `backward(inputs=...)` and `torch.autograd.grad(..., allow_unused=)`, both defaults upstream's |
| **defect prevented** | two leaves of one `add` sharing a gradient tensor (§2.1) -- found by measurement, would have been a wrong number one step later |
| **defect prevented** | a `.grad` no in-place write can reach, for any loss ending in `.sum()` (§2.2) |
| **tests added** | 7: engine answers, two leaves, dense grad, accumulation, retain_graph, allow_unused, create_graph/multi-root refusals |
| **tests added** | 1 road test: a six-step training loop against upstream, element-wise |
| **tests inverted** | 3 (§8), none deleted, none weakened |
| **documentation corrected** | four docstrings in `bootstrap.py` and two comments in `capture.rs` that described a wall that is no longer there |

## 10. Gates

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh
    463 ok, 0 FAIL          (456 before; +8 added, 1 replaced by its inversion)
    DOCWATCH: PASS -- 456/456 evaluated marker(s) hold   (442 before; +14, all in this document)
    EXIT=0

TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib $PY tools/golden/compare.py
    SUMMARY: 8921/8921 cases passed, 0 failed, ops covered=222, pending case builders=0
    EXIT=0
```

`ops=222` is unchanged on purpose: no kernel landed here.

## 11. What this document does not establish

| # | not established | why |
|---|---|---|
| 1 | **That a real model trains through this.** | §1's loop is `nn.Sequential(Linear, Tanh, Linear)` on 5 rows. `docs/training/BACKWARD8.md` §2.1 differentiates SmolLM2-135M through `_eager_backward`, but nothing here runs an `optimizer.step()` loop over a transformer, and `docs/training/BACKWARD8.md` §8 row 2's missing convolution derivative still blocks the vision half |
| 2 | **Anything about `torch.autograd.Function`.** | §6. Refused by name and sized, not started. It wants its own round |
| 3 | **That `create_graph` is only unbuilt rather than hard.** | §6 argues it needs the backward to run *on* the tape. Nothing here tried it |
| 4 | **Any per-dispatch or per-backward cost.** | Six agents were on the machine. `docs/training/BACKWARD7.md` §10 row 1 stands unchanged, and this round reports no time at all |
| 5 | **That `.grad`'s interaction with the storage guard has no case left.** | §2 argues it structurally cannot fire, because the tape is taken before the write and the write is off grad mode. That is an argument about two code paths, not a sweep |
| 6 | **`backward()` across a capture-region boundary.** | `docs/training/BACKWARD7.md` §10 row 7, unchanged. `eager_record` returns early while a region is open, so those ops are absent from the eager graph, and now that `.backward()` exists the question is askable and still unasked |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_engine present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _accumulate_into_grad present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _dense_copy_of present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs duplicate_for_retained_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_engine_answers_now_that_an_eager_graph_exists present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_two_leaves_of_one_add_do_not_share_one_gradient_tensor present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_accumulated_grad_is_dense_enough_to_be_written_in_place present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_run_backward_accumulates_rather_than_assigning_on_the_second_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_retain_graph_differentiates_the_same_forward_twice present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_allow_unused_is_the_allow_unreachable_slot_and_both_defaults_are_upstreams present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_engine_refuses_create_graph_and_several_roots_by_name present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_a_real_training_loop_runs_through_loss_backward_and_agrees_with_upstream present -->
<!-- DOCWATCH: count smoke_ok ge 463 -->
<!-- DOCWATCH: count golden_cases_passed ge 8921 -->
