# W8 and W9: an eager recorder, and the lifetime that makes it finite

`docs/BACKWARD5.md` §4 asked what an eager recorder would have to produce and answered that the
producer need not be written, because `Recorder` is already it -- four of the five things
`tape::backward` reads off a trace, and the fifth already existing under another name:

> **`docs/BACKWARD2.md` §1.5 asked, of W8, "what keeps the intermediates alive". The answer is that
> the thing that keeps them alive already exists and is thrown away.** An eager `Env` is `keepalive`
> not dropped.

**That is true, and it was the whole round.** This document is what it made possible, what it did
not, and the one place the round found `docs/BACKWARD5.md` too pessimistic.

Environment: worktree `work/eager` on `develop` `9f4557e`, torch 2.13.0, transformers 5.15.1,
`/Volumes/macMini/caches/spike-venv/bin/python`,
`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-eager`. Every shim reading printed `shim`;
every upstream reading was taken with `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`.

---

## 0. Answers, before the evidence

| question | answer |
|---|---|
| **Is `Recorder::keepalive` really `Env`?** | **Yes, and the reshape is four lines.** `keepalive: Vec<Py<PyAny>>` became `node_objects: Vec<Vec<Option<Py<PyAny>>>>`, which is `Env::nodes` with `None` in the `Slot::Other` positions. Nothing else was needed: `Env::consts` is `const_objects`, and `Env::inputs` is empty because an eager tape has no declared inputs -- everything it reads from outside itself is a constant. §1 |
| **Did the derivative rules get reused, or rewritten?** | **Reused verbatim.** `tape::backward` was split in two at the line `let env = trace.run(...)`; the half below became `backward_in`, which takes an `Env` it did not build. All 60 rules, `derivative()`, `wrt_set()`, `reachable()` and `wanted()` are shared, and a test differentiates one program both ways and asserts the two agree element-wise. §2 |
| **`docs/BACKWARD5.md` §4 expected `reachable()` to need replacing. Did it?** | **No, and the reason is worth keeping.** With no trace inputs, `reachable`'s rule -- a node is needed if it reads a wanted constant or a needed node -- **is** the propagated `requires_grad` flag, computed from the same information one step earlier. That was the one place §4 was pessimistic. §2.1 |
| **What gates the recorder?** | `tensor::mark_from_op`'s return value, which was `()` and is now `bool`. An op is on the eager tape **exactly when upstream would have given its result a `grad_fn`**, and that question is already answered at the door. A dispatch that differentiates nothing pays a `bool` in a register. §3 |
| **W9's lifetime rule** | Upstream's `retain_graph=False` default: the backward that walks the graph frees it, and a second backward raises by name. §4 |
| **Did the aliasing verdict change?** | **No, and now for a stronger reason than `docs/BACKWARD5.md` §6 had.** A recorder exists, so W10b has a consumer for the first time -- and it is still not needed, because **a tape that refuses does not have to know what aliases what.** `docs/BACKWARD6.md` §4 keyed versions on the candle `Storage`, so "a write landed on bytes this graph depends on" is one hash lookup, and A5/A6 are refused by name instead of answered 100× wrong. §5 |
| **Does `_ImperativeEngine.run_backward` work now?** | **No. It still refuses, by name, and the refusal is tested.** W8 and W9 landed; the engine is a separate thing and §6 says exactly what is between them. |
| **Every gradient checked against upstream?** | Yes, element-wise, on programs that cannot agree by coincidence: 91 elements over three programs plus a four-step gradient-descent loop, max absolute difference `1.9e-06` at `max|upstream| = 10.4`, i.e. float32 rounding. §7 |

---

## 1. `Recorder::keepalive` is `Env`, and what that made possible

The claim `docs/BACKWARD5.md` §4 made is checkable field by field. It holds.

| `tape::backward` reads | before this round | now |
|---|---|---|
| `trace.nodes: Vec<Node>` | `Recorder::nodes`, identical type | unchanged |
| `trace.inputs: Vec<TensorMeta>` | `Recorder::inputs` | `Vec::new()` eagerly -- see below |
| `trace.consts: Vec<TensorMeta>` | `Recorder::consts` | unchanged |
| `trace.outputs: Vec<Ref>` | set at `_capture_end` | eagerly, `known[addr]` of the tensor `backward()` was called on -- one lookup, as §4 predicted |
| `trace.run(py, inputs) -> Env` | **the whole difference** | not called. The `Env` is `node_objects`, which never stopped holding the values |

The reshape §4 flagged as "not free of judgement" cost four lines:

```rust
-    keepalive: Vec<Py<PyAny>>,
+    node_objects: Vec<Vec<Option<Py<PyAny>>>>,
```

`Option` rather than a flat `Vec` because `keepalive` skipped `Slot::Other` results and `Ref` does
not, so the two were never 1:1; `None` fills those positions and `Env::get` can index the way `Ref`
indexes. The declared inputs of a *region* moved to their own `input_objects`, which is all
`keepalive` was ever doing for them.

**The interesting consequence is the one about inputs.** An eager tape has none. There is no
`_eager_begin` to declare them at, so every tensor the tape reads from outside itself becomes a
`Ref::Const` -- parameters, activations from a graph that has already been freed, and the leaves the
caller will ask for gradients of, all in one list. That is not a compromise; `Ref::Const` is
precisely "a value the record did not produce", which is what a leaf is. It is also why §2.1 below
gets `reachable()` for free.

**What §4 could not say, and this round can: the retention is the *only* thing W9 costs.**
`docs/BACKWARD5.md` §3 measured 18.9 MiB at `S=8` and 302.6 at `S=128` for SmolLM2-135M and called
it a *retention* cost rather than an *allocation* cost, because the tape already holds those bytes
during the forward. The code now says so: not one `clone()` of a tensor was added. `record()` pushed
every result into `keepalive` before this round for the identity reason, and the change is entirely
in when the vector is dropped.

## 2. The rules were reused, and the reuse is a test rather than a claim

`tape::backward` was split at one line:

```rust
pub fn backward(py, trace, inputs, grad_outputs, wrt_constants) -> ... {
    ...
    let env = trace.run(py, inputs)?;                       // the replay
    backward_in(py, trace, &env, grad_outputs, wrt_constants)
}
pub(crate) fn backward_in(py, trace, env: &Env, grad_outputs, wrt_constants) -> ...   // the walk
```

Everything below that line -- the seeds, the reverse loop, `derivative()`, `accumulate()`, the
`Slot::Other` guard, the output dict -- moved into `backward_in` **unchanged**, and `tape.rs`'s
diff for this round is that split and nothing else. `_eager_backward` projects the `Recorder` onto a
`PyCaptureTrace`, builds an `Env` out of `node_objects`, and calls `backward_in`.

`test_the_eager_tape_and_the_capture_tape_are_the_same_derivative_rules` differentiates
`sum(tanh(x*w)²)` twice -- once through `_capture_begin`/`_capture_end` and `CaptureTrace.backward()`,
which reconstructs the forward by replaying it out of shapes, and once through the eager recorder,
which never replays anything -- and asserts the two gradients are **equal, element-wise**, and that
they are not trivially equal by being all ones or all zeros. Two independent rule sets could not
produce that; one rule set called twice does.

This project has said *"do not write the second one"* about `full`/`full_like`, about
`norm`/`linalg_vector_norm` and about the tape itself. That test is where the saying becomes
checkable.

### 2.1 `reachable()` did not need replacing, and that is the round's one correction to `docs/BACKWARD5.md`

§4's last paragraph says:

> `reachable()` is the one function that would be replaced rather than reused -- its own doc comment
> says why, and `docs/BACKWARD4.md`'s propagated `requires_grad` flag is now the other of the two
> sources it names.

It is reused. The reason is §1's observation about inputs. `reachable` marks node *i* as needed if
any operand of *i* is a wanted constant or a needed node; the `Ref::Input` arm is dead eagerly,
because there are no inputs. What is left is: **a node is on the gradient path if it reads a leaf
that requires a gradient, or reads a value produced by such a node.** That is the propagated flag,
stated over the record instead of over the tensors, and it is the same information one step earlier
-- which is what `reachable`'s own doc comment already said it was.

The practical difference is that nothing has to be *stored* per tensor and nothing can go stale.
`docs/BACKWARD4.md`'s `from_op` still decides whether an op is recorded at all (§3); it is not
consulted again during the walk.

## 3. The gate: `mark_from_op` returns a `bool`

`docs/BACKWARD5.md` §7 row 2 lists the per-dispatch cost of an always-on recorder as *"the number
nobody has"*, and `docs/CAPTURE.md` §7 is the reason to care -- the capture hook is one relaxed
atomic load precisely because this door is measured against small numbers.

The recorder asks **no question of its own.** `tensor::mark_from_op` already computes upstream's
five-clause condition for a result having a `grad_fn`; it returned `()` and now returns whether it
marked anything, and the door reads that:

```rust
let on_graph = crate::tensor::mark_from_op(py, op, args, kwargs, &out);
if on_graph && crate::capture::eager_enabled() {
    crate::capture::eager_record(py, op, args, kwargs, &out);
}
```

Three things follow, and they are one decision rather than three:

* **Cost.** A dispatch with nothing requiring a gradient -- every inference forward, every golden
  case, every `torch.load` -- reaches `mark_from_op`'s existing early return and then tests a `bool`
  already in a register. No second walk of the argument tuple, no atomic load, no allocation. §8.
* **Correctness.** "On the eager tape" and "has a `grad_fn`" are the same predicate by construction
  rather than by two lists agreeing. `test_the_eager_recorder_records_exactly_the_ops_that_get_a_grad_fn`
  asserts both halves on `detach` (a `NOT_DIFFERENTIABLE` entry: no `grad_fn`, no node) and under
  `no_grad`.
* **No in-place op is ever recorded**, because `mark_from_op`'s last clause is *"the result is a new
  tensor"* and an in-place op returns its receiver. That is a silent omission unless something else
  catches it, and §5 is that something.

A capture region wins: while `_capture_begin` is open, `eager_record` returns immediately, so the
ops belong to that trace and nesting never has to be decided.

## 4. W9: the graph is freed by the backward that walks it

`docs/BACKWARD5.md` §3 established the number and, more usefully, that it is what upstream already
pays for the same program -- 18.9 MiB at `S=8`, 75.7 at 32, 302.6 at 128, distinct storages, one
iteration. And §5 of the brief for this round is right that the "cheap" option is not a lesser one:

> free at `backward()`, refuse a second -- is upstream's `retain_graph=False` **default**, not a
> lesser refusal.

So that is the rule. `_eager_backward` **takes** the recorder rather than borrowing it, before it
can fail: a backward that raises still releases the graph, because putting it back on the error
paths would keep 302 MiB alive for the length of a traceback.

A second backward raises with upstream's phrase:

```
torch._C eager: Trying to backward through the graph a second time -- the saved intermediate
values have already been freed. This is upstream's retain_graph=False default ...
retain_graph=True is not implemented
```

**"Freed" and "never had a graph" are told apart by the tensor's own `grad_fn`, not by a table of
remembered addresses**, and that is a correction the round made to itself. The first version kept
the freed tape's addresses in a side set. The first program that exercised both branches got the
wrong message -- a freed address had been reused by an unrelated tensor, and a `sum()` over a tensor
that had never required a gradient was told to pass `retain_graph=True`. `from_op` cannot be wrong
that way: it is set by the door at the moment the op ran, it is exactly upstream's condition, and a
tensor carrying one whose graph is gone is precisely the `retain_graph` case. The test asserts both
messages, which is why the reuse showed up at all.

`_eager_tape_size()` exists so that the retention is assertable directly -- the tape grows during
the forward and is `0` after the backward -- rather than inferred from a gradient.

## 5. Aliasing: the verdict is unchanged, and now for a better reason

`docs/BACKWARD5.md` §6 deferred W10b on the grounds that it *"buys nothing at all until an eager
recorder exists, because `is_mutating`'s refusal makes A5/A6 unreachable"*, and the brief for this
round said: you are the round that makes it exist, so if your work changes that verdict, say so and
stop.

**It does not change it, and the round is in a position §6 was not: A5/A6 are now reachable, and
they are refused.**

The measurement, `docs/BACKWARD5.md` §1.1's own programs, run against the recorder built here and
against torch 2.13.0 in the same session:

| | program | shim: forward | shim: gradient | upstream: forward | upstream: gradient |
|---|---|---|---|---|---|
| **A5** | `a = x*1; v = a.view(3); a.mul_(10); (v*v).sum()` | `179.0` ✅ | **refuses by name** | `179.0` | `[6.0, 14.0, 22.0]` |
| **A6** | `a = x*1; v = a.view(3); v.mul_(10); (a*a).sum()` | `179.0` ✅ | **refuses by name** | `179.0` | `[6.0, 14.0, 22.0]` |
| stale leaf | `loss = (x*w).sum(); w.add_(1)` | — | **refuses by name** | — | `RuntimeError`, *"is at version 1; expected version 0"* |

The refusal:

```
torch._C eager: cannot differentiate this graph -- an in-place operation wrote into a tensor the
eager graph holds. The graph records the *mathematics* of each op and reads the values back at
backward() time, so a write that landed after the op ran -- including one made through a view of
the same storage -- would make it differentiate a program that never ran. Upstream refuses the
same shape with its version counter; this refuses it by storage.
```

**This is not W10b and does not become it.** W10b is alias *sets* -- knowing which values in a graph
are views of which others, so that the gradient can be computed anyway. Nothing here models that.
The insight is smaller and it is the reason the verdict survives:

> **A tape that refuses does not have to know what aliases what. It has to know that a write landed
> on bytes it depends on** -- and `docs/BACKWARD6.md` §4 already made that one hash lookup, by
> keying versions on the candle `Storage` rather than on the Python object.

So the recorder keeps a set of the storage addresses of every value it holds, and `note_mutation` --
which the door already calls on every dispatch, and which already returns immediately for anything
that is not in-place -- poisons the tape when a write lands on one. Cost on the ordinary path: none.
Cost on an in-place op: one hash lookup, and only if a tape exists.

Stated precisely, because "handled" can be read too strongly:

* This is **strictly less than upstream**, deliberately. Upstream *differentiates* A5 and A6;
  this refuses them. That is the trade `docs/AUTOGRAD.md` §6 chose, and the difference between it
  and W10b is the difference between *"never a wrong answer"* and *"the right answer"*.
* It sees a write only if the write goes through the door. `docs/BACKWARD6.md` §10 row 4's
  limitation is inherited unchanged.
* It is a whole-tape refusal, not a per-value one. A write to any value the graph holds refuses the
  whole graph, including the parts a gradient would never have reached. That is conservative in the
  safe direction and it is not free of false refusals -- §9 row 3.

`docs/BACKWARD5.md` §7 row 1 asked whether A5/A6's shape occurs in a real training loop. This round
did not answer that either, and the guard above makes the question less urgent rather than more:
if it does occur, the answer is now a refusal rather than a factor of 100.

## 6. What `run_backward` still refuses, and what is between here and it

`_ImperativeEngine.run_backward` -- where both `Tensor.backward()` and `torch.autograd.grad()` land
-- **still refuses, by name**, and `test_tensor_backward_still_refuses_even_though_an_eager_graph_exists`
now pins that it does so *while a graph exists*, with the instruction to invert rather than delete.

The surface that landed is `torch._C._eager_backward(output, grad_output=None, wrt=None)`, which
returns the gradients beside the tensors they belong to. That is deliberately not `.backward()`. The
brief's rule was that the engine stays refusing unless it has been checked end to end, and these are
the things this round did not build and therefore did not check:

| between here and `run_backward` | why it is not "wire it up" |
|---|---|
| `.grad` accumulation onto leaves | `+=` into a leaf's `.grad`, which is an in-place write on a tensor -- and §5's guard is watching exactly those. The interaction was not designed |
| `allow_unused` / `materialize_grads` | `_eager_backward` returns `None` for a leaf no gradient reached; upstream *raises* unless `allow_unused=True`. Two different defaults |
| `retain_grad` on non-leaves | the tape has the value; nothing surfaces it |
| hooks (`register_hook`, `post_accumulate_grad_hook`) | not fired, not enumerated |
| `create_graph` / double backward | the backward runs under `NoGradGuard`, so its own ops are not recorded. Inherited unchanged from `docs/BACKWARD4.md` §7 row 4 |
| `torch.autograd.Function` | unchanged, and not looked at |
| the `inputs=` argument of `.backward()` | `wrt` is the same idea with a different spelling and different error semantics |

`docs/BACKWARD2.md` §1.3's `None`-seed trap is gone -- `docs/BACKWARD4.md` made `_make_grads`
produce upstream's seed -- but a removed trap is not a safe engine, and the list above is why.

**If W8 and W9 land and the engine does not, that is a good round.** This is that.

## 7. Every gradient this round produced, checked element-wise against upstream

`docs/CAPTURE.md` §9-1 records this project's instance of a plausible-but-wrong gradient --
`adapt.step()` differentiating at a different dropout draw than it reports -- and
`docs/BACKWARD5.md` §1.1 records a measurement that nearly missed a factor of 100 by choosing a
program whose gradient *could* agree by coincidence. So every program below is non-linear in every
argument, and the comparison is element-wise against real torch 2.13.0, not a digest and not finite
differences. (The suite's finite-difference oracle still covers all 60 rules; it is a second opinion
here, not the proof.)

**Program 1**, the one `docs/BACKWARD6.md` §8 used for the same reason:
`loss = ((x @ wᵀ + b).tanh() ** 2).sum()`, `x` `[2,4]`, `w` `[3,4]`, `b` `[3]`, all `requires_grad`.

| quantity | elements | max abs difference from torch 2.13.0 |
|---|---:|---|
| `loss` | 1 | `0.000e+00` |
| `d(loss)/dx` | 8 | `0.000e+00` |
| `d(loss)/dw` | 12 | `7.451e-09` |
| `d(loss)/db` | 3 | `0.000e+00` |

**Program 2**, a value used twice, a reduction and a division -- so that an unaccumulated second
contribution shows up: `h = (a*c).tanh(); loss = (h*h).sum() / (h.exp().sum() + 1)`.

| quantity | elements | max abs difference |
|---|---:|---|
| `loss` | 1 | `0.000e+00` |
| `d(loss)/da` | 6 | `3.725e-09` |
| `d(loss)/dc` | 3 | `1.118e-08` |

**Program 3**, what a training loop actually issues:
`nn.Sequential(Linear(4,6), Tanh(), LayerNorm(6), Linear(6,2))`, `loss = (m(x)**2).sum()`, gradients
for all six parameter tensors.

| quantity | elements | max abs difference | `max|upstream|` | relative |
|---|---:|---|---:|---|
| `loss` | 1 | `9.537e-07` | 8.2165 | `1.16e-07` |
| gradients, all six parameters | 56 | `1.907e-06` | 10.4325 | `1.83e-07` |

**91 elements over three programs, every one at float32 rounding.** Program 3 is the one that
matters most for the reuse claim of §2: it reaches `addmm`, `native_layer_norm` and `tanh` rules
that were written for the capture tape and had never been called by anything else.

**Program 4**, four steps of gradient descent, because a single backward does not exercise the
lifetime. `Sequential(Linear(4,5), Tanh(), Linear(5,2))`, `loss = ((m(x)-y)**2).sum()`, gradients
taken, `p.add_(g, alpha=-0.05)` under `no_grad`, repeat:

| step | shim | upstream |
|---:|---|---|
| 0 | `1.834754467010498` | `1.8347543478012085` |
| 1 | `0.6300362348556519` | `0.6300361752510071` |
| 2 | `0.26539134979248047` | `0.2653912603855133` |
| 3 | `0.13494616746902466` | `0.13494616746902466` |

Max absolute difference `1.192e-07`, after four steps of compounding. Two things are being checked
at once: that the gradients are right, and that **§5's guard does not fire on the parameter update**
-- the tape is freed by the backward that precedes each `add_`, so the write lands on storages no
graph holds. A guard that refused here would have made the recorder useless for the one loop it
exists for, and this is the half of §5 that catches it firing when it should not.

The refusals of §5 are the fifth program, and they were run against upstream in the same session --
the table there is that measurement.

## 8. Cost, and the number is not usable

`docs/BACKWARD5.md` §7 row 2 asked for the per-dispatch cost of an always-on recorder. **`CLAUDE.md`
forbids reporting one from a loaded machine, and this machine was loaded: `uptime` reported load
averages of 7.2 to 8.1 on 8 cores with five other agents running.** For scale, `aten.add.Tensor`
measures 1364-1567 ns here and `docs/BACKWARD6.md` §6 measured the same op at 889 ns at load 2.0-3.2
on the same hardware. **The absolute numbers below are contaminated and should not be quoted.**

What survives contamination is a comparison taken seconds apart in **one process and one binary** --
`_eager_set_enabled(False)` is the control, so there is no second build and no second interpreter
between the two columns. 20 000 calls of `_aten_dispatch("aten.add.Tensor", a, b)` on `float32[3]`,
9 repeats, median and min of the per-repeat mean, two independent runs:

| | recorder on | recorder off | delta |
|---|---:|---:|---:|
| **not on a gradient path** (run 1) | 1363.8 (min 957.4) | 1451.3 (min 967.4) | −87.5 (min −10.0) |
| **not on a gradient path** (run 2) | 1566.8 (min 979.5) | 1328.2 (min 951.2) | +238.6 (min +28.3) |
| **on the gradient path** (run 1) | 2267.7 (min 1636.2) | 1717.0 (min 1077.5) | +550.7 (min +558.7) |
| **on the gradient path** (run 2) | 2364.4 (min 1583.2) | 1392.2 (min 1075.6) | +972.2 (min +507.6) |

**The first pair is the result.** Off the gradient path the two columns disagree by less than the
noise **and the sign flips between runs** -- which is what §3's design predicts, since the recorder
is not consulted at all until `mark_from_op` says it marked something. A cost that changes sign
between runs is a cost that was not measured.

The second pair is real and large, and it is honest to say what is in it rather than to call it the
recorder's per-op cost: this benchmark records **20 000 nodes into one tape**, holding 20 000 tensors
and growing three `Vec`s and a `HashMap`, without ever calling `backward()` to free them. That is a
worse case than a training step, which frees every iteration. The two effects -- recording one node,
and growing a tape to 20 000 -- were not separated, and §9 says so.

**A quiet-machine measurement, with the tape growth separated out, would be worth having and this
round did not have a quiet machine.**

## 9. Gates

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh
    394 ok, 0 FAIL          (389 before; +5 added, 0 inverted, 0 removed, 0 weakened)
    SELF-TEST: PASS -- 19 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised
    DOCWATCH: PASS -- 362/362 evaluated marker(s) hold   (350 before; +12, all in this document)
    EXIT=0

TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib $PY tools/golden/compare.py
    SUMMARY: 8509/8509 cases passed, 0 failed, ops covered=203, pending case builders=0
    EXIT=0
```

`ops=203` is unchanged **on purpose** -- no kernel landed, so nothing here could have moved it. Both
gates were run with `TORCH_C_ARTEFACT` set to this worktree's artefact, for the reason
`docs/BACKWARD6.md` §9 gives: `tools/golden/loader.py:33` otherwise falls back to a shared cache path
and reports on another agent's build.

### 9.1 The forward did not move

This round edits the dispatch hot path, so `docs/BACKWARD5.md` §6.5.1's "no compile input changed"
argument is not available and the measurement is the evidence.

Real `HuggingFaceTB/SmolLM2-135M`, `float32`, greedy prefill of `torch.arange(S).unsqueeze(0)`.
The recipe is `docs/BACKWARD6.md` §7's, **and it is in the tree**: sha256 over
`struct.pack("<%df" % len(row), *row)` for each row of `logits.tolist()[0]` in order.

| S | sha256, this build | |
|---:|---|:--:|
| 6 | `606e00d1be05fccce84b23b02dae8f473d490ad0f706b9b612036dd55cee531b` | ✅ |
| 32 | `62d923807cce47cee35a5e7b6091c9544e180c6c9ff771bb4cad3f478f4cbb79` | ✅ |
| 128 | `4750428e94f7383c42853b0d547192a7946fdde9e086d3b16cebdeb2ca68e37d` | ✅ |

All three equal `docs/BACKWARD6.md` §7's, `docs/BACKWARD5.md` §6.5.1's and `docs/BACKWARD4.md`
§6.2's. `docs/BACKWARD6.md` §7 built its "before" column by reverting its sources and rebuilding
**because the recipe was not written down at the time**; it is now, and the script above was written
from it rather than copied, so the agreement is a reproduction. A four-round chain of the same three
digests, one of which was taken against a deliberately reverted build, is a stronger control than
one more stash-and-rebuild would have been.

The reason the forward *can* be unmoved despite the door changing: `mark_from_op` returns a `bool`
it already computed, and the new branch is `on_graph && eager_enabled()`. A SmolLM2 prefill under
`torch.no_grad()` never reaches the first operand of that `&&`.

### 9.2 What was checked by being switched off

`CLAUDE.md` §5.5: a verification that cannot fail is not a verification. Each of the four W8/W9
tests was re-run with `_eager_set_enabled(False)`, which nullifies the recorder without touching
the tests, and **all four fail**:

```
breaks: test_the_eager_recorder_records_exactly_the_ops_that_get_a_grad_fn      AssertionError: 0
breaks: test_the_eager_tape_and_the_capture_tape_are_the_same_derivative_rules  RuntimeError
breaks: test_the_eager_graph_is_freed_by_the_backward_that_walks_it             AssertionError: 0
breaks: test_the_eager_graph_refuses_a_write_through_a_view_of_a_value_it_holds AssertionError:
                                                                               the write was not noticed
```

That check found a real defect rather than only confirming the tests: with the recorder off, the
second test raised the *`retain_graph` message* for a graph that had never existed, because
`from_op` is set by `mark_from_op` (W5) and not by the recorder (W8). The message now consults both.
Nobody would have found that from the passing runs.

## 10. What this document does not establish

| # | not established | why |
|---|---|---|
| 1 | **A usable per-dispatch cost.** | §8. Load 7.2-8.1 on 8 cores. The *off-path* comparison survives (the sign flips between runs, so the cost is under the noise floor); the on-path number does not, and it conflates recording one node with growing a tape to 20 000 |
| 2 | **That the tape cannot grow without bound.** | It is freed by `backward()` and by nothing else. A program that runs forwards under grad mode and never differentiates retains every intermediate -- `docs/BACKWARD5.md` §3's 302.6 MiB per un-freed iteration. Upstream is bounded here by refcounting the graph from the tensors; this is not. `_eager_reset()` is the manual escape and no automatic bound was designed |
| 3 | **That §5's guard has no false refusals.** | It refuses the **whole** tape when a write lands on any storage it holds, including a value no gradient would have reached. §7's program 4 shows the ordinary training loop is unaffected -- the tape is freed before the update -- but a model that writes to a buffer *mid-forward* (a KV cache, a batch-norm running statistic) would be refused where upstream answers, and that was **not measured against a real model** |
| 4 | **Anything about `run_backward`, `.grad`, hooks, `create_graph` or `torch.autograd.Function`.** | §6 is the list. None of it was built and none of it was checked |
| 5 | **That randomness on the eager path is actually safe.** | §3's gating means a dropout mask is *held*, not redrawn, so `docs/CAPTURE.md` §9-1's failure structurally cannot occur here -- and that argument was **not tested**, because the two RNG streams make a direct upstream comparison of a dropout gradient impossible without seeding work this round did not do |
| 6 | **W10b.** | Not built, and §5 is the argument that this round did not make it necessary. It did make it *reachable*, which is more than `docs/BACKWARD5.md` §6 could say |
| 7 | **That an eager tape and a capture region interact correctly beyond "the region wins".** | `eager_record` returns immediately while a region is open, so ops inside a captured region are absent from the eager graph. Nothing tests what a `.backward()` across that boundary should do, because nothing can call one yet |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs eager_record present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs node_objects present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs eager_free present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs poison_on_write_to_recorded_storage present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs backward_in present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_eager_recorder_records_exactly_the_ops_that_get_a_grad_fn present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_eager_tape_and_the_capture_tape_are_the_same_derivative_rules present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_eager_graph_is_freed_by_the_backward_that_walks_it present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_eager_graph_refuses_a_write_through_a_view_of_a_value_it_holds present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_tensor_backward_still_refuses_even_though_an_eager_graph_exists present -->
<!-- The engine is still refused. Invert the test above rather than deleting it when it lands. -->
<!-- DOCWATCH: count smoke_ok ge 390 -->
<!-- DOCWATCH: count golden_cases_passed ge 8509 -->
