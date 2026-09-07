# W4, W6, W7: the semantic three, measured — and the refusal that follows

`docs/training/BACKWARD2.md` §2 split ten walls into cheap, semantic and structural, landed the cheap three,
and stopped at the semantic row with a claim rather than a measurement:

> **semantic** | W4, W6, W7 | one line each, **but they are one decision** | nothing on its own |
> **a lie.** `requires_grad` propagating while `grad_fn` is `None` and `is_leaf` is `True` describes
> a tensor upstream reserves for an *accumulating leaf*; `torch.optim` and `AccumulateGrad` both key
> on that triple.

This round tests that claim instead of inheriting it. The claim is **right, and for a reason
BACKWARD2 did not state**: the guard `torch.optim` actually keys on is `is_leaf`, not
`requires_grad` — and this shim **already fails it today**, before W4. What W4 would add is not a
new divergence; it is a *plausible* one.

Environment: worktree `work/bwsem` on `develop` `fcb6926`, torch 2.13.0, transformers 5.15.1,
`/Volumes/macMini/caches/spike-venv/bin/python`. Every shim reading below printed `shim`; every
upstream reading was taken with `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`.

### Answers, before the evidence

| question | answer |
|---|---|
| Does W4/W6/W7 land? | **No.** §4 |
| Does W5 have to land with it? | **W6 *is* W5.** `is_leaf` is defined upstream as `grad_fn is None`; with W5 absent there is no version of W6 to land that is not a second lie. §3 |
| What does `torch.optim` do with the triple? | Nothing — the triple is a **legal, ordinary parameter before its first backward**, and `SGD.step()` skips it. The guard that would catch an *intermediate* wearing it is `is_leaf or retains_grad`, at `torch/optim/optimizer.py:1153`, and this shim already walks past it. §1 |
| What does `transformers` do with it? | **Reads `requires_grad` zero times in a `.train()` forward.** 545 writes and 2 reads, all during `from_pretrained`, all on leaves. §2 |
| Does this repo's suite depend on non-propagation? | Yes — one assertion, `test_shim.py:406`, and it is a deliberate pin with a message that names the consequence. §2.3 |
| So what is W4's whole observable effect? | It changes the shim's answer to *"will a gradient flow through this tensor?"* from **False**, which is true of this shim, to **True**, which is not. §4 |
| What landed? | The measurement, one test that makes the trap in BACKWARD2 §1.3 loud, and the documentation of a divergence that was never written down. **No behaviour changed.** §5 |
| Did the structural group get cheaper or harder? | **Cheaper by one item and better-ordered by one.** §7 |
| Did W5 then land? | **Yes, in `docs/training/BACKWARD4.md`, and smaller than §7 budgeted.** A correct `grad_fn` *nullness* needs no tape — one `Option<Box<str>>` per tensor at the door — so W5, W6, `retains_grad` and both divergences in §1.1 landed together and W8/W9/W10 were not touched. The one correction it makes to this document is §1: **W4 is not separable from W5 either**, because upstream's `grad_fn is not None` holds exactly when the flag would have propagated |
| The stale `1723`? | **It was never stale.** `1723` and `1853` are two different questions asked of the same trace, and both are current. §6 |

---

## 1. What `torch.optim` does with the triple, measured

The triple is `requires_grad=True`, `grad_fn=None`, `is_leaf=True`, `grad=None`. Upstream, on 2.13.0:

```
p = torch.ones(3, requires_grad=True)
triple: True None True None
--- SGD.step() with grad=None ---   returned; p = [1.0, 1.0, 1.0]
--- SGD.step() with grad set ---    p = [0.9, 0.9, 0.9]
```

**Nothing happens, and that is the point.** The triple is not exotic and not an error: it is exactly
what every parameter in every model looks like between `from_pretrained` and the first
`loss.backward()`. `SGD.step()` reads `p.grad`, finds `None`, and skips. So a test that asked *"does
`optim` reject the triple"* would come back green and would have measured nothing.

The interesting question is the one BACKWARD2's sentence implies but does not ask: **what does
`optim` do with a tensor that wears the triple and is not a leaf?** Upstream:

```
y = x * 2
y triple: True MulBackward0 False None
optim.SGD([y], lr=0.1)
  ValueError: can't optimize a non-leaf Tensor
```

and the guard is one line, `torch/optim/optimizer.py:1153`, inside `add_param_group`:

```python
if not self.defaults.get("differentiable", None) and not (
    param.is_leaf or param.retains_grad
):
    raise ValueError("can't optimize a non-leaf Tensor")
```

### 1.1 The guard keys on `is_leaf`, and this shim already fails it

`requires_grad` is not in that condition. Which means the divergence BACKWARD2 attributes to landing
W4 **is already here**, and W4 is not what puts it here. Measured side by side, same script:

| | shim (today, `fcb6926`) | upstream 2.13.0 |
|---|---|---|
| `y = x*2`, `y.requires_grad` | False | True |
| `y.is_leaf` | **True** | False |
| `y.grad_fn` | `None` | `MulBackward0` |
| **`optim.SGD([y])`** | **accepted** | `ValueError: can't optimize a non-leaf Tensor` |
| `y.requires_grad_(False)` | accepted | `RuntimeError: you can only change requires_grad flags of leaf variables` |
| `setattr(y, "requires_grad", True)` | accepted | the same `RuntimeError` |
| `copy.deepcopy(y)` | refuses on `UntypedStorage.copy_` | `RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol` |
| `y.retains_grad` | **`NotImplementedError: TensorBase.retains_grad`** | False |
| `y.numpy()` | `NotImplementedError: TensorBase.numpy` | `RuntimeError: Can't call numpy() on Tensor that requires grad` |
| `torch.no_grad(): (x*2).requires_grad` | False | False *(for a different reason)* |

Every one of the first three rows is a guard upstream uses to stop a caller treating a graph
intermediate as a parameter, and every one of them keys on `is_leaf` or `grad_fn`. **W4 moves none
of them.**

### 1.2 `retains_grad` is unreachable *because* `is_leaf` lies

Note the `retains_grad` row. Upstream's guard is `param.is_leaf or param.retains_grad`, and Python
short-circuits: with `is_leaf` always `True` here, `retains_grad` is never evaluated. It is
`NotImplementedError` and nothing has ever hit it.

That is a fact about W6 rather than about W4, and it is the cost of the only version of W6 available
without W5. Make `is_leaf` report `False` for an intermediate and upstream's own optimiser stops
raising `ValueError: can't optimize a non-leaf Tensor` and starts raising a shim refusal about
`retains_grad` — a name that has nothing to do with what the caller did wrong. §3 is what follows.

### 1.3 What W4 *does* move: the seed, and it moves it toward upstream

There is exactly one place where the flag is read on a path that matters, and BACKWARD2 §1.3 already
named it. `torch/autograd/__init__.py` `_make_grads` builds a seed only `if out.requires_grad`:

| | shim | upstream |
|---|---|---|
| `_make_grads((x.sum(),))` — scalar | `(None,)` | `(tensor(1.),)` |
| `_make_grads((x*2,))` — non-scalar | `(None,)` | `RuntimeError: grad can be implicitly created only for scalar outputs` |

W4 would move **both** rows onto upstream's answer. That is the honest half of the case for landing
it, and it is worth stating plainly because it cuts against this round's conclusion. What it is
worth is bounded by the fact that the only consumer of a seed is an engine, and the engine refuses:
`_ImperativeEngine.run_backward` is reached in both rows and raises before the seed is looked at. So
the improvement is real and, today, **unobservable** — while the cost in §4 is observable at every
tensor.

§5.1 lands a test on this row anyway, because the danger is not the `None`; it is what a future
engine will be tempted to do with it.

---

## 2. What `transformers` does with the triple, measured

`requires_grad` is a Python-visible property on `TensorBase`, so it can be counted. The getter and
setter were wrapped with a counter that records the calling frame, and a real
`HuggingFaceTB/SmolLM2-135M` was loaded, put in `.train()`, and given a forward with `labels=`.

### 2.1 `from_pretrained` — 545 writes, 2 reads, all on leaves

```
=== from_pretrained: requires_grad WRITES ===
   272  =True   _make_subclass <- parameter.py:57:__new__ <- core_model_loading.py:1339:set_param_for_module
   211  =True   _make_subclass <- parameter.py:57:__new__ <- linear.py:108:__init__
    61  =True   _make_subclass <- parameter.py:57:__new__ <- modeling_llama.py:59:__init__
     1  =True   _make_subclass <- parameter.py:57:__new__ <- sparse.py:165:__init__
     1  =False  requires_grad_ <- parameter.py:270:__new__ <- modeling_llama.py:88:__init__
     1  =False  requires_grad_ <- parameter.py:270:__new__ <- modeling_llama.py:89:__init__
=== from_pretrained: requires_grad READS ===
     1  parameter.py:270:__new__ <- modeling_llama.py:88:__init__
     1  parameter.py:270:__new__ <- modeling_llama.py:89:__init__
```

Every write is `nn.Parameter.__new__` or `nn.Buffer.__new__` **setting the flag explicitly** on a
tensor it has just created. Both reads are `nn.Buffer.__new__` reading back what it was passed, on
the rotary embedding's `inv_freq`. Propagation cannot reach any of these: the source tensors are
storage loads with the flag `False`, and the destination value is a literal.

### 2.2 `.train()` and the forward — zero reads, zero writes

```
=== .train():           reads 0  writes 0
=== forward(labels=):   reads 0  writes 0
```

**A `.train()` forward of a real 135M-parameter model reads `requires_grad` not once.** So the
brief's worry — that landing propagation "could move behaviour under them without touching a
kernel" — is measured and answered: it could not. Nothing in `transformers` on this path looks.

That cuts both ways, and the second way is the one that decides the round: since nothing reads it,
**propagating it buys nothing here either.**

For completeness, the two ends of that forward:

| | shim | upstream |
|---|---|---|
| `loss` | 3.1185123920440674 | 3.1186187267303467 |
| `loss.requires_grad` / `is_leaf` / `grad_fn` | False / True / `None` | True / False / `NllLossBackward0` |
| `logits.requires_grad` / `is_leaf` | False / True | True / False |
| a parameter's triple | True, `None`, True, `None` | True, `None`, True, `None` |

The last row is the one to read twice. **The parameters already agree with upstream exactly.** The
flag is carried, `grad_fn` is `None` and `is_leaf` is `True` because it is a leaf — and that is what
`docs/training/BACKWARD2.md` §4.1 landed. The rows above it are the intermediates, and they are where the two
descriptions part.

### 2.2.1 Gradient checkpointing does not reach the question

`torch/utils/checkpoint.py:90` is the one place in torch that would *notice* propagation on an
activation —

```python
if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
    warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")
```

— and it is not reachable here, for two independent reasons. It is on the **reentrant** path and
`transformers` defaults to `use_reentrant=False`; and under the shim
`model.gradient_checkpointing_enable()` succeeds but the forward then stops at an unrelated wall:

```
NotImplementedError: not implemented in torch._C shim: torch.diff(...) -- overload resolution
has no table entry for this op
```

Recorded because it was measured, and because it is a *different* work item from this one — the
sentence that warning would print ("Gradients will be None") is true of this shim today and would
become false under W4, so if that path ever opens it belongs in this argument.

### 2.3 What this repo's suite depends on

One assertion, and it is deliberate. `rust/torch_c/pytests/test_shim.py:405`, inside
`test_the_autograd_boundary_is_where_autograd_md_says_it_is`:

```python
y = _C._aten_dispatch("aten.mul.Tensor", x, x)
assert y.requires_grad is False, (
    "an op propagated requires_grad -- graph construction has appeared; "
    "see docs/training/AUTOGRAD.md §1.3 and §6 before extending this"
)
```

Its docstring already says what to do if the boundary moves — *"invert this rather than deleting
it"* — so it is not an obstacle to W4; it is the record of the decision W4 would reverse. It is
listed here because the brief asked, and because the answer being "one line, deliberate, with an
inversion note" is itself information: nothing else in 16,500 lines of tests leans on the flag
staying put.

---

## 3. W6 is W5, and W7 is empty without W4

Three one-line walls, and only one of them is actually one line.

**W6.** `is_leaf` is `property(lambda self: True)` at `bootstrap.py`. Upstream's definition is not a
separate fact — it *is* `grad_fn is None`, and `grad_fn` is `property(lambda self: None)` (W5). So
with W5 absent, "landing W6" means one of:

* `is_leaf = (grad_fn is None)` — a **tautology**. It computes `True` for everything, byte for byte
  the behaviour that is there now. Nothing lands.
* `is_leaf = False` for anything an op produced — which is computable without a graph, since it is
  the same condition W4 tests. But it breaks upstream's own invariant `is_leaf == (grad_fn is
  None)`, and §1.2 measured what it costs concretely: upstream's optimiser stops giving
  `ValueError: can't optimize a non-leaf Tensor` and starts giving
  `NotImplementedError: TensorBase.retains_grad`. A refusal that names the wrong thing is the
  failure `docs/training/BACKWARD2.md` §1.2 spent a section on, arriving from the other direction.

So the honest statement is not "W6 needs W5 alongside it". It is **W6 has no content that is not
W5.** BACKWARD2's dependency chain `W10 → W9 → W8 → W5` should be read as ending at W5-and-W6
together.

**W7.** `no_grad()` gates propagation. With no propagation there is nothing to gate, and
`torch.is_grad_enabled()` already round-trips exactly (measured, both directions, §1.1's last row).
W7 is not a wall that can be landed early; it is a *consequence* of W4 that must land in the same
commit, which is what BACKWARD2 §2 means by "one decision" and is the only part of the semantic row
that is exactly as stated.

---

## 4. The decision, and the one sentence it rests on

**W4, W6 and W7 do not land.**

The argument is not "it would break something" — §2 measured that it would break nothing on any
path this shim actually runs. It is narrower than that, and it survives the fact that §1.3 found W4
moving two rows *toward* upstream:

> Today an intermediate under this shim reports `(requires_grad=False, grad_fn=None, is_leaf=True)`.
> Upstream reserves that description for a **constant**. And under this shim, a graph intermediate
> *is* a constant: no gradient will flow through it, nothing will accumulate into it, and
> `.backward()` refuses. **The shim's current answer is not an approximation of upstream's — it is a
> true statement about this shim.**
>
> After W4 it would report `(True, None, True)`. Upstream reserves *that* for an accumulating leaf —
> a trainable parameter. That is a true statement about upstream and a false one about this shim,
> and it is false in the one direction that matters: it answers *"will a gradient flow through
> here?"* with **yes**.

`docs/design/DESIGN.md` §6's criterion is whether landing a wall alone leaves the shim honest, and
`docs/training/BACKWARD.md` §5.2's is that *"a wrong gradient looks exactly as plausible as a right one and
the program keeps running"*. A flag that promises a gradient is one step upstream of a wrong
gradient, and it is the step where the promise is still cheap to withhold.

The measured shape of the trade, so the next round can reverse it on evidence rather than taste:

| | today | after W4 |
|---|---|---|
| what an intermediate claims | a constant — **true here** | a trainable leaf — **false here** |
| `optim.SGD([intermediate])` | accepted (upstream refuses) | accepted (upstream refuses) — **unchanged** |
| `_make_grads` scalar seed | `(None,)` (upstream `tensor(1.)`) | `tensor(1.)` — **fixed, and unreachable** |
| `_make_grads` non-scalar | `(None,)` (upstream raises) | raises — **fixed, and unreachable** |
| reads in a real `.train()` forward | 0 | 0 |
| cost at the door | none | a `.requires_grad` read per tensor operand per op, in Python, on the hot path |

The last row is not the reason for the decision, but it is not nothing: a captured SmolLM2-135M
forward at `S=8` is **1862 recorded nodes**, and every one of them would take at least one extra
attribute read plus a write on the output. `docs/training/BACKWARD2.md` §7.3 already declined to measure the
cost of a *tuple unpack* on this path; W4 is strictly more than that, and it would have to be paid
by every caller including the ones that never intend to differentiate anything.

### 4.1 What this does not say

It does not say W4 is wrong. It says W4 is **not separable**. Landed together with W5 — a `grad_fn`
that is a real node — every row of §1.1's table moves onto upstream's answer at once, `is_leaf`
becomes computable rather than asserted, `retains_grad` becomes reachable and answerable, and the
seed in §1.3 gets a consumer that can use it. That is the commit W4 belongs in, and
`docs/training/BACKWARD2.md` §2.1 has already sized it and put three structural walls in front of it.

---

## 5. What landed

**No behaviour changed.** One test, one docstring, and this document. Split the way
`CLAUDE.md` §5.3 asks for:

| | |
|---|---|
| feature added | — |
| defect fixed | — |
| test added | 1 (`test_the_backward_seed_is_absent_and_nothing_guesses_a_one`) |
| documentation corrected | `_install_autograd_shape`'s docstring (it did not mention `is_leaf` at all); `docs/training/BACKWARD2.md` §7.2 and §8 row 6 (the `1723`, §6 below) |
| deleted | — |

### 5.1 The test, and why it is this one

`docs/training/BACKWARD2.md` §1.3 states the trap in a block quote and pins nothing:

> An engine implemented at W3 alone would receive `roots=(y,)`, `grads=(None,)` and
> `accumulate_grad=True`, and the obvious defensive thing to do with a `None` seed is to substitute
> a one. For the scalar case in the brief that is *the right answer by coincidence*. For any
> non-scalar output it is silently wrong, and upstream would have raised.

A warning in prose is exactly the shape `docs/verification/DOCWATCH.md` exists because of. So the new test asserts
the trap's three parts as facts that a future engine has to confront:

1. `_make_grads` on a **scalar** output yields `(None,)` here, and `tensor(1.)` upstream.
2. `_make_grads` on a **non-scalar** output yields `(None,)` here, and upstream raises
   `grad can be implicitly created only for scalar outputs`.
3. `_ImperativeEngine.run_backward` refuses, so **nothing consumes either `None`.**

It runs through the vendored tree in a subprocess, the way the checkpoint, device and meta tests
already do, because `_make_grads` and `torch.optim` are upstream Python and only exist there.

It also pins §1.1's `optim` row — that `optim.SGD([intermediate])` is *accepted* here where upstream
raises — because that divergence was found by this round and was written down nowhere. It is
asserted as a divergence with the failure message naming the inversion, not as a desirable
behaviour.

**The test is written to fail when W5 lands**, and its message says which document to read and that
inverting it is the right response. That is the same convention
`test_the_autograd_boundary_is_where_autograd_md_says_it_is` uses and the reason that test survived
`docs/training/BACKWARD.md` landing a backward.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_backward_seed_is_absent_and_nothing_guesses_a_one present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_autograd_shape present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs reachable present -->

### 5.2 The docstring

`_install_autograd_shape` enumerates what it papers over — `requires_grad`, `grad_fn`, `grad`,
`data` — and **does not mention `is_leaf`**, which it also installs, one line below `grad_fn`. That
omission is why BACKWARD2 could write "`is_leaf` is `True`" as a semantic one-liner: nothing in the
code said what it claims or who reads it. It now says both, with §1.1's measurement and the
`optimizer.py:1153` call site.

---

## 6. §7.2's `1723`, attributed — and it was never stale

`docs/training/BACKWARD2.md` §7.2 reported `on a gradient path 1853` where `docs/training/BACKWARD.md` §4 says `1723`,
confirmed the 1853 on an unmodified tree, concluded the 1723 was stale, and left "which commit moved
it" open as §8 row 6.

**No commit moved it. They are two different questions, and both answers are current.** The same
capture, on this tree, asked both ways:

```
trace: <CaptureTrace 1862 nodes, 1 inputs, 333 constants, 1 outputs>
loss  12.871352195739746
parameters that are trace constants: 272 of 272

differentiable()                      nodes 1862, on a gradient path 1853, distinct ops 26, missing {}
differentiable(wrt_constants=params)  nodes 1862, on a gradient path 1723, distinct ops 20, missing {}

constants 333, floating constants 333, parameter constants 272
floating non-parameter constants: 61
parameters with a gradient: 272 of 272
_C._tape_rules(): 60 rules
```

`1723` **and** `distinct ops 20` are `docs/training/BACKWARD.md` §4's numbers exactly, both of them, which is
what makes this an identification rather than a coincidence. §4's script seeded the walk from the
**272 parameters**; BACKWARD2 §7.2's script called `differentiable()` with no argument, and
`wrt_set` then defaults to *every floating constant* — all 333, including the 61 non-parameter ones
the trace burns in. Six more ops become reachable and 130 more nodes with them.

The mechanism is `tape.rs`'s own, and its doc comment already says so:

> *"Without this the walk would try to differentiate the rotary embedding's `arange`, because a tape
> has no `requires_grad` of its own and every value looks alike from inside. Upstream gets the same
> pruning from the flag; here it comes from the declaration of what the caller wants gradients
> *for*, which is the same information one step earlier."*

Confirmed structurally as well as numerically: `wrt_set`, `reachable`, `wanted` and
`differentiable` are **unchanged** since `443220f`, the commit that introduced them and that
`docs/training/BACKWARD.md` §4 documents. The only tape.rs edits since are six rule arms and their table
entries, and `nodes_on_a_gradient_path` is computed from `reachable()` before `has_rule` is
consulted, so a rule cannot move it.

So `docs/training/BACKWARD.md` §4 needs no correction. What needed correcting is `docs/training/BACKWARD2.md` §7.2's
reading of it and §8 row 6, and both are corrected in place with a pointer here.

**The general lesson is the one `reachable()`'s comment is about.** `wrt_constants` is not a filter
on the answer; it is *the question*. A count of "nodes on a gradient path" is meaningless without
saying a gradient path **to what** — and §7.2 compared two numbers that had different answers to
that. This document's §1 is the same hazard in the other subject matter: "does `optim` reject the
triple" has no answer until you say whether the tensor wearing it is a leaf.

---

## 7. Whether the structural group got cheaper or harder

The brief asked. **Cheaper by one item, better-ordered by one, and harder in nothing.**

| | effect | why |
|---|---|---|
| **W5** (`grad_fn`) | **cheaper** | §3 folds W6 into it. W6 was budgeted as a separate semantic line item in BACKWARD2 §2 and has no content of its own, so the structural work is four walls and not four-plus-one. And §1.2 found a *second* thing W5 must bring: `retains_grad`, which is `NotImplementedError` today and is unreachable only because `is_leaf` short-circuits it. That is one line, and finding it now is cheaper than finding it from an optimiser traceback |
| **W8** (eager recorder) | **better ordered** | §4's last row measures what W8 costs at the door in the units it will be paid in: 1862 nodes for one `S=8` forward, every one taking a flag read. BACKWARD2 §1.5 called the recorder *"one line at one door"*, which is true of the *code* and now has a number attached to the *cost* |
| **W9** (lifetime) | unchanged | nothing here touches it |
| **W10** (mutation) | unchanged, and §1's measurement **confirms** BACKWARD2's framing rather than softening it | `optimizer.step()` is where the in-place ops are, and §1 spent the round inside `torch/optim/optimizer.py`. Nothing found there suggests a cheap answer, and BACKWARD2 §8 row 4's uncosted third option (copy the operand instead of versioning it) is still uncosted |

Nothing got harder. The one thing that would have — a propagated flag that a recorder then has to
keep consistent with `no_grad`, `detach`, `.data` and in-place ops — is precisely what did not land.

---

## 7.5 Sabotage: five faults on what landed

`CLAUDE.md`'s rule — a check that cannot fail is not a check — and it matters more than usual here,
because the *only* thing this round landed is a test. Each fault is applied to the tree, **rebuilt**,
and `test_the_backward_seed_is_absent_and_nothing_guesses_a_one` re-run alone; the tree is restored
from a `cp` backup after every one.

| # | fault | caught by |
|---|---|---|
| F1 | `_ImperativeEngine.run_backward` returns `None` instead of refusing — **the W3-only engine the test exists for** | ✅ `('backward', {'ok': 'returned'})` |
| F2 | `requires_grad` reports `True` for everything (W4, crudely) | ✅ the intermediate row, `[True, True, True]` |
| F3 | `is_leaf` reports `False` (W6 in its only W5-free form) | ✅ the **leaf** control, `[True, True, False, True]` |
| F4 | `retains_grad` is implemented as `False` without `is_leaf` moving | ✅ `{'ok': False}` where a refusal was asserted |
| F5 | the oracle arm is pointed at the vendored tree — the copy-paste mistake | ✅ `up["who"] == "upstream"` → `AssertionError: shim` |

**Five of five**, and each names which claim broke. F1 is the one the test was written for: it is
the fault that makes the two `[None]` seeds dangerous, and nothing else in the suite sees it.

F5 is worth a sentence because a *weaker* version of it was tried first and came back **NOT
CAUGHT** — removing the `env.pop` while the parent's `PYTHONPATH` named only the staged `_C.abi3.so`
changes nothing, because that directory does not shadow `torch`. The fault only bites when the arm
is pointed at a tree that *does*. That is the same shape as the round's own subject: a check
that cannot fail under the conditions you tried it is not yet known to be a check, and the second
try is what decides.

---

## 7.6 Gates

Both pass on the final artefact, and the two controls with them.

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh
    344 ok, 0 FAIL          (343 before; +1 test, none inverted, none removed)
    SELF-TEST: PASS -- 20 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised
    DOCWATCH: PASS -- 260/260 evaluated marker(s) hold        (257 before; +3, all here)
    EXIT=0

$PY tools/golden/compare.py
    SUMMARY: 7763/7763 cases passed, 0 failed, ops covered=168, pending case builders=1
    EXIT=0
```

`ops=168` is unchanged **on purpose** — no kernel landed, and this round landed no behaviour at all.

### 7.6.1 The forward did not move

`docs/numerics/SEQLEN.md` §1.3's prefill logits sha256 over real SmolLM2-135M, re-measured on the final
artefact. This round edits `bootstrap.py`, and although the edit is a docstring and a comment,
`bootstrap.py` is `include_str!`'d at compile time and the file is re-executed at import — so it is
checked rather than argued.

| S | f32 | |
|---:|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | ✅ |
| 32 | `331668f36da02f21…` | ✅ |
| 128 | `00159a9dbd308eda…` | ✅ |

All three equal `docs/training/BACKWARD2.md` §7.1, `docs/training/BACKWARD.md` §9.1 and `docs/training/LOSS.md` §10.1.

### 7.6.2 The tape did not move

```
<CaptureTrace 1862 nodes, 1 inputs, 333 constants, 1 outputs>   loss 12.871352195739746
differentiable()                      nodes 1862, on a gradient path 1853, distinct ops 26, missing {}
differentiable(wrt_constants=params)  nodes 1862, on a gradient path 1723, distinct ops 20, missing {}
parameters with a gradient: 272 of 272
_C._tape_rules(): 60 rules
```

`1862`, `333`, `272 of 272`, `60` and the loss to every digit are `docs/training/BACKWARD.md` §4's and
`docs/training/BACKWARD2.md` §7.2's. Both walk counts are printed because §6 is about the difference between
them, and printing one without the other is how §7.2 came to compare two answers to different
questions.

---

## 8. What this document does not establish

| # | not established | why |
|---|---|---|
| 1 | **That no consumer of `requires_grad` on an intermediate exists anywhere.** | §2 counted reads on one path: `from_pretrained` + `.train()` + a forward with `labels=`, on one architecture. `generate`, PEFT, `Trainer`, and every `torch.nn` module SmolLM2 does not use were not exercised. The counter is a Python property wrapper, so a read from *inside* the shim's Rust would also not be counted — there are none today, and that is asserted from the source rather than measured |
| 2 | **That W4 + W5 together would be honest.** | §4.1 asserts it and this round did not build it. The claim is that every row of §1.1 moves onto upstream's answer *at once*; four of those rows depend on machinery (a node, a refcount) that nobody here has written |
| 3 | **The cost of W4 in time.** | §4's last row is a node count, not a benchmark. Load average on this machine was 12.36 on 8 cores with three other agents running, and `CLAUDE.md`'s rule says that number is not usable. A before/after was **not attempted**, for that reason |
| 4 | **That `1723`/`1853` is the only such pair.** | §6 identified one number by reproducing it. Other counts in `docs/training/BACKWARD.md` and `docs/models/ADAPT.md` are quoted with `wrt_constants` left implicit and were not re-derived |
| 5 | **Anything about `torch.autograd.Function`, hooks, `create_graph`, or double backward.** | Inherited unchanged from `docs/training/BACKWARD2.md` §7.3: each is behind W5 rather than beside it |
