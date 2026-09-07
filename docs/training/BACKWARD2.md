# `Tensor.backward()`: the refusal chain, and what a second derivative implementation would cost

`docs/training/BACKWARD.md` §8 has one row that this document is entirely about:

> | `Tensor.backward()` | It differentiates *whatever produced this tensor*, which needs a node per op
> and a flag that propagates — `docs/training/AUTOGRAD.md` §6's `VariableType` half. The tape differentiates a
> *recorded region* and needs neither. The refusal stands and the test that pins it is unchanged |

That row is a sizing claim with no measurement behind it, made from the other side of the boundary.
This round measures it. The question is not *"can it be done"* — `docs/training/AUTOGRAD.md` §3 already settled
that `torch/csrc/autograd` defines `Py_BUILD_CORE` in **0 of 129 files**, so unlike `torch.compile`
this is not closed by abi3. The question is **what it costs, and whether it duplicates the tape.**

Environment: worktree `work/backward` on `develop` `e34f65d`, torch 2.13.0,
`/Volumes/macMini/caches/spike-venv/bin/python`. Every shim reading below printed `shim`.

### Answers, before the evidence

| question | answer |
|---|---|
| How many walls between `requires_grad=True` and a working `.backward()`? | **Ten.** Three raise, seven are silent. §1 |
| Which one does a user hit today? | `_strip_python_only_kwargs` — and then an **accidental** one, `_stash_obj_in_tls`, which is a thread-local dict and not autograd at all (§1, W2) |
| Cheap or structural? | **Three cheap** (a keyword, a dtype rule, a dict), **three semantic** (flag propagation, `is_leaf`, `no_grad`), **four structural** (graph construction, an eager recorder, lifetime, mutation). §2 |
| Would it duplicate the tape's mathematics? | **No, and the code already says so.** `derivative()` takes `&Node, &Env` and never mentions `PyCaptureTrace`; **1673 of tape.rs's 1968 lines are trace-independent**, and all 60 rules are in them. §3 |
| What landed? | The three cheap walls, and one **divergence from upstream** that the measurement turned up: the shim lets an *integer* tensor require gradients and upstream refuses. §4 |
| What still refuses, and by what name? | `_ImperativeEngine.run_backward`, which is now the **first** thing a `.backward()` hits rather than the third. §4.3 |

---

## 1. The refusal chain, in order

The literal command from the brief, on the pre-round artefact:

```
$ PYTHONPATH=.../torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 python -c \
    "import torch; torch.ones(2,2,requires_grad=True).sum().backward()"
NotImplementedError: not implemented in torch._C shim: torch.ones(requires_grad=True)
  -- there is no autograd behind this shim, and returning a tensor that quietly records
  nothing would be worse than refusing
```

That is wall 1 of ten. The rest were found by stubbing each one and re-running — the method
`docs/training/BACKWARD.md` §14.2 uses, and the only one that works here, because **seven of the ten do not
raise.** A wall that returns instead of raising is not findable by reading a traceback.

| # | wall | where | raises? | kind |
|---|---|---|:--:|---|
| **W1** | `requires_grad=True` at a factory | `bootstrap.py:2451` `_strip_python_only_kwargs` | ✅ | cheap |
| **W1b** | an **integer** tensor is allowed to require gradients | `bootstrap.py:4562` `requires_grad_` | — | cheap, **and a divergence** |
| **W2** | `torch._C._stash_obj_in_tls` | `torch/autograd/graph.py:989` `_engine_run_backward` | ✅ | cheap, **and accidental** |
| **W2b** | `torch._C._remove_obj_from_tls` | `graph.py` `finally:` | ✅ | masks W3 during unwind |
| **W3** | `_ImperativeEngine.run_backward` | `bootstrap.py:264` (the table-less stub) | ✅ | **the real one** |
| **W4** | `requires_grad` does not propagate through an op | `aten.rs:392`, the door | ❌ **silent** | semantic |
| **W5** | `grad_fn` is `property(lambda self: None)` | `bootstrap.py:4569` | ❌ **silent** | **structural** |
| **W6** | `is_leaf` is `property(lambda self: True)` | `bootstrap.py:4596` | ❌ **silent** | semantic |
| **W7** | `torch.no_grad()` gates nothing | `bootstrap.py:4648` `_set_grad_enabled` | ❌ **silent** | semantic |
| **W8** | there is no eager recorder | `capture.rs:64` `CAPTURING`, false outside a region | ❌ **silent** | **structural** |
| **W9** | nothing frees an eager tape | `capture.rs` — `Vec<Node>` indexed by `Ref`, no refcount | ❌ **silent** | **structural** |
| **W10** | mutation | `capture.rs:204` `is_mutating` — capture *refuses* it | ❌ **silent** | **structural** |

### 1.1 W1 has four more spellings than the one the brief quotes

`_strip_python_only_kwargs` covers every `_VariableFunctions` factory that goes through overload
resolution, which is most of them. Four sites carry their own copy of the same refusal:

```
torch.zeros / ones / empty / randn / full / arange / *_like   bootstrap.py:2451
torch.tensor(...)                                             bootstrap.py:4706
TensorBase.new_tensor(...)                                    bootstrap.py:3818
torch.frombuffer(...)                                         lib.rs:346
torch.asarray(...)                                            lib.rs:445
```

Measured, before the round — every one of these refuses:

```
[torch.zeros(2,requires_grad=True)]      NotImplementedError: torch.zeros(requires_grad=True) ...
[torch.empty(2,requires_grad=True)]      NotImplementedError: torch.empty(requires_grad=True) ...
[torch.randn(2,requires_grad=True)]      NotImplementedError: torch.empty(requires_grad=True) ...
[torch.arange(2,requires_grad=True)]     NotImplementedError: torch.arange(requires_grad=True) ...
[torch.full((2,),1.,requires_grad=True)] NotImplementedError: torch.full(requires_grad=True) ...
[torch.ones_like(x,requires_grad=True)]  NotImplementedError: torch.ones_like(requires_grad=True) ...
[x.new_tensor([1.],requires_grad=True)]  NotImplementedError: TensorBase.new_tensor(requires_grad=True) ...
```

**And `torch.ones(2,2).requires_grad_(True)` succeeds and always has.** So W1 refuses a *spelling*,
not a *property*: the tensor the refusal is protecting the user from is one line away and is handed
over without comment. §4.1 is what follows from that.

### 1.2 W2 is not autograd, and it masks the wall that is

Stubbing W1 gets to this, and `docs/training/AUTOGRAD.md` §1.2 already called it *incidental*:

```
File "torch/autograd/graph.py", line 989, in _engine_run_backward
NotImplementedError: not implemented in torch._C shim: torch._C._stash_obj_in_tls
```

`_stash_obj_in_tls` puts a `contextvars.Context` in a C++ thread-local so the engine's *device
threads* can read the compiler config. For a single-threaded CPU backward, a thread-local dict is
the entire observable contract. It is not a piece of autograd; it is a piece of thread plumbing that
happens to sit one line above autograd.

There is a second edge to it. Stub `_stash_obj_in_tls` alone and the next thing raised is **not**
`run_backward`:

```
reached run_backward: STOP-run_backward          <- the engine WAS called
File "torch/autograd/graph.py", line 989, in _engine_run_backward
NotImplementedError: not implemented in torch._C shim: torch._C._remove_obj_from_tls
```

`_engine_run_backward` removes the key in a `finally:`, so `_remove_obj_from_tls` raises **while the
engine's own exception is unwinding** and replaces it. This is `docs/training/BACKWARD.md` §14.1's shape
exactly — *"a refusal that a `finally` block overwrites is a refusal nobody can size"* — met a
second time, in a different file, and this time it is hiding the wall that matters most.

### 1.3 W3 is the real wall, and W4 is why it is not the last one

```
$ ... (TLS trio stubbed)
[y.backward()]        NotImplementedError: not implemented in torch._C shim: _ImperativeEngine.run_backward
[autograd.grad(y,x)]  NotImplementedError: not implemented in torch._C shim: _ImperativeEngine.run_backward
engine methods: ['is_checkpoint_valid', 'queue_callback', 'run_backward']
```

Both `Tensor.backward()` and `torch.autograd.grad()` land here, so it is one wall and not two. It is
called with everything upstream's engine is called with:

```
args:  [0] tuple (tensor(4.),)      the roots
       [1] tuple (None,)            <- the seed, and it is None
       [2] bool False               keep_graph
       [3] bool False               create_graph
       [4] tuple ()                 inputs
kwargs: allow_unreachable=True, accumulate_grad=True
```

**The seed is `None`, and nothing raised to say so.** `torch/autograd/__init__.py:208`
`_make_grads` builds `torch.ones_like(out)` only `if out.requires_grad`, and appends `None`
otherwise. `y = x.sum()` has `requires_grad False` — W4 — so the seed is silently absent:

```
[make_grads(y)]  ->  (None,)         # y.requires_grad is False; no error
[make_grads(x)]  ->  RuntimeError: grad can be implicitly created only for scalar outputs
```

This is the single most dangerous fact in the chain, and it is worth stating as a warning rather
than as a table row:

> **An engine implemented at W3 alone would receive `roots=(y,)`, `grads=(None,)` and
> `accumulate_grad=True`, and the obvious defensive thing to do with a `None` seed is to substitute
> a one.** For the scalar case in the brief that is *the right answer by coincidence*. For any
> non-scalar output it is silently wrong, and upstream would have raised. The refusal that protects
> against this today is W3 — not W1.

### 1.4 W4–W7, the silent semantic four

Measured side by side. Upstream is the same script with `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`.

| | shim | upstream 2.13.0 |
|---|---|---|
| `(x*x).requires_grad` | **False** | True |
| `x.sum().requires_grad` | **False** | True |
| `x.sum().is_leaf` | **True** | False |
| `x.detach().requires_grad` | False | False |
| inside `no_grad()`, `(x*x).requires_grad` | False | False *(for a different reason)* |
| `torch.is_grad_enabled()` round-trip | True/False | True/False |
| **`ones(2,2,dtype=int64).requires_grad_(True)`** | **True** | **`RuntimeError: only Tensors of floating point dtype can require gradients`** |

The last row is a **defect, not a boundary**, and it is the only place in the chain where the shim
is *more permissive* than upstream rather than less. `tape.rs`'s `wrt_set` already applies exactly
this rule one step later — `docs/training/BACKWARD.md` §4.1 records finding it there, when the reverse walk
reached `constant_pad_nd` on integer token ids and asked for the derivative of an integer, and calls
the fix *"upstream's own rule stated one step earlier"*. It was never stated at the step where
upstream states it. §4.2.

W7 deserves its own sentence because it is the one that looks harmless and is not. `no_grad()`
round-trips its flag and **nothing reads it**, which is honest today because there is no graph to
suppress. The moment W4 and W5 move, `no_grad()` becomes load-bearing in the most unforgiving way
available: `optimizer.step()` runs inside it, and a `step()` that built graph nodes would grow a
tape across every iteration of a training loop with nothing ever freeing it (W9).

### 1.5 W8–W10, the structural three

**W8 — there is no eager recorder.** `capture::CAPTURING` is an `AtomicBool` set by
`_capture_begin` and cleared by `_capture_end`, and `aten.rs:392` records only while it is set. The
recorder itself is not the problem: it is *one line at one door*, which is the whole of
`docs/training/AUTOGRAD.md` §6.1's argument. What is missing is a mode in which it runs without a region, and
that mode has to answer two questions a capture region never has to: when does recording start
(there is no `_capture_begin`), and what keeps the intermediates alive (`_capture_end` **drops the
keepalives**, which is why `tape::backward` replays the forward — `docs/training/BACKWARD.md` §1.3).

**W9 — lifetime.** A trace is a `Vec<Node>` addressed by `Ref { node, output }`. Indices are not
reference counts. Upstream frees a graph when the `grad_fn` chain's last reference drops, which is
`AutogradMeta` doing exactly the job `docs/training/AUTOGRAD.md` §2.5 lists and `docs/training/BACKWARD.md` §1 is
pleased not to have needed. An eager tape hung off a thread-global `Vec` grows for the life of the
process unless something clears it, and the only two honest options are (a) free at `backward()` and
refuse a second one — upstream's `retain_graph=False`, which is at least a *named* refusal — or
(b) build the refcount, which is W5 by another name.

**W10 — mutation, and this is the one that cannot be bought cheaply.** Capture gets single
assignment by **refusing** every `aten.*_.*` op (`capture.rs:204` `is_mutating`), and
`docs/training/BACKWARD.md` §1 lists that refusal as the reason no version counter was needed. An eager
`.backward()` cannot make that refusal, because the very next line of every training loop is
`optimizer.step()`, which is `add_`. So an eager tape must either

* refuse in-place ops on any tensor it has recorded — which stops `torch.optim` — or
* track a version per tensor and invalidate,

and the second is `c10::VariableVersion` plus the `ADInplaceOrView` key, i.e. precisely the layer
`docs/training/AUTOGRAD.md` §6 chose the tape *in order to avoid*. **This is the wall that makes
`Tensor.backward()` a different project from the tape rather than an extension of it**, and it is
not visible from anywhere in the first nine.

---

## 2. Cheap, semantic, structural — and each sized

| | walls | size | what it buys | what it costs |
|---|---|---|---|---|
| **cheap** | W1, W1b, W2/W2b | a keyword, a dtype rule, a thread-local dict | tutorial-shaped code reaches the wall that is real; `.requires_grad_()` stops disagreeing with the factory keyword; upstream's dtype rule holds where upstream states it | nothing — no graph, no flag propagation, no seed |
| **semantic** | W4, W6, W7 | one line each, **but they are one decision** | nothing on its own | **a lie.** `requires_grad` propagating while `grad_fn` is `None` and `is_leaf` is `True` describes a tensor upstream reserves for an *accumulating leaf*; `torch.optim` and `AccumulateGrad` both key on that triple. And it hands `_make_grads` a real seed for a graph that does not exist — §1.3 |
| **structural** | W5, W8, W9, W10 | the actual work | `Tensor.backward()` | a node per op with a refcount, an always-on recorder, a lifetime story, and version counters |

The split is not by difficulty. It is by **whether landing the wall alone leaves the shim honest**,
which is `docs/design/DESIGN.md` §6's criterion and the only one that survives contact with this area.
W1 and W2 pass that test in isolation. **W4, W6 and W7 do not** — each is one line, and each makes
the shim claim something untrue unless W5 lands in the same commit. That is why the semantic row is
not "cheap" even though it is smaller than the cheap row.

### 2.1 The order the structural four have to land in, and it is not the order they were found

W5 (a node per op) cannot land before W8 (a recorder to make the nodes), W8 cannot land usefully
before W9 (or the process leaks), and W9's cheap answer (free at `backward()`) is only sound if W10
is answered, because a tape freed at `backward()` still records operands that `step()` will mutate
*before the next* `backward()`. So the dependency is **W10 → W9 → W8 → W5**, exactly reversed from
the order a `.backward()` traceback reveals them. A round that starts at the traceback starts at the
end.

---

## 3. Whether it duplicates the tape — measured on the file, not argued

This is the question the round was called for. `trace.backward(inputs)` already differentiates a
captured region and moves all 272 SmolLM2 parameters where upstream moves them (sign agreement
99.9987%, `docs/training/BACKWARD.md` §4.2). **Two implementations of the same mathematics drift**, so if
`Tensor.backward()` would be a second one, that is a maintenance claim to make deliberately.

It would not be, and the evidence is mechanical rather than a reading. `tape.rs` is 1968 lines and
splits cleanly at line 1674:

| region | lines | `PyCaptureTrace` | `trace.` | `wrt` | `Ref::` | `&Node`/`&Env` |
|---|---:|---:|---:|---:|---:|---:|
| **rules + helpers** (`call` … `sdpa_backward`) | **1673** | **0** | **0** | **0** | **0** | 14 |
| the walk (`wrt_set` … `differentiable`) | 295 | 5 | 23 | 19 | 9 | 0 |

The two `PyCaptureTrace` hits the naive grep finds in the first region are a `use` line and a
sentence in a docstring; **the type is never used there.** All **60** entries of `RULE_OPS` and every
arm of `derivative()` live in those 1673 lines, and their entire contract with the world outside is

```rust
fn derivative<'py>(py, node: &Node, env: &Env, gouts: &[Option<Obj>], outs: &[Obj]) -> Rule<'py>
fn bind<'py>(py, node: &Node, env: &Env, names: &[&str]) -> Vec<Option<Operand<'py>>>
struct Operand<'py> { target: Option<Ref>, value: Obj<'py> }
```

`Node` is `{ op, args: Vec<Arg>, kwargs, outputs: Vec<Slot>, sequence }` and `Env` is three
`Vec<Py<PyAny>>` indexed the way `Ref` indexes them. **Neither is a trace.** An eager recorder that
produced `Node`s and an `Env` would reuse `derivative()` verbatim.

### 3.1 So the answer is: share, and the sharing is already paid for

**Judgement: `Tensor.backward()` must share the tape's derivative rules, and the code is already
factored so that it can.** Three reasons, in decreasing order of how much they would cost to
discover the hard way:

1. **The rules are the expensive, correctness-critical part, and they are already validated.** All
   60 are checked against central differences in `float64` against an oracle that shares no code
   with them, 30 sabotage faults have been run through them across two rounds, and the composition
   is checked element-wise against upstream on 134,515,008 real gradients. A second implementation
   would start that ledger at zero. `docs/training/BACKWARD.md` §12.2's warning about `native_layer_norm` —
   *"a rule that stops at `rstd * gh` has the right shape, the right dtype and the right order of
   magnitude and is wrong at every element"* — is what a second implementation would be re-earning.
2. **`RULE_OPS` is already pinned in both directions.** `_C._tape_rules()` is asserted equal to the
   gradient-case table, so a rule cannot be added without a case and a case cannot outlive its rule.
   A second rule set would need a second such pin, or it would drift silently — the exact failure
   `docs/verification/AUDIT.md` found six times.
3. **The difference between the two backwards is not the mathematics.** It is *where the nodes come
   from* (a region versus the door), *what keeps values alive* (a replay versus a live tape), and
   *what prunes the walk* (a declared `wrt_constants` set versus a propagated flag). Those are the
   295 lines, and they are the part that would legitimately be written twice.

**What would be duplicated, honestly stated:** `backward()` itself (~120 lines), `wrt_set`,
`reachable` and `wanted` (~60). `reachable()`'s doc comment already names the difference —

> *"Without this the walk would try to differentiate the rotary embedding's `arange`, because a tape
> has no `requires_grad` of its own and every value looks alike from inside. Upstream gets the same
> pruning from the flag; here it comes from the declaration of what the caller wants gradients for,
> which is the same information one step earlier."*

— so an eager twin would delete `reachable()` and read W4's flag instead. That is not duplication;
that is the same walk taking its pruning from the other of the two available sources.

### 3.2 The thing that would make sharing impossible, and it is W10

The rules are compositions issued through `aten_dispatch`, and `docs/training/BACKWARD.md` §1.1 states why
that is cheap: *"the backward runs outside a capture region, so it may use ops capture would refuse
to record, and it may recompute a value instead of reading a saved one."* Under an **always-on**
eager recorder, that sentence stops being true — the backward's own ops would be recorded by the
recorder that is now always on, including the in-place ones the rules use freely. So an eager
recorder needs a suppression scope around the backward, which is what `torch.no_grad()` *is* (W7),
which is why W7 is in the semantic row and not a footnote.

---

## 4. What landed

The three cheap walls and the divergence. **Nothing semantic and nothing structural**, for the
reason §2 gives: W4, W6 and W7 are one line each and one decision together, and taking that decision
without W5 would put the shim in the state `_install_autograd_shape` was written to argue against.

`docs/training/BACKWARD.md` §5.2's rule applied to a derivative — *"a wrong gradient looks exactly as
plausible as a right one and the program keeps running"* — is the whole reason this round stops
here rather than one wall further.

**`docs/training/BACKWARD3.md` tested that stopping decision rather than inheriting it, and upheld it — with
two corrections to the reasoning above.** The guard `torch.optim` actually keys on is **`is_leaf`,
not `requires_grad`** (`torch/optim/optimizer.py:1153`), and this shim already walks past it today,
so W4 does not *create* that divergence — it makes it plausible. And **W6 has no content that is not
W5**: `is_leaf` is defined upstream as `grad_fn is None`, so the semantic row is two walls and a
consequence, not three. BACKWARD3 §1–§4.

### 4.1 W1: the factory keyword now does what `requires_grad_` already did

The refusal's stated ground is *"returning a tensor that quietly records nothing would be worse than
refusing"*. That ground does not survive §1.1's measurement, for one reason:

```python
torch.ones(2, 2, requires_grad=True)      # refused
torch.ones(2, 2).requires_grad_(True)     # succeeds, and has always succeeded
```

Both produce the same object. The refusal therefore protects nobody — it redirects a caller to a
spelling that gives them the identical tensor **with no warning at all**, which is strictly worse
than either accepting or refusing both. `docs/training/AUTOGRAD.md` §1.2 found this and named it *"the same
tensor without the refusal"*; it did not follow it to the conclusion that the refusal is inconsistent
rather than conservative.

What replaces it is the flag being carried at the factory exactly as `requires_grad_` carries it,
with `.backward()` still refusing by name. The boundary does not move; it stops being drawn in two
different places for two spellings of one thing.

The `torchnative.adapt` stage-2 refusal at `adapt/__init__.py:265` is the load-bearing consumer of
this, and it gets **more** accurate, not less. Its probe is

```
Check: torch.ones(1, requires_grad=True).sum().backward(). If that returns instead of
refusing, an autograd exists that this refusal predates.
```

Before this round that probe refused at the **factory**, which is no evidence about autograd at all —
`.requires_grad_(True)` walks past it. It now refuses at `.backward()`, which is the thing the
sentence claims to be testing.

### 4.2 W1b: an integer tensor may not require gradients

The divergence §1.4's last row found. Upstream:

```
RuntimeError: only Tensors of floating point dtype can require gradients
```

This is not a boundary the shim was drawing deliberately — it is a rule that was never stated. The
same rule is already enforced two layers down, in `tape.rs`'s `wrt_set` and `reachable`, where
`docs/training/BACKWARD.md` §4.1 records having to add it after the reverse walk asked for the derivative of
a token id. Stating it where upstream states it means the tape's filter is a second line of defence
rather than the only one.

Upstream's message is transcribed exactly, including the fact that upstream's *factory* path uses a
differently-worded one (`Only Tensors of floating point and complex dtype can require gradients`) —
both were run on 2.13.0 and both are reproduced at the site that produces them.

### 4.3 W2: the TLS trio, so that the wall a user hits is the wall that matters

`_stash_obj_in_tls` / `_get_obj_in_tls` / `_is_key_in_tls` / `_remove_obj_from_tls` are a
thread-local key-value store, and that is the entire observable contract at this layer — the same
judgement `_install_grad_mode` records for `no_grad()`'s flag. Implemented over
`threading.local()`, not a module dict, because a global would be a *different* semantics that
happens to agree in a single-threaded process.

The point of implementing them is not that anything needs them. It is that with them absent,
**`.backward()` refuses under the name of a thread-plumbing helper**, and §1.2's `finally:` then
overwrites the engine's refusal with a second one. A user who reads that traceback learns nothing
about autograd. With them present:

```
NotImplementedError: not implemented in torch._C shim: _ImperativeEngine.run_backward
  -- Tensor.backward() differentiates whatever produced this tensor, which needs a graph
  node per op and a requires_grad flag that propagates through them; neither exists here
  (docs/training/BACKWARD2.md §1 walks all ten walls). What does exist is CaptureTrace.backward(),
  which differentiates a *captured region* and moves all 272 SmolLM2 parameters where
  upstream moves them (docs/training/BACKWARD.md §4) -- torchnative.adapt is the surface over it.
```

That is a refusal by name in `docs/design/DESIGN.md` §6's sense, and it names the alternative that works,
which the old one could not because it was generated by the table-less stub.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_thread_local_store present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _refuse_non_floating_requires_grad present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs derivative present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs is_mutating present -->

---

## 5. What still refuses, and by what name

| | name | why it is the right refusal |
|---|---|---|
| `Tensor.backward()` | `_ImperativeEngine.run_backward` | W3, and it is now the **first** wall rather than the third |
| `torch.autograd.grad()` | `_ImperativeEngine.run_backward` | the same wall; §1.3 measured that both land there |
| `requires_grad` propagation | — *(silent, by design)* | W4. `mul(x,x).requires_grad` is still `False` and the test that pins it is unchanged |
| `grad_fn` | — *(reports `None`, which is true)* | W5. No node was ever created |
| an integer tensor requiring gradients | `RuntimeError` | W1b, **new this round**, and it is upstream's message |

---

## 5.1 The command from the brief, before and after

```
before   NotImplementedError: not implemented in torch._C shim: torch.ones(requires_grad=True)
           -- there is no autograd behind this shim, and returning a tensor that quietly
           records nothing would be worse than refusing

after    NotImplementedError: not implemented in torch._C shim: _ImperativeEngine.run_backward
           -- Tensor.backward() and torch.autograd.grad() differentiate whatever produced a
           tensor, which needs a graph node per op and a requires_grad flag that propagates
           through them; neither exists here, and docs/training/BACKWARD2.md §1 walks every wall
           between them. What does exist is CaptureTrace.backward(), which differentiates a
           *captured region* -- torchnative.adapt is the surface over it, and _C._tape_rules()
           lists the derivative rules it has
```

Three walls fewer, and the one that is left is the one that is true. The frame it is raised
from moved from `torch_c_bootstrap.py:264` (the table-less stub) to
`graph.py:979 _engine_run_backward`, which is upstream's own call site.

---

## 6. Sabotage: six faults on what landed

`CLAUDE.md`'s rule — a check that cannot fail is not a check — applied before claiming the
tests as a gate. Each fault is applied to the tree, **rebuilt**, and the five affected tests
re-run; the tree is restored from a `cp` backup after every one.

| # | fault | caught by |
|---|---|---|
| F1 | the dtype rule dropped from `TensorBase::set_requires_grad` | ✅ `..._at_all_three_doors` — *"attribute setter let a torch.int64 tensor require gradients"* |
| F2 | `requires_grad_` leans on the Rust setter's message instead of carrying its own | ✅ the same test, on the **exact text** — which is what says transcribing three wordings is load-bearing rather than decorative |
| F3 | `_apply_requires_grad` silently drops the flag | ✅ `..._carried_not_dropped...` **and** `..._randn_carries...` |
| F4 | the thread-local store is a module dict | ✅ `..._is_thread_local_and_not_a_module_dict` — *"sees main's key: True"* |
| F5 | `run_backward` reverts to the generic stub | ✅ `..._autograd_boundary...`, on the missing `CaptureTrace.backward()` |
| F6 | `randn` drops the flag over its in-place `normal_` | ✅ `..._randn_carries...` — *"lost the flag over its in-place fill"* |

**Six of six**, and each names which claim broke. F6 is the one worth pointing at: `randn` is
`empty` + an in-place `normal_` here, so the flag has to survive a mutation — and while the whole
path refused, **nothing could check that it did.** A refusal removes the ability to test what is
behind it, which is the cost side of a refusal and is rarely written down.

---

## 7. Gates, and two checks that are not gates

Both gates pass on the final artefact.

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh
    335 ok, 0 FAIL          (333 before; +2 tests, and 2 inverted in place)
    SELF-TEST: PASS -- 20 comparators x 11 fault modes, 0 problem(s)
    DOCWATCH: PASS -- 252/252 evaluated marker(s) hold      (248 before; +4, all here)
    EXIT=0

$PY tools/golden/compare.py
    SUMMARY: 7763/7763 cases passed, 0 failed, ops covered=168, pending case builders=1
    EXIT=0
```

`ops=168` is unchanged **on purpose**, the same falsifiable prediction `docs/training/BACKWARD.md` §9 makes:
no kernel landed, so nothing here could have moved it.

### 7.1 The forward did not move

`docs/training/BACKWARD.md` §9.1's prefill sha256 over real SmolLM2-135M, re-measured on the final artefact.
This round edits the **dispatch hot path** (`_torch_level_function` and the `TensorBase` method
wrapper both unpack a tuple now), so this is the check that says the edit was free.

| S | f32 | |
|---:|---|:--:|
| 6 | `b9fc5553ee1bf6a2ea64d48de10c4fd214b3fd46810210e46f7bb85ce86f4a2f` | ✅ |
| 32 | `331668f36da02f217a013a40736aec2c3ccfcdeea6339495eb3610743fc1df77` | ✅ |
| 128 | `00159a9dbd308edabe6a3519e3acfa76b18181da691c8c9204d48d4270480d04` | ✅ |

All three equal `docs/training/BACKWARD.md` §9.1 and `docs/training/LOSS.md` §10.1.

### 7.2 The tape did not move

```
<CaptureTrace 1862 nodes, 1 inputs, 333 constants, 1 outputs>   loss 12.871352195739746
nodes 1862, on a gradient path 1853, missing rules {}
parameters with a gradient: 272 of 272
_C._tape_rules(): 60 rules
```

`1862`, `333`, `272 of 272` and the loss to every digit are `docs/training/BACKWARD.md` §4's.

**`on a gradient path` reads 1853 where `docs/training/BACKWARD.md` §4 says 1723, and that is not this
round.** The same script on the **unmodified** tree — `git stash push` on the four changed files,
rebuild, re-run, `git stash pop` — reports 1853 as well.

> **Corrected by `docs/training/BACKWARD3.md` §6: the 1723 was never stale, and no commit moved it.** The
> two numbers are two different questions asked of the same trace. `differentiable()` with no
> argument seeds the walk from *every floating constant* — all 333, including the 61 the trace burns
> in that are not parameters — and gives **1853, distinct ops 26**. `differentiable(wrt_constants=`
> *the 272 parameter slots*`)` gives **1723, distinct ops 20**, which is `docs/training/BACKWARD.md` §4's pair
> exactly, both numbers. The script above called the first; §4's called the second.
> `wrt_set`, `reachable`, `wanted` and `differentiable` are unchanged since `443220f`, the commit
> §4 documents, and `nodes_on_a_gradient_path` is computed before `has_rule` is consulted, so the six
> rules added since cannot have moved it.
>
> What this section got wrong is not the measurement but the comparison: a count of "nodes on a
> gradient path" has no value until you say a gradient path **to what**, and `wrt_constants` is that
> question rather than a filter on the answer — which is what `reachable()`'s own doc comment says.

The paragraph is left standing rather than rewritten because `docs/verification/DOCWATCH.md`'s thesis is that a
number nobody can check goes stale, and this one is prose inside a table — exactly the shape the
marker system structurally cannot see. It went stale in one round.

### 7.3 Two things measured and not claimed

* **Performance.** The two dispatch wrappers do one extra tuple unpack and one extra branch per
  call that reaches overload resolution; the `_fast` path returns before either. A before/after
  was attempted and **thrown away**: `uptime` reported a load average of 5.32 on 8 cores with two
  other agents running, and `docs/training/BACKWARD.md`'s own §"measurement work runs alone" rule says that
  number is not usable. §7.1's nine sha256 say the *answers* did not move; nothing here says what
  the calls cost.
* **`torch.autograd.Function`, hooks, `create_graph`, double backward.** Untouched, and each is
  behind W5 rather than beside it.

---

## 8. What this document does not establish

| # | not established | why |
|---|---|---|
| 1 | **That the structural four are the right four.** | They were found by instrumenting one command. A wall that only a *second* `.backward()` on the same graph would hit — `retain_graph`, `AccumulateGrad`'s `+=` semantics, `grad` on a non-leaf — is behind W5 and could not be reached to be counted |
| 2 | **The cost of the structural four in time.** | §2 sizes them by *kind*, which is what the round asked for. Converting that to weeks is the estimate-without-measurement this repository refuses |
| 3 | **That an eager recorder can reuse `derivative()` unchanged.** | §3 measures that it *does not depend on* `PyCaptureTrace`, which is necessary and not sufficient. Nobody has built a `Node` outside `capture::record` and fed it in |
| 4 | **That W10 has no cheap answer.** | The two options in §1.5 are the two that were thought of. A third — recording a *copy* of any operand an in-place op is about to touch, i.e. paying memory instead of a version counter — is not obviously worse and was not costed |
| 5 | **Anything about memory or about device.** | Desktop macOS, `float32`, one process. Inherited unchanged from `docs/training/BACKWARD.md` §8 |
| 6 | ~~**Which commit moved §7.2's 1853.**~~ **Closed by `docs/training/BACKWARD3.md` §6: no commit moved it.** | 1723 and 1853 are the same walk seeded from the 272 parameters and from all 333 floating constants respectively. Both are current on this tree; §7.2's "stale" reading is corrected in place |
