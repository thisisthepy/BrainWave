# W5: `grad_fn` as a nullness, measured at the callers first

`docs/training/BACKWARD3.md` closed with a claim this round had to test before writing any code:

> **W6 has no content that is not W5.** Upstream's `is_leaf` *is* `grad_fn is None`; with W5 absent,
> W6 is either a tautology or it breaks upstream's own invariant.

and left two live divergences on the table — `optim.SGD([non-leaf])` and `requires_grad_` on a
non-leaf, both accepted here and both refused upstream, because `torch/optim/optimizer.py:1153`
guards on `param.is_leaf or param.retains_grad` and `is_leaf` is hardcoded `True`.

The brief's question was whether a *correct nullness* of `grad_fn` is reachable without the eager
tape W8/W9/W10 would build. **It is**, and §1 is the caller measurement that says so, taken before
the first line of implementation.

Environment: worktree `work/w5` on `develop` `d5cb426`, torch 2.13.0, transformers 5.15.1,
`/Volumes/macMini/caches/spike-venv/bin/python`, `CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-w5`.
Every shim reading printed `shim`; every upstream reading was taken with
`env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`.

### Answers, before the evidence

| question | answer |
|---|---|
| Who reads `grad_fn`? | **546 reads in `from_pretrained`, 0 in `.train()`, 0 in a forward with `labels=`, 1 in a `repr`.** All 546 are `nn.Module.__setattr__`'s `elif param.grad_fn:` — a truthiness test. §1.1 |
| Who reads `is_leaf`? | **272, all inside `optim.SGD.__init__`.** Nothing else on the path. §1.1 |
| Does `transformers` read either? | **Zero occurrences of `grad_fn`, `is_leaf` or `retains_grad` in the whole package.** §1.2 |
| Does any caller reach past nullness? | **One on an exercised path** — `torch/_tensor_str.py:646`, which takes `type(grad_fn).__name__`. Two more exist and are reachable only if the user asks: `Tensor.register_hook` (`_tensor.py:697`, `grad_fn._register_hook_dict`) and the deprecated non-full backward hooks (`nn/modules/module.py:1867`, `grad_fn.register_hook` + `.next_functions`). §1.3 |
| So is this round small or the structural group in disguise? | **Small — but not separable from W4.** A correct nullness *is* `requires_grad` propagation, because upstream's `grad_fn is not None` holds exactly when an op ran under grad mode on an operand that requires grad. §2 |
| What landed? | `grad_fn`, `is_leaf`, `retains_grad`, `retain_grad`, and **both divergences closed**. §4 |
| What still refuses, and by what name? | `_ImperativeEngine.run_backward`, unmoved — and it is now reached with a *correct* seed rather than a `None`, which is a harder property than the one docs/training/BACKWARD3.md §5.1 pinned. §5 |
| Did an engine get built? | **No.** §5.2 |

---

## 1. The caller measurement

Taken before any implementation, because it is what decides whether the round is
small or is the structural group in disguise.

### 1.1 Counted, on a real model

`grad_fn` and `is_leaf` are Python-visible properties on `TensorBase`, so both getters were
wrapped with a counter that records the calling frame, and a real `HuggingFaceTB/SmolLM2-135M`
was loaded, put in `.train()`, and given a forward with `labels=`.

```
=== from_pretrained:      546 reads   (546 grad_fn, 0 is_leaf)
    272  grad_fn   core_model_loading.py:1361:set_param_for_module <- module.py:1992:__setattr__
    274  grad_fn   modeling_llama.py / linear.py / sparse.py __init__ <- module.py:1992:__setattr__
=== .train():               0 reads
=== forward(labels=):       0 reads
=== optim.SGD(params):    272 reads   (0 grad_fn, 272 is_leaf)
    272  is_leaf   optimizer.py:405:__init__
=== repr(an intermediate):  1 read
      1  grad_fn   _tensor.py:564:__repr__ <- _tensor_str.py:715:_str
```

Every one of the 546 is `nn.Module.__setattr__`, `torch/nn/modules/module.py:628`:

```python
elif param.grad_fn:
    raise ValueError(f"Cannot assign non-leaf Tensor to parameter '{name}'. ...")
```

— a **truthiness** test, which a correct nullness answers completely. The 272 `is_leaf` reads are
`optimizer.py:405` walking `add_param_group`, which is the guard `docs/training/BACKWARD3.md` §1 identified
and could not close. **A `.train()` forward of a real 135M model reads neither name once.**

### 1.2 `transformers` reads neither, anywhere

```
$ grep -rn --include='*.py' '\.grad_fn\|\.is_leaf\|retains_grad' transformers/
(no matches)
```

Zero occurrences in the whole package, not merely zero on the measured path. That is a stronger
statement than `docs/training/BACKWARD3.md` §2 could make about `requires_grad`, which `transformers` writes
545 times.

### 1.3 The first caller past nullness, named

Three exist in the vendored tree. One is on an exercised path and two are reachable only if the user
asks for them:

| caller | what it reaches for | reachable |
|---|---|---|
| `torch/_tensor_str.py:646` | `type(grad_fn).__name__` | **yes** — every `repr()` of an intermediate |
| `torch/_tensor.py:697` `Tensor.register_hook` | `grad_fn._register_hook_dict(self)` | only if the user registers a hook on a non-leaf |
| `torch/nn/modules/module.py:1867` | `grad_fn.register_hook`, `.next_functions` | only via the **deprecated** non-full backward hooks |

Everything else in the tree that reaches further — `nn/parallel/distributed.py:1362`,
`distributed/fsdp/_runtime_utils.py:1464`, `distributed/optim/apply_optimizer_in_backward.py:66`,
`autograd/graph.py:195` — is behind a distributed or hook API this shim does not reach.

**So the boundary is `type(grad_fn).__name__`, and a non-None opaque object clears it** provided its
class is named the way upstream names it. That is one string, and §3.1 measures it.
`register_hook` on a non-leaf is where this round stops, and it stops by refusing with the name of
the missing attribute rather than by answering.

---

## 2. Why nullness is not separable from W4, and why that is not BACKWARD3's W4

`docs/training/BACKWARD3.md` §1 separated W4 from W5 and this round has to say plainly that they cannot be.
Upstream's condition for `grad_fn is not None` is

> an op ran, under grad mode, on an operand that requires a gradient, and the result can carry one

and that predicate **is** `requires_grad` propagation. There is no version of a correct `grad_fn`
nullness that does not compute it, because upstream also holds `grad_fn is not None => requires_grad`
— a tensor cannot report a node and deny requiring a gradient.

What *is* separable is the thing BACKWARD3 actually objected to. Its §4 argued:

> After W4 it would report `(True, None, True)`. Upstream reserves *that* for an accumulating leaf —
> a trainable parameter. That is a true statement about upstream and a false one about this shim.

That objection is to the **triple**, not to the flag. With `grad_fn` and `is_leaf` moving in the same
commit the triple is `(True, <MulBackward0>, False)`, which is upstream's description of an
intermediate and a true description of this one: it did come from an op, it is not a leaf, and
nothing will accumulate into it. BACKWARD3 §4.1 said exactly this would happen — *"landed together
with W5 … every row of §1.1's table moves onto upstream's answer at once"* — and §4.1 of this
document is that table, measured.

**No tape is needed for any of it.** The field is one `Option<Box<str>>` per tensor holding the op
name; there is no node, no `next_functions`, no saved operand, and nothing to free. W8/W9/W10 are
untouched.

---

## 3. What was measured before the code was written

### 3.1 The node's class name, 48 ops against upstream

The naive rule — CamelCase the aten base name, append `Backward0` — was run against
`type(y.grad_fn).__name__` on real torch 2.13.0 for 48 ops:

```
agree 41   differ 7
```

The seven, each for a reason worth keeping:

| op | upstream | the rule | why |
|---|---|---|---|
| `mul.Scalar` | `MulBackward1` | `MulBackward0` | the trailing digit is an **overload index**, not always 0 |
| `squeeze.dim` | `SqueezeBackward1` | `SqueezeBackward0` | same |
| `max.default` | `MaxBackward1` | `MaxBackward0` | same |
| `reshape.default` | `ViewBackward0` | `ReshapeBackward0` | upstream decomposes before it records |
| `linear.default` | `AddmmBackward0` | `LinearBackward0` | same |
| `to.dtype` | `ToCopyBackward0` | `ToBackward0` | the recorded op is the copy, not the cast |
| `contiguous.default` | **no node** | — | a no-op returns its own input, which is a leaf |

So `bootstrap.py` carries a measured exception table and falls back to the rule. **A wrong fallback
is a smaller divergence than the one it replaces**: before this round the shim printed no `grad_fn=`
field at all for any intermediate. `test_grad_fn_names_and_the_grad_mode_gate_agree_with_upstream`
checks the table *and* the rule against real torch, per op, so neither can rot into a guess.

The last row is not a table entry — it is the door's identity test (§4.2) answering it for free.

### 3.2 Which ops leave a leaf

Twelve of the 197 in `_aten_implemented()` need naming, and each was checked against upstream rather
than reasoned about: `detach`, the four `*_like`, `new_ones`/`new_zeros`, `lift_fresh`, `view.dtype`,
`histc`, `multinomial`, `randperm`. The rest of the non-differentiable ops — `argmax`, `sort`'s
indices, every comparison, `one_hot` — return integer or boolean tensors and are excluded by
upstream's own dtype rule instead, stated where upstream states it.

### 3.3 `retain_grad` on a leaf

Upstream returns early for a tensor that is already an accumulating leaf:
`x.retain_grad(); x.retains_grad` is `False` on 2.13.0. Measured, and matched.

---

## 4. What landed

Split the way `CLAUDE.md` §5.3 asks for:

| | |
|---|---|
| **feature added** | `grad_fn` (a correct nullness, with upstream's class name); `is_leaf` derived from it; `retains_grad`; `retain_grad`; `ones_like`/`zeros_like`/`empty_like` accept `memory_format=preserve_format` |
| **defect fixed** | **2** — `optim.SGD([non-leaf])` and `requires_grad_` on a non-leaf, both listed as live divergences by docs/training/BACKWARD3.md §1.1 |
| **test added** | 1 (`test_grad_fn_names_and_the_grad_mode_gate_agree_with_upstream`); 4 golden cases |
| **test inverted** | 2, both **strengthened** — §5 |
| **documentation corrected** | `PyTensorBase`'s field comments, `_install_autograd_shape`'s docstring, this document |
| **deleted** | — |

### 4.1 The two divergences, before and after

Measured side by side; the shim arm printed `shim`, the upstream arm was taken with
`env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`. `x` is a leaf requiring gradients, `y = x * 2`.

| | shim, before | shim, after | upstream 2.13.0 |
|---|---|---|---|
| `y.requires_grad` | False | **True** | True |
| `y.is_leaf` | True | **False** | False |
| `type(y.grad_fn).__name__` | `NoneType` | **`MulBackward1`** | `MulBackward0` ¹ |
| `repr(y)` | `tensor([2., 2., 2.])` | `tensor([2., 2., 2.], grad_fn=<MulBackward1>)` | `tensor([2., 2., 2.], grad_fn=<MulBackward0>)` ¹ |
| **`optim.SGD([y])`** | accepted | **`ValueError: can't optimize a non-leaf Tensor`** | the same |
| **`y.requires_grad_(False)`** | accepted | **`RuntimeError: you can only change requires_grad flags of leaf variables. …`** | the same |
| `setattr(y, "requires_grad", True)` | accepted | **`RuntimeError: you can only change requires_grad flags of leaf variables.`** | the same |
| `y.retains_grad` | `NotImplementedError: TensorBase.retains_grad` | **False** | False |
| `y.retain_grad(); y.retains_grad` | `NotImplementedError` | **True** | True |
| `x.is_leaf` / `x.grad_fn` (the control) | True / None | True / None | True / None |
| `no_grad(): (x*2)` triple | (False, None, True) | (False, None, True) | (False, None, True) |
| `(int*int)` is a leaf | True | True | True |
| `_make_grads` scalar | `(None,)` | **`(tensor(1.),)`** | `(tensor(1.),)` |
| `_make_grads` non-scalar | `(None,)` | **`RuntimeError: … only for scalar outputs`** | the same |
| **`s.backward()`** | `NotImplementedError: _ImperativeEngine.run_backward` | **the same, unmoved** | returns |
| `y.grad_fn.next_functions` | `AttributeError` on `None` | `NotImplementedError: grad_fn.next_functions — …` | `((<AccumulateGrad …>, 0), (None, 0))` |
| `y.register_hook(f)` | returned a handle that could never fire | `NotImplementedError: grad_fn._register_hook_dict` | `RemovableHandle` |

¹ **The one naming divergence, and its cause is dispatch rather than naming.** Upstream's `x * 2`
wraps the scalar and records `mul.Tensor`; this shim dispatches `mul.Scalar`, and upstream's own
`torch.ops.aten.mul.Scalar` reports `MulBackward1`. So the table is right about the op it is asked
about and the two answers differ because the two frontends choose different overloads. Keyed on the
op, this shim agrees with upstream on all 41 ops the test checks.

The last two rows are the round's boundary, and both are *louder* than what they replace. The
`register_hook` row in particular: before, a hook was accepted onto a tensor with no graph and would
never have fired — a silent nothing. Now the call names the attribute that does not exist.

### 4.2 The divergence this round chose not to close, and why

**An in-place op on a leaf that requires a gradient.** Upstream refuses it
(`RuntimeError: a leaf Variable that requires grad is being used in an in-place operation`); this
shim performs it, and the door **declines to change the leafness of a tensor that already exists**.

That is the W10 boundary (`docs/training/BACKWARD2.md` §1.5) arriving through the semantic door. An in-place
op returns its own receiver, so marking the output would mark the receiver — and
`optimizer.step()` is `add_`, so the first `step()` of any training loop would turn every parameter
in the model into a non-leaf and the *next* `add_param_group` would refuse it with the `ValueError`
this round just added. Upstream can afford to mark because upstream has `c10::VariableVersion` and a
leaf-mutation refusal; this shim has neither, and the honest answer at this size is to leave leafness
alone and say so.

`test_grad_fn_names_and_the_grad_mode_gate_agree_with_upstream` asserts both halves — that the shim
performs what upstream refuses, and that performing it leaves the parameter a leaf — so the
divergence is pinned rather than described.

### 4.3 `no_grad` became load-bearing, and so did the tape's backward

`docs/training/BACKWARD2.md` §1.4 predicted this exactly:

> The moment W4 and W5 move, `no_grad()` becomes load-bearing in the most unforgiving way available.

It does, in two places. The door reads a mirror of `_install_grad_mode`'s flag (an `AtomicBool`,
the same shape as `capture::is_active`) rather than the dict, because a `PyDict_GetItem` per dispatch
is a cost every caller pays. Two copies of one truth is the shape that drifts, so the test reads
`torch.is_grad_enabled()` and the leafness of an op's output in the same breath.

And `CaptureTrace.backward()` now runs under a `NoGradGuard`. Upstream's engine executes with grad
mode off, so the gradients it produces are leaves; the tape's backward is a composition of ordinary
`aten_dispatch` calls on tensors that require gradients, so without the guard every returned gradient
would carry a `grad_fn` — and `torch/optim/optimizer.py:1064` would then call `p.grad.detach_()`,
which this shim refuses by name. **A gradient is a value, not a node.** Verified: the first parameter
gradient off a real SmolLM2 capture reports `is_leaf=True grad_fn=None requires_grad=False`.

### 4.4 One capability had to be added, and the reason is the refusal's location

`torch/autograd/__init__.py` `_make_grads` builds the backward seed with
`torch.ones_like(out, memory_format=torch.preserve_format)` and only reaches that line
`if out.requires_grad` — never true of an intermediate until this round. `zeros_or_empty_like`
refused every `memory_format`, so with the flag propagating, `loss.backward()` started failing at

```
NotImplementedError: aten.ones_like.default: argument 'memory_format' not implemented
```

**which would have moved `Tensor.backward()`'s wall off `_ImperativeEngine.run_backward`.** The brief
requires that wall to stay where it is, and a wall that stops naming the thing that is missing is the
failure `docs/training/BACKWARD.md` §14.1 is about. So `ones_like`/`zeros_like`/`empty_like` now read
`memory_format` the way `full_like` and `clone` next door already do: the two formats that mean
"leave the layout alone" are accepted, `channels_last` still refuses, and four golden cases check it.

---

## 5. What still refuses, and by what name

`_ImperativeEngine.run_backward`, unmoved, for both `Tensor.backward()` and `torch.autograd.grad()`.

**No engine was built.** `docs/training/BACKWARD2.md` §1.3's trap was the reason not to, and this round
removed the trap rather than walking into it: the seed handed to the engine is no longer `None` for a
scalar and no longer `None` for a non-scalar — it is `tensor(1.)` and a `RuntimeError` respectively,
both upstream's own answers. The defensive substitution that would have been *"right by coincidence
for a scalar loss and silently wrong otherwise"* now has nothing to defend against.

**This round produced no gradient at all.** `CaptureTrace.backward()` is the only thing here that
produces one and its numbers are byte-identical to the pre-round artefact (§6.2), so the trap
`docs/graph/CAPTURE.md` §9-1 records has no new surface.

### 5.1 The two tests that were inverted, and how they got stronger

Both docstrings asked for exactly this and named the document to read first.

**`test_the_autograd_boundary_is_where_autograd_md_says_it_is`** asserted `y.requires_grad is False`
with the message *"an op propagated requires_grad — graph construction has appeared"*. An op now
propagates it and graph construction has **not** appeared, so one assertion could not express the
claim any more. It became four: the flag propagates, `is_leaf` is `False`, the name is
`MulBackward0`, **and** reaching for `next_functions`/`apply`/`metadata`/`_saved_self` refuses — plus
a leaf control, so it cannot pass on a `grad_fn` that is simply never `None`.

**`test_the_backward_seed_is_absent_and_nothing_guesses_a_one`** held a conjunction whose first
member changed sides:

| | before | after |
|---|---|---|
| 1 | the seed is **absent** in both shapes | the seed is **present and correct** in both shapes |
| 3 | nothing consumes it — the engine refuses | unchanged |

which is a harder pair to hold, not an easier one: the old version was satisfied by a shim with no
autograd at all, and the new one requires the seed to be built correctly *and* the engine to still
have nothing to walk. Four of its rows are now compared against the upstream subprocess directly
rather than against literals.

---

## 6. Gates

### 6.1 Suite and golden, on the final artefact

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh
    374 ok, 0 FAIL          (373 before; +1 test, 2 inverted, none removed, none weakened)
    SELF-TEST: PASS -- 19 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised
    DOCWATCH: PASS -- 318/318 evaluated marker(s) hold
    EXIT=0

$PY tools/golden/compare.py
    SUMMARY: 8440/8440 cases passed, 0 failed, ops covered=197, pending case builders=0
    EXIT=0
```

`ops=197` is unchanged **on purpose** — no kernel landed. The four extra golden cases are §4.4's
`memory_format` argument on two existing ops.

### 6.2 The forward did not move, and neither did the tape

This round edits the dispatch hot path, so both were re-measured **against a baseline built from the
pre-round sources in the same target directory** rather than against a digest from another document.
`git stash push -- <the four source files>` , rebuild, measure, `git stash pop`, rebuild, measure.
That is deliberate: the recipe behind `docs/numerics/SEQLEN.md` §1.3's digests is not in the repository, and a
digest whose input cannot be reproduced proves nothing about a tree it was not taken on.

Real SmolLM2-135M, `float32`, logits sha256 over the little-endian bytes of the flattened
`[1,S,49152]`:

| S | before | after | |
|---:|---|---|:--:|
| 6 | `606e00d1be05fccc…` | `606e00d1be05fccc…` | ✅ |
| 32 | `62d923807cce47ce…` | `62d923807cce47ce…` | ✅ |
| 128 | `4750428e94f7383c…` | `4750428e94f7383c…` | ✅ |

and the tape, same capture, `S=8`, every line byte-identical before and after:

```
<CaptureTrace 1862 nodes, 1 inputs, 333 constants, 1 outputs>   loss 15.292611122131348
differentiable()                      nodes 1862, on a gradient path 1853, missing {}
differentiable(wrt_constants=params)  nodes 1862, on a gradient path 1723, missing {}
parameter constants: 272 of 272        parameters with a gradient: 272 of 272
grad sum digest: 77d9cf3469ad4269348b8e95a8f434af
grad0 is_leaf/grad_fn/requires_grad: True None False
_tape_rules(): 60 rules
```

`1862`, `333`, `1853`, `1723`, `272 of 272` and `60` are `docs/training/BACKWARD.md` §4's and
`docs/training/BACKWARD3.md` §7.6.2's, reproduced here with both walk counts printed for the reason
`docs/training/BACKWARD3.md` §6 gives. The `grad sum digest` is this round's own addition and is the row that
would move if the tape's arithmetic changed at all.

A real `.train()` forward with `labels=` still gives `loss 3.636387586593628` and
`optim.SGD(model.parameters())` is still accepted, on 272 leaves.

### 6.3 Sabotage: five faults, five caught

Each fault applied to the tree, **rebuilt and reinstalled**, the three autograd tests re-run alone,
and the tree restored from a `cp` backup after every one.

| # | fault | caught by |
|---|---|---|
| F1 | `is_leaf` back to `property(lambda self: True)` — W6 undone | ✅ the boundary test, `assert y.is_leaf is False` |
| F2 | the grad-mode gate removed from the door | ✅ `no_grad_leaf` `[True, False, False]` where `[False, True, True]` was asserted |
| F3 | the in-place identity test removed from `mark_from_op` | ✅ *on the second attempt* — see below |
| F4 | a measured name (`mul.Scalar`) deleted, so the rule answers | ✅ named against upstream, per op |
| F5 | `requires_grad` stops following `from_op` | ✅ **all three tests**, which is the invariant being one thing and not three |

**F3 is the one worth the paragraph, and it came back NOT CAUGHT the first time.** The property it
breaks is "a parameter survives `optimizer.step()` as a leaf", and the test asserted it by running a
real `SGD.step()` — which upstream decorates with `no_grad`, so F2's gate answered before F3's test
could. A guard that only the *other* guard's absence exposes is not being checked. The assertion was
rewritten to mutate a parameter in place with grad mode **on**, which is the condition F3 is the only
thing standing in, and it then failed by name. Same shape as `docs/training/BACKWARD3.md` §7.5's F5: a check
that cannot fail under the conditions you tried it is not yet known to be a check.

---

## 7. What this document does not establish

| # | not established | why |
|---|---|---|
| 1 | **The cost of the door's new pass, in time.** | Load average on this machine was 4.2 with other agents running, and `CLAUDE.md` forbids reporting a number taken there. What is structural rather than measured: the common case is one relaxed atomic load plus one `bool` borrow per tensor operand, and it returns before allocating; the SmolLM2 prefill above runs under `no_grad`, so it pays only the atomic. A before/after benchmark was **not attempted** |
| 2 | **That the naming fallback is right for the 150 ops the test does not cover.** | §3.1 measured 48. The fallback's claim is only that it is a smaller divergence than printing no field at all, and §4.1's footnote records the one op where shim and upstream disagree for a reason that is not the table's |
| 3 | **That no caller anywhere reaches past nullness.** | §1.1 counted one path on one architecture, and §1.3 read the vendored tree. `generate`, PEFT, `Trainer`, gradient checkpointing and every `torch.nn` module SmolLM2 does not use were not exercised |
| 4 | **Anything about `torch.autograd.Function`, hooks that fire, `create_graph`, or double backward.** | All four need a graph, which is W8/W9/W10 and is untouched. `register_hook` on a non-leaf refuses by name and that is the whole of the answer |
| 5 | **That W10 is closer.** | §4.2 declined it explicitly. `docs/training/BACKWARD2.md` §8 row 4's uncosted third option is still uncosted |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs mark_from_op present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs any_operand_requires_grad present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs NoGradGuard present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _GradFnNode present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _grad_fn_name present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_grad_fn_names_and_the_grad_mode_gate_agree_with_upstream present -->

---

## 8. The node names, re-measured on review — two table bugs and one that is not

The class-name table was built from 48 ops. Comparing it against upstream on 33
*user-level* expressions rather than raw aten keys found three disagreements, and
they were three different things:

| expression | upstream | was | why |
|---|---|---|---|
| `x @ x` | `MmBackward0` | `MatmulBackward0` | **table bug.** `matmul` is `CompositeImplicitAutograd`; upstream decomposes before autograd records, so `MatmulBackward0` exists nowhere in upstream. The same class as the `reshape → ViewBackward0` and `linear → AddmmBackward0` rows already in the table — simply missed. Fixed. |
| `x.clamp(0, 1)` | `ClampBackward1` | `ClampBackward0` | **table bug.** The trailing digit is an overload index and it is `1` for every clamp form measured — both bounds, min only, and the `Tensor.clamp` spelling. Fixed. |
| `x * 2` | `MulBackward0` | `MulBackward1` | **not a table bug.** `torch.ops.aten.mul.Scalar(x, 2)` is `MulBackward1` upstream too, so the table row is right. What differs is *which overload gets dispatched*: upstream wraps the Python scalar and runs `mul.Tensor`; this build runs `mul.Scalar`. The name is a faithful label on the op actually recorded. |

That last one is a **dispatch** divergence, older than this round and visible only
now that the node has a name. It is left alone rather than papered over by
mapping `mul.Scalar` to `MulBackward0`: the label would then disagree with the op
in the trace, and a user reading `torch.ops.aten.mul.Scalar` directly would see a
name upstream does not give it. Recorded here so the next round that touches
scalar dispatch knows the printed name moves with it.

**Followed up in docs/numerics/SCALAR2.md**, which enumerated the whole family by
running it. Two things there change how this row reads. First, `mul` is not one
operator's problem: upstream wraps for `add`, `sub`, `mul`, `div` and
`floor_divide` and dispatches `.Scalar` for everything else, so five families
diverge here and nothing singles `mul` out. Second — and this is the part §8
could not have known — **`add`, `sub` and `div` already take the trade this row
declined.** Measured at the raw key, upstream's `aten.add.Scalar` is
`AddBackward1` and this build's is `AddBackward0`; the same for `sub` and `div`.
They fall through the naive rule, which happens to name the *expression*
correctly and the *key* incorrectly. `mul` is the only one of the four faithful
to its key, which is why it is the only one this section caught. SCALAR2.md §4
has the table and leaves the inconsistency standing rather than smoothing it in
either direction; §3 there says why the dispatch itself was not changed.

After the two fixes, 32 of 33 agree.
