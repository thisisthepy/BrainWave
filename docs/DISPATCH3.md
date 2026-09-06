# The dispatcher entrance — `_aten_dispatch` consults the mode stack

`docs/EXPORT.md` §6 item 1, and only item 1.

**`SEEN` matches upstream's operators, in upstream's order, in a separate
process — and disagrees with upstream on the overload spelling in two of the
three.** That disagreement is not the dispatcher entrance. It is
`docs/EXPORT.md` §5's `.Scalar` / `.Tensor` split, which `capture.rs` already
had and which the mode now reports from the same place, because the mode is
handed the key overload resolution picked. §5 below sizes what closing it
would take and why this round did not do it.

```
                      before            after            upstream
shim mode SEEN        []                aten.mul.Scalar  aten.mul.Tensor
                                        aten.add.Scalar  aten.add.Tensor
                                        aten.relu.default aten.relu.default
```

A mode can now **replace** a result rather than annotate one, which is the
half `export` actually needs, and `FakeTensorMode.__torch_dispatch__` is
reached from its infra slot. It still cannot return a fake, and the reason is
one level below this round — §4.

Measured 2026-09-07, `darwin/arm64`, CPython 3.13, `work/dispatch`.
Gates: suite **826 ok** (814 + the 12 in `rust/torch_c/pytests/test_dispatch.py`),
`DOCWATCH: PASS`, golden **11307/11307 ops=298 — exactly unmoved**.

---

## 0. At a glance

| | |
|---|---|
| Does a `TorchDispatchMode` see operators? | **Yes** — three, in order, for `(x * 2 + 1).relu()` (§2) |
| Same operators as upstream? | **Yes**, compared side by side in a separate process (§2) |
| Same overloads as upstream? | **No** — 2 of 3. `docs/EXPORT.md` §5's split, unchanged (§5) |
| Can a mode return something *instead of* the kernel's result? | **Yes** (§3) |
| Is `FakeTensorMode` reached? | **Yes**, from its infra slot, which the stack length does not count (§4) |
| Can `FakeTensorMode` return a fake? | **No** — `torch.is_inference_mode_enabled`, then `aten.empty_strided` (§4) |
| Does `capture.rs` double-record with a mode on the stack? | **No** — 3 nodes either way (§6) |
| Does the eager tape / `backward()` move? | **No** — same gradient with a mode, without one, and upstream's (§6) |
| Golden | **11307/11307, ops=298** — unmoved (§7) |
| `test_export.py`'s NOTE | **stopped printing**, which is what it was for (§7) |

---

## 1. What changed, and where it is

One place: `rust/torch_c/src/aten.rs`, in `aten_dispatch_entry` — the `*args,
**kwargs` door `_aten_dispatch` is bound to. Nothing else in the crate moved,
and `capture.rs` was not touched at all.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs any_dispatch_mode_active present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs innermost_dispatch_mode present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs dispatch_through_mode present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs overriding_types present -->

```rust
let rest = args.get_slice(1, args.len());
if any_dispatch_mode_active(py) {
    if let Some(active) = innermost_dispatch_mode(py)? {
        return dispatch_through_mode(py, active, op, &rest, kwargs);
    }
}
aten_dispatch(py, op, &rest, kwargs)
```

Four decisions in it are worth stating, because each of them is a place this
could have been subtly wrong rather than loudly wrong.

### 1.1 The entry, not `aten_dispatch`

`docs/EXPORT.md` §6 says "before `aten_dispatch_inner`". The consult is one
level further out than that, in `aten_dispatch_entry`, and the reason is
`capture.rs`: it **replays a recorded graph by calling `aten_dispatch` from
Rust**, where the op is already a `&str` and there is no argument tuple to
split. A replay is below the dispatcher rather than through it, and a mode
that intercepted one would be answering a call the user did not make.

Everything Python calls — `torch.<op>`, `torch.ops.aten.<op>.<overload>`, a
tensor method, `bootstrap.py`'s own helpers — arrives at the entry, so nothing
user-visible escapes the consult. The vulkan, mps and meta branches of
`aten_dispatch` are all downstream of it and are reached exactly as before.

### 1.2 The gate is torch's own flag, and it is one attribute read

The ordinary path — every golden case, every eager forward, every
`loss.backward()` — pays **one module attribute read and a branch not taken**:

```rust
module.getattr(intern!(py, "_is_in_torch_dispatch_mode"))
```

That is `torch/utils/_python_dispatch.py`'s module global, set in
`TorchDispatchMode.__enter__` and restored in `__exit__`. Reading torch's own
bookkeeping rather than keeping a second copy means the two cannot drift.

The alternative — calling `_len_torch_dispatch_stack()` per dispatch — is a
Python call on the hottest line in the crate, and it would still be *wrong*
on its own, because that function does not count infra modes here (§4.1).

The cost of the choice, said as a cost: a mode installed by calling
`torch._C._push_on_torch_dispatch_stack` directly, without the context
manager, does not set that flag and is not seen. Upstream's C++ does not use
the flag for dispatch. Nothing in the vendored tree installs a mode that way,
and `TorchDispatchMode.__enter__` is the only documented route, so this is a
difference recorded rather than a hazard hidden.

Both module handles are cached in `OnceLock`s filled by `cached_module`, which
**does not remember a failure** — the first dispatch of a process can happen
while `import torch` is still running, and caching that would disable the mode
stack for the life of the interpreter.

### 1.3 The mode is popped for the duration

Every `__torch_dispatch__` worth the name ends by calling `func(*args,
**kwargs)`, which comes straight back through the door. Without the pop that
re-entry finds the same mode on top and recurses until the stack blows.

This is the same thing `bootstrap.py::_through_torch_function_modes` already
does for the *torch-function* stack, for the same reason, so it is the shape
this tree uses rather than a new invention. Upstream uses a C++ guard.

The pop is restored on **every** exit path, including the one where the mode
raised. A mode leaked off the stack by an exception turns one error into a
process where the next `with` block enters, reports itself active, and returns
eager results — which is `docs/COMPILE.md` §5's silent fallback, arrived at
from the other side.

### 1.4 `types` is computed, not hard-coded

`torch/_tensor.py:457`'s own test — `type(a).__torch_dispatch__ is not
torch.Tensor.__torch_dispatch__` — which gives `()` for plain tensors,
matching what upstream produces for this program (measured, §2), and a
one-element tuple for a `FakeTensor` argument. Hard-coding `()` would have
passed §2 and been wrong for the case §4 is about.

Top-level arguments and keyword values only. Upstream walks into lists;
nothing reaching a mode here passes a tensor subclass inside a list today, and
adding the recursion without a case that needs it would be a guess.

---

## 2. `SEEN`, side by side

`rust/torch_c/pytests/test_dispatch.py` runs **one script twice** — once with
the vendored tree on `PYTHONPATH`, once with it removed — and compares. Not a
list of three strings written in the test file: the failure this round exists
to rule out is a mode that enters, reports itself active, and quietly returns
eager results, and only a side-by-side tells that apart from working.

```python
x = torch.ones(3)          # outside the block: a factory call is an operator too
with Log():
    y = (x * 2 + 1).relu()
```

| | shim | upstream |
|---|---|---|
| operators, in order | `aten.mul`, `aten.add`, `aten.relu` | **same** |
| full keys | `mul.Scalar`, `add.Scalar`, `relu.default` | `mul.Tensor`, `add.Tensor`, `relu.default` |
| `y` | `[3.0, 3.0, 3.0]` | `[3.0, 3.0, 3.0]` |
| `types` for a plain tensor | `()` | `()` |
| nested: inner mode sees / outer sees | 1 / 1 | 1 / 1 |
| a mode's own internal ops seen by itself | 0 | 0 |

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dispatch.py test_a_mode_sees_the_same_operators_as_upstream_in_the_same_order present -->

The re-entrancy row is the one that would have been quiet if it were wrong.
`_Reentrant.__torch_dispatch__` makes **two operator calls of its own**; a mode
still on the stack while it runs would see them, report three entries for one
user call, and then see the ones those made. One entry is the whole claim, and
it is checked against upstream's one rather than against the literal 1.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dispatch.py test_the_mode_is_popped_so_it_does_not_see_its_own_internal_operators present -->

---

## 3. A mode replaces the result

This is the half a post-hoc recorder structurally cannot do, and the reason
`docs/EXPORT.md` §4.2 called the capture hook "the right place for a recorder
and the wrong place for a mode".

```python
class Replace(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return "REPLACED"

with Replace():
    x * 2                     # -> 'REPLACED'
```

The mode is called once and the kernel never runs. `ProxyTorchDispatchMode`
returns a `Proxy` and `FakeTensorMode` returns a `FakeTensor`; neither is what
a kernel computed, and neither front end works unless the mode's return value
*is* the result.

**Upstream is stricter here and this shim is not.** Upstream raises
`RuntimeError: Unable to cast REPLACED to Tensor` at the C boundary, because
its binding declares a `Tensor` return; the shim's door returns
`Py<PyAny>` and hands the object back. That is a real difference, it is
recorded rather than asserted equal, and it is in the permissive direction —
a mode that returns nonsense gets further here than upstream. Tightening it
would mean deciding, per op, what a return type is, which is a schema question
and not a dispatcher one.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dispatch.py test_a_mode_replaces_the_result_rather_than_annotating_it present -->

---

## 4. `FakeTensorMode` — reached, and what stops it

### 4.1 Infra modes are not on the stack, and the length does not say so

`FakeTensorMode` and `ProxyTorchDispatchMode` carry a `_mode_key`, and
`torchnative/src/main/torchnative/export/upstream.py`'s
`_push_on_torch_dispatch_stack` routes them into a keyed slot rather than onto
the ordinary stack — reproducing upstream's split. But its
`_len_torch_dispatch_stack` returns `len(mode_stack)` only, where upstream's
C++ `stack_len()` counts the infra slots too:

```
inside `with FakeTensorMode():`     shim  _len_torch_dispatch_stack() -> 0
                                upstream  _len_torch_dispatch_stack() -> 1
```

**A dispatcher that read the length alone would see zero while a fake mode was
entered and would silently run the kernel** — the exact silent-fallback shape
this ordering exists to prevent, reached through correct-looking code. So
`innermost_dispatch_mode` reads the slots as well, reproducing upstream's
`TorchDispatchModeTLS::pop_stack` order: a user mode on the ordinary stack
wins if there is one; only when that is empty do the infra slots answer, in
reverse key order.

The discrepancy itself is in `upstream.py`, which is `docs/EXPORT.md` §8's
staging module and out of this round's territory. It is named here so the
hand-off in §8 can carry it.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dispatch.py test_an_infra_mode_is_found_in_its_slot_and_not_only_on_the_stack present -->

### 4.2 It is reached, and then it stops for reasons that are not this round's

`FakeTensorMode.__torch_dispatch__` **runs**, with `aten.mul.Scalar`. It then
raises:

```
NotImplementedError: not implemented in torch._C shim:
  torch.is_inference_mode_enabled(...) -- overload resolution has no table
  entry for this op (rust/torch_c/src/overloads.json)
```

and behind that, on the `from_tensor` path, `docs/EXPORT.md` §3.1 exactly as
written:

```
NotImplementedError: not implemented in torch._C shim: torch.empty_strided(...)
  @ torch/_subclasses/meta_utils.py:2008
```

So the honest statement of the harder half: **the entrance is landed and the
fake tensor machinery is not.** What was missing before this round was the
call; what is missing now is `aten.empty_strided`, a `torch._C`-level
`is_inference_mode_enabled`, the mutable-data-ptr bit and a meta storage
handle — `docs/EXPORT.md` §3, its item 3, not its item 1. None of them is in
this round's territory and none of them is a design change.

The test is written to demand more when they land: it asserts the mode is
*reached* today and, the moment a fake comes back, requires the fake's shape
to agree with upstream's instead of accepting a refusal.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dispatch.py test_fake_tensor_mode_is_reached_and_names_what_stops_it_returning_a_fake present -->

---

## 5. The overload spelling — the size of X

This is the one place the bar is not met, and it is worth being exact about
what would meet it.

`x * 2` reaches the door as `aten.mul.Scalar`. Upstream's dispatcher never
sees a `.Scalar` overload for that expression: `PythonArgParser` binds the
Python number to `mul.Tensor`'s `Tensor other` parameter and wraps it in a
0-dim "wrapped number" tensor, and the *dispatcher* then sees `mul.Tensor`.
The difference is therefore **above** the dispatcher on both sides — in the
argument parser — and the mode reports it only because the mode faithfully
reports the key the parser chose.

That it is the same disagreement `capture.rs` already had is checked rather
than asserted in prose: `test_the_overload_spelling_...` asserts
`mode_seen == capture_ops`, so it is one gap in one place and not two.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dispatch.py test_the_overload_spelling_is_the_only_remaining_disagreement_with_upstream present -->

**Size of X.**

| | |
|---|---|
| where | `rust/torch_c/src/bootstrap.py`'s overload resolution (the `raw_parse` reproduction), plus `rust/torch_c/src/overloads.json` |
| what | let a `Scalar` argument bind a `Tensor` parameter when another argument is a tensor, as upstream's parser does, and wrap it |
| ops affected | **21** names in `overloads.json` carry both a `.Tensor` and a `.Scalar` overload: `add`, `sub`, `rsub`, `mul`, `multiply`, `div`, `fmod`, `remainder`, `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `greater`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `masked_fill`, `fill_`, `bucketize` |
| what it moves | every one of those 21 spellings changes which kernel a plain `x <op> 2` reaches, and with it the dtype rule — upstream's wrapped number promotes *weakly* (an int literal does not widen a float tensor), which the `.Scalar` kernels get for free and a real 0-dim tensor argument does not |

**Not done, deliberately, and the reason is this round's own rule.** The
tempting shortcut is to normalise at the mode boundary only — hand the mode
`aten.mul.Tensor` and a wrapped 2 while the kernel keeps taking the `.Scalar`
path. That produces a shim where the operator a mode observes is not the
operator that ran, and where weak promotion applies on one path and not the
other. It is a plausible-looking wrong answer of exactly the kind
`docs/EXPORT.md` §6's ordering exists to keep out, and it would have made this
document able to claim the bar. The change belongs in overload resolution, as
one change, with golden re-run over all 21.

---

## 6. What capture and the eager tape do with a mode on the stack

Both ride this path and neither moved.

**`capture.rs`.** Same module, `(x * 2 + 1).relu()`, recorded twice:

| | nodes |
|---|---|
| capture, no mode | `mul.Scalar`, `add.Scalar`, `relu.default` |
| capture, `Log()` entered | `mul.Scalar`, `add.Scalar`, `relu.default` |

Identical, and that is the assertion — a length check would not have caught a
recorder that fired on the right number of wrong ops. The mechanism is the pop:
the mode answers by calling `func(...)`, which re-enters the door with the mode
off the stack, so the kernel runs once and the recorder fires once. A consult
placed after the recorder, or one that did not `return` early, would have
doubled every node.

The mode-answering case is the other half: a mode that returns without calling
`func` never reaches `aten_dispatch`, so `capture::record`, `mark_from_op`,
`capture::eager_record` and `capture::note_mutation` do not fire. That is
correct — no kernel ran and no value exists to record — and it is what
"must not fire while a mode is answering" means here.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dispatch.py test_capture_records_each_operator_once_while_a_mode_is_on_the_stack present -->

**The eager tape and `backward()`.** `docs/BACKWARD7.md`'s recorder is gated
on `mark_from_op`'s answer, beside the capture hook, and both sit in
`aten_dispatch` — below the consult. `w = ones(3, requires_grad=True)`,
`(w * 3).sum().backward()`:

| | `w.grad` |
|---|---|
| no mode | `[3.0, 3.0, 3.0]` |
| under a logging mode | `[3.0, 3.0, 3.0]` |
| upstream | `[3.0, 3.0, 3.0]` |

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_dispatch.py test_autograd_runs_the_same_with_and_without_a_mode_on_the_stack present -->

---

## 7. Gates, and the check on myself

```
golden      11307/11307 cases passed, 0 failed, ops covered=298, pending case builders=0
suite       826 ok, 0 FAIL          (814 before, + 12 in test_dispatch.py)
docwatch    DOCWATCH: PASS
```

Golden is the real statement about the ordinary path: **11307 cases through
`_aten_dispatch` with no mode on the stack, byte-for-byte the results and
refusals they were.** ops=298 unmoved, which is the other half — this round
added no kernel and removed none.

`test_export.py` printed a NOTE on every run naming this exact gap:

```
NOTE: a TorchDispatchMode enters and reports depth 1 while seeing no
operators -- docs/EXPORT.md §6, the dispatcher entrance is not landed
```

**It has stopped printing**, and `test_a_graph_front_end_is_not_offered_while_modes_are_not_consulted`
has moved to its other branch: `ops_seen_by_mode` is now non-empty, so it no
longer accepts "export refuses" as an answer and instead demands that if
`torch.export` ever returns a graph, that graph contains operators. The guard
got stricter by itself, which is what it was written to do.

### Nullification

§8 records the check that the tests are load-bearing rather than
self-satisfying.

---

## 8. Nullification

`any_dispatch_mode_active` forced to `false` — the consult unreachable,
everything else in the file untouched — rebuilt and re-run:

```
test_dispatch.py            10 FAIL of 12
test_export.py              the NOTE prints again
golden                      11307/11307 ops=298   (unchanged, as it must be)
```

The ten are every claim that depends on a mode being consulted. The two that
survive are the two that do not: `with no mode on the stack the answers are
upstream's`, which is the assertion that this change is invisible when nothing
has entered a mode, and `autograd runs the same with and without a mode`,
which reads the same gradient from both columns when neither column has a
working mode. Both surviving green is the right outcome and not a hole -- they
are the tests whose subject is the *unchanged* path.
Golden not moving under the nullification is the point of it: it confirms that
the 11307 cases are measuring the path this change did not take, so their being
green afterwards is evidence about the ordinary path rather than about nothing.

---

## 9. What this document does not claim

Split the way `docs/COMPILE.md` §5.3 asks for.

| | |
|---|---|
| **feature added** | the mode-stack consult in `aten_dispatch_entry`: user stack + infra slots, upstream's pop order, the mode's return value as the result |
| **binding surface implemented** | none. No `torch._C` name was added; `bootstrap.py` was not touched |
| **defect found** | `upstream.py`'s `_len_torch_dispatch_stack` not counting infra modes, where upstream's C++ does (§4.1) — a dispatcher reading it alone sees zero under `FakeTensorMode` |
| **tests added** | 12, in `rust/torch_c/pytests/test_dispatch.py`, every one compared against upstream in a separate process |
| **measurement** | `SEEN` side by side (§2), replacement (§3), the fake path's stopping point (§4.2), capture and the tape under a mode (§6) |
| **documentation corrected** | none. `docs/EXPORT.md` §4.2 is accurate as written and this closes what it named |

And the negatives, as negatives:

* **`SEEN` does not equal upstream's list character for character.** The
  operators and their order match; two of three overloads do not. §5 sizes it
  at 21 op names in overload resolution and says why doing it here would have
  been worse than not doing it.
* **`FakeTensorMode` cannot return a fake.** It is reached, and it stops on
  `docs/EXPORT.md` §3's items, which are the next round's and not this one's.
* **`_NodeBase` was not filled and the census names were not installed**, per
  `docs/EXPORT.md` §6's ordering. With this landed, they stop being the risk
  that ordering was written to prevent.
