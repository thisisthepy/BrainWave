# Fifteen in-place ops with no kernel, and a sixteenth spelling

docs/bindings/SPELLINGS.md §9 measured 18 names that neither `torch.<name>(...)` nor `tensor.<name>(...)`
reached, and split them: 6 had a kernel already (`masked_fill clamp_ exp_ fill_ neg_` plus
`index_put_`, filled in that round), and 15 did not --

    abs_ ceil_ clamp_min_ cos_ detach_ erf_ expm1_ log_ log2_ reciprocal_ rsqrt_ sigmoid_ sin_
    sqrt_ tanh_

This round's job was the 15. A sixteenth item rode along: `native_group_norm`'s kernel already
existed (§9.0 found it had landed between checkouts) but had no `torch.<name>` spelling and,
being a function-only op upstream, no `Tensor.<name>` spelling either.

Environment: worktree `work/kern` on develop `fcb6926`, torch 2.13.0 upstream
(`/Volumes/macMini/caches/spike-venv/bin/python`). `_C._aten_implemented()` was read at runtime
before writing anything, per the task's instruction to re-verify rather than trust the list --
it matched exactly.

---

## 0. What landed

**14 of the 15 got a real kernel**: `abs_ ceil_ clamp_min_ cos_ erf_ expm1_ log_ log2_
reciprocal_ rsqrt_ sigmoid_ sin_ sqrt_ tanh_`.

**`detach_` did not, and that is a decision, not an omission** -- §5 below. It mutates
autograd metadata rather than storage, upstream refuses it unconditionally for a view (a
property this shim's `TensorBase` cannot detect), and computing an answer that is right for a
leaf and silently wrong for a view is exactly the divergence direction this file refuses
elsewhere. `detach_` is refused **by name**, with a message that states the reason, from a
dedicated dispatch arm (`aten.rs::detach_inplace_refusal`) rather than the generic "no table
entry" fallback -- reachable through both `t.detach_()` and `torch.detach_(t)`, both of which
have a table entry pointing at it.

`native_group_norm` got a **function spelling only**, in `overloads.json`. Measured (§9.0,
re-confirmed here): `hasattr(torch, "native_group_norm")` is `True`, `hasattr(torch.Tensor,
"native_group_norm")` is `False` on 2.13.0. No `methods.json` entry was added.

---

## 1. Write-through, not rebind

Every one of the 14 uses the same primitive the existing in-place ops (`exp_`, `neg_`, `clamp_`,
`relu_`, `fill_`) already use: `tensor.rs::write_into`, called through `aten.rs::write_back`. A
kernel computes the result into a **fresh** tensor of the receiver's shape and dtype, then
`write_back` writes that fresh tensor's values through the receiver's *layout* -- not through the
receiver's identity. That is what makes a view taken before the call see the write, and it is
what makes `t.sqrt_() is t` true (the wrapper is never rebound; `write_into` mutates candle's
`Arc<RwLock<Storage>>` in place and the kernel returns the same `PyTensorBase`).

Evidence, both properties in one probe (`PYTHONPATH=.../torchnative/src/main
TORCH_USE_RTLD_GLOBAL=1`, real vendored `import torch` against this shim):

```
t = torch.tensor([4.0, 9.0, 16.0])
v = t[0:2]                 # a view, before the mutation
r = t.sqrt_()
r is t                     -> True
t.tolist()                 -> [2.0, 3.0, 4.0]
v.tolist()                 -> [2.0, 3.0]        # the view sees the write
```

Same result for `clamp_min_`:

```
c = torch.tensor([1.0, -2.0, 5.0]); cv = c[0:2]
c.clamp_min_(0.0)
c.tolist()  -> [1.0, 0.0, 5.0]
cv.tolist() -> [1.0, 0.0]
```

The golden harness's `_view_write_cases` builder got one entry per new op (a `select.int` strided
view, read back through the **base** rather than the return value -- a return-value check cannot
tell write-through from rebind, since every in-place op returns `self` either way). All 14 pass.

---

## 2. Dtypes -- measured per op against upstream, not derived by analogy

Two shapes, and they split the 14 exactly along the line the out-of-place kernels already draw
between "promotes an integral input" and "does not":

### 2a. The eleven that promote out of place: in-place refuses instead

`cos_ sin_ erf_ log_ reciprocal_ tanh_ sqrt_ rsqrt_ expm1_ log2_ sigmoid_`. Their out-of-place
twins all promote an integral or boolean input to the default float (`unary_float`'s rule). An
in-place receiver has nowhere to put a wider dtype than the one it already has, so every one of
these refuses on every non-floating input, measured on all nine dtypes this shim stores
(`float64 float32 float16 bfloat16 int64 int32 int16 uint8 bool`):

```
int64/int32/int16/uint8/bool . cos_/sin_/erf_/log_/reciprocal_/tanh_/sqrt_/rsqrt_/expm1_/
    log2_/sigmoid_
        -> RuntimeError: result type Float can't be cast to the desired output type
           Long/Int/Short/Byte/Bool
```

reproduced through `inplace_cast_check` (the same door `exp_inplace`/`add_inplace` already use),
which is applied via `unary_float_tag(tag) != tag` -- a floating receiver keeps its own width and
computes; anything else refuses before candle is ever called. `sqrt_` on an integer tensor is
this row exactly, matching what the task's own example predicted.

### 2b. `abs_`/`ceil_`: no promotion out of place either, so in-place computes on ints

`abs`/`ceil` do **not** promote out of place (`int64.abs()` is `int64`, `torch.arange(3).ceil()`
is `int64`), so their in-place forms have nothing to refuse on that account and were re-measured
rather than assumed to follow the eleven above:

```
int64/int32/int16/uint8 . abs_()   -> computes, wrapping_abs (INT_MIN.abs_() is INT_MIN again)
int64/int32           . ceil_()    -> identity (already integral)
bool                   . abs_()    -> NotImplementedError: "abs_cpu" not implemented for 'Bool'
bool                   . ceil_()   -> NotImplementedError: "ceil_vml_cpu" not implemented for 'Bool'
```

Different kernel names for the two `bool` refusals (upstream reaches a different kernel), copied
rather than shared.

### 2c. `clamp_min_`: its own dtype refusal, not `clamp_`'s

`clamp_min_`'s in-place rule is `clamp_`'s *shape* (in-place cannot promote, where the
out-of-place `clamp_min` does) but not `clamp_`'s *code* -- the two diverge on exactly one row,
measured:

```
clamp_min_(bool, 0)      "result type Long can't be cast to the desired output type bool"
clamp_min_(bool, 0.0)    "result type Float can't be cast to the desired output type bool"
clamp_min_(bool, False)  NotImplementedError: "clamp_min_scalar_cpu" not implemented for 'Bool'
int32.clamp_min_(2.0)    "result type Float can't be cast to the desired output type int"
int64.clamp_min_(2.0)    "... long long"
int16.clamp_min_(2.0)    "... short"
uint8.clamp_min_(2.0)    "... unsigned char"
uint32.clamp_min_(2.0)   "... unsigned int"
uint8.clamp_min_(2)      OK, uint8 kept (an int bound never widens an int receiver)
```

The destination-side type name in that message is a **fourth** C++-scalar-type spelling this
file did not already have (`clamp_scalar_cpu_type_name` in `aten.rs`) -- distinct from
`TorchDType::name()` (`uint32`), `c10_name` (`uint32_t`), and `scalar_type_name` (`UInt32`).
Reusing `clamp_`'s own `clamp_dtype_refusals` would have reproduced `clamp_`'s wording (`Int`,
capitalised, from `scalar_type_name`) where upstream's `clamp_min_` says `int` (lowercase, the
raw C type) -- caught by writing the case builder and running it, not by reading the existing
function. **This also means the existing `clamp_dtype_refusals` (used by `clamp_`, out of this
round's scope) is missing the bool-bound-on-bool-receiver `NotImplementedError` row** --
`clamp_(bool_tensor, False, True)` measured upstream as `"clamp_scalar_cpu" not implemented for
'Bool'`, and this shim's existing `clamp_inplace_default` would give the generic "result type
Long can't be cast..." message instead. Left unfixed: `clamp_` has a kernel and golden coverage
already, and this round's file scope is the 15 named ops.

---

## 3. `detach_`: what upstream does, and why this shim refuses rather than guesses

Measured against upstream 2.13.0, `Tensor.detach_()`:

```
leaf.detach_()       requires_grad -> False, is_leaf stays True, grad_fn stays None,
                      returns self (identity)
non_leaf.detach_()    (y = x*2; y.detach_()) same triple -- y becomes indistinguishable from a
                      leaf that was never part of a graph
view.detach_()        RuntimeError: Can't detach views in-place. Use detach() instead. ...
                       -- true UNCONDITIONALLY: even a view of a tensor that never required
                       grad still refuses (measured: `c = torch.randn(4); c.view(-1).detach_()`
                       raises the identical message)
```

Two of the three leaf-case fields are **already** this shim's answer for every `TensorBase`,
independent of `detach_`, and not something this round changed: `is_leaf` is
`property(lambda self: True)` and `grad_fn` is `property(lambda self: None)`
(`bootstrap.py`, docs/training/BACKWARD2.md §1.4, W5/W6 -- neither landed). So the only field a `detach_`
kernel could actually change here is `requires_grad`, and *that* alone would be a one-line
`set_requires_grad` call -- except the view refusal, which fires **before** either flag is
touched and does not depend on `requires_grad` at all.

**This shim cannot tell a view from a non-view.** `PyTensorBase` (`tensor.rs`) carries `Repr`
(a candle `Tensor`, `Meta`, or `Quantized`), a torch dtype tag, `requires_grad`, `backward_hooks`
and `grad` -- no `_base` pointer, no `is_view` flag, nothing recording which op produced a given
wrapper. Views in this shim are a *storage* fact (candle's `Arc<RwLock<Storage>>` shared between
wrapper and base, per docs/kernels/VIEWS.md §6), not a *type* fact a kernel can read off the receiver.
Always taking the leaf branch -- `requires_grad = false`, return `self` -- would compute a value
for every receiver, including the ones upstream refuses: `y = x.view(-1); y.detach_()` would
silently succeed here and raise upstream. That is the silent-divergence direction this file
refuses everywhere else (`inplace_cast_check`'s promotion refusal, the `add_`/`sub_` narrowing
fix docs/numerics/SCALAR.md §6 records, `write_into`'s own tag/shape checks), so it is refused here too.

**Decision: refuse by name rather than implement the leaf-only subset.** `aten.rs::
detach_inplace_refusal` is a dedicated dispatch arm (not the generic `aten_not_implemented`
fallback) that checks `self` is a real `TensorBase` and then raises a message that states the
measurement above and points at the out-of-place `detach()` this shim already implements --
upstream's own suggested fix. `detach_` has table entries in both `methods.json` and
`overloads.json` (so the refusal is reachable and names itself, rather than an `AttributeError`)
but is **not** in `IMPLEMENTED` -- there is no kernel, and `_aten_implemented()`'s meaning
("has a kernel and is golden-compared") would be false for it otherwise.

---

## 4. Capture refuses all 14 by name; a control proves the check is real

`capture.rs::is_mutating` keys on whether the op name's `<op>` segment (`aten.<op>.<overload>`)
ends in `_`. It reads the string, not a hand-maintained list, so every one of the 14 new
in-place names is refused automatically -- but "automatically" is a claim worth measuring rather
than trusting, so it was measured at both layers `capture.rs` exposes:

**Raw dispatch** (`_aten_dispatch("aten.sqrt_.default", (t,), {})` inside a capture region):
poisons the recording under that exact key, for all 14 (`abs_ ceil_ clamp_min_ cos_ erf_ expm1_
log_ log2_ reciprocal_ rsqrt_ sigmoid_ sin_ sqrt_ tanh_`), each checked individually.

**Vendored Python spelling** (`x.sqrt_()`/`torch.clamp_min_(x, 0.0)` inside `_capture_begin`/
`_capture_end`, real `import torch` against this shim): every one of the 14 poisons the region;
`_capture_end` then raises

```
NotImplementedError: torch._C capture: cannot capture this region -- aten.sqrt_.default writes
in place; capture refuses mutation so that aliasing cannot be observed, which is what keeps a
trace single-assignment
```

**Control, required by the task and run rather than assumed**: the out-of-place `x.sqrt()`
inside the identical `_capture_begin`/`_capture_end` pair does **not** poison -- `_capture_end`
returns normally. This is the check that the refusal is really keyed on mutation and not on
"any op this round touched" -- an always-poisoning capture would pass every "refused" assertion
above for the wrong reason.

`capture.rs` itself needed **zero lines changed** -- the name-based rule already covered every
new op the moment its dispatch arm existed. That is itself worth stating plainly: the risk in
this round was never "capture forgets to refuse a new mutator," it was "the new mutator's kernel
computes the wrong value while capture correctly refuses it anyway" -- the two are independent,
and §1/§2 are what check the first.

---

## 5. Golden and smoke coverage

`tools/golden/cases.py`: one `CASE_BUILDERS` entry per new kernel (`cos__cases` ...
`clamp_min__cases`), covering float dtypes with domain-relevant probe values (signed grid for
`cos_`/`sin_`/`erf_`/`tanh_`/`expm1_`/`sigmoid_`, positive grid for `sqrt_`/`rsqrt_`/`log_`/
`log2_`/`reciprocal_`), the refusal rows from §2 above (`expect="both_error"`), an
`_inplace_member_cases` entry (write-through read from the *base* of a narrowed view, not the
return value -- shared machinery `exp__cases`/`neg__cases` already use), and a
`_view_write_cases` entry (a second, independent write-through proof through a different view
shape, `select.int` on a 2-D base). `detach_` has no builder -- it has no kernel, so it is not in
`_aten_implemented()`, and the harness's coverage rule only requires a builder for what is.

`rust/torch_c/pytests/test_shim.py::test_spellings_9_the_six_real_gaps_reach_their_kernels_through_the_vendored_tree`
(the §9 road script) used three of the 15 kernel-less names (`sqrt_`/`abs_`/`tanh_`) as its
"still refused, by exact key" regression pin. Implementing 14 of the 15 turned that pin red by
construction -- not a defect, the premise the pin was checking (no kernel) stopped being true.
The pin now runs the other way: all 14 are checked through **both** doors
(`torch.<name>(...)` and `t.<name>()`), value-checked against upstream and checked for
`is t` (a rebind-instead-of-write-through regression would still pass a value-only check),
`detach_` is checked to still refuse **by its own name** (`aten.detach_.default` in the message,
not a generic string), and `native_group_norm` is checked to be reachable as a function
(three-tensor result, right shapes) while `Tensor.native_group_norm` is still `AttributeError`.

**Sabotage, run rather than assumed to work**: with `overloads.json`'s `sqrt_` entry removed
(`cp` backup, restored after), rebuilt and reinstalled, the suite failed

```
FAIL test_spellings_9_the_six_real_gaps_reach_their_kernels_through_the_vendored_tree:
AssertionError: sqrt__fn: got 'ERROR:NotImplementedError:not implemented in torch._C shim:
torch.sqrt_(...) -- overload resolution has no table entry for this op
(rust/torch_c/src/overloads.json); call torch.ops.aten.sqrt_.<overload>, which carries the
overload and reaches the same dispatcher'
```

(the road script's `rec()` wrapper caught `torch.sqrt_`'s `NotImplementedError` -- the name no
longer resolves -- and recorded it as a string, which failed the `assert isinstance(got, list)`
guard and named exactly the op and file the entry was missing from). Restored from the `cp`
backup, rebuilt, reinstalled, reran: 343 ok again.

---

## 6. Gates

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh
    343 ok, 0 FAIL
    DOCWATCH: PASS -- 257/257 evaluated marker(s) hold
    EXIT=0

$PY tools/golden/compare.py
    SUMMARY: 8003/8003 cases passed, 0 failed, ops covered=182, pending case builders=1
    (168 -> 182, +14 -- exactly the 14 kernels this round added; `detach_` has no kernel so it
    does not move this number, `native_group_norm`'s kernel already existed before this round)
```

Two pre-existing regression pins needed updating (not weakening -- both assert a fact this
round's own additions changed the *count* of, and both got a paragraph explaining the new
number, following the pattern every earlier round in this file already uses):

* `test_shim.py::test_schema_text_survives_the_round_trip_through_the_transcribed_tables`:
  `len(keys) == 251` -> `268`. +17 = 13 (one schema each for the eleven promoting ops plus
  `abs_`/`ceil_`) + 2 (`clamp_min_`'s `.Tensor`/`.default` pair) + 1 (`detach_`) + 1
  (`native_group_norm`, `overloads.json`-only).
* `test_shim.py::test_the_seven_in_place_ops_say_that_they_mutate`: `_EXPECTED_MUTABLE` grew by
  the 14 kernels (their schemas really are `Tensor(a!) self`, upstream's own `is_mutable`
  agrees) -- **not** by 16: `clamp_min_.Tensor` and `detach_` are schema-mutable too but have no
  kernel, and the fixture this test reads (`report["ops"]`) is keyed off
  `torch._C._aten_implemented()`, not off the raw tables, so an op with no kernel is invisible
  to it regardless of its schema -- the same reason `clamp.Tensor`/`clamp_min.Tensor` were
  already absent from this list before this round.

---

## 7. Files touched, and what was deliberately not touched

`rust/torch_c/src/aten.rs` (14 kernels + `IMPLEMENTED` entries + `detach_inplace_refusal` +
dispatch wiring), `rust/torch_c/src/methods.json` / `overloads.json` (15 spellings + 1 function
spelling for `native_group_norm`), `tools/golden/cases.py` (14 case builders + 14
`_view_write_cases` entries), `rust/torch_c/pytests/test_shim.py` (two counters updated, one
road script/test extended).

**Not touched**: `capture.rs` (§4 -- the name rule already covered the new ops),
`clamp_dtype_refusals` in `aten.rs` (§2c names the bug it has; fixing it is `clamp_`'s file, not
this round's), `torchnative/src/main/torch/` (upstream's tree, off limits).
