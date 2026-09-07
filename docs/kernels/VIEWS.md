# VIEWS.md — the four kernel gaps docs/kernels/GROUPED_MM.md §6.4 recorded

Binding the seven `TensorBase` members Mixtral needed (docs/kernels/GROUPED_MM.md §6.4) closed every
*name*. Doing so exposed four things that are not names: they are kernels that resolve and then
refuse, or capabilities the storage model does not have. §6.4 recorded them precisely rather than
writing them, because that change could not touch `aten.rs`. This document is what happened when
someone could.

Ordered smallest first, which is also the order they were done in.

| | gap | verdict |
|---|---|---|
| §1 | `aten.ge.Tensor` has no kernel | **kernel, one arm** — closed |
| §2 | `index_put_` refuses a bool mask | **kernel** — closed |
| §3 | `index_put_` refuses non-1-D operands | **kernel** — closed |
| §4 | `select.int`/`slice.Tensor` return copies, not views | **see §4** — superseded by §6 |
| §6 | the write-through redesign §4 specified | **done** — see §6 |

Baseline, before any of it (this worktree, `e50084f`):

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh    223 ok
$PY tools/golden/compare.py                  2971/2971, ops=121
$PY tools/golden/compare.py --self-test      13 comparators x 11 fault modes, 0 problems
$PY rust/torch_c/pytests/verify_schemas.py   4231/4231
```

---

## 1. `aten.ge.Tensor` — the sixth comparison, and the only one without a Tensor overload

`le.Tensor`, `lt.Tensor`, `gt.Tensor`, `eq.Tensor` and `ne.Tensor` all had a kernel.
`ge` had only `.Scalar`. §6.4 put both of `ge`'s schema strings into `methods.json` for symmetry
with the other five, which made `x >= tensor` **resolve** — and then refuse inside
`_aten_dispatch`, by name, on a key that was not in `_aten_implemented()`.

That is the good failure mode, and it is also exactly one arm of work.

### The change

`rust/torch_c/src/aten.rs`, two lines: `"aten.ge.Tensor"` in the implemented list, and

```rust
"aten.ge.Tensor" => compare_tensor(py, args, kwargs, "aten.ge.Tensor", Cmp::Ge),
```

There is no new arithmetic. `compare_tensor` already reads both operands into one exact common
representation (`f64` if either side is floating, `i64` otherwise) and `Cmp::Ge` was already a
variant of the enum, used by `ge.Scalar`. The five siblings are the same call with a different
`Cmp`.

### What the cases pin, and what they could not have pinned

`ge_tensor_cases` mirrors `gt_tensor_cases` deliberately, because the two kernels differ in one
enum variant and a wrong variant is the realistic defect. Three cases separate them:

* **`x >= x` is all `True`** where `gt.Tensor`'s matching case is all `False`. This is the
  assertion that distinguishes `Cmp::Ge` from `Cmp::Gt` and nothing else does.
* **NaN on either side is `False`**, including `nan >= nan` — `>=` is the reflexive comparison
  everywhere except on NaN, so a kernel that "helpfully" made it reflexive would pass every other
  case here.
* **the causal-mask idiom**, `arange(S)[:,None] >= arange(S)[None]`, which is `le.Tensor`'s
  measured case written the other way round.

Plus the eight dtypes × two broadcast scenarios the whole comparison family shares, and
`bool` compared as 0/1.

**Six of the cases go through the member**, not through `_aten_dispatch`: `x >= y`, `x.ge(y)` and
`x.__ge__(y)` on three dtypes, plus a 0-d right-hand side (which still picks `ge.Tensor`, since
the overloads are told apart by the argument's *type* and not its rank) and a NaN pair. That
distinction is §6.4's own lesson: the kernel-level cases for `clamp_`/`div_`/`masked_fill_` passed
for weeks while the members raised `NotImplementedError`. Here it runs the other way — the member
bound and the kernel refused — and the member cases are what fail if either half regresses.

### Sabotage

Deliberately broken, rebuilt, and counted rather than assumed:

| injected fault | golden cases failed | smoke tests failed |
|---|---:|---:|
| `Cmp::Ge` → `Cmp::Gt` on the new arm | **30** of 3002 | 1 (`test_the_mixtral_member_names_reach_the_kernels_that_were_already_there`) |
| the arm deleted entirely (back to refusing by name) | **31** of 3002 | — |

The two numbers differ by one, and the one is the `x >= x` equality-boundary case: it is the only
case in the suite whose *answer* changes between "computed with the wrong comparison" and "refused".
Every other case fails both ways, for different reasons — value mismatch in the first, one side
raising where the other computed in the second.

Restored from a `cp` backup afterwards and confirmed with `git diff --stat`.

### Counts after §1

```
run.sh                 223 ok       (unchanged — the new assertions are inside an existing test)
compare.py             3002/3002    ops=122   (+31 cases, +1 op)
compare.py --self-test 13 x 11, 0 problems    (unchanged — no new comparator)
verify_schemas.py      4233/4233    (+2: one more `_schema` text/is_mutable row and one more
                                     `OpOverload.tags` row, both derived from the implemented set)
```

One smoke-test constant moved and it is a real consequence, not a fixture edit:
`test_core_ops_and_op_tags_agree`'s `tag_core_count` went **77 → 78**. It counts *implemented* ops
that upstream tags `core`, and `ge.Tensor` is core upstream exactly as the `le`/`lt`/`gt` siblings
already counted there are. The test asserting a number rather than a range is what made that
visible in one line.

---

## 2-3. `aten.index_put_.default` — a bool mask, and operands above rank 1

These are two entries in §6.4's list and **one cause**. The kernel did not implement `index_put_`;
it built a `scatter.src` call and let that op do the work:

```rust
// before
let scatter_args = PyTuple::new(py, [receiver, 0i64, index, values])?;
let result = scatter_src(py, &scatter_args, None)?;
```

`scatter.src` requires an int32/int64 index, and index/src/self all of the same rank. So:

* a **bool mask** hit its dtype check — `Expected dtype int32 or int64 for index, got bool` — which
  §6.4 recorded as a `c_error` golden case rather than leaving uncased;
* **anything but rank 1** hit an explicit guard put in front of it
  (`only a 1-D self/index/values is implemented in torch._C shim`), which is why `x[t] = 5`
  refused: the number lifts to a 0-d tensor, and 0 is not 1.

Borrowing another op's arithmetic was the right call for the one shape Mixtral needed
(`inv_perm[perm] = arange(...)`, int64 and 1-D on every side). It does not generalise, because
`index_put_` and `scatter` do not mean the same thing.

### A mask is a different operation, not a cast

The tempting fix is to convert the mask to an integer index. That is wrong, and wrong with a
plausible shape, which is the failure mode this op invites: `x[tensor([True,False,True,False])]`
selects positions **0 and 2**, whereas the integer index `[1,0,1,0]` selects positions
**1, 0, 1, 0**. Same tensor, four elements either way, different answer.

The correct lowering is mask -> coordinates, and it is upstream's own move
(`at::native::expandTensors`). It was **already implemented here** as `mask_to_indices`, for
`index.Tensor`, along with the shape check and upstream's two-part error message. So `index_put_`
calls it rather than growing a second mask reader that could disagree with the first.

Two consequences that the tests pin because they are easy to get wrong:

* **A `k`-dimensional mask consumes `k` axes** and contributes one axis of length `count` to the
  result. The same mask values give different answers depending on the mask's rank —
  `x(2,3)[mask(2,3)] = [1,2,3]` writes three elements and `x(2,3)[mask(2,)] = [1,2,3]` writes a
  whole row. Both are cased, with the same numbers, so a kernel that reads the contents and
  ignores the rank fails one of them.
* **`uint8` is a mask too.** Upstream treats it as a deprecated spelling of `bool` and warns; both
  spellings are cased.

### The general kernel

`index_put_` now does its own address arithmetic. One index group at axis `a` consuming `m` axes,
and the indexing result shape is `dims[..a] ++ index_shape ++ dims[a+m..]` — always the spliced
form, because with **one** group `index.Tensor`'s fronting-versus-splicing rule (which needs two
separated groups to matter) cannot apply. `values` broadcasts onto that shape right-aligned, and
the walk is row-major over it.

Everything it does was measured against torch 2.13.0 first and reproduced second, including the
refusals:

| | upstream | here |
|---|---|---|
| bool / `uint8` mask | selects true positions | same, via `mask_to_indices` |
| `self` of any rank, 1-D index | indexing result `dims[..a] ++ S ++ dims[a+1..]` | same |
| index of any shape | spliced in whole | same |
| `[None, t]` (`x[:, t] = v`) | index sits at axis 1 | same |
| `values` broadcast | right-aligned onto the result | same |
| `values` that does not fit | `shape mismatch: value tensor of shape [3] cannot be broadcast to indexing result of shape [2, 2]` | verbatim |
| dtype mismatch | `Index put requires the source and destination dtypes match, got Float for the destination and Long for the source.` | verbatim, no promotion |
| negative index | wraps | wraps (`scatter` had no rule for this and refused) |
| out-of-range index | `index 9 is out of bounds for dimension 0 with size 5` | verbatim |
| mask shape mismatch | `The shape of the mask [3] at index 0 does not match the shape of the indexed tensor [4] at index 0` | verbatim |
| float index | `tensors used as indices must be long, int, byte or bool tensors` | verbatim |
| empty index / all-false mask | writes nothing, returns `self` | same |
| repeated position | last write wins | same |
| two index tensors | computes | **refused by name** — not measured, not guessed |
| `accumulate=True` | computes | **refused by name** — unchanged |

The two remaining refusals are deliberate and unchanged: refusing where upstream computes is a
recorded gap, and computing where upstream refuses is the silent divergence this repository does
not ship.

### One thing moved that is not in `aten.rs`: `_lift`'s dtype

`bootstrap.py`'s `__setitem__` lifts a bare Python number to a 0-d tensor before dispatching.
It inferred the dtype **from the Python type** — `int` -> `int64`, `float` -> `float32`.
Upstream does not. Measured with a `TorchDispatchMode` logger:

```
float32 x;  x[t] = 5     ->  lift_fresh(float32()) ...  index_put_
int64   x;  x[t] = 5.0   ->  lift_fresh(int64())   ...  index_put_
float32 x;  x[t] = True  ->  lift_fresh(float32()) ...  index_put_
bool    x;  x[t] = 2     ->  lift_fresh(bool())    ...  index_put_
```

**It is always the receiver's dtype.** The old rule survived because `index_put_` only accepted a
1-D receiver, so the one call that reached it was int64 on both sides — in the agreeing half. Both
halves are reachable now, and `index_put_` requires the dtypes to match exactly, so the old rule
would turn a write upstream performs into a *refusal*.

### Cases

`aten.index_put_.default` goes from 11 cases to 45, and 15 of those go through the **member**
(`x[mask] = v`, `x[idx] = 5`, `x[:, idx] = v`) rather than through `_aten_dispatch`. Every fault
below fails one door-level case and one member-level case, which is what that pairing is for.

Two case shapes are worth naming separately:

* **the ones that read the original binding.** `index_put_` returns `self`, so a case that reads
  the return value passes just as well against a kernel that built a fresh tensor and handed it
  back. Two cases throw the return value away and read the name that was passed in. That is the
  only shape that can fail when a write lands in a copy — the failure mode this whole document is
  about.
* **the ones that admit they cannot fail.** `x[:] = 3.0` on an int64 receiver and `x[:] = 2` on a
  bool receiver go through `fill_.Tensor`, and `fill_` takes its dtype from the receiver on both
  sides — so **neither of them discriminates between the two lift rules.** That is written into
  their note rather than implied away. The case that does discriminate is `x[idx] = 2` on a bool
  receiver, where the old rule refuses; it was added after sabotage F showed the `fill_` case
  staying green, which is what the sabotage pass is for.

### Sabotage

Each fault was injected into the built artefact, rebuilt, and counted:

| injected fault | golden failed | which cases | smoke |
|---|---:|---|---:|
| a mask always consumes exactly one axis | *aborted* | panicked at the 2-D-mask case (index out of bounds) | — |
| mask offsets drop the axis stride | 2 / 3037 | the 2-D mask, at the door and through the member | 1 |
| `values` broadcast left-aligned instead of right | 2 / 3037 | the `(2,)` broadcast and the 1-D-mask-over-2-D row | 1 |
| negative indices not wrapped | 2 / 3037 | the negative-index pair | 1 |
| leading un-indexed axes dropped from the destination stride | 2 / 3037 | the `[None, index]` pair | 1 |
| `_lift` back to inferring from the Python type | 3 / 3037 | the three number-on-the-right cases | 2 |
| the old rank-1 restriction reinstated | **18** / 3037 | everything §3 opened | 2 |

The first row is a real result and is reported as it happened: the fault made the offsets and the
result shape disagree and the kernel **panicked** rather than failing gracefully. That is a caught
fault, but an uncountable one, so the remaining faults were chosen to stay in bounds. Reaching that
panic from a *legitimate* input is not possible: `mask_to_indices` validates the mask's shape
against the receiver before any offset is built, integer indices are bounds-checked as they are
read, and the value offset is bounded by the broadcast check — so every write is inside the flat
buffer by construction.

Two of the seven faults are ones a reasonable implementer would actually write (the left-aligned
broadcast and the dropped leading axes), and both were caught by exactly one door case and one
member case, which is the resolution the case set was built for rather than a lucky wide net.

### Counts after §2-§3

```
run.sh                 225 ok       (+2: the two new tests below)
compare.py             3037/3037    ops=122   (+35 cases over §1)
compare.py --self-test 13 x 11, 0 problems    (unchanged)
verify_schemas.py      4233/4233    (unchanged -- no new key)
```

The two new smoke tests are `test_index_put_takes_a_mask_a_matrix_and_a_number`, which pins the
shapes that used to be *refused* so a regression to the `scatter` delegation fails by name and not
only by value, and `test_index_put_writes_into_the_receiver_and_not_into_a_copy`, which is §4's
question asked of the op §2-§3 just rewrote.

---

## 4. `select.int` and `slice.Tensor` — the views question

> **Superseded by §6, which did the redesign this section specifies.** Everything §4 measures is
> still true and is the reason §6 is shaped the way it is; only its verdict — "not implemented" —
> has moved. Read §4 for *why the problem is where it is* and §6 for what was built.

**Verdict: a storage-model redesign, not a kernel change. Not implemented, and §6.4's refusal
stands.** The reason §6.4 gave for it is wrong, though, and the correct reason changes what the
redesign has to be — so the refusal is kept and its justification is replaced.

### What §6.4 said, and what is actually true

§6.4, and the docstrings that carried its wording, said:

> `aten.select.int` and `aten.slice.Tensor` return copies, not views — a candle tensor is a value.

**That is measurable, and it is false.** candle's `Tensor` is `Arc<Tensor_>` and `Tensor_` holds
`storage: Arc<RwLock<Storage>>`; `narrow` and `squeeze` clone that `Arc` and rebuild only the
`Layout` (`candle-core-0.11.0/src/tensor.rs:902` and `:880`). So `select.int` returns a tensor that
**aliases its input's buffer**.

Measured rather than read off the source. candle's `same_storage` is `pub(crate)`, but `slice_set`
consults it and bails with a distinguishable message, which makes it an oracle from outside the
crate. A temporary `_probe.same_storage` key was added to `aten.rs`, built, measured, and removed:

```
x vs select.int(x, 0, 1)         SHARED
x vs slice.Tensor(x, 0, 1, 3, 1) SHARED
x vs slice.Tensor(x, 0, 5, 2)    independent buffers   (step > 1 goes through index_select)
x vs alias(x)                    SHARED
x vs detach(x)                   SHARED
x vs unsqueeze(x, 0)             SHARED
x vs view(x, [5])                SHARED
x vs clone(x)                    independent buffers
x vs a freshly built tensor      independent buffers
```

And the write, on the same build:

```
v = select.int(x, 0, 1);  copy_(v, 3.0)
x      -> [0,0,0,0,0]      unchanged
v      -> 3.0              changed
v vs x -> no longer shared
```

So the sequence does not fail because the view is a copy. It fails because **`copy_` replaced the
view wrapper's tensor with a fresh buffer instead of writing into the one it was pointing at.**
The copy is not in `select.int`; it is in `PyTensorBase::replace_with`.

This distinction is the whole reason to write §4 down. "Teach candle about views" is not the work
— candle already does views. The work is a write path, and that is somewhere else entirely.

### Why that makes it a redesign

**`replace_with` is the write primitive for every in-place op — 26 call sites.** It is defined as
"swap the wrapper's tensor", and its doc comment already says the consequence out loud: an alias
taken before the call does not see the write. `fill_`, `zero_`, `copy_`, `add_`, `relu_`,
`clamp_`, `div_`, `masked_fill_`, `uniform_`, `normal_`, `index_put_` and `set_`/`.data` all go
through it. Four things follow, and each of them is why the kernel-sized version of this fix would
be worse than the refusal:

1. **A partial fix is inconsistent in a way the refusal is not.** Making only `copy_` write
   through gives `x[0] = v` while `x[0].fill_(3)` still silently drops. Today the rule is uniform
   ("no in-place op is visible through an alias"), stated in one place, and the one path that
   would otherwise be silent refuses by name.

2. **Making writes observable creates a new divergence in the other direction.** `detach(x)`
   already shares storage with `x` — measured above — and that is harmless *only because nothing
   writes into storage*. The moment one op writes through, the ops that still swap become
   hazardous the other way: `y = detach(x); x.fill_(0)` moves `x` onto a fresh buffer and leaves
   `y` on the old one, where upstream has `y` see the fill. The divergence does not get closed by
   a partial write-through; it gets relocated and doubled.

3. **candle's public write surface does not cover the cases that would be needed.**
   `storage()`, `storage_mut()` and `same_storage()` are all `pub(crate)`. From outside the crate
   there are exactly two write-through paths:

   * `Tensor::slice_set(&self, src, dim, offset)`, which requires **both sides contiguous**,
     equal rank, matching shape off `dim` — and **refuses when the two share storage**, which is
     precisely what `x[0:2] = x[1:3]` is. A `select.int` along dim 0 is contiguous; along dim 1 it
     is not, so `x[:, 0] = v` is outside it. Building on `slice_set` alone would make some
     subscript writes work and leave the rest silently not working, which is the failure this
     refusal exists to avoid.
   * the `InplaceOp1/2/3` custom-op traits, which *do* receive the `&Layout` and so could write
     a strided view correctly. This is the real path, and it means the redesign is possible rather
     than blocked — but a custom op is written per dtype against `CpuStorage`, so it is a new
     component, not an edit.

4. **Most in-place kernels here cannot write through without being rewritten.** The dominant shape
   in `aten.rs` is `read_flat` -> compute into a `Vec` -> `write_flat`, which produces a new buffer
   by construction. That includes the `index_put_` §2-§3 just wrote. Write-through means
   re-expressing each of them as a masked write into an existing buffer.

There is also a modelling gap, though it is the smallest of the five: `PyTensorBase` has no notion
of "this wrapper is a view of that one". The *data* half is already there — the offset and strides
live in candle's `Layout` — so this is not the obstacle it would be against a value-typed tensor.
It matters for `.data`, `set_`, and anything that would need upstream's `_base`.

### The shape of the redesign, for whoever does it

Stated so that §4 is a decision to be taken rather than a wall:

1. Give `PyTensorBase` a **write-through mutation primitive** beside `replace_with` — something
   like `write_into(&self, region: &Layout, src: &Tensor)`, implemented as an `InplaceOp2` so it
   gets the layout and can handle a non-contiguous destination.
2. **Move all 26 in-place sites onto it**, in one change rather than incrementally. The
   inconsistency in point 1 above is a property of a partial migration, so a partial migration is
   the one thing not to do.
3. Then, and only then, delete `__setitem__`'s basic-index refusal branch.

`test_setitem_refuses_the_basic_index_write_rather_than_dropping_it` is the signal for step 3: it
asserts the **probe** — that the write through `select.int` does not reach the base — rather than
the refusal alone. When step 2 lands, that test goes red on the assertion and points at the branch
to delete. It is deliberately built that way and it is still green, which is the honest report of
where this stands.

### What did change here

Nothing in `aten.rs`. Three pieces of prose that gave the wrong reason were corrected, because a
wrong reason in a live docstring is what sends the next person to build the wrong fix:

* `bootstrap.py`'s `__setitem__` docstring, which said select and slice "return a copy, because a
  candle tensor is a value";
* the `NotImplementedError` message that branch raises, which said "this shim's select and slice
  return copies rather than views" — it now says the narrowing aliases correctly and the
  write-through is what is missing;
* `test_setitem_refuses_the_basic_index_write_rather_than_dropping_it`'s docstring and its
  assertion message.

The assertions themselves are unchanged, so the test still fails exactly when mutable views arrive.

---

## 5. Where this landed

|  | before | after |
|---|---|---|
| `run.sh` | 223 ok | **225 ok** |
| `compare.py` | 2971 / 2971, ops=121 | **3037 / 3037**, ops=**122** |
| `compare.py --self-test` | 13 comparators × 11 modes, 0 problems | unchanged |
| `verify_schemas.py` | 4231 / 4231 | **4233 / 4233** |
| `aten.ge.Tensor` | resolves, refuses | kernel |
| `index_put_` bool mask | refuses (`c_error` case) | kernel, cased against upstream |
| `index_put_` rank > 1 | refuses | kernel, cased against upstream |
| mutable views | refuses by name | **still refuses, with the correct reason** |

Golden cases by key: `aten.ge.Tensor` 31 (new), `aten.index_put_.default` 11 -> 45,
`aten.fill_.Tensor` 13 -> 15. Of the new cases, 21 go through a tensor **member** rather than
through `_aten_dispatch`.

Standing check (docs/verification/DOCWATCH.md) on the "after" column's still-current, non-round-scoped facts —
not the round's own gate counts above, which are a historical snapshot per the house style
docs/verification/AUDIT.md confirms elsewhere:
<!-- DOCWATCH: op-implemented aten.ge.Tensor -->
<!-- DOCWATCH: op-implemented aten.index_put_.default -->

Sabotage totals, all measured by injecting the fault, rebuilding, and counting — never by reading
a green run as proof:

| fault | golden failed |
|---|---:|
| `ge.Tensor` computed with `Cmp::Gt` | 30 |
| `ge.Tensor` arm deleted | 31 |
| mask offsets drop the axis stride | 2 |
| `values` broadcast left-aligned | 2 |
| negative indices not wrapped | 2 |
| leading un-indexed axes dropped | 2 |
| `_lift` back to the Python type | 3 |
| the old rank-1 restriction reinstated | 18 |
| a mask forced to consume one axis | harness panicked (uncountable, but caught) |

One case was found by that pass to be incapable of failing (`x[:] = 3.0` through `fill_`, which
takes its dtype from the receiver on both sides regardless of the lift rule) and a discriminating
one was added beside it rather than the note being left to imply a guarantee it did not give.

---

## 6. The write-through primitive — §4's redesign, built

Baseline for this section is §5's landing: `run.sh` 225 ok, `compare.py` 3037/3037 ops=122,
`verify_schemas.py` 4233/4233, and `test_setitem_refuses_the_basic_index_write_rather_than_dropping_it`
green because no write reached a base.

**The change is one method.** `PyTensorBase::write_into` writes a computed replacement into the
buffer the receiver already points at, through the receiver's own `Layout`, and every in-place
kernel calls it instead of `replace_with`. Nothing about the kernels' arithmetic moved.

### 6.1 Why the kernels did not have to be rewritten

§4 point 4 predicted the opposite:

> Most in-place kernels here cannot write through without being rewritten. The dominant shape in
> `aten.rs` is `read_flat` -> compute into a `Vec` -> `write_flat`, which produces a new buffer by
> construction. Write-through means re-expressing each of them as a masked write into an existing
> buffer.

**That was wrong, and it was wrong in a way worth naming, because it is what made the job look
big.** A kernel that produces "a new buffer with the receiver's shape and the receiver's dtype"
is not an obstacle to write-through; it is *exactly* the input write-through needs. The
row-major values of that buffer are the values the destination's layout should receive, position
for position. So the primitive is:

```
write_into(dest, src):     for the i-th position of dest in row-major order,
                           dest_storage[layout.offset_of(i)] = src_flat[i]
```

and every one of the twelve kernels already computed `src_flat`. The migration is one line each.
The `read_flat -> Vec -> write_flat` shape is not a problem to solve; it is the shape that made
this a one-line change per call site instead of twelve rewrites.

**The count in §4 is also off, and the corrected one is worth having**: §4 says "26 in-place
sites", and `replace_with` had **13** callers — eleven in `aten.rs` (twelve op keys, since
`fill_.Scalar` and `fill_.Tensor` share a kernel) and two in `tensor.rs`. All eleven `aten.rs`
callers moved to `write_back`; the two in `tensor.rs` stayed, deliberately (§6.6). So the migration
is complete by the only definition that matters — **no in-place op still swaps a wrapper, and none
still refuses** — and the number to check against in future is 11 call sites over 12 keys.

What *is* new is the contract, and it is checked on every call rather than assumed:

| checked | why |
|---|---|
| `src.dims() == dest.dims()` | an in-place op changes values, not shape |
| `src.dtype() == dest.dtype()` (candle) | in-place cannot widen; the kernels already cast |
| `src.tag == dest.tag` (torch) | a write must not attach or drop the `bool` tag (BOOL.md §6.3) |

All three raise with the op's name and the words "internal error". They were not decoration: the
first run of the migrated build was 3037/3037 green, and that is a *result* — it says no in-place
kernel was quietly returning a differently shaped or differently tagged receiver, which nothing
before had asked.

### 6.2 What was rejected, and why `InplaceOp1` and not `InplaceOp2`

§4 named `InplaceOp1/2/3` as the viable route and called it a new component. It is the right
route; the arity is not free.

* **`Tensor::slice_set`** — rejected, as §4 already argued: both sides must be contiguous, and it
  *refuses a pair that shares storage*, which is precisely `x[0:2] = x[1:3]`.
* **`Tensor::inplace_op2`** — rejected, and this is the part §4 could not have known without
  reading its body:

  ```rust
  pub fn inplace_op2<C: InplaceOp2>(&self, rhs: &Self, c: &C) -> Result<()> {
      self.storage_mut().inplace_op2(self.layout(), &rhs.storage(), rhs.layout(), c)
  }
  ```

  `storage_mut()` takes the write lock on `self`'s `RwLock`; `rhs.storage()` then takes the read
  lock on `rhs`'s. **When the two operands alias, that is the same `RwLock`, and a write-then-read
  on one thread is a deadlock, not an error.** Aliasing operands are not exotic here — they are
  the case this whole section exists for. So `inplace_op2` is unusable for the one thing it looks
  built for.
* **`Tensor::inplace_op1`** — taken. It locks one storage, and the source is read out into an
  owned `CpuStorage` *before* that lock is acquired.

Reading the source first is not merely a way to dodge the lock. It is what makes an overlapping
copy mean something: `x[0:1] = x[1:2]` reads row 1 and then writes row 0, which is upstream's
answer (measured: `[[3,4,5],[3,4,5]]`). A streaming copy would read values it had already
overwritten.

**There is no `unsafe` anywhere in this change.** candle holds storage in `Arc<RwLock<Storage>>`,
so aliasing-XOR-mutability is enforced at runtime by the lock rather than by a raw pointer. That
is also why `write_into` takes `&self` and not `&mut self`, and why `write_back` in `aten.rs` uses
`borrow()` rather than `borrow_mut()`: the mutation is candle's interior mutability, so the
Python-level `RefCell` borrow stays shared and a kernel may still be holding a read borrow when it
calls.

Two details of the walk, both in `tensor.rs::write_strided`:

* **Bounds are proved once, before the loop.** The furthest position reachable is
  `start_offset + Σ (dim-1)·stride`, since every index runs `0..dim` and candle's strides are
  `usize`. One check up front means no write in the loop can be out of range, and a wrong layout
  becomes an error rather than a corrupted neighbouring tensor.
* **Contiguous destinations take a `copy_from_slice`.** The odometer would give the same answer;
  the branch exists because it is the common case (`x[0]`, `x[1:3]`, and every whole-tensor
  in-place op), and it is covered by the same expectations as the strided cases.

### 6.3 The divergences that write-through *created*, and their fixes

This is the part §4 warned about — "the divergence does not get closed by a partial write-through;
it gets relocated and doubled" — and it is real. The instrument is a probe that asks, of upstream
and of the shim with one script: *does writing into the result of this op reach its input?*
Twenty-eight relationships, both sides:

```
                                 upstream      shim (before)  shim (after)
alias / detach / lift_fresh       SHARED        SHARED         SHARED
expand / permute / t / transpose  SHARED        SHARED         SHARED
select.int (dim 0 and dim 1)      SHARED        SHARED         SHARED
slice.Tensor step 1               SHARED        SHARED         SHARED
squeeze.dim / unsqueeze           SHARED        SHARED         SHARED
view.default / _unsafe_view       SHARED        SHARED         SHARED
split / split_with_sizes/ unbind  SHARED        SHARED         SHARED
contiguous (already contiguous)   SHARED        SHARED         SHARED
contiguous (of a strided input)   independent   independent    independent
clone                             independent   independent    independent
zeros_like / empty_like           independent   independent    independent
neg / abs / relu / masked_fill    independent   independent    independent
_to_copy (dtype change)           independent   independent    independent
_to_copy (nothing to convert)     independent   SHARED  <--    independent
slice.Tensor step 2               SHARED        independent <- independent
view.dtype                        SHARED        independent <- independent
```

The "shim (before)" column is not a reconstruction — the probe was run against a build of `HEAD`
before any of this, and against upstream, with the same script. Every "SHARED" in it was **inert**,
because no write reached storage; making one write go through turns all twenty-eight into
correctness questions at once, which is why the table is a smoke test
(`test_which_ops_share_storage_with_their_input_and_which_do_not`) and not a paragraph.

Two of them were not inert afterwards:

1. **`_to_copy.default` with nothing to convert aliased its input.** `to_device` and `fast_to`
   both return `self.clone()` when there is nothing to do, and a candle clone is an `Arc` clone.
   So `y = x.to(torch.float32)` on a float32 tensor handed back an alias, and `y.fill_(0)` would
   have zeroed `x` where upstream leaves it alone. **This is the sharpest thing found in the whole
   change**: it is a corruption, not a lost write, and nothing in the golden suite could have
   caught it — the harness compares values of results, and both sides' *results* were correct.
   Fixed with `Tensor::copy()` on the no-op path only; the dtype- and device-changing paths have
   already allocated.

2. **An expanded destination.** `expand` gives stride 0, so several logical positions share one
   storage element and a write is last-write-wins. Upstream's answer is a table, measured on
   2.13.0:

   | op | upstream |
   |---|---|
   | `fill_.Scalar`, `zero_` | writes |
   | `masked_fill_`, `index_put_` | writes, with a deprecation warning |
   | `fill_.Tensor`, `copy_`, `add_`, `relu_`, `clamp_`, `div_`, `uniform_`, `normal_` | **raises** |

   The `fill_` pair is the one that had to be measured rather than reasoned about: two overloads
   of one kernel, one permitted and one refused. Reproduced as a table in `aten.rs::write_back`,
   with the detection in `tensor.rs::has_internal_overlap` written to be exactly upstream's
   `c10::has_internal_overlap` — **including its conservatism**: dense is `No`, a stride of 0 on
   an axis longer than 1 is `Yes`, and anything else is permitted. A stricter test would refuse
   strided views upstream writes into happily, which is the divergence in the other direction.

### 6.4 The two that remain, and why candle cannot close them

| | upstream | here |
|---|---|---|
| `slice.Tensor`, step > 1 | a view; a write reaches the base | materialised; the write is lost |
| `view.dtype` | a view; a write reaches the base | materialised; the write is lost |

Both are **write-lost**, not corruption, and both are blocked by candle's storage model rather
than by its visibility rules:

* A **stepped view** needs a `Layout` whose stride is `step` over the *input's* storage.
  `Layout::new` is public, but the only public pairing of a storage with a layout is
  `Tensor::from_storage`, which is documented as contiguous-only and takes a
  `candle_core::Storage` that `Tensor::storage()` (`pub(crate)`) will not hand over. So this is
  behind the `pub(crate)` boundary, and closing it means a fork, a vendored candle, or an upstream
  PR adding a strided-view constructor — none of which this change may do.
* **`view.dtype`** needs a layout that reinterprets bytes. candle's `Layout` counts *elements* of
  a storage whose dtype is fixed by the `CpuStorage` variant. There is nothing to construct at any
  visibility; this one is not a boundary problem, it is a model difference.

Both are pinned three ways rather than described: an `expect="diverge"` golden case each (which
compares the base against upstream and **fails if either silently starts agreeing**), a smoke test
that asserts the shim's half on its own, and — for the step case — `__setitem__` refusing a
`step != 1` slice by name, so the door a caller actually writes through does not reach it.

### 6.5 The one gap that is a cost decision rather than a wall

Upstream refuses a `copy_` whose source and destination **partially** overlap:
`x[0:2].copy_(x[1:3])` raises *"some elements of the input tensor and the written-to tensor refer
to a single memory location"*. Disjoint views of one buffer are fine (`x[0:1].copy_(x[1:2])`
computes, and that is the shape `x[0:1] = x[1:2]` produces, so it has to keep working).

This shim reads the source out before it takes the destination's lock, so it computes a defined
answer where upstream raises. Reproducing the refusal means upstream's `get_overlap_status`, which
compares the two storages' **data pointers** — and candle's `storage()` is `pub(crate)`.

It is reachable without a fork, and the route is worth writing down because it is not obvious:
an `InplaceOp1` whose `cpu_fwd` does nothing but read `CpuStorage::as_ptr()` recovers storage
identity through a public API. The price is **one extra write-lock acquisition per in-place op
with a tensor operand**, on the dispatcher's hot path. Not paid here; recorded as an
`expect="torch_error"` golden case so it prints on every run.

### 6.6 What `replace_with` is for now

Two callers, both of them ones where **rebinding is the operation** and upstream rebinds too:

* `TensorBase.set_` — adopts a different storage, shape and possibly dtype; there is no existing
  buffer to write into, since the point is to leave it.
* `tensor.data = other` — upstream swaps the `TensorImpl`, so a view taken before the assignment
  does not follow it there either (docs/devices/DEVICE_ABS.md §4).

Its doc comment now says so, and says that anything meaning "the receiver's values change but the
receiver stays the same tensor" must not come there.

### 6.7 `__setitem__`'s basic-index branch

The refusal is gone and the walk is `__getitem__`'s, emitting the same keys with the same
arguments — an index that reads as `x[0, 1:3]` must narrow to the same view whether it is being
read or written.

**One measurement in §4-era prose was wrong and is corrected**: the docstring recorded
`x[0] = 3.0 -> [select.int, copy_.default]`. Re-measured on 2.13.0 it is
`[lift_fresh, select.int, fill_.Tensor]`. The rule is not "number on the right versus tensor on
the right"; it is upstream's `copy_to`, and it is about **shapes**:

```
sizes equal        -> copy_          x[0,1] = 9.0   (0-d destination, 0-d source)
else source is 0-d -> fill_          x[0]   = 3.0   ((4,) destination, 0-d source)
else               -> broadcast, copy_
```

`x[0] = 3.0` and `x[0,1] = 9.0` differ only in shape and land on different ops, which is why both
are cased. Nine `fill_`/`copy_` member cases pin the arms; the previous rule would have put five
of them on the wrong key.

### 6.8 `clamp_` on a `torch.bool` receiver, which the guards found

Not a write-through question, but the contract check found it and it is a real defect: the kernel
produced a `uint8` replacement for a `bool` receiver, and `replace_with` **retagged the receiver**
from `torch.bool` to `torch.uint8`. Upstream refuses — *"result type Long can't be cast to the
desired output type bool"*, and *"Float"* in place of *"Long"* when a bound is a float. So this was
computing where upstream refuses.

Refused at the door now, with upstream's wording; the tag check underneath stays as the structural
backstop. `uint8` — the dtype `bool` shares candle's `U8` storage with — still computes, and the
pair is cased together, because that is what makes the refusal a statement about the *tag* rather
than about the bytes (BOOL.md §5-B).

### 6.9 Sabotage

Eight faults, each injected into the source, rebuilt, installed, and counted. Never a green run
read as proof.

| injected fault | golden failed | smoke failed |
|---|---:|---:|
| **A** `write_back` reverts to `replace_with` (the pre-§6 behaviour) | **27** of 3075 | 5 |
| **B** `write_strided` ignores the layout's `start_offset` | 17 | 2 |
| **C** the contiguous fast path taken unconditionally | 10 | *aborted — see below* |
| **D** `has_internal_overlap` always false | 2 | 1 |
| **E** `_to_copy` with nothing to convert aliases its input again | 2 | 1 |
| **F** `__setitem__`'s `copy_to` rule loses its `fill_` arm | **0** | **0** |
| **G** `__setitem__` stops refusing a `step > 1` slice | 1 | 1 |
| **H** `write_into` stops checking the replacement's shape and tag | **0** | **0** |

**Fault A is the number that matters.** It restores exactly the behaviour this shim shipped
before, and it fails 27 golden cases and 5 smoke tests — where before §6 the same behaviour was
3037/3037 green. That is the measurement of how invisible the defect was, and it is why every one
of the new cases reads the base rather than the in-place op's return value.

Three of the eight need their result stated rather than tabulated:

* **C aborted the smoke run** with a `PanicException` (`range end index 6 out of range for slice of
  length 2`) at the first expanded destination, because forcing the fast path invalidates the
  bounds proof the fault also bypassed. `PanicException` derives from `BaseException`, so the
  runner's `except Exception` does not catch it and the run stops at test 14 of 229. Caught, but
  uncountable. Running the two relevant tests past it separately, both fail — so the honest row is
  "10 golden, and 2 smoke that the abort prevented from being counted".

* **F and H could not fail, and both were checked rather than assumed.**

  **F** — `copy_` broadcasts a 0-d source to exactly the values `fill_` writes, so the arm choice
  is not observable by value. It is not observable by error either: the overflow refusal that
  separates the two kernels (`fill_(float16, 1e6)` raises, `copy_` gives `inf`) never fires,
  because `_lift` narrows the number to a 0-d tensor before either op sees it — measured, both
  arms give `inf` and both agree with upstream. And the one instrument that would show the op
  *name*, the capture facility, **refuses to record any region containing an in-place op**. So the
  distinction is carried because it is upstream's measured lowering, not because anything here
  guards it, and the case notes say so instead of letting the op key imply otherwise.

  **H** — the contract checks are unreachable, which is itself the measured result: every in-place
  kernel already broadcasts into the receiver's shape and casts into its dtype, so candle refuses
  first on every input that would reach them (`copy_((2,),(2,2))`, `add_((2,1),(2,2))`,
  `masked_fill_((4,1), mask (4,2))` all stop at `broadcast_as`). The one input that *did* reach
  one — `bool.clamp_(0, 5)`, §6.8 — now refuses at the door. A test for these would have to be a
  kernel that violates the contract, and the public API cannot produce one.

**One case was found by this pass to be incapable of failing and was given an instrument rather
than a note**: fault E — `_to_copy` aliasing its input — was invisible at first, 3071/3071 and 229
smoke tests green. It is the *sharpest* defect in the change (a corruption, not a lost write), and
nothing could catch it because every existing case compares the op's **result**, which was
correct. Two golden cases that write into the result and read the **input**, plus a smoke test that
asserts the whole 28-row aliasing table in both directions, now fail on it.

### 6.10 Where §6 landed

|  | after §5 | after §6 |
|---|---|---|
| `run.sh` | 225 ok | **229 ok** |
| `compare.py` | 3037 / 3037, ops=122 | **3075 / 3075**, ops=122 |
| `compare.py --self-test` | 13 × 11, 0 problems | unchanged |
| `verify_schemas.py` | 4233 / 4233 | unchanged |
| in-place ops visible through a view | none | **all twelve** |
| `x[0] = v`, `x[:,1] = v`, `x[1:3] = v` | refused by name | kernel |
| aliasing relationships agreeing with upstream | 25 of 28 | **26 of 28** |
| SmolLM2-135M float32 prefill | — | **bit-identical logits** |

The prefill check is the one that says this is not an optimisation with a tail: an aliasing change
that altered a model result would be a bug, and the sha256 of all 245 760 float32 logits is
unchanged.

---

## 7. `index_put_(accumulate=True)` — the refusal §2-§3 left, closed

§2-§3's table ends with two rows where upstream computes and this shim refuses. This closes the
second of them. It was left twice on purpose and both reasons were about *documents*, not about
the code: §2-§3 had not measured the flag, and docs/training/BACKWARD.md §4.5 named it as "one line" while
declining to write it because the refusal it would make stale lived in a file that round could not
edit. Done as its own round, with its own measurement and its own sabotage pass, it is fine.

### 7.1 It is one line, and two things about that line are not the obvious spelling

The walk was already general — the mask lowering, the spliced result shape, the right-aligned
broadcast, the negative-index wrap, the dtype check and the empty-write shortcut are all shared,
and none of them changed. Only the assignment differs:

```rust
//  before                       after
o[dest] = s[src];            o[dest] = if accumulate { narrow(o[dest] + s[src]) } else { s[src] };
```

**`narrow` is the part that is not obvious.** `read_flat` hands every floating dtype over as `f64`,
so `o[dest] + s[src]` computed there and narrowed once by `write_flat` at the end is what you would
write, and it is wrong: upstream's kernel is `*dst += *src` in C++'s `scalar_t`, so the running sum
is rounded to the receiver's dtype at every step. Measured on 2.13.0, accumulating three
`bfloat16` values into one slot:

```
index_put_(zeros(1, bf16), [[0,0,0]], [1.0, 0.005, 0.005], accumulate=True)
  upstream                       1.015625
  accumulated in f64, narrowed   1.0078125
```

`float_narrower` already exists for exactly this — it is `div_floor`/`div_trunc`'s, with its own
doc comment making the same argument for the same reason — so the fix is to call it, not to write
a second one. Separating values were found for `bfloat16`, `float16` and `float32`; `float64` has
nothing to separate.

**`torch.bool` is the second one.** `*dst += *src` on a C++ `bool` integer-promotes and converts
back, so accumulating `True` onto `True` is `True` — a logical or. Writing `o + s` in the `i64`
walk would leave a **2** in the byte, which is precisely what the `bool` tag's invariant
(docs/numerics/BOOL.md §6.3: `boolean()` is the only constructor that may attach it, and the bytes are 0 or
1) forbids. Measured: `zeros(4, bool)` accumulated at `[0, 0, 1]` with three `True` gives
`[True, True, False, False]`.

Integer dtypes wrap at the storage width, which the `i64` walk gets for free from `write_flat`'s
cast: `uint8` `200 + 100 + 100` is **144** upstream and 144 here.

### 7.2 What was measured, against 2.13.0, before anything was written

| | upstream | here |
|---|---|---|
| repeated index | the sum (`zeros(5)` at `[0,2,2,4]` with `[1,2,3,4]` is `[1,0,5,0,4]`) | same |
| a non-zero receiver | added onto, not replaced | same |
| accumulation dtype | the receiver's (`bf16` 1.0+0.005+0.005 is `1.015625`) | same, via `float_narrower` |
| `torch.bool` | logical or | same |
| `uint8` overflow | wraps (`144`) | same |
| bool / `uint8` mask | same `mask_to_indices` lowering, accumulating | same |
| `[None, t]` with a repeated column | each contribution lands | same |
| negative index | wraps, then accumulates | same |
| 0-d `values` | broadcast, added once per named position | same |
| empty index / all-false mask | writes nothing, returns `self` | same |
| dtype mismatch | refuses, same message | same |
| `values` that does not fit | refuses, same message | same |
| two index tensors | computes | **still refused by name** — not measured, not guessed |

The one remaining refusal is unchanged and deliberate.

### 7.3 There is no member or `torch.`-level spelling to pair the cases with

Every other kernel in this document was cased twice — once at the door (`_aten_dispatch`) and once
through the Python name — because golden compares by dispatch key and is structurally blind to a
missing name. **`accumulate` has no second door here, and that is a fact about the surface rather
than a gap in the cases:**

* `index_put_` is not in `methods.json`, so `tensor.index_put_(...)` does not resolve;
* `__setitem__` in `bootstrap.py` builds the call itself and always passes `accumulate=False` —
  there is no subscript syntax that means "add".

That is upstream's shape too (`x[i] += v` is a `index`/`add`/`index_put_` triple upstream, not an
accumulating put), so nothing is being routed around. Adding `index_put_` to `methods.json` would
give the member a real spelling and cost two entries in `verify_schemas.py`; it is named here
rather than done, because it is a surface change and this round is a correctness round.

### 7.4 The embedding rule: measured, and the recommendation

docs/training/BACKWARD.md §4.5 says the one-hot composition is used *because* this was refused, and asks
whether the rule should switch. It should, and the memory is the smaller of the two reasons.

Both compositions were run in this shim on the same inputs — the one-hot
(`zeros[vocab,count]` → `scatter.src` → `mm`) exactly as `tape.rs` builds it, and the
`index_put_(accumulate=True)` scatter-add:

```
float32, 11 rows x 4 wide, 23 tokens, random gradients
    max |one-hot - index_put_|   0.0                      identical

bfloat16, the separating case (three contributions into one row: 1.0, 0.005, 0.005)
    upstream embedding_dense_backward   1.015625
    index_put_(accumulate=True)         1.015625     <- upstream's answer
    one-hot -> mm                       1.0078125    <- not upstream's answer
```

**At `float32` the two agree exactly, so switching cannot move a `float32` gradient.** At
`bfloat16` they do not, and the accumulating scatter is the one that matches upstream: upstream's
`embedding_dense_backward` accumulates in the storage dtype, and the one-hot's summing is done by
a **matmul**, which accumulates in `float32` (the GEMM accumulation-dtype rule docs/architectures/ARCH.md
landed). The one-hot is not merely more expensive at reduced precision — it is answering a
different question.

Two things a switch has to carry, both cheap and both already visible in the current rule:

* **`padding_idx`.** The one-hot zeroes the *column* of the one-hot. An accumulating scatter has
  to zero the corresponding rows of the flattened gradient instead — one `ne.Scalar` and one
  `where`, the same two ops the rule already builds, applied to `flat_grad` rather than to `hot`.
* **the zero buffer is `[vocab, width]`, not `[vocab, tokens]`** — which is the 200 MB.

`tape.rs` is not this round's file and the change is not made here; this is the measurement the
recommendation rests on. The comment in `tape.rs`'s `aten.embedding.default` arm ("**that is
refused by this shim**") and docs/training/BACKWARD.md §4.5's costs table are both stale as of this
section, and are left for the agent that owns them rather than edited across a boundary.

### 7.5 Sabotage

Each fault injected into the source, rebuilt, and counted. The accumulate cases are 17 of the
`aten.index_put_.default` key's 61.

| injected fault | golden failed | smoke failed |
|---|---:|---:|
| `accumulate` ignored — the walk always overwrites | **13** | 1 |
| the sum accumulated in `f64` and narrowed once at the end | **3** | 1 |
| `torch.bool` accumulates with `+` instead of `\|` | **1** | 1 |
| the receiver is zeroed before accumulating | **4** | 1 |
| the flag is inverted | **16** | 2 |

The inverted flag is the row that says both arms are pinned: it fails the *non*-accumulate cases
too, including one that goes through the member (`x[idx] = v` with a repeated index) and one that
goes through a strided view.

**What could not fail, and why that is right.**

* The bool case does not fail on "accumulate ignored". Its receiver starts all-`False` and its
  values are all `True`, so a write and an or give the same answer — it is aimed at the `+`-versus-
  `|` fault and only at that one, and it is the only case that can see it. Reading the result *as
  bool* cannot see it either, because 2 is truthy; the case casts to `int64` and compares the ints.
* The `_bit_exact` cases are the only ones that can see the accumulation dtype. `dtypes.py` gives
  `bfloat16` `atol=6e-2` and the effect is 0.0078, so a tolerant comparator would pass an
  `f64`-accumulating kernel **for every input**, not merely for easy ones. This is the second kind
  of hole docs/training/LOSS.md §5.4 named, and `_bit_exact` is the instrument that already existed for it.
* The empty-index case does not fail on "the receiver is zeroed", because the zeroing was injected
  after the `total == 0` early return — which is what that case exists to guard, and it is the only
  case that guards it.
* The two `both_error` cases (dtype mismatch, broadcast mismatch) fail on no fault above. They are
  there to say that setting the flag does not enable promotion or relax the broadcast, i.e. they
  pin checks that run *before* the walk; a fault in the walk cannot reach them.

**One case pins behaviour this change did not write.** The `uint8` wrap is candle's `i64 -> u8`
cast, not the `wrapping_add` in the walk — the `i64` accumulator cannot overflow on these values.
It is cased because it is upstream's answer and nothing else in the suite asks for it, not because
a plausible fault in this kernel would move it.

### 7.6 Where §7 landed

|  | before | after |
|---|---|---|
| `run.sh` | 302 ok | **303 ok** |
| `compare.py` | 7447 / 7447, ops=166 | **7463 / 7463**, ops=166 |
| `compare.py --self-test` | 19 × 11, 0 problems | unchanged |
| `verify_schemas.py` | 4475 / 4475 | unchanged (no new key) |
| `index_put_(accumulate=True)` | refuses (`c_error` case) | kernel, 17 cases against upstream |

`aten.index_put_.default` goes from 45 cases to 61 — the `c_error` case that recorded the refusal
is gone, replaced by the seventeen that record the behaviour.

The prefill digests do not move, and they could not: `index_put_` is not on an inference forward at
all, and the non-accumulating arm's assignment is byte-for-byte the statement it was. Re-measured
anyway on the final artefact, at every length docs/numerics/SEQLEN.md §1.3 and docs/training/TRAIN.md §6 record —
all nine unchanged (`f32` S=6/32/128/512/1024 and `bf16` S=6/32/128/512).

The counts above are §7's own delta. Three more kernels landed in the same session, in
docs/numerics/SCALAR.md §8, and its §8.7 has the round's totals: **304 ok, 7685/7685 over 168 ops,
4479/4479 schemas**, both sweeps 26/26.
