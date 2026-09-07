# REPEAT — the last three blocked architectures, and a fourth that was never ours

Worktree `work/repeat` on develop `da9e3bf`. Territory: `rust/torch_c/src/aten.rs`,
`bootstrap.py`, `device.rs`, `capture.rs`, `methods.json`, `overloads.json`,
`tools/golden/cases.py`, `rust/torch_c/pytests/test_repeat.py`, plus the two
inversions §7 lists.

## 0. What was blocked, and what each of them actually was

`docs/architectures/ARCH300.md` left 297 architectures with 293 forwarding and four stopped.
The brief was "three ours, one not". That held, but **none of the three was
what its work item said it was**, and the shapes are different enough to be
worth keeping apart:

| wall | what the work item said | what it was |
|---|---|---|
| `TensorBase.where` (`led`, `longformer`) | "`methods.json` simply has no row" | **a row that table cannot express.** The receiver is the aten schema's *second* argument, and that table binds it into the first. §1 |
| `repeat_interleave.Tensor` (`fastspeech2_conformer`) | a kernel plus three out-of-territory files | exactly that, and all four landed together. §2, §3 |
| `sam3_lite_text_text_model` | "not our gap -- upstream refuses too" | still not our gap, and it now **forwards on both sides**. §5 |

**All four architectures forward.** Two of them (`led`, `longformer`) needed a
second thing after `where` fell, which §4 is; that one was not on anybody's
list and was found by running the sweep rather than by reading one.

Counted the way `CLAUDE.md` §5.3 asks — split rather than totalled:

| | this round |
|---|---|
| kernels added | **1** (`aten.repeat_interleave.Tensor`) |
| bindings added (no new kernel) | **1** (`TensorBase.where`) |
| argument-form rules generalised | **1** (the `SymInt` rule, per element of an int list) |
| defects fixed | **1** (§4.2 — the fast path answered a wrong *value* where the slow path raised) |
| tests added | 14 in `test_repeat.py` |
| tests **inverted** | 2 (§7) |
| golden cases added | 20, all on the new kernel |
| documents corrected | 2 (`LAST7.md` §5.2, `BIND5.md` §4.1's diagnosis) |
| deletions | 0 |

<!-- DOCWATCH: op-implemented aten.repeat_interleave.Tensor -->

---

## 1. `TensorBase.where` — the row would have been wrong, not merely absent

`docs/bindings/BIND5.md` §4.1 measured the wall correctly and diagnosed it in one line:
*"`methods.json` has no `where` row, so `TensorBase.where` falls through to the
surface stub."* Checked rather than trusted, per the brief — and the check is
what mattered, because **the row it implies computes the wrong answer.**

Upstream, measured on torch 2.13.0 in a separate process:

```text
x = [[1, 2], [3, 4]]   y = [[10, 20], [30, 40]]   c = [[T, F], [F, T]]

x.where(c, y)        ->  [[1., 20.], [30., 4.]]
torch.where(c, x, y) ->  [[1., 20.], [30., 4.]]      identical
```

So `Tensor.where(condition, other)` is `torch.where(condition, self, other)` —
**the receiver is the `x` branch, not the condition.** The aten schema is

```text
aten::where.self(Tensor condition, Tensor self, Tensor other)
                 ^ argument 0       ^ the receiver goes HERE
```

and `methods.json`'s machine binds the receiver into argument **0** and only
argument 0: `_Overloads(self_bound=True)` passes it as `args[0]` and every
positional count in `resolve` skips exactly one. A `where` row in that table
therefore computes `torch.where(x, c, y)` — the **right shape**, the **right
dtype**, and the two branches swapped. Nothing about that is visible to a shape
check, and it is invisible to any value test whose two branches broadcast the
same, which is why the golden and unit cases here use disjoint value ranges and
assert the wrong answer as a *non*-answer.

So it is Python-level surface, `_install_tensor_where`, alongside `softmax`,
`chunk`, `index_put_` and `to` — the group `methods.json`'s own `_README`
already describes as "upstream's binding for these is not a plain overload set
either". Two arms, chosen the way upstream's parser chooses:

```text
other is a Tensor   ->  aten.where.self(condition, self, other)
other is a Number   ->  aten.where.ScalarOther(condition, self, other)
anything else       ->  TypeError naming BOTH overloads, as upstream's does
```

`.ScalarSelf` and `.Scalar` are unreachable through this door by construction —
the receiver is always a Tensor — and stay reachable through `torch.where`.

**The kernels really were all there.** `docs/bindings/BINDINGS.md` was once told "`mish`
just needs a binding" and found the kernel gone, so this was measured rather
than inherited: all five of `aten.where.self`, `.default`, `.Scalar`,
`.ScalarSelf` and `.ScalarOther` are in `_aten_implemented()` and none of them
was touched. `test_the_five_where_overloads_are_untouched_by_the_new_method`
asserts that, so a future round cannot mistake this for a kernel change.

`test_tensor_where_is_not_a_methods_json_row_and_the_table_still_has_none`
asserts the **absence** of the row with the reason in its message, so somebody
tidying the tables cannot add it back without reading why.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_tensor_where present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_repeat.py test_tensor_where_takes_the_receiver_as_the_true_branch_not_the_condition present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_repeat.py test_tensor_where_is_not_a_methods_json_row_and_the_table_still_has_none present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json where present -->

---

## 2. `repeat_interleave.Tensor` — the op does not touch the data it repeats

`docs/kernels/LAST7.md` §5.2 sized this precisely and deliberately did not land it. Its
sizing held on every point. One thing it did not say, and which is the easiest
part of this op to get wrong from the name:

**`aten::repeat_interleave.Tensor(Tensor repeats, *, SymInt? output_size=None)`
never sees the tensor being repeated.** Its only argument is the `repeats`
vector, and its answer is the **index vector** a following `index_select`
gathers with. A `TorchDispatchMode` logger on torch 2.13.0:

```text
torch.repeat_interleave(x, tensor([2, 0, 3]), 0)
    aten.repeat_interleave.Tensor  (3,)
    aten.index_select.default      (3, 2), 0, (5,)

torch.ops.aten.repeat_interleave.Tensor(tensor([2, 0, 3]))
    ->  tensor([0, 0, 2, 2, 2])
```

`docs/kernels/LAST7.md` §5.2's correction to `docs/bindings/BIND4.md` was right and is why this
round is small: `aten.index_select.default` **is** implemented here, so only the
first of the two was missing.

Measured against upstream and reproduced exactly:

| | upstream | here |
|---|---|---|
| a non-uniform `repeats`, including zeros and an empty vector | — | **identical values** |
| output dtype | the *repeats* dtype (`int32` in → `int32` out) | identical |
| a 2-D `repeats` | `repeat_interleave only accept 1D vector as repeat` | identical |
| a negative element | `repeats can not be negative` | identical |
| a non-`Long`/`Int` dtype | `"repeat_interleave_cpu" not implemented for 'Float'` | identical |
| a disagreeing `output_size` | `allocated size does not match required size` | identical |

That last message is upstream's and it is about the *allocation* rather than
about the argument, because upstream allocates to `output_size` and then finds
the fill overran. It is transcribed as it is rather than improved, so a caller
grepping for it has upstream's string.

### 2.1 A constant `repeats` cannot test any of this

`[2, 2, 2]` is exactly the vector that a kernel ignoring the values and
multiplying the length by one count would also answer correctly — and it is the
vector that cannot separate this overload from the **scalar** one
(`repeat_interleave.self_int`), which is a different code path here and
upstream. It is the same blindness `docs/kernels/LAST7.md` §1.1 avoided by never
letting `step == size` in `unfold`.

So every value case in `repeat_interleave_tensor_cases` and in
`test_repeat.py` is non-uniform, two of them contain a **zero** (the position an
off-by-one emits anyway), and one pair is a vector and its reverse — which no
length-only arithmetic separates.

### 2.2 The composite had to be taught, not bypassed

`torch.repeat_interleave` is a hand-written composite installed with
`setattr(varfns, ...)` **after** the table, so it wins, and it raised on a
tensor `repeats` before any dispatch happened. That is `docs/kernels/LAST7.md` §5.2's
first bullet and it held. The new arm is transcribed from upstream's trace:

```text
dim given, repeats (n,)     repeat_interleave.Tensor, index_select
dim given, repeats (1,)     view [1], expand [n], THEN the two above
dim = None                  view [-1] first, then dim = 0
```

**The one-element broadcast is in the composite, above the kernel, because that
is where upstream does it.** Pushing it into the kernel would make a legitimate
`repeat_interleave(x, tensor([2]), 0)` on a one-row input indistinguishable
from a length error. The length check is the composite's too, and it fires
before the kernel, so a wrong-length `repeats` never becomes an index vector
that `index_select` would then reject for a different reason. Its message names
`input.size(0)` *after* the flatten, which is how the `dim=None` case is
separable at all — a `(3, 2)` input with a length-3 repeats and no `dim` says
`input.size(0) = 6`, not 3.

The **one-argument spelling** (`torch.repeat_interleave(tensor([2, 0, 3]))`) is
now the kernel and nothing else. It was unreachable before this round in a way
the old refusal hid: the composite's `repeats` parameter was *required*, so the
call died in Python before reaching the arm that claimed to refuse it. It takes
a sentinel default now, and a `dim` beside a lone repeats vector is a
combination error, as it is upstream.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs repeat_interleave_tensor present -->
<!-- DOCWATCH: symbol-in-file tools/golden/cases.py repeat_interleave_tensor_cases present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_repeat.py test_repeat_interleave_tensor_answers_the_index_vector_not_the_data present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_repeat.py test_the_composite_reaches_the_kernel_on_the_spelling_the_model_uses present -->

---

## 3. The two lists, which are why this is a four-file change

The output *length* is `repeats.sum()`, a number that exists only in the
tensor's bytes. That single fact puts the op on two lists at once, and neither
is optional.

**`device.rs::MPS_HOST_READBACK_OPS` (86 → 87).** Computing on the host while
the label says `mps` is the divergence `docs/devices/VULKAN3.md` §3 caught for
`nonzero`: a correct answer off the CPU with `mps` written on it.
`test_shim.py::test_the_mps_readback_list_is_what_the_kernels_actually_do`
re-derives the list from `aten.rs`'s own bodies and would have gone red on the
kernel alone.

**The `to_vec1` is written in the dispatched function, not behind a helper.**
That derivation follows helper calls **one level, by name**, and
`docs/architectures/VOICE3.md` found `var`/`std` nearly invisible to it for exactly that
reason. A `read_repeats_to_host()` helper here would have been tidier source and
a hole in the check.

**`capture.rs::DATA_DEPENDENT_SHAPE`.** The output *shape* is a function of
values, which is what that list refuses by name for `nonzero` one line above.
`docs/graph/CAPTURE.md` is the contract: a recorded node whose output shape is not
implied by the guards replays unsoundly on any other input — the graph was built
for the first one. So this joined the list **in the same change that gave it a
kernel**, rather than after somebody noticed a wrong replay. Asked of the
recorder rather than of the constant:

```text
_capture_begin([reps]); _aten_dispatch("aten.repeat_interleave.Tensor", reps)
_capture_reason()  ->  "aten.repeat_interleave.Tensor produces an output shape
                        that depends on tensor values; ..."
_capture_end(...)  ->  NotImplementedError
```

`index_select` is exercised in the same loop and **is** recorded, so a refusal
that fired for every op would not pass there.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs DATA_DEPENDENT_SHAPE present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs MPS_HOST_READBACK_OPS present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_repeat.py test_the_two_lists_the_kernel_had_to_join_are_both_asserted_here present -->

---

## 4. What was behind `where` — and it was not on any list

Closing `TensorBase.where` moved `led` and `longformer` **one call later**, into
the same borrowed function they had already been walking:

```text
modeling_longformer.py:862  (and modeling_led.py:506)
    chunked_value = padded_value.as_strided(size=chunked_value_size, stride=...)

TypeError: Tensor.as_strided(): no matching overload in torch._C shim for
           (size=tuple, stride=tuple)
```

`as_strided` was implemented, and both tuples were tuples of the right length.
Instrumenting the call rather than reading the message is what found it: one
element of `chunked_value_size` arrives as a **0-dim tensor**.

```text
size = (2, tensor(4), 12, 16)
              ^^^^^^^^^
```

This is upstream's `SymInt` rule — `docs/bindings/BIND5.md` §7's rule — applied **per
element of an int list** rather than to a scalar argument. Measured upstream:

```text
x.as_strided(size=(2, tensor(2)),    stride=(4, 1))  ->  works
x.as_strided(size=(2, tensor([2])),  stride=(4, 1))  ->  works
x.as_strided(size=(2, tensor(2.0)),  stride=(4, 1))  ->  TypeError
x.as_strided(size=(2, tensor([2,3])),stride=(4, 1))  ->  TypeError
x.as_strided(size=(2, tensor(True)), stride=(4, 1))  ->  RuntimeError
x.view((tensor(2), 6))                                ->  works
x.permute((tensor(1), 0))                             ->  works    (int[], not SymInt[])
```

The refusals are the measurement, not a detail: **whole-valued floats are still
refused**, and `bool` gets a *different exception class* because upstream reaches
it one layer further in, past the parser, at the scalar conversion. So the
predicate accepts `bool` and the coercion refuses it, in that order, and the two
classes fall out rather than being chosen — the same split `_symint_from_tensor`
already had for the scalar position.

`_coerce_symint_size_tensors` had this rule already, applied by hand inside
three composites (`zeros`, `ones`, `view`) by `docs/bindings/BIND4.md` §3 and corrected
by `docs/bindings/BIND5.md` §1. It is now in `_TypeChecker`'s int-list predicate and in
`resolve`'s coercion, which is where it belongs: the rule is upstream's
*parser's*, not any one op's.

### 4.1 What that rule does not do

It does not accept a multi-element tensor, a float tensor, a bare Python `bool`,
or a Tensor anywhere the schema does not say `int`/`SymInt`. `permute` is in the
tests beside `as_strided` and `view` because it is `int[]` rather than
`SymInt[]` — one spelling passing would not show the rule is on the element.

The message for the float and multi-element cases is this shim's "no matching
overload", not upstream's "failed to unpack the object at pos 2", because here
they are a *binding* failure and upstream reaches them inside a bound argument.
**The exception class is the claim; the wording is recorded as differing rather
than asserted.**

### 4.2 A defect, and it is `docs/bindings/BIND5.md` §7.2 re-firing one position over

`_compile_fast_path` generates a positional fast path that reproduces
`resolve`'s predicates **and its coercions**. Its own comment says why:

> *"It used to reproduce only `sized_int_list`'s, which was invisible while the
> predicates admitted nothing that needed the other one — the moment a scalar
> `int`/`SymInt` started accepting a single-element Tensor, this path handed the
> raw Tensor to the dispatcher and the slow path did not."*

The same trap fired again here, because teaching the slow path is the obvious
half. With only `resolve` taught:

```text
x.as_strided((2, tensor(True)), (4, 1))        ->  shape [2, 1]        WRONG
x.as_strided(size=(2, tensor(True)), stride=…) ->  RuntimeError        right
```

A **wrong value**, positionally, where the keyword spelling of the same call
raised — `bool` reached the Rust side and was unpacked as 1. It was caught by
the refusal test, not by any value test, which is the argument for writing the
refusals down: the shape `[2, 1]` is plausible and nothing else looks at it.
`_fast_symint_list_coerce` is the fix and both spellings are asserted.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _coerce_symint_list present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _fast_symint_list_coerce present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_repeat.py test_the_int_list_symint_rule_refuses_exactly_what_upstream_refuses present -->

---

## 5. `sam3_lite_text_text_model` — re-checked, and it has moved

`docs/bindings/ARGFORM.md` §3 and `docs/bindings/BIND5.md` both concluded this is **not our gap**:
upstream refuses `torch.embedding(Parameter, None, ...)` identically. Re-checked
on this tree rather than carried forward, and there are two answers, which is
why it is worth separating them.

**The call is still refused by both, identically in class.** Measured side by
side in separate processes:

```text
torch.embedding(w, None)
  upstream  TypeError: embedding(): argument 'indices' (position 2) must be
                       Tensor, not NoneType
  here      TypeError: torch.embedding(): no matching overload in torch._C
                       shim for (Tensor, NoneType)
```

Different words, same class, same refusal, and `torch.embedding` with real
indices agrees element-wise on both. So the diagnosis stands and closing it here
would be a **widening against upstream**, not a fix.

**But the architecture forwards now, on both sides.** `arch_sweep.py --one
sam3_lite_text_text_model` reports `status: ok` for the shim *and* for upstream
on this tree, so the sweep no longer reaches that call at all. That is a change
since `docs/bindings/BIND5.md` and it belongs in the count: the four remaining
architectures are no longer "three ours and one not ours", they are **four that
forward**, three of them because of this round and one of them for a reason that
was never in this repository.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_repeat.py test_sam3_lite_texts_embedding_call_is_refused_by_upstream_too present -->

---

## 6. The four architectures, measured with `arch_sweep.py --one`

```text
                          before                              after
led               forward -- TensorBase.where          FORWARD RUNS  (§1, then §4)
longformer        forward -- TensorBase.where          FORWARD RUNS  (§1, then §4)
fastspeech2_      forward -- repeat_interleave with    FORWARD RUNS  (§2, §3)
  conformer                  a tensor `repeats`
sam3_lite_text_   forward -- torch.embedding(_, None)  FORWARD RUNS  -- and upstream
  text_model                 (upstream refuses too)      does too; never ours (§5)
```

**4 of 4.** Coverage moves 293 → 297 of 297.

`led` and `longformer` each needed **two** things, and only one of them was on a
list. That is the part worth carrying forward: the work item named `where`
because `where` is where the traceback stopped, and a traceback names the first
wall, never the count.

---

## 7. The two inversions, and no pinned count moved

Nothing was weakened or deleted. Two tests asserted an absence this round closed
and both were inverted into **stronger** assertions:

1. `test_last7.py::test_repeat_interleave_with_a_tensor_repeats_is_still_refused_by_name`
   → `..._now_lands_in_all_four_files`. It does not merely check the values: it
   asserts membership in `MPS_HOST_READBACK_OPS` and the capture refusal by
   name, because those are the two halves `docs/kernels/LAST7.md` §5.2 said would go red
   and the two a value test cannot see. The `docs/kernels/LAST7.md` §7 DOCWATCH marker
   moved with it.
2. `test_shim.py::test_the_three_composites_that_opened_persimmon_and_cohere`
   asserted the tensor-`repeats` refusal inline. Inverted onto a **non-uniform**
   `repeats` in both `dim=0` and `dim=1` on a non-square input, per §2.1 — the
   `[2, 2, 2]` the scalar assertion two lines above already uses would have been
   the one vector that proves nothing.

**No pinned count moved.** `test_shim.py`'s `len(keys) == 364` is unchanged, and
that is the check rather than an omission: `TensorBase.where` is Python-level
surface (§1), not a `methods.json` row, so neither table grew. A `+1` here would
have meant somebody added the row this round exists to argue against.

`tools/golden/reach_allow.json` is untouched.

---

## 8. Verification

```text
suite            947 ok, 0 FAIL, EXIT=0        (baseline 929 ok on da9e3bf)
DOCWATCH         PASS -- 830/830               (before the markers in this file)
golden           11405/11405 cases, ops covered = 301, pending = 0
                                              (baseline 11385/11385, ops = 300)
arch_sweep       led, longformer, fastspeech2_conformer,
                 sam3_lite_text_text_model:  forward ok
```

Every value claim above was produced by running the op on real torch 2.13.0 in a
separate process with `PYTHONPATH` and `TORCH_USE_RTLD_GLOBAL` unset, and
comparing element by element — never by reading a shim result twice. Timings are
not reported: three other agents were running.

<!-- DOCWATCH: count golden_cases_total ge 11405 -->
<!-- DOCWATCH: count golden_cases_passed ge 11405 -->
<!-- DOCWATCH: count golden_ops_covered ge 301 -->
