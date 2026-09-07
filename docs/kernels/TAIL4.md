# `as_strided` stays refused, and `t_` was `rwkv`'s last wall

`docs/architectures/ARCH200.md` left 27 architectures blocked behind 16 operator names. This
round took six of them: `TensorBase.index_copy_` (`aria`, `aria_text`),
`aten.as_strided.default` (`longformer`, `led`), `TensorBase.t_` (`rwkv`),
`aten.round` (`fastspeech2_conformer`), `aten.logsumexp` (`granite_swa`), and
the item ARCH200 classified not as a gap but as a **backend limitation**,
`aten.embedding.default.unsupported` (`cpmant`).

That last one is wrong in the same way `MatMulUnexpectedStriding` was in
`docs/kernels/TAIL3.md` §1, and for the same kind of reason: the error names the wrong
operand.

A seventh op, `aten.logical_not.default`, was added to the round after the
fact: the `vmap` round landing in the same batch moved three nemotron ASR
encoders one line onto it, and `cpmant`'s *next* wall after §7 turned out to be
the same name. §10.

**Five of eight architectures now forward, two moved to a new wall, two are
still on `as_strided` — deliberately.**

---

## 1. The `as_strided` verdict: the refusal stood, and `docs/kernels/STRIDED.md` overturned it

> **Superseded, and left standing rather than rewritten.** `docs/kernels/STRIDED.md`
> implemented `aten.as_strided.default`. What follows was right about every
> fact it checked -- the single write door (§1.1), candle's fresh `TensorId`
> per shallow view, the address that is reused after a free (§1.2) -- and
> wrong about one thing, which is §1.3's sizing. It sized the fix as *"one
> field on `PyTensorBase` that survives aliasing, plus propagation through the
> view ops"*, and that would **not** have been enough: propagation runs
> forward from the mark, and an alias of the base taken *before* the
> `as_strided` call is exposed backward from it. `longformer`'s own `_chunk`
> calls `as_strided` on a `hidden_states` that is already a `.view(...)`.
>
> The fix was the structure this section rejected -- an address-keyed
> registry -- with the one addition that answers §1.2's objection: the entry
> **holds the storage alive**, so the address cannot be reused while the entry
> means anything, and `Drop` removes the entry before releasing the hold. That
> is not a leak relative to upstream, whose `as_strided` result keeps the
> base's storage alive by aliasing it.
>
> `pytests/test_strided.py` and `docs/kernels/STRIDED.md` §4 carry it. The refusal
> below is now a **narrowing**: the values are upstream's and only the writes
> upstream would have propagated are refused.


`docs/kernels/TAIL3.md` §6 refused `as_strided` because candle 0.11.0 exposes no
storage-sharing constructor. This round was asked the narrower question, which
that refusal does not answer: **can a *read-only* `as_strided` be made safe —
one that refuses the moment its result is written to, rather than one that hopes
nobody writes?** `docs/kernels/COMPLEX2.md` is the precedent for the standard: `tensor()`
refuses on non-`Dense` across ~400 call sites, and nullifying that refusal made
6 of 10 sampled ops compute silently. Unlikely is not the bar; unrepresentable
is.

**The answer is no, and the reason is two facts rather than an argument.**

### 1.1 There is exactly one write door, and it is in the right file

Every in-place op in `aten.rs` ends at `write_back`, `write_back` is the only
caller of `tensor::write_into`, and `write_into` is the only thing that writes
through a tensor's layout. Verified by counting, not by reading:

```text
aten.rs   .write_into(   1 occurrence   (inside fn write_back)
tensor.rs .write_into(   0 occurrences
```

So a guard placed in `write_back` would see every in-place write, and
`write_back` is in `aten.rs`, which was this round's file. That is the half that
works, and `test_tail4.py::test_the_write_door_is_single_which_is_the_
precondition_a_read_only_as_strided_would_need` pins it — the day `write_into`
acquires a second caller, the sizing below has to be re-read.

### 1.2 The marker such a guard would read cannot be reached from `aten.rs`

A guard needs to answer "is this receiver a materialised `as_strided` result,
or the base one was taken of". That is a per-tensor fact, and the only place a
per-tensor fact can live is a field on `PyTensorBase` — `tensor.rs`, which was
**not** this round's file. Every substitute reachable from `aten.rs` alone
fails, and each fails in a way that is worth writing down because it is the
same shape of failure:

* **candle's `TensorId`.** Public, `Hash + Eq`, monotonically issued and never
  reused, so a `HashSet<TensorId>` populated at `as_strided` time has no false
  positives. But **a shallow view gets a fresh id**: `Tensor` is
  `Arc<Tensor_>`, and `reshape`/`transpose`/`narrow` build a new `Tensor_` over
  the same storage. So

  ```python
  y = x.as_strided(...)   # registered
  z = y.view(-1)          # NOT registered -- new TensorId, same buffer
  z.fill_(0)              # writes into y's storage, guard never fires
  ```

  The guard would be defeated by the most ordinary thing a caller does next.

* **The storage address** (`storage_and_layout()`, which `capture.rs` already
  uses for `STORAGE_VERSIONS`). This one *does* survive aliasing — that is
  exactly why capture keys on it. But an address is **reused after the
  allocation is freed**, so a permanent poison set keyed on it starts refusing
  writes to unrelated tensors allocated later at the same address.
  `capture.rs`'s own comment says why that is safe there and not here: a trace
  holds a strong reference to every constant it stamped, so the storage cannot
  be freed while the stamp exists. An `as_strided` guard has no such
  reference — the whole point is that the view may be dropped while the base
  lives on.

There is a third fact that no guard on writes can address at all: upstream's
`as_strided` is a **view in both directions**. Writing to the *base* after
taking the view is visible through the view upstream and is not through a
materialised copy. Poisoning the base as well would close it, and it needs the
same unreachable marker.

### 1.3 So the sizing is `docs/kernels/TAIL3.md`'s, unchanged, and now with a
precondition attached

Closing it still means what §6 said: a stride-carrying tensor wrapper in this
crate that owns the `Arc<Storage>` and its own `Layout`, or a candle patch
exposing a storage-sharing constructor (`vendor/` already carries
`int8-candle-0.11.0-cpu.patch`, so the mechanism exists **— which is FALSE, corrected 2026-09-07.** The patch file is carried (now in `vendor/`, previously in `docs/`), but NOTHING APPLIES IT: `vendor/*.sh`, `vendor/*.py`, `Cargo.toml` and `build.rs` contain no patch step and no `[patch.crates-io]`. Carrying a diff is not a mechanism, and three documents used this sentence to argue that a candle fork would be cheap). What this round adds
is the smaller thing that would be *enough for a read-only view*: **one
`bool`/`Option<...>` field on `PyTensorBase` that survives aliasing**, plus
propagation through the view-producing ops. That is a `tensor.rs` change, and
it is a smaller change than either of §6's two, but it is not a `longformer`-
shaped task either and it was not this round's file.

`longformer` and `led` were therefore still blocked, on purpose --
**until `docs/kernels/STRIDED.md`**, which moved both past this op onto
`Tensor.new_zeros((tuple))`, an argument form. The `reach_allow.json` entry
that this paragraph described has been deleted, as its own text instructed,
and `test_the_as_strided_reach_allowlist_entry_was_removed_when_the_gap_closed`
asserts it stayed deleted.

<!-- DOCWATCH: op-implemented aten.as_strided.default -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json as_strided present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_the_as_strided_refusal_was_INVERTED_by_the_strided_round present -->

---

## 2. Alias or kernel — the split, counted honestly

`docs/architectures/ARCH200.md` and `ARCH100.md` both measured that missing *names* outnumber
missing *kernels* roughly 2:1 in this tail, so the first question per op was
"does a kernel already exist under another name". The answer here is not the
2:1 one, and the reason is instructive: **the two ops that looked most like
aliases are the two that needed the most measurement.**

| op | alias? | what actually had to be written |
|---|---|---|
| `index_copy_` / `index_copy` | **no — and it is `index_add_`'s neighbour** | a new kernel. Same *walk*, four different rules. §3 |
| `round` / `round_` (x2 overloads) | **no** | **new arithmetic**: candle's `Tensor::round` is the *other* rounding rule. §4 |
| `logsumexp` | **neither** | no new arithmetic at all — `amax`, `sub`, `exp`, `sum`, `log`, `add`, in candle. §5 |
| `t_` | **yes, for the values — no, for the write** | `t_default`'s rank rule, plus the first shape-changing in-place write path in the file. §6 |
| `embedding` (int32 index) | **not an op at all** | a widening cast and a dtype refusal: **three lines**. §7 |
| `logical_not` | **no — and `bitwise_not` is the trap** | one comparison, and the dtype rule that makes it not an alias. §10 |
| `as_strided` | **refused by name** | nothing this round; `docs/kernels/STRIDED.md` landed it. §1 |

**Counted honestly: two new kernel bodies, one genuinely new piece of
arithmetic (`nearbyint_ties_even`), one composition of existing candle
reductions, one new write path over an existing rule, one one-line comparison,
and one argument-form fix.** Nine `_aten_implemented()` keys, sixteen table
rows.

"Nine ops" would overstate it by a factor of three — the four `round` keys are
one function with two booleans, and the two `index_copy` keys are one function
with one. `test_tail4.py::test_the_only_new_arithmetic_this_round_is_ties_to_even`
asserts the sharing as source structure, so the claim in this table cannot rot
into prose.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs nearbyint_ties_even present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs index_copy_common present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_the_only_new_arithmetic_this_round_is_ties_to_even present -->

---

## 3. `index_copy_` sits beside `index_add_` and agrees with it about almost
nothing

This is the one the round would most easily have got wrong, and the repository
had already recorded the trap twice: `docs/architectures/DEMAND8.md` found that
`index_add_`'s negative indices do **not** wrap while `index_put_`'s do, and
`docs/kernels/TAIL3.md` §3 found `scatter_reduce`'s `sum` and `index_add_`'s
accumulating at different widths. Both times the first draft inherited the
neighbour's rule.

Measured on `index_copy_` itself, in the same process as `index_add_`:

```text
                        index_add_                    index_copy_
zeros(3), idx [1,1,1],  [0., 6., 0.]                  [0., 3., 0.]
  src [1.,2.,3.]        accumulates                   LAST WRITE WINS

int32 index             accepted                      RuntimeError:
                        (and golden-pinned)           "Expected a long tensor
                                                       for index, but got Int"

index -1                IndexError:                   IndexError:
                        "index out of range in self"  "index -1 is out of bounds
                        -- neither index nor extent    for dimension 0 with
                                                       size 3"

alpha                   keyword-only, exists          no alpha at all
```

**The `int32` row is the one that would have shipped.** `index_add_` accepts an
`int32` index and `tools/golden/cases.py` has a case pinning that it does, so
copying its dtype gate would have made `index_copy_` accept an index upstream
refuses — a widening, invisible to every test that passes a `torch.long` index,
which is every test anybody writes.

### 3.1 The seven checks are upstream's seven, in upstream's order

The order decides which of seven messages a caller sees, so it was transcribed
from `at::native::index_copy_` rather than reconstructed:

```text
1  dim range         IndexError, and with NO op prefix on the message
2  index.dim() < 2   "Index should have dimension 1 or 0 (got 2)"
3  index is Long     "Expected a long tensor for index, but got Float"
4  self/source dtype "self and source expected to have the same dtype"
5  scalar source     "When source is scalar, index should have one element"
   else rank match   "...their dimensionality must match..."
6  numIndices        "Number of indices (2) should be equal to
                      source.size(dim) (1)"
7  slice shapes      "Destination slice shape: 2 3 at dimension 0 and source
                      slice shape: 5 7 at dimension 0."
```

Two details of step 7 are upstream's own asymmetries and are reproduced rather
than tidied: the shapes print **space-separated** (`2 3`, not `[2, 3]`), and the
destination reports `dim` while the source always reports a literal `0`. When
the source is 0-d its slice shape prints **empty**, which is the two spaces in
`source slice shape:  at dimension 0.` — and that is a *reachable* message, not
a curiosity: `zeros(2,3).index_copy_(0, [1], tensor(5.))` produces it, while
`zeros(3).index_copy_(0, [1], tensor(5.))` is perfectly legal.

Upstream has an eighth check between 5 and 6 (`dim == 0 || dim < source.dim()`).
It is **not** reproduced, because it is unreachable: step 1 guarantees
`dim < self.dim()` and step 5 has already refused every case where `self.dim()`
and `source.dim()` differ and neither is zero.

### 3.2 candle's `scatter` is the kernel, and that was a constraint rather
than a preference

`index_add_` reads the index to the host with `read_flat` and walks the source
itself. Doing that here would put `index_copy_.default` and
`index_copy.default` on the set that
`test_the_mps_readback_list_is_what_the_kernels_actually_do` derives from these
bodies, and therefore oblige an edit to `device.rs` — **not this round's
file**. `Tensor::scatter` does the same job on the device: broadcast the 1-D
index across the source's shape along `dim`, and candle writes source elements
to the index's positions in index order, so last write wins, which is
upstream's answer for a repeated index.

Two consequences of borrowing candle's bounds check rather than writing one:

* The refusal arrives as `Error::InvalidIndex { index: usize }` with a negative
  index already reinterpreted as a huge `usize`, so it is cast back through
  `as i64` to print `-1` the way upstream does. It also arrives wrapped in
  `Error::WithBacktrace` whenever backtraces are enabled, which is invisible in
  a build where `RUST_BACKTRACE` is unset — so `invalid_index_of` peels the
  wrappers rather than matching the variant directly.
* candle's scatter **silently skips** an index equal to `i64::MAX` (its
  sentinel for "no write") where upstream raises. A caller would have to pass
  `9223372036854775807` as an index to reach it. It is a divergence and it is
  the only one.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_index_copy_OVERWRITES_where_index_add_ACCUMULATES present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_index_copy_REFUSES_an_int32_index_where_index_add_ACCEPTS_one present -->

---

## 4. `round` is banker's rounding, and that is the whole op

Upstream rounds **half to even**. On the half-integer grid, that and
round-half-away-from-zero disagree on five of nine:

```text
x            -3.5  -2.5  -1.5  -0.5   0.5   1.5   2.5   3.5   4.5
half to even -4.    -2.   -2.   -0.    0.    2.    2.    4.    4.
half away    -4.    -3.   -2.   -1.    1.    2.    3.    4.    5.
                     ^^    ..    ^^    ^^    ..    ^^    ..    ^^
```

**candle's `Tensor::round` is `f32::round`, which is the second row.** So "use
candle's round" is the plausible implementation and it is wrong on more than
half of the ties. A test on `0.3` and `0.7` agrees with both rules and would
not have seen it.

Rust has `f64::round_ties_even`, but only on the host, and a host round trip
would have put `round` on the readback list (§3.2). So the rule is written out
of candle primitives:

```text
f    = floor(x)
frac = x - f                   in [0, 1)  -- NaN when x is +-inf or NaN
inc  = (frac > 0.5) or (frac == 0.5 and f is odd)
out  = f + inc
```

`+-inf` and `NaN` fall out rather than being special-cased: `frac` is `NaN`
there, every comparison against `NaN` is false, so `inc` is zero and `out` is
whatever `floor` already made it.

### 4.1 The signed zero is not decoration

`nearbyint(-0.5)` is `-0.0` — `torch.signbit(torch.round(tensor([-0.5])))` is
`True`, the same property `floor`'s comment in `aten.rs` already records for
`floor(-0.0)`. And `f + inc` computes `-1.0 + 1.0`, which is `+0.0`. So a zero
result takes its sign from `x * 0`.

`affine(0.0, 0.0)` will **not** do, and this is the sort of thing that survives
a review: it computes `x * mul + add`, and `-0.0 + 0.0` is `+0.0`. It has to be
a multiply with nothing added. `tolist()` cannot see the difference either,
because `-0.0 == 0.0` in Python, which is why the test asserts through
`signbit`.

### 4.2 `round.decimals` scales at the storage dtype, except when it does not

Upstream's `round_decimals_kernel` is `nearbyint(a * ten) / ten` at `scalar_t`,
with the operands swapped for a negative `decimals`. **The rounding that
decides the answer happens in the multiply, before `nearbyint` is reached**:

```text
float32 2.675 is 2.6749999523162842
  at f64:      2.6749999523 * 100 = 267.4999952  -> 267 -> 2.67
  at float32:  2.6749999523 * 100 = 267.5f       -> 268 -> 2.68   <- upstream
```

So the multiply and the divide are candle's, at the input's dtype, with `ten`
as a tensor of that dtype. `broadcast_div` rather than `affine(1.0 / ten)` for
the same reason one level down: `268 * 0.01f` is `2.6800001` and `268 / 100f`
is `2.68`.

**And `float16`/`bfloat16` are the exception, which a `float32`-only test
cannot see.** `c10::BFloat16`'s arithmetic operators return `float`, so
upstream's whole expression runs in `float` and narrows once. Evaluated in
`bfloat16` throughout it rounds twice — `bfloat16(2.675)` is `2.671875`, times
100 is `267.1875`, whose nearest `bfloat16` is `268`, and the answer comes back
`2.6875` where upstream says `2.671875`, i.e. the input unchanged, which is the
right answer for two decimal places of a value with three bits after the point.
The plain overload needs no such promotion and is not given one: `nearbyint` of
a `bfloat16` is exactly representable in `bfloat16`, measured against upstream
over the whole half-integer grid.

### 4.3 The bare overload and the `.decimals` overload disagree about integers

```text
torch.round(arange(3))              -> [0, 1, 2]     identity
torch.round(arange(3), decimals=0)  -> NotImplementedError "round_vml_cpu"
torch.round(arange(3), decimals=2)  -> NotImplementedError "round_cpu"
```

The same values and an argument that makes it a no-op, and it raises — and
*which kernel the message names depends on the value*, because `decimals == 0`
reaches `round_stub` and anything else reaches `round_decimals_stub`. Both were
read off real refusals. `bool` refuses on both paths with `round_vml_cpu`.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_round_is_half_to_even_and_the_grid_separates_it_from_half_away present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_round_keeps_the_sign_of_a_zero_result present -->

Verified element-wise against upstream on 4113 values (a half-integer grid of
4097 points, 4000 random draws and 16 special values) x 4 float dtypes x
{no decimals, -2, -1, 0, 1, 2, 3}: **bit-identical everywhere**.

---

## 5. `logsumexp` adds no arithmetic, and the infinite maximum is its only
branch

It is `amax`, `sub`, `exp`, `sum`, `log`, `add` — upstream's own sequence, in
candle, at the storage dtype. Doing it on the host at `f64` would have put it
on the readback list *and* stopped it matching: the candle sequence is
bit-for-bit identical to upstream on 200 random `float32` rows of 7, and an
`f64` reformulation is only approximately identical.

The stabilised form `max + log(sum(exp(x - max)))` is `nan` whenever `max` is
infinite, because `-inf - -inf` is `nan`. Upstream is not:

```text
logsumexp([-inf, -inf, -inf])  ->  -inf        the naive form gives nan
logsumexp([inf, 0.])           ->   inf        the naive form gives nan
logsumexp([nan, 0.])           ->   nan
```

Upstream's `masked_fill_(maxes.abs() == inf, 0)` is the fix, and **the
measurement that matters is that the zero has to reach both uses** — the
subtraction as well as the addition. With `max` replaced by `0` the all `-inf`
row becomes `log(exp(-inf) + ...) = log(0) = -inf` and the `+inf` row becomes
`log(inf + 1) = inf`, with no branch on the values and therefore no host read.
A `nan` maximum is not infinite, so it is left alone and poisons the row, which
is also upstream's answer. The finite path is untouched: only infinite maxima
are zeroed, so the bit-parity above still holds.

### 5.1 Three reductions, three dtype rules

```text
amax(arange(3), 0)       -> int64     keeps the input dtype
sum(arange(3), 0)        -> int64     promotes
logsumexp(arange(3), 0)  -> float32   neither
```

Inheriting either neighbour's rule would have been wrong, and a golden case for
`logsumexp` alone cannot say that — it can only say `logsumexp` agrees with
`logsumexp`. `bool` promotes to `float32` too.

### 5.2 `dim=[]` reduces everything, and then upstream raises

`logsumexp(zeros(2,3), dim=[], keepdim=True)` is shaped `[1, 1]` and holds the
reduction over all six, so an empty list is `amax`'s reading and not `sum`'s.
But with `keepdim=False` upstream raises `output with shape [] doesn't match
the broadcast shape [1, 1]`: it sized the result for the no-reduction reading
and then reduced anyway. **That is a defect upstream and it is reproduced, not
fixed**, because the alternative is answering where upstream raises. A
repeated dimension is refused with `amax`'s message; an empty reduction axis is
`-inf`, upstream's own second branch.

### 5.3 The accumulator, which is one ulp of `bfloat16`

`at::sum` on a half tensor folds into a `float` accumulator, so `exp` and `log`
run at the storage dtype and only the sum is wide. Summing in `bfloat16`
throughout costs up to **two ulps** on a row of nine — measured, and enough to
be visible without being enough for the golden tolerance to catch, which is the
size of error worth removing rather than allowing. With the wide accumulator,
`float16` is bit-identical to upstream and `bfloat16` and `float32` are within
one ulp on every row of a 300x9 draw.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_logsumexp_is_minus_inf_where_the_naive_stabilised_form_is_nan present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_logsumexp_promotes_an_integral_input_where_amax_keeps_and_sum_widens present -->

---

## 6. `t_`, and the first in-place op here that changes the receiver's shape

`rwkv`'s fifth wall. `torch/nn/init.py:705`'s `orthogonal_` calls
`flattened.t_()` whenever the weight it is initialising has `rows < cols`, and
`_init_weights` reaches the first such weight partway down `rwkv`'s module
list — which is why `docs/architectures/VOICE3.md` saw the line number move *backwards*, from
710 to 705.

Every other in-place op in this file hands `write_back` a replacement of the
receiver's own shape and dtype, and `tensor::write_into` checks exactly that
before writing through the layout. `t_` changes the layout and touches no
bytes, so `write_into` is the wrong primitive — it would refuse a `(3, 2)`
replacement for a `(2, 3)` receiver, correctly, because that check is what stops
a kernel from silently retagging a tensor.

So `t_` is the second caller of `replace_with`, and `replace_with`'s own doc
comment warns that a rebinding breaks aliasing. **`t_` is on the right side of
that warning**, and this is why: candle's `transpose` shares the storage `Arc`
and rebuilds only the `Layout`, so the new wrapper points at the *same* buffer.
Measured on both sides:

```text
x = arange(6.).reshape(2, 3);  v = x.view(-1);  x.t_();  v[0] = 99.
x[0, 0]  ->  99.0
```

The two existing `replace_with` callers (`set_`, `tensor.data = ...`) repoint at
a *different* buffer, which is why they read as rebindings and this does not.
No value comparison can see the difference — both sides answer the same numbers
whether or not the alias survives — which is why it is asserted here.

Rank decides everything, exactly as for `aten::t`: 0-D and 1-D come back
unchanged, 2-D swaps, 3-D or more raises `t_() expects a tensor with <= 2
dimensions, but self is 3D`. Reading `t_` as `transpose(-2, -1)` would compute
on a batched input where upstream refuses.

### 6.1 Two things `t_` does not do, both deliberate

* **It does not refuse a receiver that requires a gradient.** Upstream raises
  `a leaf Variable that requires grad is being used in an in-place operation`.
  `requires_grad` is inert in this shim (`tensor.rs`) and every other in-place
  op here is silent about it; making `t_` alone strict would put two rules
  behind one convention for no measured reason. Recorded as a narrowing.
* **It does not need anything added for capture or the tape.** The name ends in
  `_`, so `capture.rs::is_mutating` refuses it inside a trace and
  `note_mutation` bumps the receiver's storage version, both by name. The bump
  is **conservative** here in a way worth writing down: `t_` changes no bytes,
  so a trace constant that is merely transposed is treated as written to and
  its replay refused. `docs/training/BACKWARD8.md`'s `forgive_own_write` is the
  mechanism that would narrow it, and narrowing it means editing `capture.rs`.
  Refusing is the safe direction, so it is left.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_t_inplace_keeps_the_alias_it_had_before_the_call present -->

---

## 7. `cpmant` was never a backend limitation

`docs/architectures/ARCH200.md` filed `aten.embedding.default.unsupported` under "backend
refusals (candle layout, not a missing op)". The message it filed it on is:

```text
aten.embedding.default: candle: unsupported dtype F32 for op index-select
```

**`F32` is the weight's dtype and the weight is not the problem.**
`modeling_cpmant.py:602` casts `input_ids` to `int32` before the lookup, and
candle's `index_select` accepts only `u8`/`u32`/`i64` — so the refusal is about
the *index*, and it names the other operand. That misnaming is the whole of the
misclassification, exactly as `MatMulUnexpectedStriding`'s "non-contiguous lhs"
was in `docs/kernels/TAIL3.md` §1.

Upstream takes `Int` and `Long` alike and refuses everything else, so the fix
is a widening cast (`i32 -> i64`, free of any question) plus the refusal
upstream has — because accepting whatever *candle* accepts would have widened
`uint8` past upstream.

The refusal spells the dtype the legacy way, which is `TORCH_CHECK`'s
`t.type()` and not the scalar-type name every other message in `aten.rs` uses:

```text
float32   torch.FloatTensor      int8    torch.CharTensor
float64   torch.DoubleTensor     int16   torch.ShortTensor
float16   torch.HalfTensor       int32   torch.IntTensor
uint8     torch.ByteTensor       int64   torch.LongTensor
bfloat16  CPUBFloat16Type   <-- no `torch.` prefix, no `Tensor` suffix
bool      CPUBoolType       <-- the same
```

The last two are the ones a generated spelling gets wrong, and `bool` is the
one a caller is most likely to hit.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs legacy_tensor_type_name present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_embedding_takes_an_int32_index_which_is_what_cpmant_hands_it present -->

---

## 8. Architectures

Ran with `pytests/arch_sweep.py --only ...`, before and after, on the same
eight.

| architecture | before | after |
|---|---|---|
| `rwkv` | `TensorBase.t_` (construction) | **constructs and forwards** |
| `aria` | `TensorBase.index_copy_` | **forward passes** |
| `aria_text` | `TensorBase.index_copy_` | **forward passes** |
| `granite_swa` | `logsumexp` | **forward passes** |
| `fastspeech2_conformer` | `round` | `torch.zeros((tuple), dtype=, device=)` — an argform, the next wall |
| `cpmant` | `aten.embedding.default.unsupported` | **forward passes** (two walls: §7, then §10) |
| `longformer` | `aten.as_strided.default` | unchanged this round; `docs/kernels/STRIDED.md` moved it — §1 |
| `led` | `aten.as_strided.default` | unchanged this round; `docs/kernels/STRIDED.md` moved it — §1 |

**Five cleared, one moved, two refused on purpose** (and the two refused were
cleared by `docs/kernels/STRIDED.md`).

### 8.1 `t_` was `rwkv`'s last wall

`rwkv` had moved through five: `ndimension` -> `new_empty` -> `torch.maximum`
(pre-`ARCH100`) -> `linalg_qr` (`ARCH100`) -> `diag` (`ARCH200`) -> `t_`. Every
one of them was inside `_init_weights` rather than in a forward, and every one
was exposed by closing the one before it, so "is this the last one" was a real
question rather than a rhetorical one.

It was. `orthogonal_` calls `t_` twice — once on the way in when `rows < cols`
and once on the result after the QR — and both, plus `view_as`, `copy_` and
`mul_` after them, were already there. `rwkv` now constructs *and* forwards.

### 8.2 The two that moved

Neither is a regression and neither is a partial result. `fastspeech2_conformer`
is one line past `torch.round`:
`torch.clamp(torch.round(...), min=0).long()` now runs, and the wall is an
argument form of `torch.zeros` further into the same forward. It is
data for the next batch, in `docs/architectures/ARCH200.md`'s sense.

`cpmant` moved twice inside this round: past the embedding index (§7) onto
`TensorBase.logical_not`, and then past that too (§10). It is the only
architecture here that needed two of this round's items, and the second was
only visible because the first landed.

---

## 9. What is verified, and how

Every op is compared against upstream element-wise, in a separate process, on
inputs where a plausible wrong implementation differs.

* **Golden harness** (`tools/golden/cases.py`):
  <!-- DOCWATCH: count golden_cases_total ge 11189 -->
  <!-- DOCWATCH: count golden_cases_passed ge 11189 -->
  <!-- DOCWATCH: count golden_ops_covered ge 296 -->
  <!-- DOCWATCH: count golden_pending eq 0 -->
  11189 cases pass of 11189, 296 ops covered, 0 pending builders (10691/287 at
  this worktree's base; develop has moved since and the coordinator
  re-measures on the merged tree). The `round` cases run on the half-integer grid and on `2.675` at
  every decimal place, because those are the inputs where the two rounding
  rules and the two scaling precisions differ; ordinary values separate
  nothing.

* **Beyond the harness**, run once during development and reported here rather
  than pinned, because the sweep is too large to keep in the suite: `round`
  compared element-wise against upstream on 4113 values x 4 float dtypes x 7
  decimal settings (**bit-identical**), `index_copy_` on 37 shape/dtype/refusal
  probes (**identical, including every message**), and `logsumexp` on 300x9
  draws at four dtypes (**bit-identical at `float16`, within one ulp
  elsewhere** — candle's sequential `sum` against upstream's vectorised one,
  which is `sum`'s own pre-existing divergence and not this kernel's).

* **`pytests/test_tail4.py`**, a new suite file, holding down what a value
  comparison structurally cannot: the `as_strided` verdict *and its
  precondition*, banker's rounding as a separator rather than as an answer, the
  three ways `index_copy_` and `index_add_` disagree (each asserted against the
  other op in the same process), `t_`'s surviving alias, and the
  alias-versus-kernel split as source structure.
  <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_the_write_door_is_single_which_is_the_precondition_a_read_only_as_strided_would_need present -->
  <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_none_of_this_rounds_kernels_reads_a_tensor_back_to_the_host present -->

* **Three pinned facts in `test_shim.py`** moved, each carrying the arithmetic
  that keeps it a check: `tag_core_count` 130 -> 132 (`round.default` and
  `logical_not.default` are the only two of the nine new keys upstream tags
  `core` — `round.decimals`, the same op with one more argument and the *same
  function* behind it, is not), distinct schema identities 351 -> 360 (**+9,
  and seven of the nine are in both tables and count once**; `index_copy_` and
  `t_` are `methods.json`-only because upstream has no `torch.index_copy_` and
  no `torch.t_`), and `_EXPECTED_MUTABLE` +4 (**four, not nine** —
  `index_copy`, `round`, `round.decimals`, `logsumexp` and `logical_not` are
  out-of-place and must not appear).

* **`device.rs::MPS_HOST_READBACK_OPS` is unchanged, and that is a result
  rather than an omission.** None of the four new kernels moves a tensor to the
  host: `index_copy_` uses candle's `scatter`, `round` is written out of candle
  primitives instead of `f64::round_ties_even`, `logsumexp` is a candle
  reduction chain, `logical_not` is one comparison, and `t_` touches no bytes
  at all.
  `test_none_of_this_rounds_kernels_reads_a_tensor_back_to_the_host` asserts
  the property directly rather than relying on
  `test_the_mps_readback_list_is_what_the_kernels_actually_do`'s derivation,
  which follows helpers only one level by name — the gap `docs/architectures/VOICE3.md`
  found `var`/`std` hiding in.

The ops now in `_aten_implemented()`:

<!-- DOCWATCH: op-implemented aten.index_copy_.default -->
<!-- DOCWATCH: op-implemented aten.index_copy.default -->
<!-- DOCWATCH: op-implemented aten.round.default -->
<!-- DOCWATCH: op-implemented aten.round.decimals -->
<!-- DOCWATCH: op-implemented aten.round_.default -->
<!-- DOCWATCH: op-implemented aten.round_.decimals -->
<!-- DOCWATCH: op-implemented aten.logsumexp.default -->
<!-- DOCWATCH: op-implemented aten.t_.default -->
<!-- DOCWATCH: op-implemented aten.logical_not.default -->

and the table rows that make them callable:

<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json round present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json round_ present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json logsumexp present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json index_copy present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json round present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json round_ present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json logsumexp present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json index_copy present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json index_copy_ present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json t_ present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json logical_not present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json logical_not present -->

Only the overloads with kernels are declared. `round.out`,
`round.decimals_out`, `logsumexp.out` and `index_copy.out` are **not** in the
tables, so `torch.round(x, out=y)` refuses with "no matching signature" rather
than naming the overload. That is a narrower choice than `floor`/`ceil` make
(both declare their `.out` form with no kernel behind it), and it is recorded
here rather than silently inconsistent: the value of declaring a kernel-less
overload is that the refusal names it, and the cost is a schema identity that
claims a door. `as_strided` is the case where the first is worth the second,
because a caller reaching it needs to know *which* overload is missing; a
caller passing `out=` does not.

---

## 10. `logical_not` is not `bitwise_not`, and `bool` is the only dtype where
that is invisible

Added to this round after the fact. The `vmap` round landing in the same batch
moved `nemotron3_5_asr`, `nemotron_asr_streaming` and
`nemotron_asr_streaming_encoder` past `_vmap_increment_nesting`, and all three
stop one line later at `modeling_nemotron_asr_streaming.py:628`,
`attention_mask.logical_not()`. §8's `cpmant` had already arrived at the same
name from the other direction.

`~` and `bitwise_not` already worked, which is exactly the trap.
`docs/kernels/TAIL1.md` measured the same distinction for the `and` pair and found the
two genuinely diverge on integers. Measured here for `not`, on
`[0, 1, 2, -1, 3]`:

```text
                logical_not                    bitwise_not
bool      [True, False, False, False, False]   the same
int64     [True, False, False, False, False]   [-1, -2, -3, 0, -4]   int64
uint8     [True, False, False, False, False]   [255, 254, 253, 0, 252]
float32   [True, False, False, False, False]   "bitwise_not_cpu" not
                                                implemented for 'Float'
```

**They agree on `bool` and on nothing else.** Every integral dtype differs in
values *and* in result dtype, and every floating dtype makes `bitwise_not`
raise outright. So a kernel that routed `logical_not` at `bitwise_not` would be
right for exactly the one dtype a test is most likely to use, which is why the
two are asserted against each other in the same process rather than each
against upstream separately.

### 10.1 The result is always `bool`

Whatever the input dtype. A version that kept the input's answers `1.0`/`0.0`
instead of `True`/`False` — plausible values with the wrong type, which is the
shape golden's dtype comparison exists to catch and which a `tolist()`
comparison alone would pass.

### 10.2 `x == 0`, not `!(x != 0)`

One value separates them and it is `nan`. `logical_not(nan)` is `False`
upstream — a NaN is truthy — and `nan != 0` is `true` in IEEE, so the
negated spelling answers `False` for the wrong reason and the equality spelling
answers `False` because a comparison against NaN is false in *both* directions.
Writing it as `eq` puts the NaN answer in the comparison rather than in a
negation that would have to be reasoned about. `-0.0` is zero and therefore
`True`; `inf` is `False`.

### 10.3 What is not implemented, and why that is a rule rather than an omission

`aten::logical_not.out` and `aten::logical_not_` both exist upstream, and
`Tensor.logical_not_` is a real bound method; `torch.logical_not_` is **not** —
checked rather than assumed, the way `docs/bindings/BIND3.md` checked `torch.im2col` and
found it does not exist at all. Neither is implemented here, and
`logical_not_` is the interesting one: it keeps the **receiver's** dtype
(`tensor([0., 1.]).logical_not_()` is `[1., 0.]`, still `float32`), which is a
different rule from this kernel's and would be a second kernel rather than a
flag on it. Nothing measured calls it. It refuses by name.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs logical_not_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_logical_not_and_bitwise_not_agree_on_bool_and_on_nothing_else present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail4.py test_logical_not_treats_nan_as_TRUE_and_minus_zero_as_FALSE present -->
