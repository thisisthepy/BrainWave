# SCATTER — the scatter family, `bucketize` and `prod`: 13 of 15 architectures through the wall

Worktree `work/scatter` on develop `eb84708`. torch 2.13.0 upstream
(`/Volumes/macMini/caches/spike-venv/bin/python`), `transformers` 5.15.1.

docs/ARCH100.md ranked the 82 architectures its sweep found blocked, and five of its rows are
this round: `TensorBase.scatter_` (7 architectures, the second-ranked name in the whole sweep),
`aten.scatter.value` (4), `TensorBase.masked_scatter` (1), `aten.bucketize` (2), `aten.prod` (1).
Fifteen architectures, five names.

**Thirteen of the fifteen now forward. The two that do not hit a second wall, each a different
op**, and that is the expected shape rather than a shortfall — docs/ARCH100.md §1 says its 31 is
a first-wall count and names `rwkv` as the precedent.

---

## 0. What landed, split the way docs/ARCH100.md §5.3 asks

```text
NEW KERNEL (arithmetic that did not exist)
    aten.masked_scatter.default      positional consumption of a source under a broadcast mask
    aten.bucketize.Tensor            binary search, lower/upper bound
    aten.bucketize.Scalar            the same, from a Python number
    aten.prod.default                the empty-product identity, stepwise accumulation
    aten.prod.dim_int                the same along one axis

BINDING OVER AN EXISTING KERNEL (a spelling, not arithmetic)
    aten.scatter_.src                scatter.src + write_back, ~15 lines

BINDING PLUS A SMALL KERNEL (the shared loop existed; the scalar path did not)
    aten.scatter.value               scatter.src's loop with the source collapsed to a number
    aten.scatter_.value              the same, + write_back

TABLE-ONLY (no code)
    methods.json    scatter_ (2 schemas), masked_scatter, prod (2 schemas)
    overloads.json  masked_scatter, bucketize (2 schemas), prod (2 schemas)
```

**`aten.scatter.value` is docs/ARCH100.md's binding/kernel split in one op, and it is worth
naming precisely.** Its schema was *already* in `overloads.json` **and** `methods.json` before
this round — transcribed correctly, resolving correctly — and the only thing missing was the
dispatch arm in `aten.rs`. So `zeros.scatter(1, idx, 1)` resolved and then raised
`aten op not implemented in torch._C shim: aten.scatter.value`, which is why it appears in the
sweep as `missing_aten_op` rather than `missing_shim_name`. Four of the fifteen architectures
needed **no table change at all**; six of the eight new dispatch keys needed no new arithmetic
either. That is the 49-to-22 finding measured from the other side.

Counts: golden **9301/9301, ops=232** (was 8921/8921, ops=222 at docs/ARCH100.md; the intervening
rounds account for the rest). Suite green. No test was weakened; three pinned counts moved and
each moved with the arithmetic that keeps it a check (§8).

<!-- DOCWATCH: op-implemented aten.scatter.value -->
<!-- DOCWATCH: op-implemented aten.scatter_.src -->
<!-- DOCWATCH: op-implemented aten.scatter_.value -->
<!-- DOCWATCH: op-implemented aten.masked_scatter.default -->
<!-- DOCWATCH: op-implemented aten.bucketize.Tensor -->
<!-- DOCWATCH: op-implemented aten.bucketize.Scalar -->
<!-- DOCWATCH: op-implemented aten.prod.default -->
<!-- DOCWATCH: op-implemented aten.prod.dim_int -->

---

## 1. The wall, re-measured

```text
                       BEFORE          AFTER
  scatter_             7 blocked       0 blocked
  scatter.value        4 blocked       1 blocked   (jetmoe -> TensorBase.index_add)
  masked_scatter       1 blocked       0 blocked
  bucketize            2 blocked       0 blocked
  prod                 1 blocked       1 blocked   (tapas -> TensorBase.scatter_reduce)
  ---------------------------------------------
                      15 blocked       2 blocked
```

`rust/torch_c/pytests/arch_sweep.py --only <the fifteen>`, run on both sides. Upstream forwards
all fifteen (the baseline docs/ARCH100.md established, re-confirmed here rather than assumed);
the shim forwards thirteen.

The two second walls, both `missing_shim_name` — a spelling, which docs/ARCH100.md's split says
is the cheaper kind:

* **`jetmoe` → `TensorBase.index_add`.** The *in-place* `aten.index_add_.default` has had a
  kernel since docs/DEMAND8.md; the out-of-place `index_add` has neither a kernel nor a
  `methods.json` entry. `jetmoe` reaches it in `moe.py`'s expert accumulation, one line after
  the `scatter` this round unblocked.
* **`tapas` → `TensorBase.scatter_reduce`.** `modeling_tapas.py:1465` is
  `out.scatter_reduce(..., reduce="sum", include_self=False)`. **`scatter_reduce` is a different
  aten op from `scatter.reduce`**, with a different argument list (`include_self` exists on one
  and not the other) and a wider set of reductions (`amin`/`amax` as well as `sum`/`prod`), so
  implementing `scatter`'s `reduce=` would not have moved `tapas` at all. That is worth stating
  because it is the kind of near-name that makes a roadmap wrong.

---

## 2. What the eleven scatter architectures actually pass — and what `reduce=` is doing here

Every row below is a `grep` of the installed `transformers` 5.15.1, not a recollection.

```text
exaone_moe    modeling_exaone_moe.py:252     group_mask.scatter_(1, group_idx, 1)
glm4_moe      modeling_glm4_moe.py:304       group_mask.scatter_(1, group_idx, 1)
glm4_moe_lite modeling_glm4_moe_lite.py:401  group_mask.scatter_(1, group_idx, 1)
mistral4      modeling_mistral4.py:165       group_mask.scatter_(1, group_idx, 1)
nemotron_h    modeling_nemotron_h.py:740     group_mask.scatter_(1, group_idx, 1)
solar_open    modeling_solar_open.py:127     group_mask.scatter_(1, group_idx, 1)
groupvit      modeling_groupvit.py:57,77     zeros_like(logits).scatter_(dim, index, 1.0)
axk2          modeling_axk2.py:642           .scatter(-1, topk_indices.long(), False)
deepseek_v32  modeling_deepseek_v32.py:465   .scatter(-1, topk_indices.long(), False)
glm_moe_dsa   modeling_glm_moe_dsa.py:442    .scatter(-1, topk_indices.long(), False)
jetmoe        modeling_jetmoe.py:202         zeros.scatter(1, top_k_indices, 1)
```

Six of the seven `scatter_` rows are the **same line** in a DeepSeek-style MoE group router,
copied between model files. That is a fact about the ranked list worth carrying: "7
architectures" is one call site, and docs/ARCH100.md's own caution about `__setitem__`'s thirteen
being dominated by one family applies here too.

**`reduce=` is not implemented, and the measurement is the reason.** Not one of the eleven passes
it. `scatter.reduce` and `scatter.value_reduce` have no kernel and no table entry; the refusal is
the *unimplemented-op* one, which is a precise work item. `pytests/test_scatter.py` has a test
that fails if that ever silently changes.

<!-- DOCWATCH: op-not-implemented aten.scatter.reduce -->
<!-- DOCWATCH: op-not-implemented aten.scatter.value_reduce -->

---

## 3. `scatter.value` is not `scatter.src` with a broadcast source

Four measured differences from the kernel beside it, each of which a shared implementation would
have got wrong:

| | `scatter.src` | `scatter.value` |
|---|---|---|
| 0-d `self` | refused by this shim | **legal**, `tensor(0.).scatter(0, tensor([0]), 7.)` is `7.0` |
| index-dtype refusal | `Expected dtype int32 or int64 for index, got <name>` | `Expected dtype int32/int64 for index` |
| dtype rule | `self.dtype == src.dtype` or refuse | the scalar is *converted* to `self`'s dtype |
| out-of-range value | n/a | `c10::checked_convert`, exactly as `full`/`fill_` |

The rank rule is upstream's `ensure_nonempty_dim`, `max(rank, 1)` on both sides — the same rule
`gather_default` already reproduces. So `(0-d self, 1-d index)` and `(1-d self, 0-d index)` both
bind and `(0-d self, 2-d index)` does not. Guessing "ranks must be equal" would have refused two
calls upstream answers.

On the scalar conversion, four measurements that split "overflow" from "truncation":

```text
int64.scatter(..., 2.5)     ->  2          truncation is not an overflow
int64.scatter(..., -2.7)    -> -2          toward zero, not toward -inf
uint8.scatter(..., -1)      -> 255         checked_convert's two's-complement wrap
float32.scatter(..., inf)   -> inf         infinity converts to infinity

uint8.scatter(..., 300)     raises  "value cannot be converted to type uint8_t without overflow"
int32.scatter(..., 2**31)   raises  "... to type int without overflow"
float16.scatter(..., 1e6)   raises  "... to type c10::Half without overflow"
int64.scatter(..., nan)     raises  "... to type int64_t without overflow"
```

**`fill_`'s `numel == 1` hole does not exist here** — a *one-element* `float16` receiver still
refuses `1e6`. That hole is upstream's CPU `fill_` fast path and nothing else has it, so
`checked_convert` is called with a `numel` that is not 1 rather than with the receiver's. Both
the one-element and the three-element case are golden `both_error` cases, because they are the
two that would come apart if the hole were copied by analogy.

### 3.1 Duplicate indices: the question has no content here

`scatter.src`'s doc comment records that duplicate indices resolve to the **last** write in
iteration order (measured: `[1,2,3]` at index `[0,0,0]` leaves `3`). Upstream documents
`scatter` as nondeterministic for duplicate indices, and that is a real caveat for the `src`
form.

**For `scatter.value` it cannot be observed at all.** Every write puts the *same* number in, so
whichever order the writes happen in, the answer is that number. Three writes of `7.0` into
column 0 leave `7.0`. The golden case says exactly that in its note rather than pinning a witness
that could not distinguish anything — which is what "do not pin a test to an accident" means
here. All eleven architectures use the value form, so none of them is exposed to the
order-dependence at all.

The `.src` form's answer was re-measured this round and is unchanged: last write wins, in
row-major order over the index's shape. It is a property of upstream's serial CPU loop, not a
guarantee, and `scatter_src_cases` already carries it with that caveat in its note.

### 3.2 One divergence, deliberate: partial writes on a bad index

Upstream's kernel bounds-checks each index **as it writes**:

```text
x = zeros(1, 3)
x.scatter_(1, tensor([[0, 5]]), 7.0)    raises -- and leaves x == [[7., 0., 0.]]
```

This shim checks every index *before* writing anything, so a refusal leaves the receiver
untouched. That is a divergence and it is recorded here rather than reproduced: upstream's
half-finished write is an artefact of loop order, not a documented behaviour, and there is no
plausible caller that wants it. It is also not reachable through any of the eleven — every one
of them scatters a `topk` index, which is in range by construction.

---

## 4. `scatter_` is in-place, and its `Overlap` answer is not `masked_fill_`'s

docs/INPLACE.md §1's shape exactly: the out-of-place kernel computes a fresh replacement and
`write_back` puts it through the receiver's **layout**. So a view taken before the call sees the
write, and `t.scatter_(...) is t` holds because the wrapper is never rebound. `_view_write_cases`
gained three entries (a stride-4 `select.int` view, a doubly non-contiguous `t()` view, and the
`.src` form through the first of those), each of which reads the **base** and discards the return
value — a return-value check passes against a rebind and is therefore not a check.

**`Overlap::Refuse`, and that is measured rather than defaulted:**

```text
zeros(1,3).expand(2,3).scatter_(1, tensor([[0]]), 5.0)
    upstream: RuntimeError, "more than one element of the written-to tensor refers to a
              single memory location. Please clone() the tensor before performing the operation."

zeros(1,3).expand(2,3).masked_fill_(tensor([True,False,False]), 5.0)
    upstream: writes.
```

So `scatter_` deliberately does **not** join `write_back`'s `Allow` list, where `masked_fill_`
and `index_put_` sit. A shim with one answer for every in-place op passes one of these two and
fails the other.

**Capture and the eager tape needed nothing.** The op key ends in `_`, so `capture::is_mutating`
already refuses a trace and `note_mutation` already stamps the storage. `forgive_own_write`
(docs/BACKWARD8.md §2.3) is for the ops that mutate *without* an underscore —
`MUTATES_WITHOUT_UNDERSCORE` holds exactly `native_batch_norm` — and widening it to reach
`scatter_` would forgive a write the guard exists to see. `scatter_` has no derivative rule in
`tape.rs`, so `loss.backward()` through one refuses by name; it does not silently answer, and it
does not poison the tape for programs that do not use it. That is the same state
`masked_fill_`/`index_put_` are in.

---

## 5. `bucketize`: `right` does not mean what its name suggests

`idefics3_vision` and `smolvlm_vision` both reach it through the identical line
`torch.bucketize(fractional_coords, boundaries, right=True)` in the patched-image position
encoder.

Swept on `boundaries = [1, 3, 5, 7]` with the values **on** the boundaries, the way docs/FIXES.md
swept `-300..300` and found the `-256` case the reported examples could not reach:

```text
value            0  1  2  3  5  7  8
right=False      0  0  1  1  2  3  4      first index with boundary >= value   (lower_bound)
right=True       0  1  1  2  3  4  4      first index with boundary >  value   (upper_bound)
```

**The only values at which the two columns differ are the boundaries themselves.** A test on
interior points passes against either flag, and against a linear scan, and against the flag
inverted. That is why every golden case here sits on a boundary or is non-finite.

Duplicate boundaries are the second witness. On `[1, 3, 3, 5]`:

```text
bucketize(3)              -> 1     the START of the run of threes
bucketize(3, right=True)  -> 3     the END of it
```

Any implementation that stopped at the first match it found would give the same number twice.

**NaN is the third, and it decides how the comparison is written.** IEEE makes every comparison
against NaN false, so the *negated* predicate upstream uses (`!(b >= v)`, `!(b > v)`) is true for
every boundary and the search runs off the end:

```text
bucketize(nan,  [1,3,5,7])              -> 4      and 4 with right=True
bucketize(inf,  [1,3,5,7])              -> 4
bucketize(-inf, [1,3,5,7])              -> 0
```

Writing the predicate the obvious way round (`b < v`) gives **0** for NaN. Both flags and all
three non-finite values are golden cases.

Three more, measured:

* **`boundaries` must be rank 1** — not 0, not 2:
  `boundaries tensor must be 1 dimension, but got dim(N)`. An *empty* `boundaries` is fine and
  every answer is `0`, which is the case a loop written `for i in 1..len` gets wrong.
* **Unsorted boundaries are not refused.** Upstream runs the binary search anyway:
  `bucketize(2, [5,1,3])` is `2`. This shim does the same search rather than sorting first —
  sorting would be a *better* answer and a different one.
* **The dtypes need not agree.** `int64` values against `float32` boundaries and the reverse both
  compare numerically. Both sides are widened to `f64` here, which is exact for every dtype this
  shim stores **except `int64` beyond 2^53**: `bucketize(tensor([2**53+1]), tensor([2**53]))` is
  `1` upstream and `0` here. Recorded, not closed — an exact path would need the comparison to
  branch on the pair of dtypes, and no measured caller is near that magnitude.

Output is `int64`, or `int32` with `out_int32=True`, with `self`'s shape (0-d for the `Scalar`
overload). `torch.bucketize` exists upstream and `Tensor.bucketize` does **not**
(`hasattr(torch.Tensor, "bucketize")` is `False` on 2.13.0), so it is `overloads.json`-only.

<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json bucketize present -->

---

## 6. `prod`: the empty product, the widening, and where the accumulator lives

`tapas` reaches it as `torch.prod(torch.tensor(list(index.batch_shape())))` — a full reduction of
a short `int64` vector (`modeling_tapas.py:1380`).

**The empty product is 1, not 0**, in every dtype and in every shape:

```text
prod(tensor([]))                  -> 1.0
prod(tensor([], dtype=int64))     -> 1
prod(zeros(2, 0), 1)              -> [1., 1.]
```

The third is the one an implementation that special-cased the full reduction still gets wrong,
so it is a separate case.

**Every integral input, and `bool`, comes back `int64`**, measured over all nine stored dtypes —
`int16`, `int32`, `int64`, `uint8` and `bool` all widen; floating inputs keep their own width,
`float16` and `bfloat16` included. (`int8` is not storable by this shim's backend at all and is
out of scope here, not an omission of `prod`'s.)

**`dtype=` casts the input, it does not cast the answer.** This one is decisive:

```text
prod(tensor([2.5, 3.0]), dtype=int64)   -> 6
```

`2.5 * 3.0` is `7.5`; truncating the product gives 7 and rounding it gives 8. Only casting first
— `2 * 3` — gives upstream's 6.

**Accumulation happens in the output dtype, step by step**, and `bfloat16` is the only dtype here
that can prove it. On eight copies of `1.1`:

```text
upstream                            2.15625
stepwise, rounding to bf16 each step 2.15625     <- this
accumulate in f64, narrow once       2.171875
```

That is `float_narrower`'s finding for `div` (docs/KERNELS26.md D2, docs/DEMAND2.md) in a
different op, and the remedy is the same closure applied after every step. `int_narrower` is its integral counterpart, and it is
needed for the same reason: `prod(tensor([2**32, 2**32]))` is **0** (`int64` wraps, where an `f64`
accumulator would give `1.8e19`), `prod(..., dtype=int32)` of `70000 * 70000` is `605032704`, and
`prod(uint8 [20, 20], dtype=uint8)` is `144`.

### 6.1 A limit, recorded rather than closed

Above roughly 32 elements upstream switches to a vectorised reduction whose lane order is neither
sequential nor a simple pairwise tree. For `float16`/`bfloat16` that is visible:

```text
n copies of 1.1, bfloat16       upstream   sequential   pairwise tree
  n = 3                          1.3359375  1.3359375    1.3359375
  n = 8                          2.15625    2.15625      2.15625
  n = 17                         5.15625    5.15625      5.125
  n = 64                       472.0      482.0        468.0        <- all three differ
  n = 129                   241664.0   258048.0     241664.0
```

This kernel is sequential. So reduced-precision products over long axes can differ in the last
places, and the golden cases stay inside the range where the two agree (`n <= 8`). Pinning a case
to upstream's lane count would be pinning a test to an accident — it is a property of the SIMD
width of the machine the measurement was taken on. `float32`/`float64`/integral dtypes are exact
at every length measured and carry no such limit.

---

## 7. `masked_scatter`: the source is consumed, not broadcast

`higgs_audio_v2` (docs/ARCH100.md's row) and `idefics3`'s language model both spell it
`hidden_states.masked_scatter(token_mask.unsqueeze(-1), replacement)` — a `(B, S, 1)` mask
against a `(B, S, H)` receiver, with `source` holding the selected rows, flat.

It is **not** `masked_fill` with a tensor value and it is not `index_put_`. `source` is consumed
*positionally*, one element per true position in row-major order, and its shape is ignored beyond
its element count. Four measurements, each a thing a plausible implementation gets wrong:

* **Row-major consumption from the source's own logical order.** A transposed source is read in
  logical, not storage, order:
  `zeros(2,3).masked_scatter(m, arange(6.).reshape(2,3).t())` takes `0, 3, 1` — not `0, 1, 2`.
* **Extra source elements are ignored; too few is an error** —
  `Number of elements of source < number of ones in mask`. An all-false mask therefore accepts an
  *empty* source, which is the degenerate case a `source[0]` would crash on.
* **Both operands broadcast, not just the mask.** `zeros(3)` against a `(2,3)` mask returns a
  **`(2,3)`** result. So this is the ordinary two-sided broadcast and not "expand the mask to
  `self`", which is what the documentation's wording suggests.
* **No dtype movement in either direction.** The mask must be exactly `bool` — a `uint8` mask is
  refused (`masked_scatter_ only supports boolean masks, but got mask with dtype Byte`) where
  `masked_fill_` only *warns* for the same mask. And `self`/`source` must agree exactly:
  `masked_scatter: expected self and source to have same dtypes but gotFloat and Long`. The
  missing space after `got` is upstream's; it is transcribed rather than tidied, for the reason
  docs/PROMOTE.md gives.

Note that upstream's refusal message says `masked_scatter_` — the **in-place** op — even for the
functional call. That text is kept as upstream writes it. It is also why the round's mutable-op
list is checked against the parsed schema and not against error text: a list built from the
message would have wrongly gained `masked_scatter`.

**`masked_scatter_` (the in-place form) is not implemented.** No measured caller uses it — both
architectures use the functional form and rebind — and adding it would be one more table
identity and one more `_view_write_cases` entry for a spelling nobody reached. It is a decision,
recorded here, not an oversight.

<!-- DOCWATCH: json-key rust/torch_c/src/methods.json masked_scatter present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json scatter_ present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json prod present -->

---

## 8. The three pinned counts that moved, and the arithmetic that keeps each a check

None was relaxed; each gained a paragraph saying what the new number is made of and what a
*different* number would have meant.

**`tag_core_count` 110 → 114.** Eight new dispatch keys, **four** of them `core` upstream, every
one read off its own `.tags` on a real torch:

```text
scatter.value          ['core', 'pt2_compliant_tag']               counted
masked_scatter.default ['core', 'pt2_compliant_tag']               counted
prod.default           ['core', 'pt2_compliant_tag', 'reduction']  counted
prod.dim_int           ['core', 'pt2_compliant_tag', 'reduction']  counted
scatter_.src           ['inplace', 'pt2_compliant_tag']
scatter_.value         ['inplace', 'pt2_compliant_tag']
bucketize.Tensor       ['pt2_compliant_tag']
bucketize.Scalar       ['pt2_compliant_tag']
```

`scatter_` is not core although `scatter` is; `bucketize` is not core at all although it is an
ordinary elementwise search that two architectures need in their forward. Neither is derivable —
they are upstream's table, and reading them one at a time is what makes this +4 and not +8.

**Table identities 295 → 302.** +7, and the split across the two tables is the check:

```text
scatter_        methods.json only     +2    upstream has no torch.scatter_
masked_scatter  both tables           +1    both doors name one identity
bucketize       overloads.json only   +2    upstream has no Tensor.bucketize
prod            both tables           +2    one per overload
scatter.value   already in both       +0    the binding was there; only the arm was missing
```

+11 would have meant the six `.out`/`reduce` schemas with no kernel behind them had been listed
too; they are deliberately absent, because a dead overload key counts against `reach_allow.json`'s
`shape1_dead_overload_keys_ceiling`, which is a ratchet.

**`_EXPECTED_MUTABLE` +2.** `scatter_.src` and `scatter_.value`. Two and not eight: the other six
keys this round added are out-of-place and must **not** appear, which is what makes the list a
check on the schema parse rather than a restatement of what changed.

---

## 9. Where the tests are, and what each can fail on

```text
tools/golden/cases.py       164 new cases across eight ops (34 scatter.value, 36 scatter_.value,
                            2 scatter_.src, 18 masked_scatter, 20 bucketize.Tensor,
                            21 bucketize.Scalar, 21 prod.default, 12 prod.dim_int), every one
                            compared against upstream element-wise, in a separate process.
                            Boundary values, dtype refusals, and the view-write cases for
                            scatter_.
rust/torch_c/pytests/test_scatter.py
                            23 tests, all through the USER-LEVEL spelling (torch.<name> /
                            Tensor.<name>), each comparing a vendored-tree subprocess against
                            an upstream subprocess with PYTHONPATH stripped. Nothing here is a
                            transcribed constant.
```

The split is deliberate and it is the §0 finding restated as a test strategy. The golden harness
compares at `_aten_dispatch`, which is the right place for dtype coverage and **cannot see a
missing binding at all** — `aten.scatter.value` would have been green there while
`zeros.scatter(1, idx, 1)` still raised. So every assertion in `test_scatter.py` goes through the
spelling, and the ones that matter are the exact lines from `transformers`.

Three of them are shaped so that they cannot pass against a plausible wrong implementation:

* `test_scatter_writes_through_a_strided_view_and_not_into_a_copy` reads the **base**, never the
  return value. Every in-place op returns `self`, so a return-value check passes against a
  kernel that computed into a fresh buffer — docs/VIEWS.md §6's failure, which was 3037 cases
  green while no in-place write was visible through any view.
* `test_scatter_refuses_a_receiver_whose_elements_share_memory` fails against a shim that shares
  one `Overlap` answer across in-place ops, because `masked_fill_` beside it wants the other one.
* `test_prod_accumulates_step_by_step_in_the_output_dtype` asserts the answer is **not**
  `2.171875`, which is precisely what the obvious f64-accumulator implementation returns.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs scatter_value present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs masked_scatter_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs bucketize_position present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs int_narrower present -->

---

## 10. What is still open

| | why it is open |
|---|---|
| `scatter.reduce`, `scatter.value_reduce` | no measured caller passes `reduce=`. §2 |
| `scatter_reduce` (a different op) | `tapas`' second wall. Wider reduction set, `include_self` |
| `index_add` (out of place) | `jetmoe`'s second wall. The in-place form has a kernel |
| `masked_scatter_` | no measured caller. §7 |
| `bucketize` beyond 2^53 in `int64` | both sides widen to `f64`. §5 |
| reduced-precision `prod` over long axes | upstream's vectorised lane order. §6.1 |
| `.out` variants of all five ops | dead keys against the `reach_allow.json` ratchet |

<!-- `smoke_ok` counts `test_shim.py` alone, which is how docwatch measures it (a fresh
     stage, `pytests/` NOT on PYTHONPATH, so the split-off suite files do not run). The whole
     suite is 504: 479 here + 23 in `test_scatter.py` + 2 in `test_split_probe.py`. -->
<!-- DOCWATCH: count smoke_ok ge 479 -->
<!-- DOCWATCH: count golden_cases_passed ge 9301 -->
<!-- DOCWATCH: count golden_ops_covered ge 232 -->
<!-- DOCWATCH: count golden_pending eq 0 -->
