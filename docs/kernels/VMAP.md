# `vmap`: what the four architectures actually ask for, and what was built

Round date 2026-09-07. Branch `work/vmap`, on develop `a523ae4`. Host Apple M1,
CPython 3.13, upstream torch 2.13.0, transformers 5.x, `candle-core` 0.11.0.

`docs/kernels/COMPLEX.md` §7 sized `torch._C._functorch._vmap_increment_nesting` as the
largest single entry on `docs/architectures/ARCH200.md`'s blocked list — four architectures
where every other entry blocks one or two — and refused to stub it, for a
reason worth repeating before anything else in this document:

> A no-op counter is worse than nothing. It lets the call proceed and the
> result is used as an attention mask, which does not raise — it just produces
> a wrong mask and a model that appears to run.

That is still the whole risk. This round did not stub it and did not build a
batching-rule system either. It built a **third thing**, which exists because
the demand turned out to be far narrower than "vmap", and it is fenced so that
anything outside that narrowness still refuses.

The assertions behind every claim here live in
`rust/torch_c/pytests/test_vmap.py`.

---

## 0. Answers first

| question | answer | where |
|---|---|---|
| What do the four pass to `vmap`? | **A scalar closure over four index scalars**, nested four deep, `in_dims` one-hot over a 1-D `arange`, `out_dims=0` throughout. | §1 |
| Is `out_dims` ever non-zero? | **No.** `masking_utils.py:348` hardcodes `out_dims=0` at every level. | §1.2 |
| Is a general `vmap` needed? | **No.** For a closure that is pointwise over the mapped indices, `vmap` *is* broadcasting — and transformers ships that identity itself, as the default path. | §2 |
| Did it need a `Repr` arm? | **No, and that is the difference from `docs/kernels/COMPLEX2.md`.** A batched value here is a plain dense tensor with extra size-1 dimensions. No arm, no tag, no kernel, no `aten.*` name. | §3 |
| Does the mask match upstream? | **Bit-identical, every element, four closures, compared against upstream torch in a separate process.** | §3.4 |
| What stops a wrong answer for closures this does not fit? | Three gates, each of which turns red on its own when nullified. | §4 |
| Which of the four clear? | **`t5gemma2` clears. The three ASR ones move to a new wall, `aten.logical_not`,** reached *after* the mask is built and used. | §6 |
| Golden? | **Exactly unmoved: 10691/10691, ops=287.** | §3.1 |
| Is this a `vmap`? | **No, and it must not be described as one.** §5 sizes the real thing. | §5 |

---

## 1. The four closures, which is what the round had to establish first

All four reach `masking_utils.py:962` (`create_causal_mask`) or `:1064`
(`create_bidirectional_mask`), where `use_vmap` is initialised to `False` and
flipped to `True` by exactly one condition: the caller supplied an
`and_mask_function` or an `or_mask_function`. Two call sites do that:

```
models/nemotron_asr_streaming/modeling_nemotron_asr_streaming.py:987
    and_mask_function=chunked_limited_mask_function(*self._resolve_attn_context(...))

models/t5gemma2/modeling_t5gemma2.py:808
    and_mask_function=sliding_window_mask_function(self.config.sliding_window,
                                                   is_causal=False)
```

`nemotron3_5_asr` and `nemotron_asr_streaming_encoder` reach the first of those
through the same encoder, so the four architectures are **two** closures.

### 1.1 What is in them

```python
# nemotron_asr_streaming, chunked-limited context
def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
    q_chunk  = torch.div(q_idx,  chunk_size, rounding_mode="trunc")
    kv_chunk = torch.div(kv_idx, chunk_size, rounding_mode="trunc")
    chunk_diff = q_chunk - kv_chunk
    return (chunk_diff >= 0) & (chunk_diff <= left_context_chunks)

# t5gemma2, non-causal sliding window
def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
    dist = q_idx - kv_idx
    left_mask  = (dist >= 0) & (dist <  left_window_size)
    right_mask = (dist <  0) & (-dist < right_window_size)
    return left_mask | right_mask
```

`and_masks` (`masking_utils.py:48`) then `&`s each with the base
(`causal_mask_function` or `bidirectional_mask_function`) starting from
`q_idx.new_ones((), dtype=torch.bool)`, and `sdpa_mask` (`:504`) `&`s in
`padding_mask_function`, which is `padding_mask[batch_idx, kv_idx]` — advanced
indexing of a real 2-D tensor by two mapped indices.

So the complete operator set reachable inside these closures is: `div`
(rounding_mode trunc), `sub`, `neg`, six comparisons, `&`, `|`, `new_ones`,
`.to(device)`, and two-axis advanced indexing. Every one is pointwise over the
mapped indices. **Nothing reduces, reshapes, concatenates or matmuls.**

### 1.2 The nesting, exactly

```python
def _vmap_expansion_sdpa(mask_function):          # masking_utils.py:339
    dimensions = [(None, None, None, 0), (None, None, 0, None),
                  (None, 0, None, None), (0, None, None, None)]
    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function
```

Four levels. Each `in_dims` is one-hot: a single `0`, the rest `None`. Every
mapped argument is a 1-D `arange` (`masking_utils.py:507-510`), so every mapped
value is a **scalar** — logical rank 0. `out_dims` is 0 at every level, never
anything else. `randomness` is never passed, so it is `torch.vmap`'s default,
`"error"`.

There is no second output, no keyword argument, no pytree structure beyond a
flat four-tuple, and no chunking.

---

## 2. `vmap` over that shape is broadcasting, and transformers says so

`masking_utils.py:352` is `_non_vmap_expansion_sdpa`, and it is the **default**
path (`:514`, "Option 1: Fast non-vmap mask creation"):

```python
batch_indices = batch_indices[:, None, None, None]
head_indices  = head_indices [None, :, None, None]
q_indices     = q_indices    [None, None, :, None]
kv_indices    = kv_indices   [None, None, None, :]
```

with its own docstring: *"Allows the usage of any index-based mask function
without relying on vmap. NOTE: This is limited to index based functions only."*

That is the identity this round rests on, so it was measured rather than
assumed — **on upstream torch, in its own process**, for all four closure
configurations `test_vmap.py` uses:

    upstream, chunked_limited              vmap == broadcast   (all 162 elements)
    upstream, sliding_window_bidirectional vmap == broadcast
    upstream, sliding_window_causal        vmap == broadcast
    upstream, chunked_limited_no_padding   vmap == broadcast

`test_vmap_and_the_broadcast_expansion_agree_on_both_sides` asserts it on
upstream as well as on the shim, deliberately: if it stopped being true
upstream, this shim would be faithfully reproducing the wrong thing.

The reason it holds is mechanical. Give level `L` tensor dimension `L-1` and
pad every batched value out to a fixed rank with size-1 dimensions; then
right-aligned broadcasting lines the levels up, an elementwise op composes them
correctly, and `out_dims=0` asks for the mapped dimension to be first — which
is where it already is. `_remove_batch_dim` becomes bookkeeping.

**It holds only for logical rank 0.** A batched value with a shape of its own
would need dimensions to sit in, and every dimension is spent on a level. That
is not a limitation to work around later; it is why §4.1 refuses a 2-D
`in_dims` input by name.

---

## 3. What was built

`rust/torch_c/src/bootstrap.py`, in the functorch section, next to the dynamic
layer stack that was already there. Four names that were raising stubs:

    _vmap_increment_nesting     pushes a level, after checking §4.1
    _vmap_decrement_nesting     pops it
    _add_batch_dim              reshape to (1,)*(L-1) + (n,) + (1,)*(RANK-L)
    _remove_batch_dim           expand the level's slot, hand it back

`_VMAP_MAX_RANK` is 8; a fifth nesting past that refuses rather than colliding.
`maybe_get_level` and `unwrap_if_dead` were widened to tolerate a stack of these
interpreters and still answer `-1`/unchanged — which is not a weakening but the
same derivation they already used: **a tensor can only carry a level if
something wrapped it, and nothing here is a wrapper.**

### 3.1 Not a `Repr` arm, which is the difference from `docs/kernels/COMPLEX2.md`

The brief for this round expected a batched dimension to be the same shape of
problem as `Repr::Complex` — 19 `match Repr` sites, 14 demanded by the
compiler, a central refusal that ~400 call sites inherit, and nullifying that
refusal made 6 of 10 sampled ops compute silently.

It is not the same shape of problem, and the reason is worth stating because it
is what made the round cheap: **a batched value here is a plain dense tensor.**
It has no tag, no arm, no storage that candle cannot hold, and no `aten.*` name.
`tensor.rs` was not touched, `aten.rs` was not touched, and no kernel or
dispatch hook was added anywhere.

That is why golden is **exactly unmoved at 10691/10691 cases and ops=287**, and
`test_vmap_landed_without_a_kernel_or_a_repr_arm` asserts the premise (no
`aten.*` name containing `vmap` or `batch_dim` in either the advertised or the
parked list) rather than the consequence.

### 3.2 The frame requirement, which is not a trick

`_vmap_increment_nesting` refuses unless it is called from
`torch/_functorch/vmap.py:_flat_vmap`. That is not a way of dodging the
refusal tests in `test_shim.py` and `test_tail2.py` — it is what §4.3 needs.
The self-check re-runs the closure, and the closure and its batched arguments
live in that frame (`func`, `batched_inputs`). A call from anywhere else has
nothing to check against, so it says so and raises.

The side effect is that `test_shim.py`'s repr-predicate fixture and
`test_tail2.py:527` are **unchanged by this round and still green**: both call
the primitives by hand, and by hand they still refuse. That was checked, not
assumed.

### 3.3 One defect the implementation had to fix on the way

`torch/_functorch/vmap.py:487` is a context manager whose `finally` calls
`_vmap_decrement_nesting()` even when the increment raised. So a refusal was
immediately followed by an unpaired decrement, and the decrement's own
`RuntimeError` **replaced** the refusal — measured: `randomness="different"`
reported *"decrement with no level on the stack"* and never mentioned
randomness. A refused increment now leaves a debt that the decrement pays. The
debt is only recorded when the call came from `_flat_vmap`; recording it for a
hand call made the *next* genuine decrement pay it and leaked a live level, and
`test_no_level_is_left_on_the_stack_by_a_refusal` is what caught that.

### 3.4 The bar: bit-identical to upstream

Four closure configurations, `B=3, H=1, Q=6, KV=9`, `q_offset=2`, with a
padding mask that is False in two places. Shim in the vendored tree, upstream
in a clean process, both dumping every element:

    chunked_limited                 162/162 elements equal, dtype bool
    sliding_window_bidirectional    162/162
    sliding_window_causal           162/162
    chunked_limited_no_padding      162/162

`test_the_vmapped_mask_is_bit_identical_to_upstream` also asserts that each
mask is **neither all-True nor all-False**, because element-wise equality
against a constant proves nothing.

---

## 4. The three gates, and what nullifying each one puts back

This is the part that matters, because the failure mode is a plausible wrong
answer. Each gate was nullified and the suite re-run.

### 4.1 The shape gate — entry

`_add_batch_dim` requires a 1-D tensor and `in_dim == 0`. `_remove_batch_dim`
requires `out_dim == 0`, one output, `randomness == "error"`, a rank within
budget, every level slot equal to that level's batch size or 1, and every
dimension past the levels equal to the already-unwrapped sizes.

    nullify the level-slot half   -> cat_widens_a_level_slot stops being a
                                     NotImplementedError and becomes a raw
                                     candle broadcast error
                                     ("cannot broadcast [2,1,6,9,...] to
                                      [3,1,6,9,...]")  -- test red
    nullify the tail half         -> cat_adds_a_logical_dimension the same
                                     ("cannot broadcast [1,1,1,9,1,1,1,108]")
                                     -- test red

So the shape gate's job is to turn "candle could not broadcast" into an
explanation of what the closure did. It is a **precondition for §4.3**, not a
second opinion on it.

### 4.2 What the shape gate structurally cannot see

A closure can read *across* a mapped dimension while leaving the shape exactly
as it was. `cumsum`, `flip`, `sort`, `roll`, and a `reshape(-1)[0]` all do.
Upstream `vmap` has batching rules for these and answers correctly; a broadcast
representation answers something else, of the right dtype and the right shape.

That is precisely the class of error `docs/kernels/COMPLEX.md` §7 was worried about, and
no shape check can be the gate for it.

### 4.3 The value gate — the definition of `vmap`, re-run

At the innermost `_remove_batch_dim`, the closure is re-run on eight sampled
index points (both corners plus six deterministic interior points) with plain
0-d scalars and **no levels on the stack**, which is what `vmap` *means*:

    chk[i1, i2, i3, i4]  ==  f(a1[i1], a2[i2], a3[i3], a4[i4])

Any disagreement raises. The closure and its arguments come from the
`_flat_vmap` frame (§3.2); if they cannot be found, that is also a refusal.

    nullify it -> the probe closure `(b + q).reshape(-1)[0] >= 3`, which is the
                  identity on the 0-d tensor vmap promises and a *constant*
                  under broadcasting, stops raising and returns
                  `Tensor(3, 1, 6, 9)` -- a mask of exactly the right shape and
                  dtype, and wrong. Test red.

That nullification result is the document. It is the failure `docs/kernels/COMPLEX.md`
§7 predicted, reproduced on demand, and the gate that stops it.

Eight points is a sample, not a proof. It is a strong one — an op that mixes
across a level almost never agrees with the definition at both corners and six
interior points at once — but §5 is what a proof would cost.

---

## 5. What a real `vmap` would be, since this is not one

There is no batching rule here. There is no rule for `sum`, none for `matmul`,
none for anything: there is one representation that happens to be correct for
closures that are pointwise over the mapped indices, plus three gates that
refuse when it is not.

A real `vmap` is a per-operator batching-rule system: for each `aten` op, a
function that takes operands carrying a batch dimension at an arbitrary
position and produces a result carrying one, without materialising the loop.
Upstream's is `BatchRulesReduceOps.cpp` and friends — roughly 30 files. Here it
would additionally need a wrapper tensor type (`_maybe_remove_batch_dim` does
`isinstance(x, torch.Tensor)`, so it would have to be a `Tensor` subclass) and
therefore a `Repr` arm after all, with the `docs/kernels/COMPLEX2.md` audit that
implies.

**Nothing in `docs/architectures/ARCH200.md` demands it.** The demand is two closures over
four index scalars. Building the general thing to serve those would be the
mistake `docs/numerics/INT8.md` names: pricing a fork of the world to close a gap that a
fence closes.

If a future architecture arrives with a closure this refuses, the refusal will
name what it did (`test_vmap.py`'s probes are the catalogue), and the decision
about batching rules can be made then, with that closure in hand.

---

## 6. Which architectures clear

`pytests/arch_sweep.py --only nemotron3_5_asr nemotron_asr_streaming
nemotron_asr_streaming_encoder t5gemma2`, shim side, after this round:

| architecture | before | after |
|---|---|---|
| `t5gemma2` | `_vmap_increment_nesting` | **ok** |
| `nemotron3_5_asr` | `_vmap_increment_nesting` | `TensorBase.logical_not` |
| `nemotron_asr_streaming` | `_vmap_increment_nesting` | `TensorBase.logical_not` |
| `nemotron_asr_streaming_encoder` | `_vmap_increment_nesting` | `TensorBase.logical_not` |

**One of four clears; the vmap wall is gone from all four.** The three ASR
models now build their mask, correctly, and then fail one line later consuming
it:

```
modeling_nemotron_asr_streaming.py:628
    matrix_bd = matrix_bd.masked_fill_(attention_mask.logical_not(), float("-inf"))
```

`aten.logical_not.default` does not exist in this shim — `~` (`bitwise_not`)
does, and answers correctly for bool, but `logical_not` has no kernel and no
spelling. That is an `aten.rs` item, deliberately not taken in this round
(`aten.rs` had two other rounds in it), and it is small: the `bool` case is
`bitwise_not` and the rest is `== 0`. Adding it moves golden's op count, which
this round's gate pinned, so it belongs to whichever round owns that number.

It is worth being exact about what this means for `docs/architectures/ARCH200.md`'s
arithmetic: **the 4 architectures blocked on `_vmap_increment_nesting` become 1
forward and 3 blocked on a new, much smaller operator.** Counting this round as
"four unblocked" would be the §5.3 error from `CLAUDE.md` — a number that went
up without the thing behind it going up as far.

---

## 7. What this round did not do

- No `aten.*` kernel, no dispatch hook, no `Repr` arm, no `tensor.rs` change.
- No change to `test_shim.py` or `test_tail2.py`. Their refusal assertions are
  about hand calls, which still refuse (§3.2), so they are still true.
- `_wrap_for_grad`, `_to_functional_tensor` and the rest of functorch are
  untouched. `grad`, `jvp` and `functionalize` still refuse.
- `torch.vmap` with `chunk_size`, with a non-flat pytree of inputs, with
  multiple outputs, with `out_dims=None`, or over anything that is not a
  scalar closure: all refuse.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _VMAP_MAX_RANK present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _vmap_self_check present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _VmapBroadcastInterpreter present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vmap.py test_the_vmapped_mask_is_bit_identical_to_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vmap.py test_a_closure_that_reads_across_the_batch_dimension_is_refused present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vmap.py test_vmap_landed_without_a_kernel_or_a_repr_arm present -->
<!-- DOCWATCH: count golden_ops_covered ge 287 -->
<!-- DOCWATCH: count golden_cases_passed ge 10691 -->
