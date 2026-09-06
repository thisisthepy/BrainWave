# INDEXSEL -- ten names, two kernels this shim already had a shape for

Worktree `work/indexsel` on develop `eb84708`. torch 2.13.0 upstream, `transformers`
5.15.1. Assigned the eleven-op slice of docs/ARCH100.md's ranked tail:
`TensorBase.index_select` (5 architectures), `aten.argsort` (3), `aten.where.Scalar` (2),
`TensorBase.reshape_as`, `TensorBase.unflatten`, `TensorBase.chunk`, `aten.new_full`,
`aten.multiply`, `aten.logical_and`, `aten.diff` (1 each).

## 1. Alias versus kernel -- the split ARCH100 asked for

**None of the eleven were name-only aliases for an existing kernel.** Every one
needed a real dispatch arm in `aten.rs`; none could be routed to a sibling's kernel
unchanged. Two, though, turned out to be pure Python-level *routing* to arithmetic
this shim already had:

```text
real kernel, new arithmetic     index_select, argsort (+.stable), where.Scalar,
                                 new_full, unflatten, chunk (free-function), diff,
                                 logical_and                              -- 9
alias to an existing kernel     multiply.Tensor / multiply.Scalar -> mul's own
                                 arith_tensor/arith_scalar, unchanged      -- 1 op,
                                                                              2 keys
not a kernel at all             reshape_as -- see §3                      -- 1
```

`multiply` is the one true alias: a `TorchDispatchMode` logger over
`torch.multiply(a, b)` records exactly `aten.mul.Tensor` (`.Scalar` likewise), and its
dispatch arm is two lines reusing `arith_tensor`/`arith_scalar` with `Arith::Mul` --
the same function `aten.mul.Tensor` itself calls. Nothing else was a same-shape
alias. `chunk` (the free-function spelling) *looks* like it should route to
`TensorBase.chunk`'s existing Python-level composite in `bootstrap.py`
(`_install_tensor_chunk`), and the two share the exact `at::native::chunk` arithmetic
-- but `bootstrap.py` was out of territory for this round, and `torch.chunk`'s
overload table key (`aten.chunk.default`) is genuinely different Python surface
from `Tensor.chunk`'s (methods.json vs overloads.json, docs/C_SURFACE.md's
distinction), so the arithmetic is repeated in `chunk_default` rather than shared.

"Eleven ops implemented" would overstate this round if read as eleven kernels: it
is nine kernels, one alias, and one op parked as not-implementable in this
worktree's territory (below).

## 2. `TensorBase.reshape_as` has no `torch.ops.aten` entry

Measured with a `TorchDispatchMode` logger: `x.reshape_as(y)` on 2.13.0 fires
exactly one record, `aten.view.default`, with `y`'s shape as the argument.
`reshape_as` is a C++-level method (`THPVariable_reshape_as`) that computes a shape
and calls `reshape`/`view` itself -- there is no `aten::reshape_as` schema for
`tools/golden/loader.py`'s `resolve_torch_overload` to find, and it refuses by
design when one is missing rather than guessing.

The correct home for this is `bootstrap.py`'s Python-level surface -- the same
place `chunk`, `flatten`, and `T` already live for the identical reason
(`methods.json`'s README: "a `methods.json` entry would name a key no dispatcher
ever sees"). `bootstrap.py` is outside this worktree's territory
(`rust/torch_c/src/aten.rs`, `methods.json`, `overloads.json`,
`tools/golden/cases.py`, `pytests/test_indexsel.py`), so it was not touched.

What *is* in territory, and what this round did instead: `methods.json` carries an
entry for `reshape_as` whose "schema" is this shim's own invention --
`aten::reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)` -- written only so the
table-driven resolver can bind `self`/`other` the way it binds every other method,
dispatched to `reshape_as_default` in `aten.rs`, which calls the same
`resolve_shape` machinery `view`/`reshape` use. This computes the right answer and
is advertised through `_aten_dispatch`, but it is **not** in `_aten_implemented()`
-- it is in `IMPLEMENTED_AWAITING_GOLDEN`, with a comment explaining why
`compare.py`'s per-op golden loop structurally cannot reach it (no
`torch.ops.aten.reshape_as` to resolve `torch_call` against). It is proven against
upstream instead in `pytests/test_indexsel.py`, through the full vendored `torch`
package, each side in its own process (§4).

**This is the two-part answer §3.5.2 of this repository's own retrospective warns
about** ("If 'my way is better,' say it, don't build it"): the *correct* structural
home for `reshape_as` is a `bootstrap.py` composite, which this round's territory
excludes. Rather than build a competing structure inside `aten.rs`/`methods.json`
that pretends to be the real thing, the entry is explicitly marked as not
golden-covered and the reason is written down here and in the source, so the next
round that touches `bootstrap.py` can either move it there or confirm the invented
schema is fine to keep.

## 3. Traps checked against upstream on their edge cases, not the happy path

* **`index_select` on a 0-d `self`.** Not refused -- upstream treats it as rank-1
  with one element: the index must carry exactly one value
  ("Index to scalar can have only 1 value, got N value(s)"), checked *before* the
  range check, and the answer is `self` unchanged. Out-of-range indices raise
  unconditionally, with no negative-index wraparound (matching `index_add_`'s rule,
  not `index_put_`'s).
* **`argsort` ties.** `nllb_moe` sorts `importance_scores` (real floats, no ties in
  practice) but the golden cases use tied inputs deliberately -- an unstable sort
  cannot be told apart from a stable one on distinct values. Measured: plain
  `argsort` (no `stable=`) already agrees with `stable=True` in both ascending and
  descending order, including on ties, so `argsort_stable` reads and discards
  `stable=False` too (`order_along`, `sort`'s own helper, is unconditionally
  stable) -- within upstream's licence, not a divergence from it.
* **`where.Scalar`'s dtype table is not `add`'s.** Measured directly rather than
  inferred from `where.ScalarOther`'s existing rule: `bool+bool -> bool`,
  `int+bool -> int64`, anything with a Python `float` on either side ->
  `float32`. `docs/SCALAR2.md`'s warning held: the rule is per-op.
* **`diff`'s `n=0` ignores `prepend`/`append` entirely** -- measured, not assumed.
  `torch.diff(x, n=0, prepend=y)` is `x` unchanged, not the concatenation. `n<0`
  raises upstream's own wording. `voxtral_realtime_encoder`'s call
  (`torch.diff(position_ids, prepend=first_dummy_value, dim=-1)`) uses `n=1`; the
  golden cases additionally cover `n=2` with a 2-element `prepend` and a 2-D
  `dim=0` case to pin the "prepend/append concatenate once, then `n` first-order
  differences run over that" algorithm rather than "diff `n` times off `self`'s own
  axis, extended by less each round."
* **`new_full`'s dtype comes from the receiver, not from the fill value.**
  `int64_t.new_full((2,), 7.5)` is `7`, not `7.5` -- `full_like`'s rule, not
  `full`'s (which infers dtype from the fill and would keep `7.5`). Pinned with a
  golden case.
* **`chunk`'s `chunks` argument is an upper bound, not a promise.**
  `arange(3).chunk(7)` returns three pieces, not seven; `empty(0).chunk(3)` returns
  three empty pieces (the one case where the promise *does* hold, because "an
  arbitrary number of empty chunks sums to zero all the same" -- upstream's own
  comment, transcribed in `_install_tensor_chunk` and repeated in `chunk_default`'s
  own doc comment since the arithmetic itself is repeated, §1). Both are golden
  cases; a plausible even-division implementation gets the first wrong.
* **`unflatten`'s `-1` inference and its refusals.** At most one `-1` in `sizes`
  ("only one dimension can be inferred", upstream's own wording, shared with
  `view`); a mismatched product refuses with upstream's exact phrasing
  ("Provided sizes [...] don't multiply up to the size of dim N (M) in the input
  tensor"). `siglip_vision_model` reaches this through the vendored
  `torch/functional.py`'s `_in_projection_packed`, which calls
  `kv_proj.unflatten(-1, (2, E))` -- a real method dispatch (`torch/_tensor.py`
  forwards `Tensor.unflatten` straight to `super().unflatten`, measured), not a
  Python composite the way `reshape_as` is, which is why this one *can* live in
  `methods.json` with a genuine schema.
* **`logical_and` is truthiness, not a bit-and.** `1 & 2` bitwise is `0` (falsy)
  while both operands are truthy -- `torch.logical_and(1, 2)` is `True`, measured,
  and the golden cases pin a case that would catch a bitwise-AND implementation
  silently returning the wrong (falsy) answer on a truthy pair. Reused `any_from`,
  the same "nonzero including NaN" mask `any`/`all` already compute, rather than
  writing a third truthiness rule.

No candidate alias turned out to *not* be one on closer inspection (the `prims.transpose`
vs `aten.permute` trap docs/PRIMS.md found does not recur here) -- the one candidate
alias, `multiply`, held on every case tried, including the promoting
`float32 x int64` case that would have caught a `multiply` wired to a
non-promoting copy of `mul`.

## 4. What the suite shows

* `rust/torch_c/pytests/run.sh` (every `test_*.py`, including the new
  `test_indexsel.py`): full pass after one fix -- `test_core_ops_and_op_tags_agree`'s
  pinned `tag_core_count` moved from 110 to 112 (two of the eleven new keys,
  `index_select.default` and `logical_and.default`, are `core` upstream; the other
  nine are not, read off each op's own `.tags` rather than inferred -- see the
  updated comment block in `test_shim.py` for the per-op table). This is the one
  permitted edit to that file, with the arithmetic that keeps it a check.
* `tools/golden/compare.py`: all ten real-kernel keys (`index_select.default`,
  `argsort.default`, `argsort.stable`, `where.Scalar`, `new_full.default`,
  `unflatten.int`, `chunk.default`, `diff.default`, `multiply.Tensor`,
  `multiply.Scalar`, `logical_and.default` -- eleven keys, ten distinct ops since
  `multiply` is two overloads) are golden-compared, and the whole suite --
  including the new cases -- clears **9183/9183 cases, ops covered=235**
  (up from the 8921/8921, ops=222 baseline this round started from: +262
  cases, +13 ops -- eleven new keys plus the two `multiply` overloads and
  `argsort` counted once each, per `_aten_implemented()`'s own count).
  `--self-test` also passes (19 comparators x 11 fault modes, 0 problems): the
  first draft of `chunk_default_cases`' own comparator checked shape and
  values through `.tolist()` alone and missed a dtype-corruption fault
  because `.tolist()` does not carry dtype -- caught by the self-test, not by
  a human reading the case, and fixed by reusing `split_cases`' own
  `_chunk_list_check` instead of writing a second dtype-blind comparator.
  `reshape_as.default` is deliberately excluded from the per-op loop (§2) and
  proven in `test_indexsel.py` instead, subprocess-isolated against upstream.
* `pytests/arch_sweep.py --one <model_type>`, run individually (never as a full
  sweep -- its own module docstring forbids wiring it into `run.sh`) against the
  shim build from this round, for every architecture ARCH100 named across all
  eleven ops, **including `longt5`/`convbert`** rather than skipping them as
  "not separately assigned":

  | architecture | wall before this round | after this round |
  |---|---|---|
  | m2m_100 | `TensorBase.index_select` | **forwards clean** (`status: ok`) |
  | nllb-moe *(ARCH100 spells it with a hyphen; `nllb_moe` is not a valid `--one` key)* | `TensorBase.index_select` | **forwards clean** |
  | xglm | `TensorBase.index_select` | **forwards clean** |
  | kosmos-2.5 | `TensorBase.index_select` | **forwards clean** |
  | seamless_m4t_v2 | `TensorBase.index_select` | **forwards clean** |
  | vit_mae | `aten.argsort` | **forwards clean** |
  | aria | `aten.argsort` | argsort's own wall is gone; a *second* wall appears immediately -- `torch.bucketize` (ARCH100's own §2 list, `idefics3_vision`'s wall too, unrelated to this round) |
  | aria_text | `aten.argsort` | argsort's own wall is gone; second wall is `TensorBase.index_copy_` |
  | git | `aten.where.Scalar` | **forwards clean** |
  | cpmant | `aten.where.Scalar` | where.Scalar's own wall is gone; second wall is a candle backend limitation (`aten.embedding.default: candle: unsupported dtype F32 for op index-select`), not a missing op |
  | roformer | `TensorBase.reshape_as` | **forwards clean** |
  | siglip_vision_model | `TensorBase.unflatten` | **forwards clean** |
  | diffllama | `chunk` (free-function) | chunk's own wall is gone; second wall is `torch.rms_norm` |
  | led | `TensorBase.new_full` | new_full's own wall is gone; second wall is an `overloads.json` gap in `div`'s own table (unrelated pre-existing gap) |
  | voxtral_realtime_encoder | `aten.diff` | **forwards clean** |
  | longt5 | `aten.logical_and` | logical_and's own wall is gone; second wall is `torch.einsum` with an ellipsis pattern |
  | convbert | `aten.multiply` | multiply's own wall is gone; second wall is `torch._C._nn.im2col` |

  **10 of the 17 architectures swept clear end to end**: all five
  `index_select` architectures (`m2m_100`, `nllb-moe`, `xglm`, `kosmos-2.5`,
  `seamless_m4t_v2`), one of the three `argsort` ones (`vit_mae`), one of the two
  `where.Scalar` ones (`git`), and `roformer`, `siglip_vision_model`,
  `voxtral_realtime_encoder`. The remaining seven (`aria`, `aria_text`, `cpmant`,
  `diffllama`, `led`, `longt5`, `convbert`) hit a *second* wall immediately behind
  the first, which docs/ARCH100.md §1 predicted as the expected shape ("closing
  the top entry reveals whatever stands behind it") rather than a failure of this
  round's kernels: each was confirmed to get past the exact traceback line
  recorded before this round (`torch_c_bootstrap.py` no longer raises
  `NotImplementedError` at that call site), and the traceback for every one of
  the seven moved to a **different** op entirely, none of which is in this
  round's eleven.

`arch_sweep.py` itself was **not** wired into `run.sh` (its own module docstring
forbids that) and was run by hand, once per architecture, outside the suite.
