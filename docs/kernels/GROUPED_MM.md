# `_grouped_mm` — the offset-based grouped GEMM, and what it took to close Mixtral

Mixtral was the one architecture of twenty that did not reach zero missing operators.
`docs/kernels/OPS4.md` §13.3 left `aten._grouped_mm.default` out of scope deliberately and said so, and
the README has carried "19 of 20 — Mixtral needs `_grouped_mm` alone" ever since. This document
is the record of implementing it: what upstream's schema and semantics actually are in torch
2.13.0, what candle supplied and what had to be written, and — the part that matters more than
the operator — whether the "alone" in that sentence was true.

Written incrementally while the work happened, so the order below is roughly the order things
were found.

---

## 1. The schema, from the vendored tree

`torchnative/src/main/torchgen/packaged/ATen/native/native_functions.yaml:7026`:

```yaml
- func: _grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor
  variants: function
  dispatch:
    CompositeExplicitAutograd: _grouped_mm
    CUDA: _grouped_mm_cuda
```

Confirmed against the installed upstream at run time — `torch._C._get_schema('aten::_grouped_mm', '')`
returns the identical string. There is **one** overload (`default`); no `.out`, no `.dtype`.

Two things follow from the `dispatch:` block and neither is obvious:

- **`CompositeExplicitAutograd` means there is a real CPU kernel.** This is not a CUDA-only op
  that we would have to define the meaning of ourselves. Everything in §2 was measured by
  calling it on this machine, and every golden case compares against that kernel.
- **The meta registration is not the CPU contract.** `torch/_meta_registrations.py:8611`
  (`_meta_grouped_mm_common`) checks `mat_a.dtype == torch.bfloat16 and mat_b.dtype == torch.bfloat16`
  and refuses everything else. The CPU kernel accepts **Float32, BFloat16 and Float16**. Reading
  the meta function as the specification — which is the natural thing to do, since it is the only
  readable implementation in the vendored tree — would have made us refuse `float32`, which is
  precisely the dtype Mixtral calls it with. Measured, not inferred.

---

## 2. Semantics, measured on torch 2.13.0 CPU

`self` and `mat2` may each be 2-D or 3-D, which gives four layouts. `offs` is required when
either operand is 2-D and forbidden when both are 3-D.

| `self` | `mat2` | `offs` partitions | output |
|---|---|---|---|
| `(M,K)` | `(G,K,N)` | the **rows** of `self`, length `G` | `(M,N)` |
| `(G,M,K)` | `(K,N)` | the **columns** of `mat2`, length `G` | `(M,N)` |
| `(M,K)` | `(K,N)` | the **contraction** `K`, length `G` | `(G,M,N)` |
| `(G,M,K)` | `(G,K,N)` | forbidden — plain `bmm` | `(G,M,N)` |

`offs` is a **cumulative end index**, not a length. Group `g` covers `[offs[g-1], offs[g])` with
`offs[-1]` read as `0`. Each of the four layouts was checked against a hand-written slice-and-`mm`
reference and agreed exactly (`torch.equal`, not a tolerance).

### 2.1 What upstream refuses, verbatim

Every message below is upstream's own, reproduced rather than paraphrased, because it is the work
item a caller reads.

| condition | message |
|---|---|
| rank not 2 or 3 | `mat_a has to be 2 or 3d` / `mat_b has to be 2 or 3d` |
| dtype not f32/bf16/f16 | ``Expected mat_a to be Float32, BFloat16 or Float16 matrix, got Double`` |
| operand dtypes differ | `expected m1 and m2 to have the same dtype, but got: c10::BFloat16 != float` |
| `offs` missing with a 2-D operand, or present with two 3-D operands | `Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d` |
| `offs` not `int32` | `Offsets have to be int32` |
| `offs` not 1-D | `offs has to be 1D` |
| `bias` given | `Bias not supported yet` |
| `out_dtype` != `mat_a.dtype` | ``Grouped gemm output dtype must match `mat_a` dtype`` |
| `offs` length != the 3-D operand's batch | `matrix batch sizes have to match` |
| 3-D × 3-D batch extents differ | `batched dimension has to match` |
| contraction extents differ (when at least one operand is 3-D) | `contraction dimension of mat_a and mat_b must match` |
| operand strides unaligned — §2.2 | `strides should be multiple of 16 bytes` |

`bias` is in the schema and is **not implemented upstream at all**: any non-`None` value is
refused. So is `out_dtype` other than the input dtype — the schema allows a `ScalarType?` but the
kernel accepts only the identity. Both are refused here for the same reason and with the same
words.

Note the 2-D × 2-D row of the table that is *absent*: with both operands 2-D there is **no**
contraction check. `_grouped_mm((8,8), (4,4), offs=[2,4])` computes, because `offs` slices both
operands with the same range and the parts outside that range are never read. Measured.

### 2.2 The 16-byte stride rule, and why we reproduce it

The CPU kernel refuses operands whose last-two-dimension strides are not a multiple of 16 bytes:

```
_grouped_mm((8,3) f32, (3,3,4) f32, offs=[2,5,8])
    RuntimeError: strides should be multiple of 16 bytes
```

The predicate is the one `check_valid_strides` in `_meta_registrations.py` spells out, and it was
re-derived here by sweeping shapes rather than copied:

```
alignment = 16 / itemsize          # 4 elements for f32, 8 for bf16/f16
if stride[-2] == 1 and stride[-1] >= max(1, shape[-2]):   check stride[-1] % alignment == 0
elif stride[-1] == 1 and stride[-2] >= max(1, shape[-1]): check stride[-2] % alignment == 0
else:                                                     refuse
```

Only the last two strides are examined. A 3-D operand's *batch* stride is not checked (a batch
slice with a padded batch stride is accepted), and — measured on 2.13.0 — the **data pointer** is
not checked either, despite `transformers`' `_can_use_grouped_mm` guarding against exactly that
for torch ≤ 2.10.

**We reproduce this.** It is not a constraint our kernel has; candle would happily multiply a
`(8,3)` by a `(3,3,4)`. But it is part of what this operator *is* on CPU in this torch version,
and `transformers` has a whole fallback path (`torch.ops.transformers.grouped_mm_fallback`) whose
only purpose is to catch programs that would hit it. A shim that computes where upstream raises
is the silent-divergence direction DESIGN.md §5 exists to keep out, and it would also make every
small hand-checkable golden case a divergence rather than a comparison.

### 2.3 Rows nobody writes, and offsets that go backwards

Two behaviours that a `cat`-of-blocks implementation gets wrong, both measured:

- **`offs[-1] < M` leaves the tail of the output unwritten.** Upstream returns whatever
  `torch.empty` gave it. `transformers` relies on this on purpose — the expert-parallel sentinel
  comment in `integrations/moe.py` says "grouped_mm skips rows beyond `offsets[-1]`" and masks
  those rows itself. We fill them with zeros, which is *a* valid answer to an uninitialised
  question but not a comparable one, so no golden case asserts on that region.
- **`offs` is not required to increase.** `offs=[9,5,24]` on `M=24` computes rows `0..9` for
  group 0, nothing for group 1 (`5 <= 9`), then rows `5..24` for group 2 — *overwriting* rows
  `5..9`. The kernel is a sequential write loop and a later group wins. Reproduced exactly; see
  §3.2.

---

## 3. The implementation

### 3.1 What candle gave

`candle_core` supplies `narrow`, `matmul`, `cat`, `stack`, `Tensor::zeros`, and — via the
existing `gemm_with_layout_fallback` — a multiply that consumes a transposed operand without
copying it. That last one is load-bearing rather than an optimisation: Mixtral's `mat2` is
`weight.transpose(-2, -1)`, a non-contiguous view, on every call.

What candle does **not** give is any notion of a grouped or ragged GEMM, so the group walk is
written here. `slice_assign` exists but was rejected: it is implemented as
`pad_with_zeros` + `where_cond` over the *whole* output per group, so an eight-expert layer would
build eight full-size masks.

### 3.2 The group walk

The kernel simulates upstream's sequential write loop over an owner map rather than concatenating
per-group blocks, which is what makes §2.3 come out right:

```
owner = [None] * extent
prev  = 0
for g, end in enumerate(offs):
    if end > prev: owner[prev:end] = g
    prev = end
```

The owner map is then compressed into maximal runs, and one `matmul` is issued per run. For the
ordinary monotonic case the runs are exactly the groups, so this costs nothing; for the
pathological case it produces upstream's answer without computing anything twice, because every
row of a 2-D × 3-D product depends only on its own row of `self`.

`None` runs — the unwritten tail — become zeros.

Accumulation goes through the same `gemm_accumulate_in` that `mm`/`bmm` use, so `float16`
accumulates in `float32` exactly as upstream does. (`bfloat16` could not be discriminated: a
`K=256` product agrees bit-for-bit with both a `float32`-accumulated and a `bfloat16`-accumulated
reference.)

Out-of-range offsets (negative, or beyond the extent) are clamped. Upstream reads out of bounds
there, which is undefined rather than a behaviour to match, so nothing asserts on it.

---

## 4. Reaching it from Python

`torch._grouped_mm` is already in `surface.json`'s `varfns`, so the name resolved before this
change — to a refusal, because `overloads.json` had no entry and
`_torch_level_function` will not guess an overload. The entry added is the single schema from §1.

That is the route `transformers` actually takes. `integrations/moe.py::_grouped_mm` prefers
`torch.nn.functional.grouped_mm`, which is vendored Python (`torch/nn/functional.py:7139`) and
whose body is `torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)` — so
the call arrives with three keyword arguments, two of them `None`.

---

## 5. Mixtral — the claim, tested

The README says Mixtral needs `_grouped_mm` **alone**. Half of that is true and half is not, and
the difference is worth stating precisely because the two halves were measured differently.

### 5.1 At the operator level, the claim holds exactly

The bar the other nineteen architectures were measured against (`docs/architectures/ARCH.md` §0,
`docs/kernels/OPS4.md` §3) is: trace a real `transformers` model on **upstream** torch with a
`TorchDispatchMode`, and diff the ops it dispatches against `_C._aten_implemented()` read out of
the built artefact. Mixtral, 2 layers · hidden 64 · heads 2 · 4 experts · top-2:

```
before   mixtral: ops=41  missing=1     aten._grouped_mm.default  x4
after    mixtral: ops=41  missing=0
after    mixtral: ops=61  missing=0     (greedy `generate`, 4 tokens)
```

One operator, four calls per forward — two per layer, the up-projection and the down-projection.
Its arguments as traced, which is what the golden cases were shaped from:

```
mat_a (16,64)  float32  stride (64,1)  contiguous
mat_b (4,64,256) float32 stride (16384,1,64)  NOT contiguous -- weight.transpose(-2,-1)
offs  (4,)     int32    [5, 10, 10, 16]   <- group 2 is EMPTY
```

Uneven groups, an empty group, and a transposed right operand, all in the first call of a
four-expert toy model.

### 5.2 Running it, and what that took

The operator sweep is a coverage measurement, not an execution. So Mixtral was also *run* on the
shim — `AutoModelForCausalLM.from_config`, forward, and greedy `generate` — and compared against
upstream with identical weights (deterministic, RNG-free, written by upstream and loaded through
the shim's own `torch.load`):

```
shim logits  [1, 8, 100] float32
MAX ABS DIFF vs upstream: 2.384e-07   over 800 logits, logit scale 7.494e-01
generate(max_new_tokens=4)  completes
```

That is `float32` GEMM noise, and it is the strongest statement available: the same weights and
the same input produce the same answer through a `_grouped_mm` that never sees upstream's kernel.

**It needed five things beyond the operator, and none of them is an operator.** Every one is a
*name* that does not resolve to the kernel already sitting behind it. They are listed in §6 with
the kernel each one needs and the file each fix belongs in; for the run above they were patched in
the test script, not in the repository.

---

## 6. What blocked Mixtral, by name — **all of it is closed** (2026-08-30)

Measured by probing each name against the built shim (`torch.ops.aten._grouped_mm.default` works;
everything below is the surface above it). Ordered by where the fix belongs. §6.1 and §6.4 were
open when this section was written and are closed now; §9 has the run that proves it, unpatched.

### 6.1 `bootstrap.py` — one predicate, and it is the `_grouped_mm` one — **fixed**

```python
overloads = {
    name: _Overloads(name, schemas, _checker_source)
    for name, schemas in json.loads(overloads_json).items()
    if not name.startswith("_")  # the table's embedded README
}
```

The comment said `_README`; the predicate said *every underscore-prefixed key*. So an aten op
whose name began with `_` could never get a `torch.<op>` binding, and `torch._grouped_mm` — which
is what `torch.nn.functional.grouped_mm` calls, which is what `transformers` calls — refused with
"overload resolution has no table entry for this op" even though the entry was right there in
`overloads.json`.

The sibling comprehension six lines below, for `methods.json`, already spelled the intent
correctly as `startswith("_README")`. Narrowing this one to match was the whole fix.

**What the widened predicate now admits, enumerated.** The two predicates differ on exactly one
key in the table as it stands:

| key in `overloads.json` | old predicate | new predicate |
|---|---|---|
| `_README` (a list of prose, not schemas) | excluded | excluded |
| `_grouped_mm` | **excluded** | **admitted** |
| the other 50 keys | admitted | admitted |

`_grouped_mm` is the only underscore-prefixed op name in the table, so nothing else changed
reachability with it. Three things move as a consequence and all three were checked:

- `torch._grouped_mm` and `torch._C._VariableFunctions._grouped_mm` resolve, and they compute
  what `torch.ops.aten._grouped_mm.default` computes — asserted element-wise, in both the
  positional and the three-keyword spellings `torch/nn/functional.py:7139` uses.
- `_C._shim_overloads` gains the key `_grouped_mm`. Anything reading that dict as "what
  `torch.<op>` can reach" now gets a truthful answer where it got a false one.
- The `(qualname, overload)` -> schema map `_get_schema` answers from gains
  `(aten::_grouped_mm, default)`. `verify_schemas.py` re-derives it from upstream and matches.

The risk that remains is forward-looking rather than present: **a future underscore-prefixed
entry in `overloads.json` will now become a live `torch.<name>` binding without anyone deciding
that it should.** `test_grouped_mm_resolves_from_the_torch_level_name` asserts the admitted set is
exactly `["_grouped_mm"]`, so that decision has to be made explicitly rather than inherited.

That test asserted the *broken* state until this landed, so that the fix could not land silently.
It now asserts the fixed state and the scope above.

### 6.2 `aten.floor_divide.Scalar` — the one missing *kernel*, now implemented

`transformers`' MoE routing does `perm // num_top_k`, a tensor and a Python `int`. Upstream sends
that to `floor_divide.default`, because torch's frontend wraps the number into a tensor before it
picks an overload; this shim's resolver does not, so it lands on `.Scalar`, which had no kernel.

Implemented here, sharing `.default`'s arithmetic, with its own golden cases against upstream's
`.Scalar` key. The key selection still differs from upstream's — closing *that* means teaching the
resolver upstream's "numbers as tensors" rule, in `bootstrap.py`. The same divergence already
exists for `add`/`sub`/`div`, whose `.Scalar` overloads are listed in `overloads.json` for exactly
this reason.

### 6.3 `torch.floor_divide` · `torch.cumsum` · `torch.histc` — table entries, added here

All three are on Mixtral's Python path (`TorchFunctionMode` trace) and all three had kernels and
no `overloads.json` entry, so the name refused. Added, and `verify_schemas.py` confirms every
string against upstream.

`floor_divide.Scalar_out` came with them, and it has a side effect worth recording: the yaml does
not declare that overload (torchgen generates it), so nothing could resolve its schema before.
With a schema to resolve, `register_decomposition(aten.floor_divide)` now reaches it and the
decomposition registry grows from 1004 to 1005 and the table from 414 to 415 — one more real
upstream decomposition, closer to upstream's own 1097.

### 6.4 `TensorBase` members — seven names, seven existing kernels — **all bound**

None of these was a missing operator. Each is a Python-level spelling with a kernel already in
`_aten_implemented()`.

| refused | kernel it needs | where | where the binding went |
|---|---|---|---|
| `TensorBase.__idiv__` | `aten::div_.Tensor` | `torch/_tensor.py:1115` spells `__itruediv__` as `_C.TensorBase.__idiv__`; MoE normalises router weights with `/=` | `methods.json` |
| `TensorBase.__ge__` | `aten::ge.Scalar` / `ge.Tensor` | sentinel mask. **`__le__`, `__gt__` and `__lt__` all work** — only `__ge__` was absent, which reads as an oversight rather than a decision | `methods.json`, with `ge` alongside it |
| `TensorBase.clamp_` | `aten::clamp_` | `expert_ids_g.clamp_(max=...)` | `methods.json` |
| `TensorBase.masked_fill_` | `aten::masked_fill_.Scalar` | the pre- and post-masks | `methods.json` |
| `TensorBase.div_` | `aten::div_.Tensor` | | `methods.json` |
| `TensorBase.chunk` | composite over `aten::split.Tensor` | `CompositeImplicitAutograd` upstream, so a Python-level composition like `linear`/`dropout` | `bootstrap.py`, `_install_tensor_chunk` |
| `TensorBase.__setitem__` | `aten::index_put_.default` | `inv_perm[perm] = arange(...)`. `__getitem__` is already a Python-level member; this is its missing half | `bootstrap.py`, `_install_tensor_indexing` |

`methods.json` covers the first five; `chunk` and `__setitem__` are Python-level members and sit
with `__getitem__` in `bootstrap.py`. Six things came out of doing it that the table above did not
say, and four of them are gaps rather than closures.

**`ge` went in beside `__ge__`.** `le`, `lt`, `gt`, `eq` and `ne` each have both a dunder and a
plain-method entry; `ge` had neither. Adding only the dunder would have left the set asymmetric
for no reason anyone could later reconstruct.

**`aten.ge.Tensor` has no kernel** — `le.Tensor`, `lt.Tensor` and `gt.Tensor` all do. So
`x >= tensor` now resolves and then refuses by name, which is a precise work item where the
missing table entry was a vague one. **This one is a missing kernel, not a missing name**, and it
is the only comparison overload in that state.

> **Fixed since — and it was already fixed when this sentence was committed.** `088e8f4` ("Feat:
> Close three kernel gaps, and disprove the stated reason for the fourth") gave `ge.Tensor` a
> kernel at 18:58, this document's own final revision (`8c07af8`) landed at 19:56 the same day, and
> the sentence above still shipped saying "has no kernel." `x >= tensor` computes today
> (`aten.ge.Tensor` is in the current 168-op `_aten_implemented()` list). Round 1 of this audit's
> `docs/kernels/VIEWS.md` finding independently confirmed the same kernel from a different document.
> <!-- DOCWATCH: op-implemented aten.ge.Tensor -->

**`chunk` is not `extent // chunks`.** Upstream's composite
(`at::native::chunk`, `aten/src/ATen/native/TensorShape.cpp`) rounds the split size *up* and then
lets `split` return however many pieces that produces, so `chunks` is an upper bound rather than a
promise — `arange(10).chunk(3)` is `(4,4,2)` and not `(3,3,3,1)`, and `arange(3).chunk(7)` returns
**three** chunks. The zero-extent case is the one branch that must return exactly `chunks`, and it
is the reason upstream has a `split_with_sizes` path at all: with a split size of 0, `split` would
discard the count, because any number of empty chunks sums to zero. All of that is transcribed
rather than reimplemented, and `Tensor.chunk` returns a `tuple` (upstream's `THPVariable_chunk`
does) where `torch.ops.aten.split.Tensor` returns a `list`.

**`__setitem__` is only half implementable here, and the other half is a missing capability rather
than a missing operator.** Measured on 2.13.0, upstream's subscript assignment has two shapes:

```
x[t] = v        -> index_put_.default        reproduced
x[boolmask] = v -> index_put_.default        resolves, then the kernel refuses (below)
x[:, t] = v     -> index_put_.default        reproduced, indices [None, t]
x[:] = tensor   -> copy_.default             reproduced
x[...] = 3.0    -> fill_.Tensor              reproduced
x[0] = 3.0      -> select.int, copy_.default REFUSED
x[1:3] = tensor -> slice.Tensor, copy_       REFUSED
```

The last two are refused because **`aten.select.int` and `aten.slice.Tensor` return copies, not
views** — a candle tensor is a value. Both kernels exist and both are correct on their own; it is
the sequence that is not reproducible. Probed on this build:

```
s = select.int(x, 0, 1);  copy_(s, v)          ->  x unchanged
s = slice.Tensor(y, 0, 1, 3, 1); copy_(s, v)   ->  y unchanged
```

Running upstream's sequence would therefore report success and write nothing, which is the
silent-divergence direction DESIGN.md §5 exists to keep out. So it refused by name and said what
was missing: **mutable views**.

> **Fixed since.** Views are mutable now — `write_into` scatters into the positions the
> destination's own `Layout` addresses, so an in-place write reaches the base instead of rebinding
> the wrapper ([`docs/kernels/VIEWS.md`](../kernels/VIEWS.md) §6). The test that pinned the refusal is now
> `test_setitem_writes_the_basic_index_through_to_the_base` and asserts the write. What still
> refuses is a `step != 1` slice, for a different reason: candle's `Tensor::from_storage` builds
> only contiguous layouts, so there is nothing to construct a stepped view with.

**`aten.index_put_.default` refused a bool-mask index**, where upstream accepts one: the kernel
was written on top of `scatter`, which wants an int32/int64 index
(`Expected dtype int32 or int64 for index, got bool`).

> **Fixed since.** The kernel does its own address arithmetic and masks lower through
> `mask_to_indices`. The same change lifted the rank-1 restriction; 33 behaviours were probed
> against upstream and all 33 match, error text included.

**`aten.index_put_.default` also refuses anything but 1-D self/index/values**, which is why
`x[t] = 5` (a Python number, lifted to a 0-d tensor the way upstream lifts it) refuses and
`x[t] = tensor_of_the_right_length` does not. Third kernel gap; the same case shape covers it.

### 6.5 Not a blocker, but on the path

`torch.manual_seed` refuses (`torch._C._dynamo.eval_frame.set_eval_frame`), because
`torch/_compile.py` routes it through Dynamo's disable wrapper.

> **Fixed since (문서 감사, 2026-09).** `torch.manual_seed(42)` succeeds today — the CPU generator
> port (`docs/numerics/RNG.md`/`docs/bindings/TENSORBASE.md`, round 2 of this audit) closed this along with
> `uniform_`/`normal_`. §9's "weights are filled by a shared LCG ... because `torch.manual_seed`
> still refuses" is stale for the same reason, though re-running §9's comparison with real seeding
> instead of the LCG workaround was not attempted here (out of scope for a documentation pass).

Nothing in a forward pass needs
it; it only means a seeded shim run cannot be set up the obvious way.

---

## 7. Numbers

| | before | after the operator | after the names (§9) |
|---|---|---|---|
| Golden cases | 2843 / 2843 | 2918 / 2918 | **2971 / 2971**, 0 failed |
| Ops covered by the golden suite | 119 | 121 | **121** (no new op — these are name bindings) |
| Smoke tests (`pytests/run.sh`) | 211 | 220 | **223** |
| Schema table entries vs upstream | 4203 / 4203 | 4217 / 4217 | **4231 / 4231** |
| Golden `--self-test` comparators | 12 | 12 | **13** |
| Mixtral, operator sweep | 1 missing | **0 missing** | 0 missing |
| Mixtral, executed on the shim | — | runs with §6 patched in the test script | **runs with nothing patched** (§9) |

The golden suite grew by 75 for the operator: 64 for `_grouped_mm` and 11 for
`floor_divide.Scalar`. It grew by a further 53 for the names, and those 53 go through the
*member* on both sides (`t.clamp_(max=3)` against `t.clamp_(max=3)`) rather than through
`_aten_dispatch`. That distinction is the whole point: the kernel-level cases passed the entire
time the members were raising `NotImplementedError`, so they could not have caught this and a
name with no case is a name nobody checks.

The 13th comparator is `_chunk_tuple_check`, which is `_chunk_list_check` plus the container
type — `t.chunk(2) + (x,)` works and `list + tuple` does not, so `tuple` is part of the answer.

Both new kernels were checked by sabotage rather than by trusting a green run:

- an off-by-one in the offset walk (`end + 1`) fails **30** cases;
- deleting the 16-byte stride refusal fails **3** cases, each reported as one side computing
  where the other raised.

`--self-test` was 12 comparators × 11 fault modes with 0 problems, and is 13 × 11 now. It had
briefly gone to 13 once before and come back, because the first version of the short-`offs` case
introduced a comparator that ignored the last rows of the result and was therefore blind to the
harness's own `value-last` injection. The harness said so, and the case now slices the unwritten
tail off with `aten.slice.Tensor` instead — so `_gemm_scale_check`, whose fault profile is already
established, does the comparing.

---

## 9. Mixtral, run with nothing patched (2026-08-30)

§5.2 ran Mixtral on the shim and got `2.384e-07` over 800 logits, **with §6's name gaps patched in
the test script**. Those gaps are closed now, so the same run works against the repository as it
stands. Nothing is monkey-patched, nothing is stubbed, and the script is byte-identical between
the two sides — it is run twice, once under upstream torch and once with `PYTHONPATH` pointed at
the vendored tree, so a difference in the output is a difference in torch.

```
MixtralForCausalLM   hidden 32 · intermediate 64 · 1 layer · 4 heads · 2 kv heads
                     4 experts · top-2 · vocab 100 · 8 input tokens

shim logits   [1, 8, 100] float32
MAX ABS DIFF vs upstream    4.4703e-08      over 800 logits, logit scale 2.2832e-01
                                             (relative 1.958e-07)
argmax, all 8 positions     identical       [18, 27, 17, 27, 53, 79, 25, 52]
generate(max_new_tokens=4)  identical       [3,17,42,8,55,1,90,23, 52,95,45,66]
```

`4.47e-08` is one ulp of float32 at that magnitude. It is smaller than §5.2's number because the
weights are smaller, not because anything got more accurate; both are float32 GEMM noise.

Weights are filled by a shared LCG keyed on the sorted `state_dict` index, RNG-free — the same
procedure §5.2 used, and for the same reason: `torch.manual_seed` still refuses (§6.5), and two
independent RNG implementations would not produce the same stream even with a matched seed.

**The batch is load-bearing, checked by reverting rather than by assuming.** With
`bootstrap.py` and `methods.json` restored to their pre-fix state and the artefact rebuilt, the
same script dies in `MixtralSparseMoeBlock.forward`:

```
transformers/models/mixtral/modeling_mixtral.py:109
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
NotImplementedError: not implemented in torch._C shim: TensorBase.__idiv__
```

Which of the seven names the model actually reaches, counted by wrapping each member and running
the same forward and `generate`:

| member | calls |
|---|---:|
| `torch._grouped_mm` | 10 |
| `masked_fill_` | 10 |
| `__ge__` | 5 |
| `clamp_` | 5 |
| `chunk` | 5 |
| `__setitem__` | 5 |
| `__idiv__` | reached, but not countable this way — `torch/_tensor.py` binds `Tensor.__itruediv__` to the unwrapped member at import time, before any wrapper can be installed. The revert above is what shows it |
| `div_` | 0 — this config never calls it directly; it is in `methods.json` because §6.4 measured it and because `__idiv__` needs the same kernel |
| `ge` | 0 — added for symmetry with `le`/`lt`/`gt`, not because Mixtral calls it |

