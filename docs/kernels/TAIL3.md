# The walls behind the walls

`docs/kernels/TAIL1.md` and `docs/TAIL2.md` closed the top of `docs/architectures/ARCH100.md`'s ranking. Each closure
exposed the next wall, and this round is those: `TensorBase.index_add` (`jetmoe`),
`TensorBase.scatter_reduce` (`tapas`), `TensorBase.view_as`, `bitwise_xor` (`gpt_neo`),
`as_strided` (`longformer`), `avg_pool2d` (`efficientnet`), `erfinv` (`gemma3n_text`),
`einsum` (`longt5`) — plus the item `docs/architectures/ARCH100.md` classified not as a gap but as a **backend
limitation**, `aten.matmul.default` / `MatMulUnexpectedStriding`, against five architectures.

That last one is first, because it was the largest single item and because the classification was
wrong.

---

## 1. The matmul striding verdict: `contiguous()` was neither the fix nor the avoidance

`docs/bindings/SETITEM.md` proposed a `contiguous()` as the plausible fix. **The operands were already
contiguous**, and this is decidable from the error text alone — the refusal prints the layout it
refused:

```text
hiera        lhs [1, 1, 49, 64, 32]   stride [100352, 100352, 2048, 32, 1]
qwen3_next   lhs [1, 32,  1, 64,128]  stride [262144,   8192, 8192,128, 1]
olmo_hybrid  lhs [1, 30,  1, 64, 96]  stride [184320,   6144, 6144, 96, 1]
```

Each of those is exactly the reversed cumulative product of the reversed shape — the contiguous
strides. `32*64 = 2048`, `2048*49 = 100352`. So `gemm_with_layout_fallback`'s existing
`.contiguous()` retry *did* run, returned the same tensor because `is_contiguous()` was already
true, and the identical error came back. `contiguous()` there is a **no-op**. It would not have
fixed anything and it would not even have papered over anything.

The real refusal is `candle_core::cpu_backend::MatMul::ab_skip`, which matches on
`stride[..rank - 2]` and recognises exactly **zero, one or two** batch axes:

```rust
let a_skip: usize = match lhs_stride[..rank - 2] {
    [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
    [_, stride] if lhs_l.dims()[0] == 1 => stride,
    [stride, _] if lhs_l.dims()[1] == 1 => stride,
    [stride] => stride,
    [] => m * k,
    _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?,
};
```

Rank 5 has three batch axes and falls off the end of that list **whatever its strides are**. The
message names contiguity because that is the only reason the other arms can fail; here it is a
misnomer, and the misnomer is what produced the wrong classification.

### The fix matches upstream rather than avoiding the error

`at::native::matmul`'s N-D × N-D branch reshapes both operands to rank 3 — `(prod(batch), m, k)`
and `(prod(batch), k, n)` — calls `bmm`, and views the result back out. `fold_batch_axes_matmul`
does exactly that. This is the distinction the round was asked to establish: the arithmetic is
identical because a batched GEMM is defined per batch element and how the batch index is spelled
cannot change any dot product, so folding is **matching upstream, not routing around candle**.
Nothing is silently copied that upstream does not copy: the reshape is free when the operand is
contiguous (all five architectures), and when the batch axes have to be broadcast to agree the
copy is the one `broadcast_matmul` was already performing one level down.

It is reached only **after** candle has refused, for `gemm_with_layout_fallback`'s stated reason —
the accepted set differs by backend and by rank, and restating candle's predicate here would mean
keeping a copy of it in sync with candle. Every shape candle already accepts takes the path it
took before, so no existing answer can change.

### And it moved zero architectures on its own

The fold let `k_beta @ key.transpose(-1, -2)` through, and `torch_chunk_gated_delta_rule` stopped
**four lines later**:

```python
attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
```

`torch.eye` had no table entry. So `aten.eye.default` and `aten.eye.m` are in this round too, and
without them the sweep would still have reported five failures with a different name on them. That
is the shape of this whole tail and the reason a round is reported by *architectures cleared*
rather than by ops landed.

---

## 2. Alias or kernel — the split, counted honestly

`docs/architectures/ARCH100.md` established that missing **bindings** outnumber missing **kernels** 49 to 22 in
this tail, so the first question per op was "does a kernel already exist under another name".

| op | alias? | what actually had to be written |
|---|---|---|
| `matmul.default` (rank ≥ 5) | **neither** | a *fallback path*, not a new op: 60 lines folding the batch axes after candle refuses |
| `index_add.default` | **yes** | `index_add_`'s body with `write_back` swapped for `finish`. Both overloads dispatch at one function |
| `view_as.default` | **yes — but of `view`, not of `reshape_as`** | `aten.view.default`'s body. See §4 |
| `bitwise_xor.{Tensor,Scalar}` | **yes** | a third arm on the `Bitwise` enum, three match lines |
| `eye.{default,m}` | **no** | new factory: candle 0.11.0 has no `Tensor::eye` |
| `erfinv.default` | **no** | **new arithmetic**: no candle kernel, no closed form, AS 241 |
| `scatter_reduce.two` | **no** | **new kernel**: `include_self`, five reductions, `amin`/`amax` |
| `as_strided.default` | **refused by name** | nothing. §6 sizes it |

**Counted honestly: two new kernels, one new factory, one fallback path, three bindings over
bodies that already existed, and one deliberate refusal.** Eight `_aten_implemented()` keys, nine
table rows. "Eight ops" would overstate it by a factor of three, which is the counting failure
`CLAUDE.md` §5.3 names.

`pytests/test_tail3.py::test_the_two_new_kernels_are_the_only_new_arithmetic` asserts the sharing
itself — `index_add_common` reached from exactly two dispatch arms, `Bitwise::Xor` as an arm and
not a function — so the claim in this table cannot rot into prose.

---

## 3. An alias is not always an alias: `scatter_reduce`'s `sum` and `index_add_`'s disagree

This is the finding the round would most easily have got wrong, and the first draft did.

The same probe on both ops — 64 accumulations of `bfloat16(0.01)` into one position:

```text
zeros(2, bf16).index_add_(0, [0]*64, [0.01]*64)                     ->  0.6523
zeros(2, bf16).scatter_reduce(0, [0]*64, [0.01]*64, reduce="sum")   ->  0.6406
```

`index_add_` accumulates **at the receiver's dtype, per step** (docs/architectures/DEMAND8.md measured it).
`scatter_reduce` accumulates **wide and narrows once**. Two adjacent ops with opposite answers, and
the first `scatter_reduce_two` inherited `index_add_`'s rule from the kernel three hundred lines
above it — wrong by one `bfloat16` step, and invisible on any index without repeats.

The same list, each measured rather than assumed from a neighbour:

* `mean` divides once at the end on the wide sum: the same 64 sources give `0.0098` = `0.6406/65`.
* `include_self=True` puts `self` in the count: `full((2,), 10.)` with sources `1,2,3` at one
  position is `(10+1+2+3)/4 = 4`.
* Integer `mean` **truncates toward zero**: `int64` zeros with `1,2,4` give `7/4 = 1`, and `7/3 = 2`
  with `include_self=False`.
* `nan` wins in `amin` and `amax`. Rust's `f64::min` discards it, so the comparison is written out.
* `bool` `sum` is a logical or; integral `sum` wraps (`uint8` `200+200 = 144`) — those two *are*
  `index_add_`'s answers, re-measured rather than inherited.

### `include_self=False` seeds only what is written

Not "ignore `self`". A position no index names keeps its `self` value either way; the flag removes
`self` from the fold **at the positions that are written**, by seeding them with the reduction's
identity on first touch:

```text
full((3,), 9.).scatter_reduce(0, [0], [1.], "amax", include_self=False)
    ->  [1., 9., 9.]        not [1., -inf, -inf]   (identity written everywhere)
                            and not [9., 9., 9.]   (flag ignored)
```

### It is not `scatter.reduce`, and implementing that would not have moved `tapas`

`docs/kernels/SCATTER.md` said so and this restates it because the names are one character apart.
`aten::scatter.reduce`'s `reduce` is `"add"`/`"multiply"` only and it has no `include_self` at all.
`tapas` (`modeling_tapas.py:1465`) calls `scatter_reduce(..., reduce="amin", include_self=False)`.

Two refusals also differ from the adjacent `scatter.src`, and were transcribed rather than shared:
the index-dtype message is `scatter(): Expected dtype int32/int64 for index` with **no `, got Long`
tail**, and a **0-d `self` is accepted here** where `scatter.src` raises `0-dim self not
implemented`.

---

## 4. `view_as` is an alias of `view`, and `view` is not `reshape_as`

Upstream's two are different ops, measured on a transposed receiver:

```text
w = arange(6.).reshape(2, 3).t()
w.reshape_as(zeros(6))  ->  tensor([0., 3., 1., 4., 2., 5.])
w.view_as(zeros(6))     ->  RuntimeError: view size is not compatible with input tensor's
                            size and stride ...
```

So spelling `view_as` as `reshape_as` would have been wrong about torch. It is spelled as
`aten.view.default` instead — which in this shim is `reshape_like`, and **that op has always
accepted the layouts upstream's `view` refuses** (`w.view(6)` returns the reshaped values here and
raises upstream). `view_as` inherits that pre-existing laxity rather than adding a second rule;
making it strict while `view` stays lax would put two answers behind one definition.

The gap is recorded, not fixed. `tools/golden/cases.py` carries it as a `torch_error` row and
`test_tail3.py::test_view_as_and_reshape_as_are_different_ops_upstream` fails the day `view` is
tightened — which is when `view_as` must be tightened with it.

---

## 5. `erfinv`, and where upstream is the one that is wrong

`gemma3n_text`'s wall is not in a model file: `torch/distributions/normal.py:111`'s `icdf` is
`loc + scale * erfinv(2*value - 1) * sqrt(2)`, and `modeling_gemma3n.py:981` calls it. So the
caller is the **vendored `torch/` tree**.

candle has no `erfinv`, `libm` is not a direct dependency of this crate, and there is no closed
form, so the arithmetic had to be chosen: **AS 241 `PPND16`** (Wichura 1988), the standard-normal
quantile, with `erfinv(y) = ndtri((1+y)/2) / sqrt(2)`. The single-precision Giles approximation
usually reached for lands around `1e-7` relative, which is *at* `float32`'s own resolution and
therefore cannot be distinguished from a correct answer by a `float32` test.

The tail branch takes `q = (1 - |y|)/2` **from `y` directly** rather than as `1 - p` after forming
`p = (1 + y)/2`, which would cancel away the digits that branch is about.

Measured against `torch.erfinv` at `float64`:

```text
|y| <= 1 - 1e-11      relative <= 5.8e-15    (1-2 ulp)
|y|  = 1 - 1e-12      relative 1.6e-06
|y|  = 1 - 1e-15      relative 5.5e-05
```

**The last two rows are upstream drifting, not this.** At `y = 0.999999999999` the answers are
`5.04203993266166` (upstream) and `5.042031898572695` (here); `erfc` of the second is
`9.999778782798894e-13` against a target of `9.999778782798785e-13` — twelve digits — while `erfc`
of upstream's is `9.998953310354861e-13`, wrong in the fourth. Asserting agreement there would be
asserting the wrong answer, so `test_tail3.py` pins it by the **`erfc` round trip** instead. The
whole region is unreachable at `float32`, whose largest value below one is `1 - 6e-8`.

At `float32` the two disagree by at most one ulp (39% of 20000 random draws differ in the last
bit), which is what `erf`'s own note in `aten.rs` already records for the forward direction.

The domain, all of it measured and none of it raising: `erfinv(±1)` is `±inf`, `|y| > 1` is `nan`,
`erfinv(int64 [0, 1])` is `float32 [0., inf]`. A Newton iteration would not terminate at `±1`, and
`normal.icdf(1.0)` is a real call.

---

## 6. `as_strided` is refused by name, and here is its size

> **Superseded by `docs/kernels/STRIDED.md`, and left standing.** The constructor this
> section says is missing is still missing — that part was never overturned and
> `docs/kernels/STRIDED.md` §1 re-verifies it in candle's source. What this section did
> not consider is the third option: implement the gather *and refuse the writes
> upstream would have propagated*, in both directions, keyed on the storage
> address with a keep-alive that makes the key un-reusable. `longformer` and
> `led` are past this op as a result. The sizing below still describes what it
> would take to remove the **narrowing**; it no longer describes what it takes
> to make the op callable.

`longformer._chunk` (`modeling_longformer.py:719`) is
`hidden_states.as_strided(size=chunk_size, stride=chunk_stride)`.

**It is not implemented, on purpose.** `as_strided` is by definition a *view* over raw strides, and
candle 0.11.0 exposes no way to build a `Tensor` over an existing storage with arbitrary strides:

* `Layout::new(shape, stride, start_offset)` is `pub`,
* but the only public path that reaches it, `Tensor::from_storage`, takes an **owned** `Storage`
  and documents "this uses contiguous strides",
* `Tensor { storage: Arc<RwLock<Storage>>, layout }` is private, and `same_storage` is
  `pub(crate)`.

So the aliasing the op promises cannot be produced. A materialising gather would return the right
values for `longformer`'s read-only `_chunk` and **silently wrong ones for any writer** — the
divergence shape `docs/devices/VULKAN2.md` requires be unrepresentable rather than merely unused, and the
same gap `docs/kernels/VIEWS.md` §6.4 already carries for `slice.Tensor` with step > 1 and for
`view.dtype`.

What was done instead is `clamp.Tensor`'s pattern: the schema is in `methods.json` with **no
kernel**, so `x.as_strided(...)` refuses with `aten op not implemented in torch._C shim:
aten.as_strided.default` — naming the overload it needed — rather than "no matching signature". It
is **not** in `_aten_implemented()`, so the surface stays honest and the golden harness does not
demand case builders for a kernel that does not exist. `tools/golden/reach_allow.json` carries the
entry and its reason; the checker matches that file exactly in both directions, so the entry has to
be deleted the day the gap closes.

**Sizing it properly** means the same work `slice.Tensor` and `view.dtype` need, and doing it once
would close all three: either a stride-carrying tensor wrapper in this crate that owns the
`Arc<Storage>` and its own `Layout`, or a candle patch exposing a storage-sharing constructor
(`vendor/` already carries `int8-candle-0.11.0-cpu.patch`, so the mechanism exists **— which is FALSE, corrected 2026-09-07.** The patch file is carried (now in `vendor/`, previously in `docs/`), but NOTHING APPLIES IT: `vendor/*.sh`, `vendor/*.py`, `Cargo.toml` and `build.rs` contain no patch step and no `[patch.crates-io]`. Carrying a diff is not a mechanism, and three documents used this sentence to argue that a candle fork would be cheap). Estimate: the
wrapper touches every kernel that calls `read_flat`/`write_flat`; the patch is perhaps twenty lines
of candle and a re-vendor. Neither is a `longformer`-shaped task.

---

## 7. Two walls in `bootstrap.py`, which this round did not own

`efficientnet` and `longt5` are **not** missing aten ops, and the sweep's names for them are
misleading in the same way `MatMulUnexpectedStriding` was:

* **`efficientnet` — `torch._C._nn.avg_pool2d`.** `aten.avg_pool2d.default` has been implemented
  and golden-compared since `sew_d`. What is missing is one binding in `bootstrap.py`'s
  `_install_nn`, beside the `upsample_bilinear2d`/`upsample_bicubic2d`/`softplus` it already
  writes. `torch/nn/modules/pooling.py:779` calls `F.avg_pool2d`, which binds straight to the `_nn`
  name.
* **`longt5` — `torch.einsum` with an ellipsis.** `einsum` exists in `bootstrap.py` and parses
  fixed subscripts; it refuses `'...qhd,...khd->...hqk'` by name, saying the ellipsis needs its own
  rank arithmetic. `longt5` (`modeling_longt5.py:662`) passes exactly one equation form, so this is
  one branch — expand `...` to the operand's leading axes given its rank, per operand — and not a
  general einsum planner.

Both live in `rust/torch_c/src/bootstrap.py`, which this worktree was told not to edit. They are
recorded here as sized work items rather than left as sweep lines.

---

## 8. Architectures

Ran with `pytests/arch_sweep.py --only ...`, before and after.

| architecture | before | after |
|---|---|---|
| `jetmoe` | `TensorBase.index_add` | **forward passes** |
| `tapas` | `TensorBase.scatter_reduce` | **forward passes** |
| `gpt_neo` | `bitwise_xor` (at *construction*) | **forward passes** |
| `hiera` | `matmul MatMulUnexpectedStriding` | **forward passes** |
| `qwen3_next` | `matmul MatMulUnexpectedStriding` | **forward passes** |
| `olmo_hybrid` | `matmul MatMulUnexpectedStriding` | **forward passes** |
| `qwen3_5_moe` | `matmul MatMulUnexpectedStriding` | **forward passes** |
| `minicpmv4_6` | `matmul MatMulUnexpectedStriding` | **forward passes** |
| `gemma3n_text` | `erfinv` | `torch.std` — the next wall |
| `longformer` | `TensorBase.as_strided` | `aten.as_strided.default` — refused by name, §6 |
| `efficientnet` | `torch._C._nn.avg_pool2d` | unchanged — `bootstrap.py`, §7 |
| `longt5` | `torch.einsum` | unchanged — `bootstrap.py`, §7 |

**Eight cleared.** Five of those eight are the matmul item, and all five needed `eye` as well as
the fold. `gemma3n_text` moved one wall (`erfinv` → `std`), which is a result rather than a
failure: `torch.std` has no `overloads.json` row and `_gaussian_topk` calls it two lines after the
`icdf` this round unblocked.

---

## 9. What is verified, and how

Every op is compared against upstream element-wise, in a separate process, on inputs where a
plausible wrong implementation differs.

* **Golden harness** (`tools/golden/cases.py`):
  <!-- DOCWATCH: count golden_cases_total ge 9870 -->
  <!-- DOCWATCH: count golden_cases_passed ge 9870 -->
  <!-- DOCWATCH: count golden_ops_covered ge 263 -->
  <!-- DOCWATCH: count golden_pending eq 0 -->
  9870 cases pass of 9870, 263 ops covered, 0 pending builders (9691/255 before).
* **`pytests/test_tail3.py`**, a new suite file, holding down the four things a value comparison
  structurally cannot: the striding verdict as arithmetic on the printed layouts, the
  alias-versus-kernel split as source structure, the two adjacent ops that must **disagree**, and
  the `erfinv` far tail judged by round trip rather than by agreement.
  <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail3.py test_the_refused_matmul_operands_were_already_contiguous present -->
  <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail3.py test_scatter_reduce_sum_and_index_add_accumulate_DIFFERENTLY present -->
  <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tail3.py test_in_the_far_float64_tail_this_shim_is_more_accurate_than_upstream present -->
* **Two pinned counts in `test_shim.py`** moved, each carrying the arithmetic that keeps it a
  check: `tag_core_count` 117 → 120 (`bitwise_xor.Tensor`, `bitwise_xor.Scalar` and
  `scatter_reduce.two` are the only three of the eight new keys upstream tags `core` — each read
  off its own `.tags`; `as_strided.default` **is** core and is deliberately absent because it has
  no kernel), and distinct schema identities 322 → 331.
* **`device.rs::MPS_HOST_READBACK_OPS` 66 → 71.** `bitwise_xor.{Scalar,Tensor}`, `erfinv.default`,
  `scatter_reduce.two` and `index_add.default` all read the tensor back to the host through
  `read_flat`, so all five are refused on `mps` rather than silently computing on the CPU. `eye`
  is not on the list: it reads nothing, it only writes.
* The fold is checked **not** to have replaced the ordinary path — it is reached from the error arm
  only, and `test_the_fold_is_reached_only_after_candle_refuses` asserts that guard is still there.

The ops now in `_aten_implemented()`:

<!-- DOCWATCH: op-implemented aten.bitwise_xor.Tensor -->
<!-- DOCWATCH: op-implemented aten.bitwise_xor.Scalar -->
<!-- DOCWATCH: op-implemented aten.erfinv.default -->
<!-- DOCWATCH: op-implemented aten.index_add.default -->
<!-- DOCWATCH: op-implemented aten.scatter_reduce.two -->
<!-- DOCWATCH: op-implemented aten.view_as.default -->
<!-- DOCWATCH: op-implemented aten.eye.default -->
<!-- DOCWATCH: op-implemented aten.eye.m -->

and the table rows that make them callable:

<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json bitwise_xor present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json erfinv present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json index_add present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json scatter_reduce present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json eye present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json bitwise_xor present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json __xor__ present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json erfinv present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json index_add present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json scatter_reduce present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json view_as present -->

and the one that is a schema with no kernel behind it, on purpose (§6):

<!-- DOCWATCH: json-key rust/torch_c/src/methods.json as_strided present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs fold_batch_axes_matmul present -->

`aten.as_strided.default` is deliberately **not** an `op-implemented` marker: the whole point of §6
is that it is listed and unimplemented, and a marker asserting otherwise would be the false claim
`docs/verification/DOCWATCH.md` exists to prevent.
