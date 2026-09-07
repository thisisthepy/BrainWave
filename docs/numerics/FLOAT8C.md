# `float8_e4m3fn`, round 3: the two buckets where upstream computes and this build did not

`docs/numerics/FLOAT8B.md` enumerated all 197 ops against upstream 2.13.0 and closed 114
divergences. Two buckets stayed open, and they were the ones pointing the other
way — the ops **upstream computes** and this build did not:

    upstream computes -> this build hangs      10   (FLOAT8B table D)
    upstream computes -> this build refuses    13   (FLOAT8B table E)

All 23 are closed. **No fork of `candle-core` was needed**, and §1 is why.

---

## 1. Can it be fixed without forking candle? Yes, and the reason is narrow

`docs/numerics/FLOAT8.md` diagnosed the hang correctly: `candle-core` 0.11.0's

```rust
with_dtype!(f8e4m3, F8E4M3, f8e4m3::from_f64, |v: f8e4m3| v.to_f64());
```

generates `fn to_f64(self) -> f64 { (|v: f8e4m3| v.to_f64())(self) }`. The
`float8` crate's inherent method is `to_f64(&self)`, so the by-value trait
method is the exact receiver match and the closure calls itself. Release-mode
LLVM turns that tail call into `.L1: jmp .L1` — a full-speed CPU spin that never
overflows a stack, holding the GIL, which is why `SIGALRM` cannot end it.

What `docs/numerics/FLOAT8B.md` §4.1 did **not** establish is how far that poison reaches.
It reaches exactly three places, and the shim needs none of them:

| candle site | poisoned? | why |
|---|---|---|
| `cpu_backend/mod.rs` `(CpuStorage::F8E4M3, DType::F64)` | **yes** — `unary_map(.., \|v\| v.to_f64())` | the one arm anything here reaches |
| `cpu_backend/mod.rs` `(CpuStorage::F8E4M3, DType::{U8,U32,I16,I32,I64,BF16,F16,F32})` | **no** | every one is written `v.to_f32()`, and `WithDType` declares no `to_f32`, so it resolves to `float8`'s inherent method |
| `scalar.rs` `Scalar::F8E4M3(v) => v.to_f64()` | yes | never on this shim's path — nothing builds a `candle_core::Scalar` from float8 |
| `op.rs` `UnaryOpT::f8e4m3` | yes | never reached: every elementwise unary is refused for this dtype at the door, because **upstream refuses them too** (FLOAT8B §2) |

So the fix is not a patch, it is a **route**:

    F8E4M3 -> F64          hangs
    F8E4M3 -> F32 -> F64   terminates, and is exact

Exact rather than tolerated. `float8_e4m3fn` is 4 exponent bits and 3 mantissa
bits with a maximum magnitude of 448 and no infinities, so all 256 of its bit
patterns are representable in `f32`, and `f32 -> f64` is exact for every one of
them. The two-step result is bit-identical to what a non-recursive `to_f64`
would have produced. `tools/golden/compare.py::_as_list` had already been
reading float8 results by this same widening since `docs/numerics/FLOAT8B.md` §5.1 — the
evidence that the route works was sitting in the harness the whole time.

### Where the route lives

`rust/torch_c/src/reduced.rs::to_dtype` — the funnel `fast_to` already goes
through, and therefore the one `aten._to_copy.default` uses. That placement is
not cosmetic. After routing every `to_dtype(DType::F64)` call site in `aten.rs`
through a new `widen_f64`, **`x.to(torch.float64)` still hung**, because the
cast op does not go through any `aten.rs` helper. Fixing the call sites and
fixing the funnel are two different fixes and this round needed both;
`widen_f64` now delegates to the funnel so they cannot drift apart.

### What a fork would have cost, and why the vendored candle is not evidence for one

`/Volumes/macMini/caches/candle-vendor/candle-0.11.0-patched` exists from an
earlier round, and this crate's `Cargo.toml` **does not use it**: there is no
`[patch]` section and `candle-core = "0.11.0"` resolves from crates.io. So
vendoring is a road this project started down and did not take, and nothing in
the shipped build depends on it. It was not touched by this round.

---

## 2. Table D re-measured: two of the ten were never what the table said

`docs/numerics/FLOAT8B.md` table D was produced by a generic recipe that synthesises
arguments from each op's schema. Two rows are artefacts of what it synthesised,
and the correction matters because both would otherwise have been closed the
wrong way.

| op | FLOAT8B table D | re-measured | what the recipe had done |
|---|---|---|---|
| `aten.pow.Scalar`, `aten.pow.Tensor_Scalar` | upstream computes | **value-dependent** | upstream short-circuits a degenerate exponent and refuses `"pow"` otherwise |
| `aten.adaptive_avg_pool1d.default` | upstream computes | **refuses every non-empty output size** | the recipe had passed `output_size=[0]`, an empty output where no kernel is ever dispatched |

Measured on 2.13.0, `torch.tensor([1.,2.,4.]).to(float8_e4m3fn)`:

| call | upstream |
|---|---|
| `pow.Tensor_Scalar(t, 0)` / `(t, 0.0)` / `(t, False)` | `[1,1,1]` |
| `pow.Tensor_Scalar(t, 1)` / `(t, 1.0)` / `(t, True)` | `[1,2,4]` |
| `pow.Tensor_Scalar(t, 2)` `(t, 3)` `(t, -1)` `(t, 0.5)` `(t, 1.5)` | `NotImplementedError: "pow" not implemented for 'Float8_e4m3fn'` |
| `pow.Scalar(1, t)` / `(1.0, t)` | `[1,1,1]` |
| `pow.Scalar(0, t)` `(2, t)` `(-1, t)` `(0.5, t)` | `"pow" not implemented for 'Float8_e4m3fn'` |

and, over 2-D and 3-D inputs alike:

| `adaptive_avg_pool1d` output size | upstream |
|---|---|
| `[0]` | a `(..., 0)` tensor — **for every dtype**, including `int64` and `bool` |
| `[1]` | `"sum_cpu" not implemented for 'Float8_e4m3fn'` |
| `[2]`, `[4]` | `"adaptive_avg_pool2d" not implemented for 'Float8_e4m3fn'` |

This is the same shape as `aten.matmul.default` in `docs/numerics/FLOAT8B.md` §2.1 — a
refusal that is a property of the *call*, not the op — so these three gates sit
beside the matmul one rather than in `FLOAT8_E4M3FN_REFUSALS`, and they carry
upstream's own wording because upstream really does refuse those calls. Note
that `adaptive_avg_pool1d` needs **two** kernel names: `[1]` is the
global-average path and says `sum_cpu`.

`FLOAT8_E4M3FN_SHIM_ONLY` — the list of ops refused in this shim's own words
because it could not answer them — is now **empty**.

---

## 3. Table E, the write-through family: eight ops, one missing match arm

Seven of the thirteen refused with

    torch._C shim cannot write through a view of candle dtype F8E4M3 --
    tensor.rs::flat_storage names the dtypes it can read, and this is not one of them

and `aten.fill_.Tensor` and `aten.index_put_.default` joined them from table D
once §1 stopped them hanging first. `flat_storage` reads a source buffer with
`to_vec1::<T>()`, which copies the storage slice and never widens, so it does
**not** touch the poisoned arm — the dtype was simply not in the `match`.

Two arms, one in `flat_storage` and one in `WriteThrough::cpu_fwd`, close all
eight. Naming `CpuStorage::F8E4M3(Vec<F8E4M3>)` requires the `float8` crate, so
it is now a direct dependency, pinned to the version `candle-core` already
resolves — the same arrangement, and the same justification, as the existing
`half` and `gemm` entries. It adds no code to the artefact.

**It is not a way to reach `float8`'s inherent `to_f64`.** Nothing in this crate
calls that function; §1's route is inside candle's own converter.

---

## 4. `mm` / `addmm`: candle has no `F8E4M3` matmul, and upstream's answer is a widened one

The last two of the thirteen refused with `candle: unsupported dtype F8E4M3 for
op matmul`, which is true — candle's GEMM dispatch has no arm for this dtype at
all. Upstream computes `mm`, `addmm`, `bmm` and 2-D `matmul` and returns a
`float8_e4m3fn` result.

Widening the operands to `f32`, multiplying and narrowing back is not a guess
about what upstream does. Over **700 random cases** — `k` from 1 to 512, `m` and
`p` from 1 to 4, three magnitude scales — the narrowed product was
**bit-identical** to upstream's `mm` in every one, and an `f64`-accumulated
route agreed with the `f32` one in every one as well. Three mantissa bits is
coarse enough that the accumulation width cannot show through the final
rounding.

The widening is scoped to `gemm_accumulate_in` and deliberately **not** added to
`opmath_in`. `opmath_in` is the elementwise opmath type, and upstream *refuses*
float8 for the elementwise kernels; widening there would compute where upstream
declines, which is the divergence direction this dtype's whole gate exists to
prevent.

---

## 5. The per-op table

Values are compared by widening both sides to `float32` through their own
module's constant — exact for this dtype, and the same route on both sides.

### A. Table D (FLOAT8B): upstream computes, this build hung (10)

| op | upstream 2.13.0 | before | after | evidence |
|---|---|---|---|---|
| `aten.all.default` | WORK | HANG | **computes, matches** | `any_from` widened to `f64`; §1 routes it |
| `aten.all.dim` | WORK | HANG | **computes, matches** | same |
| `aten.all.dims` | WORK | HANG | **computes, matches** | same |
| `aten.any.default` | WORK | HANG | **computes, matches** | same |
| `aten.any.dim` | WORK | HANG | **computes, matches** | same |
| `aten.fill_.Tensor` | WORK | HANG | **computes, matches** | §1 unhung it, §3 let it write |
| `aten.index_put_.default` | WORK | HANG | **computes, matches** | same |
| `aten.pow.Tensor_Scalar` | **value-dependent** | HANG | **computes for exponent 0 and 1; refuses `"pow"` otherwise, as upstream does** | §2 |
| `aten.pow.Scalar` | **value-dependent** | HANG | **computes for base 1; refuses `"pow"` otherwise, as upstream does** | §2 |
| `aten.adaptive_avg_pool1d.default` | **refuses every non-empty output size** | HANG | **computes `[0]`; refuses `"sum_cpu"` for `[1]` and `"adaptive_avg_pool2d"` otherwise, as upstream does** | §2 |

### B. Table E (FLOAT8B): upstream computes, this build refused (13)

| op | before | after |
|---|---|---|
| `aten._local_scalar_dense.default` | `NotImplementedError: ...: float8_e4m3fn` | **computes, matches** (§1) |
| `aten.eq.Scalar` | `...: float8_e4m3fn` | **computes, matches** (§1) |
| `aten.eq.Tensor` | `...: float8_e4m3fn` | **computes, matches** (§1) |
| `aten.ne.Scalar` | `...: float8_e4m3fn` | **computes, matches** (§1) |
| `aten.ne.Tensor` | `...: float8_e4m3fn` | **computes, matches** (§1) |
| `aten.abs_.default` | cannot write through F8E4M3 | **computes, matches** (§3) |
| `aten.copy_.default` | cannot write through F8E4M3 | **computes, matches** (§3) |
| `aten.fill_.Scalar` | cannot write through F8E4M3 | **computes, matches** (§3) |
| `aten.mul_.Scalar` | cannot write through F8E4M3 | **computes, matches** (§3) |
| `aten.mul_.Tensor` | cannot write through F8E4M3 | **computes, matches** (§3) |
| `aten.zero_.default` | cannot write through F8E4M3 | **computes, matches** (§3) |
| `aten.mm.default` | `unsupported dtype F8E4M3 for op matmul` | **computes, matches** (§4) |
| `aten.addmm.default` | `unsupported dtype F8E4M3 for op matmul` | **computes, matches** (§4) |

`aten._local_scalar_dense.default`'s FLOAT8B row recorded a `RuntimeError` about
a 2-element tensor. That was the generic recipe's shape, not a float8 property —
upstream raises the same sentence for a 2-element `float32` tensor. On a 1-element
tensor the op refused by dtype, and that is the refusal this round removed.

### C. The three paths from `docs/numerics/FLOAT8.md`, which were never aten ops

| path | before | after |
|---|---|---|
| `t.tolist()` | `NotImplementedError: tolist on float8_e4m3fn` | **computes, matches** |
| `t.item()` | refused via `_local_scalar_dense` | **computes, matches** |
| `t.to(torch.float64)` | **HANG** | **computes, matches** |

`t.to(torch.float64)` was not in any of FLOAT8B's tables — the enumeration ran
over `_aten_implemented()`, and this call reaches `_to_copy` with a dtype the
recipe never supplied. It was still hanging after the `aten.rs` call sites were
routed, and it is the reason §1's fix ended up in `reduced.rs`.

### D. What still refuses, and for what named reason

Nothing in tables D or E. The only float8 refusals left are the **114 where
upstream refuses too**, transcribed kernel by kernel (`docs/numerics/FLOAT8B.md` §2),
plus the three call-shaped ones in §2 above and `matmul` 1-D x 1-D. Every one of
them is upstream's own exception type and upstream's own sentence.

Two things this round did **not** change, both recorded rather than left to be
rediscovered:

- **`adaptive_avg_pool1d` names the wrong kernel for non-float dtypes.** For
  `int64` and `bool` upstream says `"adaptive_avg_pool2d" not implemented for
  'Long'` and this build says `"adaptive_avg_pool1d" ...`. That is a
  pre-existing divergence in wording for other dtypes; the float8 gate added
  here says `adaptive_avg_pool2d` because that is what upstream says for float8.
  Correcting the other dtypes changes behaviour outside this round's scope.
- **`bmm` and non-2-D `matmul`** now have a working float8 kernel underneath
  them (§4 is in `gemm_accumulate_in`, which all of them share), but upstream
  *refuses* `bmm` for this dtype (`"bmm" not implemented`), so the door still
  refuses it. The capability exists and is deliberately not exposed, because
  exposing it would compute where upstream declines.

---

## 6. Verification

### 6.1 Every probe ran in its own process group

`SIGALRM` cannot interrupt a tail-call loop holding the GIL, so each of the 27
probes (23 ops plus the four paths) ran as its own subprocess under a driver
that `killpg`s the group after 25 seconds and records `HANG`. That is how the
before-state was measured and how the after-state was confirmed: **zero `HANG`
results in the final run**, against 11 in the first.

Every probe process is either reaped by the driver or `killpg`ed by it before
the next one starts, so a run leaves nothing behind — which matters for this
dtype specifically, an earlier round on it having left four binaries spinning at
100% CPU for twenty-two hours.

### 6.2 The gate is load-bearing

`rust/torch_c/pytests/test_shim.py` gained seven tests, and one of the previous
round's was rewritten rather than deleted:
`test_float8_shim_only_refusals_do_not_borrow_upstreams_wording` asserted the
ten still refused, and it **failed** when they started computing — which is the
signal it was written to give. It is now
`test_float8_no_op_refuses_in_the_shims_own_words_any_more`, which fails if any
of the ten is put back on the list.

Three golden cases moved from `expect="c_error"` to `expect="match"` for the
same reason: they were written to fail if the gap silently closed, the gap
closed, and they failed. They are stricter now, not weaker — a value comparison
against upstream rather than "some error happened".

### 6.3 Golden coverage for everything newly computing

**None of the 23 ops had a single float8 golden case** before this round — a
case for an op that hangs is a case that hangs the harness. Twenty-two builders
now emit float8 cases, appended at the registry so the additions are one
readable block rather than twenty insertions into unrelated builders. Where
upstream refuses (`pow` at exponent 2, `-1`, `0.5`; `adaptive_avg_pool1d` at
`[1]` and `[2]`) the case is `expect="both_error"`, which fails if **either**
side changes its mind.

    8476 -> 8509 cases, 0 failed, ops covered 203

### 6.4 Gates

    rust/torch_c/pytests/run.sh          EXIT=0, smoke_ok = 387, DOCWATCH: PASS -- 329/329
    tools/golden/compare.py              EXIT=0, SUMMARY: 8509/8509 cases passed, 0 failed,
                                         ops covered=203, pending case builders=0

<!-- DOCWATCH: count smoke_ok ge 380 -->
<!-- DOCWATCH: count golden_cases_total ge 8509 -->
<!-- DOCWATCH: count golden_ops_covered ge 203 -->
<!-- DOCWATCH: count golden_pending eq 0 -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/reduced.rs F8E4M3 present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs widen_f64 present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs float8_pow_refuses present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs F8E4M3 present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_float8_no_op_refuses_in_the_shims_own_words_any_more present -->
<!-- DOCWATCH: symbol-in-file tools/golden/cases.py _float8_extra present -->

---

## 7. What this document does not establish

| # | not established | why |
|---|---|---|
| 1 | that candle's `WithDType for f8e4m3::to_f64` is fixed | it is not. It is still `.L1: jmp .L1`, and any new `to_dtype(DType::F64)` on a float8 tensor written anywhere in this crate will hang. The route in `reduced::to_dtype` is what must be used, and `widen_f64` is the name to grep for |
| 2 | that `f32` accumulation matches upstream's `mm` for **all** inputs | 700 random cases, `k` to 512, is evidence and not proof. The golden cases pin the values that were checked |
| 3 | that the other float8 dtypes behave this way | `float8_e5m2`, `float8_e4m3fnuz`, `float8_e5m2fnuz` and `float8_e8m0fnu` have names in `dtype.rs` and no candle storage. Nothing here was measured against them |
| 4 | that the 114 transcribed refusals are still exactly right | they were re-run as a group by the suite, not re-enumerated op by op against upstream. `docs/numerics/FLOAT8B.md` §2 is the enumeration |
