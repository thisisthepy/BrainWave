# DEMAND2 — closing ranks 2, 3 and 4 of the re-ranked demand list

Working round for `docs/architectures/DEMAND.md` §0.1 as it stood after docs/architectures/DEMAND1.md's re-ranking: rank 1
(the legacy `torch.Tensor(...)` constructor) is **out of scope** and untouched, same as
DEMAND1.md left it. This round closes ranks 2 (`aten.squeeze.default`), 3
(`aten.linalg_vector_norm.default`) and 4 (`torch.linspace`).

Method: same as DEMAND1.md — `transformers` 5.15.1, `torch` 2.13.0 upstream as the oracle,
`torch/torch._C._linalg` reached through the vendored tree with
`PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1`, upstream through
`env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`, `print("shim" if hasattr(torch._C,
"_aten_implemented") else "upstream")` as the first line of every script. Build:
`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-kern2`,
`TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib`, `cargo build --release` in
`rust/torch_c`, then `bash vendor/install_shim.sh`. Scratch scripts under `/tmp/` (not
committed).

---

## 1. `aten.squeeze.default` / `aten.squeeze.dims` — the GOLDEN.md blind spot, and it was two bugs

`squeeze` is declared in both `overloads.json` and `methods.json` with three schemas:

```text
aten::squeeze(Tensor(a) self) -> Tensor(a)
aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)
```

`aten.rs`'s dispatch `match` (checked directly in the built binary, not by grepping source) had
an arm only for `.dim`. **Both `.default` and `.dims` were unreachable** — `_aten_implemented()`
listed only `aten.squeeze.dim`, and calling `x.squeeze()` or `x.squeeze(dim=(0,1))` raised `aten
op not implemented in torch._C shim: aten.squeeze.default` / `...dims`. `mbart`'s
`shift_tokens_right` calls the bare `squeeze()` form (docs/architectures/DEMAND1.md §2's model-table row); no
model measured so far calls `.dims` — that hole was found while landing `.default`, not by a
model hitting it.

### 1.1 `.default` — every axis of size 1, not just one

Measured: `squeeze(zeros(1,3,1,2)).shape == (3, 2)` — both size-1 axes go, in one call, not one
at a time. A 0-d tensor has nothing to remove and comes back unchanged. Implemented by walking
the tensor's own dims from the last axis to the first, squeezing each size-1 one — descending so
a removal never shifts the index of an axis not yet visited (candle's `squeeze(dim)` renumbers
axes above `dim` down by one).

### 1.2 `.dims` — a per-axis version of `.dim`'s no-op rule, with its own duplicate check

Measured against upstream, `x = zeros(1,3,1,2)`:

```text
squeeze(dim=(0,2))     (3, 2)      both size-1 axes named -- both removed
squeeze(dim=(0,1))     (3, 1, 2)   axis 1 has size 3 -- PARTIAL no-op, only axis 0 removed
squeeze(dim=())        (1, 3, 1, 2) no-op -- NOT "every axis" the way linalg_vector_norm/norm's
                                    empty dim list is (§2 below)
squeeze(dim=(-4,-2))   (3, 2)      same two axes, negative
squeeze(dim=(0,0,2))   RuntimeError: dim 0 appears multiple times in the list of dims
squeeze(dim=(1,1))     RuntimeError: dim 1 appears multiple times in the list of dims
                        -- raised even though axis 1 is NOT size 1: the duplicate check runs
                        BEFORE any size is looked at, so this does not silently no-op
squeeze(dim=(0,-4))    RuntimeError: dim 0 appears multiple times in the list of dims
                        -- a positive/negative pair that normalises to the same axis
0-d, dim=(0,)           ()          torch treats a 0-d tensor as one-dimensional for indexing
0-d, dim=(0,0)          RuntimeError: dim 0 appears multiple times in the list of dims
                        -- the duplicate check fires even on a 0-d tensor
```

The duplicate-dim message (`"dim {d} appears multiple times in the list of dims"`) is the exact
wording `aten.norm.ScalarOpt_dim`'s existing kernel already reproduces for the same upstream
check, transcribed on **normalised** indices in both places (a positive/negative pair that
normalises to the same axis reports the normalised value). Factored into one shared
`refuse_duplicate_dims` helper in `aten.rs` rather than copied a third time, used by
`squeeze.dims`, `linalg_vector_norm.default` (§2) and (unchanged) `norm.ScalarOpt_dim`.

### 1.3 What landed

`aten.rs`: `squeeze_default` and `squeeze_dims`, both new functions, plus the shared
`refuse_duplicate_dims` extracted from `norm.ScalarOpt_dim`'s previously-inline duplicate check.
Both new dispatch keys added to `IMPLEMENTED` and the `match`. No `overloads.json`/`methods.json`
change — the three schemas were already there; only the dispatch arms were missing.

---

## 2. `aten.linalg_vector_norm.default` — a distinct leaf, sharing `norm.ScalarOpt_dim`'s walk

Confirmed distinct from `aten.norm.ScalarOpt_dim`, both by `_dispatch_has_kernel_for_dispatch_key`
returning `False` for it too (docs/architectures/DEMAND1.md §5) and by upstream registering it as its own
schema:

```text
aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *,
                          ScalarType? dtype=None) -> Tensor
```

Reached as `torch._C._linalg.linalg_vector_norm`, not `torch.linalg_vector_norm` — upstream has
no top-level name either (`torch.linalg.vector_norm` is a distinct, Python-level function).
`_linalg` had no stub data in `surface.json` (it is in `EXTRA_SUBMODULES`, the list of
submodules the vendored tree's stubs do not declare but the tree imports anyway), so every name
on it fell through to the generic catch-all `_Unimplemented` before this round — the refusal
`docs/architectures/DEMAND.md`'s model table quoted (`"not implemented in torch._C shim:
torch._C._linalg.linalg_vector_norm"`) is that catch-all's wording, not the aten dispatcher's.
`bootstrap.py`'s `install()` now sets `module._linalg.linalg_vector_norm` to a real resolving
function built the same way `torch.<op>` names are (`_torch_level_function`, reading the same
`overloads["linalg_vector_norm"]` table `torch.<op>` resolution uses) right after the submodule
loop creates `_linalg`.

### 2.1 Sharing the walk

docs/architectures/DEMAND1.md §7 said explicitly: "doing it properly means sharing `norm.ScalarOpt_dim`'s
existing six-arm accumulate-in-`opmath` walk rather than writing a second one — the two compute
the same `ord` family and a second copy would drift." That refactor is what this round does.
The walk (`shape`/`dims_set`/`read_flat` through the accumulate loop and `write_flat`, previously
~140 lines inlined in `norm_scalaropt_dim`) is now `norm_pow_walk(op, t, tag, dims_set, p,
keepdim) -> PyResult<Tensor>`, taking the accumulate/output dtype (`tag`), the already-normalised
reduction axes (`dims_set`) and the `ord`/`p` value, and doing nothing else — no argument
parsing, no dtype checks, no error wording. Both `norm_scalaropt_dim` and the new
`linalg_vector_norm_default` parse their own arguments (which differ, per §2.2 below) and then
call the same function. `norm_scalaropt_dim`'s own behaviour is unchanged — verified by re-running
its full golden suite (all cases in `norm_scalaropt_dim_cases`, including the accumulate-at-`acc_t`
sweep and the known 1-ULP pairwise-summation residual) after the extraction; all still pass with
the same `value_check`.

### 2.2 The four ways this op is not `norm.ScalarOpt_dim` under a different name

All four measured against upstream 2.13.0 with `torch._C._linalg.linalg_vector_norm` directly
(`from torch._C._linalg import linalg_vector_norm as lvn`):

**`dim` is `int[1]?`, not `int[1]`.** Both an omitted `dim` and an explicit `dim=[]` reduce every
axis — the same rule `norm.ScalarOpt_dim` already has for its always-present `dim` when it is
`[]`. `dim=[0, 1]` (an explicit full list on a rank-2 tensor) reduces to the same scalar.

**The dtype-mismatch wording is this op's own.** `norm()`'s is `"norm(): input dtype should be
either floating point or complex. Got {name} instead."`; this op's is `"linalg.vector_norm:
Expected a floating point or complex tensor as input. Got {name}"` — no trailing `"instead."`.
Fires for an integral/bool input regardless of whether `dtype=` is also given: `lvn(zeros(3,
dtype=int64), 2, dtype=torch.float32)` still refuses this way rather than casting first.

**`dtype=` promotes before reducing, and only widening is allowed.** Measured the full pairwise
matrix across the four floating dtypes this shim stores:

```text
float32 -> float64   OK           float16 -> float32   OK
float64 -> float32   narrows, refused        bfloat16 -> float32  OK
float32 -> bfloat16  narrows, refused        float16 -> bfloat16  narrows, refused (same width!)
bfloat16 -> float16  narrows, refused (same width, the other direction)
```

Upstream's wording: `"linalg.vector_norm: the dtype of the input ({src}) should be convertible
without narrowing to the specified dtype ({dst})"`. `Half` and `BFloat16` refuse each other
despite being the same bit width — there is no total order that makes one of them "wider," so a
simple width comparison would have got this wrong; a small tier table (`Half`/`BFloat16` at tier
1, `Float32` at 2, `Float64` at 3, refuse unless the destination tier is strictly greater) matches
every measured pairing.

**The empty-reduction split.** `ord=2` on an empty reduction is `0.0` (the same identity
`norm.ScalarOpt_dim` already has, unmeasured against emptiness before this round but not touched
here either — `norm.ScalarOpt_dim`'s own golden case for an empty `(2,0)` tensor at `p=2` still
passes unchanged). `ord=±inf` on an empty reduction **refuses**, and upstream's message has two
different shapes depending on how `dim` was spelled, not on whether the reduction happens to
cover every axis:

```text
dim=None or dim=[]   (whole tensor empty)  "...on an empty tensor because the operation
                                            does not have an identity"
dim=[explicit, non-empty]  (names an axis) "...on the dimension {d}because this dimension is
                                            empty and..."   (no space before "because" --
                                            upstream's own concatenation, transcribed)
```

The named dimension is **the first entry, in the given list's own order, whose extent is 0,
printed exactly as the caller spelled it** — not normalised, and not necessarily the smallest
index:

```text
dim=[2, 1] on (2,0,0)     names 2   (first in list order, extent 0)
dim=[1, 0] on (0,3)       names 0   (dim 1 has extent 3, nonzero -- skipped; dim 0 is empty)
dim=[-1, -2] on (2,0,0)   names -1  (the RAW value, not the normalised 2)
```

An axis that is empty but is **not** in the reduced set never raises:
`lvn(zeros(0,3), inf, dim=[1])` answers the empty `(0,)` tensor with no error — there are zero
output rows, so there is nothing to compute over and no missing identity to report.

Negative `ord` over a row of zeros — the case docs/architectures/DEMAND1.md §5's sibling doc comment on
`norm.ScalarOpt_dim` calls out (`|0|^-1 = inf`, sum is `inf`, `inf^(-1) = 0`, and a correct
implementation must **not** special-case zero to avoid this) — was re-measured on
`linalg_vector_norm` directly and gives the identical answer, confirmed by construction: it is
the same `norm_pow_walk` general-power arm, unchanged.

### 2.3 `sentence_embed` end to end

`F.normalize(v, p=2, dim=1)` fires `linalg_vector_norm.default, clamp_min.default,
expand.default, div.Tensor`; the other three already had kernels, so this one closes the model.
Verified two ways:

- Direct comparison of `torch._C._linalg.linalg_vector_norm` against upstream across the six-arm
  `ord` family, `dim=None`/`[]`/explicit, `keepdim`, `dtype=` promotion and narrowing refusal, the
  empty-reduction split (both message shapes), and the duplicate-dim refusal — all match, and
  are now golden cases (§4).
- **A full `bert`-shaped `sentence_embed` forward**, `AutoModel.from_config` (2 layers, hidden 32,
  4 heads, vocab 99) → mean-pool over `attention_mask` (one row padded, to exercise the mask
  path) → `F.normalize(mean_pooled, p=2, dim=1)`, run on both the shim (through the vendored
  tree) and upstream with **identical hand-built `input_ids`/`attention_mask`** (not
  `torch.randint` — see the aside below on why). Weights bit-identical (`torch.manual_seed(0)`
  before construction, max diff `0.0` across all 39 state-dict tensors); `last_hidden_state` max
  abs diff `3.6e-07`; final normalized embeddings max abs diff `3.0e-08` — both at the same
  float32-eps scale docs/architectures/DEMAND1.md §2's `bert` row reports (`8.9e-08`). The model runs to
  completion and matches upstream; it did not before this round (`NotImplementedError` at
  `torch/nn/functional.py:6100`).

**Aside, not ranked, found while building the end-to-end check.** The first attempt used
`torch.manual_seed(1)` + `torch.randint(0, 99, (2, 6))` for `input_ids`, the same recipe
docs/architectures/DEMAND1.md §1 describes ("`torch.manual_seed(1)` before any randomly-generated input, so
both sides see the same pixels/waveform"). The two sides' `input_ids` **did not match**
(`[[22, 14, 27, 77]]` upstream vs `[[37, 14, 47, 25]]` shim for the same seed/shape/range),
which cascaded into a large, spurious forward-value divergence that had nothing to do with this
round's kernels. Confirmed **pre-existing and unrelated to this round**: built a second artefact
from unmodified `develop` (`git stash`, a clean `cargo build --release` into a separate
`CARGO_TARGET_DIR`, then `git stash pop` to restore this round's changes) and reproduced the
identical `input_ids` mismatch against it. Weight initialisation (`torch.manual_seed` +
`nn.init`) is confirmed bit-exact (§2.3 above, `0.0` diff over 39 tensors) — this is specifically
`aten.randint.low`'s RNG stream disagreeing with upstream's for at least this
shape/low/high combination, despite having a kernel, an `_aten_implemented()` entry, and its own
golden cases (`randint_low_cases`) that apparently do not cover the disagreement. Not
investigated further and not ranked — no model in either DEMAND round has hit it, and this
round's own model table has no vote for it — recorded here so the next round does not have to
re-discover it if a model does.

### 2.4 What landed

`aten.rs`: `norm_pow_walk` (the extracted shared walk, §2.1), `linalg_vector_norm_default` (new),
`refuse_narrowing_dtype` (new, §2.2), `refuse_duplicate_dims` (shared, §1.2). `norm_scalaropt_dim`
rewritten to call `norm_pow_walk` instead of inlining the walk; its own argument parsing, dtype
check and error wording are unchanged. `overloads.json` gains a `linalg_vector_norm` entry
(`.default` and `.out`, both declared in upstream's `native_functions.yaml` — checked, so this is
not a torchgen-only table addition). `bootstrap.py`'s `install()` wires
`module._linalg.linalg_vector_norm` right after the submodule-building loop.

---

## 3. `torch.linspace` — a construction-time leaf, fetched and read rather than guessed

`ConvNextModel.__init__`'s stochastic-depth rate schedule (docs/architectures/DEMAND.md's model table:
`"NotImplementedError: not implemented in torch._C shim: torch.linspace(...)"` at
`modeling_convnext.py:219`, before any forward runs). Leaf upstream
(`_dispatch_has_kernel_for_dispatch_key("aten::linspace", "CompositeImplicitAutograd")` is
`False`), with four overloads (`default`, `Tensor_Tensor`, `Tensor_Scalar`, `Scalar_Tensor`, plus
their `.out` siblings) — only `.default` (`Scalar start, Scalar end`) is implemented, since that
is the only spelling `ConvNextModel` or any measured model calls; the Tensor-argument forms are
declared nowhere and would refuse with the ordinary "no matching overload" message if a caller
reached for one.

Given the brief's own warning that getting this from reasoning alone is easy to get wrong,
upstream's actual kernel source was fetched (network access confirmed working, same as
docs/architectures/DEMAND1.md §3's note on HF Hub reachability) and read directly:
`aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp::linspace_kernel`, torch tag `v2.13.0`.

### 3.1 `steps`

```text
linspace(0, 10, 5)     [0, 2.5, 5, 7.5, 10]     the textbook case
linspace(0, 10, 1)     [0.]                     steps=1 answers [start]; end is not read
linspace(3, 999, 1)    [3.]                     confirms end is genuinely not read, not just
                                                 coincidentally equal
linspace(0, 10, 0)     []                       steps=0 is NOT an error
linspace(0, 10, -1)    RuntimeError: number of steps must be non-negative
```

### 3.2 dtype

```text
linspace(0, 10, 5).dtype              float32   -- the default float dtype, NOT int64 the way
                                                    arange's all-integral rule would suggest,
                                                    even though every argument here is an int
linspace(0, 9, 5, dtype=int64)        [0, 2, 4, 6, 9]     truncated toward zero, from the exact
                                                            schedule 0, 2.25, 4.5, 6.75, 9 -- NOT
                                                            rounded to [0, 2, 5, 7, 9]
linspace(-9, 0, 5, dtype=int64)       [-9, -6, -4, -2, 0] truncation toward zero on the negative
                                                            side too: -6.75 -> -6, not -7 (floor)
linspace(0, 10, 5, dtype=torch.bool)  RuntimeError: "linspace_cpu" not implemented for 'Bool'
```

The bool refusal, and the same for `uint16`/`uint32`/`uint64`, comes straight from upstream's own
dispatch macro (`AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, ...)` — no `Bool`, no
wide unsigned types), reproduced with a `linspace_has_cpu_kernel` allow-list rather than
special-cased per type.

### 3.3 The endpoint is exact, by construction

The source's whole shape:

```cpp
using step_t = std::conditional_t<std::is_integral_v<scalar_t>, double, scalar_t>;
const scalar_t start = scalar_start.to<scalar_t>();
const scalar_t end = scalar_end.to<scalar_t>();
const step_t step = (static_cast<step_t>(end) - static_cast<step_t>(start)) / (steps - 1);
int64_t halfway = steps / 2;
// per element, index idx:
if (idx < halfway) { return start + step * idx; }
else               { return end - step * (steps - idx - 1); }
```

Two things this makes exact that a single forward accumulation (`start + i*step` for every `i`)
would not: `end` is reached by direct subtraction at `idx = steps - 1` rather than by summing
`step` `steps - 1` times, and the split at `steps / 2` means neither half's error can grow past
half the range. For **integral** `scalar_t`, `step_t` is always `double` regardless of the
output's own width (`start`/`end` are already truncated to the integral output type before the
`double` arithmetic runs, matching `arange`'s own `.as_i64()` convention rather than emulating
narrower integer wraparound, which `arange`'s existing int path does not do either). For
**floating** `scalar_t` (`Float`, `Double`), `step_t` is the *same* primitive type, so `start`,
`end` and `step` are ordinary hardware `float`/`double` values with no wrapper — and the literal
`a + b*c` / `a - b*c` shape of the per-element expression is exactly the pattern C++ compilers
fold into one hardware fused-multiply-add (single rounding) under ordinary `-ffp-contract`
behaviour. This was **measured**, not assumed: a plain two-step `start + step*i` computed at
matching precision disagrees with upstream on the least-significant bit for a non-trivial
fraction of cases (e.g. `linspace(0.1, 0.3, 7)` at `float32`: element 4 of 7 differs by 1 ULP),
while `f32`/`f64::mul_add` — Rust's own fused multiply-add, `self*a+b` in one rounding — matches
upstream exactly across a 500-case random sweep spanning `float32` and `float64`, varied
`start`/`end`/`steps` (including 100000-step and 257-step cases, where the forward/backward split
crosses the halfway point many times).

**`Half`/`BFloat16` do not get this treatment.** `at::Half`/`at::BFloat16` are C++ classes, not
primitives — every operator call promotes to `float`, computes, and narrows back individually,
so there is no single fused expression for a compiler to contract. Verified the same way: an FMA
simulation (single rounding for the whole `start + step*i`) does **not** match upstream at these
two dtypes, while narrowing after each individual step (using this shim's existing
`float_narrower`, already the established convention for `Half`/`BFloat16` fidelity elsewhere in
this file) matches across 200 random `float16` trials.

### 3.4 What landed

`aten.rs`: `linspace_default`, `linspace_has_cpu_kernel`, `linspace_float_values` (the
`Float`/`Double`/`Half`/`BFloat16` split, §3.3), `linspace_int_values` (the integral arm, §3.2).
`overloads.json` gains a `linspace` entry (`.default` and `.out`, both declared in upstream's
`native_functions.yaml`, same check as `linalg_vector_norm`'s §2.4). `linspace` was already in
`surface.json`'s `varfns` list (it is a real `_C._VariableFunctions` member upstream), so no
`bootstrap.py` change was needed to expose `torch.linspace` itself — only the table entry and the
kernel.

---

## 4. Golden coverage

Four new `CASE_BUILDERS` entries — one per new `_aten_implemented()` key (`squeeze.default` and
`squeeze.dims` share nothing with each other's builder despite sharing a kernel shape, since
golden compares by dispatch key):

- `squeeze_default_cases` — every-axis-of-1 removal, no-op (no size-1 axes), fully-squeezed
  (all-1s shape to 0-d), already-0-d, four dtypes.
- `squeeze_dims_cases` — full removal, partial no-op (a named axis not size 1), empty list as a
  no-op (not "every axis"), negative dims, out-of-range refusal, duplicate-dim refusal (both the
  "not size 1" and the positive/negative-pair shapes).
- `linalg_vector_norm_default_cases` — the six-arm `ord` family across `dim=None`/explicit/`[]`
  and `keepdim`; `dtype=` promotion (three widening pairs) and narrowing refusal (four pairs,
  including the `Half`/`BFloat16` same-width case); integral/bool input refusal with this op's
  own wording; the duplicate-dim refusal; the empty-reduction split at both message shapes
  (named-dimension and whole-tensor), each for `ord=inf` and `ord=-inf`. Deliberately smaller
  than `norm_scalaropt_dim_cases`'s own dtype-accumulation sweep — `norm_pow_walk` is shared and
  already pinned there, so this suite concentrates on the four ways this op differs (§2.2) rather
  than re-measuring the walk a second time.
- `linspace_default_cases` — `steps` at 0/1/many, the negative-`steps` refusal, `start == end`,
  a non-exact step (`0.1`/`0.3`/`7`, `value_check=_bit_exact`), a many-step case that exercises
  the forward/backward split (257 steps), four floating dtypes (`value_check=_bit_exact`) and
  four integer dtypes including truncation and a negative range, the `bool` refusal.

`ops covered` moved from **185 to 189** (four `_aten_implemented()` entries, four kernels — one
apiece; `squeeze.default`/`.dims` reuse `.dim`'s candle call shape but are their own dispatch
keys, `linalg_vector_norm.default` shares `norm_pow_walk`).

---

## 5. Gates

Built via the instructed pipeline throughout:
`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-kern2`,
`TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib` exported before every run, `cargo build
--release` in `rust/torch_c`, `bash vendor/install_shim.sh` after.

```
$ PYTHON=$PY sh rust/torch_c/pytests/run.sh
348 ok
SELF-TEST: PASS -- 21 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised
DOCWATCH: PASS -- 276/276 evaluated marker(s) hold

$ $PY tools/golden/compare.py
SUMMARY: 8240/8240 cases passed, 0 failed, ops covered=189, pending case builders=1

$ $PY tools/golden/compare.py --self-test
SELF-TEST: PASS -- 21 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised

$ $PY rust/torch_c/pytests/verify_schemas.py
SUMMARY: 4588/4588 table entries matched upstream, 0 failed
```

Two pre-existing running-count assertions in `rust/torch_c/pytests/test_shim.py` needed updating
for this round's four new `_aten_implemented()` entries and two new `overloads.json` schema
pairs, each with a comment explaining the movement in the file's own established style (matching
every prior round's entries in the same two tests):

- `test_core_ops_and_op_tags_agree`: `tag_core_count` `101 -> 102`. Only `squeeze.dims` is
  upstream `core`-tagged among the four new kernels (`squeeze.default`,
  `linalg_vector_norm.default` and `linspace.default` are not); `squeeze.dim`, already
  `core`-tagged, was already counted before this round.
- `test_schema_text_survives_the_round_trip_through_the_transcribed_tables`: `len(keys)`
  `271 -> 275`, from the two new `overloads.json` entries (`linspace`, `linalg_vector_norm`),
  each contributing two `(qualname, overload)` pairs (`default` and `out`). `squeeze` contributes
  nothing here — its three overloads were already in the table; only the dispatch arms were
  missing, a different table from the one this test counts.

`git status --short` was checked before, during and after this round; nothing outside
`docs/architectures/DEMAND.md`, `docs/architectures/DEMAND1.md` (one marker flip, §6 below), `docs/architectures/DEMAND2.md` (this file),
`rust/torch_c/src/aten.rs`, `rust/torch_c/src/bootstrap.py`, `rust/torch_c/src/overloads.json`,
`rust/torch_c/pytests/test_shim.py` and `tools/golden/cases.py` moved.
`rust/torch_c/src/tensor.rs`, `dtype.rs`, `flash.rs`, `bootstrap.py`'s untouched regions, and the
vendored tree were not modified (`tensor.rs` in particular — another agent's worktree was
reported editing it concurrently; nothing in this round touched it).

## 6. DEMAND.md and DEMAND1.md re-ranked

`docs/architectures/DEMAND.md` §0 re-ranked: §0.1 now carries only rank 1 (the legacy `torch.Tensor(...)`
constructor, untouched, out of scope); §0.2 gains three rows (ranks 2, 3, 4, this file) ahead of
the five DEMAND1.md already closed, renumbered 5-7 with HM unchanged. Four markers flipped from
`op-not-implemented` to `op-implemented` (`aten.linalg_vector_norm.default`,
`aten.squeeze.default`) or added fresh (`aten.squeeze.dims`, `aten.linspace.default` — neither had
a marker before this round). `docs/architectures/DEMAND1.md`'s own `op-not-implemented
aten.linalg_vector_norm.default` marker (§8 there) is flipped to `op-implemented` too, for the
same reason every other closed marker in both files is flipped rather than deleted: leaving it
asserting absence would fail DOCWATCH forever now that the kernel exists, and a silent deletion
would have quieted the gate instead of updating it.
