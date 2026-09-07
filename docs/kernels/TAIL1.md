# The tail's eight one-architecture ops, and QR

`docs/architectures/ARCH100.md` swept 297 architectures and found thirty-one operators left, twenty of which
block exactly one architecture each. This round took eight of them plus `linalg_qr`, which is
`rwkv`'s last wall and the only real numerical kernel in the set.

## 1. Alias or kernel — the split, first

`docs/architectures/ARCH100.md` established that in this tail missing **bindings** outnumber missing **kernels**
49 to 22, so the first question per op was "does a kernel already exist under another name". The
answer, op by op, and it is **not** eight table rows:

| op | alias of an existing kernel? | what actually had to be written |
|---|---|---|
| `broadcast_tensors` | **no** | new kernel: shape fold + `broadcast_as` per entry. One table row *as well*, and the row is the famous half |
| `logical_and` | **no — looks like `bitwise_and` and is not** | new kernel |
| `acos` | **no — candle 0.11.0 has no `acos`** | new kernel (`f64::acos`, element-wise) |
| `_is_all_true` | **yes, mostly** | a dtype guard in front of `all`'s existing reduction, plus two table rows |
| `argsort` (×2 overloads) | **yes** | `order_along`'s indices half, which `sort`/`topk` already share. Two table rows |
| `max_pool1d` | **no — and specifically not `max_pool2d`** | new kernel |
| `upsample_nearest2d` | **no — and specifically not `upsample_bilinear2d`** | new kernel |
| `linalg_qr` | **no** | new kernel: Householder QR, LAPACK `geqrf` + `orgqr` |

**Counted honestly: six new kernels, two reuses of an existing one, nine `_aten_implemented()`
keys, twelve table rows across `overloads.json` and `methods.json`.** The reuses (`_is_all_true`,
`argsort`) are the cheap two; `linalg_qr` alone is more work than the other eight together.

`docs/architectures/ARCH100.md`'s own classification is worth restating because it does not line up with this
table: it called three of these `missing_shim_name` (`TensorBase._is_all_true`,
`torch._C._linalg.linalg_qr`, `torch._C._nn.upsample_nearest2d`) and five `missing_aten_op`. Two
of the three "just a name" ops needed a full kernel; §4 is what happened to the names.

## 2. Two that look like aliases and are not

`docs/kernels/PRIMS.md` §6 found `prims.transpose` **stricter** than `aten.permute` — it refuses a
permutation `permute` accepts — so routing one to the other would have accepted input upstream
rejects. Two ops here are that shape, and one more turned out to be it from the other side.

### 2.1 `logical_and` versus `bitwise_and`

They agree on `bool` and part company on every integer, in the **value and the result dtype**:

```text
self = int64[0, 1, 2, 3]   other = int64[0, 3, 0, 5]
  logical_and  ->  bool  [False, True, False, True]     "each side truthy?"
  bitwise_and  ->  int64 [0, 1, 0, 1]                   "bits in common"
```

`logical_and` answers `torch.bool` for *every* input dtype, including the floating ones where
`bitwise_and` has no kernel at all. Two more measured facts a shortcut gets wrong: **`NaN` is
truthy** (`logical_and([nan, 0.0], [1, 1])` is `[True, False]`, which falls out of `x != 0` and
not out of `x > 0`), and **there is no promotion** — the operands are read at their own dtypes
and only the truthiness is combined.

A bool-only test passes whichever kernel is behind the name, which is why
`test_logical_and_is_not_bitwise_and_on_integers` uses integers and asserts that the two answers
differ on the operands it chose.

### 2.2 `upsample_nearest2d` versus `upsample_bilinear2d`

`docs/architectures/DEMAND8.md` found three separate traps in `upsample_bicubic2d`, each discovered by running
upstream rather than by reading the neighbouring kernel — including that upstream stores the
weights at the **input's** dtype rather than at `opmath_t`. Nearest neighbour has neither trap,
for one reason: **it never averages.** Every output element is one input element copied.

Two things it therefore does not have. There is **no `align_corners`** — it is not in the schema,
so there is no second index formula to choose between. And there is **no `opmath_t` question**,
because there is no arithmetic on values at all. The index rule, transcribed from
`nearest_neighbor_compute_source_index`:

```text
scale = scales.is_some() ? 1/scales : input_size / output_size
src   = min(floor(dst * scale), input_size - 1)
```

Note that an explicit `scales_h` is **inverted**, and that it does not have to agree with
`output_size`. That is the case a plausible wrong implementation passes everything else on:

```text
upsample_nearest2d(4x4, [3,3])            gathers rows [0, 1, 2]    scale 4/3
upsample_nearest2d(4x4, [3,3], 1.5, 1.5)  gathers rows [0, 0, 1]    scale 1/1.5
```

Every call `F.interpolate` makes passes a scale consistent with the sizes, so deriving the scale
from the sizes and ignoring the arguments agrees with upstream on every model path and differs
here. `test_upsample_nearest2d_inverts_an_explicit_scale_rather_than_deriving_one` pins both.

**And the trap fired in the other direction.** The first implementation in this round refused
`uint8`, by inheriting the neighbours' reasoning: `upsample_bilinear2d`/`bicubic2d` have a
separate fixed-point `uint8` kernel upstream, because the rounding of their average *is* the
kernel. Nearest has no average, upstream computes `uint8`, and the float gather answers
upstream's bytes exactly (measured, `4x4 -> 3x3` uint8: `[0,1,2,4,5,6,8,9,10]`, identical to the
`float32` gather). **Refusing it was refusing a dtype upstream answers** — the same class of
mistake as routing an op at its neighbour, arrived at by reading the neighbour instead of
running upstream.

### 2.3 `max_pool1d` versus `max_pool2d` — the same mistake, caught the same way

The shape of the op really is `max_pool2d`'s with one spatial axis, and the extent arithmetic
(including `ceil_mode`'s "drop a window that begins in the right padding") is the same. **The
dtypes are not**, and the first implementation copied the 2-D branch:

```text
this shim's max_pool2d   float32, float64 only; refuses the rest as "max_pool2d"
upstream's max_pool1d    float32, float64, float16, bfloat16
                         refuses uint8/bool/int64 as "max_pool1d_impl"
```

So delegating, or copying the guard, would have **refused two dtypes upstream computes** and
**misnamed the kernel** for the three it refuses. `docs/numerics/PROMOTE.md`'s rule — transcribe the
refusal, do not unify it — is the same rule here.

### 2.4 `argsort` ties, and `stable=`

An unstable sort and a stable one agree on distinct values, so a tie-free test cannot tell them
apart. Measured on upstream 2.13.0 with `[3, 1, 3, 1, 2, 3]`:

```text
argsort()                              [1, 3, 4, 0, 2, 5]
argsort(stable=True)                   [1, 3, 4, 0, 2, 5]
argsort(stable=False)                  [1, 3, 4, 0, 2, 5]
argsort(descending=True)               [0, 2, 5, 4, 1, 3]
argsort(stable=False, descending=True) [0, 2, 5, 4, 1, 3]
```

and an 80-element all-ties tensor answers `0..79` with and without `stable=`. **All four
combinations are stable on CPU**, in both directions, so `order_along` — already stable, and
already shared with `sort.default` — is right for both overloads. `stable=False` *licenses* an
arbitrary order; what is recorded above is that today upstream does not take the licence, which
is what makes one shared implementation a transcription rather than a guess. The test asserts
the agreement between `argsort.stable` and `argsort` rather than assuming it, so a future
upstream that stopped agreeing shows up here.

One thing about the *call* rather than the kernel, found by the golden harness on its first run:
`torch.ops.aten.argsort.stable(x, True, -1, False)` raises `takes 1 positional argument(s) but 4
was/were given`. Everything past `self` is keyword-only at that binding, not only the argument
the schema stars. Passing them positionally made sixteen golden cases report SILENT DIVERGENCE —
torch refusing while the shim computed — which is the harness working exactly as intended.

## 3. `broadcast_tensors`, and what it does and does not unblock

`docs/architectures/ARCH100.md` counts `broadcast_tensors` as one architecture (`gemma3n_text`). The reason to
lead with it is `docs/training/BACKWARD9.md` §1, which had to spell its criterion `((o - t) ** 2).mean()`
because `nn.MSELoss` reaches `torch.broadcast_tensors` and this shim refused by name.

The op itself is `CompositeImplicitAutograd` upstream, like `max_pool2d.default` and
`matmul.default` already in `aten.rs`, so implementing it is following upstream's decomposition.
Three measured facts:

* **dtypes are not unified.** `broadcast_tensors(int64, float32)` answers `(int64, float32)`;
  only the shapes meet. `cat`'s `promote_list` fold here would be the plausible wrong move and is
  invisible on any single-dtype input.
* an **empty list** answers an empty list, not an error.
* the entries have to be **promoted** on the way out. They leave inside a `PyList`, which
  `promote` does not look into — the same trap `unbind.int` and `finish_ordered` already carry a
  comment for. Without it every element is a bare `TensorBase`, and the very next thing
  `F.mse_loss` does to one is a `Tensor` method.

**`nn.MSELoss` is still not reachable, and the wall is one step further on.** `F.mse_loss` calls
`torch.broadcast_tensors` — which now works — and then `torch._C._nn.mse_loss`, which is a
raising stub in `bootstrap.py`. That is the second wall, of exactly the shape `docs/architectures/DEMAND8.md`
recorded for `new_empty` (`torch.maximum` behind it). `docs/training/BACKWARD9.md`'s hand-spelled
criterion stands until an `_nn.mse_loss` binding and an `aten.mse_loss` kernel land.
`test_two_kernels_still_have_no_python_spelling_and_it_is_one_line_each` pins that, and fails the
day it stops being true.

## 4. Three ops have kernels and no way to call them

> **Closed since.** `upsample_nearest2d` and `linalg_qr` are bound in `bootstrap.py` now and
> both allowlist entries are gone — `docs/bindings/BINDINGS.md`. `vilt` clears its wall; `rwkv` gets
> past `torch.linalg.qr` and stops one line later at `torch.diag`. `nn.MSELoss` is still shut,
> so the third of the three below stands. The section is left as written because its argument
> (why these could not be `overloads.json` rows) is what the fix was built on.

`upsample_nearest2d` and `linalg_qr` are reached upstream as `torch._C._nn.upsample_nearest2d`
and `torch._C._linalg.linalg_qr` — **submodule bindings, not `torch.<name>` entries**. There is
no `torch.upsample_nearest2d` and no `torch.linalg_qr` on 2.13.0 (checked, both directions), so
no `overloads.json` row can reach them: a row would invent a `torch.<name>` upstream does not
have, which is exactly what `docs/bindings/SPELLINGS.md` refuses.

The binding for each is one line in `bootstrap.py`, beside the `upsample_bilinear2d` /
`upsample_bicubic2d` that `_install_nn` already writes and the `_linalg.linalg_vector_norm` that
`bootstrap.py:2780` already writes. **This round did not own `bootstrap.py`,** so the state is
pinned rather than fixed, in three places that all fail when it changes:

* `tools/golden/reach_allow.json` — one `shape2_kernel_without_spelling` entry each, each naming
  the fix. An entry whose gap has closed fails the suite, so closing it deletes the entry.
* `test_tail1.py::test_two_kernels_still_have_no_python_spelling_and_it_is_one_line_each` —
  runs the vendored tree in a subprocess and asserts all three (including `nn.MSELoss`) still
  refuse.
* this section.

**So `vilt` and `rwkv` do not clear their walls this round**, and `gemma3n_text` does. Reporting
"eight ops implemented, eight architectures cleared" would be the counting `CLAUDE.md` §5.3
warns about. §7 has the sweep.

A note on `reach.py` worth carrying forward: `_nn` gives every name in its stub list a *raising*
stub, so `hasattr(torch._C._nn, "upsample_nearest2d")` is `True` while the op is unreachable.
`reach.py` was not fooled by that — it reported the gap — which is worth recording because the
attribute being present is exactly how this could have looked closed.

## 5. QR: the one that is real numerical work

`rwkv`'s wall is at *construction*: `_init_weights` calls `nn.init.orthogonal_`, which draws a
normal matrix and calls `torch.linalg.qr`. Nothing in the model runs until this answers
(`docs/kernels/PRIMS.md` §6, `docs/architectures/DEMAND8.md`).

A QR factorisation is unique only up to the signs of `R`'s diagonal, and upstream's signs are
LAPACK's. **Getting the decomposition right and the convention wrong answers a `Q` with flipped
columns that is still orthogonal and still satisfies `Q @ R == A`** — and is not upstream's
answer. The convention lives entirely in `dlarfg`:

```text
xnorm = ||x[j+1..]||
if xnorm == 0:   tau = 0, beta = alpha        <-- H is I; alpha keeps its sign
else:            beta = -sign(alpha) * hypot(alpha, xnorm)
                 tau  = (beta - alpha) / beta
                 v    = x / (alpha - beta)
```

The `xnorm == 0` short circuit is the branch that is easy to miss and it decides the whole
identity case: without it, `beta = -sign(alpha)*||x||` unconditionally, and `qr(eye(3))` answers
`R = -I` where upstream answers `+I` (measured). That is the single most likely wrong answer
here and it looks entirely reasonable.

**Agreement, measured against upstream at `float64` over eleven matrices** (square, tall, wide,
`complete` mode, `1x1`, `eye`, sign-flipped, already-triangular): `max |dQ| <= 6.4e-16` and
`max |dR| <= 2.9e-14`. That is machine precision — the same factorisation and the same
convention, not merely a compatible one. The reference was validated in Python against LAPACK
*before* any Rust was written, which is what made the `dlarfg` branch visible.

At `float32` this shim is **more accurate than upstream, not differently accurate**: upstream's
`R[0][1]` for the classic `[[12,-51,4],[6,167,-68],[-4,24,-41]]` is `-21.000003814697266` where
the exact value is `-21`, a relative `1.8e-07` — inside `float32`'s `1e-5` golden tolerance.
Arithmetic here is at `f64` for both supported dtypes.

### 5.1 The case where nothing agrees, and why that is not a defect

On a rank-deficient input (`[[1,2],[2,4],[3,6]]`) the trailing columns of `Q` are determined by
rounding noise. Upstream's own `R[1][1]` there is `8.8e-07` at `float32` and `~1e-16` at
`float64`; this implementation and LAPACK disagree by `0.104` in that column while both remain
orthogonal to `1e-16`. **Upstream promises nothing there.**

So the golden cases keep element-wise comparison to full-rank matrices, and `test_tail1.py`
covers the rank-deficient input by the properties that *are* determined: `R` upper-triangular,
`Q`'s columns orthonormal to `1e-12`, and `Q @ R == A` to `1e-12`. Choosing a well-conditioned
matrix and calling QR done is the thing this section exists to have not done.

### 5.2 Refusals

Two of them, and they cannot share a branch because upstream's do not:

```text
float16 / bfloat16   NotImplementedError: "geqrf_cpu" not implemented for 'Half'
int64 / int32 / ...  RuntimeError: linalg.qr: Expected a floating point or complex
                     tensor as input. Got Long
```

`mode="r"` answers a **one-dimensional empty** `Q` (`shape == (0,)`), batched or not — measured,
not inferred from `(m, 0)`.

## 6. What is verified, and how

Every op is compared against upstream element-wise, on inputs where a plausible wrong
implementation differs.

* **Golden harness** (`tools/golden/cases.py`, run against upstream in a separate process):
  <!-- DOCWATCH: count golden_cases_total ge 9360 -->
  <!-- DOCWATCH: count golden_cases_passed ge 9360 -->
  <!-- DOCWATCH: count golden_ops_covered ge 233 -->
  <!-- DOCWATCH: count golden_pending eq 0 -->
  9360 cases pass of 9360, 233 ops covered, 0 pending builders.
* **`pytests/test_tail1.py`**, a new suite file (`run.sh` globs `pytests/test_*.py`; the file
  shares helpers with `test_shim` the way `test_split_probe.py` does).
* Three pinned counts in `test_shim.py` moved and each carries the arithmetic that keeps it a
  check: `tag_core_count` 110 → 112 (`acos` and `logical_and` are the only two of the nine that
  upstream tags `core` — each read off its own `.tags`), distinct schema identities 295 → 302
  (+7, not +12: five schemas went into *both* tables and a pair of doors onto one schema is one
  identity), and the underscore-prefixed `_shim_overloads` keys gained `_is_all_true`.

The ops now in `_aten_implemented()`:

<!-- DOCWATCH: op-implemented aten.broadcast_tensors.default -->
<!-- DOCWATCH: op-implemented aten.logical_and.default -->
<!-- DOCWATCH: op-implemented aten.acos.default -->
<!-- DOCWATCH: op-implemented aten._is_all_true.default -->
<!-- DOCWATCH: op-implemented aten.argsort.default -->
<!-- DOCWATCH: op-implemented aten.argsort.stable -->
<!-- DOCWATCH: op-implemented aten.max_pool1d.default -->
<!-- DOCWATCH: op-implemented aten.upsample_nearest2d.default -->
<!-- DOCWATCH: op-implemented aten.linalg_qr.default -->

and the table rows that make six of them callable:

<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json broadcast_tensors present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json logical_and present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json acos present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json argsort present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json max_pool1d present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json _is_all_true present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json _is_all_true present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json argsort present -->

There is deliberately **no** `torch.linalg_qr` or `torch.upsample_nearest2d` row, and upstream
has neither name:

<!-- DOCWATCH: hasattr linalg_qr false -->
<!-- DOCWATCH: hasattr upsample_nearest2d false -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json linalg_qr absent -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json upsample_nearest2d absent -->

### 6.1 One entry outside this round's territory

`rust/torch_c/src/device.rs`'s `MPS_HOST_READBACK_OPS` gained four names (`acos`, `linalg_qr`,
`max_pool1d`, `upsample_nearest2d`) and its length went 56 → 60. That list is a **safety
ratchet**: an op that reads device bytes back to the host and is not on it computes a wrong
answer on `mps` instead of refusing, and `test_the_mps_readback_list_is_what_the_kernels_actually_do`
names the four and the file. The edit is data only — four sorted strings and the array length —
and it is recorded here rather than made quietly, because `device.rs` was outside the files this
round was given.

## 7. The sweep after

`pytests/arch_sweep.py`, re-run. What the nine ops actually move:

* **`gemma3n_text`** clears `broadcast_tensors`.
* **`longt5`** clears `logical_and`. **`yoso`** clears `acos`. **`canine`** clears `max_pool1d`.
  **`aria`, `aria_text`, `vit_mae`** clear `argsort`. **`vits`** clears `_is_all_true`.
* **`vilt` and `rwkv` do not clear**, and the reason is §4, not the kernel: the kernels are
  implemented and golden-compared, and the `_nn` / `_linalg` binding that names them is in
  `bootstrap.py`.

Several of the eight that do clear hit a **second wall immediately**, which is a result rather
than a failure — `docs/architectures/DEMAND8.md` records the same shape for `new_empty` (`torch.maximum` behind
it) and this round's own `broadcast_tensors` → `_nn.mse_loss` is another. §8 has the per-
architecture landing.
