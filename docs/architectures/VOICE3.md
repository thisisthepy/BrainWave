# VOICE3 — the walls seven rounds stopped at, by name

Worktree `work/voice3` on develop `2498122`. Every operator here was found by a round *hitting*
it, not by inventory: `rwkv`'s `diag`, `vilt`'s `_unique2`, `llama4`'s `im2col`, `f5-tts`'s
`col2im`, `bigvgan`'s `kaiser_window`, `voice`'s `upsample_nearest1d` and `var`, and — added
mid-round — `gemma3n_text`'s `std`.

Upstream torch 2.13.0 (`/Volumes/macMini/caches/spike-venv/bin/python`) is the oracle
throughout, in its own process with `PYTHONPATH` stripped.

---

## 0. The alias-versus-kernel split, first

`docs/architectures/ARCH100.md` measured missing *names* outnumbering missing *kernels* 49 to 22 in this tail,
so every operator below was looked for under another spelling **before** a kernel was written.
The answer this round is unusual: **none of the eight was an alias.**

| candidate | looked for under | verdict |
|---|---|---|
| `var` | `native_batch_norm`'s statistics | **kernel.** `native_batch_norm_default` computes a *biased* (correction 0) variance over the batch-and-spatial axes only, folds it straight into `alpha`/`beta`, and never materialises it. There is no reduction to borrow. |
| `std` | `var` | **kernel.** `torch.std` dispatches to `aten::std.correction` as a **leaf** — it is not a composite over `var` — and it is *not* bit-equal to `var(...).sqrt()`. §4. |
| `upsample_nearest1d` | `upsample_nearest2d` | **kernel.** The index arithmetic is shared and this file shares it, but the op has its own schema, its own rank check (`input_size equals to 3`) and its own dtype-refusal kernel name (`compute_indices_weights_nearest`, where the 2-D op says `upsample_nearest2d_channels_last`). |
| `col2im` | `im2col` | **kernel, and not its inverse.** `col2im` **sums** overlapping windows. §5. |
| `i0` | anything | **kernel.** Nothing in the shim had a modified Bessel function. |
| `diag` | `tril`/`triu`/`eye` | **kernel, and two of them.** §3. |
| `_unique2` | `sort` | **kernel.** Shares `cmp_torch_f64`'s float order and nothing else. |
| `kaiser_window` | `hann_window` | **kernel**, though `periodic`'s rule *is* `hann_window`'s and was re-measured rather than assumed. |

So "eight ops" understates the naming and overstates the bodies at the same time:

```
15 registrations   diag, _unique2, im2col, col2im, i0,
                   kaiser_window x3, upsample_nearest1d,
                   var x3, std x3
 8 kernel bodies   var's three and std's three all reach `var_reduce`;
                   kaiser_window's three reach `kaiser_window_default`
```

`test_this_round_landed_fifteen_registrations_over_eight_kernels` pins both numbers so a later
round cannot report fifteen kernels.

And the tags say the same thing from upstream's side. Of the fifteen, exactly **three** are
`core` upstream — `col2im.default`, `var.dim`, `var.correction` — and each was read off its own
`.tags`. `col2im` is core while `im2col`, landed in the same change, is not; `var.correction` is
core while `std.correction`, which this shim implements with the *same function*, is not.

---

## 1. What cleared

`pytests/arch_sweep.py --only rwkv vilt llama4 gemma3n_text`, both sides. Upstream is 4/4.

| architecture | before | after |
|---|---|---|
| `gemma3n_text` | forward — `torch.std` | **forwards** |
| `llama4` | — | **forwards** (see the caveat below) |
| `vilt` | forward — `aten._unique2` | forward — `torch.multinomial(Tensor, Tensor)` |
| `rwkv` | construction — `torch.diag` | construction — `TensorBase.t_` |

### 1.1 `rwkv`: `diag` was not the last one

It has moved wall by wall for four rounds (`new_empty` → `maximum` → `linalg_qr` → `diag`), and
the honest answer to "is `diag` the last one" is **no**. The new wall is *four lines earlier* in
the same function, and that is the interesting part:

```python
# torch/nn/init.py, orthogonal_
    if rows < cols:
        flattened.t_()        # <- 705, the new wall
    q, r = torch.linalg.qr(flattened)
    d = torch.diag(r, 0)      # <- 710, the wall that just fell
```

`t_()` is inside a branch that the *first* module weight `rwkv._init_weights` reaches did not
take. With `diag` working, initialisation gets further through the module list and reaches a
weight whose `rows < cols`. So this is forward progress that looks like a regression in the line
number, and `t_` — an in-place transpose, docs/kernels/INPLACE.md's territory — is what is next.

### 1.2 `vilt`: one line further

```python
# modeling_vilt.py
144:  unique_rows = valid_idx[:, 0].unique()          # <- was the wall
155:  valid_choice = torch.multinomial(torch.ones(v).float(), max_image_length)
```

The new wall is classified `unsupported_arg_form`, not a missing kernel: `multinomial` is
implemented, and `max_image_length` arrives as a **tensor** where the schema says
`SymInt num_samples`. Cheaper than a kernel.

### 1.3 `llama4`: forwards, but this round does not claim it

`im2col` is `llama4`'s **vision tower** (`modeling_llama4.py:976` is
`torch.nn.Unfold(kernel_size=..., stride=config.patch_size)`), and `F.unfold` binds
`torch._C._nn.im2col`, which **still refuses** (§6). So the sweep's `from_config` does not build
the vision tower, and `llama4` forwarding is not attributable to this round's work. Stated here
rather than counted, because "an architecture cleared" is the number this repository is most
often tempted to over-claim.

### 1.4 `gemma3n_text`: attributable

`modeling_gemma3n.py:984` is `torch.std(inputs, dim=-1, keepdim=True, unbiased=False)` — the
`std.dim` overload with the *old* spelling of the correction. Nothing else in the round could
have moved it.

---

## 2. `i0`, and why one point is not a check

**This is the op the round was warned about, and the warning was right twice.**

`calc_i0` is two Chebyshev approximations that meet at `x == 8`: one in `exp(-x) I0(x)` on
`[0, 8]`, one in `exp(-x) sqrt(x) I0(x)` above. They agree in the middle of each interval and
diverge at the ends, so the check is a **sweep** — 88,002 points over `[0, 88]` at a step of
`0.001`, through the junction and out to where `float32` overflows — compared as **raw bit
patterns**, not float reprs.

Two things had to be reproduced, and each was found by the sweep rather than by reading:

**(a) The `float` kernel runs float-rounded coefficients.** `chebyshev_coefficients_i0e_A` is
`static const T coeff[]` inside a template that upstream instantiates at `T = float` as well as
at `T = double`. So upstream answers

```
i0(0.0f) == 0.9999999403953552        not 1.0
```

Computing in `f64` and narrowing once gives exactly `1.0`, and over `[0, 8]` the shortcut is
wrong by up to **4.83e-07 relative** (at `x = 7.675`) — comfortably inside the tolerance any
value comparison at `float32` would use. This is `hann_window`'s finding (docs/architectures/VOICE.md §4.1) a
second time, which is why the rule is now written down rather than rediscovered.

**(b) clang contracts the recurrence into an FMA.** `chbevl`'s three-term step is written
`b0 = x * b1 - b2 + array[i]`, and C++ is compiled with `-ffp-contract=on` by default, so it is
emitted as `fma(x, b1, -b2) + array[i]` — *one* rounding where the source has two. Rust does not
contract. The difference is small and concentrated exactly where the recurrence cancels:

| chbevl written as | bit-identical to upstream, 88,002 points, `f32` | worst relative |
|---|---|---|
| `x * b1 - b2 + c` | 86,055 (97.79%) | 3.58e-07 at `x = 0.013` |
| `x.mul_add(b1, -b2) + c` | **88,002 (100.000%)** | 0 |

and at `f64`, likewise **88,002 of 88,002**. Both are in `test_voice3.py`, one as the sweep and
one as a single assertion on `i0(0.0f)`'s bit pattern, which is the cheapest possible witness
that the shortcut did not creep back in.

`f16`/`bf16` go through the `f32` path and narrow once — upstream's own
`calc_i0(c10::Half a) { return calc_i0(float(a)); }`. NaN and both infinities fall out of the
branch rather than being cased (`NaN <= 8.0` is false, and `exp(inf)*finite/sqrt(inf)` is NaN,
which is what upstream answers for `i0(inf)` — it is *not* `inf`).

## 2.1 `kaiser_window`

`bigvgan`'s construction wall, docs/architectures/VOICE.md rank 7, open since that round because it needs
`i0`. Upstream:

```
window_length == 0  ->  empty         (before any other check)
window_length == 1  ->  ones(1)
periodic            ->  window_length += 1, last sample narrowed off at the end
alpha  = (window_length - 1) / 2      AFTER the increment
out[i] = i0(beta * sqrt(1 - ((i - alpha)/alpha)^2)) / i0(beta)
```

`beta` defaults to `12.0`. Three things were measured rather than carried over from
`hann_window`: `periodic` **truncates** rather than re-parameterising (so `kaiser_window(L, True)`
and `kaiser_window(L, False)` are both `L` long and share exactly their first element — asserted
as a count, `shared == 1`); `beta = 0` is an answer and not a degenerate case (`i0(0)/i0(0)`, so
the window is all ones); and the refusals are upstream's own wording.

**Bit-identical to upstream** at `float32` and `float64`, and the reduced floats narrow once at
the end.

---

## 3. `diag` is two operations, and `rwkv` only uses one

A 1-D input **builds** a matrix; a 2-D input **extracts** a diagonal. `rwkv` takes the second
branch (`r` is a QR's triangular factor), and implementing only that would leave `torch.diag(v)`
— the far commoner spelling — answering nothing. Measured, and the last three are what a guess
gets wrong:

```
diag(3x4,  0)      -> 3 elements         min(rows, cols - d)
diag(3x4, -1)      -> 2                  rows - |d|
diag(3x4,  5)      -> shape (0,)         an EMPTY answer, not an error
diag(3x4, -5)      -> shape (0,)         likewise
diag([1,2,3], -2)  -> 5x5, not 3x3       the offset GROWS the matrix
```

`bool` and the integral dtypes compute. Rank 0 and rank 3 are upstream's own
`"diag(): Supports 1D or 2D tensors. Got {n}D"`.

---

## 4. `var` and `std` — three traps, two of them found by measuring

**(a) `correction` defaults to 1, not 0.** Bessel's. `unbiased=` is the old spelling
(`True` is `correction=1`). Getting it wrong scales `var` by `n/(n-1)` and `std` by
`sqrt(n/(n-1))` — 0.1% and 0.05% at `n = 1000`, which every tolerance in this repository would
accept, and a factor of 2 and 1.41 at `n = 2`. **So the test is at `n = 2`**, on
`var([1., 3.]) == 2.0` and `std([1., 3.]) == sqrt(2)`.

**(b) The divisor is CLAMPED AT ZERO, and the difference is `inf` versus `nan`.** Upstream is
`n = max(0, n - correction)` and then a plain division:

```
n - correction  > 0                  ->  m2 / (n - correction)
n - correction == 0,  m2 >  0        ->  inf     var([1., 3.], correction=2)
n - correction == 0,  m2 == 0        ->  nan     var(tensor(3.))
n - correction  < 0,  m2 >  0        ->  inf     var([1., 3.], correction=3)   -- not -inf
n - correction  < 0,  m2 == 0        ->  nan     var(zeros(0))
```

All five measured. **A first draft of this kernel answered `nan` for every non-positive
denominator** — which is right for the two cases the obvious tests reach, because both have
`m2 == 0`, and wrong for the two where `m2 > 0`. The obvious tests are `var` of a scalar and
`var` of an empty tensor, i.e. exactly the two a person writes. The negative-correction case is
the one that says it is a *clamp* and not a sign convention.

**(c) `std` shares `var`'s accumulation but not its exit, and that is the whole reason the two
share a body.** The coordinator's question was whether `std` accumulates differently. It does
not — but **the square root is taken in the wide accumulator, before the single narrowing**:

| | agrees with upstream `torch.std`, 1200 random `float32` samples |
|---|---|
| `sqrt` in the `f64` accumulator, then narrow | **1199** |
| narrow to `float32`, then `sqrt` | 1043 |

which is the same statement as: **`torch.std(x)` is not `torch.var(x).sqrt()`**. Measured, they
disagree on **53 of 288** (dtype, shape, dim, correction) combinations, always by one ULP,
because the composite narrows first. `test_std_takes_the_root_in_the_accumulator_not_on_the_narrowed_variance`
asserts that split from both sides on a sample where upstream itself shows it, so a shim that
implemented `std` as `var(...).sqrt()` fails it.

Accuracy against upstream, 3000 random `float32` reductions:

```
var    3000 / 3000 bit-identical
std    2998 / 3000 bit-identical, the two differing by 1 ULP
```

The residual is upstream's *vectorised* Welford reduction order, not the algorithm: at `f64`
neither serial Welford nor a two-pass reproduces upstream's blocking (327 and 250 mismatches in
600), and at `f32` a wide accumulator makes both exact. The two-pass is the closer of the two
and is what this kernel does.

---

## 5. `im2col` / `col2im` — the overlap *is* the op

`col2im` **sums** where windows overlap; it does not overwrite with the last one.

```
col2im(ones(1, 4, 9), output=[4,4], k=2, s=1)     1 2 2 1
                                                  2 4 4 2
                                                  2 4 4 2
                                                  1 2 2 1
```

**A stride equal to the kernel cannot see this** — every output element is covered exactly once,
and a summing and an overwriting implementation both answer all ones. So every case in the
golden set and the test file uses `stride < kernel`, and the one `stride == kernel` case is kept
only for the contrast, labelled as the case that *cannot* discriminate. `col2im(im2col(x))` is
therefore not `x` but `x` weighted by the cover count, and asserting the round trip equals `x`
would be asserting the bug.

`im2col`'s folded layout has the **channel as the slowest axis**:

```
out[n][c*kh*kw + i*kw + j][oh*out_w + ow] = in[n][c][oh*sh - ph + i*dh][ow*sw - pw + j*dw]
```

A **1-channel image cannot distinguish that from its transpose** — same shape, same numbers — so
the golden set and the test both carry a 2-channel case for exactly this.

Dtypes are backwards from most of this file and were measured, not inferred: **`bool` computes
and the integral dtypes do not** (`"im2col_out_cpu" not implemented for 'Int'`, and the same for
`'Byte'`).

## 5.1 `upsample_nearest1d`

docs/architectures/VOICE.md rank 14. The index rule is `nearest_neighbor_compute_source_index` —
`floor(dst * scale)`, clamped, no half-pixel correction — and **the `scales` argument is
inverted, not used directly**. `scales=1.5` with `output_size=[3]` gathers `[0, 0, 1]`, not
`[0, 1, 2]`; deriving the ratio from the sizes instead gives the second, and every call
`F.interpolate` makes would hide the difference. `uint8` computes (a gather never averages, so
there is no rounding for a fixed-point kernel to do differently) and `bool` does not — the one
place its dtype set differs from `upsample_nearest2d`'s.

---

## 6. What is NOT closed, and who has to close it

Two hand-offs, both in files this round was told not to edit. Each is asserted from this side so
neither can be forgotten.

**(a) `MPS_HOST_READBACK_OPS` in `rust/torch_c/src/device.rs`.** Twelve of the fifteen
registrations pull the tensor to the host and must be declared there:

```
aten.diag.default          aten.i0.default            aten.var.default
aten._unique2.default      aten.im2col.default        aten.var.dim
aten.upsample_nearest1d.default                       aten.var.correction
aten.col2im.default                                   aten.std.default
                                                      aten.std.dim
                                                      aten.std.correction
```

`kaiser_window`'s three are the exception — they build a window from scalars and never touch an
input tensor, so they must **not** be added.

`var` and `std` were nearly invisible here, and that is worth recording. The derivation in
`test_the_mps_readback_list_is_what_the_kernels_actually_do` scans each dispatch target's body
and follows helper calls **one level, by name**; `var_reduce` is not on that list, so a
`read_flat` inside it reached the derived set through nothing. `var`/`std` would have been the
one family in this file that silently computed on the CPU with an `mps` tensor, with no test
saying so. The read is now spelled at each of the six dispatch targets, which puts all twelve on
the derived set — which is how the list above was produced rather than written by hand.

**(b) Three `_install_nn` entries in `rust/torch_c/src/bootstrap.py`.** `im2col`, `col2im` and
`upsample_nearest1d` are reached upstream *only* through `torch._C._nn.*` — there is no
`torch.im2col` and no `Tensor.im2col` (checked), so a row in `overloads.json`/`methods.json`
would invent a door upstream does not have. Each needs one entry shaped like
`upsample_nearest2d`'s. They are in `tools/golden/reach_allow.json` under
`shape2_kernel_without_spelling`, in the same form and for the same stated reason as the
`reflection_pad*` entries docs/kernels/PAD.md §5 left there, and
`test_the_three_nn_bindings_are_the_only_thing_still_between_these_kernels_and_F` asserts the gap
from both sides — the aten op answers correctly, and the `_nn` name refuses **naming itself**.
That test fails the moment the bindings land, which is when it and the allowlist entries should
be deleted together.

Until (b) lands, `F.unfold`, `F.fold` and `F.interpolate(..., mode="nearest")` on a 3-D input
still refuse, which is why §1.3 does not claim `llama4`.

---

## 7. Numbers

```
suite            706 ok        (668 before)  -- 1 failing, and only (a) above
golden       10653 / 10653     (10039 before),  ops covered 285  (270 before)
docwatch       619 / 621 PASS  (616 markers before) -- the 2 failures are `smoke_ok = 479`,
                               which is the same single test in (a)
test_voice3.py    39 tests
```

No timings: four other agents were running (docs/perf/PERF.md's rule).

<!-- DOCWATCH: op-implemented aten.diag.default -->
<!-- DOCWATCH: op-implemented aten._unique2.default -->
<!-- DOCWATCH: op-implemented aten.i0.default -->
<!-- DOCWATCH: op-implemented aten.im2col.default -->
<!-- DOCWATCH: op-implemented aten.col2im.default -->
<!-- DOCWATCH: op-implemented aten.upsample_nearest1d.default -->
<!-- DOCWATCH: op-implemented aten.kaiser_window.default -->
<!-- DOCWATCH: op-implemented aten.kaiser_window.periodic -->
<!-- DOCWATCH: op-implemented aten.kaiser_window.beta -->
<!-- DOCWATCH: op-implemented aten.var.default -->
<!-- DOCWATCH: op-implemented aten.var.dim -->
<!-- DOCWATCH: op-implemented aten.var.correction -->
<!-- DOCWATCH: op-implemented aten.std.default -->
<!-- DOCWATCH: op-implemented aten.std.dim -->
<!-- DOCWATCH: op-implemented aten.std.correction -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json diag present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json i0 present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json var present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json std present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json kaiser_window present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json _unique2 present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json diag present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json var present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json std present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs calc_i0_f32 present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs kaiser_window_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs var_reduce present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs col2im_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_voice3.py test_i0_is_bit_identical_to_upstream_across_the_whole_range present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_voice3.py test_col2im_sums_overlapping_windows_rather_than_overwriting present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_voice3.py test_var_and_std_clamp_the_divisor_at_zero_so_an_overshoot_is_inf_not_nan present -->
