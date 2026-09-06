# DEMAND8 — the `mobilenet_v2` divergence, bisected; and four missing ops

Worktree `work/aten6` on develop `9f4557e`. torch 2.13.0 upstream
(`/Volumes/macMini/caches/spike-venv/bin/python`) is the oracle throughout; every number below
was produced by running it, not by choosing a tolerance.

---

## 1. `mobilenet_v2` — it is float32 rounding, not a defect

docs/DEMAND7.md §1 recorded `mobilenet_v2` as the one model on the list that **forwards and does
not match** (output max abs diff 9.40e-04), and ranked it #2 as a "correctness bug". It is not a
bug. This section is the measurement that says so, and the method is the part worth keeping,
because the golden harness structurally cannot answer this question: it compares one op at a
time, and an accumulated divergence only exists in the composition.

### 1.1 Setup

`MobileNetV2Config(num_channels=3, image_size=64, depth_multiplier=0.5)`, weights generated once
upstream and saved to `.npz`, then **loaded from that file by both runs** so that the two are
comparing arithmetic and not initialisation. One detail mattered and is worth recording: with
freshly-initialised weights and untouched BatchNorm running stats (`running_var=1`), activations
**underflow to ~1e-23** by the last block, and every relative number measured on that model is
meaningless. The BatchNorm running statistics were therefore calibrated with one forward pass in
train mode at `momentum=1.0` (i.e. the running stats become real batch statistics, which is what
a trained checkpoint has), giving activations at O(1) throughout — `last_hidden_state` scale 6.0.

Activations were captured with a forward hook on **every** module (211 tensors), on both sides.

### 1.2 End to end: the divergence is real, and it grows smoothly

Shim against upstream, relative to the tensor's own scale, in module order:

```text
conv_stem.first_conv.convolution   abs=8.94e-08  scale=4.59e-01  rel=1.95e-07
layer.0.reduce_1x1                 abs=1.91e-06  scale=4.70e+00  rel=4.06e-07
layer.7   (mid-network)            abs=1.57e-05  scale=7.00e+00  rel=2.2e-06
layer.15.reduce_1x1                abs=1.11e-04  scale=6.01e+00  rel=1.84e-05
__final__ (last_hidden_state)      abs=1.28e-04  scale=6.00e+00  rel=2.13e-05
__pooled__ (pooler_output)         abs=6.99e-05  scale=4.48e+00  rel=1.56e-05
```

There is **no step**. It starts at 1.95e-07 — one float32 ulp — after the very first convolution
and grows monotonically through 53 conv+BN layers. That shape is already an argument against a
defect (a wrong kernel produces a jump at the layer that uses it), but it is not proof, so two
further measurements were made.

### 1.3 Per-op, on identical inputs: the largest single-op error is 2 ulp

Upstream's **input** to every leaf module was recorded (140 of them), and each module was then
re-run under the shim on that recorded input, so nothing accumulates. Worst offenders, of all 140:

```text
rel=2.910e-07  layer.12.reduce_1x1.normalization  BatchNorm2d  in=(1, 80, 2, 2)
rel=2.304e-07  layer.14.conv_3x3.normalization    BatchNorm2d  in=(1, 480, 2, 2)
rel=2.265e-07  conv_stem.conv_3x3.normalization   BatchNorm2d  in=(1, 16, 32, 32)
rel=1.949e-07  conv_stem.first_conv.convolution   Conv2d       in=(1, 3, 65, 65)
rel=1.913e-07  layer.15.conv_3x3.convolution      Conv2d       in=(1, 480, 4, 4)  [depthwise]
```

`float32` eps is 1.19e-07, so the worst op in the whole network is **2.4 ulp**, and the worst
*absolute* single-op error is 1.4e-06 on a tensor of scale 4.9.

**Three of the four suspects the task named are bit-exact, measured, not assumed:**

```text
ReLU6 / hardtanh          rel = 0.000e+00 on all 35 occurrences
AdaptiveAvgPool2d         rel = 0.000e+00      (pooler, in=(1, 1280, 2, 2))
grouped/depthwise Conv2d  rel <= 1.9e-07       (e.g. layer.15.conv_3x3, groups=480)
```

The fourth, `native_batch_norm`'s fused affine, is the largest — and 2.4 ulp is exactly what
docs/DEMAND1.md predicts for a differently-associated but equally-valid arrangement of
`(x - mean) * rsqrt(var + eps) * w + b`. It is not a wrong formula; a wrong formula does not land
within 2 ulp.

### 1.4 The control that settles it: both sides against a float64 oracle

Per-op error being small does not by itself prove the *accumulated* 1.28e-04 is acceptable — the
question is whether upstream's own float32 run is any closer to the exact answer. The same model
was run upstream in **float64** and both float32 runs compared against it:

```text
                      conv_stem     layer.7      __final__            __pooled__
upstream f32 vs f64   2.323e-07     2.012e-06    1.768e-05  (1.06e-04 abs)   8.231e-06
shim     f32 vs f64   2.155e-07     1.836e-06    1.715e-05  (1.03e-04 abs)   9.867e-06
shim vs upstream      —             —            1.28e-04 abs               6.99e-05 abs
```

**Upstream's own float32 answer is 1.06e-04 away from the exact answer; the shim's is 1.03e-04
away — marginally closer.** The 1.28e-04 the two differ by is the sum of two independent float32
truncation paths, each of which is individually ~1e-04 from truth. There is nothing left for a
defect to be.

On upstream's run-to-run variation: upstream is **bit-deterministic** here — `torch.set_num_threads(1)`
and `(8)` give abs diff exactly `0.000e+00` at every one of the 211 tensors. So "upstream varies by
X" is not the right yardstick for this model; the right one is the float32 truncation error above,
which upstream incurs in full.

### 1.5 Verdict

**Not a defect. Float32 rounding, amplified by depth.** Magnitude: 1.28e-04 absolute on a tensor
of scale 6.0 (2.1e-05 relative) in this configuration; DEMAND7's 9.40e-04 is the same phenomenon
at its larger configuration's larger scale. Reference point: upstream's own float32 error against
float64 is 1.06e-04 on the same tensor — i.e. **the shim-vs-upstream gap and upstream's own
distance from the truth are the same size.**

DEMAND7 §3's rank 2 ("correctness bug") should be struck. The right statement is that
`mobilenet_v2` matches upstream to within float32 accumulation over a 53-layer network, which is
what "matches" means for every other row on that table too — the difference is only that
`mobilenet_v2` is deep enough for the noise floor to be visible at 1e-04 rather than 1e-07.

**What this measurement could not have caught**, stated because §5.4 of CLAUDE.md asks for it:
a defect that upstream's float64 path shares with the shim (it does not — the two implementations
are unrelated), or a defect that only fires on inputs outside this one calibrated configuration.
The per-op replay in §1.3 is the part that generalises least by shape and most by op: it says
these 140 module instances at these shapes agree to 2 ulp, not that every shape does.

<!-- DOCWATCH: op-implemented aten.hardtanh.default -->
<!-- DOCWATCH: op-implemented aten.adaptive_avg_pool2d.default -->
<!-- DOCWATCH: op-implemented aten.convolution.default -->

---

## 2. The four missing ops

All four were **re-verified as still open** on this checkout before anything was written, by
running them against the built shim rather than trusting DEMAND7's list:

```text
torch.floor / Tensor.floor / Tensor.floor_   no table entry for this op
TensorBase.ndimension                        not implemented in torch._C shim
TensorBase.index_add_  (and index_add)       not implemented in torch._C shim
torch._C._nn.upsample_bicubic2d              not implemented in torch._C shim
```

Split by kind, which is the split that decides how much work each is:

| op | kind | what landed |
|---|---|---|
| `floor` / `floor_` | missing kernel + missing spelling | kernel in `aten.rs`, entries in `overloads.json` and `methods.json` |
| `ndimension` | **spelling only** | one method on `tensor.rs`'s `PyTensorBase`; **no kernel, no table entry** |
| `index_add_` | missing kernel + missing member spelling | kernel in `aten.rs`, entry in `methods.json` only |
| `upsample_bicubic2d` | missing kernel + `_nn` binding | kernel in `aten.rs`, `.vec`/leaf binding in `bootstrap.py` |

### 2.1 `ndimension` is a spelling, and that was checked rather than assumed

The task's guess was right, and here is what "checked" means: on torch 2.13.0
`torch.Tensor.ndimension is torch.Tensor.dim` is **`False`** — they are two distinct method
objects — but they return the same `int` at every rank, including `0` for a 0-d tensor. There is
no `aten::ndimension` schema and `hasattr(torch, "ndimension")` is `False`, so it gets **no**
`overloads.json` entry, **no** `aten.rs` dispatch arm, and **no** golden case builder: it is one
method beside `dim` in `tensor.rs`, returning `dims().len()`. Adding a free function would have
invented a surface upstream does not have.

This is exactly the case docs/SPELLINGS.md §9 warns about: the golden harness calls
`_aten_dispatch` with a dispatch key, so a name-only addition is **invisible** to it. `ndimension`
is checked in exactly one place, `test_demand8_four_names_reach_their_kernels_through_the_vendored_tree`,
through a real `import torch` against the vendored tree.

### 2.2 `floor` — measured against `ceil`, not copied from it

`ceil` was already implemented and is the obvious template. Every rule was re-measured anyway, and
one differs in the way that matters:

```text
floor(-0.5)   -1.0     ceil(-0.5) is -0.0, and the sign bit is the point there
floor(-0.0)   -0.0     signbit True, checked with math.copysign through the vendored tree
floor(inf)    inf      floor(nan) nan
arange(3)     [0,1,2]  integral dtypes are the IDENTITY, not a refusal
bool          NotImplementedError: "floor_vml_cpu" not implemented for 'Bool'
```

The kernel name in the refusal is **`floor_vml_cpu`**, a different string from `ceil_vml_cpu`,
read off a real error. `floor.out` is declared in `overloads.json` beside the bare form and has no
kernel — `torch.floor(x, out=y)` refuses naming `aten.floor.out` — which is the same honest
half-coverage `ceil` already ships rather than a new gap.

### 2.3 `index_add_` — the in-place treatment, plus five rules that are not `index_put_`'s

Write-through via `aten.rs::write_back` (a view taken before the call sees the write;
`x.index_add_(...) is x`), and capture refuses it automatically because
`capture.rs::is_mutating` reads the trailing `_` — measured at the raw-dispatch layer rather than
assumed, with two out-of-place controls from the same round that must **not** poison.

Where it differs from its nearest neighbour `index_put_`, all measured:

```text
negative index      index_put_ WRAPS; index_add_ raises "index out of range in self"
expanded receiver   index_put_ is on write_back's Overlap::Allow list; index_add_ REFUSES
                    ("more than one element of the written-to tensor refers to a single
                     memory location") -- so no table entry was added
alpha               keyword-only (a fourth positional is a TypeError), and cast to the
                    receiver's dtype: alpha=2.5 and 2.9 both behave as 2 on int64
two shape errors    a wrong extent ALONG dim reports "Number of indices (N) should be equal
                    to source.size(dim)"; any other axis reports "source tensor shape must
                    match self tensor shape, excluding the specified dimension"
accumulation        at the receiver's dtype per step, not in f64 -- 64 accumulations of
                    bfloat16(0.01) give 0.65234375 running, 0.640625 if summed wide
```

`uint8` wraps on overflow (`200 + 200` is `144`) and `bool` accumulates as a logical or, both for
`index_put_`'s reasons and both re-measured here.

### 2.4 `upsample_bicubic2d` — the `align_corners` convention, taken from upstream by running it

Both conventions were run, both ways, on `arange(16).reshape(1,1,4,4)` -> `(6,6)`:

```text
align_corners=False   -0.4340  0.0590  0.8657  1.4398  2.2465  2.7396
align_corners=True     0.0000  0.5040  1.2480  1.7520  2.4960  3.0000
```

They agree at the four corners, which is what the flag *means*, so a corner-only or symmetric case
cannot separate them. **Three traps, each measured rather than inherited from
`upsample_bilinear2d`:**

1. **`align_corners=False` does not clamp the source index at 0 for cubic.** Bilinear does;
   upstream's `area_pixel_compute_source_index` takes a `cubic` template parameter whose only job
   is to skip that clamp. That is why the first element above is **negative** — a value the input
   does not contain and a clamped implementation cannot produce.
2. **There is no `out == in` short circuit.** Bilinear copies the axis; bicubic resamples.
   `(1,1,2,3) -> [2,3]` with `scales=(0.5, 0.5)` returns
   `[[1.9062, 3.5938, 3.5], [3.4062, 5.0938, 5.0]]` upstream. Copying bilinear's short circuit
   would have returned the input and looked entirely reasonable.
3. **The cubic weights are stored at the input's dtype, not at `opmath_t`.** Upstream's separable
   CPU path builds a `scalar_t` weight tensor in `compute_indices_weights_cubic<scalar_t>` — the
   kernel the `int64` refusal is named after. With `f32` weights and an `f32` accumulator,
   `float16` disagrees by 7.8e-03 against a 5e-03 tolerance and `bfloat16` by 6.25e-02 against
   6e-02; narrowing the weights first makes **both bit-exact**.

The remaining `float32` residual is **5.7e-07 max relative** (6.7e-06 absolute at magnitude 15),
which is accumulation order inside upstream's vectorised kernel, not a model disagreement — the
weights were extracted from upstream by pushing 16 one-hot basis inputs through it in `float64`,
and this kernel's model reproduces that 6x6x4x4 weight tensor to **1.6e-15**.

`uint8` is refused by name, with its own measurement rather than bilinear's inherited one: over 40
random shapes at both flag values (1840 elements), rounding the `float32` answer disagrees with
upstream's `uint8` answer on **140**, so upstream runs a separate fixed-point kernel there.

Both spellings of the binding are accepted, because upstream's is overloaded and both are real
calls: the four-argument `.vec` shape `F.interpolate` uses
(`torch/nn/functional.py:5286`) and the five-argument leaf with `scales_h`/`scales_w`.

### 2.5 Coverage — including what fails, run rather than assumed

**Golden**: one `CASE_BUILDERS` entry per new kernel (`floor_cases`, `floor__cases`,
`index_add__cases`, `upsample_bicubic2d_cases`), plus a `_view_write_cases` entry for `floor_`.
Each builder states in its docstring which plausible wrong implementation each case separates.
`ndimension` has **no** builder and that is correct — it has no dispatch key, so the harness's
coverage rule does not ask for one, and §2.1's road test is where it is checked instead.

One defect was found by a case rather than by review, and it is worth recording because the case
would otherwise have looked green: two `Case`s sharing the tensor names `s_t`/`i_t`/`v_t` both
closed over the *variables*, so the "a 2-D index is refused" case silently ran the 0-d tensors of
the case written after it. It reported `torch_ok=True c_ok=True` on an `expect="both_error"` row,
which is the only reason it surfaced. Both now bind their tensors as lambda defaults.

**Smoke, through the vendored tree** (`test_demand8_four_names_reach_their_kernels_through_the_vendored_tree`,
docs/SPELLINGS.md §9's road-script pattern): all four names through a real `import torch` against
this shim, both doors where two exist, every expected value transcribed from upstream 2.13.0 run
separately. The subprocess asserts it actually imported the shim rather than upstream, which is
the failure mode a road script has that a unit test does not.

**Capture** (`test_capture_refuses_demand8_inplace_names_and_lets_the_others_through`): `floor_`
and `index_add_` poison a capture region by name at the raw-dispatch layer; `floor.default` and
`upsample_bicubic2d.default`, out-of-place ops from the same round, must record cleanly. Without
the second half, a refusal rule broad enough to poison everything this round touched would pass.

**Sabotage — run, not assumed:**

```text
removed tensor.rs's `ndimension` (cp backup), rebuilt, reinstalled:
  FAIL test_demand8_four_names_reach_their_kernels_through_the_vendored_tree:
  AssertionError: ndimension_3d: expected 3, got
  'ERROR:NotImplementedError:not implemented in torch._C shim: TensorBase.ndimension'

removed overloads.json's `floor` entry (cp backup), rebuilt, reinstalled:
  FAIL test_demand8_four_names_reach_their_kernels_through_the_vendored_tree:
  AssertionError: floor_fn: expected [1.0, -3.0, -1.0, 3.0], got
  'ERROR:NotImplementedError:... torch.floor(...) -- overload resolution has no table entry
   for this op (rust/torch_c/src/overloads.json)'
  FAIL test_schema_text_survives_the_round_trip_through_the_transcribed_tables: 290
```

Each named exactly the entry it lost, and neither is a case the golden harness could have caught:
`ndimension` has no dispatch key at all, and `floor`'s golden cases go through `_aten_dispatch`
and stayed green with the spelling deleted. Both restored from the `cp` backup, rebuilt,
reinstalled, reran green.

### 2.6 Where the five models stop now

Toy `AutoConfig`s, hand-built inputs, `torch.manual_seed(0)` before construction on both sides —
the shim reproduces upstream's RNG, so the two runs hold **identical weights** and the comparison
is a real numeric one rather than a shape check.

| model | before | after |
|---|---|---|
| `swin` | refused: `torch.floor` | **forwards and matches** — 512 elements, max abs diff 7.15e-07 (scale 2.63) |
| `segformer` | refused: `torch.floor` | **forwards and matches** — 256 elements, max abs diff 1.19e-06 (scale 2.58) |
| `yolos` | refused: `torch._C._nn.upsample_bicubic2d` | **forwards and matches** — 1872 elements, max abs diff 1.52e-06 (scale 2.10) |
| `switch_transformers` | refused: `TensorBase.index_add_` | **forwards and matches** — 48 elements, max abs diff 5.36e-07 (scale 2.33) |
| `rwkv` | refused: `TensorBase.ndimension` | **still refuses, at a new and later wall**: `NotImplementedError: not implemented in torch._C shim: TensorBase.new_empty` |

**Four newly pass; `rwkv` moved rather than passed.** `ndimension` closed and the next name in
`rwkv`'s path is `TensorBase.new_empty` — an open gap for a following round, not a regression.

> **Closed in the following round — docs/PRIMS.md §6.** `new_empty` had a second name behind it
> (`torch.maximum`), which nothing could see until the first was closed. Both landed, and `rwkv`
> now forwards and matches upstream: 256 elements, max abs diff 7.15e-07 at scale 2.17. The row
> above is left as it was measured; the qualification that round added is that the two sides are
> given the *same* weights rather than initialised alike, because `_init_weights` calls
> `nn.init.orthogonal_` and this shim has no `torch.linalg.qr`. That is the next wall, and it is
> in construction rather than in the forward.

That each op is genuinely on its model's path was **counted, not inferred from the refusal
disappearing**: with the four names wrapped in counters, one shim run recorded
`_nn.upsample_bicubic2d` 2 calls (`yolos`), `TensorBase.index_add_` 11 calls
(`switch_transformers`), `TensorBase.ndimension` 1 call (`rwkv`).

`torch.floor` recorded **zero** calls in that run, and chasing it is the honest part of this
section: in `transformers`, `swin`'s and `segformer`'s only use of it is inside `drop_path`
(`modeling_swin.py:59`, `modeling_segformer.py:264`), which is an identity in `eval()` mode. So
the eval-mode configs above reach neither. Re-run in `train()` with `drop_path_rate=0.5`, which is
the route that actually refused:

```text
swin_droppath        upstream [-1.7854792, 2.4511952]   shim [-1.7854794, 2.4511952]
segformer_droppath   upstream [ 0.2494404, -0.0654515]  shim [ 0.2494402, -0.0654513]
```

Both reach `torch.floor` and both match. The eval-mode rows in the table above are therefore
"these models forward end to end and match", and the train-mode rows are "the wall DEMAND7 named
is the one that closed" — two different claims, and the first does not imply the second.

## 3. Gates

```text
rust/torch_c/pytests/run.sh   391 ok, exit 0        (389 -> 391: +2 = the road test
                                                     and the capture test)
DOCWATCH                      PASS -- 362/362 evaluated marker(s) hold
                              (353 before this document's own 9 markers)
tools/golden/compare.py       SUMMARY: 8681/8681 cases passed, 0 failed,
                              ops covered=207, pending case builders=0
```

`ops covered` **203 -> 207**, +4, one per kernel landed: `aten.floor.default`,
`aten.floor_.default`, `aten.index_add_.default`, `aten.upsample_bicubic2d.default`.
`ndimension` moves it by zero, correctly — it is a spelling, and docs/DEMAND6.md §2's note that
`ops covered` structurally undercounts spellings applies here in full.

Three pinned counts elsewhere in the suite moved with real additions and were updated with the
arithmetic that makes them checks rather than change detectors: `tag_core_count` 106 -> **107**
(only `floor.default` is `core` upstream — the other three were read off their own `.tags`),
distinct schema identities 287 -> **291** (+4: `floor` brings `default` *and* `.out`, `floor_` and
`index_add_` one each, and `upsample_bicubic2d` brings **none** because it is a `_nn` binding with
no table entry), and `_EXPECTED_MUTABLE` gained exactly the two mutating names.

<!-- DOCWATCH: op-implemented aten.floor.default -->
<!-- DOCWATCH: op-implemented aten.floor_.default -->
<!-- DOCWATCH: op-implemented aten.index_add_.default -->
<!-- DOCWATCH: op-implemented aten.upsample_bicubic2d.default -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json floor present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json index_add_ present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs ndimension present -->
<!-- DOCWATCH: count smoke_ok ge 391 -->
<!-- DOCWATCH: count golden_cases_passed ge 8681 -->
