# `float32` prefill vs upstream as a function of sequence length

**One-line result.** The gap is not one thing. Fitting the measured curve gives

```
gap(S)  =  0.50 · S  +  9.1e-4 · S²      ms      (SmolLM2-135M, f32 prefill)
             ^^^^^^^      ^^^^^^^^^^
             per-token    attention
```

which reproduces every measured point from `S=32` to `S=1024` to within 6%. At the
`S=128` the brief names, **81% of the gap is the linear term and 19% is the quadratic
one**; the quadratic only takes over past `S≈550`. So the growing ratio is mostly a
*constant per-token surcharge* that upstream does not pay, and attention is the
smaller half of the story at the length that matters.

**All of it has since been worked on, and every number in §1 and §2 is the
*original* one.** §3 removed the linear term entirely (`pow`, squared by
multiplying). §7 removed **35% of the quadratic one** (`amax`, a maximum without
the argmax), taking `S=512` from 2.01× upstream to **1.68×** and `S=1024` from
3.01× to **2.33×**. §8 removed **22% of what was then left** (scaling and
causal masking fused into one pass), taking `S=512` to **1.57×** and `S=1024`
to **2.10×**. The logits are bit-identical across all three changes, at every
length. The curve as it stands is

```
gap(S)  =  0.019 · S  +  4.65e-4 · S²      ms
```

**§8 is also where the floor is named, and that is its larger result.**
Upstream answers one `S=1024` attention in 3.79 ms; doing the identical
mathematics as separate tensor ops costs 14.83 ms *using upstream's own kernel
for every one of them*. So four fifths of what remains is not a slow kernel —
it is that we materialise the `[1, 9, S, S]` score matrix and walk it, and
upstream never builds it. §8.12 has what closing that would cost, and the
answer is a second SDPA path with its own numerics contract rather than a
faster version of this one.

> **docs/kernels/FLASH.md built that second path and it is slower, twice over.**
>
> "We materialise and upstream never does" is true, and it is the wrong way
> round as a *performance* claim on CPU. The `[1, 9, S, S]` product is one BLAS
> `gemm`; a blocked kernel gives that up. `flash.rs` — upstream's own blocked
> kernel, reproduced exactly — is **20x slower at T=512**, which is why it ships
> off by default. A second, independent tiled implementation written for
> FLASH.md measured **0.38–0.56x**. Both avoid the allocation; neither is fast.
>
> So the floor named above stands, but the remedy implied here does not.
> Upstream's 3.79 ms is a hand-written fused kernel with BLAS-quality inner
> loops, not the absence of a materialised matrix. FLASH.md §1.2.

---

## 0. What these numbers may not be used for

- **Host only** (M1, macOS 15, 8 cores). No Android or iOS measurement here.
- Upstream is the **`torch` 2.13.0 macOS arm64 wheel** in `/Volumes/macMini/caches/spike-venv`:
  `BLAS_INFO=accelerate`, `USE_MKLDNN=OFF`. A differently-built upstream would not
  give these numbers.
- The machine is not idle even when nothing else is scheduled — a window server, a
  user app and two Android emulators sit on it. Load was **2.4 – 5.5** throughout.
  The control below is what makes the readings usable, not the quiet.

---

## 1. The curve

`SmolLM2-135M`, `dtype=float32`, deterministic token ids (`(i*7919+13) % 49152`),
2 warmups then 5 timed passes, **minimum of 5 within a process, then minimum across
3 alternating rounds** (`up, shim, up, shim, up, shim`). Harness
`/Volumes/macMini/caches/f32len-scratch/sweep.{py,sh}`.

| S | upstream (ms) | ours (ms) | **ratio** | absolute gap (ms) |
|---:|---:|---:|---:|---:|
| 6 | 35.46 | 37.50 | **1.06×** | 2.0 |
| 32 | 37.68 | 55.51 | **1.47×** | 17.8 |
| 128 | 75.79 | 154.35 | **2.04×** | 78.6 |
| 512 | 231.25 | 728.66 | **3.15×** | 497.4 |
| 1024 | 465.02 | 1928.46 | **4.15×** | 1463.4 |

`S=1024` is 2 rounds × 3 timed passes rather than 3 × 5, for time.

### 1.1 The control, and the spread

Two extra shim processes were run back to back against each other after the three
rounds, with nothing else changed:

| S | ctl_c1 | ctl_c2 | ctl ratio |
|---:|---:|---:|---:|
| 6 | 37.75 | 37.81 | 1.002 |
| 32 | 55.57 | 55.36 | 0.996 |
| 128 | 155.12 | 154.38 | 0.995 |
| 512 | 730.38 | 728.72 | 0.998 |

**The control reads 1.00 to within 0.5%.** Round-to-round spread on the real
measurement is of the same size (shim `S=128`: 154.35 / 154.76 / 154.63, i.e. 0.3%;
upstream `S=128`: 76.38 / 75.79 / 76.14, i.e. 0.8%). The effects in the table are
6% to 315%, so every row except `S=6` is far outside the noise. **`S=6` at 1.06× is
about 12× the control spread**, so it is real but small — and it agrees with
`docs/bindings/BIND.md`, which measured 1.02–1.13× there across several rounds.

### 1.2 The shape, which is the finding

Marginal cost per additional token, taken between adjacent rows:

| interval | upstream ms/token | ours ms/token |
|---|---:|---:|
| 6 → 32 | 0.085 | 0.686 |
| 32 → 128 | 0.397 | 1.030 |
| 128 → 512 | 0.405 | 1.496 |
| 512 → 1024 | 0.456 | 2.343 |

**Upstream's marginal cost is flat at ~0.40 ms/token** from `S=32` up to `S=512`,
rising only 13% at `S=1024`. Prefill on this model is, for upstream, essentially
linear in `S` over this whole range — the `S²` attention term is a small correction
even at 1024.

**Ours rises monotonically, 0.69 → 2.34.** A marginal cost that itself grows linearly
in `S` is a quadratic total. Least-squares over `S = 32…1024` on the gap gives

| term | coefficient | what it is |
|---|---:|---|
| `a · S` | **0.497 ms/token** | per-token work we do and upstream does not |
| `b · S²` | **9.10e-4 ms/token²** | grows with attention's area |

with residuals: `S=32` predicted 16.8 vs 17.8 measured, `S=128` 78.1 vs 78.6,
`S=512` 493.1 vs 497.4, `S=1024` 1463.9 vs 1463.4. **Under 6% everywhere, and under
1% for `S ≥ 128`.**

Split at each length:

| S | linear part | quadratic part | quadratic share |
|---:|---:|---:|---:|
| 32 | 15.9 | 0.9 | 6% |
| 128 | 63.6 | 14.9 | **19%** |
| 512 | 254.6 | 238.5 | 48% |
| 1024 | 509.1 | 954.3 | 65% |

**So at `S=128` the answer is "mostly not attention".** A brief that chased only the
quadratic term would be chasing 15 ms of a 79 ms gap. The crossover is at
`S = a/b ≈ 546`.

### 1.3 Numerics baseline

Logits sha256 (over the little-endian f32 bytes of the flattened `[1,S,49152]`
tensor). **These are the values that must not move.** The `S=128` entry matches
`docs/perf/DTYPE_PERF.md` §6.1 exactly, which is the cross-check that this harness is
measuring the same thing the previous round did.

| S | ours (f32) |
|---:|---|
| 6 | `b9fc5553ee1bf6a2…` |
| 32 | `331668f36da02f21…` |
| 128 | `00159a9dbd308eda…` |
| 512 | `07c2797dabc4552e…` |
| 1024 | `eda1e173727bb7f5…` |

Upstream's differ (`S=128`: `71e46824c0c40f15…`) and are expected to — accumulation
order is not contractual across implementations. The contract here is that *ours*
does not change.

**Re-measured after docs/numerics/SCALAR.md, and all five still read as above.** That
round changed `mul.Scalar`, `floor_divide.Scalar`, `div.Scalar_mode` and both
`pow` overloads to read their scalar operand at the precision upstream reads it
at, and `pow`'s half of that reaches `float32` — so these were re-run as the
control rather than assumed. The `bfloat16` digests docs/training/TRAIN.md §6 records did
not move either, which was **not** expected: a `TorchDispatchMode` over this
prefill shows every Python number the forward passes is an integer or exactly
representable (`pow(·, 2)`, `scale=0.125`, and `1e-05` on `add.Tensor`, which is
the half of the family that still narrows), so no changed kernel can reach a
separating value in this model. docs/numerics/SCALAR.md §4 has the log, and §4.2 the
demonstration that the numerics do move where such a call exists.

---

## 2. Decomposition — the linear term is `pow`, entirely

### 2.1 What the sampler says

`sample` on a shim process spinning `S=512` prefill, 20 s at 1 ms
(`sample_shim_512.txt`). Main thread: **13715 samples**. Inclusive counts:

| symbol | samples | share of main thread |
|---|---:|---:|
| `aten::pow_tensor_scalar` | **4624** | **33.7%** |
| ↳ `aten::pow_from_pairs` | 4495 | 32.8% |
| ↳↳ **`pow` (in `libsystem_m.dylib`)** | **4245** | **31.0%** |
| `candle::cpu_backend::ReduceIndex::map` | 1577 | 11.5% |
| `libBLAS` sgemm inner kernels (all addresses) | ~1400 | 10.2% |
| `candle::cpu_backend::copy_strided_src_f` | 850 | 6.2% |
| `VVEXPF` (in `libvMisc.dylib`) | 786 | 5.7% |

The single hottest thing in a `float32` prefill is **a scalar `pow()` call out of
libm, one per tensor element.** The call chain places it six nested `nn.Module`
calls deep, which is `LlamaRMSNorm`:

```python
# transformers/models/llama/modeling_llama.py:65
variance = hidden_states.pow(2).mean(-1, keepdim=True)
```

SmolLM2-135M has 30 layers, so that runs **61 times per forward**
(`30 × 2 + 1` final norm).

### 2.2 What the code does

`rust/torch_c/src/aten.rs`, `pow_tensor_scalar` → `side_from_tensor` →
`pow_from_pairs`. For a `float32` tensor and any scalar exponent it makes
**six full passes over the data**:

1. `flat.to_dtype(F64)` — widen every element to `f64`
2. `.to_vec1::<f64>()` — copy into a `Vec<f64>`
3. `PowSide::as_f64()` → `v.clone()` — **copy it again**
4. `(0..n).map(|i| b[i % b.len()].powf(e[i % e.len()]))` — one **libm `pow`** call
   per element, plus two integer `%` per element for the broadcast cycling
5. `Tensor::from_vec(values)` — copy into tensor storage
6. `.fast_to(storage)` — narrow back to `f32`

Step 4 is 94% of it (4245 of 4495 samples). The other five are why the remaining
6% is not free either.

### 2.3 The microbenchmark, at the shape *and layout* the model produces

`hidden_states` in `LlamaRMSNorm` is `[1, S, 576]`, `float32`, **contiguous** —
printed by the harness (`powbench.py`) rather than assumed, because this
repository has twice been misled by a microbench at a layout the model never
makes (`docs/perf/DTYPE_PERF.md` §2 vs §4). All rows below read `contig=True`.

| S | shape | contig | upstream `pow(2)` | ours `pow(2)` | ours **`x*x`** | ratio pow |
|---:|---|:--:|---:|---:|---:|---:|
| 6 | `(1,6,576)` | True | 0.0013 | 0.0457 | 0.0016 | **35×** |
| 32 | `(1,32,576)` | True | 0.0024 | 0.2584 | 0.0027 | **108×** |
| 128 | `(1,128,576)` | True | 0.0118 | 1.0628 | 0.0082 | **90×** |
| 512 | `(1,512,576)` | True | 0.0314 | 4.2598 | 0.0271 | **136×** |
| 1024 | `(1,1024,576)` | True | 0.0497 | 8.5849 | 0.0519 | **173×** |

ms, min of 5 rounds after 2 warmups. **Our own `x*x` already matches upstream**
(0.0271 vs upstream's `x*x` 0.0289 at `S=512`) — so this is not a kernel-quality
problem anywhere except in `pow`'s own path. Upstream is fast because ATen
special-cases small integral exponents in `pow_tensor_scalar` and multiplies.

### 2.4 The sum, reconciled against the model-level gap

Weighted by the call count — 61 `pow(2)` per forward — against the fitted terms
of §1.2:

| S | `61 × (ours − upstream)` | fitted **linear** term | model gap | pow share of gap |
|---:|---:|---:|---:|---:|
| 6 | 2.7 | 3.0 | 2.0 | 133% |
| 32 | 15.6 | 15.9 | 17.8 | 88% |
| 128 | **64.1** | **63.6** | 78.6 | **82%** |
| 512 | **257.9** | **254.6** | 497.4 | **52%** |
| 1024 | **520.6** | **509.1** | 1463.4 | **36%** |

**The middle two columns agree to within 2–10% at every sequence length.** This is
the reconciliation the brief asked for, and it is not a near-miss: an independent
microbenchmark of one op, multiplied by a call count derived from the model's
layer count, reproduces a term that was fitted from model-level wall clock alone.

**So the linear term of the gap is `aten.pow.Tensor_Scalar`, and nothing else
material.** At `S=6` the prediction (2.7 ms) slightly exceeds the measured gap
(2.0 ms), which is the regime where `docs/bindings/BIND.md`'s Python-layer residual and
this cancel within the noise; at `S ≥ 32` it is clean.

What this does **not** explain is the quadratic term — 14.9 ms at `S=128`,
238.5 ms at `S=512`, 954 ms at `S=1024`. That is dealt with in §4.

---

## 3. The fix — square by multiplying

One function in `rust/torch_c/src/aten.rs`, `pow_square_fast_path`, taken before
`pow_tensor_scalar` falls into the generic path:

```rust
if exponent.as_f64() != 2.0 { return Ok(None); }
if !matches!(t.dtype(), DType::F32 | DType::F64) { return Ok(None); }
if PyDtype::new(tag).storage(op)? != t.dtype() { return Ok(None); }
t.mul(t).map(Some)
```

**`f16`/`bf16` are excluded on purpose.** For those, candle's reduced-precision
multiply is not obviously one correctly-rounded step, and `docs/perf/DTYPE_PERF.md`
owns the `bfloat16` checksum. Leaving them on the old path means that checksum
is *untouched* rather than merely expected to hold.

### 3.1 Why the answer cannot move — exactness, not tolerance

An `f32` significand is 24 bits, so `b × b` needs at most 48 and `f64` has 53.
**The exact square is representable in the old path's `f64` intermediate**, so a
correctly-rounded libm `pow(b, 2.0)` returns it with no rounding at all, and the
closing `fast_to(F32)` is then a *single* rounding of the exact product. IEEE-754
`f32` multiplication is defined as the correctly-rounded exact product — the same
rounding of the same value. There is no double-rounding step because the
intermediate was exact, and no range problem either: the smallest `f32` subnormal
squared is ~2e-90, comfortably normal in `f64`, and anything overflowing `f32`
gives `inf` on both paths.

### 3.2 Three Rust tests, and the proof they can fail

`cargo test --release` goes **10 → 13**. `squaring_matches_the_libm_round_trip_bit_for_bit`
transcribes the old path (`f32 → f64 → libm pow → f32`) and compares bit patterns
against `b*b` over signed zeros, both subnormal extremes, `f32::MAX` (which
overflows to `inf`), `±inf`, a 4096-point sweep needing real rounding, and a
200-point geometric sweep across the exponent range.
`squaring_matches_libm_for_f64_too` checks the weaker `f64` claim — the one that
rests on this platform's libm rather than on representability — over 4700 values.

**Tampered, it fails.** Multiplying the fast path's result by `1.0000001` and the
reference's exponent by `0.0000001`:

```
test aten::pow_square_tests::squaring_matches_the_libm_round_trip_bit_for_bit ... FAILED
  assertion `left == right` failed: input 3 (-1e0) squared: fast 1e0 vs libm round-trip NaN
test result: FAILED. 12 passed; 1 failed
```

### 3.3 Model level — old vs new vs upstream, alternated

`old, new, upstream` × 3 rounds with the artefact swapped on disk and the swap
`cmp`-verified before every run, then a **new-vs-new control**. Minimum across
the 3 rounds. The two artefacts are confirmed distinct (`cmp`) and the
`pow_square_fast_path` symbol is present in exactly one of them (`nm`: new 1,
old 0).

| S | old | **new** | upstream | old ratio | **new ratio** | saved |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 37.64 | **34.46** | 35.34 | 1.065× | **0.975×** | 3.2 |
| 32 | 55.42 | **39.19** | 37.83 | 1.465× | **1.036×** | 16.2 |
| 128 | 154.73 | **89.74** | 75.94 | 2.037× | **1.182×** | 65.0 |
| 512 | 728.52 | **464.48** | 231.04 | 3.153× | **2.011×** | 264.0 |

**The `S=128` case the brief opened with goes 1.84–2.04× → 1.18×, and `S=6` now
runs *faster* than upstream.**

Control (`new` vs `new`, two fresh processes):

| S | 6 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|
| ratio | 1.000 | 1.000 | 1.000 | 0.999 |

### 3.4 The saving is the predicted saving

§2.4 predicted the model-level effect from a microbenchmark of one op times a
call count. That prediction was made before the fix existed:

| S | predicted `61 × (old_pow − new_pow)` | **measured model-level saving** |
|---:|---:|---:|
| 128 | 64.3 | **65.0** |
| 512 | 258.2 | **264.0** |

**Within 1% and 2%.** The decomposition was not a plausible story that happened to
point at a fast op — it quantitatively predicted the outcome.

### 3.5 Numerics — unchanged, at every length

Every `old` and `new` run above printed the logits sha256. **All four pairs are
identical**, and `S=128` still equals `docs/perf/DTYPE_PERF.md` §6.1's recorded value:

| S | old | new | same |
|---:|---|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | `b9fc5553ee1bf6a2…` | ✅ |
| 32 | `331668f36da02f21…` | `331668f36da02f21…` | ✅ |
| 128 | `00159a9dbd308eda…` | `00159a9dbd308eda…` | ✅ |
| 512 | `07c2797dabc4552e…` | `07c2797dabc4552e…` | ✅ |

That is 24 forward passes' worth of agreement (3 rounds × 4 lengths × 2
artefacts), not a single spot check.

### 3.6 What the fit says happened

Refitting `gap(S) = a·S + b·S²` on the **new** gaps (`S=128`: 13.80,
`S=512`: 233.44):

| term | before | after |
|---|---:|---:|
| linear `a` | 0.497 ms/token | **−0.008 ms/token** |
| quadratic `b` | 9.10e-4 ms/token² | **9.07e-4 ms/token²** |

**The linear term is gone — to zero, not merely reduced — and the quadratic
coefficient is unchanged to within 0.3%.** That is the strongest available
confirmation that §2's decomposition was right about *which* term `pow` was:
removing it moved one coefficient to zero and left the other exactly where it was.

### 3.7 `S=1024`, and `bfloat16` for free

| dtype | S | old | new | upstream | new ratio |
|---|---:|---:|---:|---:|---:|
| `f32` | 1024 | 1928.46 | **1405.00** | 465.02 | 4.15× → **3.02×** |
| `bf16` | 128 | 180.34 | **113.79** | 368.7 (§DTYPE_PERF §3) | **3.24× *faster*** |

`S=1024` saved 523.5 ms against a predicted 520.5 — **0.6%**.

**`bfloat16` got 1.58× faster even though the fast path refuses `bf16`.** That is
not luck and not a leak: `LlamaRMSNorm` upcasts *first* —

```python
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
```

— so the `pow` is a `float32` `pow` in every model, whatever the model's dtype.
The `bf16` logits sha256 is `7ff8e9334449b147…`, alternated old/new/old/new, and
it still equals the value `docs/perf/DTYPE_PERF.md` §6.1 recorded.

---

## 4. What remains — the quadratic term is SDPA, and mostly one reduction

### 4.1 SDPA on the tensors the model actually passes

Not reconstructed: `sdpabench.py` monkeypatches
`F.scaled_dot_product_attention`, runs a real prefill, keeps the **first call's
actual arguments**, and times with exactly those. Shapes and contiguity printed:

```
q  (1, 9, S, 64)  k (1, 3, S, 64)  v (1, 3, S, 64)   float32
is_causal=True  scale=0.125  enable_gqa=True  attn_mask=None
30 sdpa calls per forward
```

(9 query heads, 3 kv heads, head_dim 64 — GQA. Upstream reports `q` as
non-contiguous and `v` contiguous; the shim reports the reverse. Same shapes,
same values, different stride bookkeeping.)

| S | upstream/call | ours/call | ratio | **× 30 = per forward** | remaining model gap | **share** |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.132 | 0.519 | 3.9× | **+11.6 ms** | 13.80 | **84%** |
| 512 | 1.427 | 7.898 | 5.5× | **+194.1 ms** | 233.44 | **83%** |

**So the quadratic term is SDPA, and the sum reconciles at both lengths.**

### 4.2 The sign flip the brief predicted

The brief noted `_softmax` had measured 0.15× (6.8× *faster* than upstream) at a
6-token prompt and asked whether that survived at 128. **It does not.** At the
score shape attention actually produces, `[1, 9, S, S]` contiguous:

| op, `[1,9,512,512]` f32 contiguous | upstream | ours | ratio |
|---|---:|---:|---:|
| `sum.dim_IntList(-1)` | 0.092 | 0.136 | 1.5× |
| `exp.default` | 0.631 | 1.083 | 1.7× |
| **`max.dim(-1)`** | 0.349 | **5.577** | **16.0×** |
| `amax.default(-1)` | 0.089 | *not implemented* | — |
| **`_softmax(-1)`** | 0.994 | **10.601** | **10.7×** |

An op that was 6.8× fast at `S=6` is 10.7× slow at `S=512`. That is precisely the
shape of a growing gap, and it is why the `S=6` reading in `docs/bindings/BIND.md` could
not have found this.

### 4.3 The specific thing, which is one reduction

`sample` on the new build at `S=1024` (main thread 13701 samples), top of stack:

| symbol | samples | share |
|---|---:|---:|
| **`candle::cpu_backend::ReduceIndex::map`** | **3323** | **24.3%** |
| `VVEXPF` (Accelerate vectorised exp) | 1404 | 10.2% |
| `libBLAS` sgemm kernels | ~2200 | 16% |
| `copy_strided_src_f` | 853 | 6.2% |
| `binary_map` Mul / Div / Sub | 1151 | 8.4% |
| `candle::cpu_backend::ReduceSum::map` | 204 | 1.5% |

**`ReduceIndex` and `ReduceSum` run over the same tensor, the same number of
times, and differ by 16×.** `ReduceIndex` is candle's *index-tracking*
reduction — the one behind `max`/`min`/`argmax` — and it computes and then
discards an argmax the softmax never wanted. `ReduceSum` has a vectorised path;
`ReduceIndex` is a scalar loop.

Upstream's own numbers say the same thing from the other side: its `amax`
(0.089) is **3.9× faster than its own `max.dim`** (0.349) for exactly this
reason — no indices. **The shim has no `amax` kernel at all**, so anything
needing a maximum *value* has only the index-producing path available.

There are also no `flash.rs` symbols anywhere in the profile. `flash.rs` is the
*reference* implementation (`reference_enabled()`), not the default; the default
SDPA materialises the `[1, 9, S, S]` score matrix and drives it through candle
ops — which is what every symbol in that table is.

---

## 5. What is left, and what it would take

**Fixed here:** the entire linear term, 0.497 ms/token, bit-identically.

**Not fixed:** the quadratic term, `≈9.05e-4 · S²` ms — 13.8 ms at `S=128`,
233 ms at `S=512`, 940 ms at `S=1024`. §4 attributes 83–84% of it to SDPA and
localises the largest single item inside SDPA to the softmax's max-reduction
running on candle's index-tracking `ReduceIndex`.

Sizing the next step from the measured numbers: if our max-reduction reached
upstream's `amax` (5.577 → 0.089 ms at `[1,9,512,512]`), `_softmax` would fall
from 10.601 to roughly 5.1 ms — **about half the softmax gap**, with the rest in
`exp` (1.7×) and the strided copies. Because a maximum involves no rounding, that
change is numerically free in a way the `pow` fix had to argue for.

Two reasons it is not done here:

1. **It is a candle-level concern.** The fast reduction has to exist before
   anything can call it, and `candle_core`'s `ReduceIndex` is a dependency, not
   this crate. The in-crate alternative is an `amax` kernel plus routing
   softmax/SDPA to it — a real change to the attention path rather than a
   guarded fast path in a leaf function.
2. **The remaining `exp` and copy costs are not obviously separable** from how
   SDPA materialises its scores, and a rewrite that stops materialising them is
   a different piece of work with its own numerics argument (softmax
   accumulation order is exactly what `docs/models/GENERATE.md` §6 warns about).

**The honest summary is that the growing gap had two terms, one of them is now
gone, and the other is named, measured, reconciled to 83–84%, and localised to a
single reduction — but closing it is a change to the attention path, not to a
leaf.**

> **§7 checked the two reasons above and only the second one held.**
>
> Reason 1 was wrong on both counts. `candle_core` does export a mechanism for
> writing a reduction kernel outside the crate (`CustomOp1`, the same one
> docs/kernels/VIEWS.md §6.2 used), so nothing had to exist in candle first; and routing
> SDPA to it was **one line**, not a change to the attention path. §7.1.
>
> Reason 2 stands unchanged: `exp` and the strided copies are still there, they
> are now the largest remaining items, and separating them from how SDPA
> materialises its scores is still a different piece of work with its own
> numerics argument.
>
> Sizing: §5 predicted `_softmax` 10.601 → 5.1, "about half the softmax gap".
> `_softmax` did not move at all and was never on the model's path — §7.4 has
> why, and it is a correction to §4.2's reading rather than to its measurements.
> The model-level effect was **35% of the quadratic term**, `S=512` 2.01× →
> 1.68× and `S=1024` 3.01× → 2.33×.

---

## 6. Reproducing this

```sh
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-f32len
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh
```

Harnesses live in `/Volumes/macMini/caches/f32len-scratch/`:

| file | what it does |
|---|---|
| `sweep.py` / `sweep.sh` | model-level wall time + logits sha256 across `S`, alternating |
| `ab.sh` | old/new/upstream × 3 rounds with a verified artefact swap, plus control |
| `spin.py` | prefill in a loop so `sample` can attach |
| `powbench.py` | `pow(2)` at the real RMSNorm shape, printing contiguity |
| `sdpabench.py` | SDPA on tensors captured from a real forward |
| `redbench.py` | the reductions, at the real score shape |

Gates as they stood at the end of §3 (§7.11 has the current ones):

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh      242          (unchanged)
$PY tools/golden/compare.py                    3302/3302 ops=133, pending 2
$PY tools/golden/compare.py --self-test        PASS 13 comparators x 11 fault modes
$PY rust/torch_c/pytests/verify_schemas.py     4331/4331
( cd rust/torch_c && cargo test --release )    13           (was 10, +3)
```

§7's harnesses are in `/Volumes/macMini/caches/amax-scratch/`, with
`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-amax`:

| file | what it does |
|---|---|
| `ab.sh` | model level: old/new/upstream × 3 rounds, verified artefact swap, plus control |
| `absdpa.sh` | the same alternation driving `sdpabench.py` |
| `kbench/` | a standalone crate timing the five reduction formulations of §7.3 |
| `opcount.py` | wraps `_aten_dispatch` to print a forward's work queue (see its own note: bootstrap binds the dispatcher at import, so it must be installed before `import torch`) |
| `sweep.py`, `redbench.py`, `sdpabench.py`, `spin.py` | copies of §6's, pointed at this worktree |

§8's are in `/Volumes/macMini/caches/quad-scratch/`, with
`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-quad`:

| file | what it does |
|---|---|
| `ab.sh` | model level: old/new/upstream × 3 rounds, verified artefact swap, plus control |
| `opq_up.py` | the whole work queue with **shapes and strides**, via a `TorchDispatchMode` on upstream — §8.1 |
| `stage.py` | replays SDPA stage by stage from Python, and checks the stage sum against the whole op |
| `prof.sh` | spins a prefill and attaches `sample` to it |
| `parse.py`, `attrib.py`, `attrib2.py` | turn a `sample` call graph into inclusive counts, and attribute a symbol to its nearest `aten` caller — §8.2.1 |
| `one.py` | one prefill against a throwaway artefact with `Instant::now()` around each SDPA stage — §8.2 |
| `sweep.py`, `sdpabench.py`, `spin.py` | copies of §6's, pointed at this worktree |

**`opcount.py` from §7 does not work and `opq_up.py` replaces it.** Wrapping
`torch._C._aten_dispatch` after `import torch` counts nothing: `bootstrap.install`
binds the dispatcher inside the extension's `PyInit`, so the wrapper is never
the one that is called. It reported `0 dispatched calls`.

---

## 7. The quadratic term, part one — `amax`

§5 left three routes open for the max-reduction and asked which one holds. **The
second one holds, and the first one is closed for a reason worth recording.**

### 7.1 Which route, and why the other two are not it

> **1. A public candle API that returns a maximum without the index.**
> **There is none.** `Tensor::max`, `Tensor::max_keepdim`, `Tensor::min`,
> `Tensor::min_keepdim` and `Tensor::max_all` are all `pub`, and all five are
> one line each:
>
> ```rust
> pub fn max_keepdim<D: Dim>(&self, dim: D) -> Result<Self> {
>     self.reduce_impl(dim, true, ReduceOp::Max)
> }
> ```
>
> `reduce_impl` sends `ReduceOp::Max` to `cpu_backend::ReduceIndex`, the same
> struct `ReduceOp::ArgMax` uses. So every public spelling of "maximum" in
> candle 0.11.0 is the index-tracking reduction. There is no second one to
> reach for.

> **3. Forking or vendoring candle.** Not needed, so not done.

> **2. An in-crate kernel through the same mechanism `write_into` used.**
> **Taken.** `CustomOp1` and `Tensor::apply_op1_no_bwd` are both `pub` and both
> re-exported from `candle_core`'s root. `cpu_fwd` receives `&CpuStorage` and
> `&Layout` and returns `(CpuStorage, Shape)` -- storage and layout, without
> `unsafe`, exactly as docs/kernels/VIEWS.md §6.2 described for the in-place case. The
> kernel is `tensor.rs::amax_keepdim` and it is 60 lines.

**And the change is a guarded leaf, not a rewrite of the attention path.** §5
predicted "an `amax` kernel plus routing softmax/SDPA to it -- a real change to
the attention path". The routing turned out to be **one line**:

```rust
-  let row_max = scores.max_keepdim(3)?;
+  let row_max = crate::tensor::amax_keepdim(&scores, 3)?;
```

Nothing else in `sdpa_flash_cpu` moved. The score matrix is still materialised,
`exp`/`sum`/`div` are untouched, and the reference kernel in `flash.rs` was not
opened. So the concern that closing this meant touching arithmetic that
docs/models/GENERATE.md §6 warns about does not apply to *this* half of it -- it still
applies to the `exp` and copy half, which §8 leaves open.

### 7.2 Why the answer cannot move

A maximum has no arithmetic in it, so there is nothing to reassociate and no
rounding to double. What is left is three edge cases, and each is checked rather
than asserted.

**NaN — this kernel deliberately does not match candle.** candle's predicate is
`|x, y| x < y`, and every comparison against a NaN is false, so a NaN that is
not the *first* element is skipped: `max([3, nan, 1])` is `3.0` there where
upstream answers `nan`. That is not a hypothetical — it is the fault
docs/models/E2E_REAL.md found in `aten.max.default` (which now pays for a separate
`x != x` pass to route around it) and docs/bindings/SPELLINGS.md found again in
`max.other`'s second operand. **Two ops, one predicate; the third op to use it
would have had it too.** `amax` propagates, which is upstream's rule and IEEE
`maximum`, and `test_amax_propagates_nan_where_candles_own_reduction_drops_it`
walks a NaN through five positions of a 200-element row to say so.

That divergence-from-candle cannot reach SDPA's output, and the reason is not
"scores are never NaN":

| a score row contains | old `max_keepdim` | new `amax` | the row's softmax output |
|---|---|---|---|
| no NaN | `m` | `m` | identical by construction |
| a NaN | some finite `m` | `nan` | **all NaN either way** — old: `exp` of the NaN element is NaN, so `row_sum` is NaN, so every `weights/row_sum` is NaN. New: `x - nan` is NaN for every `x`. |

**`-inf` — a fully masked row.** Both reductions answer `-inf`; the subtraction
then gives `-inf - (-inf) = nan`, which is `_softmax.default`'s documented
answer and matches upstream (`_safe_softmax` is the op that answers `0` there,
and it is a different op with its own kernel). Nothing changed here, and
`test_a_fully_masked_attention_row_reduces_to_negative_infinity` pins it at five
row widths including 512.

**Signed zero — the one place the lanes can disagree with a sequential fold.**
Sixteen accumulators mean the row is not scanned strictly left to right, and the
rule (candle's, and upstream's — measured, `amax([-0., 0.])` is `-0.` and
`amax([0., -0.])` is `0.`) keeps the *first* of two equal elements. Which of
several equal maxima is "first" can change. `-0.0` and `+0.0` compare equal, so
that is the only distinction it can make. It cannot reach the output:

```
x - (+0.0) and x - (-0.0) are the same bits for every x except x = -0.0,
where they are -0.0 and +0.0 -- and exp(-0.0) == exp(+0.0) == 1.0 exactly.
logsumexp adds row_max to log(row_sum), and log is never -0.0 (log(1) is +0),
so +0.0 and -0.0 added there give the same bits too.
```

So the SDPA claim is **bit-identity by argument, not by tolerance**, and §7.5
checks it against the artefact anyway.

### 7.3 What actually made it fast, which is not the argmax

§4.3 named `ReduceIndex` and it was right about *where*. It was wrong about
*why*, and the difference decides what the fix has to be.

Dropping the index removes one store per improvement. What removes the rest is
that candle's fold is

```rust
for (src_i, &s) in src.iter().enumerate() {
    if f(val, s) { acc = src_i; val = s }
}
```

— **one** accumulator, so every element waits on the compare-and-select of the
one before it. `ReduceSum` next to it in the same file gets a vectorised path
and that is the whole of the 16x in §4.2's table.

Five formulations, measured standalone at the real score shape
(`9·512` rows of 512 `f32`, min of 7–9 rounds, spread under 1.5%,
`/Volumes/macMini/caches/amax-scratch/kbench`):

| formulation | ms | note |
|---|---:|---|
| candle's shape — 1 accumulator, compare-and-select, index kept | **5.19** | reproduces the 5.59–5.69 ms measured *through* the shim |
| IEEE `maximum` direct — `if v > acc \|\| v.is_nan()`, 16 lanes | 2.27 | correct, and **8x slower than the next row** |
| max + NaN flag in a `bool` lane array, 16 lanes | 0.83 | correct; `bool` is 1 byte against the value's 4, so the two accumulators have different vector widths |
| **max + NaN flag in a `u32` lane array, 16 lanes** | **0.28** | **taken** |
| max alone, no NaN handling, 16 lanes | 0.19 | wrong answer; the floor this could reach |

So the NaN rule costs **0.09 ms** when carried in its own same-width
accumulator and **2.08 ms** when written the obvious way, because the compound
condition stops LLVM recognising the loop as a max reduction at all. Lane count
was tuned the same way: 8 / 16 / 32 gave 0.26 / 0.28 / 0.31 ms on the correct
formulation and 0.26 / 0.19 / 0.22 on the plain one; 16 is the pick.

**One dead end worth recording, because it looks like the answer.** The
one-comparison spelling `!(v <= acc)` propagates a NaN *in* and then loses it
again: once `acc` is a NaN, every subsequent `v <= NaN` is false, so the next
ordinary value replaces it. It is fast and wrong.

Through the shim, at the same shape, `[1, 9, 512, 512]` `float32` contiguous:

| op | upstream | before | after |
|---|---:|---:|---:|
| `amax.default(-1)` | 0.099 | *not implemented* | **0.286** |
| `max.dim(-1)` | 0.351 | 5.687 | 5.593 (untouched — it still owes an index) |

0.286 through the dispatcher against 0.28 standalone, so the Python-side
overhead of the op key is not material at this size.

### 7.4 §4.2's table was measuring a proxy, and this corrects it

§5 sized this change as "`_softmax` falls from 10.601 to roughly 5.1 ms --
about half the softmax gap". **Both halves of that sentence are wrong, and the
result is bigger than it, not smaller.** Three corrections, each measured:

**1. `aten._softmax.default` does not use candle's max.** Its kernel is
`aten.rs::softmax_default`, which reads the whole tensor out through
`read_flat` into a `Vec<f64>`, runs `softmax_body` -- a scalar loop, in `f64` or
`f32` accumulation -- and writes back. There is no candle reduction anywhere in
it. Its 10.6 ms is the `f64` round trip and that scalar loop, and it is
**unchanged by this work**: 10.512 before, 10.611 after, inside a 0.5% spread.

**2. `_softmax.default` is not on the model's path at all.** SmolLM2's attention
goes through `F.scaled_dot_product_attention`, which lowers to
`aten._scaled_dot_product_flash_attention_for_cpu.default` -- 30 calls per
forward, printed by `sdpabench.py` -- and that op writes its softmax out inline
from candle ops (`aten.rs:2767`). `softmax_body` appears **0 times** in all
three `sample` profiles taken across these two rounds, while `sdpa_flash_cpu`
appears in every one. So §4.2's `_softmax` row was a *proxy* for the softmax the
model runs, not the softmax the model runs.

**3. `max.dim` is two reductions, so it double-counted.** `aten.rs::max_dim`
calls `max(dim)` **and** `argmax(dim)` -- two independent `ReduceIndex` passes
over the same data. Its 5.687 ms is therefore about twice what SDPA's single
`max_keepdim` cost. That is confirmed from the other side: SDPA at `S=512` fell
by 2.52 ms per call and the replacement costs 0.284, so the reduction it
replaced cost **2.80 ms** -- one half of 5.6, as expected.

What was 24.3% of the profile was `ReduceIndex` reached through
`sdpa_flash_cpu`, and that is what this replaced. **The line §5 pointed at was
the right line; the op it named to size the effect was standing in for it.**

### 7.5 SDPA, on the tensors the model actually passes

`sdpabench.py` again -- monkeypatch `F.scaled_dot_product_attention`, run a real
prefill, keep the **first call's actual arguments**, time with exactly those.
`old, new, upstream` × 3 rounds with the artefact swapped on disk and
`cmp`-verified before every run, then a new-vs-new control. Minimum of 3 rounds;
the round-to-round spread on every cell below is under 1%.

```
q (1, 9, S, 64)  k (1, 3, S, 64)  v (1, 3, S, 64)  float32
is_causal=True  scale=0.125  enable_gqa=True  attn_mask=None    30 calls/forward
```

| S | upstream | old | **new** | old ratio | **new ratio** |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.122 | 0.519 | **0.399** | 4.3× | **3.3×** |
| 512 | 1.430 | 7.921 | **5.399** | 5.5× | **3.8×** |

ms per call. Control (new vs new): `S=128` 0.399/0.401 = **0.995**, `S=512`
5.408/5.430 = **0.996**.

Per forward (× 30), against the model-level gap of §7.6:

| S | upstream | old | new | SDPA's share of the remaining gap |
|---:|---:|---:|---:|---:|
| 128 | 3.67 | 15.56 | **11.96** | 8.29 of 10.51 = **79%** |
| 512 | 42.90 | 237.62 | **161.96** | 119.1 of 158.0 = **75%** |

And the two other rows of §4.2's table, for completeness -- neither was touched:

| op, `[1,9,512,512]` f32 | upstream | old | new |
|---|---:|---:|---:|
| `_softmax(-1)` | 0.999 | 10.512 | 10.611 |
| `max.dim(-1)` | 0.351 | 5.687 | 5.593 |
| `amax.default(-1)` | 0.083 | — | **0.284** |

### 7.6 Model level — old, new, upstream, alternated

`SmolLM2-135M`, `float32`, deterministic ids, 2 warmups then 5 timed passes
(3 at `S=1024`), **minimum within a process, then minimum across 3 alternating
rounds** of `old, new, upstream`. Artefact swapped and `cmp`-verified per run;
`nm` says `amax_keepdim` is present in exactly one of the two (new 1, old 0).

| S | upstream | old | **new** | old ratio | **new ratio** | saved |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 35.65 | 34.52 | 34.50 | 0.968× | **0.968×** | 0.0 |
| 32 | 37.82 | 39.12 | 39.67 | 1.034× | **1.049×** | −0.6 |
| 128 | 75.81 | 89.49 | **86.32** | 1.180× | **1.139×** | 3.2 |
| 512 | 231.00 | 463.73 | **389.01** | 2.007× | **1.684×** | 74.7 |
| 1024 | 467.55 | 1407.40 | **1087.83** | 3.010× | **2.327×** | 319.6 |

**The `old` column reproduces the previous round's landing to three significant
figures** (§3.3 recorded 1.182× at 128 and 2.011× at 512; §3.7 recorded 3.02× at
1024), which is what says this harness is measuring the same thing.

Control, two fresh `new` processes back to back:

| S | 6 | 32 | 128 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|
| ratio | 1.005 | 0.998 | 0.990 | **1.002** | **1.001** |

**`S=128` is the one row where the effect is close to the noise and it is said
so here rather than rounded up.** Round-to-round spread on `new` at 128 is 2.1%
(86.98 / 88.13 / 86.32) and the effect is 3.5%. What makes it a result anyway is
that the three `old` minima (91.11 / 90.24 / 89.49) and the three `new` minima
are **disjoint ranges**, and that the per-call SDPA measurement at the same
length -- where the spread is 0.2% and the effect 23% -- predicts
`30 × (0.519 − 0.399) = 3.60 ms` against 3.17 measured. At `S=512` the same
prediction is `30 × 2.522 = 75.7` against **74.7 measured, within 1.3%**.

### 7.7 The numbers that must not move, and did not

Every `old` and `new` run above printed the logits sha256 over the little-endian
`f32` bytes of the flattened `[1, S, 49152]` tensor. **All five pairs are
identical, and all five still equal the values §1.3 recorded**, which were taken
before either change:

| S | old | new | §1.3 |
|---:|---|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | `b9fc5553ee1bf6a2…` | ✅ |
| 32 | `331668f36da02f21…` | `331668f36da02f21…` | ✅ |
| 128 | `00159a9dbd308eda…` | `00159a9dbd308eda…` | ✅ |
| 512 | `07c2797dabc4552e…` | `07c2797dabc4552e…` | ✅ |
| 1024 | `eda1e173727bb7f5…` | `eda1e173727bb7f5…` | ✅ |

That is 30 forward passes' worth of agreement (3 rounds × 5 lengths × 2
artefacts), not a spot check. Upstream's own `S=128` digest came back
`71e46824c0c40f15…`, the value §1.3 recorded for it, so the *reference* side of
the comparison is the same one too.

**`bfloat16` as well**, because SDPA is shared code -- reduced precision widens
to `f32` for the body, so it runs this very reduction. `S=128`, same alternated
harness: `7ff8e9334449b147…` on both artefacts, still the value docs/perf/DTYPE_PERF.md
§6.1 recorded, and 114.66 → **110.66 ms** (3.5% faster, upstream 384.96).

### 7.8 The profile, after

`sample` on the new build at `S=1024`, 20 s at 1 ms, main thread **14250
samples**. Inclusive counts, aggregated over the call tree:

| symbol | before (§4.3) | **after** |
|---|---:|---:|
| `candle::cpu_backend::ReduceIndex::map` | 3323 (**24.3%**) | **0 — absent from the profile entirely** |
| `tensor::amax` (this kernel, via `apply_op1`) | — | **411 (2.9%)** |
| `libBLAS` (matmul, inclusive) | ~2200 (16%) | 5909 (41.5%) |
| `unary_map` / `VVEXPF` (the `exp`) | 1404 (10.2%) | 2759 (19.4%) / 1969 (13.8%) |
| `binary_map` (Mul/Div/Sub) | 1151 (8.4%) | 2574 (18.1%) |
| `copy_strided_src_f` | 853 (6.2%) | 1332 (9.3%) |
| `candle::cpu_backend::ReduceSum::map` | 204 (1.5%) | 289 (2.0%) |

The percentages rise because the denominator fell (1405 → 1090 ms) and because
this aggregation is inclusive over the tree where §4.3's `libBLAS` row was a sum
of leaf addresses; the two `libBLAS` figures are **not** comparable and are shown
only so the shape of the profile is. The row that is comparable is the first
one, and it reads zero.

It also reconciles: 2.9% of 1148 ms (the sampled run's own pace) is 33 ms, and
30 SDPA calls at `[1, 9, 1024, 1024]` -- four times the area of the 0.284 ms
measurement at 512 -- predicts 34 ms. The old side reconciles the same way:
24.3% of 1405 is 341 ms against 30 × 11.2 = 336 ms predicted from the 2.80 ms
per-call figure of §7.4.

### 7.9 What the fit says happened

Refitting `gap(S) = a·S + b·S²` on the two-point pair (`S=128`, `S=512`) and
then checking the fit at `S=1024`, which it was not given:

| term | §3.6 (before) | **after** |
|---|---:|---:|
| linear `a` | −0.008 ms/token | **+0.007 ms/token** (still zero) |
| quadratic `b` | 9.07e-4 ms/token² | **5.90e-4 ms/token²** |

The refit of the `old` column here gives `a = −0.009`, `b = 9.05e-4` --
§3.6's numbers to two significant figures, from an independent set of runs.
Held out, `S=1024`: the new fit predicts a 625 ms gap and 620 was measured,
**0.8%**.

**35% of the quadratic term is gone.** What is left of it:

| S | quadratic before | **quadratic after** |
|---:|---:|---:|
| 128 | 14.8 ms | **9.7 ms** |
| 512 | 237.3 ms | **154.6 ms** |
| 1024 | 949.4 ms | **618.6 ms** |

### 7.10 What is reachable, and what is not

The kernel is reachable as `torch.ops.aten.amax.default`, which is the spelling
the golden harness compares. **`torch.amax` and `Tensor.amax` are not**: routing
a Python name to an overload needs an entry in `src/overloads.json` (free
function) or `src/methods.json` (member), and this round did not own those
files. Both currently refuse by name with `NotImplementedError`, which
`test_amax_has_no_python_spelling_yet_and_says_so_by_name` pins -- **that test
fails the day the entry lands**, which is the notification wanted, since golden
dispatches by key and is structurally blind to a missing name.

The entries needed, when someone owns those files:

```
overloads.json   "amax": [["Tensor self", "int[1] dim=[]", "bool keepdim=False"] -> amax.default]
methods.json     "amax": same signature, receiver as self
```

Neither is needed for the SDPA path, which calls the kernel directly in Rust.

> **TAKEN, in docs/kernels/TRIL.md §2.** Both entries landed in `src/overloads.json` and
> `src/methods.json`; `torch.amax` and `Tensor.amax` now resolve and reach this kernel
> (reverified directly: `torch.amax(t, dim=1)` and `t.amax(dim=1)` both return the kernel's
> answer). `test_amax_has_no_python_spelling_yet_and_says_so_by_name` failed on the first run
> after the entries went in, exactly as designed, and was replaced by
> `test_amax_now_has_both_python_spellings_and_they_reach_the_kernel`.
>
> Standing check (docs/verification/DOCWATCH.md):
> <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json amax present -->
> <!-- DOCWATCH: json-key rust/torch_c/src/methods.json amax present -->
> <!-- DOCWATCH: op-implemented aten.amax.default -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_amax_now_has_both_python_spellings_and_they_reach_the_kernel present -->

### 7.11 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 242 | **246** (+4: row width at 512, NaN from five positions, all-`-inf`, the missing spelling) |
| `tools/golden/compare.py` | 3302/3302, ops=133 | **3422/3422, ops=134** (+120 cases, pending 2 unchanged) |
| `compare.py --self-test` | PASS | PASS, 13 comparators × 11 fault modes |
| `verify_schemas.py` | 4331/4331 | **4334/4334** (+3: `amax`'s schema text, its `OpOverload.tags`, its packet) |
| `cargo test --release` | 13 | **18** (+5) |

`test_core_ops_and_op_tags_agree`'s `tag_core_count` went 82 → 83, because
upstream tags `amax` `['core', 'pt2_compliant_tag', 'reduction']`. Read off the
op, not inferred -- `max.dim` sitting next to it is *not* core.

### 7.12 Sabotage — these tests fail when the kernel is wrong

Two faults, injected into the built artefact and run through the real gates.

**Fault 1 — the NaN accumulator never sets** (candle's own behaviour,
reintroduced):

```
FAIL aten.amax.default :: amax(dtype=float32, shape=(4,)) [NaN in the middle
     position propagates ...] -- value mismatch (index 0: torch=nan c=3.0)
SUMMARY: 3412/3422 cases passed, 10 failed
```

10 of the 12 NaN cases. The two that survive are the `at=0` ones, and that is
the documented reason rather than a hole: a NaN in element 0 seeds every
accumulator lane and `greater` never displaces it, so even a kernel with no NaN
handling at all gets those right. **That is exactly why the cases walk the NaN
through the middle and the end as well.**

**Fault 2 — the lane-combining loop skips one accumulator** (`for lane in
2..AMAX_LANES` instead of `0..`): caught by `cargo test` and by 10 golden cases.

**Fault 2 is also the reason a test in this change was rewritten.** The first
version of `amax_matches_a_sequential_fold_at_every_length` compared against a
sequential fold over three pseudo-random rows at every length up to 67 -- and
**it passed under fault 2**, because no seed happened to put a maximum in the
dropped lane. It now walks a distinct maximum through *every* position of every
length, and through `amax_strided` at three strides as well; re-injected, it
fails. A test that cannot fail is not a test, and this one could not.

---

## 8. The quadratic term, part two — the floor is the score matrix

§7 left `exp`, `binary_map`, matmul and "the strided copies inside how SDPA
materialises its scores" as what remained. **One of those four is not what it
was called, and the shape of the answer is different from what §5 and §7 were
looking for.**

The short version, and it is the finding rather than the change:

> **Upstream's whole fused SDPA is 3.9x faster than the sum of the very same
> stages run as separate tensor ops on upstream itself.** At `S=1024` upstream
> answers one attention in **3.79 ms**; doing the identical mathematics as
> `matmul, affine, add, amax, sub, exp, sum, div, matmul` — every one of them
> *upstream's own kernel* — costs **14.83 ms**. So even a shim whose every
> kernel matched upstream's kernel exactly would still be **3.9x slow here**.
> The remaining gap is not a slow kernel. It is that we materialise
> `[1, 9, S, S]` and walk it seven times and upstream never materialises it at
> all.

Against that floor, this round takes the part of the gap that is *not* the
floor: three of those seven passes were spent scaling and masking, and two of
them plus an `S x S` allocation are now gone.

### 8.1 The queue, so that nothing quadratic is being missed

Before decomposing SDPA, the whole work queue of one real `S=1024` `float32`
prefill, with shapes **and strides** (`opq_up.py`, a `TorchDispatchMode` — the
shim's own dispatcher cannot be wrapped from Python because `bootstrap.install`
binds it inside the extension's `PyInit`, which is the note §6 records):

```
2222 calls, 25 distinct ops, 70 distinct signatures
 333 view      274 mul.Tensor   212 _unsafe_view   211 t   211 mm   182 add.Tensor
 121 transpose 121 cat          120 slice          64 unsqueeze  61 pow/mean/rsqrt
  30 _scaled_dot_product_flash_attention_for_cpu.default        30 silu
```

**Every op in it is linear in `S` except those 30.** The largest non-attention
tensor any of them touches is `[1, 1024, 1536]`. So the quadratic term is
entirely inside `_scaled_dot_product_flash_attention_for_cpu`, and a
decomposition of that op is a decomposition of the whole quadratic term. There
is no second place to look.

### 8.2 Inside the op, measured in Rust and not in a proxy

The first attempt at this replayed each stage from Python as a separate `aten`
call. **That is a proxy and it was wrong about the largest item**: the causal
mask read 0.661 ms at `S=512` when replayed as `arange`/`where`/`full`, and
what `sdpa_flash_cpu` actually ran was a scalar `Vec<f64>` push loop and a
narrowing pass. The numbers below are from a throwaway artefact with
`Instant::now()` around each stage *in `aten.rs`*, one real prefill, the
warm pass discarded — so every row is the code that shipped.

`float32`, mean over the 30 calls of one forward, ms per call:

| stage | `S=512` | `S=1024` | upstream, same op standalone `S=1024` |
|---|---:|---:|---:|
| widen `q` + `repeat_kv` k,v | 0.064 | 0.145 | 0.177 |
| **`k.transpose(2,3).contiguous()`** | 0.645 | **1.288** | 0.134 (**9.6x**) |
| `matmul q@kT` | 0.614 | 2.627 | 2.046 |
| **`affine(scale, 0)`** | 0.415 | **1.450** | 1.328 |
| **causal mask build** | 0.518 | **2.100** | 0.385 (**5.5x**) |
| **`broadcast_add(mask)`** | 0.529 | **2.133** | 1.471 |
| `amax(-1)` | 0.318 | 1.067 | 0.711 |
| `broadcast_sub(row_max)` | 0.254 | 1.292 | 1.333 |
| `exp` | 1.357 | 4.258 | 2.417 |
| `sum_keepdim(-1)` | 0.212 | 0.662 | 0.693 |
| `broadcast_div(row_sum)` | 0.282 | 1.271 | 1.326 |
| `matmul p@v` | 1.081 | 2.914 | 2.805 |
| **sum** | **6.29** | **21.21** | **14.83** |
| **the whole op, unfused vs upstream's fused kernel** | | | **3.79** |

(The instrumented sum overstates the real call by ~16% — `S=512` reads 6.29
where the uninstrumented op measures 5.40 — because the timer breaks the
pipelining between stages. The *relative* sizes are what it is used for.)

### 8.2.1 Which constituent was not what §7 called it

§7 named "the strided copies inside how SDPA materialises its scores".
**They are not inside SDPA.** `copy_strided_src_f` is 8.9% of an `S=1024`
profile, and attributing every one of its samples to the nearest `aten`
function gives

| caller | samples | share |
|---|---:|---:|
| `cat.default` (rotary `rotate_half`, and the KV concat) | 666 | 52.3% |
| SDPA's `contiguous` calls | ~570 | 45% |
| everything else | 37 | 3% |

and the `cat` half is **linear in `S`** — it copies `[1, 9, S, 32]`, not
`[1, 9, S, S]`. So half of the item §7 listed under the quadratic term belongs
to the linear one, where it is 4.7% of a profile and not worth a kernel.

**A warning about that attribution.** The linker folds identical closures: the
symbol the profile prints for SDPA's `contiguous` is
`aten::masked_fill::{{closure}}`, because `masked_fill`'s `|t| t.contiguous()`
compiles to the same instructions. Reading that as "the model calls
`masked_fill` 30 times" would have been wrong, and §8.1's queue is what says so
— there is no `masked_fill` in it at all.

### 8.3 The change — scale and mask in one pass

`rust/torch_c/src/tensor.rs::scale_and_causal_mask`, a `CustomOp1` reached from
one place. Three stages of §8.2 become one:

```rust
-  .and_then(|s| s.affine(scale, 0.0))       // one full pass
-  if is_causal { build an S x S Vec<f64>;   // scalar push loop + narrowing
-                 scores.broadcast_add(&mask) }   // another full pass
+  if is_causal { crate::tensor::scale_and_causal_mask(&raw, scale)? }
+  else         { raw.affine(scale, 0.0)? }
```

The kernel is one pass and allocates nothing but its output:

```rust
let keep = (r + 1).min(cols);
out.extend(row[..keep].iter().map(|&v| v.mul_add_zero(mul)));
out.extend(row[keep..].iter().map(|&v| v.mul_add_zero(mul).add(T::NEG_INFINITY)));
```

**Why the answer cannot move, and it is three separate points, not one.**
Element-wise work has no summation order, so there is nothing to reassociate —
but there are three places where the *obvious* kernel is a different function
from the one it replaces, and each is spelled out rather than simplified:

1. **`v * mul + 0.0`, not `v * mul`.** candle's `Affine` is
   `unary_map(vs, l, |v| v * mul + add)` with `add = 0.0`, and `+ 0.0` is not
   the identity: it turns a `-0.0` product into `+0.0`.
2. **`+ f32::NEG_INFINITY`, not `= f32::NEG_INFINITY`.** The old path went
   through `broadcast_add`. For a finite product the two agree; for a `+inf`
   product `+inf + -inf` is a **NaN** where an assignment writes `-inf`.
3. **`T::from_f64(scale)` before the multiply, not after.** candle narrows the
   `f64` scale to the tensor's dtype and then multiplies. Multiplying in `f64`
   against the un-narrowed scale rounds the scale twice and is a different
   number — at `scale = 0.1` it differs on `1.0 + f32::EPSILON`.

Two operations and not `mul_add`: Rust does not contract this to an FMA and
neither does candle, and an FMA rounds once where these round twice.

### 8.4 Sabotage — six faults, and one that is not a fault

`cargo test --release` goes **18 -> 24**. Each fault below was injected into
the kernel and the suite re-run; `scale_causal_tests` has 6 tests.

| fault injected | tests that fail |
|---|---:|
| 1. `= -inf` above the diagonal instead of `+ -inf` | 3 of 6 |
| 2. drop the `+ 0.0` from the affine | 3 of 6 |
| 3. fuse into `mul_add` (rounds once, not twice) | 1 of 6 |
| 4. off-by-one on the diagonal (`c < r` rather than `c <= r`) | 5 of 6 |
| 5. multiply in `f64` and narrow the **product** after | **0 — and correctly so** |
| 6. process only the first matrix of the batch | 4 of 6 |
| 7. narrow the **scale** after multiplying, not before | 1 of 6 |

**Fault 5 does not fail because it is not a fault.** An `f32` significand is
24 bits, so the product of two of them needs 48 and `f64` has 53: the `f64`
product is *exact*, and narrowing it is the same single rounding the `f32`
multiply performs. Recording it here rather than deleting it, because a test
suite that failed on fault 5 would be asserting something untrue.

**The cases that make 1, 2, 3 and 7 fail are specific and were chosen for it.**
A sweep of ordinary finite values agrees with the reference under every one of
those four: only `-0.0` separates fault 2, only `+inf` separates fault 1, only
a scale needing real rounding (`0.1`) against a value needing real rounding
(`1.0 + eps`) separates 3 and 7. The fixture list in `awkward()` is annotated
with which mistake each entry is there to catch — this is the same lesson §7.12
recorded, where a NaN test stayed green because every case put the NaN at
index 0.

### 8.5 A measured negative result — the transposed GEMM

`k.transpose(2, 3).contiguous()` is 1.288 ms per call at `S=1024`, **9.6x
upstream's**, for a 2.4 MB copy. candle does not need it: its Accelerate
`matmul` accepts `rhs_m1 == k && rhs_m2 == 1` and issues `transa='T'`, so
dropping the `contiguous` compiles, runs, and is **5% faster** at `S=512`
(369.6 ms against 389.0).

**It changes the answer.** With the copy removed the prefill digests read

| S | with the copy (§1.3) | without it |
|---:|---|---|
| 6 | `b9fc5553ee1bf6a2…` | **`115c738baf3e7081…`** |
| 32 | `331668f36da02f21…` | `331668f36da02f21…` |
| 128 | `00159a9dbd308eda…` | `00159a9dbd308eda…` |
| 512 | `07c2797dabc4552e…` | `07c2797dabc4552e…` |

A transposed GEMM is a different blocking and therefore a different summation
order; that the three larger lengths agree is Accelerate's blocking happening
to coincide, not a guarantee. **Rejected**, and the `contiguous` now carries a
comment saying why it is there, because it looks exactly like something to
delete.

### 8.6 SDPA per call, old against new, on the tensors the model passes

`sdpabench.py` — monkeypatch `F.scaled_dot_product_attention`, run a real
prefill, keep the **first call's actual arguments**, time with exactly those.
Shapes and contiguity printed by the harness, not assumed:

```
q (1, 9, S, 64)  k (1, 3, S, 64)  v (1, 3, S, 64)  float32
is_causal=True  scale=0.125  enable_gqa=True  attn_mask=None   30 calls/forward
```

`old, new, old, new` with the artefact swapped on disk and `cmp`-verified
before each run. Minimum of the two rounds; the two rounds agree to **0.5% or
better on every cell**.

| S | upstream | old | **new** | old ratio | **new ratio** |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.132 | 0.396 | **0.371** | 3.0x | **2.8x** |
| 512 | 1.411 | 5.413 | **4.628** | 3.8x | **3.3x** |
| 1024 | 3.790 | 21.263 | **17.429** | 5.6x | **4.6x** |

ms per call.

### 8.7 Model level — old, new, upstream, alternated

`SmolLM2-135M`, `float32`, deterministic ids, 2 warmups then 5 timed passes,
**minimum within a process, then minimum across 3 alternating rounds** of
`old, new, upstream`. Artefact swapped and `cmp`-verified per run; `nm` finds
`ScaleCausal` in exactly one of the two (new 6 symbols, old 0).

| S | upstream | old | **new** | old ratio | **new ratio** | saved |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 75.52 | 85.71 | 85.57 | 1.135x | **1.133x** | 0.1 |
| 512 | 231.86 | 385.17 | **363.34** | 1.661x | **1.567x** | 21.8 |
| 1024 | 462.35 | 1081.38 | **969.26** | 2.339x | **2.096x** | 112.1 |

**The `old` column reproduces the brief's ratios** (1.13x / 1.64x / 2.33x) and
its remaining gaps (9.7 / 154.6 / 618.6 against 10.19 / 153.31 / 619.03), which
is what says this harness is measuring the same thing the last two rounds did.

Control, two fresh `new` processes back to back:

| S | 128 | 512 | 1024 |
|---|---:|---:|---:|
| ratio | 0.995 | 1.007 | 0.989 |

**The machine was not quiet and the control says so.** It reads 1.00 to within
**1.1%**, against §7.6's 0.5% — another agent was running throughout and load
ran 3.1 to 4.1. Round 2 of the three is visibly contaminated (every cell in it,
`old`, `new` and `upstream` alike, is ~10% slow); the minimum-across-rounds
rule is what keeps it from mattering.

### 8.7.1 `S=128` is not resolved at the model level, and that is the honest reading

The `S=128` row above reads 0.1 ms of an 85 ms pass — **0.16%, against a
control spread of 1.1%.** It is not a measurement of anything.

The per-call measurement of §8.6 *is*: at `S=128` the spread is 0.5% and the
effect is 6.3%, giving `30 x (0.396 - 0.371) = 0.75 ms` per forward. That is a
real 0.75 ms and it is **below what the model-level harness can see**. So:

> at `S=128` this change is worth about 0.75 ms of a 10.2 ms gap, measured
> per call; the model-level ratio does not move and is reported as **1.133x
> against 1.135x, i.e. unchanged**.

At the two larger lengths the same prediction lands:

| S | `30 x (old - new)` per call | **measured model-level saving** | agreement |
|---:|---:|---:|---:|
| 128 | 0.75 | 0.1 (unresolvable) | — |
| 512 | 23.6 | **21.8** | 8% |
| 1024 | 115.0 | **112.1** | **2.6%** |

### 8.8 The numbers that must not move, and did not

Logits sha256 over the little-endian `f32` bytes of the flattened `[1, S, V]`
tensor, printed by every run above. **All five `f32` lengths are identical
between the two artefacts and all five still equal §1.3's values**, which were
taken before any of the three changes:

| S | old | new | §1.3 |
|---:|---|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | `b9fc5553ee1bf6a2…` | ✅ |
| 32 | `331668f36da02f21…` | `331668f36da02f21…` | ✅ |
| 128 | `00159a9dbd308eda…` | `00159a9dbd308eda…` | ✅ |
| 512 | `07c2797dabc4552e…` | `07c2797dabc4552e…` | ✅ |
| 1024 | `eda1e173727bb7f5…` | `eda1e173727bb7f5…` | ✅ |

That is 26 forward passes' worth of agreement (3 rounds x 3 lengths x 2
artefacts, plus 2 x 2 x 2 at `S=6` and `S=32`), not a spot check. Upstream's
`S=128` digest came back `71e46824c0c40f15…`, the value §1.3 recorded, so the
*reference* side is the same one too.

**`bfloat16` as well, and it is not a formality here.** Reduced precision
widens to `f32` for SDPA's body, so `bf16` runs this very kernel — the `f32`
arm of it. `S=128`, same alternated harness, twice each way:
`7ff8e9334449b147…` on both artefacts, still the value docs/perf/DTYPE_PERF.md §6.1
recorded. Timing 119.65 -> 118.19 ms, **1.2%, inside the control spread and
reported as unresolved.**

### 8.9 What the fit says happened

Refitting `gap(S) = a·S + b·S²` on `S=512` and `S=1024` and then checking it at
`S=128`, which it was not given:

| term | §7.9 (before) | measured here on `old` | **after** |
|---|---:|---:|---:|
| linear `a` | +0.007 ms/token | −0.006 ms/token | **+0.019 ms/token** (still zero) |
| quadratic `b` | 5.90e-4 ms/token² | 5.96e-4 | **4.65e-4** |

The `old` refit reproduces §7.9's `b` to **1%** from an independent set of
runs. Held out, `S=128`: the new fit predicts a 10.00 ms gap and **10.05 was
measured, 0.5%**.

**22% of the quadratic term is gone.** What is left of it:

| S | quadratic before | **quadratic after** |
|---:|---:|---:|
| 128 | 9.8 ms | **7.6 ms** |
| 512 | 156.2 ms | **122.0 ms** |
| 1024 | 624.9 ms | **487.9 ms** |

### 8.10 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 261 | **261** (unchanged — no new Python-visible op) |
| `tools/golden/compare.py` | 4284/4284, ops=139 | **4290/4290, ops=139** (+6 cases, pending 1 unchanged) |
| `compare.py --self-test` | PASS | PASS, 13 comparators x 11 fault modes |
| `verify_schemas.py` | 4353/4353 | **4353/4353** (unchanged) |
| `cargo test --release` | 18 | **24** (+6) |

`run.sh` and `verify_schemas.py` do not move because nothing new is reachable
by name: `scale_and_causal_mask` is called from Rust and has no `aten` key, no
schema and no Python spelling. That is deliberate — it is not an operator, it
is how one operator computes.

### 8.11 What the six new golden cases can and cannot catch

**They cannot catch the numerics, and it would be wrong to claim they do.**
`_sdpa_pair_check` compares against upstream's *fused* flash kernel, which does
not perform this arithmetic in this order at all, so it is necessarily a
tolerance. Injecting fault 7 of §8.4 — narrowing the scale after multiplying —
and running the whole harness gives **4290/4290 passed**. Golden is
structurally blind to a one-ULP change in SDPA and that is not a defect in it.

What they *do* catch is the shape. Removing the clamp (`keep = r + 1`,
unclamped) and re-running:

```
pyo3_runtime.PanicException: range end index 3 out of range for slice of length 2
compare.py exit 1
```

which is the `q_len=5, kv_len=2` case. **Every SDPA case that existed before
was square (`3x3`, `4x4` for GQA) or had `q_len=2 < kv_len=5`**, so none of
them reached a row where the causal mask stops widening, and the clamp had no
case at all outside the Rust unit tests.

**So the division of labour is: `tensor.rs`'s six unit tests own the
bit-for-bit claim, because they compare against the old spelling rather than
against upstream; the golden cases own the shape.** Saying that here because
the opposite assumption — "golden covers SDPA, so the arithmetic is covered" —
is the one an honest reader would otherwise make.

### 8.12 What is left, and what closing it means

**Fixed here:** two of the seven passes over the score matrix, and the `S x S`
allocation that fed one of them. 22% of the quadratic coefficient,
bit-identically, at every length recorded.

**Not fixed, and here is the shape of it.** At `S=1024` the remaining
per-forward gap is 506.9 ms, and SDPA is **409.2 ms of it (81%)** — 30 calls at
17.43 ms against upstream's 3.79. Split that 13.64 ms per call by *cause*,
using upstream's own kernels as the yardstick (§8.2's right-hand column):

| | ms/call at `S=1024` | share of the SDPA gap |
|---|---:|---:|
| ours, as it now stands | 17.43 | |
| **the same mathematics, every kernel upstream's own, still unfused** | **14.83** | |
| upstream's actual fused kernel | 3.79 | |
| ⟹ *our kernels being slower than upstream's, op for op* | 2.60 | **19%** |
| ⟹ **not materialising the score matrix** | **11.04** | **81%** |

**Four fifths of what is left is not a kernel.** A shim in which every one of
`matmul, affine, add, amax, sub, exp, sum, div, matmul` ran at exactly
upstream's speed would still be 3.9x slow at this shape, because upstream never
builds `[1, 9, 1024, 1024]` — 37.7 MB, walked five more times after it exists —
and we build it and walk it. That is the floor, and it is a property of the
strategy, not of any line of code.

**Closing it means writing a tiled attention kernel**, which is what
`flash.rs` already is: block the query rows, block the key columns, carry a
running maximum and a running sum per row block, and never hold more than a
tile. Two things follow from that being the only route:

1. **It cannot be bit-identical, and the disagreement is not small.** A running
   maximum rescales partial sums as it grows, so the softmax denominator is
   accumulated in a different order for every tiling. docs/models/GENERATE.md §6 is
   the standing warning that upstream disagrees with *itself* in `bf16` under a
   change of accumulation order — so "close enough" is not available, and the
   `f32` digests §8.8 pins would all move. It is a different contract, not a
   faster implementation of this one.
2. **`flash.rs` already measured 20x slower** than this path when it was
   written (docs/kernels/SDPA.md §12) — being upstream's *blocked* kernel is not the
   same as being upstream's *fast* kernel, and the gap between them is
   hand-written vectorisation this crate does not have.

So the answer to "what would it take" is: **a second SDPA path with its own
numerics contract, selected explicitly**, exactly as `flash::reference_enabled`
already selects one. Not a change to this one.

**The 2.60 ms/call that *is* kernel quality**, ordered, at `S=1024`
(the second row was taken in docs/kernels/KERNELS26.md §7 and is now ~0.09 ms rather
than 1.288, so the excess on it is ~+0.03 rather than +1.15):

| item | ours | upstream's | excess |
|---|---:|---:|---:|
| `exp` | 4.258 | 2.417 | **+1.84** |
| `k.transpose(2,3).contiguous()` | 1.288 | 0.134 | **+1.15** |
| `matmul q@kT` | 2.627 | 2.046 | +0.58 |
| `amax(-1)` | 1.067 | 0.711 | +0.36 |
| scale + causal mask (this round) | ~1.45 | 3.18 as three ops | **−1.73** |

- **`exp`** is already Accelerate's `vvexpf` on both sides of the profile
  (`VVEXPF` is 13.4% of an `S=1024` sample). Upstream is faster with its own
  vectorised exponential. Replacing candle's would move every `exp` in the
  library and is a numerics change on its own terms.
- **`k.transpose(2,3).contiguous()`** is 2.4 MB moved at ~3.7 GB/s, because
  candle's `copy_strided_src_f` walks a transposed layout one element at a
  time. A blocked transpose is **pure data movement with no arithmetic in it at
  all**, so unlike everything else on this list it is bit-identical by
  construction. It is the one clean kernel win left, and it is worth about
  1.15 ms of 13.64 — **8% of the SDPA gap, 7% of the model gap at `S=1024`**.
  Not taken here; sized, and left named.

  > **TAKEN, in docs/kernels/KERNELS26.md §7.** `tensor.rs::transposed_contiguous` is
  > the same copy in 32x32 cache blocks, wired into this call site and into
  > `aten.contiguous.default`. Measured: the copy is **5.25x faster at
  > `S=512`** (0.2123 → 0.0404 ms, 1.85 → 9.73 GB/s) and **4.84x at `S=1024`**
  > (0.4232 → 0.0875 ms), which is 1.29x and 1.50x of upstream rather than
  > 6.8x and 7.3x. Per SDPA call that is **−0.50 ms at `S=512` (−10.8%)** and
  > **−1.01 ms at `S=1024` (−5.8%)** — the sizing above said 1.15 ms and was
  > right. Model level: `S=512` **1.589x → 1.527x**, `S=1024` **2.089x →
  > 2.019x**; `bf16` −2.9% and −2.7%. **All ten prefill digests unchanged**
  > (`f32` and `bf16`, at every length §1.3 records), with a new-vs-new control
  > reading 0.995–1.019 and a sabotage that moves every one of them.
  >
  > Standing check (docs/verification/DOCWATCH.md):
  > <!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs transposed_contiguous present -->
- The last row is the change this round made, and it is now **faster than
  upstream's three separate ops** for the same work.

**The honest summary is that the growing gap had two terms; the linear one is
gone (§3), 35% of the quadratic one went with `amax` (§7) and 22% of what
remained went with the scale-and-mask fusion here — and the four fifths of the
rest is upstream not building a tensor that we build. A correct account of that
floor is the result of this round; the fusion is the part of the gap that was
not the floor.**
