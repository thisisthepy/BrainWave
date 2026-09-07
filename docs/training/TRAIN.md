# Training mode: the axis nobody had measured

Every architecture sweep in this repository up to and including docs/kernels/KERNELS26.md calls `.eval()`.
`docs/architectures/ARCH20.md`'s twenty and `docs/kernels/KERNELS26.md`'s twenty-six are both eval-mode numbers, and
`README` §2/§3 say the project exists for **federated learning and test-time adaptation** — which
are training. So the axis the project is for had never been exercised, and "26 of 26" was true
only inside `torch.no_grad()` + `.eval()`.

**Written incrementally, one finding at a time**, for the reason docs/kernels/KERNELS26.md §0 gives for the
same practice.

## 0. Method, and what "training mode" means here

The sweep is `/tmp/train/sweeptrain.py`: `sed 's/\.eval()/.train()/g'` over
`/tmp/k26/sweep26.py`, nothing else changed. `torch.no_grad()` stays, so this measures **the mode
axis alone** — which kernels a `.train()` forward asks for that an `.eval()` forward does not — and
not autograd, which this shim does not have and which is a separate wall.

Upstream is the oracle and was measured first, before anything here was written:

```
llama gpt2 qwen2 mistral gemma gpt_neox opt mpt starcoder2 stablelm olmo phi mixtral
bert bloom cohere falcon gpt_bigcode mamba persimmon
deberta deberta_v2 vits zoedepth sew_d sam3_video          TOTAL 26/26   (.train())
```

so `.train()` costs upstream nothing and every failure below is the shim's.

The vendored tree needs `TORCH_USE_RTLD_GLOBAL=1` (docs/platform/VENDOR.md wall 3) — without it `import
torch` dies in `_load_global_deps` before any of this is reachable.

### The baseline, every gate, before any edit

```
pytests/run.sh                274 ok, 0 FAIL                        exit 0
tools/golden/compare.py       6374/6374, ops=161, pending=1          exit 0
compare.py --self-test        16 comparators x 11 fault modes        exit 0
verify_schemas.py             4458/4458                              exit 0
sweep26   (shim, .eval())     26/26                                  exit 0
sweeptrain (shim, .train())   18/26                                  <-- the new axis
```

**18/26**, and inside it the twenty of ARCH20 are **16/20**, which is the number that started this
work. The eight failures:

| architecture | first wall in `.train()` |
|---|---|
| `gpt2` | `aten.dropout.default` |
| `opt` | `aten.dropout.default` |
| `bert` | `aten.dropout.default` |
| `gpt_bigcode` | `aten.dropout.default` |
| `deberta` | `aten.dropout.default` |
| `deberta_v2` | `aten.dropout.default` |
| `vits` | `aten.dropout.default` |
| `sew_d` | `TensorBase.bernoulli_` |

Seven of the eight arrive through `nn.Dropout` / `nn.functional.dropout`; the eighth does not, and
that is the first finding worth having. **`sew_d` never calls `dropout` at all.**
`transformers/models/sew_d/modeling_sew_d.py:229` is

```python
mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)
```

— DeBERTa's `XDropout`, which rolls its own mask because it needs the mask itself for the backward.
So "dropout" is two requirements wearing one name: the composite, and the primitive under it.

## 1. What `nn.Dropout` actually dispatches to

Measured on upstream torch 2.13.0 with a `TorchDispatchMode` logger, before choosing what to write.
This is the whole design decision, so it is measured rather than assumed:

```
nn.Dropout(0.5).train()   ->  empty_like.default, bernoulli_.float, div_.Scalar, mul.Tensor
nn.Dropout(0.5).eval()    ->  (nothing at all)
torch.dropout(x, 1.0, True)   ->  zeros.default, mul.Tensor
torch.dropout_(x, 0.5, True)  ->  empty_like.default, bernoulli_.float, div_.Scalar, mul_.Tensor
torch.dropout_(x, 1.0, True)  ->  zeros.default, mul_.Tensor
```

**`aten.dropout.default` never fires.** It is `CompositeImplicitAutograd` — the same shape as
`aten::linear` and `aten::layer_norm`, both of which this shim already answers by *decomposition in
`bootstrap.py`* rather than by a kernel (bootstrap.py `_install_nn`'s note: "Reproducing that
decomposition here is therefore *following* upstream rather than routing around it"). It is also
**not** `native_dropout`: that is the functionalised/`torch.compile` spelling, and eager
`nn.Dropout` on CPU does not reach it (`is_fused_kernel_acceptable` requires CUDA/XPU/lazy).

So the honest answer to "implement `aten.dropout.default`" is that there is no such kernel to
implement. What upstream's `_dropout_impl` is, transcribed from
`aten/src/ATen/native/Dropout.cpp` and confirmed against the trace above:

```
TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p)
if (p == 0 || !train || input.numel() == 0)  return input;          // the SAME object
if (p == 1)                                  return input * zeros({}, input.options());
noise = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
noise.bernoulli_(1 - p)
noise.div_(1 - p)
return input * noise                                                 // mul_ for dropout_
```

Three things in there are load-bearing and none is guessable:

* **The identity return is the identity *object*.** `torch.dropout(x, 0.0, False) is x` is `True`
  upstream, measured. The shim's `bootstrap.py` short-circuit already had this exactly right.
* **`p == 1` is a multiply by zero, not a `zeros_like`.** Measured on
  `[1, -1, 0, -0, nan, inf, -inf]`:

  ```
  upstream p=1   ->  [0.0, -0.0, 0.0, -0.0, nan, nan, nan]
  zeros_like     ->  [0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 0.0]
  ```

  Signed zero survives, and `±inf` becomes **nan** because `inf * 0` is nan. A `zeros_like`
  implementation passes every `(y == 0).all()` test anyone would write first. This is §5's first
  sabotage target for exactly that reason.
* **The scale is applied to the *mask*, in the mask's dtype, and only then multiplied.** So the
  survivor value is `bfloat16(1) / bfloat16(0.3)`, not `input * float(1/0.7)`:

  ```
  p=0.7   float32 3.3333332538604736   float64 3.333333333333333
          bfloat16 3.328125            float16 3.333984375
  ```

## 2. Can a seeded comparison against upstream be bit-exact? Yes.

docs/kernels/KERNELS26.md and the `randint` golden case record that `randint`'s *sequence* cannot be
matched, so the question had to be asked of dropout rather than inherited. It was measured:

```
torch.manual_seed(1234); torch.empty(16).bernoulli_(0.5)
torch.manual_seed(1234); torch.empty(16, dtype=torch.float64).uniform_(0, 1) < 0.5
        ->  identical, elementwise
and the generator is left in the same state afterwards (the next rand(4) agrees)
```

and the same holds for **every** dtype `bernoulli_` accepts — `float64 float32 bfloat16 float16
int64 int32 uint8 bool` all match `u64 < p` and all advance the stream by `numel` 64-bit draws.
That is upstream's `bernoulli_distribution<double>`: it holds a `uniform_real_distribution<double>`,
which takes `generator->random64()` regardless of `scalar_t`. `bernoulli_` is **not**
`opmath_type<scalar_t>`-shaped the way `uniform_` is (docs/numerics/RNG.md; `uniform_` on `float16` draws one
*32-bit* word) — that asymmetry is the trap here, and reading it wrong desynchronises the stream at
half the rate rather than producing visibly wrong values.

`rust/torch_c/src/rng.rs` already has `uniform_fill_f64` — `random64()` through
`transformation::uniform_real<double>`, with the `mul_add` contraction docs/numerics/RNG.md §1.2 measured.
So **the answer is yes**: a fixed seed makes shim and upstream dropout comparable value for value,
and the golden cases below do that rather than settling for a distributional check.

`p == 0` and `p == 1` still consume `numel` draws upstream (measured: the following `rand(2)` is
the same for `p=0`, `p=1` and `p=0.5`, and different from no draw at all). A short-circuit there
would be invisible in the values and would desynchronise everything after it.

The one place a seeded comparison is *not* available is a **non-contiguous receiver**:
upstream's `TensorIterator` writes the k-th draw to the k-th element in *physical memory* order, so
`b.t().bernoulli_(p)` fills `b`'s storage in `b`'s order. This shim's `uniform_`/`normal_` already
write in logical row-major order, and `bernoulli_` below follows them rather than diverging from its
own siblings. It is out of reach of the composite (`empty_like` is always contiguous) and of
`sew_d` (likewise), so it is recorded here, not fixed, and the golden cases stay contiguous.

## 3. What landed, and the sweep after it

**One change, three names, because the composite cannot be split from its primitives.**

* `rust/torch_c/src/aten.rs` — `aten.bernoulli_.float`, a new kernel.
* `rust/torch_c/src/aten.rs` — `aten.div_.Scalar`, one line onto the existing
  `arith_inplace_scalar` helper. The out-of-place `div.Scalar` and the in-place
  `add_`/`sub_`/`mul_` scalar forms were all already there; this was the hole in the middle of them.
* `rust/torch_c/src/methods.json` — `bernoulli_`, both overloads in the vendored `.pyi`'s order
  (`.Tensor` then `.float`). Only `.float` has a kernel; `.Tensor` resolves and then refuses, which
  is what `methods.json`'s own README says an entry means.
* `rust/torch_c/src/bootstrap.py` — `torch.dropout` / `torch.dropout_` rewritten from a
  `dispatch("aten.dropout.default", ...)` stub into `at::native::_dropout_impl`, which is the
  decomposition above.

### Sweep after: **18/26 -> 23/26**

Five architectures crossed: **`opt`, `deberta`, `deberta_v2`, `vits`, `sew_d`**. Of the twenty of
ARCH20 the count went **16/20 -> 17/20**.

The `.eval()` sweep is unchanged at **26/26** and every gate is unchanged (§6).

### And the fifth wall, which is the more useful finding

`gpt2`, `bert` and `gpt_bigcode` did **not** cross. They stopped one op later, on something that has
nothing to do with `nn.Dropout`:

```
NotImplementedError: scaled_dot_product_attention(dropout_p != 0)
   -- upstream drops to the math backend here
```

**Attention dropout is not `nn.Dropout`.** `transformers`' SDPA attention passes
`dropout_p=self.attention_dropout if self.training else 0.0` straight into
`F.scaled_dot_product_attention`, and on CPU a non-zero `dropout_p` takes the whole call off the
fused kernel: `_scaled_dot_product_flash_attention_for_cpu` does not implement dropout, so upstream
falls back to `_scaled_dot_product_attention_math`, a different op sequence entirely. So in
`.eval()` these three take one fused op and in `.train()` they take twenty.

`bootstrap.py`'s refusal for that path named `aten.bernoulli_.float` and `aten.div_.Scalar` as the
missing pieces. **That is now stale** — for the third time in that one function, which its own
comments already note twice. Both are implemented as of this section; what is actually missing is
the *composite*.

### The one number that does not agree with upstream, and why it is not this change

`vits` reports `(1, 176)` here against upstream's `(1, 192)`. That is pre-existing and it is
`randint`'s: `run_vits` seeds, builds the model, then draws its input ids with
`torch.randint(0, 48, (1, 6))`, whose sequence this shim does not reproduce (docs/kernels/KERNELS26.md, and
the `randint` golden case says so). Different ids give a different predicted duration and therefore
a different waveform length. In `.eval()` the same pair is `(1, 144)` here against `(1, 176)`
upstream, so the gap predates training mode entirely. The sweep asserts a forward completes, not a
shape match.

## 4. The wall behind dropout: attention dropout is a different backend

**`transformers`' SDPA attention does not use `nn.Dropout` for attention weights.** It passes
`dropout_p=self.attention_dropout if self.training else 0.0` into
`F.scaled_dot_product_attention`, and on CPU that argument selects a *backend*:

```
dropout_p == 0.0   ->  aten._scaled_dot_product_flash_attention_for_cpu   (one op)
dropout_p != 0.0   ->  _scaled_dot_product_attention_math                 (twenty ops)
```

The fused kernel has no dropout — upstream's own kernel refuses with "Currently do not support
dropout > 0" — so a non-zero `dropout_p` is not a slower road to the same answer, it is a different
computation. `bootstrap.py` refused it, and its refusal named `aten.bernoulli_.float` and
`aten.div_.Scalar` as the missing pieces. §3 landed both, so the refusal went stale **for the third
time in that one function**, and `test_the_two_stale_sdpa_refusals_no_longer_claim_a_missing_kernel`
— a standing check a previous round wrote for exactly this — failed and said so by name.

The right answer to a refusal whose dependencies have all landed is not a fourth re-wording. The
math backend is now written, in `bootstrap.py`, transcribed op for op from a `TorchDispatchMode`
trace of torch 2.13.0:

```
[bool mask?]  scalar_tensor(-inf), scalar_tensor(0.), where.self
              mul.Scalar(query, sqrt(scale))
[is_causal?]  ones([L,S], dtype=bool), tril, scalar_tensor x2, where.self
[gqa?]        unsqueeze(2), expand, clone, view      (per key and value)
              transpose.int(key, -2, -1), mul.Scalar(., sqrt(scale))
              matmul
[mask?]       add.Tensor
              _safe_softmax(., -1)
[dropout?]    empty_like, bernoulli_.float, div_.Scalar, mul.Tensor
              matmul(., value)
```

Two details there are measured rather than reasoned, and one of them is the subject of §5's most
useful sabotage:

* **The scale is applied as its square root, to *both* operands** — `query * sqrt(s)` and
  `key.transpose(-2,-1) * sqrt(s)` — rather than once to the product. For `E=8` with no explicit
  scale the factor is `0.5946035575013605`, which is `sqrt(1/sqrt(8))`, and it is not decoration:
  in `float16` with inputs around 100, `q @ k^T` alone is 80000, past `float16`'s 65504, so the
  textbook `softmax(QK^T/sqrt(d))` overflows to `inf` and the softmax answers `nan` where upstream
  answers finite values. Measured on upstream, both ways.
* **A negative explicit `scale` is not passed through.** Upstream roots `abs(scale)` and negates the
  *query* multiplier, so the sign survives exactly once instead of being lost to the square root.

### Sweep after: **23/26 -> 26/26**

`gpt2`, `bert` and `gpt_bigcode` crossed, and the twenty of ARCH20 went **17/20 -> 20/20**.

**Training mode is now 26 of 26, the same as `.eval()`**, and the eval sweep is still 26/26.

| step | sweep (`.train()`) | ARCH20 subset | what moved |
|---|---:|---:|---|
| baseline | 18/26 | 16/20 | — |
| §3 `bernoulli_` + `div_.Scalar` + the dropout decomposition | **23/26** | 17/20 | opt, deberta, deberta_v2, vits, sew_d |
| §4 the SDPA math backend | **26/26** | **20/20** | gpt2, bert, gpt_bigcode |

### A defect found on the way, in a kernel that is not new

The golden cases for `div_.Scalar` failed on `float16` the first time they ran, and the same defect
was in the *out-of-place* `div.Scalar` that has been in the shim for months:

```
float16, ones / 0.3      upstream 3.333984375      shim 3.33203125       (one step apart)
bfloat16, ones / 0.3     upstream 3.328125         shim 3.328125         (agree)
```

`div_true_kernel`'s `Half`/`BFloat16` branch reads the **original** scalar in `float` and multiplies
by its reciprocal — `opmath_t inv_b = opmath_t(1) / iter.original_scalar_value<opmath_t>(2)` — where
`add`/`mul` narrow the scalar to the tensor's dtype first (that is docs/models/GENERATE.md §3.2's
`x + 0.3` adding `0.30078125`). The shim narrowed for all of them. **`bfloat16` cannot see the
difference** — both roads round to `3.328125` — which is why `div.Scalar` passed every case it had.
Fixed in `div_scalar_reduced_float`, for both the in-place and out-of-place forms, because a
`float16` `x /= 0.3` disagreeing with `x = x / 0.3` is a difference nobody would look for.

## 5. Sabotage: what each case can and cannot see

Nine faults, each the most plausible wrong shape for the thing it breaks. Every one was applied to
the source, rebuilt, and run through `tools/golden/compare.py` and `pytests/run.sh`.

| # | fault | golden | smoke |
|---|---|---:|---|
| S1 | `bernoulli_` draws from the **f32** stream (copies `uniform_`'s `opmath_type` rule) | 134 FAIL | 3 FAIL, incl. the training sweep at 0.105 |
| S2 | `bernoulli_` **short-circuits** `p==0`/`p==1` without drawing | 2 FAIL | 1 FAIL |
| S3 | `dropout(p=1)` returns **`zeros_like`** instead of `input * zeros({})` | 4 FAIL | 1 FAIL |
| S4 | dropout scales the **input** by `1/(1-p)` instead of the mask | **0** | **0** |
| S5 | `div_.Scalar` narrows the divisor first (the defect §4 records) | 4 FAIL | 2 FAIL |
| S6 | dropout's range check moved **after** the short-circuit | 2 FAIL | 1 FAIL |
| S7 | SDPA math scales **once, after the matmul** instead of sqrt over both | **0**, then 1 FAIL | 0 |
| S8 | SDPA math **skips the dropout step** | 19 FAIL | 2 FAIL |
| S9 | `bernoulli_` compares `u <= p` instead of `u < p` | **0** | **0** |

Which case caught what is as much the point as the count:

* **S2 was caught by two cases and only two** — `bernoulli_(p=0.0) then uniform_` and its `p=1.0`
  twin, whose *result* is the following `uniform_` fill rather than the bernoulli itself. Every
  case that looks at the returned tensor passes a short-circuit, because the returned tensor is
  right. This is the case shape a stochastic kernel needs and it is the one nobody writes first.
* **S3 was caught on the sign of a zero**, at index 1 of `[1, -1, 0, -0, inf, -inf]`, before it ever
  reached the `inf -> nan` entries. `(y == 0).all()` passes `zeros_like`; `_signed_zero_check` does
  not.
* **S6 was caught for `train=False`, and correctly not for `p=nan, train=True`** — moving the check
  after the short-circuit does not change that path, because `nan` is neither `0` nor falsy.

### The three that did not fail, and why two of them are right

**S9 (`u <= p`) cannot be caught, and should not be.** The draws are 53-bit uniforms; `u == p`
happens with probability about `2^-53` per element. The two implementations are the same function
for any input anyone will ever produce. A case that "caught" it would be testing the generator's
internals, not the kernel.

**S4 (scale the input, not the mask) cannot be caught *in this shim*, and the reason is a second
defect.** Upstream distinguishes them: over 4000 random values, `(x * (1/(1-p))) * mask` differs
from `dropout` on ~10% of the survivors by one ULP in `float16` and `bfloat16`, and not at all in
`float32`. In this shim they are bit-identical, because **`mul.Scalar` narrows its scalar to the
tensor's dtype and upstream's does not** — the same class of defect §4 fixed in `div`, in the op
next door:

```
bfloat16, [3, 5, 7, 0.7] * 0.3     upstream [0.8984375,   1.5, 2.09375,     0.2099609375]
                                   shim     [0.90234375,  1.5, 2.109375,    0.2099609375]
float16,  same                     upstream [0.89990234375, 1.5, 2.099609375,  0.2100830078125]
                                   shim     [0.900390625,   1.5, 2.099609375,  0.2100830078125]
```

`mul_kernel`'s reduced-float branch is `opmath_t b = iter.original_scalar_value<opmath_t>(2)` — the
un-narrowed scalar, exactly as `div`'s is.

> **Fixed since, in [`docs/numerics/SCALAR.md`](../numerics/SCALAR.md).** It was done as its own round for the reason
> given below, and the reason held: the prefill digests did **not** move, because a dispatch trace
> over a real `bfloat16` forward shows every Python number it passes is an integer or exactly
> representable, so `mul.Scalar` is never reached with a separating scalar. Where one does reach it
> — SDPA's math backend — 36 of 128 elements were wrong before and 0 after. The prediction two
> paragraphs down is discharged: the widened dropout cases do now catch S4, at 52/46/50/54
> differing survivors in `float16`/`bfloat16` against upstream's identical counts.
>
> The family turned out to have no rule to infer: `hardshrink` narrows and `softshrink` widens.

**This was reported and not fixed** at the time, deliberately:
`mul.Scalar` is on the eval hot path (RMSNorm, RoPE, the attention scale), so changing it would move
`bfloat16` numerics repository-wide including the prefill digest docs/numerics/SEQLEN.md records, and that
document is outside this round. It is also already golden-covered and passing, so `mul_scalar_cases`
has the same blind spot these dropout cases had — a scalar that is representable, or a tensor of
ones. Someone should widen it in the same change that fixes the kernel.

The dropout cases were widened anyway, from `[1.0] * 24` to 240 values spanning a real range, so
that they **will** catch S4 the day `mul.Scalar` is fixed. Their first draft used an all-ones input,
where the two shapes coincide by construction: a test that could not fail, found only by breaking
the kernel deliberately.

**S7 was not caught, and that one was a real gap.** Scaling once after the matmul is algebraically
equal to scaling both operands by the square root; at ordinary magnitudes the two differ by about
one ULP, and the training sweep's `1e-5` bound is sized to separate dropout *masks*, not rounding
reorders. The math backend had **no golden coverage at all** — it has no dispatch key, so
`CASE_BUILDERS` had nowhere to register it. 21 cases now hang off
`aten._scaled_dot_product_flash_attention_for_cpu.default`, the key of the other backend of the same
function, and one of them is the `float16` overflow above:

```
float16, |q| = |k| = 100, dropout_p=0.25
    upstream / correct   [-0.66650390625, 0.66650390625, ...]
    scale-once           [nan, nan, nan, ...]
```

Re-run against S7, that single case fails and the other twenty pass. It is the only thing in the
suite that can see the difference, and it exists because the sabotage went looking for one.

## 6. Gates

| gate | before | after |
|---|---|---|
| `pytests/run.sh` | 274 ok, 0 FAIL | **285 ok, 0 FAIL** |
| `tools/golden/compare.py` | 6374/6374, ops=161, pending 1 | **6587/6587, ops=163, pending 1** |
| `compare.py --self-test` | 16 comparators x 11 fault modes | **unchanged** |
| `verify_schemas.py` | 4458/4458 | **4465/4465** |
| sweep26 (`.eval()`) | 26/26 | **26/26** |
| sweeptrain (`.train()`) | 18/26 | **26/26** |

`+213` golden cases: 192 for `bernoulli_`/`div_.Scalar`/the dropout composite, 21 for the SDPA math
backend. `+11` smoke tests: ten for this work and one for the training sweep itself (§7).
`+7` schema entries: `bernoulli_`'s two overloads reaching both halves of the round trip.

### The eval-mode numbers that must not move, and did not

`docs/numerics/SEQLEN.md` §1.3's prefill logits sha256 over real SmolLM2-135M, re-measured on this artefact.
**A training-mode kernel that changes an eval-mode result is a bug**, and `mul.Scalar` above is
precisely the change that would have moved these, which is the concrete reason it was left alone.

| S | f32 | §1.3 | bf16 |
|---:|---|:--:|---|
| 6 | `b9fc5553ee1bf6a2…` | ✅ | `8ef1550ea33c4f3d…` |
| 32 | `331668f36da02f21…` | ✅ | `b81325c83a0a3d15…` |
| 128 | `00159a9dbd308eda…` | ✅ | `7ff8e9334449b147…` ✅ (docs/perf/DTYPE_PERF.md §6.1) |
| 512 | `07c2797dabc4552e…` | ✅ | `9ab1e82f01378e38…` |
| 1024 | `eda1e173727bb7f5…` | ✅ | — |

The three unstarred `bf16` digests are recorded here for the first time; only `S=128` had a prior
value to check against, and it matches.

## 7. The standing check

Before this, **`.eval()` was assumed everywhere and nothing would have noticed training regressing**
— not the smoke tests, not golden, not either sweep. `test_train_mode_forwards_the_four_
architectures_eval_mode_hid` in `pytests/test_shim.py` is that gap closed, built in the shape
`test_a_real_transformers_llama_forward_matches_upstream` set: the same `transformers` in both
interpreters, the vendored tree in a subprocess and upstream in this one, weights pushed in by one
shared procedure so neither side depends on the other's random stream.

It runs `gpt2`, `opt`, `bert` and `gpt_bigcode` — the four ARCH20 architectures that forwarded in
`.eval()` and stopped in `.train()` — from the sweep's own toy config, unchanged, so the check and
the sweep measure the same models. Three assertions per architecture, each catching a different
regression:

| assertion | what it catches |
|---|---|
| the forward completes | a missing kernel |
| `.train()` differs from `.eval()` by > 1e-3 | dropout silently becoming a no-op, which every "it runs" check passes |
| train logits match upstream's within 1e-5 | the mask itself, draw for draw |
| eval logits match upstream's within 1e-5 | a training-mode change reaching an eval-mode result |

The bound was measured at **both** ends rather than chosen. Clean: `gpt2` 5.96e-08, `opt` 1.79e-07,
`bert` 5.96e-08, `gpt_bigcode` 5.96e-08. With the shim's dropout mask drawn from a different seed —
the shape of "a plausible mask from the wrong stream": `gpt2` 0.474, `opt` 0.458, `bert` 0.592,
`gpt_bigcode` 0.444. So `1e-5` is 56x above the worst clean run and 44000x below the cheapest wrong
answer. It costs 3.6 seconds.

`test_the_two_stale_sdpa_refusals_no_longer_claim_a_missing_kernel` was kept and inverted rather
than deleted, so the file still records which refusal came down and why.

## 8. What is still not measured

* **Autograd.** This whole document is `.train()` inside `torch.no_grad()`, which isolates the mode
  axis. A real federated-learning or test-time-adaptation step needs a backward, and this shim has
  none. That is the next wall and it is much larger than this one.

  > **Closed, in two rounds.** `docs/training/BACKWARD.md` built the backward — a reverse walk over a captured
  > region rather than a `VariableType` — and its §18 landed the two derivative rules a `.train()`
  > forward needs on top of the kernels below. `gpt2` and `bert` take a real Tent step in `.train()`
  > against upstream's own autograd (`docs/models/ADAPT.md` §14). §9 is what that changed about *this*
  > document, including one thing in it that stopped being true.
* **`bernoulli_.Tensor`.** Declared in `methods.json`, no kernel. Nothing in the 26 asks for it.
* **`native_dropout`.** The functionalised spelling, which `torch.compile` and the meta/fake path
  use. Eager CPU never reaches it, so it is absent rather than wrong.

  > **Landed since, and eager CPU still never reaches it — a *capture* does.** §9.1.
* **`bernoulli_` on a non-contiguous receiver** writes in logical order where upstream writes in
  physical order (§2). It follows `uniform_`/`normal_`, which have the same property; no caller in
  the 26 can reach it, because `empty_like` is always contiguous.
* **`mul.Scalar`'s reduced-float scalar** (§5), reported and left.

  > **Fixed since, in `docs/numerics/SCALAR.md`** — the note under §5 says so, and the fix is what makes
  > §9.1's dropout *gradient* bit-identical to upstream at `bfloat16`.

## 9. The backward, and the one claim above that stopped being true

### 9.1 `native_dropout` is here, and §8's third bullet was right about the wrong thing

The bullet said eager CPU never reaches `native_dropout`, so its absence was not a defect. **Both
halves survived; the conclusion did not.** Eager CPU still never reaches it — `is_fused_kernel_
acceptable` still wants CUDA/XPU/lazy — but a *capture* does, and for the reason §1 gives about the
composite: `_dropout_impl` decomposes onto `bernoulli_` and `div_`, both of which write in place, and
capture refuses mutation so that a trace stays single-assignment. So `.train()` was uncapturable for
the four architectures §7 names, and `native_dropout` — upstream's own functionalised spelling, one
node, and it hands back the mask — is what fixed it. `bootstrap.py` takes that route **only inside a
capture region**, which is following upstream's own rewrite rather than routing around it.

It is not bit-identical to the eager branch and is not claimed to be: `_dropout_impl` divides the
*mask* by `1 - p` and `native_dropout` multiplies the *output* by a `1/(1-p)` narrowed to the
tensor's dtype. The masks agree draw for draw; the survivors can differ by an ulp at `bfloat16` and
`float16`. That is upstream's difference reproduced, not this shim's.

### 9.2 §2's answer is what makes the gradient checkable

§2 asked whether a seeded comparison against upstream can be bit-exact and answered yes, because
`bernoulli_` draws in `float64` for **every** dtype. That paid off somewhere it was not written for.
A dropout *backward* is stochastic, which is exactly where a test that cannot fail hides — the
comfortable version is a distributional check that no wrong implementation fails. §2's answer means
the gradient is comparable against upstream **draw for draw** instead:

| | elements | differing from upstream, bit for bit |
|---|---:|---:|
| `float32`, `p = 0.5` | 24 | **0** |
| `bfloat16`, `p = 0.7` | 24 | **0** |

and the mask is asserted equal on both sides *first*, because if the two had drawn different masks
the comparison below it would have been meaningless.

`bfloat16` at `p = 0.7` is not decoration: upstream has two spellings of this derivative and they are
different numbers there. §5's S4 was "the shim cannot tell `x * (1/(1-p)) * mask` from
`dropout(x)` because `mul.Scalar` narrowed and upstream's did not"; `docs/numerics/SCALAR.md` fixed that, and
the same fix is what makes `(g * mask) * scale` reach upstream's answer here rather than
`g * (mask * scale)`'s. **The forward's S4 and the backward's are the same defect met from both
sides, and only the second one is bit-exact today because the first was fixed in between.**

### 9.3 The sweep did not move, and this time that is the whole point

| gate | §6 | now |
|---|---|---|
| sweeptrain (`.train()`) | 26/26 | **26/26** |
| sweep26 (`.eval()`) | 26/26 | **26/26** |
| prefill sha256, f32 × 5 and bf16 × 4 | 9/9 | **9/9 unchanged** |
| `tools/golden/compare.py` ops covered | 163 | 168, **unchanged by this round** |

A round that gave two architectures a `.train()` backward and moved **no** forward number is the
claim: the kernels were all here already (§3, §4), and what was missing was two derivatives.

<!-- DOCWATCH: op-implemented aten.native_dropout.default -->
<!-- DOCWATCH: op-implemented aten.bernoulli_.float -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs native_dropout_backward present -->
