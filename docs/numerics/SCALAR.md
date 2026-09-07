# The reduced-float scalar rule

**One-line result.** When an op folds a Python number into a `float16` or
`bfloat16` tensor, upstream reads that number at **one of two precisions**, and
which one is a property of *the individual kernel*, not of the `.Scalar` family:
`mul` and `div` read it at `opmath_type` (`float`), `add` and `sub` read it
narrowed to the tensor's own dtype. This shim narrowed for all of them.

**The second result is that no recorded digest moved** — not `float32`, which
was the control, and not `bfloat16` or `float16`, which were expected to. That
is not luck and it is not the change failing to bite: a `TorchDispatchMode` over
a real SmolLM2-135M `bfloat16` prefill shows the forward never calls a changed
op with a scalar that separates the two rules. §4 has the log and the
demonstration that the numerics *do* move where such a call exists.

The defect was not found by reading kernels. It was found because a **sabotage
fault failed to fail** (docs/training/TRAIN.md §5, S4): scaling dropout's *input* instead
of its *mask* is detectable upstream and was not detectable here, because
`mul.Scalar`'s narrowing made the two shapes bit-identical.

---

## 0. What this document is, and what it may not be used for

- Upstream is `torch` 2.13.0, the macOS arm64 wheel in
  `/Volumes/macMini/caches/spike-venv` — the same build every other measurement
  in this repository uses.
- Every "upstream does X" below is a **measurement**, run over 420 values per
  dtype against two models built out of upstream's own arithmetic, not a reading
  of ATen source. Where a kernel line is quoted it is quoted as corroboration
  after the fact.
- Comparison is **bit-exact on the packed bytes**. The whole effect is one
  representable step, and the golden harness's `bfloat16` tolerance is `6e-2` —
  three orders of magnitude too loose to see any of this. Every case added by
  this round uses `_exact_value_check`.

---

## 1. Establishing the rule

Two models, both computed with upstream's own `float32` arithmetic so that the
*only* difference between them is the scalar's precision:

```
narrow   the scalar is first rounded to the tensor's dtype, then used
widen    the scalar is kept at float32 -- opmath_type<bfloat16> == float
```

For each op, each dtype and each of ten scalars, upstream's answer is compared
against both. `/tmp/scalar_probe2.py` is the probe; the inputs are 420 values
spanning `[-4, 4]` plus hand-picked awkward ones (`±0.3`, the two `bfloat16`
neighbours of `0.3`, signed zeros, integers).

**A third column matters as much as the two models: `same`.** For a good many
ops the two models are *identical* — the scalar's only role is to be stored or
compared, and storing it into the output narrows it whichever road you took.
Those ops cannot be got wrong this way, and saying so is what stops the next
reader from "fixing" them.

| upstream reads the scalar at | ops (measured, `float16`/`bfloat16`) |
|---|---|
| **`opmath_t` — widen** | `mul.Scalar`, `div.Scalar`, `floor_divide`, `div.*_mode` (both modes), `leaky_relu`, `elu`, `celu`, `softshrink`, `softplus` (both `beta` and `threshold`), `addcmul`/`addcdiv` `value`, `lerp` weight, `norm` `p` |
| **`scalar_t` — narrow** | `add.Scalar`, `sub.Scalar`, `rsub.Scalar`, `add`/`sub` `alpha`, `pow.Tensor_Scalar` exponent, `pow.Scalar` base, `remainder.Scalar`, `fmod.Scalar`, `hardshrink` `lambd`, every comparison (`eq`/`ne`/`lt`/`le`/`gt`/`ge`) |
| **structurally insensitive** | `clamp` (both bounds), `clamp_min`/`clamp_max`, `threshold`, `hardtanh`, `masked_fill`, `where.Scalar*`, `fill_`, `full`, `scalar_tensor`, `nan_to_num` |

`hardshrink` narrows and `softshrink` widens. `clamp` is insensitive and
`threshold` is insensitive, but `leaky_relu` — which looks like the same shape of
kernel — widens. **There is no principle here that survives contact with the
table**; the rule really is per kernel, and upstream's own source says the same
thing when you look afterwards: `softshrink_kernel` writes
`lambd.to<opmath_t>()` and `hardshrink_kernel` writes `lambd.to<scalar_t>()`,
two files apart.

### 1.1 Why `clamp` and friends cannot be got wrong

`clamp(bf16 x, min=0.3)` looks like it should be sensitive and is not. The
scalar reaches the output in only two ways: as a comparison against a value that
is already `bfloat16`, or as the returned value itself — and the return narrows
it. Take `x = 0.298828125`, the `bfloat16` neighbour below `0.3`:

```
narrow:  min = bf16(0.3) = 0.30078125,  max(x, min) = 0.30078125
widen:   0.298828125 < 0.3, so the result is 0.3, stored as bf16 -> 0.30078125
```

The two roads meet at the store. That holds for every value, which is why the
`same` column is 10/10 for those ops in every dtype. **They are listed above so
that nobody adds cases for them believing the cases prove something.**

### 1.2 `float32` is not in this story, and that is load-bearing

At `float32` the widen model and the narrow model are the *same computation*:
`opmath_type<float>` is `float`, so "narrow the scalar to the tensor's dtype" and
"widen the scalar to `opmath_type`" both say `float`. Every arithmetic op above
reads `same` 10/10 at `float32`.

**So a `float32` result that moves under this change has not been fixed, it has
been broken.** §4 uses that as the control.

Two exceptions, and neither is the scalar rule: `pow.Tensor_Scalar` keeps a
`double` exponent at `float32` (§3.3), and `celu` disagrees on 2 of 10 scalars
for its own reasons. Both are recorded here so the next reader does not
rediscover them as this rule.

---

## 2. `mul.Scalar` — the reported defect

```
bfloat16   [3, 5, 7] * 0.3    upstream [0.8984375,     1.5, 2.09375     ]
                              shim     [0.90234375,    1.5, 2.109375    ]
float16    [3, 5, 7] * 0.3    upstream [0.89990234375, 1.5, 2.099609375 ]
                              shim     [0.900390625,   1.5, 2.099609375 ]
```

`5 * 0.3` agrees in both dtypes and `7 * 0.3` agrees in `float16`. **A case set
that happened to pick those values would have found nothing**, which is the
shape of this whole document.

Fixed in `arith_scalar` by building the scalar operand at `opmath_in(storage)`
for `Mul` and `Div` and leaving `Add`/`Sub` narrowing as they were. The fix is a
three-line branch; finding out that it was only those two took the table in §1.

### 2.1 The cases that could not fail, and the ones that replace them

`mul_scalar_cases` had three scalars: `2.0`, `0.0`, `-1.5`. **All three are
exactly representable in `float16` and `bfloat16`**, so narrowing them is the
identity and the builder passed under either implementation — 6587/6587 green
over an op that had been wrong since it was written.

The replacement (`_scalar_rule_cases`) needs two things at once, and dropping
either one makes every case pass again:

* the **scalar** must not be representable in the tensor's dtype;
* the **tensor values** must be, so the scalar's rounding is the only thing
  under test.

Separating power of each candidate, measured over `[3, 5, 7, 11, 13, 96, -3, -5]`:

| scalar | narrowed to `bfloat16` | separates | narrowed to `float16` | separates |
|---|---|---:|---|---:|
| `0.3` | `0.30078125` | **5/8** | `0.300048828125` | **3/8** |
| `0.7` | `0.69921875` | 1/8 | `0.7001953125` | **5/8** |
| `1.3` | `1.296875` | **5/8** | `1.2998046875` | **4/8** |
| `0.1` | `0.10009765625` | 1/8 | `0.0999755859375` | 4/8 |
| `0.5` | `0.5` | **0/8** | `0.5` | **0/8** |
| `2.0` | `2.0` | **0/8** | `2.0` | **0/8** |

`0.3`, `0.7` and `1.3` are carried because between them they separate in *both*
reduced dtypes; `0.1` is not, because its `bfloat16` column is the same 1/8
near-miss that let `div.Scalar` pass for months (docs/training/TRAIN.md §4 —
`bfloat16` rounds both roads of `1 / 0.3` to `3.328125`). `0.5` and `2.0` are
kept **as controls**: they pass under either rule, and a run in which only the
controls pass is a run that has stopped testing anything.

Red before the fix, green after:

```
FAIL aten.mul.Scalar :: (dtype=float16,  scalar=0.3) -- 3/8 elements differ
FAIL aten.mul.Scalar :: (dtype=float16,  scalar=0.7) -- 5/8 elements differ
FAIL aten.mul.Scalar :: (dtype=float16,  scalar=1.3) -- 4/8 elements differ
FAIL aten.mul.Scalar :: (dtype=bfloat16, scalar=0.3) -- 5/8 elements differ
FAIL aten.mul.Scalar :: (dtype=bfloat16, scalar=0.7) -- 1/8 elements differ
FAIL aten.mul.Scalar :: (dtype=bfloat16, scalar=1.3) -- 5/8 elements differ
     the four control cases passed, before and after
```

### 2.2 `mul_.Scalar` is **not** the same op, and upstream is the reason

The in-place spelling was measured too, and it does not follow `mul.Scalar`:

```
bfloat16 [3,5,7,11], scalar 0.3
  torch.ops.aten.mul.Scalar        0.8984375 ...   widen
  torch.ops.aten.mul_.Scalar       0.90234375 ...  NARROW
  x * 0.3        (python)          0.8984375 ...   widen
  x *= 0.3       (python)          0.8984375 ...   widen
  torch.ops.aten.div_.Scalar       widen
```

**`torch.ops.aten.mul_.Scalar` is the only spelling of a scalar multiply in
upstream that narrows**, and `div_.Scalar` — the same shape of op, one letter
away — does not. Checked over 4096 values × 4 scalars × 2 dtypes; it is not a
tail or a vectorisation edge.

The shim reproduces it as measured, which means `mul_.Scalar` keeps the
narrowing this document removes everywhere else. That is deliberate and it has a
cost, stated plainly: a caller who writes `x *= 0.3` on a `bfloat16` tensor gets
the *narrowed* answer here and the *widened* answer upstream, because upstream's
Python `*=` does not reach `mul_.Scalar` at all — a `TorchDispatchMode` over
`x *= 0.3` reports `aten.mul_.Tensor`. The shim's parser reports `mul_.Scalar`,
so the two land on different upstream kernels for the same source line.

Golden cases pin the narrowing (`aten.mul_.Scalar`, five scalars × two dtypes,
`_exact_value_check`) so that the asymmetry is asserted rather than merely
present. Closing it properly is a resolver change, above this file, and is
listed in §6.

---

## 3. The rest of the family

`mul.Scalar` was the reported defect. It was not the only one, and the point of
establishing the rule in §1 rather than fixing two call sites is that **three
more ops were wrong and two of them were wrong in the opposite direction.**

The differential that found them is shim-vs-upstream, bit-exact, over every
`Scalar`-taking op `_aten_implemented()` advertises, at four dtypes and ten
scalars (`/tmp/shim_diff.py`). Run twice: once with scalars that are *not*
representable in the reduced dtypes and once with scalars that are. **An op
that disagrees on both is not this defect** — it has an ordinary precision
problem, and §5 lists the three that turned out to be exactly that.

| op | before | after |
|---|---:|---:|
| `mul.Scalar` | 6/10 bf16, 8/10 f16 | **0** |
| `floor_divide.Scalar` | 3/10 at *every* dtype incl. `float64` | **0** |
| `div.Scalar_mode` (floor) | 3/10 bf16, 3/10 f16 | **0** |
| `div.Scalar_mode` (trunc) | 3/10 bf16 | **0** |
| `pow.Tensor_Scalar` | 8/10 bf16, 8/10 f16 | **0** |
| `pow.Scalar` | 10/10 bf16, 10/10 f16, 10/10 f32 | **0** |
| `mul_.Scalar`, `div_.Scalar`, `add_`, `sub_`, `rsub`, `remainder`, six comparisons, `clamp`, `clamp_min`, `masked_fill`, `where.ScalarOther`, `fill_`, `leaky_relu` | 0 | 0 |

`leaky_relu` is worth naming among the ones that were already right: it widens,
and it was already widening — `leaky_relu_cases` happens to pass slopes of
`0.01` and `0.1`, neither representable, so that builder has been asserting this
rule by accident since it was written. It is the one place in this repository
where the blind spot was avoided, and not on purpose.

### 3.1 `div` was already fixed, and the fix survives a closer look

docs/training/TRAIN.md §4 fixed `div.Scalar`/`div_.Scalar` before this round. Re-checked
here against a model of upstream's branch rather than against the old cases:
upstream computes `opmath_t(1) / opmath_t(scalar)` — the reciprocal in
**`float`** — and `div_scalar_reduced_float` writes exactly
`1.0f32 / (scalar as f32)`. A model that takes the reciprocal in `f64` and
narrows disagrees with both on 2 of 10 scalars, so the `f32` in that line is
load-bearing and not incidental.

### 3.2 Floor and truncating division — where one ULP becomes one unit

`div_floor_kernel` and `div_trunc_kernel` carry the same reduced-float scalar
branch `div_true_kernel` does. The shim narrowed the divisor into the tensor's
dtype *and* narrowed every intermediate step of `div_floor_floating` with it.
Both are wrong, and here the consequence is not a rounding step:

```
bfloat16   [40, 43, 61, -49] // 0.3   upstream [133, 143, 203, -164]
                                      before   [133, 143, 202, -163]
float16    [7, 14, -49]      // 0.7   upstream [10, 20, -71]
                                      before   [ 9, 20, -71]
```

**A floor turns a fractional error into an integer one.** Whether it does so at
any particular value is erratic, though, which matters for the cases — see §4.2.

The same change surfaced a second defect that has nothing to do with scalars.
`floor_divide.Scalar` computed `floor(a / b)`; `div.Scalar_mode(floor)`
computed upstream's `fmod`-based `div_floor_floating`. `div_floor_float`'s own
doc comment has said since it was written that `floor(a / b)` "is the plausible
implementation and it is wrong", and one of the two spellings of floor division
in this file was doing it anyway:

```
float64  -3.0 // 0.3    upstream -11.0    before -10.0
```

`-3.0 / 0.3` in `f64` is `-10.000000000000002`, and the two algorithms take
different views of that. They now share one function, so the two keys cannot
drift apart again; a pytest asserts they agree as well as what they agree on.

### 3.3 `pow` narrows, and `float32` narrows too

The other half of §1's split, and the reason the rule has to be established per
kernel rather than argued from `mul`. `pow_tensor_scalar_kernel` converts the
exponent to the dispatched `scalar_t`; the shim handed `pow_from_pairs` the
`f64` the parser produced.

```
float16   [3] ** 0.3     upstream 1.390625        before 1.3896484375
bfloat16  0.3 ** [3]     upstream 0.0272216796875 before 0.0269775390625
```

**And `float32` narrows as well** — `opmath_type<float>` is `float`, so there is
no widening anywhere in this kernel and the parser's `f64` is never the value
upstream uses. That is the one place in this change where a `float32` result
moves. §4 shows it does not reach any recorded digest, and why.

**What `pow` at `float32` does *not* get is a case**, and that is deliberate.
Upstream's `float32` `pow` answers different bits for the same element
depending on the tensor's length — SLEEF's vectorised `powf` against libm's on
the tail:

```
float32  [3, 5, 7, 11, 13, 96, 2, 0.5] ** 0.3   differs from the same elements
                                                one at a time on 4 of 8
float32  0.3 ** [..., 0.5, ...]                 1.1401753425598145 in the vector
                                                1.140175461769104  alone
```

A golden case at `float32` would be pinned to whichever road an 8-element
tensor happens to take. That is the shape of a test that passes for the wrong
reason, so the `pow` scalar-rule cases run at `float16` and `bfloat16` only and
this paragraph is what stands in for them. The residual `float32`/`float64`
disagreements that remain (1 of 7 exponents, ~1 ULP) are the same
vectorisation/libm difference: on the pair that separates them, `5.0 ** -1.5`,
this shim's `f64` road is the **correctly rounded** `float32` answer and
upstream's is one step below.

---

## 4. What moved, and what did not

Every number here is an A/B against the artefact built from `develop` at
`f83f94c` (`base_C.dylib`, kept on disk and `cmp`-verified before each run),
with the same harness docs/numerics/SEQLEN.md §1 uses.

### 4.1 The digests — none of them moved

| dtype | S | recorded | base | new |
|---|---:|---|---|---|
| `float32` | 6 | `b9fc5553ee1bf6a2…` (SEQLEN §1.3) | same | **same** |
| `float32` | 32 | `331668f36da02f21…` | same | **same** |
| `float32` | 128 | `00159a9dbd308eda…` (also DTYPE_PERF §6.1) | same | **same** |
| `float32` | 512 | `07c2797dabc4552e…` | same | **same** |
| `float32` | 1024 | `eda1e173727bb7f5…` | same | **same** |
| `bfloat16` | 6 | `8ef1550ea33c4f3d…` (TRAIN §6) | same | **same** |
| `bfloat16` | 32 | `b81325c83a0a3d15…` | same | **same** |
| `bfloat16` | 128 | `7ff8e9334449b147…` | same | **same** |
| `bfloat16` | 512 | `9ab1e82f01378e38…` | same | **same** |
| `float16` | 6 / 32 / 128 / 512 | not previously recorded | `d48534af3d22e7f0…` / `1c5f53ebc584babe…` / `38a0b21c39ea24f6…` / `0eca60265a8c734e…` | **same** |

The `float32` row is the control §1.2 asks for and it holds — including through
the `pow` change, which does touch `float32`.

### 4.2 Why the `bfloat16` digests did not move either

Not luck. A `TorchDispatchMode` over a real `bfloat16` SmolLM2-135M prefill
(`S=32`) lists **every** op that receives a Python number:

```
  30  aten._scaled_dot_product_flash_attention_for_cpu   (0.0, 0.125)
  61  aten.add.Tensor                                    (1e-05,)
   2  aten.mul.Tensor                                    (1.0,)
  61  aten.pow.Tensor_Scalar                             (2,)
       ... plus arange/cat/slice/transpose/unsqueeze, all integer axes
```

**`mul.Scalar` is not called at all**, `pow`'s exponent is `2`, the attention
scale is `0.125`, and `add`'s `1e-05` goes to the `.Tensor` overload and to the
half of the family that still narrows. Every scalar in that list is either an
integer or exactly representable in `bfloat16`, so no changed kernel can move.

**That is a statement about this model, not about the change.** Where a
separating scalar does reach `mul.Scalar`, the numerics move and they move onto
upstream's answer. The SDPA *math* backend is such a place: for `E=8` it applies
`sqrt(1/sqrt(8)) = 0.5946035575013605` to the query (docs/training/TRAIN.md §4). At the
tensors it actually passes:

```
mul.Scalar(bfloat16 query [1,2,8,8], 0.5946035575013605)
    base   36 of 128 elements differ from upstream   (first: -0.2490234375 vs -0.25)
    new     0 of 128                                  BIT-EXACT
```

Run the whole composite through and it still disagrees with upstream — on 47 of
128 elements after, 44 before. **That is not a regression and it is not this
op**: the same composite at `float32`, where nothing in this change applies,
disagrees with upstream on 57 of 128 by ~1e-7, and `base` and `new` are
identical there. The math backend has a pre-existing 1-ULP disagreement of its
own (accumulation order in the `bmm`/softmax); which elements happen to coincide
at 1 ULP is arbitrary, and the maximum absolute error against upstream is
unchanged at one `bfloat16` step.

### 4.3 The cost — none that is measurable

`bfloat16` `S=128` prefill, 3 alternating rounds, 2 warmups + 5 timed passes,
minimum within a process then minimum across rounds.

| | round 1 | round 2 | round 3 | **min** |
|---|---:|---:|---:|---:|
| base | 107.52 | 106.60 | 107.10 | **106.60** |
| new | 107.73 | 106.55 | 107.67 | **106.55** |
| upstream | 382.19 | 383.09 | 385.09 | **382.19** |

```
new / base            106.55 / 106.60 = 0.9995      <- no cost
control (new vs new)  107.24 / 108.35 = 1.010       <- the noise floor
upstream / new        382.19 / 106.55 = 3.59x faster than upstream
```

**The difference between base and new is 0.05 ms on 107, twenty times inside the
control's own spread.** Widening the scalar costs one extra `full` at `f32`
instead of at `bf16` per call, and nothing in this model calls it.

docs/perf/DTYPE_PERF.md §3 recorded `2.04×` faster than upstream (180.7 against
368.7); the shim's own `bfloat16` `S=128` has since come down to 106.6 in other
rounds, so `3.59×` is that figure re-measured on today's artefact and not a
change produced here — `base` reads the same.

---

## 5. Found while establishing the rule, and *not* fixed

The differential in §3 turns up three more disagreements with upstream. None of
them is the scalar rule, and the way that was decided is worth stating: **each
one still disagrees when the scalar is exactly representable**, so narrowing
cannot be what causes it.

| op | disagrees at | with an exactly-representable scalar? | what it looks like |
|---|---|---|---|
| `softplus` | every dtype **including `float64`**, with the *default* `beta=1, threshold=20` | yes, 7/7 | the `log1p(exp(x))` formula, computed in the storage dtype where upstream uses `opmath` — `bfloat16 softplus(-3)` is `0.048583984375` upstream and `0.0458984375` here. **Closed — §8.1** |
| `norm.ScalarOpt_dim` | `bfloat16` 8/10, `float16` 8/10, `float32` 1/10 — but **`p=2` agrees exactly in all four dtypes** | yes, 4/7 | the accumulation of `|x|^p`, and `powf` for fractional `p`; the same class as docs/kernels/KERNELS26.md §9.3 |
| `leaky_relu` | one case, and only on the **sign bit of a zero** with a negative slope | yes | `leaky_relu(0.0, -1.5)` — the scalar handling itself is correct |

Fixing `softplus` or `norm` means changing where those kernels accumulate, which
is a different change with its own digest question, and folding it into a
correctness commit about scalars is exactly the thing docs/training/TRAIN.md §5 says not
to do. They are recorded here rather than filed away; `softplus`'s is the larger
of the two, since it is wrong at `float64` where nothing about reduced precision
applies.

> **§8 did that separate change.** `softplus` is closed (§8.1); `norm` is
> re-measured and still open, with the size of it stated (§8.3). `leaky_relu`'s
> signed zero was not touched.

---

## 6. Open

* **`x *= 0.3` disagrees with upstream by one step at reduced precision**, for
  §2.2's reason: upstream's Python `*=` reaches `mul_.Tensor` and widens, this
  shim's parser reaches `mul_.Scalar` and upstream's `mul_.Scalar` narrows. The
  two are pinned as they are measured. Closing it means teaching the resolver
  upstream's "numbers as tensors" rule, which lives above `aten.rs`; the same
  gap `floor_divide.Scalar`'s doc comment already describes for a different op.
* ~~**The narrowing half of the family has no golden coverage.**
  `aten.add.Scalar` and `aten.sub.Scalar` are implemented in `aten.rs` but are
  not in `_aten_implemented()`, so `CASE_BUILDERS` has nowhere to hang a builder
  and the only check on them is `test_add_and_sub_scalar_still_narrow_and_did_
  not_follow_mul`. Sabotage F3 is caught by two smoke tests and **zero golden
  cases**.~~ **Closed — §8.2**, and writing the builder found a defect.
* **`softplus` and `norm`**, per §5. `softplus` closed — §8.1; `norm` still
  open — §8.3.

---

## 7. Sabotage

Six faults in `aten.rs`, each the most plausible wrong shape for what this round
changed, plus a re-run of docs/training/TRAIN.md §5's S4 — the one that started it. Every
one was applied to the source, **rebuilt**, and run through
`tools/golden/compare.py` and `pytests/test_shim.py`.

| # | fault | golden | smoke |
|---|---|---:|---|
| F1 | `mul.Scalar` narrows its scalar again (the reported defect, put back) | **6 FAIL** | 2 FAIL |
| F2 | the fix over-applied — `mul_.Scalar` widens too | **6 FAIL** | 1 FAIL |
| F3 | the fix applied to "the `.Scalar` family" — `add`/`sub` widen as well | **0** | 2 FAIL |
| F4 | the floor/trunc scalar narrows again, both spellings | **14 FAIL** | 1 FAIL |
| F5 | `pow` keeps the parser's `f64` scalar | **12 FAIL** | 1 FAIL |
| F6 | `floor_divide` back to `floor(a / b)` | **2 FAIL** | 1 FAIL |

**Two of these were caught by nothing on the first attempt, and the cases were
wrong, not the faults.**

* **F4 failed 5 golden cases and 0 smoke tests.** The shared
  `_SCALAR_RULE_VALUES` (`[3, 5, 7, 11, 13, 96, -3, -5]`) barely separate the two
  rules under a floor — upstream's `fmod`-based algorithm reaches the same
  integer from either divisor at small magnitudes, so `bf16(3) // 0.3` is 9 both
  ways and the pytest asserting it **could not fail**. Re-measured: at
  `bfloat16` the divisor `0.3` separates 4 of `[7, 14, 40, 43, 48, 61, 100, -49]`
  and at `float16` it separates *none* — the two dtypes need different scalars,
  which is why `0.1` is in the floor family's list and not in the shared one.
  With the measured values: 5 → **14** golden, 0 → 1 smoke.
* **F3 was caught by one pre-existing smoke test and not by the new one.** The
  new test asserted `add_`/`sub_`, which go through `arith_inplace_scalar`; F3
  changes `arith_scalar`, which owns the *out-of-place* `add.Scalar`/
  `sub.Scalar`. And its operands (`[1, 3, 5, 7]`) do not separate the rules for
  `add` at all — `0.3` and `0.30078125` are under half a `bfloat16` ULP apart at
  those magnitudes. Re-measured to `[-0.546875, -0.5, -0.9375, -7.96875,
  -0.421875, -1.1875, 0.0625, 0.1875]`, which separate 3/8 for `add` and 2/8 for
  `sub`/`rsub` in `bfloat16`.

**F3's golden column stays at 0, and that is a real gap, not a blind spot in the
fault.** `aten.add.Scalar` and `aten.sub.Scalar` are not in
`_aten_implemented()`, so `CASE_BUILDERS` has nowhere to register them; the two
smoke tests are the whole coverage. §6 carries it.

### 7.1 S4 re-run — the fault that started this, and it fails now

docs/training/TRAIN.md §5 recorded S4 ("dropout scales the input by `1/(1-p)` instead of
the mask") as **0 golden, 0 smoke**, and named `mul.Scalar`'s narrowing as the
reason: the two shapes were bit-identical here where upstream separates them.

`bootstrap.py` belongs to another change, so the fault was not applied to it.
Instead both shapes were computed **with the shim's own kernels**, on the
dropout cases' own 240 values and a fixed mask, and asked whether they are still
identical:

```
                          base           new         upstream
float32   p=0.25       0 of 168      0 of 168      0 of 168
float32   p=0.7        0 of 168      0 of 168      0 of 168
float16   p=0.25       0 of 168     52 of 168     52 of 168
float16   p=0.7        0 of 168     46 of 168     46 of 168
bfloat16  p=0.25       0 of 168     50 of 168     50 of 168
bfloat16  p=0.7        0 of 168     54 of 168     54 of 168
```

**The shim now separates the two shapes on the same elements upstream does, and
still not at all in `float32` — which is also upstream.** The dropout cases
compare bit-for-bit against upstream over these exact 240 values, so ~30% of the
survivors would differ and S4 is now caught. TRAIN.md §5's closing sentence —
"the dropout cases were widened anyway, so that they **will** catch S4 the day
`mul.Scalar` is fixed" — is discharged, and the widening was necessary: on the
all-ones input it replaced, the two shapes coincide by construction in every
dtype.

---

> Standing check (docs/verification/DOCWATCH.md) — the claims above that have a single
> ground truth:
> <!-- DOCWATCH: op-implemented aten.mul.Scalar -->
> <!-- DOCWATCH: op-implemented aten.mul_.Scalar -->
> <!-- DOCWATCH: op-implemented aten.floor_divide.Scalar -->
> <!-- DOCWATCH: op-implemented aten.div.Scalar_mode -->
> <!-- DOCWATCH: op-implemented aten.pow.Tensor_Scalar -->
> <!-- DOCWATCH: op-implemented aten.pow.Scalar -->
> <!-- DOCWATCH: symbol-in-file tools/golden/cases.py _scalar_rule_cases present -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_mul_scalar_reads_the_scalar_at_opmath_not_narrowed present -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_add_and_sub_scalar_still_narrow_and_did_not_follow_mul present -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_pow_narrows_its_scalar_where_mul_widens_it present -->
>
> §6's second bullet said `aten.add.Scalar` and `aten.sub.Scalar` were
> implemented in `aten.rs` but absent from `_aten_implemented()`, which is why
> golden had no builder for them. **§8 closed it.** The markers are inverted
> rather than deleted, so that the closure cannot silently come undone:
> <!-- DOCWATCH: op-implemented aten.add.Scalar -->
> <!-- DOCWATCH: op-implemented aten.sub.Scalar -->
> <!-- DOCWATCH: symbol-in-file tools/golden/cases.py _add_sub_scalar_cases present -->

---

## 8. §5 and §6, closed as their own round

§5 and §6 both end with the same sentence in different words: *this is a change
about where a kernel accumulates, it has its own digest question, and folding it
into a scalar-rule commit would make a digest move look like a regression.* This
section is that separate round. **All nine prefill digests are unchanged** (§8.5),
which is the control the whole deferral was for.

### 8.1 `softplus` — three defects, not one

§5 called it "the `log1p(exp(x))` formula, computed in the storage dtype". The
first half of that was generous: the kernel was not computing `log1p(exp(x))` at
all.

Upstream's `softplus_kernel` is one expression:

```cpp
(a * beta) > threshold ? a : std::log1p(std::exp(a * beta)) / beta
```

The kernel here computed `max(y,0) + log(1 + exp(-|y|))` in candle tensor ops.
Three separate things are wrong with that, and each was measured on 2.13.0:

| | what | evidence |
|---|---|---|
| 1 | **the rewrite is not the same function in floating point** | upstream agrees with `math.log1p(math.exp(x))` on 10 of 10 `float64` probes and with the split on 6 |
| 2 | **and not the same function at all at the edges** | with a threshold that does not fire, upstream's `exp` overflows: `softplus(800.0, 1, 1e9)` is `inf` upstream and was `800.0` here. **No tolerance is involved in that difference** |
| 3 | **it ran in the storage dtype where upstream runs in `opmath`** | `c10::BFloat16`'s operators promote to `float`, so the whole expression is `float32` and narrowed once. `bfloat16 softplus(-3)` was `0.0458984375` against upstream's `0.048583984375` — a 6% error |

The kernel is now a scalar walk: `beta` and `threshold` narrowed to the
tensor's dtype (`beta_.to<scalar_t>()`), then `log1p(exp(y))/beta` at `f64` for
`float64` and at `f32` for everything else, then one narrowing.

**Agreement after**: over 88 measured rows — four dtypes × main values, edge
values (`±inf`, `NaN`, `±0.0`, `1e30`, `710`), and `beta`/`threshold` variants
including `beta=0`, `beta=-1`, `threshold=1e9`, `threshold=-1` — **87 are
bit-identical and one is not.**

The one is `float32`, and it is **upstream's own irreproducibility**, not a
defect here. `cpu_kernel_vec` runs a Sleef-vectorised body over full blocks and
a scalar tail over the rest, and they disagree:

```
softplus(-3.0), float32:   n < 8   0x1.8e070e0p-5      (the scalar path)
                           n >= 8  0x1.8e07100p-5      (Sleef)
      measured at n = 1, 2, 3, 4, 7, 8, 16, 17, 32, 64, 100
```

This kernel is a scalar walk, so it answers the scalar path at every length. At
`n < 8` it is bit-identical to upstream; at `n >= 8` it differs by one ULP on
that one input and on nothing else. The same class docs/training/LOSS.md §5.4 records for
`_log_softmax`, and the golden cases hold the size of it rather than its
absence.

`float64`, `float16` and `bfloat16` are stable across length and are pinned
bit-exactly.

**Cases**: 15 before, all of which passed both kernels; 16 more, for 31. The old ones
could not fail for two reasons and the new ones fix both — their inputs
(`[-5,-1,0,1,5]`) are where the two formulas agree to the last bit, and their
comparator was the per-dtype tolerance, which at `bfloat16` is `6e-2` against an
effect of 0.0027. Nine of the new cases use `_bit_exact`.

### 8.2 `add.Scalar` / `sub.Scalar` — promoted, and the builder found a defect

§6's second bullet: both ops had a kernel, neither was in `_aten_implemented()`,
so `CASE_BUILDERS` had nowhere to hang a builder and sabotage F3 — *the
narrowing half of the family widens instead* — was caught by two smoke tests and
**zero** golden cases. Both are promoted, `IMPLEMENTED_AWAITING_GOLDEN` is down
to seven entries, and there are now 84 cases across the two keys — 42 each. **F3
fails 7 of them.**

Writing the builder found something no case had been in a position to see:
**nothing had ever passed either op a non-unit `alpha`.**

`alpha` is narrowed to the tensor's dtype **separately from `other`**, and their
product is narrowed again — it is not `narrow(other * alpha)`. Measured over 300
random `(other, alpha)` pairs:

| model | `bfloat16` | `float16` |
|---|---:|---:|
| `narrow(narrow(other) * narrow(alpha))` | **300/300** | **400/400** |
| `narrow(other * alpha)` — what this shim did | 202/300 | 260/400 |

`bfloat16([0.0]) + 0.3` with `alpha=0.3` is `0x1.72p-4` upstream and was
`0x1.70p-4`. Fixed; the differential over 150 random pairs × 7 values goes:

| | before | after |
|---|---:|---:|
| `bfloat16` | 53/150 rows | **0/150** |
| `float16` | 56/150 rows | **0/150** |
| `float32` | 132/150 (add) / 126/150 (sub) | 120/150 / 111/150 |
| `float64` | 106/150 / 105/150 | unchanged |

`alpha = 1` is unaffected at every dtype, which is why no digest moves:
`narrow(1.0)` is `1.0` and `narrow(narrow(o) * 1.0)` is `narrow(o)`.

**What the `float32`/`float64` residual is, and why it is left.** Upstream's
`self + alpha * other` is compiled to a **fused multiply-add** on this host. At
`float64`, `fma(alpha, other, self)` reproduces upstream on **1050/1050**
elements over 150 random pairs; the unfused expression on 862/1050. That is a
property of how the wheel was compiled (`-ffp-contract`), like §8.1's Sleef tail.

Its size is worth stating precisely because one number is uncomfortable:

| | worst ULP distance | worst **relative** error |
|---|---:|---:|
| `float64` | 310 | 5.5e-14 |
| `float32` | 333 | **3.2e-05** |

Both worst cases are the same cancellation: `7.0 + 2.430806… × 2.881715…` is
`-0.00489`, four decades below its operands, so a last-bit disagreement in the
product is a 3e-5 disagreement in the sum. **3.2e-05 is larger than
`dtypes.py`'s `float32` rtol of 1e-5**, which is the same shape docs/training/LOSS.md
§5.4 records — so there is deliberately no `float32` case at those operands, and
the `float64` case *is* at them, where the same cancellation is inside 1e-9.

Closing it needs a per-element walk using `mul_add`, applied at
`float32`/`float64` **only** — the reduced floats are exact *without* it,
because their product is narrowed before the add, so an unconditional FMA would
break them. That is a dtype-conditional rewrite rather than a line, and it is
left rather than half-done.

### 8.3 `norm.ScalarOpt_dim` — 29 of 120 rows became 1

§5 measured this as `bfloat16` 8/10, `float16` 8/10, `float32` 1/10, with `p=2`
exact everywhere. Re-measured before the change over a random 3×4 at ten `p`
values, three `dim` lists and four dtypes: **29 of 120 rows disagreed.**

Upstream's `norm_kernel_tensor_iterator_impl` dispatches
`norm_kernel_cpu_impl<scalar_t, acc_t>` with **`acc_t = float` for `Half` and
`BFloat16`** and `acc_t = scalar_t` otherwise, so the running `|x|^p` is kept in
`float` for the reduced dtypes and narrowed once. Reducing with candle keeps
every partial sum in the storage dtype.

The kernel is now a walk in `acc_t`, with each of upstream's six reduction ops
transcribed rather than expressed through another:

```text
  p = 0      acc + (data == 0 ? 0 : 1)      project acc
  p = 1      acc + |data|                   project acc
  p = 2      acc + data*data                project sqrt(acc)
  p = +inf   max(acc, |data|)               project acc
  p = -inf   min(acc, |data|)               project acc
  otherwise  acc + pow(|data|, p)           project pow(acc, 1/p)
```

**After: 1 of 120.** The residual is `float64`, `p = 2`, four-wide rows, one
ULP — and it is **the same residual the previous kernel had**, neither
introduced nor closed here. Upstream's `p = 2` arm sums **pairwise**: on
`[0.779296, 1.757861, -2.435259, -1.179592]`, `sqrt((a²+b²)+(c²+d²))` is
upstream's `0x1.a8e67779e8296p+1` and the serial sum is `…97p+1`. Matching it
means reproducing `binary_kernel_reduce_vec`'s lane count and unrolling, which
is again a compile-time property rather than an operator property. Not
attempted; cased and watched.

**Cases**: 120 new bit-exact ones across four dtypes × ten `p` × three `dim`
lists, plus the watched `p=2` residual, for 536. The 415 existing cases could not
see any of this — their data is `[3, -4, 0, 1, -1, 2]`, integers whose every partial sum
of squares and absolute values is exact in all four dtypes.

### 8.4 Sabotage — including three faults that cannot fail

Each fault injected into `aten.rs`, rebuilt, and counted. Never read from a
green run.

| fault | golden | smoke |
|---|---:|---:|
| **S7** softplus: the whole previous kernel — split, `log(1+x)`, storage dtype | **9** | 0 |
| S1 softplus: the stable split instead of `log1p(exp(y))` | 4 | 0 |
| S2 softplus: `log(1 + x)` instead of `log1p(x)` | 5 | 0 |
| S3 softplus: computed in `f64` for every dtype — *wider* than `opmath` | 2 | 0 |
| S4 softplus: `beta`/`threshold` not narrowed to the tensor dtype first | **0** | 0 |
| S5 softplus: every interior step narrowed to the storage dtype, `log1p` kept | **0** | 0 |
| N1 norm: accumulate in the storage dtype — §5's recorded defect | **14** | 0 |
| N2 norm: `pow` at `f64` instead of `powf` at `acc_t` | **0** | 0 |
| N3 norm: `p = 2` via `powf(2)`/`powf(0.5)` instead of square/`sqrt` | **0** | 0 |
| A1 add/sub: `narrow(other * alpha)` instead of narrowing each | 4 | 0 |
| A2 add/sub: widen the scalar instead of narrowing — **this is F3** | **7** | 2 |
| B1 bool: revert to the single blanket refusal message (§8.6) | 0 | 1 |

**S7 is the one that matters**: it is the kernel this section replaced, and the
new cases fail on it with exactly the reported numbers — `bfloat16` `0.0458984375`
against upstream's `0.048583984375`. The case set would have caught the defect
docs/numerics/SCALAR.md §5 had to find with a separate differential.

**A2 is the discharge of §6's second bullet.** F3 previously failed two smoke
tests and no golden case; it now fails seven, at `bfloat16` and `float16`, on
both keys.

**Three faults could not fail, and each is a real statement rather than a gap
to fill.**

* **S4 and S5 — the reduced floats hide interior precision.** Narrowing the
  result to 8 or 11 mantissa bits absorbs almost any change in how the interior
  steps round, provided the *formula* is right. S5 narrows every step of
  `log1p(exp(y))/beta` to `bfloat16` and the answer does not move, because
  `log1p` never forms the `1 + tiny` sum that the old kernel's rounding
  destroyed. What separated the old kernel at `bfloat16` was the **formula**
  (S1, S2, S7), not the precision on its own. S4 is a further step: narrowing
  `beta` to `bfloat16` before use is genuinely a different computation
  (`bf16(0.1)` is `0.10009765625`, `0.1f32` is `0.1`) and a 100-point search
  over `(beta, x)` found **no separating pair at `bfloat16` or `float16`**. It
  is observable at `float32` — and there the explicit narrowing is redundant
  with the `as f32` in the walk, so removing the line changes nothing. The line
  is kept because it is upstream's spelling, and it is recorded here that
  nothing can fail on it.
* **N2 and N3 — `powf` versus `pow`, and `pow(·,2)` versus squaring.** Both are
  correct-rounding questions that the six chosen values do not separate:
  `powf(x, 2)` is exact squaring and `powf(x, 0.5)` agrees with `sqrt` on them.
  The arms are still written as upstream writes them, because a value that
  *does* separate them exists in general even though these do not.

The honest summary of both groups: **at `bfloat16` and `float16` this suite
separates which function a kernel computes, and does not separate at what
precision it computes the interior of that function.** The precision half is
separated at `float32` and `float64`, where nothing absorbs it.

### 8.5 The digests, which is what the deferral was for

Re-measured on the final artefact, at every length docs/numerics/SEQLEN.md §1.3 and
docs/training/TRAIN.md §6 record:

| S | `float32` | | `bfloat16` | |
|---:|---|:--:|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | ✅ | `8ef1550ea33c4f3d…` | ✅ |
| 32 | `331668f36da02f21…` | ✅ | `b81325c83a0a3d15…` | ✅ |
| 128 | `00159a9dbd308eda…` | ✅ | `7ff8e9334449b147…` | ✅ |
| 512 | `07c2797dabc4552e…` | ✅ | `9ab1e82f01378e38…` | ✅ |
| 1024 | `eda1e173727bb7f5…` | ✅ | — | |

**None moved, and the reason is checkable rather than lucky.** A
`TorchDispatchMode` over this prefill (docs/numerics/SCALAR.md §4.2's log) shows the
forward calls `add.Tensor`, `mul.Tensor`, `pow.Tensor_Scalar` and SDPA — it
calls neither `softplus` (mamba's) nor `norm.ScalarOpt_dim` (`weight_norm`'s, at
construction), and every `add` it makes has the default `alpha = 1`, which is
exactly the value §8.2's fix leaves alone.

### 8.6 The refusal message §5 did not look at

While measuring `add.Scalar(bool, ·)` for §8.2's builder, the bool row came back
`[4, 3, 4]` — upstream **computes** — and `sub.Scalar(bool, ·)` came back a
refusal. One `arith_tag` message was serving both, and it said
*"torch.bool operands are logical, not arithmetic, in torch"*, which is true of
neither. docs/kernels/TAIL.md §7 has the twelve-cell re-measurement and the fix; the
refusals are unchanged and only their stated reasons moved.

### 8.7 Gates

| gate | before this section | after |
|---|---|---|
| `pytests/run.sh` | 302 ok, DOCWATCH 159/159 | **304 ok, DOCWATCH 164/164** |
| `tools/golden/compare.py` | 7447/7447, ops=166 | **7685/7685, ops=168** |
| `compare.py --self-test` | 19 × 11, 0 problems | unchanged |
| `verify_schemas.py` | 4475/4475 | **4479/4479** |
| sweep26 / sweeptrain | 26/26 | **26/26 / 26/26** |
| prefill digests | 9 recorded | **all 9 unchanged** |

`ops` moves 166 → 168 because `add.Scalar` and `sub.Scalar` are now advertised;
`verify_schemas` moves 4475 → 4479 for the same reason.

The "after" column is the whole session's, not this section's alone: two other
rounds landed against the same tree — `index_put_(accumulate=True)`
(docs/kernels/VIEWS.md §7, +16 cases and +1 test) and the watched real-width
`_log_softmax` divergence (docs/training/LOSS.md §5.4.1, +1 case and a new comparator).
This section's own contribution is +237 cases and +1 test.
