# The kernels that stopped six architectures

docs/architectures/ARCH26.md widened the sweep from 20 architectures to 26 and found the six new ones stopped
**not by a missing name but by a missing kernel** — `aten.rs`, `tensor.rs`, `dtype.rs` and
`flash.rs` were all out of territory that round, so it named the walls and left them. This document
is that round's follow-up: the kernels themselves.

**Written incrementally, one kernel at a time**, for the reason ARCH26.md gives for the same
practice — a kill mid-task should leave the finished parts behind rather than nothing.

## 0. Method, and what "landed" means here

The acceptance test is **architectures forwarding**, not kernels passing their own cases. So the
sweep is re-run after every kernel, and each section below records the count and, when the count
does not move, **which wall replaced the one that was removed**. A kernel that moves the first wall
forward without moving the count is progress that has to be visible as progress, not folded into a
batch total at the end.

The sweep is `/tmp/k26/sweep26.py` — the 20 of docs/architectures/ARCH20.md (`/tmp/arch7/sweep.py`'s list and
toy config, unchanged) plus the six toy configs of ARCH26.md §1-5, transcribed from that
document's own scratch scripts. It is run twice: once with no `PYTHONPATH` (upstream, the oracle)
and once with the vendored tree (the shim).

**Upstream is 26/26**, measured first, before anything here was written:

```
llama gpt2 qwen2 mistral gemma gpt_neox opt mpt starcoder2 stablelm olmo phi mixtral
bert bloom cohere falcon gpt_bigcode mamba persimmon
deberta deberta_v2 vits zoedepth sew_d sam3_video          TOTAL 26/26
```

One caveat carried forward from ARCH26.md §5, and it is the architecture's property rather than a
shortcut: **`sam3_video`'s top-level `forward` does not take a tensor.** It takes a stateful
`Sam3VideoInferenceSession` advanced one video frame at a time, and `transformers` ships no toy
`ModelTester` for it (only `@slow` integration tests against the real checkpoint). Its
`tracker_model` is session-shaped too — `m.tracker_model(pixel_values)` raises
`'Tensor' object has no attribute 'get_obj_num'` **on upstream**, measured, which is how that was
established rather than assumed. So "sam3_video forwards" here means its `detector_model` — the
SAM3 detector, a plain `pixel_values`/`input_ids`/`attention_mask` in, `pred_logits` out call —
forwards, which is the deepest tensor-shaped forward this architecture has.

### The baseline, every gate, before any edit

```
pytests/run.sh                261 ok, 0 FAIL                        exit 0
tools/golden/compare.py       4290/4290, ops=139, pending=1          exit 0
compare.py --self-test        13 comparators x 11 fault modes        exit 0
verify_schemas.py             4353/4353                              exit 0
sweep26 (shim)                20/26                                  exit 0
```

and the six failures, each exactly the wall ARCH26.md recorded:

| architecture | first wall at baseline |
|---|---|
| `deberta` | `torch.sqrt(...)` — no overload table entry |
| `deberta_v2` | `torch.sqrt(...)` — no overload table entry |
| `vits` | `TensorBase.set_: expected a torch.UntypedStorage, got Parameter` |
| `zoedepth` | `aten.convolution.default: only 1-D convolution ... got 4-D` |
| `sew_d` | `TensorBase(...) takes an existing tensor to re-wrap` (legacy `torch.Tensor(int)`) |
| `sam3_video` | `TensorBase.__mod__` |

---

## 1. `aten.sqrt.default`

**Sweep after: 20/26 (unchanged). What moved: `deberta` and `deberta_v2` both advanced from
`torch.sqrt` to `TensorBase.repeat`.**

That is the honest way to report it. `sqrt` was the highest-value item on the list because it was
the first wall for two architectures, but it was not the *only* wall for either, and §6/§8 of
ARCH26.md said so in advance — `aten.repeat.default` is listed there for both. The count moves when
the last wall for an architecture falls, and attributing the count to whichever kernel happened to
be last would be the wrong story.

### 1.1 Why it was a kernel and not a spelling

ARCH26.md §1.2 established this and it was re-checked rather than inherited:
`_dispatch_has_kernel_for_dispatch_key('aten::sqrt', 'CompositeImplicitAutograd')` is `False`, and a
`TorchDispatchMode` trace of `torch.sqrt(x)` fires exactly one op. There was nothing to wire a name
to.

The asymmetry — `rsqrt` present since RMSNorm, `sqrt` absent — is why two architectures stopped
before any weight multiplied. `DebertaLayerNorm.forward` computes its own layer norm by hand,
`(h - mean) / torch.sqrt(var + eps)`, instead of calling `nn.LayerNorm`; `deberta_v2` uses a real
`nn.LayerNorm` but its `scaled_size_sqrt` computes an attention temperature through `torch.sqrt`
**unconditionally**, before the `if self.relative_attention:` branch.

### 1.2 What upstream actually does, measured

Run on upstream 2.13.0 with no shim on the path, before writing anything:

```
sqrt dtype promotion            float64->float64  float32->float32  float16->float16  bfloat16->bfloat16
                                int64->float32    int32->float32    int16->float32
                                uint8->float32    bool->float32

sqrt sign and domain, float32   sqrt(-1.0)  = nan        bits 7fc00000
                                sqrt(-0.0)  = -0.0       bits 80000000     <-- sign of zero survives
                                sqrt( 0.0)  =  0.0       bits 00000000
                                sqrt( inf)  =  inf       bits 7f800000
                                sqrt(-inf)  = nan        bits 7fc00000     <-- NOT -inf
                                sqrt( nan)  = nan
                                sqrt( 2.0)  = 1.4142135381698608
```

So the rule is `unary_float`'s — a float keeps its own width, anything else becomes the default
float — and the domain is IEEE's, with **no refusal anywhere**: a negative input answers NaN rather
than raising. Both halves are asserted, because "raises on a negative" is the plausible wrong guess
and nothing in either DeBERTa's forward has a negative variance to reveal it.

**`sqrt(-0.0) = -0.0` is the property this kernel exists to get right and the one no value
comparison can see**, because `-0.0 == 0.0` is true in Python and `abs(-0.0 - 0.0)` is `0.0`. It is
checked on the sign bit, through `math.copysign`, in both the golden case
(`_signed_zero_check`) and the pytest.

### 1.3 What it is written on, and what it is deliberately not

`Unary::Sqrt` in `aten.rs`'s existing `unary_float` family, which is candle's own `Tensor::sqrt`.
Two alternatives were considered and both are wrong for measured reasons:

- **`pow(x, 0.5)` as a `bootstrap.py` composite.** ARCH26.md §1.2 rejected it as inventing a
  computation path; there is a second, harder reason. candle's `Tensor::pow` is
  `exp(exponent * log(base))` — `aten.rs`'s own `pow_result_tag` comment records this and exists
  because of it — and that expression answers **NaN for `sqrt(+inf)`**, where upstream answers
  `+inf`. The golden case `sqrt(dtype=..., shape=(1,)) [+inf -> +inf]` fails for exactly that.
- **`x.powf(0.5)`.** Gets the values right, but is a second rounding on `float16`/`bfloat16`:
  candle's `powf` widens, exponentiates and narrows, where `sqrt` is one correctly-rounded
  operation. The four reduced-float rows of the golden builder are what would catch it.

`sqrt` joins the family rather than getting its own function the way `rsqrt` has one, because it
*is* one candle call — `rsqrt` is separate only because it is two (`sqrt` then `recip`).

### 1.4 Reachable by name, not only by dispatch key

The brief's standing warning, and ARCH20.md §9's whole inventory: **golden compares by dispatch
key and cannot see a kernel with no way in.** `deberta`'s wall was the *spelling* `torch.sqrt(...)`,
so a kernel with no `overloads.json` entry would have left that wall exactly where it was while
every golden case passed.

| table | entry | what it opens |
|---|---|---|
| `overloads.json` | `aten::sqrt.out`, `aten::sqrt` | `torch.sqrt(x)` |
| `methods.json` | `aten::sqrt` | `x.sqrt()` |

`sqrt.out` is in the table with **no kernel behind it**, exactly as `rsqrt.out` already was, so that
`torch.sqrt(x, out=y)` refuses by the right name rather than falling through to "no table entry for
this op". Asserted as a refusal, not left implicit.

Three routes are tested, and they fail independently:

1. `_C._aten_dispatch("aten.sqrt.default", ...)` — the kernel, in-process against the bare artefact
   (`test_sqrt_is_a_leaf_kernel_with_ieee_domain_and_sign`).
2. `_C._shim_overloads` / `_C._shim_methods` — the two tables, asserted by content
   (`test_sqrt_is_reachable_by_name_not_only_by_dispatch_key`).
3. `torch.sqrt(x)` and `x.sqrt()` through a **real `import torch`** in a subprocess against the
   vendored tree (`test_kernels26_road_through_the_vendored_tree`), including `deberta_v2`'s
   `scaled_size_sqrt` and `DebertaLayerNorm.forward` transcribed verbatim.

### 1.5 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 261 | **264** (+3: two `sqrt` tests, one road test) |
| `compare.py` | 4290/4290, ops=139 | **4339/4339, ops=140** (+49 cases) |
| `compare.py --self-test` | 13 comparators | **14** (+`_signed_zero_check`), 0 problems |
| `verify_schemas.py` | 4353/4353 | **4359/4359** (+6: `sqrt` in both tables, `.out` included) |
| sweep26 (shim) | 20/26 | **20/26** |

Two existing tests carried hardcoded counts and both had to move; both are counts of *upstream's*
answer, not of this shim's, which is why they are worth having:

- `test_core_ops_and_op_tags_agree`: **84 → 85**. `torch.ops.aten.sqrt.default.tags` is
  `['core', 'pointwise', 'pt2_compliant_tag']`, read off the op rather than inferred from `rsqrt`.
- `test_schema_text_survives_the_round_trip_through_the_transcribed_tables`: **221 → 223**. **+2,
  not +3** — `sqrt` went into both tables and `overloads.json`'s `aten::sqrt|default` and
  `methods.json`'s are one identity; the second is `aten::sqrt|out`.

### 1.6 Sabotage check

Three faults, each aimed at a different claim, each rebuilt and re-run in full. The rule this
round is enforcing: **a check that cannot fail is not a check.**

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| S1 | `sqrt(x)` then `affine(1.0, 0.0)` — adds `+0.0`, which turns `-0.0` into `+0.0` and changes **nothing else** | **4335/4339, 4 failed** | 2 FAIL | the sign-of-zero claim is load-bearing. Exactly the 4 signed-zero cases (one per float dtype) fail, and every other `sqrt` case still passes — a value comparison alone would have seen none of it |
| S2 | always promote to the default float, so `float16` in gives `float32` out | **4306/4339, 33 failed** | 1 FAIL | the dtype rule is checked at every width, not just at `float32` |
| S3 | **kernel left intact and correct; the `sqrt` entry removed from `overloads.json` and `methods.json`** | **4339/4339, 0 failed, exit 0** | 3 FAIL | the point of §1.4. Golden is *completely green* on a kernel no model can reach, and re-running the sweep puts `deberta` straight back on its original wall: `torch.sqrt(...) -- overload resolution has no table entry for this op`. The three tests that do fail are the two table tests and the vendored-tree road |

S3 is the one worth keeping. It is the shape of gap that hid four times in this project's history,
and it is invisible to the harness that exists to catch gaps.

A fourth fault was tried first and is recorded because its *failure mode* is informative: making
`sqrt` keep the input dtype outright (`int64` in, `int64` out) does not produce a wrong answer, it
produces `PanicException: not yet implemented: no unary function for i64` from candle. A wrong
promotion here cannot be silent, which is a smaller claim than the tests make but worth knowing.

---

## 2. `aten.repeat.default`

**Sweep after: 22/26 (+2). `deberta` and `deberta_v2` both forward.**

This is the kernel that moved the count, and `sqrt` is the one that made it reachable — neither
alone gets either architecture through. §8 of ARCH26.md listed `repeat` for four of the six
(`deberta`, `deberta_v2`, `sew_d`, `sam3_video`), which is why it was worth taking even though it
was the first wall for none of them.

### 2.1 Tiling is not broadcasting

`expand` produces a view whose strides are zero; `repeat` materialises. `[1,2,3].repeat(2, 3)` is
`(2, 9)` and not `(2, 3, 3)` — **the last repeat multiplies the existing dimension and the earlier
ones are new leading dimensions.** Measured, along with the rest:

```
[1,2,3].repeat(2)          -> (6,)      [1,2,3,1,2,3]
[1,2,3].repeat(2,3)        -> (2, 9)
[[1,2],[3,4]].repeat(2,3)  -> (4, 6)
tensor(5).repeat(3)        -> (3,)      0-D gains a dimension
tensor(5).repeat([])       -> ()        an empty repeat list is legal on a 0-D
[1,2,3].repeat(0)          -> (0,)      NOT a no-op
m.repeat([2])              raises: Number of dimensions of repeat dims can not be smaller
                                    than number of dimensions of tensor
m.repeat([2,-1])           raises: Trying to create tensor with negative dimension -2: [4, -2]
dtype                      passed through unchanged, bool included -- no promotion
result                     always a fresh copy; writing into it does not reach the source
```

The negative-repeat message reports the **product** (`2 * -1` on a size-2 axis) and the whole
computed output shape, not the repeat the caller passed. Transcribed rather than paraphrased,
because the number in the message is not the number that was handed in.

### 2.2 candle has a `Tensor::repeat` and it is not the one called

Its entire body is `for (idx, &repeat) in repeats.iter().enumerate() { if repeat > 1 { cat } }`,
and that loop disagrees with upstream in three places:

| | candle | upstream |
|---|---|---|
| a repeat of `0` | skipped — `if repeat > 1` — so `[1,2,3].repeat(0)` is `(3,)` | `(0,)`, a genuinely empty dimension |
| `len(repeats) < rank` | takes the `self.clone()` branch and concatenates along axes that no longer line up | raises |
| every repeat `1` | returns `self.clone()`, and **a candle clone is an `Arc` clone** | materialises |

The third is the dangerous one and it is not visible in any result. `x.repeat(1, 1)` would share
storage with `x`, so `x.repeat(1,1).fill_(0)` would zero `x` — the exact defect docs/kernels/VIEWS.md §6
records for `_to_copy`, wearing a new hat: correct values, corrupted input, every golden case green
because they all read the *result*.

So the tiling is written out in `repeat_default`: resolve the output shape first (which is where
both refusals live and where `0` is honoured), tile with `Tensor::cat`, and end with an explicit
`.copy()` that is load-bearing rather than defensive.

### 2.3 Reachable by name

**`repeat` is a member and not a free function.** `hasattr(torch, "repeat")` is `False` on 2.13.0 —
there is `Tensor.repeat` and the unrelated `torch.repeat_interleave`, and nothing else. So the entry
goes in `methods.json` only, and `test_sqrt_is_reachable_by_name_not_only_by_dispatch_key` asserts
`"repeat" not in _C._shim_overloads` — inventing a `torch.repeat` would be adding a name upstream
does not have.

Both call shapes are exercised through a real `import torch`: `x.repeat(2, 3)` (varargs) and
`x.repeat([2, 3])` (list).

### 2.4 The numeric comparison ARCH26.md could not make

ARCH26.md §1.4 checked `deberta`/`deberta_v2`'s `@torch.jit.script` helpers standalone under
`PYTORCH_JIT=1` and `PYTORCH_JIT=0` and found them identical, but recorded honestly that this was
corroborating evidence rather than the acceptance-grade check, and that **"the real answer to 'do
the numbers drift' is: cannot be measured until `aten.sqrt.default` lands."**

It lands here, so it was measured. Same toy config, `torch.manual_seed(0)`, a 6-token forward, run
once on upstream and once through the shim, dumping the full `state_dict` as well as the output —
because "the initialisers agree" is itself a claim and not something `manual_seed` should be
trusted for:

| | `deberta` | `deberta_v2` |
|---|---|---|
| weight tensors compared | 37 | 45 |
| weight tensors differing | **0** | **0** |
| max weight abs diff | **0.0** (bit-identical) | **0.0** (bit-identical) |
| output elements | 192 | 192 |
| bit-identical output elements | 46 | 57 |
| max abs diff | 4.77e-07 | 3.58e-07 |
| **max relative diff** | **1.61e-07** | **1.21e-07** |

Both are at float32 epsilon (1.19e-07) — accumulation-order noise, not drift. **There is no
TorchScript-versus-eager divergence in either architecture**, which is the question ARCH26.md left
open, now answered on a real forward rather than on standalone helper functions.

The weight half being *bit-identical* is the stronger half of that result: it means the shim's RNG
reproduces upstream's initialisation exactly for both architectures, so the output difference is
attributable to the forward alone and to nothing upstream of it.

### 2.5 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 264 | **265** (+1) |
| `compare.py` | 4339/4339, ops=140 | **4481/4481, ops=141** (+142 cases) |
| `verify_schemas.py` | 4359/4359 | **4363/4363** (+4) |
| sweep26 (shim) | 20/26 | **22/26** |
| `test_core_ops_and_op_tags_agree` | 85 | **86** — `repeat`'s tags are `['core', 'pt2_compliant_tag']`, core but *not* `pointwise`, read off the op |
| schema identities | 223 | **224** (+1: `methods.json` only, no `.out`) |

### 2.6 Sabotage check

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| R1 | call candle's `Tensor::repeat` — the naive implementation | **4451/4481, 30 failed** | 3 FAIL | the zero-repeat rows, the rank refusal and the aliasing entry are each load-bearing |
| R2 | **everything correct except the final `.copy()`** | **4481/4481, 0 failed, exit 0** | 2 FAIL | the same lesson as §1.6's S3 from the other direction. Golden is *completely green* while `x.repeat(1)` silently zeroes its own input; only `test_which_ops_share_storage_with_their_input_and_which_do_not` and the vendored-tree road can see it |

R2 is why the `.copy()` has a comment saying it is load-bearing. Without the aliasing-table entry
added in the same change, dropping it would have been a free 4481/4481.

---

## 3. `aten.remainder.Scalar` and `aten.remainder.Tensor`

**Sweep after: 22/26 (unchanged). What moved: `sam3_video` advanced from
`TensorBase.__mod__` to `aten.div.Scalar_mode` — two lines later, in the same
`Sam3ViTRotaryEmbedding.__init__`.**

`modeling_sam3.py:428` builds a rotary position grid as `flattened_indices % end_x` for the x axis
and `torch.div(flattened_indices, end_x, rounding_mode="floor")` for the y axis. The first of those
is this kernel; the second is `div.Scalar_mode`, which §6 leaves named and unwritten.

### 3.1 The sign of the divisor, which is the whole op

**`remainder` follows the sign of the divisor; `fmod` follows the sign of the dividend.** Measured
on upstream 2.13.0, and this table is the acceptance criterion:

```
        remainder   fmod
 7,  3      1        1     agree
 7, -3     -2        1     DISAGREE
-7,  3      2       -1     DISAGREE
-7, -3     -1       -1     agree
```

They agree in exactly half the quadrants. **A case set built from positive operands, or one that
varies only the dividend's sign, passes an `fmod` implementation completely.** Both the golden
builder and the pytest run all four quadrants on both overloads and on both categories, and the
pytest asserts the *disagreement* explicitly — for opposite-sign pairs it requires
`remainder(a, b) != math.fmod(a, b)`, computed in Python rather than taken from the shim.

The implementation is upstream's own correction, transcribed:

```
mod = fmod(a, b);
if (mod != 0) && ((b < 0) != (mod < 0)) { mod += b }
```

### 3.2 Three corners that fall out of that guard, all measured

- **`remainder(-0.0, 3.0)` is `-0.0`.** `fmod(-0.0, 3.0)` is `-0.0`, and `-0.0 != 0.0` is *false*,
  so the correction never fires and the negative zero survives. **Python's own `-0.0 % 3.0` is
  `+0.0`** — so "spell it the way Python spells it" is wrong here, and wrong invisibly, because
  `-0.0 == 0.0`. Checked on the sign bit by `_signed_zero_check`, the comparator §1 added.
- **Division by zero splits by category.** A float divisor of `0.0` gives NaN with no raise; an
  integral one raises `RuntimeError('ZeroDivisionError')` — upstream's message is that bare string.
  Same op, same shape of input, two different kinds of answer.
- **Infinite divisors.** `remainder(5.0, inf)` is `5.0`, `remainder(5.0, -inf)` is `-inf`, and
  `remainder(-5.0, inf)` is `inf`. All three are the correction firing on a finite `fmod`; none of
  them is special-cased.

One divergence from C rather than from `fmod`: **`i64::MIN % -1` panics in Rust** (the quotient
overflows) where upstream answers `0`. `wrapping_rem` is used for exactly that pair, and a golden
case plus a pytest pin it — a panic here would cross the FFI boundary as a `PanicException`, which
is not a refusal.

### 3.3 Dtype, and the one deliberate gap

`remainder.Tensor` follows `torch.promote_types` **exactly** — checked cell by cell over the eight
storable numeric dtypes rather than assumed from `mul`, with no disagreements. `remainder.Scalar`
follows the wrapped-number rule: an int scalar never widens a tensor of any category, a float scalar
floats an integral one.

**The scalar is narrowed into the result dtype before the arithmetic**, and that is observable:
`remainder(uint8(200), -3)` is `200`, because `-3` becomes `253` in `uint8` and `200 % 253` is
`200`. Building the scalar as a 0-D tensor at the result storage reproduces the narrowing instead of
restating it.

**Bool is refused on both overloads, but only one of those refusals is upstream's.**

| | upstream | here |
|---|---|---|
| `remainder.Tensor(bool, bool)` | raises `NotImplementedError: "remainder_cpu" not implemented for 'Bool'` | **same message, verbatim** |
| `remainder.Scalar(bool, 2)` | computes, `int64` | refuses |
| `remainder.Scalar(bool, 2.0)` | computes, `float32` | refuses |
| `remainder.Scalar(bool, True)` | raises | refuses |

The `Scalar` row is this shim's gap and it is deliberate, for the reason `arith_tag` already gives
for refusing `bool_tensor * 2`: the rule is a fast-path ladder keyed on the **Python type** of the
scalar, and `scalar_arg` has already erased `True` into `Scalar::Int(1)` by the time the kernel
sees it — so reproducing the third row is not possible from where the kernel stands, and
implementing the first two without it would compute where upstream raises. Two golden cases carry
it as `expect="c_error"`, so the harness watches the gap rather than filing it away.

### 3.4 Reachable by name

| table | entry | what it opens |
|---|---|---|
| `overloads.json` | `remainder.{Scalar_out, Tensor_out, Tensor, Scalar}` | `torch.remainder(x, y)`; the two `.out` forms refuse by name |
| `methods.json` | `remainder` → `[Tensor, Scalar]` | `x.remainder(y)` |
| `methods.json` | `__mod__` → `[Tensor, Scalar]` | **`x % y`**, which is how every caller actually spells it |

Order is load-bearing and asserted: the resolver picks `.Tensor` when handed a tensor and `.Scalar`
otherwise, so a table listing only one would silently answer the wrong overload for half the calls
`x % y` can make.

**`__rmod__` is deliberately absent, and asserted absent.** `3 % x` is
`aten.remainder.Scalar_Tensor`, a distinct overload with its own promotion rule, and it is not
implemented. Adding `__rmod__` without the kernel would make `3 % x` refuse by a name that resolves
to nothing.

### 3.5 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 265 | **266** (+1) |
| `compare.py` | 4481/4481, ops=141 | **4682/4682, ops=143** (+201 cases) |
| `verify_schemas.py` | 4363/4363 | **4376/4376** (+13) |
| sweep26 (shim) | 22/26 | **22/26** |
| `test_core_ops_and_op_tags_agree` | 86 | **88** — both overloads are `['core', 'pointwise', 'pt2_compliant_tag']`; +2 rather than +1 because this counts overloads, and two overloads of one op need not agree |
| schema identities | 224 | **228** (+4: all four `overloads.json` entries; `remainder`/`__mod__` add none) |

### 3.6 Sabotage check

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| M1 | drop the correction on both paths — plain `fmod`/`wrapping_rem`, the sign of the **dividend** | **4623/4682, 59 failed** | 2 FAIL | the sign rule is checked in every quadrant, on both overloads and both categories. An implementation that "looks right" and is written on `fmod` fails 59 cases |
| M2 | **Python's own `%`**: `a - floor(a/b)*b`. Gets every sign quadrant *right* | **4666/4682, 16 failed** | 2 FAIL | the corner cases are the ones doing the work. All 18 sign rows still pass; what fails is the 4 signed-zero cases (`-0.0` becomes `+0.0`) and the 12 infinity rows (`5.0 % inf` becomes NaN) |

M2 is the interesting one: it is the *plausible* wrong implementation — it satisfies the headline
property the op is named for — and it is caught only by rows a "does it follow the divisor's sign"
case set would never contain.

---

## 4. The legacy `torch.Tensor(int)` constructor — the decision, and why

**Sweep after: 22/26 (unchanged). What moved: `sew_d` advanced from the constructor to
`TensorBase.set_` — the same wall `vits` is on.**

The brief asked for a verdict rather than a patch: *is upstream's behaviour worth reproducing, or
is refusing right and `sew_d` simply unsupportable?*

**Verdict: reproduce it.** Three grounds, and the grounds matter more than the conclusion.

1. **It is not a new computation path.** A `TorchDispatchMode` trace of `torch.Tensor(3)` on 2.13.0
   fires exactly one op — `aten.empty.memory_format` — which this shim already implements and
   already golden-compares. So this is a *constructor spelling* over an existing kernel,
   structurally identical to ARCH26.md §3.1's `torch.conv2d` over `aten.convolution.default`. It is
   not the kind of invention `sqrt`-as-`pow(x, 0.5)` would have been, and that distinction is the
   one this round keeps having to make.
2. **The three forms are distinguishable at the type level**, which is what the old refusal already
   did — it extracted a `TensorBase` and refused everything else. Nothing has to be guessed:

   ```
   TensorBase(existing)   re-wrap, sharing the candle tensor
   TensorBase(2, 3)       uninitialised storage of that shape
   TensorBase([3, 4])     build from data -- a (2,) tensor of 3.0 and 4.0
   ```

3. **Refusing costs a family, not an architecture.**
   `nn.Parameter(torch.Tensor(config.hidden_size).uniform_())` is the `masked_spec_embed` idiom and
   it is constructed **unconditionally** in `__init__` whether or not `apply_spec_augment` is set —
   so no toy config avoids it. `sew_d` is where ARCH26.md §4 found it; `wav2vec2`, `sew`, `hubert`,
   `unispeech` and `wavlm` carry the same line.

### 4.1 What was reproduced, and the one divergence

```
TensorBase(3)           -> (3,)         TensorBase(3, 4)  -> (3, 4)
TensorBase()            -> (0,)         not ()  -- measured
TensorBase(0)           -> (0,)
dtype                   the default float, read at call time, so it follows
                        set_default_dtype exactly as upstream's does
TensorBase(-1)          raises: Trying to create tensor with negative dimension -1: [-1]
```

**The bytes are zeros and upstream's are uninitialised.** That is a real divergence, and it is the
one `aten.empty.memory_format` already has — its golden comparator is `_dtype_shape_only_check`,
whose docstring is "there is no correct value to diff". Reading a `torch.Tensor(n)` before writing
it is undefined upstream, so this narrows undefined behaviour rather than disagreeing about defined
behaviour, and the real caller writes it immediately with `.uniform_()`.

### 4.2 The form that stays refused, and why that is the interesting one

**`torch.Tensor([3, 4])` builds from data.** It is `tensor([3., 4.])`, a `(2,)` tensor — **not** a
`(3, 4)` empty one. It looks exactly like a size list and is not one.

That is the trap in implementing this at all: a constructor that accepted a sequence as a shape
would silently answer `(3, 4)` zeros where upstream answers two numbers, and nothing downstream
would raise. So it refuses, the message says *which* form it is refusing, and the test asserts the
refusal with a note explaining that answering it would be worse than refusing.

It could be implemented — it needs `_tensor_new_from_data`, a module-level function `PyTensorBase`
has no handle to — but no measured caller reaches it, and this round does not add unreached
surface.

### 4.3 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 266 | **267** (+1) |
| `compare.py` | 4682/4682, ops=143 | **4682/4682** (unchanged — a constructor spelling, no new dispatch key) |
| `verify_schemas.py` | 4376/4376 | **4376/4376** (unchanged — not a table entry) |
| sweep26 (shim) | 22/26 | **22/26** |

The two unchanged rows are the expected result and are worth stating: this adds no aten key and no
schema, so the only gate that can see it is the pytest — which is the same reason ARCH26.md §7's
four `bootstrap.py` names moved none of these numbers either.

---

## 5. `aten.set_.source_Tensor` and `aten.set_.default`

**Sweep after: 22/26 (unchanged). What moved: `vits` and `sew_d` both advanced past `set_` — and
what they advanced *to* is the finding in §5.3.**

### 5.1 The two forms of `set_` now have opposite aliasing behaviour, on purpose

```
a.set_(storage, offset, size, stride)   COPIES   (unchanged, docs/models/CKPT.md §4)
a.set_(b)                               ALIASES  (new)
a.set_()                                empties in place, keeping the dtype
```

That is not an inconsistency. The storage form has to copy: candle owns its memory and an
`UntypedStorage` is bytes this shim holds separately, so there is nothing to alias. The tensor form
does not have to: `Repr::Dense` **is** a candle tensor, and a candle clone is an `Arc` clone of the
same storage — so `replace_with` gives upstream's semantics for free, and `a.set_(b)` followed by a
write into `b` is visible through `a`, exactly as upstream. Both directions are asserted side by
side, because "one of these copies and the other does not" is the kind of claim that rots quietly.

The order of the two arms is load-bearing: **the tensor check comes first**, because an
`nn.Parameter` extracts as a `TensorBase` and would otherwise fall into the storage arm — producing
precisely the message ARCH26.md §2 recorded (`expected a torch.UntypedStorage, got Parameter`).

### 5.2 A fourth spelling of the dtype set

Upstream refuses a dtype mismatch:
`Could not set tensor of type long long to a tensor of type float`. Those names are the plain C++
type names and are **a fourth naming**, distinct from all three this crate already carries:

| dtype | `TorchDType::name()` | `c10_name` | `scalar_type_name` | **`set_type_name`** |
|---|---|---|---|---|
| int64 | `int64` | `int64_t` | `Long` | **`long long`** |
| int16 | `int16` | `int16_t` | `Short` | **`short`** |
| uint8 | `uint8` | `uint8_t` | `Byte` | **`unsigned char`** |
| float32 | `float32` | `float` | `Float` | `float` |

Not derivable from any of the others; every row was read off a real `RuntimeError` by provoking the
mismatch across all ten storable dtypes. The pytest asserts the **exact whole string**, not a
substring, because a substring match would pass on any of the four tables.

Silently adopting the source's dtype instead of refusing would make `set_` a `to()` with no
conversion — and the parametrize machinery would then swap a float parameter for an integer one
without complaint.

### 5.3 The finding: upstream swallows the next `NotImplementedError`

This is the part worth carrying forward. With `set_` working, `vits` and `sew_d` both fail with:

```
TypeError: _WeightNorm.forward() missing 1 required positional argument: 'weight_v'
```

which names no kernel at all and points at a line 200 frames from the cause.
`ParametrizationList.__init__` (`torch/nn/utils/parametrize.py:163`) computes:

```python
for module in reversed(self):
    if hasattr(module, "right_inverse"):
        try:
            new = module.right_inverse(new)
        except NotImplementedError:
            pass                      # <-- "we assume right_inverse is the identity"
```

`_WeightNorm.right_inverse` calls `torch.norm_except_dim`, which this shim raises
`NotImplementedError` for — **and upstream's own `except NotImplementedError: pass` eats it.** `new`
stays the original tensor, so `is_tensor` becomes `True`, `ntensors` becomes `1`, and `forward`
calls a two-argument function with one argument.

So the wall is `torch.norm_except_dim`, and the error message says nothing about it. Two things
follow:

1. **`NotImplementedError` is not a safe refusal type everywhere.** This shim uses it as its
   standard "not implemented" signal, and there is at least one place in the vendored tree where
   upstream treats it as a *control-flow* signal meaning "this optional method is absent". Anywhere
   that pattern occurs, a missing kernel becomes a confusing error somewhere else entirely.
2. **It is a worked example of this round's own trap** — "if you touch a refusal, re-read what it
   claims". Reading only the message would have sent the next round after
   `ParametrizationList.forward`.

### 5.4 What `weight_norm` actually costs, measured

Traced on upstream, so the remaining bill is a number rather than an estimate:

| name | fires | status |
|---|---|---|
| `torch.norm_except_dim(w, 2, dim)` | `view` → **`aten.norm.ScalarOpt_dim`** → `view` | `norm.ScalarOpt_dim` missing |
| `torch._weight_norm(v, g, dim)` | **`aten._weight_norm_interface.default`** (returns `(output, norms)`) | missing |

ARCH26.md §6 said `weight_norm` costs two kernels. **It costs three**, and the third
(`norm.ScalarOpt_dim`) was invisible to that document's method because `norm_except_dim` is a
composite and the trace that found `_weight_norm_interface` ran on a *forward*, not on
`register_parametrization`. §6 below carries it forward with the rest.

### 5.5 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 267 | **268** (+1) |
| `compare.py` | 4682/4682, ops=143 | **4682/4682** (unchanged) |
| `verify_schemas.py` | 4376/4376 | **4376/4376** (unchanged) |
| sweep26 (shim) | 22/26 | **22/26** |

**Golden cannot see this kernel at all, and that is structural rather than an omission.** `set_` is
a `TensorBase` method implemented directly in `tensor.rs`; it does not go through `_aten_dispatch`,
so it has no dispatch key for `compare.py` to reach and no `CASE_BUILDERS` entry is possible. The
pytest is the only gate, which is exactly why it asserts the aliasing in both directions and the
refusal string in full.

---

## 6. `aten.convolution.default`, the 2-D case

**Sweep after: 22/26 (unchanged). What moved: `zoedepth` went from the convolution refusal, through
`TensorBase.expand_as` (fixed here), to `torch.conv_transpose2d` — two walls further into the same
model.**

### 6.1 The brief's question: was it in reach?

**Yes, and it was the small piece rather than the large one.** The brief sized it as possibly
"larger than the rest combined" and said to leave it if so. It is not, for one reason that could
only be found by looking: **candle already carries `Tensor::conv2d`, with the same
`(padding, stride, dilation, groups)` signature `conv1d` has.** So the 2-D case is the same thin
wrapper written twice, not a second kernel — about forty lines of argument handling, of which the
convolution itself is one call.

What it is *not* is a general 2-D convolution, and the difference is the next section.

### 6.2 What it does and does not do

```
4-D input, symmetric stride/padding/dilation      implemented
1-element stride/padding/dilation on a 4-D input  implemented -- expands to both axes
asymmetric (stride_h != stride_w, etc.)           REFUSED by name
transposed / non-zero output_padding              refused, as before
```

**candle's `conv2d` takes one scalar per argument**, so it is symmetric only. torch allows
`(stride_h, stride_w)` to differ. An asymmetric call reaching a symmetric kernel would convolve with
the wrong geometry and produce a *wrong output shape in one axis only* — which a set of square test
cases cannot show. So it refuses, the message names which argument disagreed, and three golden cases
carry the gap as `expect="c_error"`.

Nothing measured needs it: `Dinov2`'s patch embedding, which is what ARCH26.md §3.2 stopped on, is
`nn.Conv2d(3, hidden, kernel_size=16, stride=16)` — square kernel, square stride, no padding.

**One measurement corrected a first attempt.** The kernel initially refused a 1-element
`stride`/`padding`/`dilation` on a 4-D input; the golden case written for it as `both_error` failed,
because **torch computes it** — `expand_param_if_needed` broadcasts a single value to every convolved
axis, so `padding=[2]` on a 4-D input pads both axes and gives `(1, 3, 7, 7)`. The kernel now
expands, and both spellings are live cases. The mirror rule is also upstream's and also asserted: a
2-element stride on a 3-D input **raises upstream**, so refusing it is not a shim restriction.

The non-square rows are the ones a square-only case set cannot fail on: `(1, 2, 4, 7)` input and
`(2, 1, 3, 2)` weight both appear, so an implementation that swapped the height and width axes has a
case that shows it.

### 6.3 `TensorBase.expand_as`, found behind it

With 2-D convolution working, `zoedepth` stopped on `TensorBase.expand_as` — **a name gap, not a
kernel gap**, and checked as such rather than assumed: `aten::expand_as` is
`CompositeImplicitAutograd` upstream, and a `TorchDispatchMode` trace of `x.expand_as(y)` fires
exactly one op, `aten.expand.default(x, y.shape)`, which this shim already implements and already
golden-compares. Added in `bootstrap.py` as `self.expand(list(other.shape))` — through `.expand`
rather than through `dispatch`, so it inherits that method's `-1` handling and rank rules and stays
a **view** (upstream's `expand_as` shares storage, measured by `data_ptr()`).

This is ARCH26.md §7's `torch.conv2d` fix in a new place, and it is the third time this round that
the wall behind a kernel turned out to be a spelling.

### 6.4 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 268 | **268** (the new assertions joined the existing road test) |
| `compare.py` | 4682/4682, ops=143 | **4709/4709, ops=143** (+27 cases, same op) |
| `verify_schemas.py` | 4376/4376 | **4376/4376** (unchanged — same schema) |
| sweep26 (shim) | 22/26 | **22/26** |

`ops covered` does not move because this widens an op that was already counted, which is the honest
shape of the change: `convolution.default` was never *absent*, it was 1-D only.

### 6.5 Sabotage check

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| C1 | delete the symmetry refusal and silently use axis 0 for both axes — the exactly-plausible mistake | **4706/4709, 3 failed** | 1 FAIL | all three asymmetric cases fire, and the harness's message is the right one: *"gap appears CLOSED: both sides now succeed, promote this case to expect=match and diff real values."* The `c_error` mechanism catches a refusal being dropped, not only a value being wrong |


## 7. Job 2: the blocked transposed copy

docs/numerics/SEQLEN.md §8.12 named this as the one clean kernel win left in SDPA and sized it at
**1.15 ms of a 13.64 ms per-call gap at `S=1024` — 8% of the SDPA gap, 7% of the model gap.** It is
the only entry in that table which is bit-identical by construction rather than by tolerance, and
that is why it is the one taken.

### 7.1 What was slow, and why it is safe to change

`sdpa_flash_cpu`'s default branch computes `k.transpose(2, 3).contiguous()`. candle's
`copy_strided_src` walks a transposed layout **one element at a time**, recomputing a
multi-dimensional index per element and reading `head_dim` floats away from the previous one — a
cache miss per element once the source leaves L2.

**Every output element is a copy of exactly one input element.** There is no arithmetic, so there is
no summation order to reassociate and no rounding to move; the only thing a blocked traversal
changes is the order in which the same assignments happen. That is the whole safety argument, and it
is a construction argument rather than a measurement.

**The trap recorded beside it in §8.5 is the opposite change and was not made.** Simply *dropping*
the `contiguous` is 5% faster and moves the `S=6` digest, because it lets Accelerate take a
transposed GEMM with a different accumulation order. This makes the copy faster; it does not remove
it. The `contiguous` call is still there — `transposed_contiguous` is a drop-in for it.

### 7.2 The layout, confirmed against a real forward before anything was timed

docs/perf/DTYPE_PERF.md §2-vs-§4 is the standing warning that this repository has twice been misled by a
microbench at a layout the model never produces. So `k` is captured out of a real
`SmolLM2-135M` prefill by monkeypatching `F.scaled_dot_product_attention`, and the shape **and
strides** are printed before any timing:

```
S=512    k        shape=(1, 3, 512, 64)   stride=(98304, 32768, 64, 1)   contig=True   f32   393216 bytes
         k.T(2,3) shape=(1, 3, 64, 512)   stride=(98304, 32768, 1, 64)   contig=False
S=1024   k        shape=(1, 3, 1024, 64)  stride=(196608, 65536, 64, 1)  contig=True   f32   786432 bytes
         k.T(2,3) shape=(1, 3, 64, 1024)  stride=(196608, 65536, 1, 64)  contig=False
```

That transposed layout is **exactly** the pattern `transposed_plan` recognises: `stride[-2] == 1`,
`stride[-1] == dims[-2]` (64), and the two leading strides contiguous over the pair's area
(`32768 = 512·64`, `98304 = 3·512·64`). 30 SDPA calls per forward.

`TensorBase.stride` is not implemented in this shim, so the strides above are read from the
**upstream** capture of the same tensor; both sides run the identical capture script, and the shim
side prints `contig=False` for the view and `contig=True` for `k`, which is the part it can answer.

### 7.3 The measurement

Alternating old/new with a `cmp`-verified artefact swap each time
(`/Volumes/macMini/caches/k26-scratch/ab.sh`), 3 rounds, 5 timed repetitions each, **minimum and
spread both reported**. `old` is this same tree with the kernel present but not wired in — so the
only difference between the two artefacts is the call site, and the other five kernels this round
added cannot confound it.

Machine: alone, but carrying a windowing server and a user application; `uptime` printed at each
round and was 2.24–3.08 throughout.

**The copy itself**, `aten.contiguous.default` on the captured view:

| | old (min of 3 rounds) | new | ratio | upstream |
|---|---:|---:|---:|---:|
| `S=512` | 0.2123 ms, 1.85 GB/s | **0.0404 ms, 9.73 GB/s** | **5.25x** | 0.0314 ms, 12.5 GB/s |
| `S=1024` | 0.4232 ms, 1.86 GB/s | **0.0875 ms, 8.99 GB/s** | **4.84x** | 0.0582 ms, 13.5 GB/s |

Spread on the new numbers is 0.0002–0.0008 ms; on old, 0.0005–0.0040 (one round showed 0.28, and
that round's minimum is the one reported). **Relative to upstream the copy goes from 6.8x slower to
1.29x at `S=512`, and from 7.3x to 1.50x at `S=1024`.**

**Per SDPA call**, same capture:

| | old | new | delta | spread (old / new) |
|---|---:|---:|---:|---|
| `S=512` | 4.6303 / 4.6330 / 4.6699 | **4.1324 / 4.1356 / 4.1470** | **−0.50 ms (−10.8%)** | 0.04–0.13 / 0.02–0.06 |
| `S=1024` | 17.3902 / 17.4377 / 17.4881 | **16.3991 / 16.4371 / 16.4519** | **−1.01 ms (−5.8%)** | 0.25–0.41 / 0.09–0.57 |

The `S=1024` figure is **1.01 ms against SEQLEN.md's sizing of 1.15 ms** — the sizing was right.

**The control**, the same build against itself, run last in every sweep:

| | run A | run B | ratio |
|---|---:|---:|---:|
| copy `S=512` | 0.0403 | 0.0405 | **1.005** |
| copy `S=1024` | 0.0875 | 0.0872 | **0.997** |
| SDPA `S=512` | 4.1449 | 4.1254 | **0.995** |
| SDPA `S=1024` | 16.4403 | 16.7473 | **1.019** |
| model `S=512` f32 | 349.61 | 349.64 | **1.000** |
| model `S=1024` f32 | 930.56 | 930.93 | **1.000** |
| model `S=512` bf16 | 433.63 | 433.07 | **0.999** |
| model `S=1024` bf16 | 1091.45 | 1092.52 | **1.001** |

Every control reads within 2% of 1.00 and every measured effect is larger than its control, at both
lengths and in both dtypes. The one control that is not within 1% (SDPA `S=1024`, 1.019) is still a
third of the 6.1% effect it sits beside, and that repetition carried a 32 ms outlier which the
minimum discards but the spread reports.

### 7.4 Model level, and the ratios

`f32`, min of 3 per length, 2 alternating rounds:

| S | old | new | upstream | ratio before | **ratio after** |
|---:|---:|---:|---:|---:|---:|
| 6 | 34.27 | 33.91 | 35.38 | 0.97x | 0.96x |
| 32 | 38.75 | 37.89 | 38.15 | 1.02x | 0.99x |
| 128 | 85.45 | 80.43 | 75.57 | 1.13x | **1.06x** |
| 512 | 364.27 | 350.17 | 229.31 | **1.589x** | **1.527x** |
| 1024 | 961.21 | 929.35 | 460.22 | **2.089x** | **2.019x** |

The two "before" ratios reproduce the brief's stated baseline exactly (1.58x, 2.09x), which is the
cross-check that this harness is measuring the same thing.

`bf16`, same protocol:

| S | old | new | delta |
|---:|---:|---:|---:|
| 512 | 446.86 | **433.70** | −2.9% |
| 1024 | 1122.01 | **1091.50** | −2.7% |

### 7.5 Bit-identity, proved by the digests

**All ten prefill digests are unchanged**, at every length docs/numerics/SEQLEN.md §1.3 records, in both
dtypes, on both sides of the change:

| S | `f32` | `bf16` |
|---:|---|---|
| 6 | `b9fc5553ee1bf6a2` | `8ef1550ea33c4f3d` |
| 32 | `331668f36da02f21` | `b81325c83a0a3d15` |
| 128 | `00159a9dbd308eda` | `7ff8e9334449b147` |
| 512 | `07c2797dabc4552e` | `9ab1e82f01378e38` |
| 1024 | `eda1e173727bb7f5` | `b4d9440df61212b1` |

Every `f32` row matches SEQLEN.md §1.3 and the `bf16` `S=128` row matches SEQLEN.md §4's
`7ff8e9334449b147`. Upstream's `S=128` digest came back `71e46824c0c40f15`, the value §1.3 records
for it — so the harness is reading the same model the previous rounds read.

### 7.6 Where it is wired, and the second call site

Two places, and the second is what makes the copy directly measurable:

1. `sdpa_flash_cpu`'s `k.transpose(2, 3)` — the site SEQLEN.md sized.
2. **`aten.contiguous.default`** — so `x.contiguous()` on any transposed view gets it, and so the
   microbench in §6.3 is timing the kernel rather than timing candle on both sides. (The first
   attempt at this measurement wired only site 1, and the copy row read 0.2123 vs 0.2120 — an
   accidental null control that showed the harness was not reaching the code under test.)

`contiguous.default`'s aliasing contract is unchanged and still asserted by
`test_which_ops_share_storage_with_their_input_and_which_do_not`, which has it in **both** tables:
it shares storage with a contiguous input (the fast exit returns candle's handle clone) and is
independent of a strided one (it copies). The fast exit is the first thing `transposed_contiguous`
checks, which is what keeps that true.

### 7.7 Sabotage check

| # | fault | `cargo test` | `run.sh` | golden | digests |
|---|---|---|---|---|---|
| T1 | ragged-edge bug: `r0 + BLOCK.min(cols)` instead of `(r0 + BLOCK).min(cols)` — correct inside a full block, overruns on the remainder | **1 FAILED** (panics at `3x5`) | — | — | — |
| T2 | **in-bounds and silent**: `dst[r*rows+c] = row[cols-1-r]` — right shape, right dtype, wrong values | **4 FAILED** | **3 FAIL**, including the real llama forward and `do_sample` | **4645/4682, 37 failed** | **all moved**: `S=6` `5948eac9…`, `S=128` `9954d220…`, `S=512` `a6f101ec…` |

T2 is the one that matters. It is the silent failure mode — a tensor of exactly the right shape and
dtype with the wrong numbers in it — and it is caught at four independent levels, **including the
prefill digests**. That is the proof the digests in §6.5 are a real gate on this change and not a
formality: a change to this kernel that was not bit-identical would move them, and this one does
not.

Sizes in the Rust test straddle the 32-element block on both axes in both directions
(`33x31`, `31x33`, `129x65`) and include the real `(512, 64)` SDPA shape, so an edge bug on either
side has a shape that shows it. A separate test copies `-0.0`, both infinities, both NaNs and a
subnormal and compares `to_bits()`, because `==` cannot see the first or the last.

---

## 8. Where every architecture stands, and what is left

### 8.1 The sweep, kernel by kernel

**20/26 → 22/26.** Which kernel moved which wall, in the order they were done:

| after | sweep | `deberta` / `deberta_v2` | `vits` | `sew_d` | `zoedepth` | `sam3_video` |
|---|---:|---|---|---|---|---|
| *(baseline)* | 20/26 | `torch.sqrt` | `set_` | `Tensor(int)` | `convolution` 4-D | `__mod__` |
| `sqrt` | 20/26 | → `repeat` | — | — | — | — |
| `repeat` | **22/26** | **PASS** | — | — | — | — |
| `remainder` | 22/26 | — | — | — | — | → `div.Scalar_mode` |
| `Tensor(int)` | 22/26 | — | — | → `set_` | — | — |
| `set_` | 22/26 | — | → `norm_except_dim`\* | → `norm_except_dim`\* | — | — |
| `convolution` 2-D | 22/26 | — | — | — | → `expand_as` | — |
| `expand_as` | 22/26 | — | — | — | → `conv_transpose2d` | — |

\* surfacing as `TypeError: _WeightNorm.forward() missing 1 required positional argument` — see §5.3.

**Only `repeat` moved the count**, and it could only move it because `sqrt` came first. Every other
kernel moved a first wall without closing an architecture, which is what §0 said this table would
have to show rather than crediting the batch.

### 8.2 What is left, with reasons

**Left because the next wall is a different kernel, not because the work was hard.** Each of these
was reached, named and left:

| architecture | next wall | size |
|---|---|---|
| `vits`, `sew_d` | **`aten.norm.ScalarOpt_dim`** then **`aten._weight_norm_interface.default`** | two kernels, both small; §5.4 has the traces. `norm_except_dim` is a composite over the first, `torch._weight_norm` fires exactly the second |
| `zoedepth` | **`aten.convolution.default`, `transposed=True`** (`torch.conv_transpose2d`) | candle has `conv_transpose2d`, so it is *probably* the same wrapper again — but the weight-layout convention of a transposed convolution is `(in, out/groups, kH, kW)` rather than `(out, in/groups, kH, kW)`, and getting that backwards produces a plausible tensor. It needs its own measurement round, not an extension of §6 |
| `sam3_video` | **`aten.div.Scalar_mode`** / `div.Tensor_mode` (`rounding_mode="floor"`) | one kernel, small. It is the `y_positions` line two characters from the `% end_x` §3 fixed |

**Behind those**, from ARCH26.md §6's operator sweep, re-checked against `all_implemented()` rather
than against `_aten_implemented()` — which matters, because three of that table's entries turned out
to *already exist*, parked in `IMPLEMENTED_AWAITING_GOLDEN`:

```
already implemented, contrary to ARCH26.md §6:  aten.zeros.default   aten.add.Scalar
                                                aten.masked_fill.Tensor
```

Genuinely still missing, none of them attempted here:

```
aten.sigmoid.default        aten.sign.default          aten.erf.default
aten.log2.default           aten.clamp_min.default     aten.clamp_min.Tensor
aten.flip.default           aten.leaky_relu.default    aten.ones_like.default
aten.randn_like.default     aten.all.default           aten.div.Tensor_mode
aten.div.Scalar_mode        aten.avg_pool2d.default    aten.native_group_norm.default
aten.upsample_bilinear2d.default   aten.norm.ScalarOpt_dim
aten._weight_norm_interface.default                    aten.fmod.{Tensor,Scalar}
aten.remainder.Scalar_Tensor  (and with it `__rmod__`)
aten.set_.source_Tensor_storage_offset
```

The first six of those are one-line members of families that already exist here (`unary_float` for
`sigmoid`/`erf`/`log2`, the comparison family for `clamp_min`) and are the cheapest remaining work.
`native_group_norm`, `avg_pool2d` and `upsample_bilinear2d` are real kernels with their own
measurement rounds.

**Left deliberately, with the reason rather than the size:**

- **`torch.Tensor([3, 4])`**, the build-from-data constructor (§4.2). It looks like a size list and
  is not one; refusing is better than answering `(3, 4)` zeros where upstream answers two numbers.
- **`remainder.Scalar` on a `bool` tensor** (§3.3). Upstream's promotion there is a fast-path ladder
  keyed on the scalar's *Python type*, and `scalar_arg` has erased that by the time the kernel runs.
  Two `c_error` golden cases watch it.
- **asymmetric 2-D convolution** (§6.2). candle's `conv2d` is symmetric-only; three `c_error` cases
  watch it.
- **`sqrt.out`, `remainder.{Scalar,Tensor}_out`** — in `overloads.json` with no kernel, so the `out=`
  spelling refuses by the right name instead of falling through to "no table entry for this op".

### 8.3 One correction to docs/architectures/ARCH26.md

ARCH26.md §6 says `weight_norm` costs **two** kernels. **It costs three.** The third,
`aten.norm.ScalarOpt_dim`, was invisible to that document's method for two compounding reasons:
`torch.norm_except_dim` is a *composite* (so the op name never appears in the source), and the trace
that found `_weight_norm_interface` ran on a **forward**, while `norm_except_dim` is called from
`register_parametrization` at **construction** time. That is ARCH26.md §6's own stated blind spot —
the one it demonstrated on `sam3_video` — appearing a second time in the same document.

### 8.4 Final gates

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   268 ok, 0 FAIL                    exit 0   (was 261)
$PY tools/golden/compare.py                 4709/4709, ops=143, pending 1     exit 0   (was 4290/4290, ops=139)
$PY tools/golden/compare.py --self-test     14 comparators x 11 fault modes   exit 0   (was 13)
$PY rust/torch_c/pytests/verify_schemas.py  4376/4376                         exit 0   (was 4353/4353)
( cd rust/torch_c && cargo test --release ) 28 passed                         exit 0   (was 24)
sweep26 (shim)                              22/26                             exit 0   (was 20/26)
sweep26 (upstream)                          26/26                             exit 0
```

Prefill digests on the final artefact, re-run after every change above:

```
f32   S=6 b9fc5553ee1bf6a2   S=32 331668f36da02f21   S=128 00159a9dbd308eda
      S=512 07c2797dabc4552e   S=1024 eda1e173727bb7f5
```

all five unchanged from docs/numerics/SEQLEN.md §1.3.

---

# Round two: the last four architectures

Continues the document above, with the same method: **the sweep is re-run after
every kernel**, and each section records the count and, when the count does not
move, which wall replaced the one that was removed. Written incrementally, one
kernel at a time.

The starting point is §8.4's, re-measured on this worktree before any edit and
identical to it:

```
pytests/run.sh                268 ok, 0 FAIL                        exit 0
tools/golden/compare.py       4709/4709, ops=143, pending=1          exit 0
compare.py --self-test        14 comparators x 11 fault modes        exit 0
verify_schemas.py             4376/4376                              exit 0
sweep26 (shim)                22/26                                  exit 0
```

and the four walls §8.2 left, each reproduced exactly:

| architecture | first wall at this round's baseline |
|---|---|
| `vits` | `TypeError: _WeightNorm.forward() missing 1 required positional argument` (= `norm_except_dim`, §5.3) |
| `sew_d` | the same |
| `zoedepth` | `torch.conv_transpose2d(...)` — no overload table entry |
| `sam3_video` | `aten.div.Scalar_mode` |

Prefill digests, also re-measured before any edit and matching docs/numerics/SEQLEN.md §1.3
and §8.8:

```
f32   S=6 b9fc5553ee1bf6a2   S=32 331668f36da02f21   S=128 00159a9dbd308eda
      S=512 07c2797dabc4552e   S=1024 eda1e173727bb7f5
bf16  S=128 7ff8e9334449b147
```

## 9. `aten.div.Tensor_mode` and `aten.div.Scalar_mode`

**Sweep after: 22/26 (unchanged). What moved: `sam3_video` advanced from
`aten.div.Scalar_mode` to `torch.outer` — the next line of the same
`Sam3ViTRotaryEmbedding.__init__`.**

`modeling_sam3.py:428` builds the rotary position grid three lines running:
`flattened_indices % end_x` (§3's kernel), `torch.div(flattened_indices, end_x,
rounding_mode="floor")` (this one), and then `torch.outer` on the result.

**Both keys were already in `overloads.json` and `methods.json`.** Only the
kernels were missing, which is why the baseline wall was `aten op not
implemented in torch._C shim: aten.div.Scalar_mode` — a dispatch failure, not an
overload-resolution failure. So this section adds no table entry, and
`verify_schemas.py` moves only because the two new keys become *checked*
schemas (+4, counting the two `.Scalar_mode`/`.Tensor_mode` identities on each
of the two tables).

### 9.1 The three modes are three functions

Measured on 2.13.0. This table is the acceptance criterion:

```
        a    b     None      trunc   floor
        7    3     2.333       2       2
        7   -3    -2.333      -2      -3     <- trunc/floor DISAGREE
       -7    3    -2.333      -2      -3     <- trunc/floor DISAGREE
       -7   -3     2.333       2       2
       -6    3    -2.0        -2      -2     <- opposite signs, EXACT: AGREE
    dtype        float32     int64   int64
```

They differ in the answer, in the dtype, and in whether division by zero
raises — `None` promotes an integral pair to the default float and answers
`inf`; `trunc` and `floor` preserve the dtype and raise
`RuntimeError('ZeroDivisionError')` on the same input.

**`trunc` and `floor` differ exactly when the operands' signs differ *and* the
division is inexact.** That refinement is the part worth having: "they differ
when the signs differ" is the natural statement and it is wrong, because
`-6 / 3` has differing signs and both modes answer `-2`. Established rather
than asserted, over 210 integer pairs: the two modes disagree on 64 of them,
that set contains no same-sign pair and no exact-division pair, and it is
precisely the 64 opposite-sign-inexact pairs.

So **two** different case sets would each pass both implementations — one built
from positive operands, and one built from opposite signs that divide exactly.
Both the golden builder and the road test carry all three kinds, and the road
test asserts the *disagreement pattern* explicitly
(`[False, True, True, False, False, False]`) rather than only the values.

### 9.2 `floor(a / b)` is the plausible implementation and it is wrong

Upstream's `div_floor_floating` is transcribed rather than re-derived, because
four of its answers do not follow from the name. All measured:

```
inf  /  3.0    ->  nan     not inf   -- fmod(inf, b) is NaN, and it propagates
5.0  / -inf    ->  -1.0    not -0.0  -- the sign correction, on a zero quotient
-5.0 /  inf    ->  -1.0    not -0.0
-0.5 /  3.0    ->  -1.0    not -0.0
5.0  /  0.0    ->  inf               -- and this is why `inf / 3.0` is different
-0.0 /  3.0    ->  -0.0              -- sign bit, which `==` cannot see
```

The last two are the interesting pair. `5.0 / 0.0` is `inf` and `inf / 3.0` is
`nan`, and both are non-finite results of the same op — the difference is that
upstream returns the raw IEEE quotient when `b == 0` through an **explicit early
return**, and runs the algorithm otherwise. Deleting that one branch is
sabotage D4 below.

**Verified as an algorithm, not as a table of corners.** Over 10609 `f64` pairs
built from infinities, NaNs, signed zeros, subnormals and randoms, the
transcription reproduces upstream **10609/10609 bit-identical**, compared on the
packed bytes so that `-0.0` and NaN are not read as equal to their opposites.
`trunc` likewise, 10609/10609. On the integer side, 210/210 for both modes.

One note for whoever writes the next one of these: **Rust's `%` on `f64` is C's
`fmod`; Python's `math.fmod` is not.** Python raises on `fmod(inf, 3.0)` where C
returns NaN, and the model used to verify this had to be corrected for exactly
that — it reported 200 spurious mismatches first.

### 9.3 The precision finding, and the golden case that could not fail

**Upstream computes in the tensor's own dtype, and computing in `f64` and
narrowing once is measurably wrong.** `read_flat` hands every floating dtype
over as `f64`, so computing there is the obvious thing to do. Over 42436
`float32` pairs it misses **68** of `floor` and **358** of `trunc`:

```
16777216.0 / 1.3669793605804443   upstream 12273204.0   f64-then-narrow 12273203.0
```

`float16` and `bfloat16` want their own precision too, not `f32`'s. With a
narrowing closure threaded through **every intermediate step** rather than
applied once at the end, all four dtypes match upstream on every pair
(0/3225 mismatches each); computing in `f32` instead misses 144 of `f16`'s
floor cases and 781 of its trunc cases.

**And the golden cases for this could not fail.** A 1-ULP `float32` error at
these magnitudes is ~8e-8 relative, and this harness's `float32` tolerance is
`rtol=1e-5`. Under the default comparator the precision cases pass whether the
kernel computes in `f64` or not. They use `_exact_value_check`, and sabotage D2
below measures the difference: **with the exact comparator 10 cases fail, with
the default comparator 0 of the 6 precision cases do.** That is the third time
this repository has found a test that could not fail, and it was found by
assuming there was one.

### 9.4 A measured upstream inconsistency, carried as `expect="diverge"`

For `float16` and `bfloat16` only, **upstream's answer depends on the tensor's
length**:

```
float16  floor  -1121.0 / -1.1806640625     n=1: 949     n>=2: 948
float16  trunc   1050.0 / -0.69873046875    n=1: -1502   n>=2: -1503
bfloat16 floor    187.0 /  0.83984375       n=1: 222     n>=2: 221
bfloat16 trunc    190.0 /  0.68359375       n=1: 276     n>=2: 278
```

Measured at `n` = 1, 2, 4, 7, 8, 16, 17, 32, 64, 100. Every `n >= 2` agrees with
every other, including `n = 7` and `n = 17` which are not multiples of any vector
width — so this is a **one-element fast path**, not a vectorised-body/scalar-tail
split. `float32` and `float64` are stable at every length.

This shim computes in the tensor's own dtype, which is upstream's `n >= 2`
answer and therefore the answer every real tensor gets. The four `n == 1` pairs
are carried as `expect="diverge"`, which surfaces them on every run and **fails
if they start agreeing** — so if upstream ever unifies its two paths, this says
so rather than silently healing.

Reproducing the `n == 1` behaviour instead was rejected: it would make a
kernel's answer depend on its input's length, which is a genuinely surprising
property to give a kernel, and it would encode a vectorisation artefact that
upstream is free to change.

The table is keyed on `(dtype, mode)` and not on dtype alone. Keying it on dtype
alone was the first attempt and produced a case that could not fail — a pair
that diverges at `n == 1` under `floor` need not diverge under `trunc`, and one
of the four was green for that reason until it was measured per mode.

### 9.5 What is reproduced, and the one deliberate gap

```
promotion (Tensor_mode)     torch.promote_types exactly, 49 cells, no disagreements
promotion (Scalar_mode)     the wrapped-number rule -- an int scalar never widens,
                            a float scalar floats an integral tensor
uint8(200) / -3   floor     0        -- the scalar narrows to 253 BEFORE dividing
int8(100)  / -3   floor     -34      -- and to -3 here, because int8 is signed
bf16(40)   / 0.3  floor     133      -- but a REDUCED-FLOAT scalar does NOT narrow:
                                        div_floor_kernel reads it at opmath_t just
                                        as div_true_kernel does, and narrowing
                                        answers 132. Wrong here until
                                        docs/numerics/SCALAR.md §3.2; a floor turns the
                                        rounding step into a whole unit
i64::MIN   / -1   both      i64::MIN -- wrapping_div, as remainder uses wrapping_rem
integral divisor 0          RuntimeError('ZeroDivisionError'), trunc/floor only
floating divisor 0          inf/-inf/nan, no raise, every mode
rounding_mode='ceil'        "div expected rounding_mode to be one of None, 'trunc',
                            or 'floor' but found 'ceil'" -- verbatim, case-sensitive
```

**`rounding_mode=None` is delegated, not restated.** It is true division and
therefore literally `aten.div.Tensor`/`aten.div.Scalar`, so it calls them. One
implementation rather than two that have to be kept in agreement, and the road
test asserts `torch.div(a, b, rounding_mode=None) == torch.div(a, b)`.

**The one deliberate gap is `div.Scalar_mode` on a `bool` tensor**, and it is
`remainder.Scalar`'s gap in a new place, for the same reason. Upstream computes
there through a fast-path ladder keyed on the scalar's *Python* type
(`div(bool_t, 2, "floor")` is `int64`, `div(bool_t, 2.0, "floor")` is `float32`,
`div(bool_t, True, "floor")` raises), and `scalar_arg` has erased `True` into
`Scalar::Int(1)` before the kernel runs. Four `c_error` golden cases watch it.
`div.Tensor_mode` on a bool pair raises upstream's own
`"div_trunc_cpu"`/`"div_floor_cpu" not implemented for 'Bool'`, verbatim.

### 9.6 Reachable by name

Golden compares by dispatch key and is structurally blind to a missing name, so
the spellings are asserted separately, through a real `import torch` in
`_KERNELS26_ROAD_SCRIPT`:

| spelling | lands on |
|---|---|
| `torch.div(t, t, rounding_mode="floor")` | `div.Tensor_mode` |
| `torch.div(t, 3, rounding_mode="floor")` | `div.Scalar_mode` |
| `x.div(y, rounding_mode="trunc")` | both, by the same rule |
| `torch.div(t, t)` | `div.Tensor` — *not* `Tensor_mode` |

That last row is the one that needs the order in `methods.json` to be right, and
it is asserted: the `_mode` entries come **first** and are skipped when the
keyword is absent, because `str? rounding_mode` is keyword-only *and carries no
default* in `native_functions.yaml`. Reordering them so `div.Tensor` came first
would make every `rounding_mode=` call silently answer true division. The test
pins the whole four-element list rather than checking membership.

`div_.Tensor_mode` / `div_.Scalar_mode` (the in-place forms) are in
`methods.json` with no kernel and stay that way — nothing measured reaches them.

### 9.7 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 268 | **268** (the new assertions joined the existing road test) |
| `compare.py` | 4709/4709, ops=143 | **5162/5162, ops=145** (+453 cases, +2 ops) |
| `verify_schemas.py` | 4376/4376 | **4380/4380** (+4) |
| `test_core_ops_and_op_tags_agree` | 88 | **90** — both overloads are `['core', 'pointwise', 'pt2_compliant_tag']`, read off each rather than copied from `div.Tensor` beside them |
| sweep26 (shim) | 22/26 | **22/26** |

### 9.8 Sabotage check

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| D1 | **`floor(a / b)`** — the plausible implementation, whole algorithm deleted | **5138/5162, 24 failed** | 1 FAIL | the non-finite corners are doing real work: `inf/3.0`, `5.0/-inf`, `-5.0/inf` and `-0.5/3.0` all fire, on all four float dtypes |
| D2 | **compute in `f64`**, narrow once at the end — `float_narrower` becomes the identity | **5152/5162, 10 failed** | **0 FAIL** | only golden sees this, and only through `_exact_value_check`. With the default comparator instead: **5158/5162, and 0 of the 6 precision cases fail** — the tolerance swallows the whole defect |
| D3 | drop the floor correction on both paths — **`floor` becomes `trunc`** | **5093/5162, 69 failed** | 1 FAIL | the sign quadrants, on both overloads and both categories. The opposite-sign *exact* rows still pass, which is why they are not the whole case set |
| D4 | delete the **`b == 0` early return** | **5150/5162, 12 failed** | 1 FAIL | the one branch that makes `5.0 / 0.0` (`inf`) differ from `inf / 3.0` (`nan`). A single deleted `if`, caught by 12 cases |

D2 is the one worth carrying forward: it is invisible to `run.sh` entirely, and
it was invisible to golden too until the comparator was changed. A kernel can be
wrong by 1 ULP in a way that every tolerance-based gate in the repository
accepts.

## 10. `aten.convolution.default` with `transposed=True`

**Sweep after: 22/26 (unchanged). What moved: `zoedepth` advanced from
`torch.conv_transpose2d` to `aten.upsample_bilinear2d` — the next op in the same
`ZoeDepthUpsample` block, and one §8.2 already lists as missing.**

§8.2 left this deliberately and gave the reason: candle has `conv_transpose2d`,
but a transposed convolution's weight layout is the *opposite* of a forward
one's, and **getting it backwards produces a plausible tensor rather than an
error**. That is exactly what the measurement below had to be built around.

### 10.1 The layout, established three ways

**`(in_channels, out_channels/groups, kH, kW)`** — against the forward
convolution's `(out_channels, in_channels/groups, kH, kW)`.

Established from upstream's behaviour, on a case with **unequal channels and a
non-square kernel** so that no two axes are interchangeable:

```
conv_transpose2d(x(2,3,5,7), w(3,5,2,4))  ->  (2,5,6,10)
```

1. **Shape.** `out_channels` is `w.shape[1]`, and the input's 3 channels bind
   `w.shape[0]`. Handing it the transposed `w(5,3,2,4)` raises
   `Given transposed=1, weight of size [5, 3, 2, 4], expected input[2,3,5,7] to
   have 5 channels, but got 3 channels`.
2. **The module.** `nn.ConvTranspose2d(3, 5, kernel_size=(2,4)).weight.shape` is
   `[3, 5, 2, 4]`.
3. **The definition.** A from-scratch scatter-add implementation — transposed
   convolution as the gradient of a convolution with respect to its input —
   agrees with upstream on four configurations, including `groups=2`, which the
   shim itself refuses. That third check is the one that does not depend on
   reading upstream's conventions correctly, only on the mathematics.

candle's `conv_transpose2d` reads its kernel as `(c_in_k, c_out, k_h, k_w)` and
bails if `c_in_k` disagrees with the input's channel count, so it is the same
convention and the weight is passed through unpermuted.

### 10.2 Why the model's own call cannot show any of this

`zoedepth`'s `ZoeDepthUpsample` is

```python
nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
```

**Equal in/out channels and a square kernel.** Swapping the weight's first two
axes gives a tensor of exactly the same shape, and so does flipping the kernel
spatially. Measured on `(1,2,3,3)` input with a `(2,2,3,3)` weight:

| arrangement | sum | first five elements |
|---|---:|---|
| correct | 61317 | 162, 351, 569, 413, 224 |
| first two axes swapped | 54756 | 81, 180, 299, 224, 125 |
| **kernel flipped spatially** | **61317** | 234, 493, 775, 535, 276 |

The spatial flip **keeps the sum identical** while changing every element. So a
checksum-shaped test cannot see it, and neither can a shape assertion. Both
wrong arrangements are carried as live golden cases compared element by element,
and the road test asserts the separation directly — that the three have the same
shape, that the swap and the flip both change the values, that the flip does not
change the sum, and that the swap does.

Those road assertions are there to keep the *case set* honest rather than to
detect the bug: they are computed entirely on one side, so what they prove is
that these three arrangements really are distinguishable by value and really are
not distinguishable by sum. Sabotage C3 below is the check that the comparison
against upstream catches a flip.

### 10.3 What is implemented, and what is refused

```
2-D transposed (4-D input), groups=1, symmetric geometry     implemented
grouped transposed convolution (groups != 1)                 REFUSED by name
1-D transposed (3-D input)                                   REFUSED by name
asymmetric stride/padding/dilation/output_padding            REFUSED by name
```

**candle's `conv_transpose2d` has no `groups` parameter** — its
`ParamsConvTranspose2D` has no field for it, and its `conv_transpose1d` *does*,
so this is a gap in that one function rather than in candle. A grouped call is
refused rather than computed as `groups=1`. A per-group decomposition is
possible and is left for a round that has a caller for it; `zoedepth` does not.

1-D transposed is refused for the opposite reason: candle supports it fully,
groups included, but nothing measured reaches it, and this round does not add
unreached surface.

`output_padding` carries upstream's own precondition, and the bound is not the
obvious one: **`output_padding < max(stride, dilation)`**, not
`output_padding < stride`. Measured — `output_padding=1` is accepted with
`stride=1, dilation=2` and refused with `stride=1, dilation=1`. Upstream's
message is reproduced verbatim and three `both_error` cases pin it, one per
reason it can fire.

### 10.4 The spelling, and the argument order that is not `conv2d`'s

`torch.conv_transpose2d` was the name the sweep actually stopped on
(`overload resolution has no table entry for this op`), so it is added in
`bootstrap.py` next to `conv1d` and `conv2d`, as the same kind of composite:
`aten::conv_transpose2d` is `CompositeImplicitAutograd` and fires exactly one
`aten.convolution.default` record with `transposed=True`.

**Its signature is not `conv2d`'s with an extra argument:**

```
conv2d           (input, weight, bias, stride, padding, dilation, groups)
conv_transpose2d (input, weight, bias, stride, padding, output_padding, groups, dilation)
```

`groups` comes **before** `dilation`. Transcribing `conv2d`'s order would swap
the two for every positional caller, and both are small integers that usually
produce a plausible tensor rather than an error. Read off
`torch.conv_transpose2d.__doc__` and confirmed by calling upstream positionally;
the road test calls it positionally with `groups=1, dilation=2`, which the
swapped reading turns into `dilation=1, groups=2` — a different shape, and in
fact a refusal.

`F.conv_transpose2d is torch.conv_transpose2d` is asserted rather than assumed,
because the whole spelling rests on it.

**candle's own argument order is a third one:** `(kernel, padding,
output_padding, stride, dilation)`, where `conv2d`'s is `(kernel, padding,
stride, dilation, groups)`. `output_padding` sits where `stride` sits in the
forward call. Sabotage C2 is exactly that transcription.

### 10.5 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 268 | **268** (the new assertions joined the existing road test) |
| `compare.py` | 5162/5162, ops=145 | **5184/5184, ops=145** (+22 cases, same op) |
| `verify_schemas.py` | 4380/4380 | **4380/4380** (unchanged — same schema, and `conv_transpose2d` is a spelling, not a table entry) |
| sweep26 (shim) | 22/26 | **22/26** |

`ops covered` does not move for the same reason §6.4's did not: this widens an
op that was already counted. `convolution.default` was never absent, it was
forward-only.

### 10.6 Sabotage check

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| C1 | **read the weight as `(out, in, kH, kW)`** — the forward convolution's layout, permuted before the call. The exactly-plausible mistake | **5171/5184, 13 failed** | 1 FAIL | of the 13, **9 are shape errors** from the unequal-channel cases and **4 are value mismatches** — and those 4 are precisely the equal-channel, square-kernel cases. A case set built only from `zoedepth`'s own shape would have had nothing but those 4 |
| C2 | **candle's arguments in `conv2d`'s order** — `stride` into the `output_padding` slot | **5171/5184, 13 failed** | 1 FAIL | the geometry cases vary each of stride/padding/dilation/output_padding alone and to distinct values, so a permuted call changes the output shape |
| C3 | **flip the kernel spatially** — same shape, same sum, every element different | **5171/5184, 13 failed** (all 13 value mismatches) | **0 FAIL** | only the comparison against upstream sees this. No shape moves and no checksum moves; nothing but an element-wise diff against the oracle catches it |

C1's 4-versus-9 split is the finding worth carrying: the fault the section exists
to prevent is caught loudly by cases that had to be *invented* (unequal channels,
non-square kernel) and quietly by the four that resemble the real model. Had the
case set been written from the caller, only the quiet four would exist.

## 11. `weight_norm`'s three pieces

**Sweep after: 22/26 (unchanged). What moved: `vits` and `sew_d` both advanced
past `_WeightNorm` entirely — `vits` to `torch.ones_like`, `sew_d` to
`torch.group_norm`.** Two architectures moved on one change, and they moved to
*different* walls, which is the first time this round that has happened.

§8.3's correction to ARCH26.md is confirmed and can now be stated exactly.
**`weight_norm` costs four things, of which two are kernels:**

| piece | kind | fires |
|---|---|---|
| `aten.norm.ScalarOpt_dim` | **kernel** | at construction, under `norm_except_dim` |
| `aten._weight_norm_interface.default` | **kernel** | in the forward |
| `torch.norm_except_dim` | spelling (composite) | `right_inverse`, at construction |
| `torch._weight_norm` | spelling (composite) | `_WeightNorm.forward` |

ARCH26.md §6 counted two because it counted *kernels found in a forward trace*.
`norm.ScalarOpt_dim` is invisible to that method twice over: `norm_except_dim`
is a composite, so the op name never appears in the source, and it runs at
construction, so a forward trace never reaches it.

### 11.1 The error that names nothing, confirmed

§5.3's finding held exactly. At this section's baseline both architectures
failed with

```
TypeError: _WeightNorm.forward() missing 1 required positional argument: 'weight_v'
```

and the cause was that **`torch.norm_except_dim` did not exist as a working
name**. It is worth recording *how* it did not exist: the shim generates a stub
for every `torch.*` name in its surface table, and that stub raises
`NotImplementedError` when there is no overload entry behind it. So the missing
name and the missing kernel produce the same exception type — and
`ParametrizationList.__init__` swallows precisely that type:

```python
try:
    new = module.right_inverse(new)
except NotImplementedError:
    pass          # "we assume that right_inverse is the identity"
```

`new` stays the original weight, `ntensors` becomes 1, and a two-argument
`forward` is called with one argument, two hundred frames away.

**This shim's standard refusal type is a control-flow signal on that path**, and
the generic missing-name stub raises it for every unimplemented name in the
whole surface. Anything reached through `torch.nn.utils.parametrize` will
therefore fail somewhere else entirely. The road test now asserts the three
things that swallowing defeats: that the parametrization is installed, that it
carries **both** parameters (`original0` and `original1` — one of them means
`right_inverse` was skipped), and that `right_inverse` followed by `forward`
reproduces the original weight.

### 11.2 `norm.ScalarOpt_dim` — `p` is six functions, and the caller uses one

```text
p = None    same as p = 2
p = 0       the COUNT of non-zero elements, not a sum
p = +inf    max |x|
p = -inf    min |x|
p = 1       sum |x|
otherwise   (sum |x|^p)^(1/p), fractional and NEGATIVE p included
```

`norm_except_dim` only ever passes `2`. So a case set written from the caller
would exercise one of six branches, and the golden builder runs all ten measured
`p` values against every dim/keepdim combination instead.

Two rules that are the opposite of the natural reading, both measured:

- **An empty `dim` list reduces every axis.** `norm(x, 2, [])` on a 2x2 is a
  scalar, not the input.
- **`p = 0` counts non-zeros.** The general formula gives the *element* count,
  because `|0|^0` is `1`. That is sabotage W3 below, and it is a one-line
  simplification that looks like a tidy-up.

The negative-`p` rows are the ones that catch an implementation that
special-cases its way out of trouble: `norm([[0,0],[1,2]], p=-1, dim=1)` is
`[0.0, 0.666...]` — the zero row gives `|0|^-1 = inf`, a sum of `inf`, and
`inf^(-1) = 0`. It falls out of the general formula and has to be special-cased
*not* to happen.

Integral and boolean input raise with upstream's own wording,
`norm(): input dtype should be either floating point or complex. Got Long
instead.`, and a repeated dim raises `dim 0 appears multiple times in the list
of dims`. Both reproduced and both carried as `both_error`.

### 11.3 `_weight_norm_interface` — and why its second result is `float32`

```text
norms = norm_except_dim(v, 2, dim)      keep `dim`, reduce every other axis
out   = v * (g / norms)
```

Both halves checked against upstream rather than taken from the formula.

**The finding here is a dtype rule that turned out to be a precision rule.**
Upstream returns `norms` as `float32` for a `float16`/`bfloat16` input while
`out` keeps the input's dtype. Implementing that literally — norm in the input
dtype, then cast the result to `float32` — gives the right dtype and the wrong
number, and the golden cases caught it on the first run:

```
float16 v=(2,3) dim=0    upstream 2.4494898319244385    narrow-then-widen 2.4492188
```

`2.4494898319244385` is the `float32` value of `sqrt(6)`. So upstream is not
casting a `float16` norm up; it is **doing the whole computation in `float32`**,
and the `float32` result dtype is a consequence of that rather than a separate
rule. The kernel now widens `v` and `g` once at the top, computes there, and
narrows only `out` at the end. This is §9.3's lesson in a second place: a dtype
that looks like a convention is usually a compute precision.

**`dim` must be `0` or `v.dim() - 1`.** Upstream does not raise a real error for
anything else — it trips `dim == 0 || dim == v.dim() - 1 INTERNAL ASSERT FAILED`.
Both live callers sit inside that range and they sit at opposite ends of it:
`vits` takes the default `dim=0`, `sew_d` passes `dim=2` on a 3-D `Conv1d`
weight. Both ends have their own golden case, because an implementation that
handled only `dim=0` would pass everything `vits` needs and fail `sew_d`.

A `v`/`g` dtype mismatch raises (`expected scalar type Float but found Double`)
rather than promoting, integral input raises
`"weight_norm_kernel" not implemented for 'Long'`, and a zero row is not
special-cased — upstream answers `nan` and so does this.

### 11.4 The two spellings

`torch.norm_except_dim` decomposes three different ways upstream depending on
`dim` (`view`/`norm`/`view` at the ends, a `transpose`/`clone` sandwich in the
middle, and `norm.Scalar` for `dim=-1`). Those are three routes to one
statement — *keep `dim`, reduce the rest, keepdim* — which is checked against
`v.pow(2).sum(others, keepdim=True).sqrt()` for every axis of a 3-D tensor. It
is written here as the one statement, since `norm.ScalarOpt_dim` takes a
multi-axis `dim` list directly.

**`dim=-1` is not "the last axis".** It is upstream's "no axis is exempt"
spelling and gives a 0-d whole-tensor norm — the one value of `dim` that is not
an axis index, so it is a branch rather than a normalisation. Asserted by shape
(`[]`, not `[1, 2]`) as well as by value.

`torch._weight_norm` fires exactly one record, `_weight_norm_interface`, and
discards its second result. The road test pins that `[0]` by comparing the two
spellings against each other.

### 11.5 Counts

| gate | before | after |
|---|---:|---:|
| `pytests/run.sh` | 268 | **268** (the new assertions joined the existing road test) |
| `compare.py` | 5184/5184, ops=145 | **5613/5613, ops=147** (+429 cases, +2 ops) |
| `compare.py --self-test` | 14 comparators | **15** — `_weight_norm_pair_check` |
| `verify_schemas.py` | 4380/4380 | **4386/4386** (+6) |
| sweep26 (shim) | 22/26 | **22/26** |

`_weight_norm_pair_check` is a new comparator rather than a reuse of
`_pair_result_check`: that one requires its second member to match *exactly*,
which is right for `max.dim`'s integer indices and wrong for a norm.

### 11.6 Sabotage check

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| W1 | **reduce the kept axis instead of the others** — `d == axis` for `d != axis`, the axis confusion a square weight cannot show | **5602/5613, 11 failed** | 1 FAIL | both `dim=0` and `dim=v.dim()-1` cases fire, on a non-square `v`. A square `v` would have made the two indistinguishable |
| W2 | **norm in the input dtype, result cast to `float32`** — right dtype, wrong number | **5609/5613, 4 failed** | **0 FAIL** | only the four reduced-float cases see it, and only against the oracle. This is the fault that was actually in the first version of the kernel |
| W3 | **`p = 0` as `sum(|x|^0)`** — the general formula, which counts every element because `\|0\|^0` is 1 | **5572/5613, 41 failed** | 1 FAIL | the `p=0` rows exist only because the `p` family was measured; `norm_except_dim` never passes 0, so nothing on the weight_norm path would have caught this |

W2 is the third fault this round that `run.sh` cannot see and golden can, and the
second that is a *precision* mistake wearing a dtype's clothes.

## 12. The legacy `torch.Tensor` constructor — the decision, re-examined

**Sweep after: 22/26 (unchanged). Nothing moved, and nothing was expected to:
this section is a verdict and a test, not a kernel.**

The brief for this round asked for a decision on "the legacy `torch.Tensor(int)`
uninitialised-storage constructor" and described a hardcoded refusal in
`tensor.rs`. **That refusal is no longer the one that is there.** §4 of this
document already decided that form — verdict: reproduce it — and implemented it.
Measured on this worktree before anything was changed:

```
torch.Tensor(3)        -> (3,)   float32          implemented
torch.Tensor(2, 3)     -> (2, 3) float32          implemented
torch.Tensor()         -> (0,)   float32          implemented
torch.Tensor(0)        -> (0,)                    implemented
torch.Tensor(-1)       -> RuntimeError "Trying to create tensor with negative
                          dimension -1: [-1]"     upstream's wording
torch.Tensor([3, 4])   -> NotImplementedError     STILL REFUSED
```

So the open question is the **build-from-data** form, and the answer below is
about that one.

### 12.1 Verdict: keep refusing `torch.Tensor([3, 4])`

Three grounds, and the second is the one that was measured rather than argued.

1. **The failure mode of refusing is a refusal, not a wrong answer.** This is the
   asymmetry that separates it from the size form. Not implementing
   `torch.Tensor(n)` stopped `sew_d` dead at construction with a clear message —
   a blocked architecture, which is why §4 implemented it. Not implementing
   `torch.Tensor([3, 4])` does exactly the same thing: it raises by name. Neither
   silently computes. So the whole question reduces to whether anything reaches
   it.

2. **Nothing in the 26-architecture sweep reaches it, and the two forms are used
   by different code.** Measured across `transformers/models` (82 occurrences of
   `torch.Tensor(`):

   | form | where it is called |
   |---|---|
   | size — `torch.Tensor(n)` | `sew_d`, `sew`, `hubert`, `wavlm`, `data2vec`, `mask2former`, `oneformer` — all in model `__init__`, all unconditional |
   | data — `torch.Tensor([...])` | `conditional_detr` (image **post-processing**, not a forward), `bit` (`torch.Tensor(np.linspace(...))`), `higgs_audio_v2_tokenizer` (`nn.Buffer(torch.Tensor([True]))`) |

   The size form is a construction-time idiom across a family of audio models —
   §4's ground 3, now counted. The data form's three callers are one
   post-processing path and two architectures outside this sweep. It is not
   unreachable *in principle* — it would block `bit` and
   `higgs_audio_v2_tokenizer` at construction — and that is where a future round
   should start, but this round does not add surface without a caller in the
   sweep it is measured against.

3. **The dangerous reading has to stay impossible.** `[3, 4]` looks exactly like
   a size list and is not one: `torch.Tensor([3, 4])` is a `(2,)` tensor holding
   `3.0` and `4.0`, **not** a `(3, 4)` empty one. An implementation that took a
   sequence as a shape would answer `(3, 4)` zeros where upstream answers two
   numbers, and nothing downstream would raise. So the refusal is the guard, and
   its message names *which* form it is refusing so that a future implementer
   cannot mistake one for the other. The test asserts on that specific wording
   (`"third form"`), not on the exception type alone.

`torch.tensor([3, 4])` — the lowercase spelling, which is what the intent
actually is — works, and is checked.

### 12.2 What is tested, given that the values cannot be

The brief's question, and it applies to the form that *is* implemented: the bytes
are arbitrary upstream (uninitialised) and zeros here, so **no golden case can
compare them**. `aten.empty.memory_format` has the same property and its
comparator is `_dtype_shape_only_check`, whose docstring is "there is no correct
value to diff".

What stands in for the values:

```
shape          (3,), (3,4), (2,3,4); TensorBase() is (0,) and not ()  -- measured
dtype          the default float, read at CALL time, so it follows
               set_default_dtype exactly as upstream's does
refusal        TensorBase(-1) raises with upstream's exact wording
the data form  refused, asserted by the specific message and not by type
the caller     sew_d's line verbatim -- TensorBase(32) then uniform_(0, 1),
               then every value asserted finite and in [0, 1)
independence   two calls must not share storage
```

**The independence check is the one this round added**, because it is the
property a caller actually depends on when it writes into the result, and every
other assertion above passes for a constructor that handed out one shared
buffer — including the `.uniform_()` check, since a shared buffer is still
filled with values in range.

It is checked by writing into one tensor and reading the other, rather than by
comparing `data_ptr()` (which `TensorBase` does not expose). And it was verified
to discriminate, using the re-wrap form as a positive control: two
`TensorBase(base)` re-wraps of one base genuinely *do* share, and the same
assertion run against them reads back `[2.0, 2.0, 2.0, 2.0]` where the
independent pair reads `[1.0, 1.0, 1.0, 1.0]`. An assertion about non-aliasing
that has never been shown to fail on a real alias is not evidence of anything.

## 13. The tail behind the four: `outer`, `ones_like`, `tile`, `detach`

**Sweep after each: 22/26 throughout.** These are not part of the brief's four.
They were taken because each was the *next* wall after one of the four, each
turned out to be a name over a kernel that already existed or a one-line fill,
and the acceptance test is architectures forwarding — so the honest thing was to
keep walking until the walls stopped being free.

They stopped being free after these four, and §14 says where.

| kernel | who was on it | what it cost | what moved |
|---|---|---|---|
| `torch.outer` | `sam3_video` | a spelling — fires `view` + `mul` | → `torch.ones_like` |
| `aten.ones_like.default` | `vits` **and** `sam3_video` | one branch in `zeros_or_empty_like` | `vits` → `torch.detach`, `sam3_video` → `TensorBase.tile` |
| `torch.detach` / `TensorBase.tile` | `vits` / `sam3_video` | one table entry; one spelling over `repeat` | `vits` → `torch.clamp_min`, `sam3_video` → `TensorBase.all` |

Three of the four are the pattern §6.3 named and this round hit twice more:
**the wall behind a kernel turned out to be a name.** `torch.detach` is the
starkest — `aten.detach.default` has been implemented the whole time and had no
`overloads.json` entry, so `torch.detach(x)` refused while `x.detach()` worked.

### 13.1 The two that are not aliases, and are easy to think are

**`tile` is not `repeat`.** They differ in exactly one place, and it is the
place a careless implementation would not test:

```
x is (2, 3)
x.repeat(2)   REFUSES   -- repeat needs at least as many dims as the rank
x.tile(2)     (2, 6)    -- dims are LEFT-padded with 1s to the rank
```

Too *many* dims behaves identically on both (the extras become new leading
axes), so a case set built from `len(dims) >= rank` cannot tell them apart at
all. And the padding is on the **left**: `x.tile(2)` on a `(2,3)` is `(2, 6)`,
not `(4, 3)` — sabotage T2 is that one flip, and it produces a correctly-ranked,
plausible tensor.

**`ones_like` is the only one of its three siblings whose values can be
diffed.** `zeros_like` and `empty_like` share `_dtype_shape_only_check` because
upstream's `empty` bytes are undefined; `ones` is defined to be ones. So its
golden cases run the default pipeline, and sabotage T1 — the fill falling
through to the `zeros` branch, which is what reusing a sibling by accident does —
fails on values across 20 cases while the dtype and shape stay right.

### 13.2 Counts

| gate | before §13 | after |
|---|---:|---:|
| `pytests/run.sh` | 268 | **268** |
| `compare.py` | 5613/5613, ops=147 | **5634/5634, ops=148** (+21 cases, +1 op) |
| `verify_schemas.py` | 4386/4386 | **4392/4392** (+6) |
| schema identities | 228 | **230** (+2) |
| decomposition registry | 1006 | **1007** (+1) |
| sweep26 (shim) | 22/26 | **22/26** |

The identity count is **+2, not +3**, and which entry contributed is the check:
`ones_like` is in neither table before this and adds two, while **`detach` was
already in `methods.json`** — putting it in `overloads.json` gives `torch.detach`
a second door onto a schema `Tensor.detach` already named, and a second spelling
of one schema is one identity. Getting +3 would have meant the two tables had
transcribed `detach` differently.

The registry moving by one is `zeros_like.out`'s mechanism for the third time:
`aten::ones_like.out` is torchgen-generated and the yaml does not declare it, so
`register_decomposition(aten.ones_like)` has a schema to resolve and reaches one
more overload. `registry_default` does not move, because the new one is `.out` —
which is the check that this is that mechanism and not a regression of the
`["default"]` bug. `detach` moves neither, because `aten::detach` has no `.out`.

### 13.3 Sabotage check

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| T1 | `ones_like` falls through to the **zeros** branch | **5614/5634, 20 failed** | 1 FAIL | the values are the check; dtype and shape stay right throughout |
| T2 | `tile` pads its dims on the **right** | **5634/5634, 0 failed** | 1 FAIL | golden cannot see a spelling at all — no new dispatch key exists for it |
| T3 | `outer` reshapes **vec2** instead of self — the transposed product | **5634/5634, 0 failed** | 1 FAIL | same, and the case uses `(3,)` against `(4,)` so the transposition is visible; equal lengths would have hidden it |

T2 and T3 are the clearest statement this round of why the road test exists:
**golden compares by dispatch key and is structurally blind to a name.** Two
faults that change every value a caller sees leave it entirely green.

## 14. Where the sweep stands, and what is in front of each architecture

**22/26, unchanged across all eight changes — and every one of the four
remaining architectures moved.** The count is the acceptance test and it did not
move, so this table is the result:

| architecture | wall at the start of this round | wall now | walls crossed |
|---|---|---|---|
| `vits` | `norm_except_dim` (as a `TypeError`) | `torch.clamp_min` | **4** |
| `sew_d` | the same | `torch.group_norm` | **2** |
| `zoedepth` | `torch.conv_transpose2d` | `aten.upsample_bilinear2d` | **1** |
| `sam3_video` | `aten.div.Scalar_mode` | `TensorBase.all` | **4** |

The original twenty still pass, and `deberta`/`deberta_v2` — the two §8.1 closed
— still pass.

**Every remaining wall is on §8.2's own "genuinely still missing" list.** That is
the useful result: this round did not uncover a new category of gap, it walked
four architectures through eleven known ones. What is in front of them now:

```
vits         aten.clamp_min.{default,Tensor}       one-line, comparison family
sew_d        aten.native_group_norm.default        a real kernel, own round
zoedepth     aten.upsample_bilinear2d.default      a real kernel, own round
sam3_video   aten.all.default                      one-line, reduction family
```

`vits` and `sam3_video` are each one cheap kernel from their *next* wall, not
necessarily from forwarding — the pattern of this round is that they have long
tails of one-line gaps. `sew_d` and `zoedepth` are each blocked on a genuine
kernel with its own measurement round, and those two are where the next
architecture-closing work is.

Also still missing, unchanged from §8.2 and untouched here:

```
aten.sigmoid.default   aten.sign.default    aten.erf.default    aten.log2.default
aten.flip.default      aten.leaky_relu.default   aten.randn_like.default
aten.div.Tensor_mode / Scalar_mode   <- CLOSED by §9
aten.norm.ScalarOpt_dim              <- CLOSED by §11
aten._weight_norm_interface.default  <- CLOSED by §11
aten.ones_like.default               <- CLOSED by §13
aten.avg_pool2d.default   aten.fmod.{Tensor,Scalar}
aten.remainder.Scalar_Tensor (and with it `__rmod__`)
aten.set_.source_Tensor_storage_offset
```

`TensorBase.flip` was found missing on the way (§10's road test had to build its
rearranged weights from index arithmetic instead), which is a second caller for
`aten.flip.default`.

### 14.1 Final gates

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   268 ok, 0 FAIL                    exit 0
$PY tools/golden/compare.py                 5634/5634, ops=148, pending 1     exit 0   (was 4709/4709, ops=143)
$PY tools/golden/compare.py --self-test     15 comparators x 11 fault modes   exit 0   (was 14)
$PY rust/torch_c/pytests/verify_schemas.py  4392/4392                         exit 0   (was 4376/4376)
sweep26 (shim)                              22/26                             exit 0   (was 22/26)
```

Prefill digests on the final artefact, all six unchanged from docs/numerics/SEQLEN.md
§1.3 and §8.8:

```
f32   S=6 b9fc5553ee1bf6a2   S=32 331668f36da02f21   S=128 00159a9dbd308eda
      S=512 07c2797dabc4552e   S=1024 eda1e173727bb7f5
bf16  S=128 7ff8e9334449b147
```

---

# Round three: closing the last four

Same method as the two rounds above — **the sweep is re-run after every
kernel**, never batched, and each section records the count and, when the count
does not move, which wall replaced the one that was removed. Written
incrementally.

The starting point, re-measured on this worktree before any edit and identical
to §14.1's:

```
pytests/run.sh                268 ok, 0 FAIL                        exit 0
tools/golden/compare.py       5634/5634, ops=148, pending=1         exit 0
compare.py --self-test        15 comparators x 11 fault modes       exit 0
verify_schemas.py             4392/4392                             exit 0
sweep26 (shim)                22/26                                 exit 0
```

with the four walls §14 predicted, each confirmed by running it:

| architecture | wall at the start of this round |
|---|---|
| `vits` | `torch.clamp_min(...)` — no overload table entry |
| `sew_d` | `torch.group_norm(...)` — no overload table entry |
| `zoedepth` | `torch._C._nn.upsample_bilinear2d` |
| `sam3_video` | `TensorBase.all` |

### A harness note, because it cost two runs

`run.sh` refused twice with `cmp exited 137`, and it was **not** the memory
pressure its own message suggests. The kill was deterministic and one-sided:
`cmp` on the *staged* copy under `$TMPDIR/torch-c-stage/` was SIGKILLed against
any second file including itself, while the vendored copy compared against
itself fine and `sha256` said all three copies (cargo's, the stage's, the
vendored one's) were the same bytes. Deleting the stage directory and letting
`run.sh` recreate it fixed it permanently. So the guard is right that 137 is
not a staleness report — but "re-run when quieter" is not the only remedy, and
`rm -rf "$TMPDIR/torch-c-stage"` is the one that worked here.

## 15. `aten.clamp_min.default` — `vits`

`modeling_vits.py:1352` keeps the predicted waveform at least one frame long:

```python
predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()
```

measured `clamp_min.default(float32(1,), 1)`.

### 15.1 It is `clamp(min=)`, and that was checked rather than assumed

The cheap move is to write `clamp_min` as a floor and move on. The risk
KERNELS26.md keeps hitting is the opposite one — a rule that *looks* shared and
is not — so all ten rows of `clamp`'s dtype ladder were re-measured against
`clamp_min` on 2.13.0 before anything was reused:

```
                       clamp_min(t, b)     clamp(t, min=b)
int32,   0             int32               int32
int32,   2.0           float32             float32
uint8,   2             uint8               uint8
uint8,   2.0           float32             float32
float16, 2.0           float16             float16
bool,    0             int64               int64
bool,    0.0           float32             float32
bool,    False         RAISES              RAISES
f32 [nan,1,-1], 0.0    [nan,1,0]           [nan,1,0]
int64,   -1            int64               int64
```

Identical on every row, so the kernel calls `clamp_result_tag` and
`clamp_values` rather than restating a table the golden cases had to correct
once (§ the `clamp`/`clamp_` split: the out-of-place form promotes where the
in-place one refuses).

**One thing is not shared, and it is the row a delegating implementation would
get wrong invisibly.** The `bool`-bound refusal names the kernel upstream
failed to find, and that name differs:

```
torch.clamp_min(bool_t, False)   "clamp_min_scalar_cpu" not implemented for 'Bool'
torch.clamp(bool_t, False)       "clamp_scalar_cpu"     not implemented for 'Bool'
```

The golden case for that row is `both_error`, which by design does not compare
messages — so a `clamp_min` written as a straight call through to `clamp` would
be **green in golden and wrong in the string a user reads**. That is what
`test_clamp_min_names_its_own_kernel_in_the_bool_refusal` in `test_shim.py` is
for; `clamp_result_tag` derives the name from `op` so a third caller cannot
forget to say which it is.

`min` is required here where `clamp`'s is optional, which removes `clamp`'s
"both bounds absent is an error, not a no-op" branch entirely — no spelling
reaches it, because `None` cannot bind `Scalar min`.

### 15.2 The plausible wrong implementations, and the cases that separate them

| wrong implementation | what it gets right | the case that kills it |
|---|---|---|
| "a floor is `maximum`, so promote like a binary op" | every same-dtype row | `clamp_min(float16, 2.0)` → binary promotion says `float32`, upstream says `float16` |
| "clamp both ends" (copy-paste from `clamp`) | everything with no large elements | `[1,5,10,-3]` floored at 2 → the `10` must survive |
| "a floor is a relu with an offset" | every non-negative floor | `clamp_min(int64 [1,5,-3,-9], -1)` → `[1,5,-1,-1]`, a relu gives `[1,5,0,0]` |
| `where(x > min, x, min)` | every finite input | `[nan,1,-1]` floored at 0 → upstream keeps the `nan`; that spelling returns `0` |

### 15.3 The deliberate gap

`clamp_min.Tensor` (a tensor floor) is **not** implemented. It is `maximum`
with broadcasting and binary promotion, and this shim has no
`aten.maximum.default` to delegate to, so it is a broadcast kernel and not a
one-line alias. Both tables list it so `torch.clamp_min(x, some_tensor)`
refuses by the name of the overload it needed —

```
NotImplementedError: aten op not implemented in torch._C shim: aten.clamp_min.Tensor
```

— exactly the shape `clamp.Tensor` already had beside it. A `c_error` golden
case watches it, so the day it gains a kernel this fails and gets promoted
rather than quietly starting to compute something unchecked.

`out=` is refused by the resolver on both spellings (`no matching overload ...
for (Tensor, int, out=Tensor)`), so no `.out` schema is listed — same as
`clamp`.

### 15.4 A defect in the harness, found by the first golden run

Three of the new spelling cases failed with
`c raised AttributeError("module '_C' has no attribute 'clamp_min'")`. The
free-function spelling lives on `_C._VariableFunctions`, not on `_C` — `torch`
hoists it, `_C` does not. The existing `_log_member_cases` already worked
around that inline; this round extracted it as `_free(module, name)`.

**It is worth naming because of the direction the mistake points.** The
`match` cases failed loudly, which is the harness working. But the `c_error`
case for `clamp_min.Tensor` **passed** — an `AttributeError` is an error, so
"c refused" was satisfied by a typo rather than by the overload under test.
A `c_error` case reached through a name that does not exist cannot fail, which
is §0's "a validation that cannot fail is not a validation" in a new place.

### 15.5 Counts

| gate | before §15 | after |
|---|---:|---:|
| `pytests/run.sh` | 268 | **269** (+1) |
| `compare.py` | 5634/5634, ops=148 | **5656/5656, ops=149** (+22 cases, +1 op) |
| `verify_schemas.py` | 4392/4392 | **4399/4399** (+7) |
| schema identities | 230 | **232** (+2) |
| sweep26 (shim) | 22/26 | **22/26** |

The identity count is **+2, not +4**: `clamp_min` went into both tables with
the same two schemas in the same change, so `torch.clamp_min` and
`Tensor.clamp_min` are one identity per overload — the `amax`/`tril` shape
again. Getting +4 would have meant the two tables had transcribed it
differently.

### 15.6 The sweep

**22/26. `vits` moved one wall**, from `torch.clamp_min` to `torch.flip`:

```
vits         FAIL  torch.flip(...) -- overload resolution has no table entry for this op
```

which is the next entry on §14's own list. Note that §14 recorded
`aten.flip.default` as *missing*, while this round's brief described it as
existing with only a `TensorBase.flip` spelling absent. **§14 is the correct
one** — `_aten_implemented()` has no `aten.flip.default`, so `flip` is a kernel
and not a name.

## 16. `aten.all.{default,dim,dims}` — `sam3_video`, and two rules `any` had wrong

`masking_utils.py:330` asks `padding_mask.all()` before it will skip building a
bidirectional mask. §14 called this "one-line, reduction family". It is one
line **in the reduction**, and the family it was going to be copied from had
two defects that a flipped comparison would have inherited.

### 16.1 What "flip `max` to `min`" would have got wrong

`any` reduces the 0/1 mask with `max`; `all` reduces it with `min`. That is the
whole computation. The two things that are *not* symmetric:

**Empty input.** `torch.tensor([]).any()` is `False`; `torch.tensor([]).all()`
is `True`. `any_default` had a hardcoded early return of zero for
`elem_count() == 0` — correct for `any`, and the exact wrong answer for `all`
had it been shared verbatim. Now `BoolReduce::identity()` names it, one value
per op, in the one place both read.

Over a *dimension* it is worse than a wrong value, because candle refuses the
reduction outright:

```
before:  torch.zeros(0,3).any(0)   RuntimeError: candle: empty tensor for reduce
upstream:                          [False, False, False]
         torch.zeros(0,3).all(0)   [True, True, True]     three trues out of nothing
         torch.zeros(0,3).all(1)   []                     the surviving axis is 0-long
```

One shape, two dims, two different answers — which is why the kernel computes
the reduced shape and fills it with the identity, rather than short-circuiting
on "the input has no elements". A short-circuit gets `dim=0` right and `dim=1`
wrong.

**Result dtype.** The result is `torch.bool` for every input dtype **except
`uint8`, where it is `uint8`**. This is not symmetry-guessing: upstream's own
`torch.all` docstring states it ("matches the behaviour of NumPy in returning
output of dtype `bool` for all supported dtypes except `uint8`"), and it is
measured on both ops and all three forms.

**`any` had this wrong** — it returned `torch.bool` unconditionally, and its
golden cases probe only `int64`, which is exactly the dtype that cannot tell
the rule from "always bool". So the defect was invisible and had been for as
long as `any` existed. Fixed here, with `all`, rather than after it: writing
`all` from `any`'s shape would have copied it into a second op, and then two
ops would have agreed with each other and not with upstream — which is the
failure mode §0 calls "two case sets that both pass two different
implementations", arrived at from the other direction.

### 16.2 The plausible wrong implementations

| wrong implementation | what it gets right | what separates it |
|---|---|---|
| `any` with `max`→`min` and nothing else | every non-empty case | `[]` → `True` upstream, `False` from a shared early return |
| "the result is always `bool`" | six of the seven dtypes | `uint8` in, `uint8` out — on all three forms |
| "every element is **positive**" | every case built from 0/1 mask data, which is what a mask is | `[-1,-2,-3,-4].all()` is `True` |
| "NaN is not true" | every finite input | `[nan, 1.].all()` is `True` — `nan != 0` |
| short-circuit on `elem_count() == 0` | `zeros(0,3).all(0)` | `zeros(0,3).all(1)` is `[]`, not `[True,True,True]` |

The dtype and the empty rules are stated **once** and fed to *both* ops
(`_bool_reduce_dtype_cases`, `_bool_reduce_empty_cases`,
`_bool_reduce_dim_cases` in `cases.py`). That is deliberate: a rule written
once and applied to the pair is what stops them drifting apart a second time.

### 16.3 Counts

| gate | before §16 | after |
|---|---:|---:|
| `pytests/run.sh` | 269 | **270** (+1) |
| `compare.py` | 5656/5656, ops=149 | **5784/5784, ops=152** (+128 cases, +3 ops) |
| `verify_schemas.py` | 4399/4399 | **4415/4415** (+16) |
| schema identities | 232 | **238** (+6) |
| sweep26 (shim) | 22/26 | **22/26** |

**+3 ops, not +1**: `all.dim` and `all.dims` are separate overloads with
separate kernels and separate builders, and `all.dims` is *not* parked in
`IMPLEMENTED_AWAITING_GOLDEN` the way `any.dims` is — parking it would have
left the one form whose `dim=None` means "every axis" (rather than "argument
missing") uncompared.

**+6 identities**: `any`'s exact inventory one op over — three kernels and
three `.out` siblings carried with no kernel so `torch.all(x, out=y)` refuses
by the right name. 128 of the new cases are the shared dtype/empty/dim
builders, and roughly half of them land on `any`, which is where the
regression coverage for the dtype fix lives.

### 16.4 The sweep

**22/26. `sam3_video` moved one wall**, from `TensorBase.all` to
`TensorBase.sigmoid` — the first entry on §14's "also still missing" list.
`vits`, `sew_d` and `zoedepth` are unchanged on `torch.flip`,
`torch.group_norm` and `upsample_bilinear2d`.

## 17. `aten.sigmoid.default` — `sam3_video`, and a tolerance that could not see the fault

`sam3_video`'s next wall after `all` was `TensorBase.sigmoid`. §14 filed
`sigmoid` under "one-line members of families that already exist". It is a
one-line member of **two** families and they disagree about it.

### 17.1 Two families, and which half each one supplies

```
dtype rule       unary_float's   int/bool in -> default float out     (silu REFUSES)
precision rule   silu's          f16/bf16 computed in f32, narrowed once
```

`silu` has no integral CPU kernel upstream and raises on one; `sigmoid`
promotes (measured: `int64`, `int32`, `uint8` and `bool` all give `float32`).
So it cannot be a copy of `silu`. And the `unary_float` family evaluates in the
input's own dtype, which is right for `tanh`/`exp`/`cos` and **wrong here**. So
it cannot be another `Unary` variant either. It is its own function because the
two rules it needs come from different places.

The precision half is measured, not reasoned from `silu`'s comment. Over 20 000
`randn * 8` inputs, against upstream's own answer:

```
              computed in f32, narrowed once     computed in the reduced dtype
float16       0 / 20000 differ                   6983 / 20000 differ
bfloat16      0 / 20000 differ                   5466 / 20000 differ
```

### 17.2 The tolerance could not see it, and the harness said so

Every one of those 6983 disagreements is **1 ULP**, which at `float16` is about
5e-4 relative — and golden's `float16` tolerance absorbs it completely. So the
wrong implementation here is not merely plausible, it is *green*: writing
`sigmoid` as a `Unary` variant passes every ordinary case at every dtype.

That is §0's third trap ("a tolerance that cannot see the fault") arriving in
this round, and it needed the same answer the f32 precision cases needed:
compare something the tolerance does not soften. `_bitwise_equal_check`
compares the results **bit for bit** at `float16` and `bfloat16`.

**The comparator then caught a defect in itself.** Its first draft compared
only the nested value lists, and `compare.py --self-test` reported it accepting
an injected *shape* fault and an injected *dtype* fault:

```
PROBLEM: _bitwise_equal_check + shape: the comparator accepted a wrong answer
PROBLEM: _bitwise_equal_check + dtype: the comparator accepted a wrong answer
```

A `value_check` **replaces** the default pipeline rather than extending it, so
a comparator that only looks at values is blind to everything else. It is now a
superset — dtype, then shape, then bit-exact values — and catches 7 of 11
injected fault modes, the same as `_signed_zero_check`, which is the model it
should have been written from.

### 17.3 What f32 and f64 do, and why they are not bit-compared

They are **not** bit-identical, and the residual is not this kernel's:

```
aten.exp.default vs upstream, 80 points in [-5, 5]:   12/80 differ at f32,  16/80 at f64
aten.sigmoid.default vs upstream, the same 80 points:  the SAME indices
```

Upstream computing `1/(1+exp(-x))` with its own `exp` reproduces
`torch.sigmoid` exactly at both widths (0/20000 differ), so the formula is
right; what differs is candle's `exp` against upstream's vectorised one, ~1 ULP.
Demanding bit-equality at `f32` would be demanding it of `exp`, which no other
case in this repository does. Recorded rather than hidden — and the index
agreement is what makes "inherited" a measurement rather than an excuse.

Widening `f32` to `f64` and narrowing makes it **worse** (20 of 80 differ), so
`f32` is computed in `f32`. That is the opposite of the reduced-dtype rule
above, which is why both were measured rather than one inferred from the other.

### 17.4 Counts

| gate | before §17 | after |
|---|---:|---:|
| `pytests/run.sh` | 270 | **270** |
| `compare.py` | 5784/5784, ops=152 | **5830/5830, ops=153** (+46 cases, +1 op) |
| `compare.py --self-test` | 15 comparators | **16 comparators** x 11 fault modes |
| `verify_schemas.py` | 4415/4415 | **4421/4421** (+6) |
| schema identities | 238 | **240** (+2) |
| core-tagged ops | 90 | **91** (+1) |
| sweep26 (shim) | 22/26 | **22/26** |

The core-tagged count is **+1 across five kernels landed so far this round**,
and the four that do not appear are the check that the tags are read one at a
time rather than assumed:

```
sigmoid.default    ['core', 'pointwise', 'pt2_compliant_tag']   <- counted
clamp_min.default  ['pointwise', 'pt2_compliant_tag']
all.default        ['pt2_compliant_tag', 'reduction']
all.dim            ['pt2_compliant_tag', 'reduction']
all.dims           ['pt2_compliant_tag', 'reduction']
```

`clamp_min` is not core while `clamp` is, and none of the three `all` overloads
is core while `amax` is. Neither is derivable; both were read.

### 17.5 The sweep

**22/26. `sam3_video` moved one wall**, and to a wall of a kind this round had
not seen:

```
sam3_video   FAIL  aten.div.Tensor: dtype promotion not implemented in torch._C shim: float32 vs int64
```

That is not a missing name or a missing kernel — `aten.div.Tensor` has had a
kernel since the beginning. It is `same_dtype` refusing a **mixed-dtype
operand pair**, which docs/bindings/BIND.md §9 records as a deliberate divergence.
`vits`, `sew_d` and `zoedepth` are unchanged.

## 18. `aten.flip.default` — `vits`

`modeling_vits.py:595` reverses the channel order of the residual coupling
layer's input on every flow step: `torch.flip(inputs, [1])`.

**Correction to this round's brief.** It described `aten.flip.default` as
already existing with only a `TensorBase.flip` spelling missing. §14 of this
same document said the opposite, and §14 is right — `_aten_implemented()` had
no `flip` at all, and the sweep's own error was "overload resolution has no
table entry". Both the kernel and both spellings land here.

### 18.1 What it is, and the one thing it is not

**It copies.** `torch.flip(x, [0]).data_ptr() != x.data_ptr()`, measured. That
is a piece of luck rather than a design choice: a reversed axis is a negative
stride, which is exactly what candle's `Layout` cannot express, so an op that
*had* to alias would have joined `slice.Tensor` and `view.dtype` as a
docs/kernels/VIEWS.md §6.4 divergence. It does not, so this is complete rather than
recorded.

Four rules, all measured on `x = arange(6).reshape(2, 3)`:

```
flip(x, [1])      [[2,1,0],[5,4,3]]    reverses WITHIN each row
flip(x, [0])      [[3,4,5],[0,1,2]]    reverses the row ORDER
flip(x, [-1])     [[2,1,0],[5,4,3]]    a negative dim is the LAST axis
flip(x, [])       [[0,1,2],[3,4,5]]    an empty dim list is a COPY, not an error
flip(x, [0, 0])   RAISES               "dim 0 appears multiple times in the list of dims"
```

**The duplicate refusal is the one a delegating implementation loses.**
Flipping one axis twice is the identity, so a kernel that just loops over
`dims` returns *the input* where upstream raises — a correctly-shaped,
correctly-typed, entirely plausible tensor. And the check has to run **after**
normalisation, because `[1, -1]` is the same axis twice on a rank-2; a check
written against the raw list sees two different numbers. Both spellings are
cased.

Nothing in the case set is square, on purpose: `flip([0])` and `flip([1])` on a
2×2 of distinct values differ in their values but not in their shape, and an
axis mix-up is much easier to see on `(2, 3)` and `(2, 3, 4)`.

### 18.2 The defect only the spelling cases could find

The first build passed every dispatch-key case and failed all six spelling
cases with:

```
c raised TypeError("aten.flip.default: missing required argument 'dims'")
but torch computed a value
```

Every reduction in `aten.rs` calls its axis argument **`dim`**; `aten::flip`
calls its **`dims`**, and `reduce_dims` had the name hardcoded. The resolver
binds by the *schema's* name, so `torch.flip(x, [1])` and `x.flip(1)` both
arrived with a keyword the kernel was not looking for — while
`_aten_dispatch(op, t, [1])` passes the list **positionally** and finds it
either way.

So the kernel was simultaneously green on 60 dispatch-key cases and unusable
from every spelling a caller has. This is §13.3's finding from the other side:
there, golden was blind to a name; here, golden's dispatch path was blind to an
*argument* name, and only the member/free-function cases could see it.
`reduce_dims_named` now takes the keyword, and `reduce_dims` is a two-line
wrapper that supplies `"dim"`.

### 18.3 Counts

| gate | before §18 | after |
|---|---:|---:|
| `pytests/run.sh` | 270 | **271** (+1) |
| `compare.py` | 5830/5830, ops=153 | **5892/5892, ops=154** (+62 cases, +1 op) |
| `verify_schemas.py` | 4421/4421 | **4427/4427** (+6) |
| schema identities | 240 | **242** (+2) |
| core-tagged ops | 91 | **92** (+1) |
| decomposition registry | 1007 | **1008** (+1) |
| sweep26 (shim) | 22/26 | **22/26** |

The registry moving is the `zeros_like.out`/`ones_like.out` mechanism for the
fourth time, and **which** entry moved it is the check: the yaml declares
`clamp_min.out`, `sigmoid.out`, `all.out`, `all.dims_out` and `all.all_out`
itself — every other `.out` schema this round added — and does **not** declare
`flip.out`. So `flip.out` becomes the eighth table-only entry and the other
five move nothing. Grepped in the file rather than inferred from the count
having moved by one.

### 18.4 The sweep

**22/26. `vits` moved one wall**, off the missing-name class entirely:

```
vits   FAIL  TypeError: 'IntTensor' object is not subscriptable
    modeling_vits.py:363   fused_add_tanh_sigmoid_multiply(..., num_channels_tensor[0])
```

`TensorBase.__getitem__` refusing an integer index on a rank-1 tensor. Not a
kernel and not an overload entry — a `bootstrap.py` surface gap. `sew_d`,
`zoedepth` and `sam3_video` are unchanged.

## 19. `aten.native_group_norm.default` — `sew_d`

The first of the brief's two real kernels. `nn.GroupNorm.forward` →
`F.group_norm` → `torch.group_norm`, which is `CompositeImplicitAutograd`;
measured with a `TorchDispatchMode` logger on 2.13.0, all three of
`torch.group_norm(...)`, `F.group_norm(...)` and an `nn.GroupNorm` forward emit

```
aten.native_group_norm.default
```

and nothing else. So `torch.group_norm` is installed as a composite in
`bootstrap.py` beside `layer_norm`, not as an `overloads.json` entry — the
table would name `aten.group_norm.default`, a key upstream's own dispatcher
never answers to.

### 19.1 Three results, and only the first is read

`torch.group_norm` returns `result[0]`. **The `mean` and `rstd` beside it can
be the wrong shape, the wrong dtype, or a different function entirely and every
model in the sweep still forwards.** That is the brief's warning about this
kernel and it is exactly right, so each of the three was measured separately:

| result | what it is | the wrong answer with the same shape |
|---|---|---|
| `out` | `(x - mean) * rstd` reshaped, then per-channel `weight`/`bias` | applying the affine along the *statistics* view |
| `mean` | `(N, group)` — per (sample, group) | `(N, C)` per channel; keepdim-shaped like `native_layer_norm`'s |
| `rstd` | `1/sqrt(var + eps)`, **biased** variance | `sqrt(var+eps)`; `1/(sqrt(var)+eps)`; the unbiased variance |

### 19.2 The two views, and why `C=6, group=3`

The statistics are taken over `(C/group) * HxW` elements per row — the tensor
read as `(N*group, C/group*HxW)`. The affine parameters are per **channel**,
shape `(C,)`, applied to the normalised tensor reshaped back to `(N, C, HxW)`.

Those are different views, and folding them into one is the plausible error.
**It is invisible in the two configurations a hand-written test picks first:**

```
group == C    InstanceNorm       C/group == 1, so a "group" IS a channel
group == 1    LayerNorm over CHW  one group, so the group view IS the whole tensor
```

Both coincide. Every value case here uses `C=6, group=3` (`C/group == 2`) so
that they do not, and `group=1`/`group=6` are present as the *control* — a
kernel that passes only those is the one being guarded against. The
`test_shim.py` case zeroes `weight[0]` and asserts that exactly channel 0's row
vanishes, which no shape or dtype check can substitute for.

### 19.3 What pins `eps`, and what cannot

A **constant** group has variance zero, so:

```
rstd = 1/sqrt(0 + eps)      316.2278    <- upstream, measured at eps=1e-5
       1/(sqrt(0) + eps)    100000      <- eps added to the std instead
       sqrt(0 + eps)        0.00316     <- rstd is not a reciprocal
```

Three different numbers of the same shape and dtype. But the constant case
**cannot** pin the divisor: an unbiased variance over a constant group is also
zero, so it gives `316.2278` too. And a random case cannot pin the `eps`
placement, because `eps` is negligible against a real variance. So both are
here, plus a third at `eps=0.5` on real data where the placement is visible
without being degenerate. No one of the three does the job of the other two.

### 19.4 The dtype rule, and the refusals

`mean`/`rstd` follow the **parameter** dtype, not the input's — a `float16`
input with `float32` parameters gives `float32` statistics and a `float16`
output. That combination is *supported*, not an error, and it is the row where
a kernel that tagged the statistics with the input dtype passes everything
else. `float32` input with `float16` or `float64` parameters raises. All
measured; the rule is `native_layer_norm`'s, and it was re-measured here rather
than assumed from the neighbour.

Upstream's refusals, transcribed with their own wording: the integral/bool
`"GroupNormKernelImpl"`, the divisibility message, the `X.numel() == N*C*HxW`
message, the weight-length message, and `Expected num groups to be greater than
0`. The divisibility check runs **before** the count check — measured, because
a wrong `C` that happens to be indivisible reports the first message and not
the second.

A **negative `eps` is not refused**: it gives NaN where `var + eps < 0` and a
finite answer elsewhere, and this follows rather than guarding, as
`native_layer_norm` does.

**Not implemented: `HxW == 0`.** Upstream answers `mean=0` with `rstd=nan` —
one half of the pair reporting an empty reduction and the other not. That is
the same internally-inconsistent corner `native_layer_norm` refuses for a
zero-extent `normalized_shape`, refused here for the same reason and watched by
a `c_error` case. `N == 0` **is** implemented: every result is simply empty and
there is nothing inconsistent to reproduce.

### 19.5 Counts

| gate | before §19 | after |
|---|---:|---:|
| `pytests/run.sh` | 271 | **272** (+1) |
| `compare.py` | 5892/5892, ops=154 | **5941/5941, ops=155** (+49 cases, +1 op) |
| `verify_schemas.py` | 4427/4427 | **4430/4430** (+3) |
| schema identities | 242 | **242** (unchanged) |
| core-tagged ops | 92 | **93** (+1) |
| sweep26 (shim) | 22/26 | **22/26** |

**The identity count does not move, and that is the check.** `native_group_norm`
went into neither `overloads.json` nor `methods.json` — it is reached through a
`bootstrap.py` composite and through `_aten_dispatch`, neither of which is a
schema table. `verify_schemas` still gains 3, because it counts every table
entry it can re-derive and `torch.group_norm` is a name it now knows about.
An identity count of 244 here would have meant the composite had been written
as an overload entry by mistake.

### 19.6 The sweep

**22/26. `sew_d` moved one wall**, from `torch.group_norm` to
`torch.avg_pool1d` — `aten.avg_pool2d.default`'s 1-D sibling, and §8.2's list
had `avg_pool2d` on it as "a real kernel with its own measurement round".
`vits`, `zoedepth` and `sam3_video` are unchanged.

## 20. `aten.upsample_bilinear2d.default` — `zoedepth`

The second of the brief's two real kernels, and the one where the wrong answer
is **a plausible image**.

`F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=...)` →
`torch._C._nn.upsample_bilinear2d`, whose four-argument signature is the
**`.vec`** schema. `.vec` is `CompositeImplicitAutograd`; measured with a
`TorchDispatchMode` logger on 2.13.0, a `(1,1,2,3)` input emits exactly one
record —

```
aten.upsample_bilinear2d.default((1,1,2,3), [4, 6], False, 2.0, 2.0)
```

— a *concrete* output size with the scale factors passed through beside it. So
`.vec` is a `bootstrap.py` composite in `_install_nn` and this is the leaf.

### 20.1 The two grids, and where they agree

```
align_corners=true    scale = (in-1)/(out-1)   [0 if out == 1]
                      src   = scale * d
align_corners=false   scale = 1/scale_arg  if given and > 0, else in/out
                      src   = max(scale * (d + 0.5) - 0.5, 0)
```

Both values are used in the wild — `zoedepth`'s own config carries an
`align_corners` flag — and they are different functions, not a tolerance apart.
On `arange(6).reshape(1,1,2,3)` → `(4,6)` they disagree on **20 of 24
elements**:

```
align_corners=false   0.00 0.25 0.75 1.25 1.75 2.00 | 0.75 1.00 ...
align_corners=true    0.00 0.40 0.80 1.20 1.60 2.00 | 1.00 1.40 ...
```

**They agree at the four corners.** That is what the flag means, and it is
exactly why a case set assembled from corners cannot separate them — nor can a
symmetric input, nor a square one. Every case here uses a non-symmetric ramp on
a non-square `(1,1,2,3)`, and every geometry is run through *both* values of
the flag.

The `+0.5 … −0.5` is the **half-pixel** convention the brief names, and
dropping it is the classic error: `scale * d` under `align_corners=false`
produces a slightly-shifted image rather than an error, and it is the
`align_corners=true` formula, so an implementation that used one convention for
both flag values still answers something for every input.

### 20.2 Three things inside the grid, each measured on its own

  * **`scales_h`/`scales_w` are honoured and are not `in/out`.** `1/scale` and
    `in/out` coincide whenever `out == in * scale` exactly, which is *every*
    case a `scale_factor=2` test produces. With `in=3, out=4, scales_w=1.5`
    they are `0.667` against `0.75`, and upstream answers
    `[0, 0.5, 1.1667, 1.8333]` rather than `[0, 0.625, 1.375, 2]`.
  * **a non-positive scale is ignored**, falling back to `in/out` — measured
    with `0.0` and `-1.0`, both giving the no-scale answer. A kernel that
    divided by them gives `inf` or a mirrored grid.
  * **`align_corners=true` ignores the scales entirely** — measured with
    `scales_w=9.0`, which changes nothing.

And the short circuit: **when `out == in` on an axis the axis is copied**, with
no grid at all. Not "the grid happens to be the identity" — measured, `out ==
in` with `scales_w=0.5` still copies, where the grid would have resampled.

### 20.3 Precision, and the direction that is not obvious

`opmath_t` is `f32` for `float16`/`bfloat16`/`float32` and `f64` for
`float64`. Both halves measured, and they point opposite ways:

```
float16 computed in f32 and narrowed once    0 of 143 differ from upstream
float16 computed in f64 and narrowed once    2 of 143 differ
float32 computed in f64 and narrowed once  241 of 286 differ
```

So this is **not** "compute as wide as possible". `float32` has to be computed
in `float32`, which is why the kernel casts through `f32` explicitly rather
than staying in the `f64` that `read_flat` hands over — the shape of bug the
brief's `_weight_norm_interface` note describes, met from the other side.

### 20.4 The `uint8` gap, and how it was decided

Upstream **computes** `uint8`, and not by rounding a bilinear result. Over 60
random shapes (5584 elements), `round-half-away-from-zero` applied to the
`float32` answer disagrees with upstream's `uint8` answer on **355** of them —
94% right, which is exactly the kind of number that passes a small case set.
Upstream runs a separate fixed-point kernel there. Refused by name, with a
`c_error` case watching it, rather than shipping the 94% rule.

The other refusals are upstream's, in upstream's own measured order:
`output_size` length, then input rank, then "sizes greater than 0", then the
non-empty check, then the dtype. `N == 0` is accepted (the non-empty check
looks at the product of the dims *after* the batch) and `C == 0` is not.

### 20.5 What the `.vec` cases caught

The composite was written with "an explicit `output_size` wins over
`scale_factors`". Upstream raises:

```
RuntimeError: Must specify exactly one of output_size and scale_factors
```

Two golden cases failed as `SILENT DIVERGENCE: torch raised ... but c computed
a value`, which is the harness reporting the composite computing where upstream
refuses. **No dispatch key exists for `.vec` at all**, so nothing but a
spelling case could have seen it — the third time this round (after §15.4 and
§18.2) that the door and not the kernel was the defect.

### 20.6 Counts

| gate | before §20 | after |
|---|---:|---:|
| `pytests/run.sh` | 272 | **272** |
| `compare.py` | 5941/5941, ops=155 | **6033/6033, ops=156** (+92 cases, +1 op) |
| `verify_schemas.py` | 4430/4430 | **4433/4433** (+3) |
| schema identities | 242 | **242** (unchanged) |
| core-tagged ops | 93 | **93** (unchanged) |
| sweep26 (shim) | 22/26 | **22/26** |

Two counts that do **not** move, both checked rather than noticed:
`upsample_bilinear2d.default`'s tags are `['pt2_compliant_tag']` — it is *not*
core, unlike `native_group_norm` beside it — and the identity count is
unchanged because, like `native_group_norm`, this op is reached through a
`bootstrap.py` composite and `_aten_dispatch` and appears in neither schema
table.

### 20.7 The sweep

**22/26. `zoedepth` moved one wall**, and off the kernel class entirely:

```
zoedepth   FAIL  torch.relu_(...) -- overload resolution has no table entry for this op
```

`aten.relu_.default` has had a kernel since docs/kernels/KERNELS.md; only the free
function's name is missing. `vits`, `sew_d` and `sam3_video` are unchanged.

## 21. `torch.relu_` and `torch.concat` — two doors, and **zoedepth crosses**

Neither is a kernel. Both are §13.1's finding again: *the wall behind a kernel
turned out to be a name.*

| wall | what it cost | sweep after |
|---|---|---|
| `torch.relu_(...)` — no `overloads.json` entry | one table entry. `aten.relu_.default` has had a kernel since docs/kernels/KERNELS.md and `x.relu_()` worked the whole time | 22/26 → `torch.concat` |
| `torch.concat(...)` — no entry | a `bootstrap.py` composite. `aten::concat` is `CompositeImplicitAutograd`; a `TorchDispatchMode` trace fires `aten.cat.default` and nothing else | **23/26** |

`relu_` adds **no** schema identity (`methods.json` already named
`aten::relu_`, so `overloads.json` gives the same schema a second door) and
`concat` adds none either, being a composite. `verify_schemas` moves by 1 for
`relu_` and 0 for `concat`, which is the check that the second went in as a
composite and not as a table entry.

There is deliberately no `Tensor.concat`: upstream has none
(`hasattr(torch.Tensor, "concat")` is False on 2.13.0), and adding one would
invent a surface.

### 21.1 `zoedepth` — 23/26

```
llama gpt2 qwen2 mistral gemma gpt_neox opt mpt starcoder2 stablelm olmo phi
mixtral bert bloom cohere falcon gpt_bigcode mamba persimmon
deberta deberta_v2 zoedepth                                   TOTAL 23/26
```

### 21.2 …and whether it *agrees*

A forward that runs is not a forward that agrees, so the toy model was run on
both sides under one seed and diffed. **The weights are diffed too** — "the
initialisers agree under the same seed" is itself a claim, and a forward that
matches on a model that does not is not evidence about kernels.

```
architecture      zoedepth
inputs            identical
weights           129 tensors, 196221 elements, 0 differing, max |diff| = 0
shape             [2, 32, 32]  (2048 elements)
sum               1419.565185546875 vs 1419.565185546875     bit-identical
max |diff|        2.98023e-07
```

**The final tensor's argmax is not evidence here, and saying so is part of the
result.** `predicted_depth` on this toy config has **2 distinct values across
2048 elements** — a span of `1.19e-07`, which is smaller than the disagreement
between the two runs. An argmax over that is decided by float noise on both
sides, so reporting it as a match or a mismatch would be reporting the tie.
(The comparison script prints the margin to the runner-up beside every argmax
for exactly this reason.)

So the evidence is the **115 module outputs**, captured with forward hooks on
both sides:

```
module outputs    115 compared
  worst |diff| / max|value|   1.87e-06   (neck.fusion_stage...residual_layer2.convolution2)
  worst |diff|                1.25e-06   (backbone.encoder.layer.1.norm1, max|v| = 1.73)
  deepest rich tensor         metric_head.conditional_log_binomial.mlp.0
                              163840 elements, 163690 distinct
  its argmax                  77220 vs 77220   MATCH   (max |diff| 1.11e-15)
```

`1.87e-06` relative is about 16 ULP at `float32`, accumulated through a
two-layer backbone, a four-stage fusion neck and the metric head. The
argmax **that does have signal** — over a 163840-element tensor with 163690
distinct values — matches exactly.

One trap avoided in reading those numbers: the *per-element* worst relative
error over the module outputs is `0.48`, and it is meaningless. It occurs on
`metric_head.attractors.0.conv1`, whose largest element is `3.2e-10`; the
absolute difference there is `3.3e-16`. Scaling to the tensor's own magnitude
rather than to each element is what makes the number readable, and the script
does that.

### 21.3 Where the other three stand

`sew_d` and `sam3_video` did not move (`torch.avg_pool1d`, `div.Tensor`'s mixed
dtypes); `vits` did not either (`'IntTensor' object is not subscriptable`).

## 22. `erf`, `sign`, `avg_pool2d` — and **sew_d crosses**

Three kernels, each swept on its own, in the order sew_d asked for them.

Before writing any of them, the remaining work was **measured rather than
walked**: the toy model was run on upstream under a `TorchDispatchMode` logger
and the set of ops it fires diffed against `_aten_all_implemented()`. That is
ARCH26.md §6's method, and it turns "walk one wall at a time and find out" into
a list:

```
vits    50 ops fired, 2 not implemented   leaky_relu.default   randn_like.default
sew_d   47 ops fired, 3 not implemented   avg_pool2d.default   erf.default   sign.default
```

It names kernels and not spellings, so it is a lower bound — but a lower bound
that costs one run per architecture rather than one build per wall.

### 22.1 `aten.erf.default`

`sew_d` inherits DeBERTa's GELU, which spells the error function out
(`x * 0.5 * (1 + erf(x / sqrt(2)))`) rather than calling `aten.gelu`, so the op
fires on its own — twice per forward on a `(1, 19, 37)`.

A plain member of the `unary_float` family, and that was measured rather than
assumed: `int64`, `int32`, `uint8` and `bool` all give `float32`, and each float
dtype keeps its own. `silu` — the other activation-shaped op in this file —
*refuses* an integral input, so which of the two rules applies is not derivable
from "it is an activation".

**One golden case had to have its data changed to be able to run at all.**
`erf(-0.0)` is `-0.0`, which `==` cannot see, so it needs `_signed_zero_check`
— an **exact** comparator. But candle's `erf` is `libm::erf` and lands 1.2e-07
from upstream's own kernel at `x = 1` (`gelu_default` already records the same
divergence at 4.47e-08), so an exact comparator on ordinary values fails on the
*erf* rather than on the sign bit — which is what the first run did. The case
now uses only `±0.0` and `±inf`, whose erf is exact on both sides (`-0.0`,
`+0.0`, `-1.0`, `+1.0`), so it isolates the one property the tolerant
comparator cannot see and nothing else.

### 22.2 `aten.sign.default`

`modeling_sew_d.py:160` — `torch.sign(relative_pos)` on the
disentangled-attention bucket table, once per forward on a `(19, 19)`.

**Its dtype rule is the opposite of `erf`'s, in the same section.** `sign` keeps
the input dtype on *every* dtype including `bool` (`sign(bool)` is `bool`);
`erf` promotes every integral input to `float32`. Both were measured; neither
follows from the other, and they landed together precisely so that could be
said.

Three values fix the definition, and each is where `x > 0 ? 1 : -1` — the
plausible two-way spelling — is wrong:

```
sign(0.0)    0.0    there is a zero in the range
sign(nan)    0.0    NaN is neither > 0 nor < 0
sign(-0.0)  +0.0    POSITIVE zero -- checked with copysign, not with ==
```

candle's `Sign` is `f32::from(v > 0.) - f32::from(v < 0.)` on floats and
`min(1, v)` on the unsigned types, which gives all three, so this delegates.

### 22.3 `aten.avg_pool2d.default`

`nn.AvgPool1d(kernel_size=2, stride=2)` in sew_d's encoder →
`torch.avg_pool1d` → **`aten.avg_pool2d.default`**, because `aten::avg_pool1d`
is `CompositeImplicitAutograd`: measured, `torch.avg_pool1d(x, 3, 2)` fires
`unsqueeze(-2)`, `avg_pool2d([1,3],[1,2])`, `squeeze(-2)` and nothing else. So
the 1-D name is a `bootstrap.py` composite and the 2-D op is the leaf. The
degenerate axis is **H**, not W.

**The two boundaries that are not the same boundary.** Upstream computes, per
output cell:

```
start = out*stride - pad
end   = min(start + kernel, extent + pad)        <- clipped to the PADDED extent
count = (h_end - h_start) * (w_end - w_start)    <- taken HERE
start = max(start, 0);  end = min(end, extent)   <- now clipped to the REAL extent
divisor = divisor_override, else count if count_include_pad else the clipped area
```

The order between those two clips *is* `count_include_pad`. Measured on
`arange(20).reshape(1,1,4,5)` with `kernel=2, stride=2, padding=1`, the cell at
`(0,1)` sums `1+2 = 3` and divides by **4** with `count_include_pad=True`
(`0.75`) and by **2** without (`1.5`). Same sum, same window, two answers — and
`True` is the default, so an implementation that divided by what it summed is
wrong on the *default* path.

Other rules, each measured:

  * **`stride=[]` means the kernel size, not 1.** A `stride=None` case sits
    beside the explicit one so the two must agree.
  * **`ceil_mode` has a drop rule.** `ceil` can produce a last window starting
    at or past the end of the padded input, and upstream drops it. Cased on a
    `1x5` where ceil gives 3 columns and floor gives 2.
  * **`int64` computes and every other integral dtype does not.** `int32`,
    `int16`, `int8`, `uint8` and `bool` all raise `"avg_pool2d" not implemented
    for '<Type>'`. Measured one dtype at a time — "integral is supported" would
    have been the wrong summary.
  * **the integral divide truncates toward zero.** `11/4` is `2`, `-11/4` is
    `-2`, not `-3`.

Precision, and it goes both ways again:

```
float16/bfloat16 accumulated in f32 and narrowed once   max relative 0.0 vs upstream
float32          accumulated in f64 and narrowed once   max relative 1.43e-05  -- WORSE
```

`1.43e-05` is past this repository's `float32` golden tolerance, so `f32`
accumulates in `f32`. This is the second op this round (`sigmoid`,
`upsample_bilinear2d`, and now this) where "compute as wide as possible" is the
wrong rule.

A measurement error worth recording, because it nearly set the accumulate type
the other way: the first `float16` comparison narrowed **upstream's `float32`
result on the original `float32` data** rather than on the `float16` values,
and reported 90 of 216 differing. Comparing like with like — `f16` values
widened to `f32`, pooled, narrowed — gives 0.

### 22.4 The sweep, and **sew_d — 24/26**

```
after erf          23/26   sew_d still on torch.avg_pool1d
after sign         23/26   sew_d still on torch.avg_pool1d
after avg_pool2d   24/26   sew_d PASSES
```

The count moved on the third of the three, and the first two moved no wall at
all — `erf` and `sign` fire *later* in the same forward than `avg_pool1d` does,
so the sweep's first-wall report could not see them being fixed. Only the
upstream op scan in §22 above could, which is the argument for having run it.

### 22.5 …and whether `sew_d` agrees

```
architecture      sew_d
inputs            identical
shape             [1, 39, 32]  (1248 elements, 1217 distinct)
argmax            91 vs 91      MATCH   (margin to runner-up 0.00525)
argmin            1056 vs 1056  MATCH
sum               5.7529401779174805 vs 5.7529377937316895
max |diff|        8.9407e-08          (output span 0.312, so 2.9e-07 of it)
module outputs    60 compared, worst |diff|/max|v| = 5.88e-07
  deepest rich tensor  encoder.pos_conv_embed.conv.parametrizations.weight
                       8192 elements, 8192 distinct
  its argmax           7554 vs 7554   MATCH   (max |diff| 2.24e-08)
```

Unlike `zoedepth`, this output has signal — 1217 distinct values out of 1248 —
so the argmax is real evidence and its margin to the runner-up (`0.00525`) is
five orders of magnitude larger than the disagreement (`8.9e-08`).

**The weights do not agree bit for bit, and that is a known divergence rather
than a surprise.** 11 of 27610 elements differ, all of them in one tensor:

```
encoder.pos_conv_embed.conv.parametrizations.weight.original0
n=16, 11 differing, max |diff| 2.98e-07, max |value| 0.8327
```

That is `g`, the weight-norm *magnitude*, computed at **construction** by
`aten.norm.ScalarOpt_dim` — §11's kernel, whose reduction order differs from
upstream's by float32 rounding. So the two models are not bit-identical before
the forward starts, and the 8.9e-08 the forward then shows is on top of a
2.98e-07 difference in one parameter. Saying "the forward agrees" without
saying that would be claiming more than was measured.

## 23. `log2`, `leaky_relu`, `div`'s promotion, `einsum` — and **sam3_video crosses**

The same upstream op scan §22 used, re-run for the two architectures still
standing, is what set this order:

```
vits         50 ops fired, 2 not implemented   leaky_relu.default   randn_like.default
sam3_video   69 ops fired, 3 not implemented   log2.default   rand.default   randn.default
```

### 23.1 `aten.log2.default`

`expm1`'s shape exactly: **the dtype rule is `unary_float`'s and the
computation is not.** `int64`, `uint8` and `bool` all give `float32` and each
float dtype keeps its own, so the promotion is shared. But candle has no
`log2`, and `t.log()? / ln(2)` is a different function at the last bit —
measured at `float64` it disagrees with `torch.log2` on 2 of 7 probe points,
because upstream calls `std::log2` where that divides two separately-rounded
values. `f64::log2` reproduces upstream on every `float64` probe, and
`float16`/`bfloat16` are bit-identical through `f32` over 2000 random points.

The powers of two are the cases that show it: `log2(8.0)` must be exactly `3.0`.

### 23.2 `aten.leaky_relu.default`

**`silu`'s side of the dtype split, not `relu`'s.** `relu` has an integral CPU
kernel upstream and `leaky_relu` does not — `int64`, `uint8` and `bool` all
raise `"leaky_relu_cpu" not implemented`. Two ops that differ by one
multiplication and do not share a dtype rule.

The plausible wrong implementation is `max(x, slope * x)`, and it agrees with
`x < 0 ? slope*x : x` for **every slope in [0, 1]**, which is every slope anyone
writes. It differs at a *negative* slope: `leaky_relu(-1, -0.5)` is `0.5`
upstream and `-1` from the max spelling. `negative_slope` is a `Scalar` with no
sign constraint and upstream computes it, so that case is here — along with
`x < 0` versus `x <= 0`, which differ only in the sign of the zero and need
`_signed_zero_check` to see.

It is a `torch._C._nn` composite, not a table entry: `F.leaky_relu` *is* that
binding, and upstream has neither `torch.leaky_relu` nor `Tensor.leaky_relu`
(`hasattr` is False for both), so a table entry would invent a surface.

### 23.3 `div.Tensor` promotes its operands

`sam3_video` stopped on

```
aten.div.Tensor: dtype promotion not implemented in torch._C shim: float32 vs int64
```

which is neither a missing kernel nor a missing name — it is `same_dtype`
refusing a mixed pair. `mul` has promoted since docs/kernels/OPS4.md; `div` now does
too, through the same condition.

**`add` and `sub` still refuse, and that is asserted rather than left
implicit.** Upstream promotes all four; the split here is a record of which
callers have been measured, not a principle (docs/bindings/BIND.md §9), and nothing in
the sweep reaches `add`/`sub` with a mixed pair. Two `c_error` golden cases
watch them.

The **whole 10×10 promotion grid** is re-run for `div` rather than only the one
cell, because `div`'s result rule is not `mul`'s: true division floats an
integral pair, so `int64 / int64` is `float32` where `int64 * int64` is
`int64`. A promotion rule copied from `mul` and a *result* rule copied from
`mul` are two different mistakes and only the full grid separates the second.
The same pair of assertions went into the meta test for the same reason.

One cell disagrees and it is pre-existing: **`bool / bool`**. Upstream gives
`float32`; `arith_tag` refuses `bool` arithmetic outright (BOOL.md §2.2), and
`mul`'s own grid never sees it because `bool * bool` stays `bool` and *is*
logical-and. Watched as `c_error`.

### 23.4 `torch.einsum`

`modeling_sam3.py:2113` — `torch.einsum("bqc,bchw->bqhw", mask_embeddings,
instance_embeds)`, once per forward.

`aten::einsum` is `CompositeImplicitAutograd`, and this reproduces its
decomposition rather than inventing one. Measured, `einsum("bqc,bkc->bqk", ...)`
fires `unsqueeze`, `permute`, `view`, **`bmm`**, `view`, `permute`, `view` —
every one already here, and `aten.einsum.default` never fires. The part that
matters numerically is that the contraction really is a **`bmm`**, so the
accumulation order is upstream's rather than a hand-rolled sum's.

The algorithm, per contraction:

```
batch    labels in BOTH inputs and still needed afterwards
summed   labels in BOTH inputs and not needed afterwards
free_a   labels in a only        free_b   labels in b only

a -> (batch, free_a, summed) -> (B, M, K)
b -> (batch, summed, free_b) -> (B, K, N)     bmm -> (B, M, N) -> permuted to the output
```

A label appearing in one operand only and not in the output is summed out of
that operand *first*, before the pairing.

Ten equations are cased, chosen so that a decomposition handling the common
shape and nothing else fails somewhere — including the implicit-output form
(`ij,jk`, whose output is alphabetical: `ik`, not `ki`), a fully-contracted
`i,i->` where both M and N are empty products, and `abc,abd->acd`, where `b` is
contracted while `a` is a batch, which is the case that separates the two.

Refused by name, with `c_error` cases: an **ellipsis** (a variable number of
batch axes, needing its own rank arithmetic) and a label **repeated inside one
operand** (`ii->i` is a diagonal, not a contraction, and there is no
`aten.diagonal` kernel here).

It takes **both** calling conventions, because upstream has both:
`torch/functional.py:362` unpacks "the old interface of passing the operands as
one list argument" and line 372 then calls `_VF.einsum(equation, operands)` —
with a list. Writing only the list form failed every golden spelling case,
which is how that was found.

### 23.5 The sweep

```
after log2         24/26   sam3_video moved to torch.einsum
after leaky_relu   24/26   (landed together with log2; vits' wall is earlier)
after div promote  24/26   sam3_video moved to torch.einsum
after einsum       25/26   sam3_video PASSES
```

`rand` and `randn` — on the op scan's list for `sam3_video` — were **not
needed**: they fire on a branch the detector forward does not take. Written
down because the scan is a lower bound in one direction and an over-estimate in
the other, and this is the second direction.

```
llama gpt2 qwen2 mistral gemma gpt_neox opt mpt starcoder2 stablelm olmo phi
mixtral bert bloom cohere falcon gpt_bigcode mamba persimmon
deberta deberta_v2 zoedepth sew_d sam3_video               TOTAL 25/26
```

### 23.6 …and whether `sam3_video` agrees

```
architecture      sam3_video
inputs            identical
weights           258 tensors, 266290 elements, 0 differing, max |diff| = 0
shape             [1, 5]  (5 distinct values)
argmax            0 vs 0  MATCH   (margin to runner-up 0.00189)
argmin            4 vs 4  MATCH
sum               0.025955114513635635 vs 0.025955114513635635   bit-identical
max |diff|        5.58794e-09          (1.26e-07 of the output span)
module outputs    164 compared, worst |diff|/max|v| = 1.20e-06
  deepest rich tensor  detr_encoder.layers.0.mlp.activation_fn
                       65536 elements, 29653 distinct
  its argmax           40003 vs 40003   MATCH   (max |diff| 2.53e-07)
```

**The first attempt at this comparison failed, and the reason is worth
recording because it is a live gap.** The sweep's toy script draws
`input_ids = torch.randint(0, 1000, (1, 16))`, and the shim's `randint` does
**not** reproduce upstream's sequence — its own golden case says so ("random
draw, sequence unchecked"). So the two runs were different forwards:

```
pixel_values (torch.rand)      150528 of 150528 identical
input_ids    (torch.randint)       16 of 16 DIFFERING
pred_logits                    max |diff| 0.0175 -- 40% of the output span
argmin                         4 vs 2   DIFFER
```

`torch.rand` matching bit for bit is the control that says the RNG *position*
is not the problem — it is `randint`'s draw. The comparison now uses a fixed
token list so that it measures the kernels rather than the RNG; the `randint`
gap is untouched and is the same one its golden case already records.

## 24. The four gaps behind `vits` — and **26 of 26**

`vits`' tail was not kernels. Three of the four are surface, and the fourth is
a refusal §10.3 wrote down as "no measured caller".

### 24.1 `torch.IntTensor` and its nine siblings

`modeling_vits.py:349` builds `torch.IntTensor([self.hidden_size])` and then
subscripts it. The ten legacy per-dtype classes were `_ShimMeta` placeholders,
so the model stopped on `'IntTensor' object is not subscriptable` — a
`TypeError` from a type that was never a tensor class at all.

**They must stay real types.** A factory function does not work, and the
failure is not subtle: these names appear in *annotations*, which Python
evaluates at import time.
`transformers/modeling_flash_attention_utils.py:602` is
`max_seqlen_q: int | torch.IntTensor | None = None`, and `int | <function>` is
a `TypeError` that stops `import transformers` dead — measured while probing.
So each is a class whose `__new__` returns a `TensorBase`, which is also what
upstream does: `type(torch.IntTensor([1]))` is `torch.Tensor`.

All three of the legacy constructor's forms, and the ambiguity is why §12.1
ground 3 exists:

```
IntTensor(2, 3)      a SIZE -> a (2, 3) tensor of int32
IntTensor([2, 3])    DATA   -> a (2,) tensor holding 2 and 3
IntTensor(existing)  a re-wrap, cast to the class's dtype
```

`[2, 3]` looks exactly like a size list. The data branch is therefore decided
**by type and never by shape**.

**This is §12.1's own condition being met, not overturned.** That section kept
`torch.Tensor([3, 4])` refused on the ground that "nothing in the
26-architecture sweep reaches it, and the two forms are used by different
code". `vits` now reaches the *typed* data form, so the typed classes are
implemented and `torch.Tensor([...])` — which still has no caller in the sweep
— still refuses. The inconsistency is deliberate and is the rule the repository
already states: surface follows a measured caller.

`CharTensor` refuses from one layer down (candle has no `int8` storage) and
names the *dtype*. `ShortTensor` computes — `int16` is storable here, checked
rather than assumed from `int8` being absent.

### 24.2 A `numpy` scalar has to bind where a Python number would

`modeling_vits.py:1379` is
`predicted_lengths * np.prod(self.config.upsample_rates)`, and `np.prod([4,4])`
is an `np.int64` — not a Python `int` and not a subclass of one. The resolver's
`Scalar` predicate tested `isinstance(value, (bool, int, float, complex))`, so
nothing bound; Python fell back to `np.int64.__rmul__`, which asked the tensor
for `__array__`, which landed on `TensorBase.numpy` — a raising stub.

Upstream takes it (measured: `torch.tensor([1,2]) * np.int64(16)` fires
`aten.mul.Tensor` and keeps `int64`). The fix is `numbers.Number` in the
predicate and `__index__`/`__float__` in the kernel's scalar reader — **the
protocol, not a numpy import**, because `_C` is built before numpy would be
importable.

**`__index__` is tried before `__float__` and that ordering is load-bearing.**
`np.int64` has both. Taking the float would make it a `Scalar::Float`, which
`arith_tag`'s wrapped-number rule turns into a `float32` result from an `int64`
tensor — a wrong dtype, silently. Sabotage S21 is exactly that swap.

### 24.3 1-D transposed convolution

§10.3 refused it, in these words: *"candle supports it fully, groups included,
but nothing measured reaches it, and this round does not add unreached
surface."* `vits`' HiFi-GAN decoder reaches it —
`nn.ConvTranspose1d(channels, channels // 2, kernel, stride=rate,
padding=(kernel - rate) // 2)`, once per upsample rate.

The refusal is lifted for 1-D and **stands for grouped 2-D**, and the asymmetry
is candle's rather than upstream's: `conv_transpose1d` takes a `groups`
argument and `ParamsConvTranspose2D` has no field for one. So the 1-D path
honours `groups` and the 2-D path still refuses a grouped call by name.

Two `c_error` golden cases flipped on their own when this landed —

```
gap appears CLOSED: both sides now succeed, promote this case to expect=match
```

— which is what `c_error` is for, and they are now `match` cases diffing values.
`torch.conv_transpose1d` joins its 2-D sibling in `bootstrap.py`, with the
same argument order that is **not** `conv1d`'s (`groups` before `dilation`) and
with `_as_list`'s width at **1**: passing 2 would hand
`aten.convolution.default` a two-element `stride` for a rank-3 input, which is
where a copy-paste of the 2-D body goes wrong. Sabotage S18.

### 24.4 The sweep — 26 of 26

```
llama gpt2 qwen2 mistral gemma gpt_neox opt mpt starcoder2 stablelm olmo phi
mixtral bert bloom cohere falcon gpt_bigcode mamba persimmon
deberta deberta_v2 vits zoedepth sew_d sam3_video          TOTAL 26/26
```

Upstream is 26/26 on the same script, re-run at the end.

### 24.5 …and whether `vits` agrees

**`vits`' forward is stochastic, and this is the one architecture where the
draw cannot be made to agree.** `modeling_vits.py:1373` is

```python
prior_latents = prior_means + torch.randn_like(prior_means) * torch.exp(prior_log_variances) * self.noise_scale
```

and `prior_means` is **not contiguous** there — measured, on both sides.
Upstream's `normal_` CPU kernel branches on exactly that: `size >= 16 &&
self.is_contiguous()` takes the vectorised `normal_fill`, anything else takes
the scalar Box–Muller path. Upstream's `empty_like` preserves the input's
layout, so upstream takes the **scalar** path for this 256-element draw; the
shim's tensors are always materialised (docs/kernels/VIEWS.md §6.4), so its
`empty_like` is contiguous and it takes the **vectorised** one.

Both are torch's own streams, from the same state. That was established rather
than assumed:

```
normal_ at n = 6, 15, 16, 17, 256           identical on both sides
normal_ after a prefix draw of 1, 3, 4, 5, 16  identical on both sides
the RNG state at the call site               identical (probe draw agrees)
randn_like(zeros(1,32,8)) from a fresh seed  identical
the draw inside the forward                  DIFFERENT
is_contiguous(prior_means)                   False, on both sides
```

So the difference is the **path selection**, and the shim cannot see the
property it selects on. This is a consequence of a structural divergence
already recorded, not a new one, and it is not fixable without non-contiguous
tensors.

The comparison therefore replaces `randn_like` **on both sides** with the same
contiguous draw — same seed, same distribution, same count, no layout-dependent
branch — so that what is left is the kernels:

```
architecture      vits
inputs            identical
shape             [1, 128]  (63 distinct values)
argmin            31 vs 31  MATCH
argmax            22 vs 26  DIFFER -- and it is a 47-WAY TIE
sum               46.05282974243164 vs 46.052818298339844
max |diff|        5.51343e-06        (output span 2, so 2.8e-06 of it)
module outputs    64 compared, worst |diff|/max|v| = 5.51e-06
  deepest rich tensor  flow.flows.0.wavenet.in_layers.0.parametrizations.weight
                       6144 elements, 6143 distinct
  its argmax           1734 vs 1734   MATCH   (max |diff| 1.49e-08)
```

**The argmax differing is a tie, and the count is the evidence**: the waveform
is clipped, and `1.0` occurs **47 times** and `-1.0` **18 times**. The margin to
the runner-up is exactly `0`, which the comparison script prints beside every
argmax for this reason. The argmax that does have signal — over a
6144-element tensor with 6143 distinct values — matches exactly.

**The weights are not bit-identical, in the same place `sew_d`'s were not.**
147 of 72955 elements differ, every one of them in a
`*.parametrizations.weight.original0` — the weight-norm magnitude `g` computed
at construction by `aten.norm.ScalarOpt_dim` (§11), max `|diff|` `1.19e-07`
across six tensors. No other weight differs at all.

### 24.6 The three architectures' numeric agreement, side by side

| | `zoedepth` | `sew_d` | `sam3_video` | `vits` |
|---|---|---|---|---|
| weights differing | 0 / 196221 | 11 / 27610 | 0 / 266290 | 147 / 72955 |
| …and where | — | `weight.original0` | — | `weight.original0` |
| output max \|diff\| | 2.98e-07 | 8.94e-08 | 5.59e-09 | 5.51e-06 |
| output sum | bit-identical | 5.7529402 vs 5.7529378 | bit-identical | 46.052830 vs 46.052818 |
| argmax | tie (2 distinct / 2048) | **MATCH**, margin 0.00525 | **MATCH**, margin 0.00189 | tie (47-way) |
| argmin | tie | **MATCH** | **MATCH** | **MATCH** |
| module outputs | 115, worst 1.87e-06 | 60, worst 5.88e-07 | 164, worst 1.20e-06 | 64, worst 5.51e-06 |
| deepest rich argmax | **MATCH** (163840 el.) | **MATCH** (8192 el.) | **MATCH** (65536 el.) | **MATCH** (6144 el.) |

Two of the four have a degenerate final tensor — `zoedepth`'s toy depth map has
**2 distinct values across 2048 elements** and `vits`' waveform is clipped to
±1 — so their final argmax is a tie and is reported as one rather than as a
match or a mismatch. For both, the argmax over the deepest tensor that *does*
have signal matches exactly.

## 25. Sabotage

Twenty-one faults, each the **plausible wrong implementation** the
corresponding section claims its cases separate. One at a time: patch, rebuild,
`compare.py`, `run.sh`, revert. Baseline is 6374/6374 golden and `run.sh` exit 0.

Every row was re-run against **that** baseline. The first pass ran against
6372, before §25.2's two `log2` cases existed; only the two totals moved (by
exactly 2, and only for the two faults whose case sets are disjoint from
`log2`), and no `failed` count changed. Re-running rather than adding 2 to the
totals is the difference between a measurement and an inference.

| # | fault | golden | `run.sh` | what it proves |
|---|---|---|---|---|
| S1 | `clamp_min` shares `clamp`'s bool-refusal wording | **6374/6374, 0 failed** | 1 FAIL | §15.1 exactly: the golden case is `both_error` and does not compare messages, so only `test_shim.py` can see it |
| S2 | `all` shares `any`'s empty identity (`0`) | 6366/6374, **8 failed** | 1 FAIL | the `all` over nothing is `True`; 8 cases, all of them empty inputs |
| S3 | `all`/`any` always return `bool` | 6347/6374, **27 failed** | 1 FAIL | the `uint8` rule, across **both** ops — which is why it was fixed in `any` too |
| S4 | `sigmoid` computed in the reduced dtype | 6372/6374, **2 failed** | 0 FAIL | exactly the two `BIT-EXACT` `f16`/`bf16` cases. Every other case at every dtype passes — §17.2's claim, demonstrated |
| S5 | `flip` accepts a repeated dim | 6372/6374, **2 failed** | 1 FAIL | the two `both_error` cases, `[0,0]` and `[1,-1]` |
| S6 | `flip` reads the keyword `dim` | 6368/6374, **6 failed** | 1 FAIL | exactly the six *spelling* cases; all 60 dispatch-key cases pass, because they pass the list positionally |
| S7 | `group_norm` applies the affine along the statistics view | 6351/6374, **23 failed** | 1 FAIL | the `C=6, group=3` shape is what sees it |
| S8 | `group_norm` `rstd` is `sqrt(var+eps)` | 6337/6374, **37 failed** | 1 FAIL | the result nothing reads, caught by cases written for it |
| S9 | bilinear drops the half-pixel offset | 6343/6374, **31 failed** | 0 FAIL | every `align_corners=False` case on non-symmetric data |
| S10 | bilinear ignores `scales_h`/`scales_w` | 6371/6374, **3 failed** | 0 FAIL | only the three cases whose geometry makes `1/scale` differ from `in/out` — the other 90 pass |
| S11 | `avg_pool2d` ignores `count_include_pad` | 6369/6374, **5 failed** | 0 FAIL | only the padded cases; every unpadded one is unaffected, which is right |
| S12 | `log2` is `log(x)/ln(2)` | 6372/6374, **2 failed** | 0 FAIL | **see below — this one initially missed** |
| S13 | `leaky_relu` is `max(x, slope*x)` | 6370/6374, **4 failed** | 0 FAIL | the four negative-slope cases, one per float dtype. Every slope in `[0,1]` passes |
| S14 | `sign` is `x > 0 ? 1 : -1` | 6350/6374, **24 failed** | 0 FAIL | zero, NaN and the dtype rule together |
| S15 | `div` stops promoting | 6293/6374, **81 failed** | 0 FAIL | the 10×10 grid, minus the cells that were already same-dtype |
| S16a | `einsum` treats every shared label as a batch axis | **6374/6374, 0 failed** | 0 FAIL | **correct — see below** |
| S16b | `einsum` aligns the right operand as `(batch, free_b, summed)` | 6369/6374, **5 failed** | 0 FAIL | the contraction axes transposed; five of the ten equations can see it |
| S17 | `IntTensor([32])` reads the sequence as a **shape** | **6374/6374, 0 failed** | 1 FAIL | **golden is structurally blind — see below** |
| S18 | `conv_transpose1d` pads its lists to width 2 | 6372/6374, **2 failed** | 0 FAIL | the two spelling cases; the dispatch-key cases pass their lists already-sized |
| S19 | `avg_pool1d` puts the degenerate axis on W | 6369/6374, **5 failed** | 0 FAIL | the six composite cases, minus `k=1` which is the identity either way |
| S20 | the `Scalar` predicate stops accepting a numpy scalar | **6374/6374, 0 failed** | 1 FAIL | same blindness as S17 — no dispatch key names a resolver predicate |
| S21 | the scalar reader takes `__float__` before `__index__` | **6374/6374, 0 failed** | 1 FAIL | an `int64` tensor times an `np.int64` comes out `float32`; only the shim test sees it |

### 25.1 The three faults that do **not** fail golden, and why each is right

**S1, S20, S21 — golden compares by dispatch key.** A refusal *message*, a
resolver predicate and a scalar reader's ordering are not values of an op, so
no case can compare them. All three fail `run.sh`, which is where the
assertions about them live. This is §13.3's finding for the fourth, fifth and
sixth time this round.

**S16a is not a wrong implementation.** Folding a contracted label into the
batch group makes the `bmm` an outer product with `K = 1` and leaves the
summation to the `sum_over` at the end — a different *decomposition* of the
same function, and slower, but the same answer to the last bit. Reporting it as
an uncaught fault would be reporting a case set for failing to detect
correctness. S16b is the same region of the code done *wrongly* (the
contraction axes transposed), and five cases catch it.

**S17 is a real gap in the case set, and it was closed.** The legacy
constructors have no aten op — `torch.IntTensor([32])` produces a `lift_fresh`
and nothing that names the class — so `compare.py` has no key to hang a case
on, and reading the sequence as a shape left all 6374 green. A `test_shim.py`
test was written *because of this run*, and S17 now fails `run.sh`. The same is
true of S20.

### 25.2 The fault that missed, and what it changed

**S12 initially passed every one of the 6374 cases.** `log2` written as
`log(x) / ln(2)` differs from `std::log2` by **1 ULP**, and the harness's
`float64` tolerance absorbs 1 ULP completely — while the powers of two that
made up most of the case set are exact under *both* spellings.

That is §0's third trap ("a tolerance that cannot see the fault") for the second
time this round, and it was caught by the sabotage run rather than by reasoning.
The fix is the one `sigmoid` needed: measure where the two actually differ
(`3, 9, 10, 12, 100` in `float64` and `0.3` in `float32`, on 2.13.0) and compare
those **exactly**, through `_bitwise_equal_check`. S12 now fails 2 cases.

**A case set that cannot fail is not a case set**, and the only way this was
going to be found was by breaking the kernel on purpose.

## 26. Final gates

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   274 ok, 0 FAIL                    exit 0   (was 268)
$PY tools/golden/compare.py                 6374/6374, ops=161, pending 1     exit 0   (was 5634/5634, ops=148)
$PY tools/golden/compare.py --self-test     16 comparators x 11 fault modes    exit 0   (was 15)
$PY rust/torch_c/pytests/verify_schemas.py  4458/4458                          exit 0   (was 4392/4392)
( cd rust/torch_c && cargo test --release ) 28 passed                          exit 0
sweep26 (shim)                              26/26                              exit 0   (was 22/26)
sweep26 (upstream)                          26/26                              exit 0
```

Other counted invariants, each re-derived rather than carried:

```
schema identities         230 -> 248        every one accounted for in test_shim.py
core-tagged ops            92 -> 98         read off each op's own .tags, one at a time
decomposition registry   1007 -> 1008       flip.out, the fourth time by that mechanism
registry_default          461 -> 461        unchanged, which is the check it was a .out
```

Prefill digests on the final artefact, **all six unchanged** from
docs/numerics/SEQLEN.md §1.3 and §8.8:

```
f32   S=6 b9fc5553ee1bf6a2   S=32 331668f36da02f21   S=128 00159a9dbd308eda
      S=512 07c2797dabc4552e   S=1024 eda1e173727bb7f5
bf16  S=128 7ff8e9334449b147
```

### 26.1 What is left

Nothing on §8.2's "genuinely still missing" list blocks an architecture any
more. What remains from it, untouched and with no caller in the sweep:

```
aten.fmod.{Tensor,Scalar}              aten.remainder.Scalar_Tensor (and __rmod__)
aten.set_.source_Tensor_storage_offset aten.rand.default   aten.randn.default
```

`rand`/`randn` are on `sam3_video`'s op scan and were **not** needed — they fire
on a branch the detector forward does not take (§23.5). They are reachable
today through the `bootstrap.py` composites over `empty` + `uniform_`/`normal_`;
what is missing is only the aten key.

Gaps found *this* round, each refused by name with a `c_error` case watching it:

```
aten.clamp_min.Tensor              maximum with broadcasting; no aten.maximum here (§15.3)
aten.native_group_norm HxW == 0    upstream's mean=0 with rstd=nan (§19.4)
aten.upsample_bilinear2d uint8     a separate fixed-point kernel; 355/5584 differ (§20.4)
einsum with an ellipsis            variable batch rank
einsum with a repeated label       a diagonal; no aten.diagonal here
grouped 2-D transposed convolution candle's ParamsConvTranspose2D has no groups (§24.3)
div.Tensor on bool/bool            BOOL.md §2.2 (§23.3)
add.Tensor / sub.Tensor mixed      only mul and div promote (§23.3)
torch.Tensor([3, 4])               still no caller in the sweep (§12.1, §24.1)
```

And two divergences that are **not** gaps in a kernel:

- **`randint` does not reproduce upstream's sequence** (its own golden case
  says so). It made the first `sam3_video` and `vits` comparisons different
  forwards; both now use fixed token lists (§23.6, §24.5).
- **`randn_like` on a non-contiguous input takes a different `normal_` path**,
  because the shim has no non-contiguous tensors to preserve
  (docs/kernels/VIEWS.md §6.4). `vits` is the only architecture where this reaches the
  output (§24.5).

---

## 26. What "26 of 26" is a number for

**Every sweep in this document calls `.eval()`.** So does docs/architectures/ARCH20.md's twenty, and so does the
`sweep26.py` §0 describes. That was never stated as a scope, because nobody noticed it was one.

Run the same twenty-six in `.train()` and the shim was **18 of 26** at the moment this document
closed at 26 of 26 — one missing kernel and one missing backend behind it — while README §2/§3 say
the project exists for federated learning and test-time adaptation, which are training.

**docs/training/TRAIN.md** is that axis: what `nn.Dropout` actually dispatches to (not
`aten.dropout.default`, which is `CompositeImplicitAutograd` and never fires), why a seeded
comparison against upstream *is* bit-exact here where `randint`'s is not, the SDPA math backend that
attention dropout selects, and the standing check that keeps `.train()` from regressing unnoticed.
Training mode is 26 of 26 there, and this document's 26 is unchanged.
