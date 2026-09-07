# A loss value: what stood in front of one, and what it cost

`docs/training/TRAIN.md` closes at **26 of 26 architectures forwarding in `.train()`**, and every one of
those forwards is *lossless* — the sweep feeds ids and reads logits. `docs/training/AUTOGRAD.md` §5.3 found
why that matters: a training step needs a scalar to call `.backward()` on, and this shim could not
compute a cross-entropy forward at all. It named two missing ops and called them "the smallest
genuinely useful next commit in this direction".

**They were the right two ops and they were not the whole requirement.** §1 is that finding.

This round also closes the two items `docs/training/AUTOGRAD.md` §6.6 puts beside the loss —
`optimizer.zero_grad()` (§6) and `native_dropout` (§7) — and neither of those turned out to be the
size it was measured at either. The pattern is the same one every time and it is worth naming
once: **an op scan sees the leaves a call lands on, never the names a caller uses to get there.**

### Answers, before the evidence

| question | answer |
|---|---|
| Does a real SmolLM2-135M forward produce a loss? | **Yes.** `12.871352195739746` against upstream's `12.871366500854492` — **1.11e-06 relative** (§4) |
| Which of the two kernels carries that residual? | **Neither carries it jointly.** Fed identical logits, `nll_loss_forward` is **bit-identical** to upstream on the real 49152-class tensor; all of it is `_log_softmax`'s summation order (§4.1) |
| Did the loss need more than the two ops AUTOGRAD.md names? | **Yes — two kernels and four more names**, all `CompositeImplicitAutograd` and so invisible to a dispatch trace (§1) |
| Does `optimizer.zero_grad()` complete? | **Yes**, on SGD/SGD+momentum/Adam/AdamW over all 272 real SmolLM2 parameter tensors — and **vacuously**, because `p.grad` is still `None` (§6.3) |
| What were Adam's three kernels? | `lerp_.Scalar`, `addcmul_`, `addcdiv_`, all small — but Adam stops **before** them, on `torch.is_complex`, a name (§6.4) |
| Did `native_dropout` make the four architectures capturable? | **Three of the four.** `gpt2`, `bert`, `gpt_bigcode` crossed; `opt`'s wall was never dropout (§7.5) |
| Is there a fourth wall? | Yes, and it is the same wall each time: **a missing *name*, not a missing kernel.** §1's four, §6's `DisableTorchFunctionSubclass` and §6.4's `torch.is_complex` |

Written incrementally, one kernel at a time, for the reason `docs/kernels/KERNELS26.md` §0 gives.

Environment: `/Volumes/macMini/caches/spike-venv/bin/python`, torch 2.13.0, transformers 5.15.1,
worktree at `develop` `bf54489`. Upstream C++ read from `/Volumes/macMini/caches/pytorch-spike/pytorch`.

### The baseline, every gate, before any edit

```
pytests/run.sh                293 ok, 0 FAIL, DOCWATCH 71/71     exit 0
tools/golden/compare.py       6675/6675, ops=163, pending=1      exit 0
compare.py --self-test        16 comparators x 11 fault modes    exit 0
verify_schemas.py             4465/4465                          exit 0
sweep26   (shim, .eval())     26/26                              exit 0
sweeptrain (shim, .train())   26/26                              exit 0
```

---

## 1. The op scan named two ops; the path needs eight names

`docs/training/AUTOGRAD.md` §5.3 measured the requirement with a `TorchDispatchMode` over a real
`labels=ids` forward:

```
full: forward uniq 29, MISSING from shim: 2 -> ['aten._log_softmax.default',
                                                'aten.nll_loss_forward.default']
```

That measurement is correct and it is incomplete, and the reason is structural rather than an
oversight. **`TorchDispatchMode` sits below the composite layer.** Every
`CompositeImplicitAutograd` op has already been decomposed by the time a dispatch record is
emitted, so an op scan sees the leaves a call *lands on* and never the names a caller *uses to get
there*. Climbing the real path instead of scanning it — `/tmp/loss/wall.py`, one rung at a time,
against this build:

```
WALL _aten_dispatch aten._log_softmax.default        aten op not implemented in torch._C shim
WALL torch._log_softmax                              overload resolution has no table entry
WALL torch.log_softmax                               overload resolution has no table entry
WALL Tensor.log_softmax                              not implemented in torch._C shim: TensorBase.log_softmax
WALL F.log_softmax                                   not implemented in torch._C shim: TensorBase.log_softmax
WALL _aten_dispatch aten.nll_loss_forward.default    aten op not implemented in torch._C shim
WALL torch._C._nn.nll_loss_forward                   not implemented in torch._C shim
WALL torch._C._nn.nll_loss_nd                        not implemented in torch._C shim
WALL F.nll_loss                                      (the same, via nll_loss_nd)
WALL torch._C._nn.cross_entropy_loss                 not implemented in torch._C shim
WALL F.cross_entropy                                 (the same)
WALL nn.CrossEntropyLoss()                           (the same)
OK   F.pad(int64, (0,1), value=-100)                 [[1, 2, 3, -100]]
```

**Two kernels and six names**, and `transformers` reaches the loss through the last of them:
`ForCausalLMLoss` -> `fixed_cross_entropy` -> `nn.functional.cross_entropy` ->
`torch._C._nn.cross_entropy_loss`. None of the four `_nn`/composite names appears in any dispatch
trace, because each is `CompositeImplicitAutograd`:

```
cross_entropy_loss  ->  nll_loss_nd(log_softmax(self, class_dim, self.dtype), target, ...)
nll_loss_nd         ->  nll_loss                       (for a 1-D or 2-D input)
nll_loss            ->  nll_loss_forward(...)[0]
log_softmax.int     ->  _log_softmax(converted, dim, False)
```

so the trace shows `_log_softmax` and `nll_loss_forward` and nothing else — which is exactly what
§5.3 reported. This is the sixth time in this repository a gap has been a *name* rather than a
kernel (docs/architectures/ARCH20.md §5 and §9, docs/kernels/GROUPED_MM.md §6.1, docs/kernels/TRIL.md §2, docs/bindings/SPELLINGS.md), and
it is the same blindness `tools/golden/compare.py` has by construction: golden compares by dispatch
key, so it cannot see a missing name either. The golden cases below therefore carry a **spellings**
block that calls the names instead of the key.

<!-- DOCWATCH: op-implemented aten._log_softmax.default -->
<!-- DOCWATCH: op-implemented aten.nll_loss_forward.default -->
<!-- DOCWATCH: hasattr nll_loss_forward false -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json _log_softmax present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json log_softmax absent -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py cross_entropy_loss present -->

---

## 2. `_log_softmax`: upstream has two kernels and they do different arithmetic

The formula is the one everybody knows — `x - max - log(sum(exp(x - max)))` — and transcribing it
that way is wrong for two of the four dtypes.

### 2.1 The fork

`log_softmax_cpu_out` (`ATen/native/SoftMax.cpp`) picks between two kernels on whether `dim` is the
trailing axis of the contiguous input. Both are in
`ATen/native/cpu/LogSoftmaxKernelImpl.h`, and they disagree about where the sum is rounded:

| | `serial_vec_log_softmax_lastdim_range` | `serial_vec_logsoftmax_range` |
|---|---|---|
| chosen when | `dim == ndim - 1` | every other `dim` |
| sum buffer | `scalar_t[]` | `float[]` |
| log of the sum stored in | `scalar_t[]` | `float[]` |
| so on `bfloat16` | the sum is rounded to 8 significand bits, its log is rounded again | neither is rounded |

Measured against upstream over seven shape/dim combinations per dtype, counting **elements that
differ bit-for-bit** from upstream:

```
                                sum kept in f32     sum narrowed to dtype
  bfloat16  (3,5,9)   dim -1          26                     0
  bfloat16  (3,5,9)   dim  1           0                    27
  bfloat16  (1,300)   dim -1          11                     0
  bfloat16  (4,7)     dim  0           0                    11
  float16   (2,3,4,5) dim -1          19                     0
  float16   (4,7)     dim  0           0                     8
  float32   every combination          0                     0
  float64   every combination          0                     0
```

The last two rows are why this is worth a section: for `float32` and `float64` the narrowing is the
identity, so **a float-only test cannot see the fork at all**, and the obvious single-path
implementation passes every `float32` case anyone would write first.

### 2.2 The separating input, which is not the obvious one

Knowing the rule is not the same as being able to check it. One `bfloat16` ULP is about 0.4%
relative and `tools/golden/dtypes.py` allows 6% for that dtype, so an *ordinary* input differs by
one ULP whichever way the kernel is written and `math.isclose` absorbs it. That is the shape of
miss `docs/training/TRAIN.md` §5 records against a `log2` fault, and it would have happened again here.

The case that does separate is small and was constructed rather than found:

```
bfloat16 [0.0, ln(0.002)]     sum = 1 + 0.002 = 1.00203
                              bfloat16(1.00203) = 1.0        exactly, within half a ULP

  last-dim kernel   log(1.0)      = 0          -> output[0] =  0.0
  strided kernel    log(1.00203)  = 0.0019980  -> output[0] = -0.0019836
```

A **relative** difference of 1.0, because the value the two disagree about is near zero while the
disagreement is not. Both spellings are in `tools/golden/cases.py` — `(1,2)` at `dim=-1` takes the
last-dim kernel, the same two numbers as `(2,1)` at `dim=0` take the strided one — so a shim that
uses one rule for both fails exactly one of each pair.

**`float16` does not separate**, and that was measured rather than assumed: the same input gives
`-0.00195117` and `-0.00199699`, a difference of 4.6e-05, which `float16`'s 5e-3 *atol* absorbs.
It is carried in the case list as documentation of the near miss, not as a check.

### 2.3 Three more things read off the source rather than guessed

* **The order is `x - max - logsum`, left to right.** Upstream's own comment cites pytorch#11752:
  forming `max + tmp_sum` first loses the difference the computation is about when the logits are
  large and the spread is small.
* **The integral refusal names a different kernel on each side of the fork**, and it is measurable:

  ```
  _log_softmax(int64 (4,),  dim 0)   "log_softmax_lastdim_kernel_impl" not implemented for 'Long'
  _log_softmax(int64 (2,3), dim 1)   "log_softmax_lastdim_kernel_impl" ...
  _log_softmax(int64 (2,3), dim 0)   "log_softmax_kernel_impl"         ...
  ```

  Two golden cases, not one, because a hard-coded message passes the first and fails the third.
  (`_softmax`, next door in `aten.rs`, answers `softmax_lastdim_kernel_impl` for both. Upstream
  distinguishes those two as well — measured — so that is a pre-existing near-miss in that op. It
  is recorded here and deliberately not changed: it is on the eval hot path's refusal branch and
  outside this round.)
* **The fork is `dim + 1 == rank`, not `inner == 1`.** A shape like `(3,4,1)` at `dim=1` has an
  inner extent of 1 and still takes the *strided* kernel. There is a golden case at that shape for
  exactly this reason.

### 2.4 What landed

* `rust/torch_c/src/aten.rs` — `log_softmax_default` and `log_softmax_body`, the narrowing threaded
  through as an `Option<fn(f64) -> f64>` taken from the existing `float_narrower(tag)`.
* `rust/torch_c/src/overloads.json` — `_log_softmax`, the dispatched leaf.
* `rust/torch_c/src/bootstrap.py` — `Tensor.log_softmax` beside `Tensor.softmax`, and
  `torch.log_softmax` bound to it. **Not** an `overloads.json` entry: `aten::log_softmax.int` is
  `CompositeImplicitAutograd`, the `softmax` trap one line above it in the same file. The two
  spellings of one function land on opposite sides of that boundary, one underscore apart, and
  `test_grouped_mm_resolves_from_the_torch_level_name` now asserts both halves.

### 2.5 Agreement with upstream

Every combination in `/tmp/loss/ls_check.py` — 4 dtypes x 14 shape/dim pairs, plus the separators,
the `-inf`/`+inf`/`NaN` edges, seven refusals and seven spellings:

```
bfloat16, float16   bit-identical to upstream on every case, including both separators
float64             bit-identical except 3 combinations, worst |d| = 8.9e-16
float32             bit-identical except 4 combinations, worst |d| = 9.5e-07
```

The `float32`/`float64` residual is summation order: upstream reduces with
`Vectorized<float>` lanes and a tree, this kernel sums serially. It is the same residual
`_softmax` has carried since docs/models/SAMPLING.md and it is two orders of magnitude inside the
tolerances in `dtypes.py` (1e-5 and 1e-9). **It also means no `float32` golden case can separate a
summation order**, which is stated here rather than left for someone to discover. §5.2 measures
what that residual grows to at a real vocabulary width, which is larger than this table suggests.

---

## 3. `nll_loss_forward`: the second return value, and a cascade

### 3.1 `total_weight` is the part a forward-only test cannot see

The op returns **two** tensors and every caller in `transformers` drops the second.
`nll_loss_backward` takes it as an argument — it is the divisor the mean's gradient needs — so it
is not decoration, and its rules do not follow from the loss. Measured, all five:

| call | loss | `total_weight` |
|---|---|---|
| `reduction=none`, 2-D input | `[1.0, 0.25]` | **`0.0`** |
| `reduction=none`, 2-D, weighted | `[1.0, 1.0]` | **`0.0`** |
| `reduction=none`, 1-D input | `3.0` (a *scalar*) | `1.0` |
| `reduction=mean`, weighted | `0.4` | `5.0` (the weights, summed) |
| empty batch, `reduction=mean` | `nan` | `0.0` |

The first two are upstream writing `*total_weight_data = 0` at the top of `nll_loss_out_frame` and
then taking an early return that never updates it. The third is the 1-D input falling through to
the *reduce* path regardless of `reduction`, which also makes `reduction=none` produce a scalar
there and a vector two rows above it.

`_nll_pair_check` in `tools/golden/cases.py` therefore checks both members with equal weight. A
sabotage that computes the "obvious" count for the first row fails **146 cases** (§5.1, N1).

### 3.2 The summation is an eight-level cascade

Upstream does not sum the per-element losses in a loop. It accumulates into **eight partial sums
with a carry every 2^4 elements**, all in `scalar_t`. That is observable, not an implementation
detail — measured against a plain left-to-right sum in the same dtype:

```
  n=300   bfloat16   upstream -225        naive -226       (f64 reference -226.61255)
  n=300   float32    upstream 373.92365   naive 373.92377  (f64 373.92358)
  n=4096  bfloat16   upstream -1528       naive -1384      (f64 -1545.9946)
  n=8     bfloat16   upstream 33          naive 33.25      (f64 33.287109)
```

They diverge from **n=8** in `bfloat16`. Three details inside it are load-bearing, and each was
found by a mismatch rather than by reading:

1. **An ignored target `continue`s past the carry loop**, not just past the addition. So
   `ignore_index` changes *where* the carries land, not only what is summed. Running the carry
   anyway agrees on unweighted runs and drifts on `ignore_index=3` ones.
2. **`float32`/`float64` contract `sum -= data * weight` into an FMA and the reduced dtypes
   cannot.** `c10::BFloat16::operator*` returns a `BFloat16`, so the product is rounded before the
   subtraction; native `float`/`double` are contracted by the compiler. Using an FMA everywhere and
   using it nowhere both mismatch, **in opposite dtypes**, which is how the split was found — the
   first transcription was FMA-free and failed only `float32`-with-weight.
3. **`total_weight` for the unweighted case is a cast of a count**,
   `static_cast<scalar_t>(batch_size - num_ignored)`, so it rounds exactly once and never goes
   through the cascade. §5.1's N8 is the fault for this and it is the one that needed a new case.

With all three, the transcription agrees with upstream **1200 of 1200** combinations, bit for bit:
25 batch sizes from 1 to 5000 × 4 dtypes × {sum, mean} × 3 `ignore_index` × {weighted, unweighted}.
`/tmp/loss/cascade.py` is that harness.

Because the agreement is exact, `_nll_pair_check` compares **exactly, with no tolerance**. That is
not strictness for its own sake: a naive sum differs from upstream by far less than `float32`'s
1e-5 for any small batch, so a tolerance would let the wrong summation through for every case in
the list.

### 3.3 Checks, in upstream's order, and two that are not there

`ignore_index` is tested **before** the bounds check, so a target equal to an out-of-range
`ignore_index` is legal — `nll_loss_forward(x, [0, 77], None, mean, 77)` succeeds where the same
call with `ignore_index=-100` raises `IndexError: Target 77 is out of bounds.` Reversing those two
lines is §5.1's N6 and it fails 3 cases.

`reduction` is **not validated**. Upstream accepts `3` and treats it as anything-but-Mean, i.e. a
sum (measured: loss `1.25`, `total_weight` `2.0`, identical to `reduction=2`). This reproduces that
rather than adding a refusal upstream does not have.

The weight's dtype must match the input's **exactly** — it is `data_ptr<scalar_t>()` that raises,
not a promotion rule, and the message names both dtypes (`expected scalar type Float but found
Double`). It fires on the elementwise path as well as the reduce path, so both have cases.

### 3.4 What landed

* `rust/torch_c/src/aten.rs` — `nll_loss_forward_default` and `nll_cascade`.
* `rust/torch_c/src/bootstrap.py` — `torch._C._nn.nll_loss`, `nll_loss_nd` and
  `cross_entropy_loss`, the three composites §1 found. `nll_loss_nd`'s 4-D and >4-D arms refuse by
  naming `aten.nll_loss2d_forward.default`: that op reduces over a spatial extent, so `nll_loss` is
  not a slower road to the same answer.
* **`torch._C._nn.nll_loss_forward` is deliberately absent.** Upstream has no such name
  (`hasattr` is `False` on 2.13.0). The first climb in §1 listed it as a wall, and it was the
  probe that was wrong, not the shim — recorded because a shim that invented the name to make a
  probe green would have been worse than the refusal.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs log_softmax_body present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs nll_cascade present -->
<!-- DOCWATCH: symbol-in-file tools/golden/cases.py _bit_exact present -->
<!-- DOCWATCH: symbol-in-file tools/golden/cases.py _nll_pair_check present -->

---

## 4. The result: a real SmolLM2-135M loss

Real weights from the HF cache, `float32`, `.train()`, deterministic ids `(i*7919+13) % 49152`,
`labels=ids`, `S=8`, 134,515,008 parameters — the recipe docs/training/AUTOGRAD.md §5 used.

| | upstream | shim | |
|---|---|---|---|
| **`out.loss`** | `12.871366500854492` | `12.871352195739746` | **abs 1.43e-05, rel 1.11e-06** |
| `F.cross_entropy` on the same logits | `12.871366500854492` | `12.871352195739746` | the same value, so the two paths agree |
| `nll_loss_forward` `total_weight`, mean | `7.0` | `7.0` | **exact** |
| `nll_loss_forward` `total_weight`, none | `0.0` | `0.0` | **exact**, on the real model |

### 4.1 Attributing the 1.1e-06

The end-to-end number mixes this round's kernels with the shim's pre-existing forward, so it was
attributed rather than reported alone: the shim's own logits were dumped as exact `float32` bit
patterns and **rebuilt upstream**, so both sides ran the two loss ops on identical input.

```
quantity                            upstream                   shim          |diff|
nll_of_raw_logits          -14.182351112365723    -14.182351112365723      0.0000e+00
log_softmax[0,0]            -7.9750566482543945     -7.975008010864258      4.8637e-05
nll_of_shared_ls            12.871373176574707     12.871352195739746      2.0981e-05
nll_tw                                      7.0                    7.0      0.0000e+00
```

> **`nll_loss_forward` is bit-identical to upstream on the real thing** — a 49152-class, 8-row
> tensor, gathered and cascade-summed, `0.0000e+00`. The entire residual is `_log_softmax`'s
> summation order over the vocabulary.

That is worth having because it says which of the two kernels to look at if the number ever moves,
and it says the answer is not the one with the cascade in it.

---

## 5. Sabotage

18 faults, each the most plausible wrong shape for the thing it breaks. Every one was applied to
the source, **rebuilt**, and run through `tools/golden/compare.py` and `pytests/run.sh`.

### 5.1 The table

| # | fault | golden | smoke |
|---|---|---:|---|
| L1 | `_log_softmax` keeps the sum in f32 on both paths (never narrows) — *the obvious implementation* | 5 FAIL | 0 |
| L2 | …narrows on both paths | 6 FAIL | 0 |
| L3 | `x - (max + logsum)` instead of `x - max - logsum` | 1 FAIL | 0 |
| L4 | no max subtraction — `log(sum(exp(x)))` | 2 FAIL | 0 |
| L5 | last-dim chosen by `inner == 1` instead of `dim + 1 == rank` | 4 FAIL | 0 |
| L6 | one hard-coded refusal message (the `_softmax` shape) | **0** | 1 FAIL |
| L7 | `half_to_float=True` honoured instead of refused | 2 FAIL | 0 |
| L8 | kernel present, `Tensor.log_softmax` absent — *the §1 gap* | 4 FAIL | 0 |
| N1 | `total_weight` counted instead of 0 for `reduction=none`, 2-D | 146 FAIL | 0 |
| N2 | plain left-to-right sum instead of the cascade | 151 FAIL | 0 |
| N3 | the cascade carry runs for ignored targets too | 22 FAIL | 0 |
| N4 | no FMA in any dtype | 24 FAIL | 0 |
| N5 | FMA in every dtype, including the reduced ones | 28 FAIL | 0 |
| N6 | bounds check before the `ignore_index` test | 3 FAIL | 0 |
| N7 | mean divides by the kept count instead of `total_weight` | 73 FAIL | 0 |
| N8 | `total_weight` summed through the cascade instead of cast once | **0**, then 2 FAIL | 0 |
| N9 | empty batch answers 0 for mean instead of NaN | 1 FAIL | 0 |
| N10 | kernel present, `_nn.cross_entropy_loss` absent — *the §1 gap again* | 4 FAIL | 0 |

### 5.2 The four results that are more interesting than the counts

**L1 and L2 could not fail on the first attempt, and the reason is a tolerance.** Both came back
**0 golden failures** against a case list that already contained the `[0, ln(0.002)]` separator —
the case built specifically to see them. `dtypes.py` gives `bfloat16` `atol=6e-2`; the effect is
0.002. `math.isclose` absorbed it.

Widening the input does not fix that, and the bound is structural rather than a matter of searching
harder: the `bfloat16` rounding of the sum is at most 2^-9 relative, so `log(sum)` moves by at most
**0.00195 absolute** for *any* input, and reaching an absolute 6e-2 would need `log(sum) > 16` —
a reduction over e^16 elements. Nor can `rtol` help: the effect is one ULP, which is 0.4% relative
against a 6% tolerance.

The fix is `_bit_exact`, a `value_check` with no tolerance at all, applied to the separators and to
the whole reduced-dtype grid. It is a bound this kernel actually meets: measured, `bfloat16` and
`float16` agree with upstream bit for bit on all 14 shape/dim combinations **and at real
vocabulary width** — 0 of 393,216 elements differ on a `[8, 49152]` SmolLM2 logits tensor. With
`_bit_exact` in place L1 fails 5 cases and L2 fails 6.

*(The first `_bit_exact` checked values only, and `compare.py --self-test` immediately reported it
as accepting a wrong answer under the `shape` and `dtype` fault modes — a `value_check` replaces the
default pipeline entirely rather than adding to it. That is the self-test doing its job on a
comparator written in the same hour.)*

**N8 could not fail either, and needed a searched separator rather than a wider grid.** Summing
`1.0` through the cascade agrees with casting the count for every batch size in the grid, including
`n=300`. It stops agreeing where a `bfloat16` partial saturates — at 256 the ULP is 2, so `256 + 1`
is `256`. Searching `n` over `[250,270] ∪ [500,530] ∪ {1000,1023,1024,1025,2049,4097,300,301,320,
384,385}` found **ten** separating batch sizes and `n=300` is not one of them:

```
n=258 bfloat16   cast 258   cascade-of-ones 256
n=515 bfloat16   cast 516   cascade-of-ones 512
```

Two cases at `n=258` were added on that measurement, and N8 then fails both. `float16` never
separates in that range — its 11-bit significand counts exactly past 2048 — so this is a
`bfloat16`-only check and is labelled as one.

**L6 cannot be caught by golden, and that is correct.** `expect="both_error"` asserts that both
sides refuse, not that they refuse alike — deliberately, since this shim prefixes its messages with
the op key. So the two `log_softmax_lastdim_kernel_impl` / `log_softmax_kernel_impl` cases pass
with a single hard-coded message. It is checked in
`test_log_softmax_names_the_kernel_its_dim_actually_selected` instead, where a string can be
compared, and that test fails on L6.

**L8 and N10 are the §1 finding as a fault.** Both leave the kernel completely intact and remove
only a *name*, and both are caught **only** by the spellings cases — every dispatch-key case in the
op's list passes. That is the blindness §1 describes, demonstrated rather than asserted: golden
compares by dispatch key, so without those cases a shim with both kernels and no
`torch._C._nn.cross_entropy_loss` would show 7232/7232.

### 5.3 What each case would still pass

Asked of each, as the brief requires:

* No `float32` case can separate a **summation order** — see §2.5 and §5.4. The `float32` entries
  in the grid check the formula, the refusals and the shape rules, not the accumulation.
* The refusal cases check *that* both sides refuse and not *how*; only the smoke test checks
  message text, and only for `_log_softmax`'s two kernel names.
* `_nll_pair_check` compares exactly, so it would catch any arithmetic change — but it says nothing
  about **performance**: the cascade is reproduced for its rounding, and the shim's version is a
  scalar loop either way.
* Nothing here exercises `nll_loss2d`, `nll_loss_backward`, or the two `cross_entropy_loss`
  branches §6 refuses.

### 5.4 The one number that is outside tolerance, stated rather than buried

At a **real vocabulary width** the `float32` summation-order residual is larger than §2.5's table
suggests. Measured on the SmolLM2 logits, `[8, 49152]`, against upstream:

```
bfloat16   0 of 393216 elements differ                      (bit-exact)
float16    0 of 393216 elements differ                      (bit-exact)
float32    392256 of 393216 differ, worst |d| = 1.49e-04, worst rel = 5.38e-04
```

**`5.38e-04` exceeds `dtypes.py`'s `float32` `rtol` of 1e-5**, and no case in this file sees it
because the widest `float32` case is 6 elements. It is a serial sum of 49,152 terms against
upstream's 4-lane one; the error grows with `n` and the case list does not.

It is left rather than fixed, for a reason worth stating: **upstream's own `float32` answer is not
reproducible across ISAs.** `map_reduce_all` accumulates in `Vectorized<float>` lanes — 4 on NEON,
8 on AVX2, 16 on AVX512 — so matching it bit for bit would mean matching *this machine's* upstream
and diverging from another's. Bit-exactness at `float32` is not a well-defined target here, which
is exactly why the reduced dtypes' bit-exactness above is worth having: those go through the
narrowing path, where rounding the sum to 8 significand bits erases the lane-order difference
entirely. That is the mechanism, and the 0-of-393216 at vocabulary width is the evidence for it.

The consequence for §4 is bounded and measured: it moves the SmolLM2 loss by 2.1e-05, i.e.
1.6e-06 relative.

#### 5.4.1 It is now watched, and the fix is still not attempted

Everything above stands. The `float32` residual is **not** fixed and should not be — the argument
in the paragraph above is the whole reason, and it has not weakened.

What did change is the last clause of the second paragraph: *"no case in this file sees it because
the widest `float32` case is 6 elements."* That was true, and it meant this number lived only in
this document. **A number in a document does not fail when it moves.** There is now a case at real
vocabulary width whose comparator asserts a *ceiling* on the disagreement rather than its absence:

```
_log_softmax(float32, [1, 49152] -- REAL VOCABULARY WIDTH)
    49152 deterministic LCG values in [-8, 8], with one 15.0 so the row's largest
    log-probability is -1.330504 and the relative error is not divided away

    measured   all 49152 elements differ
               worst |d| = 4.39e-05      ceiling 9e-05
               worst rel = 3.29e-05      ceiling 7e-05
```

**3.29e-05 is three times `dtypes.py`'s `float32` rtol of 1e-5**, which is the point: under the
default pipeline this case is a failure, and it is not one. The ceilings are the measured values
with roughly 2× headroom, so re-measurement noise does not move them and a real regression does.

The comparator is `_bounded_divergence(max_abs, max_rel)`, and it is new. Two comparators already
existed for the two ordinary answers — the default pipeline ("agree within the dtype's tolerance")
and `_bit_exact` ("agree exactly") — and neither can say the third thing this repository keeps
having to say: *these do not agree, here is by how much, and the cause is a property of how
upstream's wheel was compiled rather than of the operator.* `--self-test` puts it through all
eleven fault modes and it catches the same seven the default pipeline does, so it is a ceiling and
not a hole.

Three other divergences of exactly this shape were measured in the same round and are recorded the
same way, in `docs/numerics/SCALAR.md` §8: `softplus`'s Sleef-versus-scalar tail at `float32` (one ULP, at
`n >= 8` only), `add.Scalar`/`sub.Scalar`'s fused multiply-add (worst 3.2e-05 relative at
`float32`, by the same cancellation mechanism as this one), and `norm.ScalarOpt_dim`'s pairwise
`p = 2` sum (one ULP at `float64`). The pattern is worth naming: **wherever upstream's answer
depends on its vectoriser, the honest test is a ceiling, and the honest fix is usually none.**

<!-- DOCWATCH: symbol-in-file tools/golden/cases.py _bounded_divergence present -->

---

## 6. `zero_grad()`: a profiler marker, and it gates SGD too

`docs/training/AUTOGRAD.md` §7 measured this and named it exactly:

```
optimiser.zero_grad()   FAIL  profiler._record_function_enter_new.default
```

`torch.optim.Optimizer.zero_grad` and `_patch_step_function` both wrap their body in
`with torch.autograd.profiler.record_function(...)`, so **two profiler markers gate every optimiser
in `torch.optim`, SGD included** — and neither is arithmetic. Climbing it found a third name on the
same road that §7 did not list: `record_function.__exit__` opens with
`with torch._C.DisableTorchFunctionSubclass():`, which `surface.json` had harvested from the `.pyi`
as a `"function"` and turned into a raising stub.

### 6.1 A no-op is the honest body here, and it is checkable

The difference between this and a silent stub is that **nothing in this build can observe a
record**. Measured on this artefact, all three ways in:

```
torch.profiler.profile()              NotImplementedError: _supported_activities
torch.autograd.profiler.profile()     NotImplementedError: _ExperimentalConfig.trace_only
torch.autograd._profiler_enabled()    NotImplementedError: _profiler_enabled
```

There is no profiler to start, no way to ask whether one is running, and therefore no callback a
marker could reach. A stub for `bernoulli_` would lose a draw somebody wanted; a marker with no
listener loses nothing that exists. `test_the_profiler_markers_are_no_ops_and_nothing_could_observe_one`
pins **that claim** rather than the return value, and says by name what has gone wrong if a
profiler ever lands.

`DisableTorchFunctionSubclass` is a no-op for a separate reason, and it is worth keeping the two
apart: what it disables is subclass `__torch_function__` dispatch, and `_is_torch_function_enabled()`
already returns `False` here because no type in the vendored tree overrides the protocol. It is
**not** wired to `_MODE_STACK` — that is the torch-function *mode* stack, which upstream's other
name (`DisableTorchFunction`) governs, and clearing it would silently drop a
`with torch.device(...)` block spanning a `record_function` region, i.e. every `optimizer.step()`.
§8's Z4 is that mis-wiring and a test catches it.

### 6.2 What landed, and where

The two markers are answered **above** `_aten_dispatch`, in `_op_callable`, not in `aten.rs`. They
take a `str`, return an object with no storage, and have no dtype, device or shape;
`_aten_implemented()` means "has a kernel *and* `tools/golden/cases.py` compares it against
upstream", and golden cannot compare a marker. `_C._shim_profiler_markers` lists the three keys so
the size of that bypass is readable rather than inferred.

### 6.3 It works, and it works vacuously — which is the honest statement

On real SmolLM2-135M, 272 parameter tensors, 134,515,008 parameters:

```
SGD    zero_grad() and zero_grad(set_to_none=False)   OK
AdamW  zero_grad() and zero_grad(set_to_none=False)   OK
SGD/SGD+momentum/Adam/AdamW  .step()                  OK
```

**and every one of those completes without touching a gradient**, because `p.grad` is `None` for
every parameter, so `if p.grad is not None` skips the body. That is not a defect of this change —
it is `docs/training/AUTOGRAD.md` §7's second gap, `.grad` having no setter, and that document argues
deliberately for leaving it: *"making `.grad` writable while nothing writes to it would move the
shim from 'honestly reports no gradient' to 'has a slot that is always empty'"*. That argument is
still right and this round does not overturn it. What changed is that the **loop now runs**;
what it iterates over is still empty.

### 6.4 A real step: SGD works, Adam needs four things and not three

`docs/training/AUTOGRAD.md` §6.6 measured the optimiser step on upstream and concluded *SGD needs zero new
aten kernels, Adam needs three*. The first half is now demonstrated rather than predicted — through
the functional API, which takes gradients as an explicit list and so does not need a `.grad` setter:

```
torch.optim.sgd.sgd(params, grads, bufs, lr=0.1, ...)            OK   1.0 - 0.1*0.5 = 0.95
torch.optim.sgd.sgd(..., momentum=0.9)                           OK
```

**A real SGD step runs and is numerically right.** Adam does not, and it stops *before* reaching
any of the three:

```
torch.optim.adam.adam(...)   NotImplementedError: torch.is_complex(...) -- overload
                             resolution has no table entry for this op
```

`torch.is_complex` is a **name**, not a kernel — `Tensor.is_complex()` already works, and
`aten.is_complex.default` has no kernel either. It is the §1 finding a fourth time. Stubbing past
it, the next wall is `TensorBase.lerp_`, which *is* one of the three. So the requirement is:

| what Adam needs | size |
|---|---|
| `torch.is_complex` | a name; the member exists, so this is a spelling |
| `aten.lerp_.Scalar` | `self += weight * (end - self)`, elementwise in place, one scalar |
| `aten.addcmul_.default` | `self += value * t1 * t2`, elementwise in place |
| `aten.addcdiv_.default` | `self += value * t1 / t2`, elementwise in place |

All three kernels are the `add_`/`mul_`/`div_` family this shim already has, so **yes, they are
small** — but the count in §6.6 was three and the list is four, and the fourth is again the kind of
gap an op scan cannot see. None of them was implemented here: the brief's bar for this item is
`zero_grad()` completing, and the three kernels are only reachable once something writes a
gradient.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _PROFILER_MARKERS present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_profiler_markers_are_no_ops_and_nothing_could_observe_one present -->
<!-- DOCWATCH: op-not-implemented aten.lerp_.Scalar -->
<!-- DOCWATCH: op-not-implemented aten.addcmul_.default -->
<!-- DOCWATCH: op-not-implemented aten.addcdiv_.default -->
<!-- DOCWATCH: op-not-implemented aten.is_complex.default -->

---

## 7. `native_dropout`: capture can record a `.train()` forward

`docs/training/AUTOGRAD.md` §6.5 lists this as a *medium, and specific* limitation of the tape:

```
torch._C capture: cannot capture this region -- aten.bernoulli_.float writes in place;
capture refuses mutation so that aliasing cannot be observed, which is what keeps a trace
single-assignment
```

Reproduced on this build before the change, and it is the eager composite's fault rather than
dropout's: `torch.dropout`'s decomposition fires `empty_like`, `bernoulli_`, `div_.Scalar`,
`mul.Tensor`, and two of those four mutate.

### 7.1 It is not the composite with the mutation removed

`at::native::native_dropout_cpu` and `_dropout_impl` are different functions, and the differences
are the case list:

| | `_dropout_impl` (`torch.dropout`) | `native_dropout_cpu` |
|---|---|---|
| mask dtype | the input's | **`bool`** |
| where the scale goes | on the **mask**: `noise.div_(1-p)` | on the **output**: `.mul_(scale)` |
| `p` outside [0,1] | `TORCH_CHECK` naming **`p`** | no check of its own — `bernoulli_`'s, naming **`1-p`** |
| `train=False` | returns the input **object** | returns a **clone** |
| `numel == 0` | returns the input object | returns the input object, and a mask of the **input's dtype** |

The third is the sharpest and is measured: `native_dropout(x, 1.5, True)` raises
`bernoulli_ expects p to be in [0, 1], but got p=-0.5` — the *survival* probability — while
`native_dropout(x, 1.5, False)` **succeeds**, because that branch never reaches `bernoulli_`. A
range check written into the op itself would refuse a call upstream accepts; §8's D7 is that fault
and two cases catch it.

### 7.2 The scale is narrowed, and that was not readable from the source

`output.mul_(scale)` reads like an ordinary in-place scalar multiply, and a standalone
`Tensor.mul_(python_float)` on a reduced dtype does **not** narrow its scalar — `mul_kernel`'s
reduced-float branch takes `original_scalar_value<opmath_t>`, which is `float`
(docs/training/TRAIN.md §5, docs/numerics/SCALAR.md). Stepping the C++ body from Python and calling the kernel give
different answers:

```
bfloat16, x = -9.875, p = 0.7, on a survivor
  x.mul(mask).mul_(scale)   step by step from Python  ->  -33.0
  native_dropout(x, 0.7, True)                        ->  -32.75
  -9.875 * bfloat16(3.3333...) = -9.875 * 3.328125    ->  -32.75
```

The narrowed reading reproduces upstream on **1280 of 1280** combinations (4 dtypes × 5 values of
`p` × 64 elements); the un-narrowed one misses 41 of 377 in the development harness and **0 of 377
in `float32` alone**. This is the family docs/numerics/SCALAR.md closed by recording that it *has no rule to
infer* — `hardshrink` narrows, `softshrink` widens — so it is measured per op, and this op narrows.

### 7.3 Which makes the substitution safe, and that is the point

The two spellings put the scale in different places and **agree bit for bit anyway**. Measured
upstream over 4 dtypes × 6 values of `p`:

```
0 of 25872 survivors differ between torch.dropout and native_dropout
```

The reason is the narrowing: the eager path stores its scale *into a mask of the input's dtype*, so
it multiplies by `dtype(1/(1-p))`; the functional path narrows the scalar and multiplies by the
same thing. **So rewriting `torch.dropout` to `native_dropout` inside a capture region does not
change any number** — and it would have, silently, on exactly the architectures this is for, if the
scale had not been narrowed. That is the property `test_capture_takes_the_functional_dropout_and_
only_inside_a_region` asserts, and D3 fails it.

The document's first draft claimed the opposite here — that the survivors *differ* by an ULP — and
the test written on that claim failed immediately. It was the assertion that was wrong.

### 7.4 What landed

* `rust/torch_c/src/aten.rs` — `native_dropout_default`, **one kernel and not a decomposition**. A
  `bootstrap.py` decomposition would emit its steps through the one door and capture would record
  `bernoulli_` among them, which is the thing being fixed.
* `rust/torch_c/src/bootstrap.py` — `_dropout_impl` takes the `native_dropout` route **only while
  `_capture_active()`**, and `torch.native_dropout` is spelled (`hasattr(torch,
  'native_dropout')` is `True` upstream). Outside a region eager keeps the eager path, because
  upstream's CPU eager never reaches `native_dropout` either
  (`is_fused_kernel_acceptable` wants CUDA/XPU/lazy). Taking it *inside* one is following upstream:
  `native_dropout` is precisely what functionalisation rewrites eager dropout to.

### 7.5 The result: 14 of 23 to 17 of 23, and it is three architectures, not four

The `.train()` forward of each architecture, captured — **model built outside the region**, because
weight initialisation calls `normal_`/`uniform_` and a tape over a training step records the
forward, not the constructor:

| | before | after |
|---|---:|---:|
| architectures capturable in `.train()` | **14/23** | **17/23** |

The three that crossed are `gpt2`, `bert` and `gpt_bigcode`, each recording **4
`native_dropout` nodes** — and they are exactly the three that refused on `aten.bernoulli_.float`.

**`opt` did not cross, and it is not a dropout wall.** It refused before and after on
`aten._local_scalar_dense.default reads a tensor value onto the host`, a data-dependent host read,
so it never reached `bernoulli_` at all. `docs/training/AUTOGRAD.md` §6.5 names four architectures; the
measurement says three of them were blocked on this and the fourth was blocked on something else.

The six still refusing, with the wall each stops at — none of them dropout:

```
opt, deberta, deberta_v2   aten._local_scalar_dense.default   a host read
mixtral                    aten.div_.Tensor                   in place
falcon, vits               aten.add_.Tensor                   in place
```

<!-- DOCWATCH: op-implemented aten.native_dropout.default -->
<!-- DOCWATCH: hasattr native_dropout true -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs native_dropout_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_capture_takes_the_functional_dropout_and_only_inside_a_region present -->

---

## 8. Sabotage: 26 faults

Every one applied to the source, **rebuilt**, and run through `tools/golden/compare.py` and
`pytests/run.sh`. §5 has L1–L8 and N1–N10; this section adds Z1–Z4 and D1–D8 and then says what
none of them can see.

| # | fault | golden | smoke |
|---|---|---:|---|
| Z1 | `_record_function_enter_new` returns `None` | 0 | 1 FAIL |
| Z2 | only the `.default` exit overload registered, not `._RecordFunction` | 0 | 1 FAIL |
| Z3 | `DisableTorchFunctionSubclass` left as the harvested raising stub | 0 | 1 FAIL |
| Z4 | …wired to the torch-function **mode** stack instead of being a no-op | 0 | 1 FAIL |
| D1 | the capture route removed — kernel present, wiring absent | 0 | 1 FAIL |
| D2 | scale the **mask** instead of the output | **0** | **0** |
| D3 | the scale **not narrowed** to the input's dtype | 40 FAIL | 1 FAIL |
| D4 | the `numel == 0` mask made `bool`, tidying upstream's quirk | 2 FAIL | 0 |
| D5 | `train=False` returns the input object instead of a clone | **0**, then 1 FAIL | 0 |
| D6 | the reciprocal left unguarded, so `p == 1` gives `inf` | 25 FAIL | 0 |
| D7 | `torch.dropout`'s range check, on `p`, at the top | **0**, then 2 FAIL | 0 |
| D8 | `train=False` draws from the stream anyway | 3 FAIL | 0 |

### 8.1 Z1–Z4 are caught only by the smoke test, and that is structural

None of the four moves a number, so golden — which compares values by dispatch key — reports
0 failures for all of them. They are caught by
`test_the_profiler_markers_are_no_ops_and_nothing_could_observe_one`, which is where a claim about
*names and context managers* can live. Z4 in particular is caught by an assertion that exists only
because the comment next to the code named the mis-wiring: the mode stack must be the same depth
inside the block as outside it.

### 8.2 D2 cannot fail, and it is right that it cannot

Scaling the mask and scaling the output are **the same function once the scale is narrowed**:

```
mine  narrow(narrow(x * 1.0) * s)        s = narrow(1/(1-p))
D2    narrow(x * narrow(1.0 * s))
```

`narrow` is idempotent and `s` is already narrowed, so both are `narrow(x * s)` for every input —
including the `inf * 0 = NaN` and signed-zero paths, checked. This is §5.2's shape again: a fault
that cannot be caught because it is not a different computation. It is worth having in the table
because *the difference between the two spellings is real in the source* and only the narrowing
makes them coincide; D3, which removes the narrowing, fails 40 cases.

### 8.3 D5 and D7 could not fail on the first attempt

**D5** changes only object identity, which no value comparison sees. Fixed by four cases that
return `float(out is input)` as a one-element tensor — the only shape this harness compares — and
they pin both answers: a clone on the `train` and `train=False` branches, the input itself on the
`numel == 0` one.

**D7**, as first written, inserted the range check just above `let p1m`, which is *below* the
`train=False` early return — so the branch the check was supposed to break never reached it, and
golden's `both_error` cases pass whatever message the other branch raises. Rewritten to put the
check where `_dropout_impl` has it (at the top, above every branch), it fails the two
`native_dropout(p=1.5, train=False) [ACCEPTED on both sides]` cases. **The first version of the
fault was wrong, not the case list** — but it was only visible because the fault was run.

### 8.4 What this suite still cannot see

* **No `float32` case separates a summation order** (§5.4), and at real vocabulary width the
  `float32` `_log_softmax` residual exceeds the harness's own tolerance.
* **Nothing here differentiates.** Every kernel is a forward; `nll_loss_backward`,
  `native_dropout_backward` and `_log_softmax_backward_data` are all absent, and the mask
  `native_dropout` now returns has no consumer in this repository yet.
* **The capture sweep is not a gate.** It is a measurement in `/tmp/loss/capsweep.py`, not a test
  in `pytests/`, so 17/23 can regress to 14/23 without anything going red except the single
  `torch.dropout` capture assertion in `test_capture_takes_the_functional_dropout_and_only_inside_a_region`.
  That assertion covers the *mechanism*; it does not cover the architectures.
* **`cross_entropy_loss`'s other two branches** (§9) are refused, not implemented, so nothing here
  says what they would answer.

---

## 9. What this round did not do

| | why |
|---|---|
| `cross_entropy_loss(probability targets)` | upstream selects it on `self.sizes() == target.sizes()` and it is a different formula, `-(log_softmax(self) * target).sum()`. Refused by name; it uses only ops this shim has, so it is a small piece of work, not a blocked one |
| `cross_entropy_loss(label_smoothing > 0)` | a third branch, blending a uniform term into the target. `F.cross_entropy`'s default is `0.0` and `transformers` never overrides it |
| `nll_loss_nd` for rank 3, 4 and above | routes to `aten.nll_loss2d_forward.default`, which reduces over a spatial extent. Substituting `nll_loss` would be silently wrong rather than slow |
| Adam's three kernels and `torch.is_complex` | §6.4. Reachable only once something writes a gradient |
| `.grad`'s setter | docs/training/AUTOGRAD.md §7 argues for leaving it, and that argument still holds. It is the next decision, not an oversight |
| the `profiler::` schema table | the three keys answered above the door still report a placeholder schema. `_NON_ATEN_SCHEMA_TEXT` would need `profiler` added to `verify_schemas.py`'s `NON_ATEN_NAMESPACES`, and that check demands *every* op in a namespace — six here, two of them carrying `Future(t)` types the shim's parser has not been asked for |
| any backward | the whole document is still a forward. docs/training/AUTOGRAD.md §6.6 step 3 is unmoved |

<!-- DOCWATCH: op-not-implemented aten.nll_loss2d_forward.default -->
<!-- DOCWATCH: op-not-implemented aten.nll_loss_backward.default -->
<!-- DOCWATCH: op-not-implemented aten.native_dropout_backward.default -->
<!-- DOCWATCH: op-not-implemented aten._log_softmax_backward_data.default -->

---

## 10. Gates

| gate | before | after |
|---|---|---|
| `pytests/run.sh` | 293 ok, 0 FAIL | **296 ok, 0 FAIL** |
| `run.sh` DOCWATCH | 71/71 | **71/71** |
| `tools/golden/compare.py` | 6675/6675, ops=163, pending 1 | **7447/7447, ops=166, pending 1** |
| `compare.py --self-test` | 16 comparators × 11 fault modes | **19 comparators × 11 fault modes** |
| `verify_schemas.py` | 4465/4465 | **4475/4475** |
| sweep26 (`.eval()`) | 26/26 | **26/26** |
| sweeptrain (`.train()`) | 26/26 | **26/26** |
| capture, `.train()`, per architecture | 14/23 | **17/23** |

`+772` golden cases across three ops; `+3` smoke tests; `+10` schema entries. `ops=166` is `+3`:
`_log_softmax`, `nll_loss_forward`, `native_dropout`.

### 10.1 The eval-mode numbers that must not move, and did not

`docs/numerics/SEQLEN.md` §1.3's prefill logits sha256 over real SmolLM2-135M, re-measured on the final
artefact. A training-mode kernel that changes an eval-mode result is a bug, and `native_dropout`'s
wiring sits inside `_dropout_impl`, which every eval forward calls.

| S | f32 | | bf16 | |
|---:|---|:--:|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | ✅ | `8ef1550ea33c4f3d…` | ✅ |
| 32 | `331668f36da02f21…` | ✅ | `b81325c83a0a3d15…` | ✅ |
| 128 | `00159a9dbd308eda…` | ✅ | `7ff8e9334449b147…` | ✅ |
| 512 | `07c2797dabc4552e…` | ✅ | `9ab1e82f01378e38…` | ✅ |
| 1024 | `eda1e173727bb7f5…` | ✅ | — | |

All nine equal `docs/training/TRAIN.md` §6's values.

---

## 11. Every command in this document

```sh
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-trainstep
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh
PY=/Volumes/macMini/caches/spike-venv/bin/python
SHIM="PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY"     # VENDOR.md wall 3

# §1  the walls, climbed one rung at a time
$SHIM /tmp/loss/wall.py

# §2  _log_softmax
$PY   /tmp/loss/probe_ls.py                      # which accumulation, per dtype x dim
$SHIM /tmp/loss/ls_check.py shim  &&  $PY /tmp/loss/ls_check.py up

# §3  nll_loss_forward
$PY   /tmp/loss/probe_nll.py                     # total_weight, reductions, refusals
$PY   /tmp/loss/probe_nll2.py                    # the cascade, first transcription
$PY   /tmp/loss/cascade.py                       # 1200/1200 after the three fixes
$SHIM /tmp/loss/nll_check.py shim &&  $PY /tmp/loss/nll_check.py up

# §4  the real loss, and its attribution
$SHIM /tmp/loss/smol_loss.py shim 8  &&  $PY /tmp/loss/smol_loss.py up 8
$SHIM /tmp/loss/attribute.py dump    &&  $SHIM /tmp/loss/attribute.py shim
$PY   /tmp/loss/attribute.py up                  # identical logits on both sides
$SHIM /tmp/loss/wide_bf16.py shim    &&  $PY /tmp/loss/wide_bf16.py up   # §5.4

# §6  zero_grad
$SHIM /tmp/loss/zg.py                            # the markers, then all four optimisers
$SHIM /tmp/loss/step.py                          # a real SGD step, and Adam's fourth wall
$SHIM /tmp/loss/adam_depth.py

# §7  native_dropout and capture
$SHIM /tmp/loss/cap2.py                          # the refusal, then the acceptance
$SHIM /tmp/loss/nd_check.py shim &&  $PY /tmp/loss/nd_check.py up   # 377/377
$SHIM /tmp/loss/capsweep.py /tmp/loss/cs         # 14/23 -> 17/23

# §8  sabotage: 26 faults, each rebuilt
sh /tmp/loss/sab.sh <tag> /tmp/loss/faults/<tag>.py

# §10 gates
PYTHON=$PY sh rust/torch_c/pytests/run.sh
$PY tools/golden/compare.py  ;  $PY tools/golden/compare.py --self-test
$PY rust/torch_c/pytests/verify_schemas.py
$SHIM /tmp/train/sweeptrain.py /tmp/loss/F/tr    ;  $SHIM /tmp/k26/sweep26.py /tmp/loss/F/ev
$SHIM /tmp/loss/seqlen.py f32                    ;  $SHIM /tmp/loss/seqlen.py bf16
```

The scratch harnesses live under `/tmp/loss/` and are reproduced nowhere else; every number they
produce is quoted above with the command that made it.
