# A training step: the tape, the gradients, and where they stop agreeing

`docs/training/AUTOGRAD.md` §6.6 ended with a recommendation and an order of work:

> **Build the tape.** Reverse-walk a captured, Core-ATen-lowered trace, with a `grad` map keyed on
> the trace's existing value identities, and derivative rules written against Core ATen only.

Steps 1 and 2 of that order landed in `docs/training/LOSS.md` — a real loss, and `native_dropout` so a
`.train()` forward captures. This is step 3, and it reached step 4 and step 5 as well.

**The target was one optimiser step on a real model that moves the weights the way upstream moves
them.** Not full autograd. The smallest thing that is honestly a training step:

```
forward with labels -> loss -> backward -> p.grad populated -> SGD step -> weights moved
```

Environment: `/Volumes/macMini/caches/spike-venv/bin/python`, torch 2.13.0, transformers 5.15.1,
worktree at `develop` `a39d0b4`. Upstream is the oracle for every gradient below.

### Answers, before the evidence

| question | answer |
|---|---|
| Does a training step run? | **Yes, on the whole of SmolLM2-135M.** 1862 nodes, 272 of 272 parameters get a gradient, SGD moves all 272 (§4) |
| Do the gradients agree with upstream? | **All 134,515,008 of them were compared.** Median relative L2 **8.8e-05**, element sign agreement **99.9987%** (§4.2) |
| Is SDPA's backward the wall `docs/training/AUTOGRAD.md` §5.1 predicted? | **No.** It is the one op there that has no Core ATen decomposition — but a *rule* is not a kernel, and the tape recomputes the attention it needs. §3.4, and §4.4 measures that removing SDPA entirely changes the residual by 1% |
| Where does the 8.8e-05 come from, then? | **The forward, not the tape.** The same tape on the same model in `float64` agrees to **8.5e-07** (§4.4) |
| Are the rules right in isolation? | Every one of the **56** is checked against central differences in `float64`, an oracle that shares no code with them; a decoder-shaped `nn.Linear`/MLP case is **bit-identical** to upstream (§2, §5) |
| Did a check fail? | **Yes, and it was the check.** A finite-difference probe on the real model disagrees with the tape by 600× — and with **upstream's own autograd** by the same 600× (§6) |
| How much of upstream's machinery did this need? | **None of `torch/csrc/autograd`.** No `VariableType` wrappers, no `AutogradMeta`, no engine, no version counters. One reverse walk of a list, and 56 rules that are compositions over ops that already existed (§1) |

Written incrementally, one stage at a time, for the reason `docs/kernels/KERNELS26.md` §0 gives.

### The baseline, every gate, before any edit

```
pytests/run.sh                296 ok, 0 FAIL, DOCWATCH 95/95      exit 0
tools/golden/compare.py       7447/7447, ops=166, pending=1       exit 0
compare.py --self-test        19 comparators x 11 fault modes     exit 0
verify_schemas.py             4475/4475                           exit 0
sweep26   (shim, .eval())     26/26                               exit 0
sweeptrain (shim, .train())   26/26                               exit 0
```

---

## 1. What the tape is, and what it did not need

`rust/torch_c/src/tape.rs`, 1644 lines, and the shape of it is the point:

```
replay   the forward, keeping every intermediate     (PyCaptureTrace::run)
seed     the declared outputs' gradients
walk     the node list backwards, once, accumulating into a map keyed on the trace's own Ref
```

Three properties come from **capture** rather than from anything in the new file, and they are
exactly the properties upstream spends `torch/csrc/autograd` on:

| property | how upstream gets it | how this gets it |
|---|---|---|
| single assignment | `ADInplaceOrView` + a version counter per tensor | capture refuses in-place ops (docs/graph/CAPTURE.md §4), so nothing recorded is ever overwritten |
| a graph to walk | a generated `VariableType` wrapper per op allocates a `Node` and links it | the record **is** the graph; the recorder is one line at one door |
| a reverse order | ready queue, dependency counts, `GraphTask`, 1862 lines of `engine.cpp` | the region is straight-line, so it is the forward order reversed |

`docs/training/AUTOGRAD.md` §2.5 listed four layers this shim would have to provide. It provided **one and a
half**: the walk, and a `.grad` slot. `VariableType`'s 163 wrappers, `AutogradMeta`, the engine's
device threads and reentrancy, and version counters are all absent and none of them was missed.

### 1.1 The rules are compositions, and that is what dissolves §4's bill

`docs/training/AUTOGRAD.md` §4 counted the derivative requirement as **25 formulas needing their own
kernel**, reaching 24 distinct backward ops of which **4 must be hand-written**. That is the bill
for a *kernel* set. It is not the bill for a *rule* set, for one reason that only became clear once
the backward existed:

> **The backward runs outside a capture region.** It may use ops capture would refuse to record, it
> may mutate, and it may recompute a value instead of reading a saved one.

So `_softmax_backward_data` is not needed — `g - out * rowsum(g * out)` is. `nll_loss_backward` is
not needed — a `scatter` into a zero buffer is.
`_scaled_dot_product_flash_attention_for_cpu_backward`, the one op on SmolLM2's path with neither a
CPU kernel here nor a Core ATen decomposition, is not needed either: §3.4.

**Zero new aten kernels were written for this document.** `ops=166` before and after, and
`tools/golden/compare.py` is unchanged at 7447/7447 — which is the check that says so, because a
new kernel would have had to appear there.

### 1.2 The surface

```python
trace.backward(inputs, grad_outputs=None, wrt_constants=None)
    -> {"inputs": [Tensor | None, ...], "constants": [Tensor | None, ...]}

trace.differentiable(wrt_constants=None)
    -> {"nodes": int, "nodes_on_a_gradient_path": int,
        "covered": {op: count}, "missing": {op: count}}

_C._tape_rules() -> [op, ...]        # the 56 rules, readable from Python
```

`constants` is where a model's weights are: capture burns in every tensor it was not handed as an
input (docs/graph/CAPTURE.md §2), which is the same split `ExportedProgram.graph_signature` makes between
user inputs and lifted parameters. So *"the gradient of the loss with respect to this parameter"* is
*"the gradient at this constant"*, matched by object identity through `trace.constant_values`.

`differentiable()` exists so that **"what stops this model" is answerable without running a backward
and reading an exception.** Every wall table in this document is generated from it. Hand-kept op
lists have gone stale in this repository five times.

### 1.3 The one thing that is paid twice

`PyCaptureTrace` keeps the *shape* of every intermediate, not the intermediate — capture drops its
keepalives at `_capture_end` (docs/graph/CAPTURE.md §6). So `backward()` **replays the forward first**, to
materialise the activations it needs, and a backward therefore costs two forwards rather than one.
That is a deliberate default: holding every activation for the life of every trace would be wrong
for the traces that are never differentiated. Measured on SmolLM2-135M at S=8, the replay plus the
whole reverse walk is **0.4 s**.

`run()` is a refactor of `replay()`, not a second interpreter — `replay` is now `run` plus a
projection onto the declared outputs. A backward that materialised activations its own way would be
differentiating a different forward from the one `replay` proves equal to eager.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs RULE_OPS present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs sdpa_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs nll_loss_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs crate::tape::backward present -->

---

## 2. Stage one: a `nn.Linear`, and a gradient that is bit-identical

The smallest thing that is a gradient. `nn.Linear(8, 4)` on a `[3, 8]` input, `.sum()` as the loss,
weights built from a deterministic generator so both sides see identical bytes with no shared RNG.

The trace is three nodes and it is worth reading, because it is the whole design in miniature:

```
%0 = aten.t.default(%c0)                    ->  [8, 4]
%1 = aten.addmm.default(%c1, %in0, %0)      ->  [3, 4]
%2 = aten.sum.default() {'self': %1}        ->  []
constants  [[4, 8], [4]]      <- the weight and the bias
```

`%c0` and `%c1` are the parameters. Two things follow immediately: the arguments arrive **all
keyword** for one node and **all positional** for the next (docs/graph/CAPTURE.md §2 measured that; the
vendored tree binds by name before dispatching), so every rule binds against a schema's parameter
names rather than reading `args[0]`; and a gradient for a parameter is a gradient at a constant.

Against upstream, comparing **bit patterns** rather than with a tolerance:

| | n | elements differing bit-for-bit | max abs |
|---|---:|---:|---|
| `grad weight` | 32 | **0** | 0.0 |
| `grad bias` | 4 | **0** | 0.0 |
| `grad input` | 24 | **0** | 0.0 |

The loss itself is *not* bit-identical (4.8e-07) — the shim's `sum` reduces serially and upstream's
does not — and the gradients are, because `sum`'s derivative never touches the sum.

---

## 3. Stage two: a loss, an activation, and attention

Four more cases, each adding one thing that a `Linear` does not have. Weights and inputs are again
built from the same deterministic generator on both sides. `|d|/max|g|` is the largest elementwise
disagreement scaled by the largest gradient in that tensor — a plain relative error on a gradient
whose smallest entries are near zero says more about those entries than about the rule.

| case | what it adds | worst bit-diff | worst `|d|` | worst `|d|/max|g|` |
|---|---|---:|---|---|
| `linear_sum` | — | 0 of 60 | 0.0 | **0.0** |
| `linear_ce` | `cross_entropy`, i.e. `_log_softmax` + `nll_loss_forward` | 20 of 40 | 2.98e-08 | 1.27e-07 |
| `mlp_ce` | a hidden layer and `silu` | 85 of 128 | 5.96e-08 | 2.06e-07 |
| `mlp_ce_ignore` | `ignore_index=-100` on one row | 57 of 128 | 2.98e-08 | 1.03e-07 |
| `rmsnorm` | `pow`/`mean.dim`/`rsqrt`, and a `[6]` weight broadcast against a `[2, 3, 6]` | 19 of 36 | 4.77e-07 | 7.87e-08 |
| `attention` | SDPA, causal, **grouped-query 9:3** | 246 of 360 | 7.15e-07 | 8.54e-07 |
| `attention_mask` | SDPA with an additive `attn_mask` | 87 of 96 | 2.38e-07 | 4.49e-07 |

Everything is inside one `float32` ULP of the gradient's own scale. The bit-diff column is there to
say that these are *not* bit-identical and that the tolerance is doing work; §5's `float64` check is
what says the rules are right rather than merely close.

### 3.1 `nll_loss_forward`'s gradient reads the op's **second** result

`docs/training/LOSS.md` §3.1 found that `total_weight` is not decoration and that every caller in
`transformers` drops it. The backward is the caller that does not: for `reduction=Mean` the divisor
is `total_weight`, and **not** the number of rows. The two are equal for every unweighted call with
nothing ignored, which is every case anyone writes first — §7's T5 is that fault and the first
version of the test could not catch it.

The other two things the rule has to get right, both from the same section read backwards:

* an ignored row contributes nothing, and
* its target cannot be scattered with, because it is routinely out of range (`-100`). It is sent to
  column 0, which is only safe *because* its value is already zero. Zeroing after clamping rather
  than before is §7's T6.

### 3.2 The broadcast that is invisible until it is not

`reduce_to` — undo broadcasting by summing the axes it expanded — is the one piece of arithmetic
every elementwise rule needs and the easiest to leave out, because it does nothing whenever the two
operands already have the same shape. An RMSNorm is where it shows: `weight * hidden` is a `[576]`
against a `[1, 8, 576]`. §7's T3 removes it; without the `rmsnorm` and `mul.Tensor` broadcast cases
it fails nothing.

### 3.3 The `float32` residual is not where a `float32` test can see it

None of the seven cases above separates a *summation order*, for the reason `docs/training/LOSS.md` §5.3
gives about the forward: at `float32` the shim's serial reductions and upstream's vectorised ones
differ by less than any tolerance a small case can set. §5 is where that is answered, in `float64`,
against an oracle that is not upstream at all.

### 3.4 SDPA's backward: a rule, not a kernel

`docs/training/AUTOGRAD.md` §5.1 named `_scaled_dot_product_flash_attention_for_cpu_backward` as *"the only
genuinely new kernel on this list"* and §4.2 as one of four with neither a CPU kernel here nor a
Core ATen decomposition. **It is still true that the kernel does not exist, and it stopped nothing**,
because the tape needs a derivative and not a kernel:

```
  P  = softmax(scale * q k^T + mask)        out = P v
  dv = P^T dout                             dP  = dout v^T
  dS = P * (dP - rowsum(dP * P))
  dq = scale * dS k                         dk  = scale * dS^T q
```

Every op in that already existed. Three details are not optional and each is a measured property of
the forward rather than a choice:

* **`is_causal` is upper-left aligned**, not bottom-right — `aten.rs`'s own comment measured that
  on a (q=2, kv=5) pair. The rule rebuilds the mask the same way.
* **Grouped-query attention is inside the kernel**, so key and value are repeated to the query's
  head count before anything touches them, and the gradient has to be **summed back down over each
  group**. Getting this wrong yields a `[1, 9, ...]` gradient where a `[1, 3, ...]` was wanted, or —
  worse, and this is what §7's T8 does — the right shape holding one group's contribution instead of
  the sum of three.
* `dropout_p > 0` is refused by name, because the forward refuses it too and a gradient would need a
  draw that was never made.

What it costs is a second attention and one `[B, H, T, S]` probability matrix per layer — upstream's
fused kernel exists to avoid exactly that, so this is correctness bought with memory. §4.5.

<!-- DOCWATCH: op-implemented aten._scaled_dot_product_flash_attention_for_cpu.default -->
<!-- DOCWATCH: op-not-implemented aten._scaled_dot_product_flash_attention_for_cpu_backward.default -->
<!-- DOCWATCH: op-not-implemented aten.nll_loss_backward.default -->
<!-- DOCWATCH: op-not-implemented aten._log_softmax_backward_data.default -->
<!-- DOCWATCH: op-not-implemented aten.embedding_dense_backward.default -->

---

## 4. Stage three: the whole of SmolLM2-135M

Real weights from the HF cache, `float32`, `.train()`, the deterministic ids `(i*7919+13) % 49152`,
`labels=ids`, `S=8`, 134,515,008 parameters — the recipe `docs/training/AUTOGRAD.md` §5 and `docs/training/LOSS.md` §4
use. The **entire** `model(input_ids=ids, labels=ids)` call is captured, embedding included.

```
captured <CaptureTrace 1862 nodes, 1 inputs, 333 constants, 1 outputs>   0.1 s
loss 12.871352195739746           (upstream 12.871366500854492, docs/training/LOSS.md §4)
parameters that are trace constants: 272 of 272
nodes 1862, on a gradient path 1723, distinct ops 20, missing rules {}
backward 0.4 s
params with a grad: 272 of 272
```

The twenty ops a gradient flows through, from `differentiable()` rather than by hand:

```
211 aten.t.default           211 aten.matmul.default    272 aten.mul.Tensor
120 aten.add.Tensor          120 aten.cat.default       121 aten.slice.Tensor
120 aten.transpose.int        92 aten.view.default       61 aten.add.Scalar
 61 aten.mean.dim             61 aten.pow.Tensor_Scalar  61 aten.rsqrt.default
 60 aten.contiguous.default   60 aten.neg.default        30 aten.reshape.default
 30 aten.silu.default         30 aten._scaled_dot_product_flash_attention_for_cpu.default
  1 aten.embedding.default     1 aten._log_softmax.default   1 aten.nll_loss_forward.default
```

### 4.1 Two things the model taught that the small cases could not

**`cat` has an empty-tensor exemption and `DynamicCache` relies on it.** A 1-D size-0 operand is
skipped rather than checked against the concatenation axis, so layer 0 concatenates a `[0]` past key
with a `[1, 3, 8, 64]` present one. A rule that assumed every operand had the output's rank panicked
on the first real model it saw. It is handled by name now, and a non-empty operand of the wrong rank
is refused rather than guessed at.

**Token ids are constants too, and they are not differentiable.** With every constant a gradient
target, the reverse walk reached the `constant_pad_nd` that builds `transformers`' shifted labels
and asked for a derivative of an integer. The fix is upstream's own rule stated one step earlier: a
non-floating value is dropped from the wanted set whichever way it was named — which is exactly what
`requires_grad_` refuses on a non-floating tensor upstream. `constant_pad_nd` has a rule anyway,
because `F.pad` is genuinely on the path of any `labels=` forward whose *activations* are padded.

### 4.2 The gradient comparison: all 134,515,008 elements

There is no way to write a tensor out of this shim — `torch.save`, `Tensor.numpy` and
`untyped_storage` all refuse — so the comparison is done the other way round: **upstream writes its
gradients to a `.safetensors` file and the shim loads it**, and every number below is computed
inside the shim process against upstream's own bytes.

| | value |
|---|---|
| tensors compared | **272 of 272** |
| elements compared | **134,515,008** |
| relative L2 `‖u−s‖/‖u‖`, median over tensors | **8.780e-05** |
| relative L2, worst tensor | 3.031e-04 (`model.layers.24.input_layernorm.weight`) |
| worst single element | 3.933e-03 (`model.layers.0.self_attn.v_proj.weight`) |
| **elements agreeing in sign** | **134,513,262 / 134,515,008 = 0.999987** |
| tensors where upstream is exactly zero and the shim is not | **0** |

### 4.3 The step, and whether the weights move the way upstream moves them

`p.grad = ...` for all 272, then `torch.optim.SGD(lr=0.1).step()` — the real optimiser, through the
real `torch.optim`, which `docs/training/LOSS.md` §6 got as far as running vacuously.

```
tensors whose weights moved: 272 of 272
```

| | relative L2 (median) | worst | element sign agreement |
|---|---|---|---|
| the step `w_after − w_before` | **9.632e-05** | 3.032e-04 | 134,511,760 / 134,515,008 = **0.999976** |
| the weights after the step | 1.043e-07 | 1.233e-04 | 134,515,006 / 134,515,008 = **1.000000** |

**The weights move, and they move where upstream moves them**: 99.9976% of the 134.5M step
components point the same way, and the post-step weights agree to a median relative 1.0e-07.

### 4.4 Where the 8.8e-05 comes from, and where it does not

Two attributions, both measured rather than argued.

**It is not the SDPA composition.** Running the same model with `attn_implementation="eager"` —
which replaces the one fused op with `matmul`/`softmax`/`matmul` and removes §3.4's rule from the
path entirely — moves the residual by about one percent:

| | nodes on a gradient path | distinct ops | gradient rel-L2 (median) | sign agreement |
|---|---:|---:|---|---|
| `sdpa` | 1723 | 20 | 8.780e-05 | 0.999987 |
| `eager` | 2053 | 23 | 8.701e-05 | 0.999987 |

**It is the forward's arithmetic.** The same tape, the same model, all 30 layers, in `float64`:

| dtype | gradient rel-L2 (median) | worst |
|---|---|---|
| `float32` | 8.780e-05 | 3.031e-04 |
| **`float64`** | **8.451e-07** | 4.511e-05 |

A hundredfold drop from changing nothing but the arithmetic width says the rules are not what the
`float32` number is measuring. `docs/kernels/SDPA.md` §3 already recorded that this shim's flat attention
disagrees with upstream's blocked kernel on 3562 of 4096 `float32` elements *in the forward*, and
`docs/training/LOSS.md` §5.4 that `_log_softmax` at vocabulary width exceeds the harness's own `float32`
tolerance; a backward inherits both and amplifies them by the network's condition number.

*(The `float64` run needs its loss computed explicitly rather than through `labels=`. `transformers`'
`fixed_cross_entropy` calls `logits.float()`, so a `float64` model returns a **`float32`** loss on
both sides — which is upstream's behaviour, not a defect here, and it hid the whole experiment until
the fd probe in §6 turned up a loss quantised to `float32` ULPs.)*

### 4.5 The costs, stated

| | |
|---|---|
| capture | 0.1 s, 1862 nodes, 333 constants held by reference |
| backward, including the replay it needs | **0.4 s** at S=8 |
| the extra forward | §1.3 — a backward is two forwards, by design |
| SDPA's rule | a `[B, H, T, S]` probability matrix per layer, recomputed. §3.4 |
| the embedding's rule | ~~a `[vocab, tokens]` one-hot — 1.5 MB at S=8, 200 MB at S=1024~~ → **§13** |

**The embedding row is closed and this paragraph is superseded by §13.** It used to say that the
one-hot was used because `index_put_(accumulate=True)` was refused, that switching was "one line",
and that *"it is the memory that is wrong, not the answer"*. `docs/kernels/VIEWS.md` §7 landed the flag and
§13 took the switch. Two of those three statements held; the third did not. **The answer was wrong
too**, at `bfloat16`, and §13.2 measures it — the claim had been made from a `float32` reading, and
`float32` is exactly where the two compositions are provably identical.

---

## 5. The rules, against an oracle that is not upstream

Comparing against upstream is necessary and it is not sufficient, for a reason specific to this
file: a derivative rule here is a short expression over ops that already work, so a test that
recomputed the same expression would agree with a rule that had its operands the wrong way round.
Every one of the **56** rules is therefore also checked against **central differences in
`float64`**, which shares no code with the rule at all.

```
test_every_tape_rule_agrees_with_central_differences_in_float64
    56 cases, |tape - (f(x+h) - f(x-h))/2h| / max|fd| < 1e-5 for every element
```

`float64` with `h = 1e-6` puts truncation near 1e-13 and cancellation near 2e-9 on the O(1)
functions used, so the bound has two orders of margin.

Three things about the case construction, each of which was a hole first:

* **`sum(t)` is not enough to scalarise.** Its gradient is all ones through any shape op, which is
  exactly what a transposed, reversed or mis-sliced rule still returns. Every case ends
  `sum(exp(t))` so that each input element has a different gradient.
* **`sum(exp(log_softmax(x)))` is identically 1**, so its gradient is zero for every input *and
  every rule*, correct or not. The case weights the log-probabilities first. A test asserting
  "the gradient is not all zeros" is what caught it.
* **Symmetry hides pads and divisors.** `slice(x, 0, 1, 3)` of a `[4, 3]` leaves one row of padding
  on each side, so swapping them is invisible; an `nll_loss` with nothing ignored has
  `total_weight == rows`, so a mean that divides by the wrong one is invisible. Both cases are
  deliberately asymmetric now, and §7 is where that shows.

### 5.1 The rule table cannot go stale, structurally

`RULE_OPS` is a second list of op names, which is the exact shape of the failure `docs/verification/AUDIT.md`
found six times. So it is not read by a human: `_C._tape_rules()` exposes it,
`test_the_tape_has_a_gradient_case_for_every_rule_it_claims` asserts that the case table equals it
in both directions, and `differentiable()` reports against it. Adding a rule without a gradient case
fails the suite; so does leaving a case for a rule that was removed.

### 5.2 An op with no rule is refused by name

`docs/design/DESIGN.md` §6, applied to a derivative — and it matters more here than for a kernel, because a
wrong gradient looks exactly as plausible as a right one and the program keeps running.

```
NotImplementedError: torch._C tape: no derivative rule for aten.topk.default -- a gradient
reached it, and the tape refuses to guess. Add a rule in tape.rs and a gradient case in
pytests/test_shim.py; trace.differentiable() lists every op in a trace that would need one
```

`differentiable()` names it *before* a backward is run, which is what makes "what stops this model"
answerable without reading an exception.

---

## 6. A check that failed, and it was the check

The obvious way to validate a backward on a real model is finite differences, and it was run:
perturb one weight of SmolLM2 by ±1e-6 in `float64`, recompute the loss, compare. It disagreed with
the tape by a factor of **600**.

```
model.layers.0.self_attn.q_proj.weight[0,0]   tape 3.66e-04   central difference 6.53e-02
model.norm.weight[0]                          tape 2.94e-02   central difference 2.94e-02
```

The final norm agrees to eight digits; everything inside the transformer layers does not. Before
believing either side, the same probe was run **on upstream**, against upstream's own autograd:

```
upstream, same model, same weights, same h
model.layers.0.self_attn.q_proj.weight[0,0]   autograd 3.41e-04   central difference 2.09e-01
model.norm.weight[0]                          autograd -3.5588e-02  central difference -3.5588e-02
```

**Upstream's autograd fails the probe in the same place and by the same order.** So the probe is not
an oracle for this function at this scale, and the conclusion it invited — "the tape is wrong deep in
the network" — was not available. What made it usable at all in §5 is exactly what is missing here:
small, well-scaled, smooth functions where the linear term dominates the quadratic one at
`h = 1e-6`.

This is recorded rather than dropped because the failure mode it belongs to is the one this
repository keeps meeting from the other side. `docs/training/AUTOGRAD.md` §5.4's lesson was *a criterion I
wrote decided the answer*; this is the same lesson with the sign flipped — a criterion I wrote was
about to condemn code that was right, and the thing that stopped it was running the criterion
against a known-good implementation first.

---

## 7. Sabotage: 17 faults

Every one applied to `rust/torch_c/src/tape.rs`, **rebuilt**, and run through the five tape tests.

| # | fault | caught |
|---|---|---|
| T1 | `addmm`: `mat1 @ g` instead of `mat1^T @ g` | ✅ shape mismatch at `mm` |
| T2 | `mul.Tensor`: operands swapped, so `grad_self = g * self` | ✅ |
| T3 | `reduce_to` short-circuited — every broadcast left unreduced | ✅ |
| T4 | `_log_softmax`: the `exp(out) * rowsum(g)` term dropped, gradient passed through | ✅ |
| T5 | `nll_loss`: `reduction=Mean` divides by the row count, not `total_weight` | ✅ *(after §5's third bullet)* |
| T6 | `nll_loss`: the ignored row not zeroed, so its clamped column 0 collects a real gradient | ✅ *(same)* |
| T7 | SDPA: `dS = P * dP`, without the softmax Jacobian's row-sum | ✅ |
| T8 | SDPA: the grouped-query fold replaced by "take group 0" | ✅ *(after §5's third bullet)* |
| T9 | `slice`: the two pads swapped | ✅ *(same)* |
| T10 | `cat`: the offset never advances, so every operand reads the first piece | ✅ |
| T11 | `silu`: the naive `g * sigmoid(x)` | ✅ |
| T12 | `embedding`: a write instead of a sum — `index_put_(accumulate=False)`, which loses a repeated token's second contribution | ✅ |
| T13 | `rsqrt`: the sign of the exponent rule | ✅ |
| T14 | `mean.dim`: the division by the reduced extent dropped | ✅ |
| T15 | `detach`: a gradient flows through the op whose purpose is that none does | ✅ |
| T16 | `where.self`: the two branches swapped | ✅ |
| T17 | `contiguous`/`clone` answered through `alias` instead of returning `g` | **❌, and correctly** |

**Sixteen of seventeen.** T17 cannot be caught and it is right that it cannot: `contiguous`, `clone`,
`alias` and `lift_fresh` all have the identity as their derivative, so substituting one for another
is not a different computation. It is in the table because the *forward* ops differ, and only the
derivative makes them coincide — the same shape as `docs/training/LOSS.md` §8.2's D2.

**Four of the sixteen could not fail when they were first run** — T5, T6, T8, T9 — and none of them
was a defect in the rule. Each was a case that could not separate the fault from the fix: an
`nll_loss` with nothing ignored, an attention case where the input was the query alone so the
grouped-query fold never entered the checked gradient, and a slice whose two pads were equal. The
cases were made asymmetric and the faults then failed. This is the pattern `docs/training/LOSS.md` §5.2 and
§8.3 record twice, met twice more.

### 7.1 What this suite still cannot see

* **Nothing here compares against upstream.** The tape tests use finite differences, because
  `pytests/test_shim.py` runs against bare `_C` with no upstream torch in the process. The
  upstream comparison is §2, §3 and §4, and those are measurements in this document rather than
  tests in `pytests/` — so §4's 8.8e-05 can move without anything going red.
* **`tools/golden/compare.py` cannot see the tape at all.** It compares *ops* by dispatch key, and a
  derivative rule is not an op. That is why the rule table is pinned by a test instead.
* **No `float32` case separates a summation order** — the same statement `docs/training/LOSS.md` §5.3 makes,
  for the same reason, and it is why §4.4's attribution had to be done in `float64`.
* **Nothing checks memory.** §4.5's numbers are arithmetic on shapes, not measurements.
* **One backward, once.** No accumulation across steps, no `zero_grad` between them, no second
  optimiser, no convergence.

---

## 8. What this round did not do

| | why |
|---|---|
| `Tensor.backward()` | It differentiates *whatever produced this tensor*, which needs a node per op and a flag that propagates — `docs/training/AUTOGRAD.md` §6's `VariableType` half. The tape differentiates a *recorded region* and needs neither. The refusal stands and the test that pins it is unchanged |
| `requires_grad` propagation | Same boundary. Still inert, still nothing reads it, and `test_the_autograd_boundary_is_where_autograd_md_says_it_is` still passes unmodified |
| `torch.autograd.grad`, hooks, `create_graph` | None is on the path of a federated or test-time-adaptation step, which is what README §2 and §3 describe |
| double backward | The tape can in principle record its own backward, but the backward runs outside a capture region today (§1.1) — which is exactly what makes the rules cheap. The two would have to be reconciled |
| Adam and AdamW | `docs/training/LOSS.md` §6.4's four items are still open: `torch.is_complex` (a name), `lerp_.Scalar`, `addcmul_`, `addcdiv_`. **SGD needed none of them** and that is what a first training step wanted |
| ~~`index_put_(accumulate=True)`~~ | **done, §13.** `docs/kernels/VIEWS.md` §7 landed the flag; the rule uses it and the `bfloat16` gradient became bit-identical to upstream |
| ~~`native_layer_norm`'s derivative~~ | **done, §12.** And the sizing in this row was wrong: `gpt2` needed `aten.split.Tensor` as well, which §12.1 explains |
| anything on device | Desktop macOS only. A tape walker generates no code, so `docs/design/DESIGN.md` §5's iOS W^X constraint does not obviously apply — but that is reasoning, not a measurement |
| training more than one step | §7.1's last bullet |

---

## 9. Gates

| gate | before | after |
|---|---|---|
| `pytests/run.sh` | 296 ok, 0 FAIL | **302 ok, 0 FAIL** |
| `run.sh` DOCWATCH | 95/95 | **109/109** (14 new markers, all in this document) |
| `tools/golden/compare.py` | 7447/7447, ops=166, pending 1 | **7447/7447, ops=166, pending 1** |
| `compare.py --self-test` | 19 comparators × 11 fault modes | **unchanged** |
| `verify_schemas.py` | 4475/4475 | **4475/4475** |
| sweep26 (`.eval()`) | 26/26 | **26/26** |
| sweeptrain (`.train()`) | 26/26 | **26/26** |

`ops=166` is unchanged **on purpose**: no kernel landed (§1.1), so nothing here could have moved it,
and a change in that number would have meant the rules had stopped being compositions.

### 9.1 The forward did not move

`docs/numerics/SEQLEN.md` §1.3's prefill logits sha256 over real SmolLM2-135M, re-measured on the final
artefact. A backward that moves a forward result is a bug.

| S | f32 | | bf16 | |
|---:|---|:--:|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | ✅ | `8ef1550ea33c4f3d…` | ✅ |
| 32 | `331668f36da02f21…` | ✅ | `b81325c83a0a3d15…` | ✅ |
| 128 | `00159a9dbd308eda…` | ✅ | `7ff8e9334449b147…` | ✅ |
| 512 | `07c2797dabc4552e…` | ✅ | `9ab1e82f01378e38…` | ✅ |
| 1024 | `eda1e173727bb7f5…` | ✅ | — | |

All nine equal `docs/training/LOSS.md` §10.1 and `docs/training/TRAIN.md` §6.

### 9.2 The six new tests

```
test_the_tape_has_a_gradient_case_for_every_rule_it_claims
test_every_tape_rule_agrees_with_central_differences_in_float64
test_the_tape_refuses_an_op_it_has_no_rule_for_by_name
test_a_gradient_reaches_a_burned_in_constant_and_only_the_ones_asked_for
test_the_tape_seeds_a_one_only_for_a_scalar_and_says_so_otherwise
test_grad_is_a_real_slot_now_and_takes_only_a_tensor_or_none
```

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_every_tape_rule_agrees_with_central_differences_in_float64 present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_tape_has_a_gradient_case_for_every_rule_it_claims present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_grad_is_a_real_slot_now_and_takes_only_a_tensor_or_none present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _set_grad present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs _shim_grad present -->

---

## 10. `.grad` is a slot now, and why that reverses a decision

`docs/training/AUTOGRAD.md` §7 argued explicitly against this, and the argument was right at the time:

> making `.grad` writable while nothing writes to it would move the shim from "honestly reports no
> gradient" to "has a slot that is always empty"

The antecedent is gone. The tape writes gradients, so the slot is not always empty, and what would
be dishonest *now* is a `torch.optim` step that silently skipped all 272 parameters because the slot
it reads cannot be filled — which is precisely what `docs/training/LOSS.md` §6.3 had to report.

Two things did **not** change with it, and they are what keeps the reversal narrow:

* **Nothing fills it implicitly.** `CaptureTrace.backward()` *returns* gradients and the caller
  assigns them, the same shape `torch.optim.sgd.sgd` already had.
* **A clone does not inherit one.** A gradient belongs to the leaf it was accumulated into, and
  upstream gives a non-leaf no `.grad` at all.

---

## 11. Every command in this document

```sh
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-tape
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh
PY=/Volumes/macMini/caches/spike-venv/bin/python
SHIM="PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY"     # VENDOR.md wall 3

# §2, §3  the small cases, both sides, then the element-wise comparison
$PY   /tmp/tape/run_up.py   <case> /tmp/tape/up_<case>.json
$SHIM /tmp/tape/run_shim.py <case> /tmp/tape/shim_<case>.json
$PY   /tmp/tape/cmp.py      <case>

# §4  the whole model: gradients, the step, and the weights
$PY   /tmp/tape/smol_up2.py   sdpa 8 sdpa8      # writes upstream's grads/weights
$SHIM /tmp/tape/smol_shim2.py sdpa 8 sdpa8      # loads them and compares in-process
$PY   /tmp/tape/smol_up2.py   eager 8 eager8    # §4.4, the SDPA attribution
$SHIM /tmp/tape/smol_shim2.py eager 8 eager8
$PY   /tmp/tape/clean64.py up 0    &&  $SHIM /tmp/tape/clean64.py shim 0   # §4.4, float64

# §6  the probe that failed, on both implementations
$SHIM /tmp/tape/clean64.py shim 4 fd
$PY   /tmp/tape/fdup.py

# §7  sabotage: 17 faults, each rebuilt
$PY /tmp/tape/sab.py            # or /tmp/tape/sab.py T5 T8 for one

# §9  gates
PYTHON=$PY sh rust/torch_c/pytests/run.sh
$PY tools/golden/compare.py  ;  $PY tools/golden/compare.py --self-test
$PY rust/torch_c/pytests/verify_schemas.py
$SHIM /tmp/k26/sweep26.py /tmp/tape/ev   ;  $SHIM /tmp/train/sweeptrain.py /tmp/tape/tr
$SHIM /tmp/loss/seqlen.py f32            ;  $SHIM /tmp/loss/seqlen.py bf16
```

The scratch harnesses live under `/tmp/tape/` and are reproduced nowhere else; every number they
produce is quoted above with the command that made it.

---

## 12. Two more rules: `native_layer_norm`, and the one nobody had counted

58 rules now, not 56. `docs/models/ADAPT.md` §13 is what they open — `gpt2` and `bert` taking a Tent step —
and this section is the rules themselves.

### 12.1 The bill was one arm and it was two

§8's row and `docs/models/ADAPT.md` §8.1 both say closing `nn.LayerNorm` is *"one arm in `tape.rs` and one
gradient case"*. **Neither had asked a `gpt2`.** Both were reading a four-line `nn.LayerNorm` toy in
`pytests/`, and `trace.differentiable()` on the real checkpoints says:

| | nodes | on a gradient path | missing rules |
|---|---:|---:|---|
| `gpt2` | 492 | 460 | `native_layer_norm` ×25, **`aten.split.Tensor` ×12** |
| `bert` | 494 | 412 | `native_layer_norm` ×25 |

`split` is GPT-2's fused qkv projection — `c_attn(x).split(n, dim=2)` — and it is invisible from a
toy normalisation module by construction. The cost of finding this was one call to a surface §1.2
already provided; the cost of not finding it would have been landing a rule, declaring the
architecture open, and having `gpt2` fail on the next op.

### 12.2 `native_layer_norm`: three gradients, one of them difficult

With `N` the width the statistics were taken over, `y = (x − mean)·rstd` and `gh = g·weight`:

```text
  grad_bias   = sum over the outer axes of  g
  grad_weight = sum over the outer axes of  g * y
  grad_input  = rstd * ( gh - mean(gh) - y * mean(gh * y) )
```

The two reductions are `reduce_to` under another name and would come out right by accident. **The
last two terms of `grad_input` are the ones a plausible implementation omits**: `mean` and `rstd`
are themselves functions of every element of the row, so a rule that stops at `rstd * gh` has the
right shape, the right dtype and the right order of magnitude and is wrong at every element. That
is the failure mode this op is notorious for, and §14's L1 is it.

**`mean` and `rstd` are read off the forward's second and third results**, not recomputed. This is
`docs/training/LOSS.md` §3.1's reading of `nll_loss_forward`'s `total_weight` met a second time: the op
returns them *because* a backward wants them, and `aten.rs` measured that they follow the
**parameter** dtype rather than the input's — so under mixed precision recomputing them would
silently substitute the input's precision. §14's L5 is the fault that tests this, and it is one of
the ones that could not fail.

### 12.3 `split.Tensor`: the derivative is a `cat`, and the zero is the part that matters

The one op here whose forward answers with a **list**, so `gouts` has one slot per chunk rather than
per tuple position — which the walk already supported, `node.outputs` being a `Vec<Slot>` sized by
`sequence_items`. The rule concatenates the chunk gradients along the split axis.

**A chunk that no gradient reached still occupies its width in the input**, so it has to appear in
the `cat` at full size as a zero. GPT-2 uses all three of its chunks, so *the model that needs this
rule cannot exercise that zero* — the gradient case in `pytests/` is what does, and §14's S2 is the
fault.

### 12.4 The gradient cases, and the hole both of them would have had

Both cases are checked against central differences in `float64`, the oracle §5 describes, and both
had to be built against the trap §5's third bullet and §7's T8 record:

* **`native_layer_norm`'s `weight` and `bias` are built from the input.** A case with constant
  affine parameters exercises `grad_input` alone — `grad_weight` and `grad_bias` could be anything
  at all and the case would pass. They are two *different* functions of `x` (a mean, and a mean of
  a sine) so that swapping them is visible too.
* **`split`'s chunks are scaled differently and reassembled out of order.** `cat(split(x))` is the
  identity and so is its derivative, so a rule that ignored the chunk widths entirely would pass the
  obvious case. The split is `7 = 3 + 3 + 1`, the short last chunk deliberately, because equal
  chunks let a rule that recovered every width from the first one pass.

| case | worst \|tape − fd\| / scale | bound |
|---|---|---|
| `aten.native_layer_norm.default` | **8.211e-10** | 1e-5 |
| `aten.split.Tensor` | **3.451e-11** | 1e-5 |
| `aten.embedding.default` (rewritten in §13) | 5.651e-10 | 1e-5 |
| (`_scaled_dot_product_flash_attention_for_cpu`, for comparison) | 1.741e-09 | 1e-5 |

The `split` case splits along **dim 1**, not dim 0, and that is §15's S3: a case
that splits along 0 cannot tell the split axis from the `cat` axis, because the rule's `cat` is then
right for the wrong reason.

### 12.5 SmolLM2 did not move, and that is the point of saying so

Neither op is on SmolLM2's path, so §4.2's numbers must be **unchanged to every digit**, and they
are: median relative L2 `8.780e-05`, worst `3.031e-04` at `model.layers.24.input_layernorm.weight`,
sign agreement `134513262/134515008 = 0.999987`. A rule that changed a model that does not use it
would mean the walk had started doing something other than what the trace says.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs layer_norm_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs aten.split.Tensor present -->
<!-- DOCWATCH: op-implemented aten.native_layer_norm.default -->
<!-- DOCWATCH: op-implemented aten.split.Tensor -->

---

## 13. The embedding rule switched, and the memory was the smaller reason

§4.5 named `index_put_(accumulate=True)` as the right fix and left it, for a reason about a
document rather than about the code. `docs/kernels/VIEWS.md` §7 landed the flag and measured both
compositions; this is the switch.

### 13.1 What the rule is now

```text
   before   zeros[vocab, tokens] -> scatter.src -> mm(one_hot, grad)
   after    zeros[vocab, width]  -> index_put_(indices, grad, accumulate=True)
```

`padding_idx` is carried, and the switch moves *where* it is applied rather than whether: the
one-hot zeroed a **column of the one-hot**, and with no one-hot to zero, the same two ops —
one `ne.Scalar` and one `where` — zero the **contributions** instead. That is the same statement one
step later, and it is checked: with `padding_idx = 13`, row 13 of the gradient is exactly `0.0` at
both dtypes, before and after.

### 13.2 The `bfloat16` gradient became bit-identical to upstream

SmolLM2's real `[49152, 576]` embedding table, `S = 1024` tokens drawn from 64 distinct ids so that
**16 contributions accumulate into every row**, and gradient magnitudes spanning `2^-11` so that
summing them in the receiver's dtype and summing them in `float32` are different numbers.
Equal-magnitude contributions round the same way either side and separate nothing — the same reason
`docs/kernels/VIEWS.md` §7.4's separating case is `1.0 + 0.005 + 0.005`.

Upstream's `embedding_dense_backward` is the oracle; 28,311,552 elements compared.

| | one-hot → `mm` | `index_put_(accumulate=True)` |
|---|---|---|
| **`float32`** rel L2 | 0.000000e+00 | **0.000000e+00** |
| `float32` elements differing bit-for-bit | 0 / 28,311,552 | **0 / 28,311,552** |
| **`bfloat16`** rel L2 | 4.722342e-03 | **0.000000e+00** |
| `bfloat16` worst \|d\| | 3.125e-02 | **0.0** |
| `bfloat16` elements differing bit-for-bit | **20,958** / 28,311,552 | **0 / 28,311,552** |

**At `bfloat16` the gradient went from 20,958 wrong elements to bit-identical.** With
`padding_idx = 13` the same thing: 20,651 → 0.

The mechanism is `docs/kernels/VIEWS.md` §7.4's and is worth restating because it is *not* a rounding
accident. Upstream's kernel is `*dst += *src` in the receiver's `scalar_t`, so the running sum is
rounded at every step. A matmul accumulates in `float32` and rounds **once**. The one-hot was
therefore not a more expensive way to compute the same thing — it was **computing a different
function**, one that is arguably more accurate and is not upstream's. At `float32` the two coincide
exactly, which is why §4.5's claim that only the memory was wrong survived a whole round: it was
made from the only reading in which it is true.

### 13.3 And `float32` did not move — which is the check that nothing else came with it

`docs/numerics/SCALAR.md`'s round showed the two compositions identical at `float32`, so the whole-model
`float32` comparison is a **falsifiable prediction** and not a formality: if it had moved, something
other than the composition had changed.

| §4.2, all 134,515,008 SmolLM2 gradients | before the switch | after |
|---|---|---|
| relative L2, median | 8.780e-05 | **8.780e-05** |
| relative L2, worst | 3.031e-04 | **3.031e-04** |
| element sign agreement | 0.999987 | **0.999987** |

Unchanged in every digit, including the worst-tensor name.

### 13.4 The memory, measured rather than computed from shapes

§7.1's fourth bullet says *"nothing checks memory — §4.5's numbers are arithmetic on shapes"*. They
are measurements now. One embedding backward at `S = 1024` over the real `[49152, 576]` table,
peak RSS and wall time, same process, same harness, the only difference being which composition the
rule builds:

| | peak RSS | rise over pre-backward | backward |
|---|---:|---:|---|
| one-hot → `mm` | 1287 MB | 1169 MB | 0.177 s |
| `index_put_(accumulate=True)` | **898 MB** | **780 MB** | **0.080 s** |

**389 MB and 2.2× less time.** The saving is larger than §4.5's predicted 200 MB because `scatter.src`
is out-of-place: the `[49152, 1024]` zeros *and* the `[49152, 1024]` one-hot are both live at
201 MB each, which the arithmetic-on-shapes estimate counted once.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs aten.index_put_.default present -->
<!-- DOCWATCH: op-implemented aten.index_put_.default -->

---

## 14. `torch.save`: what it needs, and why it is the wrong instrument anyway

`docs/models/ADAPT.md` §1 and §8.3 name `torch.save` as the wall under `Delta.persist` and
`Delta.publish` — *"a delta on the wire is not about distribution"*. This sizes it, and the sizing
changed the answer.

### 14.1 The wall that was named is a masking exception

`docs/models/ADAPT.md` reports the refusal at `PyTorchFileWriter.write_end_of_file`. That is not where
`torch.save` stops. It is what `_open_zipfile_writer.__exit__` raises **while the real exception is
unwinding**, so it replaces it:

```
NotImplementedError: not implemented in torch._C shim: torch._C._has_storage
  ^ the real one, from torch/_tensor.py:328 in _reduce_ex_internal

During handling of the above exception, another exception occurred:
NotImplementedError: not implemented in torch._C shim: PyTorchFileWriter.write_end_of_file
  ^ from serialization.py:834 in __exit__ -- the one that gets reported
```

A refusal that a `finally` block overwrites is a refusal nobody can size, and this one had been
quoted in two documents for a round. Driving the two halves separately is what shows the chain.

### 14.2 The chain, measured

Each name below was discovered by stubbing the previous one and re-running, which is the same
experiment `_ZipRecords`' docstring records for the reader.

| | what it needs | what it is |
|---|---|---|
| 1 | `torch._C._has_storage(t)` | a bool | one line |
| 2 | `torch._C._get_tensor_metadata(t)` | a dict, empty here | one line |
| 3 | `TensorBase.storage_offset()` | ~~always 0 — storages here are copies, never windows~~ **wrong, see below** | one line |
| 4 | `TensorBase.stride()` | ~~contiguous strides from the shape~~ **wrong, see below** | a few lines |
| 5 | `UntypedStorage._cdata` | an identity key, for storage de-duplication | one line |
| 6 | **`TensorBase.untyped_storage()`** | the tensor's bytes as an `UntypedStorage` | **§14.3** |
| 7 | **`PyTorchFileWriter`** | ctor + `write_record` + `write_end_of_file` | the container |

**Item 7 is smaller than it looks and item 6 is bigger.** With everything else stood in for,
`torch.save` runs to completion and calls the writer exactly ten times:

```
 1 __init__(BytesIO, bool, int)
 2 write_end_of_file()                              <- the probing first pass
 3 __init__(BytesIO, bool, int)
 4 write_record('data.pkl',            bytes(224 B), 224)
 5 write_record('.format_version',     '1',            1)
 6 write_record('.storage_alignment',  '64',           2)
 7 write_record('byteorder',           'little',       6)
 8 write_record('data/0',              <StorageBase 12 bytes>, 12)
 9 write_record('data/1',              <StorageBase 16 bytes>, 16)
10 write_end_of_file()
```

`set_min_version`, `serialization_id`, `archive_name` and `get_all_written_records` are **never
called**. Those are exactly the records `_ZipRecords` reads, so the writer is that class mirrored —
in `bootstrap.py`, over CPython's `zipfile`, for the reason `docs/models/CKPT.md` gives for the reader
(container parsing is not a tensor operation and CPython already ships a validated zip). The one
non-obvious piece is `.storage_alignment`: `torch.save` pads the local header's *extra* field to
align payloads to 64 bytes, which `_ZipRecords.get_record_offset` reads back out and which
CPython's `zipfile` does not do on its own.

### 14.3 Item 6 is a semantic problem, not a kernel — and it is the reason to stop

`storage.rs`'s module docstring states the difference in its first paragraph:

> Upstream a storage is the *owner* of a tensor's memory, and a tensor is a view onto it — `set_`
> makes the tensor alias the storage, so writing to the storage afterwards changes the tensor.
> candle owns its own storage and has no way to express that aliasing, so here a storage is a byte
> buffer that `TensorBase.set_` **copies out of**.

So an `untyped_storage()` on this stack can only return a **copy**. `torch.save` would never
notice, because it only reads. Every other caller of `untyped_storage()` would — a write through it
would land nowhere, silently. That is the same class of failure the `filled` invariant in
`storage.rs` was built to make impossible, arriving from the other direction: implementing item 6
for `torch.save`'s sake would put a lie on the public surface to satisfy the one caller that cannot
detect it.

**So the honest sizing is not "seven items".** It is *five one-liners, one container, and one
decision this round is not entitled to take* — and the decision is the whole of it.

> **docs/models/SAVE.md took that decision, and this section is right about the danger and wrong about
> the remedy.** A copy that silently swallows writes is exactly the lie described — but the thing
> that has to refuse is the **write**, not the storage. All four write doors (`__setitem__`,
> `copy_`, `resize_`, `_shim_fill`) now refuse and name the snapshot, and `torch.save` works:
> upstream reads what it writes bit-for-bit, and storage sharing survives.
>
> Rows 3 and 4 of the table above were wrong for a related reason — both assume
> `untyped_storage()` returns *the tensor's* bytes. Upstream's storage is the **whole buffer** and
> the three numbers index into it, so a materialised view produces a file whose stride and offset
> lie about their own payload. `_cdata` likewise had to be the buffer's identity and not the Python
> object's, or de-duplication inverts. SAVE.md §2.

### 14.4 The right instrument, and it was taken

A delta on the wire needs "tensor → little-endian bytes" and a container. It does **not** need a
pickle, a zip, or a storage object. safetensors is a container that is a JSON header and a
concatenated blob, and this stack already *reads* it (`docs/models/CKPT.md` §1, bit-identical with
`torch.load`). So `Delta.persist` writes safetensors, through `tolist()` — the one way out of this
shim that is public, and the one `docs/numerics/SEQLEN.md`'s logits sha256 has always used.

**A Tent delta on SmolLM2-135M, ten steps, written and read back:**

| | |
|---|---|
| tensors | 61 |
| elements | 35,136 |
| `delta.value` in memory | 140,544 B |
| on disk | 146,912 B (6,368 B of JSON header) |
| write | 0.00 s | 
| read, via `safetensors.torch.load_file` | 0.001 s |
| **elements differing after the round trip** | **0 / 35,136** |
| worst \|d\| | **0.0** |

`persist` is not free and the cost is named rather than hidden: `tolist()` is a conversion, one
Python float per element. At 35,136 elements it does not register; at a 134M-parameter checkpoint it
would. **This is a road for a delta, which is the object that has to travel** — 137 KiB against the
model's 513 MiB, `docs/models/ADAPT.md` §5.2's 3828× — and explicitly not a road for a checkpoint.

### 14.5 The round trip is exact and re-applying it is not, for a reason already measured

Reverting the model and adding the *loaded* delta back on gives an entropy of `2.98279691` where the
adapted model had `2.98279548` — a gap of `1.43e-06`. That is not the file:

```
applying the delta read off disk    2.98279691
applying the in-memory delta        2.98279691     <- the same weights, exactly
```

**The file and memory apply to bit-identical weights.** The gap is `base + (w − base) ≠ w`, which
`docs/models/ADAPT.md` §5.1 measured a round ago as 3 elements of 35,136 landing 5.821e-11 from where they
started. Worth separating explicitly, because "the delta survived the wire" and "re-applying a delta
is exact" are different claims and only the first one is true.

### 14.6 What is still refused, and it is the right one

> **Correction (2026-09-02, `docs/distributed/FEDERATED.md`).** It does not any more, and it stopped in the
> way this section hoped: someone ran the check the refusal named. `world_size = 2` landed
> (`docs/distributed/TRANSPORT.md`) and `Delta.publish` was implemented on it — it sends the delta to the
> other rank and returns the group's weighted average, checked against the same average computed
> centrally. All three of `docs/design/DESIGN.md` §3's lifetime questions now answer by doing the thing.
> The paragraph below is left as the record of what was blocked and by what.

`Delta.publish` still refuses. It needs a second rank, `ProcessGroupLocal` refuses `world_size != 1`,
and there is no backend here that does not — `docs/models/ADAPT.md` §1's table, unchanged. That refusal
names a check that can be run, and it is now the only one of `docs/design/DESIGN.md` §3's three lifetime
questions that answers with a refusal rather than by doing the thing.

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/delta/__init__.py persist present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_a_delta_is_written_and_read_back_bit_for_bit present -->

---

## 15. Sabotage: 12 faults on the three rules this round touched

Each applied to `rust/torch_c/src/tape.rs`, **rebuilt**, and run through the eight tape tests plus
the four adaptation-road tests that go through the vendored tree.

| # | fault | caught |
|---|---|---|
| L1 | layer norm: `grad_input = rstd * gh` — **both** correction terms dropped | ✅ fd, worst 2.78 |
| L2 | layer norm: only the `y * mean(gh·y)` term dropped, mean-centering kept | ✅ fd, worst 1.52 |
| L3 | layer norm: `grad_weight` computed as `sum(g)`, i.e. equal to `grad_bias` | ✅ fd, worst 0.070 |
| L4 | layer norm: the division by `N` dropped in both row-means | ✅ fd **and** the LayerNorm Tent curve stops falling monotonically |
| L5 | layer norm: `mean`/`rstd` **recomputed** instead of read off the op's own results | **❌ — §15.1** |
| L6 | layer norm: `grad_input` never narrowed back to the input's dtype | ✅ *(after §15.2)* |
| S1 | split: the chunk gradients concatenated in reverse | ✅ fd, worst 0.996 |
| S2 | split: a chunk no gradient reached contributes nothing instead of zeros | ✅ *(after §15.2)* |
| S3 | split: always concatenated along axis 0 rather than the split axis | ✅ *(after §15.2)* |
| E1 | embedding: `accumulate=False` — a write instead of a sum | ✅ fd, worst 0.500 |
| E2 | embedding: `padding_idx` no longer zeroed | ✅ *(after §15.2)* |
| E3 | embedding: the padding mask left `[count]` instead of `[count, 1]`, so it lines up against the width rather than the tokens | ✅ *(after §15.2)* |

**Eleven of twelve**, and the four marked *(after §15.2)* could not fail on the first run.

### 15.1 The one that cannot fail — and it is a hole, not a coincidence

§7's T17 and `docs/models/ADAPT.md` §9's S12 were faults that **could not be caught and were right not to
be**: substituting one spelling of the identity for another is not a different computation. **L5 is
not that.** It is a real difference that this file has no oracle for, and the distinction is worth
making because "correctly uncatchable" was the comfortable answer available.

Measured directly, running the same case under both builds:

| | elements differing | worst \|d\| |
|---|---:|---|
| **`float32` input, `float32` params** — `grad_input` | **0 / 24** | 0.0 |
| `float32`/`float32` — `grad_weight`, `grad_bias` | **0 / 4**, 0 / 4 | 0.0 |
| **`bfloat16` input, `float32` params** — `grad_input` | **22 / 24** | 6.25 |
| `bfloat16`/`float32` — `grad_weight` | **4 / 4** | 0.399 |

* **At matched dtypes, recomputing is exactly a no-op** — 0 of 32 elements differ, bit for bit. No
  `float32` or `float64` case can *ever* catch L5, because at those dtypes there is nothing to
  catch: the `float64` finite-difference oracle §5 is built on is structurally blind to it.
* **At mixed precision it moves 22 of 24 `grad_input` elements by up to 6.25**, because the forward
  computes its statistics at the *parameter* dtype and a recomputation computes them at the input's.
  That is exactly why the rule reads them, and `aten.rs` measured the dtype rule it depends on.

What would close it is an oracle for mixed-precision *values*, and `pytests/test_shim.py` has none —
§7.1's first bullet already says these tests run against bare `_C` with no upstream in the process.
The two-interpreter shape `docs/models/ADAPT.md` §11.1 uses would provide one. **It is named as a hole, not
as a property.** This is `docs/numerics/SCALAR.md`'s statement arriving a third time: this suite separates
*which* function a reduced-precision kernel computes and not *at what precision it computes its
interior*.

### 15.2 Four faults that could not fail, and none was a defect in a rule

Every one was a *case* that could not separate the fault from the fix — the pattern §7 met four
times and `docs/training/LOSS.md` §5.2 twice.

| | why it could not fail | what closed it |
|---|---|---|
| **S3** | the case split along **dim 0**, so the split axis and the `cat` axis were the same number and a rule that hard-coded 0 was right by accident | the case splits along dim 1 now, on a `[2, 7]` |
| **S2** | the case uses all three chunks — and so does GPT-2, so *the model that needs the rule cannot exercise its zero* | `test_the_split_rule_supplies_a_zero_for_a_chunk_no_gradient_reached`, which uses only the middle chunk |
| **E2, E3** | the embedding case carries no `padding_idx`, and **it cannot** — see below | `test_the_embedding_rule_zeroes_the_padding_row_and_only_that_row` |
| **L6** | `cast_like` is the identity whenever the dtypes already agree, and everything in a `float64` case agrees | `test_a_layer_norm_gradient_keeps_the_dtype_it_was_asked_for` |

**`padding_idx` cannot be checked by finite differences at all, and that is a fact about the
semantic rather than about the case.** The *forward* reads the padding row like any other, so
perturbing it does change the output: the oracle would report a gradient there, and the correct rule
deliberately returns zero. **A finite-difference case with `padding_idx` would fail on the correct
implementation.** So the new test asserts the structural claim instead — the padding row is exactly
zero, the other used rows are identical to what they are without `padding_idx`, and the repeated row
still carries both of its contributions. §13.1 checks the same property against upstream on the real
49,152-row table.

**L6 is the fault that improved the rule.** Writing its test found that mixed precision did not
merely lose precision — it **refused**, at `x - mean` with a `bfloat16` x against a `float32` mean,
a promotion this shim declines by name. The rule now computes its interior in the statistics' dtype
and narrows each result to the dtype of the thing it is a gradient for, which is what upstream does.
`cast_like` went from dead code to the thing L6 removes.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_split_rule_supplies_a_zero_for_a_chunk_no_gradient_reached present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_embedding_rule_zeroes_the_padding_row_and_only_that_row present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_a_layer_norm_gradient_keeps_the_dtype_it_was_asked_for present -->

---

## 16. Gates

| gate | before this round | after |
|---|---|---|
| `pytests/run.sh` | 312 ok, 0 FAIL | **317 ok, 0 FAIL** |
| `run.sh` DOCWATCH | 178/178 | **190/190** |
| `tools/golden/compare.py` | 7685/7685, ops=168, pending 1 | **7685/7685, ops=168, pending 1** |
| `compare.py --self-test` | 20 comparators × 11 fault modes | **unchanged** |
| `verify_schemas.py` | 4479/4479 | **4479/4479** |
| sweep26 (`.eval()`) | 26/26 | **26/26** |
| sweeptrain (`.train()`) | 26/26 | **26/26** |
| tape rules | 56 | **58** |

`ops=168` is unchanged **on purpose**, and this is the round where that had to be checked rather
than assumed: `index_put_(accumulate=True)` is an op the embedding rule now calls, and had it not
already existed the number would have moved. It existed — `docs/kernels/VIEWS.md` §7 landed it — so the two
new rules and the rewritten one are still compositions over kernels that were already there.

### 16.1 The forward did not move

`docs/numerics/SEQLEN.md` §1.3's prefill logits sha256 over real SmolLM2-135M, re-measured on the final
artefact. Three rule changes that moved a forward result would be a bug.

| S | f32 | | bf16 | |
|---:|---|:--:|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | ✅ | `8ef1550ea33c4f3d…` | ✅ |
| 32 | `331668f36da02f21…` | ✅ | `b81325c83a0a3d15…` | ✅ |
| 128 | `00159a9dbd308eda…` | ✅ | `7ff8e9334449b147…` | ✅ |
| 512 | `07c2797dabc4552e…` | ✅ | `9ab1e82f01378e38…` | ✅ |
| 1024 | `eda1e173727bb7f5…` | ✅ | — | |

All nine equal §9.1, `docs/models/ADAPT.md` §7, `docs/training/LOSS.md` §10.1 and `docs/training/TRAIN.md` §6.

### 16.2 The seven new tests, and the two they replace

```
test_the_split_rule_supplies_a_zero_for_a_chunk_no_gradient_reached
test_the_embedding_rule_zeroes_the_padding_row_and_only_that_row
test_a_layer_norm_gradient_keeps_the_dtype_it_was_asked_for
test_tent_adapts_an_nn_layer_norm_model_and_the_wrong_sign_does_not
test_an_op_with_no_derivative_rule_is_refused_by_naming_it
test_a_delta_is_written_and_read_back_bit_for_bit
test_a_delta_names_a_check_for_the_destination_it_cannot_reach
```

`test_tent_refuses_a_layer_norm_model_by_naming_the_missing_rule` and the `persist` half of
`test_a_delta_names_a_check_for_the_destinations_it_cannot_reach` were both assertions that
something **refuses**, and both refusals fell this round. Neither was simply deleted. §8 predicted
the first one's inversion and said the fix was to delete it — and that would have been wrong: what
is worth pinning is not that Tent refuses but that a LayerNorm model's entropy actually *falls*,
because a layer-norm backward with the right shape for all three gradients and the wrong values
passes any test that only checks a step ran. `test_an_op_with_no_derivative_rule_is_refused_by_naming_it`
keeps the refusal machinery exercised and asserts that the op its message names is genuinely absent
from `_tape_rules()`, so the day `abs` gains a rule it fails on its own premise instead of quietly
checking nothing.

---

## 17. Every command in §12–§16

```sh
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-rules
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh
PY=/Volumes/macMini/caches/spike-venv/bin/python
SHIM="PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY"

# §12.1  the sizing check, before any rule was written
$SHIM /tmp/rules/wall.py gpt2   ;  $SHIM /tmp/rules/wall.py bert

# §13.2  the embedding gradient, both compositions, both dtypes
$PY   /tmp/rules/emb_up.py 1024 64     ;  $SHIM /tmp/rules/emb_shim.py 1024 64
$PY   /tmp/rules/emb_up.py 1024 64 13  ;  $SHIM /tmp/rules/emb_shim.py 1024 64 13
# §13.3  the float32 guard: all 134,515,008 SmolLM2 gradients
$SHIM /tmp/tape/smol_shim2.py sdpa 8 sdpa8
# §13.4  peak RSS and time, one embedding backward at S=1024
$SHIM /tmp/rules/emb_mem.py

# §14  what torch.save needs, in three probes
$SHIM /tmp/rules/save_probe.py ; $SHIM /tmp/rules/save_probe2.py ; $SHIM /tmp/rules/save_probe3.py
$SHIM /tmp/rules/st_probe.py            # the flat container, round-tripped
$SHIM /tmp/rules/persist_real.py        # a Tent delta on SmolLM2, written and read back

# §15  sabotage: 12 faults, each rebuilt
$PY /tmp/rules/sab.py                   # or /tmp/rules/sab.py L5 S2 for one

# §16  gates
PYTHON=$PY sh rust/torch_c/pytests/run.sh
$PY tools/golden/compare.py  ;  $PY tools/golden/compare.py --self-test
$PY rust/torch_c/pytests/verify_schemas.py
$SHIM /tmp/k26/sweep26.py /tmp/rules/ev  ;  $SHIM /tmp/train/sweeptrain.py /tmp/rules/tr
$SHIM /tmp/loss/seqlen.py f32            ;  $SHIM /tmp/loss/seqlen.py bf16
```

The scratch harnesses for §12–§16 live under `/tmp/rules/` and are reproduced nowhere else; every
number they produce is quoted above with the command that made it.

---

## 18. Two rules for `.train()`, and the oracle §15.1 said was missing

`docs/models/ADAPT.md` §14 is what these open — `gpt2` and `bert` taking a Tent step **in `.train()`**, which
is the ordinary case for test-time adaptation and was refused until now. 60 rules, not 58.

### 18.1 The wall, asked rather than assumed

The same check §12.1 used, on the same checkpoints, with `.train()` instead of `.eval()`:

| | nodes | on a gradient path | missing rules |
|---|---:|---:|---|
| `gpt2` `.eval()` | 492 | 460 | — (§12 closed them) |
| `gpt2` `.train()` | **661** | **568** | `aten.native_dropout.default` ×36, `aten._safe_softmax.default` ×12 |

**Training mode is 169 more nodes and two more rules, and the second one is not about dropout at
all.** `_safe_softmax` ×12 — one per layer — arrives because a non-zero `dropout_p` takes SDPA off
the fused kernel and onto the math backend, which is a different op sequence (`docs/training/TRAIN.md` §4).
So the two rules are one requirement: there is no way to have attention dropout without the softmax
it sits behind, and a round that landed only `native_dropout` would have moved the wall by one op.

Both ops already had kernels (`docs/training/TRAIN.md` §3 and §4 landed them). What was missing was only the
derivative, which is why this round adds no ops: `ops=168` is unchanged.

### 18.2 `native_dropout`: the mask is the point, and the scalar is not narrowed

```text
  grad_input = (g * mask) * scale        scale = !train ? 1 : (p == 1 ? 0 : 1/(1-p))
```

**The mask is read off the forward's second result, and here that is not an optimisation.** `mean`
and `rstd` (§12.2) and `total_weight` (`docs/training/LOSS.md` §3.1) could all be recomputed at a price; a
dropout mask cannot be recomputed at any price, because no function of the input says which elements
were dropped. A rule that did not read it would have to draw again, and would then be the gradient of
a different function than the forward computed. §19's **D2** is that fault.

Three details are upstream's and two are invisible in `float32`:

* **The association is `(g * mask) * scale`, not `g * (mask * scale)`.** Upstream has both, for two
  callers: `native_dropout_backward` and `infinitely_differentiable_native_dropout_backward`, the
  second taken only when `GradMode` is on (`create_graph=True`). Measured on 2.13.0 they are
  *different numbers* at `bfloat16` and `float16` and identical at `float32`/`float64` — at
  `p = 0.7`, `g = 7.0`: **23.375** against **23.328125**. A plain `loss.backward()` takes the first,
  so this rule does. §19's **D3**.
* **The scalar is not narrowed to the tensor's dtype**, because `mul.Scalar` follows upstream's
  un-narrowed `original_scalar_value<opmath_t>` since `docs/numerics/SCALAR.md`. This is the **opposite** of
  what the forward does — `native_dropout`'s own kernel narrows before it multiplies (`aten.rs`,
  1280 combinations) — so the two halves of one op answer the question differently and neither is a
  guess.
* **Both guards on the scale are real.** Without the `p == 1` guard the reciprocal is `inf` and every
  dropped element's gradient is `0 * inf = nan`; without the `train` guard an eval-mode dropout's
  gradient is scaled by `1/(1-p)` against an unscaled forward. Neither is reachable from a case at an
  ordinary `p`, and both were green until they had their own arms — §19's **D4** and **D5**.

### 18.3 `_safe_softmax`: the safety is in the forward, and the backward inherits it

The expression is `_softmax`'s, `(g − rowsum(g·p))·p`. It is a **separate arm** anyway, because the
third argument is `dtype` and not `half_to_float`, and the vendored tree binds arguments by name
before dispatching (`docs/graph/CAPTURE.md` §2) — a rule reading `half_to_float` would find nothing there.

**What upstream does on a fully-masked row is the finding, and it was checked rather than assumed.**
On a row that is entirely `-inf`:

```text
  _softmax        forward  nan      backward  nan
  _safe_softmax   forward  0        backward  0        (SafeSoftmaxBackward0)
```

and upstream's `0` is **not a special case in its backward**. It is `(g − 0)·0`: the forward already
wrote zeros there, so the backward's arithmetic gives zero without knowing why. That is only true of
a rule that *reads* `outs[0]`. This tape is explicitly allowed to recompute — it runs outside a
capture region and `sdpa_backward` two functions away does exactly that — and a `_safe_softmax` rule
that recomputed would exponentiate `-inf − (-inf)` and answer `nan` on precisely the rows the op
exists for, while agreeing everywhere the `float64` oracle can look. §19's **V2**.

**The case that catches it took two tries, and the first one is the more useful half.** Written with
`masked_fill(x, mask, -inf)` the fault passed: `masked_fill`'s own rule is `masked_fill(g, mask, 0)`,
so it *replaces the `nan` with zero* on exactly the entries the test reads. Written with
`add.Tensor(x, bias)` and a `-inf` bias it fails — and the additive spelling is also the one the
model uses, since `bootstrap.py`'s math backend is `attn = add.Tensor(attn, attn_mask)`. A test that
checks a `nan` has to be built so the `nan` can get out.

### 18.4 The gradient cases, and the seed that all fifty-nine now share

| case | worst \|tape − fd\| / scale | bound |
|---|---|---|
| `aten.native_dropout.default` | **1.123e-10** | 1e-5 |
| `aten._safe_softmax.default` | **3.822e-09** | 1e-5 |

* **`native_dropout` is the first case here whose forward is stochastic**, and the tape's backward
  **replays** the forward rather than keeping the capture's intermediates. So three different masks
  were in play — the capture's, the replay's, and a fresh one per finite difference — and the
  comparison would have failed for a correct rule. `_tape_gradient` and `_tape_central_differences`
  now seed before **every** forward, for all fifty-nine cases, so the next random rule inherits it.
  §18.5 is what that seeding is hiding, said out loud.
* `p = 0.5` on twelve elements leaves **7 survivors and 5 casualties**, checked. A `p` near zero
  gives an all-ones mask, and against an all-ones mask a rule that ignored the mask is exactly right.
* **`_safe_softmax`'s case is rank 3 on `dim = 1`, not the trailing axis.** `_softmax`'s existing case
  is on `-1`, so neither of them could tell the rule's `dim` from a hardcoded `-1`; this one can.
  §19's **V1**.

### 18.5 The tape replays, so a dropout gradient is a *second* draw — measured

`trace.backward()` calls `PyCaptureTrace::run`. `native_dropout` consumes the generator and is
deliberately **not** on `capture.rs`'s `RANDOM` list — it was added so that a `.train()` forward could
be captured at all (`docs/graph/CAPTURE.md` §9). The consequence: a `torchnative.adapt` step computes the
gradient of the objective it reports **at a different sample of the noise**.

It is not a small term. One Tent gradient on `gpt2` in `.train()`, 50 tensors / 38,400 elements,
against upstream's `loss.backward()`:

| | rel L2 median | worst | sign agreement |
|---|---|---|---|
| replay put back on the capture's draw | **9.593e-03** | 1.151e-01 | 0.997214 |
| replay left to draw again | 8.731e-01 | 4.197e+00 | 0.685312 |
| *(the same model in `.eval()`, as the control)* | 8.331e-03 | 1.731e-02 | 0.997786 |

**Two orders of magnitude, and the derivative rule is not what it is about**: with the draw held the
`.train()` gradient agrees with upstream as well as the `.eval()` one does, on a path with 169 more
nodes and twenty ops of recomputed attention in it. `test_the_tape_replays_a_dropout_forward_and_
therefore_redraws_its_mask` pins the property so it cannot go quiet.

### 18.6 The oracle §15.1 asked for, and L5 is caught

§15.1 named a hole rather than a property: **L5** — the layer-norm rule recomputing its statistics
instead of reading the two the forward returns — is *exactly* a no-op at matched dtypes and moves 22
of 24 `grad_input` elements at mixed precision, so no `float32` or `float64` case could ever catch
it. It said what would close it: *"an oracle for mixed-precision values, and `pytests/test_shim.py`
has none"*.

It has one, and **not the two-interpreter shape §15.1 guessed at**: upstream `torch` is importable
**in this process** — `_E2EBackend` has been comparing against it that way since `docs/models/E2E.md`, and
§15.1 read §7.1's "no upstream in the process" as a fact about the file when it is a fact about the
*tape section* of the file. The comparison costs no subprocess.

`bfloat16` input, `float32` weight and bias, a **linear** objective so the gradient arriving at the op
is `gout` exactly on both sides:

| | elements | differing from upstream, bit for bit | worst \|d\| |
|---|---:|---:|---|
| `grad_input` | 24 | **0** | 0.0 |
| `grad_input`, **with L5 applied** | 24 | **24** | 0.0029 |

**L5 now fails, at 24 of 24.** The assertion is bit-for-bit rather than a tolerance and the reason
bounds what it can see: the rule's last step narrows to the input's dtype, so any `float32`
disagreement below `bfloat16`'s eight significand bits is rounded away first. It is a strong
assertion for a weak reason, and a fault moving `grad_input` by less than half a `bfloat16` ulp is
still invisible. L5 moves it by 6.25 in §15.1's units.

### 18.7 The oracle immediately found a second thing, and it is upstream's

`grad_weight` and `grad_bias` are **not** upstream's at mixed precision, and the shim's are the exact
ones. One column of six `bfloat16` gradients, `sum(dY)`:

```text
  exact sum, float64              1.141357421875
  this rule (float32 interior)    1.141357421875     equal at every digit
  upstream, bfloat16 in / f32 params   1.13671875
  upstream, float32 in / f32 params    1.141357421875     equal again
  upstream, all bfloat16                1.140625          one rounding
```

`1.13671875` is **not on the `bfloat16` grid**, so it is not "the exact sum rounded once", and it is
not the all-`bfloat16` kernel's answer either. It is a third accumulation taken only on the mixed
path. It is a function of `dY` alone (checked against three different inputs), so it is arithmetic
and not a leak from the input. A `bfloat16` `sum`, a sequential `bfloat16` accumulation and a
pairwise one were all tried and none reproduces it.

Relative to upstream the gap is **3.099e-03** on `grad_weight` and **1.797e-03** on `grad_bias`,
against `bfloat16`'s eps of 7.8e-03. It is recorded and bounded rather than chased into upstream's
kernel: `test_mixed_precision_layer_norm_dgamma_is_not_upstreams_and_the_gap_is_upstreams` asserts
the gap stays under one eps **and stays non-zero**, so the day somebody makes the two agree the test
says which measurement has to be retaken.

**This is §13.2's story with the sides swapped**, and it is not being treated as settled by that:
there the shim's `float32`-accumulating matmul was the more accurate composition and was *replaced*
by upstream's rounding, on the reasoning that "arguably more accurate and not upstream's" is still
not upstream. The same reasoning applies here and the same change is not made, for one reason and it
is stated so it can be argued with: **§13.2's replacement was a composition swap that made the
gradient bit-identical, and no spelling reproduces this one** — so the choice here is between a
measured 3e-03 gap and an unmeasured guess at upstream's accumulation order. That is a real item,
not a closed one, and it belongs to whoever owns `native_layer_norm` next.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs native_dropout_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs aten._safe_softmax.default present -->
<!-- DOCWATCH: op-implemented aten.native_dropout.default -->
<!-- DOCWATCH: op-implemented aten._safe_softmax.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_a_mixed_precision_layer_norm_grad_input_is_upstreams_bit_for_bit present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_dropout_gradient_is_upstreams_draw_for_draw_and_reads_the_mask present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_safe_softmax_gradient_of_a_fully_masked_row_is_zero_not_nan present -->

---

## 19. Sabotage: 13 faults, and this time L5 has something to fail on

Each applied to the source, **rebuilt**, and run through the ten `_C`-level checks plus the four
adaptation-road tests that go through the vendored tree.

| # | fault | caught by |
|---|---|---|
| D1 | dropout: the scale is `1 - p` rather than `1 / (1 - p)` | ✅ fd (0.750) **and** upstream, 14 of 24 elements |
| D2 | dropout: **the mask ignored** — `g * scale` | ✅ fd (0.048), upstream 10 of 24, and the replay test |
| D3 | dropout: `g * (mask * scale)` — upstream's *other* spelling | ✅ **`bfloat16` only**, 1 of 24 elements |
| D4 | dropout: the `p == 1` guard dropped | ✅ the whole gradient is `nan` |
| D5 | dropout: `train=False` ignored, so the scale is still `1/(1-p)` | ✅ every element off by 2× |
| V1 | safe softmax: the axis hardcoded to `-1` rather than read off `dim` | ✅ fd, worst 0.459 |
| V2 | safe softmax: the softmax **recomputed** instead of read off the forward | ✅ `nan` on the fully-masked row |
| V3 | safe softmax: the subtraction's operands swapped — a sign flip | ✅ fd (0.082) and the masked-row case |
| L5 | layer norm: `mean`/`rstd` recomputed — §15.1's uncatchable one | ✅ **24 of 24 at mixed precision** |
| T1 | `Tensor.type()` answers the dtype rather than the legacy name | ✅ |
| T2 | `Tensor.type(dtype)` always copies, losing `x.type(x.dtype) is x` | ✅ |
| T3 | a bad type name raises `TypeError` where upstream raises `ValueError` | ✅ |
| T4 | a type *object* read by `__name__`, which drops the module `tp_name` carries | ✅ |

**Thirteen of thirteen**, and three of them are worth the space:

* **D3 is caught by one dtype and one element.** `float32` and `float64` cannot see it at all, which
  is why the case runs at both — the `float32` arm is the control that says the disagreement is about
  precision and not about the formula. This is `docs/training/TRAIN.md` §5's S4 met from the other side.
* **D4 and D5 could not fail on the first run**, and neither was a defect in the rule: the case ran at
  an ordinary `p` with `train=True`, so neither guard was on the road at all.
  `test_the_dropout_gradients_two_guarded_scales_are_the_forwards_two` is the two arms, and it is the
  fifth time in this file that a guard added *because upstream has it* turned out to have no case.
* **V2 passed against the first version of its own test** (§18.3), which is the failure this
  repository keeps meeting: a test whose oracle is downstream of something that erases the symptom.

`test_the_tape_has_a_gradient_case_for_every_rule_it_claims` keeps the rule table and the case table
equal, so neither new rule could have landed without a case.

---

## 20. Gates

| gate | §16 | now |
|---|---|---|
| `pytests/run.sh` | 317 ok, 0 FAIL | **325 ok, 0 FAIL** |
| `run.sh` DOCWATCH | 190/190 | **210/210** |
| `tools/golden/compare.py` | 7685/7685, ops=168, pending 1 | **7685/7685, ops=168, pending 1** |
| `compare.py --self-test` | 20 comparators × 11 fault modes | **unchanged** |
| `verify_schemas.py` | 4479/4479 | **4479/4479** |
| sweep26 (`.eval()`) | 26/26 | **26/26** |
| sweeptrain (`.train()`) | 26/26 | **26/26** |
| tape rules | 58 | **60** |

`ops=168` is unchanged **and that is the claim of this round**: `native_dropout` and `_safe_softmax`
have had kernels since `docs/training/TRAIN.md`, so two whole architectures' worth of `.train()` adaptation
was one derivative apiece and no new arithmetic.

### 20.1 SmolLM2 did not move

Neither op is on SmolLM2's path — its config has no dropout — so §13.3's numbers must be unchanged to
every digit, and they are: median relative L2 `8.780e-05`, worst `3.031e-04` at
`model.layers.24.input_layernorm.weight`, sign agreement `134513262/134515008 = 0.999987`. The
`float64` finite-difference bounds for the other 58 rules are unchanged too, including under the new
per-forward seeding, which is the check that the seeding is a no-op where nothing draws.

### 20.2 The forward did not move

`docs/numerics/SEQLEN.md` §1.3's prefill logits sha256, re-measured on the final artefact. All nine equal
§16.1's.

| S | f32 | | bf16 | |
|---:|---|:--:|---|:--:|
| 6 | `b9fc5553ee1bf6a2…` | ✅ | `8ef1550ea33c4f3d…` | ✅ |
| 32 | `331668f36da02f21…` | ✅ | `b81325c83a0a3d15…` | ✅ |
| 128 | `00159a9dbd308eda…` | ✅ | `7ff8e9334449b147…` | ✅ |
| 512 | `07c2797dabc4552e…` | ✅ | `9ab1e82f01378e38…` | ✅ |
| 1024 | `eda1e173727bb7f5…` | ✅ | — | |

### 20.3 The eight new tests

```
test_the_dropout_gradient_is_upstreams_draw_for_draw_and_reads_the_mask
test_the_dropout_gradients_two_guarded_scales_are_the_forwards_two
test_the_tape_replays_a_dropout_forward_and_therefore_redraws_its_mask
test_the_safe_softmax_gradient_of_a_fully_masked_row_is_zero_not_nan
test_a_mixed_precision_layer_norm_grad_input_is_upstreams_bit_for_bit
test_mixed_precision_layer_norm_dgamma_is_not_upstreams_and_the_gap_is_upstreams
test_tent_in_training_mode_adapts_and_the_dropout_is_really_on
test_tensor_type_answers_a_name_a_dtype_and_a_legacy_class
```

Nothing was deleted. `test_a_layer_norm_gradient_keeps_the_dtype_it_was_asked_for` stays beside the
new mixed-precision pair: it asserts the *dtype* of `grad_input` and they assert its *values*, and
§15.2's L6 is caught by the first and not by the second.

<!-- DOCWATCH: count smoke_ok ge 325 -->
<!-- DOCWATCH: count golden_ops_covered ge 168 -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_tape_replays_a_dropout_forward_and_therefore_redraws_its_mask present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_dropout_gradients_two_guarded_scales_are_the_forwards_two present -->

---

## 21. Every command in §18–§20

```sh
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-trainrules
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh
PY=/Volumes/macMini/caches/spike-venv/bin/python
SHIM="PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY"

# §18.1  the .train() sizing, before any rule was written
$SHIM /tmp/trrules/wall_train.py gpt2

# §18.2/§18.3  what upstream's two derivatives are, and what it does on a masked row
$PY /tmp/trrules/nd_up.py   ;  $PY /tmp/trrules/ss_up.py  ;  $PY /tmp/trrules/ss_up2.py

# §18.5  one Tent gradient on gpt2, with and without the replay reseeded
$PY   /tmp/trrules/tr_up.py gpt2 1e-3 10
$SHIM /tmp/trrules/grad1.py gpt2            # reseeded
$SHIM /tmp/trrules/grad1.py gpt2 noreseed
TRMODE=eval $PY   /tmp/trrules/tr_up.py gpt2 1e-3 10     # the .eval() control
TRMODE=eval $SHIM /tmp/trrules/grad1.py gpt2

# §18.7  what upstream's mixed-precision dgamma/dbeta actually are
$PY /tmp/trrules/mp_gw.py ; $PY /tmp/trrules/mp_gw3.py ; $PY /tmp/trrules/mp_gw5.py

# §19  sabotage: 13 faults, each rebuilt
$PY /tmp/trrules/sab.py                  # or /tmp/trrules/sab.py L5 V2 for one

# §20  gates
PYTHON=$PY sh rust/torch_c/pytests/run.sh
$PY tools/golden/compare.py  ;  $PY tools/golden/compare.py --self-test
$PY rust/torch_c/pytests/verify_schemas.py
$SHIM /tmp/k26/sweep26.py /tmp/trrules/ev  ;  $SHIM /tmp/train/sweeptrain.py /tmp/trrules/tr
$SHIM /tmp/loss/seqlen.py f32            ;  $SHIM /tmp/loss/seqlen.py bf16
$SHIM /tmp/tape/smol_shim2.py sdpa 8 sdpa8
```
