# RNN — `lstm`, `upsample_linear1d`, and the name that was not a kernel

Worktree `work/rnn` on develop `a523ae4`, vendored tree assembled fresh. torch 2.13.0 upstream
(`/Volumes/macMini/caches/spike-venv/bin/python`). Territory: `rust/torch_c/src/aten.rs`,
`overloads.json`, `methods.json`, `tools/golden/cases.py`, and a new
`rust/torch_c/pytests/test_rnn.py`. `tensor.rs`, `dtype.rs`, `bootstrap.py`, `capture.rs`,
`tape.rs`, `device.rs`, `tools/wheel/` and `torchnative/` were not touched.

docs/architectures/ARCH200.md §2 named three operators worth six of its twenty-seven blocked architectures:

```text
   2  lstm                                    parakeet_rnnt, parakeet_tdt
   2  torch.conv1d                            lasr_ctc, lasr_encoder
   2  torch._C._nn.upsample_linear1d          sam_vision_model, sam_hq_vision_model
```

Two of the three were kernels. One was not, and it was not a name either — it was **one branch
of a name that already existed**.

---

## 1. `conv1d` — a NAME, and it has been spelled since docs/architectures/ARCH20.md

**Answer first: `conv1d` needed no kernel and no new spelling.** `aten::conv1d` is
`CompositeImplicitAutograd` upstream; `aten.convolution.default` has had a kernel and golden
cases since docs/kernels/OPS4.md; and `bootstrap.py` has bound `torch.conv1d` to it since
docs/architectures/ARCH20.md (`bootstrap.py:8883`). docs/architectures/ARCH200.md's `missing_shim_name torch.conv1d` row
was measured on a checkout where that work had not yet merged — its own §1 says so — so the
classifier attributed a different refusal to the name it saw in the traceback.

`pytests/test_rnn.py::test_conv1d_is_a_name_not_a_kernel` asserts there is no `aten.conv1d.*`
key in this build and that there does not need to be, so the finding cannot rot back into a
kernel request.

### 1.1 What `lasr_ctc` and `lasr_encoder` actually stop on

Re-run under this build, the two lasr architectures still fail — but the last frame is not the
name:

```text
File "torch_c_bootstrap.py", line 8933, in conv1d
NotImplementedError: not implemented in torch._C shim: torch.conv1d(padding='same') where
dilation*(kernel-1) is odd -- upstream pads the input asymmetrically with
aten::constant_pad_nd before convolving
```

`LasrEncoderConvolutionModule` builds `nn.Conv1d(..., padding="same", groups=channels)`. Its
source comment says *"kernel_size should be an odd number for 'SAME' padding"* — and
`LasrEncoderConfig.conv_kernel_size` defaults to **32**. So `dilation * (kernel - 1)` is 31,
odd, and the branch `bootstrap.py` refuses is the one every lasr checkpoint takes. **This is
not an artefact of `arch_sweep.py`'s config shrinking**; `conv_kernel_size` is not in `_SHRINK`
and the value is the real default.

### 1.2 Upstream's lowering, measured, so the fix is a transcription

A `TorchDispatchMode` logger on 2.13.0, `conv1d(x(2,3,9), w(3,1,4), b, 1, "same", 1, 3)`:

```text
   aten.constant_pad_nd.default   [(2, 3, 9), [0, 1]]
   aten.convolution.default       [(2, 3, 10), (3, 1, 4), (3,), [1], [1], [1], False, [0], 3]
```

The whole of the odd case is **one extra element of zero padding on the RIGHT**, then the
symmetric convolution with `padding = [total // 2]`. Both of those kernels are already in this
shim and both are already golden-compared. The left/right choice is not free: padding the left
instead gives a different answer, which `test_rnn.py::test_conv1d_same_odd_the_other_half_is_wrong`
pins against upstream rather than asserting in prose.

So `bootstrap.py`'s composite needs, in place of its `raise`:

```python
            total = dilation[0] * (weight.shape[-1] - 1)
            if total % 2:
                input = dispatch("aten.constant_pad_nd.default", input, [0, 1], 0.0)
            padding = [total // 2]
```

**`bootstrap.py` is not this round's territory, so that change was not made here.** It was
*measured* instead: a probe that installs exactly those four lines in its own process — using
only kernels this build already has — and then runs the two architectures through
`arch_sweep.run_one` reports

```text
lasr_ctc      ok
lasr_encoder  ok
```

That is the whole remaining wall for both, and it costs no kernel.
`test_rnn.py::test_conv1d_same_with_odd_total_padding_is_still_refused` keeps the refusal
honest and says, in its own message, to convert it to an agreement check rather than delete it
when the composite lands.

---

## 2. `aten::lstm` — the real work, and which overload

`aten::lstm` has two overloads. **Which one the parakeet decoders take was measured, not
inferred**: their traceback ends at `torch/nn/modules/rnn.py:1164`,

```python
    result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
                      self.dropout, self.training, self.bidirectional, self.batch_first)
```

nine positional arguments, i.e. `aten::lstm.input`. `ParakeetRNNTDecoder` is
`nn.LSTM(input_size=H, hidden_size=H, num_layers=config.num_decoder_layers, batch_first=True)`
— unidirectional, biased, no projection, `dropout=0`, `eval()`.

| | |
|---|---|
| **Implemented** | `num_layers > 1`, `bidirectional`, `batch_first`, `has_biases` both ways, a supplied `(h_0, c_0)`, `dropout > 0` with `train=False` |
| **Refused by name** | `aten::lstm.data` (packed sequences), `proj_size != 0`, `dropout > 0` with `train=True`, integral dtypes, a 2-D input |

`.data` is refused rather than answered with `.input`'s kernel because it is a *different
iteration order over the same weights*: a ragged batch described by `batch_sizes`, with no
`batch_first`. Answering it here would return a right-shaped wrong tensor — the failure shape
docs/kernels/TAIL3.md (`view_as` is not `reshape_as`) and docs/architectures/VOICE3.md (`col2im` is not `im2col`'s
inverse) both record.

`proj_size` is detected by **counting** `params` rather than guessed: a projection adds a fifth
weight `w_hr` per layer-direction, so `len(params) == layers * directions * (per_set + 1)` is
its signature and the refusal names it instead of saying "wrong number of parameters".

### 2.1 The four places a plausible implementation is wrong and still returns the right shape

1. **Gate order.** `i, f, g, o` along the `4H` axis of both `w_ih` and `w_hh`. A transposed
   order is a wrong number, not a wrong shape.
2. **Both biases.** `b_ih` and `b_hh` are separately stored and separately added. Folding them
   is invisible against a reference that folds them too.
3. **`batch_first` does not move `h_0`.** The input and the output transpose; `h_0`/`c_0` are
   `(layers * directions, batch, H)` either way.
4. **The hidden row must be staged.** Every unit's gates read the *whole* previous `h` row.
   Writing `h[unit]` before unit *k+1* is computed feeds a half-stepped hidden state into the
   rest of the row.

Number 4 was **this kernel's own first bug**, and it is the one that motivates the brief's
"multi-step sequence with a non-trivial hidden state". It survived a shape check, it survived
`h_n`/`c_n` having the right dimensions, and it was caught the first time the output was
compared element-wise against upstream on a `seq=3, hidden=2` fixture — max difference 0.059.
A single timestep would not have shown it: with `hidden` units updated once from a state
nothing else reads, in-place and staged are the same computation.

### 2.2 Agreement with upstream

96 combinations (four dtypes x {1,2,3} layers x bidirectional x has_biases x batch_first), each
a `seq=4` sequence from a non-zero `(h_0, c_0)`, compared element-wise against upstream in a
separate process:

```text
float64    exact
float32    2.4e-07   max absolute difference   (golden tolerance 1e-5)
float16    2.0e-03                             (5e-3)
bfloat16   1.6e-02                             (6e-2)
```

Not bit-exact, and it should not be claimed as such: upstream's CPU path uses a gemm and this
kernel uses a plain dot product. What *is* reproduced deliberately is the **storage rounding**:
`h` and `c` are narrowed to the input dtype after every timestep, which is where upstream's CPU
path keeps them, and which is what stops a `float16` run from drifting over a long sequence.

`nn.LSTM` itself, and the bare `torch.lstm(...)` spelling, are both exercised: `overloads.json`
lists `lstm.data` **first** (the vendored `_VariableFunctions.pyi` declaration order) and both
overloads take nine positional arguments, so the earlier row swallowing the call was a real
possibility. It does not, and
`test_rnn.py::test_torch_lstm_spelling_resolves_to_the_input_overload` is what says so.

<!-- DOCWATCH: op-implemented aten.lstm.input -->

---

## 3. `aten::upsample_linear1d` — the kinship was verified, and it does not fully hold

docs/kernels/GLU.md §2 recorded this op's schema and flagged docs/architectures/DEMAND8.md's three bicubic traps as
*"likely but unverified"* for it. Verified, one at a time, against upstream:

| docs/architectures/DEMAND8.md's bicubic trap | does it transfer to `linear1d`? |
|---|---|
| `align_corners=False` does not clamp the source index at 0 | **No.** linear *does* clamp, like bilinear. The `scales=2.0` case has an unclamped first index of `-0.25` and upstream returns exactly `0.0`. |
| no `out == in` short circuit | **Not observable.** With `align_corners=False` and scale 1 the index arithmetic is exact anyway, so the branch cannot be seen by value. Kept for bilinear's reason, and the doc says so rather than claiming a measurement. |
| weights stored at the input's dtype, not `opmath_t` | **No.** Weights are `opmath_t`. And the reason it does not transfer is structural: linear is a **two**-tap kernel whose taps share one lambda pair, so storing the lambdas narrower is indistinguishable. Bicubic's trap lives in having *four* taps. |

So none of the three transferred, which is exactly why they were run rather than read.

**Two differences from `upsample_bilinear2d` that a transcription would have got wrong**, both
found by comparing raw float32 bit patterns rather than by tolerance:

* **The source index is one FUSED multiply-add.** `fma(scale, index + 0.5, -0.5)`, not a
  multiply followed by a subtraction. On the float32 4 → 7 resample, the unfused form rounds
  `0.5714286 * 3.5` to exactly `2.0` and yields lambdas `(0.5, 0.5)`; upstream yields
  `(0.49999988, 0.50000012)`. Those were recovered by feeding one-hot inputs, so it is the
  weights themselves and not an accumulation artefact.
* **The accumulation is `fma(l0, v0, l1 * v1)`** — fused on the *first* tap, with `l1 * v1`
  rounded first. On the 4 → 5 resample, output column 3, that gives upstream's
  `14.299999237060547` where the plain sum, the other fusion, and the `v0 + l1 * (v1 - v0)`
  lerp all give `14.300000190734863`.

Both are ~1–3 ULP. **The golden harness's float32 tolerance is 1e-5 and would have passed
either way**, which is why `test_rnn.py::test_upsample_linear1d_is_bit_exact_not_merely_close`
compares `struct.pack` bit patterns instead, over nine output sizes x both `align_corners`
values. With the fusions, the kernel is bit-identical to upstream on every one of them.

Two more differences that are structure rather than arithmetic:

* **`uint8` is refused here.** `upsample_nearest1d` computes it (a gather never averages) and
  `upsample_bilinear2d` has a separate fixed-point kernel. `upsample_linear1d` raises
  `"compute_indices_weights_linear" not implemented for 'Byte'` — and that refusal *name*, not
  bilinear's `upsample_bilinear2d_channels_last`, is the one transcribed.
* **The lower tap needs an upper clamp.** `upsample_bilinear2d` clamps only `i1`, because
  nothing in its callers reaches past the end. An explicit `scales=0.5` on a 4 → 8 resample
  reaches a source index of 14.5; the bilinear transcription would have read out of bounds.

<!-- DOCWATCH: op-implemented aten.upsample_linear1d.default -->

### 3.1 What is NOT done: the `torch._C._nn.upsample_linear1d` binding

The same gap docs/kernels/GLU.md §1.1 recorded for `glu` and `tools/golden/reach_allow.json` already
carries for `im2col`, `col2im` and `upsample_nearest1d`: `F.interpolate(x_3d, mode="linear")`
binds `torch._C._nn.upsample_linear1d`, which is a `bootstrap.py::_install_nn` entry, and
`bootstrap.py` is not this round's territory. Upstream has no `torch.upsample_linear1d` and no
`Tensor.upsample_linear1d`, so an `overloads.json` row would invent a door upstream lacks.

The kernel is implemented, golden-compared and bit-compared; the allowlist entry says so, names
the missing `_install_nn` line, and **fails the suite the day that line lands** — which is when
it should be deleted. So `sam_vision_model` and `sam_hq_vision_model` are not claimed as
cleared here.

---

## 4. Which of the four architectures clear

`arch_sweep.py --only parakeet_rnnt parakeet_tdt lasr_ctc lasr_encoder`, this build:

```text
before   0/4       parakeet x2: lstm            lasr x2: torch.conv1d
after    2/4  ok   parakeet_rnnt, parakeet_tdt
              --   lasr_ctc, lasr_encoder still stop in bootstrap.py's conv1d composite (§1.1)
```

`lasr_ctc` and `lasr_encoder` clear with the four lines in §1.2 and **no new kernel**, proven
in-process. `sam_vision_model` / `sam_hq_vision_model` need the one `_install_nn` line in §3.1.

Neither parakeet hit a further wall, which was not the expected outcome — the brief's "several
will hit a further wall" held for zero of the two here.

---

## 5. Verification, and the one gate this round cannot pull

```text
cargo build --release                                  EXIT=0
pytests (all test_*.py)                                763 ok
tools/golden/compare.py                                10809/10809, ops=289, 0 failed
docwatch                                               669 PASS
arch_sweep --only <the four>                           2/4 forward
```

**One test fails, and it is out of territory by construction:**

```text
FAIL test_the_mps_readback_list_is_what_the_kernels_actually_do:
  these ops read an mps tensor back to the host and are NOT refused -- add them to
  MPS_HOST_READBACK_OPS in device.rs: ['aten.lstm.input', 'aten.upsample_linear1d.default']
```

Both kernels do exactly what that check says: they read their operands to the host with
`read_flat` and compute in Rust. The readbacks were deliberately placed **in the dispatched
function** rather than behind a second helper, because docs/architectures/VOICE3.md found the derivation
follows helpers only one level by name — so the check *sees* them, which is the point. The two
entries belong in `device.rs`, which this round was told not to touch, so they are reported
rather than added:

```rust
pub const MPS_HOST_READBACK_OPS: [&str; 86] = [   // was 84
    ...
    "aten.log2_.default",
    "aten.lstm.input",                            // <- insert
    "aten.masked_scatter.default",
    ...
    "aten.upsample_bilinear2d.default",
    "aten.upsample_linear1d.default",             // <- insert
    "aten.upsample_nearest1d.default",
```

That single failure is also the only reason docwatch reports `smoke_ok = 479` against its
`ge 480` claim in three documents. Both go green with the two lines above and nothing else.

### 5.1 Counted the way docs/architectures/ARCH200.md §3.3 asks

```text
functionality added     2 kernels (aten.lstm.input, aten.upsample_linear1d.default)
                        1 overload table entry (torch.lstm, both overloads)
defects fixed           0 pre-existing (the staged-hidden-row bug was this round's own)
tests added             25 in pytests/test_rnn.py; 770 golden cases (10039 -> 10809 is
                        this round plus what merged before it -- the round's own
                        contribution is the two builders)
documentation corrected 1 (docs/architectures/ARCH200.md's `torch.conv1d` row: a name, and already spelled)
deleted                 nothing
```

"No unimplemented items" is **not** claimed. Three named gaps remain and each has an owner:
the four lines in `bootstrap.py`'s `conv1d` (§1.2), one `_install_nn` line for
`upsample_linear1d` (§3.1), and two lines in `device.rs` (§5). None of them is a kernel.
