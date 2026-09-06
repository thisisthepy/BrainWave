# PAD — the op in front of the speech roadmap, from two directions

Worktree `work/pad` on develop. Territory: `rust/torch_c/src/aten.rs`,
`methods.json`, `overloads.json`, `tools/golden/cases.py`, and the new
`rust/torch_c/pytests/test_pad.py`. `bootstrap.py`, `tensor.rs`, `dtype.rs`,
`device.rs`, `capture.rs`, `tape.rs`, `tools/wheel/` and `torchnative/` were
not touched. Two files outside that list were edited and both are named with
their reasons in §6.

Two rounds arrived at the same operator independently. docs/COMPLEX.md §5
measured that `torch.stft` raises the **identical** error for
`return_complex=True` and `return_complex=False`, before any transform, and
that the error is `torch._C._nn.pad(mode='reflect')` — real-valued, nothing to
do with complex. docs/VOICE.md §1 ranks `F.pad(reflect)` (#12) as blocking four
of its five speech models and `F.pad(replicate)` (#13) a fifth case in bigvgan.

## 0. Answers first

| question | answer | where |
|---|---|---|
| Does reflect repeat the edge element? | **No. Replicate does.** `[1,2,3]` padded by 2 is `[3,2,1,2,3,2,1]` vs `[1,1,1,2,3,3,3]`. | §1 |
| Are 1-D, 2-D and 3-D separate kernels? | **Yes, six separate schemas** — and all six are implemented, because both modes are *separable* and the arithmetic is one gather per axis. | §2 |
| Is negative padding a crop? | **Not exactly, and this is the trap.** It is a negative offset into one gather, so a reflection can read elements a crop would have discarded. | §2.2 |
| Does upstream refuse a pad wider than the input? | **Reflect yes, replicate no, circular yes-but-differently.** Measured on all three. | §2.3 |
| Does `torch.stft` work now? | **No.** The wall moved from the pad to the FFT behind it, which is what this round claims and all it claims. | §4 |
| What is still needed for `F.pad` to reach these kernels? | Four lines in `bootstrap.py`, written out in full. | §5 |

Landed, split the way docs/VOICE.md §3 asks for rather than as one number:

* **feature added** — 7 kernels: 6 padding
  (`reflection_pad{1,2,3}d`, `replication_pad{1,2,3}d`, one shared gather) and
  `rms_norm` (§8).
* **tests added** — 23 in `pytests/test_pad.py`; 169 golden cases (9691 → 9860).
* **defect fixed** — none; nothing here existed to be broken.
* **not done** — `circular` (§3), the `bootstrap.py` binding (§5), and the four
  other voice ops (§9).

<!-- DOCWATCH: op-implemented aten.reflection_pad1d.default -->
<!-- DOCWATCH: op-implemented aten.reflection_pad2d.default -->
<!-- DOCWATCH: op-implemented aten.reflection_pad3d.default -->
<!-- DOCWATCH: op-implemented aten.replication_pad1d.default -->
<!-- DOCWATCH: op-implemented aten.replication_pad2d.default -->
<!-- DOCWATCH: op-implemented aten.replication_pad3d.default -->
<!-- DOCWATCH: op-implemented aten.rms_norm.default -->

## 1. Reflect versus replicate, proven against upstream

The headline, on one input, both modes, upstream 2.13.0:

```text
x = [1, 2, 3]                       padded by 2 on each side

reflect     [3, 2, 1, 2, 3, 2, 1]   mirror; the edge element is NOT repeated
replicate   [1, 1, 1, 2, 3, 3, 3]   the edge element IS repeated
circular    [2, 3, 1, 2, 3, 1, 2]   wraps around
constant    [0, 0, 1, 2, 3, 0, 0]   already worked, via constant_pad_nd
```

**Getting reflect and replicate backwards produces plausible-looking output** —
right shape, right dtype, every element drawn from the input — so a check that
only asserts "it grew and the middle survived" passes either way.
`test_pad.py::test_reflect_does_not_repeat_the_edge_and_replicate_does`
therefore asserts three things, not one: each mode against its transcribed
literal, that the two **disagree**, and the structural property (reflect's
neighbour of the edge is the interior, replicate's is the edge itself) that
survives a change of fixture.
`test_the_two_modes_disagree_on_every_rank` repeats the disagreement check in
all three ranks, because the swap is possible in each independently.

`constant` was already correct and does route to `constant_pad_nd`, confirmed
with a `TorchDispatchMode` trace: `F.pad(x, (1,1), "constant", 0)` produces
exactly one record, `aten.constant_pad_nd.default`. Nothing there needed
changing.

## 2. What the kernel actually is: one gather, applied per axis

Both modes are a **gather over the original axis**:

```text
output position j reads input index i, where   i = j - left, then
    reflect     fold i back into [0, w-1]:  i < 0 -> -i ;  i > w-1 -> 2(w-1) - i
    replicate   clamp i into [0, w-1]
```

That is `pad_source_index` in `aten.rs`. It was written from the `[1,2,3]`
cases, then **predicted eight hostile cases before they were run** —
`[4,4]`, `[0,4]`, `[4,0]`, `[-2,4]`, `[4,-2]`, `[-4,4]`, `[2,2]`, `[-1,1]` on
`[1,2,3,4,5]` — and upstream agreed with all eight.

**Both modes are separable**, checked rather than assumed:
`reflection_pad2d(x, [1,2,1,0])` equals `reflection_pad1d` on the last axis
followed by `reflection_pad1d` on the one before it. That is why one function
serves six schemas, and it is pinned by
`test_pad2d_equals_padding_the_last_axis_then_the_one_before` rather than
asserted here. `padding` is read **back to front in pairs** — `padding[0:2]`
is the last axis — the same convention `constant_pad_nd` documents.

The gather is candle's `index_select` with an index tensor built directly on
the input's device. **No element of the input is read back to the host** —
only its shape is inspected — so this op does *not* belong in
`device.rs::MPS_HOST_READBACK_OPS`.

### 2.1 Which rank do the blocked models use

`torch.stft` needs **1-D** (`reflection_pad1d`); the trace in §4 shows it.
bigvgan/vocos/f5 are conv1d stacks and also want 1-D. 2-D and 3-D were
implemented anyway because, given a separable per-axis gather, they are three
extra dispatch arms rather than three extra kernels — and refusing them by
name would have cost more code than implementing them.

### 2.2 Negative padding is not a crop pass

`constant_pad_nd` crops with `narrow` and then pads, and that is right *for
constant*. It is wrong here:

```text
reflection_pad1d([1,2,3,4], [-1, 3])   ->   [2, 3, 4, 3, 2, 1]      upstream
                                             crop-then-mirror gives [2,3,4,3,2,3]
```

The trailing `1` is the element the `-1` crop removed. A two-pass
implementation has already discarded it and must invent something; the gather
reads the original axis and gets it right for free. Pinned by
`test_a_reflection_reads_past_what_a_crop_would_have_discarded`.

### 2.3 The refusals, all transcribed from a run

| behaviour | reflect | replicate |
|---|---|---|
| pad `>=` extent | **refused** — `Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (4, 0) at dimension 2 of input [1, 1, 4]` | **allowed** — `[99,99]` on `W=4` returns a length-202 tensor |
| `padding` wrong length | `padding size is expected to be 2, but got: 4` | `padding size is expected to be 2` — **no suffix** |
| empty output, 1-D | `input (W: 4) is too small. Calculated output W: 0` | same |
| empty output, 2-D/3-D | **returns a `(1,3,0)` tensor**, no error | same |
| negative output dim | `reflection_pad2d` alone: `numel: integer multiplication overflow` | `Trying to create tensor with negative dimension -1: [1, 3, -1]` |
| `Bool` | `"reflection_pad1d" not implemented for 'Bool'` | `"replication_pad1d" not implemented for 'Bool'` |
| every other dtype | computes (int32, int64, float16, bfloat16, float32, float64) | same |

Four of those rows are upstream **inconsistencies**, reproduced rather than
tidied:

* the `", but got: N"` suffix exists on `reflection_pad*` and not on
  `replication_pad*` — all six were run before this was believed;
* `pad1d` refuses an empty output where `pad2d`/`pad3d` return one;
* `reflection_pad2d` reports an overflow where its own 3-D sibling and both
  replicate siblings report a negative dimension;
* `Argument #N` is `4 + 2k` for the k-th pair, so a `pad2d`'s height pair is
  `#6` and a `pad3d`'s depth pair is `#8`.

A shim that made any of these consistent would be wrong in whichever direction
it chose, and the caller reading the message is exactly who would be misled.

**Rank**: `ndim + 1` and `ndim + 2` are both legal, and **only the batch axis
may be empty** — `(0,2,4)` pads, `(1,0,4)` is refused with "possibly 0 batch
size and other non-zero dimensions". Measured, not read off the sentence.

### 2.4 Result of the comparison

388 cases — six ops, seven shapes, seventeen paddings, six dtypes — run on
upstream and on the shim and diffed:

```text
total=388   identical=387   (245 value cases, 142 refusals)   MISMATCH=1
```

The single divergence is upstream returning a **tensor with a negative
dimension**:

```text
reflection_pad2d(zeros(0,2,3,4), [-5,0,0,0])
  upstream  ->  OK, shape [0, 2, 3, -1]
  shim      ->  RuntimeError: numel: integer multiplication overflow
```

With an empty batch the `numel` product is 0, so upstream's overflow guard
does not fire and it constructs a shape it would reject anywhere else. That is
an upstream defect reachable only by combining an empty batch with a crop past
the axis; candle cannot represent the result and this shim refuses instead. It
is recorded here rather than worked around.

## 3. `circular` — not done, and why it is a different shape of work

`circular` did **not** fall out cheaply. Traced, it is not an aten op at all:

```text
F.pad(x, (1,1), mode="circular")
  -> new_empty, slice x2, copy_, slice x2, copy_, slice x2, copy_
```

It is a Python-level composite above the dispatcher, built from
`new_empty`/`slice`/`copy_` — so there is no kernel for this round to add.
Implementing it means writing that composition in `bootstrap.py`, which was
not this round's file. Its semantics were measured anyway so the next round
does not re-derive them: it wraps (`[1,2,3]` by 2 gives `[2,3,1,2,3,1,2]`), it
allows `pad == extent`, and it refuses `pad > extent` with
`Padding value causes wrapping around more than once.`

## 4. Does `torch.stft` move? Yes — from the pad to the FFT. It does not work.

`torch/functional.py:172` is where the pad happens, and it is Python, not C++:

```python
    if center:
        ...
        input = F.pad(input.view(extended_shape), [pad, pad], pad_mode)   # line 172
        input = input.view(input.shape[-signal_dim:])
    return _VF.stft(...)                                                  # line 174
```

**Before** (this tree, `bootstrap.py` unmodified — the pad kernels exist but
`F.pad` cannot reach them yet):

```text
NotImplementedError: not implemented in torch._C shim: torch._C._nn.pad(mode='reflect')
-- upstream routes this to aten::reflection_pad*/replication_pad*/a circular
composition rather than to aten::constant_pad_nd, and none of those has a kernel
here; mode='constant' is implemented
```

**After** (same probe, with §5's four-line composite installed in-process so
the routing is exercised without editing another agent's file), verbatim, and
identical for `return_complex=True`, `return_complex=False` **and**
`center=False`:

```text
NotImplementedError: not implemented in torch._C shim: torch.stft(...)
-- overload resolution has no table entry for this op
(rust/torch_c/src/overloads.json); call torch.ops.aten.stft.<overload>, which
carries the overload and reaches the same dispatcher
```

The pad and both `view`s now succeed; the failure is at `_VF.stft` on line 174,
one line later. **This does not make `torch.stft` work and nothing here should
be read as saying it does.** What is behind that line is the transform, and
docs/VOICE.md §3 established that candle 0.11.0 has no FFT of any kind:

```text
aten._fft_r2c.default:   NotImplementedError: aten op not implemented in torch._C shim
aten.as_strided.default: NotImplementedError: aten op not implemented in torch._C shim
implemented ops matching "fft":  NONE
```

So docs/COMPLEX.md's ordering — **reflect pad → `Repr::Complex` → an FFT** —
holds, and exactly the first of the three is now done. `center=False` gives the
same error, which is worth stating because COMPLEX.md §5 listed it as the thing
that "may" route around the pad: it does route around the pad, and it arrives
at the same next wall regardless.

No `stft` row was added to `overloads.json`. A row would move the refusal from
the binding to the aten name without moving the wall, and `aten.stft.center`
has no kernel either.

## 5. What `bootstrap.py` still needs — the exact patch

`torch._C._nn.pad` is a `bootstrap.py::_install_nn` composite (around line
7437) that today refuses every non-constant mode by name. `bootstrap.py` was
another agent's file this round, so the kernels are proven through
`_C._aten_dispatch` and `tools/golden/compare.py` instead — the same hand-off
docs/GLU.md §1.1 made, which worked.

Replace the `if mode != "constant": raise ...` guard with:

```python
        if mode != "constant":
            n = len(pad) // 2
            if mode == "reflect":
                return dispatch(f"aten.reflection_pad{n}d.default", input, list(pad))
            if mode == "replicate":
                return dispatch(f"aten.replication_pad{n}d.default", input, list(pad))
            raise NotImplementedError(
                f"not implemented in torch._C shim: torch._C._nn.pad(mode={mode!r}) "
                f"-- reflect and replicate are implemented; circular is a "
                f"new_empty/slice/copy_ composite upstream (docs/PAD.md §3)"
            )
```

`n` is `len(pad) // 2` and not the input's rank: upstream picks the kernel by
how many pairs it was given, and refuses the combinations that do not line up
with the rank *inside* the kernel (`Padding size 2 is not supported for 4D
input tensor`, with its own three-line table). Passing the count through and
letting the kernel judge reproduces that; deriving `n` from the rank would
accept combinations upstream rejects.

Whoever lands this should then **delete the six
`aten.*_pad*d.default` entries from `tools/golden/reach_allow.json`** — they
are written to fail the suite the moment the binding lands — and re-run
`pytests/arch_sweep.py`, since `univnet` (docs/ARCH100.md lists `_nn.pad` as
its wall) and docs/VOICE.md ranks 12 and 13 only clear at that point. **Until
then, "four speech models clear `F.pad`" is not true end-to-end**; what is true
is that the kernels they need exist and are golden-compared, and the remaining
step is binding surface, not numerics.

## 6. The two files edited outside the stated territory, and why

* **`rust/torch_c/pytests/test_shim.py`** — one pinned count,
  `tag_core_count`, 117 → 122. This is the one edit the round's rules allow
  there, and the arithmetic is what keeps it a check: **the delta is five, not
  six.** `replication_pad1d` is `['pt2_compliant_tag']` upstream while
  `replication_pad2d` and `replication_pad3d` — the same op one and two ranks
  up, served here by the *same function* — are both `core`. Each of the six was
  read off its own `.tags`; inferring from a sibling would have written 123,
  and the test would still have passed the day it was written.
* **`tools/golden/reach_allow.json`** — six `shape2_kernel_without_spelling`
  entries. `reach.py` fails the suite for a kernel with no Python spelling, and
  these have none: upstream has `torch._C._nn.reflection_pad1d` but **no**
  `torch.reflection_pad1d` and no `Tensor.reflection_pad1d` (checked for all
  six), so an `overloads.json` or `methods.json` row would invent a door
  upstream does not have — the reasoning `aten.upsample_nearest2d.default`
  already records in that file. The allowlist is the sanctioned mechanism for
  exactly this state, and each entry names the `bootstrap.py` work item and
  deletes itself when it lands.

## 7. Gates

```text
suite      610 ok, EXIT=0            (gate: 587+)
docwatch   PASS -- 573/573           (gate: 562+)
golden     9860/9860, ops=262        (gate: 9691+, ops 255+)
           pending case builders=0
```

<!-- DOCWATCH: count golden_cases_total ge 9860 -->
<!-- DOCWATCH: count golden_cases_passed ge 9860 -->
<!-- DOCWATCH: count golden_ops_covered ge 262 -->
<!-- DOCWATCH: count golden_pending eq 0 -->

## 8. `rms_norm` — taken, and the trap was the default `eps`

`aten::rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None,
float? eps=None)`. docs/VOICE.md §1 rank 16, `f5`'s forward; real-valued and
independent of the complex question, which is why it fitted this round.

```text
out = input * rsqrt(mean(input^2, over the trailing k axes) + eps) * weight
```

No mean subtraction — that is the whole difference from `layer_norm` — and
**`eps` is inside the square root**, checked against the outside form rather
than assumed.

**The finding worth carrying forward: the default `eps` is the *accumulation*
dtype's epsilon, not the input's.** A `float16` or `bfloat16` input uses
`finfo(float32).eps` = 1.1920929e-07, not `finfo(float16).eps` = 9.765625e-04.

```text
rms_norm([[1, 2, 3, 4]], [4])              float16 input
    finfo(f32).eps and finfo(f16).eps agree to EVERY printed digit

rms_norm([[1e-3, 2e-3, 3e-3, 4e-3]], [4])  float16 input
    upstream                   0.362305     <- finfo(f32).eps
    with finfo(f16).eps        0.031891     <- an order of magnitude out
```

This is the shape docs/GLU.md warns about and the round's brief warns about for
`i0`: two candidate implementations that agree in the ordinary range and
diverge at one end. A golden suite built on `arange`-sized values would have
passed with the wrong constant, so both the golden cases and
`test_the_default_eps_is_the_accumulation_dtypes_epsilon_not_the_inputs` use
`1e-3` values where the mean square is comparable to epsilon. `float64` takes
`finfo(float64).eps`, so the rule is the accumulation dtype throughout.

Three refusals differ from `native_layer_norm`'s despite validating the same
argument, and all three were run side by side on the same input to be sure the
difference is upstream's:

| | `rms_norm` | `native_layer_norm` |
|---|---|---|
| shape mismatch | `expected input with shape [*3]` | `expected input with shape [*, 3]` |
| `normalized_shape` longer than rank | **`ValueError`**: `Input tensor must have at least 3 dimensions, but got 2` | (no such check) |
| integer input | `"rms_norm" not implemented for 'Long'` | `"LayerNormKernelImpl" not implemented for 'Long'` |
| mismatched `weight` dtype | **accepted** (f32 weight on a bf16 input computes) | a "mixed dtype (CPU)" refusal ladder |

113 cases were run on both sides and diffed: **103 identical, 10 differing only
in the last `float32` digit** (~5e-7 relative, against a 1e-5 golden
tolerance). Every dtype, every shape and every refusal agrees exactly; the
residue is that upstream uses a fused `rsqrt` where this shim writes
`sqrt().recip()` — the same association `aten.rsqrt.default` and
`native_layer_norm` already use here, so it is house-consistent rather than a
choice made for this op.

Unlike the six padding kernels, `torch.rms_norm` **does** exist upstream, so it
gets a real `overloads.json` row rather than a `reach_allow.json` entry, and
`test_torch_rms_norm_reaches_its_kernel_in_the_vendored_tree` runs the vendored
shim in a subprocess to prove the row resolves. It is **not** `core` upstream
(`['pt2_compliant_tag']`), so `tag_core_count` stays at 122.

## 9. Not taken this round

docs/VOICE.md §1's other real-valued ops — `kaiser_window` (needs a modified
Bessel `i0`), `upsample_nearest1d`, `var`, `col2im`/`im2col` — were left. None
is blocked on anything here.

`kaiser_window`'s `i0` in particular should be checked **across the argument
range** rather than at one point: series and asymptotic implementations agree
in the middle and diverge at the ends. §8's `eps` finding is the same shape and
is the reason to take that warning literally — the wrong constant there was
invisible at ordinary magnitudes and an order of magnitude out at small ones.

`var` was scoped and skipped: it has five overloads upstream (`default`,
`dim`, `correction`, and two `out` forms) with two different spellings of the
same parameter (`unbiased` and `correction`), so it is a resolution-table
question as much as a kernel one and did not fit beside this round's work.
