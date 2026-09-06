# The FFT — the third wall, and `torch.stft` produces upstream's numbers

Worktree `work/fft` on develop `2498122`. Territory: `rust/torch_c/src/aten.rs`,
`src/overloads.json`, `tools/golden/cases.py` and the new
`rust/torch_c/pytests/test_fft.py`. Five files outside that list were edited and
each is named with its reason in §8. `tensor.rs`, `dtype.rs`, `bootstrap.py`,
`capture.rs`, `tape.rs`, `tools/wheel/` and `torchnative/` were not touched.

`docs/COMPLEX.md` set an ordering — **reflect pad → complex tensors → the
FFT** — and the two previous rounds cleared the first two (`docs/PAD.md`,
`docs/COMPLEX2.md`). This is the third and last of them.

## 0. Answers first

| question | answer | where |
|---|---|---|
| Does `torch.stft` produce upstream's numbers? | **Yes, on a real signal, through the public Python entry point, with `center=False`.** Element-wise against a live upstream, worst relative disagreement **4.9e-7**. | §1 |
| And with `center=True`? | **Yes, but one line short of reachable.** `torch/functional.py:678` pads in Python through `torch._C._nn.pad`, whose `reflect` branch is still missing from `bootstrap.py` — another agent's file, and docs/PAD.md §5 already wrote the four-line patch. With that composite installed in-process the centred output agrees with upstream to 1.2e-7. | §6 |
| Is `n_fft` a power of two in every model? | **No, and this was measured rather than assumed.** `whisper`, `qwen3_asr` and `voxtral_realtime` all use **400**. So no size is refused: radix-2 where it applies, **Bluestein** everywhere else. | §4 |
| What are the normalisation conventions? | The `normalization` argument is **not** a `norm=` string. `0 → 1`, `1 → 1/sqrt(n)`, `2 → 1/n`, and the **same three codes serve both directions** — `rfft`'s `backward` default is `0` and `ifft`'s `backward` default is `2`. `stft(normalized=True)` is code **1**. | §2.2 |
| And `onesided`? | `n // 2 + 1` bins. Default `True` for a real input. `_fft_c2r` additionally reads only the first `last_dim_size // 2 + 1` bins, **even when more are supplied**. | §3 |
| Was candle forked? | **No.** candle 0.11.0 still has no FFT (docs/VOICE.md §3) and now still has none — the arithmetic is written in `aten.rs`. | §2 |
| Did `as_strided` have to land? | **No, and that is asserted.** Upstream's `stft` frames its input with `as_strided`; this one gathers. `aten.as_strided.default` was still unimplemented after this round; `docs/STRIDED.md` landed it later and `stft` still does not use it. | §5.2 |
| What did the test catch? | **Two defects, both of which return a plausible spectrum**: a conjugated Bluestein transform, and `_fft_c2r` reading one bin too many. | §2.1, §3 |

Split the way docs/ARCH100.md §5.3 asks, rather than as one number:

* **feature added** — 5 kernels: `_fft_r2c`, `_fft_c2c`, `_fft_c2r`,
  `stft.default`, `stft.center`; one `overloads.json` row (two schemas).
* **tests added** — 14 in `pytests/test_fft.py`; 38 golden cases
  (10039 → 10077, ops 270 → 272).
* **defect fixed** — none pre-existing; the two in §2.1 and §3 were this
  round's own, found before landing.
* **tests inverted** — 1 in `pytests/test_tail2.py` (the `fft_` absence
  assertion), 1 example moved in `test_shim.py`, 2 pinned counts.
* **not done** — `istft` (§7.2), `fft_fftn`'s *spelling* (§5.3), a
  device-side transform (§7.1), `torch.stft(center=True)`'s last line (§6).

<!-- The three `_fft_*` keys are in IMPLEMENTED_AWAITING_GOLDEN, not in
     `_aten_implemented()` (§3, §7.3), so `op-implemented` is the wrong marker
     for them and `op-not-implemented` would be a lie in the other direction.
     Their dispatch arms are watched directly instead; `pytests/test_fft.py`
     and `pytests/test_tail2.py` both pin the parked list itself. -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs fft_r2c_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs fft_c2c_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs fft_c2r_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs fft_bluestein present -->
<!-- DOCWATCH: op-implemented aten.stft.default -->
<!-- DOCWATCH: op-implemented aten.stft.center -->
<!-- DOCWATCH: op-implemented aten.as_strided.default -->
<!-- DOCWATCH: op-not-implemented aten.istft.default -->

## 1. The bar: `torch.stft`, on a real signal

Not `torch.ops.aten.stft` — the public function, which goes through
`torch/functional.py` and back into the shim's `_VF.stft`:

```python
sig = torch.sin(torch.arange(64.) * 0.31) + 0.4 * torch.cos(torch.arange(64.) * 1.7) \
      + torch.arange(64.) * 0.01
torch.stft(sig, 16, 4, window=torch.hann_window(16), center=False, return_complex=True)
```

```text
                                  shape        worst relative disagreement
torch.stft(center=False)          (9, 13)      1.2e-07
  return_complex=False            (9, 13, 2)   1.2e-07
  normalized=True                 (9, 13)      4.9e-07
whisper's shape, n_fft=400        (201, 8)     6.3e-08
```

**The shim is the more accurate of the two.** It transforms in `f64` on the
host where upstream transforms in `f32`, so the residual above is upstream's
rounding rather than this code's — which is why `test_fft.py`'s tolerance of
`1e-5` is twenty times the observed worst case and still nowhere near wide
enough to hide any of the failure modes §2 lists.

`torch.stft(center=True)` is **not** in that table. §6 says exactly where it
stops and what it does when that line is supplied.

## 2. The transform, which is arithmetic and not a binding

`docs/VOICE.md` §3 grepped candle 0.11.0 and found no FFT of any kind — the
only hits for "fft" are a comment in `conv.rs` and a commented-out row in
`npy.rs`'s dtype table. That was re-checked rather than inherited, and it still
holds. So there is no candle call to reach for.

**It runs on the host, in `f64`, and that is a choice with a real cost.** Every
other kernel in `aten.rs` keeps its data on the device. This one reads it back.
The reason is that the alternatives are worse rather than that this is good:

* a **DFT matrix multiply** stays on-device and is one `matmul` — but it
  accumulates 1024 `f32` products per output bin and diverges from upstream by
  ~1e-4 relative. That is a *uniform* error, which is the size that a loose
  tolerance hides and a "looks like a spectrum" eyeball check cannot see.
* a **device-side radix-2** would need a scatter/gather kernel per stage, and
  candle has no primitive that expresses a butterfly.

`f64` on the host costs a round trip and buys agreement at ~1e-7. §7.1 is the
standing item.

### 2.1 The defect that a symmetric input cannot see

`fft_bluestein`'s first build defined its chirp as `exp(-sign·iπm²/N)` where
the identity needs `exp(+sign·iπm²/N)`. Every non-power-of-two transform came
out as the **complex conjugate** of upstream's: same shape, same dtype, same
magnitudes, every real part correct.

It was invisible to every power-of-two case (they never reach that function)
and it would have been invisible to a real *symmetric* input, whose spectrum is
its own conjugate. It was caught on the first run because `test_fft.py`'s
`x5 = [1, 2, 4, 8, 3]` is deliberately both asymmetric **and** of length 5.

```text
_fft_r2c([1,2,4,8,3], [0], 0, True)
  upstream   [18, 0]  [-7.163,  3.302]  [0.663, -3.216]
  shim       [18, 0]  [-7.163, -3.302]  [0.663,  3.216]
                              ^ sign
```

### 2.2 Normalisation, measured in both directions

The `normalization` argument is an `int`, and it does **not** name a `norm=`
string. Traced with a `TorchDispatchMode`:

```text
torch.fft.rfft(x, norm="backward")   -> _fft_r2c(x, [0], 0, True)     default
torch.fft.rfft(x, norm="ortho")      -> _fft_r2c(x, [0], 1, True)
torch.fft.rfft(x, norm="forward")    -> _fft_r2c(x, [0], 2, True)

torch.fft.ifft(X, norm="backward")   -> _fft_c2c(X, [0], 2, False)    default
torch.fft.ifft(X, norm="ortho")      -> _fft_c2c(X, [0], 1, False)
torch.fft.ifft(X, norm="forward")    -> _fft_c2c(X, [0], 0, False)
```

So the code means **the factor**, and the same three codes serve both
directions:

```text
0 -> 1        1 -> 1/sqrt(n)        2 -> 1/n
```

A shim that read the code as "which `norm=` string" would be right forward and
wrong by a factor of `n` inverse. `forward` and `inverse` select the **sign of
the exponent and nothing else**; the scale is applied once, at the end, from
the code.

**`torch.stft(normalized=True)` is code 1**, traced —
`_fft_r2c(frames, [2], 1, True)`. Code 2 would divide by `n_fft` where this
divides by `sqrt(n_fft)`; both produce a spectrum, and only the ratio
distinguishes them, which is how `test_fft.py` pins it.

### 2.3 The refusals, transcribed rather than tidied

```text
float16 / bfloat16   expected scalar type Double but found Half
int64                Only supports floating-point dtypes, but found: Long
complex into r2c     Only supports floating-point dtypes, but found: ComplexFloat
real into c2c        Only supports complex dtypes, but found: Float
```

The first reads like an upstream defect — it names `Double`, which is not what
it can take — and it is reproduced anyway, because a caller diagnosing a dtype
problem matches on the message it actually gets. All four are compared against
a **running** upstream in `test_fft.py` rather than against this transcription,
so they cannot drift.

**An empty transformed axis is not an error.**
`_fft_r2c(zeros(0), [0], 0, True)` is a length-1 tensor upstream, because
`0 // 2 + 1 == 1` and the sum of no elements is zero. Measured; "an empty input
is an error" is the obvious guess and it is wrong.

## 3. `onesided`, and the bin count that is wrong in the other direction

`_fft_r2c` with `onesided=True` returns **`n // 2 + 1`** bins. A full-length
output is self-consistent, has the right rank and dtype, and is wrong. Pinned
for an even length (8 → 5) and an odd one (5 → 3), because `n // 2 + 1` and
`(n + 1) // 2` agree on odd lengths and disagree on even ones.

**`_fft_c2r` has the mirror trap and it cost a second defect.**
`last_dim_size` is authoritative, not the input's bin count — a 5-bin input
with `last_dim_size = 9` gives a length-9 tensor and with `last_dim_size = 7` a
length-7 one. The first build rebuilt the Hermitian spectrum from *all* the
bins it was given:

```text
_fft_c2r(rfft([1,2,4,8,3,-1,0.5,7]), [0], 2, 7)
  upstream   2.214  1.737  6.706  7.504  0.054  -0.591  ...
  shim       1.630  2.305  6.267  7.727  0.090  -0.880  ...
```

Bin 4 is simply not part of a 7-point Hermitian spectrum. The rule is
`used = min(last_dim_size // 2 + 1, bins)`, and everything past it is dropped
rather than mirrored. Wrong by 0.58 on values of order 10 — a signal of the
right length and the wrong values, with no shape to give it away.

**`dim` may name more than one axis.** Upstream does a `c2c` over every axis
except the last named one and an `r2c` over that one, and the `onesided`
truncation applies to the last entry only:
`_fft_r2c(randn(4,8), [0,1], 0, True)` is `(4, 5)`. That is `rfft2`'s shape, so
the multi-axis path is implemented rather than refused — §5.3 on why that is
not the same as landing `fft_fftn`.

## 4. "n_fft is a power of two" — measured, and false

This round was scoped to *measure* that claim and, if it held, to implement the
power-of-two path well and **refuse other sizes by name**. It does not hold.

Every `transformers` feature extractor was scanned for an `n_fft` /
`filter_length` default (`inspect.signature` over
`transformers.models.*.feature_extraction_*`):

```text
n_fft = 400    whisper, qwen3_asr, voxtral_realtime
n_fft = 512    cohere_asr, granite_speech, lasr, nemotron_asr_streaming,
               parakeet, phi4_multimodal
n_fft = 1024   clvp;   filter_length = 1024   univnet
n_fft = 16384  musicgen_melody
```

**400 is not a power of two, and `whisper` is in this tree's own HF cache.**
Refusing non-powers of two by name would have refused the mel front-end of the
most-used speech model here, and would have done it in a way that reads like a
deliberate narrowing rather than a gap.

So no size is refused. `fft_radix2` handles powers of two, and **Bluestein's
chirp-z algorithm** handles everything else by turning the DFT into a
convolution the radix-2 path can do:

```text
n k = (n^2 + k^2 - (k - n)^2) / 2
X[k] = conj(chirp[k]) * sum_n (x[n] * conj(chirp[n])) * chirp[k - n]
```

padded to the next power of two at least `2n - 1` long. `m*m % (2n)` rather
than `m*m` keeps the angle small — for `n = 16384`, `m^2` leaves an `f64`'s
exact-integer range around `m = 2^26`, and long before that the argument
reduction inside `sin_cos` is discarding the bits this function's accuracy
rests on. The residue is exact, since `exp(-iπm²/n)` has period `2n` in `m²`.

Twiddles use `sin_cos` per stage rather than the recurrence `w *= w_step`. The
recurrence is faster and drifts with the stage length; for `n_fft = 1024` that
is the difference between agreeing with upstream at 1e-7 and at 1e-5. It costs
`n - 1` transcendental calls in total, not `n log n`, because each stage's
table is built once and reused across that stage's blocks.

## 5. `torch.stft`'s framing

### 5.1 What the kernel does

```text
hop_length  defaults to n_fft // 4          win_length defaults to n_fft
n_frames    1 + (len - n_fft) / hop
frames      (batch, n_frames, n_fft), frame f starting at f * hop
window      centred inside n_fft when win_length < n_fft:
              left = (n_fft - win_length) // 2, so the odd element goes RIGHT
transform   _fft_r2c(frames, dim=[-1], normalization = normalized ? 1 : 0, onesided)
layout      transpose to (batch, n_freq, n_frames); squeeze the batch axis the
            1-D form invented
result      return_complex=True  -> Repr::Complex
            return_complex=False -> stack([re, im], -1)
```

`torch.stft` reaches the **`default`** overload, not `center`:
`torch/functional.py:672-680` does the centring in Python and then calls
`_VF.stft` with the nine-argument form. `aten::stft.center` is implemented
anyway because it is a real upstream overload with a real schema, and because
having it lets the centred transform be exercised at the aten level while
`bootstrap.py`'s pad branch is missing (§6).

### 5.2 The frames are a gather, not `as_strided` — and that is asserted

`docs/VOICE.md` §3 named `aten.as_strided.default` alongside `_fft_r2c` as
`stft`'s walls. **Only one of the two fell.** Upstream's frames are a strided
view onto the input's storage; these are copied by an `index_select`. That is a
narrowing in the same family as `view_as_complex`'s (docs/COMPLEX2.md §6.1) and
unobservable for the same reason: the next thing that happens to the frames is
the window multiply, and nothing in this shim can write through them in
between.

`test_fft.py::test_as_strided_landed_and_stft_still_does_not_use_it` asserts
it, so the claim "`as_strided` was not needed" stays true by measurement rather
than by this paragraph.

**`docs/STRIDED.md` implemented `as_strided`, and `stft` still does not use
it** — which is the note this paragraph was written to leave, arriving at its
addressee. The test was inverted rather than deleted and now checks the
kernel's source instead of the op list: `stft`'s frames must remain an
`index_select` gather. The reason is no longer only "the aliasing is
unobservable" but a cost — an `as_strided` result bars in-place writes to its
base's storage while it lives (`docs/STRIDED.md` §2), so routing `stft` through
it would bar `stft`'s own input for the duration of the window multiply in
exchange for aliasing nothing here can observe.

### 5.3 What `fft_fftn` still needs

`_fft_r2c`'s multi-axis `dim` is the arithmetic `torch.fft.fftn` /
`torch.fft.rfft2` need, and `_fft_c2c` takes a multi-axis `dim` too. What is
**not** here is `aten.fft_fftn.default` — the op `fnet` dispatches, which is a
separate kernel above these with its own `s=`/`dim=`/`norm=` argument handling.
`docs/COMPLEX.md` §3.3 step 5 is therefore still open, and it is now a binding
rather than arithmetic.

### 5.4 The refusals

Upstream prefixes **every** `stft` refusal with the whole resolved call and the
reason after a ` : `. The details are not guessable and were transcribed from
runs of each branch:

```text
stft(torch.FloatTensor[64], n_fft=16, hop_length=4, win_length=16,
     window=torch.FloatTensor{[16]}, normalized=0, onesided=None,
     return_complex=1, align_to_window=None) : expected hop_length > 0, but got hop_length=0
```

The tensor prints with square brackets and the window with braces *around*
brackets; `normalized` is `0`/`1` and `onesided` is `None`/`0`/`1`;
`win_length` shows the **resolved** default, so `n_fft=0` prints
`win_length=0`.

Ordering, measured rather than guessed:

| check | message |
|---|---|
| **`return_complex` missing, first of all** | `stft requires the return_complex parameter be given for real inputs, ...` — no prefix, and it fires **ahead of** the dtype and rank checks |
| `align_to_window` with `center=True` | `stft align_to_window should only be set when center = false.` |
| dtype | `expected a tensor of floating point or complex values` |
| rank | `expected a 1D or 2D tensor` |
| `n_fft` | `expected 0 < n_fft < {len}, but got n_fft={n}` |
| `hop_length` | `expected hop_length > 0, but got hop_length={h}` |
| `win_length` | `expected 0 < win_length <= n_fft, but got win_length={w}` |
| window | `expected a 1D window tensor of size equal to win_length={w}, but got window with size [8]` |

**`expected 0 < n_fft < 64` is what the message says and `n_fft <= 64` is what
it enforces** — `stft(randn(16), n_fft=16)` computes and returns a single
frame. Reproduced with its own bound rather than corrected, because a caller
matching on the text gets upstream's text. An empty signal reports
`expected 0 < n_fft < 0`.

All eight are compared against a **live** upstream in `test_fft.py`.

## 6. `center=True`: where it stops, and what it does when that line is supplied

`torch/functional.py:674-679` is Python, not C++:

```python
    if center:
        signal_dim = input.dim()
        extended_shape = [1] * (3 - signal_dim) + list(input.size())
        pad = int(n_fft // 2)
        input = F.pad(input.view(extended_shape), [pad, pad], pad_mode)   # line 678
        input = input.view(input.shape[-signal_dim:])
    return _VF.stft(...)                                                  # line 680
```

Line 680 now works. **Line 678 does not**, verbatim:

```text
NotImplementedError: not implemented in torch._C shim: torch._C._nn.pad(mode='reflect')
-- upstream routes this to aten::reflection_pad*/replication_pad*/a circular
composition rather than to aten::constant_pad_nd, and none of those has a kernel
here; mode='constant' is implemented
```

The six pad kernels behind that message exist and are golden-compared —
docs/PAD.md landed them — and `bootstrap.py`'s `_install_nn` composite is the
one line between them and `F.pad`. docs/PAD.md §5 contains the exact patch;
`bootstrap.py` was not that round's file and it is not this one's either.

**So the wall has moved by exactly one line, backwards, and it is the line the
previous round already wrote the fix for.** With that composite installed
in-process — which is what `test_fft.py`'s probe does, and what docs/PAD.md §4
did — the centred transform agrees with upstream:

```text
torch.stft(sig, 16, 4, window=hann(16), center=True)          (9, 17)    1.2e-07
  normalized=True                                             (9, 17)    4.9e-07
  pad_mode="replicate"                                        (9, 17)    9.7e-08
whisper's shape, n_fft=400, center=True                       (201, 11)  6.3e-08
```

**This does not make `torch.stft(center=True)` work in this tree and nothing
here should be read as saying it does.** What is true is that the kernels are
right and the remaining step is binding surface in another file. Whoever lands
docs/PAD.md §5's patch should also delete the six `aten.*_pad*d.default`
entries from `tools/golden/reach_allow.json`, which are written to fail the
suite the moment the branch lands.

## 7. Left standing, each a decision rather than an omission

### 7.1 The host readback

All five kernels read their input back to the CPU (§2), so all five are in
`MPS_HOST_READBACK_OPS` and **refuse on `mps` rather than computing on the
GPU's data on the CPU**. Unlike almost everything else on that list, these do
not read back on one dtype path and stay on the device on another: there is no
device path at all until candle grows a butterfly primitive or this crate
writes a Metal kernel. The refusal is the honest answer; it is not a bug to be
worked around by removing the entry.

The read is written out **in the body of each of the four dispatched
functions** rather than behind a shared helper, and that is deliberate:
`test_shim.py`'s mps gate scans each kernel's own body for `.to_vec*` and
follows helper calls exactly one level, by name. A two-level chain through a
new helper is invisible to it, and the op would go on reading device bytes with
the suite green. The first version of this code had exactly that shape and the
classification test caught it.

### 7.2 `istft`

Not implemented, and it is a genuinely different problem rather than this one
unwritten: an overlap-add with a window-sum normalisation on top of `_fft_c2r`.
`_fft_c2r` — the transform half — **is** implemented and compared. `istft` is
now `test_shim.py`'s standing example of a name with no `overloads.json` entry,
which `stft` was until this round.

### 7.3 `torch.fft.rfft` and friends have no `torch._fft_*` door

`torch._fft_r2c` exists upstream as a `_VariableFunctions` member and this shim
deliberately does **not** put a row in `overloads.json` for it. The door nobody
uses is not the one that reaches the kernel: `torch.fft.rfft`/`fft`/`ifft`/
`irfft` are C++-level composites that land on `aten::_fft_*` through the
dispatcher, and `torch.stft` reaches it through `aten::stft`, which is spelled
and exercised. Three entries in `tools/golden/reach_allow.json` record this
with the reason; delete them if a caller for the bare spelling is found.

### 7.4 Complex input to `stft`

Upstream routes a complex input to `_fft_c2c`; this shim's `stft` is built on
`_fft_r2c` only and refuses a complex input by name. No measured caller passes
one — `torch.stft` on a real waveform is the whole of docs/VOICE.md's demand.

## 8. The five files edited outside the stated territory

| file | edit | why |
|---|---|---|
| `rust/torch_c/src/device.rs` | +5 op names in `MPS_HOST_READBACK_OPS`, `71` → `76` | **Required, not optional.** The gate is symmetric: an op that reads back and is not declared fails, *and* a declared op that does not read back fails. There is no way to classify these five from inside `aten.rs`. Data only; no logic changed. §7.1. |
| `tools/golden/reach_allow.json` | +3 entries for the `_fft_*` keys | Same class of edit docs/PAD.md made. Each carries its reason and an `upstream_absent` claim that is put to a real upstream by `reach.py`. §7.3. |
| `rust/torch_c/pytests/test_shim.py` | 2 pinned counts, 1 example moved | §8.1. |
| `rust/torch_c/pytests/test_tail2.py` | 1 assertion inverted | That file's docstring asks an implementing round to invert rather than delete. Nothing was removed: the list is now pinned to exactly the three, so a fourth still fails. |
| `docs/VOICE.md` | 1 DOCWATCH marker inverted | `op-not-implemented aten.stft.default` was true when written and is now false. Inverted to `op-implemented` for both overloads rather than deleted, with a note pointing at this file. |

### 8.1 The two pinned counts, with the arithmetic that keeps them checks

**`tag_core_count` 125 → 127. The delta is two, not five.** Each of the five
new keys was read off its own `.tags`:

```text
_fft_r2c.default   ['core', 'pt2_compliant_tag']   <- counted
_fft_c2r.default   ['core', 'pt2_compliant_tag']   <- counted
_fft_c2c.default   ['pt2_compliant_tag']           <- NOT core
stft.default       ['pt2_compliant_tag']           <- NOT core
stft.center        ['pt2_compliant_tag']           <- NOT core
```

`_fft_c2c` not being core while its two siblings are — all three implemented
here by the same `dft_in_place` — is upstream's table and not derivable.
Inferring from the round would have written 130.

**Schema identities 337 → 339. The delta is two, not five.** `overloads.json`
gains the two real `aten::stft` overloads because upstream really does have
`torch.stft`. The three `_fft_*` kernels contribute **zero**: a row for any of
them would invent a door upstream's callers do not use (§7.3), the same
reasoning the six pad kernels record above it in that file.

## 9. Gates

```text
suite      684 ok  (baseline 668 + 16 in pytests/test_fft.py), EXIT=0
DOCWATCH   PASS -- 625/625
golden     10077/10077 cases passed, 0 failed, ops covered=272, pending 0
           (baseline 10039/10039, ops 270)
```

The 38 new golden cases are all `aten.stft.*`. `ops covered` moves by **two**
and not five, which is the check that §3's parking is real: the three `_fft_*`
keys are dispatchable and unadvertised, exactly as `view_as_complex` and its
four neighbours have been since docs/COMPLEX2.md.
