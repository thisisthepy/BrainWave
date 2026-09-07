# BIND4 — four small `bootstrap.py` items, plus one handed off mid-round

Worktree `work/bind4` on develop `26ef12c`, vendored tree assembled fresh. torch 2.13.0
upstream (`/Volumes/macMini/caches/spike-venv/bin/python`). Territory: `rust/torch_c/src/
bootstrap.py`, `tools/golden/reach_allow.json`, and a new `rust/torch_c/pytests/
test_bind4.py`. `aten.rs`, `tensor.rs`, `dtype.rs`, `device.rs`, `capture.rs`, `tape.rs` were
not touched. `test_rnn.py` and `test_tail2.py` were each touched once, at the exact spot their
own docstrings said to invert when the fix landed (§1, §5) — `test_shim.py` was not touched at
all.

Every item here was located precisely by a previous round that could not fix it because
`bootstrap.py` was not its file. **Each binding's kernel was checked as really present before
the binding was written** — `docs/bindings/BINDINGS.md`'s `mish` is what happens when that check is
skipped, and it is restated per item below rather than assumed once.

---

## 0. At a glance

| item | kernel really there? | architectures |
|---|---|---|
| 1. `conv1d(padding="same")`, odd total | Yes — both `constant_pad_nd` and `convolution` | `lasr_ctc`, `lasr_encoder`: **ok** |
| 2. `torch._C._nn.upsample_linear1d` | Yes — golden- and bit-compared | `sam_vision_model`, `sam_hq_vision_model`: **ok** |
| 3. `torch.zeros` with a 0-dim Tensor in `size` | N/A — argument form, not a kernel | `fastspeech2_conformer`: moves past this wall onto `repeat_interleave` (needs a kernel, `aten.rs`, **not actionable here**) |
| 4a. `torch._C._nn.avg_pool2d` | Yes, already bound (docs/bindings/BIND2.md) | confirmed still bound |
| 4b. `nystromformer` | N/A | `aten.convolution.default`'s asymmetric-padding refusal — candle backend limitation, `aten.rs`, **not actionable here** (re-confirms docs/bindings/ARGFORM.md §1) |
| 4c. `univnet` | N/A | `TensorBase.unfold` — missing kernel, `aten.rs`, **not actionable here** |
| 5. `torch._C._fft.fft_fftn` / `Tensor.real` / `Tensor.imag` (handed off mid-round) | Yes — `_fft_c2c`/`real`/`imag` all implemented | `fnet`: binding lands and is reached; numeric claim belongs to the merged tree (§5) |

Suite: **833 ok**, 0 FAIL, `DOCWATCH: PASS` 742/742, `EXIT=0`. Golden: **11307/11307,
ops=298** — exactly unmoved. No kernel added anywhere in this round.

---

## 1. `conv1d(padding="same")`, odd `dilation * (kernel - 1)` — `lasr_ctc`, `lasr_encoder`

**Both kernels were already there.** `aten.constant_pad_nd.default` and
`aten.convolution.default` have been implemented and golden-compared since long before this
round (`docs/kernels/OPS4.md`). `bootstrap.py`'s `conv1d` composite already handled every branch except
this one, which it refused by name (docs/kernels/RNN.md §1).

The fix is the four-line transcription docs/kernels/RNN.md §1.2 measured and could not land (RNN was not
`bootstrap.py`'s owner that round): pad one zero on the RIGHT with `constant_pad_nd`, then
convolve symmetrically with `total // 2`, in place of the `raise`. Landed verbatim, with the
docstring above it updated to describe the lowering instead of the refusal.

`test_rnn.py::test_conv1d_same_with_odd_total_padding_is_still_refused` was written, per its own
docstring, to flip to `_agree(...)` "when it lands" — renamed to
`test_conv1d_same_with_odd_total_padding_now_agrees` and inverted accordingly, not deleted.
`test_bind4.py` adds five more cases through the actual spelling (`F.conv1d`, not the raw aten
op): the odd-total case that now computes, the even-total case that already worked and is
unaffected, a second odd-total case at a different dilation (not overfit to `dilation=1`), the
`"valid"` branch, and — the one that must NOT get swallowed by the new branch — non-unit stride
with `padding="same"`, which is upstream's own refusal and stays a refusal.

Measured against upstream, `F.conv1d(x, w_even, b, 1, "same", 1, 3)` on the exact `lasr`-shaped
fixture: shim `972.3858`, upstream `972.3857` (float32 tolerance). `F.conv1d(..., "valid", ...)`
still matches unchanged.

`arch_sweep.py --one lasr_ctc` / `--one lasr_encoder`: both **ok**.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind4.py test_conv1d_same_odd_total_now_computes present -->

---

## 2. `torch._C._nn.upsample_linear1d` — `sam_vision_model`, `sam_hq_vision_model`

**The kernel was already there.** `aten.upsample_linear1d.default` was implemented, golden-
compared, and bit-compared against upstream by docs/kernels/RNN.md §3 — the gap was one `_install_nn`
line, shaped like `upsample_nearest1d`'s one line above it, and specifically not landed there
because `bootstrap.py` was not that round's file either (docs/kernels/RNN.md §3.1).

**The discriminator is the fourth argument's TYPE, not the arity** — docs/bindings/BIND3.md §3.1's exact
trap for `upsample_nearest1d`, one name above this in `torch._C._nn`. Both the `.vec` schema
(`input, output_size, align_corners, scale_factors`) and the leaf schema (`self, output_size,
align_corners, scales`) take four arguments here, unlike the 2-D op, whose leaf has a fifth
argument to key on. Measured on upstream 2.13.0:

```text
_nn.upsample_linear1d(x, [9], False, None)     -> .default(x, [9], False)
_nn.upsample_linear1d(x, None, True, [1.5])    -> .default(x, [9], True, 1.5)
_nn.upsample_linear1d(x, [9], True, 1.5)       -> .default(x, [9], True, 1.5)
_nn.upsample_linear1d(x, [9], False)           -> .default(x, [9], False)
```

A sequence fourth argument is `.vec`'s `scale_factors`; a float is the leaf's `scales`. Written
following `upsample_bilinear2d`'s shape (mutual-exclusion refusal, scale forwarded rather than
only used to size the output).

`tools/golden/reach_allow.json`'s `aten.upsample_linear1d.default` entry is deleted — its own
text said "fails the suite the moment the entry lands", and it did (`REACH` failure), confirming
the entry described a real, now-closed gap and not a stale one.

`test_bind4.py` proves it through `F.interpolate(x_3d, mode="linear")`, both `align_corners`
values, seven output widths (crossing the FUSED-multiply-add ULP boundary docs/kernels/RNN.md §3
measured), two scale factors including a non-integral one (`2.25`, so the forwarded-vs-
recomputed grids separate), the three-argument leaf form, both mutual-exclusion refusals
(compared to upstream's own message text, not merely "refuses"), and that `uint8` still refuses
at the kernel rather than being papered over by the new binding.

`arch_sweep.py --one sam_vision_model` / `--one sam_hq_vision_model`: both **ok**.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind4.py test_f_interpolate_linear_agrees_with_upstream present -->

---

## 3. `torch.zeros` with a 0-dim Tensor inside the size tuple — `fastspeech2_conformer`

Not a kernel question at all — an argument-form question, and it was measured against real
upstream before being added, per docs/bindings/ARGFORM.md's own bar.

`FastSpeech2ConformerLengthRegulator.length_regulator` does exactly this (`modeling_
fastspeech2_conformer.py:112`):

```python
max_len = torch.sum(duration_labels, dim=1).max()          # a 0-dim Tensor, not an int
hidden_states = torch.zeros(
    (encoded_embeddings.size(0), max_len, encoded_embeddings.size(2)),
    dtype=torch.float, device=encoded_embeddings.device,
)
```

Measured directly: `torch.zeros((2, torch.tensor(15), 5), dtype=torch.float)` **succeeds**
upstream and gives shape `(2, 15, 5)`. This shim's overload resolver refused it —
`TypeError: torch.zeros(): no matching overload ... for (tuple, dtype=dtype, device=str)` —
because its `SymInt[]` list checker only accepted `int` elements.

**Measured to be general, not `zeros`-specific**, the same way docs/bindings/ARGFORM.md §2 requires before
generalising a rule: a 0-dim integral Tensor inside a `SymInt[]` size list is upstream's own
argument-parser rule, confirmed against `torch.ones`, `torch.empty`, and `Tensor.view` as well
(all four accept it, unprompted). It is still **installed only for `zeros`**, the same scoping
`div`'s wrapped-number override uses one function above it in `bootstrap.py` — folding it into
`_TypeChecker` would accept a Tensor in every `SymInt[]` position table-wide with no matching
measurement for the others, exactly the silent-divergence trap docs/bindings/ARGFORM.md §2 names.

### 3.1 The trap this one hid: `DeviceContext` matches by object identity

The first version of this override was a bare wrapper —
`def zeros(size, *a, **kw): return _table_zeros(_coerce(size), *a, **kw)` — and it broke
`with torch.device("meta"): torch.zeros(2)`, silently returning a **CPU** tensor. Caught by
`test_meta_road_through_the_vendored_tree`, not designed for.

The reason: `torch/utils/_device.py`'s `DeviceContext.__torch_function__` decides whether to
inject `device=` by testing `func in _device_constructors()`, where `_device_constructors()`
reads `torch.zeros` etc. **fresh off the `torch` module at call time** — object identity, not a
name. Every table-driven factory in `bootstrap.py` carries its own `if _MODE_STACK: return
_through_torch_function_modes(fn, args, kwargs)` guard, passing **itself** as `func`, which is
exactly what makes the identity check line up. A wrapper that skips straight to the *inner*
table-driven closure hands the mode `_table_zeros`'s inner `fn` as `func` — no longer the object
now sitting at `torch.zeros` (that is the wrapper) — so the identity check fails and no device
gets injected, silently.

The fix reproduces the same guard in the wrapper, passing the wrapper itself as `func`:

```python
def zeros(size, *args, **kwargs):
    if _MODE_STACK:
        return _through_torch_function_modes(zeros, (size,) + args, kwargs)
    return _table_zeros(_coerce_symint_size_tensors(size), *args, **kwargs)
```

`_through_torch_function_modes` pops the mode for the duration of the call, so the re-entrant
`zeros(*args, **kwargs)` inside `DeviceContext.__torch_function__` lands on the second branch,
not an infinite loop. Verified directly: `with torch.device("meta"): torch.zeros(2)` gives
`meta` again, and the tensor-in-size-tuple case still works inside that same context.

`test_bind4.py` covers the argument form (`zeros((2, tensor, 5), dtype=, device=)`, a
single-tensor-only size, multiple tensor elements mixed with plain ints, the pre-existing plain
and varargs forms unaffected, and a multi-element Tensor still refusing). The device-context
identity trap is what the *existing* `test_meta_road_through_the_vendored_tree` in `test_shim.py`
catches — that test was not touched, and passing it unmodified is the proof the wrapper is
transparent to it.

### 3.2 Not actionable further from here

`arch_sweep.py --one fastspeech2_conformer`, before this round, stopped at `torch.zeros`. After:
it moves one wall further, into `torch.repeat_interleave` with a **Tensor** `repeats` argument
(`length_regulator`'s very next line, `torch.repeat_interleave(encoded_embedding,
target_duration, dim=0)`):

```text
NotImplementedError: not implemented in torch._C shim: torch.repeat_interleave with a tensor
`repeats` -- upstream lowers it to aten::repeat_interleave.Tensor followed by aten::index_select,
and this shim has neither kernel; the integer `repeats` spelling is implemented
```

Both `aten::repeat_interleave.Tensor` and `aten::index_select`'s tensor-repeats path are missing
kernels in `aten.rs`, not `bootstrap.py`. `fastspeech2_conformer` is therefore **not clearable
past this point from this file** — recorded rather than left unattributed, per docs/kernels/TAIL4.md
§8.2's own prediction that this was "the next wall" and not the last one.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind4.py test_zeros_tensor_in_size_tuple_agrees present -->

---

## 4. `avg_pool2d` confirmed bound; `nystromformer` and `univnet` confirmed NOT actionable here

`torch._C._nn.avg_pool2d` is bound (`docs/bindings/BIND2.md`), confirmed present in `_shim_nn_implemented`
and its kernel in `_aten_implemented()` — `test_bind4.py::test_avg_pool2d_is_still_bound`, so
this is checked rather than assumed to have survived every round since.

`nystromformer` (`arch_sweep.py --one nystromformer`, re-run this round): stops at
`aten.convolution.default: an asymmetric padding [32, 0] is not implemented in torch._C shim --
candle's conv2d takes one value per argument, not one per axis`. This is `docs/bindings/ARGFORM.md` §1's
exact finding, re-measured and re-confirmed rather than assumed still true: a genuine candle
backend limitation in `aten.rs`, not an argument-form gap and not `bootstrap.py`'s to fix.

`univnet` (`arch_sweep.py --one univnet`, re-run this round): its `torch._C._nn.pad` wall
(ARCH200.md's original classification) is already closed — `pad` is bound in this tree. The
architecture now stops one line later, at `hidden_states.unfold(2, hop_size + 2 * padding,
hop_size)`: `NotImplementedError: not implemented in torch._C shim: TensorBase.unfold`. That is a
missing **kernel** (the tensor sliding-window VIEW method, distinct from `F.unfold`/`im2col`,
which this shim does have), in `aten.rs`, not a binding.

Neither has a line to add in `bootstrap.py`. Both are reported as such rather than left
unattributed.

---

## 5. `torch._C._fft.fft_fftn`, `Tensor.real`, `Tensor.imag` — `fnet` (handed off mid-round)

Handed off by the coordinating session mid-round: the complex round (docs/kernels/COMPLEX3.md, merging
into `develop` after this worktree branched) taught `_to_copy(dtype=complex64)`, `slice`,
`constant_pad_nd`, `view`, and `aten.complex` on `Repr::Complex`, and expected `fft_fftn` to then
work — right about the operators, wrong about the door. `docs/kernels/COMPLEX3.md` §6.2 carries the
two-line patch verbatim; it is landed here **exactly as written**, in `bootstrap.py`, which is
this round's file.

**Both kernels were already there.** `aten.real.default`, `aten.imag.default`, and
`aten._fft_c2c.default` are all implemented in this worktree's `aten.rs` (confirmed by grep and
by direct call — `torch.polar(...).real` computes end to end right now, since `polar` already
produces a `complex64` tensor without going through `_to_copy`). What was missing was the door:
`torch/fft/__init__.py` is `fftn = _add_docstr(_fft.fft_fftn, ...)`, `_fft` had no stub data, so
every name on it fell to the catch-all `_Unimplemented` — and `Tensor.real`/`Tensor.imag` are
**properties**, so `methods.json` (keyed on names called with `()`) could never carry them.

Landed:

* `module._fft.fft_fftn` — the decomposition docs/kernels/COMPLEX3.md §6.1 read off a
  `TorchDispatchMode` logger: widen to complex via `_to_copy`, optionally slice/pad each
  transformed axis to `s`, then `_fft_c2c` with `norm` mapped to `{backward: 0, ortho: 1,
  forward: 2}`. Installed next to `linalg_vector_norm`, the same shape of fix for the same reason
  (`_fft`/`_linalg` both have no stub data).
* `tensorbase.real` / `tensorbase.imag` — `property(lambda self: dispatch("aten.real.default",
  self))` and the `imag` sibling, installed from a new `_install_tensor_complex_parts`, called
  from `_install_tensor_methods` next to `_install_tensor_T`.

**No `aten.fft_fftn.default` kernel was added**, deliberately, per docs/kernels/COMPLEX3.md §6.3's own
reasoning kept intact: it would duplicate a decomposition upstream already has, would still not
reach `torch.fft.fftn` (which never routes through `torch.ops.aten`), and would read its input
back to the host through `_fft_c2c` — a `device.rs` row invisible to the readback derivation,
which follows helpers only one level by name. `grep 'aten.fft_fftn'` over `_aten_implemented()`
confirms empty.

### 5.1 What is checked here, and what belongs to the merged tree

This worktree was cut **before** the complex round's `_to_copy(dtype=complex64)` gate merged, so
on this tree the numeric path still refuses one level in:

```text
Tensor.real on a real (non-complex) tensor:
  RuntimeError: aten.real.default: expected a complex tensor, got a float32 one
    -- the KERNEL's own type check, not the catch-all. Correct behaviour.

Tensor.real on a genuinely complex tensor (produced by `torch.polar`, which does not
need `_to_copy(complex64)`):
  computes. torch.polar(...).real / .imag both agree with the values `polar` itself produced.

torch._C._fft.fft_fftn(torch.ones(4)):
  NotImplementedError: aten._to_copy.default: dtype not storable by the candle backend in
  torch._C shim: torch.complex64
    -- refuses from INSIDE the real decomposition (the widen-to-complex step), not from a
    name lookup. `fft_fftn` is no longer the catch-all.

arch_sweep.py --one fnet:
  stops at the SAME `_to_copy` line, called from inside `fft_fftn` (`bootstrap.py:2890`,
  frame reads "in fft_fftn") rather than from the old catch-all frame -- the wall moved
  one level down, exactly as expected for a worktree missing the complex round's merge.
```

That is exactly what a correctly-landed binding onto a not-yet-merged kernel looks like: the
name resolves, is reached, and fails deeper in, at the piece this round did not own. The full
numeric claim — a two-layer `FNetModel` forward agreeing with upstream element-wise, max
absolute difference 7.15e-07 over 256 outputs — is docs/kernels/COMPLEX3.md §6.1's own measurement, from
a tree where both halves are present, and is **not re-claimed here**.

`test_tail2.py::test_stft_computes_and_fft_fftn_is_still_unspelled` was inverted a third time
(its own docstring anticipated exactly this: "fails, demanding its own inversion, the moment
someone lands that name"). The new assertion checks the narrower thing this tree can prove: the
refusal is no longer the catch-all (`"fft_fftn" not in msg`), and is specifically the `_to_copy`
complex64 gate (`"_to_copy" in msg and "complex64" in msg`) — not merely "still raises something".

`tools/golden/reach_allow.json`'s `aten._fft_c2c.default` entry is deleted: its own text said
"delete this entry if a caller for the bare spelling is found", and `fft_fftn` is now that
caller — confirmed by `reach.py`'s static scan failing exactly there before the entry was
removed (`"a spelling now reaches it (or its kernel is gone)"`).

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_tensor_complex_parts present -->

---

## 6. Golden, suite, docwatch

```text
cargo build --release                                          EXIT=0
PYTHON=$PY sh rust/torch_c/pytests/run.sh                       833 ok, 0 FAIL, DOCWATCH 742/742, EXIT=0
tools/golden/compare.py                                         11307/11307, ops=298, 0 failed -- exactly unmoved
arch_sweep --one lasr_ctc / lasr_encoder / sam_vision_model      ok / ok / ok
arch_sweep --one fastspeech2_conformer                            moved wall (torch.zeros -> repeat_interleave)
arch_sweep --one nystromformer / univnet                          unchanged (both aten.rs, not actionable here)
arch_sweep --one fnet                                             moved wall one level deeper (into fft_fftn's own body)
```

No kernel was added anywhere in this round. Golden's case count and ops-covered figure are
identical to the round's starting point, which is the expected result for a round that is
entirely `bootstrap.py` bindings, argument-form fixes, and one type-checker guard.

### 6.1 Counted the way CLAUDE.md §5.3 asks

```text
functionality added     3 bindings (upsample_linear1d, fft_fftn, real/imag) that reach
                         kernels already present; 1 composite branch (conv1d's odd-total
                         "same" padding) built from kernels already present; 1 argument-form
                         fix (zeros accepting a 0-dim Tensor in its size list), scoped to
                         zeros alone
defects fixed            0 pre-existing in this round's own code -- the DeviceContext identity
                         trap (§3.1) was this round's own bug, caught before landing
tests added              19 in test_bind4.py, all comparing the actual user-facing spelling
                         against a live upstream in a separate process
tests inverted            2: test_rnn.py's conv1d refusal (renamed to _now_agrees),
                         test_tail2.py's fft_fftn-is-still-unspelled half (narrowed rather
                         than simply flipped, since the numeric claim needs the un-merged gate)
documentation corrected  0 -- ARCH200.md's avg_pool2d/im2col/pad rows for
                         efficientnet/convbert/univnet were already stale before this round
                         (their bindings landed elsewhere); not re-corrected here since only
                         nystromformer/univnet were in this round's brief
deleted                  2 reach_allow.json entries (upsample_linear1d, _fft_c2c), each because
                         its own text said to delete it once a caller/binding landed
not actionable here      3: fastspeech2_conformer past torch.zeros (repeat_interleave, aten.rs),
                         nystromformer (asymmetric conv padding, aten.rs), univnet
                         (TensorBase.unfold, aten.rs)
```

"No unimplemented items" is not claimed. Three named gaps remain outside this file's reach, each
with an owner (`aten.rs`, three different ops), and `fnet`'s numeric claim is explicitly deferred
to the tree where the complex round's merge and this round's binding are both present.
