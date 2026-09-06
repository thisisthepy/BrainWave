# Complex, widened to the ops two architectures need

Round date 2026-09-07. Branch `work/cplx2`, on develop `26ef12c`. Host Apple M1,
CPython 3.13, upstream torch 2.13.0, `candle-core` 0.11.0.

`docs/COMPLEX2.md` built `Repr::Complex { re, im }` and taught it twelve ops.
`docs/FFT.md` built `_fft_r2c` / `_fft_c2c` / `_fft_c2r` and matched
`torch.stft` against upstream. `docs/BIND3.md` §6 then measured, op by op,
**exactly which further complex ops two architectures still stop on** — and
this round taught those.

The proof is `rust/torch_c/pytests/test_cplx2.py`: ten tests, every value
compared element-wise against a live upstream torch in a separate process, on
**both** components.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs complex_to_copy present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs complex_slice present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs complex_constant_pad_nd present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs complex_view present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs complex_default present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json complex present -->

---

## 0. Answers first

| question | answer | where |
|---|---|---|
| Which ops were taught? | **Five.** `aten._to_copy.default` (real → complex), `aten.slice.Tensor`, `aten.constant_pad_nd.default`, `aten.view.default` / `aten._unsafe_view.default`, and one new key, `aten.complex.default`. | §1 |
| How is "the imaginary part survived" proved? | Element-wise against upstream in a separate process, on inputs where `re ≠ im` **and the signs differ**, plus a per-op check that the shim's own `im` column is neither zeros nor a copy of `re`. | §2 |
| Was that verification shown to be able to fail? | **Yes, built and run.** Three one-line nullifications (`view` rebuilding `im` from `re`; `pad` filling both halves with `value`; `slice` narrowing `re` twice) turn four tests red and name the element and both values. | §2.3 |
| Did teaching five ops open the four hundred that were not taught? | **No.** Twelve probes still refuse, including `reshape` and `select` — the untaught *neighbours* of the two ops taught (`view`, `slice`). | §3 |
| `llama4`'s vision tower? | **The full `Llama4VisionModel` forward now completes and agrees with upstream element-wise**, max abs diff `1.7e-08` over 128 outputs, with `im2col` instrumented and measurably on the path (1 call). Measured directly, not from a sweep. | §5 |
| `fnet`? | **Not landed, and the remainder is two lines in `bootstrap.py`, which was not this round's file.** Every aten op it needs now computes and matches upstream. Supplying those two lines from outside makes `arch_sweep.py --one fnet` report `status: ok`, and a full two-layer `FNetModel` forward then agrees with upstream element-wise, max abs diff `7.2e-07`. | §6 |
| Narrowings? | **Two, both asserted as narrowings.** Upstream's `slice` and `view` return aliases of their base; these return copies, for the reason `view_as_complex` already copies. | §4 |
| Anything read back to the host? | **Nothing new.** All five kernels are candle shape calls on two tensors, so none needs a `MPS_HOST_READBACK_OPS` row in `device.rs`. | §7 |

---

## 1. The five ops, and why these five

`docs/BIND3.md` §6 is the specification for this round. It traced
`torch.fft.fftn` on upstream with a `TorchDispatchMode` and found the
decomposition, then ran each step against the installed shim and recorded which
refused. That table is reproduced here with this round's column added:

| step | `docs/BIND3.md` §6 | now |
|---|---|---|
| `aten._to_copy(x, dtype=torch.complex64)` | refuses — `dtype not storable by the candle backend` | **computes** |
| `torch.complex(re, im)` | refuses — no `overloads.json` row | **computes** |
| `aten._fft_c2c` on a genuine shim complex tensor | works | works (unchanged) |
| `aten.constant_pad_nd` on that same tensor | refuses | **computes** |
| `aten.slice.Tensor` on that same tensor | refuses | **computes** |
| `Tensor.real` | not implemented | **still not implemented — `bootstrap.py`, §6** |

and `docs/BIND3.md` §5's separate finding:

| step | §5 | now |
|---|---|---|
| `Llama4VisionRotaryEmbedding` → `reshape_for_broadcast` → `freqs_ci.view(...)` | refuses — `view` is not a taught op | **computes**, and the tower completes (§5) |

`torch.zeros(2, 2, dtype=torch.complex64)` is the one row of that table this
round deliberately did **not** close. `docs/COMPLEX2.md` §4's answer stands —
`complex64._has_storage` is `False` and must stay so — and a complex factory
would be a fourth entrance to the representation with no measured caller.
Nothing in either architecture asks for one.

### 1.1 Where they live, and why not in `tensor::complex_ops`

`docs/COMPLEX2.md` put its twelve in `tensor.rs::complex_ops`, beside the
representation. These five are in **one contiguous block at the end of
`aten.rs`**, with **one contiguous run of dispatch arms** near the top of
`dispatch_impl`, immediately after the six guarded arms that round added.

The split is not arbitrary. `complex_ops` holds the ops that are *arithmetic
over the representation* — `polar`, `mul`, `view_as_real` — and those belong
next to the thing they are defined over. These five are **variants of kernels
that already exist in `aten.rs`**: each re-uses that file's argument readers,
its clamping rule, its `-1` wildcard resolution and its crop-then-pad ordering.
Putting the complex arm in the same file as the dense arm it mirrors is what
makes "these two agree about what `start=-1` means" checkable by reading rather
than by running.

Four of the five are **guards on an existing key**, not new keys:

```rust
"aten.slice.Tensor" if first_arg_is_complex(args, kwargs) => complex_slice(py, args, kwargs),
```

The guard is a discriminant test on an already-parsed wrapper and is `false`
for every real tensor, so **no dense path changes and no key moves between
`IMPLEMENTED` and `IMPLEMENTED_AWAITING_GOLDEN`** — golden's cases go through
the dense kernels exactly as before. `test_cplx2.py::test_the_five_taught_keys_are_all_still_dispatchable`
asserts that directly, because an op silently leaving `_aten_implemented()` is
how it silently stops being golden-compared.

`_to_copy`'s guard is the one that is not `first_arg_is_complex`, and the
difference matters: its new case is a **real** input with a complex `dtype=`.
That is the direction that had no route at all, and it is why nothing
downstream of it was reachable.

`aten.complex.default` is the one genuinely new key. It is parked in
`IMPLEMENTED_AWAITING_GOLDEN` for the same reason the eight complex/FFT keys
before it are: it **returns** a complex tensor, and the golden harness compares
by reading both sides as dense tensors.

### 1.2 The three places the arithmetic could have been plausible and wrong

Worth naming, because each one returns the right shape and the right dtype:

* **`constant_pad_nd`'s fill splits.** Upstream pads a complex tensor with
  `complex(value, 0)` — measured on 2.13.0. So `re` is padded with `value` and
  `im` with **zero**. Padding both halves with `value` puts `value + value·i`
  in the pad, which passes any magnitude check. **With the schema default
  `value=0` the right answer and that wrong one are the same tensor**, which is
  why the test uses a non-zero fill and asserts the padded entries by position.
* **`view`'s wildcard resolves against the complex element count.**
  `Repr::Complex`'s shape *is* `re`'s shape (`docs/COMPLEX2.md` §1.2), so there
  is no trailing 2 and no factor of two to correct for. Resolving `-1` against
  `re.elem_count() * 2` would put the wrong extent in and still satisfy every
  check made on `re` alone.
* **A strided `slice` must index both halves with the *same* picks.** The
  kernel builds one index tensor and uses it twice. Two independently built
  ones would agree for every symmetric case; `test_cplx2.py`'s `slice_step3`
  (picks 1 and 4, neither adjacent nor symmetric) is the case chosen to catch
  them drifting.

---

## 2. The bar: proving the imaginary part survives

Every positive assertion in `pytests/test_cplx2.py` is:

1. reported through `torch.view_as_real(z).tolist()` — the only spelling that
   shows **both** halves. `.real` alone is precisely the read that cannot see
   the bug;
2. compared element-wise against upstream torch run in a **separate process**,
   with each side asserting its own marker so a mis-wired environment fails
   loudly instead of comparing something against itself;
3. **on an input where `re ≠ im` at every element and the signs differ.**
   `docs/COMPLEX2.md` §3 already made this point and this round makes it
   sharper: the base is `[[1,-2],[3,-7],[5,11],[-4,9],[6,-13],[0.5,8]]`. An
   input with `im == re` cannot distinguish "moved both halves" from "moved
   `re` twice", and an input with `im == 0` cannot distinguish either from
   "rebuilt `im` as zeros".

### 2.1 The check that is not a comparison

`_close` compares shim to upstream. That is necessary and not sufficient: it
would still pass if both sides had been asked for something whose imaginary
part happens to be zero. So each op is *also* run through `_imag_survived`,
which asserts of the **shim's own answer** that:

* the imaginary column is not identically zero, and
* the imaginary column is not equal to the real column.

Those two are exactly what the two plausible half-written shape ops produce.

### 2.2 Op by op

| op | what proves `im` survived |
|---|---|
| `_to_copy` real → complex | `im` is zero here on both sides — this is the one op where that is *correct*, so the widening case below carries the burden instead |
| `_to_copy` complex64 → complex128 | both halves widen; the result's `im` column is asserted to be `[-2,-7,11,9,-13,8]` by value. A kernel that converted only `re` cannot even build its result — `PyTensorBase::complex` refuses a pair whose candle dtypes disagree |
| `slice.Tensor` | five forms — mid, negative range, stride 2, stride 3, and a slice along dim 1 of a 2-D complex tensor. `slice_step3`'s pairs are asserted as `[(3,-7),(6,-13)]` |
| `constant_pad_nd` | five forms — default fill, **non-zero fill**, pure crop, mixed crop-and-pad, 2-D two-axis. The `5.0` case asserts the padded entries are `(5,0)` and the interior is untouched |
| `view` / `_unsafe_view` | four forms including a wildcard and `llama4`'s own `(1, S, 1, -1)`; the resulting shape is asserted as a value as well as compared |
| `complex(re, im)` | the pairs are asserted as `[(1,-4),(2,5),(3,-6)]`, plus broadcasting, plus both upstream refusals compared **byte for byte** against the same upstream process |
| pad → slice → `_fft_c2c`, and real → `_to_copy` → `_fft_c2c` | the composition, because the failure a per-op test cannot see is a result that is individually right and does not compose — a non-contiguous half the next kernel reads through the wrong layout |

### 2.3 Nullified, and it goes red

A verification that cannot fail is not a verification. Three one-line changes
were made to `aten.rs`, the crate rebuilt, the shim reinstalled and the suite
re-run:

| nullification | result |
|---|---|
| `complex_view` returns `cast(re)` for both halves | `test_view_reshapes_both_halves_at_llama4s_own_shape` red — `view_flat[1]: shim 1.0 vs upstream -2.0` |
| `complex_constant_pad_nd` pads `im` with `value` too | `test_constant_pad_nd_pads_both_halves_and_splits_a_non_zero_fill` red — `pad_five[1]: shim 5.0 vs upstream 0.0` |
| `complex_slice` returns `cut(re)` for both halves | `test_slice_keeps_the_imaginary_part_on_every_form` red — `slice_mid[1]: shim 3.0 vs upstream -7.0` |

and `test_the_fftn_shape_path_composes_end_to_end` went red as well
(`fftn_shaped[1]: shim 6.0 vs upstream -13.0`), which is the composition test
doing its job.

Two tests stayed **green** under all three nullifications, and that is correct
and worth recording: `test_slice_and_view_copy_where_upstream_aliases` is a
narrowing check rather than a value check, and
`test_to_copy_builds_a_complex_tensor_from_a_real_one` does not exercise any of
the three. Neither is evidence for the arithmetic and neither is claimed as
such.

The nullifications were reverted and the artefact rebuilt; the reverted file is
byte-identical in the three edited regions (checked by `grep -c` on all three
sites).

---

## 3. Teaching five did not open four hundred

`docs/COMPLEX2.md` §2.1's property is the thing this round could most easily
have broken: `PyTensorBase::tensor()` refuses on the complex arm, which is how
~400 kernels read their inputs, and that refusal is what makes a dropped
imaginary part *unrepresentable* rather than merely avoided.

`test_cplx2.py::test_the_untaught_ops_still_refuse` samples twelve ops. Nine
are `test_complex.py`'s (minus `slice`, see below); **three are new and were
chosen because they are the untaught neighbours of the ops this round taught**:

```
transpose  permute  index_select
```

— shape ops on the same tensors, reached through the same dispatcher. If a
guard had been written at the wrong level (on the *tag* rather than on the
*arm*, or on a shared helper rather than on the five keys), these are what
would have started computing from `re` alone. `reshape` and `select` are the
sharpest pair in the list for the same reason: `view` is now taught and
`reshape` is not; `slice` is now taught and `select` is not.

All twelve refuse, all twelve raise `NotImplementedError`, and all twelve name
`complex64` in the message.

### 3.1 One test inverted, not deleted

`test_complex.py::test_every_untaught_op_refuses_rather_than_dropping_the_imaginary_part`
sampled ten ops and **`slice` was one of them.** That assertion is now false, so
it was **inverted**: `slice` moved out of that list and into
`test_cplx2.py::test_slice_keeps_the_imaginary_part_on_every_form`, which
proves it element-wise on both components in five forms including two strides.
The docstring of the original test was updated to say where the coverage went
and why `reshape` and `select` stay.

`z.to(torch.float32)` stays in that list, and deliberately. Upstream *does*
answer it — it discards the imaginary part with a warning — and this shim does
not. `complex_to_copy` routes that case straight into `tensor()`'s own refusal
rather than writing a second wording of it, so the reader gets the one message
that names the dtype and says which half would have been lost.

`test_tail2.py` was **not** touched. Its
`test_only_the_expected_complex_ops_landed` pins the parked list by keyword
(`view_as_complex`, `view_as_real`, `polar`, `real`, `imag`, `fft_`), and
`aten.complex.default` matches none of them; its assertion that no
`fft_fftn` key exists is **still true**, because none was added (§6).

---

## 4. The narrowings, written down

`docs/COMPLEX2.md` §6 recorded one: upstream's `view_as_complex` aliases its
base and a pair copies. Two more are added here, and they have the same single
cause.

| op | upstream | here |
|---|---|---|
| `slice.Tensor` on a complex tensor | returns a **view**; writing through the base shows through | returns a **copy** |
| `view` / `_unsafe_view` on a complex tensor | returns a **view** | returns a **copy** |

A pair of real tensors cannot alias an interleaved buffer, and
`view_as_complex` already allocates unconditionally — so **no complex tensor in
this shim ever shares storage with anything**, which makes the difference
unobservable for anything built through the shim's own constructors. That is
the same argument `docs/COMPLEX2.md` §6 made for `copy_`, and it holds for the
same reason.

It is asserted as a narrowing in
`test_cplx2.py::test_slice_and_view_copy_where_upstream_aliases`, which checks
**both** directions: that upstream still aliases (measured, not assumed, so the
narrowing is re-derived rather than remembered) and that the shim does not. If
either shim answer ever becomes `True` that is an improvement, and the test
says to remove the narrowing from this document first.

---

## 5. `llama4`'s vision tower — measured directly

`docs/BIND3.md` §5 is emphatic about why the sweep cannot answer this:
`arch_sweep.py`'s `main_only` inputs for `llama4` are **text**, the vision
branch is never built, and instrumenting `im2col` during that forward records
**zero** calls. A green sweep is not evidence. So the tower was run directly,
on both libraries, from the same seed, with `torch._C._nn.im2col` wrapped to
count calls.

```
Llama4VisionModel(hidden=32, intermediate=128, layers=1, heads=4,
                  image=16, patch=8, channels=3,
                  vision_output_dim=128, projector_input_dim=128,
                  projector_output_dim=128, pixel_shuffle_ratio=0.5)
input  torch.arange(1*3*16*16).reshape(1,3,16,16) / 1000

  shim      status ok   shape (1, 1, 128)   im2col calls 1   sum -0.166985035
  upstream  status ok   shape (1, 1, 128)   im2col calls 1   sum -0.166984960

  element-wise over all 128 outputs:  max abs diff  1.676e-08
```

**The full `Llama4VisionModel` forward completes**, not just the patch
embedding. `docs/BIND3.md` got `Llama4UnfoldConvolution` agreeing to the last
digit and then stopped past `im2col` at `freqs_ci.view(...)`; that wall is the
one `complex_view` removes, and it was the only complex op the tower needed.

The residual is float32 rounding on two libraries that do not share a libm —
the same order as every other element-wise agreement in this repository.

### 5.1 The config, and why it is not the stock one

`docs/BIND3.md` §5 noted that `Llama4VisionConfig`'s own defaults fail
**upstream** too, with `mat1 and mat2 shapes cannot be multiplied`, so that
configuration proves nothing in either direction. That was reproduced first —
shim and upstream failing at the same line of `modeling_llama4.py` with the
same shapes — and then the adapter dimensions were made self-consistent
(`intermediate_size = projector_input_dim = projector_output_dim =
vision_output_dim = 128`, which is `hidden_size · (1/ratio)²`). Both sides then
run. The point of saying this is that **the config was changed to make upstream
work, not to make the shim work**: the two were failing identically before and
agree identically after.

Like `docs/BIND3.md`'s own runs, this is not in `run.sh` — it constructs a
`transformers` model, which `arch_sweep.py`'s header says must stay out of the
suite.

---

## 6. `fnet` — what computes, and the two lines that are left

**`fnet` does not forward, and the remainder is not in `aten.rs`.**

`arch_sweep.py --one fnet` on the shim, before and after this round, stops in
the same place:

```
modeling_fnet.py:170   outputs = self.fourier_transform(hidden_states).real
NotImplementedError: not implemented in torch._C shim: torch._C._fft.fft_fftn
```

`docs/BIND3.md` §6 forecast that landing the three representation ops would
make `fft_fftn` "just work", because `aten::fft_fftn` is
`CompositeImplicitAutograd` and decomposes. **That forecast is right about the
operators and wrong about the door.** There is no decomposition engine on this
path: `torch/fft/__init__.py` is `fftn = _add_docstr(_fft.fft_fftn, ...)`, so
the shim needs `torch._C._fft.fft_fftn` to be a real function, and
`_fft` has no stub data — every name on it is `bootstrap.py`'s catch-all
`_Unimplemented`. Same for `Tensor.real`, which is a **property** and is
currently the raising stub `bootstrap.py` installs from the surface list.

Both are in `bootstrap.py`, which was not this round's file.

### 6.1 What was proved instead

The op side is complete and was measured, not argued. `fnet`'s actual line —
`torch.fft.fftn(x, dim=(1,2)).real` — spelled through the aten ops that
upstream's own dispatch trace records, on a `(2,5,8)` input:

```
shim      _to_copy(x, complex64) -> _fft_c2c(c, [1,2], 0, True) -> torch.real
upstream  torch.fft.fftn(x, dim=(1,2)).real

  80 real outputs      max abs diff  9.54e-07
  160 view_as_real     max abs diff  9.54e-07     (both components)
```

on values of magnitude ~10, i.e. ~1e-7 relative — float32 rounding.

And the whole architecture, with the two missing lines supplied **from
outside** the shim in the probe process rather than committed:

```
arch_sweep.run_one("fnet")
  {"model_type": "fnet", "side": "shim", "arch_class": "FNetModel",
   "stage": "forward", "status": "ok", "inputs": "main_only"}
```

and with the same two lines supplied the same way, the **whole `FNetModel`**
forward compared element-wise against upstream from the same seed — which the
RNG parity work in this repository makes a fair comparison, since both sides
initialise identically:

```
FNetModel(vocab=64, hidden=32, layers=2, intermediate=64, positions=32)
input  torch.arange(8).reshape(1, 8) % 64

  shim / upstream  shape (1, 8, 32)
  element-wise over all 256 outputs:  max abs diff  7.153e-07
```

So the claim is narrow and checkable: **every operator `fnet` needs now
computes and matches upstream element-wise through a full two-layer forward;
the two missing things are a function binding and a property**, neither of
which is in `aten.rs`.

### 6.2 The two lines, for whoever owns `bootstrap.py`

Verified working in the probe above.

```python
# torch._C._fft.fft_fftn -- fnet's wall (docs/ARCH200.md, docs/BIND3.md §6).
# The decomposition upstream's own TorchDispatchMode records, in the same
# shape as _linalg.linalg_vector_norm above: a Python-level spelling over
# aten ops this shim implements.
_NORM = {None: 0, "backward": 0, "ortho": 1, "forward": 2}

def fft_fftn(self, s=None, dim=None, norm=None, *, out=None):
    rank = self.dim()
    if dim is None:
        dim = list(range(rank)) if s is None else list(range(rank - len(s), rank))
    dim = [d + rank if d < 0 else d for d in dim]
    c = dispatch("aten._to_copy.default", self,
                 dtype=module.complex128 if self.dtype == module.float64
                 else module.complex64)
    if s is not None:
        for d, want in zip(dim, s):
            have = c.shape[d]
            if want < have:
                c = dispatch("aten.slice.Tensor", c, d, 0, want)
            elif want > have:
                pad = [0] * (2 * (c.dim() - d))
                pad[-1] = want - have
                c = dispatch("aten.constant_pad_nd.default", c, pad)
    return dispatch("aten._fft_c2c.default", c, dim, _NORM[norm], True)

module._fft.fft_fftn = fft_fftn

# Tensor.real / Tensor.imag -- properties, so `methods.json` cannot carry them.
# Both aten keys exist and are proven in test_complex.py.
tensorbase.real = property(lambda self: dispatch("aten.real.default", self))
tensorbase.imag = property(lambda self: dispatch("aten.imag.default", self))
```

Two things a reader should not have to rediscover:

* the `s=` branch is the *only* reason `slice.Tensor` and `constant_pad_nd`
  had to be taught; without `s=` neither is on the path. `docs/BIND3.md` §6
  measured that and it held.
* `norm` maps to `_fft_c2c`'s integer `normalization` as
  `{backward: 0, ortho: 1, forward: 2}`, read off upstream's dispatch trace in
  `docs/BIND3.md` §6, not off the C++.

### 6.3 `aten.fft_fftn.default` was deliberately not added as a kernel

It would have been a second implementation of a decomposition upstream already
has, it still would not have reached `torch.fft.fftn` (which does not go
through `torch.ops.aten`), and — the deciding reason — a Rust `fft_fftn` reads
its input back to the host through `_fft_c2c`, which would make it a
`MPS_HOST_READBACK_OPS` case in `device.rs`. `docs/VOICE3.md` established that
that derivation follows helpers **one level by name**, so a two-level chain
through the dispatcher is invisible to it: the op would go on reading device
bytes on the CPU with nothing saying so, in a file this round could not edit to
say it.

`test_tail2.py::test_only_the_expected_complex_ops_landed` asserts no
`fft_fftn` key exists. **That assertion is still true and was left alone** —
inverting it would have been the lie its own docstring warns about.

---

## 7. Device classification

**Nothing new belongs in `device.rs`.** All five kernels are candle shape calls
on two tensors — `narrow`, `index_select`, `reshape`, `cat`, `to_dtype`,
`to_device`, `affine`, `broadcast_add` — and not one of them calls `to_vec*` or
otherwise moves data to the host. The only kernels in `aten.rs` that do are the
FFT ones, and `docs/FFT.md` §7 classified those.

Listed explicitly because `docs/VOICE3.md` found `var`/`std` nearly invisible
to that derivation, and "I checked and there are none" is the answer that has
to be *written down* rather than inferred from the absence of a row.

---

## 8. What changed, split by kind

`docs/COMPLEX2.md` §5.3's lesson — "test count is not progress" — so:

| kind | what |
|---|---|
| **functionality added** | 5 ops: `_to_copy` (real→complex and complex→complex), `slice.Tensor`, `constant_pad_nd`, `view`/`_unsafe_view`, `complex` |
| **architectures moved** | **1** — `llama4`'s full vision tower, measured directly and element-wise (§5). `fnet` **not** claimed (§6) |
| **defects fixed** | none. Nothing found wrong in existing code this round |
| **tests added** | `pytests/test_cplx2.py`, 10 tests. Nine of the ten are element-wise comparisons against a live upstream; one is the refusal sweep. **Four of them were shown red by nullification** (§2.3) |
| **tests inverted** | 1 — `slice` out of `test_complex.py`'s untaught-op sweep, with the coverage moved rather than dropped (§3.1) |
| **documentation** | this file. `docs/BIND3.md` §5 and §6 are now partly superseded and say so from here rather than being edited, since they are a record of what that round measured |
| **deleted** | nothing |

### 8.1 Not done, and why

* **`fnet`** — two `bootstrap.py` lines, written out and verified in §6.2.
* **`torch.zeros(..., dtype=complex64)`** — no measured caller, and a factory
  would be a fourth entrance to the representation (§1).
* **`reshape`, `select`, `cat`, `stack`, `transpose`, `permute`,
  `index_select` on complex** — untaught on purpose. Each is one more arm of
  the same shape, and none is on either architecture's path; they are in
  §3's refusal sweep so that adding one has to come with its own proof.
* **`tensor.rs::COMPLEX_OPS`** — the list `no_real_storage`'s refusal points a
  reader at does **not** yet name these five, because `tensor.rs` was not this
  round's file. `test_complex.py` only checks that the list is sorted, unique
  and reachable, so nothing fails — but the list is now *incomplete*, which is
  the "refusal that names a stale list" failure that file's own docstring
  warns about. The five names to append, in sorted position, are
  `aten._to_copy.default`, `aten._unsafe_view.default`,
  `aten.complex.default`, `aten.constant_pad_nd.default`,
  `aten.slice.Tensor`, `aten.view.default`.
