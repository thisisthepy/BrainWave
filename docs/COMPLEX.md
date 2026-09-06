# Complex tensors: whether they are representable, and where the wall is

Round date 2026-09-06. Branch `work/tail2`, on develop `eb84708`. Host Apple M1,
CPython 3.13, upstream torch 2.13.0, `candle-core` 0.11.0.

**This is a sizing document, and the answer is negative.** It exists because four
names in `docs/ARCH100.md`'s tail — `view_as_complex` (`llama4`), `polar`
(`llama4_text`), `fft_fftn` (`fnet`), and the `torch.stft` a concurrent speech
round wants — are not four work items. They are one question, and until that
question is settled none of the four can be estimated at all.

`docs/INT8.md` is the model for the shape of this document, and §2 says why the
answer is *not* the same shape as its answer.

The assertions behind every claim here live in
`rust/torch_c/pytests/test_tail2.py`, which is written to go red when any of
this stops being true.

---

## 0. Answers first

| question | answer | where |
|---|---|---|
| Does `TorchDType` have complex variants? | **Yes — `Complex32`, `Complex64`, `Complex128`, plus the `chalf`/`cfloat`/`cdouble` aliases.** They are import-blocking names and have been present since the dtype tag was split from storage. | §1 |
| Does candle? | **No.** 0.11.0 and `main` both enumerate ten-plus real dtypes and not one complex one. Same negative as `I8`. | §2.1 |
| Is it the same size of gap as `torch.int8`? | **No — it is categorically larger.** `I8` was a new arm on an enum. Complex requires *removing* `PartialOrd` from the `WithDType` bound and `min`/`max` from `VecOps`, which every comparison, reduction, sort and clamp kernel in candle is generic over. | §2.2 |
| So is complex representable here at all? | **Yes, but not through candle.** The design is a fourth `Repr` arm holding two real tensors, exactly as `Repr::Quantized` and `Repr::Vulkan` already hold things candle cannot. | §3 |
| Can this round land it? | **No.** `Repr`, the `tag` field and every constructor live in `tensor.rs`, which this round does not own. §3.3 is the exact change list for the round that does. | §3.3 |
| Does the shim silently lose imaginary parts today? | **No, and that is the thing to protect.** Complex tags report `_has_storage == False` and every constructor refuses by name. | §4 |
| What does this mean for `torch.stft`? | **STFT's first wall is not complex.** It is `torch._C._nn.pad(mode='reflect')`, reached before any transform, and `return_complex=False` does not route around it. | §5 |
| `linalg_norm`? | A **binding** gap, not a kernel gap — the kernel exists. Its install site is in `bootstrap.py`, which this round does not own. | §6 |
| `_vmap_increment_nesting`? | **Real vmap, all four architectures.** Not an import path, and a no-op counter would produce wrong masks rather than unblock anything. | §7 |

---

## 1. The tag half is already done

`rust/torch_c/src/dtype.rs` enumerates `Complex32`, `Complex64` and `Complex128`
alongside every other name the vendored tree uses, with the three aliases
(`chalf`, `cfloat`, `cdouble`), the `abbr` entries (`c32`/`c64`/`c128`), correct
`itemsize` (2/8/16), and both directions of the real↔complex mapping:

```rust
fn to_real(&self)    { Complex64 => Float32, ... }
fn to_complex(&self) { Float32 => Complex64, ... }
```

This is `docs/BOOL.md`'s split doing exactly the job it was built for — **`_C`
owns the dtype tag and candle owns the storage**, related by `storage()`, which
returns `None` for every dtype candle cannot hold. The three complex tags are in
that `None` set.

That is not a partial implementation. It is the *whole* half that does not need
storage, and it is load-bearing: `torch/_prims_common`, `torch/_tensor_str.py`
and `torch/utils/_dtype_abbrs.py` build tables over `torch.complex64` while
`import torch` is still running. A shim without these names cannot finish the
import.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/dtype.rs Complex128 present -->

---

## 2. The storage half, and why it is not `docs/INT8.md` again

### 2.1 candle has no complex dtype, in any version

`candle-core` 0.11.0's `src/dtype.rs` declares

    U8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3, F6E2M3, F6E3M2, F4, F8E8M0

`docs/INT8.md` §1.1 already established that 0.11.0 is `max_version` on
crates.io and that `main` at `ddf1b879` (2026-09-04) adds only the MX float
formats. There is no complex variant in either, so — as with `I8` — the cheap
answer of bumping the pin does not exist.

### 2.2 …and unlike `I8`, it cannot be added as one more arm

This is the finding that decides the round, and it is a *trait bound*, not a
missing kernel.

`candle-core/src/dtype.rs:146`:

```rust
pub trait WithDType:
    Sized + Copy
    + num_traits::NumAssign
    + std::cmp::PartialOrd          // <-- here
    + std::fmt::Display
    + ...
    + crate::cpu::kernels::VecOps
```

and `candle-core/src/cpu/kernels.rs:1`:

```rust
pub trait VecOps: num_traits::NumAssign + Copy {
    fn min(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;
```

`num_complex::Complex<f32>` satisfies `NumAssign`. It does **not** satisfy
`PartialOrd`, and it cannot: the complex numbers are not an ordered field.
`min`/`max` are likewise undefined.

So the candle change is not "add an arm and 252 lines of CPU backend", which is
what `I8` measured. It is:

1. **Relax `WithDType`** to drop `PartialOrd`, and split `VecOps` so that
   `min`/`max` live on a narrower ordered sub-trait.
2. **Re-bound every generic that relied on it.** Comparison (`eq`/`ne`/`lt`/
   `le`/`gt`/`ge`), `min`/`max`/`argmin`/`argmax`, `clamp`, the sort and cmp
   kernels, and the reduce paths are all generic over `WithDType` today and all
   become ill-typed for the complex instantiation. That is a change spread over
   candle's 39 494 lines of source and its 27 files that match on `DType`, not
   over 11 files.
3. **Then** write the actual complex arithmetic, which for a useful `fft_fftn`
   means a transform candle does not have in any dtype.

Steps 1 and 2 are a **refactor of candle's core numeric trait**, not an
addition to it. `docs/INT8.md` was able to produce a patch and measure candle's
own suite at 0 failed precisely because `I8` touched nothing structural. A
complex fork would change the type signature of the trait every backend
implements — CPU, CUDA, Metal — and would have to be carried against upstream
forever.

**No patch is offered and none should be landed.** The standing rule holds with
more force here than it did for `I8`: this dependency is pinned and reaches
Android, iOS, wasm and every wheel target, and a fork of a core trait is not
something to carry for four architectures in the tail.

<!-- DOCWATCH: symbol-in-file rust/torch_c/Cargo.toml '[patch' absent -->

---

## 3. What can be done without candle, and it is more than nothing

### 3.1 The precedent already exists, twice

`Repr` in `tensor.rs` is not `Tensor`. It is:

```rust
pub enum Repr {
    Dense(Tensor),
    Meta { shape: Vec<usize> },
    Quantized(...),      // candle's QTensor is not a DType (docs/QUANT.md §5.1)
    Vulkan(...),         // docs/VULKAN2.md
}
```

Both of the non-obvious arms exist for the same reason a complex arm would:
**candle cannot hold the thing, and the shim can.** `Repr::Quantized`'s comment
says it outright — "the reason this is a third arm and not a `Tensor` wearing a
label is that candle's quantisation is not a `DType`". Complex is not a `DType`
either.

### 3.2 Two real tensors, not interleaved storage

The obvious representation is upstream's own: interleave real and imaginary in a
trailing dimension of size 2, which is *exactly* what `view_as_complex` views.
That is the wrong choice here, for one concrete reason: `Repr::Dense`'s shape is
candle's shape, so an interleaved complex tensor would report a trailing `2` in
`.shape` that must then be hidden at every shape-reporting site. Hiding a
dimension in one place and not another is the silent-wrong-answer shape.

    Repr::Complex { re: Tensor, im: Tensor }     // both real, both the true shape

gets the shape right by construction — `re.shape()` *is* the complex tensor's
shape — and makes each operator's obligation explicit rather than implicit.

Measured against upstream, this carries llama4's entire complex surface:

| op | on `Repr::Complex { re, im }` |
|---|---|
| `polar(abs, angle)` | `re = abs*cos(angle)`, `im = abs*sin(angle)` |
| `view_as_complex(x)` | `re = x[..., 0]`, `im = x[..., 1]` |
| `view_as_real(z)` | `stack([re, im], -1)` |
| `z * w` | `(re·wre − im·wim, re·wim + im·wre)` |
| `z.real` / `z.imag` / `conj` / `abs` | field access; `(re, −im)`; `hypot(re, im)` |
| indexing, `None`-unsqueeze, slicing, `to(device)` | applied to both fields |

### 3.3 The one narrowing, stated rather than discovered

**Upstream's `view_as_complex` is a view; this is a copy.** Measured:

```python
base = torch.tensor([[1., 2.]])
v = torch.view_as_complex(base)
base[0, 0] = 99.
v.tolist()            # [(99+2j)]   -- aliases
```

A pair-of-tensors representation loses that. `llama4` does not depend on it —
all three of its call sites feed a freshly computed expression
(`xq.float().reshape(...)`, `torch.stack([cos, sin], -1)`) that is never written
to again — so the narrowing is safe *for the models measured*, and it belongs in
`docs/VIEWS.md`'s ledger rather than in a footnote.

### 3.4 Why this round did not land it

`Repr`, the private `tag` field, and every tagging constructor (`boolean`,
`meta`, `quantized`, `vulkan`) are in **`tensor.rs`**, which this round was
explicitly scoped out of, and which concurrent rounds are editing. A fifth
`Repr` arm is a cross-cutting change: every `match` on `Repr` in `tensor.rs`,
`capture.rs` and `tape.rs` must gain an arm, and each one is a decision about
whether that operation is defined for complex or must refuse.

**That refusal work is the majority of the cost and it is the part that must not
be skipped.** An arm added with a `_ => unreachable!()` or a fallthrough that
operates on `re` alone would return plausible numbers with the imaginary part
dropped — the exact failure `docs/VULKAN2.md` required be made
*unrepresentable* rather than merely unused.

For the round that picks this up, in order:

1. `Repr::Complex { re, im }` + the arm in every existing `match` on `Repr`,
   defaulting to **refuse**, not to `re`.
2. A `PyTensorBase::complex(re, im)` constructor, the only way to attach a
   complex tag — mirroring how `boolean()` is the only way to attach `Bool`.
3. `polar`, `view_as_complex`, `view_as_real`, `mul` in `aten.rs`, plus the four
   upstream error messages transcribed in `test_tail2.py::_UPSTREAM_REFUSALS`.
4. Flip `test_tail2.py`'s absence assertions into element-wise comparisons
   against `_LLAMA4_ROPE_SPEC`, whose checksums are already in the tree.
5. Only then `fft_fftn`, which needs a transform on top of all of the above.

Steps 1–4 unblock `llama4` and `llama4_text`. Step 5 is `fnet` alone and is a
separate decision.

---

## 4. The refusal is correct today, and that is worth asserting

The failure mode this whole area has is that a complex tensor which silently
loses its imaginary part **still returns plausible numbers**. It survives a
smoke test, it survives a shape check, and it is only caught by an element-wise
comparison nobody runs on an architecture that was reported as passing.

Measured, the shim does not have that failure:

```
torch.complex64._has_storage                     False
torch.zeros(2, dtype=torch.complex64)            NotImplementedError:
    aten.zeros.default: dtype not storable by the candle backend in
    torch._C shim: torch.complex64
torch.view_as_complex(...) / torch.polar(...)    NotImplementedError, by name
```

The refusal names the **dtype**, not a near neighbour, which is what lets a
reader tell whether their gap is the dtype or the operator.

`test_tail2.py` asserts `_has_storage is False` directly rather than through any
operator that happens to consult it, because that single fact is what the whole
refusal rests on — and because an implementation is more likely to break it by
accident than deliberately. It also asserts `float32`/`bool` still report
`True`, so a change that made `_has_storage` uniformly `False` is caught rather
than passing the loop.

---

## 5. `torch.stft` — for the concurrent speech round

**The complex answer is not your first blocker.** Measured on the shim:

```python
torch.stft(torch.arange(64).float(), n_fft=16, return_complex=True)
torch.stft(torch.arange(64).float(), n_fft=16, return_complex=False)
```

Both raise, and both raise the *same* error, before any transform runs:

    NotImplementedError: not implemented in torch._C shim:
    torch._C._nn.pad(mode='reflect')

`torch.stft` reflect-pads its input by `n_fft // 2` when `center=True` (the
default), and that happens first. So:

* **`return_complex=False` does not route around this round.** It is the obvious
  next thing to try and it hits the identical wall.
* **`center=False` may.** That skips the pad entirely; the next wall would then
  be the transform itself, which is §3.3 step 5 and is gated on everything
  before it.
* The cheap, useful, independent work is **`_nn.pad(mode='reflect')`**. It is
  real-valued, it has nothing to do with complex, and it is shared with
  `univnet` (`docs/ARCH100.md` lists `_nn.pad` as its wall). A speech round can
  take it today without waiting for any of this.

Treat STFT as gated on: `reflect` pad → `Repr::Complex` → an FFT. Three
independent things, in that order, and only the first is available now.

---

## 6. `linalg_norm` is a binding, and the kernel is already there

> **Closed since.** The install below was made — `docs/BINDINGS.md` §1.3. The vector cases are
> compared element-wise against upstream; the matrix `ord`s (`'fro'`, `'nuc'`, `±2`, a 2-tuple
> `dim`) refuse by name, because they are `linalg_matrix_norm` upstream and the flattened
> vector norm would answer them with the right shape and the wrong number.

`torch._C._linalg.linalg_norm` blocks `owlv2` and `owlvit`
(`docs/ARCH100.md:76`). It is `missing_shim_name`, and the classification is
accurate: `aten.linalg_vector_norm.default` **has a kernel** in `aten.rs` and is
in `_aten_implemented()` (asserted in `test_tail2.py`). `linalg.norm` over a
vector, or with `ord=None`, is `vector_norm`.

The install site is `bootstrap.py:2780`, immediately beside the existing

```python
module._linalg.linalg_vector_norm = _torch_level_function(...)
```

**This round did not make that edit**, because `bootstrap.py` is outside its
territory and is being edited concurrently by the round taking `_nn.glu` — two
rounds appending to the same install function is exactly the merge collision
the suite file was split to avoid. It is a ~15-line addition and belongs to
whichever round owns `bootstrap.py` next; the kernel it needs is already
present, so nothing blocks it but the file.

---

## 7. `_vmap_increment_nesting`: real vmap, in all four

> **Superseded by `docs/VMAP.md` (2026-09-07).** The sizing below is
> correct — they do genuinely vmap, and a no-op counter would have
> produced a wrong mask — but the conclusion that closing it needs "a
> batching rule for every op reachable inside a vmapped closure" was
> too pessimistic for *these* closures. They are pointwise over four
> index scalars, and for that shape vmap is broadcasting. VMAP.md §2 is
> the measurement, §4 is the fence, and §5 sizes the general thing,
> which is still unbuilt.

`nemotron3_5_asr`, `nemotron_asr_streaming`, `nemotron_asr_streaming_encoder`
and `t5gemma2` (`docs/ARCH100.md:71`). The question worth asking was whether
they genuinely vmap or merely touch an import path — because a counter with no
observable effect would have been a four-architecture win for almost nothing.

**They genuinely vmap.** The chain, identical in all four:

```
transformers/masking_utils.py:962   use_vmap = True, but only when the caller
                                    passes and_mask_function / or_mask_function
masking_utils.py:526  -> _vmap_expansion_sdpa(mask_function)
masking_utils.py:348  -> torch.vmap(mask_function, in_dims=..., out_dims=0)
torch/_functorch/apis.py:68 -> vmap_impl
torch/_functorch/vmap.py:487  _vmap_increment_nesting(batch_size, randomness)
                       :490  _vmap_decrement_nesting()
                       :173  _add_batch_dim      <-- the actual work
                       :204  _remove_batch_dim
```

The `use_vmap` flag is off by default — `masking_utils.py:514` is the "fast
non-vmap mask creation" path — and these four turn it on because they supply a
genuine mask closure: chunked-limited attention context
(`modeling_nemotron_asr_streaming.py:876-881`, which does `torch.div(...,
rounding_mode="trunc")`, a subtraction, two comparisons and a `&` per index) and
non-causal sliding-window masking (`modeling_t5gemma2.py:805`).

So the cheap fix is not available, and it is worse than unavailable — it is
actively harmful. A `_vmap_increment_nesting` that returns a level and does
nothing lets the call proceed to `_add_batch_dim`, and whatever comes back is
then used **as an attention mask**. A wrong attention mask does not raise; it
produces a plausible tensor of the right dtype and a model that appears to run.
That is the same failure shape as a dropped imaginary part, which is why
`test_tail2.py` asserts the refusal *deliberately* and asserts that
`_add_batch_dim` and the counter never disagree about whether they work.

**Sized, not built.** Closing this means a batched-tensor level in the shim:
`_add_batch_dim`/`_remove_batch_dim`, the nesting stack, and a batching rule for
every op reachable inside a vmapped closure — for these four that is `div`
(trunc), `sub`, comparison and `bitwise_and`, but the surface is open-ended
because the closure is user-supplied. That is a `tensor.rs`-and-dispatch round
of its own, comparable in scope to `Repr::Complex`, and it should not be started
inside a tail-clearing round.

---

## 8. What this round changed

Split as `docs/ARCH100.md` §5.3 asks, because "four names investigated" is not
four of anything:

| class | what |
|---|---|
| feature added | **none** |
| defect fixed | none |
| tests added | `pytests/test_tail2.py`, 10 tests |
| documentation | this file |
| deleted | nothing |

The honest summary is that this round **converted four unestimated names into
one measured negative, one design with a step list, and two hand-offs**. No
architecture moved from blocked to passing, and the count in `docs/ARCH100.md`
is unchanged at 215/297.

The tests are not decoration and were checked against the standard that a
verification which cannot fail is not a verification: mutating the recorded
`element_size` and the `fft_fftn` message text turns two of the ten red, and the
subprocess probe asserts its own marker (`shim`, never `upstream`) so that it
cannot pass by having tested the wrong library.
