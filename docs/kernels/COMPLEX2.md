# Complex tensors: the representation, built

Round date 2026-09-06. Branch `work/complex`, on develop `8a60d2d4`. Host Apple
M1, CPython 3.13, upstream torch 2.13.0, `candle-core` 0.11.0.

`docs/kernels/COMPLEX.md` is the sizing document and it is the specification for this
one. It settled the hard question — **do not fork candle** — and deliberately
left the buildable half unstarted, because `Repr`, the `tag` field and every
tagging constructor live in `tensor.rs`, which that round did not own. This
round owned it.

Its §3.3 gave a five-step list. Steps 1 through 4 are done. Step 5 (`fft_fftn`,
for `fnet`) is untouched and remains a separate decision.

The assertions behind everything here live in
`rust/torch_c/pytests/test_complex.py` (11 tests, all element-wise against a
live upstream in a separate process) and in the two tests of
`pytests/test_tail2.py` that this round inverted.

---

## 0. Answers first

| question | answer | where |
|---|---|---|
| Can a complex tensor be held? | **Yes.** `Repr::Complex { re, im }`, a fifth arm holding two real candle tensors. | §1 |
| Does the imaginary part survive a round trip? | **Yes, checked element-wise against upstream**, not by shape. `view_as_complex` → `view_as_real` reproduces `[[1,2],[3,4],[5,6]]` exactly, and `torch.imag` gives `[2,4,6]`. | §3 |
| Was candle forked? | **No, and no `[patch]` was added.** COMPLEX.md §2.2's verdict was re-checked, not assumed: `WithDType` still requires `PartialOrd` and `VecOps` still requires `min`/`max` in 0.11.0. | §1.1 |
| How many `match Repr` sites needed a decision? | **19** — 16 in `tensor.rs`, 2 in `aten.rs`, 1 in `capture.rs`. **14 of them the compiler demanded**; the other 5 have a `_` fallthrough and were checked by hand rather than assumed. | §2 |
| What does nullifying the central one put back? | Changing `tensor()`'s complex arm to `Ok(re)` makes **6 of 10** sampled untaught ops compute silently — `sum`, `tolist`, `reshape`, `slice`, `select`, `to(float32)`. All plausible, all wrong. | §2.3 |
| Does `llama4` construct? | **Yes**, and it now completes a forward pass. So does `llama4_text`. | §5 |
| Is `complex64` storable by candle now? | **No, and it must never be.** `torch.complex64._has_storage` is still `False`; `torch.zeros(2, dtype=torch.complex64)` still refuses by name. | §4 |
| Anything found that was already wrong? | **Two, both in `dtype.rs`.** `complex32.itemsize` was 2 where upstream is 4, and `bfloat16.to_complex()` returned `bfloat16` where upstream returns `complex64`. | §7 |
| Ops taught the arm | **12**, listed by `torch._C._complex_ops()`. Six were predicted; six were found by running the sweep. | §5 |

---

## 1. The representation

```rust
pub enum Repr {
    Dense(Tensor),
    Meta { shape: Vec<usize> },
    Quantized(Arc<QTensor>),
    Vulkan(crate::vulkan::VkTensor),
    Complex { re: Tensor, im: Tensor },   // this round
}
```

A fifth arm, for the same structural reason as the third and the fourth:
**candle cannot hold the thing and the shim can.** `Repr::Quantized` exists
because candle's quantisation is not a `DType`; `Repr::Vulkan` exists because
`candle_core::Device` has no Vulkan variant; this exists because
`candle_core::DType` has no complex variant and cannot get one cheaply.

### 1.1 The candle verdict, re-checked rather than inherited

COMPLEX.md §2.2's argument was verified against the pinned source rather than
taken on trust, because a document that decides *not* to do something is
exactly the kind that goes stale silently:

```
candle-core-0.11.0/src/dtype.rs:146   pub trait WithDType: ... + std::cmp::PartialOrd + ... + VecOps
candle-core-0.11.0/src/cpu/kernels.rs pub trait VecOps { fn min(..); fn max(..); }
```

Both bounds are still there. `num_complex::Complex<f32>` satisfies `NumAssign`
and cannot satisfy either of those, so the change is still a refactor of
candle's core numeric trait across three backends rather than one more enum
arm. **No `[patch]` was added and none should be.**

<!-- DOCWATCH: symbol-in-file rust/torch_c/Cargo.toml '[patch' absent -->

### 1.2 A pair, not interleaving — and the shape site that proves it

COMPLEX.md §3.2 chose the pair over upstream's interleaved trailing-2 because
`Repr::Dense`'s shape *is* candle's shape, so interleaving would report a
trailing `2` that then has to be hidden at every shape-reporting site.

That is not an abstract argument, and `dims()` is where it cashes out:

```rust
Repr::Complex { re, .. } => re.dims(),
```

`re.shape()` **is** the complex tensor's shape. There is no dimension to hide,
and this is the one site that would have had to lie about it. Measured against
upstream: `torch.view_as_complex(torch.ones(3,2)).shape` is `(3,)` and `.numel()`
is `3`, both agreeing.

`element_size()` is the mirror: it answers from the *tag*, so `complex64` is 8
bytes rather than 4, and `numel() * element_size()` sizes the two buffers
together and correctly.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs no_real_storage present -->

### 1.3 One entrance

`PyTensorBase::complex(re, im)` is the only way to attach a complex tag,
mirroring `boolean()`'s role for `torch.bool`. It establishes the four
invariants every consumer of the arm relies on — same dims, same candle dtype,
same device, and a dtype with a complex partner — and it derives the tag from
the component dtype (`TorchDType::complex_for_component`) rather than accepting
one from the caller, so "the tag agrees with the storage" is true by
construction.

---

## 2. The refusal work, which is most of the work

COMPLEX.md §3.4 said it plainly and it was right: **the majority of the cost is
the refusal work, not the arithmetic.** The arithmetic is nine lines; the arms
are nineteen decisions.

### 2.1 The one that carries everything

```rust
pub fn tensor(&self) -> PyResult<&Tensor> {
    match &self.inner {
        Repr::Dense(tensor) => Ok(tensor),
        ...
        Repr::Complex { .. } => Err(no_real_storage(self.tag)),
    }
}
```

`tensor()` is how every kernel in `aten.rs` reads its inputs — ~400 call sites.
Returning `re` here would compile, would type-check at every one of them, and
would make each of those kernels compute the real part of a complex expression
and return a plausible number. So it refuses, and **a kernel added tomorrow
without being told about complex is safe by inheriting the type**, exactly as it
is for `Meta`, `Quantized` and `Vulkan`.

The message names the dtype and says *which half would have been lost*, rather
than "no storage" — because the reader's next question is whether their gap is
the dtype or the operator, and `docs/kernels/COMPLEX.md` §4 established that naming the
dtype is what lets them tell.

### 2.2 The nineteen sites

| file | site | arm |
|---|---|---|
| `tensor.rs` | `tensor()` | **refuse**, §2.1 |
| | `qtensor()` | refuse, naming `"complex"` |
| | `dims()` | `re.dims()` — the true shape |
| | `elem_count()` | `re.elem_count()` — complex elements, not floats |
| | `device_label()` | `re`'s device (both halves agree by construction) |
| | `element_size()` | the tag's width: 8 for `complex64` |
| | `is_contiguous()` | `re && im`, read rather than assumed |
| | `is_nested` / `is_sparse` / `is_quantized` / `_is_zerotensor` / `is_neg` | `false` ×5 |
| | `_layout_name()` | `"strided"` (upstream agrees, measured) |
| | `has_storage()` | `true` — it has two buffers |
| `aten.rs` | `Where::of` | `Where::Dense(re.device())` |
| | device-agreement pair match | agrees with a dense argument on the same device |
| `tensor.rs` | `vk_tensor()` | `_` fallthrough — refuses, naming the device |
| | `qscheme()` | `_` fallthrough — refuses, as upstream does on a dense tensor |
| `aten.rs` | device-agreement `_ => false` | unreachable now that the pair arm above is explicit; left as the backstop |
| `capture.rs` | `storage_key` | already `_ => None`, and that is the right answer: there is no single buffer to version-stamp, and nothing can write into one in place |

**Fourteen of the nineteen were demanded by the compiler**, because those
matches are exhaustive rather than `_ => false`. The other five were read by
hand; each already answers correctly for a fifth arm, and saying so is the price
of not having been asked. That is the property
`tensor.rs`'s own docstring promised would pay off — "an arm cannot be added to
`Repr` without the compiler asking what these six answer for it" — and this is
the second time it has been collected (`Repr::Quantized` was the first).

### 2.3 Nullified, and it goes red

A verification that cannot fail is not a verification, so the central arm was
nullified and the suite re-run.

**`tensor()`'s complex arm → `Ok(re)`:** six of the ten ops sampled in
`test_complex.py::test_every_untaught_op_refuses_...` stop raising and start
returning values —

```
reshape  select  slice  sum  to_float32  tolist
```

— and the test names all six. Every one of those returns something plausible:
`sum` is numerically wrong with no shape to give it away, `reshape` and `slice`
are right in shape and silently half the data. That is the failure this
representation exists to make unrepresentable, and it is one line away.

**`view_as_real` stacking `[re, re]` instead of `[re, im]`:** four tests in
`test_complex.py` and one in `test_tail2.py` go red, naming the element and the
two values (`shim 1.0 vs upstream 2.0`).

Both nullifications were built and run, not reasoned about.

---

## 3. The bar: the imaginary part survives

`pytests/test_complex.py` runs one probe script under **two interpreters** — the
vendored shim on `PYTHONPATH`, and upstream torch with the environment stripped
— and compares element-wise. Each side asserts its own marker, so a mis-wired
environment fails loudly instead of comparing something against itself.

```
view_as_complex([[1,2],[3,4],[5,6]])   shape (3,)  dtype complex64  element_size 8
  torch.real                            [1, 3, 5]        == upstream
  torch.imag                            [2, 4, 6]        == upstream
  view_as_real (round trip)             [[1,2],[3,4],[5,6]]  == upstream
polar / z*w / z*2.5                     both components  == upstream
llama4 rope, B,S,H,D = 2,3,4,8          192 elements     == upstream
copy_ / detach                          both components  == upstream
```

Everything complex is reported through `torch.view_as_real(...)`, which is the
only spelling that shows both halves — reading `.real` alone is precisely the
read that cannot see the bug.

**Tolerance is `1e-6`, and the reason is measured**: candle's `cos` returns
`0.99999994` for `cos(0.0)` where upstream returns exactly `1.0`, so `polar`
differs by ~8 ulp for reasons that have nothing to do with the representation.
A dropped imaginary part moves a value by its whole magnitude, not by an ulp, so
the tolerance is nowhere near wide enough to hide one — which is the property
that matters, and is why the round trip additionally asserts
`imag == [2.0, 4.0, 6.0]` as a literal.

The four upstream refusal messages are compared against a **live** upstream
process rather than against COMPLEX.md's transcription of them, so they cannot
drift:

```
Tensor must have a last dimension of size 2
view_as_complex is only supported for half, float and double tensors, but got a tensor of scalar type: Long
view_as_real is only supported for complex tensors
Expected object of scalar type Float but got scalar type Double for second argument
```

---

## 4. What did *not* change, deliberately

**`torch.complex64._has_storage` is still `False`.** That flag means "candle can
store this dtype", and candle still cannot — the pair lives outside candle's
`DType` entirely, exactly as `Repr::Quantized` does. So:

```python
torch.zeros(2, dtype=torch.complex64)     # still NotImplementedError, naming complex64
torch.empty(2, dtype=torch.complex64)     # still NotImplementedError, naming complex64
```

`docs/kernels/COMPLEX.md` §4 called the existing refusal "the thing to protect", and it
is intact: `test_tail2.py::test_complex_tags_report_no_storage` and
`test_constructing_a_complex_tensor_refuses_by_name` are unchanged and green.

Complex tensors enter through `view_as_complex` and `polar` only. That is not a
limitation to be worked around later — it is what makes
`PyTensorBase::complex` the single entrance.

---

## 5. The twelve ops, and how six of them were found

`torch._C._complex_ops()` is a constant in `tensor.rs`, exported, and checked
against the dispatch table by `test_complex.py` — because a refusal that names a
stale list is worse than one that names none, and this repository has been
bitten by exactly that (`overloads.json`'s note on `min.other`).

**COMPLEX.md predicted six.** They were right:

| op | on `Repr::Complex { re, im }` |
|---|---|
| `view_as_complex(x)` | `re = x[..., 0].copy()`, `im = x[..., 1].copy()` |
| `view_as_real(z)` | `stack([re, im], -1)` |
| `polar(abs, angle)` | `abs*cos(angle)`, `abs*sin(angle)` |
| `mul.Tensor` | `(ac − bd, ad + bc)`, plus the mixed complex×real case |
| `real` / `imag` | field access |

**Six more were found by running `arch_sweep.py --only llama4` and reading the
wall, one at a time.** None of them is arithmetic, and none was predictable from
reading the model source:

| op | why `llama4` needs it |
|---|---|
| `detach` (+ `alias`, `clone`, `contiguous`, `lift_fresh`) | `nn.Buffer(...)` → `torch/nn/parameter.py:270` → `data.detach()`. This is reached *before* anything can look at the tensor. |
| `copy_` | `transformers/initialization.py:169` — `_init_weights` refills the buffer |
| `mul.Scalar` | `freqs_cis * self.attention_scaling` in `llama4_text`'s rope forward |
| `unsqueeze` | `freqs_cis[:, :, None, :]`; `bootstrap.py`'s `__getitem__` turns a `None` index into exactly one `aten.unsqueeze.default` and skips the three full slices without dispatching |

That is the round's own §5.5 lesson in miniature: the six predicted ops were the
ones anyone would predict, and the model stopped on six others. **Each wall was
found by running the thing, not by reading it.**

`slice`, `select`, `index`, `cat`, `reshape` and `to` are deliberately **not**
taught, even though the same one-line "do it to both halves" would work for each.
Every op added is surface that has to be compared against upstream, and no
measured caller reaches them on a complex tensor; they refuse at `tensor()`
until one does.

### 5.1 Where the twelve live, and why golden did not move

| op | list | why |
|---|---|---|
| `view_as_complex`, `view_as_real`, `polar`, `real`, `imag` | `IMPLEMENTED_AWAITING_GOLDEN` | Golden compares a shim result against an upstream one by reading both as real tensors. Four of these five accept or return a `complex64` tensor, so there is no dense storage to read *on either side* — a case builder would have to compare something other than the thing the op produced. They are proven element-wise in `test_complex.py` instead, exactly as `aten.reshape_as.default` is proven in `test_indexsel.py`. |
| `mul.Tensor`, `mul.Scalar`, `copy_`, `detach`, `alias`, `clone`, `contiguous`, `lift_fresh` | `IMPLEMENTED` (unchanged) | Their dense paths are untouched and stay golden-compared. The complex branch is a match guard that is `false` for every real tensor. |

**`ops covered=255` and `9691/9691` are exactly unmoved**, which is the check
that the guard really is inert on the dense path.

---

## 6. The narrowings, stated rather than discovered

### 6.1 `view_as_complex` copies where upstream aliases

Upstream's is a view:

```python
base = torch.tensor([[1., 2.]]); v = torch.view_as_complex(base)
base[0, 0] = 99.;  torch.view_as_real(v)[0][0]      # 99.0 upstream, 1.0 here
```

A pair-of-tensors representation cannot alias an interleaved buffer, and
choosing the pair is what bought the correct `.shape` (§1.2). `llama4` does not
depend on the aliasing — all three of its call sites feed a freshly computed
expression that is never written to again — so the narrowing is safe for the
models measured.

It is asserted as a **divergence** in two places
(`test_complex.py::test_view_as_complex_copies_where_upstream_aliases`,
`test_tail2.py::test_view_as_complex_copies_rather_than_aliasing`): both sides
are required to disagree, so the note fails if *either* side changes.

**And it was nearly worse than a narrowing.** The first implementation used
`.contiguous()` on the two narrowed halves — and candle's `contiguous()` returns
`self.clone()` when the layout is already contiguous. A narrow to length 1 on
the last axis *is* contiguous, so for `[[1., 2.]]` the halves still shared the
base's storage and `base[0,0] = 99.` showed through, while for other shapes the
same code copied. **An aliasing rule that holds for some shapes and not others
is worse than either answer**, and the narrowing test is what caught it — it was
red on the first run for exactly that reason. The fix is candle's `copy()`,
which allocates unconditionally, and it makes "a complex tensor in this shim
never shares storage with anything" true by construction.

### 6.2 `copy_` replaces rather than writes through

`write_into` is the write primitive for a tensor that may be a view, and it
begins with `self.tensor()?` — which refuses on the complex arm, correctly:
there is no single buffer a complex tensor's layout addresses. So the complex
`copy_` uses `replace_with`.

That is only safe because of §6.1: since no complex tensor shares storage with
anything, no view exists that could observe the difference. The two narrowings
are one narrowing, and if §6.1 is ever removed this one has to be revisited in
the same change.

### 6.3 Real ↔ complex casts are not implemented

`dst.copy_(src)` refuses when exactly one side is complex, naming which was
which. Upstream will cast a real source into a complex destination (imaginary
part zero); doing that here would be a *second constructor* for the
representation, and there is exactly one by design.

---

## 7. Two defects found in `dtype.rs`, and one divergence left standing

`docs/kernels/COMPLEX.md` §1 reported the tag half as "already done… with correct
itemsize (2/8/16)". Checked against upstream rather than against that sentence:

| | this tree, before | upstream 2.13.0 |
|---|---|---|
| `torch.complex32.itemsize` | **2** | **4** |
| `torch.bfloat16.to_complex()` | **`torch.bfloat16`** | **`torch.complex64`** |

Both are fixed in this round. The first is the silent-wrong-number shape:
`numel * element_size` would have sized a `complex32` buffer at half its bytes,
and nothing caught it because the only assertion in the tree was about
`complex64` (`test_tail2.py` asserts `element_size == 2 * float32.itemsize ==
complex64.itemsize`, which says nothing about `complex32`). Both now have
assertions.

**Left standing, and recorded rather than fixed:** upstream *raises*
`RuntimeError` for `to_complex()` on an integral, bool or float8 dtype, where
this tree returns the input unchanged. That is a wider change than this round's
subject — it turns a total function partial for thirty tags with no measured
caller — and it belongs to whoever next owns `dtype.rs` with a reason to make
it.

**Also left standing: `print(z)` on a complex tensor refuses.**
`torch/_tensor_str.py:356` calls `self.resolve_conj()`, which is not implemented,
so formatting a complex tensor raises `NotImplementedError: TensorBase
.resolve_conj`. It is an honest refusal rather than a wrong answer, and no
measured caller formats one — but it is a live gap and it had a real cost: it
crashed `test_tail2.py`'s probe script the moment `view_as_complex` started
*succeeding*, because that script called `repr()` on every successful probe. A
probe that cannot survive its own subject succeeding is not a probe, and that
line is now a type-and-shape summary.

---

## 8. `llama4`, measured

```
pytests/arch_sweep.py --only llama4 llama4_text --out ...
```

| | before | after |
|---|---|---|
| `llama4` | stops in **construction** — `Llama4VisionRotaryEmbedding.__init__` | **forward completes** |
| `llama4_text` | stops in forward at `polar` | **forward completes** |

`llama4` was blocked at construction, which is why COMPLEX.md said nothing about
it could be tested until this moved. It now constructs, and the sweep reports
`2/2 forward` for the pair, reproduced across three runs.

One honest note on the path there: an intermediate build (everything except
`unsqueeze`) had `llama4` reaching `torch._C._nn.im2col` in
`F.unfold`, and the final build does not stop there. `unsqueeze` cannot have
fixed `im2col`, so the vision branch is evidently not exercised on every run of
this fixture. **`llama4` constructs and this sweep's forward passes; whether the
vision tower is covered by that forward is not established here**, and
`_nn.im2col` should be treated as a live wall for it (it is `univnet`'s
neighbour in `docs/architectures/ARCH100.md` and is real-valued — nothing to do with complex).

---

## 9. What this round changed

Split as `docs/architectures/ARCH100.md` §5.3 asks:

| class | what |
|---|---|
| **feature added** | `Repr::Complex { re, im }` and its constructor; 12 ops taught the arm (`view_as_complex`, `view_as_real`, `polar`, `real`, `imag`, `mul.Tensor`, `mul.Scalar`, `copy_`, `detach`, `alias`/`clone`/`contiguous`/`lift_fresh`, `unsqueeze`); `torch._C._complex_ops()`; five `overloads.json` entries |
| **defect fixed** | `complex32.itemsize` 2 → 4; `bfloat16.to_complex()` identity → `complex64`; `view_as_complex` aliasing its base for shapes where the narrow stayed contiguous |
| **tests added** | `pytests/test_complex.py`, 11 tests, every positive one element-wise against a live upstream |
| **tests inverted** | 2 in `pytests/test_tail2.py` — the two that asserted these operators *absent*, which is what that file's docstring asks an implementing round to do. Neither was deleted. |
| **documentation** | this file; the `Repr::Complex` and `complex_ops` doc comments; `overloads.json`'s `polar` note |
| **deleted** | nothing |
| **architectures moved** | `llama4` blocked → forward; `llama4_text` blocked → forward |

Not done, and each is a decision rather than an omission: `fft_fftn`
(COMPLEX.md §3.3 step 5, `fnet` alone), `torch.stft` (gated on
`_nn.pad(mode='reflect')` first — COMPLEX.md §5 is unchanged), general complex
indexing, complex `abs`/`conj`/`sum`, real↔complex casts, and printing a complex
tensor.

### Gates

```
suite            599 ok  (baseline 587), EXIT=0
DOCWATCH         PASS -- 562/562
golden           9691/9691 cases passed, 0 failed, ops covered=255, pending 0
cross            aarch64-linux-android      links (cargo ndk -t arm64-v8a --platform 21)
                 aarch64-apple-ios-sim      links (PYO3_CONFIG_FILE for the simulator)
```

`ops covered=255` and `9691/9691` are **exactly** the pre-round numbers. That is
the check that adding a `Repr` arm changed nothing for tensors that are not
complex.
