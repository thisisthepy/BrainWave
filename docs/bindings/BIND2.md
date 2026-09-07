# Four bindings that were "one binding away" from someone else's file

Four rounds each stopped with a wall inside `rust/torch_c/src/bootstrap.py`,
which was another agent's file at the time and stayed unowned until this
round. This is that: three of the four are bindings this round landed there,
proven against upstream element-wise, in a separate process, through the
spelling a user writes. The fourth needs a kernel and is recorded, not
implemented, per the territory this round was given.

Per docs/bindings/BINDINGS.md's warning, **every kernel below was confirmed present in
`_aten_implemented()` before its binding was written** -- the `mish` miss was
two lines onto nothing, and that is not repeated here.

---

## 1. `torch._C._nn.avg_pool2d` -- `efficientnet`

**Kernel: present.** `aten.avg_pool2d.default` is dispatched in
`rust/torch_c/src/aten.rs` (`avg_pool2d_default`, confirmed both by grep and
by `"aten.avg_pool2d.default" in _C._aten_implemented()` at runtime) and has
been golden-compared since `sew_d` -- `docs/kernels/TAIL3.md` §7 already established
this; this round only re-confirmed it before writing the binding.

**What was missing:** the `_install_nn` entry itself. `torch/nn/modules/
pooling.py:779` (`nn.AvgPool2d.forward`) calls `F.avg_pool2d`, which binds
straight to `torch._C._nn.avg_pool2d` with the **leaf** schema's own seven
arguments (`input, kernel_size, stride=None, padding=0, ceil_mode=False,
count_include_pad=True, divisor_override=None`) -- measured via
`torch._C._nn.avg_pool2d.__doc__` on 2.13.0, and unlike `upsample_bilinear2d`/
`upsample_bicubic2d` a few lines above it in the same file, there is no
separate `.vec` overload to choose between (the arity trap docs/bindings/BINDINGS.md
records for `upsample_nearest2d` does not recur here, but was checked for).
The kernel already treats an omitted/`None` `stride` as "the kernel size"
(measured with a `TorchDispatchMode` logger: `F.avg_pool2d(x, 2)` fires
`aten.avg_pool2d.default(x, [2, 2])` with no third positional argument at
all), so the binding forwards all seven arguments unconditionally rather than
defaulting `stride` itself.

**Proof, through the user spelling:**

```python
F.avg_pool2d(x, 3, 2, 1, ceil_mode=True, count_include_pad=False)
```

against upstream, element-wise, in a separate process: **max abs diff 0.0**
over 75 elements (kernel_size=3, stride=2, padding=1, ceil_mode/
count_include_pad both exercised, since those are exactly the branches
`avg_pool2d_default`'s docstring calls out as easy to get backwards).

---

## 2. `torch.einsum`'s ellipsis branch -- `longt5`

**Kernel: not applicable -- this is a `bootstrap.py` composite, not a leaf.**
`aten::einsum` is `CompositeImplicitAutograd` (docs/kernels/TAIL3.md §7 already
established this for the non-ellipsis path); the composite decomposes into
`unsqueeze`/`permute`/`view`/`bmm`, every one of which already has a kernel
here and is exercised by every other `einsum` call the existing composite
already passes.

**What was missing:** `equation` containing `...` was refused by name
unconditionally. `longt5` (`modeling_longt5.py:662`) passes exactly one form,
`'...qhd,...khd->...hqk'`. The fix expands `...` to fresh, real labels before
the existing contraction machinery ever runs -- drawn from ASCII letters that
appear nowhere else in the equation -- so the composite above needed no
change at all; only the equation text reaching it changed. Each operand's
ellipsis rank is `len(tensor.shape) - (len(term) - 3)`; **operands must
agree on that rank** (this round's own arithmetic, not copied from anywhere),
because upstream's own ellipsis additionally broadcasts mismatched ranks
against each other, and that broadcasting is refused by name here as out of
scope -- `longt5`'s call has no such mismatch, and reproducing the broadcast
rule untested would be matching the *summary* ("ellipsis support") rather
than the measured call, which is docs/bindings/BINDINGS.md's `upsample_nearest2d`
trap from the other direction.

**Proof, through the user spelling:**

```python
torch.einsum('...qhd,...khd->...hqk', a, b)   # a: (2,3,4,8), b: (2,5,4,8)
```

against upstream, element-wise, in a separate process: max abs diff
`4.77e-07` over 120 elements -- float32 rounding between two different
reduction orders (this shim's `bmm`-based decomposition vs. whatever
upstream's own composite picks), well inside golden tolerance, not a
divergence.

---

## 3. The six padding modes -- speech models

**Kernel: present, all six.** `aten.reflection_pad{1,2,3}d.default` and
`aten.replication_pad{1,2,3}d.default` are golden-compared already
(docs/kernels/PAD.md §4); this round re-confirmed all six in `_aten_implemented()`
before touching `pad`'s dispatch.

**What was missing:** exactly the patch docs/kernels/PAD.md §5 wrote out in full,
landed verbatim -- the `if mode != "constant": raise` guard in `_install_nn`'s
`pad` replaced with a `reflect`/`replicate` dispatch by `n = len(pad) // 2`
(not the input's rank: the kernel itself judges rank/pad-count combinations
upstream rejects, and deriving `n` from rank here would accept combinations
upstream refuses). `circular` remains refused by name -- it is a
`new_empty`/`slice`/`copy_` composite upstream, a genuinely different code
path, not one of these six leaves.

**Proof, through the user spelling**, all four rank/mode combinations that
exist as real ops, element-wise, in a separate process:

| call | max abs diff | n |
|---|---|---|
| `F.pad(x2d, [1,1,1,1], mode='reflect')` | 0.0 | 300 |
| `F.pad(x2d, [1,1,1,1], mode='replicate')` | 0.0 | 300 |
| `F.pad(x1d, [1,1], mode='reflect')` | 0.0 | 30 |
| `F.pad(x1d, [1,1], mode='replicate')` | 0.0 | 30 |
| `F.pad(x3d, [1,1,1,1,1,1], mode='reflect')` | 0.0 | 1536 |
| `F.pad(x3d, [1,1,1,1,1,1], mode='replicate')` | 0.0 | 1536 |

**The six `tools/golden/reach_allow.json` entries were deleted** --
`aten.reflection_pad{1,2,3}d.default` and `aten.replication_pad{1,2,3}d
.default` under `shape2_kernel_without_spelling` -- because they now
self-fail the reach suite the moment `F.pad` reaches them, which is the
design docs/kernels/PAD.md §5 and the entries' own `reason` text both call for. Golden
itself (`tools/golden/compare.py`) is unmoved: these six entries gated
*reach*, not the kernel comparison, and no kernel changed.

Per docs/kernels/PAD.md §5's own caveat: this is the step that makes "four speech
models clear `F.pad`" true end-to-end rather than just numerically true --
see §5 below for which architectures actually clear.

---

## 4. `torch.std` -- `gemma3n_text`

**Kernel: absent. This is `aten.rs` work, explicitly out of this round's
territory, and is not implemented here.**

Measured with a `TorchDispatchMode` logger on 2.13.0:

```python
torch.std(x)                          -> aten.std.correction((4,5))
torch.std(x, dim=1, unbiased=False)   -> aten.std.correction((4,5), [1], correction=0)
```

`torch.std` is **not** a composite over `var` the way `avg_pool1d` is a
composite over `avg_pool2d` -- it dispatches straight to `aten::std
.correction`, a leaf. Neither `aten.std.correction` nor `aten.var.correction`
appears anywhere in `rust/torch_c/src/aten.rs`'s dispatch table (grepped for
`"aten.std`, `"aten.var`, `aten::std`, `aten::var` -- zero hits, and zero
hits for `"std"`/`"var"` as op names in `bootstrap.py` too), and `std` has no
`overloads.json` row (confirming docs/kernels/TAIL3.md §8's own note). So per this
round's instruction -- check whether a kernel exists under another name
before assuming the gap is a kernel -- the checked alternatives are:

* **`var` is `std`'s square** (docs/kernels/TAIL3.md's own framing) -- also absent,
  same search, so aliasing to it buys nothing.
* **`linalg_vector_norm`** (`aten::linalg_vector_norm`, present, has an
  `overloads.json` row at line 827) is *not* a substitute: it is an
  unweighted p-norm without the "subtract the mean first" step `std`'s
  formula requires, so composing `std` from it would need a mean-subtract
  and a reduction upstream's own binding does not perform through that op --
  it would be a different numerical path than the one measured above, which
  is exactly the kind of divergence docs/kernels/PAD.md §3's `circular` refusal and
  this file's einsum note both refuse to approximate.

**Sized work item, for whoever owns `aten.rs` next:** one kernel,
`aten::std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=1,
bool keepdim=False) -> Tensor` (Welford or two-pass variance, `sqrt` at the
end; `correction` generalizes old `unbiased` -- `correction=1` is `unbiased`,
`correction=0` is not, verified above), one `overloads.json` row for `std`,
and -- if `var` is wanted too, since `_gaussian_topk`-adjacent code in
`modeling_gemma3n.py` may reach it next -- `aten::var.correction` is the same
shape of kernel minus the final `sqrt`. Not attempted here: `aten.rs` is
`docs/architectures/ARCH100.md`/`docs/kernels/TAIL3.md`'s territory boundary, not this round's, and
this round's rule is that a wrong numeric approximation is worse than a
named refusal.

---

## 5. Architecture sweep, before and after

`PYTHONPATH=<vendored tree> TORCH_USE_RTLD_GLOBAL=1 pytests/arch_sweep.py
--out ... --only efficientnet longt5 gemma3n_text`, run after the
rebuild+reinstall above (`TORCH_C_ARTEFACT` pointed at the freshly built
`lib_C.dylib`, not a stale one -- and `side=shim` printed and checked, since
the script silently reports `side=upstream` and no-ops the trace-dependent
checks if `PYTHONPATH` is not set):

| architecture | before this round | after |
|---|---|---|
| `efficientnet` | `torch._C._nn.avg_pool2d` | **forward passes** |
| `longt5` | `torch.einsum` (ellipsis) | **forward passes** |
| `gemma3n_text` | `torch.std` | **unchanged** -- `torch.std` has no kernel; §4 above (`modeling_gemma3n.py:1745`'s decoder layer is the frame the sweep's traceback names) |

**Two of three clear.** `gemma3n_text` staying on `torch.std` is a result,
not a failure: the item this round could do for it (confirm the wall,
confirm it is a leaf not a composite, and rule out an alias through `var` or
`linalg_vector_norm`) is done, and the remaining step is a kernel in
`aten.rs`, a file explicitly outside this round's territory.

`efficientnet` and `longt5` clearing `F.avg_pool2d`/`einsum` was not a given
just because the golden element-wise proofs in §1 and §2 passed -- a forward
pass exercises many more ops than either isolated call does, and
docs/kernels/TAIL3.md §8 already recorded `gemma3n_text` itself moving from `erfinv`
to `std`, one wall to the next, in the same sweep last round. This round's
sweep output (`/tmp/bind2-sweep-shim.json` at the time of this write-up) is
what settled it, not an assumption that binding the named call was sufficient.
