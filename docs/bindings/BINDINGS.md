# Four binding debts, three paid

A kernel that exists, is golden-compared, and cannot be called from Python is
not a feature. `tools/golden/compare.py` calls `_C._aten_dispatch(key, ...)`
with a key it took from its own case table, so it proves the arithmetic and is
structurally blind to whether anything a user writes arrives there —
`docs/bindings/REACH.md` is the argument and `tools/golden/reach.py` is the check.

Three such kernels had been sitting green and unreachable, each for the same
non-technical reason: the round that wrote the kernel did not own
`bootstrap.py`, and the missing piece was a composite in that file.
`tools/golden/reach_allow.json` recorded two of them by name, with reasons that
said in as many words that the gap was **owed, not deliberate**, and that the
entry should be deleted the moment the binding landed. This round landed them.

| debt | spelling | blocked | state |
|---|---|---|---|
| `aten.linalg_qr.default` | `torch.linalg.qr` / `torch._C._linalg.linalg_qr` | `rwkv` (construction) | **landed** |
| `aten.upsample_nearest2d.default` | `F.interpolate(mode="nearest")` / `torch._C._nn.upsample_nearest2d` | `vilt` | **landed** |
| `aten.linalg_vector_norm.default` | `torch.linalg.norm` / `torch._C._linalg.linalg_norm` | `owlv2`, `owlvit` | **landed** |
| `aten.mish.default` | `torch._C._nn.mish` | F5-TTS | **not payable — §5** |

Split the way `CLAUDE.md` §5.3 asks, so that "four names" is not four of
anything:

| class | what |
|---|---|
| feature added | three `bootstrap.py` composites: `_linalg.linalg_qr`, `_linalg.linalg_norm`, `_nn.upsample_nearest2d` |
| defect fixed | none |
| tests added | `pytests/test_bindings.py`, 13 tests, differential against upstream in a second subprocess |
| tests inverted | two — the pins in `test_tail1.py` and `test_tail2.py` that asserted these gaps *existed* |
| documentation | this file |
| deleted | two `reach_allow.json` entries |
| kernels added | **none** — golden is unmoved by construction |

---

## 1. How each was proved

Not by `_aten_dispatch`. Every proof in `pytests/test_bindings.py` goes through
**the spelling a user writes**, in a subprocess with the vendored tree on
`PYTHONPATH`, and compares element-wise against **upstream torch run in a
separate subprocess** with `PYTHONPATH` stripped. Both subprocesses print their
own marker (`shim` / `upstream`) and the suite asserts it, because a
differential test that ran the same library twice would pass perfectly and mean
nothing.

Comparison is against a library that is installed and can be asked, not against
frozen constants. A constant records what somebody once saw; the claim being
made here is agreement.

### 1.1 `torch.linalg.qr`

`torch/linalg/__init__.py:2823` is `qr = _add_docstr(_linalg.linalg_qr, ...)`,
so `torch.linalg.qr` **is** the binding rather than a wrapper around it. There
is no `torch.linalg_qr` and no `Tensor.linalg_qr` on 2.13.0, so this could not
have been an `overloads.json` row without inventing a door upstream does not
have (`docs/bindings/SPELLINGS.md`).

Compared element-wise against upstream at `float64` on the classic
`[[12,-51,4],…]`, a tall `3x2`, `mode="complete"`, and the identity — plus the
named-tuple access (`.Q` / `.R`, which upstream's binding returns and this one
preserves), the `mode="r"` **one-dimensional** empty `Q`, and both refusals
(`mode="banana"`, an `int64` input) message-for-message.

The identity case is asserted against a literal as well as against upstream:
`dlarfg`'s `xnorm == 0` short circuit is what makes `qr(eye(3))` answer `+I`
rather than `-I`, and `-I` is orthogonal, satisfies `Q @ R == A`, and is wrong
(`docs/kernels/TAIL1.md` §5).

### 1.2 `F.interpolate(mode="nearest")`

`torch/nn/functional.py:5188` calls `torch._C._nn.upsample_nearest2d(input,
output_size, scale_factors)` — **three** arguments, the `.vec` schema. Bilinear's
`.vec` has four; nearest has no `align_corners` at all, so the argument counts
differ and the shape was measured rather than copied from the neighbour:

```text
F.interpolate((1,1,3,5), scale_factor=1.5, mode="nearest")
    -> aten.upsample_nearest2d.default((1,1,3,5), [4, 7], 1.5, 1.5)
F.interpolate((1,1,3,5), size=(7,11), mode="nearest")
    -> aten.upsample_nearest2d.default((1,1,3,5), [7, 11])
torch._C._nn.upsample_nearest2d(x, [4,7], [1.5,1.5])
    -> RuntimeError: Must specify exactly one of output_size and scale_factors
```

That third line is upstream refusing to be given both, which is what
establishes that three positional arguments is `.vec` and not the leaf. The
leaf spelling (`output_size`, `scales_h`, `scales_w`) is four arguments, so a
fourth argument is the only discriminator and the composite uses it as one —
the same sentinel trick `upsample_bicubic2d` uses one argument further along.

**The scale factors are forwarded, not merely used to size the output.**
`1/scale` and `in/out` coincide whenever the product is integral, which is every
case a `scale_factor=2` test produces. On the `3x5 -> 4x7` case above they
happen to agree as well, which is exactly why a test built only from that case
would not have noticed a composite that dropped them — and did not: mutating
the composite to pass `None, None` left every other test in this file green.
`4x7` at factor `1.7` gives `6x11`, where `1/1.7 = 0.588` and `in/out = 0.667`
are different grids, and that case is now in the suite in both shapes.

§7 records what looking for such a case turned up on the kernel side.

### 1.3 `torch.linalg.norm`

`torch/linalg/__init__.py:1353` is `norm = _add_docstr(_linalg.linalg_norm,
...)`, the same shape as `qr`, and again with no `torch.linalg_norm` to hang a
table row on. `docs/kernels/COMPLEX.md` §6 classified it correctly as a **binding** gap:
`aten.linalg_vector_norm.default` already had a kernel.

**`ord` is not a parameter, it is a choice of computation**, so the composite
forwards the ones it can and refuses the rest *by name*:

```text
dim is an int (or a 1-list)     -> vector norm, any numeric ord     FORWARDED
dim is None, ord is None        -> flatten to 1-D, 2-norm           FORWARDED
dim is a 2-tuple                -> matrix norm                      REFUSED
dim is None, ord given, A is 2-D-> matrix norm                       REFUSED
ord is 'fro' or 'nuc'           -> matrix norm                       REFUSED
```

The matrix cases are `linalg_matrix_norm` upstream: `nuc` and `ord=±2` need
singular values, `fro` and `±1`/`±inf` are row/column reductions. None of them
is the flattened vector norm — which is the point, because the flattened vector
norm has the **right shape and the wrong number** and would therefore be
believed.

What the two blocked architectures actually spell, read out of `transformers`
rather than guessed:

```text
modeling_owlv2.py:975    torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
modeling_owlv2.py:1048   torch.linalg.norm(x, dim=-1, keepdim=True)
modeling_owlvit.py:957   (identical)
modeling_owlvit.py:1028  (identical)
```

Both are vector norms over an `int` `dim`, i.e. inside the forwarded branch.
Ten vector cases are compared element-wise against upstream (`ord` ∈ {None, 1,
2, 3, ±inf}, `dim` ∈ {-1, 0, None}, with and without `keepdim`, and with
`dtype=float64`), and the four refusals are asserted to raise **and** to be
cases upstream answers — so the day a matrix-norm kernel lands, the test that
records the divergence turns red rather than staying quietly stale.

---

## 2. The allowlist entries, and why removing them is the design

`reach.py` matches `reach_allow.json` **in both directions**: an unlisted gap
fails, and an entry whose gap has closed fails too. So a landed binding makes
the suite red until its excuse is deleted. That is not a regression; it is the
file refusing to describe a past that no longer exists.

Removed:

* `shape2_kernel_without_spelling["aten.linalg_qr.default"]`
* `shape2_kernel_without_spelling["aten.upsample_nearest2d.default"]`

`aten.alias.default` stays. It is the one entry in that section that is a
design decision rather than a work item — upstream exposes no `torch.alias` and
no `Tensor.alias`, so a spelling would be invented rather than restored.

`linalg_norm` never had an entry, because the gap was on the *name* side of a
kernel that other spellings already reached; it was pinned by a test in
`test_tail2.py` instead.

---

## 3. Two tests were inverted, not deleted

Both were written to fail when the work landed and to say so in their own
failure messages. Deleting them would have thrown away the coverage along with
the pin, so each was turned around to assert the other side:

* `test_tail1.py::test_two_kernels_still_have_no_python_spelling_and_it_is_one_line_each`
  → `test_the_two_submodule_bindings_landed_and_mse_loss_is_the_one_still_open`.
  It probes the vendored tree in a subprocess; it now asserts both names
  compute, and checks the values. **`mse_loss` is still shut** and that
  assertion is unchanged — `docs/training/BACKWARD9.md` §1's hand-spelled criterion
  stands.
* `test_tail2.py::test_linalg_norm_is_a_binding_gap_not_a_kernel_gap`
  → `test_linalg_norm_binding_landed_on_the_kernel_that_was_already_there`.
  Its own message said "if it was installed, COMPLEX.md §6 is done and this
  test should compare it against upstream element-wise instead". It keeps the
  *pairing* it was really about — the name and the kernel it depends on must
  not come apart — and the element-wise comparison lives in `test_bindings.py`
  where upstream is available.

One probe fixture in `test_tail1.py` was also corrected: it built its QR input
with `torch.eye(3) if hasattr(torch, "eye") else …`, and `hasattr` is `True`
for a *raising stub*, so the guard chose the branch that could not run. It is a
literal now. This is the trap `docs/kernels/TAIL1.md` §4 already names in another form.

---

## 4. `rwkv` and `vilt`

* **`vilt` clears `upsample_nearest2d`.**
* **`rwkv` gets past `torch.linalg.qr` and stops one line later.**
  `nn.init.orthogonal_` is `torch/nn/init.py`, and the wall moved from line 709
  (`q, r = torch.linalg.qr(flattened)`) to line 710 (`d = torch.diag(r, 0)`) —
  `torch.diag` has no `overloads.json` row. That is a result, not a failure:
  the debt this round owed is paid and the next one is a different, smaller
  shape (a table row over an existing kernel family, not a binding).

`test_bindings.py` asserts this two-sidedly rather than as an `xfail`: if
`orthogonal_` starts working, the orthonormality of its answer is checked; if
it fails for a reason that names `linalg` or `qr`, the test goes red. Only
"stops at `torch.diag`" passes.

---

## 5. `mish` was the fourth debt and it could not be paid

`docs/architectures/VOICE.md` §4.4 records that `aten.mish.default` was implemented, matched
to upstream within one ULP across `float64`/`float32`/`float16`/`bfloat16`
including the saturating tail, and then **removed**, because the only spelling
is `torch._C._nn.mish` and `bootstrap.py` belonged to another round.

The kernel is not in the tree. `mish` appears nowhere in `rust/torch_c/src`, it
is not in `_aten_implemented()`, and it has no golden cases. So the two lines
beside `silu`'s would install a door onto nothing: `_nn.mish` would dispatch
`aten.mish.default` and raise `aten op not implemented in torch._C shim` — a
strictly worse failure than the raising stub that is there now, because the
stub names itself and the dispatch failure names the dispatcher.

Landing it needs the kernel back, which is an `aten.rs` change, which is a
`golden` change (`ops` would go 255 → 256), and this round adds no kernel by
construction. **It is still the cheapest item on the list**, and it now needs
one round rather than two: whoever restores the kernel body from
`docs/architectures/VOICE.md`'s git history can land the binding in the same change, because
nothing about `bootstrap.py` blocks it any more — `_install_nn` is where `silu`
lives and the pattern is two lines.

`test_bindings.py::test_mish_has_neither_a_kernel_nor_a_binding` asserts the
**pair**: kernel absent *and* binding absent. Landing either one alone turns it
red, so the two cannot come apart the way they did last time.

---

## 6. What was checked, and what these checks cannot see

* Golden is **exactly unmoved**: no kernel was added, no dispatch arm changed.
  A move would have meant something was changed that was not meant to be.
* The suite, `compare.py --self-test` and the documentation checker all run
  against the artefact built from this tree, with `TORCH_C_ARTEFACT` set —
  without it, `compare.py` reads another checkout's binary and reports a
  plausible green.
* `bootstrap.py` is `include_str!`'d at **compile** time. Editing it and
  re-running without a rebuild tests the old binary. Every result here is from
  a rebuild followed by `vendor/install_shim.sh`, because the vendored-tree
  subprocess tests read the installed shim and not the staged one.

What they cannot see: whether `vilt` and `rwkv` produce *correct outputs*
end-to-end. `arch_sweep.py` answers "does it get further", not "is the answer
right"; the element-wise agreement here is per-op, at the spelling, and stops
there.

---

## 7. One kernel divergence found on the way, not fixed here

Looking for a case where a forwarded scale and a recomputed one differ turned
up a place where **`aten.upsample_nearest2d.default` and upstream disagree**,
and it is in `aten.rs`, not in the binding. Recorded rather than fixed: this
round adds no kernel and changes none, and golden is unmoved by construction.

Upstream's `nearest_idx` (`ATen/native/UpSample.h`) short circuits before it
ever computes a scale:

```text
if (output_size == input_size)      return output_index;
if (output_size == 2 * input_size)  return output_index >> 1;
                                    /* else use the scale */
```

So on an exact 1x or 2x resize upstream **ignores a supplied scale entirely**.
Measured on `3x5 -> 6x10`, where `scales = 1.5`, `3.0`, `0.5` and `None` all
answer the identical tensor. This shim's kernel honours the scale in that
branch, so `torch._C._nn.upsample_nearest2d(x, [6, 10], 1.5, 1.5)` differs.

Why it is not urgent, and why it is worth writing down anyway:

* **No caller reaches it.** `F.interpolate` computes `output_size` from
  `scale_factors` itself, so a 2x output always arrives with `scale = 2` and
  `1/2 == in/out` — the two agree and the short circuit is invisible. It takes
  a hand-written call with an output size that contradicts its own scale.
* It is nevertheless a real difference in an op that is golden-compared, and
  golden did not see it — the case table has no case that supplies a scale
  inconsistent with the output size, which is the only shape that shows it.
  That is a **gap in the golden cases**, not just in the kernel, and it is the
  more useful half of this finding.

`test_bindings.py` deliberately steers around it (its forwarding case is
`4x7 -> 6x11`, neither 1x nor 2x) and says so in a comment, so that nobody
later "fixes" the test by picking a 2x case that cannot distinguish anything.
