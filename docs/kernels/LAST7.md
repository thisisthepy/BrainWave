# LAST7 — four of the last seven, and only two of them were kernels

Worktree `work/last7` on develop `b33e2ee`. Territory: `rust/torch_c/src/aten.rs`,
`overloads.json`, `methods.json`, `tools/golden/cases.py`,
`rust/torch_c/pytests/test_last7.py`, plus the three inversions §7 lists.

## 0. The alias-versus-kernel split, first

`docs/architectures/ARCH200.md` left 7 of 297 architectures blocked and four came here as
"needs `aten.rs`". **Two of the four needed `aten.rs` and two did not**, and the
two that did needed it for different reasons:

| wall | what it actually was | where the fix lives |
|---|---|---|
| `TensorBase.unfold` (`univnet`) | **a real missing kernel** | `aten.rs` + `methods.json`. Landed. |
| per-axis convolution padding (`nystromformer`) | **not a kernel and not an argument form** — a real candle refusal with a lowering around it | `aten.rs`, inside the existing `convolution` kernel. Landed. |
| `torch.multinomial(Tensor, Tensor)` (`vilt`) | **an argument form** — upstream's *parser*, not its schema | `bootstrap.py`'s `_TypeChecker`. Out of territory here; **landed by `docs/bindings/BIND5.md` in the same batch, and `vilt` forwards.** §5.1 |
| `torch.repeat_interleave(Tensor repeats)` (`fastspeech2_conformer`) | **a real missing kernel**, and the only one of the four that is genuinely data-dependent | `aten.rs` **plus** `bootstrap.py`, `device.rs` and `capture.rs`. **Not landed**, and §5.2 is why landing the kernel alone would have been worse than not landing it. |

**`univnet` and `nystromformer` run a complete forward.** `vilt` and
`fastspeech2_conformer` stop where they stopped, for reasons measured and written
down here rather than left as "not yet gotten to". §6.

The pattern this batch was warned about — *candle's error message names the
wrong thing* — did not recur. Both refusals that fell were this shim's own
messages, and one of them (`"an asymmetric padding"`) **named the wrong thing
anyway**: `docs/bindings/ARGFORM.md` §1 had already measured that `padding=[0, 5]` is not
asymmetric padding at all. §4.

<!-- DOCWATCH: op-implemented aten.unfold.default -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json unfold present -->

---

## 1. What `Tensor.unfold` is, and what it is not

**It is not `nn.Unfold`.** `F.unfold` is `im2col`, which landed in
`docs/bindings/BIND3.md` and is a different kernel in this same file.
`Tensor.unfold(dimension, size, step)` slides a window of `size` along one axis
with a stride of `step` and appends the window as a **new trailing dimension**:

```text
torch.arange(6.).unfold(0, 3, 1)   ->  [[0,1,2], [1,2,3], [2,3,4], [3,4,5]]
                                       shape (4, 3), stride (1, 1)
```

That stride is the whole difficulty. `1 < 3` — consecutive windows overlap in
`size - step` elements — so the second stride is **smaller than the extent it
indexes**, and `docs/kernels/STRIDED.md` §1 already established that no composition of
`narrow`/`transpose`/`reshape` produces such a stride and that candle 0.11.0 has
no public constructor for an arbitrary `Layout` over a shared `Storage`.

`univnet` is the caller. `modeling_univnet.py:312`:

```python
hidden_states = hidden_states.unfold(2, hop_size + 2 * padding, hop_size)
```

`size = hop_size + 2 * padding` and `step = hop_size`, so its windows overlap by
`2 * padding` — this is not a case where `step == size` and the op degenerates
into a reshape.

### 1.1 Why `step == size` cannot test any of this

A step equal to the size is a plain reshape, and a reshape agrees with **four**
different wrong implementations at once: a stride read off the wrong axis, the
window axis inserted next to its source instead of appended, a gather that walks
storage order rather than the receiver's logical order, and an off-by-one in the
window count. Every value case in `tools/golden/cases.py::unfold_cases` and in
`test_last7.py` overlaps its windows for that reason, and one case deliberately
uses `step > size` (windows that *skip*) so that the window-count arithmetic is
exercised in both directions from `step == size`.

---

## 2. The narrowing: it is a copy, stated as a copy

Upstream's `unfold` is a **two-way view**. Measured on real torch 2.13.0 in a
separate process:

```text
y = torch.arange(6.); v = y.unfold(0, 3, 1); v[0, 0] = 99.  ->  y[0] is 99.
z = torch.arange(6.); w = z.unfold(0, 3, 1); z[1]    = -5.  ->  w[0, 1] is -5.
```

This one gathers, so it is a copy, and every write upstream would have
propagated — **in either direction** — is a named refusal here rather than a
wrong number:

| | upstream | here |
|---|---|---|
| values, including overlapping windows, skipping windows, empty windows | — | **identical** |
| the four refusals, including their **order** (§3.2) | — | **identical messages** |
| a non-contiguous receiver | reads the logical order | **identical** (§3.1) |
| write through the window, base read after | propagates | `RuntimeError` |
| write to the base, window read after | propagates | `RuntimeError` |

Both directions are registered in `tools/golden/cases.py` as `expect="c_error"`,
one case each so that closing one cannot hide the other — the same register
`as_strided_cases` uses, and a stronger one than `aten.slice.Tensor` (step > 1)
and `aten.view.dtype`, which are `expect="diverge"` and lose a write *silently*.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs unfold_default present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_last7.py test_writing_through_an_unfold_window_is_refused_rather_than_lost present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_last7.py test_writing_to_the_base_of_a_live_unfold_is_refused_too present -->

---

## 3. The barrier, and the test that had to be inverted for it

`storage.rs::StridedBarrier` is reused unchanged. It is keyed on the **storage
address** and holds a clone of both candle tensors, so the key cannot be reused
while it means anything; `docs/kernels/STRIDED.md` §2 and §3 are the argument and none of
it is re-derived here. `unfold_default` takes the barrier over the receiver and
the result exactly as `as_strided_default` does.

`test_strided.py::test_as_strided_is_the_only_op_that_takes_the_barrier`
asserted `len(calls) == 1` and went red the moment this landed. Its own docstring
said what to do:

> *"If a second op ever needs it, that is a decision worth making explicitly —
> an op that bars its input's storage is an op that can make an unrelated later
> write fail."*

So it is **inverted, not deleted**:
`test_as_strided_and_unfold_are_the_two_ops_that_take_the_barrier` asserts
`len(calls) == 2` and names both producers, so a *third* one still fails there
until somebody writes it down. That is the fifth inversion this repository has
kept rather than dropped.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_strided.py test_as_strided_and_unfold_are_the_two_ops_that_take_the_barrier present -->

### 3.1 Where `unfold` and `as_strided` genuinely differ

`as_strided` refuses a non-contiguous receiver by name (`docs/kernels/STRIDED.md` §4.1)
because upstream's `as_strided` addresses the **raw storage** and ignores the
receiver's layout — gathering the logical order would return upstream's shape
with elements read from the wrong places. **`unfold` has no such refusal, and
that is not a shortcut.** It is defined on the receiver's *logical* index space:

```text
torch.arange(12.).reshape(3, 4).t().unfold(0, 2, 1)
  -> [[[0,1],[4,5],[8,9]], [[1,2],[5,6],[9,10]], [[2,3],[6,7],[10,11]]]
     the TRANSPOSED order, not the storage order
```

So a `.contiguous()` before the gather is *correct* here and would have been
*wrong* there. `test_unfold_reads_the_receivers_logical_order_not_its_storage`
asserts both halves in one test — the values through `unfold`, and that
`as_strided` on the same receiver still refuses — so the two ops cannot drift
into agreeing.

### 3.2 Two things that had to be measured rather than derived

**The refusal order is not the guessable one.** On a length-6 tensor,
`unfold(0, 7, 0)` is wrong twice and upstream reports the **size**:

```text
maximum size for tensor at dimension 0 is 6 but size is 7
```

not the step. The order is `dimension` → `size >= 0` → `size <= extent` →
`step > 0`, and every input where only one argument is wrong is blind to it.

**A zero-dimensional receiver is a special case, not a corollary.**
`torch.tensor(3.).unfold(0, 1, 1)` is `tensor([3.])` — shape `[1]`. The general
rule (replace `dims[dim]` with the window count, then append the size) gives
`[1, 1]`, which is wrong. Upstream's answer for rank 0 is simply `[size]`;
`unfold(0, 0, 1)` on it is shape `[0]` and `unfold(0, 2, 1)` refuses with a
maximum of 1.

---

## 4. The convolution padding, which `docs/architectures/ARCH100.md` mis-named and `docs/bindings/ARGFORM.md` half-corrected

`nystromformer`'s wall was `aten.convolution.default: an asymmetric padding
[32, 0] is not implemented`. `docs/bindings/ARGFORM.md` §1 already found the first error:
**it is not asymmetric padding.** It is two axes each padded *symmetrically* by a
different amount, which upstream computes fine. The caller is
`NystromformerSelfAttention`:

```python
self.conv = nn.Conv2d(heads, heads, kernel_size=(conv_kernel_size, 1),
                      padding=(conv_kernel_size // 2, 0), groups=heads)
```

`docs/bindings/ARGFORM.md` then classified it as *"a genuine backend limitation, not an
argument-form gap"* and stopped. **That was right about candle and wrong about
the conclusion.** candle's `conv2d` really does take one scalar padding, but the
difference between the axes can be spent as explicit zero padding on the input —
the same move `docs/kernels/RNN.md` §2 made for `conv1d(padding='same')` with an odd
total, one rank up:

```text
padding = [a, b]      ->      common = min(a, b)
                              pad the input by (a - common) on each side of H
                                          and (b - common) on each side of W
                              convolve with padding = [common, common]
```

One of the two extras is zero by construction, because the common part is the
minimum. `pad_with_zeros` is candle's `aten::constant_pad_nd` with a value of 0,
one axis at a time.

**Only padding.** A per-axis-differing `stride` or `dilation` has no such
lowering — there is nothing to add to an input that makes an unequal stride
equal — so both still refuse **by name**, which is what keeps "this backend
cannot" separable from "this round did not". The transposed case is left alone
too: padding a transposed convolution's input is not the same operation as
reducing its output-side padding, and nothing measured reaches it.
`test_last7.py` pins all three of those boundaries.

The golden case that asserted the refusal (`asymmetric padding -- c_error, torch
computes`) went red the day the lowering landed and is **inverted into the
stronger form**: five live per-axis-padding rows across two dtypes that diff real
values against upstream, including both orderings of the difference, two
non-square inputs, and `nystromformer`'s own depthwise `(k, 1)`-kernel shape.
Both orderings are there because on a square input and a square kernel the two
possible wrong axes swap into each other's shapes; the non-square rows are what
make the axis choice falsifiable on shape alone.

`test_argform.py::test_asymmetric_conv_padding_is_a_backend_limit_not_an_argument_form`
is inverted to `test_per_axis_conv_padding_now_computes_and_agrees_with_upstream`
(values, against a reference measured on upstream in a separate process) plus
`test_a_per_axis_differing_stride_is_still_refused_by_name` for the half that did
not close, so the two halves of `docs/bindings/ARGFORM.md` §1's finding stay separable.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_argform.py test_per_axis_conv_padding_now_computes_and_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_argform.py test_a_per_axis_differing_stride_is_still_refused_by_name present -->
<!-- DOCWATCH: symbol-in-file tools/golden/cases.py unfold_cases present -->

---

## 5. The two that were not landed, and exactly what they need

### 5.1 `multinomial` — an argument form, and `docs/architectures/VOICE3.md` had it right

> **Closed.** `docs/bindings/BIND5.md` taught `_TypeChecker` upstream's `SymInt` rule in
> the same batch, and `vilt` forwards. The test below was written to fail the
> moment that happened and to name this section; it is inverted rather than
> deleted, and now asserts the **shape** of the rule — one element by
> `numel()`, integral, not `bool`, with `bool` refused at the coercion so
> upstream's two exception classes fall out. BIND5 also found that the same
> rule landed one position over in `docs/bindings/BIND4.md` had used `int()` on any
> tensor, so this build accepted `tensor(3.5)` and `tensor(True)` where
> upstream raises. The refusals are the test.


`modeling_vilt.py:155` is
`torch.multinomial(torch.ones(v).float(), max_image_length)` with
`max_image_length` arriving as a **tensor**. Measured on real torch 2.13.0:

```text
multinomial(p, torch.tensor(3))       ->  works, shape (3,)
multinomial(p, torch.tensor([3]))     ->  works, shape (3,)
multinomial(p, torch.tensor(3.0))     ->  TypeError: 'num_samples' must be int, not Tensor
multinomial(p, torch.tensor([3, 4]))  ->  TypeError: same
```

So upstream's **schema** takes `SymInt num_samples` and upstream's **argument
parser** implicitly converts a single-element *integral* tensor. That is the
general `SymInt` rule, not something about `multinomial` — which means the fix is
`bootstrap.py`'s `_TypeChecker`, out of this round's territory, and `aten.rs`
needs nothing: `aten.multinomial.default` is implemented and golden-compared.

**An `overloads.json` row is not an available workaround.** Every schema string
in both tables is checked against upstream by `pytests/verify_schemas.py`, so a
fabricated `aten::multinomial.num_samples_tensor` would fail that check rather
than route around the type checker. Recorded so the next round does not re-derive
it, and `test_multinomial_with_a_tensor_num_samples_is_an_argument_form_not_a_kernel`
asserts the *refusal*, so the pin inverts the day the rule lands.

`docs/bindings/ARGFORM.md`'s own warning applies to whoever lands it: upstream accepts
this on some argument positions and not others, so the coercion belongs on
`SymInt` positions specifically and its **integral-and-single-element**
precondition is part of the measurement, not a detail.

### 5.2 `repeat_interleave.Tensor` — a real kernel, and three files this round may not touch

`modeling_fastspeech2_conformer.py:123` is
`torch.repeat_interleave(encoded_embedding, target_duration, dim=0)`.
Upstream's lowering, from a `TorchDispatchMode` logger:
`aten::repeat_interleave.Tensor` then `aten::index_select`. The second exists
here; the first does not. Its real schema is

```text
aten::repeat_interleave.Tensor(Tensor repeats, *, SymInt? output_size=None) -> Tensor
```

and it was measured end to end (`[2, 0, 3]` → `[0, 0, 2, 2, 2]`, dtype preserved
through int32, `repeats` must be 1-D, negatives and floats refused, a disagreeing
`output_size` refused). **The kernel was not landed**, because writing it alone
would have been worse than not writing it:

* **`bootstrap.py`** — `torch.repeat_interleave` is a hand-written composite that
  raises on a tensor `repeats` before any dispatch happens, and it is installed
  with `setattr(varfns, ...)` *after* the table, so it wins. A kernel behind it
  would be unreachable through the only spelling `fastspeech2_conformer` uses.
  **The architecture stays blocked either way.**
* **`device.rs`** — the output length is `repeats.sum()`, so the kernel must read
  the repeats back to the host.
  `test_shim.py::test_the_mps_readback_list_is_what_the_kernels_actually_do`
  re-derives `MPS_HOST_READBACK_OPS` from `aten.rs`'s own bodies and goes red
  until the op is declared there.
* **`capture.rs`** — the output *shape* is a function of tensor values, which is
  exactly what `DATA_DEPENDENT_SHAPE` refuses by name for `nonzero`. The brief
  asked whether it must join that list: **it must.** A recorded node whose output
  shape is not a function of its inputs replays unsoundly on any other input, and
  the refusal already exists for the identical reason one line above.

So landing the kernel would have turned two of those three into a red suite and
left the architecture blocked anyway. It is written down instead, and
`test_repeat_interleave_with_a_tensor_repeats_is_still_refused_by_name` pins the
refusal with the three-file list in its docstring so the write-down cannot rot
into "forgotten".

> **Closed.** `docs/kernels/REPEAT.md` landed all four files in one change --
> the kernel in `aten.rs`, the composite's new arm in `bootstrap.py`, the
> readback declaration in `device.rs` and the `DATA_DEPENDENT_SHAPE` entry in
> `capture.rs` -- and `fastspeech2_conformer` forwards. Every item in the
> three-bullet list above held when it was reached: the composite really did
> win over the table, the readback derivation really did demand `device.rs`,
> and `capture.rs` really did have to refuse it. The pin is **inverted**, not
> deleted, into
> `test_repeat_interleave_with_a_tensor_repeats_now_lands_in_all_four_files`,
> which asserts the values *and* the two list memberships -- because a test
> that checked only the values would pass with `device.rs` and `capture.rs`
> untouched, which are precisely the two this section said would go red.
>
> One correction to what is above: this section names *three* out-of-territory
> files, and the count that matters is **four**, because `aten.rs` is the
> fourth and it is the one that cannot be done without the other three.
> `docs/kernels/REPEAT.md` §2 states it that way.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_last7.py test_multinomial_with_a_tensor_num_samples_now_matches_upstreams_symint_rule present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_last7.py test_repeat_interleave_with_a_tensor_repeats_now_lands_in_all_four_files present -->

---

## 6. The four architectures, measured with `arch_sweep.py --one`

```text
                          before                              after
univnet          forward -- TensorBase.unfold         FORWARD RUNS
nystromformer    forward -- per-axis conv padding     FORWARD RUNS
vilt             forward -- multinomial(Tensor,       SAME WALL -- argument form in
                            Tensor)                     bootstrap.py, §5.1
fastspeech2_     forward -- repeat_interleave with    SAME WALL -- kernel + three
  conformer                 a tensor `repeats`          out-of-territory files, §5.2
```

**2 of 4 run a complete forward.** Coverage moves 290 → 292 of 297. The other two
are unmoved *for stated reasons*, not for lack of reaching them: both were
measured against upstream, both have their fix located to a named file and a
named function, and both have a test that inverts when that fix lands.

Counted the way `CLAUDE.md` §5.3 asks — split rather than totalled:

| | this round |
|---|---|
| kernels added | **1** (`aten.unfold.default`) |
| lowerings added inside an existing kernel | **1** (per-axis conv padding) |
| defects fixed | 0 |
| tests added | 16 in `test_last7.py`, 1 in `test_argform.py` |
| tests **inverted** | 3 (§7) |
| golden cases added | 49 (39 `unfold`, 10 net on `convolution`) |
| documents corrected | 2 (`ARGFORM.md` §1's conclusion, `STRIDED.md`'s producer count) |
| deletions | 0 |

---

## 7. The three inversions, and one pinned count

Nothing was weakened or deleted. Three tests asserted an absence this round
closed and each was inverted into a **stronger** assertion:

1. `test_strided.py::test_as_strided_is_the_only_op_that_takes_the_barrier`
   → `test_as_strided_and_unfold_are_the_two_ops_that_take_the_barrier`.
   Still a closed list; a third producer still fails it. §3.
2. `test_argform.py::test_asymmetric_conv_padding_is_a_backend_limit_not_an_argument_form`
   → `test_per_axis_conv_padding_now_computes_and_agrees_with_upstream`, which
   diffs values instead of asserting a refusal, **plus** a second test keeping
   the stride half of the refusal named. §4.
3. `tools/golden/cases.py`'s `asymmetric padding -- c_error, torch computes`
   → five live value-diffing rows across two dtypes. §4.

One pinned count moved, with the arithmetic that keeps it a check:
`test_shim.py`'s `assert len(keys) == 363` → `364`. **+1, and which table it comes
from is the check**: `methods.json`-only, because upstream has `Tensor.unfold` and
**no** `torch.unfold` (measured: `hasattr(torch, "unfold")` is `False`), so
`aten::unfold|default` is a new identity rather than a second spelling. `+2` would
have meant an `overloads.json` row for a door upstream does not have — the trap
that note already records in the other direction for `complex`.

`tools/golden/reach_allow.json` lost its `multinomial` entry, which the reach test
demanded by name once §5.1's test spelled the function. The allowlist's stated
reason was that a value assertion on a random op is unsound; the test that
replaced it asserts a **shape**, so the reason does not survive the entry.

---

## 8. Verification

```text
suite            886 ok, 0 FAIL, EXIT=0        (baseline 868 ok on b33e2ee)
DOCWATCH         PASS -- 777/777
golden           11385/11385 cases, ops covered = 300, pending = 0
                                              (baseline 11336/11336, ops = 299)
arch_sweep       univnet: forward ok
                 nystromformer: forward ok
```

Every value claim above was produced by running the op on real torch 2.13.0 in a
separate process with `PYTHONPATH` and `TORCH_USE_RTLD_GLOBAL` unset, and
comparing element by element — never by reading a shim result twice.

<!-- DOCWATCH: count golden_cases_total ge 11385 -->
<!-- DOCWATCH: count golden_cases_passed ge 11385 -->
<!-- DOCWATCH: count golden_ops_covered ge 300 -->
<!-- DOCWATCH: count golden_pending eq 0 -->
