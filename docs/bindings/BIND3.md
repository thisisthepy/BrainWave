# Three `_nn` bindings, one `fft_fftn` that is still a kernel, and why zero export names moved

`docs/architectures/VOICE3.md` landed three kernels — `im2col`, `col2im`, `upsample_nearest1d` —
and could not bind any of them, because `rust/torch_c/src/bootstrap.py` was another
round's file. It recorded the gap in `tools/golden/reach_allow.json` and asserted it
from both sides in `pytests/test_voice3.py`. This round closes it.

The bar was set by `docs/bindings/BINDINGS.md`, which was told "`mish` just needs a binding"
and found the kernel gone: **each binding here says whether its kernel was really
there, checked before the binding was written**, not after.

Measured 2026-09-07, `darwin/arm64`, CPython 3.13, `work/bind3`.

<!-- DOCWATCH: count smoke_ok ge 480 -->
<!-- DOCWATCH: count golden_cases_passed ge 10691 -->
<!-- DOCWATCH: count golden_ops_covered ge 287 -->

---

## 0. At a glance

| | |
|---|---|
| `aten.im2col.default` — kernel really present? | **Yes**, `aten.rs:26469`, in `IMPLEMENTED` and golden-compared |
| `aten.col2im.default` — kernel really present? | **Yes**, `aten.rs:26588`, same |
| `aten.upsample_nearest1d.default` — kernel really present? | **Yes**, `aten.rs:26730`, same |
| Bindings landed | 3, all in `_install_nn`, none as an `overloads.json`/`methods.json` row |
| Allowlist entries retired | 3 (§4) |
| `llama4`'s vision tower | **The patch embedding now runs and agrees with upstream to the last digit** (§5). The full `Llama4VisionModel` stops later, on complex `view`, not on `im2col` |
| `aten.fft_fftn.default` | **Not landed.** It is a composite, its three parts are not all reachable, and §6 names which one is missing |
| Export names moved into `bootstrap.py` | **Zero.** §7, and the measurement that decides it |
| Golden | **10691/10691, ops=287** — exactly unmoved; no kernel changed |
| Suite | **755 ok** across `pytests/` (`test_shim.py`'s own share, which `smoke_ok` counts, is 480), `DOCWATCH: PASS` |

---

## 1. `torch._C._nn.im2col` — `F.unfold`, `llama4`'s vision tower

**The kernel was really there.** `im2col_default` at `rust/torch_c/src/aten.rs:26469`,
listed in `IMPLEMENTED`, with its own dtype check, its own sliding-block refusal, and
golden cases including a multi-channel input. Checked before a line of the binding was
written, because `docs/bindings/BINDINGS.md`'s `mish` is what happens when it is not.

<!-- DOCWATCH: op-implemented aten.im2col.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs im2col_default present -->

`torch/nn/functional.py`'s `unfold` ends in a straight forward with no branch:

```python
return torch._C._nn.im2col(
    input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
```

so the binding has nothing to discriminate and nothing to get wrong that the kernel
does not already own. All four `int[2]` arguments are **required** — upstream raises
`TypeError: im2col() missing 3 required positional argument` for
`torch._C._nn.im2col(x, [2,2])` — so no defaults were invented.

**One thing the binding does add.** `torch._C._nn.im2col(x, 2, 1, 0, 1)` with bare ints
computes upstream, because an `int[N]` schema broadcasts a scalar. The Rust `pair_arg`
broadcasts a *length-1 list* but reads through `shape_arg`, which wants a sequence, so
a bare int would have diverged. `_int_pair` normalises it on the Python side; `aten.rs`
was not this round's file and the calling convention belongs on the side that knows it
is one.

**Not an `overloads.json` or `methods.json` row.** There is no `torch.im2col` and no
`Tensor.im2col` on 2.13.0 — `reach_allow.json` staked its reason on exactly that and
`reach.py --verify-upstream` checks it in a PYTHONPATH-stripped subprocess. A table row
would have been the easier change and would have put a door on this shim that upstream
does not have. `test_no_torch_level_spelling_was_invented_for_these_three` holds it from
this side too.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind3.py test_no_torch_level_spelling_was_invented_for_these_three present -->

## 2. `torch._C._nn.col2im` — `F.fold`, `f5-tts`

**The kernel was really there.** `col2im_default` at `aten.rs:26588`, with the property
that is the op: **overlapping windows are summed, not overwritten.**

<!-- DOCWATCH: op-implemented aten.col2im.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs col2im_default present -->

`F.fold` forwards the same way `F.unfold` does, one argument longer. The binding adds
nothing to the kernel's arithmetic; what it adds is that `F.fold` reaches it.

Every `F.fold` case in `test_bind3.py` uses `stride < kernel`, and `fold_no_overlap` is
present as the **control rather than the evidence**: with `stride == kernel` every output
element is covered exactly once, and a summing implementation and an overwriting one
produce byte-identical output. `test_fold_of_unfold_is_the_cover_count_not_the_identity`
asserts the `1 2 2 1 / 2 4 4 2 / 2 4 4 2 / 1 2 2 1` cover pattern directly and then
asserts that pattern is **not** all ones, so the day someone edits the input the check
fails as a broken control instead of passing vacuously.

## 3. `torch._C._nn.upsample_nearest1d` — `F.interpolate(..., mode="nearest")` on 3-D

**The kernel was really there.** `upsample_nearest1d_default` at `aten.rs:26730`, and
`aten.rs`'s own comment records that it was checked *not* to be an alias of the 2-D op.

<!-- DOCWATCH: op-implemented aten.upsample_nearest1d.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs upsample_nearest1d_default present -->

### 3.1 The discriminator is the third argument's TYPE, not the arity

This is the one place the binding could not be transcribed from `upsample_nearest2d`
sitting immediately above it, and it is worth writing down because the transcription
looks correct.

`upsample_nearest2d` has

    .vec   (input, output_size, scale_factors)          3 arguments
    leaf   (self,  output_size, scales_h, scales_w)     4 arguments

so a sentinel on a fourth parameter tells them apart. `upsample_nearest1d` has

    .vec   (input, output_size, scale_factors)          3 arguments
    leaf   (self,  output_size, scales)                 3 arguments

**the same count.** Measured on 2.13.0, all four of these are accepted upstream:

| call | dispatched |
|---|---|
| `_nn.upsample_nearest1d(x, [7], None)` | `.default(x, [7])` |
| `_nn.upsample_nearest1d(x, None, [1.5])` | `.default(x, [7], 1.5)` |
| `_nn.upsample_nearest1d(x, [7], 1.5)` | `.default(x, [7], 1.5)` |
| `_nn.upsample_nearest1d(x, [7])` | `.default(x, [7])` |

A *sequence* third argument is `.vec`'s `scale_factors`; a float is the leaf's `scales`.
Discriminating on arity reads the third row as `.vec` and subscripts a bare float.
`test_upsample_nearest1d_tells_leaf_from_vec_by_type_not_by_arity` sends all four and
then asserts that the two three-argument shapes **disagree about the output shape**, so
the test cannot survive the discriminator being removed.

Both wrong ways to ask `.vec` refuse with upstream's words, and the second is not
guessable from the first: `Must specify exactly one of output_size and scale_factors`
reads like a check against over-specification, and upstream raises the identical string
for *under*-specification. Both messages are compared, not merely their presence.

### 3.2 The scale factor is forwarded — and the obvious test for that does not work

The factor is passed to the kernel rather than only used to size the output, because
`floor(i / scale)` and `floor(i * in_w / out_w)` diverge whenever `in_w * scale` is not
integral. That much was already in `upsample_nearest2d`'s docstring.

What is new here is that **the obvious case does not separate them.** `3 -> 4 at 1.5`
gives `0,0,1,2` under both readings; so does `5 -> 7 at 1.5`. A test built on either —
which is what a reader reaches for first — would have asserted a difference that is not
there and passed for the wrong reason.

Searching `in_w` in `2..12` against scale in eighths over `0.25..5`, **158 pairs
separate the two readings and no integral scale does.** The case used is `in_w = 2` at
`scale = 2.25`, where forwarding gives `0,0,0,1` and recomputing gives `0,0,1,1`:

```
F.interpolate(arange(4).reshape(1,2,2), scale_factor=2.25, mode="nearest")
    upstream -> [0, 0, 0, 1,  2, 2, 2, 3]      (forwarded)
    recomputed from the shapes -> [0, 0, 1, 1,  2, 2, 3, 3]
```

`test_the_scale_factor_is_forwarded_and_not_merely_used_to_size_the_output` computes
both readings, **asserts they differ before asserting anything else**, and then requires
upstream to equal one and not the other. That is the `§5.5` shape — a verification that
can fail.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind3.py test_the_scale_factor_is_forwarded_and_not_merely_used_to_size_the_output present -->

## 4. What the check then deleted

`reach.py`'s shape-2 question is "this kernel is implemented — can any Python spelling
reach it?", and it answers it by scanning `bootstrap.py`'s **literal** string constants
among other sources. `docs/bindings/BIND2.md` had to write literal dict lookups rather than an
f-string for that reason, and these three bindings follow it: each `dispatch(...)` call
names its op as a plain literal.

So the moment the bindings landed, `reach.py` stopped reporting the three as gaps — and
an allowlist entry whose gap has closed **fails the suite too**, which is what keeps
that file describing the present. All three entries are deleted in this change:

```
aten.im2col.default
aten.col2im.default
aten.upsample_nearest1d.default
```

`reach.py` now reports shape 2 as four entries (`_fft_c2c`, `_fft_c2r`, `_fft_r2c`,
`alias`), down from seven, and `REACH: PASS`.

<!-- DOCWATCH: json-key tools/golden/reach_allow.json shape2_kernel_without_spelling present -->

**`test_voice3.py`'s negative test was inverted, not deleted.** It asserted that the
three `_nn` names refused while their kernels answered; it now asserts that the same
three probe cases agree with upstream element-wise. Its docstring said "delete this
test", and the coverage was kept instead: those are the only cases in that file that go
through the `F.*` spelling rather than through `torch.ops.aten.*`.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_voice3.py test_the_three_nn_bindings_now_carry_these_kernels_all_the_way_to_F present -->

## 5. `llama4`'s vision tower — what is and is not claimed

`pytests/arch_sweep.py --only llama4` reports **1/1 forward** on the shim and 1/1 on
upstream. **That is not evidence for this round**, and saying so is the point of this
section.

The sweep's `main_only` inputs are text. Instrumenting `torch._C._nn.im2col` and running
`arch_sweep.run_one("llama4")` in-process records **zero** calls to it — the vision
branch is not exercised by that forward at all. `docs/architectures/VOICE3.md` §1.3 and
`docs/kernels/COMPLEX2.md` §8 both flagged this ambiguity and declined to claim it; the
instrumentation settles it.

What *is* attributable is the module that holds the `nn.Unfold`
(`modeling_llama4.py:976`, `Llama4UnfoldConvolution`), run directly on both sides with
the same weights:

```
Llama4UnfoldConvolution(patch=8, image=16, channels=3, hidden=16)
    shim      (1, 4, 16)   im2col calls 1   sum 7378.503753662
    upstream  (1, 4, 16)   im2col calls 1   sum 7378.503753662
```

Identical to the last digit, with the binding measurably on the path. **The patch
embedding of `llama4`'s vision tower runs; the vision tower as a whole does not.**

The full `Llama4VisionModel` now stops **past** `im2col` (one call recorded), at
`Llama4VisionRotaryEmbedding` → `reshape_for_broadcast` → `freqs_ci.view(*shape)`, which
refuses because `view` is not among the ops taught the complex representation
(`torch._C._complex_ops()`, `docs/kernels/COMPLEX.md` §2). That is the next wall for this tower
and it belongs to the complex road, not to this one. It is not compared against upstream
because `Llama4VisionConfig`'s own defaults fail upstream too
(`mat1 and mat2 shapes cannot be multiplied`), so that configuration proves nothing in
either direction.

Neither of these runs is in `run.sh`: they construct `transformers` models, which
`arch_sweep.py`'s own header says must stay out of the suite.

## 6. `aten.fft_fftn.default` — not landed, and exactly which piece is missing

`docs/kernels/FFT.md` §5.3 called this "now a binding rather than arithmetic", and
`docs/architectures/ARCH200.md` lists `torch._C._fft.fft_fftn` as `fnet`'s wall. It is *nearly* right:
`aten::fft_fftn` is `CompositeImplicitAutograd`, and a `TorchDispatchMode` trace of
`torch.fft.fftn` upstream fires **no `fft_fftn` at all**:

```
torch.fft.fftn(x)                        -> _to_copy(dtype=complex64), _fft_c2c(c, [0,1], 0, True)
torch.fft.fftn(x, s=(2,8))               -> _to_copy, slice.Tensor, constant_pad_nd, _fft_c2c
torch.fft.fftn(x, norm="ortho")          -> _to_copy, _fft_c2c(..., 1, True)
torch.fft.fftn(x, norm="forward")        -> _to_copy, _fft_c2c(..., 2, True)
```

So it is a spelling over three ops this shim already implements — `aten._to_copy.default`,
`aten.slice.Tensor`, `aten.constant_pad_nd.default`, `aten._fft_c2c.default` — and the
argument handling (`s`, `dim`, `norm`, the `-1` sentinel, negative dims) is ordinary
Python. On the strength of the op list alone it looks landable this round.

**It is not, and the missing piece is the complex representation, not the transform.**
Measured against the installed shim:

| step | result |
|---|---|
| `aten._to_copy(x, dtype=torch.complex64)` | **refuses** — `dtype not storable by the candle backend` |
| `torch.complex(re, im)` | refuses — no `overloads.json` row |
| `torch.zeros(2, 2, dtype=torch.complex64)` | refuses — same backend reason |
| `torch.view_as_complex(torch.stack([x, zeros], -1))` | **builds** a `complex64` tensor … |
| … then `aten._fft_c2c(that, [0,1], 0, True)` | **refuses** — held as a pair of real tensors |
| `aten._fft_c2c` on a genuine shim complex tensor (`torch.stft` output) | **works** |
| `aten.constant_pad_nd` on that same genuine complex tensor | **refuses** |
| `aten.slice.Tensor` on that same genuine complex tensor | **refuses** |
| `Tensor.real` | **not implemented** |

Three separate gaps, in the order a `fft_fftn` binding would hit them:

1. **There is no real → complex conversion.** `_to_copy` to a complex dtype is the first
   op in *every* `fftn` decomposition and it refuses, because complex is a
   `Repr::Complex { re, im }` pair here and not a candle storage dtype
   (`docs/kernels/COMPLEX.md` §2). `view_as_complex` produces something the C2C kernel then
   rejects, so it is not a way round.
2. **`slice.Tensor` and `constant_pad_nd` refuse complex inputs**, and those two are how
   `s=` truncates and pads. Even given (1), any `s=` argument would be unreachable.
3. **`Tensor.real` is missing**, so `fnet`'s actual line —
   `torch.fft.fftn(x, dim=(1,2)).real` — could not be spelled even if `fftn` returned.

`_fft_c2c` itself is fine: it takes a multi-axis `dim`, it is golden-compared, and it
computes correctly on a real shim complex tensor. **The transform is not what is
missing.** Landing `fft_fftn` is a `Repr::Complex` job in `aten.rs` — teaching
`_to_copy`, `constant_pad_nd` and `slice.Tensor` the complex arm, the same way
`docs/kernels/COMPLEX2.md` taught twelve other ops — and `aten.rs` was not this round's file.

`test_tail2.py::test_stft_computes_and_fft_fftn_is_still_unspelled` therefore stays
as it is. Its half asserting the absence is still true, and inverting it would have been
the lie.

## 7. `torch.export` — how many of the 29 names moved, and why the answer is zero

`docs/graph/EXPORT.md` §6 orders the work: **dispatcher mode entrance first, `_NodeBase`
second, the three of §3 third, the 29 names last.** The entrance is in `aten.rs`, which
was not this round's file, so the question put to this round was the narrow one: of the
29 implementations staged in `torchnative/src/main/torchnative/export/upstream.py`, how
many can move into `bootstrap.py` **without making the empty-graph path reachable**?

**None.** Not because the names are individually dangerous — three of the eight groups
are inert — but because of what the group that is *not* inert does, which is measurable
today and is worse than the failure §6 warns about.

### 7.1 The measurement

The same script, against the installed shim, with and without `upstream.install()`:

```
without install()
    with TorchDispatchMode():          NotImplementedError: torch._C._dynamo.guards.
                                       set_is_in_mode_without_ignore_compile_internals
    with FakeTensorMode():             NotImplementedError: torch._C._only_lift_cpu_tensors
    torch.fx.Graph()                   NotImplementedError: _NodeBase._update_args_kwargs

with install()   (29 replaced, 15 overridden)
    with TorchDispatchMode():          ENTERS
        _len_torch_dispatch_stack()      1
        (ones(3) * 2 + 1).relu()         [3.0, 3.0, 3.0]
        operators seen by the mode       []
    with FakeTensorMode():             ENTERS, and torch.ones(3) * 2 is a real Tensor
    torch.fx.Graph()                   still refuses
```

Today every one of those blocks **refuses by name**. After the move, `with
FakeTensorMode():` becomes a block that enters, reports itself active through
`_len_torch_dispatch_stack() == 1`, hands back eager results, and says nothing. The
fake-tensor abstraction is silently off. That is `docs/graph/COMPILE.md` §5's silent eager
fallback, reached without any `ExportedProgram` being produced at all — **one level below
the empty graph, and correspondingly harder to see.**

### 7.2 Why the existing guard does not catch it

`test_export.py::test_a_graph_front_end_is_not_offered_while_modes_are_not_consulted`
is the load-bearing test, and it is written as

    if no mode sees an operator, no graph front end may return a graph

which after the move is still satisfied: `modes_work` is `False` (the mode sees nothing),
`export_returns` is `False` and `fx_graph_builds` is `False`, because `_NodeBase` is
untouched. The test passes in both columns above. It guards the *graph*; the failure
here is a mode that claims to be active. **A check that cannot distinguish the two
columns is not a check on this change**, and adding the missing one is work for the
round that does item 1, since the invariant it needs — "no mode may report itself active
while the dispatcher ignores it" — is only expressible once there is a dispatcher
entrance to point at.

### 7.3 The subset that looks safe, and why it was not taken either

Three of `upstream.py`'s eight installer groups are genuinely off the
`TorchDispatchMode.__enter__` path: `_install_profiler`, `_install_functorch`,
`_install_tensor_predicates`. Moving those three alone would leave the dynamo bool a
stub, and the mode entrance would go on refusing at exactly the name it refuses at today.

It was still not done, for the reason §6 gives for putting the names last: **they change
nothing on their own.** What a partial move does produce is a `bootstrap.py` and an
`upstream.py` that each hold part of one census, so the round that does items 1–3 has to
reconstruct which half went where before it can apply §8's hand-off. That is a cost with
no matching benefit, and `docs/graph/EXPORT.md` §8 is written as one patch for a reason.

`upstream.py` is unchanged by this round, `test_export.py` is unchanged, and
`set_eval_frame`'s refusal was not approached.

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/upstream.py install present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export.py test_a_graph_front_end_is_not_offered_while_modes_are_not_consulted present -->

---

## 8. What this round changed

Split as `docs/architectures/ARCH100.md` §5.3 asks, because "landed" is not one thing:

| class | what |
|---|---|
| **feature added** | 3 `_install_nn` bindings — `im2col`, `col2im`, `upsample_nearest1d` — and the `_int_pair` helper they share. No kernel, no `aten.rs` change, no new dispatch key |
| **defect fixed** | none |
| **tests added** | `pytests/test_bind3.py`, 17 tests, every positive one element-wise against a live upstream in its own process |
| **tests inverted** | 1 in `pytests/test_voice3.py` — the one asserting these three unreachable from `F.*`. Not deleted |
| **documentation** | this file |
| **deleted** | 3 entries from `tools/golden/reach_allow.json`, deleted *because* the gaps closed and `reach.py` fails on a stale entry |
| **architectures moved** | none claimed. `llama4`'s vision **patch embedding** runs and matches upstream (§5); the tower as a whole still stops, on complex `view` |
| **not done, and why** | `fft_fftn` (§6 — three complex-representation gaps in `aten.rs`); the 29 export names (§7 — measured, zero) |

Golden is **exactly unmoved**: 10691/10691, ops covered 287, 6 recorded divergences —
which is the right result, because nothing in this round is a kernel.

## 9. Reproduction

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-bind3
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export TORCH_C_STAGE=/tmp/stage-bind3
PY=/Volumes/macMini/caches/spike-venv/bin/python

cd rust/torch_c && cargo build --release && cd ../..     # bootstrap.py is include_str!'d
bash vendor/install_shim.sh
PYTHON=$PY sh rust/torch_c/pytests/run.sh                # 755 ok, DOCWATCH: PASS
TORCH_C_ARTEFACT=$TORCH_C_STAGE/_C.abi3.so $PY tools/golden/compare.py   # 10691/10691 ops=287
TORCH_C_ARTEFACT=$TORCH_C_STAGE/_C.abi3.so $PY tools/golden/reach.py     # REACH: PASS, shape 2 = 4

cd rust/torch_c/pytests
PYTHONPATH=$PWD/../../../torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
    $PY arch_sweep.py --only llama4 --out /tmp/shim.json                 # 1/1 forward
```

§5's `im2col`-call instrumentation and §7.1's install comparison are the two runs worth
repeating by hand; both are short scripts and both are quoted in full above.

## 10. What this document does not claim

* **Not that `llama4` works.** Its vision patch embedding does, measured (§5). The tower
  stops one module later and the sweep's green does not cover either.
* **Not that `f5-tts` or `bigvgan` forward.** `col2im` and `upsample_nearest1d` were
  their recorded first walls; whether a second one is behind each is not measured here.
* **Not that `_fft_c2c` is short of anything.** §6's three gaps are all in the complex
  representation of `_to_copy`, `constant_pad_nd` and `slice.Tensor`.
* **Not that the 29 export names are wrong.** §7 is about *when*, not about whether.
