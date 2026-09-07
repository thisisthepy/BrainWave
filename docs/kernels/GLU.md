# GLU — the ASR encoders' shared wall, and the six ops behind it

Worktree `work/glu` on develop `eb84708`. Territory: `rust/torch_c/src/aten.rs`,
`methods.json`, `overloads.json`, `tools/golden/cases.py`,
`rust/torch_c/pytests/test_glu.py`. `bootstrap.py`, `capture.rs`, `tape.rs`,
`tensor.rs`, `device.rs`, `tools/wheel/`, `torchnative/` were not touched, per
this round's territory split.

docs/architectures/ARCH100.md's sweep named seven blocked architectures at one operator,
`torch._C._nn.glu`, all ASR encoders: `parakeet` x3, `lasr` x2, `cohere_asr`,
`parakeet_tdt`. This project has never run a speech model, and a concurrent
round is adding TTS models from `voicestudio` — so the 1-D and speech-shaped
operators this round covers are about to be in demand from two directions.

## 1. `aten.glu.default` — DONE, kernel + golden cases + regression tests

`aten::glu(Tensor self, int dim=-1) -> Tensor`: split `self` in half along
`dim`, return `a * sigmoid(b)`. Implemented in `aten.rs` (`glu_default`,
next to `silu_default`), added to `IMPLEMENTED`, wired into the dispatch
match, golden-compared in `tools/golden/cases.py` (`glu_cases`, registered
in `CASE_BUILDERS`), and pinned with regression tests in the new
`rust/torch_c/pytests/test_glu.py`.

Measured against upstream 2.13.0 (`torch.ops.aten.glu.default`, `F.glu`),
not assumed:

```text
glu(arange(12).reshape(2,6))            splits dim=-1 (the DEFAULT)
glu(arange(12).reshape(2,6), dim=0)     splits the OTHER axis -- different answer, different shape
glu(arange(6).reshape(1,3,2), dim=1)    RAISES "Halving dimension must be even, but dimension 1 is size 3"
glu(arange(4, dtype=int64))             RAISES "\"glu_cpu\" not implemented for 'Long'"
hasattr(torch, "glu")                   False -- no torch.glu, no Tensor.glu upstream
```

Two traps, both checked rather than assumed:

* **The default is `dim=-1`, not `dim=0`.** All seven blocked architectures
  gate the trailing (channel) axis after a conv or linear that puts it last.
  A `dim=0` default would silently split the wrong axis for every one of
  them while still passing a golden suite that only exercises 2-D inputs
  with the batch axis first — which is why `test_glu.py` asserts the
  default disagrees with `dim=0` on a non-square input, not just that it
  matches `dim=-1`.
* **An odd size at `dim` must raise, not floor-divide.** A kernel that
  computed `extent // 2` for both halves without checking evenness would
  pass every even-sized golden case and only diverge the first time a real
  checkpoint's projection width is odd.

Dtype/precision rule is `silu`'s, not `sigmoid`'s: float only (no integral
promotion — `glu_cpu` has no kernel for `Long` upstream, same refusal shape
as `silu_cpu`), and `float16`/`bfloat16` accumulate in `float32` and narrow
once at the end (the `sigmoid`/`silu` precision rule, inherited rather than
re-derived since `glu`'s internal sigmoid is the same fused op).

### 1.1 What is NOT done: the `torch._C._nn.glu` Python binding

`torch._C._nn.glu` (what `F.glu` actually calls, per
`torch/nn/functional.py`) is a `bootstrap.py::_install_nn` composite, the
same shape as `pad`/`upsample_bilinear2d`/`leaky_relu` already there.
**`bootstrap.py` is out of this round's territory** (nine other agents are
editing `aten.rs`-adjacent files concurrently, and `bootstrap.py` is the file
most likely to collide). The kernel this file lands, `aten.glu.default`, is
proven directly against upstream through `_C._aten_dispatch("aten.glu.default",
...)` in `test_glu.py` and through `tools/golden/compare.py`, exactly as the
gates ask — but a checkpoint calling `F.glu` will not reach it until someone
adds to `_install_nn`:

```python
def glu(input, dim=-1):
    """torch._C._nn.glu -- F.glu(x, dim) IS this binding, not a wrapper
    around it (same relationship gelu/leaky_relu have to _nn)."""
    return dispatch("aten.glu.default", input, dim)
```

registered in the `for fn, name in (...)` table and `_shim_nn_implemented`
alongside the existing entries. That is a three-line addition once
`bootstrap.py` is free to edit — the arithmetic and its edge cases are
already proven here. **So "seven ASR encoders clear glu" is not yet true
end-to-end**; what is true is that the kernel they all need now exists,
golden-compared, and the remaining step is binding surface, not numerics —
the same `missing_shim_name` vs. kernel distinction ARCH100.md §2 draws.
Whoever owns `bootstrap.py` next should re-run `pytests/arch_sweep.py`
after adding the three lines above; that is the check this round could not
finish itself.

## 2. The other six ops — scoped out this round, schemas recorded so the next round does not re-derive them

Budget did not stretch to implementing kernels for the remaining six ops
this round named. Recorded here, from upstream 2.13.0, so re-deriving them
is not the next round's first step:

```text
aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[],
                  int[1] padding=[0], int[1] dilation=[1], bool ceil_mode=False) -> Tensor
aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
aten::upsample_linear1d(Tensor self, SymInt[1] output_size, bool align_corners,
                         float? scales=None) -> Tensor
aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None,
                          bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
torch._C._nn.pad                        -- dispatcher over constant/reflect/replicate/circular;
                                            `constant_pad_nd` already exists in this shim and
                                            is what `pad(mode="constant")` should route to
                                            (bootstrap.py already has a `pad` composite that
                                            refuses every mode by name -- see line ~7216)
```

Notes for whoever picks these up next, from this round's reading rather than
guessing:

* **`logsumexp` must subtract the max before summing exp**, and per the
  task brief a real check needs a case where the naive form overflows and
  the stable form does not (e.g. values around 1e4+ in float32) — that is
  what makes it a check rather than a restatement of the formula.
* **`upsample_linear1d`'s traps are `upsample_bicubic2d`'s, per docs/architectures/DEMAND8.md
  §2.4, and none of the three should be assumed to transfer without
  re-measuring against upstream on `upsample_linear1d` specifically**:
  `align_corners=False` did not clamp the source index at 0 for cubic;
  there is no `out == in` short circuit for bicubic; the weights are stored
  at the input's dtype, not `opmath_t`. `upsample_linear1d` is 1-D bilinear's
  sibling, not bicubic's, so its source-index rule is closer to
  `upsample_bilinear2d`'s (already in this shim) than to bicubic's — but
  that is exactly the kind of "should transfer" claim this project has been
  burned by assuming rather than running.
* **`linalg_vector_norm`** is the real leaf upstream's `torch.linalg.norm`
  reaches for the `ord=`/`None`/`'fro'` cases relevant here (measured:
  `torch.linalg.norm(x)` with no `ord` computes a plain 2-norm over the
  flattened tensor). `owlv2`/`owlvit` are the two blocked architectures;
  which `ord` they actually pass was not checked this round — that is the
  first thing the next round should measure, not assume `ord=2`.
* **`_nn.pad`**: `bootstrap.py`'s existing `pad` composite (this round did
  not touch it) already refuses every mode by name and notes that
  `constant_pad_nd` — which has a kernel and golden cases in this shim
  already — is the route for `mode="constant"`. `univnet` is the one
  blocked architecture; which mode it calls was not checked this round.

## 3. Verification run

`cargo build --release` (`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-glu`),
then `PYTHON=$PY sh rust/torch_c/pytests/run.sh` and
`TORCH_C_ARTEFACT=.../lib_C.dylib $PY tools/golden/compare.py`. Results are
in the report this document accompanies rather than duplicated here, per
the rule that a number superseded by a later run should not be pinned twice.
