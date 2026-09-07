"""docs/bindings/BIND4.md -- four small `bootstrap.py` items, each located by a round
that could not fix it because `bootstrap.py` was not its file.

  1. `torch.conv1d(padding="same")` with an ODD `dilation * (kernel - 1)` --
     `lasr_ctc`/`lasr_encoder`'s wall (docs/kernels/RNN.md §1.2). Not a kernel: both
     `aten.constant_pad_nd.default` and `aten.convolution.default` were
     already implemented and golden-compared. The fix is a four-line
     transcription of upstream's own lowering, measured with a
     `TorchDispatchMode` logger: pad one zero on the RIGHT, then convolve
     symmetrically with `total // 2`.

  2. `torch._C._nn.upsample_linear1d` -- `sam_vision_model`/
     `sam_hq_vision_model`'s wall (docs/kernels/RNN.md §3.1). Not a kernel either:
     `aten.upsample_linear1d.default` was already implemented, golden- and
     bit-compared. The gap was one `_install_nn` line, shaped like
     `upsample_nearest1d`'s one line above it -- and the discriminator is the
     same trap docs/bindings/BIND3.md §3.1 found there: the fourth argument's TYPE,
     not the arity, since both the `.vec` and leaf schemas here take four
     arguments.

  3. `torch.zeros((..., 0-dim int Tensor, ...), dtype=..., device=...)` --
     `fastspeech2_conformer`'s wall (docs/kernels/TAIL4.md §8.2). Measured against
     real upstream torch 2.13.0 before being added, per docs/bindings/ARGFORM.md's own
     bar: a 0-dim integral Tensor inside a `SymInt[]` size list is upstream's
     general size-list parsing rule (also measured against `torch.ones`,
     `torch.empty`, `Tensor.view`), but installed only for `zeros`, the same
     scoping `div`'s wrapped-number rule uses one function above it.

  4. `torch._C._nn.avg_pool2d` -- confirmed still bound (docs/bindings/BIND2.md).
     `nystromformer` and `univnet` were checked and found NOT actionable from
     this file: `nystromformer` stops on `aten.convolution.default`'s
     asymmetric-padding refusal, a candle backend limitation in `aten.rs`
     (docs/bindings/ARGFORM.md §1's own verdict, re-measured here); `univnet` stops on
     `TensorBase.unfold`, a missing kernel, also in `aten.rs`. Neither has a
     line to add here.

Every case below goes through the spelling a user actually writes --
`F.conv1d(..., padding="same")`, `F.interpolate(..., mode='linear')`,
`torch.zeros((tensor_size,), ...)` -- and is compared against a **live
upstream torch in a separate process** (`env -u PYTHONPATH -u
TORCH_USE_RTLD_GLOBAL`), element-wise, shape and dtype included. Golden
compares at the dispatch key and is structurally blind to whether any Python
spelling reaches it, which is the whole reason this file exists rather than a
golden case.

Nothing here needs numpy or a network.
"""

import json
import os
import subprocess
import sys

from test_shim import _C

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


_PROBE_SCRIPT = r"""
import json, sys
import torch
import torch.nn.functional as F

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


def rec(name, fn):
    try:
        r = fn()
    except Exception as e:
        out[name] = {"raised": type(e).__name__, "msg": str(e)}
        return
    if isinstance(r, torch.Tensor):
        out[name] = {"ok": [float(v) for v in r.reshape(-1).double()],
                     "shape": list(r.shape), "dtype": str(r.dtype)}
    else:
        out[name] = {"ok": r}


# -- 1. F.conv1d(padding="same"), odd dilation*(kernel-1) -------------------
# `LasrEncoderConvolutionModule`'s exact shape: depthwise (`groups=channels`),
# an EVEN `conv_kernel_size` despite the model's own comment saying it should
# be odd, so `dilation * (kernel - 1)` is odd and upstream pads
# asymmetrically (one extra zero on the right) before convolving.
cx = torch.arange(2 * 3 * 9, dtype=torch.float32).reshape(2, 3, 9) / 7
cw_even = torch.arange(3 * 1 * 4, dtype=torch.float32).reshape(3, 1, 4) / 5
cw_odd = torch.arange(3 * 1 * 3, dtype=torch.float32).reshape(3, 1, 3) / 5
cb = torch.tensor([0.5, -0.25, 2.0])
rec("conv1d_same_odd_total", lambda: F.conv1d(cx, cw_even, cb, 1, "same", 1, 3))
rec("conv1d_same_even_total", lambda: F.conv1d(cx, cw_odd, cb, 1, "same", 1, 3))
rec("conv1d_same_odd_dilation2",
    lambda: F.conv1d(cx, cw_odd, cb, 1, "same", 2, 3))
rec("conv1d_valid_unaffected", lambda: F.conv1d(cx, cw_even, cb, 1, "valid", 1, 3))
# Non-unit stride with "same" is upstream's OWN refusal, not this shim's --
# held so the new branch does not silently swallow it.
rec("conv1d_same_strided_refused",
    lambda: F.conv1d(cx, cw_even, cb, 2, "same", 1, 3))

# -- 2. F.interpolate(mode="linear") -> torch._C._nn.upsample_linear1d ------
# The four call shapes upstream's binding itself accepts, discriminated by
# the TYPE of the fourth argument (a sequence is `.vec`'s `scale_factors`; a
# float is the leaf's `scales`) -- docs/bindings/BIND3.md §3.1's trap, one line above
# this name in `torch._C._nn`.
line = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
for w in (1, 2, 3, 5, 7, 9, 16):
    rec("interp_linear_size_%d_ac0" % w,
        lambda w=w: F.interpolate(line, size=w, mode="linear", align_corners=False))
    rec("interp_linear_size_%d_ac1" % w,
        lambda w=w: F.interpolate(line, size=w, mode="linear", align_corners=True))
for s in (0.5, 1.5, 2.25):
    rec("interp_linear_scale_%s" % s,
        lambda s=s: F.interpolate(line, scale_factor=s, mode="linear", align_corners=False))
rec("nn_ul1d_vec_size",
    lambda: torch._C._nn.upsample_linear1d(line, [9], False, None))
rec("nn_ul1d_vec_factors",
    lambda: torch._C._nn.upsample_linear1d(line, None, True, [1.5]))
rec("nn_ul1d_leaf_scales",
    lambda: torch._C._nn.upsample_linear1d(line, [9], True, 1.5))
rec("nn_ul1d_three_args",
    lambda: torch._C._nn.upsample_linear1d(line, [9], False))
rec("nn_ul1d_both_refused",
    lambda: torch._C._nn.upsample_linear1d(line, [9], False, [1.5]))
rec("nn_ul1d_neither_refused",
    lambda: torch._C._nn.upsample_linear1d(line, None, False, None))
rec("interp_linear_uint8_refused", lambda: F.interpolate(
    torch.arange(4, dtype=torch.uint8).reshape(1, 1, 4), size=7, mode="linear",
    align_corners=False))

# -- 3. torch.zeros with a 0-dim Tensor inside the size tuple ---------------
# `fastspeech2_conformer`'s exact shape: `max_len =
# torch.sum(duration_labels, dim=1).max()` (a 0-dim Tensor), then
# `torch.zeros((batch, max_len, dim), dtype=..., device=...)` straight from
# it -- upstream never narrows it to a Python int first.
max_len = torch.sum(torch.tensor([[1, 2, 3], [4, 5, 6]]), dim=1).max()
rec("zeros_tensor_in_size_tuple",
    lambda: torch.zeros((2, max_len, 5), dtype=torch.float32, device="cpu"))
rec("zeros_tensor_only_element", lambda: torch.zeros((max_len,)))
rec("zeros_tensor_and_plain_ints_mixed",
    lambda: torch.zeros((max_len, 3, max_len)))
# The plain-int and varargs spellings must still work unchanged.
rec("zeros_plain_tuple", lambda: torch.zeros((2, 3, 5), dtype=torch.float32))
rec("zeros_varargs", lambda: torch.zeros(2, 3))
# A multi-element Tensor is still refused (by `__int__`, not specially
# handled) -- not silently accepted as "take the first element" or similar.
rec("zeros_multi_element_tensor_refused",
    lambda: torch.zeros((torch.tensor([1, 2]), 3)))

json.dump(out, sys.stdout)
"""

_cache = {}


def _run(side):
    """`side` is `"shim"` or `"upstream"`. `None` on the shim side when the
    vendored shim is not installed, matching `test_bind3.py`/`test_voice3.py`:
    `run.sh` builds the standalone `_C`, and only `vendor/install_shim.sh`
    writes the one the vendored tree loads."""
    if side in _cache:
        return _cache[side]
    env = dict(os.environ)
    if side == "shim":
        if not os.path.isfile(_VENDOR_SHIM):
            _cache[side] = None
            return None
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"the {side} probe failed to run at all:\n{proc.stderr[-3000:]}"
    )
    data = json.loads(proc.stdout)
    assert data["_marker"] == side, (
        f"the {side} probe imported the wrong torch ({data['_marker']}) -- every "
        "assertion below would have been about the wrong library"
    )
    _cache[side] = data
    return data


def _both(name):
    s = _run("shim")
    if s is None:
        return "skip"
    up = _run("upstream")
    assert name in up, f"{name} is not in the upstream probe output"
    return s[name], up[name]


def _agree(name, atol=1e-5, rtol=1e-5):
    pair = _both(name)
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, f"{name}: upstream itself refused: {want}"
    assert "raised" not in got, (
        f"{name}: the shim refused where upstream computed: {got}"
    )
    assert got.get("shape") == want.get("shape"), f"{name}: shape {got} != {want}"
    assert got.get("dtype") == want.get("dtype"), f"{name}: dtype {got} != {want}"
    gv, wv = got["ok"], want["ok"]
    assert len(gv) == len(wv), f"{name}: length {len(gv)} != {len(wv)}"
    for a, b in zip(gv, wv):
        assert abs(a - b) <= atol + rtol * abs(b), f"{name}: {a} != {b}"


def _both_refuse(name):
    pair = _both(name)
    if pair == "skip":
        return
    got, want = pair
    assert "raised" in want, f"{name}: upstream did NOT refuse: {want}"
    assert "raised" in got, (
        f"{name}: upstream refuses and the shim computed something instead: {got}"
    )
    return got, want


# ---------------------------------------------------------------------------
# 1. conv1d(padding="same"), odd dilation*(kernel-1)
# ---------------------------------------------------------------------------


def test_conv1d_same_odd_total_now_computes():
    """`lasr_ctc`/`lasr_encoder`'s actual wall, through `F.conv1d` -- the
    spelling `nn.Conv1d.forward` writes, not the raw aten op."""
    _agree("conv1d_same_odd_total")


def test_conv1d_same_even_total_still_agrees():
    """The branch that already worked before this round is unaffected."""
    _agree("conv1d_same_even_total")


def test_conv1d_same_odd_total_with_dilation():
    """Same branch, a different dilation, so the fix is not overfit to
    dilation=1."""
    _agree("conv1d_same_odd_dilation2")


def test_conv1d_valid_padding_unaffected():
    _agree("conv1d_valid_unaffected")


def test_conv1d_same_strided_is_upstreams_own_refusal():
    """Non-unit stride with `padding="same"` refuses upstream too -- the new
    odd-total branch must not swallow this refusal."""
    _both_refuse("conv1d_same_strided_refused")


def test_conv1d_is_still_a_name_not_a_kernel():
    """The fix adds no new dispatch key: it composes
    `aten.constant_pad_nd.default` and `aten.convolution.default`, both
    already implemented."""
    both = set(_C._aten_implemented()) | set(_C._aten_implemented_awaiting_golden())
    assert not [k for k in both if k.startswith("aten.conv1d.")], sorted(both)
    assert "aten.constant_pad_nd.default" in both
    assert "aten.convolution.default" in both


# ---------------------------------------------------------------------------
# 2. torch._C._nn.upsample_linear1d
# ---------------------------------------------------------------------------


def test_f_interpolate_linear_agrees_with_upstream():
    """The spelling `sam_vision_model`/`sam_hq_vision_model` actually write:
    `F.interpolate(x_3d, mode="linear")`, both `align_corners` values, over
    enough output widths to cross the FUSED-multiply-add ULP boundary
    docs/kernels/RNN.md §3 measured (golden's float32 tolerance would not catch it,
    but this compares upstream's own float64-widened output, not bits)."""
    for w in (1, 2, 3, 5, 7, 9, 16):
        _agree("interp_linear_size_%d_ac0" % w)
        _agree("interp_linear_size_%d_ac1" % w)


def test_f_interpolate_linear_scale_factor_is_forwarded():
    """`1/scale` and `in/out` diverge as soon as the product is not integral
    -- `scale_factor=2.25` on a width-4 input is that case."""
    for s in (0.5, 1.5, 2.25):
        _agree("interp_linear_scale_%s" % s)


def test_nn_upsample_linear1d_discriminates_leaf_from_vec_by_type():
    """Both schemas take four arguments here -- the discriminator that
    `upsample_nearest1d` needs one line above this name, and that a
    transcription from `upsample_nearest2d` (which uses arity) would miss."""
    vec_size, vec_factors = _both("nn_ul1d_vec_size")[0], _both("nn_ul1d_vec_factors")[0]
    leaf = _both("nn_ul1d_leaf_scales")[0]
    if _run("shim") is None:
        return
    assert "raised" not in vec_size and "raised" not in vec_factors
    assert "raised" not in leaf
    # `.vec` with an explicit size and `.vec` with a scale factor must not
    # collapse to the same output width by construction of the fixture.
    assert vec_size["shape"] == leaf["shape"] == [2, 3, 9]


def test_nn_upsample_linear1d_three_arg_leaf_form():
    _agree("nn_ul1d_three_args")


def test_nn_upsample_linear1d_mutual_exclusion_refusals():
    """Both wrong ways to ask -- output_size and scale_factors both given, or
    neither -- refuse with upstream's own words, measured rather than
    assumed from `upsample_nearest1d`'s copy of the same message."""
    both = _both_refuse("nn_ul1d_both_refused")
    neither = _both_refuse("nn_ul1d_neither_refused")
    if both is None or neither is None:
        return
    got_b, want_b = both
    assert got_b["msg"] == want_b["msg"], (got_b["msg"], want_b["msg"])
    got_n, want_n = neither
    assert got_n["msg"] == want_n["msg"], (got_n["msg"], want_n["msg"])


def test_upsample_linear1d_uint8_still_refused():
    """`uint8` is refused by the KERNEL (`compute_indices_weights_linear`
    not implemented for `Byte`), unaffected by adding the binding on top --
    the binding must not paper over a real kernel refusal."""
    _both_refuse("interp_linear_uint8_refused")


def test_upsample_linear1d_binding_reaches_the_kernel_key():
    """The kernel was really there before this binding was written, checked
    the way docs/bindings/BINDINGS.md's `mish` was not."""
    assert "upsample_linear1d" in _C._shim_nn_implemented
    assert "aten.upsample_linear1d.default" in set(_C._aten_implemented())


# ---------------------------------------------------------------------------
# 3. torch.zeros with a 0-dim Tensor inside the size tuple
# ---------------------------------------------------------------------------


def test_zeros_tensor_in_size_tuple_agrees():
    """`fastspeech2_conformer`'s exact call: a 0-dim `Tensor` (`.max()`'s
    result) sitting in the middle of a `torch.zeros` size tuple, alongside
    plain ints, with `dtype=`/`device=` also given."""
    _agree("zeros_tensor_in_size_tuple")


def test_zeros_tensor_only_element_agrees():
    _agree("zeros_tensor_only_element")


def test_zeros_multiple_tensor_elements_agree():
    _agree("zeros_tensor_and_plain_ints_mixed")


def test_zeros_plain_forms_unaffected():
    """The pre-existing spellings this round did not touch."""
    _agree("zeros_plain_tuple")
    _agree("zeros_varargs")


def test_zeros_multi_element_tensor_still_refused():
    """A multi-element Tensor in the size position must still refuse -- not
    silently take its first element or its length."""
    _both_refuse("zeros_multi_element_tensor_refused")


# ---------------------------------------------------------------------------
# 4. avg_pool2d confirmed bound; nystromformer/univnet confirmed NOT
#    actionable here
# ---------------------------------------------------------------------------


def test_avg_pool2d_is_still_bound():
    """docs/bindings/BIND2.md's binding, confirmed still present rather than assumed
    to have survived every round since."""
    assert "avg_pool2d" in _C._shim_nn_implemented
    assert "aten.avg_pool2d.default" in set(_C._aten_implemented())


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
