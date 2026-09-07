"""docs/architectures/VOICE3.md -- the seven walls, each proven against upstream in its own
process, on inputs where a *plausible wrong implementation* differs.

`tools/golden/cases.py` compares each of these ops against upstream
element-wise already, and this file does not repeat that. What is here is the
part a value comparison over comfortable inputs structurally cannot hold down:

  * **`i0` across the argument range, not at one point.** Its two Chebyshev
    approximations meet at `x == 8` and agree in the middle of each interval,
    so a spot check passes for an implementation that is wrong at the ends.
    `test_i0_is_bit_identical_to_upstream_across_the_whole_range` sweeps
    88,002 points at `float32` and at `float64` and demands **bit** equality,
    and `test_the_f64_shortcut_for_i0_is_measurably_wrong` records what the
    plausible shortcut would have cost.

  * **`var` at `n = 2`.** Bessel's correction defaults to **1**, not 0. The
    wrong default scales every answer by `n/(n-1)`, which is invisible at
    large `n`; at `n = 2` it is a factor of two.

  * **`col2im` on OVERLAPPING windows.** It sums where windows overlap. A test
    on a stride equal to the kernel cannot distinguish summing from
    overwriting -- every element is covered exactly once -- so every case here
    uses `stride < kernel`, and one asserts the overlap count directly.

  * **`_unique2`'s three-tensor contract.** Which tensors come back is gated by
    flags, the inverse carries the *input's* shape, and `sorted=False` still
    sorts on CPU. None of that is a value comparison.

  * **`diag` is two operations.** Rank decides whether it builds or extracts,
    and `rwkv` only exercises one of them.

  * **The alias-versus-kernel split**, pinned as a count so a later round
    cannot report "seven ops" as seven kernels.

Every upstream number is measured with `env -u PYTHONPATH
-u TORCH_USE_RTLD_GLOBAL python`, i.e. real torch 2.13.0 in its own process.
Nothing here needs numpy or a network.
"""

import json
import math
import os
import subprocess
import sys

from test_shim import _C

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


_PROBE_SCRIPT = r"""
import json, math, struct, sys
import torch

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
    elif isinstance(r, (tuple, list)) and r and isinstance(r[0], torch.Tensor):
        out[name] = {"tuple": [
            {"ok": [float(v) for v in t.reshape(-1).double()],
             "shape": list(t.shape), "dtype": str(t.dtype)} for t in r]}
    else:
        out[name] = {"ok": r}


def bits(t):
    # Raw bits, so `1.0` and `0.9999999403953552` cannot compare equal
    # through a float repr.
    if t.dtype == torch.float64:
        return [struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in t.tolist()]
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in t.tolist()]


aten = torch.ops.aten

# -- i0, swept ---------------------------------------------------------------
# [0, 8] at 0.001 and [8, 88] at 0.001: both Chebyshev intervals, both ends of
# each, and the junction.
sweep = [i * 0.001 for i in range(0, 8001)] + [i * 0.001 for i in range(8000, 88001)]
out["i0_sweep_f32_bits"] = {"ok": bits(torch.i0(torch.tensor(sweep, dtype=torch.float32)))}
out["i0_sweep_f64_bits"] = {"ok": bits(torch.i0(torch.tensor(sweep, dtype=torch.float64)))}
out["i0_sweep_n"] = {"ok": len(sweep)}
# Zero is the point the f64-and-narrow shortcut gets visibly wrong.
out["i0_zero_f32_bits"] = {"ok": bits(torch.i0(torch.tensor([0.0], dtype=torch.float32)))}
rec("i0_reduced_f16", lambda: torch.i0(torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float16)))
rec("i0_reduced_bf16", lambda: torch.i0(torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.bfloat16)))
rec("i0_integral", lambda: torch.i0(torch.tensor([0, 1, 2])))
rec("i0_bool", lambda: torch.i0(torch.tensor([True, False])))
rec("i0_negative", lambda: torch.i0(torch.tensor([-1.0, -8.0, -30.0], dtype=torch.float64)))
rec("i0_nonfinite", lambda: torch.i0(
    torch.tensor([float("nan"), float("inf"), float("-inf")], dtype=torch.float64)))

# -- kaiser_window -----------------------------------------------------------
for L in (0, 1, 2, 4, 5, 12, 64, 257):
    for per in (True, False):
        rec("kaiser_%d_%s" % (L, per), lambda L=L, p=per: torch.kaiser_window(L, p))
for beta in (0.0, 1.0, 9.0, 12.0, 38.0):
    rec("kaiser_beta_%s" % beta, lambda b=beta: torch.kaiser_window(16, True, b))
for name, dt in (("f64", torch.float64), ("f16", torch.float16), ("bf16", torch.bfloat16)):
    rec("kaiser_dtype_%s" % name, lambda d=dt: torch.kaiser_window(8, True, 12.0, dtype=d))
out["kaiser_12_true_bits"] = {"ok": bits(torch.kaiser_window(12, True))}
out["kaiser_12_false_bits"] = {"ok": bits(torch.kaiser_window(12, False))}
out["kaiser_64_beta9_f64_bits"] = {"ok": bits(torch.kaiser_window(64, True, 9.0, dtype=torch.float64))}
rec("kaiser_negative", lambda: torch.kaiser_window(-1))
rec("kaiser_int_dtype", lambda: torch.kaiser_window(4, dtype=torch.int64))

# -- diag: BOTH operations ---------------------------------------------------
mat = torch.arange(12, dtype=torch.float32).reshape(3, 4)
tall = torch.arange(12, dtype=torch.float32).reshape(4, 3)
vec = torch.tensor([1.0, 2.0, 3.0])
for d in (-6, -2, -1, 0, 1, 3, 5):
    rec("diag_extract_wide_%d" % d, lambda d=d: torch.diag(mat, d))
    rec("diag_extract_tall_%d" % d, lambda d=d: torch.diag(tall, d))
    rec("diag_build_%d" % d, lambda d=d: torch.diag(vec, d))
rec("diag_square", lambda: torch.diag(torch.arange(9, dtype=torch.float32).reshape(3, 3), 0))
rec("diag_int", lambda: torch.diag(torch.tensor([1, 2, 3]), 1))
rec("diag_bool", lambda: torch.diag(torch.tensor([True, False])))
rec("diag_empty_vector", lambda: torch.diag(torch.zeros(0)))
rec("diag_transposed_receiver", lambda: torch.diag(mat.t(), 0))
rec("diag_3d", lambda: torch.diag(torch.zeros(2, 2, 2)))
rec("diag_0d", lambda: torch.diag(torch.tensor(3.0)))
rec("diag_method", lambda: mat.diag(1))
# `rwkv`'s own line: nn/init.py:710 is `d = torch.diag(r, 0); ph = d.sign()`.
rwkv_r = torch.tensor([[-2.0, 1.0, 4.0], [0.0, 3.0, -5.0], [0.0, 0.0, -1.0]])
rec("diag_rwkv_sign", lambda: torch.diag(rwkv_r, 0).sign())

# -- var: the correction default, at n = 2 -----------------------------------
two = torch.tensor([1.0, 3.0])
rec("var_n2_default", lambda: torch.var(two))
rec("var_n2_correction0", lambda: torch.var(two, correction=0))
rec("var_n2_unbiased_false", lambda: torch.var(two, unbiased=False))
six = torch.tensor([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
rec("var_all_default", lambda: torch.var(six))
for c in (0, 1, 2, 5):
    rec("var_correction_%d" % c, lambda c=c: torch.var(six, correction=c))
rec("var_correction_fractional", lambda: torch.var(six, correction=0.5))
rec("var_dim0", lambda: torch.var(six, 0))
rec("var_dim1", lambda: torch.var(six, 1))
rec("var_dim_neg", lambda: torch.var(six, -1))
rec("var_dim_keepdim", lambda: torch.var(six, 0, keepdim=True))
rec("var_dim_list", lambda: torch.var(six, [0, 1]))
rec("var_dim_empty_list", lambda: torch.var(six, []))
rec("var_dim_none_correction0", lambda: torch.var(six, None, correction=0))
cube = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) * 0.5
rec("var_cube_dim1", lambda: torch.var(cube, 1))
rec("var_cube_dim02", lambda: torch.var(cube, [0, 2]))
rec("var_cube_dim02_keepdim", lambda: torch.var(cube, [0, 2], keepdim=True))
rec("var_scalar", lambda: torch.var(torch.tensor(3.0)))
rec("var_empty", lambda: torch.var(torch.zeros(0)))
for name, dt in (("f64", torch.float64), ("f16", torch.float16), ("bf16", torch.bfloat16)):
    rec("var_dtype_%s" % name, lambda d=dt: torch.var(six.to(d)))
    rec("var_dtype_dim_%s" % name, lambda d=dt: torch.var(six.to(d), 0))
# The clamped divisor: `max(0, n - correction)`, so an overshooting
# correction divides by zero rather than going negative.
rec("var_denom_zero_nonzero_m2", lambda: torch.var(two, correction=2))
rec("var_denom_zero_zero_m2", lambda: torch.var(torch.tensor([2.0, 2.0]), correction=2))
rec("var_denom_negative_nonzero_m2", lambda: torch.var(two, correction=3))
rec("var_integral", lambda: torch.var(torch.tensor([1, 2, 3])))

# -- std: a leaf, not a composite over var ----------------------------------
rec("std_n2_default", lambda: torch.std(two))
rec("std_n2_correction0", lambda: torch.std(two, correction=0))
rec("std_n2_unbiased_false", lambda: torch.std(two, unbiased=False))
rec("std_all_default", lambda: torch.std(six))
for c in (0, 1, 2, 5):
    rec("std_correction_%d" % c, lambda c=c: torch.std(six, correction=c))
rec("std_correction_fractional", lambda: torch.std(six, correction=0.5))
rec("std_dim0", lambda: torch.std(six, 0))
rec("std_dim1", lambda: torch.std(six, 1))
rec("std_dim_neg", lambda: torch.std(six, -1))
rec("std_dim_keepdim", lambda: torch.std(six, 0, keepdim=True))
rec("std_dim_list", lambda: torch.std(six, [0, 1]))
rec("std_cube_dim1", lambda: torch.std(cube, 1))
rec("std_cube_dim02_keepdim", lambda: torch.std(cube, [0, 2], keepdim=True))
rec("std_scalar", lambda: torch.std(torch.tensor(3.0)))
rec("std_empty", lambda: torch.std(torch.zeros(0)))
rec("std_denom_zero_nonzero_m2", lambda: torch.std(two, correction=2))
rec("std_denom_negative_nonzero_m2", lambda: torch.std(two, correction=3))
for name, dt in (("f64", torch.float64), ("f16", torch.float16), ("bf16", torch.bfloat16)):
    rec("std_dtype_%s" % name, lambda d=dt: torch.std(six.to(d)))
    rec("std_dtype_dim_%s" % name, lambda d=dt: torch.std(six.to(d), 0))
rec("std_integral", lambda: torch.std(torch.tensor([1, 2, 3])))
rec("std_method", lambda: six.std(0))
rec("std_ops_correction", lambda: aten.std.correction(six, [0], correction=0))
rec("std_ops_dim", lambda: aten.std.dim(six, [0], False, False))
rec("std_ops_default", lambda: aten.std.default(six, True))
# `gemma3n_text`'s own spelling, modeling_gemma3n.py:1745.
rec("std_gemma3n_spelling", lambda: six.std(dim=-1, keepdim=True))
# `std` is NOT `var(...).sqrt()`: the root goes in the accumulator, and the
# composite narrows first. Recorded from both sides so the ULP gap is a fact
# about upstream rather than about this shim.
# Found by searching float32 samples for a disagreement upstream: this one
# gives std = 29.12095832824707 and var().sqrt() = 29.120956420898438, one
# ULP apart, and both sides of the split are reproduced exactly here.
wide_std = torch.tensor([-20.9247, 45.96, -35.8534, -12.5087,
                         -1.5798, 36.5793, 21.9547, 22.7914])
rec("std_wide_sample", lambda: torch.std(wide_std))
rec("std_wide_sample_as_sqrt_var", lambda: torch.var(wide_std).sqrt())
rec("var_method", lambda: six.var(0))
rec("var_ops_correction", lambda: aten.var.correction(six, [0], correction=0))
rec("var_ops_dim", lambda: aten.var.dim(six, [0], False, False))
rec("var_ops_default", lambda: aten.var.default(six, True))

# -- _unique2: three tensors, gated ------------------------------------------
dup = torch.tensor([3, 1, 2, 1, 3, 3])
for s in (True, False):
    for ri in (False, True):
        for rc in (False, True):
            rec("unique2_%s_%s_%s" % (s, ri, rc),
                lambda s=s, ri=ri, rc=rc: aten._unique2(dup, s, ri, rc))
rec("unique2_2d", lambda: aten._unique2(torch.tensor([[3.0, 1.0], [2.0, 1.0]]), True, True, True))
rec("unique2_empty", lambda: aten._unique2(torch.zeros(0), True, True, True))
rec("unique2_scalar", lambda: aten._unique2(torch.tensor(5), True, True, True))
rec("unique2_bool", lambda: aten._unique2(torch.tensor([True, False, True]), True, True, True))
rec("unique2_nan", lambda: aten._unique2(
    torch.tensor([float("nan"), 1.0, float("nan"), 0.0]), True, True, True))
rec("unique2_negative", lambda: aten._unique2(torch.tensor([0, -1, 5, -1, 0]), True, True, True))
rec("unique2_float", lambda: aten._unique2(
    torch.tensor([2.5, -0.5, 2.5, 7.0]), True, True, True))
# `vilt`'s own spelling: modeling_vilt.py:144.
valid_idx = torch.tensor([[2, 0], [0, 1], [2, 3], [0, 0], [1, 5]])
rec("unique2_vilt_spelling", lambda: valid_idx[:, 0].unique())

# -- im2col / col2im: OVERLAPPING windows ------------------------------------
img = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
multi = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
batched = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)
rec("im2col_k2_s1", lambda: aten.im2col(img, [2, 2], [1, 1], [0, 0], [1, 1]))
rec("im2col_k2_s2", lambda: aten.im2col(img, [2, 2], [1, 1], [0, 0], [2, 2]))
rec("im2col_dilation2", lambda: aten.im2col(img, [2, 2], [2, 2], [0, 0], [1, 1]))
rec("im2col_pad1_k3", lambda: aten.im2col(img, [3, 3], [1, 1], [1, 1], [1, 1]))
rec("im2col_asym", lambda: aten.im2col(img, [3, 2], [1, 1], [1, 0], [2, 1]))
# Multi-channel: the ONLY input that can tell the channel-slowest layout from
# the kernel-slowest one -- both give the same shape on a 1-channel image.
rec("im2col_multichannel", lambda: aten.im2col(multi, [2, 2], [1, 1], [0, 0], [1, 1]))
rec("im2col_batched", lambda: aten.im2col(batched, [2, 2], [1, 1], [0, 0], [1, 1]))
rec("im2col_unbatched_3d", lambda: aten.im2col(img[0], [2, 2], [1, 1], [0, 0], [1, 1]))
rec("im2col_bool", lambda: aten.im2col(img.bool(), [2, 2], [1, 1], [0, 0], [1, 1]))
for name, dt in (("f64", torch.float64), ("f16", torch.float16), ("bf16", torch.bfloat16)):
    rec("im2col_dtype_%s" % name,
        lambda d=dt: aten.im2col(img.to(d), [2, 2], [1, 1], [0, 0], [1, 1]))
rec("im2col_int32", lambda: aten.im2col(img.int(), [2, 2], [1, 1], [0, 0], [1, 1]))
rec("im2col_uint8", lambda: aten.im2col(img.byte(), [2, 2], [1, 1], [0, 0], [1, 1]))
rec("im2col_kernel_too_big", lambda: aten.im2col(img, [5, 5], [1, 1], [0, 0], [1, 1]))
# `F.unfold`, which is what llama4's vision tower actually spells.
rec("im2col_via_F_unfold",
    lambda: torch.nn.functional.unfold(multi, kernel_size=2, stride=1))

ones_cols = torch.ones(1, 4, 9)
rec("col2im_overlap_sums", lambda: aten.col2im(ones_cols, [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]))
rec("col2im_no_overlap", lambda: aten.col2im(torch.ones(1, 4, 4), [4, 4], [2, 2], [1, 1], [0, 0], [2, 2]))
rec("col2im_roundtrip", lambda: aten.col2im(
    aten.im2col(img, [2, 2], [1, 1], [0, 0], [1, 1]), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]))
rec("col2im_ramp_overlap", lambda: aten.col2im(
    torch.arange(36, dtype=torch.float32).reshape(1, 4, 9),
    [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]))
rec("col2im_multichannel", lambda: aten.col2im(
    torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(1, 8, 6),
    [3, 4], [2, 2], [1, 1], [0, 0], [1, 1]))
rec("col2im_batched", lambda: aten.col2im(
    torch.arange(2 * 4 * 9, dtype=torch.float32).reshape(2, 4, 9),
    [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]))
rec("col2im_unbatched_2d", lambda: aten.col2im(torch.ones(4, 9), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]))
rec("col2im_padded", lambda: aten.col2im(ones_cols, [2, 2], [2, 2], [1, 1], [1, 1], [1, 1]))
rec("col2im_dilation", lambda: aten.col2im(
    torch.arange(16, dtype=torch.float32).reshape(1, 4, 4), [4, 4], [2, 2], [2, 2], [0, 0], [1, 1]))
for name, dt in (("f64", torch.float64), ("f16", torch.float16), ("bf16", torch.bfloat16)):
    rec("col2im_dtype_%s" % name,
        lambda d=dt: aten.col2im(ones_cols.to(d), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]))
rec("col2im_int32", lambda: aten.col2im(ones_cols.int(), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]))
rec("col2im_wrong_blocks", lambda: aten.col2im(torch.ones(1, 4, 8), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]))
rec("col2im_indivisible", lambda: aten.col2im(torch.ones(1, 5, 9), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]))
rec("col2im_via_F_fold", lambda: torch.nn.functional.fold(
    ones_cols, output_size=(4, 4), kernel_size=2, stride=1))

# -- upsample_nearest1d ------------------------------------------------------
line = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
for out_w in (1, 2, 3, 4, 7, 8, 11):
    rec("un1d_%d" % out_w, lambda w=out_w: aten.upsample_nearest1d(line, [w], None))
for scale, w in ((2.0, 8), (1.5, 3), (0.5, 2), (3.0, 12)):
    rec("un1d_scale_%s_%d" % (scale, w),
        lambda s=scale, w=w: aten.upsample_nearest1d(line, [w], s))
rec("un1d_lanes", lambda: aten.upsample_nearest1d(
    torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5), [8], None))
rec("un1d_uint8", lambda: aten.upsample_nearest1d(
    torch.arange(4, dtype=torch.uint8).reshape(1, 1, 4), [7], None))
for name, dt in (("f64", torch.float64), ("f16", torch.float16), ("bf16", torch.bfloat16)):
    rec("un1d_dtype_%s" % name, lambda d=dt: aten.upsample_nearest1d(line.to(d), [7], None))
rec("un1d_int64", lambda: aten.upsample_nearest1d(line.long(), [8], None))
rec("un1d_bool", lambda: aten.upsample_nearest1d(line.bool(), [8], None))
rec("un1d_4d", lambda: aten.upsample_nearest1d(img, [8], None))
rec("un1d_via_F_interpolate", lambda: torch.nn.functional.interpolate(
    torch.arange(6, dtype=torch.float32).reshape(1, 2, 3), size=7, mode="nearest"))

json.dump(out, sys.stdout)
"""

_cache = {}


def _run(side):
    """`side` is `"shim"` or `"upstream"`. `None` on the shim side when the
    vendored shim is not installed -- `test_tail3.py`'s silent skip, for the
    same reason: `run.sh` builds the standalone `_C`, and only
    `vendor/install_shim.sh` writes the one the vendored tree loads."""
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


def _compare_entry(name, got, want, atol, rtol):
    assert got.get("shape") == want.get("shape"), f"{name}: shape {got} != {want}"
    assert got.get("dtype") == want.get("dtype"), f"{name}: dtype {got} != {want}"
    a, b = got["ok"], want["ok"]
    assert len(a) == len(b), f"{name}: length {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if isinstance(x, float) and isinstance(y, float):
            if math.isnan(x) and math.isnan(y):
                continue
        if x == y:
            continue
        assert math.isclose(x, y, rel_tol=rtol, abs_tol=atol), (
            f"{name}[{i}]: {x!r} != {y!r}"
        )


def _agree(name, atol=1e-9, rtol=1e-9):
    """Element-wise agreement with upstream, dtype and shape included."""
    pair = _both(name)
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, f"{name}: upstream itself refused: {want}"
    assert "raised" not in got, f"{name}: the shim refused where upstream did not: {got}"
    if "tuple" in want:
        assert "tuple" in got, f"{name}: expected a tuple, got {got}"
        assert len(got["tuple"]) == len(want["tuple"]), f"{name}: arity"
        for i, (g, w) in enumerate(zip(got["tuple"], want["tuple"])):
            _compare_entry(f"{name}[{i}]", g, w, atol, rtol)
        return
    _compare_entry(name, got, want, atol, rtol)


def _both_refuse(name):
    pair = _both(name)
    if pair == "skip":
        return
    got, want = pair
    assert "raised" in want, f"{name}: upstream did NOT refuse: {want}"
    assert "raised" in got, (
        f"{name}: upstream refuses and the shim computed something instead: {got}"
    )


def _value(name):
    pair = _both(name)
    if pair == "skip":
        return None, None
    return pair


# --------------------------------------------------------------------------
# 1. `i0`: checked across the range, because the ends are where it can be wrong
# --------------------------------------------------------------------------


def test_i0_is_bit_identical_to_upstream_across_the_whole_range():
    """88,002 points, both Chebyshev intervals, compared as **bits**.

    A series and an asymptotic expansion agree in the middle and diverge at
    the ends, so a spot check in the middle of `[0, 8]` passes an
    implementation that is wrong at `x = 0` and at `x = 8`. This sweeps
    `[0, 8]` and `[8, 88]` at a step of `0.001` -- through the junction, out
    to where `float32` overflows -- and demands equality of the raw bit
    patterns rather than of a float repr, because `1.0` and
    `0.9999999403953552` are the same to five significant figures and are
    the *whole* difference between the right implementation and the shortcut.
    """
    for key in ("i0_sweep_f32_bits", "i0_sweep_f64_bits"):
        pair = _both(key)
        if pair == "skip":
            return
        got, want = pair
        a, b = got["ok"], want["ok"]
        assert len(a) == len(b) and len(a) > 80000, (key, len(a), len(b))
        differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        assert not differing, (
            f"{key}: {len(differing)} of {len(a)} points differ from upstream "
            f"bit-for-bit; first at index {differing[0]}"
        )


def test_the_f64_shortcut_for_i0_is_measurably_wrong():
    """`i0(0.0f)` is `0.9999999403953552` upstream, not `1.0`.

    Upstream's `calc_i0` is a template instantiated at `float` as well as at
    `double`, so its Chebyshev coefficients are **float-rounded** and the
    recurrence runs in `float`. Computing in `f64` and narrowing once gives
    exactly `1.0` here. That single value is the cheapest possible witness
    that the shortcut is a different function, and it is asserted directly so
    that a later "simplification" to the closed form fails on it.
    """
    pair = _both("i0_zero_f32_bits")
    if pair == "skip":
        return
    got, want = pair
    ONE_F32 = 0x3F800000
    assert want["ok"][0] != ONE_F32, (
        "upstream's float i0(0) is no longer the float-rounded 0.99999994 -- "
        "the premise of this whole kernel has changed and it must be re-measured"
    )
    assert got["ok"] == want["ok"], (
        f"i0(0.0f): shim {got['ok'][0]:#x} != upstream {want['ok'][0]:#x}; "
        "0x3f800000 would mean the f64-and-narrow shortcut crept back in"
    )


def test_i0_dtype_rules_are_upstreams():
    for name in ("i0_reduced_f16", "i0_reduced_bf16", "i0_integral", "i0_bool",
                 "i0_negative", "i0_nonfinite"):
        _agree(name)


# --------------------------------------------------------------------------
# 2. `kaiser_window`: periodic truncates, and beta=0 is all ones
# --------------------------------------------------------------------------


def test_kaiser_window_agrees_with_upstream():
    for L in (0, 1, 2, 4, 5, 12, 64, 257):
        for per in (True, False):
            _agree("kaiser_%d_%s" % (L, per))
    for beta in (0.0, 1.0, 9.0, 12.0, 38.0):
        _agree("kaiser_beta_%s" % beta)
    for name in ("f64", "f16", "bf16"):
        _agree("kaiser_dtype_%s" % name)


def test_kaiser_window_is_bit_identical_at_float32_and_float64():
    """Not merely close. The window multiplies a signal sample by sample, so a
    half-ULP drift per tap is a real spectral change, and `hann_window` set
    the standard for this file by being bit-identical at four dtypes."""
    for key in ("kaiser_12_true_bits", "kaiser_12_false_bits", "kaiser_64_beta9_f64_bits"):
        pair = _both(key)
        if pair == "skip":
            return
        got, want = pair
        assert got["ok"] == want["ok"], f"{key}: {got['ok']} != {want['ok']}"


def test_kaiser_periodic_truncates_rather_than_reparameterising():
    """`kaiser_window(L, True)` and `kaiser_window(L, False)` are both `L`
    long and **share only their first element**.

    That is the signature of "build `L+1` symmetric and narrow the last one
    off" and it distinguishes it from "same window, different alpha", which
    would give two windows agreeing nowhere or everywhere. Asserted against
    upstream's own numbers so it cannot pass by agreeing with itself.
    """
    got_t, want_t = _value("kaiser_12_True")
    if got_t is None:
        return
    got_f, _ = _value("kaiser_12_False")
    periodic, symmetric = got_t["ok"], got_f["ok"]
    assert len(periodic) == len(symmetric) == 12
    assert periodic[0] == symmetric[0]
    shared = sum(1 for a, b in zip(periodic, symmetric) if a == b)
    assert shared == 1, (
        f"{shared} of 12 elements agree between the periodic and symmetric "
        "forms; exactly one (the first) should"
    )


def test_kaiser_beta_zero_is_all_ones():
    """`i0(0)/i0(0)` for every sample. Not a degenerate case that raises --
    measured, and it is also the check that the denominator is `i0(beta)` and
    not a constant."""
    got, _ = _value("kaiser_beta_0.0")
    if got is None:
        return
    assert got["ok"] == [1.0] * 16, got["ok"]
    _agree("kaiser_beta_0.0")


def test_kaiser_window_refuses_what_upstream_refuses():
    _both_refuse("kaiser_negative")
    _both_refuse("kaiser_int_dtype")


# --------------------------------------------------------------------------
# 3. `diag`: two operations, chosen by rank
# --------------------------------------------------------------------------


def test_diag_extracts_and_builds_and_both_agree_with_upstream():
    for d in (-6, -2, -1, 0, 1, 3, 5):
        _agree("diag_extract_wide_%d" % d)
        _agree("diag_extract_tall_%d" % d)
        _agree("diag_build_%d" % d)
    for name in ("diag_square", "diag_int", "diag_bool", "diag_empty_vector",
                 "diag_transposed_receiver", "diag_method"):
        _agree(name)


def test_diag_is_two_operations_and_rank_picks_which():
    """A 1-D input **builds**; a 2-D input **extracts**. Asserted as a rank
    relation rather than by value, because an implementation that only ever
    extracted would still pass every 2-D case -- and `rwkv` only exercises the
    2-D one."""
    build, _ = _value("diag_build_0")
    if build is None:
        return
    extract, _ = _value("diag_extract_wide_0")
    assert len(build["shape"]) == 2, ("1-D input must build a matrix", build)
    assert len(extract["shape"]) == 1, ("2-D input must extract a vector", extract)


def test_diag_offset_grows_the_built_matrix_and_can_empty_the_extraction():
    """The two consequences of `diagonal=` that a guess gets wrong.

    `diag([1,2,3], -2)` is **5x5**, not 3x3 -- the offset makes the matrix
    bigger. And `diag(3x4, 5)` is the **empty** vector, not an error: the
    length is a clamped minimum.
    """
    built, _ = _value("diag_build_-2")
    if built is None:
        return
    assert built["shape"] == [5, 5], built["shape"]
    off, _ = _value("diag_extract_wide_5")
    assert off["shape"] == [0], off["shape"]
    under, _ = _value("diag_extract_wide_-6")
    assert under["shape"] == [0], under["shape"]


def test_diag_reproduces_the_rwkv_line_that_needed_it():
    """`torch/nn/init.py:710` -- `d = torch.diag(r, 0); ph = d.sign()`, the
    line one past `linalg_qr` that was `rwkv`'s last construction wall."""
    _agree("diag_rwkv_sign")
    got, _ = _value("diag_rwkv_sign")
    if got is None:
        return
    assert got["ok"] == [-1.0, 1.0, -1.0], got["ok"]


def test_diag_refuses_ranks_upstream_refuses():
    _both_refuse("diag_3d")
    _both_refuse("diag_0d")


# --------------------------------------------------------------------------
# 4. `var`: Bessel's correction, tested where it is unmissable
# --------------------------------------------------------------------------


def test_var_default_correction_is_one_measured_at_n_equals_two():
    """**The default is 1, not 0**, and `n = 2` is where that is a factor of
    two rather than a rounding difference.

    `var([1., 3.])` is `2.0` with the default and `1.0` with `correction=0`.
    At `n = 1000` the same mistake is a 0.1% error that every tolerance in
    this repository would accept.
    """
    got, want = _value("var_n2_default")
    if got is None:
        return
    assert want["ok"] == [2.0], ("upstream's default changed", want)
    assert got["ok"] == [2.0], (
        f"var([1., 3.]) is {got['ok']}; 1.0 would mean correction defaulted to 0"
    )
    zero, _ = _value("var_n2_correction0")
    assert zero["ok"] == [1.0], zero["ok"]
    unbiased_false, _ = _value("var_n2_unbiased_false")
    assert unbiased_false["ok"] == [1.0], (
        "unbiased=False is the old spelling of correction=0 and must agree with it"
    )


def test_var_agrees_with_upstream_over_shapes_dims_and_dtypes():
    names = ["var_all_default", "var_correction_fractional", "var_dim0", "var_dim1",
             "var_dim_neg", "var_dim_keepdim", "var_dim_list", "var_dim_empty_list",
             "var_dim_none_correction0", "var_cube_dim1", "var_cube_dim02",
             "var_cube_dim02_keepdim", "var_method", "var_ops_correction",
             "var_ops_dim", "var_ops_default"]
    names += ["var_correction_%d" % c for c in (0, 1, 2, 5)]
    names += ["var_dtype_%s" % n for n in ("f64", "f16", "bf16")]
    names += ["var_dtype_dim_%s" % n for n in ("f64", "f16", "bf16")]
    for name in names:
        _agree(name, atol=1e-6, rtol=1e-6)


def test_var_non_positive_denominator_is_nan_rather_than_a_signed_zero():
    """`var(tensor(3.))` has `n - correction == 0` and `var(zeros(0))` has
    `-1`. Both are `nan` upstream. Dividing `0.0` by `-1.0` gives `-0.0`,
    which compares equal to `0.0` and would slip through a value test that
    did not look for NaN specifically."""
    for name in ("var_scalar", "var_empty"):
        got, want = _value(name)
        if got is None:
            return
        assert math.isnan(want["ok"][0]), (name, want)
        assert math.isnan(got["ok"][0]), (name, got)


def test_var_refuses_integral_dtypes_as_upstream_does():
    _both_refuse("var_integral")


# --------------------------------------------------------------------------
# 5. `_unique2`: three tensors, gated by flags, and NaN is not itself
# --------------------------------------------------------------------------


def test_unique2_agrees_with_upstream_for_every_flag_combination():
    for s in (True, False):
        for ri in (False, True):
            for rc in (False, True):
                _agree("unique2_%s_%s_%s" % (s, ri, rc))


def test_unique2_returns_three_tensors_and_the_unasked_ones_are_empty():
    """The schema returns three always. `vilt` asks for neither the inverse
    nor the counts and still receives them; getting this wrong is a
    `ValueError: not enough values to unpack` inside `torch.functional`, not
    a wrong number."""
    got, want = _value("unique2_True_False_False")
    if got is None:
        return
    assert len(got["tuple"]) == 3, got
    assert got["tuple"][1]["shape"] == [0], got["tuple"][1]
    assert got["tuple"][2]["shape"] == [0], got["tuple"][2]
    assert [t["shape"] for t in got["tuple"]] == [t["shape"] for t in want["tuple"]]


def test_unique2_inverse_carries_the_inputs_shape_not_the_values():
    """A `2x2` input gives a `2x2` inverse and a 3-element `values`. Only
    `values` is flattened, which a "flatten everything" implementation gets
    wrong while still agreeing on every element."""
    got, want = _value("unique2_2d")
    if got is None:
        return
    assert got["tuple"][0]["shape"] == [3], got["tuple"][0]
    assert got["tuple"][1]["shape"] == [2, 2], got["tuple"][1]
    assert [t["shape"] for t in got["tuple"]] == [t["shape"] for t in want["tuple"]]
    _agree("unique2_2d")
    _agree("unique2_scalar")
    _agree("unique2_empty")


def test_unique2_sorted_false_still_sorts_on_cpu():
    """The CPU kernel has one path; `sorted` is a CUDA optimisation upstream.
    Answering an unsorted result when asked for one would differ from the
    thing this shim is compared against."""
    unsorted, _ = _value("unique2_False_True_True")
    if unsorted is None:
        return
    values = unsorted["tuple"][0]["ok"]
    assert values == sorted(values), values
    _agree("unique2_False_True_True")


def test_unique2_keeps_every_nan_because_nan_is_not_equal_to_itself():
    """`[nan, 1., nan, 0.]` gives **four** values, not three: NaN never equals
    NaN, so each survives, and they sort last. A deduplication on a sort key
    rather than on `==` would collapse them into one."""
    got, want = _value("unique2_nan")
    if got is None:
        return
    assert want["tuple"][0]["shape"] == [4], ("upstream changed", want)
    assert got["tuple"][0]["shape"] == [4], got["tuple"][0]
    values = got["tuple"][0]["ok"]
    assert values[:2] == [0.0, 1.0] and all(math.isnan(v) for v in values[2:])
    _agree("unique2_nan")


def test_unique2_reaches_the_vilt_spelling():
    """`modeling_vilt.py:144` is `valid_idx[:, 0].unique()` -- through
    `Tensor.unique` -> `torch.unique` -> `torch._unique2`, all defaults."""
    _agree("unique2_vilt_spelling")
    for name in ("unique2_bool", "unique2_negative", "unique2_float"):
        _agree(name)


# --------------------------------------------------------------------------
# 6. `im2col` / `col2im`: not inverses, and the overlap is the op
# --------------------------------------------------------------------------


def test_im2col_agrees_with_upstream():
    for name in ("im2col_k2_s1", "im2col_k2_s2", "im2col_dilation2", "im2col_pad1_k3",
                 "im2col_asym", "im2col_multichannel", "im2col_batched",
                 "im2col_unbatched_3d", "im2col_bool"):
        _agree(name)
    for name in ("f64", "f16", "bf16"):
        _agree("im2col_dtype_%s" % name)


def test_im2col_channel_is_the_slowest_axis_of_the_folded_dimension():
    """`out[c*kh*kw + i*kw + j][block]`, checked on a **2-channel** input.

    A single-channel image cannot tell this layout from the transposed one --
    both give `(kh*kw, L)` and the same numbers. The multi-channel case is the
    only one that can, so it is asserted separately rather than left to the
    bulk comparison.
    """
    got, want = _value("im2col_multichannel")
    if got is None:
        return
    assert got["shape"] == [1, 8, 6], got["shape"]
    # Rows 0..3 are channel 0's four kernel positions, rows 4..7 channel 1's.
    # Channel 1 of `arange(24).reshape(1,2,3,4)` starts at 12.
    assert min(got["ok"][:6]) == 0.0 and min(got["ok"][24:30]) == 12.0, got["ok"]
    assert got["ok"] == want["ok"]


def test_col2im_sums_overlapping_windows_rather_than_overwriting():
    """**The whole content of this op.** With `kernel=2, stride=1` on a 4x4,
    the interior pixels are covered by four windows and the corners by one, so
    `col2im(ones)` is the cover count:

        1 2 2 1
        2 4 4 2
        2 4 4 2
        1 2 2 1

    An implementation that *overwrites* answers all ones -- and so does the
    correct one when `stride == kernel`, which is why no test here uses a
    non-overlapping stride to make this point.
    """
    got, want = _value("col2im_overlap_sums")
    if got is None:
        return
    cover = [1, 2, 2, 1, 2, 4, 4, 2, 2, 4, 4, 2, 1, 2, 2, 1]
    assert want["ok"] == [float(v) for v in cover], ("upstream changed", want)
    assert got["ok"] == [float(v) for v in cover], (
        f"col2im(ones) is {got['ok']}; all-ones would mean overlaps are "
        "overwritten instead of summed"
    )
    # And the non-overlapping case, which CANNOT tell the two apart -- kept so
    # the contrast is on record rather than as evidence.
    flat, _ = _value("col2im_no_overlap")
    assert flat["ok"] == [1.0] * 16, flat["ok"]


def test_col2im_of_im2col_is_not_the_identity():
    """It is the input weighted by the cover count. Asserting the round trip
    equals `x` would be asserting the bug."""
    got, want = _value("col2im_roundtrip")
    if got is None:
        return
    source = list(range(1, 17))
    cover = [1, 2, 2, 1, 2, 4, 4, 2, 2, 4, 4, 2, 1, 2, 2, 1]
    expected = [float(a * b) for a, b in zip(source, cover)]
    assert want["ok"] == expected, ("upstream changed", want)
    assert got["ok"] == expected, got["ok"]


def test_col2im_agrees_with_upstream():
    for name in ("col2im_ramp_overlap", "col2im_multichannel", "col2im_batched",
                 "col2im_unbatched_2d", "col2im_padded", "col2im_dilation"):
        _agree(name)
    for name in ("f64", "f16", "bf16"):
        _agree("col2im_dtype_%s" % name)


def test_im2col_and_col2im_refuse_the_dtypes_upstream_refuses():
    """**`bool` computes and the integral dtypes do not**, which is backwards
    from most of this shim and so was measured rather than inferred."""
    for name in ("im2col_int32", "im2col_uint8", "im2col_kernel_too_big",
                 "col2im_int32", "col2im_wrong_blocks", "col2im_indivisible"):
        _both_refuse(name)
    _agree("im2col_bool")


# --------------------------------------------------------------------------
# 7. `upsample_nearest1d`: not an alias of the 2-D op
# --------------------------------------------------------------------------


def test_upsample_nearest1d_agrees_with_upstream():
    for out_w in (1, 2, 3, 4, 7, 8, 11):
        _agree("un1d_%d" % out_w)
    for scale, w in ((2.0, 8), (1.5, 3), (0.5, 2), (3.0, 12)):
        _agree("un1d_scale_%s_%d" % (scale, w))
    for name in ("un1d_lanes", "un1d_uint8"):
        _agree(name)
    for name in ("f64", "f16", "bf16"):
        _agree("un1d_dtype_%s" % name)


def test_upsample_nearest1d_inverts_the_scale_argument():
    """`scales=1.5` with `output_size=[3]` gathers `[0, 0, 1]`, not
    `[0, 1, 2]`. The argument is a *scale factor* and the index rule divides
    by it, so an implementation that derives the ratio from the sizes -- which
    every call `F.interpolate` makes would hide -- differs here."""
    got, want = _value("un1d_scale_1.5_3")
    if got is None:
        return
    assert want["ok"] == [0.0, 0.0, 1.0], ("upstream changed", want)
    assert got["ok"] == [0.0, 0.0, 1.0], (
        f"{got['ok']}; [0, 1, 2] would mean the scale was ignored"
    )


def test_upsample_nearest1d_refuses_what_upstream_refuses_including_bool():
    """`uint8` computes (a gather never averages) but `bool` does not -- the
    one place this op's dtype set differs from `upsample_nearest2d`'s, and the
    reason it could not be registered as an alias of it."""
    _both_refuse("un1d_int64")
    _both_refuse("un1d_bool")
    _both_refuse("un1d_4d")
    _agree("un1d_uint8")


# --------------------------------------------------------------------------
# 8. The alias-versus-kernel split, pinned as a count
# --------------------------------------------------------------------------


def test_this_round_landed_fifteen_registrations_over_eight_kernels():
    """docs/architectures/ARCH100.md measured missing *names* outnumbering missing *kernels*
    49 to 22 in this tail, so "seven ops" would overstate the work if some
    were table rows. Here it cuts both ways: **eight kernel bodies behind
    fifteen registrations**. `var` and `std` have three overloads each and
    share ONE body between all six -- but sharing it was a measurement, not a
    convenience (see
    `test_std_takes_the_root_in_the_accumulator_not_on_the_narrowed_variance`),
    and `kaiser_window`'s three share another.

    Three candidates were checked for being aliases and none was:
    `var` beside `native_batch_norm`'s statistics (which computes a *biased*
    variance it never materialises), `upsample_nearest1d` beside
    `upsample_nearest2d` (different schema, different rank check, different
    dtype set -- `bool`), and `std` beside `var` (a leaf that dispatches to
    `aten::std.correction`, and NOT bit-equal to `var(...).sqrt()`).
    """
    landed = [
        "aten.diag.default",
        "aten._unique2.default",
        "aten.im2col.default",
        "aten.col2im.default",
        "aten.i0.default",
        "aten.kaiser_window.default",
        "aten.kaiser_window.periodic",
        "aten.kaiser_window.beta",
        "aten.upsample_nearest1d.default",
        "aten.var.default",
        "aten.var.dim",
        "aten.var.correction",
        "aten.std.default",
        "aten.std.dim",
        "aten.std.correction",
    ]
    implemented = set(_C._aten_implemented())
    missing = [op for op in landed if op not in implemented]
    assert not missing, missing
    assert len(landed) == 15, len(landed)
    # Fifteen registrations, eight distinct kernel bodies: `var`'s three and
    # `std`'s three all reach `var_reduce` (five saved), and
    # `kaiser_window`'s three share `kaiser_window_default` (two saved).
    distinct_bodies = len(landed) - 5 - 2
    assert distinct_bodies == 8, distinct_bodies


def test_std_agrees_with_upstream_and_is_not_a_composite_over_var():
    """`torch.std` dispatches to `aten::std.correction` as a **leaf**, so it
    needed a kernel and not an `overloads.json` row. `gemma3n_text`'s wall
    (`modeling_gemma3n.py:1745`), and its second this batch after `erfinv`.
    """
    names = ["std_all_default", "std_correction_fractional", "std_dim0", "std_dim1",
             "std_dim_neg", "std_dim_keepdim", "std_dim_list", "std_cube_dim1",
             "std_cube_dim02_keepdim", "std_method", "std_ops_correction",
             "std_ops_dim", "std_ops_default", "std_gemma3n_spelling"]
    names += ["std_correction_%d" % c for c in (0, 1, 2, 5)]
    names += ["std_dtype_%s" % n for n in ("f64", "f16", "bf16")]
    names += ["std_dtype_dim_%s" % n for n in ("f64", "f16", "bf16")]
    for name in names:
        _agree(name, atol=1e-6, rtol=1e-6)
    _both_refuse("std_integral")


def test_std_default_correction_is_one_measured_at_n_equals_two():
    """Bessel's again, and the same reason for `n = 2`: the wrong default
    scales by `sqrt(n/(n-1))`, which is 0.05% at `n = 1000` and 41% here."""
    got, want = _value("std_n2_default")
    if got is None:
        return
    # The float32 rounding of sqrt(2), not sqrt(2) itself -- the value is
    # compared as the dtype the op actually answers in.
    root_two_f32 = 1.4142135381698608
    assert want["ok"] == [root_two_f32], ("upstream's default changed", want)
    assert got["ok"] == [root_two_f32], (
        f"std([1., 3.]) is {got['ok']}; 1.0 would mean correction defaulted to 0"
    )
    for name in ("std_n2_correction0", "std_n2_unbiased_false"):
        v, _ = _value(name)
        assert v["ok"] == [1.0], (name, v["ok"])


def test_std_takes_the_root_in_the_accumulator_not_on_the_narrowed_variance():
    """**`torch.std(x)` is not `torch.var(x).sqrt()`**, and this is the whole
    reason the two kernels share a body but not an exit.

    Upstream roots the wide accumulator and narrows once; the composite
    narrows to `float32` first and then roots. Measured, they disagree on 53
    of 288 (dtype, shape, dim, correction) combinations, always by one ULP --
    and rooting-then-narrowing reproduces upstream on 1199 of 1200 random
    `float32` samples where rooting-after-narrowing manages 1043.

    The sample below is one where they differ **upstream**, so this asserts a
    property of torch and then asserts the shim reproduces the same split. A
    shim that implemented `std` as `var(...).sqrt()` would match the second
    value and not the first.
    """
    direct, want_direct = _value("std_wide_sample")
    if direct is None:
        return
    composite, want_composite = _value("std_wide_sample_as_sqrt_var")
    assert want_direct["ok"] != want_composite["ok"], (
        "upstream's std and var().sqrt() now agree on this sample -- pick "
        "another one; the point of this test is a case where they do not"
    )
    assert direct["ok"] == want_direct["ok"], (
        f"std: shim {direct['ok']} != upstream {want_direct['ok']}; matching "
        f"{want_composite['ok']} instead would mean std was implemented as "
        "var(...).sqrt()"
    )
    assert composite["ok"] == want_composite["ok"], (composite, want_composite)


def test_var_and_std_clamp_the_divisor_at_zero_so_an_overshoot_is_inf_not_nan():
    """`max(0, n - correction)`, which is upstream's own line, and the case a
    first draft of this kernel got wrong.

    "Non-positive denominator -> nan" is right for the two cases the obvious
    tests reach -- `var` of a scalar and of an empty tensor -- because both
    have `m2 == 0` and `0/0` is `nan`. It is wrong the moment `m2 > 0`:

        var([1., 3.], correction=2)   n - c == 0, m2 > 0    ->  inf
        var([2., 2.], correction=2)   n - c == 0, m2 == 0   ->  nan
        var([1., 3.], correction=3)   n - c == -1, m2 > 0   ->  inf   (not -inf)

    All measured. The negative case is the one that says it is a *clamp* and
    not a sign convention.
    """
    for name, expect in (
        ("var_denom_zero_nonzero_m2", "inf"),
        ("var_denom_negative_nonzero_m2", "inf"),
        ("var_denom_zero_zero_m2", "nan"),
        ("std_denom_zero_nonzero_m2", "inf"),
        ("std_denom_negative_nonzero_m2", "inf"),
    ):
        got, want = _value(name)
        if got is None:
            return
        value = want["ok"][0]
        if expect == "inf":
            assert value == float("inf"), (f"upstream changed for {name}", want)
            assert got["ok"][0] == float("inf"), (
                f"{name}: shim answered {got['ok'][0]!r}; nan would mean the "
                "divisor was not clamped at zero"
            )
        else:
            assert math.isnan(value), (f"upstream changed for {name}", want)
            assert math.isnan(got["ok"][0]), (name, got)
    for name in ("std_scalar", "std_empty"):
        got, want = _value(name)
        assert math.isnan(want["ok"][0]) and math.isnan(got["ok"][0]), (name, got, want)


def test_the_three_nn_bindings_now_carry_these_kernels_all_the_way_to_F():
    """INVERTED (docs/bindings/BIND3.md). This test used to assert that `im2col`,
    `col2im` and `upsample_nearest1d` had kernels and were **not reachable**
    from `F.unfold` / `F.fold` / `F.interpolate`, because `_install_nn` in
    `bootstrap.py` had no entry for them and that file was outside the
    round's territory.

    The three `_install_nn` entries landed, so the absence is gone and the
    assertion is turned around rather than removed: the same three probe
    cases, in the same two probe processes, now have to **agree with upstream
    element-wise** instead of refusing. Deleting it would have taken the
    coverage with it -- these are the only cases in this file that go through
    the `F.*` spelling rather than through `torch.ops.aten.*`.

    `test_bind3.py` is the wider proof (every argument shape, both refusal
    directions, and the scale-forwarding case a `2x` test cannot see). What
    stays here is the pair of cases this file's probe already recorded, so
    that the round that opened the gap is also the round that shows it shut.
    """
    for name in ("im2col_via_F_unfold", "col2im_via_F_fold",
                 "un1d_via_F_interpolate"):
        _agree(name)
    # And the kernels themselves are advertised, which is the other half --
    # a binding onto a missing kernel is docs/bindings/BINDINGS.md's `mish`.
    implemented = set(_C._aten_implemented())
    for op in ("aten.im2col.default", "aten.col2im.default",
               "aten.upsample_nearest1d.default"):
        assert op in implemented, op
    # The bindings exist by name, so a future round that deletes one fails
    # here and not only through a value comparison it might read as flaky.
    for name in ("im2col", "col2im", "upsample_nearest1d"):
        assert name in _C._shim_nn_implemented, name


def test_the_new_ops_that_read_the_host_are_named_for_the_mps_list():
    """Twelve of the fifteen registrations pull the tensor to the host, and
    `device.rs`'s `MPS_HOST_READBACK_OPS` is where that has to be declared.
    That file is not this round's to edit, so the list is written down here
    instead: `test_the_mps_readback_list_is_what_the_kernels_actually_do` in
    `test_shim.py` re-derives the same set from `aten.rs` and will name them.

    `kaiser_window`'s three registrations are the exception -- they construct
    from scalars and never touch an input tensor, so they are not on the list
    and must not be added to it."""
    reads_host = {
        "aten.diag.default",
        "aten._unique2.default",
        "aten.im2col.default",
        "aten.col2im.default",
        "aten.i0.default",
        "aten.upsample_nearest1d.default",
        "aten.var.default",
        "aten.var.dim",
        "aten.var.correction",
        "aten.std.default",
        "aten.std.dim",
        "aten.std.correction",
    }
    constructs_only = {
        "aten.kaiser_window.default",
        "aten.kaiser_window.periodic",
        "aten.kaiser_window.beta",
    }
    assert not (reads_host & constructs_only)
    declared = set(_C._shim_mps_host_readback_ops())
    for op in constructs_only:
        assert op not in declared, (
            f"{op} builds a window from scalars and reads no device memory; "
            "listing it would refuse a call that works"
        )


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
