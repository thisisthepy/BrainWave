"""The four binding debts: a kernel that exists and a spelling that reaches it.

`tools/golden/compare.py` calls `_C._aten_dispatch(key, ...)` with a key it
took from its own case table, so it proves the *kernel* and is structurally
blind to whether anything a user writes arrives there (docs/REACH.md). Three
kernels in this tree were golden-green and unreachable for exactly that reason
-- each needed a composite in `bootstrap.py`, and `bootstrap.py` belonged to a
different round every time. `tools/golden/reach_allow.json` carried two of them
by name with the reason "OWED, not deliberate".

So every test here goes through **the spelling a user writes** --
`torch.linalg.qr(A)`, `F.interpolate(x, mode="nearest")`,
`torch.linalg.norm(x, ord=2, dim=-1)` -- in a subprocess with the vendored
tree on `PYTHONPATH`, and compares element-wise against **upstream torch run
in a separate subprocess** with `PYTHONPATH` stripped. Not against frozen
constants: a constant records what somebody once saw, and the thing being
tested here is agreement with a library that is installed and can be asked.

Both subprocesses print their own marker (`shim` / `upstream`) and this file
asserts it, so the suite cannot pass by having run the same library twice --
which is the failure shape that makes a comparison test meaningless while
looking perfect.

`mish` is the fourth debt and it is **not** here. docs/BINDINGS.md §5 says why:
its kernel was written, matched to 1 ULP, and then removed, so there is no
`aten.mish.default` in `_aten_implemented()` for a binding to reach. A
`_nn.mish` composite would be a door onto nothing.
"""

import json
import os
import subprocess
import sys

from test_shim import _C

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


# One script, run twice: once against the vendored (shim-backed) `torch` and
# once against upstream. Sharing the source is the point -- a shim-side script
# and an upstream-side script that had drifted apart would compare two
# different computations and call the difference agreement.
_SCRIPT = r"""
import json, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

out = {"marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


def rec(name, fn):
    try:
        v = fn()
    except BaseException as e:
        out[name] = {"error": type(e).__name__, "msg": str(e)}
        return
    if isinstance(v, torch.Tensor):
        out[name] = {"shape": list(v.shape), "flat": [float(x) for x in v.flatten().tolist()]}
    else:
        out[name] = v


# -- torch.linalg.qr, which is _add_docstr(_C._linalg.linalg_qr, ...) --------
A = torch.tensor([[12., -51., 4.], [6., 167., -68.], [-4., 24., -41.]], dtype=torch.float64)
TALL = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
# Written out rather than `torch.eye`, which this shim has no table row for.
EYE = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float64)

rec("qr_Q", lambda: torch.linalg.qr(A)[0])
rec("qr_R", lambda: torch.linalg.qr(A)[1])
rec("qr_named_Q", lambda: torch.linalg.qr(A).Q)
rec("qr_named_R", lambda: torch.linalg.qr(A).R)
rec("qr_tall_Q", lambda: torch.linalg.qr(TALL)[0])
rec("qr_tall_R", lambda: torch.linalg.qr(TALL)[1])
rec("qr_complete_Q", lambda: torch.linalg.qr(TALL, mode="complete")[0])
rec("qr_complete_R", lambda: torch.linalg.qr(TALL, mode="complete")[1])
# The `dlarfg` xnorm==0 branch: get it wrong and this is -I, orthogonal and
# wrong (docs/TAIL1.md §5).
rec("qr_eye_R", lambda: torch.linalg.qr(EYE)[1])
# mode="r" answers a *one-dimensional* empty Q, not (m, 0) -- measured.
rec("qr_r_mode_Q_shape", lambda: list(torch.linalg.qr(TALL, mode="r")[0].shape))
rec("qr_r_mode_R", lambda: torch.linalg.qr(TALL, mode="r")[1])
rec("qr_bad_mode", lambda: torch.linalg.qr(A, mode="banana"))
rec("qr_long", lambda: torch.linalg.qr(torch.ones(2, 2, dtype=torch.int64)))

# -- F.interpolate(mode="nearest") -> torch._C._nn.upsample_nearest2d -------
x = torch.arange(15., dtype=torch.float32).reshape(1, 1, 3, 5)
rec("near_scale", lambda: F.interpolate(x, scale_factor=1.5, mode="nearest"))
rec("near_size", lambda: F.interpolate(x, size=(7, 11), mode="nearest"))
rec("near_2x", lambda: F.interpolate(x, scale_factor=2, mode="nearest"))
# The binding itself, in both of upstream's shapes: `.vec` (three arguments,
# output_size may be None) and the leaf (four).
rec("near_vec", lambda: torch._C._nn.upsample_nearest2d(x, None, [1.5, 1.5]))
rec("near_leaf", lambda: torch._C._nn.upsample_nearest2d(x, [4, 7], 1.5, 1.5))
rec("near_leaf_noscale", lambda: torch._C._nn.upsample_nearest2d(x, [7, 11], None, None))
rec("near_both", lambda: torch._C._nn.upsample_nearest2d(x, [4, 7], [1.5, 1.5]))
# The cases that separate a forwarded scale from a recomputed one. On the
# `3x5 -> 4x7` grid above, `1/scale` and `in/out` happen to agree, so a
# composite that dropped the scale factors entirely passed every case before
# this line -- measured, by mutating the composite. `4x7` at factor 1.7 gives
# `6x11`, where `1/1.7 = 0.588` and `in/out = 0.667` are different grids.
#
# NOT a 2x case, and that is deliberate: upstream's `nearest_idx` short
# circuits when `output_size == input_size` or `output_size == 2 * input_size`
# and IGNORES the scale entirely in those two branches (measured -- `3x5 ->
# 6x10` answers the same thing for scales 1.5, 3.0, 0.5 and None). So a 2x
# case cannot distinguish forwarding from recomputing on either side.
x2 = torch.arange(28., dtype=torch.float32).reshape(1, 1, 4, 7)
rec("near_vec_17", lambda: torch._C._nn.upsample_nearest2d(x2, None, [1.7, 1.7]))
rec("near_leaf_17", lambda: torch._C._nn.upsample_nearest2d(x2, [6, 11], 1.7, 1.7))
rec("near_leaf_17_recomputed", lambda: torch._C._nn.upsample_nearest2d(
    x2, [6, 11], None, None))

# -- torch.linalg.norm -> torch._C._linalg.linalg_norm ----------------------
# The four spellings owlv2/owlvit reach, and the flattened default.
y = (torch.arange(12., dtype=torch.float32) + 1.).reshape(3, 4)
rec("norm_ord2_dim", lambda: torch.linalg.norm(y, ord=2, dim=-1, keepdim=True))
rec("norm_default_dim", lambda: torch.linalg.norm(y, dim=-1, keepdim=True))
rec("norm_nokeepdim", lambda: torch.linalg.norm(y, dim=-1))
rec("norm_flat", lambda: torch.linalg.norm(y))
rec("norm_dim0", lambda: torch.linalg.norm(y, ord=2, dim=0))
rec("norm_ord1", lambda: torch.linalg.norm(y, ord=1, dim=-1))
rec("norm_inf", lambda: torch.linalg.norm(y, ord=float("inf"), dim=-1))
rec("norm_neginf", lambda: torch.linalg.norm(y, ord=float("-inf"), dim=-1))
rec("norm_1d_ord3", lambda: torch.linalg.norm(y[0], ord=3))
rec("norm_dtype", lambda: torch.linalg.norm(y, ord=2, dim=-1, dtype=torch.float64))
# The refusals. Upstream *answers* these (they are matrix norms); the shim has
# no matrix-norm kernel and says so by name. Recorded from both sides on
# purpose -- the divergence is the point, and pinning it here is what makes it
# visible the day a matrix-norm kernel lands.
rec("norm_fro", lambda: torch.linalg.norm(y, ord="fro"))
rec("norm_nuc", lambda: torch.linalg.norm(y, ord="nuc"))
rec("norm_matrix_ord2", lambda: torch.linalg.norm(y, ord=2))
rec("norm_matrix_dim", lambda: torch.linalg.norm(y, ord=1, dim=(0, 1)))

# -- the composite user: nn.init.orthogonal_, rwkv's construction wall ------
def _orthogonal():
    w = torch.empty(4, 4)
    nn.init.orthogonal_(w)
    # Report the property, not the values: the input is drawn from an RNG and
    # the two libraries do not share one, so element-wise comparison here would
    # be meaningless. Orthonormality is what `orthogonal_` promises.
    prod = w @ w.t()
    ident = torch.tensor([[1. if i == j else 0. for j in range(4)] for i in range(4)])
    err = float((prod - ident).abs().max().item())
    return {"ok": err < 1e-4, "err": err}

rec("orthogonal_", _orthogonal)

# `mish`: asserted absent, with the kernel it would need also absent, so that
# landing one without the other cannot pass here.
out["mish_kernel"] = (
    "aten.mish.default" in torch._C._aten_implemented()
    if hasattr(torch._C, "_aten_implemented") else None
)

json.dump(out, sys.stdout)
"""


def _run(vendored):
    env = dict(os.environ)
    if vendored:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bindings subprocess (vendored={vendored}) exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


_CACHE = {}


def _fixtures():
    if not _CACHE:
        _CACHE["shim"] = _run(True)
        _CACHE["upstream"] = _run(False)
    return _CACHE["shim"], _CACHE["upstream"]


def _available():
    return os.path.isfile(_VENDOR_SHIM)


def _close(got, want, tol, where):
    assert isinstance(got, dict) and "flat" in got, f"{where}: shim raised {got}"
    assert isinstance(want, dict) and "flat" in want, f"{where}: upstream raised {want}"
    assert got["shape"] == want["shape"], f"{where}: shape {got['shape']} != {want['shape']}"
    assert len(got["flat"]) == len(want["flat"]), where
    for i, (a, b) in enumerate(zip(got["flat"], want["flat"])):
        assert abs(a - b) <= tol + tol * abs(b), (
            f"{where}[{i}]: {a!r} != {b!r} (tol {tol})"
        )


def test_the_two_subprocesses_are_not_the_same_library():
    """Without this every comparison below could pass by having run upstream
    twice, which is the way a differential test dies silently."""
    if not _available():
        return  # vendor tree not installed -- see vendor/install_shim.sh
    shim, up = _fixtures()
    assert shim["marker"] == "shim", shim["marker"]
    assert up["marker"] == "upstream", up["marker"]


def test_linalg_qr_matches_upstream_through_torch_linalg_qr():
    if not _available():
        return
    shim, up = _fixtures()
    for key in ("qr_Q", "qr_R", "qr_tall_Q", "qr_tall_R",
                "qr_complete_Q", "qr_complete_R", "qr_eye_R",
                "qr_named_Q", "qr_named_R", "qr_r_mode_R"):
        _close(shim[key], up[key], 1e-12, key)


def test_linalg_qr_answers_upstreams_sign_convention_on_the_identity():
    """`dlarfg`'s `xnorm == 0` short circuit. Without it `qr(eye(3))` is `-I`
    -- orthogonal, satisfying `Q @ R == A`, and not upstream's answer. This is
    asserted against a literal as well as against upstream because it is the
    single most likely wrong answer here (docs/TAIL1.md §5)."""
    if not _available():
        return
    shim, _ = _fixtures()
    assert shim["qr_eye_R"]["flat"] == [1., 0., 0., 0., 1., 0., 0., 0., 1.]


def test_linalg_qr_r_mode_answers_a_one_dimensional_empty_q():
    if not _available():
        return
    shim, up = _fixtures()
    assert shim["qr_r_mode_Q_shape"] == [0], shim["qr_r_mode_Q_shape"]
    assert shim["qr_r_mode_Q_shape"] == up["qr_r_mode_Q_shape"]


def test_linalg_qr_refuses_what_upstream_refuses_and_says_the_same_thing():
    if not _available():
        return
    shim, up = _fixtures()
    for key in ("qr_bad_mode", "qr_long"):
        assert "error" in shim[key], f"{key}: shim did not refuse"
        assert "error" in up[key], f"{key}: upstream did not refuse"
        assert shim[key]["msg"] == up[key]["msg"], (key, shim[key], up[key])


def test_upsample_nearest2d_matches_upstream_through_F_interpolate():
    if not _available():
        return
    shim, up = _fixtures()
    for key in ("near_scale", "near_size", "near_2x"):
        _close(shim[key], up[key], 1e-6, key)


def test_upsample_nearest2d_binding_accepts_both_of_upstreams_shapes():
    """`.vec` is three arguments here (nearest has no `align_corners`, where
    bilinear's `.vec` does) and the leaf is four. A fourth argument is the only
    discriminator, so both shapes are exercised."""
    if not _available():
        return
    shim, up = _fixtures()
    for key in ("near_vec", "near_leaf", "near_leaf_noscale"):
        _close(shim[key], up[key], 1e-6, key)


def test_upsample_nearest2d_forwards_the_scale_factors_it_was_given():
    """The scales are **forwarded**, not merely used to size the output.
    `1/scale` and `in/out` are the same grid whenever the product is integral,
    which is every case a `scale_factor=2` test produces and also the
    `3x5 -> 4x7` case above -- so dropping them is invisible until a case where
    they differ. This is that case, in both of upstream's shapes, and it is
    here because mutating the composite to pass `None, None` left every other
    test in this file green."""
    if not _available():
        return
    shim, up = _fixtures()
    for key in ("near_vec_17", "near_leaf_17"):
        _close(shim[key], up[key], 1e-6, key)
    # ...and the two grids really are different, so the comparison above has
    # something to fail on.
    for given, recomputed in (("near_vec_17", "near_leaf_17_recomputed"),
                              ("near_leaf_17", "near_leaf_17_recomputed")):
        assert shim[given]["shape"] == shim[recomputed]["shape"], (given, recomputed)
        assert shim[given]["flat"] != shim[recomputed]["flat"], (
            f"{given} and {recomputed} sample the same grid, so this test "
            "cannot tell a forwarded scale from a recomputed one"
        )


def test_upsample_nearest2d_refuses_both_output_size_and_scale_factors():
    if not _available():
        return
    shim, up = _fixtures()
    assert "error" in shim["near_both"], shim["near_both"]
    assert shim["near_both"]["msg"] == up["near_both"]["msg"], (
        shim["near_both"], up["near_both"]
    )


def test_linalg_norm_matches_upstream_on_every_vector_case():
    if not _available():
        return
    shim, up = _fixtures()
    for key in ("norm_ord2_dim", "norm_default_dim", "norm_nokeepdim",
                "norm_flat", "norm_dim0", "norm_ord1", "norm_inf",
                "norm_neginf", "norm_1d_ord3", "norm_dtype"):
        _close(shim[key], up[key], 1e-6, key)


def test_linalg_norm_is_what_owlv2_and_owlvit_spell():
    """`ord=2, dim=-1, keepdim=True` and `dim=-1, keepdim=True` -- the two
    forms at modeling_owlv2.py:975/1048 and modeling_owlvit.py:957/1028. They
    are the same computation, which is worth pinning: `ord=None` defaulting to
    anything but 2 would silently change both models' embedding
    normalisation."""
    if not _available():
        return
    shim, _ = _fixtures()
    assert shim["norm_ord2_dim"]["flat"] == shim["norm_default_dim"]["flat"]
    assert shim["norm_ord2_dim"]["shape"] == [3, 1]


def test_linalg_norm_refuses_the_matrix_norms_by_name_rather_than_guessing():
    """Upstream answers all four of these; this shim has no matrix-norm kernel.
    The failure that matters is not raising -- it is answering the *flattened
    vector* norm, which has the right shape and the wrong number. So each
    refusal is asserted to name itself and to name `linalg_norm`."""
    if not _available():
        return
    shim, up = _fixtures()
    for key in ("norm_fro", "norm_nuc", "norm_matrix_ord2", "norm_matrix_dim"):
        assert "error" in shim[key], f"{key}: shim answered {shim[key]}"
        assert shim[key]["error"] == "NotImplementedError", shim[key]
        assert "linalg_norm" in shim[key]["msg"], shim[key]
        assert "error" not in up[key], (
            f"{key}: upstream refused too, so this is no longer a divergence "
            f"and the refusal text should be reconsidered: {up[key]}"
        )


def test_nn_init_orthogonal_gets_past_linalg_qr_and_stops_at_the_next_wall():
    """`rwkv`'s construction wall runs `_init_weights` -> `nn.init.orthogonal_`
    -> `torch.linalg.qr`, and until this round `torch.linalg.qr` was the wall.

    It is not any more, and `orthogonal_` still does not complete: it now stops
    **one line later**, at `torch/nn/init.py:710`'s `d = torch.diag(r, 0)`,
    which is *after* `q, r = torch.linalg.qr(flattened)` has returned. That is
    the result, not a failure -- the binding this round owed is paid and the
    next debt is a different one (`torch.diag` has no `overloads.json` row).

    Written as a two-sided assertion rather than an `xfail`: if `orthogonal_`
    starts working it goes down the success branch and the orthonormality of
    the answer is checked, and if it fails for a *different* reason -- in
    particular anything naming `linalg` or `qr` -- this turns red rather than
    quietly accepting a regression of the thing it exists to prove.
    """
    if not _available():
        return
    shim, up = _fixtures()
    assert "error" not in up["orthogonal_"], up["orthogonal_"]
    got = shim["orthogonal_"]
    if "error" not in got:
        assert got["ok"], got
        return
    msg = got["msg"]
    for forbidden in ("linalg_qr", "linalg.qr", "_linalg"):
        assert forbidden not in msg, (
            f"nn.init.orthogonal_ still stops at the QR binding: {got}"
        )
    assert "torch.diag" in msg, (
        "nn.init.orthogonal_ stops somewhere new. The wall after linalg.qr was "
        f"`torch.diag` when this was written; now it is: {got}"
    )


def test_mish_has_neither_a_kernel_nor_a_binding():
    """docs/BINDINGS.md §5. `mish` was the fourth debt on this round's list and
    it could not be paid: its kernel was removed (docs/VOICE.md §4.4), so a
    `_nn.mish` composite would dispatch to an op that is not there. This
    asserts the *pair* -- kernel absent and binding absent -- so that landing
    either one alone turns it red and whoever does it has to land the other."""
    if not _available():
        return
    shim, _ = _fixtures()
    assert shim["mish_kernel"] is False, (
        "aten.mish.default is implemented now -- land the `_nn.mish` binding "
        "beside `silu`'s and delete this test"
    )
    assert "mish" not in _C._shim_nn_implemented


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
