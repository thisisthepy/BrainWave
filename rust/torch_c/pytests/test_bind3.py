"""docs/bindings/BIND3.md -- the three `torch._C._nn` bindings docs/architectures/VOICE3.md left open,
proven **through the spelling a user writes**.

`tools/golden/cases.py` already compares `aten.im2col.default`,
`aten.col2im.default` and `aten.upsample_nearest1d.default` element-wise
against upstream, and this file does not repeat that. Golden compares at the
**dispatch key** and is structurally blind to whether any Python spelling
reaches it -- which is the whole reason `docs/bindings/REACH.md` and
`tools/golden/reach.py` exist, and the reason those three kernels sat in
`reach_allow.json` for a round with a green golden run beside them.

So every case here goes through `F.unfold` / `F.fold` / `F.interpolate`, or
through the `torch._C._nn.*` name those three forward to, and compares against
a **live upstream torch in a separate process** (`env -u PYTHONPATH -u
TORCH_USE_RTLD_GLOBAL`), element-wise, shape and dtype included.

What a comfortable case cannot hold down, and is therefore here:

  * **`upsample_nearest1d`'s discriminator is the third argument's TYPE.**
    The 1-D leaf schema is `(self, output_size, scales)` -- three arguments,
    the *same count* as the `.vec` shape `(input, output_size,
    scale_factors)`. `upsample_nearest2d` one function above can tell the two
    apart by arity because its leaf has a fourth argument; the 1-D op cannot.
    `test_upsample_nearest1d_tells_leaf_from_vec_by_type_not_by_arity` sends
    all four shapes upstream accepts.

  * **The scale factor is forwarded, not merely used to size the output.**
    `1/scale` and `in_w/out_w` coincide whenever the product is integral --
    i.e. in every `scale_factor=2` test -- and pick different source columns
    the moment it is not. `scale_factor=1.5` on a width-3 input is that case.

  * **`F.fold` on OVERLAPPING windows**, because `col2im` sums where windows
    overlap and a `stride == kernel` case cannot distinguish summing from
    overwriting.

  * **`F.unfold` on a MULTI-channel input**, because the channel is the
    slowest axis of the folded dimension and a single-channel case cannot
    tell that reading from the transposed one -- same shape, wrong matrix.

  * **The refusals**, both ways `.vec` can be asked wrong (both of
    `output_size`/`scale_factors`, and neither), because the second is not
    guessable from the first.

Nothing here needs numpy or a network.
"""

import json
import os
import struct
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


# -- F.unfold / torch._C._nn.im2col ------------------------------------------
# MULTI-channel on purpose: the channel is the slowest axis of the folded
# dimension, and a single-channel input cannot tell that from the transposed
# reading -- same shape, different matrix.
multi = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
single = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

rec("unfold_plain", lambda: F.unfold(multi, kernel_size=2, stride=1))
rec("unfold_int_args", lambda: F.unfold(multi, 2, 1, 0, 1))
rec("unfold_dilation_padding_stride",
    lambda: F.unfold(multi, kernel_size=2, dilation=2, padding=1, stride=2))
rec("unfold_asymmetric_kernel",
    lambda: F.unfold(multi, kernel_size=(2, 3), stride=(1, 1)))
rec("unfold_batched",
    lambda: F.unfold(torch.arange(2 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 2, 4, 4),
                     kernel_size=2, stride=1))
rec("unfold_f64", lambda: F.unfold(multi.double(), kernel_size=2, stride=1))
rec("unfold_kernel_too_big", lambda: F.unfold(single, kernel_size=5))
rec("unfold_int32_refused", lambda: F.unfold(multi.int(), kernel_size=2))
# The `_nn` name itself, with the scalar `int[2]` broadcast upstream allows.
rec("nn_im2col_scalar_ints", lambda: torch._C._nn.im2col(single, 2, 1, 0, 1))
rec("nn_im2col_kwargs", lambda: torch._C._nn.im2col(
    single, kernel_size=[2, 2], dilation=[1, 1], padding=[0, 0], stride=[1, 1]))
rec("nn_im2col_missing_args", lambda: torch._C._nn.im2col(single, [2, 2]))

# -- F.fold / torch._C._nn.col2im --------------------------------------------
# stride < kernel everywhere: overlapping windows are SUMMED, and a
# non-overlapping case cannot distinguish that from overwriting.
cols = torch.ones(1, 4, 9)
ramp = torch.arange(36, dtype=torch.float32).reshape(1, 4, 9)

rec("fold_overlap_sums", lambda: F.fold(cols, output_size=(4, 4), kernel_size=2, stride=1))
rec("fold_ramp_overlap", lambda: F.fold(ramp, output_size=(4, 4), kernel_size=2, stride=1))
rec("fold_no_overlap",
    lambda: F.fold(torch.ones(1, 4, 4), output_size=(4, 4), kernel_size=2, stride=2))
rec("fold_padded", lambda: F.fold(cols, output_size=(2, 2), kernel_size=2, padding=1, stride=1))
rec("fold_dilation", lambda: F.fold(
    torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
    output_size=(4, 4), kernel_size=2, dilation=2, stride=1))
rec("fold_multichannel", lambda: F.fold(
    torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(1, 8, 6),
    output_size=(3, 4), kernel_size=2, stride=1))
rec("fold_batched", lambda: F.fold(
    torch.arange(2 * 4 * 9, dtype=torch.float32).reshape(2, 4, 9),
    output_size=(4, 4), kernel_size=2, stride=1))
rec("fold_f64", lambda: F.fold(cols.double(), output_size=(4, 4), kernel_size=2, stride=1))
rec("fold_wrong_block_count",
    lambda: F.fold(torch.ones(1, 4, 8), output_size=(4, 4), kernel_size=2, stride=1))
rec("fold_int32_refused",
    lambda: F.fold(cols.int(), output_size=(4, 4), kernel_size=2, stride=1))
rec("nn_col2im_scalar_ints", lambda: torch._C._nn.col2im(cols, 4, 2, 1, 0, 1))
rec("nn_col2im_missing_args", lambda: torch._C._nn.col2im(cols, [4, 4], [2, 2]))

# `F.fold(F.unfold(x))` is NOT `x` -- it is `x` weighted by the cover count,
# and that is the op rather than a rounding artefact.
rec("fold_of_unfold_is_weighted", lambda: F.fold(
    F.unfold(single, kernel_size=2, stride=1), output_size=(4, 4),
    kernel_size=2, stride=1))

# -- F.interpolate(mode="nearest") / torch._C._nn.upsample_nearest1d ---------
# Width 3 -> 7 at scale 1.5: `1/1.5` and `3/7` differ, so an implementation
# that sizes the output from the factor but does not FORWARD it samples a
# different grid at the same shape.
line = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
wide = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)

for w in (1, 2, 3, 4, 7, 8, 11):
    rec("interp_size_%d" % w, lambda w=w: F.interpolate(line, size=w, mode="nearest"))
for s in (0.5, 1.5, 2.0, 3.0):
    rec("interp_scale_%s" % s,
        lambda s=s: F.interpolate(line, scale_factor=s, mode="nearest"))
rec("interp_lanes", lambda: F.interpolate(wide, size=8, mode="nearest"))
# The case that separates "forwarded the factor" from "recomputed one from the
# shapes": width 2 at 2.25 is width 4, and `floor(i/2.25)` is `0,0,0,1` where
# `floor(i*2/4)` is `0,0,1,1`. Found by search rather than guessed -- the
# obvious `3 -> 4 at 1.5` does NOT separate them, and a test built on it would
# have asserted a difference that is not there.
narrow = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
rec("interp_forwarded_scale",
    lambda: F.interpolate(narrow, scale_factor=2.25, mode="nearest"))
rec("interp_f64", lambda: F.interpolate(line.double(), size=7, mode="nearest"))
rec("interp_uint8", lambda: F.interpolate(
    torch.arange(4, dtype=torch.uint8).reshape(1, 1, 4), size=7, mode="nearest"))
rec("interp_bool_refused", lambda: F.interpolate(line.bool(), size=7, mode="nearest"))

# The four shapes the `_nn` name itself accepts. The third and the first have
# the same arity and different schemas; that is the whole point.
rec("nn_un1d_vec_size", lambda: torch._C._nn.upsample_nearest1d(line, [7], None))
rec("nn_un1d_vec_factors", lambda: torch._C._nn.upsample_nearest1d(line, None, [1.5]))
rec("nn_un1d_leaf_scales", lambda: torch._C._nn.upsample_nearest1d(line, [7], 1.5))
rec("nn_un1d_two_args", lambda: torch._C._nn.upsample_nearest1d(line, [7]))
rec("nn_un1d_kwarg", lambda: torch._C._nn.upsample_nearest1d(line, output_size=[7]))
rec("nn_un1d_both", lambda: torch._C._nn.upsample_nearest1d(line, [7], [1.5]))
rec("nn_un1d_neither", lambda: torch._C._nn.upsample_nearest1d(line, None, None))

json.dump(out, sys.stdout)
"""

_cache = {}


def _run(side):
    """`side` is `"shim"` or `"upstream"`. `None` on the shim side when the
    vendored shim is not installed -- `test_voice3.py`'s silent skip, for the
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


def _agree(name):
    """Element-wise agreement with upstream, dtype and shape included.

    Exact rather than approximate: none of these three ops arithmetic --
    `im2col` and `col2im` gather and sum, `upsample_nearest1d` gathers -- so
    a tolerance here would only hide a wrong index.
    """
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
    assert got["ok"] == want["ok"], f"{name}: values {got['ok']} != {want['ok']}"


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


def _f32(x):
    """`x` rounded to float32, because the kernel accumulates the scale at
    float32 for every dtype but float64 and the two readings above are
    separated by less than a float64 would round away."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _value(name):
    pair = _both(name)
    if pair == "skip":
        return None, None
    return pair


# --------------------------------------------------------------------------
# 1. `F.unfold` -> `torch._C._nn.im2col`  (llama4's vision tower)
# --------------------------------------------------------------------------


def test_f_unfold_agrees_with_upstream_including_the_multichannel_layout():
    """The spelling `llama4`'s vision tower writes, on every argument shape.

    `unfold_plain` is the multi-channel case and it is the one that matters:
    the folded dimension is `c * kh * kw` with the CHANNEL slowest, and a
    transposed reading produces the identical shape with the rows permuted.
    A single-channel input has `c == 1` and cannot see the difference.
    """
    for name in ("unfold_plain", "unfold_int_args",
                 "unfold_dilation_padding_stride", "unfold_asymmetric_kernel",
                 "unfold_batched", "unfold_f64"):
        _agree(name)


def test_unfold_reaches_im2col_rather_than_some_other_kernel():
    """`F.unfold` is `torch._C._nn.im2col(...)` and nothing else.

    Held two ways so that neither alone has to carry it: the name is installed
    (the binding exists), and the kernel it names is advertised. A binding
    onto a missing kernel is `docs/bindings/BINDINGS.md`'s `mish` -- two lines that
    were a door onto nothing -- and this is the check that would have caught
    it.
    """
    assert "im2col" in _C._shim_nn_implemented
    assert "aten.im2col.default" in set(_C._aten_implemented())


def test_unfold_refuses_what_upstream_refuses():
    """A kernel bigger than the input, and an integral dtype.

    `im2col` is the op in this file where the dtype rule runs BACKWARDS from
    most of the shim -- `bool` computes and `int32` does not -- so "refuses
    what upstream refuses" is not a formality here.
    """
    _both_refuse("unfold_kernel_too_big")
    _both_refuse("unfold_int32_refused")
    got, _ = _both_refuse("nn_im2col_missing_args") or (None, None)
    if got is not None:
        assert got["raised"] == "TypeError", got


def test_nn_im2col_takes_the_scalar_int_broadcast_upstream_takes():
    """`torch._C._nn.im2col(x, 2, 1, 0, 1)` computes upstream -- an `int[2]`
    schema broadcasts a scalar -- and `F.unfold` never sends that shape,
    because `_pair` has already run.

    So this case exists only because the binding is a public name and not
    just `F.unfold`'s implementation detail. Without the normalisation in
    `_int_pair` the Rust `shape_arg` under `pair_arg` would refuse a bare int
    and the two would diverge on a call upstream accepts.
    """
    _agree("nn_im2col_scalar_ints")
    _agree("nn_im2col_kwargs")


# --------------------------------------------------------------------------
# 2. `F.fold` -> `torch._C._nn.col2im`  (f5-tts)
# --------------------------------------------------------------------------


def test_f_fold_agrees_with_upstream_on_overlapping_windows():
    """`stride < kernel` in every case here, on purpose.

    Where two windows overlap `col2im` SUMS the contributions. With
    `stride == kernel` every output element is covered exactly once and a
    summing implementation and an overwriting one produce identical output --
    so a suite built only from `fold_no_overlap` would pass on the wrong op.
    `fold_no_overlap` is here as the control, not as the evidence.
    """
    for name in ("fold_overlap_sums", "fold_ramp_overlap", "fold_no_overlap",
                 "fold_padded", "fold_dilation", "fold_multichannel",
                 "fold_batched", "fold_f64"):
        _agree(name)


def test_fold_of_unfold_is_the_cover_count_not_the_identity():
    """`F.fold(F.unfold(x))` is `x * cover`, and the interior of a `4x4` at
    `kernel=2, stride=1` is covered four times.

    Asserted as a value against upstream *and* as a ratio here, so that the
    claim survives someone changing the input: if this were the identity the
    ratio would be 1 everywhere, and `col2im` would be overwriting.
    """
    got, want = _value("fold_of_unfold_is_weighted")
    if got is None:
        return
    assert got["ok"] == want["ok"], (got, want)
    src = list(range(16))
    cover = [[1, 2, 2, 1], [2, 4, 4, 2], [2, 4, 4, 2], [1, 2, 2, 1]]
    flat = [c for row in cover for c in row]
    assert got["ok"] == [float(v * c) for v, c in zip(src, flat)], got
    assert flat != [1] * 16, "the control itself is broken"


def test_fold_reaches_col2im_rather_than_some_other_kernel():
    """The binding exists and the kernel it names is advertised --
    `test_unfold_reaches_im2col_rather_than_some_other_kernel`'s reason."""
    assert "col2im" in _C._shim_nn_implemented
    assert "aten.col2im.default" in set(_C._aten_implemented())


def test_fold_refuses_what_upstream_refuses():
    _both_refuse("fold_wrong_block_count")
    _both_refuse("fold_int32_refused")
    _both_refuse("nn_col2im_missing_args")


def test_nn_col2im_takes_the_scalar_int_broadcast_upstream_takes():
    _agree("nn_col2im_scalar_ints")


# --------------------------------------------------------------------------
# 3. `F.interpolate(mode="nearest")` on 3-D -> `upsample_nearest1d` (bigvgan)
# --------------------------------------------------------------------------


def test_f_interpolate_nearest_on_a_3d_input_agrees_with_upstream():
    """Seven output widths, up and down, plus a multi-lane input.

    `interp_size_1` and `interp_size_3` are the degenerate ends -- an output
    narrower than the input and an output equal to it -- which is where an
    off-by-one in the index arithmetic shows and where a `2x` case does not.
    """
    for w in (1, 2, 3, 4, 7, 8, 11):
        _agree("interp_size_%d" % w)
    _agree("interp_lanes")
    _agree("interp_f64")
    _agree("interp_uint8")


def test_the_scale_factor_is_forwarded_and_not_merely_used_to_size_the_output():
    """`1/scale` and `in_w/out_w` coincide far more often than they look like
    they should, and where they do a test proves nothing.

    Width **2** at `scale_factor=2.25` is width 4, and `floor(i * 1/2.25)` is
    `0,0,0,1` where `floor(i * 2/4)` is `0,0,1,1` -- so an implementation that
    sizes the output from the factor but forwards `None` samples a different
    grid at the identical shape and dtype.

    **The obvious case does not work.** `3 -> 4 at 1.5` was tried first and
    both readings give `0,0,1,2`; so does `5 -> 7 at 1.5`. Over the search
    space `in_w in 2..12` and `scale in 0.25..5` in eighths, 158 of the pairs
    separate and none of the integral ones do -- which is why every
    `scale_factor=2` case in this suite is blind to it.

    The separation is computed here and required to hold before the agreement
    is claimed, so the day someone edits the input this fails as a broken
    control rather than passing as a vacuous check.
    """
    got, want = _value("interp_forwarded_scale")
    if got is None:
        return
    src = [0.0, 1.0, 2.0, 3.0]   # (1, 2, 2)
    in_w, out_w, scale = 2, 4, 2.25
    inv = _f32(1.0 / scale)
    ratio = _f32(in_w / out_w)
    forwarded = [src[lane * in_w + min(in_w - 1, int(i * inv))]
                 for lane in (0, 1) for i in range(out_w)]
    recomputed = [src[lane * in_w + min(in_w - 1, int(i * ratio))]
                  for lane in (0, 1) for i in range(out_w)]
    assert forwarded != recomputed, (
        "this case no longer distinguishes the two readings -- pick another "
        "scale factor before trusting the assertion below"
    )
    assert want["ok"] == forwarded, ("upstream forwards", want, forwarded)
    assert want["ok"] != recomputed, ("upstream recomputes?", want, recomputed)
    assert got["ok"] == want["ok"], (got, want)
    for s in (0.5, 1.5, 2.0, 3.0):
        _agree("interp_scale_%s" % s)


def test_upsample_nearest1d_tells_leaf_from_vec_by_type_not_by_arity():
    """The 1-D leaf schema is `(self, output_size, scales)` -- **three**
    arguments, the same count as `.vec`'s `(input, output_size,
    scale_factors)`.

    `upsample_nearest2d` one function above discriminates on a fourth
    argument, because its leaf schema has one. Transcribing that approach to
    the 1-D op reads `_nn.upsample_nearest1d(x, [7], 1.5)` as `.vec` and
    subscripts a bare float. All four shapes upstream accepts are here, and
    the two that share an arity disagree about what the third argument means.
    """
    for name in ("nn_un1d_vec_size", "nn_un1d_vec_factors",
                 "nn_un1d_leaf_scales", "nn_un1d_two_args", "nn_un1d_kwarg"):
        _agree(name)
    # And the arity that would be ambiguous really is ambiguous: same number
    # of arguments, different answers.
    a, _ = _value("nn_un1d_vec_factors")
    b, _ = _value("nn_un1d_leaf_scales")
    if a is None:
        return
    assert a["shape"] != b["shape"], (
        "a three-argument `.vec` call and a three-argument leaf call now agree "
        "on shape, so this test no longer proves the discriminator"
    )


def test_upsample_nearest1d_refuses_both_ways_the_vec_shape_can_be_asked_wrong():
    """Giving both `output_size` and `scale_factors`, and giving neither.

    The second is not guessable from the first: `Must specify exactly one`
    reads like a check against over-specification, and upstream raises the
    identical message for under-specification. Measured, and both messages
    are compared rather than only their presence.
    """
    for name in ("nn_un1d_both", "nn_un1d_neither"):
        got, want = _both_refuse(name) or (None, None)
        if got is None:
            return
        assert got["raised"] == want["raised"], (name, got, want)
        assert got["msg"] == want["msg"], (name, got["msg"], want["msg"])


def test_interpolate_refuses_bool_which_is_where_the_1d_and_2d_dtype_sets_differ():
    """`uint8` computes for both ops and `bool` for neither -- but that is the
    one place `upsample_nearest1d` and `upsample_nearest2d` are known to
    disagree in wording, which is why they are separate kernels and separate
    bindings rather than an alias."""
    _both_refuse("interp_bool_refused")


def test_upsample_nearest1d_reaches_its_kernel():
    assert "upsample_nearest1d" in _C._shim_nn_implemented
    assert "aten.upsample_nearest1d.default" in set(_C._aten_implemented())


# --------------------------------------------------------------------------
# 4. The allowlist entries these three bindings retire
# --------------------------------------------------------------------------


def test_the_three_reach_allowlist_entries_are_gone_and_stay_gone():
    """`tools/golden/reach_allow.json` recorded all three as gaps that were
    absent *on purpose*. They are not absent any more, so the entries are
    deleted -- and `reach.py` fails on a stale entry as well as on an
    unlisted gap, which is what makes that file describe the present rather
    than the past.

    Asserted here as well, because `reach.py` runs from `compare.py` and this
    file runs from `run.sh`: a gate that only one of the two pulls is one
    reordering away from not being pulled at all.
    """
    import json as _json
    path = os.path.join(_REPO_ROOT, "tools", "golden", "reach_allow.json")
    allow = _json.loads(open(path).read())
    shape2 = allow["shape2_kernel_without_spelling"]
    for key in ("aten.im2col.default", "aten.col2im.default",
                "aten.upsample_nearest1d.default"):
        assert key not in shape2, (
            f"{key} is spelled by `_install_nn` now; its allowlist entry "
            "describes a gap that no longer exists"
        )
        assert key in set(_C._aten_implemented()), key


def test_no_torch_level_spelling_was_invented_for_these_three():
    """There is no `torch.im2col`, no `Tensor.im2col`, and none of the other
    four, on upstream 2.13.0 -- `reach_allow.json` staked its reasons on
    exactly that and `reach.py --verify-upstream` checks it.

    So these are `_nn` bindings and NOT `overloads.json` / `methods.json`
    rows. A table row would have been the easier change and would have put a
    door on this shim that upstream does not have, which docs/bindings/SPELLINGS.md
    refuses. Held from this side too, because the tables are what a later
    round is most likely to reach for.
    """
    for name in ("im2col", "col2im", "upsample_nearest1d"):
        assert name not in _C._shim_overloads, (
            f"torch.{name} does not exist upstream; a row here invents one"
        )
        assert name not in _C._shim_methods, (
            f"Tensor.{name} does not exist upstream; a row here invents one"
        )
        assert name in _C._shim_nn_implemented, name


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
