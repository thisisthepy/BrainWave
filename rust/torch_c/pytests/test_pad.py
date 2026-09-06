"""`reflection_pad{1,2,3}d` / `replication_pad{1,2,3}d` -- docs/PAD.md.

The op that sits in front of the speech roadmap from two directions.
docs/COMPLEX.md §5 measured that `torch.stft` raises the *identical* error for
`return_complex=True` and `return_complex=False`, before any transform, because
`stft` reflect-pads by `n_fft // 2` when `center=True`. Independently,
docs/VOICE.md §1 ranks `F.pad(mode="reflect")` as blocking four of its five
speech models and `mode="replicate"` a fifth case. Neither has anything to do
with complex numbers.

These tests go through `_C._aten_dispatch`, the same door `test_shim.py` uses,
because `torch._C._nn.pad` is a `bootstrap.py::_install_nn` composite and
`bootstrap.py` was another agent's file this round -- see docs/PAD.md §5 for
the exact patch that makes `F.pad` reach these kernels.

What is pinned here, all of it measured against upstream 2.13.0 rather than
derived from the name:

  * reflect does NOT repeat the edge element; replicate DOES -- and the two
    are asserted to *disagree* on the same input, not merely to match their
    own literals, because swapping them produces plausible-looking output
  * reflect refuses `pad >= extent`; replicate has no such limit
  * negative padding is a negative offset into ONE gather, so a reflection can
    read elements a crop would have discarded
  * only the batch axis may be empty
  * `pad1d` refuses an empty output where `pad2d`/`pad3d` return one
"""

import json
import os
import subprocess
import sys

from test_shim import _C, _CKPT_VENDOR_DIR, _CKPT_VENDOR_SHIM

REFLECT = [f"aten.reflection_pad{n}d.default" for n in (1, 2, 3)]
REPLICATE = [f"aten.replication_pad{n}d.default" for n in (1, 2, 3)]


def _t(values, shape, dtype=None):
    """A tensor with exactly `values`, built through the shim's own door."""
    t = _C._tensor_from_flat([float(v) for v in values], list(shape),
                             dtype=dtype or _C.float32)
    return t


def _flat(t):
    out = t.tolist()
    while out and isinstance(out[0], list):
        out = [v for row in out for v in row]
    return out


def test_all_six_pad_ops_are_advertised():
    implemented = _C._aten_implemented()
    for op in REFLECT + REPLICATE:
        assert op in implemented, op


# --------------------------------------------------------------------------
# The headline: reflect vs replicate, on the same input, in one assertion.
# --------------------------------------------------------------------------

def test_reflect_does_not_repeat_the_edge_and_replicate_does():
    """Padding `[1,2,3]` by 2 each side.

        reflect    [3, 2, 1, 2, 3, 2, 1]     mirror, edge NOT repeated
        replicate  [1, 1, 1, 2, 3, 3, 3]     edge repeated

    Both were read off upstream. Getting them backwards produces output of
    the right shape with the right elements in it, which is exactly the kind
    of wrong a shape-and-dtype check does not catch.
    """
    x = _t([1, 2, 3], (1, 1, 3))
    ref = _flat(_C._aten_dispatch("aten.reflection_pad1d.default", x, [2, 2]))
    rep = _flat(_C._aten_dispatch("aten.replication_pad1d.default", x, [2, 2]))

    assert ref == [3, 2, 1, 2, 3, 2, 1], ref
    assert rep == [1, 1, 1, 2, 3, 3, 3], rep
    # The check that fails if the two kernels are swapped, which the two
    # assertions above would ALSO catch -- but only because their literals
    # were transcribed correctly. This one holds even if they were not.
    assert ref != rep
    # Reflect's immediate neighbours of the edge are the interior; replicate's
    # are the edge itself. Stated structurally so it survives a change of
    # fixture.
    assert ref[1] == 2 and ref[2] == 1, ref     # ...2,1 | 1 is the edge
    assert rep[0] == rep[1] == rep[2] == 1, rep  # the edge, three times


def test_the_two_modes_disagree_on_every_rank():
    """The 2-D and 3-D kernels are separate upstream schemas, and the same
    swap is possible in each of them independently."""
    for n, shape in ((1, (1, 1, 4)), (2, (1, 3, 4)), (3, (1, 2, 3, 4))):
        size = 1
        for d in shape:
            size *= d
        x = _t(range(1, size + 1), shape)
        padding = [1, 1] * n
        ref = _flat(_C._aten_dispatch(f"aten.reflection_pad{n}d.default", x, padding))
        rep = _flat(_C._aten_dispatch(f"aten.replication_pad{n}d.default", x, padding))
        assert ref != rep, (n, ref)


# --------------------------------------------------------------------------
# The width limit, which the two modes do NOT share.
# --------------------------------------------------------------------------

def test_reflect_refuses_a_pad_as_wide_as_the_axis_and_replicate_does_not():
    x = _t([1, 2, 3, 4], (1, 1, 4))
    # pad == extent - 1 is the widest reflect accepts.
    ok = _flat(_C._aten_dispatch("aten.reflection_pad1d.default", x, [3, 3]))
    assert ok == [4, 3, 2, 1, 2, 3, 4, 3, 2, 1], ok

    try:
        _C._aten_dispatch("aten.reflection_pad1d.default", x, [4, 0])
    except Exception as e:
        assert "Padding size should be less than the corresponding" in str(e), e
        assert "at dimension 2 of input [1, 1, 4]" in str(e), e
    else:
        raise AssertionError("reflect accepted a pad as wide as the axis")

    # Replicate has no limit at all -- upstream returns a length-202 tensor.
    wide = _C._aten_dispatch("aten.replication_pad1d.default", x, [99, 99])
    assert list(wide.shape) == [1, 1, 202], wide.shape
    flat = _flat(wide)
    assert flat[:99] == [1] * 99 and flat[-99:] == [4] * 99


def test_the_argument_number_in_the_width_refusal_tracks_the_axis():
    """`4 + 2k` for the k-th pair, so a `pad2d`'s height pair is `#6`. Read
    off upstream for all three ranks; a kernel that always said `#4` would
    misdirect every caller who padded a non-last axis too wide."""
    x = _t(range(12), (1, 3, 4))
    try:
        _C._aten_dispatch("aten.reflection_pad2d.default", x, [4, 0, 0, 0])
    except Exception as e:
        assert "Argument #4:" in str(e), e
    try:
        _C._aten_dispatch("aten.reflection_pad2d.default", x, [0, 0, 3, 0])
    except Exception as e:
        assert "Argument #6:" in str(e), e
        assert "at dimension 1 of input [1, 3, 4]" in str(e), e


# --------------------------------------------------------------------------
# Negative padding: one gather, not a crop pass.
# --------------------------------------------------------------------------

def test_negative_padding_crops():
    x = _t([1, 2, 3, 4], (1, 1, 4))
    assert _flat(_C._aten_dispatch("aten.reflection_pad1d.default", x, [-1, 0])) == [2, 3, 4]
    assert _flat(_C._aten_dispatch("aten.replication_pad1d.default", x, [-1, -1])) == [2, 3]


def test_a_reflection_reads_past_what_a_crop_would_have_discarded():
    """THE case that separates a gather from a crop-then-mirror.

    `reflection_pad1d([1,2,3,4], [-1, 3])` is `[2,3,4,3,2,1]` upstream. The
    trailing `1` is the element the `-1` crop removed -- so an implementation
    that cropped first and mirrored the remainder cannot produce it (it would
    have to invent a value, and the plausible one is `[2,3,4,3,2,3]`).
    """
    x = _t([1, 2, 3, 4], (1, 1, 4))
    got = _flat(_C._aten_dispatch("aten.reflection_pad1d.default", x, [-1, 3]))
    assert got == [2, 3, 4, 3, 2, 1], got
    assert got[-1] == 1, "the reflection did not reach past the crop"

    # The mirror image, cropping the back instead.
    got = _flat(_C._aten_dispatch("aten.reflection_pad1d.default", x, [3, -1]))
    assert got == [4, 3, 2, 1, 2, 3], got


def test_replicate_clamps_rather_than_reflecting_under_a_crop():
    x = _t([1, 2, 3, 4], (1, 1, 4))
    got = _flat(_C._aten_dispatch("aten.replication_pad1d.default", x, [-1, 2]))
    assert got == [2, 3, 4, 4, 4], got


# --------------------------------------------------------------------------
# Rank, emptiness, and the pad1d/pad2d inconsistency -- all upstream's.
# --------------------------------------------------------------------------

def test_both_the_batched_and_unbatched_rank_are_accepted():
    """`ndim + 1` and `ndim + 2`. A kernel that hardcoded one would fail only
    on the other, and `torch.stft` uses the batched one (it `view`s its 1-D
    signal up to 3-D first, which is why `stft` on a bare signal works)."""
    unbatched = _t([1, 2, 3], (1, 3))
    batched = _t([1, 2, 3], (1, 1, 3))
    a = _flat(_C._aten_dispatch("aten.reflection_pad1d.default", unbatched, [1, 1]))
    b = _flat(_C._aten_dispatch("aten.reflection_pad1d.default", batched, [1, 1]))
    assert a == b == [2, 1, 2, 3, 2], (a, b)


def test_only_the_batch_axis_may_be_empty():
    empty_batch = _C._aten_dispatch("aten.reflection_pad1d.default",
                                    _t([], (0, 2, 4)), [1, 1])
    assert list(empty_batch.shape) == [0, 2, 6], empty_batch.shape

    for shape in ((1, 0, 4), (0, 3)):
        try:
            _C._aten_dispatch("aten.reflection_pad1d.default", _t([], shape), [1, 1])
        except Exception as e:
            assert "possibly 0 batch size" in str(e), e
        else:
            raise AssertionError(f"accepted an empty non-batch axis: {shape}")


def test_pad1d_refuses_an_empty_output_but_pad2d_returns_one():
    """An inconsistency in upstream, reproduced rather than tidied. A shim
    that made the two consistent would be wrong in whichever direction it
    chose."""
    x1 = _t(range(12), (1, 3, 4))
    try:
        _C._aten_dispatch("aten.replication_pad1d.default", x1, [-4, 0])
    except Exception as e:
        assert "is too small. Calculated output W: 0" in str(e), e
    else:
        raise AssertionError("pad1d accepted an empty output")

    empty = _C._aten_dispatch("aten.replication_pad2d.default", x1, [-4, 0, 0, 0])
    assert list(empty.shape) == [1, 3, 0], empty.shape


def test_the_padding_length_refusal_is_worded_differently_per_mode():
    """`reflection_pad*` appends ", but got: N" and `replication_pad*` does
    not. All six were run upstream; it is not a transcription slip here."""
    x = _t([1, 2, 3], (1, 1, 3))
    try:
        _C._aten_dispatch("aten.reflection_pad1d.default", x, [1, 1, 1, 1])
    except Exception as e:
        assert str(e).endswith("padding size is expected to be 2, but got: 4"), e
    try:
        _C._aten_dispatch("aten.replication_pad1d.default", x, [1, 1, 1, 1])
    except Exception as e:
        assert str(e).endswith("padding size is expected to be 2"), e


def test_bool_is_the_only_dtype_refused():
    """Unlike `glu`, these are pure data movement and upstream implements
    them for integers -- so a float-only gate copied from `silu`/`glu` would
    refuse cases upstream computes."""
    for dtype in (_C.int64, _C.int32, _C.float64, _C.float16, _C.bfloat16):
        x = _C._tensor_from_flat([1.0, 2.0, 3.0], [1, 1, 3], dtype=dtype)
        got = _flat(_C._aten_dispatch("aten.reflection_pad1d.default", x, [1, 1]))
        assert got == [2, 1, 2, 3, 2], (dtype, got)

    b = _C._tensor_from_flat([1.0, 0.0, 1.0], [1, 1, 3], dtype=_C.bool)
    try:
        _C._aten_dispatch("aten.reflection_pad1d.default", b, [1, 1])
    except Exception as e:
        assert 'not implemented for \'Bool\'' in str(e), e
        assert '"reflection_pad1d"' in str(e), e
    else:
        raise AssertionError("bool was accepted")


# --------------------------------------------------------------------------
# Separability, and the stft shape.
# --------------------------------------------------------------------------

def test_pad2d_equals_padding_the_last_axis_then_the_one_before():
    """Both modes are separable, checked rather than assumed -- it is what
    lets one gather serve all six schemas."""
    x = _t(range(12), (1, 3, 4))
    for one, two in (("reflection_pad1d", "reflection_pad2d"),
                     ("replication_pad1d", "replication_pad2d")):
        both = _C._aten_dispatch(f"aten.{two}.default", x, [1, 2, 1, 0])
        # Last axis first, exactly as `padding` is ordered.
        step = _C._aten_dispatch(f"aten.{one}.default", x, [1, 2])
        step = _C._aten_dispatch("aten.transpose.int", step, 1, 2)
        step = _C._aten_dispatch(f"aten.{one}.default", step.contiguous(), [1, 0])
        step = _C._aten_dispatch("aten.transpose.int", step, 1, 2)
        assert _flat(both) == _flat(step), (one, _flat(both), _flat(step))


def test_the_stft_pad_shape():
    """What `torch.stft(signal, n_fft=64, center=True)` asks for: a reflect
    pad of `n_fft // 2` on a signal `view`ed up to `(1, 1, L)`.
    docs/PAD.md §4 -- this is the call that used to be `torch.stft`'s first
    wall, and the wall is now the FFT behind it."""
    n_fft, length = 64, 400
    sig = _C._aten_dispatch("aten.arange.start_step", 0, length, 1)
    sig = _C._aten_dispatch("aten._to_copy.default", sig, _C.float32)
    sig = _C._aten_dispatch("aten.view.default", sig, [1, 1, length])
    padded = _C._aten_dispatch("aten.reflection_pad1d.default",
                               sig, [n_fft // 2, n_fft // 2])
    assert list(padded.shape) == [1, 1, length + n_fft], padded.shape
    flat = _flat(padded)
    # The reflection, checked at both seams rather than by shape alone.
    assert flat[:32] == list(range(32, 0, -1)), flat[:32]
    assert flat[32:36] == [0, 1, 2, 3], flat[32:36]
    assert flat[-1] == length - 1 - 32, flat[-1]


# --------------------------------------------------------------------------
# aten.rms_norm.default -- docs/VOICE.md rank 16 (f5), taken in the same round
# because it is real-valued and independent of the complex question.
# docs/PAD.md §8.
# --------------------------------------------------------------------------

def _rms(x, ns, weight=None, eps=None):
    return _C._aten_dispatch("aten.rms_norm.default", x, list(ns), weight, eps)


def test_rms_norm_is_advertised():
    assert "aten.rms_norm.default" in _C._aten_implemented()


def test_rms_norm_does_not_subtract_the_mean():
    """The whole difference from `layer_norm`. A mean-subtracting kernel gives
    a zero-mean row; this one does not, and on an all-positive row every
    output stays positive."""
    x = _t([1, 2, 3, 4], (1, 4))
    got = _flat(_rms(x, [4]))
    assert all(v > 0 for v in got), got
    # mean-square is 7.5, so the scale is 1/sqrt(7.5) = 0.3651484...
    assert abs(got[0] - 0.3651484) < 1e-5, got
    assert abs(got[3] - 4 * got[0]) < 1e-5, got  # exactly proportional to input


def test_rms_norm_eps_is_inside_the_square_root():
    """`rsqrt(mean_square + eps)`, not `1 / (sqrt(mean_square) + eps)`. With a
    large eps the two are far apart, which is what makes this checkable
    without relying on the last float32 digit."""
    x = _t([1, 2, 3, 4], (1, 4))
    got = _flat(_rms(x, [4], None, 1.0))
    inside = 1.0 / (7.5 + 1.0) ** 0.5
    outside = 1.0 / (7.5 ** 0.5 + 1.0)
    assert abs(got[0] - 1 * inside) < 1e-5, (got[0], inside, outside)
    assert abs(got[0] - 1 * outside) > 1e-3, "eps was added to the ROOT"


def test_the_default_eps_is_the_accumulation_dtypes_epsilon_not_the_inputs():
    """docs/PAD.md §8's measured trap.

    For a float16 input upstream uses `finfo(float32).eps` (1.19e-07), not
    `finfo(float16).eps` (9.77e-04). The two agree to every printed digit on
    ordinary-magnitude input, so this uses 1e-3 values where the mean square
    is comparable to epsilon and the choice is visible.
    """
    for dtype in (_C.float16, _C.bfloat16):
        x = _C._tensor_from_flat([1e-3, 2e-3, 3e-3, 4e-3], [1, 4], dtype=dtype)
        got = _flat(_rms(x, [4]))
        # mean square = 7.5e-06. With f32 eps: 1e-3/sqrt(7.5e-6 + 1.19e-7).
        with_f32_eps = 1e-3 / (7.5e-06 + 1.1920928955078125e-07) ** 0.5
        with_f16_eps = 1e-3 / (7.5e-06 + 9.765625e-04) ** 0.5
        assert abs(got[0] - with_f32_eps) < 5e-3, (dtype, got[0], with_f32_eps)
        assert abs(got[0] - with_f16_eps) > 0.1, (
            "the default eps came from the INPUT dtype, not the accumulation "
            "dtype -- an order of magnitude wrong, and invisible at ordinary "
            "magnitudes")


def test_rms_norm_weight_multiplies_after_the_normalisation():
    x = _t([1, 2, 3, 4], (1, 4))
    plain = _flat(_rms(x, [4]))
    w = _t([1, 2, 3, 4], (4,))
    scaled = _flat(_rms(x, [4], w))
    for i, (p, sc) in enumerate(zip(plain, scaled)):
        assert abs(sc - p * (i + 1)) < 1e-5, (i, p, sc)


def test_rms_norm_normalized_shape_selects_how_many_trailing_axes():
    x = _t(range(1, 9), (2, 4))
    last_only = _flat(_rms(x, [4]))
    whole = _flat(_rms(x, [2, 4]))
    assert last_only != whole, "normalized_shape=[2,4] reduced only the last axis"
    # [2,4] is one row of eight: mean square = (1+4+...+64)/8 = 25.5
    assert abs(whole[0] - 1 / 25.5 ** 0.5) < 1e-5, whole[0]


def test_rms_norm_refusals_are_upstreams_including_the_ones_layer_norm_words_differently():
    x = _t([0] * 8, (2, 4))
    try:
        _rms(x, [3])
    except Exception as e:
        # `[*3]`, NOT layer_norm's `[*, 3]`.
        assert "expected input with shape [*3]" in str(e), e
    else:
        raise AssertionError("accepted a mismatched normalized_shape")

    try:
        _rms(x, [1, 2, 4])
    except ValueError as e:
        assert "at least 3 dimensions, but got 2" in str(e), e
    except Exception as e:
        raise AssertionError(f"expected ValueError, got {type(e).__name__}: {e}")

    try:
        _rms(x, [])
    except Exception as e:
        assert "at least 1-dimensional" in str(e), e

    try:
        _rms(x, [4], _t([1, 1, 1], (3,)))
    except Exception as e:
        assert "same shape as normalized_shape" in str(e), e

    b = _C._tensor_from_flat([1.0] * 8, [2, 4], dtype=_C.int64)
    try:
        _rms(b, [4])
    except Exception as e:
        # Named for the OP, where layer_norm names its kernel.
        assert '"rms_norm" not implemented for' in str(e), e
    else:
        raise AssertionError("an integer input was accepted")


_RMS_VENDOR_PROBE = """
import json, torch
out = {"is_shim": hasattr(torch._C, "_aten_implemented")}
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
try:
    # The free-function spelling a model reaches the kernel through.
    r = torch.rms_norm(x, [4])
    out["rms_norm"] = ["ok", [round(float(v), 6) for v in r.flatten()]]
except Exception as e:
    out["rms_norm"] = ["raised", f"{type(e).__name__}: {e}"]
try:
    r = torch.rms_norm(x, [4], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    out["rms_norm_weight"] = ["ok", [round(float(v), 6) for v in r.flatten()]]
except Exception as e:
    out["rms_norm_weight"] = ["raised", f"{type(e).__name__}: {e}"]
print(json.dumps(out))
"""


def test_torch_rms_norm_reaches_its_kernel_in_the_vendored_tree():
    """The spelling, not the kernel.

    `tools/golden/compare.py` dispatches by key, so it cannot see whether any
    Python name reaches the arm -- docs/REACH.md §1's shape 3, which has bitten
    four times. `torch.rms_norm` exists upstream, so unlike this round's six
    padding kernels it gets a real `overloads.json` row rather than an
    allowlist entry, and this is what proves the row resolves.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _RMS_VENDOR_PROBE],
                          capture_output=True, text=True, env=env, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"vendored probe exited {proc.returncode}\n"
                           f"{proc.stdout}\n{proc.stderr}")
    r = json.loads(proc.stdout)
    assert r["is_shim"] is True, "the probe imported upstream torch, not the shim"
    assert r["rms_norm"][0] == "ok", r["rms_norm"]
    # 1/sqrt(7.5) scaled by the input -- upstream's values.
    assert r["rms_norm"][1] == [0.365148, 0.730297, 1.095445, 1.460594], r["rms_norm"]
    assert r["rms_norm_weight"][0] == "ok", r["rms_norm_weight"]


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
