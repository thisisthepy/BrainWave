"""The FFT: `torch.stft` produces upstream's numbers, and how that was checked.

`docs/kernels/COMPLEX.md` set an ordering -- reflect pad, then complex tensors, then the
transform. `docs/kernels/PAD.md` cleared the first and `docs/kernels/COMPLEX2.md` the second;
this file is the third one's evidence. `docs/kernels/FFT.md` is the write-up.

**Nothing here is a shape check, and that is the point.** Every failure mode
this round can have produces a plausible spectrum:

* a **wrong twiddle sign** returns the complex conjugate -- same magnitudes,
  same shape, same dtype. A real *symmetric* input cannot see it, because its
  spectrum is its own conjugate. Every input below is deliberately asymmetric,
  and `[1, 2, 4, 8, 3]` is there because it is asymmetric AND of length 5,
  which is the only way to reach the Bluestein path. **This test caught exactly
  that defect on its first run** (docs/kernels/FFT.md §2).
* a **wrong normalisation** scales everything by `n` or `sqrt(n)`. Uniform, so
  it still looks like a spectrum. All three codes are pinned, in both
  directions, and the sizes 8 and 5 are used so that `n` and `sqrt(n)` are
  distinguishable from each other and from 1.
* **`onesided` returning `n` bins instead of `n // 2 + 1`** is self-consistent.
  Both the count and the values are asserted.
* a **wrong window placement** when `win_length < n_fft` shifts energy between
  bins without changing the shape.

Every positive assertion runs the same probe script under two interpreters --
the vendored shim, and upstream torch with the environment stripped -- and
compares element-wise. Each side asserts its own marker, so a mis-wired
environment fails loudly instead of comparing something against itself. That is
`test_complex.py`'s harness and it is reused verbatim.
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

# The shim transforms in `f64` on the host and upstream transforms in `f32`, so
# the residual here is *upstream's* rounding rather than this code's. Measured
# across the 27 stft cases and the 29 `_fft_*` ones, the worst relative
# disagreement is 4.9e-7 -- about 4 ulp of float32. 1e-5 is twenty times that
# and still nowhere near wide enough to hide any of the four failure modes
# above: a conjugation moves a value by twice its own magnitude, and the
# narrowest wrong normalisation (sqrt(16) vs 1) is a factor of four.
_TOL = 1e-5


_PROBE = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
import torch

MARKER = "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"
out = {"_marker": MARKER}

if MARKER == "shim":
    # docs/kernels/PAD.md section 5's four-line composite, installed IN THIS PROCESS
    # only. `torch._C._nn.pad` lives in `bootstrap.py`, which was not this
    # round's file, and its `reflect` branch is the one line between this shim
    # and `torch.stft(center=True)`. Installing it here exercises the routing
    # without editing another agent's file -- exactly what docs/kernels/PAD.md section
    # 4 did, and the keys below whose names start with `center_` are the ones
    # that depend on it.
    _d = torch._C._aten_dispatch
    def _pad(input, pad, mode="constant", value=None):
        n = len(pad) // 2
        if mode == "constant":
            return _d("aten.constant_pad_nd.default", input, list(pad),
                      0.0 if value is None else value)
        if mode == "reflect":
            return _d("aten.reflection_pad%dd.default" % n, input, list(pad))
        if mode == "replicate":
            return _d("aten.replication_pad%dd.default" % n, input, list(pad))
        raise NotImplementedError(mode)
    torch._C._nn.pad = _pad

def rec(key, fn):
    try:
        r = fn()
        if r.is_complex():
            r = torch.view_as_real(r)
        out[key] = {"shape": list(r.shape), "v": r.flatten().tolist()}
    except Exception as exc:
        out[key] = {"err": "%s: %s" % (type(exc).__name__, exc)}

# ---- the transform -------------------------------------------------------
# Asymmetric on purpose. x8 is a power of two (radix-2); x5 is not (Bluestein).
x8 = torch.tensor([1.0, 2.0, 4.0, 8.0, 3.0, -1.0, 0.5, 7.0])
x5 = torch.tensor([1.0, 2.0, 4.0, 8.0, 3.0])
for nrm in (0, 1, 2):
    rec("r2c8_n%d" % nrm, lambda n=nrm: torch.ops.aten._fft_r2c.default(x8, [0], n, True))
    rec("r2c5_n%d" % nrm, lambda n=nrm: torch.ops.aten._fft_r2c.default(x5, [0], n, True))
rec("r2c8_twosided", lambda: torch.ops.aten._fft_r2c.default(x8, [0], 0, False))
rec("r2c5_twosided", lambda: torch.ops.aten._fft_r2c.default(x5, [0], 0, False))
rec("r2c2d_dim0", lambda: torch.ops.aten._fft_r2c.default(x8.reshape(2, 4), [0], 0, True))
rec("r2c2d_dim1", lambda: torch.ops.aten._fft_r2c.default(x8.reshape(2, 4), [1], 0, True))
rec("r2c2d_both", lambda: torch.ops.aten._fft_r2c.default(x8.reshape(2, 4), [0, 1], 0, True))
rec("r2c_f64", lambda: torch.ops.aten._fft_r2c.default(x8.double(), [0], 0, True))
rec("r2c_n0", lambda: torch.ops.aten._fft_r2c.default(torch.zeros(0), [0], 0, True))
rec("r2c_half", lambda: torch.ops.aten._fft_r2c.default(x8.half(), [0], 0, True))
rec("r2c_long", lambda: torch.ops.aten._fft_r2c.default(torch.arange(8), [0], 0, True))
# whisper's length: 400 is NOT a power of two (docs/kernels/FFT.md section 4)
x400 = torch.sin(torch.arange(400.) * 0.017) + 0.3 * torch.cos(torch.arange(400.) * 0.11)
rec("r2c400", lambda: torch.ops.aten._fft_r2c.default(x400, [0], 0, True))

X8 = torch.view_as_complex(torch.stack(
    [x8, torch.tensor([0.5, -2.0, 1.0, 0.0, 3.0, 4.0, -5.0, 6.0])], -1))
for nrm in (0, 1, 2):
    for fwd in (True, False):
        rec("c2c8_n%d_f%d" % (nrm, fwd),
            lambda n=nrm, f=fwd: torch.ops.aten._fft_c2c.default(X8, [0], n, f))
X5 = torch.view_as_complex(torch.stack(
    [x5, torch.tensor([0.5, -2.0, 1.0, 0.0, 3.0])], -1))
rec("c2c5", lambda: torch.ops.aten._fft_c2c.default(X5, [0], 0, True))
rec("c2c_on_real", lambda: torch.ops.aten._fft_c2c.default(x8, [0], 0, True))
rec("r2c_on_complex", lambda: torch.ops.aten._fft_r2c.default(X8, [0], 0, True))

R8 = torch.ops.aten._fft_r2c.default(x8, [0], 0, True)
for nrm in (0, 1, 2):
    rec("c2r8_n%d" % nrm, lambda n=nrm: torch.ops.aten._fft_c2r.default(R8, [0], n, 8))
rec("c2r_last9", lambda: torch.ops.aten._fft_c2r.default(R8, [0], 2, 9))
rec("c2r_last7", lambda: torch.ops.aten._fft_c2r.default(R8, [0], 2, 7))
R5 = torch.ops.aten._fft_r2c.default(x5, [0], 0, True)
rec("c2r5", lambda: torch.ops.aten._fft_c2r.default(R5, [0], 2, 5))

# ---- the framing ---------------------------------------------------------
sig = (torch.sin(torch.arange(64.) * 0.31)
       + 0.4 * torch.cos(torch.arange(64.) * 1.7)
       + torch.arange(64.) * 0.01)
w16 = torch.hann_window(16)
A = torch.ops.aten.stft
rec("stft_basic", lambda: A.default(sig, 16, 4, 16, w16, False, True, True, None))
rec("stft_normalized", lambda: A.default(sig, 16, 4, 16, w16, True, True, True, None))
rec("stft_twosided", lambda: A.default(sig, 16, 4, 16, w16, False, False, True, None))
rec("stft_return_real", lambda: A.default(sig, 16, 4, 16, w16, False, True, False, None))
rec("stft_no_window", lambda: A.default(sig, 16, 4, 16, None, False, True, True, None))
rec("stft_win_short", lambda: A.default(sig, 16, 4, 8, torch.hann_window(8), False, True, True, None))
rec("stft_win_short_odd", lambda: A.default(sig, 16, 4, 7, torch.hann_window(7), False, True, True, None))
rec("stft_defaults", lambda: A.default(sig, 16, None, None, w16, False, None, True, None))
rec("stft_hop1", lambda: A.default(sig, 16, 1, 16, w16, False, True, True, None))
rec("stft_batch", lambda: A.default(torch.stack([sig, sig * 0.5 + 1]), 16, 4, 16, w16, False, True, True, None))
rec("stft_f64", lambda: A.default(sig.double(), 16, 4, 16, w16.double(), False, True, True, None))
rec("stft_nfft20", lambda: A.default(sig, 20, 5, 20, torch.hann_window(20), False, True, True, None))
rec("stft_c_reflect", lambda: A.center(sig, 16, 4, 16, w16, True, "reflect", False, True, True, None))
rec("stft_c_replicate", lambda: A.center(sig, 16, 4, 16, w16, True, "replicate", False, True, True, None))
rec("stft_c_off", lambda: A.center(sig, 16, 4, 16, w16, False, "reflect", False, True, True, None))

# refusals, all eight compared against a LIVE upstream rather than a transcript
rec("e_return_complex_missing", lambda: A.default(sig, 16, 4, 16, w16, False, True, None, None))
rec("e_nfft_too_big", lambda: A.default(torch.ones(8), 16, 4, 16, w16, False, True, True, None))
rec("e_hop_zero", lambda: A.default(sig, 16, 0, 16, w16, False, True, True, None))
rec("e_win_gt_nfft", lambda: A.default(sig, 16, 4, 20, torch.hann_window(20), False, True, True, None))
rec("e_window_size", lambda: A.default(sig, 16, 4, 16, torch.hann_window(8), False, True, True, None))
rec("e_rank3", lambda: A.default(torch.ones(2, 2, 64), 16, 4, 16, w16, False, True, True, None))
rec("e_integral", lambda: A.default(torch.arange(64), 16, 4, 16, w16, False, True, True, None))
rec("e_align_to_window", lambda: A.center(sig, 16, 4, 16, w16, True, "reflect", False, True, True, True))

# ---- torch.stft, the public entry point ----------------------------------
rec("public_center_false", lambda: torch.stft(sig, 16, 4, window=w16, center=False, return_complex=True))
rec("public_center_false_real", lambda: torch.stft(sig, 16, 4, window=w16, center=False, return_complex=False))
rec("center_true", lambda: torch.stft(sig, 16, 4, window=w16, center=True, return_complex=True))
rec("center_true_normalized", lambda: torch.stft(sig, 16, 4, window=w16, center=True, normalized=True, return_complex=True))
rec("center_true_replicate", lambda: torch.stft(sig, 16, 4, window=w16, center=True, pad_mode="replicate", return_complex=True))
wsig = torch.sin(torch.arange(1600.) * 0.02) + 0.2 * torch.cos(torch.arange(1600.) * 0.9)
rec("public_whisper400", lambda: torch.stft(wsig, 400, 160, window=torch.hann_window(400), center=False, return_complex=True))
rec("center_true_whisper400", lambda: torch.stft(wsig, 400, 160, window=torch.hann_window(400), center=True, return_complex=True))

json.dump(out, sys.stdout)
"""

_cache = {}


def _run(side):
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
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"the {side} probe failed to run at all:\n{proc.stderr[-3000:]}"
    )
    data = json.loads(proc.stdout)
    assert data["_marker"] == side, (
        f"the {side} probe imported the other library ({data['_marker']}) -- "
        f"every comparison below would have been about the wrong torch"
    )
    _cache[side] = data
    return data


def _both():
    s = _run("shim")
    if s is None:
        return None, None
    return s, _run("upstream")


def _same(shim, up, key):
    """Element-wise, both halves, and it refuses to pass on an empty result."""
    a, b = shim[key], up[key]
    assert "err" not in b, f"{key}: upstream itself refused ({b['err']}) -- fix the probe"
    assert "err" not in a, f"{key}: the shim refused where upstream computed: {a['err']}"
    assert a["shape"] == b["shape"], f"{key}: shape {a['shape']} vs upstream {b['shape']}"
    assert a["v"], f"{key}: nothing was compared, which is not a passing test"
    scale = max((abs(v) for v in b["v"]), default=1.0) or 1.0
    for i, (x, y) in enumerate(zip(a["v"], b["v"])):
        assert abs(x - y) <= _TOL * scale, (
            f"{key}[{i}]: shim {x!r} vs upstream {y!r} "
            f"(tolerance {_TOL} relative to {scale})"
        )


def _same_error(shim, up, key):
    a, b = shim[key], up[key]
    assert "err" in b, f"{key}: upstream did NOT refuse -- this is not a refusal case"
    assert "err" in a, f"{key}: the shim computed where upstream refused: {a}"
    assert a["err"] == b["err"], f"{key}:\n  shim     {a['err']}\n  upstream {b['err']}"


# ---------------------------------------------------------------------------
# 1. The bar: torch.stft, on a real signal, element-wise
# ---------------------------------------------------------------------------


def test_torch_stft_produces_upstreams_numbers():
    """**This is the test the round is for.**

    `torch.stft` on a real signal, through the public Python entry point --
    not through `torch.ops.aten` -- with `center=False`, which is the form that
    needs nothing outside this round's territory. Both the complex form and the
    `return_complex=False` form, since they are different code paths at the
    end of the kernel.
    """
    shim, up = _both()
    if shim is None:
        return
    _same(shim, up, "public_center_false")
    _same(shim, up, "public_center_false_real")
    # `return_complex=False` is `view_as_real` of the complex answer, so the
    # two must agree with each other as well as with upstream. If they did not,
    # one of the two paths would be being compared against nothing.
    assert shim["public_center_false"]["v"] == shim["public_center_false_real"]["v"]


def test_torch_stft_with_centring_needs_bootstrap_pys_pad_branch_and_then_agrees():
    """`center=True` -- and the honest statement of where it stands.

    `torch/functional.py:678` does the centring in Python, as
    `F.pad(input.view(...), [n_fft // 2] * 2, pad_mode)`. That reaches
    `torch._C._nn.pad`, which lives in `bootstrap.py` and still refuses every
    non-constant mode by name; docs/kernels/PAD.md section 5 has the exact four-line
    patch and `bootstrap.py` was not this round's file either.

    So the probe installs that composite **in its own process** and compares.
    It agrees with upstream. What this test asserts is therefore precise: the
    kernels are right and the remaining step is binding surface in another
    file. It is written this way rather than skipped so that it **starts
    failing** if either the pad kernels or the transform drift.
    """
    shim, up = _both()
    if shim is None:
        return
    _same(shim, up, "center_true")
    _same(shim, up, "center_true_normalized")
    _same(shim, up, "center_true_replicate")
    _same(shim, up, "center_true_whisper400")
    # The centred output is genuinely longer than the uncentred one -- if the
    # pad had silently not happened, both would be 13 frames wide.
    assert shim["center_true"]["shape"][1] > shim["public_center_false"]["shape"][1]


def test_whisper_length_400_is_not_a_power_of_two_and_still_agrees():
    """**The measured refutation of "n_fft is always a power of two".**

    Scanning every `transformers` feature extractor for an `n_fft` default
    (docs/kernels/FFT.md section 4) finds `whisper`, `qwen3_asr` and `voxtral_realtime`
    at **400**. So this round did not refuse non-powers of two by name; it
    implemented Bluestein's algorithm, and this is the case that exercises it
    at a real model's size.
    """
    shim, up = _both()
    if shim is None:
        return
    assert 400 & 399, "400 is a power of two, which would make this test vacuous"
    _same(shim, up, "public_whisper400")
    _same(shim, up, "r2c400")
    assert shim["public_whisper400"]["shape"][0] == 201, shim["public_whisper400"]["shape"]


# ---------------------------------------------------------------------------
# 2. The transform, and the errors that look like answers
# ---------------------------------------------------------------------------


def test_a_wrong_twiddle_sign_would_show_here():
    """Asymmetric inputs, both the radix-2 and the Bluestein path.

    A conjugation error negates every imaginary component and leaves every
    real one and every magnitude alone. `x5` is length 5, so it is the ONLY
    input in this file that reaches `fft_bluestein` -- and that function
    shipped its first build with exactly this defect, caught by exactly this
    case (docs/kernels/FFT.md section 2).
    """
    shim, up = _both()
    if shim is None:
        return
    for key in ("r2c8_n0", "r2c5_n0", "r2c8_twosided", "r2c5_twosided",
                "r2c2d_dim0", "r2c2d_dim1", "r2c2d_both", "r2c_f64"):
        _same(shim, up, key)
    # Said as a property as well as a comparison: the imaginary parts are not
    # all zero and not all of one sign, so a conjugation is observable.
    im = up["r2c5_n0"]["v"][1::2]
    assert any(v > 1e-6 for v in im) and any(v < -1e-6 for v in im), im


def test_all_three_normalisation_codes_in_both_directions():
    """`0 -> 1`, `1 -> 1/sqrt(n)`, `2 -> 1/n`, forward and inverse.

    The codes are **not** a `norm=` string: `torch.fft.rfft`'s default reaches
    `_fft_r2c(..., 0, ...)` and `torch.fft.ifft`'s default reaches
    `_fft_c2c(..., 2, False)`. Both are `norm="backward"`. A shim that read the
    code as "which norm= string" would be right in one direction and wrong by a
    factor of `n` in the other -- and `n = 8` here, so the three codes give
    three visibly different answers.
    """
    shim, up = _both()
    if shim is None:
        return
    for nrm in (0, 1, 2):
        _same(shim, up, f"r2c8_n{nrm}")
        _same(shim, up, f"r2c5_n{nrm}")
        _same(shim, up, f"c2r8_n{nrm}")
        for fwd in (0, 1):
            _same(shim, up, f"c2c8_n{nrm}_f{fwd}")
    # The three codes really are three different answers, so the loop above is
    # not comparing the same numbers three times.
    dc = [up[f"r2c8_n{n}"]["v"][0] for n in (0, 1, 2)]
    assert len(set(dc)) == 3, dc
    assert math.isclose(dc[1], dc[0] / math.sqrt(8), rel_tol=1e-6), dc
    assert math.isclose(dc[2], dc[0] / 8, rel_tol=1e-6), dc


def test_onesided_returns_n_over_two_plus_one_bins():
    """`n // 2 + 1`, not `n`.

    A full-length output would have the right rank, the right dtype and a
    correct-looking first half. The count is asserted for an even length (8 ->
    5) and an odd one (5 -> 3), because `n // 2 + 1` and `(n + 1) // 2` agree
    on odd lengths and disagree on even ones.
    """
    shim, up = _both()
    if shim is None:
        return
    assert shim["r2c8_n0"]["shape"] == [5, 2], shim["r2c8_n0"]["shape"]
    assert shim["r2c5_n0"]["shape"] == [3, 2], shim["r2c5_n0"]["shape"]
    assert shim["r2c8_twosided"]["shape"] == [8, 2]
    assert shim["r2c5_twosided"]["shape"] == [5, 2]
    for key in ("r2c8_n0", "r2c5_n0", "r2c8_twosided", "r2c5_twosided"):
        assert shim[key]["shape"] == up[key]["shape"], key


def test_c2r_reads_last_dim_size_and_not_the_bin_count():
    """`last_dim_size` decides the output length AND how many bins are read.

    Measured: a 5-bin input with `last_dim_size=7` uses bins 0..3 and mirrors
    1..3 -- bin 4 is not part of a 7-point Hermitian spectrum. An
    implementation that reads all five bins returns a signal of the right
    length with the wrong values, and it was wrong by 0.58 on values of order
    10 before this was measured (docs/kernels/FFT.md section 3).
    """
    shim, up = _both()
    if shim is None:
        return
    for key in ("c2r8_n2", "c2r_last9", "c2r_last7", "c2r5"):
        _same(shim, up, key)
    assert shim["c2r_last7"]["shape"] == [7]
    assert shim["c2r_last9"]["shape"] == [9]
    # `_fft_c2r(_fft_r2c(x), 2, len(x))` is the identity, which is the property
    # a wrong bin count breaks without changing any shape.
    x8 = [1.0, 2.0, 4.0, 8.0, 3.0, -1.0, 0.5, 7.0]
    for got, want in zip(shim["c2r8_n2"]["v"], x8):
        assert abs(got - want) < 1e-4, (shim["c2r8_n2"]["v"], x8)


def test_the_dtype_refusals_are_upstreams_own_wording():
    """`float16`, `bfloat16`, integral and complex, against a live upstream.

    Upstream refuses `float16` with `expected scalar type Double but found
    Half`, which reads like a defect and is reproduced rather than tidied: a
    caller diagnosing a dtype problem matches on the message it actually gets.
    Compared against a running upstream rather than a transcription, so the
    assertion cannot go stale.
    """
    shim, up = _both()
    if shim is None:
        return
    for key in ("r2c_half", "r2c_long", "r2c_on_complex", "c2c_on_real"):
        _same_error(shim, up, key)
    assert "Half" in up["r2c_half"]["err"], up["r2c_half"]["err"]


def test_an_empty_transformed_axis_is_one_bin_and_not_an_error():
    """`_fft_r2c(zeros(0), [0], 0, True)` is a length-1 tensor upstream.

    `0 // 2 + 1 == 1`, and the sum of no elements is zero. Measured rather than
    reasoned about, because "an empty input is an error" is the obvious guess
    and it is wrong.
    """
    shim, up = _both()
    if shim is None:
        return
    _same(shim, up, "r2c_n0")
    assert shim["r2c_n0"]["shape"] == [1, 2], shim["r2c_n0"]["shape"]


# ---------------------------------------------------------------------------
# 3. The framing
# ---------------------------------------------------------------------------


def test_the_stft_framing_matches_upstream_across_the_options():
    """hop, window length, batching, dtype, both overloads and both norms."""
    shim, up = _both()
    if shim is None:
        return
    for key in ("stft_basic", "stft_normalized", "stft_twosided",
                "stft_return_real", "stft_no_window", "stft_win_short",
                "stft_win_short_odd", "stft_defaults", "stft_hop1",
                "stft_batch", "stft_f64", "stft_nfft20",
                "stft_c_reflect", "stft_c_replicate", "stft_c_off"):
        _same(shim, up, key)


def test_normalized_is_one_over_sqrt_n_fft_and_not_one_over_n_fft():
    """The uniform-factor error, pinned by a ratio rather than by a comparison.

    `normalized=True` is normalisation code **1**, traced. Code 2 would divide
    by 16 where this divides by 4 -- both produce a spectrum, and only the
    ratio distinguishes them.
    """
    shim, up = _both()
    if shim is None:
        return
    _same(shim, up, "stft_normalized")
    a = shim["stft_basic"]["v"]
    b = shim["stft_normalized"]["v"]
    biggest = max(range(len(a)), key=lambda i: abs(a[i]))
    ratio = a[biggest] / b[biggest]
    assert math.isclose(ratio, math.sqrt(16.0), rel_tol=1e-4), (
        f"normalized=True scaled by 1/{ratio}, expected 1/sqrt(16)=1/4"
    )


def test_a_short_window_is_centred_inside_n_fft():
    """`win_length < n_fft` pads the window, and the odd element goes RIGHT.

    `left = (n_fft - win_length) // 2`. With `n_fft=16` and `win_length=7` the
    split is 4 and 5, not 5 and 4 -- and either choice produces a spectrum of
    the right shape. The odd case is here for exactly that reason.
    """
    shim, up = _both()
    if shim is None:
        return
    _same(shim, up, "stft_win_short")
    _same(shim, up, "stft_win_short_odd")
    # A window that was not padded at all, or padded on one side only, gives a
    # different answer from the full-length window. Assert they differ, so that
    # a kernel which ignored `win_length` could not pass the comparison above
    # by accident.
    assert shim["stft_win_short"]["v"] != shim["stft_basic"]["v"]
    assert shim["stft_win_short_odd"]["v"] != shim["stft_win_short"]["v"]


def test_every_stft_refusal_is_upstreams_own_message():
    """Eight refusals, each compared against a live upstream.

    Upstream prefixes every one with the whole resolved call, and the details
    are not guessable: the tensor prints as `torch.FloatTensor[64]` and the
    window as `torch.FloatTensor{[16]}`, `normalized` prints as `0`/`1` and
    `onesided` as `None`/`0`/`1`, and `win_length` shows the resolved default.
    """
    shim, up = _both()
    if shim is None:
        return
    for key in ("e_return_complex_missing", "e_nfft_too_big", "e_hop_zero",
                "e_win_gt_nfft", "e_window_size", "e_rank3", "e_integral",
                "e_align_to_window"):
        _same_error(shim, up, key)
    # The `return_complex` check runs FIRST -- `e_nfft_too_big` would also be
    # an n_fft error, and upstream reports the missing parameter instead.
    assert "return_complex" in up["e_return_complex_missing"]["err"]


def test_n_fft_equal_to_the_signal_length_computes_despite_the_message():
    """`stft(randn(16), n_fft=16)` returns one frame, and the refusal that
    does NOT fire says `expected 0 < n_fft < 16`.

    The message states a strict bound and the code enforces `<=`. Reproduced
    rather than corrected; a caller matching on the text gets upstream's text.
    """
    shim, up = _both()
    if shim is None:
        return
    _same_error(shim, up, "e_nfft_too_big")
    assert "expected 0 < n_fft < 8" in up["e_nfft_too_big"]["err"], (
        up["e_nfft_too_big"]["err"]
    )


# ---------------------------------------------------------------------------
# 4. Surface bookkeeping
# ---------------------------------------------------------------------------


def test_the_three_fft_ops_are_parked_and_stft_is_advertised():
    """Where each of the five new ops sits, and why.

    `_fft_r2c` and `_fft_c2c` **return** a complex tensor and `_fft_c2r`
    **takes** one, so on at least one side of a golden comparison there is no
    dense storage to read -- the same reason `view_as_complex` is parked
    (docs/kernels/COMPLEX2.md section 5.1). `aten.stft.*` is real on both sides in its
    `return_complex=False` form, so it is golden-compared, which is what keeps
    the transform itself inside the harness rather than only inside this file.
    """
    parked = set(_C._aten_implemented_awaiting_golden())
    advertised = set(_C._aten_implemented())
    for op in ("aten._fft_r2c.default", "aten._fft_c2c.default",
               "aten._fft_c2r.default"):
        assert op in parked, f"{op} is advertised to golden but has no builder"
        assert op not in advertised, op
    for op in ("aten.stft.default", "aten.stft.center"):
        assert op in advertised, f"{op} left _aten_implemented()"
        assert op not in parked, op


def test_as_strided_landed_and_stft_still_does_not_use_it():
    """**Inverted, not deleted** -- this test's previous body asked for exactly
    this, and `docs/kernels/STRIDED.md` is the round it was addressed to.

    What it said: `as_strided` is unimplemented, `torch.stft` upstream reaches
    it and this `stft` does not, because the frames are extracted with a
    **gather** rather than a strided view. `docs/architectures/VOICE.md` §3 had named the two
    as `stft`'s walls and only `_fft_r2c` fell.

    `as_strided` has now landed (`docs/kernels/STRIDED.md`), and the note that body was
    written to leave still holds: **`stft` does not have to change.** That is
    the claim worth keeping, so it is what is asserted now, and it is asserted
    about the kernel rather than about the op list -- `stft`'s frames must
    still be an `index_select` gather and not a call into the new op.

    The reason it should stay that way is `docs/kernels/STRIDED.md` §2: an
    `as_strided` result bars in-place writes to its base's storage for as long
    as it lives. `stft` would be barring its own input for the duration of the
    window multiply, in exchange for aliasing nothing can observe -- the
    frames are consumed immediately. A gather is the right primitive here even
    now that the other one exists.
    """
    assert "aten.as_strided.default" in set(_C._aten_implemented()), (
        "as_strided left _aten_implemented(); if it was reverted, invert this "
        "test back rather than deleting it -- docs/kernels/STRIDED.md §1"
    )
    path = os.path.join(_REPO_ROOT, "rust", "torch_c", "src", "aten.rs")
    if not os.path.isfile(path):
        print("   (skipped: aten.rs is not beside this file)")
        return
    text = open(path, encoding="utf-8").read()
    start = text.index("fn stft_kernel(") if "fn stft_kernel(" in text else None
    assert start is not None, "stft's kernel was renamed; re-point this check"
    body = text[start:]
    body = body[:body.index("\nfn ", 10)]
    # Comments stripped first. `stft_kernel` says "Upstream is `as_strided`;
    # this is the same index arithmetic" in a comment, and reading that as a
    # call is the mistake this repository has made in the other direction
    # (commented-out code read as live).
    body = "\n".join(line for line in body.splitlines()
                     if not line.lstrip().startswith("//"))
    assert "as_strided" not in body, (
        "stft now routes through as_strided. That is a real change and may be "
        "right, but it makes stft bar its input's storage for the duration -- "
        "docs/kernels/STRIDED.md §2, docs/kernels/FFT.md §5.2."
    )
    assert "index_select" in body, body[:400]


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
