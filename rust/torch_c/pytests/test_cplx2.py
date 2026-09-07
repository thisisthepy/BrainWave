"""The complex ops `fnet` and `llama4`'s vision tower stop on.

`docs/kernels/COMPLEX2.md` built `Repr::Complex { re, im }` and taught it twelve ops.
`docs/bindings/BIND3.md` §6 then measured *exactly* which further ops two architectures
still stop on, and this file is the proof for the round that taught them:
`aten._to_copy.default` (real -> complex), `aten.slice.Tensor`,
`aten.constant_pad_nd.default`, `aten.view.default` / `aten._unsafe_view.default`,
and `aten.complex.default`. docs/kernels/COMPLEX3.md.

**Why every assertion here is about the imaginary part.**

Four of the five are *shape* ops, and a shape op is the easiest place in this
representation to be wrong while looking right. Writing `slice` as "narrow
`re`, hand back a `Dense`" gives a result with the correct shape, the correct
element count and a plausible magnitude; so does "narrow both halves but pad
only `re`". `docs/kernels/COMPLEX2.md` §2.3 measured what that class of mistake costs
-- six of ten sampled ops computing silently once the central refusal was
nullified -- and none of those six was detectable by a shape check.

So:

*   every value is reported through `torch.view_as_real(z).tolist()`, the one
    spelling that shows **both** halves, and compared element-wise against a
    live upstream torch **in a separate process**;
*   every complex input has `re != im` at every element, and several have
    `im` of the opposite sign, because an input with `im == re` cannot tell a
    kernel that moved both halves apart from one that moved `re` twice;
*   the fill test uses a **non-zero** pad value, because upstream pads with
    `complex(value, 0)` and with `value = 0` the right answer and the wrong
    one (`value + value*i`) are the same tensor;
*   and `test_the_untaught_ops_still_refuse` re-checks that closing five ops
    did not open the four hundred that were never taught.

Two narrowings are asserted *as narrowings* rather than left to be found:
upstream's `slice` and `view` return aliases of their base and these return
copies, for the reason `view_as_complex` already copies (docs/kernels/COMPLEX2.md §6)
-- a pair of tensors cannot alias an interleaved buffer.

Nothing here needs numpy or a network. It does need the vendored shim
(`vendor/install_shim.sh`); without it every probe skips.
"""

import json
import math
import os
import subprocess
import sys

from test_shim import _C  # noqa: F401  (import-time marker, as the siblings do)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

# The same tolerance and the same reasoning as `test_complex.py`: float32 on two
# sides that do not share a libm. It is ~8 ulp at unit magnitude and is orders
# of magnitude tighter than any dropped imaginary part, which moves a value by
# its whole size rather than by an ulp.
_TOL = 1e-6


_PROBE = r"""
import json, sys
import torch

MARKER = "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"

def pair(z):
    # The ONLY reporting spelling used in this file. `.real` alone is exactly
    # the read that cannot see the bug this file exists to catch.
    return torch.view_as_real(z).tolist()

def refusal(fn):
    try:
        r = fn()
    except Exception as e:
        return {"raised": type(e).__name__, "msg": str(e)}
    return {"ok": repr(r)[:200]}

out = {"_marker": MARKER}

# -- the inputs -----------------------------------------------------------
#
# `im` differs from `re` at every element and changes sign, so a kernel that
# applied its shape change to `re` twice, or that rebuilt `im` from `re`,
# cannot produce these numbers.
BASE1 = [[1., -2.], [3., -7.], [5., 11.], [-4., 9.], [6., -13.], [0.5, 8.]]
z = torch.view_as_complex(torch.tensor(BASE1))              # (6,)
BASE2 = [[[1., -2.], [3., -7.], [5., 11.]],
         [[-4., 9.], [6., -13.], [0.5, 8.]]]
z2 = torch.view_as_complex(torch.tensor(BASE2))             # (2, 3)

out["z"] = pair(z)
out["z2"] = pair(z2)

# -- 1. _to_copy: real -> complex, the first op of every fftn decomposition --
x = torch.tensor([0.5, -1.5, 2.25, -3.0])
out["to_copy_f32_c64"] = pair(torch.ops.aten._to_copy.default(x, dtype=torch.complex64))
out["to_copy_f32_c64_dtype"] = str(
    torch.ops.aten._to_copy.default(x, dtype=torch.complex64).dtype)
out["to_copy_f64_c128"] = pair(
    torch.ops.aten._to_copy.default(x.double(), dtype=torch.complex128))
out["to_copy_bool_c64"] = pair(torch.ops.aten._to_copy.default(
    torch.tensor([True, False, True]), dtype=torch.complex64))
# complex -> complex: BOTH halves have to widen. Converting only `re` would
# not survive `PyTensorBase::complex`'s same-dtype invariant, and that is the
# point -- the invariant is what makes the failure impossible rather than
# merely untested.
w = torch.ops.aten._to_copy.default(z, dtype=torch.complex128)
out["to_copy_c64_c128"] = pair(w)
out["to_copy_c64_c128_dtype"] = str(w.dtype)

# -- 2. slice.Tensor ------------------------------------------------------
MAX = 9223372036854775807
out["slice_mid"] = pair(torch.ops.aten.slice.Tensor(z, 0, 1, 4))
out["slice_neg"] = pair(torch.ops.aten.slice.Tensor(z, 0, -4, -1))
out["slice_step2"] = pair(torch.ops.aten.slice.Tensor(z, 0, 0, MAX, 2))
out["slice_step3"] = pair(torch.ops.aten.slice.Tensor(z, 0, 1, MAX, 3))
out["slice_dim1"] = pair(torch.ops.aten.slice.Tensor(z2, 1, 1, 3))
out["slice_empty_shape"] = list(torch.ops.aten.slice.Tensor(z, 0, 2, 2).shape)

# -- 3. constant_pad_nd ---------------------------------------------------
out["pad_zero"] = pair(torch.ops.aten.constant_pad_nd.default(z, [1, 2]))
# **The one that separates the halves.** Upstream fills with complex(5, 0), so
# the padded `im` entries are 0 while the padded `re` entries are 5. Filling
# both with 5 gives 5+5j and would pass any magnitude or shape check.
out["pad_five"] = pair(torch.ops.aten.constant_pad_nd.default(z, [2, 1], 5.0))
out["pad_crop"] = pair(torch.ops.aten.constant_pad_nd.default(z, [-1, -2]))
out["pad_mixed"] = pair(torch.ops.aten.constant_pad_nd.default(z, [-1, 2]))
out["pad_2d"] = pair(torch.ops.aten.constant_pad_nd.default(z2, [1, 1, 0, 1]))

# -- 4. view --------------------------------------------------------------
out["view_flat"] = pair(torch.ops.aten.view.default(z2, [-1]))
out["view_23"] = pair(torch.ops.aten.view.default(z, [2, 3]))
out["view_wildcard"] = pair(torch.ops.aten.view.default(z, [3, -1]))
# `llama4`'s `reshape_for_broadcast` shape, which is where its vision tower
# stopped: `freqs_ci.view(1, S, 1, -1)`.
out["view_llama4"] = pair(torch.ops.aten.view.default(z2, [1, 2, 1, -1]))
out["view_llama4_shape"] = list(
    torch.ops.aten.view.default(z2, [1, 2, 1, -1]).shape)

# -- 5. complex(re, im) ---------------------------------------------------
re_t = torch.tensor([1., 2., 3.])
im_t = torch.tensor([-4., 5., -6.])
out["complex_ctor"] = pair(torch.complex(re_t, im_t))
out["complex_ctor_dtype"] = str(torch.complex(re_t, im_t).dtype)
out["complex_broadcast"] = pair(torch.complex(re_t, torch.tensor([[7.], [-8.]])))
out["ref_complex_mismatch"] = refusal(lambda: torch.complex(re_t, im_t.double()))
out["ref_complex_int"] = refusal(lambda: torch.complex(re_t.long(), im_t.long()))

# -- 6. the composition the two architectures actually walk ---------------
#
# `fftn`'s `s=` path in one line: pad an axis, trim another, transform.
padded = torch.ops.aten.constant_pad_nd.default(z2, [0, 0, 0, 1])
trimmed = torch.ops.aten.slice.Tensor(padded, 1, 0, 2)
out["fftn_shaped"] = pair(torch.ops.aten._fft_c2c.default(trimmed, [0, 1], 0, True))
c = torch.ops.aten._to_copy.default(
    torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7.0,
    dtype=torch.complex64)
out["fftn_from_real"] = pair(torch.ops.aten._fft_c2c.default(c, [0, 1], 0, True))
out["fftn_from_real_real"] = torch.real(
    torch.ops.aten._fft_c2c.default(c, [0, 1], 0, True)).tolist()

# -- 7. the narrowing: do slice and view alias their base? ----------------
b = torch.tensor([[1., 2.], [3., 4.]])
zb = torch.view_as_complex(b)
sl = torch.ops.aten.slice.Tensor(zb, 0, 0, 2)
vw = torch.ops.aten.view.default(zb, [2, 1])
b[0, 0] = 99.
out["slice_aliases_its_base"] = pair(sl)[0][0] == 99.0
out["view_aliases_its_base"] = pair(vw)[0][0][0] == 99.0

json.dump(out, sys.stdout)
"""


_cache = {}


def _run(side):
    """`_PROBE`'s result from `side`, or `None` to skip.

    Two processes, never one: this interpreter has real torch installed, so an
    in-process `import torch` would compare upstream against itself. The marker
    is asserted on both sides so a mis-wired environment fails loudly rather
    than passing vacuously.
    """
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
        f"the {side} probe failed to run at all:\n{proc.stderr[-4000:]}"
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


def _flat(x):
    if isinstance(x, list):
        for i in x:
            yield from _flat(i)
    else:
        yield x


def _close(a, b, key):
    fa, fb = list(_flat(a)), list(_flat(b))
    assert len(fa) == len(fb), f"{key}: {len(fa)} values vs {len(fb)}"
    assert fa, f"{key}: nothing was compared, which is not a passing test"
    for i, (x, y) in enumerate(zip(fa, fb)):
        assert math.isclose(x, y, rel_tol=_TOL, abs_tol=_TOL), (
            f"{key}[{i}]: shim {x!r} vs upstream {y!r}"
        )


def _imag_survived(pairs, key, expect_nonzero=True):
    """The assertion that is not a comparison.

    `_close` would still pass if the shim and upstream had *both* been asked
    for something whose imaginary part happens to be zero. This says, of the
    shim's own answer, that the imaginary column carries information: it is not
    identically zero, and it is not a copy of the real column.

    Every op in this file is checked with it as well as against upstream,
    because these two failures -- "im is zeros" and "im is re" -- are what a
    half-written shape op produces, and the second one survives a comparison
    against an input where `re == im`.
    """
    res = [v[0] for v in _pairs_of(pairs)]
    ims = [v[1] for v in _pairs_of(pairs)]
    assert res, f"{key}: no elements, which is not a passing check"
    if expect_nonzero:
        assert any(v != 0.0 for v in ims), (
            f"{key}: every imaginary component is zero. That is exactly what a "
            f"kernel that moved only `re` and rebuilt `im` as zeros produces, "
            f"and the input to this op had a non-zero imaginary part."
        )
    assert res != ims, (
        f"{key}: the imaginary column equals the real column. That is what a "
        f"kernel that applied its shape change to `re` twice produces."
    )


def _pairs_of(value):
    """The trailing `[re, im]` couples of a `view_as_real` result, flattened."""
    flat = list(_flat(value))
    assert len(flat) % 2 == 0, f"not a view_as_real result: {len(flat)} values"
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


# ---------------------------------------------------------------------------
# 1. `_to_copy` -- the op that made every `fftn` decomposition unreachable
# ---------------------------------------------------------------------------


def test_to_copy_builds_a_complex_tensor_from_a_real_one():
    """`x.to(torch.complex64)`, element-wise against upstream on both halves.

    `docs/bindings/BIND3.md` §6 item 1: this refuses with "dtype not storable by the
    candle backend", it is the *first* op of every `torch.fft.fftn`
    decomposition, and nothing downstream of it was reachable.

    The imaginary part of the result is genuinely zero here -- that is what
    upstream produces too -- so this is the one op whose `_imag_survived` check
    is switched off, and the widening test below is the one that shows both
    halves moving.
    """
    shim, up = _both()
    if shim is None:
        return
    for key in ("to_copy_f32_c64", "to_copy_f64_c128", "to_copy_bool_c64"):
        _close(shim[key], up[key], key)
    assert shim["to_copy_f32_c64_dtype"] == up["to_copy_f32_c64_dtype"] == "torch.complex64"
    # Said as a value as well as a comparison: a probe that somehow compared
    # upstream against itself still has to produce these.
    assert [re for re, _ in _pairs_of(shim["to_copy_f32_c64"])] == [0.5, -1.5, 2.25, -3.0]
    assert all(im == 0.0 for _, im in _pairs_of(shim["to_copy_f32_c64"]))


def test_to_copy_widens_both_halves_and_not_just_the_real_one():
    """`complex64 -> complex128` on a tensor whose halves differ.

    The case that separates "converted the pair" from "converted `re`": the
    result's imaginary column has to be `BASE1`'s second column, widened. A
    kernel that converted only `re` cannot even build its result --
    `PyTensorBase::complex` refuses a pair whose candle dtypes disagree -- so
    this test is checking that the invariant is where it is claimed to be as
    much as it is checking the arithmetic.
    """
    shim, up = _both()
    if shim is None:
        return
    _close(shim["to_copy_c64_c128"], up["to_copy_c64_c128"], "to_copy_c64_c128")
    assert shim["to_copy_c64_c128_dtype"] == up["to_copy_c64_c128_dtype"] == "torch.complex128"
    _imag_survived(shim["to_copy_c64_c128"], "to_copy_c64_c128")
    assert [im for _, im in _pairs_of(shim["to_copy_c64_c128"])] == [
        -2.0, -7.0, 11.0, 9.0, -13.0, 8.0
    ]


# ---------------------------------------------------------------------------
# 2. `slice.Tensor`
# ---------------------------------------------------------------------------


def test_slice_keeps_the_imaginary_part_on_every_form():
    """Five slices, including a negative range and two strides.

    `docs/bindings/BIND3.md` §6 measured this refusing on a genuine shim complex tensor
    (an `stft` output), which is half of why `fft_fftn`'s `s=` was out of reach
    independently of `_to_copy`.

    A strided slice is the form where "narrow both halves" is not enough on its
    own: `re` and `im` have to be indexed by the *same* picks. The kernel
    builds one index tensor and uses it twice for that reason, and
    `slice_step3` -- whose picks are 1 and 4, neither adjacent nor symmetric --
    is what would catch two independent index tensors drifting.
    """
    shim, up = _both()
    if shim is None:
        return
    for key in ("slice_mid", "slice_neg", "slice_step2", "slice_step3", "slice_dim1"):
        _close(shim[key], up[key], key)
        _imag_survived(shim[key], key)
    assert shim["slice_empty_shape"] == up["slice_empty_shape"] == [0]
    # Transcribed, so this cannot pass by comparing two identical wrongs.
    assert _pairs_of(shim["slice_step3"]) == [(3.0, -7.0), (6.0, -13.0)]


# ---------------------------------------------------------------------------
# 3. `constant_pad_nd`
# ---------------------------------------------------------------------------


def test_constant_pad_nd_pads_both_halves_and_splits_a_non_zero_fill():
    """The other half of `s=`, and the one op here whose two halves are *not*
    treated identically.

    Upstream pads a complex tensor with `complex(value, 0)` -- measured -- so a
    fill of `5.0` puts `5` in the padded real entries and `0` in the padded
    imaginary ones. Filling both halves with `5` gives `5+5j`, which has the
    right shape, the right dtype and a plausible magnitude.

    **With the default `value=0` the right answer and that wrong one are the
    same tensor**, which is why `pad_five` exists and why the assertion below
    names the padded entries by position rather than trusting the comparison.
    """
    shim, up = _both()
    if shim is None:
        return
    for key in ("pad_zero", "pad_five", "pad_crop", "pad_mixed", "pad_2d"):
        _close(shim[key], up[key], key)
        _imag_survived(shim[key], key)
    five = _pairs_of(shim["pad_five"])
    assert five[0] == (5.0, 0.0) and five[1] == (5.0, 0.0), five[:2]
    assert five[-1] == (5.0, 0.0), five[-1]
    assert five[2:-1] == [(1.0, -2.0), (3.0, -7.0), (5.0, 11.0),
                          (-4.0, 9.0), (6.0, -13.0), (0.5, 8.0)], five


# ---------------------------------------------------------------------------
# 4. `view` -- `llama4`'s vision tower
# ---------------------------------------------------------------------------


def test_view_reshapes_both_halves_at_llama4s_own_shape():
    """`docs/bindings/BIND3.md` §5's wall, and the only complex op that tower needed.

    `Llama4VisionRotaryEmbedding` -> `reshape_for_broadcast` ->
    `freqs_ci.view(*shape)`, refused because `view` was not among the taught
    ops. `view_llama4` is that call's shape.

    The wildcard is resolved against `re.elem_count()`, which is the *complex*
    element count: `Repr::Complex`'s shape is `re`'s shape (docs/kernels/COMPLEX2.md
    §1.2), so there is no trailing 2 to correct for. Resolving against twice
    that would put the wrong extent in the wildcard and still satisfy every
    check made on `re` alone, which is why the shape is asserted as a value
    here as well as compared.
    """
    shim, up = _both()
    if shim is None:
        return
    for key in ("view_flat", "view_23", "view_wildcard", "view_llama4"):
        _close(shim[key], up[key], key)
        _imag_survived(shim[key], key)
    assert shim["view_llama4_shape"] == up["view_llama4_shape"] == [1, 2, 1, 3]
    assert _pairs_of(shim["view_23"]) == [(1.0, -2.0), (3.0, -7.0), (5.0, 11.0),
                                          (-4.0, 9.0), (6.0, -13.0), (0.5, 8.0)]


# ---------------------------------------------------------------------------
# 5. `torch.complex(re, im)`
# ---------------------------------------------------------------------------


def test_complex_constructor_and_its_two_refusals():
    """The third row of `docs/bindings/BIND3.md` §6's table.

    Both refusals are transcribed from a live upstream in the same run rather
    than from the C++, and compared byte for byte, because a message that has
    drifted is how a caller learns the wrong thing about their gap.
    """
    shim, up = _both()
    if shim is None:
        return
    _close(shim["complex_ctor"], up["complex_ctor"], "complex_ctor")
    _close(shim["complex_broadcast"], up["complex_broadcast"], "complex_broadcast")
    _imag_survived(shim["complex_ctor"], "complex_ctor")
    _imag_survived(shim["complex_broadcast"], "complex_broadcast")
    assert shim["complex_ctor_dtype"] == up["complex_ctor_dtype"] == "torch.complex64"
    assert _pairs_of(shim["complex_ctor"]) == [(1.0, -4.0), (2.0, 5.0), (3.0, -6.0)]
    for key in ("ref_complex_mismatch", "ref_complex_int"):
        assert "raised" in up[key], f"upstream no longer refuses {key}: {up[key]}"
        assert "raised" in shim[key], f"the shim computed {key}: {shim[key]}"
        assert shim[key]["msg"] == up[key]["msg"], (
            f"{key}:\n  shim     {shim[key]['msg']!r}\n"
            f"  upstream {up[key]['msg']!r}"
        )


# ---------------------------------------------------------------------------
# 6. The composition -- what `fft_fftn` walks
# ---------------------------------------------------------------------------


def test_the_fftn_shape_path_composes_end_to_end():
    """pad -> slice -> `_fft_c2c`, and real -> `_to_copy` -> `_fft_c2c`.

    `docs/bindings/BIND3.md` §6 traced `torch.fft.fftn` upstream and got
    `_to_copy(dtype=complex64)`, then `slice.Tensor` and `constant_pad_nd` when
    `s=` is given, then `_fft_c2c`. Each of those four is proven separately
    above; this runs them in sequence, because the failure a per-op test cannot
    see is a result that is individually right and does not compose -- a
    non-contiguous half that the next kernel reads through the wrong layout.

    `_fft_c2c` was already correct on a complex tensor (docs/kernels/FFT.md); what is
    new is that its *input* can now be built from real data.
    """
    shim, up = _both()
    if shim is None:
        return
    _close(shim["fftn_shaped"], up["fftn_shaped"], "fftn_shaped")
    _close(shim["fftn_from_real"], up["fftn_from_real"], "fftn_from_real")
    _close(shim["fftn_from_real_real"], up["fftn_from_real_real"], "fftn_from_real_real")
    _imag_survived(shim["fftn_shaped"], "fftn_shaped")
    _imag_survived(shim["fftn_from_real"], "fftn_from_real")


# ---------------------------------------------------------------------------
# 7. The narrowings, written down rather than discovered
# ---------------------------------------------------------------------------


def test_slice_and_view_copy_where_upstream_aliases():
    """Upstream's `slice` and `view` return views of their base; these copy.

    Same narrowing as `view_as_complex` (docs/kernels/COMPLEX2.md §6) and the same
    cause: a pair of real tensors cannot alias an interleaved buffer, and
    `view_as_complex` allocates unconditionally, so **no complex tensor in this
    shim shares storage with anything.** That makes the difference
    unobservable for anything built through the shim's own constructors -- but
    it is a difference, and this asserts it in the direction it actually holds
    rather than leaving a reader to find it.

    If either of these ever becomes `True`, that is an *improvement* and the
    fix is to remove the narrowing from docs/kernels/COMPLEX3.md first.
    """
    shim, up = _both()
    if shim is None:
        return
    assert up["slice_aliases_its_base"] is True, (
        "upstream's slice stopped aliasing -- the narrowing this asserts is "
        "measured against upstream, not assumed, so re-measure before editing"
    )
    assert up["view_aliases_its_base"] is True
    assert shim["slice_aliases_its_base"] is False, (
        "the shim's complex slice now aliases its base. docs/kernels/COMPLEX3.md "
        "records the copy as a narrowing; remove it there before removing it "
        "here."
    )
    assert shim["view_aliases_its_base"] is False


# ---------------------------------------------------------------------------
# 8. Closing five ops did not open four hundred
# ---------------------------------------------------------------------------


_UNTAUGHT = """
import json, sys
import torch
assert hasattr(torch._C, "_aten_implemented")
z = torch.view_as_complex(torch.tensor([[1., -2.], [3., -7.]]))
PROBES = {
    "sum": lambda: z.sum(),
    "add": lambda: z + z,
    "abs": lambda: z.abs(),
    "tolist": lambda: z.tolist(),
    "matmul": lambda: z @ z,
    "reshape": lambda: z.reshape(2, 1),
    "to_float32": lambda: z.to(torch.float32),
    "select": lambda: z[0],
    "cat": lambda: torch.cat([z, z]),
    "transpose": lambda: torch.ops.aten.transpose.int(z.view(1, 2), 0, 1),
    "permute": lambda: torch.ops.aten.permute.default(z.view(1, 2), [1, 0]),
    "index_select": lambda: torch.ops.aten.index_select.default(
        z, 0, torch.tensor([0])),
}
out = {}
for name, fn in PROBES.items():
    try:
        r = fn()
    except Exception as e:
        out[name] = {"raised": type(e).__name__, "msg": str(e)}
    else:
        out[name] = {"ok": repr(r)[:200]}
json.dump(out, sys.stdout)
"""


def test_the_untaught_ops_still_refuse():
    """**The property the representation rests on, re-checked after widening it.**

    `test_complex.py` samples ten ops that must refuse; this samples twelve,
    including three that are *neighbours of the ops this round taught* --
    `transpose`, `permute` and `index_select` are shape ops on the same
    tensors, reached through the same dispatcher, and if the guards had been
    written at the wrong level (on the tag rather than on the arm, or on a
    shared helper rather than on the five keys) they would have started
    computing from `re` alone.

    `reshape` and `select` are the sharpest: `view` is now taught and `reshape`
    is not, `slice` is now taught and `select` is not. A guard that leaked from
    one to the other is the specific mistake this widening could make, and each
    of those two would return a plausible half-tensor rather than raising.

    Every refusal must also **name the dtype**, because the reader's next
    question is whether their gap is the dtype or the operator.
    """
    if not os.path.isfile(_VENDOR_SHIM):
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _UNTAUGHT],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    data = json.loads(proc.stdout)
    computed = sorted(k for k, v in data.items() if "ok" in v)
    assert not computed, (
        f"these ops computed something on a complex tensor without being "
        f"taught the representation: {computed}. Whatever came back was "
        f"derived from the real part alone. If one of them was genuinely "
        f"implemented, move it out of this list and give it an element-wise "
        f"both-components comparison against upstream above."
    )
    for name, r in sorted(data.items()):
        assert r["raised"] == "NotImplementedError", f"{name}: {r}"
        assert "complex64" in r["msg"], (
            f"{name} refused without naming the dtype, so the reader cannot "
            f"tell whether the gap is the dtype or the operator: {r['msg']}"
        )


def test_the_five_taught_keys_are_all_still_dispatchable():
    """A cheap structural check that the guards did not move a key.

    Each of the four guarded keys has a dense kernel that golden compares, so
    it must stay in `_aten_implemented()`; `aten.complex.default` returns a
    complex tensor, which golden cannot read, so it must stay *out* of it and
    in the parked list. Getting this backwards is how an op silently stops
    being compared (docs/kernels/COMPLEX2.md's note on `IMPLEMENTED_AWAITING_GOLDEN`).
    """
    implemented = set(_C._aten_implemented())
    parked = set(_C._aten_implemented_awaiting_golden())
    for op in ("aten._to_copy.default", "aten.slice.Tensor",
               "aten.constant_pad_nd.default", "aten.view.default",
               "aten._unsafe_view.default"):
        assert op in implemented, (
            f"{op} left _aten_implemented(). Its dense path is unchanged by "
            f"the complex guard; if it moved, golden stopped comparing it."
        )
        assert op not in parked, op
    assert "aten.complex.default" in parked, (
        "aten.complex.default is advertised to golden, which has no way to "
        "read a complex result"
    )
    assert "aten.complex.default" not in implemented


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
