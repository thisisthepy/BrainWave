"""`torch.equal` and `torch.allclose` -- the gap a fixture found.

Neither had a table entry in `overloads.json` before this round, and both
raised `NotImplementedError` from `torch.<name>(...)` the moment a test called
them. They matter because they are what user code and test suites reach for
to compare tensors -- their absence shows up in *somebody else's* code, not
this shim's.

Measured first, against real torch 2.13.0 in
`/Volumes/macMini/caches/spike-venv` (`import torch as _upstream_torch`,
guarded exactly the way `test_shim.py`'s own "against real upstream torch,
live in the same process" section already does it -- `_C` here is the shim,
loaded standalone and never as `torch._C`, so the two do not collide):

  * A `TorchDispatchMode` logger shows both are **leaf ops**:
    `torch.equal(a, b)` fires exactly `aten.equal.default` and
    `torch.allclose(a, b)` fires exactly `aten.allclose.default`, with
    nothing decomposed under either. So both are genuine `overloads.json`
    entries (the single schema each carries resolves and calls straight
    through `_C._aten_dispatch`), not `bootstrap.py` composites -- unlike
    `dropout`/`layer_norm`/`isfinite`, which fire a *different* key than
    their public name and have to be composed from pieces that do exist.

  * Both **reduce to a Python `bool`, not a `Tensor`.** `_torch_level_function`
    in `bootstrap.py` does not care -- `dispatch(key, **bound)` returns
    whatever `aten.rs` hands back -- so the return leaving the tensor world
    changes nothing about where the *arguments* resolve, only what
    `aten.rs`'s kernel is allowed to answer with. `aten.rs`'s own
    `_local_scalar_dense` kernel (`item()`/`__bool__`'s door) already answers
    a bare Python value the same way; these two are the first *table*
    entries that do.

  * `torch.equal` does **not** check dtype -- `equal(int64([1,2,3]),
    int32([1,2,3]))` is `True`, `equal(int64([1,2,3]), int32([1,2,4]))` is
    `False`. It compares promoted *values*, the same rule `eq.Tensor`
    promotes by. Shape mismatch answers `False` and does not raise.

  * `torch.allclose` is the opposite on both counts: dtype mismatch
    **raises** (`RuntimeError: Float did not match Double`, upstream's own
    wording), and shape mismatch **broadcasts**, raising only if the
    broadcast itself refuses. Its tolerance is `|self - other| <= atol +
    rtol * |other|` -- asymmetric, verified both ways below.

This file is deliberately not part of `tools/golden/cases.py`: the golden
harness reads both sides as dense tensors and diffs values element-wise, and
neither op returns a tensor to diff. The comparison equivalent here is this
file's `test_*_agrees_with_upstream_*` pair, which counts agreements against
upstream directly rather than trusting a value written down once.
"""

import math
import os

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _C

# Real upstream torch, in the same process, never shadowing `_C` (this file
# never adds a vendored `torch/` package to `sys.path`). Every test in the
# agreement sections below no-ops if it is not importable, exactly as
# `test_shim.py`'s own e2e section does.
try:
    import torch as _upstream_torch
except ImportError:  # pragma: no cover - most interpreters running this file
    _upstream_torch = None


_DTYPES = {
    "float32": (_C.float32, "float32"),
    "float64": (_C.float64, "float64"),
    "int64": (_C.int64, "int64"),
    "int32": (_C.int32, "int32"),
    "int16": (_C.int16, "int16"),
    "uint8": (_C.uint8, "uint8"),
    "bool": (_C.bool, "bool"),
}


def _shim_tensor(values, shape, dtype_name):
    c_dtype, _ = _DTYPES[dtype_name]
    return _C._tensor_from_flat(values, shape, c_dtype)


def _upstream_tensor(values, shape, dtype_name):
    _, up_name = _DTYPES[dtype_name]
    up_dtype = getattr(_upstream_torch, up_name)
    flat = _upstream_torch.tensor(values, dtype=_upstream_torch.float64)
    return flat.to(up_dtype).reshape(shape)


# --------------------------------------------------------------------------
# Non-vacuous, pinned behaviour -- runs with only `_C`, no upstream needed.
# --------------------------------------------------------------------------


def test_equal_is_advertised_and_reachable_through_the_one_door():
    listed = _C._aten_implemented()
    assert "aten.equal.default" in listed, listed
    assert "aten.allclose.default" in listed, listed
    # Reachable: resolves and executes rather than raising NotImplementedError.
    a = _shim_tensor([1.0, 2.0], [2], "float32")
    b = _shim_tensor([1.0, 2.0], [2], "float32")
    assert _C._aten_dispatch("aten.equal.default", a, b) is True
    assert _C._aten_dispatch("aten.allclose.default", a, b) is True


def test_equal_true_for_identical_values():
    a = _shim_tensor([1.0, 2.0, 3.0], [3], "float32")
    b = _shim_tensor([1.0, 2.0, 3.0], [3], "float32")
    got = _C._aten_dispatch("aten.equal.default", a, b)
    assert got is True, got


def test_equal_false_for_one_differing_element():
    a = _shim_tensor([1.0, 2.0, 3.0], [3], "float32")
    b = _shim_tensor([1.0, 2.0, 3.5], [3], "float32")
    got = _C._aten_dispatch("aten.equal.default", a, b)
    assert got is False, got


def test_equal_ignores_dtype_and_compares_promoted_values():
    a = _shim_tensor([1.0, 2.0, 3.0], [3], "int64")
    b = _shim_tensor([1.0, 2.0, 3.0], [3], "int32")
    assert _C._aten_dispatch("aten.equal.default", a, b) is True
    c = _shim_tensor([1.0, 2.0, 4.0], [3], "int32")
    assert _C._aten_dispatch("aten.equal.default", a, c) is False


def test_equal_false_not_raise_on_shape_mismatch():
    a = _shim_tensor([1.0, 2.0], [2], "float32")
    b = _shim_tensor([1.0, 2.0, 3.0], [3], "float32")
    got = _C._aten_dispatch("aten.equal.default", a, b)
    assert got is False, got


def test_equal_true_for_matching_empty_shapes_false_for_mismatched_empty_shapes():
    a = _shim_tensor([], [0, 3], "float32")
    b = _shim_tensor([], [0, 3], "float32")
    assert _C._aten_dispatch("aten.equal.default", a, b) is True
    c = _shim_tensor([], [0, 4], "float32")
    assert _C._aten_dispatch("aten.equal.default", a, c) is False


def test_equal_nan_never_equals_itself():
    nan = float("nan")
    a = _shim_tensor([nan], [1], "float32")
    b = _shim_tensor([nan], [1], "float32")
    got = _C._aten_dispatch("aten.equal.default", a, b)
    assert got is False, got


def test_equal_same_signed_infinities_are_equal_opposite_signs_are_not():
    inf = float("inf")
    a = _shim_tensor([inf], [1], "float32")
    b = _shim_tensor([inf], [1], "float32")
    c = _shim_tensor([-inf], [1], "float32")
    assert _C._aten_dispatch("aten.equal.default", a, b) is True
    assert _C._aten_dispatch("aten.equal.default", a, c) is False


def test_allclose_default_tolerance_close_and_not_close():
    a = _shim_tensor([1.0], [1], "float32")
    b = _shim_tensor([1.0 + 1e-9], [1], "float32")
    assert _C._aten_dispatch("aten.allclose.default", a, b) is True
    c = _shim_tensor([1.5], [1], "float32")
    assert _C._aten_dispatch("aten.allclose.default", a, c) is False


def test_allclose_boundary_is_inclusive():
    # 2.0 - 2.5 is exactly -0.5 in binary, so this boundary is not a rounding
    # trap: atol=0.5 must include it, atol just under must exclude it.
    a = _shim_tensor([2.0], [1], "float32")
    b = _shim_tensor([2.5], [1], "float32")
    assert _C._aten_dispatch("aten.allclose.default", a, b, 0.0, 0.5, False) is True
    assert (
        _C._aten_dispatch("aten.allclose.default", a, b, 0.0, 0.4999999, False)
        is False
    )


def test_allclose_tolerance_is_asymmetric_in_self_and_other():
    # |1 - 100| = 99 <= 1.0 * |100| = 100  ->  True
    # |100 - 1| = 99 <= 1.0 * |1|   = 1    ->  False
    a = _shim_tensor([1.0], [1], "float32")
    b = _shim_tensor([100.0], [1], "float32")
    assert _C._aten_dispatch("aten.allclose.default", a, b, 1.0, 0.0, False) is True
    assert _C._aten_dispatch("aten.allclose.default", b, a, 1.0, 0.0, False) is False


def test_allclose_equal_nan_flag():
    nan = float("nan")
    a = _shim_tensor([nan], [1], "float32")
    b = _shim_tensor([nan], [1], "float32")
    assert _C._aten_dispatch("aten.allclose.default", a, b, 1e-05, 1e-08, False) is False
    assert _C._aten_dispatch("aten.allclose.default", a, b, 1e-05, 1e-08, True) is True
    # A NaN is never close to a non-NaN, equal_nan or not.
    c = _shim_tensor([1.0], [1], "float32")
    assert _C._aten_dispatch("aten.allclose.default", a, c, 1e-05, 1e-08, True) is False


def test_allclose_infinities_same_sign_close_opposite_sign_and_finite_not():
    inf = float("inf")
    a = _shim_tensor([inf], [1], "float32")
    b = _shim_tensor([inf], [1], "float32")
    c = _shim_tensor([-inf], [1], "float32")
    d = _shim_tensor([1.0], [1], "float32")
    assert _C._aten_dispatch("aten.allclose.default", a, b) is True
    assert _C._aten_dispatch("aten.allclose.default", a, c) is False
    assert _C._aten_dispatch("aten.allclose.default", a, d) is False


def test_allclose_dtype_mismatch_raises_with_upstreams_wording():
    a = _shim_tensor([1.0], [1], "float32")
    b = _shim_tensor([1.0], [1], "float64")
    try:
        _C._aten_dispatch("aten.allclose.default", a, b)
    except RuntimeError as e:
        assert str(e) == "Float did not match Double", str(e)
    else:
        raise AssertionError("dtype mismatch must raise")


def test_allclose_broadcasts_shape_and_raises_on_a_genuine_mismatch():
    a = _shim_tensor([1.0, 1.0, 1.0], [3], "float32")
    b = _shim_tensor([1.0], [], "float32")
    assert _C._aten_dispatch("aten.allclose.default", a, b) is True

    c = _shim_tensor([1.0, 1.0], [2], "float32")
    d = _shim_tensor([1.0, 1.0, 1.0], [3], "float32")
    try:
        _C._aten_dispatch("aten.allclose.default", c, d)
    except RuntimeError as e:
        assert "must match the size of tensor b" in str(e), str(e)
    else:
        raise AssertionError("shape mismatch that cannot broadcast must raise")


def test_allclose_empty_tensors_are_close():
    a = _shim_tensor([], [0], "float32")
    b = _shim_tensor([], [0], "float32")
    assert _C._aten_dispatch("aten.allclose.default", a, b) is True


# --------------------------------------------------------------------------
# Element-wise agreement against real upstream torch 2.13.0.
# --------------------------------------------------------------------------


def test_equal_agrees_with_upstream_across_dtypes_and_edges():
    if _upstream_torch is None:
        return
    nan, inf = float("nan"), float("inf")
    cases = []
    for dt in ("float32", "float64", "int64", "int32", "int16", "uint8", "bool"):
        vals = [0.0, 1.0, 1.0, 0.0] if dt == "bool" else [1.0, 2.0, 3.0, 4.0]
        cases.append((vals, [2, 2], dt, vals, [2, 2], dt))
        other = [1.0, 2.0, 3.0, 5.0] if dt != "bool" else [0.0, 1.0, 1.0, 1.0]
        cases.append((vals, [2, 2], dt, other, [2, 2], dt))
    # cross-dtype, same values and differing values
    cases.append(([1.0, 2.0, 3.0], [3], "int64", [1.0, 2.0, 3.0], [3], "int32"))
    cases.append(([1.0, 2.0, 3.0], [3], "int64", [1.0, 2.0, 4.0], [3], "int32"))
    # shape mismatch
    cases.append(([1.0, 2.0], [2], "float32", [1.0, 2.0, 3.0], [3], "float32"))
    cases.append(([], [0, 3], "float32", [], [0, 4], "float32"))
    # empty, matching
    cases.append(([], [0], "float32", [], [0], "float32"))
    # 0-d
    cases.append(([1.0], [], "float32", [1.0], [], "float32"))
    cases.append(([1.0], [], "float32", [2.0], [], "float32"))
    # nan / inf
    cases.append(([nan, 1.0], [2], "float32", [nan, 1.0], [2], "float32"))
    cases.append(([inf, 1.0], [2], "float32", [inf, 1.0], [2], "float32"))
    cases.append(([inf, 1.0], [2], "float32", [-inf, 1.0], [2], "float32"))

    agree = 0
    total = 0
    disagreements = []
    for a_vals, a_shape, a_dt, b_vals, b_shape, b_dt in cases:
        total += 1
        sa = _shim_tensor(a_vals, a_shape, a_dt)
        sb = _shim_tensor(b_vals, b_shape, b_dt)
        shim = _C._aten_dispatch("aten.equal.default", sa, sb)
        ua = _upstream_tensor(a_vals, a_shape, a_dt)
        ub = _upstream_tensor(b_vals, b_shape, b_dt)
        up = bool(_upstream_torch.equal(ua, ub))
        if shim == up:
            agree += 1
        else:
            disagreements.append((a_dt, a_shape, b_dt, b_shape, shim, up))
    assert not disagreements, f"{agree}/{total} agreed; disagreements: {disagreements}"
    assert agree == total == len(cases)
    print(f"ok   torch.equal agrees with upstream on {agree}/{total} cases")


def test_allclose_agrees_with_upstream_across_dtypes_and_edges():
    if _upstream_torch is None:
        return
    nan, inf = float("nan"), float("inf")
    # (a_vals, a_shape, a_dt, b_vals, b_shape, b_dt, rtol, atol, equal_nan)
    cases = []
    for dt in ("float32", "float64"):
        cases.append(([1.0, 2.0, 3.0], [3], dt, [1.0, 2.0, 3.0], [3], dt, 1e-05, 1e-08, False))
        cases.append(([1.0, 2.0, 3.0], [3], dt, [1.0, 2.0, 3.5], [3], dt, 1e-05, 1e-08, False))
        cases.append(([1.0], [1], dt, [1.0 + 1e-9], [1], dt, 1e-05, 1e-08, False))
    for dt in ("int64", "int32", "int16", "uint8", "bool"):
        vals = [0.0, 1.0, 1.0, 0.0] if dt == "bool" else [1.0, 2.0, 3.0, 4.0]
        cases.append((vals, [2, 2], dt, vals, [2, 2], dt, 1e-05, 1e-08, False))
        other = [1.0, 2.0, 3.0, 5.0] if dt != "bool" else [0.0, 1.0, 1.0, 1.0]
        cases.append((vals, [2, 2], dt, other, [2, 2], dt, 1e-05, 1e-08, False))
    # rtol/atol boundary, exact in binary
    cases.append(([2.0], [1], "float32", [2.5], [1], "float32", 0.0, 0.5, False))
    cases.append(([2.0], [1], "float32", [2.5], [1], "float32", 0.0, 0.4999999, False))
    # asymmetry
    cases.append(([1.0], [1], "float32", [100.0], [1], "float32", 1.0, 0.0, False))
    cases.append(([100.0], [1], "float32", [1.0], [1], "float32", 1.0, 0.0, False))
    # equal_nan both ways
    cases.append(([nan], [1], "float32", [nan], [1], "float32", 1e-05, 1e-08, False))
    cases.append(([nan], [1], "float32", [nan], [1], "float32", 1e-05, 1e-08, True))
    cases.append(([nan], [1], "float32", [1.0], [1], "float32", 1e-05, 1e-08, True))
    # infinities
    cases.append(([inf], [1], "float32", [inf], [1], "float32", 1e-05, 1e-08, False))
    cases.append(([inf], [1], "float32", [-inf], [1], "float32", 1e-05, 1e-08, False))
    cases.append(([inf], [1], "float32", [1.0], [1], "float32", 1e-05, 1e-08, False))
    # broadcasting
    cases.append(([1.0, 1.0, 1.0], [3], "float32", [1.0], [], "float32", 1e-05, 1e-08, False))
    # empty
    cases.append(([], [0], "float32", [], [0], "float32", 1e-05, 1e-08, False))

    agree = 0
    total = 0
    disagreements = []
    for a_vals, a_shape, a_dt, b_vals, b_shape, b_dt, rtol, atol, equal_nan in cases:
        total += 1
        sa = _shim_tensor(a_vals, a_shape, a_dt)
        sb = _shim_tensor(b_vals, b_shape, b_dt)
        try:
            shim = _C._aten_dispatch(
                "aten.allclose.default", sa, sb, rtol, atol, equal_nan
            )
            shim_raised = None
        except RuntimeError as e:
            shim, shim_raised = None, str(e)

        ua = _upstream_tensor(a_vals, a_shape, a_dt)
        ub = _upstream_tensor(b_vals, b_shape, b_dt)
        try:
            up = bool(_upstream_torch.allclose(ua, ub, rtol=rtol, atol=atol, equal_nan=equal_nan))
            up_raised = None
        except RuntimeError as e:
            up, up_raised = None, str(e)

        ok = (shim_raised is None) == (up_raised is None) and (
            shim == up if shim_raised is None else True
        )
        if ok:
            agree += 1
        else:
            disagreements.append(
                (a_dt, a_shape, b_dt, b_shape, rtol, atol, equal_nan, shim, shim_raised, up, up_raised)
            )
    assert not disagreements, f"{agree}/{total} agreed; disagreements: {disagreements}"
    assert agree == total == len(cases)
    print(f"ok   torch.allclose agrees with upstream on {agree}/{total} cases")


# --------------------------------------------------------------------------
# Dispatch-table shape: dtype mismatch and shape mismatch raise the same
# RuntimeError family upstream does, checked with the real message content
# rather than only the exception type.
# --------------------------------------------------------------------------


def test_allclose_dtype_mismatch_message_matches_upstream():
    if _upstream_torch is None:
        return
    a = _shim_tensor([1.0], [1], "float32")
    b = _shim_tensor([1.0], [1], "float64")
    try:
        _C._aten_dispatch("aten.allclose.default", a, b)
        raise AssertionError("must raise")
    except RuntimeError as e:
        shim_msg = str(e)

    ua = _upstream_tensor([1.0], [1], "float32")
    ub = _upstream_tensor([1.0], [1], "float64")
    try:
        _upstream_torch.allclose(ua, ub)
        raise AssertionError("upstream must raise too")
    except RuntimeError as e:
        up_msg = str(e)
    assert shim_msg == up_msg, (shim_msg, up_msg)


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
