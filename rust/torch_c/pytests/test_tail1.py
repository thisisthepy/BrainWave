"""docs/kernels/TAIL1.md: the eight one-architecture ops of docs/architectures/ARCH100.md's tail,
plus `linalg_qr`.

Every value here is held against **upstream torch**, not against a number
typed into this file. `_upstream_torch` is the real torch on the venv's path;
`_C` is the shim's own extension module. They are two different modules in one
interpreter, which is the arrangement `test_shim.py`'s own e2e checks use --
the golden harness goes to a subprocess because it loads the *vendored* torch,
and nothing here does except the two reachability tests at the bottom, which
have to.

What each op's cases are shaped to catch is the point, and it is different per
op (docs/kernels/TAIL1.md §2):

  * `logical_and`  -- integers, where `bitwise_and` gives a different answer
                      and a different dtype. A bool-only test passes on both.
  * `argsort`      -- duplicates, and `stable=` in both settings. A test on
                      distinct values cannot tell a stable sort from an
                      unstable one.
  * `upsample_nearest2d` -- an explicit `scales_h` that *disagrees* with
                      `output_size`. Every call `F.interpolate` makes hides
                      the difference between inverting the argument and
                      deriving the scale from the sizes.
  * `linalg_qr`    -- the identity matrix, where dropping LAPACK's
                      `xnorm == 0` short circuit answers `-I` for `+I`; and a
                      rank-deficient matrix, where nothing is determined and
                      the check has to be a property rather than a value.
"""

import json
import math
import os
import subprocess
import sys

from test_shim import _C, _upstream_torch, _CKPT_VENDOR_DIR, _CKPT_VENDOR_SHIM


def _have_upstream():
    return _upstream_torch is not None


def _pair(flat, shape, dtype_name):
    """The same numbers on both sides, without either building the other's."""
    torch = _upstream_torch
    t = torch.tensor(list(flat), dtype=getattr(torch, dtype_name)).reshape(list(shape))
    c = _C._tensor_from_flat(list(flat), list(shape), dtype=getattr(_C, dtype_name))
    return t, c


def _flat(x):
    out = []
    stack = [x.tolist() if hasattr(x, "tolist") else x]
    while stack:
        v = stack.pop(0)
        if isinstance(v, list):
            stack = list(v) + stack
        else:
            out.append(v)
    return out


def _same(t_res, c_res, label, atol=0.0, rtol=0.0):
    torch = _upstream_torch
    assert str(t_res.dtype) == str(c_res.dtype).replace("torch.", "torch."), (
        f"{label}: dtype {t_res.dtype} vs {c_res.dtype}"
    )
    assert tuple(t_res.shape) == tuple(c_res.shape), (
        f"{label}: shape {tuple(t_res.shape)} vs {tuple(c_res.shape)}"
    )
    a, b = _flat(t_res), _flat(c_res)
    for i, (x, y) in enumerate(zip(a, b)):
        if isinstance(x, bool) or isinstance(y, bool):
            assert x == y, f"{label}[{i}]: {x!r} vs {y!r}"
            continue
        x, y = float(x), float(y)
        if math.isnan(x) or math.isnan(y):
            assert math.isnan(x) and math.isnan(y), f"{label}[{i}]: {x!r} vs {y!r}"
            continue
        assert math.isclose(x, y, rel_tol=rtol, abs_tol=atol), (
            f"{label}[{i}]: upstream {x!r} vs shim {y!r}"
        )


# --- aten.broadcast_tensors.default ------------------------------------------


def test_broadcast_tensors_agrees_with_upstream_and_keeps_each_entry_dtype():
    if not _have_upstream():
        return
    torch = _upstream_torch
    for flats, note in (
        ([([0, 1, 2], (3, 1), "float32"), ([0, 1], (1, 2), "float32")], "outer product"),
        ([([5.0], (), "float32"), (list(range(6)), (2, 3), "float32")], "0-d entry"),
        ([(list(range(6)), (2, 1, 3), "float32"), (list(range(12)), (4, 3), "float32")],
         "ranks differ -- right aligned"),
        ([(list(range(6)), (2, 3), "float32")], "single entry"),
    ):
        pairs = [_pair(*f) for f in flats]
        t_out = torch.ops.aten.broadcast_tensors([p[0] for p in pairs])
        c_out = _C._aten_dispatch(
            "aten.broadcast_tensors.default", [p[1] for p in pairs]
        )
        assert len(t_out) == len(c_out), note
        for i, (t_x, c_x) in enumerate(zip(t_out, c_out)):
            _same(t_x, c_x, f"broadcast_tensors[{i}] ({note})")

    # **Dtypes are not unified.** `cat`'s promote here would be the plausible
    # wrong move, and it would be invisible on any single-dtype input.
    a_t, a_c = _pair([1, 2], (2,), "int64")
    b_t, b_c = _pair([1.0, 2.0], (2,), "float32")
    t_out = torch.ops.aten.broadcast_tensors([a_t, b_t])
    c_out = _C._aten_dispatch("aten.broadcast_tensors.default", [a_c, b_c])
    assert [str(x.dtype) for x in t_out] == ["torch.int64", "torch.float32"]
    assert [str(x.dtype) for x in c_out] == ["torch.int64", "torch.float32"], (
        [str(x.dtype) for x in c_out]
    )


def test_broadcast_tensors_empty_list_and_incompatible_shapes_match_upstream():
    if not _have_upstream():
        return
    torch = _upstream_torch
    assert list(torch.ops.aten.broadcast_tensors([])) == []
    assert list(_C._aten_dispatch("aten.broadcast_tensors.default", [])) == []

    a_t, a_c = _pair([1, 2, 3], (3,), "float32")
    b_t, b_c = _pair([1, 2], (2,), "float32")
    raised_upstream = False
    try:
        torch.ops.aten.broadcast_tensors([a_t, b_t])
    except Exception:
        raised_upstream = True
    raised_shim = False
    try:
        _C._aten_dispatch("aten.broadcast_tensors.default", [a_c, b_c])
    except Exception:
        raised_shim = True
    assert raised_upstream and raised_shim, (raised_upstream, raised_shim)


def test_broadcast_tensors_entries_are_promoted_exactly_as_much_as_unbind_int():
    """The list leaves through `PyList`, which `promote` does not look into --
    the same trap `unbind.int` and `finish_ordered` already carry a comment
    for. `promote` is only non-identity once `torch`'s bootstrap has called
    `_set_tensor_class` (`tensor.rs`'s own doc comment: with no `torch`
    package around `_C`, as here, `promote` is the identity and results stay
    `TensorBase`). So this cannot assert a fixed class name in this file --
    it asserts *parity* with `unbind.int`, which already calls `promote` on
    each element the same way (`finish_ordered`'s pair does too): whatever
    `TENSOR_CLASS` registration state this process is in, `broadcast_tensors`
    must land in the same state as `unbind.int`, not fall one promotion step
    behind it."""
    a_c = _C._tensor_from_flat([0, 1, 2], [3, 1])
    b_c = _C._tensor_from_flat([0, 1], [1, 2])
    out = _C._aten_dispatch("aten.broadcast_tensors.default", [a_c, b_c])
    unbound = _C._aten_dispatch("aten.unbind.int", a_c, 0)
    want = type(unbound[0]).__name__
    for x in out:
        assert type(x).__name__ == want, (
            f"broadcast_tensors element is {type(x).__name__!r}, "
            f"unbind.int's is {want!r} -- they must promote the same way"
        )


# --- aten.logical_and.default ------------------------------------------------


def test_logical_and_is_not_bitwise_and_on_integers():
    """The one input that separates the two. On bools they agree, so a
    bool-only test passes whichever kernel is behind the name."""
    if not _have_upstream():
        return
    torch = _upstream_torch
    for dtype_name in ("int64", "int32", "int16", "uint8"):
        # Every pair here has zero bitwise overlap (`1&2`, `2&1`, `4&3`,
        # `3&4` are all `0`) while both operands are nonzero, so
        # `bitwise_and` answers all-zero/`False` and `logical_and` answers
        # all-`True` on the same input -- the maximal divergence, not one
        # that happens to land the same way by coincidence.
        a_t, a_c = _pair([1, 2, 4, 3], (4,), dtype_name)
        b_t, b_c = _pair([2, 1, 3, 4], (4,), dtype_name)
        t_res = torch.ops.aten.logical_and(a_t, b_t)
        c_res = _C._aten_dispatch("aten.logical_and.default", a_c, b_c)
        _same(t_res, c_res, f"logical_and({dtype_name})")
        assert str(t_res.dtype) == "torch.bool"
        # And the difference, stated: `bitwise_and` on the same operands is a
        # different dtype carrying different values.
        bit = torch.ops.aten.bitwise_and.Tensor(a_t, b_t)
        assert str(bit.dtype) == f"torch.{dtype_name}", bit.dtype
        assert _flat(bit) != [bool(v) for v in _flat(t_res)], (
            "logical_and and bitwise_and agreed on this input, so it does not "
            "separate them -- pick different operands"
        )


def test_logical_and_treats_nan_as_true_and_negative_zero_as_false():
    if not _have_upstream():
        return
    torch = _upstream_torch
    a_t, a_c = _pair([float("nan"), 0.0, -0.0, float("inf"), -1.0], (5,), "float32")
    b_t, b_c = _pair([1.0, 1.0, 1.0, 1.0, 1.0], (5,), "float32")
    t_res = torch.ops.aten.logical_and(a_t, b_t)
    c_res = _C._aten_dispatch("aten.logical_and.default", a_c, b_c)
    _same(t_res, c_res, "logical_and(NaN/-0.0)")
    # Spelled out so the property survives a change to the operands: this is
    # `x != 0`, not `x > 0` and not a cast through an integer.
    assert _flat(t_res) == [True, False, False, True, True], _flat(t_res)


def test_logical_and_does_not_build_a_common_dtype():
    if not _have_upstream():
        return
    torch = _upstream_torch
    for left, right in (
        (([0, 1, 2], (3,), "int64"), ([1, 0, 1], (3,), "bool")),
        (([0.0, 1.0, 2.0], (3,), "float32"), ([0, 1, 0], (3,), "int64")),
        (([0, 1], (2,), "uint8"), ([1.5, 0.0], (2,), "float64")),
    ):
        a_t, a_c = _pair(*left)
        b_t, b_c = _pair(*right)
        _same(
            torch.ops.aten.logical_and(a_t, b_t),
            _C._aten_dispatch("aten.logical_and.default", a_c, b_c),
            f"logical_and({left[2]}, {right[2]})",
        )


def test_logical_and_broadcasts():
    if not _have_upstream():
        return
    torch = _upstream_torch
    a_t, a_c = _pair([1, 0, 1], (3, 1), "bool")
    b_t, b_c = _pair([1, 0], (1, 2), "bool")
    t_res = torch.ops.aten.logical_and(a_t, b_t)
    c_res = _C._aten_dispatch("aten.logical_and.default", a_c, b_c)
    assert tuple(t_res.shape) == (3, 2)
    _same(t_res, c_res, "logical_and broadcast")


# --- aten.acos.default -------------------------------------------------------


_ACOS_DOMAIN = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0, float("nan"), float("inf")]
_ACOS_TOL = {"float64": 1e-12, "float32": 1e-6, "float16": 5e-3, "bfloat16": 6e-2}


def test_acos_matches_upstream_including_outside_its_domain():
    if not _have_upstream():
        return
    torch = _upstream_torch
    for dtype_name, tol in _ACOS_TOL.items():
        a_t, a_c = _pair(_ACOS_DOMAIN, (9,), dtype_name)
        t_res = torch.ops.aten.acos(a_t)
        c_res = _C._aten_dispatch("aten.acos.default", a_c)
        _same(t_res, c_res, f"acos({dtype_name})", atol=tol, rtol=tol)
    # NaN outside [-1, 1] rather than an exception -- the plausible wrong
    # guess is a domain refusal, and `_same` above would pass either way if
    # both sides raised, so the shape of the answer is asserted here.
    a_t, a_c = _pair([1.5, -2.0, float("inf")], (3,), "float32")
    assert all(math.isnan(v) for v in _flat(_C._aten_dispatch("aten.acos.default", a_c)))
    assert all(math.isnan(v) for v in _flat(torch.ops.aten.acos(a_t)))


def test_acos_promotes_every_non_floating_dtype_to_the_default_float():
    if not _have_upstream():
        return
    torch = _upstream_torch
    for dtype_name in ("int64", "int32", "int16", "uint8", "bool"):
        a_t, a_c = _pair([0, 1], (2,), dtype_name)
        t_res = torch.ops.aten.acos(a_t)
        c_res = _C._aten_dispatch("aten.acos.default", a_c)
        assert str(t_res.dtype) == "torch.float32", (dtype_name, t_res.dtype)
        _same(t_res, c_res, f"acos({dtype_name})", atol=1e-6, rtol=1e-6)


# --- aten._is_all_true.default -----------------------------------------------


def test_is_all_true_matches_upstream_and_answers_a_zero_d_bool_tensor():
    if not _have_upstream():
        return
    torch = _upstream_torch
    for flat, shape in (([1, 1], (2,)), ([1, 0], (2,)), ([0, 0], (2,)),
                        ([], (0,)), ([1], ()), ([0], ()), ([1, 1, 1, 0], (2, 2))):
        a_t, a_c = _pair(flat, shape, "bool")
        t_res = torch.ops.aten._is_all_true(a_t)
        c_res = _C._aten_dispatch("aten._is_all_true.default", a_c)
        assert tuple(t_res.shape) == (), tuple(t_res.shape)
        _same(t_res, c_res, f"_is_all_true({flat})")
    # The empty conjunction is True, which `any`'s identity is not -- a kernel
    # that shared one early return between the two would get this wrong and
    # only here.
    e_t, e_c = _pair([], (0,), "bool")
    assert bool(torch.ops.aten._is_all_true(e_t)) is True
    assert bool(_C._aten_dispatch("aten._is_all_true.default", e_c)) is True


def test_is_all_true_refuses_every_dtype_but_bool_as_upstream_does():
    if not _have_upstream():
        return
    torch = _upstream_torch
    for dtype_name in ("int64", "float32", "uint8", "float64"):
        a_t, a_c = _pair([1, 1], (2,), dtype_name)
        up = shim = None
        try:
            torch.ops.aten._is_all_true(a_t)
        except Exception as e:
            up = type(e).__name__
        try:
            _C._aten_dispatch("aten._is_all_true.default", a_c)
        except Exception as e:
            shim = type(e).__name__
        assert up is not None, f"upstream computed _is_all_true({dtype_name})"
        assert shim is not None, f"the shim computed _is_all_true({dtype_name})"


# --- aten.argsort.default / aten.argsort.stable ------------------------------


_TIED = [3, 1, 3, 1, 2, 3]


def test_argsort_ties_match_upstream_in_both_directions():
    """Distinct values cannot separate a stable sort from an unstable one.
    These six can: the two 1s and the three 3s have to come back in their
    original relative order, ascending *and* descending."""
    if not _have_upstream():
        return
    torch = _upstream_torch
    for dtype_name in ("float32", "float64", "int64", "int32"):
        for descending in (False, True):
            a_t, a_c = _pair(_TIED, (6,), dtype_name)
            t_res = torch.ops.aten.argsort(a_t, -1, descending)
            c_res = _C._aten_dispatch("aten.argsort.default", a_c, -1, descending)
            _same(t_res, c_res, f"argsort({dtype_name}, desc={descending})")
    # Named, so the property survives someone changing `_TIED`.
    a_t, _ = _pair(_TIED, (6,), "float32")
    assert _flat(torch.ops.aten.argsort(a_t, -1, False)) == [1, 3, 4, 0, 2, 5]
    assert _flat(torch.ops.aten.argsort(a_t, -1, True)) == [0, 2, 5, 4, 1, 3]


def test_argsort_stable_overload_agrees_with_the_default_one_in_both_settings():
    """`stable=` is the argument that cannot be tested by taking it on faith.
    Upstream answers the same permutation with `stable=True`, `stable=False`
    and no `stable` at all on CPU (measured); this asserts that, and asserts
    that this shim answers the same thing, so a future upstream that stopped
    agreeing would show up here rather than in a model."""
    if not _have_upstream():
        return
    torch = _upstream_torch
    for stable in (True, False):
        for descending in (False, True):
            a_t, a_c = _pair(_TIED, (6,), "float32")
            t_res = torch.ops.aten.argsort.stable(
                a_t, stable=stable, dim=-1, descending=descending
            )
            c_res = _C._aten_dispatch(
                "aten.argsort.stable", a_c, stable=stable, dim=-1, descending=descending
            )
            _same(t_res, c_res, f"argsort.stable(stable={stable}, desc={descending})")
            plain = torch.ops.aten.argsort(a_t, -1, descending)
            assert _flat(t_res) == _flat(plain), (
                f"upstream's argsort.stable(stable={stable}) no longer agrees with "
                f"argsort() -- this shim shares one implementation between them and "
                f"docs/kernels/TAIL1.md §3 records why that was licensed"
            )


def test_eighty_identical_elements_come_back_in_index_order():
    if not _have_upstream():
        return
    torch = _upstream_torch
    a_t, a_c = _pair([1.0] * 80, (80,), "float32")
    assert _flat(torch.ops.aten.argsort(a_t, -1, False)) == list(range(80))
    _same(
        torch.ops.aten.argsort(a_t, -1, False),
        _C._aten_dispatch("aten.argsort.default", a_c, -1, False),
        "argsort(80 ties)",
    )


def test_argsort_dims_nan_and_zero_d_match_upstream():
    if not _have_upstream():
        return
    torch = _upstream_torch
    for dim in (0, 1, -1):
        a_t, a_c = _pair([3, 1, 2, 0, 5, 4], (2, 3), "int64")
        _same(
            torch.ops.aten.argsort(a_t, dim, False),
            _C._aten_dispatch("aten.argsort.default", a_c, dim, False),
            f"argsort(dim={dim})",
        )
    nan_flat = [1.0, float("nan"), 0.0, float("inf"), float("-inf")]
    for descending in (False, True):
        a_t, a_c = _pair(nan_flat, (5,), "float32")
        _same(
            torch.ops.aten.argsort(a_t, -1, descending),
            _C._aten_dispatch("aten.argsort.default", a_c, -1, descending),
            f"argsort(NaN, desc={descending})",
        )
    a_t, a_c = _pair([5.0], (), "float32")
    _same(
        torch.ops.aten.argsort(a_t, -1, False),
        _C._aten_dispatch("aten.argsort.default", a_c, -1, False),
        "argsort(0-d)",
    )


# --- aten.max_pool1d.default -------------------------------------------------


def test_max_pool1d_matches_upstream_across_kernel_stride_padding_dilation_ceil():
    if not _have_upstream():
        return
    torch = _upstream_torch
    twenty = [float(v) for v in range(20)]
    tol = {"float32": 0.0, "float64": 0.0, "float16": 0.0, "bfloat16": 0.0}
    for dtype_name in tol:
        for args in ((2,), (3, 2), (3, 2, 1), (2, 1, 0, 2), (3, 3, 0, 1, True),
                     (3, 2, 0, 1, True)):
            a_t, a_c = _pair(twenty, (1, 2, 10), dtype_name)
            _same(
                torch.ops.aten.max_pool1d(a_t, *args),
                _C._aten_dispatch("aten.max_pool1d.default", a_c, *args),
                f"max_pool1d({dtype_name}, {args})",
            )


def test_max_pool1d_accepts_float16_and_bfloat16_which_max_pool2d_here_refuses():
    """The dtype rule is `max_pool1d`'s own, not the neighbour's. This shim's
    `max_pool2d` refuses `float16`; upstream's `max_pool1d` computes it, and
    inheriting the 2-D branch would have refused two dtypes upstream answers."""
    if not _have_upstream():
        return
    torch = _upstream_torch
    for dtype_name in ("float16", "bfloat16"):
        a_t, a_c = _pair([float(v) for v in range(10)], (1, 1, 10), dtype_name)
        t_res = torch.ops.aten.max_pool1d(a_t, 2)
        c_res = _C._aten_dispatch("aten.max_pool1d.default", a_c, 2)
        assert str(t_res.dtype) == f"torch.{dtype_name}"
        _same(t_res, c_res, f"max_pool1d({dtype_name})")
    # And `max_pool2d` in this shim really does refuse it, which is what makes
    # the sentence above a measurement rather than a claim about upstream.
    b_c = _C._tensor_from_flat([float(v) for v in range(16)], [1, 1, 4, 4],
                               dtype=_C.float16)
    refused = False
    try:
        _C._aten_dispatch("aten.max_pool2d.default", b_c, 2)
    except Exception:
        refused = True
    assert refused, (
        "max_pool2d now accepts float16 here -- if that is deliberate, the "
        "asymmetry docs/kernels/TAIL1.md §2 records is gone and this test should say so"
    )


def test_max_pool1d_refuses_the_integral_dtypes_upstream_refuses():
    if not _have_upstream():
        return
    torch = _upstream_torch
    for dtype_name in ("int64", "int32", "uint8", "bool"):
        a_t, a_c = _pair([1, 0, 1, 1], (1, 1, 4), dtype_name)
        up = shim = None
        try:
            torch.ops.aten.max_pool1d(a_t, 2)
        except Exception as e:
            up = str(e)
        try:
            _C._aten_dispatch("aten.max_pool1d.default", a_c, 2)
        except Exception as e:
            shim = str(e)
        assert up is not None and shim is not None, (dtype_name, up, shim)
        # Upstream names `max_pool1d_impl`, not `max_pool2d`. Copying the
        # neighbour's message would misname the kernel for these four dtypes.
        assert "max_pool1d_impl" in up, up
        assert "max_pool1d_impl" in shim, shim


def test_max_pool1d_takes_both_ranks_and_a_nan_wins_the_window():
    if not _have_upstream():
        return
    torch = _upstream_torch
    a_t, a_c = _pair([float(v) for v in range(10)], (2, 5), "float32")
    _same(
        torch.ops.aten.max_pool1d(a_t, 2),
        _C._aten_dispatch("aten.max_pool1d.default", a_c, 2),
        "max_pool1d unbatched (C, L)",
    )
    nan_flat = [1.0, float("nan"), 0.0, 3.0, float("-inf"), 2.0]
    b_t, b_c = _pair(nan_flat, (1, 1, 6), "float32")
    _same(
        torch.ops.aten.max_pool1d(b_t, 2),
        _C._aten_dispatch("aten.max_pool1d.default", b_c, 2),
        "max_pool1d NaN window",
    )


# --- aten.upsample_nearest2d.default -----------------------------------------


def test_upsample_nearest2d_index_rule_matches_upstream():
    if not _have_upstream():
        return
    torch = _upstream_torch
    sixteen = [float(v) for v in range(16)]
    for dtype_name in ("float32", "float64", "float16", "bfloat16"):
        for out_size in ([2, 2], [3, 3], [8, 8], [5, 7], [4, 4], [1, 1], [6, 3]):
            a_t, a_c = _pair(sixteen, (1, 1, 4, 4), dtype_name)
            _same(
                torch.ops.aten.upsample_nearest2d(a_t, out_size),
                _C._aten_dispatch("aten.upsample_nearest2d.default", a_c, out_size),
                f"upsample_nearest2d({dtype_name}, {out_size})",
            )


def test_upsample_nearest2d_inverts_an_explicit_scale_rather_than_deriving_one():
    """THE case for this op. With `output_size=[3,3]` alone the scale is 4/3
    and rows [0,1,2] are gathered; with `scales_h=1.5` the scale is 1/1.5 and
    rows [0,0,1] are. A kernel that derived the scale from the sizes and
    ignored the argument agrees with upstream on every call `F.interpolate`
    makes and differs here."""
    if not _have_upstream():
        return
    torch = _upstream_torch
    sixteen = [float(v) for v in range(16)]
    for scales, out_size in (((2.0, 2.0), [8, 8]), ((1.5, 1.5), [3, 3]),
                             ((0.5, 0.5), [2, 2]), ((1.5, 3.0), [3, 6])):
        a_t, a_c = _pair(sixteen, (1, 1, 4, 4), "float32")
        _same(
            torch.ops.aten.upsample_nearest2d(a_t, out_size, scales[0], scales[1]),
            _C._aten_dispatch(
                "aten.upsample_nearest2d.default", a_c, out_size, scales[0], scales[1]
            ),
            f"upsample_nearest2d(scales={scales}, out={out_size})",
        )
    # Named, so the difference is visible without running upstream.
    a_t, a_c = _pair(sixteen, (1, 1, 4, 4), "float32")
    with_scale = _flat(_C._aten_dispatch(
        "aten.upsample_nearest2d.default", a_c, [3, 3], 1.5, 1.5))
    from_sizes = _flat(_C._aten_dispatch(
        "aten.upsample_nearest2d.default", a_c, [3, 3]))
    assert with_scale == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 4.0, 4.0, 5.0], with_scale
    assert from_sizes == [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0], from_sizes
    assert with_scale != from_sizes


def test_upsample_nearest2d_computes_uint8_because_it_never_averages():
    """Its interpolating neighbours have a separate fixed-point `uint8` kernel
    upstream; nearest neighbour has no rounding to do differently, and
    upstream computes `uint8` here. Refusing it -- which this file first did,
    by inheriting the neighbours' reasoning -- refuses a dtype upstream
    answers."""
    if not _have_upstream():
        return
    torch = _upstream_torch
    for out_size in ([3, 3], [8, 8], [5, 7]):
        a_t, a_c = _pair(list(range(16)), (1, 1, 4, 4), "uint8")
        t_res = torch.ops.aten.upsample_nearest2d(a_t, out_size)
        c_res = _C._aten_dispatch("aten.upsample_nearest2d.default", a_c, out_size)
        assert str(t_res.dtype) == "torch.uint8"
        _same(t_res, c_res, f"upsample_nearest2d(uint8, {out_size})")


def test_upsample_nearest2d_refuses_what_upstream_refuses():
    if not _have_upstream():
        return
    torch = _upstream_torch
    for dtype_name in ("int64", "int32", "bool"):
        a_t, a_c = _pair(list(range(16)), (1, 1, 4, 4), dtype_name)
        up = shim = None
        try:
            torch.ops.aten.upsample_nearest2d(a_t, [3, 3])
        except Exception as e:
            up = str(e)
        try:
            _C._aten_dispatch("aten.upsample_nearest2d.default", a_c, [3, 3])
        except Exception as e:
            shim = str(e)
        assert up is not None and shim is not None, (dtype_name, up, shim)
    # Rank, too: upstream wants exactly 4 dimensions.
    b_t, b_c = _pair([1.0, 2.0, 3.0, 4.0], (2, 2), "float32")
    for run in (lambda: torch.ops.aten.upsample_nearest2d(b_t, [3, 3]),
                lambda: _C._aten_dispatch("aten.upsample_nearest2d.default", b_c, [3, 3])):
        raised = False
        try:
            run()
        except Exception:
            raised = True
        assert raised


def test_upsample_nearest2d_covers_several_planes():
    if not _have_upstream():
        return
    torch = _upstream_torch
    a_t, a_c = _pair([float(v) for v in range(32)], (2, 2, 2, 4), "float32")
    _same(
        torch.ops.aten.upsample_nearest2d(a_t, [3, 5]),
        _C._aten_dispatch("aten.upsample_nearest2d.default", a_c, [3, 5]),
        "upsample_nearest2d (2,2,2,4) -> [3,5]",
    )


# --- aten.linalg_qr.default --------------------------------------------------


_QR_MATRICES = [
    ("3x3", [12, -51, 4, 6, 167, -68, -4, 24, -41], (3, 3)),
    ("4x2 tall", [1, 2, 3, 4, 5, 6, 7, 8], (4, 2)),
    ("2x4 wide", [1, 2, 3, 4, 5, 6, 7, 9], (2, 4)),
    ("1x1", [2], (1, 1)),
    ("1x1 negative", [-2], (1, 1)),
    ("3x3 identity", [1, 0, 0, 0, 1, 0, 0, 0, 1], (3, 3)),
    ("2x2 negative diagonal", [-1, 0, 0, -1], (2, 2)),
    ("3x2 already triangular", [1, 2, 0, 3, 0, 0], (3, 2)),
]


def test_linalg_qr_agrees_with_lapack_element_wise_including_the_signs():
    """A QR is unique only up to the signs of R's diagonal, so this is a
    *convention* check as much as a numeric one. `float64` is held to 1e-12,
    which no merely-compatible factorisation would meet."""
    if not _have_upstream():
        return
    torch = _upstream_torch
    for dtype_name, tol in (("float64", 1e-12), ("float32", 1e-5)):
        for label, flat, shape in _QR_MATRICES:
            for mode in ("reduced", "complete", "r"):
                a_t, a_c = _pair(flat, shape, dtype_name)
                t_q, t_r = torch.ops.aten.linalg_qr(a_t, mode)
                c_q, c_r = _C._aten_dispatch("aten.linalg_qr.default", a_c, mode)
                _same(t_q, c_q, f"Q {label} {mode} {dtype_name}", atol=tol, rtol=tol)
                _same(t_r, c_r, f"R {label} {mode} {dtype_name}", atol=tol, rtol=tol)


def test_linalg_qr_answers_plus_i_for_the_identity_not_minus_i():
    """The single most likely wrong answer here, and it looks reasonable.
    LAPACK's `dlarfg` returns `tau = 0` and leaves `alpha` alone when the
    sub-column is already zero; the plausible `beta = -sign(alpha)*norm`
    without that short circuit answers `R = -I`. Both are orthogonal
    factorisations of the identity; only one is upstream's."""
    if not _have_upstream():
        return
    torch = _upstream_torch
    a_t, a_c = _pair([1, 0, 0, 0, 1, 0, 0, 0, 1], (3, 3), "float64")
    t_q, t_r = torch.ops.aten.linalg_qr(a_t, "reduced")
    c_q, c_r = _C._aten_dispatch("aten.linalg_qr.default", a_c, "reduced")
    assert [t_r[i][i].item() for i in range(3)] == [1.0, 1.0, 1.0]
    assert [c_r[i][i].item() for i in range(3)] == [1.0, 1.0, 1.0], (
        "R's diagonal is not +1 for the identity -- dlarfg's xnorm == 0 short "
        "circuit is missing (docs/kernels/TAIL1.md §5)"
    )
    _same(t_q, c_q, "Q(eye)", atol=0.0, rtol=0.0)


def test_linalg_qr_r_mode_answers_a_one_dimensional_empty_q():
    if not _have_upstream():
        return
    torch = _upstream_torch
    for flat, shape in (([12, -51, 4, 6, 167, -68, -4, 24, -41], (3, 3)),
                        ([1, 2, 3, 4, 5, 6, 7, 8], (4, 2))):
        a_t, a_c = _pair(flat, shape, "float64")
        t_q, _ = torch.ops.aten.linalg_qr(a_t, "r")
        c_q, _ = _C._aten_dispatch("aten.linalg_qr.default", a_c, "r")
        assert tuple(t_q.shape) == (0,), tuple(t_q.shape)
        assert tuple(c_q.shape) == (0,), tuple(c_q.shape)


def test_linalg_qr_batches_over_the_leading_dimensions():
    if not _have_upstream():
        return
    torch = _upstream_torch
    batch = [1, 2, 3, 4, 5, 6, 7, 8, 2, 1, 4, 3, 6, 5, 8, 7]
    for shape in ((2, 4, 2), (2, 2, 2, 2)):
        a_t, a_c = _pair(batch, shape, "float64")
        t_q, t_r = torch.ops.aten.linalg_qr(a_t, "reduced")
        c_q, c_r = _C._aten_dispatch("aten.linalg_qr.default", a_c, "reduced")
        _same(t_q, c_q, f"Q batched {shape}", atol=1e-12, rtol=1e-12)
        _same(t_r, c_r, f"R batched {shape}", atol=1e-12, rtol=1e-12)


def test_linalg_qr_on_a_rank_deficient_matrix_is_checked_by_property_not_value():
    """docs/kernels/TAIL1.md §5. On `[[1,2],[2,4],[3,6]]` the second column of Q is
    determined by rounding noise -- upstream's own R[1][1] is 8.8e-07 at
    float32 -- so an element-wise comparison there compares two arbitrary
    answers. What *is* determined is checked instead: R upper-triangular, Q's
    columns orthonormal, and Q @ R == A.

    This is the case the round could most easily have skipped by choosing a
    well-conditioned matrix and calling QR done."""
    if not _have_upstream():
        return
    flat, m, n = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 3, 2
    a_c = _C._tensor_from_flat(flat, [m, n], dtype=_C.float64)
    q, r = _C._aten_dispatch("aten.linalg_qr.default", a_c, "reduced")
    q_l, r_l = q.tolist(), r.tolist()
    for i in range(len(r_l)):
        for j in range(i):
            assert r_l[i][j] == 0.0, f"R is not upper triangular at ({i},{j})"
    k = min(m, n)
    for c1 in range(k):
        for c2 in range(k):
            dot = sum(q_l[i][c1] * q_l[i][c2] for i in range(m))
            want = 1.0 if c1 == c2 else 0.0
            assert abs(dot - want) < 1e-12, f"Q columns not orthonormal: {c1},{c2} -> {dot}"
    for i in range(m):
        for j in range(n):
            got = sum(q_l[i][c] * r_l[c][j] for c in range(k))
            assert abs(got - flat[i * n + j]) < 1e-12, f"Q@R != A at ({i},{j}): {got}"


def test_linalg_qr_refusals_are_two_different_ones_because_upstream_has_two():
    if not _have_upstream():
        return
    torch = _upstream_torch
    for dtype_name, needle in (("float16", "geqrf_cpu"), ("bfloat16", "geqrf_cpu"),
                               ("int64", "floating point"), ("int32", "floating point")):
        a_t, a_c = _pair([1, 2, 3, 4], (2, 2), dtype_name)
        up = shim = None
        try:
            torch.ops.aten.linalg_qr(a_t, "reduced")
        except Exception as e:
            up = str(e)
        try:
            _C._aten_dispatch("aten.linalg_qr.default", a_c, "reduced")
        except Exception as e:
            shim = str(e)
        assert up is not None and shim is not None, (dtype_name, up, shim)
        assert needle in up, (dtype_name, up)
        assert needle in shim, (dtype_name, shim)
    # A dtype with no LAPACK kernel and a dtype rejected before LAPACK are not
    # the same refusal, which is why they cannot share a branch.
    a_t, a_c = _pair([1, 2, 3, 4], (2, 2), "float64")
    for run in (lambda: torch.ops.aten.linalg_qr(a_t, "bogus"),
                lambda: _C._aten_dispatch("aten.linalg_qr.default", a_c, "bogus")):
        raised = False
        try:
            run()
        except Exception:
            raised = True
        assert raised
    b_t, b_c = _pair([1.0, 2.0, 3.0], (3,), "float64")
    for run in (lambda: torch.ops.aten.linalg_qr(b_t, "reduced"),
                lambda: _C._aten_dispatch("aten.linalg_qr.default", b_c, "reduced")):
        raised = False
        try:
            run()
        except Exception:
            raised = True
        assert raised


# --- what is reachable from Python, and what is not --------------------------
#
# Two of the nine ops have a kernel here and **no Python spelling**, because
# upstream reaches them through `torch._C._nn` / `torch._C._linalg` submodule
# bindings that live in `bootstrap.py`. This round did not own that file, so
# the state below is pinned rather than fixed: each assertion fails when the
# binding lands, which is when docs/kernels/TAIL1.md §4 should be rewritten.

_VENDOR_PROBE = r"""
import json, sys
import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}


def probe(name, fn):
    try:
        out[name] = ["ok", fn()]
    except Exception as e:
        out[name] = ["raised", type(e).__name__, str(e)[:200]]


probe("broadcast_tensors", lambda: [
    list(t.shape) for t in torch.broadcast_tensors(
        torch.zeros(3, 1), torch.zeros(1, 2))])
probe("acos", lambda: [round(v, 6) for v in torch.acos(
    torch.tensor([0.0, 1.0])).tolist()])
probe("acos_method", lambda: round(torch.tensor([0.5]).acos().item(), 6))
probe("logical_and", lambda: torch.logical_and(
    torch.tensor([0, 1, 2]), torch.tensor([0, 3, 0])).tolist())
probe("logical_and_method", lambda: torch.tensor([1, 0]).logical_and(
    torch.tensor([1, 1])).tolist())
probe("argsort", lambda: torch.argsort(torch.tensor([3.0, 1.0, 2.0])).tolist())
probe("argsort_method", lambda: torch.tensor([3.0, 1.0, 2.0]).argsort().tolist())
probe("argsort_stable", lambda: torch.argsort(
    torch.tensor([3.0, 1.0, 3.0]), stable=True).tolist())
probe("max_pool1d", lambda: torch.max_pool1d(
    torch.arange(10, dtype=torch.float32).reshape(1, 1, 10), 2).tolist())
probe("is_all_true", lambda: bool(torch._is_all_true(torch.tensor([True, True]))))
probe("nn_upsample_nearest2d", lambda: torch._C._nn.upsample_nearest2d(
    torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4), [2, 2]).tolist())
probe("linalg_qr", lambda: [list(t.shape) for t in torch.linalg.qr(
    torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))])
probe("mse_loss", lambda: round(float(torch.nn.MSELoss()(
    torch.zeros(2, 2), torch.ones(2, 2))), 6))

json.dump(out, sys.stdout)
"""


def _vendor_probe():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _VENDOR_PROBE],
        capture_output=True, text=True, env=env, timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"vendored probe exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_the_new_torch_level_spellings_reach_their_kernels_in_the_vendored_tree():
    """A kernel with no spelling is invisible from Python -- the shape
    `docs/bindings/SPELLINGS.md` exists for. This runs the *vendored* torch (the shim
    wearing torch's name) in a subprocess and calls each op the way a model
    would."""
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _vendor_probe()
    assert r["is_shim"] is True, "the probe imported upstream torch, not the shim"
    for name, expected in (
        ("broadcast_tensors", [[3, 2], [3, 2]]),
        ("acos", [1.570796, 0.0]),
        ("acos_method", 1.047198),
        ("logical_and", [False, True, False]),
        ("logical_and_method", [True, False]),
        ("argsort", [1, 2, 0]),
        ("argsort_method", [1, 2, 0]),
        ("argsort_stable", [1, 0, 2]),
        ("max_pool1d", [[[1.0, 3.0, 5.0, 7.0, 9.0]]]),
        ("is_all_true", True),
    ):
        assert r[name][0] == "ok", f"torch.{name} does not reach its kernel: {r[name]}"
        assert r[name][1] == expected, (name, r[name][1], expected)


def test_the_two_submodule_bindings_landed_and_mse_loss_is_the_one_still_open():
    """This test used to pin the *opposite* -- that `torch._C._nn.
    upsample_nearest2d` and `torch._C._linalg.linalg_qr` had kernels and no way
    to call them, because the round that wrote the kernels did not own
    `bootstrap.py`. docs/kernels/TAIL1.md §4 said it would fail the moment the bindings
    landed, and it did. They are inverted here rather than deleted: the
    coverage that mattered was never "the gap exists", it was "somebody checks
    these two names through the vendored tree", and that is worth keeping now
    that they answer. docs/bindings/BINDINGS.md.

    `mse_loss` is the one still open, and it is a `bootstrap.py` stub of the
    same shape -- docs/training/BACKWARD9.md §1's hand-spelled criterion stands until it
    lands."""
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _vendor_probe()
    assert r["nn_upsample_nearest2d"][0] == "ok", r["nn_upsample_nearest2d"]
    assert r["nn_upsample_nearest2d"][1] == [[[[0.0, 2.0], [8.0, 10.0]]]], (
        r["nn_upsample_nearest2d"][1]
    )
    assert r["linalg_qr"][0] == "ok", r["linalg_qr"]
    assert r["linalg_qr"][1] == [[3, 3], [3, 3]], r["linalg_qr"][1]
    # And the loss that motivated `broadcast_tensors`: the table row landed and
    # the *next* wall is `torch._C._nn.mse_loss`, also a bootstrap.py stub.
    assert r["mse_loss"][0] == "raised", (
        "nn.MSELoss now computes -- the _nn.mse_loss binding landed. "
        "docs/training/BACKWARD9.md §1 can stop spelling its criterion out by hand, and "
        "docs/kernels/TAIL1.md §4 should say so"
    )
    assert "mse_loss" in r["mse_loss"][2], r["mse_loss"]


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
