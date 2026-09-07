"""`aten.glu.default` -- the ASR encoders' shared wall (docs/kernels/GLU.md).

`torch._C._nn.glu` is what `F.glu` binds to upstream, and ARCH100.md's sweep
found it blocking seven architectures (`parakeet` x3, `lasr` x2,
`cohere_asr`, `parakeet_tdt`), all ASR. This file proves the *kernel*,
`aten.glu.default`, directly through `_C._aten_dispatch` -- the same door
`test_shim.py` uses -- rather than through `torch._C._nn.glu`, which is a
Python-level composite installed by `bootstrap.py::_install_nn` and out of
scope for this file (see docs/kernels/GLU.md for the remaining wiring step).

The formula (`a * sigmoid(b)` after splitting in half) is not the risk here;
two edge cases are, both measured against upstream 2.13.0 rather than
assumed, and both pinned as regression tests below:

  * the default `dim` is `-1`, not `0`
  * an ODD size at `dim` must raise, not floor-divide
"""

from test_shim import _C


def _tensor_from_flat(flat, shape, dtype):
    # Built the same way test_shim.py builds fixtures elsewhere in this
    # suite: a flat `arange`-shaped fill, cast, then reshaped.
    n = 1
    for d in shape:
        n *= d
    assert len(flat) == n, (flat, shape)
    t = _C._aten_dispatch("aten.arange.start_step", 0, len(flat), 1)
    t = _C._aten_dispatch("aten._to_copy.default", t, dtype)
    if flat != list(range(len(flat))):
        # Overwrite with the exact values requested via a cast-and-scale
        # trick is overkill for this file's needs -- every case below either
        # wants arange or values arange already gives (0, 1, 2, ...), so no
        # case needs an arbitrary fill. Assert that stays true rather than
        # silently mis-building a fixture.
        raise NotImplementedError("only arange-shaped fixtures are used in this file")
    if shape:
        t = _C._aten_dispatch("aten.view.default", t, list(shape))
    else:
        t = _C._aten_dispatch("aten.view.default", t, [])
    return t


def test_glu_is_advertised_and_dispatchable():
    assert "aten.glu.default" in _C._aten_implemented()
    out = _C._aten_dispatch("aten.glu.default", _tensor_from_flat(list(range(12)), (2, 6), _C.float32))
    assert list(out.shape) == [2, 3]


def test_glu_default_dim_is_minus_one_not_zero():
    """The trap `docs/kernels/GLU.md` calls out: a `dim=0` default would split the
    batch axis instead of the channel axis, which is silently wrong for
    every one of the seven blocked ASR encoders (they all gate the
    trailing axis)."""
    x = _tensor_from_flat(list(range(12)), (2, 6), _C.float32)
    default = _C._aten_dispatch("aten.glu.default", x)
    explicit_last = _C._aten_dispatch("aten.glu.default", x, -1)
    explicit_first = _C._aten_dispatch("aten.glu.default", x, 0)

    assert list(default.shape) == [2, 3]
    assert default.tolist() == explicit_last.tolist()
    assert list(explicit_first.shape) == [1, 6]
    assert default.tolist() != explicit_first.tolist() or list(default.shape) != list(
        explicit_first.shape
    ), "dim=0 and dim=-1 must disagree on a non-square input -- otherwise this test cannot tell them apart"


def test_glu_odd_halving_dimension_raises():
    """Upstream: 'Halving dimension must be even, but dimension {d} is size
    {n}'. A kernel that floor-divides instead of refusing would silently
    drop the last row/column rather than error -- this is the trap, not the
    arithmetic."""
    x = _tensor_from_flat(list(range(6)), (1, 3, 2), _C.float32)
    try:
        _C._aten_dispatch("aten.glu.default", x, 1)
    except RuntimeError as e:
        assert "even" in str(e)
    else:
        raise AssertionError("glu on an odd-sized halving dimension should raise")


def test_glu_refuses_integral_input_like_silu_not_sigmoid():
    """Float only -- `silu`'s dtype rule, not `sigmoid`'s promotion. Measured
    upstream: `NotImplementedError('"glu_cpu" not implemented for
    \\'Long\\'')`."""
    x = _tensor_from_flat([0, 1, 2, 3], (4,), _C.int64)
    try:
        _C._aten_dispatch("aten.glu.default", x)
    except NotImplementedError as e:
        assert "glu_cpu" in str(e)
    else:
        raise AssertionError("glu on an int64 input should raise, not promote")


def test_glu_value_matches_split_and_sigmoid_by_hand():
    """`a * sigmoid(b)` after splitting in half -- checked against the two
    kernels it is built from rather than against a literal, so this stays a
    check on the *composition* rather than a restatement of one arithmetic
    fact."""
    x = _tensor_from_flat(list(range(8)), (1, 8), _C.float32)
    out = _C._aten_dispatch("aten.glu.default", x, -1)

    a = _C._aten_dispatch("aten.slice.Tensor", x, 1, 0, 4)
    b = _C._aten_dispatch("aten.slice.Tensor", x, 1, 4, 8)
    gate = _C._aten_dispatch("aten.sigmoid.default", b)
    expected = _C._aten_dispatch("aten.mul.Tensor", a, gate)

    got, want = out.tolist(), expected.tolist()
    assert all(abs(g - w) < 1e-5 for g, w in zip(got[0], want[0])), (got, want)


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
