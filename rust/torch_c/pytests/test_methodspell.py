"""The method spelling of a function that already works.

`torch.mul(a, b)` and `a.mul(b)` are *different bindings* upstream, and this
shim keeps two tables for that reason: `overloads.json` for `torch.<name>` and
`methods.json` for `tensor.<name>`. `methods.json`'s own `_README` says it
outright -- "A method is not reachable by putting its name in
`overloads.json`; nothing looks there."

So a name can be fully implemented, agree with upstream, have a kernel and a
table entry, and still refuse through the member door. That is not a visible
failure of anything this repo tests: `test_equal_allclose.py` measures
`torch.equal` and `torch.allclose` against upstream across dtypes and edges,
and every one of its calls goes through `torch.<name>`. `docs/platform/
RELEASE_0_1_0b2.md` recorded the two as missing operators, they were added, and
the member half was never wired -- because nothing asked.

**What makes the hole invisible rather than loud.** Every upstream
`TensorBase` member that this shim does not implement is installed as a
*raising stub* (`_make_function`, driven by `surface.json`), so `hasattr(
torch.Tensor, "equal")` is `True` and `dir()` lists it. Only a call tells the
truth, and it tells it as `NotImplementedError: not implemented in torch._C
shim: TensorBase.equal`. That message is what this file keys on.

`test_no_implemented_function_is_missing_its_method_spelling` is the general
form and the one that matters: it walks `_C._shim_overloads` -- the table of
what `torch.<name>` can reach -- and for each name that upstream also carries
as a `TensorBase` member, calls the member and classifies the result. A stub
is a gap. It is derived at runtime from the two tables rather than from a list
written down here, so a *future* `overloads.json` entry that forgets its
sibling goes red without anyone remembering this file exists.

Measured on 2026-09-12 before the fix, that walk named five:

    allclose  equal  diff  fmod  multiply

and no others. A plain diff of the two JSON files reports **nine**, and is
wrong about four of them: `chunk`, `where` and `is_floating_point` are
installed by hand (`_install_tensor_chunk`, `_install_tensor_where`,
`_install_tensor_predicates`) and do reach a kernel, and `stft` is rebound in
Python by `torch/_tensor.py` in the vendored tree, so its `TensorBase` stub is
unreachable from a caller. That is why this test *calls* rather than diffing,
and why its candidate set is upstream's `torch._C.TensorBase` filtered to the
members `torch.Tensor` has not replaced.

Nullifications this file is meant to catch, each verified by making the break
and watching it go red:

* a name removed from `methods.json`      -> `test_no_implemented_function_is_missing_its_method_spelling`
* the member door answering something
  other than the function door             -> `test_the_five_members_answer_what_their_functions_answer`
* the stub detector matching nothing
  (a vacuous pass)                         -> `test_the_stub_detector_can_still_see_a_stub`
"""

import os

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _C


#: The wording `_make_function`'s stub raises with. Keyed on rather than on the
#: exception type alone, because a real method handed the wrong arguments may
#: also raise `NotImplementedError` -- from a kernel, with a different message.
_STUB_MARK = "not implemented in torch._C shim: TensorBase."

#: Names whose member door this file measures by hand, below. The general walk
#: finds them too; these are here so a regression says *which* one broke and
#: what the right answer was, rather than only that the count moved.
_FIVE = ("allclose", "equal", "diff", "fmod", "multiply")


def _t(values, shape=None, dtype=None):
    return _C._tensor_from_flat(
        values, shape if shape is not None else [len(values)], dtype or _C.float32
    )


def _is_stub(name, *args, **kwargs):
    """Did calling `TensorBase.<name>` land on the raising placeholder?

    Any other outcome -- a value, a `TypeError` from argument binding, a
    kernel's own refusal -- means the member is wired, which is all this asks.
    """
    a = _t([1.0, 2.0])
    try:
        getattr(a, name)(*args, **kwargs)
    except NotImplementedError as exc:
        return _STUB_MARK in str(exc)
    except Exception:  # noqa: BLE001
        return False
    return False


# --------------------------------------------------------------------------
# The general form
# --------------------------------------------------------------------------


def test_the_stub_detector_can_still_see_a_stub():
    """Non-vacuity. If `_is_stub` stopped recognising a stub, the walk below
    would report zero gaps for the same reason it reports zero when there are
    none, and the two are indistinguishable from the outside."""
    # `TensorBase.__deepcopy__` is in surface.json and is not implemented here;
    # any member that is a stub will do, and this one is not a candidate of the
    # walk below (it is not in `overloads.json`), so fixing it would not
    # silently retire this check.
    known_stub = None
    for name in ("__deepcopy__", "_values", "coalesce", "to_sparse"):
        if hasattr(_C.TensorBase, name) and _is_stub(name):
            known_stub = name
            break
    assert known_stub is not None, (
        "no raising stub could be found on TensorBase at all -- the detector "
        "cannot be trusted to find one in the walk below"
    )
    print(f"ok   methodspell: the stub detector still recognises one ({known_stub})")


def test_no_implemented_function_is_missing_its_method_spelling():
    try:
        import torch as _upstream
    except ImportError:  # pragma: no cover
        print("ok   methodspell: upstream torch absent -- walk skipped")
        return

    # Upstream's *C binding*, not `torch.Tensor`. The two differ, and the
    # difference is the point: `stft` is on `TensorBase` upstream and yet
    # `torch.Tensor.stft` is a different object -- `torch/_tensor.py` rebinds
    # it to a Python function. The vendored tree carries that same Python
    # layer, so a `TensorBase.stft` stub is invisible to a caller and is not a
    # gap. Derived by identity rather than by a list written down here,
    # because a list would decide the answer.
    TB = _upstream._C.TensorBase
    candidates = [
        name
        for name in _C._shim_overloads
        if not name.startswith("_")
        and callable(getattr(TB, name, None))
        and getattr(_upstream.Tensor, name, None) is getattr(TB, name)
    ]
    assert len(candidates) > 50, candidates  # the walk must not be empty
    gaps = sorted(name for name in candidates if _is_stub(name))
    assert not gaps, (
        "these have a working torch.<name> and a raising TensorBase.<name>: "
        f"{gaps}. Add the same schemas to methods.json; nothing looks in "
        "overloads.json for a member."
    )
    print(
        f"ok   methodspell: all {len(candidates)} table names that upstream "
        f"spells as a method are reachable as one"
    )


# --------------------------------------------------------------------------
# The five, by hand
# --------------------------------------------------------------------------


def test_the_five_members_are_reachable():
    for name in _FIVE:
        assert not _is_stub(name), name
    print(f"ok   methodspell: {', '.join(_FIVE)} reach a kernel through the member")


def test_the_five_members_answer_what_the_one_door_answers():
    """The member has to land on the same kernel, not merely on *a* kernel.

    Compared against `_aten_dispatch` rather than against `torch.<name>`,
    because the `torch` module's functions are built onto the vendored package
    and this suite loads `_C` standalone. `_aten_dispatch` is the door both
    spellings end at, so a member wired to the wrong overload shows up here.
    """
    a = _t([1.0, 2.0, 4.0])
    b = _t([1.0, 2.0, 4.0])
    c = _t([1.0, 2.0, 5.0])

    assert a.equal(b) is _C._aten_dispatch("aten.equal.default", a, b) is True
    assert a.equal(c) is _C._aten_dispatch("aten.equal.default", a, c) is False
    assert a.allclose(b) is True and a.allclose(c) is False
    # Tolerance reaches the member too, rather than the defaults being wired
    # and the keywords dropped.
    assert a.allclose(c, atol=2.0) is True

    assert a.diff().tolist() == _C._aten_dispatch("aten.diff.default", a).tolist()
    assert (
        a.fmod(3.0).tolist()
        == _C._aten_dispatch("aten.fmod.Scalar", a, 3.0).tolist()
    )
    assert (
        a.multiply(b).tolist()
        == _C._aten_dispatch("aten.multiply.Tensor", a, b).tolist()
    )
    print("ok   methodspell: the five members land on the same kernels the door does")


def test_the_five_members_agree_with_upstream():
    try:
        import torch as _upstream
    except ImportError:  # pragma: no cover
        print("ok   methodspell: upstream torch absent -- agreement skipped")
        return

    a = _t([1.0, 2.0, 4.0])
    c = _t([1.0, 2.0, 5.0])
    ua = _upstream.tensor([1.0, 2.0, 4.0])
    uc = _upstream.tensor([1.0, 2.0, 5.0])

    assert a.equal(c) == ua.equal(uc)
    assert a.allclose(c) == ua.allclose(uc)
    assert a.allclose(c, atol=2.0) == ua.allclose(uc, atol=2.0)
    assert a.diff().tolist() == ua.diff().tolist()
    assert a.fmod(3.0).tolist() == ua.fmod(3.0).tolist()
    assert a.multiply(c).tolist() == ua.multiply(uc).tolist()
    print("ok   methodspell: the five members agree with upstream torch")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
