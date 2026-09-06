"""`TensorBase.__setitem__` -- the stepped-slice write, and the one form the
297-architecture sweep is actually blocked on.

docs/ARCH100.md §2 ranks `TensorBase.__setitem__` first: 13 of the 82 blocked
architectures stop there, more than any other single name. Running those 13
(docs/SETITEM.md §1) shows they do not spread across `__setitem__`'s many
overloaded index forms at all. **All 13 stop on one form** -- a slice with
`step != 1` on the left of an assignment -- in exactly two shapes:

    pe[:, 0::2]        = <matrix>   3 arch, all at CONSTRUCTION
    freqs_t[..., k::3] = <rank-3>   10 arch, in the forward

So this file is not a survey of `__setitem__`. It is about the stepped write,
and it is written so that it stays a check on **both** sides of the change
docs/SETITEM.md §2 describes: while the form still refuses, it must refuse by
name and must never silently no-op; once the lowering lands, it must produce
upstream's values element for element.

That two-sided shape is deliberate. The refusal exists because
`aten.slice.Tensor` materialises above step 1 (docs/VIEWS.md §6.4), so the
natural walk narrows to a tensor that does *not* share storage with the
receiver -- and a write into that tensor is lost with no error at all. A test
that only pinned "it raises" would have to be deleted when the lowering lands,
and a test that only pinned the values would be red until then. Neither is a
check across the transition; this is.

Every number below was measured against upstream torch 2.13.0 in a separate
process (`env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`), not derived.
"""

import pathlib

from test_shim import _C


REFUSAL_NAME = "TensorBase.__setitem__"


def _from(flat, shape, dtype_name="float32"):
    """`flat` reshaped to `shape`, through the shim's own scaffolding.

    `_C._tensor_from_flat` is the documented way to get real data into a
    `TensorBase` (tools/golden/build.py uses it for the same reason): there is
    no aten op that takes a Python list of numbers. It matters here that the
    builder does not itself go through `__setitem__`, which is the member
    under test.
    """
    return _C._tensor_from_flat(list(flat), list(shape), dtype=getattr(_C, dtype_name))


def _stepped_write(receiver, index, value):
    """`receiver[index] = value`, reporting which of the two régimes happened.

    Returns `("wrote", values)` or `("refused", message)`. Anything else --
    including a write that raises nothing and changes nothing -- is the silent
    loss this whole area exists to prevent, and the callers below reject it.
    """
    before = receiver.tolist()
    try:
        receiver[index] = value
    except NotImplementedError as e:
        after = receiver.tolist()
        assert after == before, (
            "__setitem__ refused and yet the receiver changed -- a partial "
            f"write happened before the refusal: {before} -> {after}"
        )
        return "refused", str(e)
    return "wrote", receiver.tolist()


def _expect(receiver, index, value, upstream):
    """Either upstream's exact answer, or a refusal that names itself.

    The silent no-op is rejected on the `wrote` branch: if the shim ever
    routes a stepped write into a materialised copy, it returns normally and
    leaves the receiver untouched, which lands here as `wrote` with the
    unchanged contents and fails against `upstream`.
    """
    kind, payload = _stepped_write(receiver, index, value)
    if kind == "wrote":
        assert payload == upstream, (
            f"stepped write produced {payload}, upstream gives {upstream}"
        )
        return "wrote"
    assert REFUSAL_NAME in payload, (
        "a stepped write must refuse BY NAME, so the architecture sweep can "
        f"classify it; got: {payload}"
    )
    assert "step" in payload, (
        "the refusal must say which form was asked for -- the step is the "
        f"whole reason it refuses; got: {payload}"
    )
    return "refused"


# --- the two shapes the 13 architectures use ---------------------------------


def test_the_conformer_positional_encoding_write():
    """`pe_positive[:, 0::2] = torch.sin(...)`.

    fastspeech2_conformer, seamless_m4t and wav2vec2-conformer all execute
    this line inside `__init__`, so it is a CONSTRUCTION wall: nothing at all
    about those three can be measured until it falls. The stepped slice is at
    axis 1, not axis 0 -- a lowering that handled only `x[0::2]` would leave
    all three exactly where they are.

    Upstream, measured: zeros(3,4) with [[1,2],[3,4],[5,6]] into `[:, 0::2]`
    gives [[1,0,2,0],[3,0,4,0],[5,0,6,0]].
    """
    x = _from([0.0] * 12, (3, 4))
    src = _from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
    _expect(x, (slice(None), slice(0, None, 2)), src,
            [[1.0, 0.0, 2.0, 0.0], [3.0, 0.0, 4.0, 0.0], [5.0, 0.0, 6.0, 0.0]])


def test_the_interleaved_mrope_write():
    """`freqs_t[..., idx] = freqs[dim, ..., idx]` with `idx = slice(k, None, 3)`.

    The qwen3_5 / qwen3_5_moe / qwen3_vl / qwen3_vl_moe family plus
    cosmos3_omni and minicpmv4_6 -- ten of the thirteen -- stop on this line
    in `apply_interleaved_mrope`. The ellipsis expands to full slices, so the
    stepped slice lands at the LAST axis of a rank-3 receiver, and the step is
    3 rather than 2.

    Upstream, measured: zeros(2,1,6) with [[[1,2]],[[3,4]]] into `[..., 1::3]`
    gives [[[0,1,0,0,2,0]],[[0,3,0,0,4,0]]].
    """
    x = _from([0.0] * 12, (2, 1, 6))
    src = _from([1.0, 2.0, 3.0, 4.0], (2, 1, 2))
    _expect(x, (Ellipsis, slice(1, None, 3)), src,
            [[[0.0, 1.0, 0.0, 0.0, 2.0, 0.0]], [[0.0, 3.0, 0.0, 0.0, 4.0, 0.0]]])


def test_a_stepped_write_never_silently_loses_the_write():
    """The failure mode this area is shaped around, asked directly.

    docs/VIEWS.md §6.4: above step 1, `slice.Tensor` goes through
    `index_select` and materialises, so the obvious walk writes into a buffer
    nobody is holding and returns normally. That is worse than a refusal and
    worse than a wrong number, because nothing anywhere reports it.

    `_expect` rejects it on both branches -- an unchanged receiver fails the
    value comparison, and a refusal that mutated anything fails inside
    `_stepped_write`.
    """
    x = _from([0.0] * 6, (6,))
    src = _from([7.0, 8.0, 9.0], (3,))
    _expect(x, slice(None, None, 2), src, [7.0, 0.0, 8.0, 0.0, 9.0, 0.0])


# --- the semantics a lowering has to get right, measured upstream ------------


def test_upstream_casts_the_value_into_the_receivers_dtype():
    """`int64 x; x[0::2] = [1.7, 2.7, 3.7]` is [1,0,2,0,3,0] upstream.

    Not a refusal -- upstream reaches a stepped write through `copy_`, which
    casts. `aten.index_put_.default` does NOT cast; it refuses a dtype
    mismatch with upstream's own wording, and its refusal is right for the
    calls that really are `index_put_` (`x[t] = v`). So a lowering that routes
    a stepped write to `index_put_` has to cast first, and this pins which of
    the two rules wins here. Getting it backwards would refuse a write
    upstream performs, which is the direction that stops an architecture.
    """
    x = _from([0, 0, 0, 0, 0, 0], (6,), "int64")
    src = _from([1.7, 2.7, 3.7], (3,), "float32")
    _expect(x, slice(None, None, 2), src, [1, 0, 2, 0, 3, 0])


def test_a_number_on_the_right_of_a_stepped_write_broadcasts():
    """`zeros(6)[1::2] = 5.0` is [0,5,0,5,0,5] upstream."""
    x = _from([0.0] * 6, (6,))
    _expect(x, slice(1, None, 2), 5.0, [0.0, 5.0, 0.0, 5.0, 0.0, 5.0])


def test_a_negative_step_is_a_value_error_and_not_a_shim_refusal():
    """Upstream: `ValueError: step must be greater than zero`.

    This is not a gap and must never become one. A lowering that turned the
    slice into positions with Python's own `slice.indices` would happily
    accept a negative step and write in reverse, which upstream refuses -- so
    the check has to be explicit, and it has to raise the same class.
    """
    x = _from([0.0] * 5, (5,))
    src = _from([1.0, 2.0, 3.0, 4.0, 5.0], (5,))
    try:
        x[::-1] = src
    except ValueError as e:
        assert "step must be greater than zero" in str(e), str(e)
    except NotImplementedError as e:  # pragma: no cover - the pre-patch route
        assert REFUSAL_NAME in str(e), str(e)
    else:
        raise AssertionError(
            "a negative step must not write; upstream raises ValueError"
        )


def test_a_negative_start_resolves_against_the_extent():
    """`zeros(6)[-6::2] = [1,2,3]` is [1,0,2,0,3,0] upstream.

    Negative bounds are the trap docs/DEMAND8.md warns about by name: the
    neighbouring op is not evidence, because `index_add_`'s indices do not
    wrap while `index_put_`'s do. Here the negative number is a slice BOUND
    rather than an index, and a slice bound resolves against the extent
    before anything else sees it -- so this and the wrap rule are two
    different questions and both are measured rather than assumed.
    """
    x = _from([0.0] * 6, (6,))
    src = _from([1.0, 2.0, 3.0], (3,))
    _expect(x, slice(-6, None, 2), src, [1.0, 0.0, 2.0, 0.0, 3.0, 0.0])


# --- what the lowering rests on, which holds today ---------------------------


def test_index_put_answers_for_the_shapes_a_stepped_write_becomes():
    """A stepped slice is a set of positions, and `index_put_` already writes
    a set of positions through the receiver's own storage.

    Both architecture shapes become an `index_put_` call with the stepped axis
    as an integer index and every other axis a `None`. Neither had coverage
    before docs/SETITEM.md: every existing `[None, index]` case has exactly
    one leading `None`, and the mrope shape needs two.
    """
    # `pe[:, 0::2] = matrix`  ->  [None, [0, 2]]
    x = _from([0.0] * 12, (3, 4))
    idx = _from([0, 2], (2,), "int64")
    val = _from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
    _C._aten_dispatch("aten.index_put_.default", x, [None, idx], val, False)
    assert x.tolist() == [[1.0, 0.0, 2.0, 0.0],
                          [3.0, 0.0, 4.0, 0.0],
                          [5.0, 0.0, 6.0, 0.0]], x.tolist()

    # `freqs_t[..., 1::3] = src`  ->  [None, None, [1, 4]]
    y = _from([0.0] * 12, (2, 1, 6))
    idy = _from([1, 4], (2,), "int64")
    valy = _from([1.0, 2.0, 3.0, 4.0], (2, 1, 2))
    _C._aten_dispatch("aten.index_put_.default", y, [None, None, idy], valy, False)
    assert y.tolist() == [[[0.0, 1.0, 0.0, 0.0, 2.0, 0.0]],
                          [[0.0, 3.0, 0.0, 0.0, 4.0, 0.0]]], y.tolist()


def test_index_put_writes_through_a_view_taken_before_the_call():
    """The property that makes the lowering legitimate at all.

    Routing a stepped write to `index_put_` is only equivalent to upstream's
    `slice + copy_` if `index_put_` writes into the buffer the receiver
    already points at. If it swapped the wrapper instead, this lowering would
    reproduce upstream's *values* through the receiver and lose them through
    every alias -- a divergence no value comparison could see.
    """
    x = _from([0.0] * 6, (6,))
    v = _C._aten_dispatch("aten.slice.Tensor", x, 0, 0, 4, 1)
    idx = _from([0, 2, 4], (3,), "int64")
    val = _from([7.0, 8.0, 9.0], (3,))
    _C._aten_dispatch("aten.index_put_.default", x, [idx], val, False)
    assert x.tolist() == [7.0, 0.0, 8.0, 0.0, 9.0, 0.0], x.tolist()
    assert v.tolist() == [7.0, 0.0, 8.0, 0.0], (
        "the step-1 view taken before the write did not see it -- "
        "index_put_ is rebinding rather than writing through"
    )


def test_index_put_refuses_a_second_index_group_by_name():
    """Why the lowering is restricted to ONE stepped slice.

    `index_put_` implements a single index group. `x[0::2, 1::3] = v` would
    need two, and the honest answer is a refusal that says so rather than an
    approximation. This pins the kernel-level refusal the Python-level one
    stands on, so the restriction cannot quietly become an incorrect write.
    """
    x = _from([0.0] * 12, (3, 4))
    a = _from([0, 2], (2,), "int64")
    b = _from([1, 3], (2,), "int64")
    val = _from([1.0, 2.0], (2,))
    try:
        _C._aten_dispatch("aten.index_put_.default", x, [a, b], val, False)
    except NotImplementedError as e:
        assert "more than one index tensor" in str(e), str(e)
    else:
        raise AssertionError(
            "index_put_ accepted two index groups; the __setitem__ lowering's "
            "one-group restriction is resting on a refusal that is gone"
        )


# --- the handoff itself ------------------------------------------------------


def test_the_setitem_document_carries_the_patch_it_promises():
    """docs/SETITEM.md exists to be applied, so its applicability is checked.

    The translation lives in `bootstrap.py`, which belonged to another round;
    the deliverable is therefore a document with an exact patch in it. A
    document is not a check, so this makes the one part that can be checked
    into one: that §2 is present and holds the anchor it claims to replace,
    verbatim as it appears in `bootstrap.py`. If a later edit moves that
    anchor, the patch stops applying and this fails, rather than the patch
    silently rotting until somebody tries it.
    """
    root = pathlib.Path(__file__).resolve().parents[3]
    doc = root / "docs" / "SETITEM.md"
    assert doc.exists(), "docs/SETITEM.md is missing -- the patch has no home"
    text = doc.read_text()
    assert "## 2." in text, "SETITEM.md has no §2 (the patch)"

    anchor = "        index = _index_tuple(index)\n        index = _expand_ellipsis(self, index)"
    assert anchor in text, (
        "SETITEM.md no longer quotes the anchor its patch replaces"
    )
    bootstrap = (root / "rust" / "torch_c" / "src" / "bootstrap.py").read_text()
    assert bootstrap.count(anchor) == 2, (
        f"the anchor SETITEM.md's patch keys on appears {bootstrap.count(anchor)} "
        "times in bootstrap.py, not the 2 it did when the patch was written "
        "(__getitem__ and __setitem__) -- the patch may no longer apply cleanly"
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
