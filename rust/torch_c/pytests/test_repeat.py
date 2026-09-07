"""docs/kernels/REPEAT.md -- the last three blocked architectures.

Three things landed and they are different in kind, so they are kept apart
here rather than counted together:

  1. `TensorBase.where` (`led`, `longformer`) -- **not a kernel and not a
     table row.** Every `where` overload was already implemented; what was
     missing was the *method*, and the reason it could not be a `methods.json`
     row is the whole of §1: the receiver is the `x` branch, which is the aten
     schema's **second** argument, and that table binds the receiver into the
     **first**.
  2. `aten::repeat_interleave.Tensor` (`fastspeech2_conformer`) -- a real
     kernel, in four files at once (docs/kernels/LAST7.md §5.2 sized it and
     deliberately did not land it).
  3. `sam3_lite_text_text_model` -- re-checked, still **not our gap**.

Every value below was measured on real torch 2.13.0 in a separate process,
element by element, on inputs where a plausible wrong implementation differs.
The two properties that shape the choice of inputs:

  * for `where`, the branches must **disagree**, or swapping them is invisible;
  * for `repeat_interleave`, the `repeats` must be **non-uniform**, or the
    tensor overload and the scalar one answer the same thing.
"""

import json
import os

from test_shim import _C


def _flat(t):
    """Every element of `t`, as Python floats, through the dispatcher only."""
    flat = _C._aten_dispatch("aten.reshape.default", t, [-1])
    n = int(flat.shape[0])
    return [
        float(_C._aten_dispatch(
            "aten._local_scalar_dense.default",
            _C._aten_dispatch("aten.select.int", flat, 0, k)))
        for k in range(n)
    ]


def _t(flat, shape, dtype=None):
    if dtype is None:
        return _C._tensor_from_flat(list(flat), list(shape))
    return _C._tensor_from_flat(list(flat), list(shape), dtype=dtype)


def _cond(flat, shape):
    """A bool tensor, which is what `where` refuses to work without."""
    return _C._aten_dispatch(
        "aten._to_copy.default", _t(flat, shape), dtype=_C.bool
    )


# --- 1. TensorBase.where: the argument ORDER, which is the whole risk -------


def test_tensor_where_takes_the_receiver_as_the_true_branch_not_the_condition():
    """`x.where(c, y)` is `torch.where(c, x, y)`, measured on upstream.

    The receiver is **`x`**, not the condition. Upstream, for
    `x = [[1, 2], [3, 4]]`, `y = [[10, 20], [30, 40]]` and
    `c = [[True, False], [False, True]]`:

        x.where(c, y)      ->  [[1., 20.], [30., 4.]]
        torch.where(c, x, y) ->  [[1., 20.], [30., 4.]]

    A binding that put the receiver in the condition slot answers the **same
    shape** and the **same dtype** with the branches swapped, and there is no
    shape check anywhere that can see that. So the two branches here are
    disjoint value ranges and the assertion is element-wise: the wrong order
    would give `[[10., 2.], [3., 40.]]`, and that list is written into the
    test so a reader can see the two answers are not each other's rounding.
    """
    x = _t([1.0, 2.0, 3.0, 4.0], [2, 2])
    y = _t([10.0, 20.0, 30.0, 40.0], [2, 2])
    c = _cond([1.0, 0.0, 0.0, 1.0], [2, 2])

    got = _flat(x.where(c, y))
    assert got == [1.0, 20.0, 30.0, 4.0], got
    # The answer a receiver-into-argument-0 binding would give. Asserted as a
    # non-answer so the two cannot be confused by a future reader.
    assert got != [10.0, 2.0, 3.0, 40.0]

    # And it agrees with the free function on the same three operands, which
    # is the relationship the method *is*.
    assert _flat(_C._VariableFunctions.where(c, x, y)) == got


def test_tensor_where_is_not_a_methods_json_row_and_the_table_still_has_none():
    """The absence is deliberate; this states why, so nobody adds the row.

    `methods.json`'s machine binds the receiver into schema argument **0** --
    `_Overloads(self_bound=True)` passes it as `args[0]` and every positional
    count skips exactly one. `aten::where.self`'s argument 0 is `condition`.
    So a `where` row in that table is not a smaller version of the right fix,
    it is the swapped-branch bug of the test above, installed by a table.

    Asserted from the loaded build rather than from the source file, and the
    method is asserted present in the same test so this cannot pass by the
    method having gone missing too.
    """
    # The method is present -- asserted first, so the absence below cannot
    # pass by `where` having gone missing from both places at once.
    assert callable(getattr(_C.TensorBase, "where", None))

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "src", "methods.json"
    )
    if not os.path.exists(path):
        print("   (skipped the methods.json half: the tree is not beside this "
              "file -- installed rather than in-tree)")
        return
    with open(path, encoding="utf-8") as handle:
        table = json.load(handle)
    assert "where" not in table, (
        "methods.json grew a `where` row. That table binds the receiver into "
        "schema argument 0, and `aten::where.self`'s argument 0 is the "
        "CONDITION -- the row computes torch.where(x, c, y), which is the "
        "right shape with the branches swapped. See _install_tensor_where."
    )
    # And the row it would have been made from is still in the *function*
    # table, where argument 0 genuinely is the condition.
    over = os.path.join(os.path.dirname(path), "overloads.json")
    with open(over, encoding="utf-8") as handle:
        assert any(
            "where.self(" in schema for schema in json.load(handle)["where"]
        )


def test_tensor_where_takes_a_scalar_other_through_the_scalarother_overload():
    """`x.where(c, 0.5)` -- upstream accepts a Number and so does this.

    The scalar arm is a *different aten overload* (`where.ScalarOther`), so it
    is a second binding decision rather than a corollary of the first, and the
    receiver is in the same second slot there. An int and a bool are included
    because upstream treats both as wrapped numbers here and neither promotes
    the float receiver -- measured: `x.where(c, 3)` is float, not int.
    """
    x = _t([1.0, 2.0, 3.0, 4.0], [2, 2])
    c = _cond([1.0, 0.0, 0.0, 1.0], [2, 2])
    assert _flat(x.where(c, 0.5)) == [1.0, 0.5, 0.5, 4.0]
    out = x.where(c, 3)
    assert out.dtype == _C.float32, out.dtype
    assert _flat(out) == [1.0, 3.0, 3.0, 4.0]
    assert _flat(x.where(c, True)) == [1.0, 1.0, 1.0, 4.0]


def test_tensor_where_broadcasts_all_three_operands_not_just_two():
    """Upstream broadcasts condition, receiver and other together.

    `arange(3).where([T, F, T], [[9], [8]])` is `(2, 3)` upstream -- the
    *other* operand's leading axis survives, which a kernel that took its
    output shape from the condition alone would drop. Measured value:
    `[[0., 9., 2.], [0., 8., 2.]]`.
    """
    x = _t([0.0, 1.0, 2.0], [3])
    c = _cond([1.0, 0.0, 1.0], [3])
    y = _t([9.0, 8.0], [2, 1])
    out = x.where(c, y)
    assert list(out.shape) == [2, 3], list(out.shape)
    assert _flat(out) == [0.0, 9.0, 2.0, 0.0, 8.0, 2.0]


def test_tensor_where_refuses_what_upstream_refuses_and_in_upstreams_words():
    """A non-bool condition, and an `other` that is neither Tensor nor Number.

    Both are upstream's, measured. The float condition matters because a
    kernel that treated it as truthy would compute a plausible answer for
    every model that ever passed a mask as floats.
    """
    x = _t([1.0, 2.0, 3.0, 4.0], [2, 2])
    y = _t([10.0, 20.0, 30.0, 40.0], [2, 2])
    try:
        x.where(_t([1.0, 0.0, 0.0, 1.0], [2, 2]), y)
    except RuntimeError as e:
        assert "boolean tensor" in str(e), str(e)
    else:
        raise AssertionError("a float condition must refuse")

    c = _cond([1.0, 0.0, 0.0, 1.0], [2, 2])
    try:
        x.where(c, None)
    except TypeError as e:
        # Upstream names *both* overloads, not only the nearer one.
        assert "Tensor other" in str(e) and "Number other" in str(e), str(e)
    else:
        raise AssertionError("a None `other` must refuse")


def test_the_five_where_overloads_are_untouched_by_the_new_method():
    """The method is surface over kernels that already existed.

    docs/bindings/BIND5.md §4.1's claim, re-checked on this tree rather than inherited
    -- docs/bindings/BINDINGS.md was told "`mish` just needs a binding" and found the
    kernel gone, so "the kernels are there" is measured here.
    """
    for overload in ("aten.where.self", "aten.where.default",
                     "aten.where.Scalar", "aten.where.ScalarSelf",
                     "aten.where.ScalarOther"):
        assert overload in _C._aten_implemented(), overload


# --- 2. repeat_interleave.Tensor -------------------------------------------


def _ri(flat, dtype=None, **kw):
    reps = _t(flat, [len(flat)], dtype=dtype or _C.int64)
    return _C._aten_dispatch("aten.repeat_interleave.Tensor", reps, **kw)


def test_repeat_interleave_tensor_answers_the_index_vector_not_the_data():
    """The op's argument is `repeats` alone and its answer is an INDEX vector.

    This is the shape of the op that is easiest to get wrong from the name:
    `aten::repeat_interleave.Tensor(Tensor repeats, *, SymInt? output_size)`
    never sees the data being repeated. A `TorchDispatchMode` logger on
    upstream reports `repeat_interleave.Tensor` then `index_select.default`
    for `torch.repeat_interleave(x, tensor([2, 0, 3]), 0)`, and the kernel
    alone answers `tensor([0, 0, 2, 2, 2])` -- the positions `index_select`
    then gathers.

    The `repeats` here is **non-uniform and contains a zero**. A constant one
    would agree with a kernel that ignored the values entirely, and the zero
    is the position an off-by-one emits anyway.
    """
    assert _flat(_ri([2.0, 0.0, 3.0])) == [0.0, 0.0, 2.0, 2.0, 2.0]
    # A vector and its reverse, which no length-only arithmetic separates.
    assert _flat(_ri([1.0, 2.0, 3.0])) == [0.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert _flat(_ri([3.0, 2.0, 1.0])) == [0.0, 0.0, 0.0, 1.0, 1.0, 2.0]
    # All zeros is empty, not length 3.
    assert list(_ri([0.0, 0.0, 0.0]).shape) == [0]
    assert list(_ri([]).shape) == [0]


def test_repeat_interleave_tensor_keeps_the_repeats_dtype():
    """`int32` in, `int32` out -- upstream's answer, measured.

    Not cosmetic: `index_select` here accepts `int32` and `int64` and refuses
    everything else, so a kernel that always answered `int64` would hide a
    dtype it was never handed.
    """
    assert _ri([2.0, 0.0, 3.0]).dtype == _C.int64
    assert _ri([2.0, 0.0, 3.0], dtype=_C.int32).dtype == _C.int32


def test_repeat_interleave_tensor_refusals_are_upstreams_own_words():
    """Four refusals, each measured on torch 2.13.0.

    The float one is worth naming: upstream refuses by *kernel* name
    (`"repeat_interleave_cpu" not implemented for 'Float'`) rather than by
    schema, so a caller who greps for that string has upstream's.
    """
    for flat, shape, dtype, kw, exc, fragment in [
        ([2.0, 3.0], [1, 2], _C.int64, {}, RuntimeError,
         "only accept 1D vector as repeat"),
        ([2.0, -1.0], [2], _C.int64, {}, RuntimeError,
         "repeats can not be negative"),
        ([2.0, 3.0], [2], None, {}, NotImplementedError,
         '"repeat_interleave_cpu" not implemented for \'Float\''),
        ([2.0, 0.0, 3.0], [3], _C.int64, {"output_size": 6}, RuntimeError,
         "allocated size does not match required size"),
    ]:
        reps = _t(flat, shape, dtype=dtype)
        try:
            _C._aten_dispatch("aten.repeat_interleave.Tensor", reps, **kw)
        except exc as e:
            assert fragment in str(e), (fragment, str(e))
        else:
            raise AssertionError(f"expected a refusal for {flat} {kw}")

    # An AGREEING output_size is accepted and changes nothing.
    reps = _t([2.0, 0.0, 3.0], [3], dtype=_C.int64)
    out = _C._aten_dispatch("aten.repeat_interleave.Tensor", reps, output_size=5)
    assert _flat(out) == [0.0, 0.0, 2.0, 2.0, 2.0]


def test_the_composite_reaches_the_kernel_on_the_spelling_the_model_uses():
    """`torch.repeat_interleave(input, tensor, dim=0)` -- the only spelling.

    docs/kernels/LAST7.md §5.2's first bullet: the composite is `setattr` onto the
    varfns **after** the table, so it wins, and it used to raise before any
    dispatch happened. A kernel behind it would have been unreachable through
    `modeling_fastspeech2_conformer.py:123`.

    `dim=0` and `dim=1` are both asserted on a NON-SQUARE input, so a wrong
    axis fails on shape before it fails on values.
    """
    vf = _C._VariableFunctions
    x = _t([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [3, 2])

    out = vf.repeat_interleave(x, _t([2.0, 0.0, 3.0], [3], dtype=_C.int64), 0)
    assert list(out.shape) == [5, 2], list(out.shape)
    assert _flat(out) == [0.0, 1.0, 0.0, 1.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0]

    out = vf.repeat_interleave(x, _t([2.0, 1.0], [2], dtype=_C.int64), 1)
    assert list(out.shape) == [3, 3], list(out.shape)
    assert _flat(out) == [0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0]

    # The member spelling binds the same closure and cannot drift.
    assert _flat(x.repeat_interleave(_t([2.0, 0.0, 3.0], [3], dtype=_C.int64), 0)) \
        == [0.0, 1.0, 0.0, 1.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0]


def test_a_one_element_repeats_broadcasts_above_the_kernel_not_inside_it():
    """`repeat_interleave(x, tensor([2]), 0)` on a 3-row input.

    Upstream's trace makes the split visible: `view [1]`, `expand [3]`, *then*
    `repeat_interleave.Tensor` on a length-3 vector. So the broadcast is the
    composite's and the length check is the kernel's caller's. Pushing the
    broadcast down into the kernel would make a legitimate `[2]` for a
    one-row input indistinguishable from a length error.
    """
    vf = _C._VariableFunctions
    x = _t([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [3, 2])
    out = vf.repeat_interleave(x, _t([2.0], [1], dtype=_C.int64), 0)
    assert list(out.shape) == [6, 2], list(out.shape)
    assert _flat(out) == [0.0, 1.0, 0.0, 1.0, 2.0, 3.0,
                          2.0, 3.0, 4.0, 5.0, 4.0, 5.0]


def test_a_dimless_call_flattens_first_and_a_wrong_length_says_so():
    """`dim=None` flattens, which changes what "the same size" means.

    Upstream's message on a `(3, 2)` input with a length-3 repeats and no
    `dim` names `input.size(0) = 6`, not 3 -- the flatten has already
    happened. That number is the check that the flatten is in the right place.
    """
    vf = _C._VariableFunctions
    x = _t([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [3, 2])
    out = vf.repeat_interleave(x, _t([1.0] * 6, [6], dtype=_C.int64))
    assert list(out.shape) == [6], list(out.shape)

    try:
        vf.repeat_interleave(x, _t([2.0, 0.0, 3.0], [3], dtype=_C.int64))
    except RuntimeError as e:
        assert "input.size(0) = 6" in str(e), str(e)
    else:
        raise AssertionError("a length-3 repeats on a flattened 6 must refuse")

    # And with `dim=0` the same repeats is correct, so the message above is
    # about the flatten rather than about the vector.
    assert list(vf.repeat_interleave(
        x, _t([2.0, 0.0, 3.0], [3], dtype=_C.int64), 0).shape) == [5, 2]


def test_the_one_argument_spelling_is_the_kernel_and_a_dim_beside_it_refuses():
    """`torch.repeat_interleave(tensor([2, 0, 3]))` is the kernel, alone.

    Upstream has a third overload whose single argument *is* the repeats
    vector; it answers `tensor([0, 0, 2, 2, 2])`. It used to raise here with
    the same words as the two-argument form. A `dim` beside it is a
    combination error upstream, because the overload that takes `dim` is the
    one that takes an input too.
    """
    vf = _C._VariableFunctions
    reps = _t([2.0, 0.0, 3.0], [3], dtype=_C.int64)
    assert _flat(vf.repeat_interleave(reps)) == [0.0, 0.0, 2.0, 2.0, 2.0]
    assert _flat(vf.repeat_interleave(reps, output_size=5)) == \
        [0.0, 0.0, 2.0, 2.0, 2.0]
    try:
        vf.repeat_interleave(reps, dim=0)
    except TypeError as e:
        assert "invalid combination" in str(e), str(e)
    else:
        raise AssertionError("a dim beside a lone repeats must refuse")


def test_the_integer_repeats_path_is_unchanged_by_the_tensor_one():
    """The branch that already worked, asserted because it is the one at risk.

    Teaching a composite a new arm is how the old arm breaks. `dim=-1` and
    `dim=0` disagree on this input, and `dim=None` flattens, so all three of
    the integer path's decisions are separated.
    """
    vf = _C._VariableFunctions
    m = _t([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3])
    assert _flat(vf.repeat_interleave(m, 2, dim=-1)) == \
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
    assert _flat(vf.repeat_interleave(m, 2, dim=0)) == \
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0]
    assert list(vf.repeat_interleave(m, 2).shape) == [12]
    try:
        vf.repeat_interleave(m, -1, dim=0)
    except RuntimeError as e:
        assert "non-negative" in str(e), str(e)
    else:
        raise AssertionError("a negative integer repeats must refuse")


def test_the_two_lists_the_kernel_had_to_join_are_both_asserted_here():
    """`device.rs` and `capture.rs`, which are what LAST7 §5.2 said would go red.

    Kept in this file as well as in the inverted pin in `test_last7.py`,
    because these are the two claims a value test cannot make and the two a
    later change is most likely to undo while the values still pass.

    The readback list is asserted against the **loaded artefact**, not against
    the source constant beside it, so a change nobody rebuilt cannot agree
    with itself. The capture refusal is asked of the recorder for the same
    reason.
    """
    assert "aten.repeat_interleave.Tensor" in _C._shim_mps_host_readback_ops()

    reps = _t([2.0, 0.0, 3.0], [3], dtype=_C.int64)
    _C._capture_begin([reps])
    result = _C._aten_dispatch("aten.repeat_interleave.Tensor", reps)
    reason = _C._capture_reason()
    assert reason is not None, "capture recorded a data-dependent shape"
    assert "aten.repeat_interleave.Tensor" in reason, reason
    assert "shape that depends on tensor values" in reason, reason
    try:
        _C._capture_end(result)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("capture returned a trace it had refused")


# --- 3. sam3_lite_text_text_model, re-checked ------------------------------


def test_sam3_lite_texts_embedding_call_is_refused_by_upstream_too():
    """docs/bindings/ARGFORM.md §3 and docs/bindings/BIND5.md both said "not our gap". Re-checked.

    The call is `torch.embedding(weight, None, ...)` -- a `None` where the
    schema wants `Tensor indices`. Upstream refuses it identically, so closing
    it here would be a *widening* against upstream rather than a fix, and the
    architecture count is "three ours, one not" rather than four.

    Asserted as a refusal with the right shape (a `TypeError`, from overload
    resolution, before any kernel runs) rather than by comparing message text,
    since the two parsers word it differently and only the refusal is the
    claim. The same call with real indices is asserted to work in the same
    test, so a `torch.embedding` that had broken entirely would not pass here.
    """
    vf = _C._VariableFunctions
    weight = _t([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [3, 2])
    try:
        vf.embedding(weight, None)
    except TypeError:
        pass
    else:
        raise AssertionError("embedding(weight, None) must refuse, as upstream does")
    out = vf.embedding(weight, _t([2.0, 0.0], [2], dtype=_C.int64))
    assert _flat(out) == [4.0, 5.0, 0.0, 1.0], _flat(out)


# --- 4. the SymInt rule inside an int list ---------------------------------


def test_a_single_element_integral_tensor_is_accepted_inside_a_size_list():
    """`led` and `longformer`'s wall **after** `where` fell.

    Closing `TensorBase.where` moved both architectures one call later, into
    `_sliding_chunks_matmul_attn_probs_value`:

        padded_value.as_strided(size=chunked_value_size, stride=...)

    where one element of `chunked_value_size` arrives as a **0-dim tensor**.
    docs/bindings/BIND5.md §7 taught `_TypeChecker` upstream's `SymInt` rule for a
    *scalar* position; this is the same rule applied per **element** of an int
    list, which is what upstream's parser does and what
    `_coerce_symint_size_tensors` already did for three hand-written
    composites. docs/kernels/REPEAT.md §4.

    A 0-dim tensor and a one-element 1-D tensor are both accepted upstream,
    and both are asserted, because they take different branches of `numel()`.
    """
    x = _t([float(i) for i in range(12)], [3, 4])
    two = _t([2.0], [], dtype=_C.int64)
    two_1d = _t([2.0], [1], dtype=_C.int64)

    assert list(x.as_strided((2, two), (4, 1)).shape) == [2, 2]
    assert list(x.as_strided(size=(2, two_1d), stride=(4, 1)).shape) == [2, 2]
    # The rule is on the list *element*, not on `as_strided`, so it holds for
    # every int-list argument. `view` and `permute` are `SymInt[]` and `int[]`
    # respectively, which is the pair that shows it is not one spelling.
    assert list(x.view((two, 6)).shape) == [2, 6]
    assert list(x.permute((_t([1.0], [], dtype=_C.int64), 0)).shape) == [4, 3]
    # And in the `stride` position too, which is a second list on the same
    # call -- a rule wired only to the argument named `size` would pass every
    # assertion above and fail this one.
    assert list(x.as_strided((2, 2), (_t([4.0], [], dtype=_C.int64), 1)).shape) \
        == [2, 2]


def test_the_int_list_symint_rule_refuses_exactly_what_upstream_refuses():
    """Three refusals, and the two exception CLASSES are upstream's.

    Measured on torch 2.13.0 for `x.as_strided(size=(2, X), stride=(4, 1))`:

        X = tensor(2.0)     -> TypeError      even though it is whole-valued
        X = tensor([2, 3])  -> TypeError      more than one element
        X = tensor(True)    -> RuntimeError   'Expected scalar.isIntegral(...)'

    The split is structural rather than chosen: the predicate accepts `bool`
    and the *coercion* refuses it, one layer further in, which is where
    upstream refuses it too. A version that refused `bool` in the predicate
    would answer `TypeError` for all three and pass any test that only checked
    that something raised.

    The float and multi-element cases raise this shim's "no matching overload"
    text rather than upstream's "failed to unpack the object at pos 2",
    because here they are a *binding* failure and upstream reaches them inside
    a bound argument. The class is the claim; the wording is recorded as
    differing rather than asserted.
    """
    x = _t([float(i) for i in range(12)], [3, 4])
    bad_float = _t([2.0], [])
    bad_two = _t([2.0, 3.0], [2], dtype=_C.int64)
    bad_bool = _C._aten_dispatch(
        "aten._to_copy.default", _t([1.0], []), dtype=_C.bool
    )

    for element, exc in ((bad_float, TypeError), (bad_two, TypeError),
                         (bad_bool, RuntimeError)):
        try:
            x.as_strided((2, element), (4, 1))
        except exc:
            pass
        else:
            raise AssertionError(f"{element.dtype} must refuse with {exc}")

    # A plain `bool` in a size list is still refused, which is the rule the
    # tensor case must not have widened: `bool` subclasses `int` in Python and
    # torch excludes it explicitly.
    try:
        x.as_strided((2, True), (4, 1))
    except TypeError:
        pass
    else:
        raise AssertionError("a python bool in a size list must refuse")


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
