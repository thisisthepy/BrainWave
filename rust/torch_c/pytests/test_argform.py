"""Tests for the argument-form gaps docs/ARGFORM.md closes.

docs/ARCH100.md swept 297 architectures and found six blocked not by a
missing operator but by an argument *form* of an op that already exists:

    torch.ones(dtype=bool)                                     gpt_neo
    Tensor.mean(axis=int, keepdim=bool)                        ibert
    torch.mean(Tensor, axis=int, keepdim=bool)                 imagegpt
    torch.div(int, int, rounding_mode=str)                     longformer
    aten.convolution.default: asymmetric per-axis padding      nystromformer
    torch.embedding(Parameter, NoneType, int, bool, bool)      sam3_lite_text_text_model

plus two more found separately and left in `bootstrap.py`'s territory:

    F.adaptive_avg_pool2d(x, 2) -- a bare int                  dinov3_convnext, efficientnet

Every alias here was checked against real torch 2.13.0 *before* being added
(CLAUDE.md's own warning: upstream accepts a numpy spelling on some ops and
refuses it on others, so accepting it everywhere would be a shim more
permissive than the thing it replaces). What each measurement found and what
was done about it is in docs/ARGFORM.md; this file is the proof the code
actually does what that document claims.

Two of the six are NOT touched here, and are not bugs in this file:

  * the "asymmetric" conv padding was `aten.rs`'s candle backend refusing a
    per-axis-differing value by name -- out of that round's territory
    (`aten.rs`) and a backend limit, not an argument-form gap. **That refusal
    is gone**: docs/LAST7.md §4 spends the difference as explicit zero padding
    on the input and convolves with the common remainder, so the pin here is
    now `test_per_axis_conv_padding_now_computes_and_agrees_with_upstream`,
    which diffs values, plus
    `test_a_per_axis_differing_stride_is_still_refused_by_name` for the half
    that has no such lowering.
  * the `torch.embedding` case reproduces upstream's OWN refusal (measured:
    `TypeError: embedding(): argument 'indices' ... must be Tensor, not
    NoneType`) -- not an argument-form gap at all, so there is nothing to
    close here; see docs/ARGFORM.md's note on it.

Also covers the `torch._C._nn.glu` binding wired up in the same round --
docs/ARCH100.md's `_nn.glu` blocker, seven ASR encoders. The kernel
(`aten.glu.default`) lands on a different branch and is not in this
worktree, so the numeric behaviour cannot be asserted here; only that the
binding exists, is advertised, and reaches the dispatcher under the right
key with the right default `dim`.
"""

from test_shim import _C


def _vf(name):
    return getattr(_C._VariableFunctions, name)


# --- torch.ones(dtype=<python type>) ----------------------------------------


def test_ones_dtype_accepts_python_bool_int_float_like_upstream():
    # Measured against torch 2.13.0: `torch.ones(2, dtype=bool)` is a bool
    # tensor of True, `dtype=int` is int64, `dtype=float` is float64 -- not
    # float32, which is what a *tensor* built from Python floats gets
    # (measured separately: `torch.tensor([1.0]).dtype` is float32). The two
    # rules disagree and this is upstream's `dtype=` rule, not the literal
    # inference rule.
    bools = _vf("ones")(2, 3, dtype=bool)
    assert bools.dtype == _C.bool, bools.dtype
    assert bools.tolist() == [[True, True, True], [True, True, True]]

    ints = _vf("ones")(2, dtype=int)
    assert ints.dtype == _C.int64, ints.dtype
    assert ints.tolist() == [1, 1]

    floats = _vf("ones")(2, dtype=float)
    assert floats.dtype == _C.float64, floats.dtype
    assert floats.tolist() == [1.0, 1.0]


def test_dtype_python_type_alias_does_not_touch_an_explicit_torch_dtype():
    # The translation only fires for the three bare Python types. A caller
    # spelling the ordinary way must see no behaviour change at all -- this
    # is the "does the new path ever shadow the old one" check CLAUDE.md
    # asks for.
    explicit = _vf("ones")(2, dtype=_C.float32)
    assert explicit.dtype == _C.float32, explicit.dtype


# --- Tensor.mean(axis=)/torch.mean(axis=) and keepdims= ---------------------


def test_mean_axis_alias_matches_dim_on_both_the_method_and_the_free_function():
    # Measured against torch 2.13.0: `x.mean(axis=1, keepdim=True)` and
    # `torch.mean(x, axis=1, keepdim=True)` both resolve, and both give
    # exactly what `dim=1` gives -- confirmed by comparing the two calls
    # against each other and against a hand-computed row mean below.
    x = _C._tensor_from_flat(
        [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [3, 4],
    )
    by_dim = x.mean(dim=1, keepdim=True)
    by_axis = x.mean(axis=1, keepdim=True)
    assert by_axis.tolist() == by_dim.tolist() == [[-3.5], [0.5], [4.5]]

    top_level = _vf("mean")(x, axis=1, keepdim=True)
    assert top_level.tolist() == by_dim.tolist()


def test_mean_keepdims_alias_matches_keepdim():
    x = _C._tensor_from_flat([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert x.mean(dim=1, keepdim=True).tolist() == x.mean(
        dim=1, keepdims=True
    ).tolist()


def test_axis_alias_is_not_installed_for_an_op_that_was_not_measured():
    # docs/ARGFORM.md's whole point: `axis=` is accepted on some ops and
    # refused on others upstream, so it must not be installed as a blanket
    # rule. `sum` was not one of the six architectures and was not measured
    # here, so it must still refuse -- proving the alias table is scoped to
    # what was actually checked, not "reductions in general".
    x = _C._tensor_from_flat([1.0, 2.0, 3.0, 4.0], [2, 2])
    try:
        x.sum(axis=1)
    except TypeError as e:
        assert "no matching overload" in str(e), str(e)
    else:
        raise AssertionError(
            "Tensor.sum(axis=...) resolved -- the numpy-alias table has "
            "leaked into an op docs/ARGFORM.md never measured"
        )


def test_giving_both_dim_and_axis_does_not_silently_pick_one():
    # Measured against upstream: `mean(dim=1, axis=1)` raises "multiple
    # values for argument 'dim'". This shim's alias rewrite (rename rather
    # than merge) reaches the same outcome for a different mechanical
    # reason -- `dim` ends up bound twice, which the schema binder already
    # refuses -- but the observable contract (refuse, do not guess) matches.
    x = _C._tensor_from_flat([1.0, 2.0, 3.0, 4.0], [2, 2])
    try:
        x.mean(dim=1, axis=1)
    except TypeError:
        pass
    else:
        raise AssertionError("mean(dim=1, axis=1) must not silently resolve")


# --- torch.div(int, int, rounding_mode=...) ---------------------------------


def test_div_wraps_bare_python_numbers_the_way_upstream_wraps_them():
    # Measured against torch 2.13.0 with a TorchDispatchMode logger:
    # `torch.div(7, 2, rounding_mode='trunc')` reaches `aten.div.Tensor_mode`
    # -- upstream wraps the two bare Python ints into 0-dim tensors rather
    # than refusing for want of a `Scalar, Scalar` overload. The wrapped
    # dtype is upstream's own rule too: int -> int64, float -> float32.
    assert _vf("div")(7, 2, rounding_mode="trunc").item() == 3
    assert _vf("div")(-7, 2, rounding_mode="floor").item() == -4
    # No rounding_mode: true division, which promotes even an int/int divide
    # to float32 -- measured, and exactly what the real `aten.div.Tensor`
    # kernel already does once both operands are real tensors, which is the
    # whole point of wrapping rather than hand-computing the answer here.
    plain = _vf("div")(7, 2)
    assert plain.dtype == _C.float32, plain.dtype
    assert plain.item() == 3.5


def test_div_scalar_wrapping_does_not_disturb_the_tensor_tensor_path():
    # The wrapping only applies when `input` itself is a bare Python number.
    # A real Tensor `self` must reach the table exactly as before -- proof
    # that the override does not shadow the existing (already-working)
    # Tensor/Tensor and Tensor/Scalar overloads.
    a = _C._tensor_from_flat([7.0], [1])
    b = _C._tensor_from_flat([2.0], [1])
    assert _vf("div")(a, b).tolist() == [3.5]
    assert _vf("div")(a, 2).tolist() == [3.5]


# --- F.adaptive_avg_pool2d(x, 2) -- a bare int ------------------------------


def test_adaptive_avg_pool2d_bare_int_normalises_to_a_pair():
    # docs/FIXES.md §3: the raw aten op refuses a bare int (matching
    # upstream's raw aten op, measured) but upstream's `torch._C._nn.
    # adaptive_avg_pool2d` -- a *different* binding, and the one `F.
    # adaptive_avg_pool2d` actually calls -- accepts one and expands it to a
    # pair. This is that expansion, one door up from the raw op, which stays
    # untouched and still refuses a scalar (the second assertion below).
    x = _C._tensor_from_flat(list(range(64)), [1, 1, 8, 8]).to(dtype=_C.float32)
    bare = _C._nn.adaptive_avg_pool2d(x, 2)
    pair = _C._nn.adaptive_avg_pool2d(x, (2, 2))
    assert bare.tolist() == pair.tolist()
    assert bare.tolist() == [[[[13.5, 17.5], [45.5, 49.5]]]]

    try:
        _C._aten_dispatch("aten.adaptive_avg_pool2d.default", x, 2)
    except Exception:
        pass
    else:
        raise AssertionError(
            "the raw aten op accepted a bare int -- this is the exact "
            "SILENT DIVERGENCE docs/FIXES.md §3 measured and reverted; the "
            "expansion belongs in the _nn binding, not the aten op"
        )


def test_adaptive_avg_pool2d_none_entries_keep_the_input_size():
    # `_list_with_default`'s other rule, measured alongside the bare-int one:
    # a `None` entry in the output_size tuple means "keep this axis as is".
    x = _C._tensor_from_flat(list(range(64)), [1, 1, 8, 8]).to(dtype=_C.float32)
    kept_height = _C._nn.adaptive_avg_pool2d(x, (None, 2))
    assert kept_height.shape == (1, 1, 8, 2)


# --- the backend-limited one, pinned so it is not mistaken for fixed -------


def test_per_axis_conv_padding_now_computes_and_agrees_with_upstream():
    """nystromformer's wall, inverted.

    **This test was `test_asymmetric_conv_padding_is_a_backend_limit_not_an_
    argument_form` and asserted the refusal.** docs/ARGFORM.md §1 measured that
    `padding=[0, 5]` is not asymmetric padding at all -- it is two axes each
    padded symmetrically by a different amount, which upstream computes fine --
    and classified this shim's refusal as a genuine candle limitation rather
    than an argument-form gap. That classification was right about candle and
    wrong about the conclusion: `conv2d`'s padding really is one scalar, but the
    difference between the axes can be spent as explicit zero padding on the
    input, exactly as docs/RNN.md §2 spent the odd half of `padding='same'`.
    docs/LAST7.md §4 lands that lowering, so the assertion above became an
    assertion that a working op is missing.

    Inverted into the stronger form: the values, against a reference measured on
    real torch 2.13.0 in a separate process rather than against this shim's own
    other spelling of the same lowering, which would be circular. The kernel is
    `1x6` and the padding is `(0, 5)`, so a lowering that padded the wrong
    candle axis gives `(1, 1, 20, 5)` instead of `(1, 1, 10, 15)` -- the shape
    separates that one -- and a lowering that padded only one side gives the
    right shape with every element shifted, which is what the element checks
    below are for.
    """
    inp = _C._tensor_from_flat([i / 4 for i in range(100)], [1, 1, 10, 10])
    wgt = _C._tensor_from_flat([i / 3 for i in range(6)], [1, 1, 1, 6])
    out = _C._aten_dispatch(
        "aten.convolution.default",
        inp, wgt, None, [1, 1], [0, 5], [1, 1], False, [0, 0], 1,
    )
    assert list(out.shape) == [1, 1, 10, 15], list(out.shape)
    flat = _C._aten_dispatch("aten.reshape.default", out, [-1])
    got = [round(float(_C._aten_dispatch("aten._local_scalar_dense.default",
                                         _C._aten_dispatch("aten.select.int", flat, 0, k))), 4)
           for k in range(15)]
    # Upstream, measured: the first output row. The leading zero and the
    # trailing zero are the two ends of the width padding.
    want = [0.0, 0.4167, 1.1667, 2.1667, 3.3333, 4.5833, 5.8333, 7.0833,
            8.3333, 9.5833, 6.6667, 4.1667, 2.1667, 0.75, 0.0]
    assert got == want, (got, want)
    total = float(_C._aten_dispatch(
        "aten._local_scalar_dense.default",
        _C._aten_dispatch("aten.sum.default", out)))
    assert round(total, 2) == 6187.50, total


def test_a_per_axis_differing_stride_is_still_refused_by_name():
    """The half of docs/ARGFORM.md §1's finding that did NOT close.

    Padding has a lowering because zeros can be added to the input; a stride has
    none -- there is nothing to add that makes an unequal stride equal. So the
    refusal stays, and it stays *named*, which is what keeps the two halves
    distinguishable. docs/LAST7.md §4.
    """
    inp = _C._tensor_from_flat([0.0] * 98, [1, 2, 7, 7])
    wgt = _C._tensor_from_flat([0.0] * 36, [2, 2, 3, 3])
    try:
        _C._aten_dispatch(
            "aten.convolution.default",
            inp, wgt, None, [2, 1], [0, 0], [1, 1], False, [0, 0], 1,
        )
    except NotImplementedError as e:
        assert "asymmetric stride" in str(e), str(e)
    else:
        raise AssertionError(
            "a per-axis-differing stride resolved -- if the backend grew this, "
            "update docs/LAST7.md §4 and invert this test too"
        )


# --- torch._C._nn.glu -------------------------------------------------------


def test_glu_is_advertised_and_reaches_the_kernel_key_with_the_right_default_dim():
    """The binding, now that the kernel it reaches is in the same tree.

    This test was written in a worktree that had the binding and *not* the
    kernel, so it could only assert that `_nn.glu` reached
    `aten.glu.default` -- by catching the refusal and reading the key out of
    it. The kernel merged, so that spelling became an assertion that the op is
    missing, and it went red the moment both halves met.

    Inverted rather than deleted, and asserting the stronger thing: the values,
    against upstream's own definition. `glu(x, dim)` splits `x` in half along
    `dim` and returns `a * sigmoid(b)`, so a binding that reached the right key
    with the wrong `dim` -- the trap docs/GLU.md measured, since the default is
    `-1` and not `0` -- gives different numbers here rather than passing.
    """
    import math

    x = _C._tensor_from_flat([1.0, 2.0, 3.0, 4.0], [2, 2])

    # dim=-1 (the default): halves are columns. Row 0 is [1, 2] -> 1*sig(2).
    got = [round(float(v), 6) for v in _C._nn.glu(x).flatten()]
    want = [round(1.0 * (1 / (1 + math.exp(-2.0))), 6),
            round(3.0 * (1 / (1 + math.exp(-4.0))), 6)]
    assert got == want, (got, want)

    # dim=0: halves are rows. [1, 2] * sigmoid([3, 4]).
    got0 = [round(float(v), 6) for v in _C._nn.glu(x, 0).flatten()]
    want0 = [round(1.0 * (1 / (1 + math.exp(-3.0))), 6),
             round(2.0 * (1 / (1 + math.exp(-4.0))), 6)]
    assert got0 == want0, (got0, want0)

    # And the two are different, so "the default is -1" is a claim this test
    # can actually falsify.
    assert got != got0, got

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
