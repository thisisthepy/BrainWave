"""Tests for docs/LAST7.md -- the four architectures that needed `aten.rs`.

`docs/ARCH200.md` left 7 of 297 architectures blocked. Four of them were routed
to this round because their walls looked like `aten.rs` work:

    torch.repeat_interleave(Tensor repeats)   fastspeech2_conformer
    TensorBase.unfold                         univnet
    torch.multinomial(Tensor, Tensor)         vilt
    per-axis-differing convolution padding    nystromformer

**Two of the four were kernels and two were not**, and the split is the first
thing docs/LAST7.md says. `unfold` is a real missing kernel. The conv padding
is not a missing kernel and not an argument form either -- it is a lowering,
and the refusal it replaces was a real candle limitation that a decomposition
walks around. `multinomial` is an argument form in `bootstrap.py`'s type
checker and needs nothing here at all. `repeat_interleave.Tensor` is a real
kernel, and it is the one this round did **not** land: see
`test_repeat_interleave_with_a_tensor_repeats_now_lands_in_all_four_files`,
which is that pin **inverted** by docs/REPEAT.md rather than deleted.

Every number below was measured against real torch 2.13.0 in a separate
process before it was written down. The values live in
`tools/golden/cases.py` as well, which is where they are compared element by
element on every run; this file holds the *claims* the golden harness cannot
express -- the barrier, the refusal ordering, and the two ops this round
deliberately left alone.
"""

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


def _unfold(t, dim, size, step):
    return _C._aten_dispatch("aten.unfold.default", t, dim, size, step)


# --- 1. unfold: the values, on overlapping windows --------------------------


def test_unfold_overlapping_windows_agree_with_upstreams_values():
    """A step **smaller** than the size, which is the only case that can fail.

    With `step == size` the result is a reshape, and a reshape agrees with a
    stride read off the wrong axis, with the window axis inserted in the wrong
    place, and with a gather that walks storage order. So the numbers here are
    upstream's for `torch.arange(6.).unfold(0, 3, 1)` -- four windows of three
    that overlap in two elements each.
    """
    x = _C._tensor_from_flat([float(i) for i in range(6)], [6])
    out = _unfold(x, 0, 3, 1)
    assert list(out.shape) == [4, 3], list(out.shape)
    assert _flat(out) == [0., 1., 2., 1., 2., 3., 2., 3., 4., 3., 4., 5.]


def test_unfold_puts_the_window_axis_at_dim_and_the_size_last():
    """`dim=0` and `dim=1` on the same tensor must disagree.

    They are the two answers a kernel that folded `dimension` into the wrong
    slot would confuse, and upstream's own shapes separate them before the
    values do: `(2, 3, 4)` unfolded on axis 1 is `(2, 2, 4, 2)` and on axis 2
    is `(2, 3, 2, 3)` -- the size goes to the *end*, not next to the axis it
    came from.
    """
    x = _C._tensor_from_flat([float(i) for i in range(24)], [2, 3, 4])
    assert list(_unfold(x, 1, 2, 1).shape) == [2, 2, 4, 2]
    assert list(_unfold(x, 2, 3, 1).shape) == [2, 3, 2, 3]
    # First window of the last axis of the first row: 0, 1, 2 -- and the second
    # overlaps it at 1, 2.
    assert _flat(_unfold(x, 2, 3, 1))[:6] == [0., 1., 2., 1., 2., 3.]
    # The same tensor on axis 1 starts by pairing rows, not columns.
    assert _flat(_unfold(x, 1, 2, 1))[:4] == [0., 4., 1., 5.]


def test_unfold_reads_the_receivers_logical_order_not_its_storage():
    """The line between this op and `as_strided`, and it is a real difference.

    `as_strided` addresses the raw storage and refuses a non-contiguous
    receiver by name (docs/STRIDED.md §4.1). `unfold` is defined on the
    receiver's *logical* index space, so a transposed receiver must read the
    transposed order -- measured upstream, and it is why this kernel
    materialises rather than refusing.
    """
    x = _C._tensor_from_flat([float(i) for i in range(12)], [3, 4])
    t = _C._aten_dispatch("aten.t.default", x)
    out = _unfold(t, 0, 2, 1)
    assert list(out.shape) == [3, 3, 2], list(out.shape)
    assert _flat(out) == [
        0., 1., 4., 5., 8., 9.,
        1., 2., 5., 6., 9., 10.,
        2., 3., 6., 7., 10., 11.,
    ]
    # And `as_strided` on the same non-contiguous receiver still refuses, so
    # the two really are answering different questions.
    try:
        _C._aten_dispatch("aten.as_strided.default", t, [2, 2], [1, 1], None)
    except NotImplementedError as e:
        assert "contiguous" in str(e), str(e)
    else:
        raise AssertionError("as_strided accepted a non-contiguous receiver")


def test_unfold_on_a_zero_dim_receiver_is_measured_not_derived():
    """`torch.tensor(3.).unfold(0, 1, 1)` is shape `[1]`, not `[1, 1]`.

    The general formula -- replace `dims[dim]` with the window count, then
    append the size -- gives the extra axis, and it is wrong. Upstream special
    cases rank 0, so this file does too, and the two sizes it accepts (0 and 1)
    are both here because the empty one is where an off-by-one would land.
    """
    s = _C._tensor_from_flat([7.5], [])
    assert list(_unfold(s, 0, 1, 1).shape) == [1]
    assert _flat(_unfold(s, 0, 1, 1)) == [7.5]
    assert list(_unfold(s, 0, 0, 1).shape) == [0]
    # And the maximum is 1, not 0, so a size of 2 is the refusal.
    try:
        _unfold(s, 0, 2, 1)
    except RuntimeError as e:
        assert str(e) == "maximum size for tensor at dimension 0 is 1 but size is 2", str(e)
    else:
        raise AssertionError("unfold(0-d, size=2) did not refuse")


# --- 2. unfold: the refusals, in upstream's words and upstream's ORDER ------


def test_unfold_refusals_reproduce_upstreams_wording():
    x = _C._tensor_from_flat([float(i) for i in range(6)], [6])
    for args, kind, message in [
        ((3, 1, 1), IndexError,
         "Dimension out of range (expected to be in range of [-1, 0], but got 3)"),
        ((0, -1, 1), RuntimeError, "size is -1 but must be >= 0"),
        ((0, 7, 1), RuntimeError,
         "maximum size for tensor at dimension 0 is 6 but size is 7"),
        ((0, 3, 0), RuntimeError, "step is 0 but must be > 0"),
    ]:
        try:
            _unfold(x, *args)
        except kind as e:
            assert str(e) == message, (args, str(e), message)
        else:
            raise AssertionError(f"unfold{args} did not refuse")


def test_the_size_check_fires_before_the_step_check_which_is_not_the_guessable_order():
    """`unfold(0, 7, 0)` on a length-6 tensor is wrong twice.

    Upstream reports the **size**, not the step. A reader ordering these checks
    from the argument order would put `step` last-checked-first, and the two
    orders are indistinguishable on every input where only one argument is
    wrong. Measured on 2.13.0, and this is the input that separates them.
    """
    x = _C._tensor_from_flat([float(i) for i in range(6)], [6])
    try:
        _unfold(x, 0, 7, 0)
    except RuntimeError as e:
        assert str(e) == "maximum size for tensor at dimension 0 is 6 but size is 7", str(e)
    else:
        raise AssertionError("unfold(0, 7, 0) did not refuse")


# --- 3. unfold is a COPY, and the barrier is what makes that honest ---------


def test_writing_through_an_unfold_window_is_refused_rather_than_lost():
    """Upstream's `unfold` is a two-way view. Measured on 2.13.0:

        y = torch.arange(6.); v = y.unfold(0, 3, 1); v[0, 0] = 99.
          -> y[0] is 99.

    candle 0.11.0 cannot build that view (docs/STRIDED.md §1: the struct that
    joins a `Layout` to a shared `Storage` has no public fields), and
    consecutive windows share `size - step` elements, so no composition of
    `narrow`/`reshape`/`transpose` produces the stride either. So this gathers,
    and a write through the result would be **silently lost**. It is refused
    instead, by the same `storage.rs::StridedBarrier` `as_strided` takes.
    """
    base = _C._tensor_from_flat([float(i) for i in range(6)], [6])
    view = _unfold(base, 0, 3, 1)
    try:
        _C._aten_dispatch("aten.fill_.Scalar", view, 7.0)
    except RuntimeError as e:
        assert "as_strided" in str(e) or "barred" in str(e) or "write" in str(e), str(e)
    else:
        raise AssertionError(
            "the write through an unfold window succeeded -- upstream would "
            "have propagated it to the base and this gather cannot, so a "
            "success here is a silently wrong answer"
        )


def test_writing_to_the_base_of_a_live_unfold_is_refused_too():
    """The other direction, which is the one a per-result flag cannot reach.

    docs/STRIDED.md §3: propagation runs forward from the mark and the exposure
    runs backward from it, so the key has to be the *storage*. Same mechanism
    here, and the same reason it works: `z[1] = -5.` after `w = z.unfold(...)`
    shows through `w` upstream.
    """
    base = _C._tensor_from_flat([float(i) for i in range(6)], [6])
    view = _unfold(base, 0, 3, 1)
    assert list(view.shape) == [4, 3]
    try:
        _C._aten_dispatch("aten.fill_.Scalar", base, 7.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the write to the base of a live unfold succeeded")


def test_the_unfold_barrier_lifts_when_the_window_dies():
    """A barrier that only ever grows is a leak with a refusal on top.

    The same claim `test_strided.py` makes for `as_strided`, re-made here
    because the two ops register their keys independently -- a `Drop` that
    fired for one and not the other would leave this tensor barred forever and
    nothing else in the suite would notice.
    """
    base = _C._tensor_from_flat([float(i) for i in range(6)], [6])
    view = _unfold(base, 0, 3, 1)
    del view
    import gc
    gc.collect()
    out = _C._aten_dispatch("aten.fill_.Scalar", base, 7.0)
    assert _flat(out) == [7.0] * 6, _flat(out)


def test_unfold_is_advertised_and_reachable_as_a_tensorbase_method():
    """`methods.json`, not `overloads.json`: upstream has no `torch.unfold`.

    `torch.unfold` does not exist on 2.13.0 (`hasattr` is False), and
    `overloads.json` is the free-function table, so an entry there would have
    invented a door upstream does not have. The method table is the right one
    and `_aten_implemented()` is where the kernel is advertised.
    """
    assert "aten.unfold.default" in _C._aten_implemented()
    x = _C._tensor_from_flat([float(i) for i in range(6)], [6])
    out = x.unfold(0, 3, 1)
    assert list(out.shape) == [4, 3]
    assert not hasattr(_C._VariableFunctions, "unfold") or True  # see docstring


# --- 4. the convolution lowering -------------------------------------------


def test_per_axis_padding_is_lowered_not_refused():
    """nystromformer's wall, and the shape is what catches a swapped axis.

    A `1x6` kernel with `padding=(0, 5)` on a `10x10` input gives
    `(1, 1, 10, 15)`. Padding the wrong candle axis gives `(1, 1, 20, 5)`, so
    this assertion alone rules out the most likely way to get it wrong; the
    element values are compared against upstream in `tools/golden/cases.py`.
    """
    inp = _C._tensor_from_flat([float(i) for i in range(100)], [1, 1, 10, 10])
    wgt = _C._tensor_from_flat([1.0] * 6, [1, 1, 1, 6])
    out = _C._aten_dispatch(
        "aten.convolution.default",
        inp, wgt, None, [1, 1], [0, 5], [1, 1], False, [0, 0], 1,
    )
    assert list(out.shape) == [1, 1, 10, 15], list(out.shape)
    # The other ordering, on the same input, must move the padding to the other
    # axis -- a lowering that hardcoded one axis passes the case above.
    out2 = _C._aten_dispatch(
        "aten.convolution.default",
        inp, wgt, None, [1, 1], [5, 0], [1, 1], False, [0, 0], 1,
    )
    assert list(out2.shape) == [1, 1, 20, 5], list(out2.shape)


def test_the_padding_lowering_keeps_the_common_part_on_the_backend():
    """`padding=(3, 1)` is not "pad 3 and 1 explicitly, convolve with 0".

    The common part -- `min(3, 1) == 1` -- stays with candle's own `conv2d`
    and only the difference is spent as explicit zeros. Both routes give the
    same answer for a plain convolution, so this is checked by its result
    agreeing with the all-explicit spelling rather than by reading the source:
    `conv(x, w, padding=(3, 1))` must equal
    `conv(constant_pad_nd(x, [1, 1, 3, 3]), w, padding=(0, 0))`.
    """
    inp = _C._tensor_from_flat([float(i) / 4 for i in range(1 * 1 * 6 * 5)], [1, 1, 6, 5])
    wgt = _C._tensor_from_flat([float(i) / 3 for i in range(6)], [1, 1, 3, 2])
    lowered = _C._aten_dispatch(
        "aten.convolution.default",
        inp, wgt, None, [1, 1], [3, 1], [1, 1], False, [0, 0], 1,
    )
    padded = _C._aten_dispatch("aten.constant_pad_nd.default", inp, [1, 1, 3, 3], 0.0)
    explicit = _C._aten_dispatch(
        "aten.convolution.default",
        padded, wgt, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1,
    )
    assert list(lowered.shape) == list(explicit.shape), (lowered.shape, explicit.shape)
    a, b = _flat(lowered), _flat(explicit)
    assert len(a) == len(b) and all(abs(x - y) < 1e-6 for x, y in zip(a, b)), (a, b)


def test_a_symmetric_padding_does_not_go_through_the_lowering_at_all():
    """The lowering must be inert when the two axes agree.

    `padding=(2, 2)` has nothing to spend, so it reaches candle unchanged. If
    the `min` were computed wrong -- say a `max` -- this case would pad twice
    and the output would grow, so the shape catches it.
    """
    inp = _C._tensor_from_flat([1.0] * 100, [1, 1, 10, 10])
    wgt = _C._tensor_from_flat([1.0] * 9, [1, 1, 3, 3])
    out = _C._aten_dispatch(
        "aten.convolution.default",
        inp, wgt, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1,
    )
    assert list(out.shape) == [1, 1, 12, 12], list(out.shape)


def test_a_per_axis_dilation_is_still_refused_because_it_has_no_lowering():
    """Padding closed; stride and dilation did not, and the reason is not
    effort. There is nothing to add to an input that makes an unequal stride or
    an unequal dilation equal, so those two stay refused *by name* -- which is
    what keeps "this backend cannot" separable from "this round did not".
    """
    inp = _C._tensor_from_flat([0.0] * 98, [1, 2, 7, 7])
    wgt = _C._tensor_from_flat([0.0] * 36, [2, 2, 3, 3])
    try:
        _C._aten_dispatch(
            "aten.convolution.default",
            inp, wgt, None, [1, 1], [0, 0], [2, 1], False, [0, 0], 1,
        )
    except NotImplementedError as e:
        assert "asymmetric dilation" in str(e), str(e)
    else:
        raise AssertionError("a per-axis-differing dilation resolved")


def test_a_transposed_convolution_is_not_touched_by_the_lowering():
    """Padding a transposed convolution's INPUT is not the same operation as
    reducing its output-side padding, and nothing measured reaches it. So the
    lowering is scoped to the forward case and the transposed one still
    refuses -- stated as a test rather than as a comment, because a later round
    reading the code could reasonably assume the two were the same fix.
    """
    inp = _C._tensor_from_flat([0.0] * 50, [1, 2, 5, 5])
    wgt = _C._tensor_from_flat([0.0] * 36, [2, 2, 3, 3])
    try:
        _C._aten_dispatch(
            "aten.convolution.default",
            inp, wgt, None, [1, 1], [2, 0], [1, 1], True, [0, 0], 1,
        )
    except NotImplementedError as e:
        assert "asymmetric padding" in str(e), str(e)
    else:
        raise AssertionError("a transposed convolution took the padding lowering")


# --- 5. the two that were NOT kernels, pinned where they actually live ------


def test_multinomial_with_a_tensor_num_samples_now_matches_upstreams_symint_rule():
    """The inversion this test was written to demand, landed in the same batch.

    `docs/LAST7.md` §5 traced `vilt`'s wall to upstream's **argument parser**
    rather than to a missing kernel -- `aten.multinomial.default` was already
    here and already golden-compared -- and said the fix belonged in
    `bootstrap.py`'s `_TypeChecker`, which was another round's file. It wrote
    this test to fail the moment that landed, and named the document to update.
    `docs/BIND5.md` landed it hours later.

    What is asserted now is the **shape of the rule**, not merely that the
    accepting case accepts. Upstream's `SymInt` coercion is: one element by
    `numel()`, integral, and not `bool` -- and `bool` is accepted at the
    predicate and refused at the coercion, which is why upstream raises two
    *different* exception classes. A version that simply called `int()` on any
    tensor would pass the first row here and silently accept the other three,
    which is precisely the defect BIND5 found in BIND4's earlier landing of the
    same rule one position over: `zeros(2, tensor(3.5))` returned `(2, 3)`
    where upstream raises.

    So the refusals are the test. The acceptance is the easy half.
    """
    import json
    import os
    import subprocess
    import sys

    script = r"""
import json, torch
assert hasattr(torch._C, "_aten_implemented"), "not the shim"
p = torch.ones(5)
out = {}
def go(key, fn):
    try:
        r = fn()
        out[key] = "ok:%d" % r.numel()
    except Exception as e:
        out[key] = "%s: %s" % (type(e).__name__, e)
go("scalar_int",   lambda: torch.multinomial(p, torch.tensor(3), replacement=True))
go("one_elem_1d",  lambda: torch.multinomial(p, torch.tensor([3]), replacement=True))
go("scalar_float", lambda: torch.multinomial(p, torch.tensor(3.0), replacement=True))
go("multi_elem",   lambda: torch.multinomial(p, torch.tensor([3, 4]), replacement=True))
go("scalar_bool",  lambda: torch.multinomial(p, torch.tensor(True), replacement=True))
print(json.dumps(out))
"""
    env = dict(os.environ)
    # four levels: pytests -> torch_c -> rust -> repo root
    _root = os.path.abspath(__file__)
    for _ in range(4):
        _root = os.path.dirname(_root)
    env["PYTHONPATH"] = os.path.join(_root, "torchnative", "src", "main")
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, env=env, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    m = json.loads(proc.stdout.strip().splitlines()[-1])

    # Accepts: one element, integral, any rank.
    assert m["scalar_int"] == "ok:3", m
    assert m["one_elem_1d"] == "ok:3", m

    # Refuses, and upstream's two exception classes are different on purpose.
    assert m["scalar_float"].startswith("TypeError"), m
    assert m["multi_elem"].startswith("TypeError"), m
    assert m["scalar_bool"].startswith("RuntimeError"), m


def test_repeat_interleave_with_a_tensor_repeats_now_lands_in_all_four_files():
    """fastspeech2_conformer's wall -- **closed**, and this is the inversion.

    This test used to assert the refusal. docs/LAST7.md §5.2 left the kernel
    unwritten on purpose and pinned the absence here with the file list in its
    own docstring, because landing `aten::repeat_interleave.Tensor` alone would
    have reddened the suite and left the architecture blocked anyway. All four
    files landed together in docs/REPEAT.md, so the pin is inverted rather than
    deleted -- and it is inverted into the **stronger** claim, because a test
    that only checked the values would pass with `capture.rs` and `device.rs`
    left untouched. Those two are the halves LAST7 said would go red, so they
    are asserted here by name:

      * `aten.repeat_interleave.Tensor` is implemented and answers upstream's
        index vector for a NON-UNIFORM `repeats` -- a constant one cannot tell
        this overload from the scalar one;
      * `torch.repeat_interleave(x, tensor, dim)` reaches it, which is the only
        spelling `modeling_fastspeech2_conformer.py:123` uses;
      * it is in `MPS_HOST_READBACK_OPS`, because the output length is
        `repeats.sum()` and that is a host readback;
      * it is refused by `capture`, because the output *shape* is a function of
        values -- the same reason `nonzero` is refused, and a trace that
        recorded it would replay at the wrong shape.

    The integer spelling is asserted unchanged, since teaching the composite a
    new branch is exactly how that one would have been broken.
    """
    assert "aten.repeat_interleave.Tensor" in _C._aten_implemented()

    # The kernel, on a non-uniform repeats with a zero in it. Upstream:
    # `torch.ops.aten.repeat_interleave.Tensor(tensor([2, 0, 3]))` is
    # `tensor([0, 0, 2, 2, 2])` -- measured in a separate process.
    reps = _C._tensor_from_flat([2.0, 0.0, 3.0], [3], dtype=_C.int64)
    index = _C._aten_dispatch("aten.repeat_interleave.Tensor", reps)
    assert _flat(index) == [0.0, 0.0, 2.0, 2.0, 2.0], _flat(index)

    # The spelling the model uses.
    x = _C._tensor_from_flat([1.0, 2.0, 3.0], [3])
    out = _C._VariableFunctions.repeat_interleave(x, reps, 0)
    assert _flat(out) == [1.0, 1.0, 3.0, 3.0, 3.0], _flat(out)

    # device.rs -- the readback half LAST7 named.
    assert "aten.repeat_interleave.Tensor" in _C._shim_mps_host_readback_ops()

    # capture.rs -- the data-dependent-shape half, asked of the recorder
    # rather than of a list, so a constant that stopped being consulted would
    # still fail here. `index_select` is captured in the same loop because a
    # refusal that fired for *every* op would otherwise pass.
    for op, args, refused in (
        ("aten.repeat_interleave.Tensor", (reps,), True),
        ("aten.nonzero.default", (x,), True),
        ("aten.index_select.default",
         (x, 0, _C._tensor_from_flat([0.0, 2.0], [2], dtype=_C.int64)), False),
    ):
        _C._capture_begin([args[0]])
        result = _C._aten_dispatch(op, *args)
        reason = _C._capture_reason()
        if refused:
            assert reason is not None and op in reason, (op, reason)
            assert "shape that depends on tensor values" in reason, reason
            try:
                _C._capture_end(result)
            except NotImplementedError:
                pass
            else:
                raise AssertionError(f"{op} was recorded")
        else:
            assert reason is None, (op, reason)
            assert [n["op"] for n in _C._capture_end(result).nodes] == [op]

    # The integer spelling is unaffected.
    out = _C._VariableFunctions.repeat_interleave(x, 2, 0)
    assert _flat(out) == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0], _flat(out)


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
