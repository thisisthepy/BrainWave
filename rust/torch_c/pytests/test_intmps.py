"""`int16` and `int32` on Metal: the wrap-identity argument, and a refusal that
names the dtype instead of a candle shader symbol.

docs/numerics/DTYPEDEV.md section 4.2 left this cell open with two roads out of
it, and this file is the round that walked both and says which one it took.

**The measurement it started from.** `int64` reaches four of `add mul sum neg`
on `mps`; `int16` and `int32` reach none. And the refusal was this:

    aten.add.Tensor: candle: Metal error Error while loading function: badd_i16

`badd_i16` is a candle-internal Metal shader name. It names neither the dtype
nor the device nor what to do instead, and it is the *whole* family that speaks
this way -- `bmul_i32`, `Metal contiguous to_dtype I16 I64 not implemented`,
`Metal copy_strided I32 not implemented`, `mlx matmul doesn't support I32`.

**The design question, answered before any kernel.** The obvious move is to
promote to `int64`, compute, and narrow back. Integer arithmetic in torch wraps
on overflow, so a promotion is only allowed where the wrapped answer survives
the round trip. The argument, per operator, is in docs/numerics/DTYPEDEV.md
section 4.2 and is checked below by
`test_the_ring_operators_survive_a_promotion_round_trip` and
`test_the_integer_reductions_return_int64_and_so_are_never_narrowed`. In short:

  add sub mul neg   safe.  Reduction mod 2**N is a ring homomorphism and
                    `2**16` and `2**32` both divide `2**64`, so Z -> Z/2**64 ->
                    Z/2**N composes; and torch's narrowing integer cast was
                    measured to truncate, not saturate.
  sum prod cumsum   safe and must NOT be narrowed -- upstream returns `int64`
                    for these on an `int16`/`int32` input and does not wrap.
  div abs max       not claimed. Not ring operations; and every one of them is
                    already refused on `mps` for `int64` too, by the
                    host-readback gate, so they are not a dtype gap.

**And the promotion is still not done, for a reason the argument did not
reach.** An `int16`/`int32` buffer on Metal is *sealed*: candle instantiates its
Metal kernels for `f32 f16 bf16 u8 u32 i64` only and
`candle_metal_kernels::DType` has no `I16`/`I32` variant at all, so there is no
cast **off** an `I16` buffer in any direction -- not to `I64`, not to `F32`, not
even to `I32`. The widening cast the promotion depends on is itself one of the
missing kernels. Promoting would mean reading the tensor back to the host,
which is exactly what `mps_host_readback_gate` exists to refuse: a correct value
the GPU did not compute, under an `mps` label.

So this round delivers the refusal rather than the kernel, which is the
outcome docs/numerics/DTYPEDEV.md section 1 ranks above a quietly wrong one.

Nullifications this file is meant to catch:

* `name_metal_int_refusal` returning the error unchanged
      -> test_int16_and_int32_on_metal_refuse_by_name_rather_than_by_shader_symbol
* the refusal applied to `add` only, not the family
      -> test_the_named_refusal_covers_the_whole_family
* the escape hatch closing
      -> test_the_escape_hatch_off_a_sealed_int_metal_tensor_stays_open
* a promotion added for a reduction, narrowing where upstream does not
      -> test_the_integer_reductions_return_int64_and_so_are_never_narrowed
"""

import os

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C

try:
    import torch as _upstream_torch
except ImportError:  # pragma: no cover
    _upstream_torch = None


_DTYPES = ("int16", "int32")

# The bit widths, so the wrap identity below is arithmetic rather than a table
# of constants somebody could mistype.
_BITS = {"int16": 16, "int32": 32}


def _mps_or_skip(what):
    """A live `mps` device, or None having said why not.

    Same shape as `test_dtypedev._mps_or_skip` and deliberately not imported
    from it: the skip line is what this file promises to keep truthful, so it
    says which file skipped.
    """
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        print("   (skipped %s: no mps device on this machine -- %s)"
              % (what, str(e).splitlines()[0]))
        return None
    return _C.device("mps")


def _cpu(name, values):
    return _C._tensor_from_flat(
        [float(v) for v in values], [len(values)], _C.float64
    ).to(getattr(_C, name))


def _vals(t):
    h = t.cpu() if str(t.device) != "cpu" else t
    return [int(v) for v in h.to(_C.float64).flatten().tolist()]


# ---------------------------------------------------------------------------
# The wrap identity, which had to be established before any promotion
# ---------------------------------------------------------------------------


def _wrap(x, bits):
    """Two's-complement reduction of a Python int into `bits` bits.

    This is the `Z -> Z/2**N` half of the homomorphism the argument rests on,
    written out so the test compares against arithmetic rather than against
    another implementation of the same idea.
    """
    m = 1 << bits
    x &= m - 1
    return x - m if x >= (m >> 1) else x


def test_the_narrowing_integer_cast_truncates_rather_than_saturating():
    """The load-bearing measurement under the whole promotion argument.

    `promote, compute, narrow` is only wrap-identical if the narrowing step is
    a truncation -- keep the low N bits and reinterpret. If it saturated,
    `int16(32767) + int16(1)` would come back as `32767` where upstream wraps
    to `-32768`, and every promoted operator would be wrong at the boundary
    while looking right everywhere else.

    Asked of both this build and upstream, because the argument needs it of
    both: upstream is what the answer is compared against, and this build is
    what would perform the narrowing.
    """
    for name in _DTYPES:
        bits = _BITS[name]
        probes = [1 << (bits - 1), (1 << bits) - 1, -1, 0, (1 << bits)]
        want = [_wrap(v, bits) for v in probes]
        got = _vals(_C._tensor_from_flat(
            [float(v) for v in probes], [len(probes)], _C.float64
        ).to(_C.int64).to(getattr(_C, name)))
        assert got == want, (
            "narrowing int64 -> %s gave %s, not the truncating answer %s. If "
            "this build saturates, no promotion in this file is wrap-identical "
            "and none of them may be applied." % (name, got, want))
        if _upstream_torch is not None:
            up = _upstream_torch.tensor(probes, dtype=_upstream_torch.int64).to(
                getattr(_upstream_torch, name)).tolist()
            assert up == want, (
                "upstream narrowing int64 -> %s gave %s, not %s. The oracle "
                "disagrees with the arithmetic the argument is written in."
                % (name, up, want))


def test_the_ring_operators_survive_a_promotion_round_trip():
    """Grade: **agrees**, at the boundaries, for `add sub mul neg`.

    Reduction mod `2**N` is a ring homomorphism, and `2**16` and `2**32` both
    divide `2**64`, so `Z -> Z/2**64 -> Z/2**N` is a composition of ring
    homomorphisms. Any expression built only from `+`, `-`, `*` and constants
    therefore has the same image whether it is evaluated at width N throughout
    or evaluated at width 64 and reduced at the end -- **including when the
    width-64 evaluation itself overflows**, since `2**N` divides `2**64`.

    That is the argument. This test is the check of it, at the values where an
    argument about wrapping either holds or does not: `iinfo(dtype).max`,
    `.min`, and products far past both.

    `neg` is in the list because it is `0 - x` and therefore a ring operation,
    and it is the case most likely to be got wrong by hand: `-iinfo.min` is
    `iinfo.min` again, at both widths.
    """
    if _upstream_torch is None:
        print("   (skipped: no upstream torch in this interpreter -- the "
              "oracle half of this test cannot run)")
        return
    for name in _DTYPES:
        bits = _BITS[name]
        hi, lo = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
        probes = [hi, lo, 1, -1, hi - 1, lo + 1]
        a = _cpu(name, probes)
        ua = _upstream_torch.tensor(probes, dtype=getattr(_upstream_torch, name))
        cases = {
            "add": (lambda t: t + t, lambda v: v + v),
            "sub": (lambda t: t - t, lambda v: v - v),
            "mul": (lambda t: t * t, lambda v: v * v),
            "neg": (lambda t: -t, lambda v: -v),
        }
        for op, (fn, ref) in sorted(cases.items()):
            want = [_wrap(ref(v), bits) for v in probes]
            up = fn(ua)
            assert up.tolist() == want, (
                "upstream %s on %s gave %s, not the wrapped answer %s -- the "
                "premise that torch integer arithmetic wraps is what the whole "
                "promotion argument rests on." % (op, name, up.tolist(), want))
            assert str(up.dtype).replace("torch.", "") == name, (
                "upstream %s on %s returned %s. A promotion argument for an "
                "operator that changes dtype is a different argument."
                % (op, name, up.dtype))
            got = fn(a)
            assert _vals(got) == want, (
                "this build's %s on %s (cpu) gave %s, not %s"
                % (op, name, _vals(got), want))
            # The identity itself: compute promoted, narrow, and land on the
            # same values. This is the operation a Metal promotion *would*
            # perform, checked on the cpu where both widths have kernels.
            promoted = fn(a.to(_C.int64)).to(getattr(_C, name))
            assert _vals(promoted) == want, (
                "promoting %s to int64, computing %s and narrowing back gave "
                "%s, not %s. The wrap identity does not hold for this "
                "operator and it must not be promoted."
                % (name, op, _vals(promoted), want))


def test_the_integer_reductions_return_int64_and_so_are_never_narrowed():
    """`sum`, `prod` and `cumsum` on an `int16`/`int32` input return `int64`
    **and do not wrap**, upstream and here.

    This is why they need no wrap identity: there is no narrowing step to
    prove anything about. It is recorded as a test rather than a remark
    because the failure it guards is a promotion added later that helpfully
    narrows the result back to the input dtype -- which would look tidy, agree
    with the input dtype, and be wrong for every reduction that leaves the
    range.

    The mid-reduction overflow case is the one that shows it: `max + max + min`
    is `32766` for `int16`, which is *not* what accumulating at width 16 would
    give, and upstream returns exactly that.
    """
    if _upstream_torch is None:
        print("   (skipped: no upstream torch in this interpreter -- the "
              "oracle half of this test cannot run)")
        return
    for name in _DTYPES:
        bits = _BITS[name]
        hi, lo = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
        probes = [hi, hi, lo]
        ua = _upstream_torch.tensor(probes, dtype=getattr(_upstream_torch, name))
        assert str(ua.sum().dtype) == "torch.int64", ua.sum().dtype
        assert ua.sum().item() == sum(probes), (
            "upstream sum on %s gave %s, not the unwrapped %s"
            % (name, ua.sum().item(), sum(probes)))
        assert ua.sum().item() != _wrap(sum(probes), bits) or sum(probes) == \
            _wrap(sum(probes), bits)
        a = _cpu(name, probes)
        got = a.sum()
        assert str(got.dtype).replace("torch.", "") == "int64", (
            "this build's sum on %s returned %s. Upstream returns int64; a "
            "narrowing here would wrap a reduction upstream does not wrap."
            % (name, got.dtype))
        assert _vals(got) == [sum(probes)], (
            "this build's sum on %s gave %s, not %s"
            % (name, _vals(got), sum(probes)))


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------

# The candle-internal spellings that must never reach a caller. Each was
# measured on this machine on 2026-09-12 as the *entire* message for an
# `int16`/`int32` tensor on `mps`.
_SHADER_SYMBOLS = (
    "badd_i16", "badd_i32", "bmul_i16", "bmul_i32",
    "bsub_i16", "bsub_i32", "bdiv_i16", "bdiv_i32",
    "to_dtype I16", "to_dtype I32",
    "copy_strided I16", "copy_strided I32",
    "matmul doesn't support I16", "matmul doesn't support I32",
    "Error while loading function",
)

_OPS = {
    "add": lambda t: t + t,
    "sub": lambda t: t - t,
    "mul": lambda t: t * t,
    "div": lambda t: t / t,
    "neg": lambda t: -t,
    "sum": lambda t: t.sum(),
    "cumsum": lambda t: t.cumsum(0),
    "matmul": lambda t: t.reshape(2, 2) @ t.reshape(2, 2),
    "to_int64": lambda t: t.to(_C.int64),
    "to_f32": lambda t: t.to(_C.float32),
    "exp": lambda t: t.exp(),
}


def _refusals(name):
    """`{op: message}` for every op in `_OPS` that refuses on `mps`."""
    out = {}
    for op, fn in sorted(_OPS.items()):
        t = _cpu(name, [1, 2, 3, 4]).to("mps")
        try:
            fn(t)
        except BaseException as e:  # noqa: BLE001
            out[op] = str(e)
    return out


def test_int16_and_int32_on_metal_refuse_by_name_rather_than_by_shader_symbol():
    """Grade: **refuses by name**. Every refusal names the dtype, names the
    device, and says why -- and none of them hands back a candle shader symbol.

    This is the deliverable this round is most sure of. A refusal that says
    `badd_i16` sends the reader into candle's Metal sources to discover a fact
    about *this build's dtype support*, which is a fact this build is the one
    that knows.
    """
    if _mps_or_skip("the int16/int32 refusal text") is None:
        return
    for name in _DTYPES:
        got = _refusals(name)
        assert got, (
            "%s on mps refused nothing in %s. Either candle grew the kernels "
            "-- in which case docs/numerics/DTYPEDEV.md section 4.2 closes and "
            "test_dtypedev's frozen mps column is what grades it -- or this "
            "test is asking the wrong ops." % (name, sorted(_OPS)))
        for op, msg in sorted(got.items()):
            first = msg.splitlines()[0]
            for symbol in _SHADER_SYMBOLS:
                assert symbol not in msg, (
                    "%s on mps refused %s with a candle shader symbol %r:\n  %s\n"
                    "A caller cannot act on that: it names neither the dtype "
                    "nor the device nor what to do instead."
                    % (name, op, symbol, first))
            assert name in msg, (
                "%s on mps refused %s without naming the dtype:\n  %s"
                % (name, op, first))
            assert "mps" in msg, (
                "%s on mps refused %s without naming the device:\n  %s"
                % (name, op, first))


def test_the_named_refusal_covers_the_whole_family():
    """Not `add` only.

    The gap is one fact -- candle instantiates its Metal kernels for
    `f32 f16 bf16 u8 u32 i64` and nothing else -- and it surfaces through
    binary kernels, unary kernels, casts, strided copies and matmul, each with
    its own candle spelling. A refusal installed on the op that happened to be
    measured first would leave the other spellings leaking, which is how the
    `badd_i16` message survived a round that had already read it.
    """
    if _mps_or_skip("the family of int refusals") is None:
        return
    for name in _DTYPES:
        got = _refusals(name)
        # Four different candle *sources* of the same fact: a binary kernel, a
        # unary kernel, a cast and matmul. If the refusal only wraps one of
        # them the others come back in candle's words.
        for op in ("add", "neg", "to_int64", "matmul"):
            assert op in got, (
                "%s on mps did not refuse %s, so this test cannot check that "
                "its refusal is named. If it now computes, the frozen mps "
                "column in test_dtypedev.py is what grades the new cell."
                % (name, op))
            # `candle: ` is the prefix this crate puts on an error it is
            # passing through verbatim, so its absence is the check that the
            # message was reworded rather than relayed. The word "candle" on
            # its own is *expected* in the new text -- naming which backend
            # lacks the kernels is the reason, and a refusal that hid it would
            # be vaguer, not better.
            assert "candle: " not in got[op] and "Metal error" not in got[op], (
                "%s on mps refused %s in candle's words:\n  %s"
                % (name, op, got[op].splitlines()[0]))
            assert "not implemented for %s tensors on the mps device" % name \
                in got[op], (
                "%s on mps refused %s with something other than the named "
                "refusal:\n  %s" % (name, op, got[op].splitlines()[0]))


def test_the_refusal_says_what_to_do_instead():
    """A refusal that names the thing and not the way out is half a refusal.

    Two ways out exist and both are real: `.cpu()`, which computes on the host
    and says so, and `.to(torch.int64)` *before* the move, which computes on
    the GPU because `int64` is one of the six dtypes candle instantiates. The
    message has to carry both, because they are not interchangeable -- one
    keeps the dtype and loses the device, the other keeps the device and
    changes the dtype.
    """
    if _mps_or_skip("the remedy in the refusal") is None:
        return
    for name in _DTYPES:
        for op, msg in sorted(_refusals(name).items()):
            assert ".cpu()" in msg, (
                "%s on mps refused %s without offering the host road:\n  %s"
                % (name, op, msg.splitlines()[0]))
            assert "int64" in msg, (
                "%s on mps refused %s without offering the int64 road, which "
                "is the one that keeps the computation on the GPU:\n  %s"
                % (name, op, msg.splitlines()[0]))


def test_the_int64_road_named_in_the_refusal_actually_works():
    """The remedy is checked, not just spelled.

    A refusal that recommends a road nobody walked is worse than one that
    recommends nothing, and this project has shipped one of those before. So
    `x.to(int64).to("mps")` is run here, for the four operators the ask names
    that `int64` reaches, and compared against upstream.
    """
    if _mps_or_skip("the int64 remedy") is None:
        return
    if _upstream_torch is None:
        print("   (skipped: no upstream torch in this interpreter)")
        return
    probes = [1, 2, 3, 4]
    for name in _DTYPES:
        on_mps = _cpu(name, probes).to(_C.int64).to("mps")
        ua = _upstream_torch.tensor(probes, dtype=_upstream_torch.int64)
        for op, fn, uref in (
            ("add", lambda t: t + t, lambda t: t + t),
            ("mul", lambda t: t * t, lambda t: t * t),
            ("neg", lambda t: -t, lambda t: -t),
            ("sum", lambda t: t.sum(), lambda t: t.sum()),
        ):
            got = fn(on_mps)
            assert str(got.device).startswith("mps"), (
                "%s via the int64 road landed on %s, not mps -- the remedy "
                "moved the computation off the device it promised to keep it "
                "on." % (op, got.device))
            want = [int(v) for v in uref(ua).flatten().tolist()]
            assert _vals(got) == want, (
                "the int64 road for %s (from %s) gave %s, upstream %s"
                % (op, name, _vals(got), want))


def test_the_escape_hatch_off_a_sealed_int_metal_tensor_stays_open():
    """`float64` on Metal was gated at construction (docs/numerics/DTYPEDEV.md
    section 4.1) because its escape hatch was closed too: the object could be
    made and could not be converted back. `int16`/`int32` are **not** that
    case, and this test is why the same gate was not applied to them.

    `.cpu()` is a device move, not a cast, so it needs no Metal kernel and
    works. An `int16` tensor on `mps` is therefore storage that can be put down
    and picked back up -- useful as staging even though nothing can compute on
    it -- and gating construction would remove a capability to punish a
    missing one.
    """
    if _mps_or_skip("the escape hatch") is None:
        return
    probes = [1, 2, 3, 4]
    for name in _DTYPES:
        t = _cpu(name, probes).to("mps")
        assert str(t.device).startswith("mps"), t.device
        back = t.cpu()
        assert str(back.device) == "cpu", back.device
        assert _vals(back) == probes, (
            "%s made the round trip to mps and back as %s, not %s"
            % (name, _vals(back), probes))
        assert str(back.dtype).replace("torch.", "") == name, back.dtype
        # The two operators the frozen mps column records as reaching. They are
        # buffer moves rather than kernels, which is why they survive.
        assert _vals(t.clone().cpu()) == probes
        assert _vals(t.reshape(2, 2)[0].cpu()) == probes[:2]


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
