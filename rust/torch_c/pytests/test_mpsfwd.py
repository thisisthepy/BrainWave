"""A transformer forward on `mps`, and the four kernels that were in its way.

docs/RELEASE_0_0_13a0.md §5 lists "no transformer forwards on `mps`" as a gap.
docs/MPS.md named the shape of it: the ops whose kernels in `aten.rs` read a
dispatched tensor back to host memory are refused on `mps` rather than allowed
to return a correct value the GPU did not compute -- and a transformer's path
went through some of them.

**Which ones was measured rather than guessed** (docs/MPSFWD.md §2): the gate
was temporarily made to log-and-allow, SmolLM2-135M was run on `mps`, and
exactly three refusals fired -- `aten.pow.Tensor_Scalar` (RMSNorm's
`x.pow(2)`), `aten.neg.default` (`rotate_half`, twice a layer) and
`aten.cumsum.default` (the attention mask). Not `aten._softmax.default`, which
docs/MPS.md expected to be the one: SmolLM2 goes through
`F.scaled_dot_product_attention`, whose kernel writes its softmax out from
candle ops (docs/SEQLEN.md §7.4 recorded that and it still holds).

All three were rewritten to stay on the device and are off
`MPS_HOST_READBACK_OPS`. This file holds that down from both sides:

  * the values, on a live Metal device, against the same call on the CPU;
  * the *source*, so a machine with no Metal still fails if a readback comes
    back into one of those kernels -- which is the direction this can decay,
    because the runners that cannot exercise the gate are the ones most likely
    to change a kernel under it.

Everything that needs Metal skips by name and says why, on the runners that
have none.
"""

import os
import re

from test_shim import _C, _MPS_READBACK_MARKERS, _MPS_READBACK_HELPERS
from test_shim import _aten_rs_functions, _aten_dispatch_targets


# The three that left the refusal list by being rewritten, plus `prims.neg`,
# which is `neg_default` under another name and so left with it.
MOVED_ONTO_THE_DEVICE = (
    "aten.neg.default",
    "prims.neg.default",
    "aten.pow.Tensor_Scalar",
    "aten.cumsum.default",
)


def _mps_or_skip(what):
    """A live `mps` device, or None having said why not.

    Same shape as `test_shim._mps_or_skip` and deliberately not imported from
    it: the skip line is the thing this file promises to keep truthful
    (docs/VULKAN3.md §6.1 is what a lying skip line cost), so it says which
    file skipped.
    """
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except NotImplementedError as e:
        print(f"   (skipped {what}: no mps device on this machine -- "
              f"{str(e).splitlines()[0]})")
        return None
    return _C.device("mps")


def _f32(values, shape):
    return _C._tensor_from_flat([float(v) for v in values], shape, _C.float32)


def _i64(values, shape):
    return _C._tensor_from_flat([float(v) for v in values], shape, _C.int64)


def _flat(t):
    out = []
    stack = [t.tolist()]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# The source claim. Runs everywhere, Metal or no Metal.
# ---------------------------------------------------------------------------


def test_the_ops_that_left_the_refusal_list_no_longer_read_back():
    """They came off the list by being rewritten, not by being excused.

    `test_shim.py::test_the_mps_readback_list_is_what_the_kernels_actually_do`
    already re-derives the whole list, so this does not re-check the list. It
    checks the *four names this round moved*, individually, because a set
    comparison reports "they agree" whichever way a future edit moves both
    sides together -- and the failure mode being guarded here is somebody
    putting a `to_vec1` back into one of these kernels and taking the op off
    the list to match.

    A readback anywhere in the reachable body of these four fails here even on
    a machine with no Metal device, which is the point: the gate cannot be
    exercised on those machines, so the source claim has to be checkable there.
    """
    parsed = _aten_rs_functions()
    if parsed is None:
        print("   (skipped: rust/torch_c/src/aten.rs is not beside this file "
              "-- installed rather than in-tree)")
        return
    bodies, text = parsed
    ops = _aten_dispatch_targets(text)
    refused = set(_C._shim_mps_host_readback_ops())

    for op in MOVED_ONTO_THE_DEVICE:
        assert op in ops, (op, "not routed by aten_dispatch at all")
        assert op not in refused, (
            f"{op} is refused on mps again -- if its kernel went back to "
            "reading the tensor to the host that is the right call, but "
            "docs/MPSFWD.md §3 says it does not"
        )
        body = bodies.get(ops[op], "")
        assert not _MPS_READBACK_MARKERS.search(body), (
            op, ops[op], "reads device bytes to the host again"
        )
        for helper in _MPS_READBACK_HELPERS:
            assert not re.search(r"\b" + helper + r"\s*\(", body), (
                op, ops[op], f"calls {helper}, which reads back"
            )


def test_the_refusal_list_did_not_grow_a_way_around_itself():
    """The gate is a list of *refusals*; it has no exemptions this round.

    `MPS_READBACK_BUT_ALLOWED` is the one place an op can hold a readback and
    still be dispatched on `mps`, and docs/MPS.md §3.3 argues for its two
    entries one at a time. Nothing this round belongs there -- the four ops
    above were rewritten instead -- so the check is that it is still exactly
    those two.

    Without this, "moved onto the device" and "excused" look identical from
    outside: both end with the op dispatching on mps.
    """
    assert sorted(_C._shim_mps_readback_but_allowed()) == [
        "aten._local_scalar_dense.default",
        "aten.uniform_.default",
    ], _C._shim_mps_readback_but_allowed()
    for op in MOVED_ONTO_THE_DEVICE:
        assert op not in _C._shim_mps_readback_but_allowed(), op


# ---------------------------------------------------------------------------
# The values, on a live Metal device.
# ---------------------------------------------------------------------------


def test_neg_on_mps_is_bit_identical_to_cpu_including_the_wrap():
    """`rotate_half`'s op, on the integral path that used to read back.

    `int64` negation is exact at every input, so this is equality and not a
    tolerance -- and `int64` min is included because that is the one value
    where the rewrite could have differed: the host loop used `wrapping_neg`
    and `0 - x` has to wrap the same way.
    """
    mps = _mps_or_skip("neg on mps")
    if mps is None:
        return
    d = _C._aten_dispatch
    x = _i64([0, 1, -1, 7, -9], [5])
    x_g = d("aten._to_copy.default", x, device=mps)
    got = d("aten.neg.default", x_g)
    assert got.device.type == "mps", got.device
    assert got.cpu().tolist() == d("aten.neg.default", x).tolist()
    assert got.cpu().tolist() == [0, -1, 1, -7, 9], got.cpu().tolist()

    # The wrap. `-(-2**63)` is `-2**63` on both sides.
    lo = _C._aten_dispatch(
        "aten.full.default", [1], -(2 ** 63), dtype=_C.int64
    )
    lo_g = d("aten._to_copy.default", lo, device=mps)
    assert d("aten.neg.default", lo_g).cpu().tolist() == \
        d("aten.neg.default", lo).tolist()

    # And the float path, which was always on the GPU, still is.
    f = _f32([1.5, -2.25, 0.0], [3])
    f_g = d("aten._to_copy.default", f, device=mps)
    assert d("aten.neg.default", f_g).cpu().tolist() == \
        d("aten.neg.default", f).tolist()


def test_pow_square_on_mps_is_bit_identical_to_cpu():
    """RMSNorm's op. `x.pow(2)` is `x * x`, which the GPU can do exactly.

    Equality rather than a tolerance is a claim about *this exponent*: for
    `2.0` the kernel multiplies, and IEEE multiplication is the correctly
    rounded exact product on any conforming device. A general exponent is a
    different matter and the next test is about that.
    """
    mps = _mps_or_skip("pow on mps")
    if mps is None:
        return
    d = _C._aten_dispatch
    x = _f32([1.5, -2.25, 0.0, 3.125, 1e-8, 1e8], [6])
    x_g = d("aten._to_copy.default", x, device=mps)
    got = d("aten.pow.Tensor_Scalar", x_g, 2)
    assert got.device.type == "mps", got.device
    assert got.cpu().tolist() == d("aten.pow.Tensor_Scalar", x, 2).tolist()

    # The integral path is exponentiation by squaring in int64, on the device.
    n = _i64([0, 1, 2, 3, -4], [5])
    n_g = d("aten._to_copy.default", n, device=mps)
    for e in (0, 1, 3, 5):
        assert d("aten.pow.Tensor_Scalar", n_g, e).cpu().tolist() == \
            d("aten.pow.Tensor_Scalar", n, e).tolist(), e


def test_pow_with_a_general_exponent_on_mps_refuses_rather_than_lowering_precision():
    """The other half of the pow rewrite, and it is a refusal on purpose.

    The float path computes in `f64` -- candle's `F64` `powf` arm is
    `f64::powf`, the same call the host loop made, which is why the CPU answer
    did not move. Metal has no `f64` at all, so a non-square exponent raises
    there instead of computing in `f32` and handing back numbers the CPU would
    not have produced.

    That is the same choice docs/MPS.md made at the door, one layer down: a
    loud failure rather than a quiet difference. Asserted so that a later
    "just use f32 on Metal" is a test change and not a silent one.
    """
    mps = _mps_or_skip("pow precision on mps")
    if mps is None:
        return
    d = _C._aten_dispatch
    x_g = d("aten._to_copy.default", _f32([4.0, 9.0], [2]), device=mps)
    try:
        d("aten.pow.Tensor_Scalar", x_g, 0.5)
    except RuntimeError as e:
        assert "F64" in str(e) or "f64" in str(e), str(e)
    else:
        raise AssertionError(
            "pow with a non-square exponent must not quietly compute in f32 "
            "on mps -- see docs/MPSFWD.md §3"
        )


def test_cumsum_on_mps_is_bit_identical_to_cpu():
    """The attention mask's op, rewritten as `n - 1` narrow/add pairs.

    The integral case is exact. The order is the host loop's order, so this is
    equality; a parallel scan would have been a different summation order and
    would have had to be argued for rather than asserted.
    """
    mps = _mps_or_skip("cumsum on mps")
    if mps is None:
        return
    d = _C._aten_dispatch
    x = _i64([1, 1, 1, 1, 0, 1, 1, 1], [2, 4])
    x_g = d("aten._to_copy.default", x, device=mps)
    for dim in (0, 1):
        got = d("aten.cumsum.default", x_g, dim)
        assert got.device.type == "mps", got.device
        assert got.cpu().tolist() == d("aten.cumsum.default", x, dim).tolist(), dim
    assert d("aten.cumsum.default", x_g, 1).cpu().tolist() == \
        [[1, 2, 3, 4], [0, 1, 2, 3]]


def test_all_on_mps_reduces_without_widening_the_mask_to_f64():
    """`aten.all.default` was never refused -- it *died* on the device.

    `any_from` widened its input to `f64` to compare it against zero, and Metal
    has no `U8 -> F64` conversion, so a bool mask reduced with `.all()` raised
    `Metal contiguous to_dtype U8 F64 not implemented`. That is where a
    SmolLM2 forward stopped after the refused kernels were off its path.

    An integral input is compared in its own dtype now, which is the same
    answer -- `x != 0` does not depend on the width it is asked in.
    """
    mps = _mps_or_skip("all on mps")
    if mps is None:
        return
    d = _C._aten_dispatch
    for values, expected in (([1, 1, 1], True), ([1, 0, 1], False)):
        x = _C._aten_dispatch(
            "aten._to_copy.default", _i64(values, [3]), dtype=_C.bool
        )
        x_g = d("aten._to_copy.default", x, device=mps)
        got = d("aten.all.default", x_g)
        assert got.device.type == "mps", got.device
        assert bool(got.cpu().tolist()) is expected, (values, got.cpu().tolist())
        assert got.cpu().tolist() == d("aten.all.default", x).tolist()


def test_sdpa_on_mps_agrees_with_cpu():
    """The three `cpu_fwd`-only custom ops on SDPA's path, from the caller.

    `transposed_contiguous`, `scale_and_causal_mask` and `amax_keepdim` are
    `CustomOp1`s with a CPU kernel and no Metal one, so candle answered `no
    metal implementation for torch._C shim: ...` for each in turn -- three
    walls after the refused kernels were cleared. Each has a fallback on Metal
    that is documented as bit-identical to it on the CPU (docs/MPSFWD.md §4).

    A tolerance and not equality, and the reason is the matmuls either side of
    the softmax rather than any of the three: a GPU GEMM sums in a different
    order. `1e-5` on inputs of this size is that reassociation; a wrong mask or
    a wrong transpose is not a small number.
    """
    mps = _mps_or_skip("sdpa on mps")
    if mps is None:
        return
    d = _C._aten_dispatch
    b, h, s, e = 1, 2, 5, 4
    n = b * h * s * e
    q = _f32([((i * 37) % 19) / 8.0 - 1.0 for i in range(n)], [b, h, s, e])
    k = _f32([((i * 53) % 23) / 8.0 - 1.0 for i in range(n)], [b, h, s, e])
    v = _f32([((i * 71) % 17) / 8.0 - 1.0 for i in range(n)], [b, h, s, e])
    args = dict(dropout_p=0.0, is_causal=True)
    want = d("aten._scaled_dot_product_flash_attention_for_cpu.default",
             q, k, v, **args)[0]
    got = d("aten._scaled_dot_product_flash_attention_for_cpu.default",
            *[d("aten._to_copy.default", t, device=mps) for t in (q, k, v)],
            **args)[0]
    assert got.device.type == "mps", got.device
    a, c = _flat(want), _flat(got.cpu())
    assert len(a) == len(c) and a, (len(a), len(c))
    worst = max(abs(x - y) for x, y in zip(a, c))
    assert worst < 1e-5, (worst, a, c)


def test_a_transformer_block_forwards_on_mps_and_agrees_with_cpu():
    """The gap in docs/RELEASE_0_0_13a0.md §5, at a size that fits in a suite.

    Every op SmolLM2's decoder layer reaches is here in the order it reaches
    them -- RMSNorm (`pow(2)`, `mean`, `rsqrt`, `mul`), the rotary halves
    (`neg`, `cat`), SDPA, and the gated MLP (`silu`, `mul`) -- run twice from
    the same weights, once on the CPU and once on `mps`, and compared.

    It is not a substitute for the real checkpoint: docs/MPSFWD.md §5 has
    SmolLM2-135M itself, 393,216 logits, agreeing to 5.3e-6 relative. It is
    the part of that which can run without a 135M-parameter download, and it
    fails for the same reasons.
    """
    mps = _mps_or_skip("a transformer block on mps")
    if mps is None:
        return
    d = _C._aten_dispatch

    b, h, s, e = 1, 2, 6, 4
    n = b * h * s * e
    x = _f32([((i * 31) % 13) / 16.0 - 0.4 for i in range(n)], [b, h, s, e])
    w = _f32([1.0 + ((i * 7) % 5) / 32.0 for i in range(e)], [e])

    def block(x, w):
        # RMSNorm.
        sq = d("aten.pow.Tensor_Scalar", x, 2)
        mean = d("aten.mean.dim", sq, [-1], True)
        norm = d("aten.mul.Tensor", x, d("aten.rsqrt.default",
                                        d("aten.add.Scalar", mean, 1e-5)))
        norm = d("aten.mul.Tensor", norm, w)
        # rotate_half: (-x2, x1).
        half = e // 2
        x1 = d("aten.slice.Tensor", norm, -1, 0, half)
        x2 = d("aten.slice.Tensor", norm, -1, half, e)
        rot = d("aten.cat.default", [d("aten.neg.default", x2), x1], -1)
        q = d("aten.add.Tensor", norm, d("aten.mul.Scalar", rot, 0.5))
        # Attention.
        att = d("aten._scaled_dot_product_flash_attention_for_cpu.default",
                q, norm, norm, dropout_p=0.0, is_causal=True)[0]
        h1 = d("aten.add.Tensor", x, att)
        # Gated MLP.
        gate = d("aten.silu.default", h1)
        return d("aten.add.Tensor", h1, d("aten.mul.Tensor", gate, h1))

    want = block(x, w)
    got = block(d("aten._to_copy.default", x, device=mps),
                d("aten._to_copy.default", w, device=mps))
    assert got.device.type == "mps", got.device
    a, c = _flat(want), _flat(got.cpu())
    assert len(a) == len(c) and a, (len(a), len(c))
    worst = max(abs(p - q_) for p, q_ in zip(a, c))
    scale = max(abs(p) for p in a) or 1.0
    assert worst / scale < 1e-5, (worst, scale, a, c)


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
