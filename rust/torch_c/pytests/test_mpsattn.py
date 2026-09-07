"""An eager attention block on `mps`, and the four walls that were in its way.

docs/platform/RELEASE_0_1_0b0.md §5 says:

    No transformer forwards on `mps`. `aten._softmax.default` is in the set of
    ops refused there, and every attention block passes through it.

**Both halves of that sentence were re-measured rather than trusted, and each
turned out to be half right** (docs/devices/MPSATTN.md §1):

  * "no transformer forwards on `mps`" was already false when it was written --
    docs/devices/MPSFWD.md ran SmolLM2-135M there, and §5 was not updated.
  * "every attention block passes through `_softmax`" is false for the SDPA
    path, which is what SmolLM2 takes and what docs/devices/MPSFWD.md concluded from.
    It is **true** for an *eager* attention block: a BERT built with
    `attn_implementation="eager"` reaches `aten._softmax.default` twice a
    layer, and that is exactly where it stopped.

So the gap named by §5 was real, and it was reachable only by a model
docs/devices/MPSFWD.md did not run. This file holds the closure down from both sides,
the split docs/devices/MPS.md established:

  * **the source**, so a machine with no Metal device still fails if a readback
    comes back into one of these kernels -- the runners that cannot exercise
    the gate are the ones most likely to change a kernel under it;
  * **the values**, on a live Metal device, against the same call on the CPU,
    and (through a vendored-tree subprocess) a real BERT encoder against
    upstream torch with a tolerance derived from upstream's own float32 error.

Everything that needs Metal skips by name and says why on the runners that
have none.
"""

import json
import os
import re
import subprocess
import sys

from test_shim import _C, _MPS_READBACK_MARKERS, _MPS_READBACK_HELPERS
from test_shim import _aten_rs_functions, _aten_dispatch_targets


# The two ops that left `MPS_HOST_READBACK_OPS` this round, by being rewritten
# out of `read_flat` + a scalar loop and into candle ops on the device.
MOVED_ONTO_THE_DEVICE = (
    "aten._softmax.default",
    "aten._safe_softmax.default",
)


def _mps_or_skip(what):
    """A live `mps` device, or None having said why not.

    Deliberately not imported from `test_shim` or `test_mpsfwd`, for the reason
    docs/devices/VULKAN3.md §6.1 paid for: the skip line is the thing this file
    promises to keep truthful, so it names the file that skipped.
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


def _flat(t):
    out, stack = [], [t.tolist()]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# The source claims. These run everywhere, Metal or no Metal.
# ---------------------------------------------------------------------------


def test_the_softmax_ops_left_the_refusal_list_by_being_rewritten():
    """They came off the gate's list because the readback is gone, not excused.

    `test_shim.py::test_the_mps_readback_list_is_what_the_kernels_actually_do`
    re-derives the whole list already, so this does not re-check the list. It
    checks *the two names this round moved*, individually -- a set comparison
    reports "they agree" whichever way a future edit moves both sides together,
    and the failure being guarded here is somebody putting a `read_flat` back
    into `softmax_default` and taking the op off the list to match.

    **This is also the runtime assertion behind docs/devices/MPSATTN.md §5's claim that
    `_softmax` computed on the GPU.** candle's Metal backend has no silent CPU
    fallback (docs/devices/MPS.md §1.1): every op it cannot do bails. So an `mps`
    tensor's arithmetic can only have happened on the host if a kernel *here*
    read it back, and that is what this asserts is absent -- from the source,
    against the list the loaded artefact is gating on.
    """
    parsed = _aten_rs_functions()
    if parsed is None:
        print("   (skipped mpsattn source claim: rust/torch_c/src/aten.rs is "
              "not beside this file -- installed rather than in-tree)")
        return
    bodies, text = parsed
    ops = _aten_dispatch_targets(text)
    refused = set(_C._shim_mps_host_readback_ops())

    for op in MOVED_ONTO_THE_DEVICE:
        assert op in ops, (op, "not routed by aten_dispatch at all")
        assert op not in refused, (
            f"{op} is refused on mps again -- if its kernel went back to "
            "reading the tensor to the host that is the right call, but "
            "docs/devices/MPSATTN.md §2 says it does not"
        )
        body = bodies.get(ops[op], "")
        assert not _MPS_READBACK_MARKERS.search(body), (
            op, ops[op], "reads device bytes to the host again")
        for helper in _MPS_READBACK_HELPERS:
            assert not re.search(r"\b" + helper + r"\s*\(", body), (
                op, ops[op], f"calls {helper}, which reads back")


def test_the_shared_softmax_reduction_is_candle_ops_and_not_a_scalar_loop():
    """`softmax_on_device` is the one body both ops share, so it is named here.

    The per-op scan above follows helper calls **one level, by name**
    (docs/architectures/VOICE3.md records what that nearly cost for `var`/`std`), and
    `softmax_on_device` is exactly one level below both kernels. A `to_vec1`
    added *inside it* would be invisible to the per-op derivation: the kernels'
    own bodies would still be clean, both ops would stay off the list, and the
    gate would stop covering a readback that had come back.

    `test_shim.py::test_every_host_readback_in_aten_is_classified` is the net
    for that in general; this is the same claim aimed at the function this
    round created, so the failure names it.
    """
    parsed = _aten_rs_functions()
    if parsed is None:
        print("   (skipped mpsattn helper claim: aten.rs is not beside this "
              "file -- installed rather than in-tree)")
        return
    bodies, _ = parsed
    body = bodies.get("softmax_on_device")
    assert body is not None, (
        "softmax_on_device is gone from aten.rs -- if the softmax reduction "
        "moved, this test has to move with it rather than be deleted")
    assert not _MPS_READBACK_MARKERS.search(body), (
        "softmax_on_device reads device bytes to the host; both softmax ops "
        "are off MPS_HOST_READBACK_OPS on the strength of it not doing that")
    for helper in _MPS_READBACK_HELPERS:
        assert not re.search(r"\b" + helper + r"\s*\(", body), helper
    # The ops it is built from, named so that a rewrite has to restate them.
    for candle_op in ("max_keepdim", "broadcast_sub", "exp", "sum_keepdim",
                      "broadcast_div"):
        assert candle_op in body, (candle_op, "no longer in softmax_on_device")


def test_the_refusal_list_shrank_and_grew_no_exemption():
    """Two off the list, and nothing moved into the excused column instead.

    `MPS_READBACK_BUT_ALLOWED` is the one place an op can hold a readback and
    still dispatch on `mps`; docs/devices/MPS.md §3.3 argues for its two entries one at
    a time. Nothing this round belongs there. Without this check, "rewritten
    onto the device" and "quietly excused" look identical from outside.

    `<=` rather than `==` on the count: other rounds legitimately add refusals
    when they add a kernel that reads back, and this test must not be the thing
    that fails when they do.
    """
    refused = _C._shim_mps_host_readback_ops()
    assert sorted(set(refused)) == sorted(refused), "duplicate entries"
    assert set(_C._shim_mps_readback_but_allowed()) == {
        "aten._local_scalar_dense.default",
        "aten.uniform_.default",
    }, "the excused list changed -- docs/devices/MPS.md §3.3 argues for exactly two"
    for op in MOVED_ONTO_THE_DEVICE:
        assert op not in refused


def test_the_metal_matmul_refusal_is_recognised_as_a_striding_refusal():
    """The second wall, and it is a source claim because its arm is one.

    `gemm_with_layout_fallback` retries with contiguous operands when candle
    refuses a layout, and it recognised only the CPU backends' spelling
    (`MatMulUnexpectedStriding`). Metal answers `MetalKernelError::
    MatMulNonContiguous`, which arrives as `Error::Metal`, so the retry never
    ran there -- and `query @ key.transpose(2, 3)` inside an eager attention
    block has a permuted left operand (docs/devices/MPSATTN.md §3).
    """
    parsed = _aten_rs_functions()
    if parsed is None:
        print("   (skipped mpsattn matmul arm claim: aten.rs is not beside "
              "this file -- installed rather than in-tree)")
        return
    bodies, _ = parsed
    body = bodies.get("is_matmul_striding_refusal", "")
    assert "Error::Metal" in body and "Invalid matmul arguments" in body, (
        "the Metal arm of is_matmul_striding_refusal is gone -- without it "
        "gemm_with_layout_fallback's retry does not run on mps")


# ---------------------------------------------------------------------------
# The values, on a live Metal device.
# ---------------------------------------------------------------------------


def test_softmax_on_mps_agrees_with_the_same_call_on_cpu():
    """The op §5 named, run on the device it said refused it.

    Compared with a **tolerance and not equality**, and the boundary is the one
    docs/devices/MPSFWD.md §5 measured: `matmul` and `mul` are bit-identical between the
    backends, but `exp` and float reductions are not -- Metal's transcendental
    kernels differ from the host libm in the last bit and a reduction's
    summation order differs. A softmax is an `exp` and two reductions, so
    equality here would go red on the GPU's `expf` and say nothing about this
    round.
    """
    device = _mps_or_skip("softmax on mps")
    if device is None:
        return
    values = [0.5, -1.25, 3.0, -0.75, 2.5, 0.0, -3.5, 1.75,
              -2.0, 4.25, 1.0, -0.5, 0.25, -4.0, 2.25, -1.0]
    for shape, dim in (([16], 0), ([2, 8], 1), ([2, 8], 0), ([2, 2, 4], -1)):
        host = _f32(values, shape)
        on_gpu = host.to(device)
        want = _flat(_C._aten_dispatch("aten._softmax.default", host, dim, False))
        got_t = _C._aten_dispatch("aten._softmax.default", on_gpu, dim, False)
        assert str(got_t.device).startswith("mps"), got_t.device
        got = _flat(got_t.cpu())
        assert len(got) == len(want)
        for a, b in zip(got, want):
            assert abs(a - b) <= 2e-7, (shape, dim, a, b)
        # A softmax's rows sum to one, checked on the device's own answer.
        # Stated as a count of rows so it cannot degenerate into something
        # always true: `outer * inner` rows, each summing to 1.
        size = shape[dim]
        rows = len(got) // size
        stride = 1
        for extent in shape[dim + 1:] if dim >= 0 else shape[len(shape) + dim + 1:]:
            stride *= extent
        for r in range(rows):
            base = (r // stride) * size * stride + (r % stride)
            total = sum(got[base + j * stride] for j in range(size))
            assert abs(total - 1.0) <= 1e-6, (shape, dim, r, total)


def test_safe_softmax_on_mps_answers_zero_for_an_all_minus_infinity_row():
    """`_safe_softmax`'s only divergence from `_softmax`, on the device.

    docs/devices/MPS.md's aten docs record the measurement: plain `_softmax` answers
    `nan` for a row that is entirely `-inf` (because `-inf - (-inf)` is NaN)
    and `_safe_softmax` answers `0`. That branch used to be an `if` inside a
    scalar loop; on the device it is a `where_cond`, and this is the test that
    it survived the move -- a fully-masked attention row is where it fires.
    """
    device = _mps_or_skip("safe_softmax on mps")
    if device is None:
        return
    inf = float("-inf")
    host = _f32([inf, inf, inf, inf, 0.0, 1.0, 2.0, 3.0], [2, 4])
    got = _flat(_C._aten_dispatch("aten._safe_softmax.default",
                                  host.to(device), -1, None).cpu())
    assert got[:4] == [0.0, 0.0, 0.0, 0.0], got[:4]
    want = _flat(_C._aten_dispatch("aten._safe_softmax.default", host, -1, None))
    for a, b in zip(got[4:], want[4:]):
        assert abs(a - b) <= 2e-7, (a, b)
    # and plain `_softmax` still answers NaN on that row, on the device too.
    nan_row = _flat(_C._aten_dispatch("aten._softmax.default",
                                      host.to(device), -1, False).cpu())
    assert all(v != v for v in nan_row[:4]), nan_row[:4]


def test_softmax_of_a_scalar_and_of_an_empty_tensor_still_answer():
    """The two shapes with no reduction to do, on the CPU as well as on mps.

    Written because **nullifying the guard for these did not fail this file** --
    it failed the golden corpus, nine cases across every float dtype, and
    nothing here noticed (docs/devices/MPSATTN.md §5). Golden is in the gate, so the
    regression was caught; but a reader of this file would have concluded the
    edge was covered here, and it was not.

    candle's reductions refuse both shapes rather than answering
    (`max: dimension index 0 out of range for shape []` and
    `empty tensor for reduce`), which the scalar loop this replaced never had
    to think about -- it folded them into `(outer, n, inner) = (1, 1, 1)`.
    """
    for op, extra in (("aten._softmax.default", (False,)),
                      ("aten._safe_softmax.default", (None,))):
        scalar = _C._tensor_from_flat([2.5], [], _C.float32)
        assert _C._aten_dispatch(op, scalar, -1, *extra).tolist() == 1.0, op
        empty = _C._tensor_from_flat([], [0], _C.float32)
        assert _C._aten_dispatch(op, empty, -1, *extra).tolist() == [], op

    device = _mps_or_skip("scalar and empty softmax on mps")
    if device is None:
        return
    for op, extra in (("aten._softmax.default", (False,)),
                      ("aten._safe_softmax.default", (None,))):
        scalar = _C._tensor_from_flat([2.5], [], _C.float32).to(device)
        got = _C._aten_dispatch(op, scalar, -1, *extra)
        assert str(got.device).startswith("mps"), got.device
        assert got.cpu().tolist() == 1.0, op


def test_a_permuted_matmul_on_mps_no_longer_refuses_its_own_layout():
    """`query @ key.transpose(2, 3)` with a `view`+`permute` left operand.

    This is the exact shape an eager attention block hands `matmul`: the left
    operand's strides are `[256, 8, 32, 1]`, which is what a `[1, 8, 4, 8]`
    view permuted to `[1, 4, 8, 8]` leaves behind. candle's Metal matmul
    refused it and the retry did not recognise the refusal.
    """
    device = _mps_or_skip("permuted matmul on mps")
    if device is None:
        return
    n = 1 * 8 * 4 * 8
    q = _f32([(i % 13) * 0.25 - 1.5 for i in range(n)], [1, 8, 4, 8])
    k = _f32([(i % 7) * 0.5 - 1.0 for i in range(n)], [1, 8, 4, 8])
    def score(a, b):
        a = _C._aten_dispatch("aten.transpose.int", a, 1, 2)
        b = _C._aten_dispatch("aten.transpose.int", b, 1, 2)
        b = _C._aten_dispatch("aten.transpose.int", b, 2, 3)
        return _C._aten_dispatch("aten.matmul.default", a, b)
    want = _flat(score(q, k))
    got_t = score(q.to(device), k.to(device))
    assert str(got_t.device).startswith("mps"), got_t.device
    got = _flat(got_t.cpu())
    assert len(got) == len(want) == 1 * 4 * 8 * 8
    # `matmul` is bit-identical between the backends at this shape
    # (docs/devices/MPSFWD.md §5 measured 331,776/331,776), so this is equality.
    assert got == want


def test_a_float_literal_and_a_scalar_where_reach_the_device_at_all():
    """The third and fourth walls, neither of which is a readback.

    Both are the same mistake in two places: a constant was materialised **by
    the device** in `f64`, and Metal has no `f64`. `torch.tensor([...],
    device="mps")` answered `Metal contiguous to_dtype F64 F32 not implemented`
    and `aten.where.ScalarOther` answered `unsupported const-set f64`. Building
    the constant on the host and moving it is `host_const`'s argument, and it
    is an upload, not a readback -- nothing of any dispatched tensor travels.
    """
    device = _mps_or_skip("host-built constants on mps")
    if device is None:
        return
    literal = _C._tensor_new_from_data([[1.5, -2.5], [0.0, 3.25]],
                                       _C.float32, device)
    assert str(literal.device).startswith("mps"), literal.device
    assert _flat(literal.cpu()) == [1.5, -2.5, 0.0, 3.25]

    # The mask is a bool literal rather than `literal > 0`: `gt.Scalar` widens
    # to `f64` to compare exactly (`compare_common`) and Metal has no `f64`, so
    # it is refused on `mps` for a reason that has nothing to do with this test
    # and is not on an attention block's path (docs/devices/MPSATTN.md §6).
    mask = _C._tensor_new_from_data([[True, False], [False, True]],
                                    _C.bool, device)
    picked = _C._aten_dispatch("aten.where.ScalarOther", mask, literal, -1.0)
    assert str(picked.device).startswith("mps"), picked.device
    assert _flat(picked.cpu()) == [1.5, -1.0, -1.0, 3.25]


# ---------------------------------------------------------------------------
# A real BERT encoder, through the vendored tree, against upstream.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

# Shrunk, for `arch_sweep.py`'s reason: the operators reached are the same and
# the wall time is not.
_BERT_CFG = dict(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                 num_attention_heads=4, intermediate_size=64,
                 max_position_embeddings=32)
_BERT_IDS = [[1, 2, 3, 4, 5, 6, 7, 8]]
# Supplied rather than left to default: `BertEmbeddings` otherwise builds them
# with `torch.gather`, which is still refused on `mps` (docs/devices/MPSATTN.md §6).
# That op is in the *embeddings*, not the attention block.
_BERT_TT = [[0] * 8]

_SHIM_SIDE = r"""
import json, sys
import torch
from transformers import BertConfig, BertModel

cfg = json.load(sys.stdin)
model = BertModel(BertConfig(attn_implementation="eager", **cfg["config"])).eval()
model.load_state_dict({k: torch.as_tensor(v, dtype=torch.float32)
                       for k, v in cfg["weights"].items()})
ids = torch.as_tensor(cfg["ids"], dtype=torch.int64)
tt = torch.as_tensor(cfg["token_type_ids"], dtype=torch.int64)
out = {}
with torch.no_grad():
    out["cpu"] = model(input_ids=ids, token_type_ids=tt).last_hidden_state.tolist()
model = model.to("mps")
with torch.no_grad():
    mps = model(input_ids=ids.to("mps"), token_type_ids=tt.to("mps")).last_hidden_state
out["mps_device"] = str(mps.device)
out["mps"] = mps.cpu().tolist()
# The runtime half of the "it computed on the GPU" claim: every op this forward
# dispatched under an mps label is absent from the gate's table, and the gate
# would have raised before the kernel ran if it were not.
out["refused"] = list(torch._C._shim_mps_host_readback_ops())
json.dump(out, sys.stdout)
"""


def _upstream_bert():
    """Weights, the float32 answer, and the float64 oracle -- from upstream."""
    try:
        import torch as upstream
        from transformers import BertConfig, BertModel
    except Exception as e:  # noqa: BLE001
        print(f"   (skipped the BERT-on-mps agreement: no upstream torch or "
              f"transformers here -- {type(e).__name__})")
        return None
    upstream.manual_seed(0)
    model = BertModel(BertConfig(attn_implementation="eager", **_BERT_CFG)).eval()
    ids = upstream.tensor(_BERT_IDS)
    tt = upstream.tensor(_BERT_TT)
    with upstream.no_grad():
        f32 = model(input_ids=ids, token_type_ids=tt).last_hidden_state
        wide = BertModel(BertConfig(attn_implementation="eager", **_BERT_CFG)).eval()
        wide.load_state_dict(model.state_dict())
        f64 = wide.double()(input_ids=ids, token_type_ids=tt).last_hidden_state
    return (
        {k: v.tolist() for k, v in model.state_dict().items()},
        _flatten_nested(f32.tolist()),
        _flatten_nested(f64.tolist()),
    )


def _flatten_nested(v):
    out, stack = [], [v]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def test_a_bert_encoder_forwards_on_mps_and_agrees_with_upstream():
    """The claim docs/platform/RELEASE_0_1_0b0.md §5 said could not be made.

    A shrunk BERT with `attn_implementation="eager"` -- so its attention block
    really does go through `aten._softmax.default`, twice a layer -- forwards
    on `mps` and is compared **three ways**: against the same shim model on
    `cpu`, against upstream's `float32`, and against upstream's `float64`.

    **The tolerance is derived, not chosen**, by docs/numerics/AGREE.md's rule: upstream
    is run again in `float64` and both `float32` answers are scored against it,
    and a difference is not a defect if it is within **4x upstream's own
    distance from the float64 truth** for the same output. Anything tighter
    would have to call upstream wrong. Measured on this model
    (docs/devices/MPSATTN.md §4): upstream's own float32 error is 4.32e-07, the shim on
    `mps` is 5.26e-07 from the same truth -- 1.22x, not 4x -- and `mps` differs
    from the shim's own `cpu` answer by at most 2.384e-07, which is exactly one
    float32 ulp at that magnitude.

    Runs in a subprocess against the vendored tree, for `test_shim.py`'s
    reason: the shim-backed `torch` and upstream `torch` cannot both be the
    `torch` module in one interpreter.
    """
    produced = _upstream_bert()
    if produced is None:
        return
    weights, up32, up64 = produced
    if not os.path.isfile(_VENDOR_SHIM):
        print("   (skipped the BERT-on-mps agreement: no vendored shim at "
              f"{_VENDOR_SHIM} -- run vendor/install_shim.sh)")
        return
    env = dict(os.environ, PYTHONPATH=_VENDOR_DIR, TORCH_USE_RTLD_GLOBAL="1")
    payload = json.dumps({"config": _BERT_CFG, "weights": weights,
                          "ids": _BERT_IDS, "token_type_ids": _BERT_TT})
    proc = subprocess.run([sys.executable, "-c", _SHIM_SIDE], input=payload,
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        if "not implemented for the mps device" in tail or "no mps" in tail:
            print(f"   (skipped the BERT-on-mps agreement: {tail})")
            return
        raise AssertionError(
            "the shim-side BERT forward failed:\n" + proc.stderr[-3000:])
    got = json.loads(proc.stdout)

    assert got["mps_device"].startswith("mps"), got["mps_device"]
    # The runtime assertion behind "it computed on the GPU": the two softmax
    # ops this forward goes through are not in the table the loaded artefact
    # gates on, so no kernel on the path reads a tensor back -- and candle's
    # Metal backend has no silent CPU fallback to do it instead.
    for op in MOVED_ONTO_THE_DEVICE:
        assert op not in got["refused"], op

    mps = _flatten_nested(got["mps"])
    cpu = _flatten_nested(got["cpu"])
    assert len(mps) == len(cpu) == len(up32) == len(up64)

    worst_upstream = max(abs(a - b) for a, b in zip(up32, up64))
    worst_mps = max(abs(a - b) for a, b in zip(mps, up64))
    worst_cpu = max(abs(a - b) for a, b in zip(cpu, up64))
    worst_backends = max(abs(a - b) for a, b in zip(mps, cpu))
    assert worst_upstream > 0, (
        "upstream's float32 answer is exactly its float64 one, so there is no "
        "error distribution to derive a tolerance from -- pick a harder model "
        "rather than a constant")
    assert worst_mps <= 4 * worst_upstream, (
        f"mps is {worst_mps:.3e} from the float64 truth against upstream's own "
        f"{worst_upstream:.3e}; docs/numerics/AGREE.md's rule allows 4x")
    assert worst_cpu <= 4 * worst_upstream, (worst_cpu, worst_upstream)
    # `mps` against the shim's own `cpu`, which is the backend comparison and
    # is tighter than the upstream one: one float32 ulp at this magnitude.
    ulp = max(abs(v) for v in up64) * 2 ** -23
    assert worst_backends <= 2 * ulp, (worst_backends, ulp)
    print(f"   BERT-eager on mps: upstream f32 err {worst_upstream:.3e}, "
          f"shim cpu {worst_cpu:.3e}, shim mps {worst_mps:.3e}, "
          f"mps-vs-cpu {worst_backends:.3e} ({worst_backends / ulp:.2f} ulp)")


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
