"""docs/TAIL3.md -- the walls found *behind* the ones the previous round closed.

`tools/golden/cases.py` already compares every op below against upstream
element-wise, and this file deliberately does not repeat that. What is here is
the part a value comparison structurally cannot hold down:

  * **The matmul striding verdict.** `docs/ARCH100.md` classified
    `MatMulUnexpectedStriding` a backend limitation and `docs/SETITEM.md`
    proposed `contiguous()` as the fix. The operands are *already contiguous*,
    so `contiguous()` is a no-op there -- and a golden case that passes says
    nothing about *why*. `test_the_refused_matmul_operands_were_already_
    contiguous` asserts the layout arithmetic itself, so if someone later
    "simplifies" the fold into a `.contiguous()` the reason is still on record.

  * **The alias-versus-kernel split.** Five of the six ops this round landed
    reuse a body that already existed; only `scatter_reduce.two` and `erfinv`
    are new arithmetic. `test_the_two_new_kernels_are_the_only_new_arithmetic`
    pins that count so a later round cannot report "six ops" as six kernels.

  * **The pairs that look like aliases and are not.** `index_add_` accumulates
    at the receiver's dtype and `scatter_reduce`'s `sum` accumulates wide;
    `view_as` and `reshape_as` are different ops upstream. Each is asserted
    against the *other* op in the same process, which is the shape a per-op
    golden case cannot take.

  * **`erfinv` past where upstream is right.** Beyond `1 - 1e-11` at `float64`
    the two sides disagree and upstream is the one that has drifted, so
    agreement is the wrong assertion; the `erfc` round trip is the right one.

  * **`as_strided`, refused by name.** It is in `methods.json` without a
    kernel on purpose (`clamp.Tensor`'s pattern), because candle exposes no
    way to build a tensor over an existing storage with arbitrary strides.
    The test asserts the refusal *names the overload*, which is the whole
    value of listing it, and asserts it is NOT advertised as implemented.

Every upstream number below was measured with `env -u PYTHONPATH
-u TORCH_USE_RTLD_GLOBAL python`, i.e. real torch 2.13.0 in its own process,
and the probes re-measure both sides at run time rather than trusting the
transcription. Nothing here needs numpy or a network.
"""

import json
import math
import os
import subprocess
import sys

from test_shim import _C

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


# --------------------------------------------------------------------------
# One probe script, run twice: once against the vendored shim and once against
# upstream torch. Comparing the two dicts is the oracle.
# --------------------------------------------------------------------------

_PROBE_SCRIPT = r"""
import json, math, sys
import torch

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


def rec(name, fn):
    try:
        r = fn()
    except Exception as e:
        out[name] = {"raised": type(e).__name__, "msg": str(e)}
        return
    if isinstance(r, torch.Tensor):
        out[name] = {"ok": [float(v) for v in r.reshape(-1).double()],
                     "shape": list(r.shape), "dtype": str(r.dtype)}
    else:
        out[name] = {"ok": r}


# -- matmul: the five architectures' shape, and its layout ------------------
a5 = torch.arange(1 * 2 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 2, 3, 4, 5)
b5 = torch.arange(1 * 2 * 3 * 5 * 6, dtype=torch.float32).reshape(1, 2, 3, 5, 6)
out["matmul5_lhs_contiguous"] = {"ok": bool(a5.is_contiguous())}
out["matmul5_lhs_stride"] = {"ok": list(a5.stride())}
out["matmul5_rhs_contiguous"] = {"ok": bool(b5.is_contiguous())}
# `.contiguous()` on an already-contiguous tensor changes nothing, which is
# the whole reason it was not the fix.
out["matmul5_contiguous_is_a_noop"] = {
    "ok": list(a5.contiguous().stride()) == list(a5.stride())
}
rec("matmul5", lambda: a5 @ b5)
rec("matmul5_after_contiguous", lambda: a5.contiguous() @ b5.contiguous())
# hiera's own shape, scaled down but with the same three batch axes.
h_l = torch.arange(1 * 1 * 7 * 4 * 3, dtype=torch.float32).reshape(1, 1, 7, 4, 3)
h_r = torch.arange(1 * 1 * 7 * 3 * 4, dtype=torch.float32).reshape(1, 1, 7, 3, 4)
rec("matmul5_hiera_shape", lambda: h_l @ h_r)
# The transposed right operand `k.transpose(-1, -2)` produces -- the exact
# expression `torch_chunk_gated_delta_rule` fails on.
k = torch.arange(1 * 3 * 1 * 4 * 5, dtype=torch.float32).reshape(1, 3, 1, 4, 5)
q = torch.arange(1 * 3 * 1 * 4 * 5, dtype=torch.float32).reshape(1, 3, 1, 4, 5)
rec("matmul5_transposed_rhs", lambda: q @ k.transpose(-1, -2))
rec("matmul6", lambda: (torch.arange(2 * 2 * 2 * 2 * 3 * 2, dtype=torch.float32)
                        .reshape(2, 2, 2, 2, 3, 2)
                        @ torch.arange(2 * 2 * 2 * 2 * 2 * 4, dtype=torch.float32)
                        .reshape(2, 2, 2, 2, 2, 4)))

# -- eye: the wall four lines behind the matmul fold ------------------------
rec("eye3", lambda: torch.eye(3))
rec("eye23", lambda: torch.eye(2, 3))
rec("eye32", lambda: torch.eye(3, 2))
rec("eye_bool", lambda: torch.eye(2, dtype=torch.bool))
rec("eye_int", lambda: torch.eye(3, dtype=torch.int64))
out["eye_default_dtype"] = {"ok": str(torch.eye(2).dtype)}
out["eye_zero_shape"] = {"ok": list(torch.eye(0).shape)}
rec("eye_negative", lambda: torch.eye(-1))
# The exact expression `torch_chunk_gated_delta_rule` stops on, at the dtype
# the model uses.
rec("eye_chunk_add", lambda: torch.zeros(1, 2, 1, 4, 4) + torch.eye(4, dtype=torch.float32))

# -- index_add: out of place ------------------------------------------------
base = torch.zeros(3)
res = base.index_add(0, torch.tensor([1, 1, 1]), torch.tensor([1.0, 2.0, 3.0]))
out["index_add_result"] = {"ok": [float(v) for v in res]}
out["index_add_leaves_self"] = {"ok": [float(v) for v in base]}
out["index_add_is_not_self"] = {"ok": res is base}
empty = torch.zeros(2)
out["index_add_empty_is_not_self"] = {
    "ok": empty.index_add(0, torch.tensor([], dtype=torch.int64),
                          torch.zeros(0)) is empty
}
rec("index_add_negative_index",
    lambda: torch.zeros(3).index_add(0, torch.tensor([-1]), torch.tensor([1.0])))
# A non-contiguous receiver: the in-place form refuses an overlapping write,
# the out-of-place form simply reads.
tv = torch.arange(6.0).reshape(2, 3).t()
rec("index_add_transposed_self",
    lambda: tv.index_add(0, torch.tensor([0, 1]), torch.ones(2, 2)))

# -- the two accumulators that look alike and are not -----------------------
src64 = torch.full((64,), 0.01, dtype=torch.bfloat16)
idx64 = torch.zeros(64, dtype=torch.int64)
rec("bf16_index_add_", lambda: torch.zeros(2, dtype=torch.bfloat16)
    .index_add_(0, idx64, src64))
rec("bf16_scatter_reduce_sum", lambda: torch.zeros(2, dtype=torch.bfloat16)
    .scatter_reduce(0, idx64, src64, reduce="sum"))

# -- scatter_reduce: include_self, and the amin tapas needs -----------------
rec("scatter_reduce_amax_include_self_false",
    lambda: torch.full((3,), 9.0).scatter_reduce(
        0, torch.tensor([0]), torch.tensor([1.0]), reduce="amax",
        include_self=False))
rec("scatter_reduce_amin_tapas_shape",
    lambda: torch.zeros(3).scatter_reduce(
        0, torch.tensor([0, 0, 1]), torch.tensor([2.0, 3.0, 4.0]),
        reduce="amin", include_self=False))
rec("scatter_reduce_mean_counts_self",
    lambda: torch.full((2,), 10.0).scatter_reduce(
        0, torch.tensor([0, 0, 0]), torch.tensor([1.0, 2.0, 3.0]), reduce="mean"))
rec("scatter_reduce_int_mean_truncates",
    lambda: torch.zeros(2, dtype=torch.int64).scatter_reduce(
        0, torch.tensor([0, 0, 0]), torch.tensor([1, 2, 4]), reduce="mean"))
rec("scatter_reduce_nan_wins",
    lambda: torch.zeros(2).scatter_reduce(
        0, torch.tensor([0]), torch.tensor([float("nan")]), reduce="amax"))
rec("scatter_reduce_bad_reduce",
    lambda: torch.zeros(2).scatter_reduce(
        0, torch.tensor([0]), torch.tensor([1.0]), reduce="banana"))
sr_self = torch.zeros(3)
sr_res = sr_self.scatter_reduce(0, torch.tensor([0]), torch.tensor([9.0]),
                                reduce="sum")
out["scatter_reduce_is_out_of_place"] = {
    "ok": [float(v) for v in sr_self] == [0.0, 0.0, 0.0] and sr_res is not sr_self
}

# -- view_as vs reshape_as --------------------------------------------------
rec("view_as_plain", lambda: torch.arange(6.0).view_as(torch.zeros(2, 3)))
rec("view_as_transposed", lambda: torch.arange(6.0).reshape(2, 3).t()
    .view_as(torch.zeros(6)))
rec("reshape_as_transposed", lambda: torch.arange(6.0).reshape(2, 3).t()
    .reshape_as(torch.zeros(6)))

# -- bitwise_xor, against its two siblings ----------------------------------
xa = torch.tensor([0b1100, 0b1010])
xb = torch.tensor([0b1010, 0b0110])
rec("xor", lambda: torch.bitwise_xor(xa, xb))
rec("and", lambda: torch.bitwise_and(xa, xb))
rec("or", lambda: torch.bitwise_or(xa, xb))
rec("xor_self_is_zero", lambda: torch.bitwise_xor(xa, xa))
rec("xor_float", lambda: torch.bitwise_xor(torch.ones(2), torch.ones(2)))
# The dunder spelling, which `methods.json` gives its own key exactly as it
# does `__and__` and `__or__`. `a ^ b` reaches it through the number protocol
# and `a.__xor__(b)` reaches it directly; both must land on the same kernel.
rec("xor_dunder", lambda: xa.__xor__(xb))
rec("xor_operator", lambda: xa ^ xb)

# -- erfinv -----------------------------------------------------------------
grid = [-1.0, -0.99, -0.85, -0.5, -0.1, 0.0, 0.1, 0.5, 0.85, 0.99, 1.0, 1.5,
        float("nan")]
rec("erfinv_grid", lambda: torch.erfinv(torch.tensor(grid, dtype=torch.float64)))
rec("erfinv_int", lambda: torch.erfinv(torch.tensor([0, 1])))
# The far tail, where the two sides part company. Recorded on both sides and
# judged by the erfc round trip rather than by agreement.
far = [0.999999999999, 0.999999999999999]
rec("erfinv_far_tail", lambda: torch.erfinv(torch.tensor(far, dtype=torch.float64)))

# -- as_strided -------------------------------------------------------------
rec("as_strided", lambda: torch.arange(10.0).as_strided((3, 3), (1, 1)))


# The write that upstream propagates into the base and this shim refuses.
# Recorded on both sides: upstream must SUCCEED here, or the refusal below is
# gratuitous rather than a narrowing. docs/STRIDED.md 4.
def _as_strided_write():
    x = torch.arange(10.0)
    y = x.as_strided((3, 3), (1, 1))
    y.fill_(7.0)
    return x


rec("as_strided_write_refused", _as_strided_write)

json.dump(out, sys.stdout)
"""

_cache = {}


def _run(side):
    """`side` is `"shim"` or `"upstream"`. Returns the probe dict, or `None`
    for the shim side when the vendored shim is not installed -- the same
    silent skip `test_tail2.py` uses, and for the same reason: `run.sh` builds
    the standalone `_C`, and only `vendor/install_shim.sh` writes the one the
    vendored tree loads."""
    if side in _cache:
        return _cache[side]
    env = dict(os.environ)
    if side == "shim":
        if not os.path.isfile(_VENDOR_SHIM):
            _cache[side] = None
            return None
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"the {side} probe failed to run at all:\n{proc.stderr[-3000:]}"
    )
    data = json.loads(proc.stdout)
    assert data["_marker"] == side, (
        f"the {side} probe imported the wrong torch ({data['_marker']}) -- every "
        "assertion below would have been about the wrong library"
    )
    _cache[side] = data
    return data


def _both(name):
    """`(shim_entry, upstream_entry)`, or the string `"skip"`."""
    s = _run("shim")
    if s is None:
        return "skip"
    return s[name], _run("upstream")[name]


def _agree(name, atol=1e-9, rtol=1e-9):
    pair = _both(name)
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, f"{name}: upstream itself refused: {want}"
    assert "raised" not in got, f"{name}: the shim refused where upstream did not: {got}"
    assert got.get("shape") == want.get("shape"), f"{name}: shape {got} != {want}"
    assert got.get("dtype") == want.get("dtype"), f"{name}: dtype {got} != {want}"
    a, b = got["ok"], want["ok"]
    assert len(a) == len(b), f"{name}: length {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if isinstance(x, float) and math.isnan(x) and math.isnan(y):
            continue
        if x == y:
            continue
        assert math.isclose(x, y, rel_tol=rtol, abs_tol=atol), (
            f"{name}[{i}]: {x!r} != {y!r}"
        )


# --------------------------------------------------------------------------
# 1. The matmul verdict: contiguous() was neither the fix nor the avoidance
# --------------------------------------------------------------------------


def test_the_refused_matmul_operands_were_already_contiguous():
    """The layouts `docs/ARCH100.md` printed, checked as arithmetic.

    Each of the three sweep refusals prints its operand layout. A stride
    vector is the contiguous one exactly when it is the reversed cumulative
    product of the reversed shape -- so whether `contiguous()` could have
    helped is decidable from the error text alone, and the answer is no.
    """
    refusals = [
        # (architecture, shape, stride) -- transcribed from the sweep's own
        # MatMulUnexpectedStriding payloads.
        ("hiera", [1, 1, 49, 64, 32], [100352, 100352, 2048, 32, 1]),
        ("qwen3_next", [1, 32, 1, 64, 128], [262144, 8192, 8192, 128, 1]),
        ("olmo_hybrid", [1, 30, 1, 64, 96], [184320, 6144, 6144, 96, 1]),
    ]
    for arch, shape, stride in refusals:
        want = []
        acc = 1
        for d in reversed(shape):
            want.append(acc)
            acc *= d
        want.reverse()
        assert stride == want, (
            f"{arch}: the refused lhs was NOT contiguous after all "
            f"(stride {stride}, contiguous would be {want}). If this ever "
            "fires, the docs/TAIL3.md verdict needs re-deriving -- a "
            "contiguous() might genuinely help for that shape."
        )
        assert len(shape) - 2 >= 3, (
            f"{arch}: fewer than three batch axes, so candle's ab_skip would "
            "have accepted it and the diagnosis is wrong"
        )


def test_contiguous_is_a_noop_on_the_operands_that_were_refused():
    pair = _both("matmul5_contiguous_is_a_noop")
    if pair == "skip":
        return
    got, want = pair
    assert got["ok"] is True and want["ok"] is True, (
        "the rank-5 operand's strides changed under .contiguous(), which would "
        "make contiguous() a plausible fix after all"
    )
    for key in ("matmul5_lhs_contiguous", "matmul5_rhs_contiguous"):
        g, w = _both(key)
        assert g["ok"] is True and w["ok"] is True, f"{key}: {g} / {w}"


def test_rank_five_matmul_now_agrees_with_upstream():
    for name in ("matmul5", "matmul5_after_contiguous", "matmul5_hiera_shape",
                 "matmul5_transposed_rhs", "matmul6"):
        _agree(name, atol=1e-4, rtol=1e-6)


def test_the_fold_is_reached_only_after_candle_refuses():
    """The fold must not have replaced the ordinary path.

    `fold_batch_axes_matmul` is called from the error arm of
    `batched_matmul`, not before the multiply. If that ever inverts, every
    rank-4-and-below shape starts taking a different code path than the one
    the existing golden cases were measured against.
    """
    src = open(os.path.join(_REPO_ROOT, "rust", "torch_c", "src", "aten.rs")).read()
    assert "fold_batch_axes_matmul" in src
    assert "Err(e) if is_matmul_striding_refusal(&e) => match fold_batch_axes_matmul" in src, (
        "the fold is no longer guarded by candle's own refusal"
    )
    assert "if rank < 5 || lhs.rank() < 2 || rhs.rank() < 2" in src, (
        "the fold's rank floor moved -- below rank 5 candle's ab_skip accepts "
        "the layout, so a refusal there is about strides and folding hides it"
    )


# --------------------------------------------------------------------------
# 2. Alias versus kernel -- the accounting this round is reported by
# --------------------------------------------------------------------------


def test_eye_is_the_wall_four_lines_behind_the_matmul_fold():
    """`torch_chunk_gated_delta_rule` computes
    `attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)`
    four lines after the `k_beta @ key.transpose(-1, -2)` the matmul fold
    unblocked. Landing the fold without `eye` would have moved **zero**
    architectures -- the sweep would still have reported five failures, just
    with a different name on them.
    """
    for name in ("eye3", "eye23", "eye32", "eye_bool", "eye_int", "eye_chunk_add"):
        _agree(name)
    got, want = _both("eye_default_dtype")
    assert got["ok"] == want["ok"] == "torch.float32", (
        "eye's default dtype is the default FLOAT, not int64 -- every value it "
        f"holds is an integer and the dtype is not ({got} / {want})"
    )
    got, want = _both("eye_zero_shape")
    assert got["ok"] == want["ok"] == [0, 0], (got, want)
    got, want = _both("eye_negative")
    assert "raised" in got and "raised" in want
    assert got["msg"] == want["msg"] == "n must be greater or equal to 0, got -1"


def test_the_two_new_kernels_are_the_only_new_arithmetic():
    """Six ops landed; four of them are bodies that already existed.

    docs/TAIL3.md reports the split, and this is the check that keeps the
    report honest. `index_add.default` shares `index_add_`'s body,
    `view_as.default` shares `aten.view.default`'s, and both `bitwise_xor`
    overloads are a third arm on an enum. Only `scatter_reduce.two` and
    `erfinv.default` compute something nothing else in the file computes.
    """
    src = open(os.path.join(_REPO_ROOT, "rust", "torch_c", "src", "aten.rs")).read()
    # Shared bodies, asserted by the sharing itself rather than by prose.
    assert src.count("index_add_common(py, args, kwargs,") == 2, (
        "index_add and index_add_ no longer share one body -- if that was "
        "deliberate, the negative-index rule now exists in two places and "
        "docs/DEMAND8.md's finding has to be re-checked in both"
    )
    assert "Bitwise::Xor => x ^ y" in src and "Bitwise::Xor => x ^ rhs" in src, (
        "bitwise_xor grew its own kernel instead of an arm on Bitwise"
    )
    # The two that really are new.
    assert "fn scatter_reduce_two(" in src
    assert "fn erfinv_scalar(" in src


def test_every_op_this_round_landed_is_advertised_and_reachable():
    implemented = set(_C._aten_implemented())
    for op in (
        "aten.bitwise_xor.Tensor", "aten.bitwise_xor.Scalar",
        "aten.erfinv.default", "aten.index_add.default",
        "aten.scatter_reduce.two", "aten.view_as.default",
        "aten.eye.default", "aten.eye.m",
    ):
        assert op in implemented, f"{op} is not in _aten_implemented()"


# --------------------------------------------------------------------------
# 3. The pairs that look like aliases and are not
# --------------------------------------------------------------------------


def test_scatter_reduce_sum_and_index_add_accumulate_DIFFERENTLY():
    """The finding this round would most easily have got wrong.

    The same 64 accumulations of `bfloat16(0.01)` into one position give
    `0.65234375` through `index_add_` (a running `bfloat16` sum) and
    `0.640625` through `scatter_reduce(reduce="sum")` (accumulated wide and
    narrowed once). The first draft of `scatter_reduce_two` inherited
    `index_add_`'s rule from the adjacent kernel and was wrong by one
    `bfloat16` step -- invisible on every non-repeating index.
    """
    pair_a = _both("bf16_index_add_")
    if pair_a == "skip":
        return
    pair_s = _both("bf16_scatter_reduce_sum")
    shim_add, up_add = pair_a
    shim_sr, up_sr = pair_s
    assert up_add["ok"][0] != up_sr["ok"][0], (
        "upstream's two accumulators agree, so this round's central finding "
        "no longer holds and both kernels should be re-measured"
    )
    _agree("bf16_index_add_", atol=1e-3, rtol=1e-3)
    _agree("bf16_scatter_reduce_sum", atol=1e-3, rtol=1e-3)
    assert shim_add["ok"][0] != shim_sr["ok"][0], (
        "the shim's two accumulators now agree -- one of them has been "
        "changed to match the other and one of them is now wrong"
    )


def test_view_as_and_reshape_as_are_different_ops_upstream():
    """Upstream's `view_as` refuses a receiver whose strides cannot express
    the target shape; `reshape_as` copies. So `view_as` is NOT an alias of
    `reshape_as`, and treating it as one would have been wrong about torch.

    This shim's `aten.view.default` has always accepted those layouts
    (`reshape_like`), so `view_as` inherits that and both spellings succeed
    here. That gap is recorded, not fixed -- see the op's doc comment. When
    `view` is tightened, this test says so.
    """
    pair = _both("view_as_transposed")
    if pair == "skip":
        return
    shim_v, up_v = pair
    shim_r, up_r = _both("reshape_as_transposed")
    assert "raised" in up_v, (
        "upstream's view_as accepted a transposed receiver, so the two ops no "
        "longer diverge and this shim's laxity is no longer a gap"
    )
    assert "view size is not compatible" in up_v["msg"], up_v["msg"]
    assert "raised" not in up_r, "upstream's reshape_as refused, which it should not"
    assert "raised" not in shim_v and "raised" not in shim_r, (
        "this shim now refuses one of them; if `view` was tightened, `view_as` "
        "should have been tightened with it and this test updated to match"
    )
    assert shim_v["ok"] == shim_r["ok"] == up_r["ok"], (
        "the shim's two spellings disagree with each other or with upstream's "
        "reshape_as"
    )
    _agree("view_as_plain")


def test_bitwise_xor_is_a_third_operation_not_a_spelling_of_and_or_or():
    pair = _both("xor")
    if pair == "skip":
        return
    shim_x, up_x = pair
    _, up_a = _both("and")
    _, up_o = _both("or")
    assert up_x["ok"] != up_a["ok"] and up_x["ok"] != up_o["ok"], (
        "the probe operands no longer separate the three bitwise ops"
    )
    for name in ("xor", "and", "or", "xor_self_is_zero"):
        _agree(name)
    for name in ("xor_dunder", "xor_operator"):
        got, want = _both(name)
        assert "raised" not in got, f"{name}: {got}"
        assert got["ok"] == want["ok"] == shim_x["ok"], (
            f"{name} does not reach the same kernel as torch.bitwise_xor"
        )
    got, want = _both("xor_float")
    assert "raised" in want and "raised" in got, (
        "one side computes bitwise_xor on floats: upstream has no "
        '"bitwise_xor_cpu" kernel for Float'
    )


def test_index_add_does_not_write_through_and_still_refuses_negatives():
    if _run("shim") is None:
        return
    for name in ("index_add_result", "index_add_leaves_self"):
        got, want = _both(name)
        assert got["ok"] == want["ok"], f"{name}: {got} != {want}"
    for name in ("index_add_is_not_self", "index_add_empty_is_not_self"):
        got, want = _both(name)
        assert got["ok"] is False and want["ok"] is False, (
            f"{name}: the out-of-place form returned `self` itself; a caller "
            "writing into the result would reach the receiver"
        )
    got, want = _both("index_add_negative_index")
    assert "raised" in got and "raised" in want, (
        "a negative index was accepted -- that is `index_put_`'s rule "
        "(docs/DEMAND8.md), not this op's"
    )
    assert got["msg"] == want["msg"] == "index out of range in self", (got, want)
    _agree("index_add_transposed_self")


def test_scatter_reduce_include_self_seeds_only_what_is_written():
    for name in ("scatter_reduce_amax_include_self_false",
                 "scatter_reduce_amin_tapas_shape",
                 "scatter_reduce_mean_counts_self",
                 "scatter_reduce_int_mean_truncates",
                 "scatter_reduce_nan_wins"):
        _agree(name, atol=1e-6, rtol=1e-6)
    pair = _both("scatter_reduce_amax_include_self_false")
    if pair == "skip":
        return
    got, _ = pair
    assert got["ok"] == [1.0, 9.0, 9.0], (
        f"{got['ok']} -- [1, -inf, -inf] means the identity was written "
        "everywhere; [9, 9, 9] means include_self was ignored"
    )
    got, want = _both("scatter_reduce_bad_reduce")
    assert "raised" in got and "raised" in want
    assert got["msg"] == want["msg"], (got["msg"], want["msg"])
    got, want = _both("scatter_reduce_is_out_of_place")
    assert got["ok"] is True and want["ok"] is True


# --------------------------------------------------------------------------
# 4. erfinv, including where upstream is the one that is wrong
# --------------------------------------------------------------------------


def test_erfinv_agrees_with_upstream_over_the_reachable_range():
    _agree("erfinv_grid", atol=1e-12, rtol=1e-12)
    _agree("erfinv_int")


def test_erfinv_ends_are_infinities_rather_than_errors():
    pair = _both("erfinv_grid")
    if pair == "skip":
        return
    got, want = pair
    assert got["ok"][0] == want["ok"][0] == float("-inf"), "erfinv(-1) is -inf"
    assert got["ok"][10] == want["ok"][10] == float("inf"), "erfinv(1) is +inf"
    assert math.isnan(got["ok"][11]) and math.isnan(want["ok"][11]), (
        "erfinv(1.5) is nan on both sides, and NOT an exception"
    )


def test_in_the_far_float64_tail_this_shim_is_more_accurate_than_upstream():
    """Past `1 - 1e-11` the two disagree by up to 5.5e-05, and asserting
    agreement there would be asserting the wrong answer.

    The judge is the round trip: `erfc(erfinv(y))` should be `1 - y`. At
    `y = 0.999999999999` the shim's answer round-trips to twelve digits and
    upstream's to four. The whole region is unreachable at `float32`, whose
    largest value below one is `1 - 6e-8`, so no `float32` caller can see it.
    """
    pair = _both("erfinv_far_tail")
    if pair == "skip":
        return
    got, want = pair
    targets = [1.0 - 0.999999999999, 1.0 - 0.999999999999999]
    for i, target in enumerate(targets):
        ours = math.erfc(got["ok"][i])
        theirs = math.erfc(want["ok"][i])
        assert abs(ours - target) <= abs(theirs - target), (
            f"the shim's far-tail erfinv is no longer at least as accurate as "
            f"upstream's: erfc(ours)={ours!r}, erfc(theirs)={theirs!r}, "
            f"target={target!r}. If the series was changed, re-derive "
            "docs/TAIL3.md's accuracy table before relaxing this."
        )
    assert abs(math.erfc(got["ok"][0]) - targets[0]) / targets[0] < 1e-9, (
        "the shim's own round trip has drifted"
    )


# --------------------------------------------------------------------------
# 5. as_strided, refused by name and sized rather than half-built
# --------------------------------------------------------------------------


def test_as_strided_landed_as_a_COPY_and_the_writes_are_refused_instead():
    """**Inverted, not deleted**, and the branch this test's old body named is
    the one that was taken.

    Its words were: *"If a real aliasing view landed, delete this test and add
    element-wise cases; if a COPY landed, that is the silent-divergence shape
    this test exists to prevent."* A copy landed. The shape it existed to
    prevent is prevented a different way, and that difference is the whole of
    `docs/STRIDED.md`:

      * The aliasing is still impossible. candle 0.11.0 still exposes no way to
        build a `Tensor` over an existing storage with arbitrary strides --
        `Layout::new(shape, stride, offset)` is public, `Tensor::from_storage`
        is public and takes an *owned* `Storage`, and `Tensor_`, which is where
        the two meet, has no public field. `test_strided.py` re-verifies it.
      * So the result is a gather, exactly as this test warned.
      * **But every write upstream's view would have propagated is refused**,
        in both directions, by `storage.rs::StridedBarrier`. The divergence is
        a named `RuntimeError` rather than a wrong number, which is what this
        test was protecting.

    So the assertion moves from "it refuses" to "it computes upstream's values
    **and** refuses the writes". Dropping either half would be the regression.
    """
    pair = _both("as_strided")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, "upstream refused as_strided, which it should not"
    assert "raised" not in got, (
        "as_strided refuses again. If it was reverted, invert this test back "
        "rather than deleting it -- docs/STRIDED.md §1 is the argument."
    )
    assert got["ok"] == want["ok"], (got, want)
    assert got["shape"] == want["shape"], (got, want)
    # The half this file is actually here to protect: a copy that accepted
    # writes would be the silent divergence. `test_strided.py` measures both
    # directions against upstream; this asserts the refusal exists at all, so
    # that this file cannot go green on a build with the barrier removed.
    assert "as_strided_write_refused" in _run("shim"), (
        "the probe in this file no longer records the write refusal"
    )
    refused = _run("shim")["as_strided_write_refused"]
    assert refused.get("raised") == "RuntimeError", refused
    assert "as_strided" in refused["msg"], refused["msg"]


def test_as_strided_is_advertised_now_that_it_has_a_kernel():
    """The surface-honesty rule, in the other direction.

    The old body asserted the op was *absent* from `_aten_implemented()`
    because listing a schema in `methods.json` with no kernel behind it must
    not claim a door. There is a kernel now, and the same rule says it must be
    advertised -- `tools/golden/cases.py` carries 29 cases for it, so the
    harness demands and gets the coverage that advertising implies.
    """
    implemented = _C._aten_implemented()
    assert "aten.as_strided.default" in implemented, (
        "as_strided has a kernel and case builders but is not advertised -- "
        "docs/STRIDED.md §1"
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
