"""`torch.vmap` over the four architectures' mask closures, checked by value.

`docs/kernels/VMAP.md` is the round; this file is the part of it that goes red.

`docs/kernels/COMPLEX.md` §7 sized `torch._C._functorch._vmap_increment_nesting` as the
largest single item on `docs/architectures/ARCH200.md`'s blocked list -- four architectures,
`nemotron3_5_asr`, `nemotron_asr_streaming`, `nemotron_asr_streaming_encoder`
and `t5gemma2` -- and said the one thing that must *not* happen is a cheap
stub, because the value that comes back is used as an attention mask. A wrong
attention mask does not raise. It changes what the model attends to and the
model still runs.

So the bar here is not "a mask of the right shape". It is **the same bits as
upstream torch, computed in a separate process, for the same inputs**, and
that is what `test_the_vmapped_mask_is_bit_identical_to_upstream` asserts.

The four closures are transcribed rather than imported. `transformers` is not a
dependency of this suite, and more importantly the point of the round is what
these *particular* shapes of closure need -- an import would let a
`transformers` upgrade silently change the subject. The transcriptions are
verbatim from transformers 5.x:

    masking_utils.py:48    and_masks
    masking_utils.py:75    causal_mask_function
    masking_utils.py:83    bidirectional_mask_function
    masking_utils.py:168   padding_mask_function
    masking_utils.py:339   _vmap_expansion_sdpa       <-- the nesting under test
    masking_utils.py:352   _non_vmap_expansion_sdpa   <-- the same thing, unbatched
    models/nemotron_asr_streaming/...:870  chunked_limited_mask_function
    models/t5gemma2/...:719                sliding_window_mask_function

Nothing here needs numpy or a network.
"""

import json
import os
import subprocess
import sys

from test_shim import _C  # noqa: F401  (asserts the staged artefact loads)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


# --------------------------------------------------------------------------
# The four closures, and both expansions of them
# --------------------------------------------------------------------------

_CLOSURES = r'''
import torch

def and_masks(*mask_functions):
    def and_mask(b, h, q, kv):
        result = q.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(b, h, q, kv).to(result.device)
        return result
    return and_mask

def causal_mask_function(b, h, q, kv):
    return kv <= q

def bidirectional_mask_function(b, h, q, kv):
    return q >= 0

def padding_mask_function(padding_mask):
    def inner_mask(b, h, q, kv):
        return padding_mask[b, kv]
    return inner_mask

def chunked_limited_mask_function(left_ctx, right_ctx):
    chunk_size = right_ctx + 1
    left_context_chunks = left_ctx // chunk_size if left_ctx >= 0 else 10_000
    def inner_mask(b, h, q, kv):
        q_chunk = torch.div(q, chunk_size, rounding_mode="trunc")
        kv_chunk = torch.div(kv, chunk_size, rounding_mode="trunc")
        chunk_diff = q_chunk - kv_chunk
        return (chunk_diff >= 0) & (chunk_diff <= left_context_chunks)
    return inner_mask

def sliding_window_mask_function(sliding_window, is_causal=True):
    def inner_mask(b, h, q, kv):
        if is_causal:
            left_window_size, right_window_size = sliding_window, 0
        else:
            left_window_size, right_window_size = (
                (sliding_window + 1) // 2, (sliding_window) // 2 + 1)
        dist = q - kv
        left_mask = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask
    return inner_mask

def vmap_expansion(mask_function):
    dimensions = [(None, None, None, 0), (None, None, 0, None),
                  (None, 0, None, None), (0, None, None, None)]
    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function

def broadcast_expansion(b, h, q, kv):
    return (b[:, None, None, None], h[None, :, None, None],
            q[None, None, :, None], kv[None, None, None, :])

B, H, Q, KV = 3, 1, 6, 9
Q_OFFSET = 2

padding = torch.ones(B, KV, dtype=torch.bool)
padding[1, 7:] = False
padding[2, 0] = False

def arange_args():
    return (torch.arange(B), torch.arange(H),
            torch.arange(Q) + Q_OFFSET, torch.arange(KV))

CASES = {
    # nemotron_asr_streaming / nemotron3_5_asr: an `and_mask_function` of
    # chunked-limited context, on top of causal, with the 2-D padding mask
    # that `sdpa_mask` folds in at masking_utils.py:504.
    "chunked_limited": and_masks(
        causal_mask_function,
        chunked_limited_mask_function(4, 1),
        padding_mask_function(padding),
    ),
    # t5gemma2: a non-causal sliding window on top of a bidirectional base.
    "sliding_window_bidirectional": and_masks(
        bidirectional_mask_function,
        sliding_window_mask_function(5, is_causal=False),
        padding_mask_function(padding),
    ),
    # The same, causal -- t5gemma2's decoder side takes this branch.
    "sliding_window_causal": and_masks(
        causal_mask_function,
        sliding_window_mask_function(4, is_causal=True),
    ),
    # No padding mask, so nothing indexes a tensor by a batched index: the
    # closure is pure index arithmetic. Kept separate because it is the one
    # case that would still work if advanced indexing under the level broke.
    "chunked_limited_no_padding": and_masks(
        causal_mask_function,
        chunked_limited_mask_function(-1, 3),
    ),
}
'''


_MASK_SCRIPT = _CLOSURES + r'''
import json, sys

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}
for name, fn in CASES.items():
    entry = {}
    for how in ("vmap", "broadcast"):
        try:
            args = arange_args()
            if how == "vmap":
                m = vmap_expansion(fn)(*args)
            else:
                m = fn(*broadcast_expansion(*args))
                m = m.expand(B, H, Q, KV)
            entry[how] = {
                "shape": list(m.shape),
                "dtype": str(m.dtype),
                "values": [int(v) for v in m.reshape(-1).to(torch.int64).tolist()],
            }
        except Exception as e:
            entry[how] = {"raised": type(e).__name__, "msg": str(e)[:300]}
    out[name] = entry
json.dump(out, sys.stdout)
'''


# Everything that must still refuse. `_vmap_increment_nesting` called by hand
# is in here deliberately: the batched representation is checked against the
# closure it batches, and the closure is only reachable from `_flat_vmap`'s
# frame, so a hand call has nothing to check against and says so.
_REFUSAL_SCRIPT = _CLOSURES + r'''
import json, sys

def leaky(b, h, q, kv):
    """Well-formed on scalars, and reads *across* the batch dimensions.

    On the 0-d tensor `vmap` actually promises, `reshape(-1)[0]` is the
    identity, so this closure is legal and its answer varies with `b` and `q`.
    Under a broadcast representation the reshape flattens every level together
    and `[0]` takes the corner, so the answer is a *constant* -- with the
    right dtype, the right shape, and no error. Nothing about the shape can
    see that, which is why the value check exists.
    """
    return (b + q).reshape(-1)[0] >= 3

def cat_widens_a_level_slot(b, h, q, kv):
    """`cat` along dimension 0, which under this representation is a level.

    The result has 2 where the batch level's size belongs, so the slot half of
    the shape gate is the thing that catches it -- before the value check gets
    a chance, and before `cat` of two 0-d tensors would have raised.
    """
    x = kv <= q
    return torch.cat([x, x], dim=0)

def cat_adds_a_logical_dimension(b, h, q, kv):
    """The same op after a flatten: the levels stay 1 and a logical dimension
    appears past them, which is the tail half of the shape gate."""
    x = (kv <= q).reshape(-1)
    return torch.cat([x, x])

def two_outputs(b, h, q, kv):
    return (kv <= q), (kv >= q)

PROBES = {
    # First, deliberately: a refused `_vmap_increment_nesting` leaves an
    # unpaired decrement behind (the context manager's `finally` runs anyway),
    # and this probe is about the *unprovoked* call.
    "decrement_by_hand":
        lambda: torch._C._functorch._vmap_decrement_nesting(),
    # Never landed, and named separately so a round that lands `grad` does
    # not get to inherit this file's greenness.
    "wrap_for_grad": lambda: torch._C._functorch._wrap_for_grad(torch.ones(2), 0),
    # A hand call, outside `_flat_vmap`.
    "increment_by_hand":
        lambda: torch._C._functorch._vmap_increment_nesting(2, "error"),
    "add_batch_dim_by_hand":
        lambda: torch._C._functorch._add_batch_dim(torch.ones(2, 2), 0, 1),
    # Through `torch.vmap`, but shapes this representation does not hold.
    "in_dims_on_a_2d_tensor":
        lambda: torch.vmap(lambda x: x.sum(), in_dims=0)(torch.ones(3, 4)),
    "out_dims_one":
        lambda: torch.vmap(lambda x, y: x + y, in_dims=(0, None),
                           out_dims=1)(torch.arange(3), torch.arange(4)),
    "randomness_different":
        lambda: torch.vmap(lambda x: x + 1, in_dims=0,
                           randomness="different")(torch.arange(3)),
    "two_outputs": lambda: vmap_expansion(two_outputs)(*arange_args()),
    "cat_widens_a_level_slot":
        lambda: vmap_expansion(cat_widens_a_level_slot)(*arange_args()),
    "cat_adds_a_logical_dimension":
        lambda: vmap_expansion(cat_adds_a_logical_dimension)(*arange_args()),
    # The one the self-check exists for.
    "reads_across_the_batch_dim":
        lambda: vmap_expansion(leaky)(*arange_args()),
}

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}
for name, fn in PROBES.items():
    try:
        r = fn()
    except Exception as e:
        out[name] = {"raised": type(e).__name__, "msg": str(e)[:400]}
    else:
        shape = tuple(getattr(r, "shape", ()))
        out[name] = {"ok": f"{type(r).__name__}{shape}"}

# The dynamic layer stack has to be empty again after all of that: every
# refusal above happens with a level either not yet pushed or already popped
# by `vmap_increment_nesting`'s `finally`. A leaked level would make every
# later tensor operation in the process run under a transform.
out["stack_depth_after"] = torch._C._functorch.get_dynamic_layer_stack_depth()
out["level_after"] = torch._C._functorch.maybe_get_level(torch.ones(2))
json.dump(out, sys.stdout)
'''


def _run(script, shim):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, env=env, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{'shim' if shim else 'upstream'} subprocess exited "
            f"{proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}")
    out = json.loads(proc.stdout)
    want = "shim" if shim else "upstream"
    assert out["_marker"] == want, f"asked for {want}, got {out['_marker']}"
    return out


_CACHE = {}


def _available():
    return os.path.exists(_VENDOR_SHIM)


def _masks(shim):
    key = ("masks", shim)
    if key not in _CACHE:
        _CACHE[key] = _run(_MASK_SCRIPT, shim)
    return _CACHE[key]


def _refusals():
    if "refusals" not in _CACHE:
        _CACHE["refusals"] = _run(_REFUSAL_SCRIPT, True)
    return _CACHE["refusals"]


def _case_names():
    return ["chunked_limited", "sliding_window_bidirectional",
            "sliding_window_causal", "chunked_limited_no_padding"]


# --------------------------------------------------------------------------
# 1. The bar: the same bits as upstream
# --------------------------------------------------------------------------

def test_the_vmapped_mask_is_bit_identical_to_upstream():
    """The whole point of the round.

    `docs/kernels/COMPLEX.md` §7: "a no-op counter is worse than nothing ... it just
    produces a wrong mask and a model that appears to run." So this compares
    the *values*, not the shape and not the dtype alone -- every element of
    every case, against upstream torch in its own process.

    If this file is ever reduced to a shape assertion, the thing it was
    written to prevent is back.
    """
    if not _available():
        return
    shim, upstream = _masks(True), _masks(False)
    for name in _case_names():
        up = upstream[name]["vmap"]
        sh = shim[name]["vmap"]
        assert "raised" not in up, (name, up)
        assert "raised" not in sh, (name, sh)
        assert sh["shape"] == up["shape"], (name, sh["shape"], up["shape"])
        assert sh["dtype"] == up["dtype"], (name, sh["dtype"], up["dtype"])
        assert sh["values"] == up["values"], (
            f"{name}: the vmapped mask differs from upstream in "
            f"{sum(1 for a, b in zip(sh['values'], up['values']) if a != b)} "
            f"of {len(up['values'])} elements")
        # Not vacuous: a mask that is all-True or all-False would compare
        # equal to an all-True or all-False bug.
        assert 0 < sum(up["values"]) < len(up["values"]), (
            f"{name} is a constant mask, so element-wise equality proves "
            f"nothing -- pick different parameters")


def test_vmap_and_the_broadcast_expansion_agree_on_both_sides():
    """The identity the representation rests on, asserted on upstream too.

    `masking_utils.py:352`'s `_non_vmap_expansion_sdpa` is the default path
    and `:339`'s `_vmap_expansion_sdpa` is the `use_vmap=True` one. For an
    index-based closure they are the same function -- that is the claim this
    shim's `vmap` implements. It is checked *on upstream* as well, because if
    it stopped being true there, the shim would be right about the wrong
    thing.
    """
    if not _available():
        return
    for shim in (False, True):
        got = _masks(shim)
        for name in _case_names():
            a, b = got[name]["vmap"], got[name]["broadcast"]
            assert "raised" not in a, (shim, name, a)
            assert "raised" not in b, (shim, name, b)
            assert a["values"] == b["values"], (
                f"{'shim' if shim else 'upstream'}: {name}: vmap and the "
                f"broadcast expansion disagree")


def test_the_padding_mask_is_indexed_by_a_batched_index():
    """Three of the four cases index a real tensor by a batched index.

    `padding_mask_function` is `padding_mask[b_idx, kv_idx]`, and it is folded
    into the closure by `sdpa_mask` itself, not by the model. So it is not
    optional to support: under the level, `b` and `kv` are tensors and this is
    advanced indexing on two axes at once. The fourth case leaves it out so
    that a failure here is attributable.
    """
    if not _available():
        return
    shim = _masks(True)
    padded = shim["chunked_limited"]["vmap"]["values"]
    unpadded = shim["chunked_limited_no_padding"]["vmap"]["values"]
    assert padded != unpadded, (
        "the padding mask made no difference to the result, so nothing here "
        "shows that indexing by a batched index works")
    B, H, Q, KV = 3, 1, 6, 9
    # padding[2, 0] is False, so every query in batch 2 must be masked at kv 0.
    for q in range(Q):
        assert padded[((2 * H) * Q + q) * KV + 0] == 0, (
            "padding_mask[2, 0] is False but the mask lets batch 2 attend to "
            "key 0 -- the batched index did not reach the right row")


# --------------------------------------------------------------------------
# 2. What still refuses, and why each one has to
# --------------------------------------------------------------------------

def test_a_closure_that_reads_across_the_batch_dimension_is_refused():
    """The gate the shape check cannot be.

    A broadcast representation is only `vmap` for closures that are pointwise
    over the mapped indices. `sum(0, keepdim=True)` is the identity on the 0-d
    tensor vmap promises, so the closure is legal -- but under broadcasting
    dimension 0 *is* the batch, and the sum mixes elements while leaving the
    shape untouched. Nothing about the shape can catch it.

    So `_remove_batch_dim` re-runs the closure on sampled index points with
    plain scalars, which is what `vmap` means, and refuses on disagreement.
    Deleting that check turns this test red, which is the point of it.
    """
    if not _available():
        return
    r = _refusals()["reads_across_the_batch_dim"]
    assert "raised" in r, (
        "a closure that sums across the mapped dimension returned an answer. "
        "That answer is wrong, and if it were a mask nothing would say so: "
        f"got {r}")
    assert r["raised"] == "NotImplementedError", r
    assert "disagrees with the closure" in r["msg"], (
        "the closure was refused, but not by the value check -- something "
        f"else raised first, so this test is not exercising it: {r}")


def test_the_shapes_this_representation_cannot_hold_are_refused():
    """Each one is a different way the broadcast identity stops holding.

    * a 2-D `in_dims` input -- the value would have a logical shape, and this
      representation spends every dimension on levels;
    * `out_dims=1` -- the mapped dimension is already at 0 here and moving it
      is a claim about a shape that was never built;
    * `randomness != "error"` -- a statement about how random ops behave under
      the transform, and there are no batching rules to make it true;
    * more than one output -- the level is unwound once;
    * a `cat` along a level dimension, and a `cat` that adds a dimension past
      the levels -- the two halves of the shape gate, which is a precondition
      for the value check rather than a second opinion on it (VMAP.md §4.2).
    """
    if not _available():
        return
    r = _refusals()
    for name in ("in_dims_on_a_2d_tensor", "out_dims_one",
                 "randomness_different", "two_outputs",
                 "cat_widens_a_level_slot", "cat_adds_a_logical_dimension"):
        assert "raised" in r[name], (name, r[name])
        assert r[name]["raised"] == "NotImplementedError", (name, r[name])
        assert "torch._C shim" in r[name]["msg"], (name, r[name])


def test_grad_and_the_hand_calls_still_refuse():
    """vmap landing does not make the rest of functorch land.

    `_wrap_for_grad` is untouched, and the three `_vmap_*` primitives called
    by hand -- outside `torch/_functorch/vmap.py:_flat_vmap` -- refuse, because
    the batched answer is checked against the closure it batches and a hand
    call has no closure to check against. `test_shim.py` and `test_tail2.py`
    assert the same two refusals from the other direction and are *unchanged*
    by this round; that is deliberate, not an oversight.
    """
    if not _available():
        return
    r = _refusals()
    for name in ("wrap_for_grad", "increment_by_hand", "add_batch_dim_by_hand"):
        assert "raised" in r[name], (name, r[name])
        assert r[name]["raised"] == "NotImplementedError", (name, r[name])
    assert r["decrement_by_hand"]["raised"] == "RuntimeError", r["decrement_by_hand"]


def test_no_level_is_left_on_the_stack_by_a_refusal():
    """A leaked level would put every later op in the process under a transform.

    Every refusal above happens either before the level is pushed or inside
    the `with` whose `finally` pops it, so the stack is empty afterwards and
    `maybe_get_level` is back to -1 -- the value `test_shim.py` pins.
    """
    if not _available():
        return
    r = _refusals()
    assert r["stack_depth_after"] == 0, r["stack_depth_after"]
    assert r["level_after"] == -1, r["level_after"]


# --------------------------------------------------------------------------
# 3. This round added no operator
# --------------------------------------------------------------------------

def test_vmap_landed_without_a_kernel_or_a_repr_arm():
    """The negative half of the claim, and the reason golden did not move.

    `docs/kernels/COMPLEX2.md` is the cautionary example: a new `Repr` arm reached 19
    `match` sites and nullifying the central refusal made 6 of 10 sampled ops
    compute silently. A batched dimension looked like the same shape of
    problem and turned out not to be one -- the representation is a plain
    dense tensor with extra size-1 dimensions, so there is no arm, no tag, no
    kernel and no new `aten.*` name. Golden is unmoved at 287 ops for exactly
    that reason, and this asserts the premise rather than the consequence.
    """
    implemented = set(_C._aten_implemented())
    parked = set(_C._aten_implemented_awaiting_golden())
    assert not [op for op in implemented | parked
                if "vmap" in op or "batch_dim" in op], (
        "an aten op named for vmap appeared; this round claims to have added "
        "none, and golden's op count is pinned on that claim")


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
