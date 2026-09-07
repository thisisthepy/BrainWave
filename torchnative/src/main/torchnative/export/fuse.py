"""Fold an inference-mode batch norm into the convolution in front of it.

## Why this is not a decomposition

docs/graph/DECOMP.md §12.5 measured the thing that makes this pass necessary, and
docs/graph/NPU.md §7 confirmed it a second time on a different network: running
upstream's post-autograd table over `mobilenet_v2` **makes it worse**. Ops
outside NNAPI go 3 -> 6 and nodes go 203 -> 1191, because `native_batch_norm`
decomposes into `sqrt`, `reciprocal` and `new_zeros`, and NNAPI has none of the
three. More lowering is not more progress; a rule can point *away* from a
target.

The right treatment for an inference batch norm on an NPU is §12.7's
*target-dependent* category: not to express it in smaller ops but to make it
disappear, by pushing its affine into the weights of the convolution that
produced its input. A convolution is linear, so

    BN(Conv(x, W, B)) = Conv(x, W, B) * alpha + beta
                      = Conv(x, W * alpha, B * alpha + beta)

and the fused convolution is one node where there were two. Nothing is
approximated: the affine is exactly the one the batch-norm kernel applies.

## The arithmetic, and the reason it is quoted rather than derived

docs/architectures/DEMAND1.md §5 records a defect this project already had, found by its own
golden harness. Upstream's inference batch norm applies a **fused** affine:

    alpha = invstd * weight          invstd = rsqrt(running_var + eps)
    beta  = bias - mean * alpha
    out   = x * alpha + beta

and the last line is a cancellation of two large nearly-equal numbers. For a
constant channel with `bias = 0.1` upstream answers `0.0999755859375`, one
float32 ULP grid at magnitude 632 -- not `0.1`. The algebraically identical
`(x - mean) * invstd * weight + bias` **is exact there, and is therefore
wrong**: it disagrees with upstream on precisely the inputs where upstream
loses precision.

So this module uses the fused form, and
`test_the_batch_norm_affine_this_fold_uses_is_upstreams` re-derives `alpha` and
`beta` here and requires `x * alpha + beta` to agree with
`aten.native_batch_norm.default` **bit for bit**, with the obvious form
measured beside it and required to differ. That check fails if either form is
substituted for the other, which is the only reason it is worth having.

Note what that check does and does not establish. It pins the *affine*. It does
not pin the fold, because folding into the weights changes the order of the
arithmetic fundamentally -- the scale now happens inside the convolution's
accumulation instead of after it -- so the fused graph is **not** expected to be
bit-exact against the unfolded one. That claim is measured separately, by
replaying both graphs on real inputs, and docs/graph/REFOLD.md carries the number.

## What it refuses

Silently declining to fuse is safe (the graph still computes the right thing);
fusing when one of these does not hold is not. Each is checked:

* the batch norm must be in **inference** mode (`training=False`). A training
  batch norm normalises by batch statistics, which depend on `x`, and there is
  no constant `alpha` at all.
* `weight`, `bias`, `running_mean` and `running_var` must all be **constants**
  of the trace. A parameter reachable from an input is not known at fold time.
* the producer must be `aten.convolution.default` with a **constant weight**,
  and not `transposed` -- a transposed convolution's weight has the output
  channels on axis 1, so scaling axis 0 would scale the wrong thing.
* the convolution's result must be used **only** by this batch norm and must
  not be a trace output. Otherwise the unfused value is still needed.
* the batch norm's `save_mean` / `save_invstd` outputs (slots 1 and 2) must be
  unused. In eval they are empty (docs/architectures/DEMAND1.md §1), but a graph that reads
  them is asking for something the fused convolution does not produce.
"""

from __future__ import annotations

from typing import Any

from .decompose import (
    DecomposedTrace,
    _as_the_dispatcher_would_call_it,
    _value,
    _walk,
)

__all__ = [
    "FusionRefused",
    "batch_norm_affine",
    "fold_conv_batch_norm",
]

#: The batch-norm spellings this pass recognises, and where `training` sits in
#: each one's positional schema. `_native_batch_norm_legit_no_training` has no
#: `training` argument because its name is the assertion.
_BATCH_NORMS = {
    "aten.native_batch_norm.default": 5,
    "aten._native_batch_norm_legit_no_training.default": None,
}


class FusionRefused(NotImplementedError):
    """The fold cannot be applied here, and this says why."""


def batch_norm_affine(weight, bias, mean, var, eps):
    """`(alpha, beta)` such that `x * alpha + beta` is upstream's eval BN.

    The fused form from docs/architectures/DEMAND1.md §5, not the algebraically identical
    unfused one -- see the module docstring for why the difference is not
    cosmetic. `weight` and `bias` may be `None`, which is how a
    `BatchNorm2d(affine=False)` records.
    """
    import torch

    invstd = torch.rsqrt(var + eps)
    alpha = invstd if weight is None else invstd * weight
    zero = torch.zeros_like(alpha) if bias is None else bias
    beta = zero - mean * alpha
    return alpha, beta


def _channel_view(alpha, rank):
    """`alpha` shaped to multiply a conv weight's output-channel axis."""
    return alpha.reshape([-1] + [1] * (rank - 1))


def fold_conv_batch_norm(trace):
    """Fuse every eligible `conv -> batch_norm` pair of `trace` into one conv.

    Returns `(trace, fused)` where `fused` counts one entry per pair folded,
    each naming the batch-norm op it removed. A pair that fails any of the
    conditions in the module docstring is left exactly as recorded; this pass
    never raises on a graph it cannot improve, because "no fusion was possible"
    is a correct answer and an exception there would stop a caller that was
    only asking.
    """
    import torch

    constants = list(trace.constants)
    constant_values = list(trace.constant_values)
    nodes: list[dict] = []
    mapping: dict[tuple, Any] = {}
    fused: list[str] = []
    #: new node index -> that node's position in `trace.nodes`, so a fused conv
    #: can be found again by the batch norm that follows it.
    conv_at: dict[int, int] = {}

    uses = _use_counts(trace)

    def remap(ref):
        return mapping.get((ref.kind, ref.index, ref.output), ref)

    def const_value(ref):
        if ref is None:
            return None
        if not isinstance(ref, torch._C.CaptureValue) or ref.kind != "const":
            raise FusionRefused("not a constant")
        return constant_values[ref.index]

    def add_constant(tensor):
        index = len(constants)
        constants.append(
            {
                "index": index,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
        )
        constant_values.append(tensor)
        return _value("const", index)

    for position, node in enumerate(trace.nodes):
        op = node["op"]
        args = [_walk(a, remap) for a in node["args"]]
        kwargs = {k: _walk(v, remap) for k, v in node["kwargs"].items()}

        if op in _BATCH_NORMS:
            try:
                new_index = _try_fuse(
                    torch, op, node, args, kwargs, nodes, conv_at, uses,
                    trace, position, const_value, add_constant,
                )
            except FusionRefused:
                new_index = None
            if new_index is not None:
                fused.append(op)
                # The batch norm's tensor result *is* the fused convolution's.
                mapping[("node", position, 0)] = _value("node", new_index, 0)
                continue

        index = len(nodes)
        nodes.append(
            {
                "op": op,
                "args": args,
                "kwargs": kwargs,
                "outputs": list(node["outputs"]),
                "sequence": node["sequence"],
            }
        )
        if op == "aten.convolution.default":
            conv_at[index] = position
        for slot in range(len(node["outputs"])):
            mapping[("node", position, slot)] = _value("node", index, slot)

    folded = DecomposedTrace(
        list(trace.guards),
        constants,
        constant_values,
        nodes,
        [_walk(ref, remap) for ref in trace.outputs],
    )
    return folded, fused


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _use_counts(trace) -> dict[tuple, int]:
    """How many places read each `("node", position, slot)` reference.

    Trace outputs count as a use. A convolution whose result is read anywhere
    other than the batch norm being fused still has to be computed unfused, so
    the fold would have to keep both nodes -- which is a different pass.
    """
    import torch

    counts: dict[tuple, int] = {}

    def tally(arg):
        if isinstance(arg, torch._C.CaptureValue):
            key = (arg.kind, arg.index, arg.output)
            counts[key] = counts.get(key, 0) + 1
        return arg

    for node in trace.nodes:
        for value in list(node["args"]) + list(node["kwargs"].values()):
            _walk(value, tally)
    for ref in trace.outputs:
        _walk(ref, tally)
    return counts


def _try_fuse(
    torch, op, node, args, kwargs, nodes, conv_at, uses, trace, position,
    const_value, add_constant,
):
    """Fold this batch norm into its producer, or raise `FusionRefused`."""
    flat, rest = _as_the_dispatcher_would_call_it(op, args, kwargs)
    if rest:
        raise FusionRefused(f"{op} kept keyword arguments {sorted(rest)}")
    training_at = _BATCH_NORMS[op]
    if training_at is not None:
        if len(flat) <= training_at:
            raise FusionRefused(f"{op} recorded without its training flag")
        if flat[training_at] is not False:
            raise FusionRefused(
                f"{op} is in training mode; its statistics depend on the input "
                f"and there is no constant affine to fold"
            )
        source, weight, bias, mean, var = flat[0], flat[1], flat[2], flat[3], flat[4]
        eps = flat[7] if len(flat) > 7 else 1e-5
    else:
        source, weight, bias, mean, var = flat[0], flat[1], flat[2], flat[3], flat[4]
        eps = flat[6] if len(flat) > 6 else 1e-5

    # Slots 1 and 2 (save_mean / save_invstd) must be dead.
    for slot in (1, 2):
        if uses.get(("node", position, slot), 0):
            raise FusionRefused(
                f"{op} slot {slot} is read; the fused convolution does not "
                f"produce save_mean/save_invstd"
            )

    if not isinstance(source, torch._C.CaptureValue) or source.kind != "node":
        raise FusionRefused("batch norm input is not the result of a node")
    conv_index, conv_slot = source.index, source.output
    if conv_slot != 0 or conv_index not in conv_at:
        raise FusionRefused("batch norm input was not produced by a convolution")
    if uses.get(("node", conv_at[conv_index], 0), 0) != 1:
        raise FusionRefused(
            "the convolution's result is read somewhere other than this batch "
            "norm, so the unfused value is still needed"
        )

    conv = nodes[conv_index]
    conv_args, conv_rest = _as_the_dispatcher_would_call_it(
        conv["op"], conv["args"], conv["kwargs"]
    )
    if conv_rest or len(conv_args) < 9:
        raise FusionRefused("convolution not recorded in full positional form")
    if conv_args[6] is not False:
        raise FusionRefused(
            "transposed convolution: its weight carries output channels on "
            "axis 1, so scaling axis 0 would scale the wrong thing"
        )

    w_bn = const_value(weight)
    b_bn = const_value(bias)
    mean_v = const_value(mean)
    var_v = const_value(var)
    if mean_v is None or var_v is None:
        raise FusionRefused("running statistics are absent")
    w_conv = const_value(conv_args[1])
    b_conv = const_value(conv_args[2])
    if w_conv is None:
        raise FusionRefused("convolution weight is not a constant")

    alpha, beta = batch_norm_affine(w_bn, b_bn, mean_v, var_v, eps)
    if list(alpha.shape) != [w_conv.shape[0]]:
        raise FusionRefused(
            f"batch norm has {list(alpha.shape)} channels and the convolution "
            f"produces {w_conv.shape[0]}"
        )

    new_weight = w_conv * _channel_view(alpha, w_conv.dim())
    new_bias = beta if b_conv is None else b_conv * alpha + beta

    conv_args = list(conv_args)
    conv_args[1] = add_constant(new_weight)
    conv_args[2] = add_constant(new_bias)
    conv["args"] = conv_args
    conv["kwargs"] = {}
    conv["outputs"] = list(node["outputs"][:1])
    return conv_index
