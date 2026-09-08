"""Ground truth for what the Qualcomm QNN / Hexagon HTP backend actually supports.

This module is **not** a lowering path. `torchnative.export.qnn` and
`torchnative.export.qnn_device` already do the AOT/device work and
`docs/devices/QNN.md` records what that round measured (no artefact could be
built on this arm64 Mac -- ExecuTorch's QNN Python bindings are Linux-only,
see `docs/devices/QNN.md` section 1.3). This module answers a narrower,
purely-static question that does not need an artefact, a Linux host, or a
device: **for a given ATen op, does ExecuTorch's QNN partitioner take it at
all**, going by the source of that partitioner itself.

Why this has to be honest about its sourcing, not just correct. QNN's failure
mode is silent: the partitioner declines a node at COMPILE time and the graph
falls back to CPU. The program still runs and still produces correct numbers,
so "it ran and the answer was right" proves nothing about whether Hexagon was
used at all -- the opposite of NNAPI/CoreML, which fail closed at *run* time.
A table entry that is guessed rather than sourced would make this module
produce exactly that false confidence in writing. So every entry here is
either sourced to a specific file in a specific installed package, or it is
absent from the table and named in `UNVERIFIED` instead.

Where these facts came from (see `docs/devices/QNNOPS.md` for the full audit
trail): a real ExecuTorch wheel is installed at
`/Volumes/macMini/caches/qnn-venv` --- `executorch==1.4.1`
(`executorch-1.4.1.dist-info`), macosx_14_0_arm64. Its
`executorch/backends/qualcomm/builders/op_*.py` files are the QNN backend's
**node visitors**: each one registers itself (`@register_node_visitor`) against
a `target = [...]` list of ATen op overload strings, and the partitioner
(`executorch/backends/qualcomm/partition/qnn_partitioner.py`) takes a node iff
a visitor is registered for its target. That `target` list is therefore the
same list the partitioner itself consults -- not a description of it, the
input to it -- which is why it is trusted here as ground truth rather than
paraphrased from documentation.

What this module deliberately does NOT claim, mirroring
`torchnative.export.intelnpu.supported_ops`'s honesty about its own scope:
it does not claim these ops run *quantized* (`kHtpQuantized`, QNN's default
runtime mode for phones, needs a calibrated quantizer this file does not
touch -- see `constraints()["quantization"]`), it does not claim any numeric
shape/rank limit beyond what the source actually asserts (QNN SDK headers
were searched for on this machine and are not present -- see `UNVERIFIED` in
the doc), and it does not claim anything ran on real Hexagon silicon --
`docs/devices/QNN.md` section 5 is the one round that touched real silicon
(SM8550, HTP v73) and even that round could not push an artefact.
"""

from __future__ import annotations

__all__ = [
    "EXECUTORCH_VERSION",
    "SOURCE_PACKAGE",
    "supported_ops",
    "not_supported_ops",
    "to_be_implemented_ops",
    "constraints",
    "check_leaf",
]

#: The exact installed package this table was read out of. Recorded so a
#: later round can tell whether the table is stale against a newer wheel.
EXECUTORCH_VERSION = "1.4.1"
SOURCE_PACKAGE = (
    "/Volumes/macMini/caches/qnn-venv/lib/python3.13/site-packages/executorch"
    "-1.4.1.dist-info (executorch==1.4.1, macosx_14_0_arm64 wheel)"
)

# --------------------------------------------------------------------------
# The supported table.
#
# Every string below is a `target =` entry copied verbatim out of one
# `executorch/backends/qualcomm/builders/op_*.py` file in the installed
# package named by SOURCE_PACKAGE. 127 unique ATen/edge targets across 115
# builder files, enumerated by walking every op_*.py and extracting its
# `target = [...]` list. `docs/devices/QNNOPS.md` section 2 has the
# file-by-file listing this was extracted from, so a disagreement here can be
# checked against a fresh install rather than trusted on prose.
# --------------------------------------------------------------------------
_SUPPORTED_OPS = frozenset({
    "aten._log_softmax.default",
    "aten._native_batch_norm_legit.no_stats",
    "aten._native_batch_norm_legit_no_training.default",
    "aten._safe_softmax.default",
    "aten._softmax.default",
    "aten._to_copy.default",
    "aten.abs.default",
    "aten.adaptive_avg_pool2d.default",
    "aten.adaptive_max_pool2d.default",
    "aten.add.Tensor",
    "aten.amax.default",
    "aten.amin.default",
    "aten.arange.start_step",
    "aten.argmax.default",
    "aten.argmin.default",
    "aten.asin.default",
    "aten.atan.default",
    "aten.avg_pool2d.default",
    "aten.avg_pool3d.default",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_or.Tensor",
    "aten.bitwise_xor.Tensor",
    "aten.bmm.default",
    "aten.cat.default",
    "aten.ceil.default",
    "aten.channel_shuffle.default",
    "aten.clamp.default",
    "aten.constant_pad_nd.default",
    "aten.convolution.default",
    "aten.copy.default",
    "aten.cos.default",
    "aten.cumsum.default",
    "aten.div.Tensor",
    "aten.elu.default",
    "aten.embedding.default",
    "aten.eq.Tensor",
    "aten.exp.default",
    "aten.expand_copy.default",
    "aten.flip.default",
    "aten.floor.default",
    "aten.floor_divide.default",
    "aten.full.default",
    "aten.full_like.default",
    "aten.gather.default",
    "aten.ge.Tensor",
    "aten.gelu.default",
    "aten.grid_sampler_2d.default",
    "aten.grid_sampler_3d.default",
    "aten.gt.Tensor",
    "aten.hardsigmoid.default",
    "aten.hardswish.default",
    "aten.hardtanh.default",
    "aten.index.Tensor",
    "aten.index_put.default",
    "aten.index_select.default",
    "aten.instance_norm.default",
    "aten.isinf.default",
    "aten.isnan.default",
    "aten.le.Tensor",
    "aten.linear.default",
    "aten.log.default",
    "aten.logical_and.default",
    "aten.logical_not.default",
    "aten.lt.Tensor",
    "aten.matmul.default",
    "aten.max.dim",
    "aten.max_pool2d_with_indices.default",
    "aten.maximum.default",
    "aten.mean.dim",
    "aten.min.dim",
    "aten.minimum.default",
    "aten.mm.default",
    "aten.mul.Tensor",
    "aten.native_group_norm.default",
    "aten.native_layer_norm.default",
    "aten.ne.Tensor",
    "aten.neg.default",
    "aten.permute_copy.default",
    "aten.pixel_shuffle.default",
    "aten.pixel_unshuffle.default",
    "aten.pow.Tensor_Tensor",
    "aten.prelu.default",
    "aten.rand.default",
    "aten.rand_like.default",
    "aten.randn.default",
    "aten.randn_like.default",
    "aten.reciprocal.default",
    "aten.reflection_pad1d.default",
    "aten.reflection_pad2d.default",
    "aten.relu.default",
    "aten.repeat.default",
    "aten.rms_norm.default",
    "aten.round.default",
    "aten.rsqrt.default",
    "aten.scatter.src",
    "aten.select.int",
    "aten.select_copy.int",
    "aten.sigmoid.default",
    "aten.sign.default",
    "aten.sin.default",
    "aten.slice_copy.Tensor",
    "aten.slice_scatter.default",
    "aten.split_with_sizes.default",
    "aten.split_with_sizes_copy.default",
    "aten.sqrt.default",
    "aten.squeeze.dims",
    "aten.squeeze_copy.dims",
    "aten.stack.default",
    "aten.sub.Tensor",
    "aten.sum.dim_IntList",
    "aten.tanh.default",
    "aten.topk.default",
    "aten.unbind.int",
    "aten.unsqueeze_copy.default",
    "aten.upsample_bicubic2d.vec",
    "aten.upsample_bilinear2d.default",
    "aten.upsample_bilinear2d.vec",
    "aten.upsample_nearest2d.default",
    "aten.upsample_nearest2d.vec",
    "aten.view_copy.default",
    "aten.where.self",
    "dim_order_ops._to_dim_order_copy.default",
    "getitem",
    "quantized_decomposed.dequantize_per_tensor.default",
    "quantized_decomposed.dequantize_per_tensor.tensor",
    "quantized_decomposed.quantize_per_tensor.default",
    "scalar_tensor.default",
})

# --------------------------------------------------------------------------
# The explicit refusal lists. Source:
# executorch/backends/qualcomm/partition/common_defs.py in the same package.
# These are read by qnn_partitioner.py to REMOVE nodes from partitioning even
# though (for `to_be_implemented_operator`) no visitor error would otherwise
# stop them, and (for `not_supported_operator`) with a stated reason in the
# source comments -- reproduced here verbatim.
# --------------------------------------------------------------------------

#: Never partitioned, with the reason given in the source comment.
_NOT_SUPPORTED_OPS = {
    "aten._embedding_bag.default": "output size is data dependent on the slice index",
    "dim_order_ops._clone_dim_order.default": (
        "for graph sharding purpose, different from the op used in decoder models"
    ),
    "quantized_decomposed.embedding_4bit.dtype": "QNN does not support 4-bit embedding",
}

#: Named in the partitioner's own `to_be_implemented_operator` list -- i.e.
#: upstream ExecuTorch itself records these as not yet done, not merely
#: absent from this table.
_TO_BE_IMPLEMENTED_OPS = frozenset({
    "aten.adaptive_max_pool3d.default",
    "aten.max_pool3d_with_indices.default",
    "aten.median.default",
    "aten.median.dim",
    "aten.round.decimals",
    "aten.le.Scalar",
})

# --------------------------------------------------------------------------
# Dtype support. Source: executorch/backends/qualcomm/builders/node_visitor.py
# QNN_TENSOR_TYPE_MAP (unquantized I/O tensors) and QNN_QUANT_TYPE_MAP
# (quantized tensors), lines ~48-74.
# --------------------------------------------------------------------------

#: dtypes QNN has a tensor type for at all, unquantized path.
_UNQUANTIZED_DTYPES = frozenset({
    "bool", "float16", "float32", "float64", "int8", "int16", "int32",
    "int64", "uint8", "uint16", "uint32",
})

#: dtypes valid on the QUANTIZED path. Notably narrower: no int64, no
#: uint32, no bool, no float. Source comment: "there is no int64 tensor data
#: type in Qnn" (QNN_QUANT_TYPE_MAP maps torch.int64 to QNN_DATATYPE_UNDEFINED).
_QUANTIZED_DTYPES = frozenset({"int8", "int16", "int32", "uint8", "uint16"})

#: float64 has no native QNN type either; QNN_TENSOR_TYPE_MAP silently
#: downcasts it to QNN_DATATYPE_FLOAT_32. Recorded so `check_leaf` can name
#: this as a narrowing rather than a straightforward accept.
_DOWNCAST_DTYPES = {"float64": "float32"}


def supported_ops() -> frozenset:
    """ATen/edge op target strings a QNN node visitor is registered for.

    This is the same `target` value the real partitioner consults --
    extracted from installed source, not paraphrased. It says nothing about
    whether a *particular* node of one of these ops will be accepted: shape,
    dtype and the explicit refusal lists in `constraints()` all still apply,
    which is what `check_leaf` is for.
    """
    return _SUPPORTED_OPS


def not_supported_ops() -> dict:
    """Ops explicitly excluded by `common_defs.not_supported_operator`, with reasons."""
    return dict(_NOT_SUPPORTED_OPS)


def to_be_implemented_ops() -> frozenset:
    """Ops ExecuTorch's own partitioner lists as not yet implemented for QNN."""
    return _TO_BE_IMPLEMENTED_OPS


def constraints() -> dict:
    """The numeric/structural limits sourced from the installed package.

    Deliberately does NOT include a max-rank or max-tensor-dimension figure:
    unlike `torchnative.export.intelnpu.MAX_DIM` (sourced to a specific
    linear-IR assertion), no such limit was found in
    `executorch/backends/qualcomm/builders/` or `partition/` at this
    ExecuTorch version, and no QNN SDK header (`QnnTypes.h`, `QnnOpDef.h`,
    HTP op-package headers) is present on this machine to source one from
    directly -- `mdfind`/`find` for both came back empty. See
    `docs/devices/QNNOPS.md`'s `UNVERIFIED` section. Do not fill this in by
    guessing a number from Qualcomm marketing material; source it or leave it
    out.
    """
    return {
        "unquantized_dtypes": _UNQUANTIZED_DTYPES,
        "quantized_dtypes": _QUANTIZED_DTYPES,
        "float64_downcast_to": dict(_DOWNCAST_DTYPES),
        "not_supported": dict(_NOT_SUPPORTED_OPS),
        "to_be_implemented": _TO_BE_IMPLEMENTED_OPS,
        "quantization": (
            "kHtpQuantized is QNN's default runtime mode and the interesting "
            "one for a phone; this table says nothing about whether a "
            "*calibrated* quantized graph is accepted, only which ops have a "
            "quantized dtype mapping at all. A real calibration/quantizer "
            "pass (executorch.backends.qualcomm.quantizer) was not exercised "
            "-- docs/devices/QNN.md section 11 records the same gap for the "
            "AOT lowering path."
        ),
        "static_shape": (
            "UNVERIFIED at this ExecuTorch version: no assertion enforcing "
            "static shapes was found in builders/ or partition/ during this "
            "audit, and no QNN SDK header was available to check against "
            "directly. Treat QNN/HTP's well-known static-shape requirement "
            "as industry knowledge, not as something this table sourced -- "
            "see docs/devices/QNNOPS.md UNVERIFIED."
        ),
    }


def check_leaf(op_name: str, shapes=None, dtype=None):
    """Per-leaf verdict for whether QNN/HTP would take a node of this op.

    Returns a dict `{"accepted": bool, "reason": str}`. `reason` is always a
    string, sourced to one of the tables above -- never a bare `False` --
    matching `torchnative.export.intelnpu.plan_lowering`'s convention that
    every skip names why. `shapes`/`dtype` are optional because this table
    carries no numeric shape constraint to check them against (see
    `constraints()["static_shape"]`); they are accepted now so a later round
    that sources one does not have to change this function's signature.

    This function's absence of a shape check is a real gap, not silent
    optimism: `constraints()["static_shape"]` names it explicitly, and
    `accepted=True` here means only "a node visitor is registered and no
    known refusal reason applies" -- it is not a claim that any specific
    shape or quantization configuration will be accepted by the real
    partitioner, and it is not a claim that anything will actually run on
    Hexagon silicon (that is a runtime question this module does not answer;
    see `torchnative.export.qnn` and `docs/devices/QNN.md`).
    """
    if op_name in _NOT_SUPPORTED_OPS:
        return {
            "accepted": False,
            "reason": (
                f"{op_name} is in common_defs.not_supported_operator: "
                f"{_NOT_SUPPORTED_OPS[op_name]}"
            ),
        }
    if op_name in _TO_BE_IMPLEMENTED_OPS:
        return {
            "accepted": False,
            "reason": (
                f"{op_name} is in common_defs.to_be_implemented_operator -- "
                "ExecuTorch's own partitioner names this as not yet done for "
                "QNN, not merely absent from the builder set."
            ),
        }
    if op_name not in _SUPPORTED_OPS:
        return {
            "accepted": False,
            "reason": (
                f"no QNN node visitor is registered for {op_name} in "
                f"{SOURCE_PACKAGE}; it is absent from every op_*.py "
                "`target` list this table was built from."
            ),
        }
    if dtype is not None:
        dtype = str(dtype).replace("torch.", "")
        if dtype in _DOWNCAST_DTYPES:
            return {
                "accepted": True,
                "reason": (
                    f"{op_name} is supported, but {dtype} has no native QNN "
                    f"tensor type and is silently downcast to "
                    f"{_DOWNCAST_DTYPES[dtype]} (QNN_TENSOR_TYPE_MAP, "
                    "node_visitor.py)."
                ),
            }
        if dtype not in _UNQUANTIZED_DTYPES:
            return {
                "accepted": False,
                "reason": (
                    f"{dtype} has no entry in QNN_TENSOR_TYPE_MAP "
                    f"(node_visitor.py) for {op_name}."
                ),
            }
    return {
        "accepted": True,
        "reason": (
            f"a QNN node visitor is registered for {op_name} and no known "
            "refusal reason applies. This is not a shape or quantized-"
            "calibration check -- see this function's docstring."
        ),
    }
