"""docs/kernels/METAEMB.md -- the meta kernels real models actually stop on.

docs/kernels/METAFAM.md §2.2 MEASURED that the top wall for a `forward` under
`torch.device("meta")` was `aten.embedding.default`, for five of seven
architectures, and left it undone because it belonged to a different family.
This file is the tests for closing it -- and for the multi-output ops
METAFAM.md §6 named (`max.dim`, `argmax`, `topk`, `sort`), whose meta kernels
have to invent an INDEX shape and dtype as well as a value one.

**The measured priority was re-derived, not inherited.** METAFAM.md §2.2's
list had gone stale: two of its entries (`slice.Tensor`, `cumsum.default`)
were closed by that very round, so bert and roberta had already advanced to a
different wall. METAEMB.md §2 records the ranking measured here.

**The rule this file exists to pin down** is the index dtype. Upstream returns
`int64` indices from every multi-output reduction, unconditionally, and a meta
kernel that returns any other index dtype produces a perfectly plausible pair
-- right shape, right value dtype -- that fails arbitrarily far away, the
first time somebody uses the index. `.shape`-only assertions cannot see it,
which is why every comparison here records shape, dtype AND stride.

Method, unchanged from METAFAM.md §4 and VOICE4.md §6: one probe script, run
in its own subprocess against this shim (vendored tree) and against upstream
torch 2.13.0 (`PYTHONPATH` stripped), the transcript diffed key by key.
"""

import json
import os
import subprocess
import sys

from test_shim import _C

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


_PROBE = r"""
import json, sys
import torch

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}
DTYPES = {
    "f32": torch.float32, "f64": torch.float64, "f16": torch.float16,
    "bf16": torch.bfloat16, "i64": torch.int64, "i32": torch.int32,
    "i16": torch.int16, "u8": torch.uint8, "bool": torch.bool,
}


def one(t):
    # Shape, dtype AND stride. The stride is here because METAFAM.md's own
    # round found its first draft reading `.tensor()` (candle storage) where
    # meta needs `.dims()` (metadata only): a meta kernel's entire output is
    # metadata, so every piece of it that upstream publishes has to be
    # compared or it is not being checked at all.
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "stride": list(t.stride()),
        "is_meta": bool(t.is_meta),
    }


def rec(name, fn):
    try:
        r = fn()
    except Exception as e:
        out[name] = {"error": type(e).__name__}
        return
    if isinstance(r, (tuple, list)) or type(r).__module__ == "torch.return_types":
        out[name] = {"outputs": [one(t) for t in r]}
    else:
        out[name] = one(r)


def m(*shape, dtype=torch.float32):
    return torch.empty(*shape, dtype=dtype, device="meta")


def mi(*shape, dtype=torch.int64):
    return torch.empty(*shape, dtype=dtype, device="meta")


# --- embedding -------------------------------------------------------------

for tag, dt in DTYPES.items():
    rec(f"emb_w_{tag}", lambda dt=dt: torch.embedding(m(5, 3, dtype=dt), mi(2)))
rec("emb_idx_2d", lambda: torch.embedding(m(5, 3), mi(2, 4)))
rec("emb_idx_rank0", lambda: torch.embedding(m(5, 3), mi()))
rec("emb_idx_empty", lambda: torch.embedding(m(5, 3), mi(0)))
rec("emb_idx_i32", lambda: torch.embedding(m(5, 3), mi(2, dtype=torch.int32)))
rec("emb_idx_f32_refuses", lambda: torch.embedding(m(5, 3), m(2)))
rec("emb_idx_bool_refuses", lambda: torch.embedding(m(5, 3), mi(2, dtype=torch.bool)))
rec("emb_w_rank1_refuses", lambda: torch.embedding(m(5), mi(2)))
rec("emb_w_rank3_refuses", lambda: torch.embedding(m(5, 3, 2), mi(2)))
rec("emb_padding_idx_ignored", lambda: torch.embedding(m(5, 3), mi(2), 0))
rec("emb_wide", lambda: torch.embedding(m(50257, 768), mi(1, 8)))
# Excluded from the strict diff: this shim's dense kernel refuses these two
# (no autograd), upstream's forward accepts them. A deliberate shim-wide
# refusal, not a meta question.
rec("emb_sparse", lambda: torch.embedding(m(5, 3), mi(2), -1, False, True))
rec("emb_scale_grad", lambda: torch.embedding(m(5, 3), mi(2), -1, True, False))

# --- gather ----------------------------------------------------------------

for tag, dt in DTYPES.items():
    rec(f"gather_{tag}", lambda dt=dt: torch.gather(m(3, 4, dtype=dt), 1, mi(3, 2)))
rec("gather_neg_dim", lambda: torch.gather(m(3, 4), -1, mi(3, 2)))
rec("gather_rank0", lambda: torch.gather(m(), 0, mi()))
rec("gather_idx_i32", lambda: torch.gather(m(3, 4), 1, mi(3, 2, dtype=torch.int32)))
rec("gather_idx_f32_refuses", lambda: torch.gather(m(3, 4), 1, m(3, 2)))
rec("gather_rank_mismatch_refuses", lambda: torch.gather(m(3, 4), 1, mi(3)))
rec("gather_too_big_refuses", lambda: torch.gather(m(3, 4), 1, mi(5, 2)))
rec("gather_bad_dim_refuses", lambda: torch.gather(m(3, 4), 7, mi(3, 2)))
rec("gather_idx_bigger_at_dim", lambda: torch.gather(m(3, 4), 1, mi(3, 9)))
# Stride only: upstream reports (1, 1) for a (3, 0) result where the
# contiguous strides this shim computes are (0, 1). See _STRIDE_DIVERGENCE.
rec("gather_empty_idx", lambda: torch.gather(m(3, 4), 1, mi(3, 0)))

# --- the four multi-output ops, and the INDEX-DTYPE RULE -------------------

for tag, dt in DTYPES.items():
    rec(f"maxdim_{tag}", lambda dt=dt: torch.max(m(3, 4, dtype=dt), dim=1))
    rec(f"argmax_{tag}", lambda dt=dt: torch.argmax(m(3, 4, dtype=dt), dim=1))
    rec(f"topk_{tag}", lambda dt=dt: torch.topk(m(3, 4, dtype=dt), 2, dim=1))
    rec(f"sort_{tag}", lambda dt=dt: torch.sort(m(3, 4, dtype=dt), dim=1))
rec("maxdim_keepdim", lambda: torch.max(m(3, 4), dim=1, keepdim=True))
rec("maxdim_neg_dim", lambda: torch.max(m(3, 4), dim=-2))
rec("maxdim_rank0", lambda: torch.max(m(), dim=0))
rec("maxdim_rank0_neg", lambda: torch.max(m(), dim=-1))
rec("maxdim_rank0_keepdim", lambda: torch.max(m(), dim=0, keepdim=True))
rec("maxdim_empty_other_dim", lambda: torch.max(m(0, 4), dim=1))
rec("maxdim_bad_dim_refuses", lambda: torch.max(m(3, 4), dim=5))
rec("mindim_basic", lambda: torch.min(m(3, 4), dim=1))
# Divergence 1: upstream's meta answers a shape, its cpu raises IndexError.
rec("maxdim_zero_extent", lambda: torch.max(m(0, 4), dim=0))

rec("argmax_no_dim", lambda: torch.argmax(m(3, 4)))
rec("argmax_no_dim_keepdim", lambda: torch.argmax(m(3, 4), keepdim=True))
rec("argmax_keepdim", lambda: torch.argmax(m(3, 4), dim=1, keepdim=True))
rec("argmax_neg_dim", lambda: torch.argmax(m(3, 4), dim=-2))
rec("argmax_rank0", lambda: torch.argmax(m()))
rec("argmax_rank0_dim", lambda: torch.argmax(m(), dim=0))
rec("argmax_empty_other_dim", lambda: torch.argmax(m(0, 4), dim=1))
rec("argmax_empty_reduced_dim_refuses", lambda: torch.argmax(m(0, 4), dim=0))
rec("argmax_bad_dim_refuses", lambda: torch.argmax(m(3, 4), dim=7))
# Divergence 2: upstream's meta raises RuntimeError here, its cpu IndexError.
rec("argmax_empty_no_dim", lambda: torch.argmax(m(0, 4)))

rec("topk_k0", lambda: torch.topk(m(3, 4), 0, dim=1))
rec("topk_k_all", lambda: torch.topk(m(3, 4), 4, dim=1))
rec("topk_neg_dim", lambda: torch.topk(m(3, 4), 2, dim=-1))
rec("topk_rank0", lambda: torch.topk(m(), 1))
rec("topk_largest_false", lambda: torch.topk(m(3, 4), 2, dim=1, largest=False))
rec("topk_sorted_false", lambda: torch.topk(m(3, 4), 2, dim=1, sorted=False))
rec("topk_k_too_big_refuses", lambda: torch.topk(m(3, 4), 9, dim=1))

rec("sort_neg_dim", lambda: torch.sort(m(3, 4), dim=-2))
rec("sort_descending", lambda: torch.sort(m(3, 4), descending=True))
rec("sort_rank0", lambda: torch.sort(m(), dim=0))
rec("sort_empty", lambda: torch.sort(m(0, 4), dim=0))
# Divergence 5: upstream's meta SILENTLY ACCEPTS an out-of-range dim here.
rec("sort_bad_dim", lambda: torch.sort(m(3, 4), dim=9))

# --- contraction -----------------------------------------------------------

rec("mm_basic", lambda: torch.mm(m(3, 4), m(4, 5)))
rec("mm_mismatch_refuses", lambda: torch.mm(m(3, 4), m(5, 6)))
rec("mm_rank3_refuses", lambda: torch.mm(m(2, 3, 4), m(4, 5)))
rec("bmm_basic", lambda: torch.bmm(m(2, 3, 4), m(2, 4, 5)))
rec("bmm_batch_mismatch_refuses", lambda: torch.bmm(m(2, 3, 4), m(3, 4, 5)))
rec("addmm_basic", lambda: torch.addmm(m(5), m(3, 4), m(4, 5)))
rec("addmm_full_bias", lambda: torch.addmm(m(3, 5), m(3, 4), m(4, 5)))
for tag, dt in DTYPES.items():
    rec(f"matmul_{tag}", lambda dt=dt: torch.matmul(m(3, 4, dtype=dt), m(4, 5, dtype=dt)))
rec("matmul_batched", lambda: torch.matmul(m(2, 3, 4), m(2, 4, 5)))
rec("matmul_broadcast", lambda: torch.matmul(m(2, 1, 3, 4), m(5, 4, 6)))
rec("matmul_mismatch_refuses", lambda: torch.matmul(m(3, 4), m(5, 6)))
# Excluded: this shim's dense kernel refuses 1-D operands by name.
rec("matmul_vector", lambda: torch.matmul(m(4), m(4)))

# --- native_layer_norm: three outputs, two shapes, two dtypes --------------

rec("lnorm_f32", lambda: torch.native_layer_norm(m(2, 3, 4), [4], m(4), m(4), 1e-5))
rec("lnorm_no_params", lambda: torch.native_layer_norm(m(2, 3, 4), [4], None, None, 1e-5))
rec("lnorm_two_axes", lambda: torch.native_layer_norm(m(2, 3, 4), [3, 4], None, None, 1e-5))
rec("lnorm_all_axes", lambda: torch.native_layer_norm(m(2, 3, 4), [2, 3, 4], None, None, 1e-5))
rec("lnorm_f64", lambda: torch.native_layer_norm(m(2, 3, 4, dtype=torch.float64), [4], None, None, 1e-5))
rec("lnorm_bad_shape_refuses", lambda: torch.native_layer_norm(m(2, 3, 4), [5], None, None, 1e-5))
rec("lnorm_too_many_axes_refuses", lambda: torch.native_layer_norm(m(2, 3, 4), [2, 3, 4, 5], None, None, 1e-5))
rec("lnorm_empty_ns_refuses", lambda: torch.native_layer_norm(m(2, 3, 4), [], None, None, 1e-5))
rec("lnorm_int_refuses", lambda: torch.native_layer_norm(m(2, 3, 4, dtype=torch.int64), [4], None, None, 1e-5))
# Divergence 6: upstream's META answers float32 statistics for a reduced
# input; its CPU answers the input's own dtype, and so does this shim.
rec("lnorm_f16_stats", lambda: torch.native_layer_norm(m(2, 3, 4, dtype=torch.float16), [4], None, None, 1e-5))
rec("lnorm_bf16_stats", lambda: torch.native_layer_norm(m(2, 3, 4, dtype=torch.bfloat16), [4], None, None, 1e-5))

# --- combine / split -------------------------------------------------------

rec("cat_dim0", lambda: torch.cat([m(2, 3), m(4, 3)], 0))
rec("cat_dim1", lambda: torch.cat([m(2, 3), m(2, 5)], 1))
rec("cat_neg_dim", lambda: torch.cat([m(2, 3), m(2, 5)], -1))
rec("cat_single", lambda: torch.cat([m(2, 3)], 0))
rec("cat_promotes", lambda: torch.cat([m(2, 3, dtype=torch.float16), m(2, 3)], 0))
rec("cat_int_bool_promotes", lambda: torch.cat([mi(2, 3), mi(2, 3, dtype=torch.bool)], 0))
rec("cat_skips_1d_empty", lambda: torch.cat([m(0), m(3)], 0))
rec("cat_mismatch_refuses", lambda: torch.cat([m(2, 3), m(4, 5)], 0))
rec("cat_rank_mismatch_refuses", lambda: torch.cat([m(2, 3), m(3)], 0))
rec("cat_empty_list_refuses", lambda: torch.cat([], 0))

rec("split_even", lambda: torch.split(m(6, 3), 2, 0))
rec("split_uneven", lambda: torch.split(m(7, 3), 3, 0))
rec("split_dim1", lambda: torch.split(m(6, 4), 3, 1))
rec("split_neg_dim", lambda: torch.split(m(6, 4), 3, -1))
rec("split_bigger_than_extent", lambda: torch.split(m(6, 3), 9, 0))
rec("split_zero_refuses", lambda: torch.split(m(6, 3), 0, 0))
rec("split_empty_tensor", lambda: torch.split(m(0, 3), 2, 0))
rec("split_with_sizes", lambda: torch.split_with_sizes(m(6, 3), [1, 2, 3], 0))

# --- convolution -----------------------------------------------------------

rec("conv1d", lambda: torch.convolution(m(1, 4, 10), m(8, 4, 3), None, [1], [0], [1], False, [0], 1))
rec("conv1d_stride_pad", lambda: torch.convolution(m(1, 4, 10), m(8, 4, 3), None, [2], [2], [1], False, [0], 1))
rec("conv1d_dilated", lambda: torch.convolution(m(1, 4, 10), m(8, 4, 3), None, [1], [0], [2], False, [0], 1))
rec("conv2d", lambda: torch.convolution(m(2, 3, 8, 8), m(6, 3, 3, 3), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1))
rec("conv2d_groups", lambda: torch.convolution(m(2, 3, 8, 8), m(6, 1, 3, 3), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 3))
rec("conv1d_transposed", lambda: torch.convolution(m(1, 4, 10), m(4, 8, 3), None, [2], [1], [1], True, [1], 1))
rec("conv1d_bias", lambda: torch.convolution(m(1, 4, 10), m(8, 4, 3), m(8), [1], [0], [1], False, [0], 1))
rec("conv1d_f16", lambda: torch.convolution(m(1, 4, 10, dtype=torch.float16), m(8, 4, 3, dtype=torch.float16), None, [1], [0], [1], False, [0], 1))
rec("conv_kernel_too_big_refuses", lambda: torch.convolution(m(1, 4, 2), m(8, 4, 3), None, [1], [0], [1], False, [0], 1))

# --- SDPA (cpu flash), activations, repeat, advanced indexing --------------

# Spelled through `torch.ops.aten`, not `torch._scaled_dot_...`: the bare
# `torch._C` name has no entry in this shim's overload table, so calling it
# that way refuses before the dispatcher is reached and the probe would be
# measuring the name table rather than the meta kernel. Models reach it
# through `F.scaled_dot_product_attention`, which lands on the same op.
_sdpa = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default
rec("sdpa", lambda: _sdpa(m(2, 4, 8, 16), m(2, 4, 8, 16), m(2, 4, 8, 16)))
rec("sdpa_f16_logsumexp_stays_f32", lambda: _sdpa(
    m(2, 4, 8, 16, dtype=torch.float16), m(2, 4, 8, 16, dtype=torch.float16), m(2, 4, 8, 16, dtype=torch.float16)))
rec("sdpa_longer_kv", lambda: _sdpa(m(2, 4, 8, 16), m(2, 4, 12, 16), m(2, 4, 12, 16)))
# Divergence 7: upstream's meta answers the query's head size; its cpu refuses.
rec("sdpa_head_mismatch", lambda: _sdpa(m(2, 4, 8, 16), m(2, 4, 12, 16), m(2, 4, 12, 32)))

for name, fn in (("gelu", torch.nn.functional.gelu), ("silu", torch.nn.functional.silu), ("relu", torch.relu)):
    for tag in ("f32", "f64", "f16", "bf16"):
        rec(f"{name}_{tag}", lambda fn=fn, tag=tag: fn(m(2, 3, dtype=DTYPES[tag])))
    # Divergence 8: upstream's meta accepts a non-floating input for all
    # three; its cpu refuses, and so does this shim's dense kernel.
    rec(f"{name}_i64_div", lambda fn=fn: fn(m(2, 3, dtype=torch.int64)))
    rec(f"{name}_bool_div", lambda fn=fn: fn(m(2, 3, dtype=torch.bool)))
rec("gelu_approximate_tanh", lambda: torch.nn.functional.gelu(m(2, 3), approximate="tanh"))

rec("repeat_basic", lambda: m(2, 3).repeat(2, 3))
rec("repeat_extra_leading", lambda: m(3).repeat(4, 2))
rec("repeat_zero", lambda: m(2, 3).repeat(0, 3))
rec("repeat_rank0", lambda: m().repeat(2, 2))
rec("repeat_i64", lambda: mi(2, 3).repeat(2, 1))
rec("repeat_too_few_refuses", lambda: m(2, 3).repeat(2))

rec("index_one", lambda: torch.ops.aten.index.Tensor(m(5, 3), [mi(2)]))
rec("index_one_2d", lambda: torch.ops.aten.index.Tensor(m(5, 3), [mi(2, 4)]))
rec("index_none_then", lambda: torch.ops.aten.index.Tensor(m(5, 3), [None, mi(2)]))
rec("index_two_adjacent", lambda: torch.ops.aten.index.Tensor(m(5, 3, 7), [mi(2), mi(2)]))
rec("index_separated_moves_front", lambda: torch.ops.aten.index.Tensor(m(5, 3, 7), [mi(2), None, mi(2)]))
rec("index_broadcasts", lambda: torch.ops.aten.index.Tensor(m(5, 3, 7), [mi(2, 1), mi(1, 4)]))
rec("index_rank0_index", lambda: torch.ops.aten.index.Tensor(m(5, 3), [mi()]))
rec("index_i32", lambda: torch.ops.aten.index.Tensor(m(5, 3), [mi(2, dtype=torch.int32)]))
rec("index_too_many_refuses", lambda: torch.ops.aten.index.Tensor(m(5, 3), [mi(2), mi(2), mi(2)]))
rec("index_bool_mask_refuses", lambda: torch.ops.aten.index.Tensor(m(5, 3), [mi(5, dtype=torch.bool)]))

# --- the three that must never get a meta kernel ---------------------------

rec("masked_select_refuses", lambda: torch.masked_select(m(3, 4), mi(3, 4, dtype=torch.bool)))
rec("unique_refuses", lambda: torch.unique(m(3, 4)))
rec("repeat_interleave_refuses", lambda: torch.repeat_interleave(mi(3)))

json.dump(out, sys.stdout)
"""

_cache = {}


def _run(side):
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
        [sys.executable, "-c", _PROBE], capture_output=True, text=True,
        env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, f"the {side} metaemb probe failed:\n{proc.stderr[-4000:]}"
    data = json.loads(proc.stdout)
    assert data["_marker"] == side, (
        f"the {side} metaemb probe imported the wrong torch ({data['_marker']})"
    )
    _cache[side] = data
    return data


# Cases where upstream's OWN meta kernel disagrees with upstream's OWN cpu
# kernel, so "agree with upstream" has two answers and the diff cannot be
# strict. This shim has one door and follows its dense kernel, which follows
# cpu -- docs/devices/META.md §7.3's standing rule, applied here to nine more
# instances of it. Every one is checked separately BY NAME below, in
# `test_the_nine_upstream_self_disagreements_are_followed_to_the_dense_side`,
# so that "excluded" never means "unchecked".
_KNOWN_META_VS_CPU_DIVERGENCE = {
    "maxdim_zero_extent",
    "emb_w_rank1_refuses", "emb_w_rank3_refuses",
    "lnorm_int_refuses",
    "index_too_many_refuses",
    "argmax_empty_no_dim",
    "sort_bad_dim",
    "lnorm_f16_stats",
    "lnorm_bf16_stats",
    "sdpa_head_mismatch",
    "gelu_i64_div", "gelu_bool_div",
    "silu_i64_div", "silu_bool_div",
    "relu_bool_div",
}

# Cases excluded for a different reason: this shim refuses something upstream
# accepts on BOTH devices, by a deliberate shim-wide decision that has nothing
# to do with meta (no autograd; 1-D matmul rules never measured). Also checked
# by name below.
_KNOWN_SHIM_WIDE_REFUSAL = {
    "emb_sparse",
    "emb_scale_grad",
    "matmul_vector",
}

# Cases that never reach `meta_dispatch` in the first place, so they say
# nothing about a meta kernel either way. `torch.cat([], 0)` carries no
# tensor, so `check_devices_agree` finds no meta device to route on and the
# call lands in the DENSE kernel -- which answers `RuntimeError` where
# upstream answers `ValueError`. A real (pre-existing, dense) divergence,
# recorded in docs/kernels/METAEMB.md §3.4, and not a meta one. The meta arm's own
# `ValueError` for an empty list is unreachable and says so here rather than
# being quietly deleted: it would become reachable the day a meta device can
# be named without a tensor.
_UNREACHABLE_FROM_META = {"cat_empty_list_refuses", "repeat_interleave_refuses"}

# Stride only. This shim's meta tensors report CONTIGUOUS strides computed
# from the shape (docs/devices/META.md §12 -- meta carries no stride field of its
# own), which is right everywhere upstream's result is contiguous and differs
# on degenerate extents, where upstream reports a stride this shim has no
# field to remember. Named rather than dropped: the shape and dtype of these
# cases ARE compared.
_STRIDE_DIVERGENCE = {
    # Degenerate extents: a zero-length axis, where upstream reports a stride
    # this shim has no field to remember.
    "gather_empty_idx", "topk_k0",
    # NOT degenerate, and the more interesting half: upstream's `split`
    # chunks are VIEWS and keep the parent's stride (`(4, 1)` for a `(6, 4)`
    # parent, on every chunk), and its SDPA `logsumexp` is laid out
    # transposed (`(32, 1, 4)` for a `(2, 4, 8)` shape). This shim's meta
    # tensors carry no stride field at all and report contiguous strides
    # computed from the shape -- docs/devices/META.md §12's standing gap, whose first
    # instance was `expand`. These are its second and third, and the first
    # where the divergence is not confined to an empty tensor.
    "split_dim1", "split_neg_dim",
    "sdpa", "sdpa_f16_logsumexp_stays_f32", "sdpa_longer_kv",
}


def _compare(prefix, min_cases):
    shim = _run("shim")
    if shim is None:
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return 0
    upstream = _run("upstream")
    excluded = _KNOWN_META_VS_CPU_DIVERGENCE | _KNOWN_SHIM_WIDE_REFUSAL | _UNREACHABLE_FROM_META
    keys = sorted(k for k in upstream if k.startswith(prefix) and k not in excluded)
    assert keys, f"the probe recorded no {prefix}* cases"
    diffs = []
    for key in keys:
        want, got = upstream[key], shim.get(key)
        if key in _STRIDE_DIVERGENCE:
            want, got = _drop_stride(want), _drop_stride(got)
        if got != want:
            diffs.append(f"{key}: shim {got} != upstream {want}")
    assert not diffs, "\n".join(diffs)
    assert len(keys) >= min_cases, f"only {len(keys)} {prefix}* cases were compared"
    return len(keys)


def _drop_stride(record):
    if not isinstance(record, dict):
        return record
    if "outputs" in record:
        return {"outputs": [_drop_stride(o) for o in record["outputs"]]}
    return {k: v for k, v in record.items() if k != "stride"}


# --------------------------------------------------------------------------
# The measured wall: embedding and gather
# --------------------------------------------------------------------------


def test_embedding_answers_what_upstream_answers_across_every_argument_form():
    """The op METAFAM.md §2.2 measured as the top wall and left undone.

    Shape is the INDEX tensor's with the embedding width appended; dtype is
    the WEIGHT's, never the index's. Both halves are wrong in a plausible
    way if swapped, so both are compared across all nine weight dtypes, a
    0-dim index, an empty index, `int32` indices (cpmant's spelling) and the
    two refusals.
    """
    assert _compare("emb_", 15) >= 15


def test_gather_answers_the_index_shape_with_the_inputs_dtype():
    """bert's and roberta's wall, once METAFAM closed slice and cumsum.

    Note what is NOT here and cannot be: the dense kernel bounds-checks every
    index VALUE against `self`'s extent, and a meta tensor holds no values.
    Upstream's meta kernel omits the same check. That is the one thing a
    caller loses by running `gather` on meta, and it is stated in
    docs/kernels/METAEMB.md §4 rather than papered over.
    """
    assert _compare("gather_", 15) >= 15


# --------------------------------------------------------------------------
# The index-dtype rule
# --------------------------------------------------------------------------


def test_the_index_half_is_int64_whatever_the_input_dtype_was():
    """THE rule this round exists to pin down, asserted directly.

    Not "the shapes agree with upstream" -- that is the test below. This one
    reads the index dtype out of the shim's own answer for all nine input
    dtypes and asserts `int64` literally, so that a change making both sides
    wrong in the same way still fails here.
    """
    shim = _run("shim")
    if shim is None:
        return
    checked = 0
    for op in ("maxdim", "topk", "sort"):
        for tag in ("f32", "f64", "f16", "bf16", "i64", "i32", "i16", "u8", "bool"):
            record = shim[f"{op}_{tag}"]
            assert "outputs" in record, f"{op}_{tag} did not answer a pair: {record}"
            values, indices = record["outputs"]
            assert indices["dtype"] == "torch.int64", (
                f"{op}_{tag}: index dtype is {indices['dtype']}, not torch.int64"
            )
            assert indices["shape"] == values["shape"], (
                f"{op}_{tag}: index shape {indices['shape']} != value shape {values['shape']}"
            )
            checked += 1
    for tag in ("f32", "f64", "f16", "bf16", "i64", "i32", "i16", "u8", "bool"):
        record = shim[f"argmax_{tag}"]
        assert record.get("dtype") == "torch.int64", (
            f"argmax_{tag} answered {record}, not a lone int64 tensor"
        )
        checked += 1
    assert checked == 36, checked


def test_the_value_half_keeps_the_inputs_own_dtype_and_does_not_widen():
    """The other half of the same rule, and the one a `sum`-shaped habit gets
    wrong: `sum` widens an integral input to `int64`, and `max.dim`/`topk`/
    `sort` do NOT. A kernel that reused `sum_natural_tag` here would pass
    every float case and fail every integral one."""
    shim = _run("shim")
    if shim is None:
        return
    want = {
        "f32": "torch.float32", "f64": "torch.float64", "f16": "torch.float16",
        "bf16": "torch.bfloat16", "i64": "torch.int64", "i32": "torch.int32",
        "i16": "torch.int16", "u8": "torch.uint8", "bool": "torch.bool",
    }
    for op in ("maxdim", "topk", "sort"):
        for tag, dtype in want.items():
            values = shim[f"{op}_{tag}"]["outputs"][0]
            assert values["dtype"] == dtype, (
                f"{op}_{tag}: value dtype {values['dtype']} != input's {dtype}"
            )


def test_max_dim_answers_what_upstream_answers_including_keepdim_and_rank0():
    assert _compare("maxdim_", 14) >= 14
    assert _compare("mindim_", 1) >= 1


def test_argmax_answers_what_upstream_answers_including_the_flatten_form():
    """`dim=None, keepdim=True` answers a shape of `rank` ones -- `(1, 1)` for
    a `(3, 4)` input, measured on both upstream devices. This shim's DENSE
    kernel answers `(1,)` there, which is a pre-existing dense defect this
    round found and reports (METAEMB.md §3.4) rather than copying."""
    assert _compare("argmax_", 14) >= 14


def test_topk_answers_what_upstream_answers_including_k0_and_the_k_refusal():
    assert _compare("topk_", 14) >= 14


def test_sort_keeps_the_shape_and_answers_int64_indices():
    assert _compare("sort_", 12) >= 12


# --------------------------------------------------------------------------
# The families the same measurement named after embedding fell
# --------------------------------------------------------------------------


def test_the_contraction_family_answers_what_upstream_answers():
    assert _compare("mm_", 3) >= 3
    assert _compare("bmm_", 2) >= 2
    assert _compare("addmm_", 2) >= 2
    assert _compare("matmul_", 11) >= 11


def test_native_layer_norm_answers_three_outputs_with_two_shapes():
    """`mean` and `rstd` are the input's shape with the normalised axes
    COLLAPSED TO 1, not removed -- `keepdim=True`'s shape. Answering
    `reduced_dims(..., keepdim=False)` there gives a plausible tensor of the
    wrong rank."""
    assert _compare("lnorm_", 8) >= 8


def test_cat_and_split_answer_what_upstream_answers():
    assert _compare("cat_", 8) >= 8
    assert _compare("split_", 8) >= 8


def test_convolution_answers_both_directions_of_the_arithmetic():
    assert _compare("conv", 9) >= 9


def test_sdpa_logsumexp_is_float32_whatever_the_query_dtype_was():
    """The index-dtype rule's shape with `float32` in place of `int64`: the
    second output has a different rank AND a different dtype from the first,
    and stays `float32` for a `float16` query. Measured on both upstream
    devices."""
    assert _compare("sdpa", 3) >= 3
    shim = _run("shim")
    if shim is None:
        return
    out, lse = shim["sdpa_f16_logsumexp_stays_f32"]["outputs"]
    assert out["dtype"] == "torch.float16", out
    assert lse["dtype"] == "torch.float32", lse
    assert lse["shape"] == out["shape"][:3], (lse, out)


def test_the_activations_and_repeat_and_advanced_indexing_agree():
    assert _compare("gelu_", 5) >= 5
    assert _compare("silu_", 4) >= 4
    assert _compare("relu_", 4) >= 4
    assert _compare("repeat_", 5) >= 5
    assert _compare("index_", 9) >= 9


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_the_three_data_dependent_ops_refuse_by_name_with_the_reason():
    """CLAUDE.md's rule that a refusal names itself, applied to the ops that
    must NEVER get a meta kernel.

    `masked_select`, `_unique2` and `repeat_interleave.Tensor` have an output
    shape that is a function of the input's VALUES. No meta kernel can exist
    for them, and upstream has none either. Before this round they fell
    through to the generic message, which says the op "would have to infer
    its output shape without computing -- which is a real kernel" -- true of
    everything else in `meta_dispatch`'s fallthrough and FALSE of these
    three, which is worse than saying nothing.
    """
    shim = _run("shim")
    if shim is None:
        return
    for key in ("masked_select_refuses", "unique_refuses", "repeat_interleave_refuses"):
        assert shim[key] == {"error": "NotImplementedError"}, (key, shim[key])
    # And the message itself, read through the dispatcher rather than the
    # transcript, so the *reason* is pinned and not just the exception class.
    import subprocess as sp
    probe = (
        "import torch\n"
        "t = torch.zeros(3, 4, device='meta')\n"
        "m = torch.zeros(3, 4, dtype=torch.bool, device='meta')\n"
        "try:\n"
        "    torch.masked_select(t, m)\n"
        "except NotImplementedError as e:\n"
        "    print(str(e))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    got = sp.run([sys.executable, "-c", probe], capture_output=True, text=True,
                 env=env, cwd=_REPO_ROOT).stdout
    assert "aten.masked_select.default" in got, got
    assert "will not have a meta kernel" in got, got
    assert "VALUES" in got, got
    assert "docs/kernels/METAEMB.md" in got, got


def test_index_tensor_answers_integer_indices_and_refuses_a_bool_mask():
    """The one PARTIAL kernel in this round, and the split is deliberate.

    With integer indices the output shape is a function of shapes alone.
    With a bool mask the extent is the number of true entries -- a property
    of the values. So the integer half answers and the bool half refuses, by
    name, which is exactly where upstream draws the same line (its bool-mask
    meta path lands in `torch.nonzero`'s refusal).
    """
    shim = _run("shim")
    if shim is None:
        return
    assert shim["index_bool_mask_refuses"] == {"error": "NotImplementedError"}, shim["index_bool_mask_refuses"]
    assert shim["index_separated_moves_front"]["shape"] == [2, 3], shim["index_separated_moves_front"]
    assert shim["index_none_then"]["shape"] == [5, 2], shim["index_none_then"]


def test_the_thirteen_upstream_self_disagreements_are_followed_to_the_dense_side():
    """"Excluded from the diff" must never mean "unchecked".

    Each key below is a call where upstream's own meta kernel and its own cpu
    kernel answer differently -- the docs/devices/META.md §7.3 shape of divergence.
    This shim follows the dense/cpu side in every one, and this test asserts
    the CHOSEN answer literally, so the exclusion list cannot quietly grow to
    cover a real disagreement.
    """
    shim = _run("shim")
    if shim is None:
        return
    upstream = _run("upstream")
    expected = {
        # cpu refuses a zero-extent reduction dim; upstream's meta answers.
        "maxdim_zero_extent": {"error": "IndexError"},
        # cpu raises IndexError; upstream's meta raises RuntimeError.
        "argmax_empty_no_dim": {"error": "IndexError"},
        # cpu refuses an out-of-range dim; upstream's meta SILENTLY ACCEPTS.
        "sort_bad_dim": {"error": "IndexError"},
        # cpu refuses a non-floating input to all three; meta accepts.
        # cpu raises RuntimeError ("'weight' must be 2-D"); upstream's meta
        # raises AssertionError from its own registration.
        "emb_w_rank1_refuses": {"error": "RuntimeError"},
        "emb_w_rank3_refuses": {"error": "RuntimeError"},
        # cpu raises NotImplementedError; upstream's meta raises RuntimeError.
        "lnorm_int_refuses": {"error": "NotImplementedError"},
        # cpu raises IndexError; upstream's meta raises RuntimeError.
        "index_too_many_refuses": {"error": "IndexError"},
        "gelu_i64_div": {"error": "NotImplementedError"},
        "gelu_bool_div": {"error": "NotImplementedError"},
        "silu_i64_div": {"error": "NotImplementedError"},
        "silu_bool_div": {"error": "NotImplementedError"},
        "relu_bool_div": {"error": "RuntimeError"},
        # cpu refuses unequal head sizes; upstream's meta answers the query's.
        "sdpa_head_mismatch": {"error": "RuntimeError"},
    }
    for key, want in expected.items():
        assert shim[key] == want, f"{key}: shim {shim[key]} != chosen {want}"
        assert shim[key] != upstream[key], (
            f"{key} is listed as an upstream self-disagreement but the shim and "
            f"upstream's meta now AGREE ({shim[key]}) -- the exclusion is stale "
            "and should be removed from _KNOWN_META_VS_CPU_DIVERGENCE"
        )
    # The two statistics divergences are dtype rather than exception.
    for key, dtype in (("lnorm_f16_stats", "torch.float16"),
                       ("lnorm_bf16_stats", "torch.bfloat16")):
        outputs = shim[key]["outputs"]
        assert outputs[1]["dtype"] == dtype, (key, outputs[1])
        assert outputs[2]["dtype"] == dtype, (key, outputs[2])
        assert upstream[key]["outputs"][1]["dtype"] == "torch.float32", (
            f"{key}: upstream's meta no longer answers float32 statistics -- "
            "this exclusion is stale"
        )
    assert set(expected) | {"lnorm_f16_stats", "lnorm_bf16_stats"} == _KNOWN_META_VS_CPU_DIVERGENCE, (
        "every excluded key must be checked here by name"
    )


def test_the_shim_wide_refusals_are_not_meta_decisions():
    """Three cases where this shim refuses something upstream accepts on both
    devices. None is a meta question -- two are the no-autograd refusal
    `embedding_default` already made, one is `matmul_default`'s "torch's
    vector rules were not measured". The meta arms follow the dense arms, and
    that is the property asserted."""
    shim = _run("shim")
    if shim is None:
        return
    assert shim["emb_sparse"] == {"error": "NotImplementedError"}, shim["emb_sparse"]
    assert shim["emb_scale_grad"] == {"error": "NotImplementedError"}, shim["emb_scale_grad"]
    assert shim["matmul_vector"] == {"error": "NotImplementedError"}, shim["matmul_vector"]
    assert set(_KNOWN_SHIM_WIDE_REFUSAL) == {"emb_sparse", "emb_scale_grad", "matmul_vector"}


def test_every_op_this_round_gave_a_meta_kernel_is_already_a_dense_op():
    """METAFAM.md §4's move, and VOICE4.md §4.1's before it: the claim "zero
    new operators, only their meta halves" is checked rather than written
    down. `_aten_implemented()` means "has a dense kernel and golden cases",
    and a meta kernel adds no values to compare -- so if any name below were
    NOT already there, this round would have added an operator without a
    golden case, which is exactly the shape of the failure
    `golden_cases_failed eq 0` was added to catch.
    """
    implemented = set(_C._aten_implemented())
    names = [
        "aten.embedding.default", "aten.gather.default",
        "aten.max.dim", "aten.min.dim", "aten.argmax.default",
        "aten.topk.default", "aten.sort.default",
        "aten.mm.default", "aten.bmm.default", "aten.addmm.default",
        "aten.matmul.default", "aten.native_layer_norm.default",
        "aten.cat.default", "aten.split.Tensor", "aten.split_with_sizes.default",
        "aten.convolution.default",
        "aten._scaled_dot_product_flash_attention_for_cpu.default",
        "aten.gelu.default", "aten.silu.default", "aten.relu.default",
        "aten.repeat.default", "aten.index.Tensor",
    ]
    missing = [n for n in names if n not in implemented]
    assert not missing, f"expected already-implemented (dense) ops, missing: {missing}"
    assert len(names) == 22, len(names)


def test_the_stride_exclusion_covers_exactly_one_named_case():
    """`_STRIDE_DIVERGENCE` is an escape hatch, so it is bounded here rather
    than left to grow. Every other case in this file compares strides."""
    assert _STRIDE_DIVERGENCE == {
        "gather_empty_idx", "topk_k0",
        "split_dim1", "split_neg_dim",
        "sdpa", "sdpa_f16_logsumexp_stays_f32", "sdpa_longer_kv",
    }
    # And the exclusion is stride-ONLY: shape and dtype are still compared
    # for every one of them, which the `_compare` calls above already do.
    # Asserted here so that "excluded" can never widen into "unchecked".
    shim, upstream = _run("shim"), _run("upstream")
    if shim is None:
        return
    for key in _STRIDE_DIVERGENCE:
        assert _drop_stride(shim[key]) == _drop_stride(upstream[key]), key


if __name__ == "__main__":
    import traceback

    failures = 0
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
        except Exception:
            failures += 1
            print(f"FAIL {name}:")
            traceback.print_exc()
    print(f"\n{len(tests) - failures} ok / {failures} fail")
    sys.exit(1 if failures else 0)
