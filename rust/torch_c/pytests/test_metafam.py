"""docs/kernels/METAFAM.md -- closing the FAMILIES docs/architectures/VOICE4.md left open.

VOICE4.md §4 gave `aten.sum.default` and `aten.view.default` meta kernels
because BigVGAN's `__init__` needed exactly those two overloads, and said so
explicitly: "이 라운드가 닫은 것은 계열이 아니라 각 계열의 한 멤버입니다" -- the round
closed two MEMBERS, not their families. `sum.dim_IntList`, `mean.dim`,
`mean.default`, `cumsum.default`, `any.default` and `amax.default` (the
reduction family) and `reshape.default`, `t.default`, `transpose.int`,
`permute.default`, `unsqueeze.default`, `squeeze.dim`, `squeeze.default` and
`slice.Tensor` (the view/shape family) sat in the same
`docs/devices/META.md` §7.4 table with no meta kernel, despite every one of them
already carrying a dense kernel and golden cases (`_aten_implemented()`
included all of them before this round -- the same finding VOICE4.md made
about `sum`/`view`, extended to their siblings).

Every kernel here calls the SAME shape/dtype helper its dense sibling already
factored out (`reduced_dims`, `sum_natural_tag`, `resolve_shape`,
`normalise_dim`, ...) rather than restating the rule -- docs/devices/META.md §7.1's
argument for why a meta kernel that disagrees with its own dense kernel is
worse than no meta kernel at all: nothing downstream can tell the difference
until a value is read, and by then the wrong shape has already propagated.

Method, unchanged from VOICE4.md §6: every comparison below runs the SAME
probe script against upstream torch 2.13.0 and against this shim, each in its
own subprocess, and asserts they agree. Shape and dtype are the whole of what
a meta kernel can promise, so that is the whole of what is compared.
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


# --------------------------------------------------------------------------
# The probe. One script, run twice (shim / upstream), each in its own process
# with the other's PYTHONPATH scrubbed -- the same shape VOICE4.md §6 uses,
# for the same reason: a meta tensor is a real `torch.Tensor` with real
# dtype objects, and building those without the whole of `torch` imported is
# more machinery than reusing it.
# --------------------------------------------------------------------------

_PROBE = r"""
import json, sys
import torch

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}
DTYPES = {
    "f32": torch.float32, "f64": torch.float64, "f16": torch.float16,
    "bf16": torch.bfloat16, "i64": torch.int64, "i32": torch.int32,
    "i16": torch.int16, "u8": torch.uint8, "bool": torch.bool,
}


def rec(name, fn):
    try:
        t = fn()
    except Exception as e:
        out[name] = {"error": type(e).__name__}
    else:
        if isinstance(t, tuple):
            out[name] = {
                "shape": list(t[0].shape) if hasattr(t[0], "shape") else t[0],
            }
        else:
            out[name] = {
                "shape": list(t.shape), "dtype": str(t.dtype), "is_meta": bool(t.is_meta),
            }


def m(*shape, dtype=torch.float32):
    return torch.empty(*shape, dtype=dtype, device="meta")


# --- reduction family -------------------------------------------------------

for tag, dt in DTYPES.items():
    rec(f"sumdim_{tag}", lambda dt=dt: m(2, 3, dtype=dt).sum(dim=1))
rec("sumdim_keepdim", lambda: m(2, 3, 4).sum(dim=1, keepdim=True))
rec("sumdim_multi", lambda: m(2, 3, 4).sum(dim=[0, 2]))
rec("sumdim_neg", lambda: m(2, 3, 4).sum(dim=-1))
rec("sumdim_empty_list", lambda: m(2, 3).sum(dim=[]))
rec("sumdim_explicit_dtype", lambda: m(2, 3, dtype=torch.int32).sum(dim=0, dtype=torch.float64))
rec("sumdim_rank0", lambda: m().sum(dim=[]))

rec("meandim_f32", lambda: m(2, 3).mean(dim=1))
rec("meandim_f64", lambda: m(2, 3, dtype=torch.float64).mean(dim=1))
rec("meandim_f16", lambda: m(2, 3, dtype=torch.float16).mean(dim=1))
rec("meandim_bf16", lambda: m(2, 3, dtype=torch.bfloat16).mean(dim=1))
rec("meandim_keepdim", lambda: m(2, 3, 4).mean(dim=[1, 2], keepdim=True))
rec("meandim_int_refuses", lambda: m(2, 3, dtype=torch.int64).mean(dim=0))
rec("meandim_bool_refuses", lambda: m(2, 3, dtype=torch.bool).mean(dim=0))

rec("meandefault_f32", lambda: m(2, 3).mean())
rec("meandefault_f64", lambda: m(2, 3, dtype=torch.float64).mean())
rec("meandefault_int_refuses", lambda: m(2, 3, dtype=torch.int64).mean())
rec("meandefault_rank0", lambda: m().mean())

for tag, dt in DTYPES.items():
    rec(f"cumsum_{tag}", lambda dt=dt: m(2, 3, dtype=dt).cumsum(dim=0))
rec("cumsum_neg_dim", lambda: m(2, 3, 4).cumsum(dim=-1))
rec("cumsum_rank1", lambda: m(5).cumsum(dim=0))
rec("cumsum_explicit_dtype", lambda: m(2, 3, dtype=torch.int32).cumsum(dim=0, dtype=torch.float32))

for tag, dt in DTYPES.items():
    rec(f"any_{tag}", lambda dt=dt: torch.any(m(2, 3, dtype=dt)))
rec("any_rank0", lambda: torch.any(m()))
rec("any_empty", lambda: torch.any(m(0, 3)))

for tag, dt in DTYPES.items():
    rec(f"amax_{tag}", lambda dt=dt: m(2, 3, dtype=dt).amax(dim=1))
rec("amax_keepdim", lambda: m(2, 3, 4).amax(dim=[0, 2], keepdim=True))
rec("amax_neg_dim", lambda: m(2, 3).amax(dim=-1))
rec("amax_dup_dim_refuses", lambda: m(2, 3).amax(dim=[0, 0]))
rec("amax_no_dim_empty_refuses", lambda: m(0, 3).amax())
rec("amax_dim_on_nonzero_of_empty", lambda: m(0, 3).amax(dim=1))
# NOT compared against upstream below (see test_amax_...): upstream's OWN
# `meta` kernel answers a different exception class than its `cpu` kernel
# does for this exact call (RuntimeError vs IndexError, measured) -- the
# same shape of divergence META.md §7.3 already catalogues for
# `bitwise_not`/`clamp`/`where`. This shim has one door, so it follows its
# own dense kernel (`amax_default`, IndexError) rather than upstream's
# meta-specific wording.
rec("amax_dim_on_zero_extent", lambda: m(0, 3).amax(dim=0))

# --- view/shape family -------------------------------------------------------

for tag, dt in DTYPES.items():
    rec(f"reshape_{tag}", lambda dt=dt: m(2, 3, dtype=dt).reshape(3, 2))
rec("reshape_wildcard", lambda: m(2, 3, 4).reshape(4, -1))
rec("reshape_to_rank0", lambda: m(1).reshape(()))
rec("reshape_bad_numel", lambda: m(2, 3).reshape(4, 2))
rec("reshape_two_wildcards", lambda: m(2, 3).reshape(-1, -1))
rec("reshape_empty", lambda: m(0, 3).reshape(0, 3))

for tag, dt in DTYPES.items():
    rec(f"t_{tag}", lambda dt=dt: m(2, 3, dtype=dt).t())
rec("t_rank0", lambda: m().t())
rec("t_rank1", lambda: m(5).t())
rec("t_rank3_refuses", lambda: m(2, 3, 4).t())

rec("transpose_basic", lambda: m(2, 3, 4).transpose(0, 2))
rec("transpose_neg", lambda: m(2, 3, 4).transpose(-1, -2))
rec("transpose_same_dim_noop", lambda: m(2, 3).transpose(0, 0))
rec("transpose_out_of_range_refuses", lambda: m(2, 3).transpose(0, 5))

rec("permute_basic", lambda: m(2, 3, 4).permute(2, 0, 1))
rec("permute_neg", lambda: m(2, 3, 4).permute(-1, -3, -2))
rec("permute_rank0", lambda: m().permute())
rec("permute_dup_refuses", lambda: m(2, 3, 4).permute(0, 0, 1))
rec("permute_wrong_len_refuses", lambda: m(2, 3).permute(0, 1, 2))

rec("unsqueeze_mid", lambda: m(2, 3).unsqueeze(1))
rec("unsqueeze_neg", lambda: m(2, 3).unsqueeze(-1))
rec("unsqueeze_last", lambda: m(2, 3).unsqueeze(2))
rec("unsqueeze_out_of_range_refuses", lambda: m(2, 3).unsqueeze(3))

rec("squeezedim_removes", lambda: m(1, 3).squeeze(0))
rec("squeezedim_noop", lambda: m(2, 3).squeeze(0))
rec("squeezedim_neg", lambda: m(3, 1).squeeze(-1))
rec("squeezedim_rank0", lambda: m().squeeze(0))

rec("squeezedefault_multi", lambda: m(1, 3, 1, 2).squeeze())
rec("squeezedefault_none", lambda: m(2, 3).squeeze())
rec("squeezedefault_all_ones", lambda: m(1, 1).squeeze())

for tag, dt in DTYPES.items():
    rec(f"slice_{tag}", lambda dt=dt: m(6, 3, dtype=dt)[1:4])
rec("slice_default", lambda: m(6, 3)[:])
rec("slice_neg_indices", lambda: m(6, 3)[-4:-1])
rec("slice_step", lambda: m(6, 3)[0:6:2])
rec("slice_out_of_range_clamps", lambda: m(6, 3)[2:100])
rec("slice_empty_result", lambda: m(6, 3)[4:2])
rec("slice_dim1", lambda: m(6, 3)[:, 1:2])

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
    assert proc.returncode == 0, f"the {side} metafam probe failed:\n{proc.stderr[-4000:]}"
    data = json.loads(proc.stdout)
    assert data["_marker"] == side, (
        f"the {side} metafam probe imported the wrong torch ({data['_marker']})"
    )
    _cache[side] = data
    return data


# Cases excluded from the strict shim-equals-upstream comparison because
# upstream's OWN `meta` kernel disagrees with its OWN `cpu` kernel on them
# (measured) -- the META.md §7.3 shape of divergence. This shim has one
# door and follows its dense kernel, so it is checked separately, by name,
# in the test that owns it (`test_amax_...`), rather than silently excluded.
_KNOWN_META_VS_CPU_DIVERGENCE = {"amax_dim_on_zero_extent"}


def _compare(prefix, min_cases):
    shim = _run("shim")
    if shim is None:
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return 0
    upstream = _run("upstream")
    keys = sorted(
        k for k in upstream
        if k.startswith(prefix) and k not in _KNOWN_META_VS_CPU_DIVERGENCE
    )
    assert keys, f"the probe recorded no {prefix}* cases"
    diffs = []
    for key in keys:
        if shim.get(key) != upstream[key]:
            diffs.append(f"{key}: shim {shim.get(key)} != upstream {upstream[key]}")
    assert not diffs, "\n".join(diffs)
    assert len(keys) >= min_cases, f"only {len(keys)} {prefix}* cases were compared"
    return len(keys)


# --------------------------------------------------------------------------
# Reduction family
# --------------------------------------------------------------------------


def test_sum_dim_int_list_answers_what_upstream_answers():
    """Shape drops the reduced axes (or keeps them at size 1); dtype is
    `sum_natural_tag` -- the SAME function `aten.sum.default`'s meta kernel
    already calls, per docs/devices/META.md §7.1."""
    _compare("sumdim_", 10)


def test_mean_dim_answers_what_upstream_answers_and_refuses_non_floating():
    _compare("meandim_", 6)


def test_mean_default_answers_what_upstream_answers():
    """The whole-tensor sibling of `sum.default`: rank-0 shape, but `mean`
    refuses an integral input rather than widening it like `sum` does."""
    _compare("meandefault_", 4)


def test_cumsum_default_keeps_shape_and_widens_dtype_like_sum():
    """`cumsum` does not reduce rank -- the shape is unchanged -- but the
    dtype rule is `sum`'s own (`sum_natural_tag`, integral widens to int64)."""
    _compare("cumsum_", 9)


def test_any_default_is_rank_0_bool():
    _compare("any_", 9)


def test_amax_preserves_dtype_and_refuses_the_empty_reductions_dense_does():
    """Unlike `sum`, `amax` does NOT widen the dtype -- and it reproduces the
    two zero-numel refusals `amax_default`'s dense kernel already raises,
    because a meta tensor's extents can be zero too and there is no dense
    call afterwards to catch it."""
    _compare("amax_", 10)
    shim = _run("shim")
    if shim is None:
        return
    # The one excluded case, checked against the shim's OWN dense kernel's
    # answer rather than upstream's meta-specific one -- see
    # `_KNOWN_META_VS_CPU_DIVERGENCE`.
    assert shim["amax_dim_on_zero_extent"] == {"error": "IndexError"}


# --------------------------------------------------------------------------
# View/shape family
# --------------------------------------------------------------------------


def test_reshape_default_answers_what_upstream_answers_including_the_numel_check():
    """`reshape` was the one META.md §7.4 explicitly kept refused after
    VOICE4.md's round, on the grounds that it may copy where `view` never
    does. On a meta tensor that distinction is not observable -- this shim's
    meta tensors carry no strides at all -- so shape and dtype (the whole of
    what either kernel can promise) are identical, and `reshape` now answers
    from the same `resolve_shape` helper `view` already uses."""
    _compare("reshape_", 8)


def test_t_default_answers_what_upstream_answers_including_the_rank3_refusal():
    _compare("t_", 10)


def test_transpose_int_swaps_the_named_axes():
    _compare("transpose_", 3)


def test_permute_default_reorders_and_refuses_what_dense_refuses():
    _compare("permute_", 4)


def test_unsqueeze_default_inserts_at_every_legal_position():
    _compare("unsqueeze_", 3)


def test_squeeze_dim_is_a_noop_on_a_non_1_axis():
    """The one thing to get right, per `squeeze_dim`'s own comment: a
    dimension whose size is not 1 is unchanged, not refused."""
    _compare("squeezedim_", 3)


def test_squeeze_default_removes_every_size_1_axis():
    _compare("squeezedefault_", 2)


def test_slice_tensor_answers_what_upstream_answers():
    _compare("slice_", 10)


# --------------------------------------------------------------------------
# The manifest claim itself, asserted rather than only written down --
# docs/architectures/VOICE4.md §4.1's move, extended to every op this round adds.
# --------------------------------------------------------------------------

_NEW_META_OPS = (
    "aten.sum.dim_IntList",
    "aten.mean.dim",
    "aten.mean.default",
    "aten.cumsum.default",
    "aten.any.default",
    "aten.amax.default",
    "aten.reshape.default",
    "aten.t.default",
    "aten.transpose.int",
    "aten.permute.default",
    "aten.unsqueeze.default",
    "aten.squeeze.dim",
    "aten.squeeze.default",
    "aten.slice.Tensor",
)


def test_the_new_meta_kernels_are_the_meta_half_of_ops_already_implemented():
    """Every op above was in `_aten_implemented()` before this round -- it had
    a dense kernel and golden cases already. `_aten_implemented()` does not
    change size here for the same reason VOICE4.md §7 gives: meta support is
    a property of an op already on the list, not a new op."""
    implemented = set(_C._aten_implemented())
    missing = [op for op in _NEW_META_OPS if op not in implemented]
    assert not missing, f"expected already-implemented (dense) ops, missing: {missing}"


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
