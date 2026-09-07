"""Run a fixed battery of aten ops and emit every result as raw IEEE-754 bits.

Runs unmodified on the host and on an Android device; `device_android.sh
parity` diffs the two JSON documents. The point is that the comparison is on
*bit patterns*, not on repr strings and not on a tolerance: `aarch64` on both
ends makes bit equality a reasonable thing to test for, but only a measurement
can say whether it holds. Several ops here deliberately route through libm
(`cos`, `sin`, `tanh`, `gelu`, `silu`, `rsqrt`, `pow`) because the host links
Apple's libm and the device links bionic's, and those are different
implementations of the same specification.

That bet paid, on the original 33-case battery: 30 of 32 comparable cases came
out bit-identical, and the two that did not are `tanh.default` and
`_softmax.default` (whose kernel calls `expf`) -- by exactly 1 ULP. `cos`,
`sin`, `gelu`, `silu`, `rsqrt` and `pow` were identical, so the divergence is
per-function, not a blanket libm effect. docs/devices/DEVICE.md records that run and
the reference-value analysis, and the runs after each battery expansion.

Inputs avoid exactly-representable values on purpose. `1.5 + 2.25 = 3.75` is
bit-identical under any arithmetic that is not actively broken, so it proves
nothing; the tensors here are built from `arange`-derived fractions that do not
terminate in binary.

The battery grows as ops land that this file never ran on real hardware --
`_aten_implemented()` reached 116 while this file still tested 33 (mamba and
mixtral's 12, plus a `cat`/`max` semantics fix, went un-verified on device for
days). New cases below target what a bigger battery is *for*: an edge each op
is specifically known to get wrong if it is not careful (`cat`'s legacy-empty
skip, `max`'s NaN propagation, a matmul size that crossed the gemm threading
threshold in 25a79df) rather than another easy interior point. Uninitialized-
memory ops (`empty`, `empty_like`) are deliberately absent -- there is no bit
pattern to compare, the same reasoning tools/golden/cases.py's
`_dtype_shape_only_check` encodes for the golden suite.

Judgement is by exit code and by the emitted JSON, never by scraping stdout for
a success word (IMPORT_WALLS 2차 lost a round to a traceback that echoed its own
source line).

Usage:
    python device_parity.py <output.json>
"""

from __future__ import annotations

import json
import os
import struct
import sys
import traceback


def _install_android_stubs() -> bool:
    """Stand in for the two extension modules Android's CPython does not ship.

    This is an instrument, not a fix. The Android CPython distribution builds
    neither `_multiprocessing` nor `_posixshmem` -- Android has no SysV IPC and
    no usable POSIX named semaphores -- and `torch/multiprocessing/__init__.py`
    imports `multiprocessing.resource_tracker` unconditionally at import time.

    `_multiprocessing` is left empty because `resource_tracker.py:49` guards it
    with `hasattr(_multiprocessing, 'sem_unlink')`, and absence is the honest
    answer: a build with no named semaphores has none to clean up.
    `_posixshmem.shm_unlink` is read *unguarded* at `resource_tracker.py:54`, so
    that one name must exist. It is wired to raise rather than to no-op, so a
    real use fails loudly instead of silently leaking.
    """
    if os.environ.get("BW_STUB_MULTIPROCESSING") != "1":
        return False
    import types

    def _unavailable(*_args, **_kwargs):
        raise OSError("shared memory is unavailable on Android")

    sys.modules["_multiprocessing"] = types.ModuleType("_multiprocessing")
    shm = types.ModuleType("_posixshmem")
    shm.shm_unlink = _unavailable
    shm.shm_open = _unavailable
    sys.modules["_posixshmem"] = shm
    return True


STUBBED = _install_android_stubs()

import torch  # noqa: E402  -- must follow the stub install


def _flatten(nested):
    if isinstance(nested, list):
        for item in nested:
            yield from _flatten(item)
    else:
        yield nested


def encode(tensor) -> dict:
    """Describe a tensor by shape, dtype and the exact bits of every element.

    Floats go out as big-endian IEEE-754 hex so that a diff of two JSON files is
    a diff of bit patterns. `float('nan')` and infinities survive this encoding,
    which `repr` and `json` do not both do.
    """
    values = list(_flatten(tensor.tolist()))
    dtype = str(tensor.dtype)
    if dtype in ("torch.float32", "torch.float16", "torch.bfloat16"):
        encoded = [struct.pack(">f", float(v)).hex() for v in values]
    elif dtype == "torch.float64":
        encoded = [struct.pack(">d", float(v)).hex() for v in values]
    elif dtype == "torch.bool":
        encoded = [bool(v) for v in values]
    else:
        encoded = [int(v) for v in values]
    return {"shape": list(tensor.shape), "dtype": dtype, "bits": encoded}


# Inputs whose binary expansions do not terminate, so that rounding is exercised.
def _inputs():
    # Spelled with explicit `.<overload>` throughout: the shim's bare
    # `torch.div(...)` raises NotImplementedError because overload resolution
    # has no table entry for it, and the message points here.
    base = torch.ops.aten.view.default(torch.ops.aten.arange.start_step(0.0, 12.0, 1.0), [3, 4])
    ones = torch.ops.aten.full.default([3, 4], 1.0)
    sevens = torch.ops.aten.full.default([3, 4], 7.0)
    threes = torch.ops.aten.full.default([3, 4], 3.0)
    fives = torch.ops.aten.full.default([3, 4], 5.0)
    a = torch.ops.aten.div.Tensor(torch.ops.aten.add.Tensor(base, ones), sevens)
    b = torch.ops.aten.div.Tensor(torch.ops.aten.sub.Tensor(fives, base), threes)
    return a, b


CASES = {}


def case(name):
    def register(fn):
        CASES[name] = fn
        return fn

    return register


@case("add.Tensor")
def _(a, b):
    return torch.ops.aten.add.Tensor(a, b)


@case("sub.Tensor")
def _(a, b):
    return torch.ops.aten.sub.Tensor(a, b)


@case("mul.Tensor")
def _(a, b):
    return torch.ops.aten.mul.Tensor(a, b)


@case("div.Tensor")
def _(a, b):
    return torch.ops.aten.div.Tensor(a, b)


@case("mm.default")
def _(a, b):
    return torch.ops.aten.mm.default(a, torch.ops.aten.t.default(b))


@case("addmm.default")
def _(a, b):
    return torch.ops.aten.addmm.default(
        torch.ops.aten.full.default([3, 3], 0.125),
        a,
        torch.ops.aten.t.default(b),
    )


@case("bmm.default")
def _(a, b):
    left = torch.ops.aten.unsqueeze.default(a, 0)
    right = torch.ops.aten.unsqueeze.default(torch.ops.aten.t.default(b), 0)
    return torch.ops.aten.bmm.default(left, right)


# --- libm-backed: the ops most likely to diverge between Apple libm and bionic.
@case("cos.default")
def _(a, b):
    return torch.ops.aten.cos.default(a)


@case("sin.default")
def _(a, b):
    return torch.ops.aten.sin.default(a)


@case("tanh.default")
def _(a, b):
    return torch.ops.aten.tanh.default(b)


@case("gelu.default")
def _(a, b):
    return torch.ops.aten.gelu.default(b)


@case("silu.default")
def _(a, b):
    return torch.ops.aten.silu.default(b)


@case("rsqrt.default")
def _(a, b):
    return torch.ops.aten.rsqrt.default(a)


@case("reciprocal.default")
def _(a, b):
    return torch.ops.aten.reciprocal.default(a)


@case("pow.Tensor_Scalar")
def _(a, b):
    return torch.ops.aten.pow.Tensor_Scalar(a, 2.5)


@case("relu.default")
def _(a, b):
    return torch.ops.aten.relu.default(b)


# --- reductions: order of accumulation shows up here if it differs at all.
@case("sum.default")
def _(a, b):
    return torch.ops.aten.sum.default(a)


@case("sum.dim_IntList (dim=[])")
def _(a, b):
    # The fix docs/graph/DECOMP.md §6.1 records: an empty `dim` list means "reduce
    # every dimension", not "reduce none". `rust/torch_c/src/aten.rs` is
    # ordinary CPU-only Rust with no platform-conditional code in this
    # function, so there is no reason to expect the device to disagree with
    # the host here -- this case exists to check that expectation rather
    # than assume it.
    return torch.ops.aten.sum.dim_IntList(a, [], False)


@case("mean.dim")
def _(a, b):
    return torch.ops.aten.mean.dim(a, [1], False, None)


@case("cumsum.default")
def _(a, b):
    return torch.ops.aten.cumsum.default(a, 1, None)


@case("_softmax.default")
def _(a, b):
    return torch.ops.aten._softmax.default(a, 1, False)


@case("native_layer_norm.default")
def _(a, b):
    out = torch.ops.aten.native_layer_norm.default(
        a,
        [4],
        torch.ops.aten.full.default([4], 1.25),
        torch.ops.aten.full.default([4], -0.5),
        1e-5,
    )
    return out[0]


@case("max.dim")
def _(a, b):
    return torch.ops.aten.max.dim(b, 1, False)[0]


@case("argmax.default")
def _(a, b):
    return torch.ops.aten.argmax.default(b, 1, False)


@case("topk.default")
def _(a, b):
    return torch.ops.aten.topk.default(b, 2, 1, True, True)[0]


@case("sort.default")
def _(a, b):
    return torch.ops.aten.sort.default(b, 1, False)[0]


# --- shape/index ops, where a bug is a wrong element rather than a wrong bit.
@case("cat.default")
def _(a, b):
    return torch.ops.aten.cat.default([a, b], 0)


@case("permute.default")
def _(a, b):
    return torch.ops.aten.permute.default(a, [1, 0])


@case("slice.Tensor")
def _(a, b):
    return torch.ops.aten.slice.Tensor(a, 1, 1, 3, 1)


@case("embedding.default")
def _(a, b):
    return torch.ops.aten.embedding.default(a, torch.tensor([2, 0, 1]), -1, False, False)


@case("where.self")
def _(a, b):
    return torch.ops.aten.where.self(torch.ops.aten.lt.Scalar(b, 0.0), a, b)


@case("nn.Linear forward")
def _(a, b):
    import torch.nn as nn

    layer = nn.Linear(4, 3)
    with torch.no_grad():
        layer.weight.copy_(b[:, :4] if b.shape[1] >= 4 else b)
        layer.bias.copy_(torch.tensor([0.125, -0.375, 0.625]))
    return layer(a)


@case("nn.ReLU forward")
def _(a, b):
    # Recorded because it fails, and it fails the same way on both ends. The
    # module calls `torch.relu(...)` (no overload suffix) via `F.relu`, and the
    # shim's overload table has no entry for that spelling. It is a gap in
    # `rust/torch_c/src/overloads.json`, not a device problem -- which is
    # exactly what running it on both host and device establishes.
    import torch.nn as nn

    return nn.ReLU()(b)


@case("nn.Sequential 2-layer")
def _(a, b):
    import torch.nn as nn

    # `nn.Tanh` rather than `nn.ReLU`: see the `nn.ReLU forward` case above.
    net = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2))
    with torch.no_grad():
        net[0].weight.copy_(b)
        net[0].bias.copy_(torch.tensor([0.125, -0.375, 0.625]))
        net[2].weight.copy_(torch.tensor([[0.5, -0.25, 0.75], [-1.5, 0.125, 0.25]]))
        net[2].bias.copy_(torch.tensor([-0.5, 0.25]))
    return net(a)


# --- expansion: the "repr" kernels (docs/graph/DECOMP.md's naming) -----------------
#
# `abs`, `ceil`, `gt`, `masked_select`, `min` and `unbind` landed together and
# were never run on the device before this expansion.


@case("abs.default")
def _(a, b):
    return torch.ops.aten.abs.default(b)


@case("ceil.default")
def _(a, b):
    return torch.ops.aten.ceil.default(a)


@case("gt.Scalar")
def _(a, b):
    return torch.ops.aten.gt.Scalar(a, 1.0)


@case("masked_select.default")
def _(a, b):
    mask = torch.ops.aten.lt.Scalar(b, 0.0)
    return torch.ops.aten.masked_select.default(b, mask)


@case("min.default")
def _(a, b):
    return torch.ops.aten.min.default(a)


@case("unbind.int")
def _(a, b):
    # `unbind` hands back a tuple of tensors, not one -- `encode()` only
    # knows how to describe a single tensor, so this reads the first row.
    # Same pattern as `max.dim`/`topk.default`/`sort.default` above.
    return torch.ops.aten.unbind.int(a, 0)[0]


@case("max.default")
def _(a, b):
    # The bug this pins: candle's reduction used to *skip* NaN, so
    # `max([3, nan, 1])` came back `3.0` where upstream (and IEEE `maximum`)
    # says the result is `nan`. Fixed on the host; never run on device.
    # `min.default` shares the same reduction path -- `min.default` above
    # covers the ordinary case, this covers the propagation rule.
    nan_input = torch.tensor([3.0, float("nan"), 1.0, 0.5, -2.0, 4.0], dtype=torch.float32)
    return torch.ops.aten.max.default(nan_input)


@case("cat.default (empty)")
def _(a, b):
    # torch's "legacy empty" rule: a 1-D tensor of shape exactly (0,) is
    # skipped by `cat` regardless of the other operands' rank; anything else
    # empty (e.g. shape (0, 5)) is NOT exempt. `docs/models/E2E_REAL.md` -- every
    # transformers KV cache's first decoder step concatenates against
    # exactly this shape, so getting the *rule*, not just the ordinary case
    # `cat.default` above already covers, right matters on device too.
    empty = torch.ops.aten.full.default([0], 1.0)
    return torch.ops.aten.cat.default([empty, a], 0)


# --- expansion: mamba's and mixtral's 12 ops (never run on device) -----------


@case("clamp_.default")
def _(a, b):
    # mixtral's exact call shape: max only, min absent. Clone first --
    # in-place, and `a`/`b` are shared across every case in this file.
    x = torch.ops.aten.clone.default(a)
    return torch.ops.aten.clamp_.default(x, None, 3.0)


@case("convolution.default")
def _(a, b):
    # mamba's exact shape: depthwise causal 1-D conv -- groups == in_channels
    # == out_channels, padding == kernel_size - 1. Values from
    # tools/golden/cases.py::convolution_cases, the same fixture already
    # measured against upstream.
    x = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 0.5, 2.0, -2.0, 1.0, 0.0, 1.0, -1.0, 2.0, 3.0]
    ).reshape(1, 3, 5)
    w = torch.tensor(
        [1.0, -1.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, -1.0, 1.0, 0.0, 2.0]
    ).reshape(3, 1, 4)
    bias = torch.tensor([0.1, -0.2, 0.3])
    return torch.ops.aten.convolution.default(x, w, bias, [1], [3], [1], False, [0], 3)


@case("div_.Tensor")
def _(a, b):
    # mixtral's exact call shape: top_k_weights.div_(top_k_weights.sum(-1,
    # keepdim=True)). Clone first -- in-place.
    x = torch.ops.aten.clone.default(a)
    denom = torch.ops.aten.sum.dim_IntList(x, [-1], True)
    return torch.ops.aten.div_.Tensor(x, denom)


@case("exp.default")
def _(a, b):
    return torch.ops.aten.exp.default(a)


@case("floor_divide.default")
def _(a, b):
    return torch.ops.aten.floor_divide.default(a, b)


@case("ge.Scalar")
def _(a, b):
    return torch.ops.aten.ge.Scalar(a, 1.0)


@case("histc.default")
def _(a, b):
    # mixtral's exact call shape: bins=num_experts, min=0, max=num_experts-1.
    return torch.ops.aten.histc.default(a, 4, 0, 3)


@case("index_put_.default")
def _(a, b):
    # mixtral's exact call shape: inv_perm[perm] = torch.arange(perm.size(0)).
    # A single int64 index tensor, no accumulate. Freshly built, not a
    # clone of `a`/`b` -- nothing here is shared with another case.
    self_ = torch.ops.aten.full.default([5], 0.0)
    index = torch.tensor([4, 3, 2, 1, 0], dtype=torch.int64)
    values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    return torch.ops.aten.index_put_.default(self_, [index], values, False)


@case("masked_fill_.Scalar")
def _(a, b):
    # mixtral's exact shape: a (N,1) sentinel mask broadcasts into an
    # (N,hidden_dim) receiver, filled with a large negative sentinel. Clone
    # first -- in-place.
    x = torch.ops.aten.clone.default(a)
    mask = torch.tensor([[True], [False], [True]])
    return torch.ops.aten.masked_fill_.Scalar(x, mask, -1e9)


@case("softplus.default")
def _(a, b):
    return torch.ops.aten.softplus.default(a)


@case("zeros_like.default")
def _(a, b):
    return torch.ops.aten.zeros_like.default(a)


# --- expansion: the gemm threading threshold (25a79df) ------------------------
#
# Correctness only -- CLAUDE.md says not to measure performance on a shared,
# loaded emulator, and this script never has. 25a79df raised the threading
# crossover from 589,824 to 4,000,000 total multiply-adds; n=128 (2,097,152)
# sits between the two, so a matmul this size took the threaded path before
# and takes the sequential path now. The commit argued the answer cannot
# change (columns split between threads, k-accumulation stays whole) and
# checked it by hashing host output at three thresholds -- this is the
# device-side half of that same claim, on the artefact that actually shipped
# the change.
@case("mm.default (n=128, gemm threading threshold)")
def _(a, b):
    n = 128
    x = torch.ops.aten.view.default(
        torch.ops.aten.div.Tensor(
            torch.ops.aten.arange.start_step(0.0, float(n * n), 1.0),
            torch.ops.aten.full.default([n * n], 7.0),
        ),
        [n, n],
    )
    y = torch.ops.aten.view.default(
        torch.ops.aten.div.Tensor(
            torch.ops.aten.arange.start_step(0.0, float(n * n), 1.0),
            torch.ops.aten.full.default([n * n], 5.0),
        ),
        [n, n],
    )
    return torch.ops.aten.mm.default(x, torch.ops.aten.t.default(y))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: device_parity.py <output.json>", file=sys.stderr)
        return 2

    report = {
        "platform": sys.platform,
        "machine": os.uname().machine,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "c_extension": torch._C.__file__,
        "aten_implemented": len(torch._C._aten_implemented()),
        "stubbed_multiprocessing": STUBBED,
        "results": {},
        "failures": {},
    }

    a, b = _inputs()
    for name, fn in CASES.items():
        try:
            report["results"][name] = encode(fn(a, b))
        except BaseException as exc:  # noqa: BLE001 -- a failing case is data
            report["failures"][name] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    with open(sys.argv[1], "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)

    print(f"cases={len(CASES)} ok={len(report['results'])} failed={len(report['failures'])}")
    for name, err in report["failures"].items():
        print(f"  FAIL {name}: {err}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
