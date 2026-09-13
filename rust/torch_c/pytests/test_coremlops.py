"""What the CoreML arm lowers beyond `Linear`, and what each type costs.

docs/graph/NPU2.md landed `to(torchnative.device.npu)` for **one** leaf type.
`_compile_model`'s own docstring named the obstacle for the next one: a conv
leaf's MIL program cannot be built without its input's spatial dimensions, and
those are not knowable at `to()` time, so conv was left on the CPU and said so.

This round answers that with a measurement rather than a guess, and the
measurement decides which types are worth lowering at all. `MLComputePlan` was
read at float16 for every candidate op (docs/graph/NPU2.md §7):

    conv          preferred NeuralEngine from 128->256 at 32x32 upward
    linear        preferred NeuralEngine at 128x1024          (already shipped)
    layer_norm    supported NeuralEngine, preferred CPU/GPU at every size tried
    relu, gelu    supported NeuralEngine, preferred CPU/GPU at every size tried
    softmax       supported NeuralEngine, preferred CPU/GPU at every size tried
    max_pool      supported NeuralEngine, preferred CPU/GPU at every size tried
    gather        NeuralEngine not in the supported column at all

So **conv is the one that reaches the unit**, and the memory-bound leaves are
refused with that number rather than shipped because they compiled. An op
lowered to CoreML that CoreML then runs on the CPU is a tensor round trip
bought for nothing, and docs/graph/NPU2.md §1 is the record of exactly that
going unnoticed.

The shape obstacle generalises rather than blocking: `_CoreMLLinear` already
compiles per shape and caches, because batch is free there. For conv the free
dimensions are `(N, H, W)` instead of `(N,)`, which makes the cache key the
whole input shape and nothing else changes -- except that the `to()`-time
eager probe cannot run, because batch 1 is a shape this library can choose and
a spatial size is not. A conv leaf is therefore **deferred**, and that is on
the report by name instead of being a surprise.

Skips say by name what is missing, for docs/devices/VULKAN3.md §6.1's reason.
"""

import os

from test_shim import _CKPT_VENDOR_SHIM, _npu_fixture


_COREMLOPS_SCRIPT = r"""
import json
import os
import warnings

import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}

try:
    import coremltools as ct
    out["coremltools"] = ct.__version__
except Exception as error:
    out["coremltools"] = None
    out["import_error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(out))
    raise SystemExit(0)

import numpy as np

import torchnative
from torchnative import device as tn_device
from torchnative.export import coreml as C
from torchnative.export.decompose import DecomposedTrace


def capture(fn, *inputs):
    torch._C._capture_begin(list(inputs))
    with torch.no_grad():
        produced = fn(*inputs)
    produced = produced if isinstance(produced, (list, tuple)) else [produced]
    trace = torch._C._capture_end(list(produced))
    return DecomposedTrace(
        trace.guards, trace.constants, trace.constant_values,
        trace.nodes, trace.outputs,
    )


_OURS = ("torchnative ", "nn.Module.to(")


def ours(caught):
    return [str(w.message) for w in caught if str(w.message).startswith(_OURS)]


# The size the measurement chose, not the smallest one that compiles.
# docs/graph/NPU2.md 7 read MLComputePlan across conv sizes: at
# (1, 64, 32, 32) with 64->128 channels CoreML still *prefers* the CPU, and at
# 128->256 it prefers the Neural Engine. Both list the unit as supported. So
# "preferred == NeuralEngine" only means something at a size above that
# threshold, and choosing the size deliberately is the difference between a
# test that measures the unit and one that measures which conv was typed first.
class Conv(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.c = torch.nn.Conv2d(128, 256, 3, padding=1, **kwargs)

    def forward(self, x):
        return self.c(x)


class ConvNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.n = torch.nn.LayerNorm(32)

    def forward(self, x):
        return self.n(self.c(x))


class ActOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.ReLU()
        self.norm = torch.nn.LayerNorm(32)

    def forward(self, x):
        return self.norm(self.act(x))


try:
    torch.manual_seed(0)
    out["resolution"] = {
        "backend": tn_device.npu.resolve().backend,
        "unit": tn_device.npu.resolve().unit,
    }

    image = torch.randn(1, 128, 32, 32)

    # -- 1. the selection, with no CoreML involved -------------------------
    out["plan_lowering_conv_norm"] = C.plan_lowering(ConvNorm())
    out["plan_lowering_reflect"] = C.plan_lowering(
        Conv(padding_mode="reflect"))

    # -- 2. to(device.npu) on a conv, at each precision --------------------
    for label, kwargs in (("float16", {}), ("float32", {"precision": "float32"})):
        model = Conv().eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            returned = model.to(tn_device.npu, **kwargs)
        entry = {
            "returned_is_self": returned is model,
            "leaf_type": type(model.c).__name__,
            "warnings_at_to": ours(caught),
            "deferred": model.torchnative_offload["deferred"],
            "plans_at_to": list(model.torchnative_offload["plans"]),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            produced = model(image)
        entry["warnings_at_forward"] = ours(caught)
        report = model.torchnative_offload
        entry["plans"] = report["plans"]
        entry["fully_offloaded"] = report["fully_offloaded"]
        entry["fraction_moved"] = report["fraction_moved"]
        entry["output_shape"] = list(produced.shape)
        reference = Conv().eval()
        reference.load_state_dict({
            "c.weight": model.c.weight, "c.bias": model.c.bias})
        with torch.no_grad():
            entry["max_abs_diff_vs_torch"] = float(
                (produced - reference(image)).abs().max().item())
        out.setdefault("lowered", {})[label] = entry

    # -- 3. the agreement grade for conv, through coreml.verify ------------
    trace = capture(Conv().eval(), image)
    for label, float32 in (("float16", False), ("float32", True)):
        out.setdefault("verify_conv", {})[label] = C.verify(
            trace, [image], float32=float32, tolerance=1.0,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

    # -- 4. the types that are NOT lowered, and the number that decided ----
    #
    # relu and gelu have MIL lowerings here and CoreML lists the Neural Engine
    # as supported for both -- and prefers something else at every size. That
    # is the measurement behind leaving them on the CPU, taken through the
    # same `compute_plan` the lowered types are graded by.
    # 256x1024, not the 1024x4096 the sweep in docs/graph/NPU2.md §7 used.
    # `_np` marshals through `tolist()`, so a four-million-element tensor
    # becomes four million Python floats, and this fixture process reproducibly
    # **segfaulted at interpreter shutdown** after that -- with the whole JSON
    # already printed and every claim in it made. The answer does not depend on
    # the size (the sweep measured CPU/GPU preferred at 1024x4096 and at
    # 1x256x64x64 as well), so the fixture asks it at a size it survives, and
    # the crash is recorded rather than hidden: see docs/graph/NPU2.md §7.3.
    #
    # **Asked at more than one shape, and that is not belt-and-braces.**
    # docs/graph/NPU2.md §9.1: at exactly (256, 1024) the compiled artefact
    # for each of these two programs comes back from MLComputePlan with no
    # device for any operation -- while (255, 1024), (128, 1024), (64, 4096)
    # and (256, 1023) all answer normally for the same op at the same
    # precision in the same process. So a fixture that asks once can have its
    # measurement taken away by one artefact, and the claim "relu is supported
    # and not preferred" would silently become untested. It is asked at every
    # shape, each answer is kept as it came -- including `unknown`, which is
    # now a row rather than an absence -- and the test grades them.
    for name, module in (("relu", torch.nn.ReLU()), ("gelu", torch.nn.GELU())):
        by_shape = {}
        for shape in ((256, 1024), (255, 1024), (128, 1024), (256, 1023)):
            model_, _names, _emitted = C.compile_model(
                capture(module.eval(), torch.randn(*shape)), float32=False)
            by_shape[str(list(shape))] = C.computes(
                C.compute_plan(model_, compute_units=ct.ComputeUnit.ALL))
        out.setdefault("rejected_plans", {})[name] = by_shape
    out["layer_norm_has_no_lowering"] = sorted(
        op for op in C.supported_ops() if "layer_norm" in op)

    # -- 5. a model with no lowerable leaf is still a refusal --------------
    try:
        ActOnly().to(tn_device.npu)
    except Exception as error:
        out["act_only_refusal"] = f"{type(error).__name__}: {error}"
    else:
        out["act_only_refusal"] = None

    # -- 6. a partial offload: conv lowered, LayerNorm named ---------------
    mixed = ConvNorm().eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mixed.to(tn_device.npu)
    out["mixed"] = {
        "warnings": ours(caught),
        "report": {k: v for k, v in mixed.torchnative_offload.items()
                   if k not in ("plans", "_said")},
    }
except Exception as error:  # noqa: BLE001
    import traceback
    out["coremlops_error"] = traceback.format_exc()

print(json.dumps(out))
"""


_CACHE = {}


def _fixture_or_skip():
    """The fixture, or `None` with the reason printed by name."""
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return None
    if "f" not in _CACHE:
        _CACHE["f"] = _npu_fixture(_COREMLOPS_SCRIPT)
    result = _CACHE["f"]
    if result["coremltools"] is None:
        print("   (skipped: coremltools not installed for this interpreter)")
        return None
    if result.get("resolution", {}).get("backend") != "coreml":
        print("   (skipped: this host's npu does not resolve to the coreml "
              "backend)")
        return None
    if "coremlops_error" in result:
        raise AssertionError(
            "the CoreML op-coverage fixture raised; this is a failure and not "
            "a skip:\n" + result["coremlops_error"]
        )
    return result


def test_conv2d_lowers_to_coreml_instead_of_being_left_on_the_cpu():
    """The gap this round closed: `Conv2d` is a second lowered leaf type.

    `to()` still returns `self` and still hands back an `nn.Module` -- the leaf
    is swapped, nothing is wrapped -- and the forward produces the shape the
    torch conv produces.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    for label in ("float16", "float32"):
        entry = r["lowered"][label]
        assert entry["returned_is_self"] is True, (label, entry)
        assert entry["leaf_type"] == "_CoreMLConv2d", (label, entry)
        assert entry["output_shape"] == [1, 256, 32, 32], (label, entry)
        assert entry["fully_offloaded"] is True, (label, entry)


def test_the_neural_engine_runs_the_conv_at_float16_and_cannot_at_float32():
    """The measurement that makes conv worth lowering, in both directions.

    At float16 `MLComputePlan` puts the Neural Engine in the supported column
    **and prefers it** for this conv. At float32 the unit is not in the
    supported column at all, exactly as docs/graph/NPU2.md §1.1 measured for
    `linear` -- so the float32 spelling reaches CPU/GPU and says so.

    Asserting the float32 absence is what stops this passing if precision had
    no effect.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    f16 = [row for plan in r["lowered"]["float16"]["plans"]
           for row in plan["rows"] if row["op"].endswith("conv")]
    assert f16, r["lowered"]["float16"]["plans"]
    for row in f16:
        assert "NeuralEngine" in row["supported"], row
        assert row["preferred"] == "NeuralEngine", row
    f32 = [row for plan in r["lowered"]["float32"]["plans"]
           for row in plan["rows"] if row["op"].endswith("conv")]
    assert f32, r["lowered"]["float32"]["plans"]
    for row in f32:
        assert "NeuralEngine" not in row["supported"], row
    # And the caller is told, rather than left with a model they believe is on
    # the unit. A deferred leaf has no plan at `to()` time, so for conv the
    # sentence lands at the first forward -- but it does land.
    entry = r["lowered"]["float32"]
    assert any("Neural Engine" in w for w in
               entry["warnings_at_to"] + entry["warnings_at_forward"]), entry


def test_a_conv_leaf_is_deferred_because_its_shape_is_not_known_at_to_time():
    """The obstacle `_compile_model` named, solved by admitting it.

    A `Linear` is probed eagerly at batch 1 -- a shape the library can choose.
    A conv's spatial dimensions are not choosable, so its program is built at
    the first forward and the leaf is listed under `deferred` at `to()` time.
    The plan for the shape that actually ran is then on the report, keyed by
    that shape, so "which unit ran this" is never unanswered.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    entry = r["lowered"]["float16"]
    assert entry["deferred"] == ["c"], entry["deferred"]
    assert entry["plans_at_to"] == [], entry["plans_at_to"]
    assert len(entry["plans"]) == 1, entry["plans"]
    assert entry["plans"][0]["shape"] == [1, 128, 32, 32], entry["plans"][0]
    assert entry["plans"][0]["probe"] is False, entry["plans"][0]


def test_the_conv_float32_spelling_agrees_and_float16_only_nearly_does():
    """Graded through `coreml.verify`, at its own tolerance, not a new one.

    float32 must meet verify's own 2e-05, which is where docs/graph/NPU.md set
    the word *agrees*. float16 does not and is not made to: a 3x3x64 conv sums
    576 terms in half precision, so it gets its own, weaker, named claim.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    f32 = r["verify_conv"]["float32"]
    f16 = r["verify_conv"]["float16"]
    assert f32["executed"] is True and f16["executed"] is True, r["verify_conv"]
    assert f32["max_abs_diff"] <= 2e-5, f32
    assert f16["max_abs_diff"] > 2e-5, f16
    assert f16["max_abs_diff"] < 5e-2, f16


def test_the_rejected_types_are_rejected_by_a_number_and_not_by_omission():
    """Why relu, gelu and LayerNorm did not get a leaf, measured.

    `relu` and `gelu` already have MIL lowerings and CoreML lists the Neural
    Engine as **supported** for both -- and prefers the CPU or the GPU at
    every elementwise shape measured, from 256x1024 here up to 1024x4096 and
    1x256x64x64 in docs/graph/NPU2.md §7's sweep. A leaf swapped for one of
    those buys a tensor round trip through CoreML and does not reach the unit,
    which is the outcome docs/graph/NPU2.md §1 exists to make visible rather
    than ship.

    `LayerNorm` is refused a step earlier: there is no MIL lowering here for
    `aten.native_layer_norm.default` at all, so it is absent from
    `supported_ops()` and a leaf for it would have to invent one.

    **Graded per shape, and an unanswered shape is a named `unknown`, not a
    missing row.** This test was red on develop for days because CoreML
    returned no device for any operation of the (256, 1024) artefact and the
    fixture's single `assert rows` had nothing left to stand on
    (docs/graph/NPU2.md §9.1). The rule is not weakened to fit that: the
    measurement is still required, at every shape CoreML answered, and at
    least two shapes must answer or the claim is untested and this fails.
    What changed is that the shape CoreML declined is *visible* -- it is a row
    reading `preferred == "unknown"` -- instead of being an empty list that
    reads exactly like "no measurement was taken".
    """
    r = _fixture_or_skip()
    if r is None:
        return
    for name in ("relu", "gelu"):
        by_shape = r["rejected_plans"][name]
        assert len(by_shape) == 4, (name, sorted(by_shape))
        answered = 0
        for shape, rows in sorted(by_shape.items()):
            # Never absent. One row per computing operation, whatever CoreML
            # was willing to say about it.
            assert len(rows) == 1, (name, shape, rows)
            row = rows[0]
            if row["preferred"] == "unknown":
                assert row["supported"] == [], (name, shape, row)
                continue
            answered += 1
            assert "NeuralEngine" in row["supported"], (name, shape, row)
            assert row["preferred"] != "NeuralEngine", (name, shape, row)
        assert answered >= 2, (name, by_shape)
    assert r["layer_norm_has_no_lowering"] == [], r["layer_norm_has_no_lowering"]


def test_every_leaf_not_lowered_is_named_and_a_partial_offload_warns():
    """A leaf left behind is on the report by type, and the caller is told.

    `plan_lowering` answers this with no coremltools and no compile, which is
    what lets the selection be inspected on a machine that cannot run any of
    it. A `padding_mode` this module cannot express is a *skip with a reason*,
    not a silent success.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    plan = r["plan_lowering_conv_norm"]
    assert plan["eligible"] == ["c"], plan
    assert plan["left_on_cpu"].get("LayerNorm") == 1, plan
    assert plan["fully_offloaded"] is False, plan
    reflect = r["plan_lowering_reflect"]
    assert reflect["eligible"] == [], reflect
    assert reflect["skipped"], reflect
    assert "reflect" in reflect["skipped"][0][1], reflect
    mixed = r["mixed"]
    assert mixed["report"]["swapped"] == ["c"], mixed
    assert mixed["report"]["left_on_cpu"].get("LayerNorm") == 1, mixed
    assert any("PARTIAL offload" in w for w in mixed["warnings"]), mixed
    assert any("LayerNorm" in w for w in mixed["warnings"]), mixed


def test_a_model_with_no_lowerable_leaf_is_still_a_refusal():
    """Zero leaves lowered stays a refusal now that there are two leaf types.

    Widening the table is the change most likely to turn "nothing lowered"
    into "something lowered, badly"; this asserts the refusal still fires and
    still names the leaf types it found.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    refusal = r["act_only_refusal"]
    assert refusal is not None, "to(device.npu) on a conv-free model succeeded"
    assert "nothing was lowered" in refusal, refusal
    assert "ReLU" in refusal and "LayerNorm" in refusal, refusal


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
