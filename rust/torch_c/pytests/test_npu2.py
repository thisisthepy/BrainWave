"""Tests for docs/graph/NPU2.md -- the round that put a graph on an NPU.

docs/platform/RELEASE_0_0_13a0.md §5 carried one entry no round had moved: *nothing has
run on an NPU*. docs/graph/NPU.md had compiled and run a CoreML model, and had
serialised an NNAPI blob and decoded it back -- but it drew the line between
**executed** and **structurally validated** and put NNAPI on the wrong side of
it, because there is no NNAPI runtime on a Mac.

This file moves both halves, and it separates two claims docs/graph/NPU.md's line
does not distinguish:

1. **"Ran through CoreML" is not "ran on the NPU."** `MLModel` chooses among
   CPU, GPU and Neural Engine. The models docs/graph/NPU.md executed ran on the
   **CPU**, and the reason is the same `float32=True` that made their numerical
   claim meaningful: the Neural Engine is float16 hardware and CoreML does not
   offer it for a float32 program at all. Read from `MLComputePlan`, which is
   CoreML's own answer, not a guess.
2. **NNAPI is executed now.** `nnapi_device.py` ships the blob upstream's
   serialiser wrote to an Android device and `nnapi_runner.c` replays it into
   `ANeuralNetworksModel`. Which driver answered is named rather than left as
   "an NPU".

Every test here skips **by name** where the thing it needs is absent -- no
`coremltools`, no `ANDROID_SERIAL`, no NDK. docs/devices/VULKAN3.md §6.1 is why the
skip lines say what is missing: a skip with a false reason is counted as a
pass and is worse than a failure.
"""

import os

from test_shim import _CKPT_VENDOR_SHIM, _npu_fixture


# ---------------------------------------------------------------------------
# The fixture: one subprocess in the vendored tree, both halves.
# ---------------------------------------------------------------------------
#
# Split into two scripts rather than one, so that a machine with coremltools
# and no device (or the reverse) still gets the half it can run. A single
# script would make each half's availability depend on the other's.

_NPU2_COREML_SCRIPT = r"""
import json
import os
import tempfile

import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}

try:
    import coremltools as ct
    import coremltools.models.utils as ct_utils
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.compute_device import (
        MLCPUComputeDevice, MLGPUComputeDevice, MLNeuralEngineComputeDevice)
    out["coremltools"] = ct.__version__
except Exception as error:
    out["coremltools"] = None
    out["import_error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(out))
    raise SystemExit(0)

import numpy as np

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


def device_name(device):
    if isinstance(device, MLNeuralEngineComputeDevice):
        return "NeuralEngine"
    if isinstance(device, MLGPUComputeDevice):
        return "GPU"
    if isinstance(device, MLCPUComputeDevice):
        return "CPU"
    return type(device).__name__


out["available_devices"] = sorted(
    device_name(d) for d in
    __import__("coremltools").models.compute_device.MLComputeDevice
    .get_all_compute_devices()
)


def plan_for(trace, *, float32, compute_units):
    # Compile, save, compile *again* with the OS, and read the compute plan.
    #
    # `MLComputePlan.load_from_path` wants a compiled `.mlmodelc`, not the
    # `.mlpackage` that `MLModel.save` writes -- handed the package it aborts
    # the process with a C++ exception rather than raising, so the second
    # compile is not optional tidiness.
    model, names, emitted = C.compile_model(trace, float32=float32)
    directory = tempfile.mkdtemp()
    package = os.path.join(directory, "m.mlpackage")
    model.save(package)
    compiled = ct_utils.compile_model(package)
    plan = MLComputePlan.load_from_path(compiled, compute_units=compute_units)
    function = plan.model_structure.program.functions["main"]
    rows = []
    for operation in function.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None:
            # `const` has no compute device: it is not executed anywhere.
            continue
        rows.append({
            "op": operation.operator_name,
            "preferred": device_name(usage.preferred_compute_device),
            "supported": sorted(
                device_name(d) for d in usage.supported_compute_devices),
        })
    return rows, package, names, emitted


try:
    torch.manual_seed(0)

    # -- 1. the graphs docs/graph/NPU.md actually executed -----------------------
    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 8)
            self.b = torch.nn.Linear(8, 3)

        def forward(self, x):
            return torch.nn.functional.softmax(
                self.b(torch.nn.functional.gelu(self.a(x))), dim=1)

    class Cnn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            return torch.relu(self.pool(self.conv(x)))

    npu_md_cases = {
        "mlp_gelu_softmax": (Mlp().eval(), torch.randn(2, 4)),
        "cnn_conv_pool_relu": (Cnn().eval(), torch.randn(1, 3, 16, 16)),
        "sigmoid": (torch.nn.Sigmoid(), torch.randn(2, 3)),
    }
    as_executed = {}
    for name, (module, example) in npu_md_cases.items():
        rows, _pkg, _n, _e = plan_for(
            capture(module, example), float32=True,
            compute_units=ct.ComputeUnit.ALL)
        as_executed[name] = rows
    out["npu_md_float32_plans"] = as_executed

    # -- 2. what float32 costs: the same model, both precisions ------------
    cnn = capture(Cnn().eval(), torch.randn(1, 3, 16, 16))
    out["same_model"] = {
        "float32": plan_for(cnn, float32=True,
                            compute_units=ct.ComputeUnit.ALL)[0],
        "float16": plan_for(cnn, float32=False,
                            compute_units=ct.ComputeUnit.ALL)[0],
    }

    # -- 3. a graph that does reach the Neural Engine ----------------------
    class Wide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.c2 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.c3 = torch.nn.Conv2d(128, 128, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = torch.relu(self.c1(x))
            x = torch.relu(self.c2(x))
            x = torch.relu(self.c3(x))
            return torch.relu(self.pool(x))

    example = torch.randn(1, 3, 64, 64)
    wide = capture(Wide().eval(), example)
    rows, package, names, emitted = plan_for(
        wide, float32=False, compute_units=ct.ComputeUnit.CPU_AND_NE)
    out["wide_float16_cpu_and_ne_plan"] = rows

    feed = {names[0]: np.asarray(example.tolist(), dtype=np.float32)}
    reference = [np.asarray(t.tolist(), dtype=np.float64)
                 for t in emitted.replay([example])]
    produced = {}
    for label, units in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
                         ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)):
        loaded = ct.models.MLModel(package, compute_units=units)
        got = [np.asarray(v, dtype=np.float64)
               for v in loaded.predict(feed).values()]
        produced[label] = got
        out.setdefault("wide_executed", {})[label] = {
            "elements": int(got[0].size),
            "shape": list(got[0].shape),
            "max_abs_diff": max(float(np.max(np.abs(g - r)))
                                for g, r in zip(got, reference)),
        }
    out["wide_ne_vs_cpu"] = max(
        float(np.max(np.abs(a - b)))
        for a, b in zip(produced["CPU_AND_NE"], produced["CPU_ONLY"]))
except Exception as error:  # noqa: BLE001
    import traceback
    out["coreml_error"] = traceback.format_exc()

print(json.dumps(out))
"""


_NPU2_NNAPI_SCRIPT = r"""
import json

import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}

from torchnative.export import fuse as FU
from torchnative.export import nnapi as N
from torchnative.export import nnapi_device as D
from torchnative.export.decompose import DecomposedTrace

out["serial"] = D.serial()
out["adb"] = D.adb_available()
out["ndk_clang"] = D.ndk_clang()
if not out["adb"] or out["ndk_clang"] is None:
    print(json.dumps(out))
    raise SystemExit(0)


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


try:
    out["devices"] = D.devices()
    names = [d["name"] for d in out["devices"]]

    torch.manual_seed(0)

    # -- 1. the smallest thing that is arithmetic and not layout -----------
    class ConvRelu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x):
            return torch.relu(self.conv(x))

    image = torch.randn(1, 3, 8, 8)
    conv_relu = capture(ConvRelu().eval(), image)
    out["conv_relu"] = {
        name: D.verify_on_device(conv_relu, [image], device_name=name,
                                 tolerance=1e-5)
        for name in names if "quant" not in name
    }

    # -- 2. docs/graph/REFOLD.md §4's whole model, on the device ------------------
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, stride=2, padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 4, 3, padding=1),
        torch.nn.ReLU6(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 5),
        torch.nn.Softmax(dim=1),
    ).eval()
    with torch.no_grad():
        net[1].running_mean.uniform_(-1.0, 1.0)
        net[1].running_var.uniform_(0.5, 2.0)
        net[1].weight.uniform_(0.5, 1.5)
        net[1].bias.uniform_(-0.5, 0.5)
        # A softmax over a randomly-initialised Linear is nearly uniform, and
        # then a *wrong* answer is only ~7e-4 from a right one -- an agreement
        # threshold under that would be passing on the model's flatness rather
        # than on the device's arithmetic. Widening the last layer spreads the
        # output so the control below is six orders above the agreement.
        net[7].weight.uniform_(-4.0, 4.0)
        net[7].bias.uniform_(-2.0, 2.0)
    picture = torch.randn(1, 3, 16, 16)
    whole = capture(net, picture)
    fused, pairs = FU.fold_conv_batch_norm(whole)
    model = N.serialize(fused)
    out["whole"] = {
        "pairs": len(pairs),
        "bytes": len(model.as_bytes()),
        "opcodes": [op["opcode"] for op in N.parse_model(model)["operations"]],
        # After constant folding, which is part of the serialisation path
        # rather than an optimisation on top of it (docs/graph/NPU.md §5).
        "outside": sorted(
            {n["op"] for n in N.fold_constants(fused)[0].nodes}
            - N.supported_ops()),
    }
    out["whole"]["executed"] = D.verify_on_device(
        fused, [picture], device_name="nnapi-reference", tolerance=1e-5)

    # The control: the same device, the same blob, a *different* input. If the
    # comparison above cannot tell these apart it is not measuring anything.
    reference = fused.replay([picture])[0].tolist()[0]
    worst = 0.0
    for _ in range(3):
        other = torch.randn(1, 3, 16, 16)
        answer = D.run_on_device(model, [other], device_name="nnapi-reference")
        worst = max(worst, max(abs(a - b)
                               for a, b in zip(answer["floats"], reference)))
    out["whole"]["control_wrong_input"] = worst

    # -- 3. a driver that does not support the ops must refuse -------------
    quant = [n for n in names if "quant" in n]
    if quant:
        try:
            D.verify_on_device(fused, [picture], device_name=quant[0])
            out["quant_driver"] = "ACCEPTED"
        except D.NnapiDeviceRefused as error:
            out["quant_driver"] = str(error)

    # -- 4. nothing was widened (docs/graph/REFOLD.md's live warning) ------------
    out["supported_ops"] = sorted(N.supported_ops())
except Exception as error:  # noqa: BLE001
    import traceback
    out["nnapi_error"] = traceback.format_exc()

print(json.dumps(out))
"""


_CACHE = {}


def _coreml_fixture():
    if "coreml" not in _CACHE:
        _CACHE["coreml"] = _npu_fixture(_NPU2_COREML_SCRIPT)
    return _CACHE["coreml"]


def _nnapi_fixture():
    if "nnapi" not in _CACHE:
        _CACHE["nnapi"] = _npu_fixture(_NPU2_NNAPI_SCRIPT)
    return _CACHE["nnapi"]


def _coreml_or_skip():
    """The CoreML fixture, or `None` with the reason printed by name."""
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return None
    result = _coreml_fixture()
    if result["coremltools"] is None:
        print("   (skipped: coremltools not installed for this interpreter)")
        return None
    if "coreml_error" in result:
        raise AssertionError(
            "the CoreML compute-plan fixture raised; this is a failure and "
            "not a skip:\n" + result["coreml_error"]
        )
    return result


def _nnapi_or_skip():
    """The NNAPI fixture, or `None` with the reason printed by name.

    The two reasons are kept apart. "No `ANDROID_SERIAL`" is a *refusal to
    guess*: several emulators are shared on this machine, so this module never
    picks "the attached one". "No NDK" is a missing toolchain. Reporting either
    as the other would send the next reader to the wrong place, which is
    docs/devices/VULKAN3.md §6.1's whole lesson.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return None
    result = _nnapi_fixture()
    if not result["adb"]:
        print("   (skipped: no adb on PATH, or ANDROID_SERIAL is unset -- "
              "this module will not guess which shared emulator to use)")
        return None
    if result["ndk_clang"] is None:
        print("   (skipped: no Android NDK clang for arm64, so nnapi_runner.c "
              "cannot be built)")
        return None
    if "nnapi_error" in result:
        raise AssertionError(
            "the NNAPI device fixture raised; this is a failure and not a "
            "skip:\n" + result["nnapi_error"]
        )
    return result


# --- 1. CoreML: which unit executed -----------------------------------------


def test_the_coreml_models_docs_npu_executed_ran_on_the_cpu():
    """docs/graph/NPU.md §2's executed claim was a **CPU** claim, and did not say so.

    "Compiled by macOS and run through `MLModel.predict`" is true and is not
    the same sentence as "ran on the NPU". `MLModel` picks among CPU, GPU and
    Neural Engine, and `MLComputePlan` is CoreML's own answer to which. For
    all three of docs/graph/NPU.md's float32 graphs, every compute operation's
    preferred device is the CPU.

    This is not a defect in that document -- it is a distinction it did not
    draw, and drawing it is what this round is for.
    """
    result = _coreml_or_skip()
    if result is None:
        return
    plans = result["npu_md_float32_plans"]
    assert set(plans) == {"mlp_gelu_softmax", "cnn_conv_pool_relu", "sigmoid"}, \
        sorted(plans)
    for name, rows in plans.items():
        assert rows, (name, "no compute operations in the plan at all")
        for row in rows:
            assert row["preferred"] == "CPU", (name, row)
            assert "NeuralEngine" not in row["supported"], (name, row)


def test_pinning_float32_is_what_puts_the_neural_engine_out_of_reach():
    """The same model, both precisions -- and the difference is not a preference.

    At float16 the Neural Engine is in the *supported* set for every compute
    op. At float32 it is not in the supported set at all: CoreML does not
    offer the unit, so no `compute_units` setting can reach it. The Neural
    Engine is float16 hardware.

    That makes docs/graph/NPU.md's `float32=True` and NPU execution mutually
    exclusive, which is worth stating as a measured fact rather than as an
    inference from the previous test. Both halves are measured on one model so
    the comparison cannot be confounded by graph shape.
    """
    result = _coreml_or_skip()
    if result is None:
        return
    half = result["same_model"]["float16"]
    full = result["same_model"]["float32"]
    assert half and full, result["same_model"]
    assert any("NeuralEngine" in row["supported"] for row in half), half
    for row in full:
        assert "NeuralEngine" not in row["supported"], row
    # And the machine really has one, so this is about precision rather than
    # about there being no Neural Engine to reach.
    assert "NeuralEngine" in result["available_devices"], \
        result["available_devices"]


def test_a_graph_executes_on_the_neural_engine_and_agrees_with_replay():
    """The executed NPU claim on the CoreML side.

    Three convolutions and a pool at float16, compiled with the GPU excluded so
    the only alternative to the CPU is the Neural Engine. `MLComputePlan` puts
    every compute operation on the Neural Engine -- only the two `cast`
    operations at the boundary stay on the CPU -- and the model then runs and
    agrees with `DecomposedTrace.replay` at float16 precision.

    A plan is a statement of intent, so it is not the only evidence here: the
    same package run with `CPU_ONLY` gives a **different** answer. Identical
    outputs would be consistent with the Neural Engine plan having been
    ignored; a difference of the size half precision produces is not.
    """
    result = _coreml_or_skip()
    if result is None:
        return
    rows = result["wide_float16_cpu_and_ne_plan"]
    on_ne = {row["op"] for row in rows if row["preferred"] == "NeuralEngine"}
    assert {"ios16.conv", "ios16.relu", "ios16.reduce_mean"} <= on_ne, rows
    for row in rows:
        if row["preferred"] != "NeuralEngine":
            assert row["op"].endswith("cast"), row

    executed = result["wide_executed"]
    assert executed["CPU_AND_NE"]["elements"] == 128, executed
    # float16 arithmetic: docs/graph/NPU.md measured 2.3e-04 for the same reason.
    assert executed["CPU_AND_NE"]["max_abs_diff"] < 1e-2, executed
    assert result["wide_ne_vs_cpu"] > 0.0, (
        "the Neural Engine and CPU-only runs of the same package agreed bit "
        "for bit, which is what it would look like if the Neural Engine plan "
        "were not being followed -- re-examine before believing the plan"
    )


# --- 2. NNAPI: executed on a device -----------------------------------------


def test_nnapi_drivers_are_read_from_the_device_rather_than_assumed():
    """Which driver answers is a fact about the image, so it is read.

    An emulator's NNAPI is backed by software: a reference implementation plus
    whatever sample drivers the system image ships. That is still execution
    and docs/graph/NPU2.md counts it as such -- but it names the driver, because
    "NNAPI ran it" and "a CPU reference driver ran it" are different claims and
    this project's method is not to blur them.
    """
    result = _nnapi_or_skip()
    if result is None:
        return
    devices = result["devices"]
    assert devices, "the runtime reported no NNAPI devices at all"
    names = {d["name"] for d in devices}
    assert "nnapi-reference" in names, sorted(names)


def test_a_conv_relu_blob_executes_on_nnapi_and_agrees_with_replay():
    """The claim docs/graph/NPU.md §2 could not make.

    The blob is upstream's -- `torch/backends/_nnapi/serializer.py` wrote every
    byte of it -- and `nnapi_runner.c` replays it operand by operand into
    `ANeuralNetworksModel`. There is no second lowering on the device that
    could agree with the first by sharing a mistake, which is the same reason
    `verify_shapes` compares against capture rather than a recomputation.

    Every driver that claims the operations is run, not just one, so a result
    that depended on a particular software implementation would show up as a
    disagreement between them.
    """
    result = _nnapi_or_skip()
    if result is None:
        return
    runs = result["conv_relu"]
    assert runs, "no driver claimed CONV_2D and RELU"
    assert "nnapi-reference" in runs, sorted(runs)
    for name, run in runs.items():
        assert run["executed"] is True, (name, run)
        assert run["elements"] == 256, (name, run)
        assert run["operations_supported"] == "2/2", (name, run)
        assert run["within_tolerance"], (name, run)
        assert run["max_abs_diff"] < 1e-6, (name, run)


def test_the_whole_model_executes_on_nnapi_and_the_control_is_orders_larger():
    """docs/graph/REFOLD.md §4's deliverable, executed rather than decoded.

    The same network, the same fold, the same 1,156-byte blob and the same
    eight opcodes by value -- and now the numbers it computes on a device,
    compared element by element against `DecomposedTrace.replay`.

    The control is the half that makes this a measurement. A softmax over a
    randomly-initialised Linear is nearly uniform, so on the original weights a
    *wrong* answer sat about 7e-4 from a right one and any threshold under
    that would have been passing on the model's flatness. The last layer is
    widened so the output spreads, and the test then requires the gap between
    "same input" and "different input" to be at least four orders of
    magnitude. CLAUDE.md §5.5: a check that cannot fail is not a check.
    """
    result = _nnapi_or_skip()
    if result is None:
        return
    whole = result["whole"]
    assert whole["pairs"] == 1, whole
    assert whole["bytes"] == 1156, whole
    assert whole["opcodes"] == [3, 19, 3, 21, 1, 22, 9, 25], whole
    executed = whole["executed"]
    assert executed["executed"] is True, executed
    assert executed["device"] == "nnapi-reference", executed
    assert executed["operations_supported"] == "8/8", executed
    assert executed["elements"] == 5, executed
    assert executed["within_tolerance"], executed
    agreement = executed["max_abs_diff"]
    control = whole["control_wrong_input"]
    assert agreement < 1e-5, agreement
    assert control > agreement * 1e4, (agreement, control)


def test_a_driver_that_does_not_claim_the_operations_refuses_by_name():
    """The negative control for the driver selection itself.

    `nnapi-sample_quant` is a quantised-only sample driver, so it claims 0 of
    the 8 operations and compiling for it must fail. If it ever succeeded, the
    device name passed to `createForDevices` would not be deciding anything and
    every "this driver ran it" claim in docs/graph/NPU2.md would be unfounded.
    """
    result = _nnapi_or_skip()
    if result is None:
        return
    if "quant_driver" not in result:
        print("   (skipped: this image ships no quantised-only sample driver)")
        return
    assert result["quant_driver"] != "ACCEPTED", result["quant_driver"]
    assert "supports 0/8" in result["quant_driver"], result["quant_driver"]


def test_executing_on_a_device_widened_nothing():
    """docs/graph/REFOLD.md's live warning, honoured rather than quoted.

    That document measured `mobilenet_v2` getting *worse* under a bigger
    table -- 203 nodes to 1,191 -- and left the standing instruction that more
    ops lowering is not automatically progress. This round adds execution and
    **no coverage**: `nnapi.supported_ops()` is the same 25 overloads it was,
    and the whole model still has nothing outside it. Pinned by value so that
    a later round which does widen it has to come back and say so.
    """
    result = _nnapi_or_skip()
    if result is None:
        return
    assert len(result["supported_ops"]) == 25, result["supported_ops"]
    assert "aten.convolution.default" in result["supported_ops"]
    assert result["whole"]["outside"] == [], result["whole"]["outside"]


def test_the_device_module_refuses_to_guess_which_emulator_to_use():
    """Runnable everywhere, including where there is no device at all.

    The emulators on this machine are shared. `nnapi_device.serial()` reads
    `ANDROID_SERIAL` and nothing else -- never "the one attached device" --
    so a run here cannot land on another project's emulator. This is checked
    in-process rather than through the fixture, because the property being
    checked is what happens when the environment is *empty*.
    """
    import importlib
    import sys

    root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))),
        "torchnative", "src", "main")
    if not os.path.isdir(os.path.join(root, "torchnative", "export")):
        print(f"   (skipped: no vendored torchnative tree under {root})")
        return
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    saved = os.environ.pop("ANDROID_SERIAL", None)
    try:
        module = importlib.import_module("torchnative.export.nnapi_device")
        assert module.serial() is None, module.serial()
        assert module.adb_available() is False
        try:
            module._adb("devices")
        except module.NnapiDeviceRefused as error:
            assert "ANDROID_SERIAL" in str(error), str(error)
        else:
            raise AssertionError(
                "_adb ran with no ANDROID_SERIAL set -- it would have picked a "
                "shared emulator on its own"
            )
    finally:
        if saved is not None:
            os.environ["ANDROID_SERIAL"] = saved
        if inserted and root in sys.path:
            sys.path.remove(root)


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
