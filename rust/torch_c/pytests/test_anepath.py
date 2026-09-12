"""Tests for the CoreML arm of `nn.Module.to(torchnative.device.npu)`.

docs/graph/NPU2.md measured the thing that made this round hard. On this
machine `device.npu` resolves to the Apple Neural Engine through the `coreml`
backend and reports `available=True, kind=measured` --- and then
`model.to(device.npu)` raised `NotImplementedError`, because only the
`openvino` arm had been wired. The device namespace could see the hardware and
could not use it.

The obstacle is not plumbing. `coreml.compile_model(float32=True)` is pinned
that way because docs/graph/NPU.md §6 measured coremltools' float16 default
disagreeing with `DecomposedTrace.replay` by 2.3e-04 against 3.0e-08. That pin
is also what puts the Neural Engine out of reach: for a float32 program
CoreML does not list the unit in the *supported* column at all, so no
`compute_units` setting reaches it.

So the two things this project wants are two products and not one:

* **float16** reaches the Neural Engine, and agrees to about 1e-03 relative.
* **float32** agrees at the float32 tolerance, and runs on CPU/GPU only.

They are two spellings here --- `to(device.npu, precision=...)` --- and never
one spelling with a silent mode. Every test below asserts on `MLComputePlan`,
CoreML's own answer to which unit ran, rather than on the fact that a number
came back: docs/graph/NPU2.md exists because a partial offload went unnoticed
while every answer it produced was right.

Skips say by name what is missing, for docs/devices/VULKAN3.md §6.1's reason.
"""

import os

from test_shim import _CKPT_VENDOR_SHIM, _npu_fixture


_ANEPATH_SCRIPT = r"""
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


# This module's warnings, kept apart from everybody else's. coremltools emits
# an "Implicitly cleaning up <TemporaryDirectory ...>" warning of its own, and
# a test that asserted "no warnings at all" would be asserting something this
# round does not control -- and would go red the day an unrelated dependency
# got noisier. So the two are collected separately and both are reported: the
# claim is that *this* path is silent when it has nothing to say, and the
# foreign list is carried along so nothing is quietly dropped.
_OURS = ("torchnative ", "nn.Module.to(")


def ours(caught):
    return [str(w.message) for w in caught
            if str(w.message).startswith(_OURS)]


def foreign(caught):
    return [str(w.message) for w in caught
            if not str(w.message).startswith(_OURS)]


class Big(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        return self.fc(x)


class Mixed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1024, 1024)
        self.norm = torch.nn.LayerNorm(1024)

    def forward(self, x):
        return self.norm(self.fc(x))


class NoLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.ReLU()

    def forward(self, x):
        return self.act(x)


try:
    torch.manual_seed(0)
    out["resolution"] = {
        "backend": tn_device.npu.resolve().backend,
        "unit": tn_device.npu.resolve().unit,
    }

    # -- 1. the selection, with no CoreML involved -------------------------
    out["plan_lowering"] = {
        k: v for k, v in C.plan_lowering(Mixed()).items()
        if k != "skipped"
    }
    out["plan_lowering_no_linear"] = {
        k: v for k, v in C.plan_lowering(NoLinear()).items()
        if k != "skipped"
    }

    example = torch.randn(128, 1024)

    # -- 2. to(device.npu) at each precision -------------------------------
    for label, kwargs in (("float16", {}),
                          ("float32", {"precision": "float32"})):
        model = Big().eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            returned = model.to(tn_device.npu, **kwargs)
        entry = {
            "returned_is_self": returned is model,
            "is_nn_module": isinstance(returned, torch.nn.Module),
            "leaf_type": type(model.fc).__name__,
            "warnings_at_to": ours(caught),
            "foreign_warnings_at_to": foreign(caught),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            produced = model(example)
        entry["warnings_at_forward"] = ours(caught)
        report = model.torchnative_offload
        entry["report_precision"] = report["precision"]
        entry["fully_offloaded"] = report["fully_offloaded"]
        entry["fraction_moved"] = report["fraction_moved"]
        entry["plans"] = report["plans"]
        entry["output_shape"] = list(produced.shape)
        out.setdefault("lowered", {})[label] = entry

    # -- 3. the agreement grade, through coreml.verify ---------------------
    reference = Big().eval()
    trace = capture(reference, example)
    for label, float32 in (("float16", False), ("float32", True)):
        out.setdefault("verify", {})[label] = C.verify(
            trace, [example], float32=float32, tolerance=1.0,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

    # -- 4. nothing lowered is a refusal -----------------------------------
    try:
        NoLinear().to(tn_device.npu)
    except Exception as error:
        out["no_linear_refusal"] = f"{type(error).__name__}: {error}"
    else:
        out["no_linear_refusal"] = None

    # -- 5. a bad precision spelling is refused by name --------------------
    try:
        Big().to(tn_device.npu, precision="bfloat16")
    except Exception as error:
        out["bad_precision_refusal"] = f"{type(error).__name__}: {error}"
    else:
        out["bad_precision_refusal"] = None
except Exception as error:  # noqa: BLE001
    import traceback
    out["anepath_error"] = traceback.format_exc()

print(json.dumps(out))
"""


_CACHE = {}


def _fixture_or_skip():
    """The fixture, or `None` with the reason printed by name."""
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return None
    if "f" not in _CACHE:
        _CACHE["f"] = _npu_fixture(_ANEPATH_SCRIPT)
    result = _CACHE["f"]
    if result["coremltools"] is None:
        print("   (skipped: coremltools not installed for this interpreter)")
        return None
    if result.get("resolution", {}).get("backend") != "coreml":
        print("   (skipped: this host's npu does not resolve to the coreml "
              "backend)")
        return None
    if "anepath_error" in result:
        raise AssertionError(
            "the CoreML lowering fixture raised; this is a failure and not a "
            "skip:\n" + result["anepath_error"]
        )
    return result


def _compute_ops(rows):
    """Plan rows for the ops that compute, dropping the boundary `cast`s."""
    return [r for r in rows if not r["op"].endswith("cast")]


def test_to_the_npu_lowers_for_coreml_instead_of_refusing():
    """The gap this round closed, stated as the behaviour and not the diff.

    `device.npu.availability()` said `available=True, kind=measured` and
    `resolve()` named the Apple Neural Engine, and then `to()` raised
    `NotImplementedError` -- a namespace that could see the hardware and not
    use it. `to()` returns `self`, still an `nn.Module`, because
    `_module_to.py`'s "no wrapping, ever" is what separates this project from
    an inference-object wrapper.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    for label in ("float16", "float32"):
        entry = r["lowered"][label]
        assert entry["returned_is_self"] is True, (label, entry)
        assert entry["is_nn_module"] is True, (label, entry)
        assert entry["leaf_type"] == "_CoreMLLinear", (label, entry)
        assert entry["output_shape"] == [128, 1024], (label, entry)


def test_the_neural_engine_is_supported_at_float16_and_absent_at_float32():
    """The measurement the whole round turns on, asserted in both directions.

    docs/graph/NPU2.md §1.1 read this for a conv; it holds for `linear` too and
    at every size tried. For a float32 program CoreML's supported column is
    `CPU, GPU` -- the Neural Engine is not in it, so no `compute_units`
    setting reaches the unit. At float16 it is.

    Asserting the *absence* at float32 is what makes this non-vacuous: a test
    that only checked float16 would pass just as well if precision had no
    effect at all.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    for label, expected in (("float16", True), ("float32", False)):
        rows = _compute_ops(r["lowered"][label]["plans"][0]["rows"])
        assert rows, (label, r["lowered"][label]["plans"])
        for row in rows:
            present = "NeuralEngine" in row["supported"]
            assert present is expected, (label, row)


def test_a_lowered_leaf_records_which_unit_coreml_actually_preferred():
    """No fourth outcome: `to()` succeeded, and the report says what ran.

    `MLComputePlan` is read at each compile and stored on the report, so the
    answer to "which unit" is CoreML's own and is available without rerunning
    anything. At float16 and batch 128 the `linear` is preferred on the Neural
    Engine; at float32 it cannot be.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    f16 = _compute_ops(r["lowered"]["float16"]["plans"][-1]["rows"])
    assert [row["preferred"] for row in f16] == ["NeuralEngine"], f16
    f32 = _compute_ops(r["lowered"]["float32"]["plans"][-1]["rows"])
    assert all(row["preferred"] != "NeuralEngine" for row in f32), f32
    assert r["lowered"]["float16"]["report_precision"] == "float16", r["lowered"]
    assert r["lowered"]["float32"]["report_precision"] == "float32", r["lowered"]


def test_the_float32_spelling_agrees_and_the_float16_one_only_nearly_does():
    """Two products, graded honestly, through the comparison that already exists.

    `coreml.verify` runs the compiled model and compares against
    `DecomposedTrace.replay`. The float32 tolerance is 2e-05, which is
    `verify`'s own default and is where docs/graph/NPU.md set it. float16 does
    not meet it and this test says so rather than widening one number to cover
    both: docs/graph/NPU.md §6 measured the float16 default at 2.3e-04 and
    docs/graph/NPU2.md §2 at 2.0e-04, and a `Linear(1024, 1024)` accumulates
    over 1024 terms so it is looser still.

    The grade for the Neural Engine path is therefore **float16 agreement**,
    not `agrees` at the project's usual bar, and it is reported as such.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    f32 = r["verify"]["float32"]
    f16 = r["verify"]["float16"]
    assert f32["executed"] is True and f16["executed"] is True, r["verify"]
    assert f32["max_abs_diff"] <= 2e-5, f32
    assert f16["max_abs_diff"] > 2e-5, f16
    assert f16["max_abs_diff"] < 5e-2, f16


def test_lowering_nothing_is_a_refusal_and_not_a_quiet_success():
    """Zero leaves lowered is the silent CPU fallback, so it raises.

    The same position `intelnpu._compile_model` takes: returning an untouched
    model with a success message would leave a caller holding a model they
    believe is on the Neural Engine and which is entirely on the CPU. A bad
    `precision` spelling is refused by name for the neighbouring reason -- an
    argument accepted and dropped is how a mode goes silent.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    refusal = r["no_linear_refusal"]
    assert refusal is not None, "to(device.npu) on a Linear-free model succeeded"
    assert "nothing was lowered" in refusal, refusal
    bad = r["bad_precision_refusal"]
    assert bad is not None, "precision='bfloat16' was accepted"
    assert "bfloat16" in bad and "float16" in bad, bad


def test_a_partial_offload_warns_and_a_complete_one_does_not():
    """Silence is reserved for the case with nothing to report.

    `plan_lowering` answers "what would happen" with no coremltools and no
    compile -- the same separation `intelnpu.plan_lowering` draws, and for the
    same reason: the selection is where a granularity defect lives, and it must
    be inspectable on a machine that cannot run the thing.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    mixed = r["plan_lowering"]
    assert mixed["eligible"] == ["fc"], mixed
    assert mixed["fully_offloaded"] is False, mixed
    assert mixed["left_on_cpu"].get("LayerNorm") == 1, mixed
    assert 0.0 < mixed["fraction_moved"] < 1.0, mixed
    none = r["plan_lowering_no_linear"]
    assert none["eligible"] == [], none
    assert none["fully_offloaded"] is False, none
    # A complete offload: `Big` is one Linear and nothing else, so `to()` has
    # nothing to warn about and says nothing.
    assert r["lowered"]["float16"]["fully_offloaded"] is True, r["lowered"]
    assert r["lowered"]["float16"]["warnings_at_to"] == [], r["lowered"]
    # float32 cannot reach the unit the caller named, so that one is not silent.
    assert any("Neural Engine" in w
               for w in r["lowered"]["float32"]["warnings_at_to"]), r["lowered"]


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
