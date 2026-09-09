"""Intel NPU: docs/devices/INTELNPU.md, `torchnative.export.intelnpu`.

What is checkable here and what is not, stated first because the difference is
the whole design of this file. This round was written on an arm64 Mac. There is
no Intel NPU here and there never will be. But there **is** an OpenVINO --- the
runtime ships macOS arm64 wheels, with a CPU plugin and the same
`libopenvino_c` C API a Windows machine has --- and that changes what this file
can honestly assert.

So the tests are three kinds, not two:

* **Pure.** Platform refusal, library-name candidates, `EXECUTION_DEVICES`
  parsing, the verdict rules, the IR documents' shape, `judge`'s refusals, and
  every refusal-by-name. No OpenVINO, no hardware.
* **Against a real OpenVINO, for device `"CPU"`.** The C bindings, the IR
  actually loading, the weights blob, inference, and the numerics of a
  `torch.nn.Linear` lowered by `compile_model` and compared against the shim's
  own eager answer. These need `TORCHNATIVE_OPENVINO_C` pointing at an
  `openvino_c` shared library; they skip **by name** otherwise, saying exactly
  what is missing --- docs/devices/VULKAN3.md section 6.1: a skip with a false reason is
  counted as a pass.
* **Against a real Intel NPU.** Only `test_probe_on_real_hardware`, which needs
  OpenVINO to list an `NPU` device. This file cannot fake it and does not try.

The gap between the second kind and the third is exactly one string: whether
`EXECUTION_DEVICES` reads back `NPU` or `CPU`. Everything else on the path is
exercised here.

The verdict tests are the load-bearing ones. If `verdict_execution_devices` were
relaxed to "NPU appears somewhere in the list",
`test_verdict_refuses_partial_offload` goes red; if `judge`'s numeric floor were
dropped, `test_the_numeric_control_has_a_floor_and_not_just_a_ratio` goes red.
"""

import json
import os
import subprocess
import sys
import tempfile

# Resolved from __file__, not from the cwd: run.sh invokes each test file
# directly and the cwd it uses is not this file's business.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
sys.path.insert(0, _VENDOR_DIR)

import torchnative.export.intelnpu as intelnpu  # noqa: E402
from torchnative.export.intelnpu import (  # noqa: E402
    EXECUTION_DEVICES,
    LIBRARY_ENV,
    MAX_DIM,
    OV_STATUS,
    SUPPORTED_MODULES,
    IntelNPUExecutionError,
    IntelNPUUnavailable,
    IntelNPUUnsupported,
    IntelNPUWithdrawn,
    judge,
    library_candidates,
    linear_ir,
    minimal_ir,
    openvino_package_libs_dir,
    pack_f16,
    parse_execution_devices,
    supported_ops,
    unpack_f16,
    verdict_execution_devices,
)

_OV_ENV = "TORCHNATIVE_OPENVINO_C"


# ---------------------------------------------------------------- platform


def test_library_candidates_windows_and_linux():
    win = library_candidates("win32")
    assert win[0] == "openvino_c.dll", win
    lin = library_candidates("linux")
    assert lin[0] == "libopenvino_c.so", lin
    assert all(n.startswith("libopenvino_c.so") for n in lin), lin
    print("ok   intelnpu: library candidates named for win32 and linux")


def _fake_openvino_libs_dir(root, files):
    """Build a fake `<pkg>/libs/` directory mirroring the real pip wheel layout."""
    libs = os.path.join(root, "openvino", "libs")
    os.makedirs(libs, exist_ok=True)
    for name in files:
        with open(os.path.join(libs, name), "wb"):
            pass
    return libs


def test_library_candidates_globs_the_real_linux_wheel_suffix():
    """The real pip wheel ships `libopenvino_c.so.2541` -- a version number no
    hardcoded list ever had. `library_candidates` must find it by globbing the
    package directory, not by growing the hardcoded fallback list, since that
    list rots the same way every time OpenVINO ships a new build number."""
    with tempfile.TemporaryDirectory() as root:
        libs_dir = _fake_openvino_libs_dir(
            root,
            ["libopenvino_c.so.2541", "libopenvino.so.2541", "libopenvino_intel_npu_plugin.so"],
        )
        found = library_candidates("linux", libs_dir=libs_dir)
        assert found == (os.path.join(libs_dir, "libopenvino_c.so.2541"),), found
    print("ok   intelnpu: library_candidates globs the real libopenvino_c.so.NNNN suffix")


def test_library_candidates_globs_the_real_windows_wheel_name():
    with tempfile.TemporaryDirectory() as root:
        libs_dir = _fake_openvino_libs_dir(
            root, ["openvino_c.dll", "openvino.dll", "openvino_intel_npu_plugin.dll"]
        )
        found = library_candidates("win32", libs_dir=libs_dir)
        assert found == (os.path.join(libs_dir, "openvino_c.dll"),), found
    print("ok   intelnpu: library_candidates globs the real openvino_c.dll in a pip libs dir")


def test_library_candidates_falls_back_to_system_names_when_libs_dir_has_nothing():
    """A `libs_dir` that exists but has no matching file (or is `None`) must fall
    back to the hardcoded system-install names rather than returning empty --
    an empty candidate list would make `load_openvino_c` fail with an unhelpful
    "tried []" instead of trying the OS loader path."""
    assert library_candidates("linux", libs_dir=None) == library_candidates("linux")
    print("ok   intelnpu: no libs_dir falls back to the system-install candidate names")


def test_openvino_package_libs_dir_finds_the_injected_fake_layout():
    with tempfile.TemporaryDirectory() as root:
        libs_dir = _fake_openvino_libs_dir(root, ["libopenvino_c.so.2541"])
        pkg_root = os.path.dirname(libs_dir)
        found = openvino_package_libs_dir(search_locations=[pkg_root])
        assert found == libs_dir, (found, libs_dir)
    print("ok   intelnpu: openvino_package_libs_dir finds libs/ under an injected package root")


def test_openvino_package_libs_dir_returns_none_without_a_libs_directory():
    with tempfile.TemporaryDirectory() as root:
        pkg_root = os.path.join(root, "openvino")
        os.makedirs(pkg_root)
        with open(os.path.join(pkg_root, "__init__.py"), "wb"):
            pass
        found = openvino_package_libs_dir(search_locations=[pkg_root])
        assert found is None, found
    print("ok   intelnpu: openvino_package_libs_dir returns None when there is no libs/")


def test_openvino_package_libs_dir_returns_none_when_package_not_installed():
    """`search_locations=[]` stands in for `importlib.util.find_spec` returning
    nothing -- an environment with no `openvino` package installed at all."""
    assert openvino_package_libs_dir(search_locations=[]) is None
    print("ok   intelnpu: openvino_package_libs_dir returns None when nothing is found")


def test_explicit_path_and_env_var_still_win_over_package_discovery():
    """A user who names a library by hand must still get that one -- pip package
    discovery is a fallback source of candidates, not a source of truth that
    overrides an explicit choice. This is a load-bearing ordering claim about
    `load_openvino_c`, checked here without touching hardware: it loads a name
    that cannot possibly resolve and asserts the refusal names exactly that
    tried name, proving the explicit path reached `ctypes.CDLL` unchanged."""
    bogus = "definitely-not-a-real-openvino-c-library-name"
    try:
        intelnpu.load_openvino_c(path=bogus)
    except IntelNPUUnavailable as exc:
        assert repr([bogus]) in str(exc) or bogus in str(exc), str(exc)
        print("ok   intelnpu: an explicit path is tried as-is, ahead of package discovery")
        return
    raise AssertionError("load_openvino_c(path=bogus) unexpectedly loaded something")


def test_env_var_still_wins_over_package_discovery():
    bogus = "definitely-not-a-real-openvino-c-library-name-either"
    old = os.environ.get(LIBRARY_ENV)
    os.environ[LIBRARY_ENV] = bogus
    try:
        intelnpu.load_openvino_c()
    except IntelNPUUnavailable as exc:
        assert bogus in str(exc), str(exc)
        print("ok   intelnpu: TORCHNATIVE_OPENVINO_C still wins over package discovery")
        return
    finally:
        if old is None:
            os.environ.pop(LIBRARY_ENV, None)
        else:
            os.environ[LIBRARY_ENV] = old
    raise AssertionError(f"load_openvino_c() with {LIBRARY_ENV} set unexpectedly loaded something")


def test_library_candidates_refuses_darwin_by_name():
    """macOS has no Intel NPU. The refusal must say so, and say why, not just fail."""
    try:
        library_candidates("darwin")
    except IntelNPUUnavailable as exc:
        text = str(exc)
        assert "torchnative intelnpu:" in text, text
        assert "'darwin'" in text, text
        assert "Windows and Linux" in text, text
        assert "docs/devices/INTELNPU.md" in text, text
        print("ok   intelnpu: darwin refused by name with a reason")
        return
    raise AssertionError("library_candidates('darwin') did not refuse")


def test_library_candidates_refuses_unknown_platform():
    for platform in ("emscripten", "ios", "android"):
        try:
            library_candidates(platform)
        except IntelNPUUnavailable as exc:
            assert repr(platform) in str(exc), (platform, str(exc))
            continue
        raise AssertionError(f"library_candidates({platform!r}) did not refuse")
    print("ok   intelnpu: unknown platforms refused by name")


# ------------------------------------------------------- EXECUTION_DEVICES


def test_property_key_is_the_openvino_spelling():
    """`ov::execution_devices{"EXECUTION_DEVICES"}`, properties.hpp:1409."""
    assert EXECUTION_DEVICES == "EXECUTION_DEVICES", EXECUTION_DEVICES
    print("ok   intelnpu: EXECUTION_DEVICES key matches OpenVINO's spelling")


def test_ov_status_table_matches_the_c_header():
    """ov_status_e, ov_common.h:135-163. A wrong name here misreports a real failure.

    Read against the header rather than remembered: 18 entries, contiguous from 0
    down to -17, including upstream's own misspelling `UNKNOW_EXCEPTION` (-17) next
    to the correctly spelled `UNKNOWN_C_ERROR` (-15). Normalising either one would
    make a real OpenVINO failure report a status name that is not in the header.
    """
    assert OV_STATUS[0] == "OK"
    assert OV_STATUS[-1] == "GENERAL_ERROR"
    assert OV_STATUS[-5] == "NOT_FOUND"
    assert OV_STATUS[-14] == "INVALID_C_PARAM"
    assert OV_STATUS[-15] == "UNKNOWN_C_ERROR"
    assert OV_STATUS[-17] == "UNKNOW_EXCEPTION"  # sic, upstream's spelling
    assert sorted(OV_STATUS) == list(range(-17, 1)), sorted(OV_STATUS)
    assert len(OV_STATUS) == 18, sorted(OV_STATUS)
    print(f"ok   intelnpu: ov_status_e table has {len(OV_STATUS)} entries matching the header")


def test_parse_execution_devices_accepts_every_observed_spelling():
    cases = {
        "NPU": ("NPU",),
        " NPU ": ("NPU",),
        "NPU,CPU": ("NPU", "CPU"),
        "NPU CPU": ("NPU", "CPU"),
        "[NPU, CPU]": ("NPU", "CPU"),
        "['NPU']": ("NPU",),
        "": (),
    }
    for raw, want in cases.items():
        got = parse_execution_devices(raw)
        assert got == want, (raw, got, want)
    print(f"ok   intelnpu: {len(cases)} EXECUTION_DEVICES spellings parse")


def test_verdict_accepts_a_lone_npu():
    assert verdict_execution_devices(("NPU",)) == "NPU"
    assert verdict_execution_devices(("CPU",), expect="CPU") == "CPU"
    print("ok   intelnpu: a lone expected device is accepted")


def test_verdict_refuses_cpu_fallback_naming_the_mechanism():
    """The silent fallback: inference.h:77-79 rewrites device to "CPU" and only warns."""
    try:
        verdict_execution_devices(("CPU",))
    except IntelNPUExecutionError as exc:
        text = str(exc)
        assert "torchnative intelnpu:" in text, text
        assert "CPU" in text, text
        assert "inference.h:77-79" in text, text
        print("ok   intelnpu: CPU fallback refused, naming the upstream mechanism")
        return
    raise AssertionError("a CPU-only EXECUTION_DEVICES was accepted as NPU execution")


def test_verdict_refuses_partial_offload():
    """NPU present but not alone means part of the graph runs elsewhere.

    This is the test that goes red if the assertion is ever weakened to
    "NPU in devices". docs/graph/NPU2.md caught exactly this on the CoreML side.
    """
    try:
        verdict_execution_devices(("NPU", "CPU"))
    except IntelNPUExecutionError as exc:
        assert "split" in str(exc) or "other" in str(exc), str(exc)
        print("ok   intelnpu: heterogeneous NPU+CPU execution refused")
        return
    raise AssertionError("a split NPU/CPU execution was accepted as NPU execution")


def test_verdict_refuses_empty_evidence():
    try:
        verdict_execution_devices(())
    except IntelNPUExecutionError as exc:
        assert "no evidence" in str(exc), str(exc)
        print("ok   intelnpu: empty EXECUTION_DEVICES refused rather than assumed")
        return
    raise AssertionError("an empty EXECUTION_DEVICES was accepted")


# ------------------------------------------------------------- judge, pure


def _good_bundle():
    """An evidence bundle shaped like a successful NPU run on real hardware.

    Written by hand on purpose: no machine here can produce one, and the
    refusals below have to be checkable against the shapes that *would* be
    wrong. The numbers are the shape of a real reading -- f16 agreement, a
    control that moves by orders of magnitude more.
    """
    return {
        "execution_devices": ["NPU"],
        "execution_devices_control": ["CPU"],
        "linear_execution_devices": ["NPU"],
        "linear_max_abs_diff": 2.4e-4,
        "linear_control_diff": 3.1875,
    }


def test_judge_accepts_the_shape_of_a_real_npu_run():
    got = judge(_good_bundle(), "NPU", "CPU")
    assert got["verdict"] == "npu", got
    assert got["assert_device"] == "NPU", got
    assert got["control_moved"] is True and got["numeric_control_moved"] is True, got
    print("ok   intelnpu: judge accepts a well-formed NPU evidence bundle")


def test_judge_refuses_when_the_device_control_did_not_move():
    """If both compiles report the same device, the property is not tracking the request."""
    bundle = _good_bundle()
    bundle["execution_devices_control"] = ["NPU"]
    try:
        judge(bundle, "NPU", "CPU")
    except IntelNPUExecutionError as exc:
        assert "device control failed" in str(exc), str(exc)
        assert "not tracking" in str(exc), str(exc)
        print("ok   intelnpu: judge refuses when the device reading does not move")
        return
    raise AssertionError("judge accepted a reading that did not move with the request")


def test_judge_refuses_a_linear_that_ran_somewhere_else():
    """The probe IR and the Linear are compiled separately; both have to land on NPU."""
    bundle = _good_bundle()
    bundle["linear_execution_devices"] = ["CPU"]
    try:
        judge(bundle, "NPU", "CPU")
    except IntelNPUExecutionError as exc:
        assert "CPU" in str(exc), str(exc)
        print("ok   intelnpu: judge refuses when the Linear itself ran on the CPU")
        return
    raise AssertionError("judge accepted an NPU verdict for a Linear that ran on the CPU")


def test_judge_refuses_arithmetic_outside_the_f16_tolerance():
    bundle = _good_bundle()
    bundle["linear_max_abs_diff"] = 0.5
    try:
        judge(bundle, "NPU", "CPU")
    except IntelNPUExecutionError as exc:
        assert "arithmetic is wrong" in str(exc), str(exc)
        print("ok   intelnpu: judge refuses a right-device wrong-answer run")
        return
    raise AssertionError("judge accepted an answer outside the f16 tolerance")


def test_the_numeric_control_has_a_floor_and_not_just_a_ratio():
    """CLAUDE.md section 5.5, in the form this round nearly shipped.

    `evidence`'s weights and inputs are quarter-integers, exactly representable
    in f16, so a correct device gives `linear_max_abs_diff == 0.0`. A pure ratio
    check -- `control > agreement * 100` -- is then `control > 0`, which passes
    for any two answers that are not bit-identical, including a device whose two
    answers differ in the last bit. The floor is what the check rests on, and
    this test is the reason it exists: a device returning a near-constant answer
    against a zero agreement must still be refused.
    """
    bundle = _good_bundle()
    bundle["linear_max_abs_diff"] = 0.0
    bundle["linear_control_diff"] = 1e-6  # would pass `> 0 * 100`
    try:
        judge(bundle, "NPU", "CPU")
    except IntelNPUExecutionError as exc:
        assert "numeric control failed" in str(exc), str(exc)
        print("ok   intelnpu: the numeric control has an absolute floor, not just a ratio")
        return
    raise AssertionError(
        "judge accepted a 1e-6 control against a 0.0 agreement -- the ratio check "
        "degenerates to `control > 0` and is not load-bearing"
    )


# --------------------------------------------------------------------- IR


def test_minimal_ir_is_well_formed_and_shaped_as_declared():
    import xml.etree.ElementTree as ET

    root = ET.fromstring(minimal_ir(size=16))
    assert root.tag == "net", root.tag
    assert root.get("version") == "11", root.get("version")
    layers = root.find("layers")
    types = [layer.get("type") for layer in layers]
    assert types == ["Parameter", "ReLU", "Result"], types
    data = layers[0].find("data")
    assert data.get("element_type") == "f16", data.attrib
    assert data.get("shape") == "1,16", data.attrib
    assert len(root.find("edges")) == 2
    print("ok   intelnpu: minimal IR is well-formed f16 Parameter->ReLU->Result")


def test_minimal_ir_is_f16_not_f32():
    """f16 on purpose: docs/graph/NPU2.md's CoreML models ran on the CPU because of f32."""
    assert 'element_type="f16"' in minimal_ir()
    assert "f32" not in minimal_ir()
    print("ok   intelnpu: minimal IR asks for f16, not f32")


def test_linear_ir_declares_byte_counts_that_match_its_shapes():
    """A `Const` whose declared shape and `size` disagree is a load-time refusal.

    Checked here because it is arithmetic this module does, not OpenVINO: the
    weight is `out*in` halves at offset 0 and the bias is `out` halves
    immediately after it, and `pack_f16` has to produce exactly that many bytes
    in exactly that order. Getting the offset wrong reads the bias as weight
    and produces numbers that look like an arithmetic bug.
    """
    import xml.etree.ElementTree as ET

    in_f, out_f = 5, 3
    root = ET.fromstring(linear_ir(in_f, out_f, batch=2, bias=True))
    consts = [ly for ly in root.find("layers") if ly.get("type") == "Const"]
    assert len(consts) == 2, [c.get("name") for c in consts]
    weight, bias = consts
    assert weight.find("data").get("offset") == "0"
    assert int(weight.find("data").get("size")) == out_f * in_f * 2
    assert int(bias.find("data").get("offset")) == out_f * in_f * 2
    assert int(bias.find("data").get("size")) == out_f * 2
    total = out_f * in_f * 2 + out_f * 2
    assert len(pack_f16([0.0] * (out_f * in_f + out_f))) == total
    print(f"ok   intelnpu: linear_ir Const offsets and sizes tile the {total}-byte blob exactly")


def test_linear_ir_stores_the_weight_as_out_by_in_and_transposes_in_the_op():
    """`transpose_b="true"` with a `[out, in]` weight -- torch's own layout.

    If this were flipped, a square Linear would still load and would compute
    `x @ W` instead of `x @ W.T`: correct-looking numbers for the wrong function.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(linear_ir(5, 3, batch=1, bias=False))
    matmul = [ly for ly in root.find("layers") if ly.get("type") == "MatMul"][0]
    assert matmul.find("data").get("transpose_b") == "true", matmul.find("data").attrib
    assert matmul.find("data").get("transpose_a") == "false", matmul.find("data").attrib
    weight = [ly for ly in root.find("layers") if ly.get("name") == "weight"][0]
    dims = [int(d.text) for d in weight.find("output").find("port")]
    assert dims == [3, 5], dims  # [out, in]
    print("ok   intelnpu: linear_ir stores [out, in] and sets transpose_b, as torch does")


def test_linear_ir_without_bias_drops_the_add_and_rewires_the_result():
    import xml.etree.ElementTree as ET

    root = ET.fromstring(linear_ir(4, 2, bias=False))
    types = [ly.get("type") for ly in root.find("layers")]
    assert types == ["Parameter", "Const", "MatMul", "Result"], types
    assert len(root.find("edges")) == 3, len(root.find("edges"))
    print("ok   intelnpu: bias=False emits no Add and rewires the Result")


def test_linear_ir_refuses_an_oversized_layer_by_name():
    """The archived library silently returns the torch layer here (nn/linear.py:66)."""
    try:
        linear_ir(MAX_DIM + 1, 4)
    except IntelNPUUnsupported as exc:
        text = str(exc)
        assert "MAX_DIM" in text, text
        assert "nn/linear.py:66" in text, text
        assert "silently" in text or "unchanged" in text, text
        print("ok   intelnpu: an oversized Linear refuses by name instead of staying on the CPU")
        return
    raise AssertionError("an oversized Linear was lowered")


def test_pack_f16_round_trips_and_unpack_refuses_an_odd_byte_count():
    values = [0.0, 1.0, -0.25, 3.5, -100.0]
    assert unpack_f16(pack_f16(values)) == values, unpack_f16(pack_f16(values))
    assert len(pack_f16(values)) == 2 * len(values)
    try:
        unpack_f16(b"\x00\x00\x00")
    except IntelNPUExecutionError as exc:
        assert "not a whole number" in str(exc), str(exc)
        print("ok   intelnpu: f16 packing round-trips and an odd byte count is refused")
        return
    raise AssertionError("unpack_f16 accepted an odd byte count")


# --------------------------------------------------------------- refusals


def _assert_refuses(fn, *needles):
    try:
        fn()
    except IntelNPUUnsupported as exc:
        text = str(exc)
        assert text.startswith("torchnative intelnpu:"), text
        for needle in needles:
            assert needle in text, (needle, text)
        return text
    raise AssertionError(f"{fn} did not refuse")


def test_compile_module_refuses_by_name_and_distinguishes_itself_from_compile_model():
    """Two doors, different coverage. Redirecting one to the other would misreport."""
    text = _assert_refuses(
        lambda: intelnpu.compile_module,
        "compile_module",
        "docs/graph/NPU2.md",
    )
    assert "compile_model" in text, text
    assert "captured-graph" in text, text
    print("ok   intelnpu: compile_module refuses by name and does not redirect to compile_model")


def test_quantize_refuses_and_points_at_torchnative_quant():
    _assert_refuses(
        lambda: intelnpu.quantize_,
        "neural-compressor",
        "torchnative.quant.quantize_",
        "docs/graph/QUANT2.md",
    )
    print("ok   intelnpu: neural-compressor quantization refused, redirected to quant.quantize_")


def test_dynamo_backend_refuses_permanently():
    """Must not weaken: torch.compile is a permanent refusal, docs/graph/COMPILE.md."""
    text = _assert_refuses(
        lambda: intelnpu.dynamo_backend, "PEP 523", "abi3", "docs/graph/COMPILE.md"
    )
    assert "will not" in text, text
    print("ok   intelnpu: torch.compile backend refused permanently, naming PEP 523")


def test_supported_ops_is_smaller_than_what_openvino_accepts_and_says_so():
    """The distinction coreml.py draws: what a target accepts vs what we can emit."""
    ops = supported_ops()
    assert ops == frozenset({"MatMul", "Add"}), ops
    assert SUPPORTED_MODULES == frozenset({"torch.nn.Linear"}), SUPPORTED_MODULES
    print(f"ok   intelnpu: supported_ops() is {sorted(ops)} over {sorted(SUPPORTED_MODULES)}")


# ------------------------------------------- against a real OpenVINO, CPU


_SUBPROCESS = r"""
import json, os, sys
sys.path.insert(0, {vendor!r})
import torch

out = {{"is_shim": hasattr(torch._C, "_aten_implemented")}}
if not out["is_shim"]:
    print(json.dumps(out)); raise SystemExit(0)

from torchnative.export import intelnpu as I

path = os.environ["TORCHNATIVE_OPENVINO_C"]
ov = I.OpenVINO(path)
try:
    out["devices"] = list(ov.devices())
    # The whole evidence bundle, for CPU, against a real runtime.
    out["evidence"] = I.evidence(ov, "CPU", "CPU")
finally:
    ov.close()

# A real torch.nn.Linear tree, lowered and run, against the shim's eager answer.
torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(8, 5), torch.nn.ReLU(), torch.nn.Linear(5, 3)
)
x = torch.randn(4, 8)
eager = model(x)
model, report = I._compile_model(model, device="CPU", library=path)
lowered = model(x)
out["report"] = report
out["max_abs_diff"] = float((lowered - eager).abs().max())
out["control_diff"] = float((model(torch.randn(4, 8)) - eager).abs().max())
out["leaf_type"] = type(model[0]).__name__
out["is_nn_module"] = isinstance(model[0], torch.nn.Module)

# A *square* Linear, separately. On a non-square layer a flipped `transpose_b`
# is a shape error and OpenVINO refuses to load the IR at all; on a square one
# it loads and computes `x @ W` instead of `x @ W.T` -- the same shape, plausible
# magnitudes, wrong function. This is the only case where that fault is silent,
# so it needs its own arithmetic check rather than riding on the one above.
square_layer = torch.nn.Linear(6, 6)
weight, bias = square_layer.weight.detach(), square_layer.bias.detach()
sx = torch.randn(2, 6)
square_eager = square_layer(sx)
# What the flipped-transpose fault would produce: x @ W rather than x @ W.T.
square_wrong = sx @ weight + bias
square, _ = I._compile_model(
    torch.nn.Sequential(square_layer), device="CPU", library=path
)
square_out = square(sx)
out["square_diff"] = float((square_out - square_eager).abs().max())
out["square_wrong_diff"] = float((square_out - square_wrong).abs().max())
out["square_fault_is_visible"] = float((square_eager - square_wrong).abs().max())

# The numpy bridge the archived library's whole FFI boundary is built on.
try:
    x.to(torch.float16).numpy()
    out["has_numpy_method"] = True
except NotImplementedError as error:
    out["has_numpy_method"] = False
    out["numpy_error"] = str(error)[:120]
try:
    import numpy
    torch.from_numpy(numpy.zeros((2, 2), dtype="float16"))
    out["has_from_numpy"] = True
except NotImplementedError as error:
    out["has_from_numpy"] = False
    out["from_numpy_error"] = str(error)[:120]
except ImportError:
    out["has_from_numpy"] = None

print(json.dumps(out))
"""

_cached = []


def _openvino_run():
    """Run the real-OpenVINO half in the vendored tree, or return None with a reason.

    A subprocess for the reason docs/graph/NPU2.md's fixture uses one: the shim torch
    lives in the vendored tree, not on this process's path, and `is_shim` in the
    payload is what proves the measurement was taken against it rather than
    against some other torch that happened to import.
    """
    if _cached:
        return _cached[0]
    path = os.environ.get(_OV_ENV)
    if not path:
        _cached.append(
            (None, f"{_OV_ENV} is not set, so there is no OpenVINO C runtime to load")
        )
        return _cached[0]
    if not os.path.isfile(path):
        _cached.append((None, f"{_OV_ENV}={path!r} does not name a file"))
        return _cached[0]
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")):
        _cached.append(
            (None, "the vendored shim is not installed (run vendor/install_shim.sh)")
        )
        return _cached[0]
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS.format(vendor=_VENDOR_DIR)],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"openvino subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    _cached.append((payload, None))
    return _cached[0]


def _openvino_or_skip(what):
    payload, reason = _openvino_run()
    if payload is None:
        print(f"   (skipped {what}: {reason})")
        return None
    assert payload["is_shim"], (
        "the subprocess did not load torchnative's shim -- torch._C has no "
        "_aten_implemented(), so this measurement is against some other torch"
    )
    return payload


def test_the_hand_written_ir_is_accepted_by_a_real_openvino():
    """The claim this round would otherwise have had to leave unverified.

    Both IR documents in this module were written by this project, not emitted by
    OpenVINO's serialiser. Whether OpenVINO accepts them is not a matter of
    opinion, and until it was checked the honest status of every downstream claim
    was "assuming the IR loads". It loads.
    """
    payload = _openvino_or_skip("IR acceptance")
    if payload is None:
        return
    evidence = payload["evidence"]
    assert evidence["execution_devices"] == ["CPU"], evidence
    assert evidence["linear_execution_devices"] == ["CPU"], evidence
    print(
        f"ok   intelnpu: both hand-written IR documents load and compile on a real "
        f"OpenVINO (devices {payload['devices']}), EXECUTION_DEVICES={evidence['execution_devices']}"
    )


def test_a_linear_runs_through_openvino_and_agrees_with_the_reference():
    """Inference, not just compilation -- and the control that makes it evidence."""
    payload = _openvino_or_skip("OpenVINO inference")
    if payload is None:
        return
    evidence = payload["evidence"]
    agreement = evidence["linear_max_abs_diff"]
    control = evidence["linear_control_diff"]
    assert agreement <= 1e-2, evidence
    assert control > 1e-2, (
        f"a different input moved the answer by only {control}; the device could be "
        f"returning a constant and the agreement would prove nothing"
    )
    print(
        f"ok   intelnpu: a Linear ran through openvino_c and agreed with the reference "
        f"to {agreement:g}, while a different input moved it by {control:g}"
    )


def test_compile_model_lowers_a_real_nn_linear_tree_and_the_numbers_agree():
    """The user-facing shape: swap the leaves, call the model, compare to eager.

    This is the mechanism `NPUModelForCausalLM` rests on, exercised on a real
    `torch.nn.Sequential` under the shim. What a Windows machine changes is the
    device string; the lowering, the FFI crossing and the arithmetic are these.
    """
    payload = _openvino_or_skip("compile_model lowering")
    if payload is None:
        return
    report = payload["report"]
    assert report["swapped"] == ["0", "2"], report
    assert report["left_on_cpu"] == {"ReLU": 1}, report
    assert report["fully_offloaded"] is False, report
    assert report["execution_devices"] == ["CPU"], report
    assert payload["leaf_type"] == "_NPULinear", payload["leaf_type"]
    assert payload["is_nn_module"] is True, payload
    assert payload["max_abs_diff"] <= 1e-2, payload["max_abs_diff"]
    assert payload["control_diff"] > payload["max_abs_diff"] * 100, payload
    print(
        f"ok   intelnpu: compile_model swapped {report['swapped']}, left "
        f"{report['left_on_cpu']} on the CPU, and the lowered model agrees with eager "
        f"to {payload['max_abs_diff']:g} (different input: {payload['control_diff']:g})"
    )


def test_a_square_linear_pins_the_transpose_that_would_otherwise_be_silent():
    """`transpose_b="true"` with a `[out, in]` weight, checked by arithmetic.

    On a non-square Linear, flipping that attribute is a shape error and OpenVINO
    refuses the IR -- so the existing tests catch it, but they catch it as a load
    failure and would keep passing if the shapes ever happened to line up. On a
    **square** Linear it loads and computes `x @ W` instead of `x @ W.T`: right
    shape, plausible magnitudes, wrong function. This is the case where that
    fault is silent, which is the only case worth a dedicated check.

    The third assertion is what makes the first two mean anything: if the right
    and wrong answers were close together, agreeing with one would not be
    evidence of not being the other.
    """
    payload = _openvino_or_skip("square Linear transpose")
    if payload is None:
        return
    visible = payload["square_fault_is_visible"]
    assert visible > 1e-2, (
        f"x @ W and x @ W.T differ by only {visible} on this weight, so agreeing "
        f"with one does not rule out the other and this check proves nothing"
    )
    assert payload["square_diff"] <= 1e-2, payload["square_diff"]
    assert payload["square_wrong_diff"] > 1e-2, (
        f"the lowered square Linear is within {payload['square_wrong_diff']} of the "
        f"*transposed* answer -- transpose_b may be emitting the wrong function"
    )
    print(
        f"ok   intelnpu: a square Linear agrees with x@W.T to "
        f"{payload['square_diff']:g} and differs from x@W by "
        f"{payload['square_wrong_diff']:g} (the two are {visible:g} apart)"
    )


def test_the_report_names_what_was_left_behind_rather_than_claiming_the_model():
    """`fully_offloaded` is the field that stops "the model is on the NPU" being said.

    The archived library returns `None` from `lower_linear` for anything it does
    not recognise (`compiler.py:173`) and the caller gets a model it believes is
    offloaded. docs/graph/NPU2.md is a whole document about that going unnoticed.
    """
    payload = _openvino_or_skip("offload report")
    if payload is None:
        return
    report = payload["report"]
    assert "ReLU" in report["left_on_cpu"], report
    assert report["fully_offloaded"] is False, report
    print(f"ok   intelnpu: the report names {sorted(report['left_on_cpu'])} as left on the CPU")


def test_the_shim_has_no_numpy_bridge_which_is_why_this_packs_bytes():
    """docs/devices/INTELNPU.md section 1.5, measured rather than assumed.

    The archived library's entire FFI boundary is numpy: `np.ctypeslib.ndpointer`
    argtypes (`backend/bindings.py:14-18`), `.numpy()` at every call site
    (`backend/runtime.py:64,76,97,183-184`), `torch.from_numpy` on the way back
    (`runtime.py:134`). Neither half exists on this shim. That -- not libtorch --
    is what would actually stop that package being hosted here, and it is why
    this module crosses in `struct`-packed bytes.

    Checked by behaviour, so the day a numpy bridge lands this goes red and the
    conclusion gets revisited instead of inherited.
    """
    payload = _openvino_or_skip("numpy bridge")
    if payload is None:
        return
    assert payload["has_numpy_method"] is False, payload
    assert "numpy" in payload["numpy_error"], payload["numpy_error"]
    if payload["has_from_numpy"] is not None:
        assert payload["has_from_numpy"] is False, payload
    print(
        f"ok   intelnpu: the shim has no .numpy() and no torch.from_numpy "
        f"({payload['numpy_error'][:60]}...), so the FFI crossing is bytes"
    )


# ------------------------------------------------------- real hardware only


def test_probe_on_real_hardware():
    """The only test that can prove NPU execution. Skips loudly everywhere else."""
    if sys.platform != "win32" and not sys.platform.startswith("linux"):
        print(
            f"   (skipped: sys.platform is {sys.platform!r}; the OpenVINO NPU plugin "
            f"ships for Windows and Linux on x86-64 only, so there is no Intel NPU "
            f"to reach from here -- this is not a missing install)"
        )
        return
    from torchnative.export.intelnpu import OpenVINO, probe

    try:
        ov = OpenVINO()
    except IntelNPUUnavailable as exc:
        print(f"   (skipped: OpenVINO C runtime not loadable -- {exc})")
        return
    try:
        devices = ov.devices()
    finally:
        ov.close()
    if "NPU" not in devices:
        print(
            f"   (skipped: OpenVINO loaded and reports devices {list(devices)!r}, which "
            f"does not include 'NPU'; no Intel NPU driver / NPU plugin on this machine)"
        )
        return

    report = probe()
    assert report["verdict"] == "npu", report
    assert report["assert_device"] == "NPU", report
    assert report["execution_devices"] == ["NPU"], report
    assert report["linear_execution_devices"] == ["NPU"], report
    assert report["control_moved"] is True, report
    assert report["numeric_control_moved"] is True, report
    assert report["execution_devices"] != report["execution_devices_control"], report
    print(
        f"ok   intelnpu: a model compiled for NPU reports EXECUTION_DEVICES="
        f"{report['execution_devices']} on {report.get('FULL_DEVICE_NAME')!r}, the CPU "
        f"control reports {report['execution_devices_control']}, and the Linear it ran "
        f"agrees to {report['agreement']:g} with a control of {report['control_diff']:g}"
    )


# --------------------------------------------------------- the withdrawal

#: Every name `intelnpu` used to export as "call this to run your model".
#: Listed here rather than read off `intelnpu._WITHDRAWN`, deliberately: a test
#: that asks the module which names it withdrew cannot fail when a name is
#: quietly put back. This list is the independent statement of the claim
#: (CLAUDE.md §5.5).
WITHDRAWN_INTELNPU = (
    "compile_model",
    "NPULinear",
    "compile_module",
    "quantize_",
    "dynamo_backend",
)


def test_every_withdrawn_intelnpu_name_refuses_by_name_and_names_the_replacement():
    """Withdrawn is not deleted: reaching for one says so, and says what instead.

    A silent `AttributeError` teaches nothing. Each of these has to name itself,
    say it was withdrawn, name `AutoModelForCausalLM` as the replacement, say
    plainly that the replacement does not exist yet, and -- for this Intel path
    -- name `optimum-intel`, which ships today what this module was reaching for.
    """
    for name in WITHDRAWN_INTELNPU:
        try:
            getattr(intelnpu, name)
        except IntelNPUWithdrawn as exc:
            text = str(exc)
        else:
            raise AssertionError(f"intelnpu.{name} did not refuse")
        assert text.startswith(f"torchnative intelnpu: {name} was withdrawn"), text
        assert "AutoModelForCausalLM" in text, (name, text)
        # See test_qnn.py: the replacement landed, so "not implemented yet" is
        # no longer true of it. The compile step is what remains unimplemented.
        assert "now EXISTS" in text, (name, text)
        assert "not implemented" in text, (name, text)
        assert "optimum-intel" in text or "optimum.intel" in text, (name, text)
        assert "OVModelForCausalLM" in text, (name, text)
    print(
        f"ok   intelnpu: {len(WITHDRAWN_INTELNPU)} withdrawn names refuse by name, "
        f"naming AutoModelForCausalLM (which now exists, though the compile "
        f"step does not) and optimum-intel"
    )


def test_the_withdrawal_does_not_promise_a_working_npu_device_string():
    """`.to("npu")` appears in the message as a shape, never as a working call.

    `torch.device("npu")` raises on this shim -- `_rename_privateuse1_backend`
    is a stub -- so a refusal that showed the snippet without saying so would
    send the reader at a second wall with no name on it.
    """
    text = str(intelnpu._withdrawal_message("compile_model"))
    assert "model.to(torchnative.device.npu)" in text, text
    assert 'model.to("npu")' not in text, text
    assert "torch.device" in text and "raises" in text, text
    assert "_rename_privateuse1_backend" in text, text
    assert "on purpose" in text, text
    print("ok   intelnpu: the withdrawal names the second wall (no `npu` device) too")


def _tiny_torch():
    """`import torch` from the vendored tree, past VENDOR.md wall 1.

    This file's other tests never imported torch, so nothing here had needed
    `TORCH_USE_RTLD_GLOBAL` before: the vendored tree's `_load_global_deps()`
    looks for a `libtorch_global_deps` that this build does not ship, and
    upstream's own escape hatch is that variable.
    """
    os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
    import torch

    assert hasattr(torch._C, "_aten_implemented"), (
        "not the torchnative shim -- torch._C has no _aten_implemented"
    )
    return torch


def test_an_oversized_leaf_is_left_behind_and_named_not_fatal():
    """The Qwen3-4B wall: one oversized `lm_head` must not refuse a whole model.

    A real user ran Qwen3-4B-Instruct-2507 on an Intel NPU and the whole model
    was refused because `lm_head` is 151936 x 2560 and 151936 > MAX_DIM. Every
    model with a large vocabulary has that layer and it is usually the single
    largest weight, so a fatal refusal makes every real LLM unreachable.

    The fix is the archived Intel library's OUTCOME with the opposite
    epistemics: the layer stays on the CPU (as `nn/linear.py:66` does) but is
    **named in the report with its shape and the limit it exceeded**, and
    `fully_offloaded` goes False -- where the archived library returns the torch
    layer silently and leaves an unannounced CPU layer inside a model the caller
    believes is offloaded. That silence is what docs/graph/NPU2.md is about.
    """
    torch = _tiny_torch()
    from torchnative.export.intelnpu import MAX_DIM, plan_lowering

    model = torch.nn.Sequential()
    model.add_module("ok", torch.nn.Linear(16, 16))
    model.add_module("lm_head", torch.nn.Linear(64, MAX_DIM + 1, bias=False))

    plan = plan_lowering(model)
    assert plan["eligible"] == ["ok"], plan["eligible"]
    assert len(plan["skipped"]) == 1, plan["skipped"]
    name, reason = plan["skipped"][0]
    assert name == "lm_head", name
    assert str(MAX_DIM + 1) in reason, reason          # its shape
    assert f"MAX_DIM={MAX_DIM}" in reason, reason      # the limit it exceeded
    assert "stays on the CPU" in reason, reason
    assert plan["fully_offloaded"] is False, plan
    print(
        f"ok   intelnpu: an out_features={MAX_DIM + 1} leaf is left behind and "
        f"named rather than refusing the model, and fully_offloaded is False"
    )


def test_how_much_moved_is_a_value_and_not_only_prose():
    """A caller who ignores the list must still not conclude a full offload."""
    torch = _tiny_torch()
    from torchnative.export.intelnpu import MAX_DIM, plan_lowering

    model = torch.nn.Sequential()
    model.add_module("a", torch.nn.Linear(100, 100, bias=False))     # 10000
    model.add_module("big", torch.nn.Linear(100, MAX_DIM + 1, bias=False))

    plan = plan_lowering(model)
    assert plan["parameters_total"] == 10000 + 100 * (MAX_DIM + 1), plan
    assert plan["parameters_moved"] == 10000, plan
    assert 0.0 < plan["fraction_moved"] < 0.01, plan["fraction_moved"]
    assert plan["fraction_moved"] == plan["parameters_moved"] / plan["parameters_total"]

    whole = torch.nn.Sequential()
    whole.add_module("a", torch.nn.Linear(8, 8))
    full = plan_lowering(whole)
    assert full["fraction_moved"] == 1.0, full
    assert full["fully_offloaded"] is True, full
    print(
        f"ok   intelnpu: fraction_moved is a value -- {plan['fraction_moved']:.5f} "
        f"when the largest leaf stays behind, 1.0 when nothing does"
    )


def test_the_predicate_matches_quantize_s_signature_and_narrows_selection():
    """One idea, one spelling: `predicate(name, module) -> bool`, as quantize_ has."""
    torch = _tiny_torch()
    import inspect

    from torchnative.export.intelnpu import _compile_model, plan_lowering
    from torchnative.quant import quantize_

    assert "predicate" in inspect.signature(quantize_).parameters
    for fn in (plan_lowering, _compile_model):
        assert "predicate" in inspect.signature(fn).parameters, fn

    model = torch.nn.Sequential()
    model.add_module("keep", torch.nn.Linear(8, 8))
    model.add_module("drop", torch.nn.Linear(8, 8))

    seen = []

    def predicate(name, module):
        seen.append((name, type(module).__name__))
        return name != "drop"

    plan = plan_lowering(model, predicate=predicate)
    assert plan["eligible"] == ["keep"], plan["eligible"]
    assert plan["skipped"] == [("drop", "excluded by predicate")], plan["skipped"]
    assert plan["fully_offloaded"] is False, plan
    assert sorted(seen) == [("drop", "Linear"), ("keep", "Linear")], seen
    print(
        "ok   intelnpu: predicate(name, module) narrows the selection and is the "
        "same spelling torchnative.quant.quantize_ uses"
    )


def test_the_plan_and_the_real_lowering_cannot_drift_apart():
    """`plan_lowering` must not be a second copy of the eligibility rule.

    It calls `linear_ir` -- the same pure function `_NPULinear.__init__` calls.
    This asserts they agree on both sides of the limit, so a change to the rule
    cannot make the plan lie.
    """
    torch = _tiny_torch()
    from torchnative.export.intelnpu import (
        MAX_DIM,
        IntelNPUUnsupported,
        _NPULinear,
        plan_lowering,
    )

    for out, expect_eligible in ((16, True), (MAX_DIM + 1, False)):
        layer = torch.nn.Linear(8, out, bias=False)
        holder = torch.nn.Sequential()
        holder.add_module("x", layer)
        planned = plan_lowering(holder)["eligible"] == ["x"]
        try:
            _NPULinear.from_torch(layer, "NPU", None)
            really = True
        except IntelNPUUnsupported:
            really = False
        assert planned == expect_eligible, (out, planned)
        assert planned == really, (
            f"out_features={out}: plan says eligible={planned} but the real "
            f"constructor says {really} -- the plan has drifted from the rule"
        )
    print("ok   intelnpu: the plan agrees with _NPULinear's own refusal at the limit")


def test_the_kept_machinery_is_still_exported():
    """The verdict logic, the emitters and the probe are kept -- they are evidence.

    This is the other half of the withdrawal and the half that can rot: it would
    be easy to take the whole file out with the API. These names are how this
    project tells "it ran on the NPU" from "the answer happened to be right".
    """
    kept = (
        "assert_execution_device",
        "verdict_execution_devices",
        "parse_execution_devices",
        "EXECUTION_DEVICES",
        "probe",
        "available_devices",
        "npu_available",
        "library_candidates",
        "load_openvino_c",
        "OpenVINO",
        "minimal_ir",
        "linear_ir",
        "pack_f16",
        "unpack_f16",
        "supported_ops",
    )
    for name in kept:
        assert name in intelnpu.__all__, f"{name} fell out of __all__"
        assert getattr(intelnpu, name) is not None, name
    for name in WITHDRAWN_INTELNPU:
        assert name not in intelnpu.__all__, f"{name} is still exported"
    print(f"ok   intelnpu: {len(kept)} verification names kept, {len(WITHDRAWN_INTELNPU)} withdrawn")


def test_an_unknown_name_is_still_a_plain_attribute_error():
    """The withdrawal `__getattr__` must not swallow ordinary typos."""
    try:
        intelnpu.no_such_name_at_all
    except IntelNPUWithdrawn as exc:  # noqa: BLE001
        raise AssertionError(f"a typo was reported as a withdrawal: {exc}")
    except AttributeError:
        pass
    else:
        raise AssertionError("a missing attribute did not raise")
    print("ok   intelnpu: an unknown name is an AttributeError, not a withdrawal")



if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
