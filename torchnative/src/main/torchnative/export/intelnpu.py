"""Intel NPU (Core Ultra / AI Boost) reach-the-device layer.

`docs/devices/INTELNPU.md` is the design record. This module reaches the Intel NPU through
the OpenVINO **C** API over `ctypes`, compiles a model for device `"NPU"`, runs it,
and **asserts where it actually ran** by reading OpenVINO's own `EXECUTION_DEVICES`
property back off the compiled model. The assertion, not the arithmetic, is the
point --- see below.

Why that assertion is the whole point. `intel_npu_acceleration_library`, whose
source `docs/devices/INTELNPU.md` §1 dissects, silently rewrites the target device in C++
when the NPU is missing --- `inference.h:77-79`:

    if (!_isNPUAvailable(core)) {
        // Fallback to auto in case there is no NPU device. Handle this situation at python level
        device = "CPU";
    }

and the Python side only calls `warnings.warn` about it (`backend/utils.py:56-60`).
The answers stay correct, so nothing downstream notices. `docs/graph/NPU2.md` records the
CoreML round making exactly this mistake and only catching it by asking CoreML's own
`MLComputePlan` which unit it picked. `EXECUTION_DEVICES` is this stack's equivalent
question, and this module raises on the wrong answer instead of warning.

Why the C API and not the C++ one, and not the archived library. Per
`docs/devices/INTELNPU.md` §1.1, the archived library's own extension links **only**
`openvino::runtime` --- no libtorch, no ATen, no `c10`, no `PyInit_`; it is a plain
`extern "C"` DLL that Python loads with `ctypes` and feeds numpy pointers. That is
what makes any of this hostable by torchnative at all, since torchnative replaces
`torch._C` with a Rust extension and ships no libtorch. `openvino_c` gives us the
same property without taking an archived, Windows-only, end-of-life dependency:
a stable `extern "C"` surface, loadable by `ctypes`, no C++ of our own to ship,
no build-time OpenVINO SDK, and no threat to the abi3 single-wheel discipline.

What it reaches the device *with*. `compile_model(model, device="NPU")` walks a
module tree and replaces every `torch.nn.Linear` with an `NPULinear` whose forward
runs on the device. That is not an approximation of what the archived library does
--- it is the same mechanism. `intel_npu_acceleration_library.compile`
(`compiler.py:42-81`) does no tracing at all; it is `named_children()` +
`add_module()` (`compiler.py:103-141`) swapping `torch.nn.Linear` for a leaf that
dispatches over the FFI (`compiler.py:144-160`, `nn/linear.py:35-51`). That single
substitution is the whole of what makes `NPUModelForCausalLM.from_pretrained(...)`
followed by `model.generate(...)` run on an NPU: `generate()` never learns
anything about the hardware.

What this module does NOT do, and says so by name rather than pretending: it does
not lower a *captured graph* (`compile_module` --- a different door from
`compile_model`, and the distinction matters because they cover different amounts
of a model), it does not quantize, and it offers no `torch.compile` backend. See
the refusals at the bottom of this file.

Where the claims in this file were measured. The IR, the C bindings, the weights
blob, the inference and the numerics were all exercised against a real OpenVINO
2026.3.1 --- on an arm64 Mac, for device `"CPU"`. What an Intel NPU machine adds
is the string `"NPU"` coming back out of `EXECUTION_DEVICES`. docs/devices/INTELNPU.md
section 3.3 draws that line precisely and section 4 is the Windows procedure.
"""

from __future__ import annotations

import ctypes
import struct
import os
import sys

__all__ = [
    "IntelNPUUnavailable",
    "IntelNPUExecutionError",
    "IntelNPUUnsupported",
    "OV_STATUS",
    "EXECUTION_DEVICES",
    "MAX_DIM",
    "library_candidates",
    "parse_execution_devices",
    "verdict_execution_devices",
    "minimal_ir",
    "linear_ir",
    "pack_f16",
    "unpack_f16",
    "load_openvino_c",
    "OpenVINO",
    "available_devices",
    "npu_available",
    "assert_execution_device",
    "probe",
    "supported_ops",
    "NPULinear",
    "compile_model",
    "compile_module",
    "quantize_",
    "dynamo_backend",
]


# --------------------------------------------------------------------------
# Refusals. One exception class per failure kind, message prefixed
# "torchnative intelnpu: ", naming the thing and the reason -- the house style
# set by nnapi.JitFacadeRefused and coreml.CoreMLRefused.
# --------------------------------------------------------------------------


class IntelNPUUnavailable(RuntimeError):
    """The Intel NPU could not be reached: wrong OS, no OpenVINO, or no NPU device."""


class IntelNPUExecutionError(RuntimeError):
    """A model compiled, but OpenVINO says it will not run where we asked.

    This is the anti-silent-fallback exception. It exists because a correct
    answer computed on the CPU is indistinguishable from a correct answer
    computed on the NPU.
    """


class IntelNPUUnsupported(NotImplementedError):
    """Something is refused by name, permanently or for now, with the reason given."""


# --------------------------------------------------------------------------
# Pure helpers. Everything in this section is testable without an NPU, without
# OpenVINO, and on any platform -- which is the point, since the machine this
# was written on is an arm64 Mac.
# --------------------------------------------------------------------------

#: `ov_status_e`, openvino/src/bindings/c/include/openvino/c/ov_common.h:135-163.
OV_STATUS = {
    0: "OK",
    -1: "GENERAL_ERROR",
    -2: "NOT_IMPLEMENTED",
    -3: "NETWORK_NOT_LOADED",
    -4: "PARAMETER_MISMATCH",
    -5: "NOT_FOUND",
    -6: "OUT_OF_BOUNDS",
    -7: "UNEXPECTED",
    -8: "REQUEST_BUSY",
    -9: "RESULT_NOT_READY",
    -10: "NOT_ALLOCATED",
    -11: "INFER_NOT_STARTED",
    -12: "NETWORK_NOT_READ",
    -13: "INFER_CANCELLED",
    -14: "INVALID_C_PARAM",
    -15: "UNKNOWN_C_ERROR",
    -16: "NOT_IMPLEMENT_C_METHOD",
    -17: "UNKNOW_EXCEPTION",
}

#: The property key. `ov::execution_devices` is declared as
#: `Property<std::vector<std::string>, PropertyMutability::RO>{"EXECUTION_DEVICES"}`
#: at openvino/src/inference/include/openvino/runtime/properties.hpp:1409, and is
#: readable through `ov_compiled_model_get_property`
#: (openvino/src/bindings/c/include/openvino/c/ov_compiled_model.h:173-176).
EXECUTION_DEVICES = "EXECUTION_DEVICES"

#: Env var to point at `openvino_c` explicitly, for installs we cannot guess.
#: It is consulted *before* `library_candidates()`, so a machine that has an
#: OpenVINO but is not one of the two NPU platforms can still exercise the whole
#: ctypes layer against a real runtime. That is not a loophole in the NPU
#: refusal -- `assert_execution_device` still refuses unless OpenVINO itself
#: lists an `NPU` device -- it is what let the IR, the C bindings and the
#: numerics in this file be measured on an arm64 Mac rather than asserted.
LIBRARY_ENV = "TORCHNATIVE_OPENVINO_C"

#: The largest weight dimension `linear_ir` will emit. The archived library
#: draws the same line at `nn/linear.py:66` (`if any(dim > 2**17 for dim in
#: layer.weight.shape): return layer`) -- except that it *silently hands the
#: torch layer back*, so an oversized layer stays on the CPU and nothing says
#: so. Here it refuses by name instead, for the reason this whole module
#: exists: a layer that quietly did not move is a silent CPU fallback.
MAX_DIM = 2 ** 17


def library_candidates(platform: str | None = None) -> tuple[str, ...]:
    """Shared-library filenames to try for the OpenVINO C API, newest naming first.

    Pure: takes the platform string rather than reading `sys.platform`, so the
    Windows and Linux answers are both checkable from a Mac.

    Raises:
        IntelNPUUnavailable: on any platform where the OpenVINO NPU plugin does
            not exist. The plugin ships for Windows and Linux on x86-64 only; on
            macOS there is no Intel NPU to reach and no plugin to reach it with.
    """
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return ("openvino_c.dll", "openvino_c_d.dll")
    if platform.startswith("linux"):
        return ("libopenvino_c.so", "libopenvino_c.so.2025", "libopenvino_c.so.2024")
    raise IntelNPUUnavailable(
        f"torchnative intelnpu: platform {platform!r} has no Intel NPU path. The NPU "
        f"is reached through OpenVINO's NPU plugin, which Intel ships for Windows and "
        f"Linux on x86-64 only -- see docs/devices/INTELNPU.md section 1.3. The archived "
        f"intel_npu_acceleration_library draws the same line explicitly at "
        f"backend/bindings.py:56-59, refusing every sys.platform that is not 'win32' or "
        f"'linux'. This is not a missing feature; there is no such hardware here."
    )


def parse_execution_devices(value: str) -> tuple[str, ...]:
    """Split the `EXECUTION_DEVICES` property string into device names.

    OpenVINO renders a `std::vector<std::string>` property through the C API as a
    single string. Observed spellings vary by version and by plugin -- a bare
    ``NPU``, a comma-joined ``NPU,CPU``, a space-joined ``NPU CPU``, and some
    versions bracket the list. All four are accepted; anything left over after
    stripping separators is preserved verbatim so a surprise is visible rather
    than silently dropped.
    """
    text = value.strip().strip("[]{}()")
    parts = [p.strip().strip("'\"") for p in text.replace(",", " ").split()]
    return tuple(p for p in parts if p)


def verdict_execution_devices(devices, expect: str = "NPU") -> str:
    """Return the matching device, or raise `IntelNPUExecutionError`.

    The rule is strict on purpose:

    * the expected device must be present, and
    * it must be the *only* device.

    The second half matters. A ``HETERO:NPU,CPU`` compilation reports both, which
    means part of the graph runs on the CPU -- exactly the partial-offload case
    `docs/graph/NPU2.md` caught on the CoreML side, where results were right and the
    Neural Engine was never touched. Accepting "NPU is in the list" would let that
    through.
    """
    devices = tuple(devices)
    if not devices:
        raise IntelNPUExecutionError(
            "torchnative intelnpu: OpenVINO reported an empty EXECUTION_DEVICES for "
            "this compiled model, so there is no evidence of where it will run. "
            "Refusing to claim NPU execution on no evidence."
        )
    if expect not in devices:
        raise IntelNPUExecutionError(
            f"torchnative intelnpu: asked OpenVINO to compile for {expect!r}, but the "
            f"compiled model reports EXECUTION_DEVICES = {list(devices)!r}. This is the "
            f"silent fallback described in docs/devices/INTELNPU.md section 1.3 -- "
            f"intel_npu_acceleration_library's inference.h:77-79 rewrites the device to "
            f"\"CPU\" when the NPU is absent and only warns. The results would still be "
            f"correct, which is why this has to be checked and not inferred."
        )
    if len(devices) > 1:
        raise IntelNPUExecutionError(
            f"torchnative intelnpu: compiled model reports EXECUTION_DEVICES = "
            f"{list(devices)!r}. {expect!r} is present but so is at least one other "
            f"device, meaning the graph is split and part of it runs elsewhere. "
            f"docs/graph/NPU2.md records a partial offload of exactly this kind going "
            f"unnoticed because the answers were still right. Refused."
        )
    return devices[0]


def minimal_ir(name: str = "torchnative_intelnpu_probe", size: int = 8) -> str:
    """An OpenVINO IR v11 document for `Parameter -> ReLU -> Result`, f16, no weights.

    This exists so the probe has something real to compile. Nothing is written to
    disk: the string goes to `ov_core_read_model_from_memory_buffer`
    (ov_core.h:189-194) with a NULL weights tensor, which is legal because the graph
    has no constants.

    f16 is deliberate. Per `docs/graph/NPU2.md`, the CoreML round's models ran on the CPU
    precisely *because* they were compiled float32, and the Neural Engine is
    float16-only; the Intel NPU is likewise a low-precision engine, and asking it for
    f32 is a good way to get a compile-time redirect to the CPU and never notice.

    This document is **accepted by a real OpenVINO**: `test_the_hand_written_ir_is
    _accepted_by_a_real_openvino` reads it back through
    `ov_core_read_model_from_memory_buffer` and compiles it. See docs/devices/INTELNPU.md
    section 3.3 for where that was measured and what it does and does not settle.
    """
    dims = f"<dim>1</dim><dim>{int(size)}</dim>"
    return (
        '<?xml version="1.0"?>\n'
        f'<net name="{name}" version="11">\n'
        "  <layers>\n"
        '    <layer id="0" name="input" type="Parameter" version="opset1">\n'
        f'      <data shape="1,{int(size)}" element_type="f16"/>\n'
        f'      <output><port id="0" precision="FP16">{dims}</port></output>\n'
        "    </layer>\n"
        '    <layer id="1" name="act" type="ReLU" version="opset1">\n'
        f'      <input><port id="0" precision="FP16">{dims}</port></input>\n'
        f'      <output><port id="1" precision="FP16">{dims}</port></output>\n'
        "    </layer>\n"
        '    <layer id="2" name="output" type="Result" version="opset1">\n'
        f'      <input><port id="0" precision="FP16">{dims}</port></input>\n'
        "    </layer>\n"
        "  </layers>\n"
        "  <edges>\n"
        '    <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>\n'
        '    <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>\n'
        "  </edges>\n"
        "</net>\n"
    )


def _port(pid: int, dims, names: str = "") -> str:
    tag = f' names="{names}"' if names else ""
    body = "".join(f"<dim>{int(d)}</dim>" for d in dims)
    return f'<port id="{pid}" precision="FP16"{tag}>{body}</port>'


def linear_ir(in_features: int, out_features: int, batch: int = 1, bias: bool = True) -> str:
    """OpenVINO IR v11 for `y = x @ W.T + b`, f16, weights in the companion blob.

    This is `torch.nn.Linear` and nothing else, which is deliberate: it is the
    exact leaf `intel_npu_acceleration_library` replaces. `lower_linear`
    (`compiler.py:144-173`) walks the module tree and swaps `torch.nn.Linear` for
    its own `nn.Linear`, whose `forward` (`nn/linear.py:35-51`) calls `run_matmul`
    over the FFI. Everything else about that library -- the LLM fast paths, the
    horizontal fusion, the dynamo backend -- sits on top of that one substitution.

    Two layout facts, both of which would produce a plausible-looking wrong answer
    if got wrong:

    * **`transpose_b="true"`, so the weight is stored `[out, in]`** -- which is
      already `torch.nn.Linear.weight`'s layout. No transpose happens anywhere;
      getting this backwards yields a shape error for non-square layers and
      silently wrong numbers for square ones.
    * **The bias is `[1, out]` with `auto_broadcast="numpy"`**, not `[out]`. The
      constant's declared shape and its byte count are read independently by
      OpenVINO, so a shape that does not match `size` is a load-time refusal
      rather than a misread.

    The emitted shape was not written from the schema. It was **read off a
    reference document** that OpenVINO's own `ov.save_model` produced for this
    exact graph, and then checked back through `ov_core_read_model_from_memory_buffer`
    -- docs/devices/INTELNPU.md section 3.3.

    Raises:
        IntelNPUUnsupported: for a dimension above `MAX_DIM`, by name.
    """
    for label, value in (("in_features", in_features), ("out_features", out_features)):
        if int(value) > MAX_DIM:
            raise IntelNPUUnsupported(
                f"torchnative intelnpu: {label}={int(value)} exceeds MAX_DIM={MAX_DIM}, "
                f"so this Linear is not lowered. intel_npu_acceleration_library draws "
                f"the same line at nn/linear.py:66 but *returns the torch layer "
                f"unchanged*, which leaves it running on the CPU inside a model the "
                f"caller believes is on the NPU. Refusing by name instead -- an "
                f"unannounced CPU layer is the failure docs/graph/NPU2.md is about."
            )
        if int(value) < 1:
            raise IntelNPUUnsupported(
                f"torchnative intelnpu: {label}={int(value)} is not a positive "
                f"dimension, so there is no Linear to lower."
            )
    in_features, out_features, batch = int(in_features), int(out_features), int(batch)
    weight_bytes = out_features * in_features * 2

    layers = [
        f'<layer id="0" name="input" type="Parameter" version="opset1">'
        f'<data shape="{batch},{in_features}" element_type="f16"/>'
        f"<output>{_port(0, (batch, in_features), 'input')}</output></layer>",
        f'<layer id="1" name="weight" type="Const" version="opset1">'
        f'<data element_type="f16" shape="{out_features}, {in_features}" '
        f'offset="0" size="{weight_bytes}"/>'
        f"<output>{_port(0, (out_features, in_features))}</output></layer>",
        f'<layer id="2" name="matmul" type="MatMul" version="opset1">'
        f'<data transpose_a="false" transpose_b="true"/>'
        f"<input>{_port(0, (batch, in_features))}"
        f"{_port(1, (out_features, in_features))}</input>"
        f"<output>{_port(2, (batch, out_features), '' if bias else 'output')}</output>"
        f"</layer>",
    ]
    edges = [
        '<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>',
        '<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>',
    ]
    tail, tail_port, result_id = 2, 2, 3
    if bias:
        layers.append(
            f'<layer id="3" name="bias" type="Const" version="opset1">'
            f'<data element_type="f16" shape="1, {out_features}" '
            f'offset="{weight_bytes}" size="{out_features * 2}"/>'
            f"<output>{_port(0, (1, out_features))}</output></layer>"
        )
        layers.append(
            f'<layer id="4" name="add" type="Add" version="opset1">'
            f'<data auto_broadcast="numpy"/>'
            f"<input>{_port(0, (batch, out_features))}"
            f"{_port(1, (1, out_features))}</input>"
            f"<output>{_port(2, (batch, out_features), 'output')}</output></layer>"
        )
        edges.append('<edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>')
        edges.append('<edge from-layer="3" from-port="0" to-layer="4" to-port="1"/>')
        tail, tail_port, result_id = 4, 2, 5
    layers.append(
        f'<layer id="{result_id}" name="output" type="Result" version="opset1" '
        f'output_names="output"><input>{_port(0, (batch, out_features))}</input></layer>'
    )
    edges.append(
        f'<edge from-layer="{tail}" from-port="{tail_port}" '
        f'to-layer="{result_id}" to-port="0"/>'
    )
    return (
        '<?xml version="1.0"?>\n<net name="torchnative_linear" version="11">\n'
        "  <layers>\n    " + "\n    ".join(layers) + "\n  </layers>\n"
        "  <edges>\n    " + "\n    ".join(edges) + "\n  </edges>\n</net>\n"
    )


def pack_f16(values) -> bytes:
    """Pack a flat sequence of Python floats as little-endian IEEE half.

    `struct`'s `<e`, not numpy -- and that is the interesting part.

    The archived library's FFI boundary is **numpy all the way down**: its
    `argtypes` are declared as `np.ctypeslib.ndpointer` (`backend/bindings.py:14-18`)
    and every call site converts with `.numpy()` (`backend/runtime.py:64,76,97,
    183-184`), reading results back with `torch.from_numpy` (`runtime.py:134`).
    Neither of those exists on torchnative's shim -- both raise
    `NotImplementedError`, measured in `test_the_shim_has_no_numpy_bridge_which_is
    _why_this_packs_bytes`. So the numpy boundary, not libtorch, is what would
    actually stop that library from being hosted here (docs/devices/INTELNPU.md section 1.5),
    and it is why this module crosses in bytes.
    """
    values = list(values)
    return struct.pack(f"<{len(values)}e", *values)


def unpack_f16(blob: bytes) -> list:
    """Inverse of `pack_f16`. Refuses a byte count that is not a whole number of halves."""
    if len(blob) % 2:
        raise IntelNPUExecutionError(
            f"torchnative intelnpu: OpenVINO returned {len(blob)} bytes for an f16 "
            f"tensor, which is not a whole number of 2-byte halves. Refusing to "
            f"reinterpret a buffer whose element type is not what was asked for."
        )
    return list(struct.unpack(f"<{len(blob) // 2}e", blob))


# --------------------------------------------------------------------------
# The ctypes layer.
# --------------------------------------------------------------------------


class _AvailableDevices(ctypes.Structure):
    """`ov_available_devices_t`, ov_core.h:64-67."""

    _fields_ = [("devices", ctypes.POINTER(ctypes.c_char_p)), ("size", ctypes.c_size_t)]


class _Shape(ctypes.Structure):
    """`ov_shape_t`, ov_shape.h:20-23. Passed **by value** to
    `ov_tensor_create_from_host_ptr` (ov_tensor.h:34-37), hence a Structure
    rather than a pointer."""

    _fields_ = [("rank", ctypes.c_int64), ("dims", ctypes.POINTER(ctypes.c_int64))]


#: `ov_element_type_e`, ov_common.h:171-198. Only `U8` is needed: the weights
#: blob is handed over as an opaque byte tensor and OpenVINO reads the real
#: element types out of the IR's `Const` layers. Binding the whole enum would be
#: 26 values that could drift; binding one that is positionally stable (it has
#: not moved since the enum was introduced) is the smaller thing to be wrong about.
OV_ELEMENT_U8 = 16


def load_openvino_c(path: str | None = None) -> ctypes.CDLL:
    """Load `openvino_c` and declare the argument types we use.

    Only long-standing entry points are bound. In particular the non-variadic
    `ov_core_compile_model_props` is *not* used: it is present on OpenVINO master
    but not necessarily in the release a user has installed, and
    `ov_core_compile_model(core, model, device, 0, &out)` with zero varargs does
    the same job on every version.

    Raises:
        IntelNPUUnavailable: if the library cannot be found or loaded.
    """
    tried = []
    if path:
        names = [path]
    elif os.environ.get(LIBRARY_ENV):
        # Consulted *before* `library_candidates()`, which refuses outright on any
        # platform that is not Windows or Linux. Reading the env var afterwards
        # made it unreachable on exactly the machines where it is the only way in,
        # and that was not a policy -- `library_candidates` raises, so the
        # `names.insert(0, ...)` that used to follow it never ran.
        names = [os.environ[LIBRARY_ENV]]
    else:
        names = list(library_candidates())
    lib = None
    for name in names:
        tried.append(name)
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        raise IntelNPUUnavailable(
            f"torchnative intelnpu: could not load the OpenVINO C runtime. Tried "
            f"{tried!r}. Install the OpenVINO runtime (the archived Intel library "
            f"pinned 2024.4; any release with an NPU plugin will do) and either put "
            f"its bin directory on PATH / LD_LIBRARY_PATH, or set {LIBRARY_ENV} to the "
            f"full path of openvino_c.dll / libopenvino_c.so. On Windows the runtime "
            f"also needs its sibling DLLs resolvable -- see docs/devices/INTELNPU.md section 4."
        )

    c_char_pp = ctypes.POINTER(ctypes.c_char_p)
    lib.ov_core_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.ov_core_create.restype = ctypes.c_int
    lib.ov_core_free.argtypes = [ctypes.c_void_p]
    lib.ov_core_free.restype = None
    lib.ov_core_get_available_devices.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AvailableDevices)]
    lib.ov_core_get_available_devices.restype = ctypes.c_int
    lib.ov_available_devices_free.argtypes = [ctypes.POINTER(_AvailableDevices)]
    lib.ov_available_devices_free.restype = None
    lib.ov_core_get_property.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, c_char_pp]
    lib.ov_core_get_property.restype = ctypes.c_int
    lib.ov_core_read_model_from_memory_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ov_core_read_model_from_memory_buffer.restype = ctypes.c_int
    lib.ov_model_free.argtypes = [ctypes.c_void_p]
    lib.ov_model_free.restype = None
    # Variadic: argtypes covers the fixed prefix only, which is what ctypes wants.
    lib.ov_core_compile_model.restype = ctypes.c_int
    lib.ov_compiled_model_get_property.argtypes = [ctypes.c_void_p, ctypes.c_char_p, c_char_pp]
    lib.ov_compiled_model_get_property.restype = ctypes.c_int
    lib.ov_compiled_model_free.argtypes = [ctypes.c_void_p]
    lib.ov_compiled_model_free.restype = None
    lib.ov_free.argtypes = [ctypes.c_char_p]
    lib.ov_free.restype = None
    # -- tensors and inference (ov_tensor.h, ov_infer_request.h) --------------
    lib.ov_tensor_create_from_host_ptr.argtypes = [
        ctypes.c_int,
        _Shape,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ov_tensor_create_from_host_ptr.restype = ctypes.c_int
    lib.ov_tensor_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.ov_tensor_data.restype = ctypes.c_int
    lib.ov_tensor_get_byte_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.ov_tensor_get_byte_size.restype = ctypes.c_int
    lib.ov_tensor_free.argtypes = [ctypes.c_void_p]
    lib.ov_tensor_free.restype = None
    lib.ov_compiled_model_create_infer_request.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ov_compiled_model_create_infer_request.restype = ctypes.c_int
    lib.ov_infer_request_get_input_tensor.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.ov_infer_request_get_input_tensor.restype = ctypes.c_int
    lib.ov_infer_request_get_output_tensor.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.ov_infer_request_get_output_tensor.restype = ctypes.c_int
    lib.ov_infer_request_infer.argtypes = [ctypes.c_void_p]
    lib.ov_infer_request_infer.restype = ctypes.c_int
    lib.ov_infer_request_free.argtypes = [ctypes.c_void_p]
    lib.ov_infer_request_free.restype = None
    if hasattr(lib, "ov_get_last_err_msg"):
        lib.ov_get_last_err_msg.argtypes = []
        lib.ov_get_last_err_msg.restype = ctypes.c_char_p
    return lib


class OpenVINO:
    """A borrowed `ov_core_t` with the few operations this stage needs.

    Usable as a context manager. Every OpenVINO call goes through `_check`, which
    turns a non-zero `ov_status_e` into an exception carrying the symbolic status
    name and, when the runtime offers it, `ov_get_last_err_msg()` -- so a failure
    on the user's machine reports OpenVINO's account of itself rather than ours.
    """

    def __init__(self, path: str | None = None):
        self._lib = load_openvino_c(path)
        core = ctypes.c_void_p()
        self._check(self._lib.ov_core_create(ctypes.byref(core)), "ov_core_create")
        self._core = core

    # -- plumbing ---------------------------------------------------------
    def _check(self, status: int, what: str) -> None:
        if status == 0:
            return
        detail = ""
        if hasattr(self._lib, "ov_get_last_err_msg"):
            try:
                msg = self._lib.ov_get_last_err_msg()
                if msg:
                    detail = f" -- {msg.decode('utf-8', 'replace')}"
            except Exception:  # pragma: no cover - defensive around a C call
                pass
        raise IntelNPUUnavailable(
            f"torchnative intelnpu: {what} failed with ov_status_e "
            f"{OV_STATUS.get(status, status)} ({status}){detail}"
        )

    def _take_string(self, ptr) -> str:
        try:
            return ptr.value.decode("utf-8", "replace") if ptr.value else ""
        finally:
            if ptr.value:
                self._lib.ov_free(ptr)

    def __enter__(self) -> "OpenVINO":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_core", None) is not None:
            self._lib.ov_core_free(self._core)
            self._core = None

    # -- queries ----------------------------------------------------------
    def devices(self) -> tuple[str, ...]:
        """`ov_core_get_available_devices` -- the same question `common.h:36-39` asks."""
        out = _AvailableDevices()
        self._check(
            self._lib.ov_core_get_available_devices(self._core, ctypes.byref(out)),
            "ov_core_get_available_devices",
        )
        try:
            return tuple(out.devices[i].decode("utf-8", "replace") for i in range(out.size))
        finally:
            self._lib.ov_available_devices_free(ctypes.byref(out))

    def device_property(self, device: str, key: str) -> str:
        value = ctypes.c_char_p()
        self._check(
            self._lib.ov_core_get_property(
                self._core, device.encode(), key.encode(), ctypes.byref(value)
            ),
            f"ov_core_get_property({device}, {key})",
        )
        return self._take_string(value)

    def compile_ir(self, xml: str, device: str = "NPU", weights: bytes | None = None):
        """Read IR from memory and compile it for `device`. Returns an opaque handle.

        **Nothing touches the filesystem**, and that is the sense in which this is
        not an "OpenVINO export path": no `.xml`/`.bin` pair is written, no `ovc`
        or Model Optimizer runs, and the `openvino` Python package is never
        imported. `ov_core_read_model_from_memory_buffer` (ov_core.h:189-194) takes
        the document as a buffer and the weights as an `ov_tensor_t*`, which is
        NULL for a graph with no constants.

        The weights buffer is kept alive on the returned handle rather than
        dropped at the end of this call. `ov_tensor_create_from_host_ptr` does not
        copy -- OpenVINO reads through the pointer during `read_model`, and letting
        Python collect the buffer first is a use-after-free that presents as
        plausible-looking wrong numbers rather than as a crash.
        """
        blob = xml.encode("utf-8")
        keepalive = None
        weights_tensor = None
        if weights is not None:
            keepalive = ctypes.create_string_buffer(weights, len(weights))
            dims = (ctypes.c_int64 * 1)(len(weights))
            shape = _Shape(1, ctypes.cast(dims, ctypes.POINTER(ctypes.c_int64)))
            weights_tensor = ctypes.c_void_p()
            self._check(
                self._lib.ov_tensor_create_from_host_ptr(
                    OV_ELEMENT_U8,
                    shape,
                    ctypes.cast(keepalive, ctypes.c_void_p),
                    ctypes.byref(weights_tensor),
                ),
                "ov_tensor_create_from_host_ptr(weights)",
            )
        model = ctypes.c_void_p()
        try:
            self._check(
                self._lib.ov_core_read_model_from_memory_buffer(
                    self._core, blob, len(blob), weights_tensor, ctypes.byref(model)
                ),
                "ov_core_read_model_from_memory_buffer",
            )
            try:
                compiled = ctypes.c_void_p()
                self._check(
                    self._lib.ov_core_compile_model(
                        self._core,
                        model,
                        device.encode(),
                        ctypes.c_size_t(0),
                        ctypes.byref(compiled),
                    ),
                    f"ov_core_compile_model(device={device})",
                )
                compiled._torchnative_weights = keepalive
                return compiled
            finally:
                self._lib.ov_model_free(model)
        finally:
            if weights_tensor is not None:
                self._lib.ov_tensor_free(weights_tensor)

    def infer(self, compiled, input_bytes: bytes) -> bytes:
        """Run one synchronous inference and return the output tensor's raw bytes.

        The input tensor is **borrowed from the infer request** rather than created:
        `ov_infer_request_get_input_tensor` hands back the buffer OpenVINO already
        allocated at the model's declared shape and element type, and this copies
        into it. That avoids binding `ov_element_type_e` and `ov_shape_t` for the
        activation path at all -- the element type is whatever the IR said, which
        is the only place it should be decided.

        The byte count is checked against what OpenVINO reports rather than
        assumed. A mismatch here is a lowering bug (a shape in the IR that is not
        the shape being fed) and it would otherwise show up as garbage numbers.
        """
        request = ctypes.c_void_p()
        self._check(
            self._lib.ov_compiled_model_create_infer_request(compiled, ctypes.byref(request)),
            "ov_compiled_model_create_infer_request",
        )
        try:
            tensor = ctypes.c_void_p()
            self._check(
                self._lib.ov_infer_request_get_input_tensor(request, ctypes.byref(tensor)),
                "ov_infer_request_get_input_tensor",
            )
            size = ctypes.c_size_t()
            self._check(
                self._lib.ov_tensor_get_byte_size(tensor, ctypes.byref(size)),
                "ov_tensor_get_byte_size(input)",
            )
            if size.value != len(input_bytes):
                raise IntelNPUExecutionError(
                    f"torchnative intelnpu: the compiled model's input tensor is "
                    f"{size.value} bytes but {len(input_bytes)} were supplied. The IR "
                    f"this module emitted declares a shape that is not the shape being "
                    f"fed to it. Refusing rather than truncating -- a short write here "
                    f"produces numbers that look like an arithmetic error."
                )
            data = ctypes.c_void_p()
            self._check(self._lib.ov_tensor_data(tensor, ctypes.byref(data)), "ov_tensor_data(input)")
            ctypes.memmove(data, input_bytes, len(input_bytes))
            self._check(self._lib.ov_infer_request_infer(request), "ov_infer_request_infer")
            out = ctypes.c_void_p()
            self._check(
                self._lib.ov_infer_request_get_output_tensor(request, ctypes.byref(out)),
                "ov_infer_request_get_output_tensor",
            )
            out_size = ctypes.c_size_t()
            self._check(
                self._lib.ov_tensor_get_byte_size(out, ctypes.byref(out_size)),
                "ov_tensor_get_byte_size(output)",
            )
            out_data = ctypes.c_void_p()
            self._check(self._lib.ov_tensor_data(out, ctypes.byref(out_data)), "ov_tensor_data(output)")
            return ctypes.string_at(out_data, out_size.value)
        finally:
            self._lib.ov_infer_request_free(request)

    def execution_devices(self, compiled) -> tuple[str, ...]:
        """Read `EXECUTION_DEVICES` off a compiled model. The evidence, not an inference."""
        value = ctypes.c_char_p()
        self._check(
            self._lib.ov_compiled_model_get_property(
                compiled, EXECUTION_DEVICES.encode(), ctypes.byref(value)
            ),
            "ov_compiled_model_get_property(EXECUTION_DEVICES)",
        )
        return parse_execution_devices(self._take_string(value))

    def free_compiled(self, compiled) -> None:
        self._lib.ov_compiled_model_free(compiled)


def available_devices(path: str | None = None) -> tuple[str, ...]:
    with OpenVINO(path) as ov:
        return ov.devices()


def npu_available(path: str | None = None) -> bool:
    """True if OpenVINO lists an `NPU` device. The direct analogue of `common.h:36-39`."""
    return "NPU" in available_devices(path)


def assert_execution_device(xml: str | None = None, expect: str = "NPU", path: str | None = None) -> str:
    """Compile `xml` for `expect` and assert OpenVINO says it will run there.

    Returns the device name on success; raises `IntelNPUExecutionError` otherwise.

    This is the single-question form, and on its own it is **not** enough to claim
    NPU execution: a device reading that never moves is not tracking the request,
    and a right device with a cached answer is not running anything. `probe()` is
    the form with all four questions and their controls (docs/devices/INTELNPU.md section
    3.2). Use this when you already have the controls elsewhere; use `probe()` to
    make the claim.
    """
    xml = minimal_ir() if xml is None else xml
    with OpenVINO(path) as ov:
        found = ov.devices()
        if expect not in found:
            raise IntelNPUUnavailable(
                f"torchnative intelnpu: OpenVINO loaded, but {expect!r} is not among its "
                f"available devices {list(found)!r}. Either the machine has no Intel NPU, "
                f"or the NPU driver / OpenVINO NPU plugin is not installed. Refusing here "
                f"rather than compiling anyway -- intel_npu_acceleration_library only warns "
                f"at this point (backend/utils.py:56-60) and then silently uses the CPU."
            )
        compiled = ov.compile_ir(xml, expect)
        try:
            return verdict_execution_devices(ov.execution_devices(compiled), expect)
        finally:
            ov.free_compiled(compiled)


def evidence(ov, device: str, control_device: str) -> dict:
    """Gather the whole evidence bundle for `device`, with `control_device` as its foil.

    Split out of `probe` and parameterised on the device for one reason: it means
    the *entire* bundle -- the property read, the IR, the weights blob, the
    inference, both controls and both refusals -- is exercised on any machine with
    an OpenVINO, by asking for `"CPU"` with `"NPU"` as the foil. On the arm64 Mac
    this was written on, every line below runs. What Windows adds is the string
    `"NPU"` coming back out of `EXECUTION_DEVICES`, and nothing else.

    Pure Python floats throughout: no torch, no numpy. `probe()` therefore runs
    standalone on a machine that has only OpenVINO installed, which is what makes
    it the first thing to run on the user's laptop.
    """
    out: dict = {"requested": device}

    # 1. Where does OpenVINO say a model compiled for `device` will run?
    xml = minimal_ir()
    compiled = ov.compile_ir(xml, device)
    try:
        devices = ov.execution_devices(compiled)
    finally:
        ov.free_compiled(compiled)
    out["execution_devices"] = list(devices)

    # 2. The same document compiled for the *other* device. A property that
    #    reports the same thing either way is not tracking the request, and the
    #    first reading would then be evidence of nothing. This is the direct
    #    analogue of docs/graph/NPU2.md section 2's third row.
    control = ov.compile_ir(xml, control_device)
    try:
        control_devices = ov.execution_devices(control)
    finally:
        ov.free_compiled(control)
    out["execution_devices_control"] = list(control_devices)

    # 3. Now run something. A compute plan is a statement of intent; docs/graph/NPU2.md
    #    is explicit that identical outputs are what an ignored plan would look
    #    like, so the arithmetic has to be shown too.
    rows, cols = 4, 3
    weight = [((r * cols + c) % 7 - 3) * 0.25 for r in range(rows) for c in range(cols)]
    bias = [0.5, -0.25, 0.125, 0.0]
    blob = pack_f16(weight) + pack_f16(bias)
    x = [0.5, -1.5, 2.0]
    other = [-2.0, 0.75, 1.25]

    def reference(vec):
        return [
            sum(weight[r * cols + c] * vec[c] for c in range(cols)) + bias[r]
            for r in range(rows)
        ]

    linear = ov.compile_ir(linear_ir(cols, rows, 1, True), device, blob)
    try:
        linear_devices = ov.execution_devices(linear)
        got = unpack_f16(ov.infer(linear, pack_f16(x)))
        moved = unpack_f16(ov.infer(linear, pack_f16(other)))
    finally:
        ov.free_compiled(linear)
    out["linear_execution_devices"] = list(linear_devices)
    out["linear_out"] = got
    out["linear_reference"] = reference(x)
    out["linear_max_abs_diff"] = max(abs(a - b) for a, b in zip(got, reference(x)))
    out["linear_control_diff"] = max(abs(a - b) for a, b in zip(moved, got))
    return out


def judge(bundle: dict, device: str, control_device: str, tolerance: float = 1e-2) -> dict:
    """Turn an `evidence` bundle into a verdict, or refuse by name. Pure.

    Separated from the gathering so the refusals can be tested against
    hand-written bundles -- including bundles no hardware here can produce, such
    as one that reports `NPU` for both the request and its control.
    """
    verdict_execution_devices(bundle["execution_devices"], device)
    verdict_execution_devices(bundle["linear_execution_devices"], device)

    if tuple(bundle["execution_devices"]) == tuple(bundle["execution_devices_control"]):
        raise IntelNPUExecutionError(
            f"torchnative intelnpu: the device control failed. Compiling for "
            f"{device!r} and compiling for {control_device!r} both report "
            f"EXECUTION_DEVICES = {bundle['execution_devices']!r}. The property is "
            f"not tracking the requested device, so the {device} reading is not "
            f"evidence of anything. Refusing to claim {device} execution."
        )

    agreement = bundle["linear_max_abs_diff"]
    control = bundle["linear_control_diff"]
    if agreement > tolerance:
        raise IntelNPUExecutionError(
            f"torchnative intelnpu: the model ran on {device} but disagreed with the "
            f"reference by {agreement}, outside the f16 tolerance {tolerance}. The "
            f"device assignment is not the question here -- the arithmetic is wrong."
        )
    # docs/graph/NPU2.md section 3.5: a tolerance is only worth something if a wrong
    # input fails it by a wide margin. That round's first attempt at this check
    # ran on a model so flat that feeding it an entirely different picture moved
    # the answer by less than a thousandth, and it would have passed on the
    # model's flatness rather than on the device's arithmetic.
    #
    # The `tolerance` floor is not belt and braces. `evidence`'s weights and
    # inputs are quarter-integers, all exactly representable in f16, so a correct
    # device gives `agreement == 0.0` -- at which point `control > agreement * 100`
    # is `control > 0` and passes for *any* non-identical pair of answers,
    # including two that differ in the last bit. A relative margin against zero is
    # not a margin. The floor is what the check actually rests on here, and it is
    # what a device returning a constant would fail.
    floor = max(agreement * 100, tolerance)
    if not control > floor:
        raise IntelNPUExecutionError(
            f"torchnative intelnpu: the numeric control failed. Two different inputs "
            f"produced answers {control} apart, against an agreement of {agreement} "
            f"with the reference; this requires the gap to exceed {floor}. A device "
            f"returning a cached or constant answer looks exactly like this. Refusing "
            f"to report the agreement as evidence (docs/graph/NPU2.md section 3.5)."
        )
    return {
        "verdict": device.lower(),
        "assert_device": bundle["execution_devices"][0],
        "control_moved": True,
        "numeric_control_moved": True,
        "agreement": agreement,
        "control_diff": control,
    }


def probe(path: str | None = None, device: str = "NPU") -> dict:
    """Full evidence bundle plus verdict. The thing to run on the Intel NPU laptop.

    Returns a dict; raises `IntelNPUUnavailable` or `IntelNPUExecutionError` rather
    than reporting a soft failure. docs/devices/INTELNPU.md section 4 is what to read the
    output against, field by field, including what each way of failing means.
    """
    control_device = "CPU" if device != "CPU" else "NPU"
    result: dict = {"platform": sys.platform, "requested": device}
    with OpenVINO(path) as ov:
        result["devices"] = list(ov.devices())
        result["npu_available"] = "NPU" in result["devices"]
        for needed in (device, control_device):
            if needed not in result["devices"]:
                result["verdict"] = "no-npu" if needed == "NPU" else "no-device"
                result["missing"] = needed
                return result
        for key in ("FULL_DEVICE_NAME", "NPU_DRIVER_VERSION", "DEVICE_ARCHITECTURE"):
            try:
                result[key] = ov.device_property(device, key)
            except IntelNPUUnavailable as exc:
                result[key] = f"<unavailable: {exc}>"
        result.update(evidence(ov, device, control_device))
    result.update(judge(result, device, control_device))
    return result


# --------------------------------------------------------------------------
# The module-tree rewrite: the mechanism docs/devices/INTELNPU.md section 1.2 found.
#
# `intel_npu_acceleration_library.compile(model, config)` (`compiler.py:42-81`)
# is not a compiler in the torch.compile sense and does no tracing at all. It
# walks the module tree with `named_children()` / `add_module()` and swaps
# `torch.nn.Linear` for its own leaf whose `forward` dispatches over the FFI.
# That is the whole mechanism behind `NPUModelForCausalLM.from_pretrained` +
# `model.generate(...)`: `generate()` never learns anything about the NPU.
#
# It is reproduced here rather than imported, because §2 rejects taking that
# archived package as a dependency -- and §1.5 shows it could not run on this
# shim anyway, its FFI boundary being numpy.
# --------------------------------------------------------------------------


def _torch():
    """Import torch lazily, so `import torchnative.export.intelnpu` stays cheap."""
    import torch

    return torch


class NPULinear:
    """A `torch.nn.Linear` replacement whose forward runs on the OpenVINO device.

    Constructed through `from_torch`, never directly from a caller's shapes, so the
    weight it holds is the one the original layer held.

    The compiled model is built **once, lazily, on the first forward** and then
    reused, because compilation for the NPU goes through the driver-resident
    compiler (`NPU_COMPILER_TYPE="DRIVER"`, `inference.h:86`) and is the expensive
    part; the archived library caches at two levels for the same reason
    (`ov::cache_dir` at `inference.h:82`, and a pickle at `modelling.py:95-97,112`).

    **The device assertion happens at compile time, not at forward time, and it
    raises.** A compiled model that reports anything but the requested device is
    refused before it ever produces a number -- because once it has produced a
    number the number is correct and there is nothing left to notice.

    The batch dimension is part of the compiled shape, so a differently-shaped
    input recompiles. That is stated rather than hidden: it is the cost of a
    static-shape IR and it is why this is a leaf-replacement stage and not yet a
    whole-model one.
    """

    def __new__(cls, *args, **kwargs):
        # Built as a subclass of the *shim's* nn.Module at first use rather than at
        # import, so this module can be imported (and its refusals tested) without
        # torch being importable at all.
        torch = _torch()
        if not issubclass(cls, torch.nn.Module):
            cls = type("NPULinear", (NPULinear, torch.nn.Module), {})
            obj = torch.nn.Module.__new__(cls)
            return obj
        return super().__new__(cls)

    def __init__(self, weight, bias=None, device: str = "NPU", library: str | None = None):
        torch = _torch()
        torch.nn.Module.__init__(self)
        if weight.dim() != 2:
            raise IntelNPUUnsupported(
                f"torchnative intelnpu: NPULinear needs a 2-D weight, got shape "
                f"{tuple(weight.shape)}. There is no Linear here to lower."
            )
        if not weight.dtype.is_floating_point:
            raise IntelNPUUnsupported(
                f"torchnative intelnpu: NPULinear will not lower a {weight.dtype} "
                f"weight. This stage emits f16 IR only; integer weights need the "
                f"quantized path, which is refused by name in quantize_() and "
                f"described in docs/devices/INTELNPU.md section 1.4."
            )
        self.out_features, self.in_features = int(weight.shape[0]), int(weight.shape[1])
        # Raises here, at construction, for an oversized layer -- before the model
        # is handed back looking offloaded.
        linear_ir(self.in_features, self.out_features, 1, bias is not None)
        self.weight = torch.nn.Parameter(weight.detach().to(torch.float16))
        self.bias = (
            torch.nn.Parameter(bias.detach().to(torch.float16)) if bias is not None else None
        )
        self.device_name = device
        self.library = library
        self._compiled = {}
        self._ov = None
        self.execution_devices = None

    # -- construction -----------------------------------------------------
    @classmethod
    def from_torch(cls, layer, device: str = "NPU", library: str | None = None):
        """The `lower_linear` substitution (`compiler.py:144-160`), one layer."""
        return cls(layer.weight, getattr(layer, "bias", None), device=device, library=library)

    # -- the device layer -------------------------------------------------
    def _weights_blob(self) -> bytes:
        blob = pack_f16(self.weight.detach().flatten().tolist())
        if self.bias is not None:
            blob += pack_f16(self.bias.detach().flatten().tolist())
        return blob

    def _compile_for(self, batch: int):
        if batch in self._compiled:
            return self._compiled[batch]
        if self._ov is None:
            self._ov = OpenVINO(self.library)
            found = self._ov.devices()
            if self.device_name not in found:
                raise IntelNPUUnavailable(
                    f"torchnative intelnpu: OpenVINO loaded but does not list "
                    f"{self.device_name!r} among its devices {list(found)!r}. Either "
                    f"the machine has no Intel NPU, or the NPU driver / OpenVINO NPU "
                    f"plugin is not installed. Refusing here rather than compiling "
                    f"anyway -- intel_npu_acceleration_library only warns at this "
                    f"point (backend/utils.py:56-60) and then silently uses the CPU."
                )
        xml = linear_ir(self.in_features, self.out_features, batch, self.bias is not None)
        compiled = self._ov.compile_ir(xml, self.device_name, self._weights_blob())
        devices = self._ov.execution_devices(compiled)
        # The assertion, before any number comes back.
        verdict_execution_devices(devices, self.device_name)
        self.execution_devices = list(devices)
        self._compiled[batch] = compiled
        return compiled

    def forward(self, x):
        torch = _torch()
        shape = tuple(int(d) for d in x.shape)
        if shape[-1] != self.in_features:
            raise IntelNPUUnsupported(
                f"torchnative intelnpu: input last dimension {shape[-1]} does not "
                f"match in_features={self.in_features}."
            )
        batch = 1
        for dim in shape[:-1]:
            batch *= dim
        compiled = self._compile_for(batch)
        flat = x.detach().to(torch.float16).flatten().tolist()
        out = unpack_f16(self._ov.infer(compiled, pack_f16(flat)))
        expected = batch * self.out_features
        if len(out) != expected:
            raise IntelNPUExecutionError(
                f"torchnative intelnpu: expected {expected} output elements for a "
                f"({batch}, {self.out_features}) result, got {len(out)}."
            )
        result = torch.tensor(out, dtype=torch.float32).reshape(*shape[:-1], self.out_features)
        return result.to(x.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, device={self.device_name!r}, "
            f"execution_devices={self.execution_devices}"
        )


def compile_model(model, device: str = "NPU", library: str | None = None):
    """Swap every `torch.nn.Linear` in `model` for an `NPULinear`. In place.

    This is `intel_npu_acceleration_library.compile` minus everything that is not
    the mechanism: no `torch.compile`, no dynamo, no tracing, no fx, no
    neural-compressor. Just `named_children()` + `add_module()`, which is all
    `compiler.py:103-141` ever did.

    Returns `(model, report)`. **The report is not decoration.** It names every
    leaf module type left behind, because "the model is on the NPU" is false for
    any model with a `LayerNorm` in it, and the archived library does not say so:
    `lower_linear` returns `None` for anything it does not recognise
    (`compiler.py:173`) and the caller gets a model it believes is offloaded.
    docs/graph/NPU2.md is a whole document about a partial offload that went unnoticed
    because the answers were right.

    Raises:
        IntelNPUUnsupported: if `device` is not NPU or CPU, or if the model
            contains no `torch.nn.Linear` at all -- returning an untouched model
            and calling it compiled is the silent fallback wearing a bow tie.
    """
    torch = _torch()
    if device not in ("NPU", "CPU"):
        raise IntelNPUUnsupported(
            f"torchnative intelnpu: device {device!r} is not supported. This module "
            f"targets 'NPU'; 'CPU' exists only as the negative control that proves "
            f"the EXECUTION_DEVICES reading moves when the request moves. OpenVINO's "
            f"'AUTO', 'HETERO' and 'MULTI' are deliberately excluded -- each of them "
            f"may place part of the graph elsewhere, which is precisely the outcome "
            f"verdict_execution_devices() refuses."
        )
    swapped, left = [], {}

    def walk(parent, prefix):
        for name, child in list(parent.named_children()):
            path = f"{prefix}{name}"
            if isinstance(child, torch.nn.Linear):
                parent.add_module(name, NPULinear.from_torch(child, device, library))
                swapped.append(path)
                continue
            grandchildren = list(child.named_children())
            if not grandchildren:
                left[type(child).__name__] = left.get(type(child).__name__, 0) + 1
            else:
                walk(child, f"{path}.")

    walk(model, "")
    if not swapped:
        raise IntelNPUUnsupported(
            f"torchnative intelnpu: this model has no torch.nn.Linear, so nothing was "
            f"lowered and nothing runs on {device}. Leaf module types found: "
            f"{sorted(left) or ['<none>']}. Returning the model unchanged with a "
            f"success message would be the silent CPU fallback this module exists to "
            f"prevent -- see docs/devices/INTELNPU.md section 3.1. Linear is the only leaf "
            f"lowered at this stage; intel_npu_acceleration_library's own lowering "
            f"starts at the same place (compiler.py:144-160)."
        )

    # Compile the first swapped layer here, eagerly, rather than at first
    # forward. It is what makes `device="NPU"` on a machine with no NPU a
    # failure of *this call* instead of a model that looks offloaded and only
    # discloses otherwise several layers into a generate() loop -- by which
    # point the answers are correct and nothing draws attention. The
    # EXECUTION_DEVICES assertion happens inside `_compile_for`, so the report
    # below carries OpenVINO's own answer rather than our intention.
    first = model
    for part in swapped[0].split("."):
        first = getattr(first, part) if not part.isdigit() else first[int(part)]
    first._compile_for(1)

    return model, {
        "device": device,
        "swapped": swapped,
        "left_on_cpu": dict(sorted(left.items())),
        "fully_offloaded": not left,
        "execution_devices": list(first.execution_devices),
    }


# --------------------------------------------------------------------------
# Refused by name. Each of these has a test asserting the refusal.
# --------------------------------------------------------------------------


#: The module types `compile_model` lowers, and nothing else. One entry.
#:
#: Deliberately not the same question as "what does OpenVINO's NPU plugin
#: accept" -- `coreml.py` calls that distinction out by name, and conflating the
#: two would report coverage this module does not have. OpenVINO's opset is
#: enormous; the archived library's own reflected op table is 61 entries
#: (`backend/ops.py`, `get_supported_ops()`); what *this* file can emit is
#: `MatMul` and `Add`, arranged as one Linear.
SUPPORTED_MODULES = frozenset({"torch.nn.Linear"})


def supported_ops() -> frozenset:
    """OpenVINO ops this module can emit: `MatMul` and `Add`, as one Linear.

    There is no captured-graph lowering table here and `compile_module` says so.
    This module reaches the device by *module replacement*, the mechanism
    docs/devices/INTELNPU.md section 1.2 found underneath `NPUModelForCausalLM`, not by
    serialising a `decompose` -> `refold` trace. The two are different doors and
    this function answers for the one that is open.
    """
    return frozenset({"MatMul", "Add"})


def compile_module(module=None, example_inputs=None, **kwargs):
    """Refused: **captured-graph** lowering is not implemented.

    Not to be confused with `compile_model`, which is implemented and is a
    different mechanism. The distinction is the one docs/devices/INTELNPU.md section 1.2
    turns on:

    * `compile_model` replaces `torch.nn.Linear` leaves with `NPULinear`. No
      capture, no trace, no graph -- and it is what the archived library does.
    * `compile_module` would take a `decompose` -> `refold` trace and serialise
      the whole graph to OpenVINO IR, the way `coreml.py` and `nnapi.py` do. That
      reaches fusions and inter-module structure that leaf replacement cannot,
      and it is not written.

    Refusing rather than quietly falling back to `compile_model` matters: the
    two have different coverage, and a caller who asked for the whole graph and
    silently got the Linears would be told a model was offloaded that mostly is
    not. That is the failure docs/graph/NPU2.md records.
    """
    raise IntelNPUUnsupported(
        "torchnative intelnpu: compile_module is not implemented -- there is no "
        "captured-graph lowering here. It would serialise a decompose->refold trace "
        "to OpenVINO IR in memory and compile it through the path this module "
        "already opens, the way coreml.py and nnapi.py do for their targets. What "
        "*is* implemented is compile_model(), which replaces torch.nn.Linear leaves "
        "with NPULinear -- the mechanism docs/devices/INTELNPU.md section 1.2 found "
        "underneath NPUModelForCausalLM. Use that, and read its report: it names "
        "every leaf left on the CPU, which whole-graph lowering would not have to. "
        "This refuses rather than silently redirecting to compile_model, because the "
        "two cover different amounts of the model and docs/graph/NPU2.md is about being "
        "told a model was offloaded when part of it was not."
    )


def quantize_(model=None, format=None, **kwargs):
    """Refused: use `torchnative.quant.quantize_`, not neural-compressor."""
    raise IntelNPUUnsupported(
        "torchnative intelnpu: no quantizer here. intel_npu_acceleration_library "
        "routes int4/int8 through Intel neural-compressor "
        "(quantization.py:90-109, PostTrainingQuantConfig(approach='weight_only', "
        "algorithm='RTN')), which reaches deep into PyTorch internals and is not "
        "hostable on torchnative's shim. torchnative already has the same shape at "
        "torchnative.quant.quantize_(model, format='q8_0') -- see docs/graph/QUANT2.md "
        "section 3, which cites that library's module-replacement approach as the "
        "precedent. Note that the archived library also carries a dependency-free "
        "per-row symmetric quantizer (quantization.py:15-64) that needs no "
        "neural-compressor; docs/devices/INTELNPU.md section 1.4 has the details."
    )


def dynamo_backend(*args, **kwargs):
    """Refused permanently: torch.compile does not exist here."""
    raise IntelNPUUnsupported(
        "torchnative intelnpu: there is no torch.compile backend and there will not "
        "be one. Dynamo needs CPython's PEP 523 frame-evaluation hook "
        "(_PyInterpreterState_SetEvalFrameFunc plus the _PyInterpreterFrame layout), "
        "neither of which is reachable from an abi3 extension -- see docs/graph/COMPILE.md. "
        "This costs nothing here: docs/devices/INTELNPU.md section 1.2 establishes that "
        "intel_npu_acceleration_library's own NPU path does not use torch.compile "
        "either. Its compile() (compiler.py:42-81) is plain nn.Module subtree "
        "replacement; the @register_backend npu at compiler.py:270 is a separate, "
        "optional entry point that NPUModelForCausalLM never touches."
    )


if __name__ == "__main__":  # pragma: no cover - this is the Windows entry point
    import json

    device = sys.argv[1] if len(sys.argv) > 1 else "NPU"
    try:
        report = probe(device=device)
    except (IntelNPUUnavailable, IntelNPUExecutionError, IntelNPUUnsupported) as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print(json.dumps(report, indent=2))
    if report.get("verdict") != device.lower():
        print(
            f"\nNOT PROVEN: verdict is {report.get('verdict')!r}, not {device.lower()!r}. "
            f"Missing: {report.get('missing')!r}. See docs/devices/INTELNPU.md section 4."
        )
        raise SystemExit(2)
    print(
        f"\nPROVEN: OpenVINO reports EXECUTION_DEVICES={report['execution_devices']} "
        f"for a model compiled for {device}, the control compiled for "
        f"{'CPU' if device != 'CPU' else 'NPU'} reports "
        f"{report['execution_devices_control']}, and the Linear it ran agrees with "
        f"the reference to {report['agreement']:g} while a different input moves the "
        f"answer by {report['control_diff']:g}."
    )
