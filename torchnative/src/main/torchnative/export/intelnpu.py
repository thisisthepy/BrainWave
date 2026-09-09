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

What it reaches the device *with* --- and this is now a **private** capability,
not an API. `_compile_model(model, device="NPU")` walks a module tree and
replaces every `torch.nn.Linear` with an `_NPULinear` whose forward runs on the
device. It was public as `compile_model` and was withdrawn (see the bottom of
this file); it is kept private because the measurements in this file rest on it. That is not an approximation of what the archived library does
--- it is the same mechanism. `intel_npu_acceleration_library.compile`
(`compiler.py:42-81`) does no tracing at all; it is `named_children()` +
`add_module()` (`compiler.py:103-141`) swapping `torch.nn.Linear` for a leaf that
dispatches over the FFI (`compiler.py:144-160`, `nn/linear.py:35-51`). That single
substitution is the whole of what makes `NPUModelForCausalLM.from_pretrained(...)`
followed by `model.generate(...)` run on an NPU: `generate()` never learns
anything about the hardware.

What this module does NOT do, and says so by name rather than pretending: it
does not lower a *captured graph* (there is no `decompose` -> `refold` trace
serialised to OpenVINO IR here, the way `coreml.py` and `nnapi.py` do), it does
not quantize, and it offers no `torch.compile` backend. Nor does it offer a way
to run your model: every such entry point was withdrawn, and each refuses by
name at the bottom of this file. The replacement is
`torchnative.transformers.AutoModelForCausalLM`, which now **exists** --- though
the recompile behind `model.to(torchnative.device.npu)` does not, and refuses by
name after resolving the NPU. For int4 on an Intel NPU today the answer is still
`optimum-intel`.

Where the claims in this file were measured. The IR, the C bindings, the weights
blob, the inference and the numerics were all exercised against a real OpenVINO
2026.3.1 --- on an arm64 Mac, for device `"CPU"`. What an Intel NPU machine adds
is the string `"NPU"` coming back out of `EXECUTION_DEVICES`. docs/devices/INTELNPU.md
section 3.3 draws that line precisely and section 4 is the Windows procedure.
"""

from __future__ import annotations

import ctypes
import glob as _glob
import importlib.util
import struct
import os
import sys
import warnings

from .. import _cachedir

__all__ = [
    "IntelNPUUnavailable",
    "IntelNPUExecutionError",
    "IntelNPUUnsupported",
    "IntelNPUWithdrawn",
    "IntelNPUCacheWarning",
    "OV_STATUS",
    "EXECUTION_DEVICES",
    "MAX_DIM",
    "OPENVINO_CACHE_ENV",
    "CACHE_DIR_PROPERTY",
    "openvino_cache_dir",
    "ensure_cache_dir",
    "plan_lowering",
    "openvino_package_libs_dir",
    "library_candidates",
    "parse_execution_devices",
    "verdict_execution_devices",
    "minimal_ir",
    "linear_ir",
    "pack_f16",
    "unpack_f16",
    "f16_bytes",
    "f16_tensor",
    "load_openvino_c",
    "OpenVINO",
    "available_devices",
    "npu_available",
    "assert_execution_device",
    "probe",
    "supported_ops",
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


class IntelNPUWithdrawn(IntelNPUUnsupported):
    """A user-facing name this module used to export and no longer does.

    A subclass of `IntelNPUUnsupported` so that anything already catching this
    module's refusals keeps catching it. A withdrawal *is* a refusal: it names
    itself, gives the reason, and names what to use instead (CLAUDE.md §6).
    """

class IntelNPUCacheWarning(UserWarning):
    """The compiled-model cache could not be used. Compilation still happened.

    A warning and not an exception, because a cache that cannot be written is a
    slower run and not a wrong one -- and refusing to compile because a
    directory is read-only would be worse than the problem. But it is a warning
    and not silence: `docs/graph/NPU2.md`'s position, arrived at the hard way on
    CoreML, is that the silently degraded path is the defect. A user who thinks
    compilation is cached and is paying for it on every process start should be
    told once.

    Once. `_NPULinear` compiles per leaf per shape -- 252 leaves x 2 shapes on a
    Qwen3-4B -- so a warning per compile would be 504 identical lines. The
    directory is resolved once per `OpenVINO`, and `_CACHE_ANNOUNCED` holds the
    number down even across several of those.
    """



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


#: Backend-specific cache override, and the way to switch this backend's cache
#: off on its own. Wins over `torchnative._cachedir.CACHE_ROOT_ENV`; used
#: verbatim as the directory. Spelled like `TORCHNATIVE_OPENVINO_C` above, which
#: is this module's naming precedent: `TORCHNATIVE_` + the thing + what it is.
#: `TORCHNATIVE_OPENVINO_CACHE_DIR=0` (or off/no/none/false/empty) disables it.
OPENVINO_CACHE_ENV = "TORCHNATIVE_OPENVINO_CACHE_DIR"

#: `ov::cache_dir`'s C spelling. Declared as an exported *variable*,
#: `OPENVINO_C_VAR(const char*) ov_property_key_cache_dir;` at
#: openvino/c/ov_property.h:97-98, whose definition is the string below. We read
#: the exported symbol when the loaded runtime surfaces it and fall back to this
#: literal otherwise -- see `_cache_dir_property_key`.
CACHE_DIR_PROPERTY = "CACHE_DIR"

#: Directories already announced as unusable, so the announcement is once per
#: process per directory rather than once per compile.
_CACHE_ANNOUNCED = set()


def _reset_cache_announcements() -> None:
    """Forget what has been announced. For tests, which need to hear it again."""
    _CACHE_ANNOUNCED.clear()


def openvino_cache_dir(
    platform: "str | None" = None,
    env: "dict[str, str] | None" = None,
    android: "bool | None" = None,
) -> "str | None":
    """Where OpenVINO should keep compiled blobs, or `None` for "do not cache".

    Pure, and injectable per platform for the same reason `library_candidates`
    is: the Windows answer has to be checkable from the machine this was written
    on. `torchnative._cachedir` holds the platform table and the reasoning for
    each entry, including why this does not sit under `HF_HOME`.

    Nothing is created here. `ensure_cache_dir` does that, and announces.
    """
    return _cachedir.backend_cache_dir(
        "openvino",
        backend_env=OPENVINO_CACHE_ENV,
        platform=platform,
        env=env,
        android=android,
    )


def ensure_cache_dir(path: "str | None") -> "str | None":
    """Create `path` and confirm it is writable, or degrade to `None` with a warning.

    Degraded, not broken: a read-only or unwritable cache directory must not
    stop a model compiling. `None` comes back and the caller passes no
    properties, which is exactly the call this module made before caching
    existed -- so "off" is the shipped path and not a third one.

    The writability *probe* is a real file, created and removed, not
    `os.access`. `os.access` answers with the real uid's permission bits and
    gets network filesystems, read-only mounts, ACLs, full disks and container
    overlays wrong in both directions; the failure it misses here would surface
    later as an OpenVINO error from inside the plugin, which is the worst place
    for it.
    """
    if path is None:
        return None
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".torchnative-write-probe")
        with open(probe, "wb") as handle:
            handle.write(b"")
        os.remove(probe)
        return path
    except OSError as exc:
        if path not in _CACHE_ANNOUNCED:
            _CACHE_ANNOUNCED.add(path)
            warnings.warn(
                f"torchnative intelnpu: could not use {path!r} as OpenVINO's "
                f"compiled-model cache ({type(exc).__name__}: {exc}). Compiling "
                f"anyway, WITHOUT a cache -- every model will be recompiled on "
                f"every process start, and a Qwen3-4B is 252 leaves x 2 shapes "
                f"of driver compilation. Point {OPENVINO_CACHE_ENV} at a writable "
                f"directory, or set it to 0 to turn caching off on purpose and "
                f"silence this.",
                IntelNPUCacheWarning,
                stacklevel=2,
            )
        return None


def _cache_dir_property_key(lib) -> bytes:
    """The `CACHE_DIR` key, preferring the one the loaded runtime exports.

    `ov_property_key_cache_dir` is a `const char*` **data** export, so it is read
    with `ctypes.c_char_p.in_dll` rather than called. Reading it means the key
    comes from the same binary that will consume it. The literal fallback covers
    a runtime that does not surface data symbols and an older release that does
    not have this one; both spellings are `"CACHE_DIR"`, which is the only
    reason the fallback is safe to take silently.
    """
    exported = getattr(lib, "_torchnative_exported_cache_key", None)
    if exported:
        return exported
    try:
        value = ctypes.c_char_p.in_dll(lib, "ov_property_key_cache_dir").value
    except (ValueError, AttributeError, TypeError):
        value = None
    return value or CACHE_DIR_PROPERTY.encode()


def openvino_package_libs_dir(search_locations: "list[str] | tuple[str, ...] | None" = None) -> str | None:
    """Return the `libs` directory of an installed `openvino` pip package, or `None`.

    `pip install openvino` ships the *entire* runtime -- `openvino_c` and every
    plugin, including `openvino_intel_npu_plugin` -- inside the package's `libs/`
    directory. This finds that directory without importing `openvino` (importing
    would load its native extension, `_pyopenvino`, for no reason we need here;
    `importlib.util.find_spec` walks the import machinery far enough to get the
    package's `submodule_search_locations` without executing `openvino/__init__.py`).

    `search_locations` is the injection point for tests: pass a fake package
    root (or several) directly and the `find_spec` lookup is skipped entirely,
    so this is checkable against a directory tree that mirrors the real wheel
    layout without an actual OpenVINO install.

    Returns `None` if no `openvino` package is found, or if it has no `libs`
    directory (e.g. an sdist/editable install, or a future layout change) --
    callers fall back to bare filenames and the OS loader path in that case.
    """
    if search_locations is None:
        try:
            spec = importlib.util.find_spec("openvino")
        except (ImportError, ValueError):
            spec = None
        search_locations = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    for location in search_locations:
        candidate = os.path.join(location, "libs")
        if os.path.isdir(candidate):
            return candidate
    return None


#: Fallback filenames, tried only when no pip-installed `openvino` package's
#: `libs/` directory could be found -- i.e. for a system-wide OpenVINO install
#: that ctypes must locate through the OS loader path. This list rots (the
#: real Linux wheel ships `libopenvino_c.so.2541`, a version this list never
#: had), which is exactly why `library_candidates` prefers globbing a real
#: directory over trusting this when it can.
_SYSTEM_LIBRARY_NAMES = {
    "win32": ("openvino_c.dll", "openvino_c_d.dll"),
    "linux": ("libopenvino_c.so", "libopenvino_c.so.2025", "libopenvino_c.so.2024"),
}


def library_candidates(platform: str | None = None, libs_dir: str | None = None) -> tuple[str, ...]:
    """Shared-library paths to try for the OpenVINO C API, newest naming first.

    Pure: takes the platform string rather than reading `sys.platform`, so the
    Windows and Linux answers are both checkable from a Mac. `libs_dir`, if
    given, is globbed for the real filename OpenVINO shipped there instead of
    guessing -- the pip wheel's Linux `.so` carries a build-number suffix
    (`libopenvino_c.so.2541`) that no hardcoded list keeps up with. When
    `libs_dir` is `None` or the glob finds nothing, this falls back to
    `_SYSTEM_LIBRARY_NAMES`, which is what a system-wide (non-pip) install
    needs, since there is no directory to glob in that case.

    Raises:
        IntelNPUUnavailable: on any platform where the OpenVINO NPU plugin does
            not exist. The plugin ships for Windows and Linux on x86-64 only; on
            macOS there is no Intel NPU to reach and no plugin to reach it with.
    """
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        key = "win32"
        globs = ("openvino_c.dll", "openvino_c_d.dll")
    elif platform.startswith("linux"):
        key = "linux"
        globs = ("libopenvino_c.so*",)
    else:
        raise IntelNPUUnavailable(
            f"torchnative intelnpu: platform {platform!r} has no Intel NPU path. The NPU "
            f"is reached through OpenVINO's NPU plugin, which Intel ships for Windows and "
            f"Linux on x86-64 only -- see docs/devices/INTELNPU.md section 1.3. The archived "
            f"intel_npu_acceleration_library draws the same line explicitly at "
            f"backend/bindings.py:56-59, refusing every sys.platform that is not 'win32' or "
            f"'linux'. This is not a missing feature; there is no such hardware here."
        )
    if libs_dir:
        found = []
        for pattern in globs:
            found.extend(sorted(_glob.glob(os.path.join(libs_dir, pattern))))
        if found:
            return tuple(found)
    return _SYSTEM_LIBRARY_NAMES[key]


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


def _f16_count(blob: bytes, where: str) -> int:
    """How many halves are in `blob`, refusing a byte count that is not whole.

    Split out of `unpack_f16` so that the refusal survives the routes that no
    longer build a Python list on the way past it.
    """
    if len(blob) % 2:
        raise IntelNPUExecutionError(
            f"torchnative intelnpu: {where} returned {len(blob)} bytes for an f16 "
            f"tensor, which is not a whole number of 2-byte halves. Refusing to "
            f"reinterpret a buffer whose element type is not what was asked for."
        )
    return len(blob) // 2


def f16_bytes(tensor) -> bytes:
    """A tensor's elements as little-endian f16 bytes, **without Python floats**.

    This is `pack_f16(tensor.flatten().tolist())`'s output and not its method,
    and the difference is the whole point of docs/devices/INTELNPU.md, `The weights path`.
    `tolist()` materialises one `PyFloat` per element -- about 24 bytes of
    object plus an 8-byte list slot. For a Qwen3 `down_proj` weight, 9728 x 2560
    = 24_903_680 elements, that is roughly 800 MB of CPython heap asked for in
    order to produce a 49 MB blob, and a real user's `generate()` died with
    `MemoryError` inside `_weights_blob` doing exactly that. The same round trip
    ran on **every activation of every forward**, which is why the NPU sat idle
    between matmuls.

    The route is `torch._C._shim_f16_bytes`, added for this: it flattens, makes
    contiguous, converts through the same funnel `.to(torch.float16)` uses, and
    returns one `bytes` object. `tensor.rs::shim_f16_bytes` documents why its
    encoding is byte-identical to `pack_f16`'s and where the two could differ.

    The conversion to f16 is spelled here as well as there, deliberately: with
    the tensor already f16 the Rust-side conversion is a no-op, and the one case
    where `struct`'s `<e` and `half::f16` disagree -- a magnitude above f16's
    range, where `struct` raises `OverflowError` and `half` saturates to
    infinity -- becomes unreachable.

    **There is no `tolist` fallback.** Upstream torch is served by `.numpy()`,
    which the shim does not have (`test_the_shim_has_no_numpy_bridge_which_is_
    why_this_packs_bytes` measures that). Anything else refuses by name, because
    a silent fallback to the route this function exists to remove would put the
    `MemoryError` back without anyone noticing.
    """
    torch = _torch()
    half = tensor.detach().to(torch.float16)
    reader = getattr(torch._C, "_shim_f16_bytes", None)
    if reader is not None:
        return reader(half)
    to_numpy = getattr(half, "numpy", None)
    if to_numpy is not None:
        try:
            return to_numpy().tobytes()
        except NotImplementedError:
            pass
    raise IntelNPUUnsupported(
        "torchnative intelnpu: this torch build offers neither "
        "torch._C._shim_f16_bytes nor a working Tensor.numpy(), so there is no "
        "way to reach a tensor's bytes without building one Python float per "
        "element. Refusing rather than falling back to .tolist(): that fallback "
        "is what raised MemoryError on a 24.9-million-element weight "
        "(docs/devices/INTELNPU.md, `The weights path`)."
    )


def f16_tensor(blob: bytes, shape):
    """`unpack_f16`'s inverse destination, reached without a Python list.

    `torch.tensor(unpack_f16(blob))` is the same defect as `f16_bytes` replaces,
    pointing the other way: it builds one `PyFloat` per returned element on
    every forward. `torch.frombuffer` reads the bytes directly
    (`lib.rs::_frombuffer`), so the result never passes through Python scalars.

    `bytearray(blob)` because `frombuffer` wants a writable buffer; that is one
    C-level copy, not `numel` objects.
    """
    torch = _torch()
    count = _f16_count(blob, "OpenVINO")
    expected = 1
    for dim in shape:
        expected *= int(dim)
    if count != expected:
        raise IntelNPUExecutionError(
            f"torchnative intelnpu: expected {expected} output elements for a "
            f"{tuple(int(d) for d in shape)} result, got {count}."
        )
    frombuffer = getattr(torch, "frombuffer", None)
    if frombuffer is None:
        flat = torch.tensor(unpack_f16(blob), dtype=torch.float16)
    else:
        flat = frombuffer(bytearray(blob), dtype=torch.float16)
    return flat.reshape(*[int(d) for d in shape])


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
    libs_dir = None
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
        # An explicit path or LIBRARY_ENV names a specific file and wins outright --
        # a user who names a library gets that one. Otherwise prefer the OpenVINO
        # pip package: `pip install openvino` (or `torchnative[npu]`) ships the
        # entire runtime, including the NPU plugin, in its `libs/` directory, so
        # a user should never have to hunt down a system-wide install or set PATH.
        libs_dir = openvino_package_libs_dir()
        names = list(library_candidates(libs_dir=libs_dir))
    lib = None
    # On Windows, `openvino_c.dll` pulls in sibling DLLs (`openvino.dll`, the
    # plugin DLLs) by bare name at load time. `os.add_dll_directory` is the
    # supported mechanism for making a specific directory resolvable for that --
    # unlike mutating PATH, it is scoped to this process and this call, and it
    # does not exist on non-Windows platforms, hence the guard.
    if libs_dir and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(libs_dir)
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
            f"{tried!r}. Easiest fix: `pip install torchnative[npu]` (or plain "
            f"`pip install openvino`) -- the wheel ships the whole runtime, including "
            f"the NPU plugin, with no system install and no PATH changes required. "
            f"If you have a system-wide OpenVINO install instead, either put its bin "
            f"directory on PATH / LD_LIBRARY_PATH, or set {LIBRARY_ENV} to the full "
            f"path of openvino_c.dll / libopenvino_c.so -- the env var is the escape "
            f"hatch for installs this cannot find on its own. On Windows a system "
            f"install also needs its sibling DLLs resolvable -- see "
            f"docs/devices/INTELNPU.md section 4."
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


#: Distinguishes "the caller said nothing" from "the caller said do not cache".
_UNSET = object()


class OpenVINO:
    """A borrowed `ov_core_t` with the few operations this stage needs.

    Usable as a context manager. Every OpenVINO call goes through `_check`, which
    turns a non-zero `ov_status_e` into an exception carrying the symbolic status
    name and, when the runtime offers it, `ov_get_last_err_msg()` -- so a failure
    on the user's machine reports OpenVINO's account of itself rather than ours.
    """

    def __init__(self, path: str | None = None, cache_dir: "str | None" = _UNSET):
        self._lib = load_openvino_c(path)
        core = ctypes.c_void_p()
        self._check(self._lib.ov_core_create(ctypes.byref(core)), "ov_core_create")
        self._core = core
        # Resolved and created **once, here**, not per compile. That is not a
        # micro-optimisation: `_NPULinear` compiles once per leaf per input
        # shape, which is 252 x 2 = 504 compiles on a Qwen3-4B before the second
        # generated token, and a `makedirs` + write probe on each of those is 504
        # filesystem round trips -- and, when the directory is unusable, 504
        # warnings. Doing it at construction makes both numbers 1.
        #
        # `_UNSET` rather than `None` as the default because `None` is a
        # meaningful argument here: it means "do not cache", and a caller must be
        # able to say that without being given the environment's answer instead.
        if cache_dir is _UNSET:
            cache_dir = openvino_cache_dir()
        self.cache_dir = ensure_cache_dir(cache_dir)
        self._cache_key = (
            _cache_dir_property_key(self._lib) if self.cache_dir is not None else None
        )

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
                # The properties, and the count. `ov_core.h:204` defines
                # `property_args_size` as "How many properties args will be
                # passed, each property contains 2 args: key and value" -- it is
                # the ARG count, not the pair count, and the C side rejects an
                # odd one. So one property is 2, and `len(props)` is the count by
                # construction rather than a number written twice.
                #
                # Still the variadic `ov_core_compile_model`, and still not
                # `ov_core_compile_model_props`: that entry point is on OpenVINO
                # master but not in every release a user has installed, and
                # passing properties does not change that. The reasoning in
                # `load_openvino_c` survives this round intact; only the count
                # went from 0 to 2.
                props = ()
                if self.cache_dir is not None:
                    props = (
                        ctypes.c_char_p(self._cache_key),
                        ctypes.c_char_p(self.cache_dir.encode("utf-8")),
                    )
                self._check(
                    self._lib.ov_core_compile_model(
                        self._core,
                        model,
                        device.encode(),
                        ctypes.c_size_t(len(props)),
                        ctypes.byref(compiled),
                        *props,
                    ),
                    f"ov_core_compile_model(device={device})",
                )
                compiled._torchnative_weights = keepalive
                # The lifetime anchor. `close()` calls `ov_core_free`, and a
                # compiled model whose core has been freed is a use-after-free
                # that presents as wrong numbers rather than as a crash -- the
                # same failure shape the weights keepalive above prevents, one
                # level up. A strong reference from the handle to the owning
                # Python object means no reachable compiled model can have a
                # collected core underneath it. It does not (and must not) make
                # an explicit `close()` safe: `probe()` closes deliberately, at
                # the end of a `with` block, after its compiled models are gone.
                # What this rules out is the accidental version -- the shared
                # core `_compile_model` builds having no other name.
                compiled._torchnative_core = self
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


def open_core(library: str | None, device: str) -> "OpenVINO":
    """One `ov::Core`, with the device-presence assertion done once on it.

    **Why this is a function and not two lines inside `_compile_for`.** It used
    to be those two lines, and the cost of that was 252 -- a Qwen3-4B lowers 252
    `nn.Linear` leaves, and each one built its own `ov::Core` on its first
    forward. `ov_core_create` dlopens and initialises OpenVINO's whole plugin
    registry; `ov_core_get_available_devices` then enumerates and initialises
    every plugin it found (the NPU plugin, which talks to the Level Zero driver,
    and the GPU plugin included); and `OpenVINO.__init__` resolves and creates
    the compile-cache directory. All of that ran 252 times for one model.

    It is not only waste. OpenVINO's model-cache serialisation is a per-hash
    mutex held *inside one* `CoreImpl` (`src/inference/src/dev/core_impl.cpp`,
    `m_cache_guard.get_hash_lock(...)`), so 252 separate cores do not share it
    at all -- the one-core-per-leaf shape had opted out of the only concurrency
    protection OpenVINO offers before any thread existed. That is why the shared
    core had to land before the parallelism question could even be asked;
    docs/devices/NPUPAR.md section 1 is the record.

    The core is **returned**, not stored on a module global. A process-wide
    singleton would outlive every model, could never be closed, and would freeze
    the cache-directory decision that docs/devices/NPUCACHE.md deliberately
    leaves per-core and overridable.

    Refusing here, when `device` is not among what OpenVINO reports, rather than
    compiling anyway: `intel_npu_acceleration_library` only warns at this spot
    (`backend/utils.py:56-60`) and then silently uses the CPU, which is
    docs/graph/NPU2.md's failure exactly.
    """
    core = OpenVINO(library)
    found = core.devices()
    if device not in found:
        raise IntelNPUUnavailable(
            f"torchnative intelnpu: OpenVINO loaded but does not list "
            f"{device!r} among its devices {list(found)!r}. Either "
            f"the machine has no Intel NPU, or the NPU driver / OpenVINO NPU "
            f"plugin is not installed. Refusing here rather than compiling "
            f"anyway -- intel_npu_acceleration_library only warns at this "
            f"point (backend/utils.py:56-60) and then silently uses the CPU."
        )
    return core


class _NPULinear:
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
            cls = type("_NPULinear", (_NPULinear, torch.nn.Module), {})
            obj = torch.nn.Module.__new__(cls)
            return obj
        return super().__new__(cls)

    def __init__(self, weight, bias=None, device: str = "NPU", library: str | None = None,
                 core: "OpenVINO | None" = None):
        torch = _torch()
        torch.nn.Module.__init__(self)
        if weight.dim() != 2:
            raise IntelNPUUnsupported(
                f"torchnative intelnpu: _NPULinear needs a 2-D weight, got shape "
                f"{tuple(weight.shape)}. There is no Linear here to lower."
            )
        if not weight.dtype.is_floating_point:
            raise IntelNPUUnsupported(
                f"torchnative intelnpu: _NPULinear will not lower a {weight.dtype} "
                f"weight. This stage emits f16 IR only; integer weights need the "
                f"quantized path, which is not implemented here. Use "
                f"torchnative.quant.quantize_(model, format=...); "
                f"docs/devices/INTELNPU.md section 1.4 has the details."
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
        # The shared `ov::Core`, when there is one. `_compile_model` builds
        # exactly one and hands it to every leaf; a leaf built on its own
        # through `from_torch` gets None and makes its own at first compile, so
        # sharing did not make a core mandatory. See `_ensure_core`.
        self._ov = core
        self.execution_devices = None

    # -- construction -----------------------------------------------------
    @classmethod
    def from_torch(cls, layer, device: str = "NPU", library: str | None = None,
                   core: "OpenVINO | None" = None):
        """The `lower_linear` substitution (`compiler.py:144-160`), one layer.

        `core` is the shared `ov::Core` when the caller has one. `_compile_model`
        always does; a caller lowering a single layer need not, and passing None
        keeps the old behaviour exactly -- a core built lazily, by this leaf, at
        its first compile.
        """
        return cls(layer.weight, getattr(layer, "bias", None), device=device,
                   library=library, core=core)

    # -- the device layer -------------------------------------------------
    def _weights_blob(self) -> bytes:
        """The Constant payload OpenVINO loads: weight then bias, f16, flat.

        Through `f16_bytes`, not `pack_f16(...tolist())`. The old spelling asked
        CPython for ~800 MB of `PyFloat` objects for one Qwen3 `down_proj` and
        raised `MemoryError` before OpenVINO saw a byte
        (docs/devices/INTELNPU.md, `The weights path`).
        """
        blob = f16_bytes(self.weight)
        if self.bias is not None:
            blob += f16_bytes(self.bias)
        return blob

    def _compile_for(self, batch: int):
        if batch in self._compiled:
            return self._compiled[batch]
        if self._ov is None:
            self._ov = open_core(self.library, self.device_name)
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
        # Bytes in, bytes out. Neither direction builds Python scalars: the old
        # spelling did `tolist()` on the way in and `torch.tensor(unpack_f16(...))`
        # on the way back, on *every* call, which is the reason the device was
        # idle between matmuls (docs/devices/INTELNPU.md, `The weights path`).
        blob = self._ov.infer(compiled, f16_bytes(x))
        result = f16_tensor(blob, (batch, self.out_features))
        result = result.to(torch.float32).reshape(*shape[:-1], self.out_features)
        return result.to(x.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, device={self.device_name!r}, "
            f"execution_devices={self.execution_devices}"
        )


def plan_lowering(model, predicate=None):
    """What `_compile_model` would lower and what it would leave, without OpenVINO.

    The lowering itself needs an OpenVINO runtime and, for a real claim, an
    Intel NPU. **The selection does not**, and the selection is where the
    granularity defect lived: a Qwen3-4B was refused whole because one leaf was
    oversized. This answers "what would happen" on any machine, which is what
    makes that defect testable on a host with neither.

    It is **not** evidence that anything ran on an NPU, and nothing here should
    be read as saying so --- `probe()` and `assert_execution_device()` are the
    functions that answer that question. This answers a different one: which
    leaves are eligible, which are not, and how much of the model that is.

    **The eligibility check is not a second copy.** It calls `linear_ir`, the
    same pure function `_NPULinear.__init__` calls to decide, so the plan and
    the real lowering cannot drift apart --- a test asserts they agree.

    `predicate(name, module) -> bool` narrows the selection, matching
    `torchnative.quant.quantize_` and `_compile_model`.

    Returns a dict with `eligible`, `skipped` (`(name, reason)`),
    `left_on_cpu`, `fully_offloaded`, `parameters_moved`, `parameters_total`
    and `fraction_moved`.
    """
    torch = _torch()
    eligible, skipped, left = [], [], {}
    moved = 0

    def walk(parent, prefix):
        nonlocal moved
        for name, child in list(parent.named_children()):
            path = f"{prefix}{name}"
            if isinstance(child, torch.nn.Linear):
                if predicate is not None and not predicate(path, child):
                    skipped.append((path, "excluded by predicate"))
                    continue
                try:
                    linear_ir(
                        int(child.in_features), int(child.out_features), 1,
                        getattr(child, "bias", None) is not None,
                    )
                except IntelNPUUnsupported as exc:
                    skipped.append((
                        path,
                        f"Linear(out_features={child.out_features}, "
                        f"in_features={child.in_features}) stays on the CPU: "
                        f"{str(exc).split(chr(10))[0]}",
                    ))
                    continue
                eligible.append(path)
                moved += child.weight.numel() + (
                    child.bias.numel() if getattr(child, "bias", None) is not None else 0
                )
                continue
            grandchildren = list(child.named_children())
            if not grandchildren:
                left[type(child).__name__] = left.get(type(child).__name__, 0) + 1
            else:
                walk(child, f"{path}.")

    walk(model, "")
    total = sum(p.numel() for p in model.parameters())
    return {
        "eligible": eligible,
        "skipped": skipped,
        "left_on_cpu": dict(sorted(left.items())),
        "fully_offloaded": not left and not skipped,
        "parameters_moved": moved,
        "parameters_total": total,
        "fraction_moved": moved / total if total else 0.0,
    }


def _compile_model(model, device: str = "NPU", library: str | None = None,
                   predicate=None, eager: bool = True, progress=None):
    """Swap every `torch.nn.Linear` in `model` for an `_NPULinear`. In place.

    `predicate(name, module) -> bool` narrows which leaves are lowered; the
    default takes all of them. **This is deliberately the same signature as
    `torchnative.quant.quantize_`**, and for the same reason that function
    gives: `lm_head` is both the largest single weight in a small model and the
    layer whose error lands directly on the logits with nothing after it to
    attenuate. Both facts are real and pull opposite ways, so the choice is the
    caller's. One idea, one spelling -- a second one here would be a second
    thing to learn for no gain.

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

    **An oversized leaf is left behind and named, not fatal.** A `Linear` whose
    dimensions exceed `MAX_DIM` used to raise out of the walk and take the whole
    model with it, which made every real LLM unreachable: `lm_head` in
    Qwen3-4B-Instruct-2507 is 151936 x 2560 and 151936 > MAX_DIM (131072), so a
    36-layer model was refused for one layer. Every model with a large
    vocabulary has that layer and it is usually the single largest weight.

    The archived Intel library draws the same limit (`nn/linear.py:66`) and
    responds by **silently returning the torch layer unchanged**, which leaves
    an unannounced CPU layer inside a model the caller believes is on the NPU.
    That is precisely docs/graph/NPU2.md's failure. So this takes the same
    outcome and the opposite epistemics: the layer stays on the CPU and the
    report says so **by name, with its shape and the limit it exceeded**, and
    `fully_offloaded` goes False.

    That last part is the load-bearing one. A caller who ignores the report
    must not be able to conclude the model is fully offloaded, so `report`
    carries `fraction_moved` -- a **value**, parameters lowered over parameters
    total -- rather than only prose. For Qwen3-4B, dropping `lm_head` alone is
    about 10% of the parameters, and a number says that where a list of names
    does not.

    **One `ov::Core` for the whole model.** `_NPULinear._compile_for` used to
    build its own on first forward, so a 252-leaf Qwen3-4B constructed 252 of
    them -- 252 dlopens of OpenVINO's plugin set, 252 device enumerations, 252
    cache-directory resolutions, and 252 model caches that could not share
    OpenVINO's per-hash write guard with each other. `open_core` is built once
    here and handed to every leaf.

    **`eager=True` compiles the batch=1 decode shape for every leaf before
    returning**, and `progress(done, total, name)` reports it. That is not a
    speed-up -- the same compiles happen either way -- it is a change of
    *placement*: `generate()` uses the prompt-length shape once and then batch=1
    per token, so lazily those 252 compiles land inside the second generated
    token and look like a hang. A leaf that will not compile eagerly is reported
    in `eager_failed` by name and keeps its lazy path; only the first leaf's
    failure is fatal, because that one is the assertion that the device exists.

    **These compiles are serial and stay serial.** docs/devices/NPUPAR.md is the
    record of why: two of the four things that would have to hold before running
    them concurrently are UNVERIFIED, and one of the two fails as a corrupted
    cache entry rather than as slowness.

    Raises:
        IntelNPUUnsupported: if `device` is not NPU or CPU, or if no
            `torch.nn.Linear` was lowered at all -- returning an untouched model
            and calling it compiled is the silent fallback wearing a bow tie.
        IntelNPUUnavailable: if OpenVINO does not list `device`, or if the first
            lowered leaf will not compile.
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
    swapped, left, skipped = [], {}, []
    moved_parameters = 0
    # None during the walk, deliberately. The walk needs no OpenVINO -- it is
    # `named_children()` plus the pure `linear_ir` eligibility check -- and
    # building the core first would move the "OpenVINO is not installed here"
    # failure ahead of the "nothing was lowered" refusal below, reordering two
    # errors that say different things. The core is built once, after the walk,
    # and handed to every leaf then.
    core = None

    def walk(parent, prefix):
        nonlocal moved_parameters
        for name, child in list(parent.named_children()):
            path = f"{prefix}{name}"
            if isinstance(child, torch.nn.Linear):
                if predicate is not None and not predicate(path, child):
                    skipped.append((path, "excluded by predicate"))
                    continue
                # Counted before the swap: after `add_module` the original
                # tensors are no longer reachable through `parent`.
                numel = child.weight.numel() + (
                    child.bias.numel() if child.bias is not None else 0
                )
                try:
                    lowered = _NPULinear.from_torch(child, device, library, core=core)
                except IntelNPUUnsupported as exc:
                    # Left behind and NAMED. Not fatal: one oversized leaf must
                    # not make the whole model unreachable. See this function's
                    # docstring for why, and what the archived library does
                    # instead.
                    skipped.append((
                        path,
                        f"{type(child).__name__}"
                        f"(out_features={child.out_features}, "
                        f"in_features={child.in_features}) stays on the CPU: "
                        f"{str(exc).split(chr(10))[0]}",
                    ))
                    continue
                parent.add_module(name, lowered)
                swapped.append(path)
                moved_parameters += numel
                continue
            grandchildren = list(child.named_children())
            if not grandchildren:
                left[type(child).__name__] = left.get(type(child).__name__, 0) + 1
            else:
                walk(child, f"{path}.")

    walk(model, "")
    if not swapped:
        raise IntelNPUUnsupported(
            f"torchnative intelnpu: nothing was lowered, so nothing runs on "
            f"{device}. Leaf module types found: {sorted(left) or ['<none>']}. "
            f"{len(skipped)} Linear(s) were skipped: {skipped[:4]}. If that list "
            f"is non-empty the model does have Linears and your predicate "
            f"excluded all of them, or every one exceeded MAX_DIM={MAX_DIM}. "
            f"Returning the model unchanged with a "
            f"success message would be the silent CPU fallback this module exists to "
            f"prevent -- see docs/devices/INTELNPU.md section 3.1. Linear is the only leaf "
            f"lowered at this stage; intel_npu_acceleration_library's own lowering "
            f"starts at the same place (compiler.py:144-160)."
        )

    # ONE ov::Core for the whole model, built here and shared by every leaf.
    # Not 252 of them, which is what one-per-leaf meant for a Qwen3-4B; see
    # `open_core` for what each of those 252 was actually doing. The
    # device-presence assertion runs once, on this core, rather than once per
    # leaf.
    #
    # LIFETIME. Nothing else holds this name after the function returns, and
    # that is fine: every leaf holds a strong reference in `_ov`, and every
    # compiled model holds one through `_torchnative_core`, so the core is
    # reachable exactly as long as anything made from it is. `close()` is never
    # called here -- a shared core has no single owner who could know when.
    core = open_core(library, device)

    def _leaf(path):
        node = model
        for part in path.split("."):
            node = getattr(node, part) if not part.isdigit() else node[int(part)]
        return node

    leaves = [_leaf(path) for path in swapped]
    for leaf in leaves:
        leaf._ov = core

    # Compile the first swapped layer here, eagerly, rather than at first
    # forward. It is what makes `device="NPU"` on a machine with no NPU a
    # failure of *this call* instead of a model that looks offloaded and only
    # discloses otherwise several layers into a generate() loop -- by which
    # point the answers are correct and nothing draws attention. The
    # EXECUTION_DEVICES assertion happens inside `_compile_for`, so the report
    # below carries OpenVINO's own answer rather than our intention.
    #
    # **This one is not caught by name**, unlike every compile after it. It is
    # different in kind: it is the assertion that the device is real. Absorbing
    # it into a report would turn "there is no NPU on this machine" into 252
    # named warnings attached to a model the caller believes is offloaded --
    # the silent CPU fallback wearing a report.
    first = leaves[0]
    first._compile_for(1)

    # The rest of the decode-shape compiles, up front.
    #
    # The complaint this answers is not that compilation is slow in total; it is
    # that it happens *inside* `generate()`. `generate()` uses two shapes -- the
    # prompt length once, then batch=1 per token with a KV cache -- so with lazy
    # compilation the batch=1 IR for all 252 leaves is compiled during the
    # SECOND generated token, one leaf at a time, and the model appears to hang
    # after producing one word. Compiling batch=1 here moves that cost to the
    # moment the caller asked for it, which is also the only moment at which it
    # can be reported: `progress(done, total, name)`.
    #
    # batch=1 only. The prompt-length shape is not knowable until there is a
    # prompt, and guessing one would compile an IR nothing uses.
    #
    # `eager=False` keeps the old lazy behaviour, for a caller who wants to
    # lower and inspect a model without paying minutes of compile.
    #
    # Serially. Whether these could run concurrently is a separate question with
    # four parts, and docs/devices/NPUPAR.md answers two of them UNVERIFIED --
    # so no thread pool ships. `test_ovpar.py` holds that as a standing check.
    eager_failed = []
    eager_compiled = 1
    if eager:
        total = len(leaves)
        if progress is not None:
            progress(1, total, swapped[0])
        for index, (path, leaf) in enumerate(zip(swapped[1:], leaves[1:]), start=2):
            try:
                leaf._compile_for(1)
                eager_compiled += 1
            except Exception as exc:  # noqa: BLE001
                # Named, not raised and not swallowed. Raising would mean one
                # leaf OpenVINO happens to refuse makes a model that would
                # otherwise run unreachable -- eager compilation turning a
                # working lazy path into a hard failure, which is the one thing
                # it must not do. Swallowing would mean the caller is told the
                # model is fully offloaded when it is not. The leaf keeps its
                # lazy path, so it still compiles at first forward if the
                # refusal was transient.
                eager_failed.append((path, f"{type(exc).__name__}: {exc}"))
            if progress is not None:
                progress(index, total, path)

    total_parameters = sum(p.numel() for p in model.parameters())
    return model, {
        "device": device,
        "swapped": swapped,
        "left_on_cpu": dict(sorted(left.items())),
        # `(name, reason)`, the same shape `torchnative.quant.quantize_`'s
        # report uses. Predicate exclusions and oversized leaves both land
        # here, and each reason distinguishes which it was.
        "skipped": skipped,
        # False if ANYTHING stayed behind -- a non-Linear leaf, a predicate
        # exclusion, an oversized Linear, or a leaf whose eager compile failed.
        # A caller who reads only this flag must not be told a partially
        # offloaded model is complete.
        "fully_offloaded": not left and not skipped and not eager_failed,
        # "How much actually moved", as a value rather than prose, so that a
        # caller who skims the report still cannot mistake a 90% offload for a
        # whole one.
        "parameters_moved": moved_parameters,
        "parameters_total": total_parameters,
        "fraction_moved": (
            moved_parameters / total_parameters if total_parameters else 0.0
        ),
        "execution_devices": list(first.execution_devices),
        # How many leaves have their batch=1 decode shape already compiled when
        # this returns. 1 with `eager=False` -- the device assertion -- and
        # `len(swapped) - len(eager_failed)` with it on.
        "eager_compiled": eager_compiled,
        # `(name, reason)`, the same shape as `skipped`, and for the same
        # reason: a leaf that did not compile has to be nameable. These are NOT
        # `skipped` entries -- they were lowered, they are on the NPU path, and
        # they will try again at first forward. An oversized leaf never reaches
        # here; it was refused at `from_torch` and is in `skipped` with its
        # shape and MAX_DIM (docs/graph/NPU2.md).
        "eager_failed": eager_failed,
    }


# --------------------------------------------------------------------------
# What this file can answer for. Each of these has a test.
# --------------------------------------------------------------------------


#: The module types the (now private) `_compile_model` lowers, and nothing else.
#: One entry. Kept public because it answers a question about this file's reach
#: rather than offering a way to run anything --- the same reason `supported_ops`
#: is kept.
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

    There is no captured-graph lowering table here, and the withdrawn
    `compile_module` said so by name. This module reaches the device by *module
    replacement*, the mechanism docs/devices/INTELNPU.md section 1.2 found
    underneath `NPUModelForCausalLM`, not by serialising a `decompose` ->
    `refold` trace. Both doors are now shut to callers --- one because it was
    never written, the other because it was withdrawn --- and this function
    still answers honestly for what the emitters in this file can produce.
    """
    return frozenset({"MatMul", "Add"})


# --------------------------------------------------------------------------
# Withdrawn user-facing entry points.
#
# Everything above this line is kept: the device-reading and verdict logic, the
# IR emitters, the C bindings, `probe`, and every refusal. That machinery is how
# this project tells "it ran on the NPU" from "the answer happened to be right",
# and it is reusable for CoreML and QNN.
#
# What was withdrawn is the part that presented itself as "call this to run your
# model". It was wrong in two ways at once. Its *name* copied the archived
# `intel_npu_acceleration_library`, where the ecosystem convention names the
# project (`OVModelForCausalLM`, `ORTModelForCausalLM`, `IPEXModelForCausalLM`),
# and its *shape* -- `compile_model(model, device="NPU") -> (model, report)` --
# was a second in-place module-replacement API beside this repository's own
# `torchnative.quant.quantize_(model, format=...)`, which already had torchao's
# spelling for the same move.
#
# The implementations survive privately as `_compile_model` and `_NPULinear`.
# They are not deleted because the measured claims in docs/devices/INTELNPU.md
# rest on them and the OpenVINO-gated tests still exercise them; they are
# private because they are evidence, not an API.
# --------------------------------------------------------------------------

#: The replacement, named in one place so that one line changes when it lands.
REPLACEMENT = "torchnative.transformers.AutoModelForCausalLM"

_WITHDRAWN = {
    "compile_model": (
        "it was a second in-place module-replacement entry point beside "
        "torchnative.quant.quantize_(model, format=...), which already had "
        "torchao's spelling (`_` for in-place) for the same move -- replacing "
        "leaves. Two shapes for one operation in one codebase. It also returned "
        "`(model, report)` rather than the model, so it was not in-place in the "
        "way its own mechanism was. The lowering it did survives privately as "
        "`_compile_model`, and the report it produced -- which named every leaf "
        "left on the CPU -- is the part worth keeping"
    ),
    "NPULinear": (
        "it is the leaf `compile_model` swapped in, and it reached the device "
        "for `torch.nn.Linear` only, with static shapes and element-at-a-time "
        "FFI. It survives privately as `_NPULinear` because the numbers in "
        "docs/devices/INTELNPU.md were measured through it"
    ),
    "compile_module": (
        "captured-graph lowering was never implemented here -- there is no "
        "decompose->refold trace serialised to OpenVINO IR, the way coreml.py "
        "and nnapi.py do for their targets -- and the leaf-replacement door it "
        "used to point at (compile_model) is itself now withdrawn. It refused "
        "rather than silently redirecting, because the two cover different "
        "amounts of the model and docs/graph/NPU2.md is about being told a "
        "model was offloaded when part of it was not"
    ),
    "quantize_": (
        "there is no quantizer here. intel_npu_acceleration_library routes "
        "int4/int8 through Intel neural-compressor (quantization.py:90-109, "
        "PostTrainingQuantConfig(approach='weight_only', algorithm='RTN')), "
        "which reaches deep into PyTorch internals and is not hostable on "
        "torchnative's shim. This name also collided with the real one: "
        "torchnative.quant.quantize_(model, format='q8_0') is this "
        "repository's one existing user-facing API and is unaffected by this "
        "withdrawal -- use it. See docs/graph/QUANT2.md section 3, which cites "
        "that library's module-replacement approach as the precedent. Note "
        "that the archived library also carries a dependency-free per-row "
        "symmetric quantizer (quantization.py:15-64) that needs no "
        "neural-compressor; docs/devices/INTELNPU.md section 1.4 has the details"
    ),
    "dynamo_backend": (
        "there is no torch.compile backend and there will not be one. Dynamo "
        "needs CPython's PEP 523 frame-evaluation hook "
        "(_PyInterpreterState_SetEvalFrameFunc plus the _PyInterpreterFrame "
        "layout), neither of which is reachable from an abi3 extension -- see "
        "docs/graph/COMPILE.md. This costs nothing: docs/devices/INTELNPU.md "
        "section 1.2 establishes that intel_npu_acceleration_library's own NPU "
        "path does not use torch.compile either. Its compile() "
        "(compiler.py:42-81) is plain nn.Module subtree replacement; the "
        "@register_backend npu at compiler.py:270 is a separate, optional entry "
        "point that NPUModelForCausalLM never touches. This refusal is "
        "permanent and is not lifted by the replacement API"
    ),
}


def _withdrawal_message(name):
    """The refusal text for a withdrawn `name`: what went, why, what instead."""
    return (
        f"torchnative intelnpu: {name} was withdrawn and is not available. It "
        f"was withdrawn because {_WITHDRAWN[name]}.\n"
        f"The replacement is {REPLACEMENT}, and it now EXISTS:\n"
        f"\n"
        f"    import torchnative\n"
        f"    from torchnative.transformers import AutoModelForCausalLM\n"
        f"    model = AutoModelForCausalLM.from_pretrained(model_id)\n"
        f"    model.to(torchnative.device.npu)\n"
        f"\n"
        f"`from_pretrained` returns the real model -- a genuine nn.Module that "
        f"backprops -- not a wrapper. The device is torchnative.device.npu, "
        f"which RESOLVES per host (Intel NPU on Windows) and says which. It is "
        f"NOT torch.device(\"npu\"): that spelling still raises on this shim "
        f"because torch._C._rename_privateuse1_backend is a stub, and it is a "
        f"stub on purpose -- PyTorch has no npu device type and this project "
        f"will not pretend it does (docs/devices/DEVICE_NS.md section 1).\n"
        f"\n"
        f"**Recompiling for the accelerator is still not implemented.** "
        f"model.to(torchnative.device.npu) resolves the NPU, names the unit, "
        f"and then refuses at the compile step rather than handing back an "
        f"unchanged model. `export=` and `load_in_4bit=` refuse by name too. "
        f"So the capability this name reached for is still not here; what "
        f"changed is where the wall is and that it now has a name on it.\n"
        f"\n"
        f"If what you need is int4 on an Intel NPU today, the answer is not in "
        f"this repository -- it is optimum-intel, which HuggingFace and Intel "
        f"maintain and which already ships what this module was reaching for:\n"
        f"\n"
        f"    from optimum.intel import OVModelForCausalLM\n"
        f"    model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_4bit=True).to(\"npu\")\n"
        f"\n"
        f"Its runtime model is an OpenVINO graph rather than a real nn.Module, "
        f"so it cannot backprop. optimum DOES reach mobile -- "
        f"optimum-executorch exports for Android and iOS -- but it exports "
        f"FROM a desktop and the artefact runs in ExecuTorch's C++ runtime, "
        f"with no Python on the device. This project runs Python ON the "
        f"device, which is the difference it exists for, and not NPU "
        f"coverage. docs/devices/INTELNPU.md records the comparison."
    )


def __getattr__(name):
    """Refuse a withdrawn name by name; leave every other miss as an AttributeError."""
    if name in _WITHDRAWN:
        raise IntelNPUWithdrawn(_withdrawal_message(name))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
