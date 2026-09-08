"""Execute a serialised NNAPI blob on a real NNAPI runtime, over `adb`.

docs/graph/NPU2.md is the round this belongs to. `nnapi.py` ends at
`parse_model`: it decodes the blob back through the layout upstream's
serialiser wrote, which proves the *layout* and says nothing about arithmetic.
docs/graph/NPU.md was explicit that this is where the claim stopped -- "structurally
validated", because no NNAPI runtime exists on a Mac.

This module closes that. It ships the blob, the weight buffers and the input
bytes to an Android device, runs `nnapi_runner.c` there -- which replays the
blob into `ANeuralNetworksModel` operand by operand -- and brings the output
bytes back for an element-wise comparison against `DecomposedTrace.replay`.

Three things it deliberately does not do:

* **It does not re-lower.** The runner reads the blob and nothing else. There
  is no second conversion here that could agree with the first by sharing a
  mistake, which is the same reason `verify_shapes` compares against capture
  rather than against a recomputation.
* **It does not install anything.** The binary is pushed to
  `/data/local/tmp/bw_device`, run, and its scratch files removed. No package
  is installed and no system state is touched, so the device stays as it was
  for whatever else is sharing it.
* **It does not choose the driver silently.** `devices()` lists what the
  runtime offers and the runner is told which one to compile for, because
  "NNAPI executed it" and "the CPU reference driver executed it" are different
  claims (docs/graph/NPU2.md §3).

Every entry point refuses by name when there is no device, no `adb`, or no NDK
to build the runner with, so a caller can tell "not executed here" from
"executed and disagreed".
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess

__all__ = [
    "NnapiDeviceRefused",
    "DEVICE_DIR",
    "adb_available",
    "build_runner",
    "devices",
    "ndk_clang",
    "run_on_device",
    "serial",
    "verify_on_device",
]


class NnapiDeviceRefused(RuntimeError):
    """No device, no toolchain, or the device would not run the blob."""


#: The only directory this module writes to. docs/devices/VULKAN3.md §4's precedent:
#: the emulators are shared, so use what is already there and leave nothing
#: behind outside a directory the project owns by name.
DEVICE_DIR = "/data/local/tmp/bw_device"

_RUNNER_SOURCE = os.path.join(os.path.dirname(__file__), "nnapi_runner.c")


def serial() -> str | None:
    """The device this module will talk to, or `None`.

    `ANDROID_SERIAL` only -- never "the one attached device". Several
    emulators are shared on this machine and a command that picks its own
    target can land on somebody else's.
    """
    value = os.environ.get("ANDROID_SERIAL")
    return value or None


def adb_available() -> bool:
    return shutil.which("adb") is not None and serial() is not None


def _adb(*args, check=True, binary=False):
    if not adb_available():
        raise NnapiDeviceRefused(
            "torchnative nnapi_device: no adb on PATH, or ANDROID_SERIAL is "
            "unset. The serial is required rather than inferred: the "
            "emulators here are shared."
        )
    proc = subprocess.run(
        ["adb", "-s", serial(), *args],
        capture_output=True,
        timeout=900,
    )
    if check and proc.returncode != 0:
        raise NnapiDeviceRefused(
            f"torchnative nnapi_device: adb {' '.join(args)} exited "
            f"{proc.returncode}: {proc.stderr.decode(errors='replace').strip()}"
        )
    if binary:
        return proc.stdout
    return proc.stdout.decode(errors="replace")


def ndk_clang(api: int = 31) -> str | None:
    """The NDK clang that can build `nnapi_runner.c` for arm64, or `None`.

    `libneuralnetworks.so` first appears in the NDK sysroot at API 27, which is
    where NNAPI itself was introduced; anything below that cannot link, so the
    default is well above it rather than at the floor.
    """
    root = os.environ.get("ANDROID_NDK_HOME")
    candidates = []
    if root:
        candidates.append(root)
    sdk = os.environ.get("ANDROID_SDK_ROOT") or os.path.expanduser(
        "~/Library/Android/sdk"
    )
    ndk_dir = os.path.join(sdk, "ndk")
    if os.path.isdir(ndk_dir):
        candidates.extend(
            os.path.join(ndk_dir, name) for name in sorted(os.listdir(ndk_dir),
                                                           reverse=True)
        )
    for candidate in candidates:
        prebuilt = os.path.join(candidate, "toolchains", "llvm", "prebuilt")
        if not os.path.isdir(prebuilt):
            continue
        for host in sorted(os.listdir(prebuilt)):
            path = os.path.join(
                prebuilt, host, "bin", f"aarch64-linux-android{api}-clang"
            )
            if os.path.isfile(path):
                return path
    return None


def build_runner(destination: str, api: int = 31) -> str:
    """Compile `nnapi_runner.c` for arm64 and return the host path."""
    clang = ndk_clang(api)
    if clang is None:
        raise NnapiDeviceRefused(
            "torchnative nnapi_device: no Android NDK clang for arm64. Set "
            "ANDROID_NDK_HOME, or install an NDK under $ANDROID_SDK_ROOT/ndk."
        )
    proc = subprocess.run(
        [clang, _RUNNER_SOURCE, "-o", destination, "-lneuralnetworks"],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise NnapiDeviceRefused(
            f"torchnative nnapi_device: could not build nnapi_runner.c:\n"
            f"{proc.stderr.strip()}"
        )
    return destination


def devices(binary: str | None = None, workdir: str | None = None) -> list[dict]:
    """The NNAPI drivers the device's runtime offers.

    Read from `ANeuralNetworks_getDeviceCount` on the device rather than
    assumed. On an emulator this is typically a software reference driver and
    the sample drivers that ship with the image -- which is still execution,
    and docs/graph/NPU2.md says which one answered rather than leaving it as "an
    NPU".
    """
    import tempfile

    own = workdir is None
    workdir = workdir or tempfile.mkdtemp()
    try:
        source = os.path.join(workdir, "devices.c")
        with open(source, "w") as handle:
            handle.write(_DEVICE_PROBE_C)
        clang = ndk_clang()
        if clang is None:
            raise NnapiDeviceRefused(
                "torchnative nnapi_device: no Android NDK clang for arm64."
            )
        local = os.path.join(workdir, "devices")
        proc = subprocess.run(
            [clang, source, "-o", local, "-lneuralnetworks"],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise NnapiDeviceRefused(
                f"torchnative nnapi_device: device probe would not build:\n"
                f"{proc.stderr.strip()}"
            )
        _adb("shell", "mkdir", "-p", DEVICE_DIR)
        remote = f"{DEVICE_DIR}/npu2_devices"
        _adb("push", local, remote)
        _adb("shell", "chmod", "755", remote)
        text = _adb("shell", remote)
        found = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("device "):
                continue
            fields = dict(
                part.split("=", 1) for part in line.split(" ", 2)[2].split(" ")
                if "=" in part
            )
            found.append(fields)
        _adb("shell", "rm", "-f", remote, check=False)
        return found
    finally:
        if own:
            shutil.rmtree(workdir, ignore_errors=True)


_DEVICE_PROBE_C = r"""
#include <android/NeuralNetworks.h>
#include <stdio.h>
int main(void) {
    uint32_t n = 0;
    if (ANeuralNetworks_getDeviceCount(&n) != ANEURALNETWORKS_NO_ERROR) {
        printf("error: getDeviceCount failed\n");
        return 1;
    }
    for (uint32_t i = 0; i < n; i++) {
        ANeuralNetworksDevice *d = NULL;
        const char *name = NULL; const char *version = NULL;
        int32_t type = -1; int64_t feature = 0;
        ANeuralNetworks_getDevice(i, &d);
        ANeuralNetworksDevice_getName(d, &name);
        ANeuralNetworksDevice_getType(d, &type);
        ANeuralNetworksDevice_getVersion(d, &version);
        ANeuralNetworksDevice_getFeatureLevel(d, &feature);
        printf("device %u name=%s type=%d version=%s feature=%lld\n",
               i, name, type, version, (long long)feature);
    }
    return 0;
}
"""


def _tensor_bytes(tensor) -> bytes:
    """A shim tensor's float32 payload, in row-major order.

    Via `tolist()` and `struct`, for the reason `coreml.py::_np` gives:
    `TensorBase.numpy` is not implemented in this shim, so the buffer route
    fails with a message about a method nobody called.
    """
    flat: list[float] = []

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        else:
            flat.append(float(value))

    walk(tensor.tolist())
    return struct.pack(f"<{len(flat)}f", *flat)


def _nhwc(tensor):
    """`tensor` permuted NCHW -> NHWC, through the dispatcher."""
    import torch

    return torch._C._aten_dispatch(
        "aten.contiguous.default",
        torch._C._aten_dispatch("aten.permute.default", tensor, [0, 2, 3, 1]),
    )


def run_on_device(model, inputs, *, device_name=None, workdir=None) -> dict:
    """Ship `model` and `inputs` to the device, run, and return the outputs.

    `model` is an `nnapi.NnapiModel`. `inputs` are shim tensors in the order
    the trace takes them. Returns the raw output values as lists of floats,
    plus what the runner reported about the driver.

    An input operand the serialiser marked CHANNELS_LAST is fed NHWC: the blob
    records the *NNAPI* shape for it (upstream's `fix_shape`), so sending NCHW
    bytes would be read as a differently-shaped tensor and the disagreement
    would look like an arithmetic fault rather than a layout one.
    """
    import tempfile

    own = workdir is None
    workdir = workdir or tempfile.mkdtemp()
    try:
        blob = model.as_bytes()
        blob_path = os.path.join(workdir, "model.blob")
        with open(blob_path, "wb") as handle:
            handle.write(blob)

        weights_path = os.path.join(workdir, "weights.bin")
        with open(weights_path, "wb") as handle:
            for weight in model.weights:
                payload = _tensor_bytes(weight)
                handle.write(struct.pack("<i", len(payload)))
                handle.write(payload)

        feed_path = os.path.join(workdir, "inputs.bin")
        with open(feed_path, "wb") as handle:
            for order, value in zip(model.inp_dim_orders, inputs):
                if order == 1:  # DimOrder.CHANNELS_LAST
                    value = _nhwc(value)
                handle.write(_tensor_bytes(value))

        runner = build_runner(os.path.join(workdir, "nnapi_runner"))
        _adb("shell", "mkdir", "-p", DEVICE_DIR)
        out_remote = f"{DEVICE_DIR}/npu2_outputs.bin"
        pushed = [out_remote]
        # The device is shared, so the cleanup is in a `finally`: a driver that
        # refuses (see `nnapi-sample_quant`, docs/graph/NPU2.md §3.6) raises out of
        # the middle of this, and leaving the blob and a 5 MB binary behind
        # would make a refusal cost the next user disk.
        try:
            for local, name in (
                (runner, "npu2_runner"), (blob_path, "npu2_model.blob"),
                (weights_path, "npu2_weights.bin"),
                (feed_path, "npu2_inputs.bin"),
            ):
                remote = f"{DEVICE_DIR}/{name}"
                _adb("push", local, remote)
                pushed.append(remote)
            _adb("shell", "chmod", "755", f"{DEVICE_DIR}/npu2_runner")
            report = _adb(
                "shell",
                f"{DEVICE_DIR}/npu2_runner {DEVICE_DIR}/npu2_model.blob "
                f"{DEVICE_DIR}/npu2_weights.bin {DEVICE_DIR}/npu2_inputs.bin "
                f"{out_remote} {device_name or ''}; echo RC=$?",
            )
            rc_line = [l for l in report.splitlines() if l.startswith("RC=")]
            if not rc_line or rc_line[-1].strip() != "RC=0":
                raise NnapiDeviceRefused(
                    f"torchnative nnapi_device: the runner refused on device:\n"
                    f"{report.strip()}"
                )
            local_out = os.path.join(workdir, "outputs.bin")
            _adb("pull", out_remote, local_out)
            with open(local_out, "rb") as handle:
                raw = handle.read()
        finally:
            for remote in pushed:
                _adb("shell", "rm", "-f", remote, check=False)

        supported = None
        for line in report.splitlines():
            if line.startswith("device:") and "supports" in line:
                supported = line.split("supports", 1)[1].strip().split()[0]
        return {
            "report": report.strip(),
            "device": device_name,
            "operations_supported": supported,
            "floats": list(struct.unpack(f"<{len(raw) // 4}f", raw[: len(raw) // 4 * 4])),
        }
    finally:
        if own:
            shutil.rmtree(workdir, ignore_errors=True)


def verify_on_device(trace, inputs, *, device_name=None, tolerance=1e-5,
                     fold=True) -> dict:
    """Serialise, execute on the device, and compare against `replay`.

    This is the **executed** claim for NNAPI, the counterpart of
    `coreml.verify`. Both sides see the same inputs: the NNAPI side computed by
    a driver on the device from the blob upstream's serialiser wrote, the
    reference side by our own graph through `torch._C._aten_dispatch`.

    An output operand the serialiser marked CHANNELS_LAST comes back NHWC, so
    the reference is permuted to match rather than the device's answer being
    reshaped -- reshaping would make a layout error look like agreement on the
    first element and noise afterwards.
    """
    from .nnapi import serialize

    model = serialize(trace, fold=fold)
    result = run_on_device(model, inputs, device_name=device_name)

    reference: list[float] = []
    for order, value in zip(model.out_dim_orders, trace.replay(inputs)):
        if order == 1:  # CHANNELS_LAST
            value = _nhwc(value)
        reference.extend(struct.unpack(
            f"<{len(_tensor_bytes(value)) // 4}f", _tensor_bytes(value)))

    got = result["floats"]
    if len(got) != len(reference):
        raise NnapiDeviceRefused(
            f"torchnative nnapi_device: the device returned {len(got)} value(s) "
            f"and the trace computes {len(reference)}"
        )
    worst = max((abs(a - b) for a, b in zip(got, reference)), default=0.0)
    return {
        "executed": True,
        "device": device_name,
        "elements": len(got),
        "operations_supported": result["operations_supported"],
        "max_abs_diff": worst,
        "within_tolerance": worst <= tolerance,
        "tolerance": tolerance,
        "report": result["report"],
    }
