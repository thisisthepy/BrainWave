"""Does `device.npu` on Android prove a Hexagon NPU, or only recognise a name?

`_resolve_qnn` used to succeed on `report["htp_arch"]` and nothing else.
`htp_arch` is produced by reading `ro.soc.model` off the device and looking the
string up in ExecuTorch's chipset table. It is a **part number**. It reads the
same on a unit whose compute DSP is fused off, powered down, or simply not
exposed to the uid holding the adb shell -- and it reads the same on anything
that merely claims the string.

Measured on the Galaxy Tab S9 Ultra attached to this machine (SM-X910, SM8550,
Android 16 / API 36):

    ro.soc.model      SM8550          -> htp_arch 73, chipset SM8550
    /sys/class/fastrpc  adsprpc-smd, adsprpc-smd-secure   <- and nothing else
    /dev/cdsprpc-smd  absent          (as is /dev/fastrpc-cdsp)
    /dev/adsprpc-smd  crw-rw-r-- system:system   <- shell cannot even open it rw
    QNN runtime libs on device        none (staged_complete false)

and `device.npu.resolve()` returned, confidently,
`<NpuResolution android: qnn -> Qualcomm Hexagon NPU>`. Every input to that
verdict was a name. `adsprpc` is the **audio** DSP; it is present on essentially
every Qualcomm part ever shipped and nothing that runs on it is an NPU
workload. This is `_mps_is_available` inverted -- a hardcoded yes rather than a
hardcoded no -- and it is the same shape as the NNAPI path claiming vendor
acceleration on a tablet that enumerated only `nnapi-reference`.

Nothing here needs a device, executorch, or torch. Each test drives
`_resolve_qnn` with a synthetic report, because the property under test is what
the resolver concludes from evidence -- and the dangerous evidence is exactly
what a green run against real hardware never produces.

Nullifications this file is meant to catch, each verified by making the break
and watching it go red:

* the `htp_reachable` gate deleted from `_resolve_qnn`
      -> `test_a_named_soc_with_no_compute_dsp_does_not_resolve`
* an `adsprpc` node counted as a compute DSP
      -> `test_the_audio_dsp_is_not_accepted_as_the_compute_dsp`
* `htp_reachable` hardcoded true
      -> `test_htp_reachability_is_read_from_the_device_not_declared`
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# Importing `torchnative.device` reaches `torch`, whose `_load_global_deps()`
# dlopens `torch/lib/libtorch_global_deps.*`. That file is created by
# tools/wheel/build.py for a wheel and does not exist in the source tree, so
# without this the import raises OSError -- which is how these three tests
# passed standalone (the variable happened to be set) and failed under
# run.sh. Every other suite here sets it the same way, before importing torch.
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")


def _device_ns():
    """`torchnative.device`, or `None` if the vendored tree is not here."""
    if not os.path.isdir(os.path.join(_VENDOR_DIR, "torchnative", "export")):
        return None
    import importlib

    return importlib.import_module("torchnative.device")


def _qnn_device():
    if not os.path.isdir(os.path.join(_VENDOR_DIR, "torchnative", "export")):
        return None
    import importlib

    return importlib.import_module("torchnative.export.qnn_device")


#: The report the attached Galaxy Tab S9 Ultra actually produces, transcribed.
#: Not invented: this is `device_report()`'s output on that device, with the
#: serial removed. It is the case that used to pass.
_TAB_S9_REPORT = {
    "serial": "R54W903JCSE",
    "reachable": True,
    "reason": None,
    "abi": "arm64-v8a",
    "soc_property": "ro.soc.model",
    "soc_raw": "SM8550",
    "soc_status": "known",
    "chipset": "SM8550",
    "soc_model": 43,
    "htp_arch": 73,
    "htp_libraries": (
        "libQnnHtp.so",
        "libQnnSystem.so",
        "libQnnHtpV73Stub.so",
        "libQnnHtpV73Skel.so",
    ),
    "fastrpc": ("/dev/adsprpc-smd", "/dev/adsprpc-smd-secure"),
    "cdsp_fastrpc": (),
    "htp_reachable": False,
    "htp_unreachable_reason": "no compute-DSP FastRPC endpoint on the device",
    "staged": (),
    "staged_complete": False,
}


def _resolve_with(D, report):
    """Run `_resolve_qnn` against a fixed report, whatever host this is."""
    from torchnative.export import qnn_device

    saved = qnn_device.device_report
    qnn_device.device_report = lambda: dict(report)
    try:
        return D.npu._resolve_qnn("android", "qnn", "Qualcomm Hexagon NPU")
    finally:
        qnn_device.device_report = saved


def test_a_named_soc_with_no_compute_dsp_does_not_resolve():
    """The load-bearing one. This exact report used to return a resolution."""
    D = _device_ns()
    if D is None:
        print("   (skipped: no vendored torchnative tree)")
        return
    try:
        res = _resolve_with(D, _TAB_S9_REPORT)
    except D.NpuUnresolved as exc:
        text = str(exc)
        assert "SM8550" not in text or "part number" in text, text
        assert "compute-DSP" in text or "compute DSP" in text, text
    else:
        raise AssertionError(
            "a device with htp_arch=73 and no compute-DSP FastRPC endpoint "
            f"resolved to {res!r} -- the SoC name alone was taken as proof of "
            "a Hexagon NPU"
        )
    print("ok   qnnprobe: a recognised SoC name with no reachable cDSP refuses by name")


def test_the_audio_dsp_is_not_accepted_as_the_compute_dsp():
    """`adsprpc` must not be in the set that licenses the claim."""
    Q = _qnn_device()
    if Q is None:
        print("   (skipped: no vendored torchnative tree)")
        return
    for node in Q.CDSP_FASTRPC_NODES:
        assert "adsprpc" not in node, (
            f"{node} is an audio-DSP endpoint and is present on virtually every "
            "Qualcomm device; counting it makes the probe unfalsifiable"
        )
    assert any("cdsp" in n for n in Q.CDSP_FASTRPC_NODES), Q.CDSP_FASTRPC_NODES
    # And the resolver must not read the permissive tuple instead.
    D = _device_ns()
    report = dict(_TAB_S9_REPORT, htp_reachable=False)
    report["fastrpc"] = ("/dev/adsprpc-smd",)
    try:
        _resolve_with(D, report)
    except D.NpuUnresolved:
        pass
    else:
        raise AssertionError("an adsprpc node alone was accepted as a Hexagon NPU")
    print("ok   qnnprobe: the audio DSP is not accepted in place of the compute DSP")


def test_htp_reachability_is_read_from_the_device_not_declared():
    """`htp_reachable` must come from a probe, and must be able to say no."""
    Q = _qnn_device()
    if Q is None:
        print("   (skipped: no vendored torchnative tree)")
        return
    saved = Q.cdsp_fastrpc_nodes
    try:
        Q.cdsp_fastrpc_nodes = lambda: ()
        ok, why = Q.htp_reachable()
        assert ok is False, "htp_reachable said yes with no cDSP endpoint at all"
        assert why and "cdsprpc" in why, why
        Q.cdsp_fastrpc_nodes = lambda: ("/dev/cdsprpc-smd",)
        ok, why = Q.htp_reachable()
        assert ok is True and why is None, (ok, why)
    finally:
        Q.cdsp_fastrpc_nodes = saved
    print("ok   qnnprobe: htp_reachable answers from the cDSP endpoints, both ways")


def test_a_reachable_compute_dsp_still_resolves():
    """The fix must refuse the unproven case without refusing every case."""
    D = _device_ns()
    if D is None:
        print("   (skipped: no vendored torchnative tree)")
        return
    report = dict(
        _TAB_S9_REPORT,
        cdsp_fastrpc=("/dev/cdsprpc-smd",),
        htp_reachable=True,
        htp_unreachable_reason=None,
    )
    res = _resolve_with(D, report)
    assert res.unit == "Qualcomm Hexagon NPU", res
    assert res.backend == "qnn", res
    print("ok   qnnprobe: a device with a reachable cDSP does resolve -- the gate is not a block")


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
