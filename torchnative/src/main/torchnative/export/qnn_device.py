"""Ask an attached Snapdragon device what it is, over `adb`. Never assume.

`torchnative.export.nnapi_device` is the shape this follows, including the two
rules that document earned:

* **`ANDROID_SERIAL` only, never "the one attached device".** The devices on
  this machine are shared with other work, and a command that picks its own
  target lands on somebody else's.
* **Write only under one directory this project owns by name**, install
  nothing, and remove what was pushed.

What is different is the question. NNAPI's runtime enumerates its own drivers,
so `nnapi_device.devices()` could ask the *runtime* what it had. QNN has no
such call before a backend is initialised, and initialising one needs the
Qualcomm libraries already staged on the device. So the SoC is read from
Android's own properties and mapped through **ExecuTorch's** chipset table
(`torchnative.export.qnn.resolve_soc`), and the HTP libraries are located by
looking for the files.

Nothing here executes a model. docs/devices/QNN.md §6.4 says why: this round could not
lower a QNN artefact -- the host cannot -- so there is nothing to execute, and
a module that ran *something* and reported success would be reporting on the
CPU path. What this module does is answer the two questions the device
procedure in docs/devices/QNN.md §5 starts with, from the device rather than from a
guess: **which SoC is this** and **is the HTP reachable**.

Those two are answered by different evidence and must not be run together. The
SoC is a *name*, mapped through a table; it describes the part number, and it
reads the same on a unit whose cDSP is fused off, powered down, or not exposed
to userspace at all. Only `htp_reachable` answers the second question, and it
answers it from the FastRPC compute-DSP endpoints -- see `CDSP_FASTRPC_NODES`
for the measurement that forced the two apart.
"""

from __future__ import annotations

import os
import shutil
import subprocess


__all__ = [
    "QnnDeviceRefused",
    "DEVICE_DIR",
    "SOC_PROPERTIES",
    "HTP_RUNTIME_LIBRARIES",
    "FASTRPC_NODES",
    "CDSP_FASTRPC_NODES",
    "SOC_KNOWN",
    "SOC_NOT_IN_TABLE",
    "SOC_TABLE_UNAVAILABLE",
    "serial",
    "adb_available",
    "adb",
    "getprop",
    "device_soc",
    "fastrpc_nodes",
    "cdsp_fastrpc_nodes",
    "htp_reachable",
    "device_abi",
    "htp_stub_for",
    "staged_libraries",
    "device_report",
    "cleanup",
]


class QnnDeviceRefused(RuntimeError):
    """No device, no serial, or the device cannot answer the question."""


#: The only path on the device this module writes to. Chosen to match the
#: instruction this round was given; `nnapi_device` owns `bw_device` next to it.
DEVICE_DIR = "/data/local/tmp/bw_qnn"

#: Read in order; the first that answers wins. `ro.soc.model` is the one
#: Android documents for this and the one ExecuTorch's own issue reports quote
#: (`ro.soc.model=CQ8750S` in pytorch/executorch#16465), but it is OEM
#: populated and absent on older images, so `ro.board.platform` is the
#: fallback that has been there since long before it.
SOC_PROPERTIES = ("ro.soc.model", "ro.board.platform", "ro.hardware")

#: The Qualcomm runtime libraries an HTP-delegated `.pte` needs beside it.
#: `libQnnHtp.so` is the backend, `libQnnSystem.so` its shared system layer,
#: `...V<arch>Stub.so` the CPU-side stub for one HTP generation and
#: `...V<arch>Skel.so` the matching skeleton that is loaded *on the DSP* over
#: FastRPC. The stub/skel pair is per-architecture, which is why `htp_stub_for`
#: takes the arch rather than returning a fixed list: pushing the V73 pair to a
#: V75 device stages files that will never be opened.
HTP_RUNTIME_LIBRARIES = ("libQnnHtp.so", "libQnnSystem.so")


def serial():
    """The device this module talks to, or `None`. `ANDROID_SERIAL` only."""
    return os.environ.get("ANDROID_SERIAL") or None


def adb_available():
    return shutil.which("adb") is not None and serial() is not None


def adb(*args, check=True, binary=False, timeout=300):
    if shutil.which("adb") is None:
        raise QnnDeviceRefused(
            "torchnative qnn_device: no `adb` on PATH. Add "
            "$ANDROID_SDK_ROOT/platform-tools."
        )
    if serial() is None:
        raise QnnDeviceRefused(
            "torchnative qnn_device: ANDROID_SERIAL is unset. It is required "
            "rather than inferred: the Android devices on this machine are "
            "shared, and `adb` with one device attached would silently pick "
            "whichever one that is."
        )
    proc = subprocess.run(
        ["adb", "-s", serial(), *args], capture_output=True, timeout=timeout
    )
    if check and proc.returncode != 0:
        raise QnnDeviceRefused(
            f"torchnative qnn_device: adb {' '.join(args)} exited "
            f"{proc.returncode}: {proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout if binary else proc.stdout.decode(errors="replace")


def getprop(name):
    """One Android system property, stripped. Empty string if it is unset."""
    return adb("shell", "getprop", name).strip()


def device_abi():
    """`ro.product.cpu.abi`. QNN's device libraries are `arm64-v8a` only."""
    return getprop("ro.product.cpu.abi")


#: `device_soc`'s third element. Three outcomes, and conflating any two of
#: them produces a report that is worse than no report.
SOC_KNOWN = "known"                       # in ExecuTorch's QcomChipset
SOC_NOT_IN_TABLE = "not-in-table"         # the device answered; the table does not have it
SOC_TABLE_UNAVAILABLE = "table-unavailable"   # this *host* cannot consult the table


def device_soc():
    """`(property, raw, status, chipset, soc_model, htp_arch)` read from the device.

    **`status` exists because the first version of this function did not have
    it, and it lied.** Run against a real Snapdragon 8 Gen 2 whose
    `ro.soc.model` is `SM8550` -- a chipset ExecuTorch knows perfectly well --
    it reported `chipset: null`, because `qnn.resolve_soc` needs executorch to
    read `QcomChipset` and the interpreter it was called from did not have
    executorch installed. A refusal about the *host* was rendered as a verdict
    about the *device*.

    Those two are not close. `SOC_NOT_IN_TABLE` means what
    pytorch/executorch#16465 describes -- a device reporting `CQ8750S`, on
    which QNN's own SoC detection later fails with "No Snapdragon SOC
    detected", and no lowering can help. `SOC_TABLE_UNAVAILABLE` means install
    executorch. Reporting the second as the first sends the reader to buy
    different hardware.

    The raw string is always carried, whatever the status: it is the only
    thing here that came from the device, and it is what a person needs when
    the table does not have it.
    """
    from torchnative.export import qnn

    for prop in SOC_PROPERTIES:
        raw = getprop(prop)
        if not raw:
            continue
        try:
            member, soc_model, htp_arch = qnn.resolve_soc(raw)
        except qnn.QnnRefused as exc:
            # Distinguish "the table says no" from "there is no table here" by
            # asking whether the table can be read at all, not by parsing the
            # message.
            try:
                qnn.soc_targets()
            except qnn.QnnRefused:
                return prop, raw, SOC_TABLE_UNAVAILABLE, None, None, None
            return prop, raw, SOC_NOT_IN_TABLE, None, None, None
        return prop, raw, SOC_KNOWN, member.name, soc_model, htp_arch
    raise QnnDeviceRefused(
        "torchnative qnn_device: none of "
        f"{', '.join(SOC_PROPERTIES)} is set on {serial()}. This is not a "
        "device that can be targeted by name, and guessing a chipset would "
        "produce an artefact QNN refuses at init with an architecture "
        "mismatch."
    )


def htp_stub_for(htp_arch):
    """The stub/skel pair for one HTP architecture, e.g. `(…V75Stub.so, …V75Skel.so)`.

    Derived from the architecture number rather than listed, because the list
    grows every generation and a stale copy of it here would silently omit the
    newest device -- the same reason `qnn.soc_targets()` reads ExecuTorch's
    table instead of transcribing it.
    """
    if not isinstance(htp_arch, int) or htp_arch <= 0:
        raise QnnDeviceRefused(
            f"torchnative qnn_device: {htp_arch!r} is not an HTP architecture "
            "number. It comes from ExecuTorch's `_soc_info_table` via "
            "`qnn.resolve_soc`, and there is no default that is safe to "
            "invent -- the stub must match the silicon."
        )
    return (f"libQnnHtpV{htp_arch}Stub.so", f"libQnnHtpV{htp_arch}Skel.so")


#: Every FastRPC character device this module looks for, compute and audio
#: alike. Reported for context; **not** evidence of an NPU on its own, for the
#: reason `CDSP_FASTRPC_NODES` gives.
FASTRPC_NODES = (
    "/dev/adsprpc-smd",
    "/dev/adsprpc-smd-secure",
    "/dev/fastrpc-adsp",
    "/dev/cdsprpc-smd",
    "/dev/cdsprpc-smd-secure",
    "/dev/fastrpc-cdsp",
)

#: The subset of `FASTRPC_NODES` that reaches the **compute** DSP, which is the
#: only DSP the HTP runs on. `libQnnHtpV<arch>Skel.so` is loaded *onto the cDSP*
#: over one of these, so their absence means no Hexagon NPU is reachable
#: however good the artefact is and whatever the SoC is called.
#:
#: **Separating these two tuples is the whole point.** The first version of this
#: module had one flat tuple with `/dev/adsprpc-smd` at the front of it, and
#: `device_report` reported `fastrpc: ("/dev/adsprpc-smd",)` on the Galaxy Tab
#: S9 Ultra as if that were a positive finding. It is not. `adsprpc` is the
#: **audio** DSP: it is present on essentially every Qualcomm phone and tablet
#: ever shipped, it is present when the cDSP is fused off, and nothing that
#: runs on it is an NPU workload. Measured on that tablet (SM8550, Android 16,
#: API 36): `/sys/class/fastrpc/` -- the kernel driver's own registration list,
#: which is what this should have been reading -- contains exactly
#: `adsprpc-smd` and `adsprpc-smd-secure` and **no cDSP endpoint at all**,
#: while `/dev/adsprpc-smd` is `crw-rw-r-- system:system` and so is not even
#: openable read-write by the `shell` user that would have to open it.
#:
#: Both the downstream (`cdsprpc-smd`) and mainline (`fastrpc-cdsp`) names are
#: listed because which one a kernel exposes is not fixed.
CDSP_FASTRPC_NODES = (
    "/dev/cdsprpc-smd",
    "/dev/cdsprpc-smd-secure",
    "/dev/fastrpc-cdsp",
)


def fastrpc_nodes():
    """Which of `FASTRPC_NODES` exist on the device. Stat, never `ls /dev`.

    Each path is stat'd by name rather than the directory being listed, so that
    a device whose `/dev` the `shell` user cannot enumerate is not reported as
    having no DSP.

    **A non-empty return is not an NPU.** Nearly every entry that comes back on
    a real handset is `adsprpc`, the audio DSP. Use `cdsp_fastrpc_nodes` for the
    question anybody actually means.
    """
    found = []
    for path in FASTRPC_NODES:
        out = adb("shell", f"ls -d {path} 2>/dev/null || true").strip()
        if out == path:
            found.append(path)
    return tuple(found)


def cdsp_fastrpc_nodes():
    """Which compute-DSP FastRPC endpoints exist. The HTP is reached over these.

    Two sources, and the disagreement between them matters, so both are
    consulted and the union is returned:

    * `/sys/class/fastrpc/` -- the FastRPC driver's own list of registered
      endpoints. This is the authoritative one: a name here was registered by
      the kernel driver, not merely left in a `/dev` populated by ueventd.
    * `ls -d` on each of `CDSP_FASTRPC_NODES`, for kernels that do not export
      the sysfs class.
    """
    found = []
    listing = adb(
        "shell", "ls -1 /sys/class/fastrpc 2>/dev/null || true"
    ).split()
    for path in CDSP_FASTRPC_NODES:
        name = path.rsplit("/", 1)[-1]
        if name in listing:
            found.append(path)
            continue
        out = adb("shell", f"ls -d {path} 2>/dev/null || true").strip()
        if out == path:
            found.append(path)
    return tuple(found)


def htp_reachable():
    """`(bool, reason)` -- is there a path to the Hexagon NPU on this device?

    Deliberately **not** a function of the SoC name. `device_soc` answering
    `SM8550`/`htp_arch=73` says the silicon that model number denotes has a V73
    HTP in its datasheet; it does not say this unit's cDSP is powered, fused in,
    exposed to userspace, or reachable by the uid holding the adb shell. Those
    are different claims and only the second one licenses `device.npu`.
    """
    nodes = cdsp_fastrpc_nodes()
    if not nodes:
        return False, (
            "no compute-DSP FastRPC endpoint on the device: none of "
            f"{', '.join(CDSP_FASTRPC_NODES)} exists and /sys/class/fastrpc "
            "registers none. The HTP skeleton is loaded onto the cDSP over one "
            "of these, so there is no path to a Hexagon NPU here regardless of "
            "what the SoC is called. (An `adsprpc` node is the *audio* DSP and "
            "does not substitute.)"
        )
    return True, None


def staged_libraries():
    """Names of files already under `DEVICE_DIR`, or `()` if it does not exist."""
    out = adb("shell", f"ls -1 {DEVICE_DIR} 2>/dev/null || true")
    return tuple(sorted(line.strip() for line in out.splitlines() if line.strip()))


def device_report():
    """Everything this module can establish about the attached device.

    A dict rather than an exception on the negative paths, for the reason
    `qnn.probe()` gives: when the answer is "no NPU here", the useful output is
    which half is missing.
    """
    if not adb_available():
        return {
            "serial": serial(),
            "reachable": False,
            "reason": (
                "no adb on PATH, or ANDROID_SERIAL is unset"
                if serial() is None
                else "no adb on PATH"
            ),
        }
    report = {"serial": serial(), "reachable": True, "reason": None}
    try:
        report["abi"] = device_abi()
        prop, raw, status, chipset, soc_model, htp_arch = device_soc()
        report.update(
            soc_property=prop,
            soc_raw=raw,
            soc_status=status,
            chipset=chipset,
            soc_model=soc_model,
            htp_arch=htp_arch,
        )
        report["htp_libraries"] = (
            HTP_RUNTIME_LIBRARIES + htp_stub_for(htp_arch)
            if htp_arch
            else HTP_RUNTIME_LIBRARIES
        )
        report["fastrpc"] = fastrpc_nodes()
        report["cdsp_fastrpc"] = cdsp_fastrpc_nodes()
        # The SoC name says what the datasheet has; this says what the device
        # will actually let a caller reach. `device.npu` gates on this one.
        reachable, why = htp_reachable()
        report["htp_reachable"] = reachable
        report["htp_unreachable_reason"] = why
        report["staged"] = staged_libraries()
        report["staged_complete"] = all(
            name in report["staged"] for name in report["htp_libraries"]
        )
    except QnnDeviceRefused as exc:
        report["reachable"] = False
        report["reason"] = str(exc)
    return report


def cleanup():
    """Remove `DEVICE_DIR` entirely. The devices here are shared.

    Returns True if the directory is gone afterwards. Never touches anything
    outside it and never uninstalls anything, because this module never
    installed anything.
    """
    adb("shell", f"rm -rf {DEVICE_DIR}", check=False)
    return not staged_libraries()
