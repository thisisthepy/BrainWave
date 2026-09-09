"""Which vendor's NPU is in this machine --- asked of the OS, not of the OS name.

`docs/devices/NPUVENDOR.md` is the round that wrote this. The defect it is for:
`device/__init__.py` used to read the NPU vendor off `platform.system()`, so
every Windows machine was told it had an **Intel** NPU. On a Ryzen AI laptop
that produced a refusal reading "this machine has no Intel NPU" --- true, and
the machine has an NPU. Denying present hardware is the same disease as
claiming absent hardware (`docs/devices/QNN.md` §5), pointed the other way.

To refuse by name you have to know what **is** there, not only what is not, and
this module is the "what is there" half. It does not target anything and it
must not be read as a claim that anything is reachable: it reads Windows' own
Plug and Play enumeration out of the registry with `winreg` --- stdlib, no
vendor SDK, no COM, no subprocess --- and reports which PCI functions look like
neural accelerators and who made them.

**Every identifier below is sourced.** Nothing here is from memory; see
`docs/devices/NPUVENDOR.md` §3 for the two sources and §6 for what could not be
settled on this machine (an arm64 Mac with no Windows and no NPU of any
vendor, so *none* of the registry paths below has been executed once).
"""

import platform
import sys

__all__ = [
    "NPU_PCI_CLASS_CODES",
    "PCI_VENDORS",
    "SOURCED_NPU_DEVICES",
    "VENDOR_BACKENDS",
    "enumerate_pci",
    "npu_vendor_report",
]


# PCI-SIG vendor identifiers. Source: `pci.ids` v2.2 from
# <https://pci-ids.ucw.cz/v2.2/pci.ids>, the repository Microsoft's own
# "Identifiers for PCI Devices" page links to as the list of known PCI IDs.
# Line numbers are that file's, fetched 2026-09-10.
PCI_VENDORS = {
    "1022": "Advanced Micro Devices, Inc. [AMD]",   # pci.ids:5142
    "17CB": "Qualcomm Technologies, Inc",           # pci.ids:24556
    "1DB7": "Phytium Technology Co., Ltd.",         # pci.ids:27909
    "1FF4": "DEEPX Co., Ltd.",                      # pci.ids:30221
    "8086": "Intel Corporation",                    # pci.ids:31421
}


# Individual PCI functions whose *name in pci.ids* says "NPU" or "Neural
# Processing Unit". Same file, same fetch. This is an allowlist and it is
# deliberately not the primary signal --- a device released after this snapshot
# is simply absent from it, which is why `npu_vendor_report` also matches on the
# class code below and reports the two matches separately.
#
# Gaussian & Neural Accelerator (GNA) parts are **excluded on purpose**: pci.ids
# names them "Gaussian & Neural-Network Accelerator", they are a different and
# older Intel block from the Core Ultra NPU, and OpenVINO's GNA plugin was a
# separate plugin from its NPU plugin. Treating a GNA as an NPU would be the
# claim-what-is-not-there failure again.
SOURCED_NPU_DEVICES = {
    ("1022", "17F0"): "Strix/Krackan/Strix Halo Neural Processing Unit",  # pci.ids:5710
    ("1DB7", "DC24"): "NPU Controller [X100 Series]",                     # pci.ids:27914
    ("1FF4", "0102"): "M1 [H1 V-NPU]",                                    # pci.ids:30227
    ("1FF4", "0112"): "M1M [H1M V-NPU]",                                  # pci.ids:30233
    ("1FF4", "2001"): "VPU [H1/H1M V-NPU]",                               # pci.ids:30235
    ("8086", "643E"): "Core Ultra 200V Series Processors NPU",            # pci.ids:39255
    ("8086", "7D1D"): "Core Ultra 200H/200V Series Processors NPU",       # pci.ids:39789
    ("8086", "AD1D"): "Core Ultra 200 Series Processors NPU",             # pci.ids:41281
    ("8086", "B03E"): "Core Ultra Processors (Series 3) NPU",             # pci.ids:41304
}


# PCI base class 12h, subclass 00h: "Processing accelerators". Source: the same
# `pci.ids` v2.2 class section --- `C 12  Processing accelerators` with
# `00  Processing accelerators` under it (the only other subclass defined there
# is `01  SNIA Smart Data Accelerator Interface (SDXI) controller`, which is a
# data-mover, not a neural accelerator, and is not matched).
#
# Windows exposes this as a *compatible ID* of the form `PCI\CC_c(2)s(2)`,
# documented in "Identifiers for PCI Devices"
# <https://learn.microsoft.com/en-us/windows-hardware/drivers/install/identifiers-for-pci-devices>,
# which lists `PCI\VEN_v&CC_c(2)s(2)`, `PCI\CC_c(2)s(2)p(2)` and `PCI\CC_c(2)s(2)`
# among the compatible IDs the PCI bus driver reports. So the string to look for
# is `CC_1200`, and it is vendor-neutral by construction --- which is the whole
# point, since the vendor this round cannot target is the one it must recognise.
NPU_PCI_CLASS_CODES = ("CC_1200",)


# Which vendor this project has an execution path for, and which it does not.
# `None` means recognised and **not targetable from here** --- that is a real
# entry, not a gap to be filled in silently later. See NPUVENDOR.md §2:
# recognising a vendor we cannot target is this round's deliverable; targeting
# it is explicitly not.
VENDOR_BACKENDS = {
    "8086": "openvino",
    "1022": None,
    "17CB": None,
    "1DB7": None,
    "1FF4": None,
}


class PciScanUnavailable(Exception):
    """The PnP enumeration could not be read, and the message says why."""


def _normalise(value):
    return value.strip().upper()


def parse_pci_instance_key(name):
    """`VEN_8086&DEV_7D1D&SUBSYS_...&REV_00` -> `("8086", "7D1D")`, or `None`.

    Split out from the registry walk so it can be tested on this Mac, which
    has no registry. The key spelling is Microsoft's hardware-ID format from
    "Identifiers for PCI Devices"; the registry's `Enum\\PCI` subkey names use
    the same `VEN_`/`DEV_` fields.
    """
    ven = dev = None
    for part in _normalise(name).split("&"):
        if part.startswith("VEN_"):
            ven = part[4:]
        elif part.startswith("DEV_"):
            dev = part[4:]
    if ven is None or dev is None:
        return None
    return ven, dev


def _ids(values):
    out = []
    for v in values or ():
        if isinstance(v, str):
            out.append(_normalise(v))
    return out


def enumerate_pci():
    """Every PCI function Windows' PnP manager knows about.

    Returns `(entries, source)`. Raises `PciScanUnavailable`, by name, on any
    host or interpreter that cannot answer --- an empty list would read as "no
    devices", and "I could not look" and "I looked and there is nothing" are
    the two answers this whole module exists to keep apart.
    """
    if sys.platform != "win32":
        raise PciScanUnavailable(
            f"PnP enumeration is a Windows facility and sys.platform is "
            f"{sys.platform!r}; nothing here can enumerate this machine's PCI "
            f"functions"
        )
    try:
        import winreg
    except ImportError as exc:  # pragma: no cover - Windows only
        raise PciScanUnavailable(f"winreg is not importable: {exc}") from None

    root = r"SYSTEM\CurrentControlSet\Enum\PCI"
    entries = []
    try:
        base = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root)
    except OSError as exc:  # pragma: no cover - Windows only
        raise PciScanUnavailable(
            f"HKLM\\{root} could not be opened ({type(exc).__name__}: {exc}). "
            f"On this Windows build the PnP enumeration is not readable by this "
            f"process, so this build cannot see which accelerators are present"
        ) from None

    with base:
        i = 0
        while True:
            try:
                device_key = winreg.EnumKey(base, i)
            except OSError:
                break
            i += 1
            ids = parse_pci_instance_key(device_key)
            if ids is None:
                continue
            ven, dev = ids
            try:
                dk = winreg.OpenKey(base, device_key)
            except OSError:
                continue
            with dk:
                j = 0
                while True:
                    try:
                        instance = winreg.EnumKey(dk, j)
                    except OSError:
                        break
                    j += 1
                    entry = {
                        "vendor_id": ven,
                        "device_id": dev,
                        "key": device_key + "\\" + instance,
                        "desc": None,
                        "service": None,
                        "class": None,
                        "compatible_ids": [],
                        "hardware_ids": [],
                    }
                    try:
                        ik = winreg.OpenKey(dk, instance)
                    except OSError:
                        entries.append(entry)
                        continue
                    with ik:
                        for value, field in (
                            ("DeviceDesc", "desc"),
                            ("Service", "service"),
                            ("Class", "class"),
                        ):
                            try:
                                entry[field] = winreg.QueryValueEx(ik, value)[0]
                            except OSError:
                                pass
                        for value, field in (
                            ("CompatibleIDs", "compatible_ids"),
                            ("HardwareID", "hardware_ids"),
                        ):
                            try:
                                entry[field] = _ids(winreg.QueryValueEx(ik, value)[0])
                            except OSError:
                                pass
                    entries.append(entry)
    return entries, r"winreg HKLM\SYSTEM\CurrentControlSet\Enum\PCI"


def classify(entry):
    """Why (if at all) this PCI function looks like a neural accelerator.

    Returns `None`, or a dict naming the vendor, the product where a source
    names it, and **which of the two signals matched** --- they are different
    strengths of evidence and collapsing them would hide that.
    """
    ven = _normalise(entry.get("vendor_id") or "")
    dev = _normalise(entry.get("device_id") or "")
    matched = []
    product = SOURCED_NPU_DEVICES.get((ven, dev))
    if product is not None:
        matched.append("pci.ids device name")
    ids = list(entry.get("compatible_ids") or ()) + list(entry.get("hardware_ids") or ())
    if any(code in one for code in NPU_PCI_CLASS_CODES for one in ids):
        matched.append("PCI class code 1200 (processing accelerator)")
    if not matched:
        return None
    return {
        "vendor_id": ven,
        "vendor": PCI_VENDORS.get(ven),
        "device_id": dev,
        "product": product,
        "desc": entry.get("desc"),
        "matched_by": matched,
        "backend": VENDOR_BACKENDS.get(ven),
        "key": entry.get("key"),
    }


def npu_vendor_report():
    """What accelerators this machine has, or why that could not be asked.

    Always a dict, never an exception, for `qnn_device.device_report`'s reason:
    when the answer is "no NPU here" the useful output is *which half* is
    missing. `scanned` is the field that separates the two --- `False` means
    nothing was looked at, and a caller that reads `found == []` without reading
    `scanned` would turn "could not look" into "there is none", which is the
    exact sentence this round exists to stop being printed.
    """
    report = {
        "scanned": False,
        "source": None,
        "reason": None,
        "found": [],
        "host": platform.system().lower(),
    }
    try:
        entries, source = enumerate_pci()
    except PciScanUnavailable as exc:
        report["reason"] = str(exc)
        return report
    report["scanned"] = True
    report["source"] = source
    report["examined"] = len(entries)
    found = []
    for entry in entries:
        hit = classify(entry)
        if hit is not None:
            found.append(hit)
    report["found"] = found
    return report


def describe(report):
    """One sentence a refusal can carry, in the house style: name the thing.

    The three cases are deliberately distinct sentences. "No accelerator was
    found" and "I could not look" must not compress to the same words.
    """
    if not report.get("scanned"):
        return (
            "this build cannot see whether another vendor's NPU is present -- "
            + (report.get("reason") or "the PCI/PnP enumeration was not read")
        )
    found = report.get("found") or []
    if not found:
        return (
            "the PCI/PnP enumeration was read ({} functions via {}) and no "
            "neural accelerator was among them".format(
                report.get("examined", 0), report.get("source")
            )
        )
    parts = []
    for hit in found:
        vendor = hit.get("vendor") or f"PCI vendor {hit['vendor_id']}"
        product = hit.get("product") or hit.get("desc") or f"device {hit['device_id']}"
        if hit.get("backend"):
            parts.append(f"{vendor} {product} (this project targets it via {hit['backend']})")
        else:
            parts.append(
                f"{vendor} {product} -- recognised, and this project has no "
                f"execution path for it"
            )
    return "the PCI/PnP enumeration names " + "; ".join(parts)
