"""The NPU vendor is resolved by probe, not read off the operating system.

See docs/devices/NPUVENDOR.md.

**What is checkable here and what is not.** This is an arm64 Mac. There is no
Windows, no registry, no Intel NPU, no AMD NPU and no Snapdragon, so *none* of
`_pcivendor`'s registry code can execute. What can be checked is everything that
does not need the hardware, and the split is deliberate:

* The **candidate table and its order** are pure data plus `platform.machine()`,
  and both `TORCHNATIVE_DEVICE_HOST` and `TORCHNATIVE_DEVICE_MACHINE` move, so
  win_amd64 and win_arm64 are both exercised from here. Without the second
  variable an order that ignored the ISA would pass --- it would answer the same
  on this Mac and never be asked anything else.
* The **classifier** is fed registry-shaped dictionaries directly. That is not a
  claim that a real machine produces those dictionaries (NPUVENDOR.md section 6
  says it is unverified); it is a check that *given* the shape Microsoft
  documents, an AMD NPU is recognised as an AMD NPU and not as an absence.
* The **refusal wording** is checked for the sentence a Ryzen AI owner must not
  read. That is the whole defect, so it gets a test that names it.
* The **"could not look" / "looked and found nothing"** distinction is checked
  as two different strings, because collapsing them is the defect with a
  different cause.

Nullifications this file is meant to catch, each verified by making the break
and watching it go red:

* the candidate table reduced to one backend per host -> `test_windows_tries_more_than_one_candidate_in_an_isa_dependent_order`
* the ISA ignored when ordering                        -> same test
* an AMD NPU reported as nothing                       -> `test_an_amd_npu_is_recognised_by_name_and_not_reported_as_an_absence`
* "could not look" rendered as "found nothing"         -> `test_could_not_look_is_a_different_sentence_from_found_nothing`
* the refusal telling an NPU owner they have no NPU    -> `test_the_refusal_never_tells_a_machine_with_an_npu_that_it_has_none`
* the adb probe run on Windows                         -> `test_the_hexagon_candidate_refuses_on_windows_rather_than_probing_over_adb`
* the class-code signal deleted                        -> `test_an_unknown_vendors_accelerator_is_still_caught_by_the_class_code`
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
sys.path.insert(0, _VENDOR_DIR)

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import torch  # noqa: E402

import torchnative.device as D  # noqa: E402
from torchnative.device import _pcivendor  # noqa: E402


def _shim():
    """CLAUDE.md section 3: is this the shim or did we silently get upstream?"""
    assert hasattr(torch._C, "_aten_implemented"), (
        "not the torchnative shim -- torch._C has no _aten_implemented, so the "
        "vendored tree is missing and every probe below would measure upstream"
    )


def _with_env(pairs, fn):
    old = {k: os.environ.get(k) for k, _v in pairs}
    for k, v in pairs:
        os.environ[k] = v
    try:
        return fn()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _entry(ven, dev, desc=None, compat=()):
    """A registry-shaped record, in the spelling Microsoft's PCI ID page gives."""
    return {
        "vendor_id": ven,
        "device_id": dev,
        "key": f"VEN_{ven}&DEV_{dev}&SUBSYS_00000000&REV_00\\3&11583659&0&40",
        "desc": desc,
        "service": None,
        "class": None,
        "compatible_ids": list(compat),
        "hardware_ids": [],
    }


# --------------------------------------------------------------------------
# The candidate table
# --------------------------------------------------------------------------


def test_windows_tries_more_than_one_candidate_in_an_isa_dependent_order():
    """The OS is not the vendor, so Windows cannot map to one backend.

    Both halves matter. If the table went back to one entry per host the first
    assertion fails; if the order stopped reading the ISA the last one does.
    """
    _shim()
    amd64 = D.npu_candidates("windows", "AMD64")
    arm64 = D.npu_candidates("windows", "ARM64")

    assert len(amd64) >= 2, (
        f"windows offers {amd64} -- a single candidate is the OS-as-vendor "
        f"table again, and it is what told a Ryzen AI owner they had an Intel NPU"
    )
    assert set(b for b, _u in amd64) == set(b for b, _u in arm64), (amd64, arm64)
    assert amd64[0][0] == "openvino", amd64
    assert arm64[0][0] == "qnn", arm64
    assert amd64 != arm64, (
        "the same order on x86-64 and arm64 means the ISA is not read at all"
    )
    print(
        f"ok   npuvendor: windows tries {[b for b, _ in amd64]} on AMD64 and "
        f"{[b for b, _ in arm64]} on ARM64"
    )


def test_the_single_candidate_hosts_are_unchanged():
    """darwin and android had no defect and must not acquire one."""
    _shim()
    assert D.npu_candidates("darwin", "arm64") == (("coreml", "Apple Neural Engine"),)
    assert D.npu_candidates("android", "aarch64") == (
        ("qnn", "Qualcomm Hexagon NPU"),
    )
    assert D.npu_candidates("freebsd", "amd64") == ()
    print("ok   npuvendor: darwin and android keep one candidate, an unknown host has none")


def test_the_hexagon_candidate_refuses_on_windows_rather_than_probing_over_adb():
    """The only Hexagon probe here talks to a phone. Running it on Windows
    would answer about whatever is plugged in -- QNN.md section 5 backwards."""
    _shim()
    try:
        D.npu._resolve_qnn("windows", "qnn", "Qualcomm Hexagon NPU")
    except D.NpuUnresolved as exc:
        text = str(exc)
        assert "adb" in text, text
        assert "Qualcomm Hexagon NPU" in text, text
        assert "no probe" in text, text
    else:
        raise AssertionError("the qnn candidate must refuse by name on windows")
    print("ok   npuvendor: the hexagon candidate refuses on windows and names the adb probe")


# --------------------------------------------------------------------------
# Vendor detection
# --------------------------------------------------------------------------


def test_an_amd_npu_is_recognised_by_name_and_not_reported_as_an_absence():
    """The defect, in one test. VEN_1022&DEV_17F0 is pci.ids v2.2 line 5710."""
    _shim()
    hit = _pcivendor.classify(_entry("1022", "17F0", desc="NPU Compute Accelerator Device"))
    assert hit is not None, "an AMD NPU classified as not-an-accelerator"
    assert "AMD" in hit["vendor"], hit
    assert "Neural Processing Unit" in hit["product"], hit
    assert hit["backend"] is None, (
        "this project has no AMD execution path and must not claim one"
    )
    text = _pcivendor.describe({"scanned": True, "examined": 40, "source": "x", "found": [hit]})
    assert "AMD" in text and "no execution path" in text, text
    print(f"ok   npuvendor: an AMD NPU is named -- {text[:70]}")


def test_an_intel_npu_is_recognised_and_routed_to_openvino():
    _shim()
    for dev in ("643E", "7D1D", "AD1D", "B03E"):
        hit = _pcivendor.classify(_entry("8086", dev))
        assert hit is not None, dev
        assert hit["backend"] == "openvino", hit
        assert "Intel" in hit["vendor"], hit
    print("ok   npuvendor: the four sourced Intel NPU device ids route to openvino")


def test_a_gaussian_neural_accelerator_is_not_counted_as_an_npu():
    """GNA is a different, older Intel block with its own OpenVINO plugin.
    Counting it would be claiming hardware that is not there."""
    _shim()
    for dev in ("4511", "464F", "4E11", "774C", "7E4C", "9A11"):
        assert ("8086", dev) not in _pcivendor.SOURCED_NPU_DEVICES, dev
        assert _pcivendor.classify(_entry("8086", dev)) is None, dev
    print("ok   npuvendor: the six sourced Intel GNA ids are not treated as NPUs")


def test_an_unknown_vendors_accelerator_is_still_caught_by_the_class_code():
    """The allowlist cannot know a part released after pci.ids v2.2. The
    vendor-neutral PCI class code 12h/00h can, and that is why it is here."""
    _shim()
    hit = _pcivendor.classify(
        _entry("BEEF", "0001", desc="Some Vendor AI Accelerator",
               compat=["PCI\\VEN_BEEF&CC_1200", "PCI\\CC_1200"])
    )
    assert hit is not None, "an unknown vendor's class-12 accelerator went unseen"
    assert hit["vendor"] is None, hit
    assert hit["backend"] is None, hit
    assert any("class code" in m for m in hit["matched_by"]), hit
    assert not _pcivendor.classify(_entry("BEEF", "0001")), (
        "with no class code and no sourced name there is nothing to go on, and "
        "guessing would be the disease"
    )
    print("ok   npuvendor: an unknown vendor is caught by CC_1200 and named as unrouted")


def test_an_ordinary_device_is_not_mistaken_for_an_accelerator():
    _shim()
    assert _pcivendor.classify(
        _entry("8086", "A0F0", desc="Wi-Fi 6 AX201", compat=["PCI\\CC_0280"])
    ) is None
    print("ok   npuvendor: an ordinary network controller is not classified as an NPU")


def test_could_not_look_is_a_different_sentence_from_found_nothing():
    """"I could not look" collapsed into "there is none" is the original defect
    with a new cause, so the two renderings must not be able to coincide."""
    _shim()
    could_not = _pcivendor.describe(
        {"scanned": False, "reason": "sys.platform is 'darwin'", "found": []}
    )
    nothing = _pcivendor.describe(
        {"scanned": True, "examined": 41, "source": "winreg ...", "found": []}
    )
    assert could_not != nothing
    assert "cannot see" in could_not, could_not
    assert "no neural accelerator was among them" in nothing, nothing
    assert "cannot see" not in nothing, nothing
    print("ok   npuvendor: 'could not look' and 'looked and found nothing' are distinct sentences")


def test_the_scan_refuses_to_answer_off_windows_instead_of_returning_empty():
    """On this Mac. An empty list would read as 'no accelerators here'."""
    _shim()
    report = _pcivendor.npu_vendor_report()
    if sys.platform == "win32":
        print("ok   npuvendor: on windows -- the scan ran (see tools/devices/npuvendor_verify.py)")
        return
    assert report["scanned"] is False, report
    assert report["reason"], report
    assert sys.platform in report["reason"], report
    assert report["found"] == []
    print(f"ok   npuvendor: off windows the scan says why it could not look -- {report['reason'][:60]}")


def test_the_registry_key_spelling_is_the_documented_one():
    """`parse_pci_instance_key` is the only part of the registry walk that can
    be exercised here, so it is exercised."""
    _shim()
    assert _pcivendor.parse_pci_instance_key(
        "VEN_8086&DEV_7D1D&SUBSYS_00000000&REV_00"
    ) == ("8086", "7D1D")
    assert _pcivendor.parse_pci_instance_key("ven_1022&dev_17f0") == ("1022", "17F0")
    assert _pcivendor.parse_pci_instance_key("VEN_8086") is None
    assert _pcivendor.parse_pci_instance_key("ACPI\\VEN_QCOM") is None
    print("ok   npuvendor: the VEN_/DEV_ key spelling parses, case-folded, and refuses a partial one")


# --------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------


def test_the_refusal_never_tells_a_machine_with_an_npu_that_it_has_none():
    """A Ryzen AI owner, simulated at the one seam that can be simulated here.

    The PCI scan is replaced with one answering as that machine's registry
    would; the resolver, the candidate order and the wording are the real ones.
    This is not evidence that a Ryzen AI machine enumerates that way ---
    NPUVENDOR.md section 6 lists that as unverified --- it is evidence that when
    it does, the sentence names the AMD part.
    """
    _shim()
    amd = _pcivendor.classify(_entry("1022", "17F0", desc="NPU Compute Accelerator Device"))
    saved = _pcivendor.npu_vendor_report
    try:
        _pcivendor.npu_vendor_report = lambda: {
            "scanned": True, "examined": 44, "source": "winreg (fake)",
            "found": [amd], "reason": None, "host": "windows",
        }

        def probe():
            try:
                D.npu.resolve()
            except D.NpuUnresolved as exc:
                return str(exc)
            return None

        err = _with_env(
            [("TORCHNATIVE_DEVICE_HOST", "windows"),
             ("TORCHNATIVE_DEVICE_MACHINE", "AMD64")],
            probe,
        )
    finally:
        _pcivendor.npu_vendor_report = saved

    assert err is not None, "the fake AMD machine must not resolve -- there is no path"
    assert "AMD" in err, err
    assert "Neural Processing Unit" in err, err
    assert "no execution path" in err, err
    # Both candidates named, so the message is not one backend's opinion.
    assert "openvino" in err and "qnn" in err, err
    assert "cpu" in err.lower(), err
    print(f"ok   npuvendor: a Ryzen AI machine is refused by name -- ...{err[-110:]}")


def test_the_refusal_on_a_windows_box_with_no_npu_says_so_without_overclaiming():
    _shim()
    saved = _pcivendor.npu_vendor_report
    try:
        _pcivendor.npu_vendor_report = lambda: {
            "scanned": True, "examined": 39, "source": "winreg (fake)",
            "found": [], "reason": None, "host": "windows",
        }
        try:
            _with_env(
                [("TORCHNATIVE_DEVICE_HOST", "windows"),
                 ("TORCHNATIVE_DEVICE_MACHINE", "AMD64")],
                D.npu.resolve,
            )
        except D.NpuUnresolved as exc:
            err = str(exc)
        else:
            raise AssertionError("resolved on a fake windows box with no NPU")
    finally:
        _pcivendor.npu_vendor_report = saved
    assert "no neural accelerator was among them" in err, err
    assert "cannot see" not in err, err
    print("ok   npuvendor: a windows box with no NPU is told the enumeration was read and was empty")


def test_a_snapdragon_windows_box_is_not_told_it_has_no_intel_npu():
    """win_arm64 is a wheel we ship. The old table told that machine it had an
    Intel NPU; the least it must get now is its own candidate first."""
    _shim()
    order = D.npu_candidates("windows", "ARM64")
    assert order[0] == ("qnn", "Qualcomm Hexagon NPU"), order

    def probe():
        try:
            D.npu.resolve()
        except D.NpuUnresolved as exc:
            return str(exc)
        return None

    err = _with_env(
        [("TORCHNATIVE_DEVICE_HOST", "windows"),
         ("TORCHNATIVE_DEVICE_MACHINE", "ARM64")],
        probe,
    )
    assert err is not None, err
    assert err.index("Qualcomm Hexagon NPU") < err.index("Intel NPU"), (
        "on arm64 windows the Hexagon must be the first thing named, not Intel"
    )
    print("ok   npuvendor: a win_arm64 box is refused Hexagon-first, not Intel-first")


def test_the_openvino_candidate_is_not_offered_as_an_amd_path():
    """Settled from source, NPUVENDOR.md section 1: the plugin selects its Level
    Zero driver by memcmp against Intel's own UUID. So no candidate list may
    pair openvino with an AMD unit, and no vendor mapping may route AMD to it."""
    _shim()
    for host, cands in D.NPU_CANDIDATES.items():
        for backend, unit in cands:
            if backend == "openvino":
                assert unit == "Intel NPU", (host, backend, unit)
    assert _pcivendor.VENDOR_BACKENDS["1022"] is None
    assert _pcivendor.VENDOR_BACKENDS["17CB"] is None
    assert _pcivendor.VENDOR_BACKENDS["8086"] == "openvino"
    print("ok   npuvendor: openvino is only ever paired with the Intel NPU, and AMD routes nowhere")


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
