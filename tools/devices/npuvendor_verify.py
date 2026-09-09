"""Run this on a Windows machine with an NPU. It answers two SEPARATE questions.

    python tools/devices/npuvendor_verify.py
    python tools/devices/npuvendor_verify.py --dump-all

Stage A -- SELECTION. What this build *would* do: which candidate backends it
would try on this host in which order, and what the PCI/PnP enumeration says is
in the machine. Pure Python: stdlib `winreg` only, no OpenVINO, no QNN, no
vendor SDK, nothing executed on any accelerator. Runs anywhere; on a non-Windows
box it prints the named "could not look" answer, which is itself the thing
docs/devices/NPUVENDOR.md section 3.2 wants checked.

Stage B -- EXECUTION. Whether `torchnative.device.npu` actually resolves, which
runs the real probes and needs real hardware. It is the ONLY stage that is
evidence about hardware; Stage A is not, and says so.

Keeping them apart is intelnpu_verify.py's rule and it is here for the same
reason: a green Stage A on a machine with no NPU is still a true result about
selection, and reporting the two together is how a selection result gets read as
an execution claim.

What Stage A settles, from NPUVENDOR.md section 6's UNVERIFIED list:

  * whether HKLM\\SYSTEM\\CurrentControlSet\\Enum\\PCI is readable unelevated;
  * whether an Intel NPU really appears there as VEN_8086&DEV_<id>;
  * whether its CompatibleIDs really contain CC_1200 -- the vendor-neutral
    signal, which is worth much less than claimed if it is absent;
  * what Class/ClassGUID Windows gives it, since Microsoft documents no NPU
    device setup class.

Paste the Stage A block into the NPUVENDOR.md UNVERIFIED section's place.
"""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "torchnative", "src", "main"))


def stage_a(dump_all: bool) -> int:
    from torchnative.device import _pcivendor
    import torchnative.device as D

    print("\n=== Stage A: selection (nothing executes on any accelerator) ===")
    print(f"host()    = {D.host()!r}")
    print(f"machine() = {D.machine()!r}")
    print(f"candidates, in the order they will be tried: {list(D.npu_candidates())}")

    print("\n-- PCI/PnP enumeration --")
    report = _pcivendor.npu_vendor_report()
    print(f"scanned  = {report['scanned']}")
    print(f"source   = {report['source']}")
    print(f"reason   = {report['reason']}")
    print(f"examined = {report.get('examined')}")
    print(f"found    = {len(report['found'])}")
    for hit in report["found"]:
        print(f"  VEN_{hit['vendor_id']}&DEV_{hit['device_id']}")
        print(f"    vendor     : {hit['vendor']}")
        print(f"    product    : {hit['product']}")
        print(f"    DeviceDesc : {hit['desc']}")
        print(f"    matched by : {', '.join(hit['matched_by'])}")
        print(f"    backend    : {hit['backend'] or 'NONE -- recognised, not targetable from here'}")
        print(f"    key        : {hit['key']}")
    print(f"\nsentence a refusal would carry:\n  {_pcivendor.describe(report)}")

    if dump_all:
        print("\n-- every PCI function, unfiltered (for the UNVERIFIED list) --")
        try:
            entries, _src = _pcivendor.enumerate_pci()
        except _pcivendor.PciScanUnavailable as exc:
            print(f"  unavailable: {exc}")
        else:
            for e in entries:
                print(f"  VEN_{e['vendor_id']}&DEV_{e['device_id']}  "
                      f"class={e['class']!r} service={e['service']!r}")
                print(f"    desc={e['desc']!r}")
                print(f"    compat={e['compatible_ids']}")

    print(
        "\nThis is a statement about SELECTION and about what Windows says is "
        "plugged in. Nothing has run on an NPU; Stage B is the only stage that "
        "is evidence about hardware."
    )
    return 0


def stage_b() -> int:
    import torchnative.device as D

    print("\n=== Stage B: execution (runs the real probes, needs real hardware) ===")
    try:
        res = D.npu.resolve()
    except D.NpuUnresolved as exc:
        print("UNRESOLVED, by name:")
        print(f"  {exc}")
        print(
            "\nThis is Stage B answering, not failing to start. Read whether the "
            "sentence names what is in your machine. If you have an NPU and it "
            "says the machine has none, that is the defect NPUVENDOR.md exists "
            "for -- report it."
        )
        return 0
    print(f"RESOLVED: {res!r}")
    print(f"  {res.as_dict()}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-all", action="store_true",
                    help="print every PCI function, not only the accelerators")
    ap.add_argument("--stage", choices=("a", "b", "both"), default="both")
    args = ap.parse_args(argv)
    rc = 0
    if args.stage in ("a", "both"):
        rc |= stage_a(args.dump_all)
    if args.stage in ("b", "both"):
        rc |= stage_b()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
