# NPUVENDOR — the operating system is not the silicon vendor

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/__init__.py NPU_CANDIDATES present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/__init__.py npu_candidates present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/_pcivendor.py npu_vendor_report present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/_pcivendor.py SOURCED_NPU_DEVICES present -->

`torchnative/src/main/torchnative/device/__init__.py`,
`torchnative/src/main/torchnative/device/_pcivendor.py`.
Tests: `rust/torch_c/pytests/test_npuvendor.py`, `rust/torch_c/pytests/test_devicens.py`.
Hardware script for the one machine that can settle this: `tools/devices/npuvendor_verify.py`.

## 0. The defect

`device/__init__.py` resolved the NPU vendor from `platform.system()`:

```python
NPU_BACKENDS = {
    "darwin":  ("coreml",   "Apple Neural Engine"),
    "windows": ("openvino", "Intel NPU"),
    "android": ("qnn",      "Qualcomm Hexagon NPU"),
}
```

The OS is not a proxy for the silicon vendor and this project's own wheel list is
the counter-example — we ship `win_amd64` **and** `win_arm64`. On Windows alone:

| Wheel | Machine | NPU | Reached by |
|---|---|---|---|
| `win_amd64` | Intel Core Ultra | Intel NPU | OpenVINO's `NPU` device |
| `win_amd64` | AMD Ryzen AI | XDNA | **nothing in this project** |
| `win_arm64` | Snapdragon X Elite | Hexagon | QNN — never OpenVINO (§1) |
| `win_*` | no NPU | — | must refuse honestly |

What a Ryzen AI owner got: `openvino` is pip-installable on any x86-64 Windows,
so it imported, `available_devices()` did not list `NPU`, and the refusal read
**"OpenVINO lists ['CPU', 'GPU'] with no NPU among them"** under a heading that
had already declared the unit to be the *Intel NPU* — i.e. "this machine has no
Intel NPU", on a machine that has an NPU.

This is `docs/devices/QNN.md` §5's defect run backwards. That one **claimed
hardware that was not reachable** (an `SM8550` name string became a confident
"Qualcomm Hexagon NPU" on a device with no compute-DSP endpoint). This one
**denies hardware that is present**. Both come from reading a name instead of
probing, and both are fixed the same way: name what was measured.

## 1. Settled first: OpenVINO cannot reach an AMD NPU

This was a *belief* of the coordinating session before this round and is now
traced to source, because if it were false most of this round would change shape.

**The trace.** `src/plugins/intel_npu/src/utils/src/zero/zero_init.cpp` in
openvinotoolkit/openvino (`master`, fetched 2026-09-10) is what binds the plugin
to a driver:

```cpp
constexpr ze_driver_uuid_t uuid = ze_intel_npu_driver_uuid;          // line 31
...
zeDriverGetProperties(all_drivers[i], &_driver_properties);           // line 74
if (memcmp(&_driver_properties.uuid, &uuid, sizeof(uuid)) == 0) {     // line 75
    _driver_handle = all_drivers[i];
    break;
}
...
if (_driver_handle == nullptr) {
    OPENVINO_THROW("NPU driver wasn't found!");
}
```

`ze_intel_npu_driver_uuid` is a compile-time constant, defined in
intel/level-zero-npu-extensions `ze_intel_npu_uuid.h`:

```c
#define ze_intel_npu_driver_uuid { 0x01, 0x7d, 0xe9, 0x31, 0x6b, 0x4d, 0x4f, 0xd4, \
                                   0xaa, 0x9b, 0x5b, 0xed, 0x77, 0xfc, 0x8e, 0x89 }
```

So the plugin does not enumerate "an NPU". It enumerates Level Zero drivers and
takes the one whose UUID is byte-identical to Intel's own, and throws otherwise.
The enumeration itself is also narrowed to Intel's stack: the fast path passes
`ZE_INIT_DRIVER_TYPE_FLAG_NPU` to `zeInitDrivers`, and the fallback calls
`zeInit(ZE_INIT_FLAG_VPU_ONLY)` — both Level Zero, which is Intel's API and is
not what AMD's XDNA driver exposes on Windows.

**Corroborated in the shipped binary**, not only in the source. In the wheel
`openvino-2026.3.1-22476-cp313-cp313-win_amd64.whl`
(`openvino/libs/openvino_intel_npu_plugin.dll`):

* those sixteen UUID bytes appear **exactly once** (and zero times in
  `openvino.dll`);
* the string `NPU driver wasn't found!` is present;
* its PE import table names `ze_loader.dll`, and it references ~60 `ze*` entry
  points (`zeDriverGet`, `zeCommandQueueCreate`, …);
* `strings` over **every** DLL in `openvino/libs/` matches `xdna`, `ryzen` or
  `vitis` a total of **zero** times;
* the only NPU plugin shipped is `openvino_intel_npu_plugin.dll` — there is no
  second vendor plugin to fall through to.

**Answer: no.** OpenVINO's `NPU` device is Intel's Level Zero NPU and nothing
else. `openvino` being importable on a Ryzen AI machine is therefore not weak
evidence about AMD; it is no evidence at all, which is exactly how it produced a
confident wrong refusal.

**A second, independent consequence for Snapdragon.** PyPI's `openvino`
2026.3.1 publishes four platform tags: `macosx_11_0_arm64`,
`manylinux_2_28_x86_64`, `manylinux_2_35_aarch64`, `win_amd64`
(<https://pypi.org/pypi/openvino/json>, fetched 2026-09-10). There is **no
`win_arm64` wheel**, so on the Snapdragon X machine we ship a `win_arm64` wheel
to, OpenVINO cannot even be imported — and it was that machine the old table
told it had an "Intel NPU".

## 2. What this round does and does not do

**Does:** turns the table into ordered *candidates* and lets a probe decide;
recognises the AMD and Qualcomm cases by name; makes every refusal say what the
machine actually has, or say plainly that it could not look.

**Does not:** invent an AMD execution path. There is no XDNA/Vitis code here and
`_pcivendor.VENDOR_BACKENDS` maps `1022` to `None` explicitly rather than leaving
a gap. Recognising a vendor we cannot target is the deliverable; targeting it is
not.

## 3. The detection investigation

The question: **can a pure-Python process identify an NPU on Windows without a
vendor SDK?** Four approaches were considered.

| Approach | Cost | Verdict |
|---|---|---|
| **`winreg` over `HKLM\SYSTEM\CurrentControlSet\Enum\PCI`** | stdlib, in-process, no dependency, no subprocess, no COM | **Chosen.** Key names carry `VEN_`/`DEV_`; each instance carries `DeviceDesc`, `Class`, `Service`, `HardwareID`, `CompatibleIDs` |
| `ctypes` + SetupAPI (`SetupDiGetClassDevs`/`SetupDiGetDeviceRegistryProperty`) | ~150 lines of struct/ABI declarations, 32/64-bit `DEVPROPKEY` handling, and a wrong `ctypes` signature is a crash, not an exception | Rejected: same data as the registry at much higher risk |
| WMI via COM (`Win32_PnPEntity`) | needs `pywin32`, or a `powershell.exe` subprocess (~1s, and a new failure mode when PowerShell is restricted) | Rejected: a dependency or a subprocess for data the registry already has |
| `platform.processor()` / `PROCESSOR_IDENTIFIER` | free | Rejected as the *primary* signal: it names the **CPU** vendor, not whether an NPU is present. `AuthenticAMD` on a Ryzen 5000 says nothing about an NPU |

`platform.machine()` is still used, but only to **order** the candidates
(§4) — never to decide the answer.

### 3.1 What is matched, and from which source

Two signals, kept separate in the report because they are different strengths.

**(a) Class code — vendor-neutral, the primary signal.** PCI base class `12h`
subclass `00h` is *Processing accelerators*. Source: the `pci.ids` v2.2 class
section, `C 12  Processing accelerators` / `00  Processing accelerators`
(<https://pci-ids.ucw.cz/v2.2/pci.ids>, fetched 2026-09-10 — the repository
Microsoft's own page links to). Windows reports this as a compatible ID
`PCI\CC_c(2)s(2)`, i.e. `PCI\CC_1200`; the compatible-ID formats are documented
in *Identifiers for PCI Devices*
(<https://learn.microsoft.com/en-us/windows-hardware/drivers/install/identifiers-for-pci-devices>),
which lists `PCI\VEN_v&CC_cs`, `PCI\CC_csp` and `PCI\CC_cs` among them.

This matters because it is the signal that can catch a vendor **we have never
heard of**, including parts released after this document.

**(b) Device name — a sourced allowlist, the corroborating signal.** Every PCI
function `pci.ids` v2.2 names "NPU" or "Neural Processing Unit", verbatim, with
its line number, reproduced in `_pcivendor.SOURCED_NPU_DEVICES`:

| Vendor | ID | Product (pci.ids wording) | pci.ids line |
|---|---|---|---|
| AMD (`1022`) | `17F0` | Strix/Krackan/Strix Halo Neural Processing Unit | 5710 |
| Intel (`8086`) | `643E` | Core Ultra 200V Series Processors NPU | 39255 |
| Intel (`8086`) | `7D1D` | Core Ultra 200H/200V Series Processors NPU | 39789 |
| Intel (`8086`) | `AD1D` | Core Ultra 200 Series Processors NPU | 41281 |
| Intel (`8086`) | `B03E` | Core Ultra Processors (Series 3) NPU | 41304 |
| Phytium (`1DB7`) | `DC24` | NPU Controller [X100 Series] | 27914 |
| DEEPX (`1FF4`) | `0102` / `0112` / `2001` | M1 / M1M / VPU [H1 V-NPU] | 30227 / 30233 / 30235 |

Intel's **GNA** parts (`4511`, `464F`, `4E11`, `774C`, `7E4C`, `9A11` —
"Gaussian & Neural-Network Accelerator") are deliberately **excluded**. They are a
different, older block from the Core Ultra NPU and OpenVINO shipped a separate
GNA plugin for them; counting a GNA as an NPU would be the claim-what-is-not-
there failure this round is fixing.

A small corroboration that the allowlist is aimed at the right parts: the string
`7d1d` also appears inside the shipped `openvino_intel_npu_plugin.dll`, alongside
`3720` and `4000` (the Meteor Lake / Lunar Lake NPU generation numbers).

### 3.2 What "could not look" must not become

`_pcivendor.npu_vendor_report()` returns `scanned: False` with a named `reason`
when the registry cannot be read at all — non-Windows, no `winreg`, or an
`OSError` opening `Enum\PCI`. `describe()` renders that as a *different sentence*
from "read it and found nothing". A caller that read `found == []` without
reading `scanned` would print "this machine has no NPU" after failing to look,
which is the original defect with a new cause.

## 4. The candidate order, and why

```python
NPU_CANDIDATES = {
    "darwin":  (("coreml", "Apple Neural Engine"),),
    "windows": (("openvino", "Intel NPU"), ("qnn", "Qualcomm Hexagon NPU")),
    "android": (("qnn", "Qualcomm Hexagon NPU"),),
}
```

`npu_candidates(host, machine)` reorders Windows by instruction set: on a
machine string beginning `arm`/`aarch`, Hexagon is tried first; otherwise Intel
is. The justification is that the two are **mutually exclusive by ISA** — an
Intel NPU exists only on x86-64, a Snapdragon X's Hexagon only on arm64 — so this
puts the plausible candidate first at zero risk.

The second is still probed, and that is not decoration. Excluding it would put
the ISA→vendor inference back into the table one level down, and the inference is
unsound in a specific reachable way: `platform.machine()` reports the
**interpreter's** architecture, so an x86-64 Python emulated on an arm64 Windows
box reports `AMD64` on a machine whose only NPU is a Hexagon. Trying both means
that host gets a probe rather than a guess.

`linux` deliberately gains no entry. OpenVINO does ship `manylinux` wheels with
the NPU plugin, but nothing in this project has probed that path and adding it
would be a claim, not a fix.

### 4.1 Windows-on-Snapdragon refuses by name rather than running the wrong probe

The `qnn` candidate on a non-`android` host refuses immediately with a named
reason. This project's only Hexagon probe is
`torchnative.export.qnn_device.device_report`, which shells out to **adb** and
reads `/sys/class/fastrpc` on an attached *Android* device. Neither exists on
Windows-on-Snapdragon, where QNN is reached through the Windows QNN runtime DLLs.
Running the adb probe there would answer about whatever phone happens to be
plugged in — `QNN.md` §5's failure with the cable the other way round.

So the Snapdragon X owner is told: a Hexagon NPU was the first candidate, this
build has no probe that can look for one on Windows, and here is what the PnP
enumeration says is in the machine. Not "you have no Intel NPU".

## 5. What each owner sees now

| Machine | `device.npu.available` | The refusal, in substance |
|---|---|---|
| Intel Core Ultra + OpenVINO NPU listed | `True` | — resolves to `openvino → Intel NPU` |
| **AMD Ryzen AI** | `False` | openvino refused (OpenVINO lists no NPU); qnn refused (no Windows Hexagon probe); **"the PCI/PnP enumeration names Advanced Micro Devices, Inc. [AMD] Strix/… Neural Processing Unit — recognised, and this project has no execution path for it"** |
| **Snapdragon X Elite** | `False` | qnn refused by name (adb-only probe); openvino refused (no `win_arm64` wheel, so the import fails); plus whatever the PnP scan says |
| Windows, no NPU | `False` | both candidates refused, and the scan says it read *n* PCI functions and found no accelerator |
| Windows, registry unreadable | `False` | both candidates refused, and **"this build cannot see whether another vendor's NPU is present"** — not "there is none" |

## 6. UNVERIFIED

This machine is an arm64 Mac. It has no Windows, no Intel NPU, no AMD NPU and no
Snapdragon. **Not one line of the registry path in `_pcivendor.enumerate_pci` has
ever executed.** Precisely:

* **That `HKLM\SYSTEM\CurrentControlSet\Enum\PCI` is readable by a non-elevated
  process.** The code is written to name an `OSError` here rather than return an
  empty list, so a permission failure is reported as "could not look" — but
  whether it *is* a permission failure on a stock Windows 11 install is unknown
  here. *Settled by:* running `tools/devices/npuvendor_verify.py` on any Windows
  machine, elevated and not.
* **That an Intel NPU actually appears under that key with `VEN_8086&DEV_7D1D`
  (or a sibling), and that its `CompatibleIDs` really contains `CC_1200`.** The
  ID list is from `pci.ids`; the *registry spelling* is from Microsoft's
  documented format. Neither has been read off a real machine.
  *Settled by:* the same script, on the user's Intel NPU laptop. This is the
  single most valuable missing measurement, because if the class code is absent
  the vendor-neutral signal (§3.1a) is worth less than claimed.
* **That AMD's NPU presents as class `1200`.** Widely repeated as "NPU Compute
  Accelerator Device"; the *device name* `1022:17F0` is sourced from `pci.ids`,
  the *class code* is not sourced for this part specifically. *Settled by:*
  running the script on a Ryzen AI machine.
* **AMD Phoenix / Hawk Point (Ryzen 7040/8040) NPU device IDs.** `pci.ids` v2.2
  lists only `17F0` (Strix/Krackan/Strix Halo) under `1022`. Earlier Ryzen AI
  parts are therefore covered only by the class-code signal, if at all.
  *Settled by:* a later `pci.ids`, or the script on such a machine.
* **Whether Windows has a device *setup class* for NPUs.** Both Microsoft lists —
  *System-Defined Device Setup Classes Available to Vendors* and *…Reserved for
  System Use* (fetched 2026-09-10) — contain **no** class whose name or
  description mentions Compute, Accelerator, Neural or NPU. A `ClassGUID`-based
  detection therefore has no documented basis and was not built. *Settled by:*
  reading the real `Class`/`ClassGUID` values off an NPU machine, which the
  script prints.
* **Snapdragon-on-Windows Hexagon presence.** Whether the Hexagon appears in
  `Enum\PCI` at all on `win_arm64` (it may be enumerated on ACPI rather than PCI)
  is unknown. If it is not on PCI, `_pcivendor` will honestly report "read *n*
  functions, no accelerator among them" on a machine that has one — a weaker
  answer than for Intel/AMD, and named here rather than discovered later.
* **Everything about actually executing on any NPU.** Unchanged by this round.
  `docs/devices/INTELNPU.md` and `docs/devices/QNN.md` remain the record of what
  has and has not been run on real silicon; nothing here is evidence that an NPU
  was reached.

## 7. Sources used

| Fact | Source | Fetched |
|---|---|---|
| OpenVINO binds its NPU driver by Intel UUID | openvinotoolkit/openvino `src/plugins/intel_npu/src/utils/src/zero/zero_init.cpp` lines 31, 74–75, 80–82 | 2026-09-10 |
| The UUID constant | intel/level-zero-npu-extensions `ze_intel_npu_uuid.h` | 2026-09-10 |
| Only an Intel NPU plugin ships; no XDNA string anywhere | `openvino-2026.3.1-22476-cp313-cp313-win_amd64.whl`, `openvino/libs/*.dll` | local |
| No `win_arm64` OpenVINO wheel | <https://pypi.org/pypi/openvino/json> | 2026-09-10 |
| PCI vendor and device IDs, PCI class 12h | `pci.ids` v2.2, <https://pci-ids.ucw.cz/v2.2/pci.ids> | 2026-09-10 |
| Windows PCI hardware/compatible ID formats incl. `CC_` | Microsoft Learn, *Identifiers for PCI Devices* | 2026-09-10 |
| No NPU device setup class is documented | Microsoft Learn, the two *System-Defined Device Setup Classes* pages | 2026-09-10 |

## 8. Nullification — what each new guarantee is actually held down by

Every guarantee below was broken deliberately and the suite re-run, because a
test that cannot fail is not a test (CLAUDE.md §5.5). Observed red tests:

| # | Break | Went red |
|---|---|---|
| N1 | `NPU_CANDIDATES["windows"]` back to one entry | `..._windows_tries_more_than_one_candidate_in_an_isa_dependent_order`, `..._a_snapdragon_windows_box_is_not_told_it_has_no_intel_npu`, `..._the_refusal_never_tells_a_machine_with_an_npu_that_it_has_none` |
| N2 | ISA reorder deleted from `npu_candidates` | the first two of those |
| N3 | AMD `17F0` removed from `SOURCED_NPU_DEVICES` | `..._an_amd_npu_is_recognised_by_name_and_not_reported_as_an_absence`, `..._the_refusal_never_tells_a_machine_with_an_npu_that_it_has_none` |
| N4 | `describe()` renders "could not look" as "found nothing" | `..._could_not_look_is_a_different_sentence_from_found_nothing` |
| N5 | the `h != "android"` guard removed, so the adb probe runs on Windows | `..._the_hexagon_candidate_refuses_on_windows_rather_than_probing_over_adb` |
| N6 | `NPU_PCI_CLASS_CODES` emptied | `..._an_unknown_vendors_accelerator_is_still_caught_by_the_class_code` |
| N7 | the vendor sentence dropped from the refusal | `..._the_refusal_never_tells_a_machine_with_an_npu_that_it_has_none`, `..._the_refusal_on_a_windows_box_with_no_npu_says_so_without_overclaiming` |

Before the resolver changed, 7 of this file's 15 tests were red against the old
one-backend-per-host table, and two of those failures printed the defect
verbatim: `torchnative.device.npu resolves to the Intel NPU on windows` for a
fake machine whose PCI enumeration named an AMD Neural Processing Unit.

What none of this is: evidence about hardware. §6 stands.
