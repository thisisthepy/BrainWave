#!/usr/bin/env python3
"""Install the iOS-simulator wheel into a target CPython and prove its torch computes.

    python tools/wheel/verify_ios_sim.py dist/torchnative-*iphonesimulator*.whl

`tools/wheel/verify_android.py` makes the host wheel's judgement -- *`torch.__file__`
must come out of the install location* -- on an Android device. This script makes the
same judgement inside an iOS simulator process. The judgement sentence is identical;
only the way a process is started differs.

Why this needs a launcher at all
--------------------------------

The Android CPython distribution ships `bin/python3.13`, so `verify_android.py` only
has to push a tree and run it. **The iOS distribution ships no executable** -- there is
only `Python.framework/Python`, an `MH_DYLIB` (docs/devices/DEVICE_LOAD_IOS.md). iOS supports
"an app links the framework and calls `Py_Initialize` itself", not "run the interpreter".

So this script compiles one, from three lines of C:

    #include <Python.h>
    int main(int argc, char **argv) { return Py_BytesMain(argc, argv); }

`Py_BytesMain` is CPython's entire CLI entry point and the framework exports it. The
result is a real `python3.13` for the simulator, which means the probe can be run with
the same `-s -P <script>` arguments `verify_android.py` uses -- rather than through
`PyRun_SimpleString`, whose `sys.path` differs from a script run's in exactly the way
that would make the two platforms' measurements incomparable.

It is linked against **UIKit**, and that is not incidental. On iOS, CPython's
`platform.system()` resolves through `_ios_support.get_platform_ios()`, which asks
Objective-C for `UIDevice`. In a bare `simctl spawn` process UIKit is not loaded,
`objc_getClass` returns nil, and `torch/__init__.py`'s `_load_global_deps()` dies on
`platform.system()` before any of this wheel's code runs. A real app always has UIKit
loaded; linking it makes the harness resemble an app instead of making the probe
tiptoe around a hole that only the harness has. `--no-uikit` reproduces the failure.

What "installed" means, precisely, because it is not `pip install`
------------------------------------------------------------------

Same definition as `verify_android.py`, for the same reasons:

  * the archive is unpacked into the target CPython's **site-packages**, not onto a
    `PYTHONPATH` directory -- a `PYTHONPATH` entry is exactly the thing that can shadow
    a broken install.
  * `.data/purelib/` is relocated, as an installer does, so `importlib.metadata` can
    see `torch-<v>.dist-info`.
  * the pure-Python distributions the wheel's own `Requires-Dist` names are staged
    beside it, read from the wheel's METADATA rather than a list written here.
  * the probe runs with `-s -P` from a directory outside the tree, and `torch.__file__`
    is checked to be inside site-packages anyway.

The distribution is **copied** to a scratch prefix first. The one under
`/Volumes/macMini/caches/target-python/` is shared with other work in this repository
and unpacking a wheel into its site-packages would be a side effect on everyone else.

What this cannot answer
-----------------------

The simulator runs on the host M1; it is not an iPhone. See docs/platform/IOS.md §1. Nothing
here measures throughput, and verifying the simulator wheel does not verify the device
wheel -- they differ in Mach-O `LC_BUILD_VERSION.platform` (7 vs 2) and are separate
artefacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The unpack/stage semantics are the judgement, and they must not drift between the
# two device harnesses. Import them rather than restating them.
from verify_android import unpack, stage_dependencies  # noqa: E402

TARGET_PYTHON = Path(os.environ.get(
    "TARGET_PYTHON_IOS_SIM",
    "/Volumes/macMini/caches/target-python/arm64-iphonesimulator"))
SCRATCH = Path(os.environ.get(
    "IOS_SIM_SCRATCH", "/Volumes/macMini/caches/ios-wheel-check"))

LAUNCHER_C = r'''/* A real python3.13 for the iOS simulator.
   Py_BytesMain is CPython's whole CLI entry point, so this binary parses
   -s/-P/-c/<script> exactly as the host python does. UIKit is linked because
   platform.system() on iOS asks Objective-C for UIDevice; see the module docstring. */
#include <Python.h>
int main(int argc, char **argv) { return Py_BytesMain(argc, argv); }
'''

# Runs inside the simulator. One JSON object on the last line. Every failure is caught
# and reported inside it so that a crash and a failed assertion are distinguishable.
PROBE = r'''
import json, os, sys, traceback, types

def install_stubs():
    """The two extension modules the iOS CPython distribution does not build.

    Same shape as verify_android.py's: `_multiprocessing` stays empty because
    `resource_tracker.py` guards it with `hasattr(..., 'sem_unlink')`, while
    `_posixshmem.shm_unlink` is read unguarded and so must exist -- wired to raise,
    so a real use fails loudly instead of silently doing nothing.
    """
    def unavailable(*a, **k):
        raise OSError("shared memory is unavailable on iOS")
    sys.modules["_multiprocessing"] = types.ModuleType("_multiprocessing")
    shm = types.ModuleType("_posixshmem")
    shm.shm_unlink = unavailable
    shm.shm_open = unavailable
    sys.modules["_posixshmem"] = shm

out = {"stubbed": sys.argv[1] == "stubbed",
       "rtld_global": os.environ.get("TORCH_USE_RTLD_GLOBAL"),
       "sys_path": sys.path,
       "platform": sys.platform,
       "py": sys.version.split()[0]}
if out["stubbed"]:
    install_stubs()
try:
    import platform as _p
    out["platform_system"] = _p.system()
    # The host kernel is visible through this field even though system/release are
    # the simulator's. docs/platform/IOS.md §4 keeps the evidence.
    out["uname_version"] = _p.uname().version
except Exception as exc:
    out["platform_system"] = "<%s: %s>" % (type(exc).__name__, exc)
try:
    import torch
    out["torch_file"] = torch.__file__
    out["torch_version"] = torch.__version__
    out["C_file"] = sys.modules["torch._C"].__file__
    out["C_names"] = len(dir(sys.modules["torch._C"]))
    out["aten_ops"] = len(dir(torch.ops.aten))
    x = torch.ones(3, 4)
    y = torch.ones(4, 2)
    out["mm"] = torch.ops.aten.mm.default(x, y).tolist()
    out["add"] = (x + x).tolist()[0]
    import torch.nn as nn
    m = nn.Linear(4, 3)
    z = m(torch.ones(2, 4))
    out["linear_shape"] = list(z.shape)
    out["linear_dtype"] = str(z.dtype)
    import importlib.metadata as md
    try:
        out["metadata_version_torch"] = md.version("torch")
    except Exception as exc:
        out["metadata_version_torch"] = "<%s: %s>" % (type(exc).__name__, exc)
    out["ok"] = True
except BaseException as exc:
    out["ok"] = False
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
    out["traceback"] = traceback.format_exc().splitlines()[-8:]

print("BW_JSON " + json.dumps(out))
'''


def sh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(args, capture_output=True, text=True)
    if check and proc.returncode != 0:
        sys.exit(f"{' '.join(args)} failed:\n{proc.stdout}{proc.stderr}")
    return proc


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    listed = json.loads(sh("xcrun", "simctl", "list", "devices", "available",
                           "--json").stdout)
    for runtime, devices in sorted(listed["devices"].items()):
        if "iOS" not in runtime:
            continue
        for device in devices:
            if device.get("isAvailable") and "iPhone" in device["name"]:
                print(f"+ device {device['name']} ({runtime.split('.')[-1]}) "
                      f"{device['udid']}")
                return device["udid"]
    sys.exit("no available iPhone simulator; pass --udid")


def build_launcher(scratch: Path, uikit: bool) -> Path:
    """Compile a simulator-platform python3.13 against the distribution's framework.

    The distribution's own `bin/arm64-apple-ios-simulator-clang` wrapper is used rather
    than a hand-written `-target` line, so the deployment target and SDK are whatever
    the distribution was built to expect.
    """
    source = scratch / "launcher.c"
    source.write_text(LAUNCHER_C)
    out = scratch / ("python3.13" if uikit else "python3.13.nouikit")
    prefix = scratch / "prefix"
    frameworks = ["-framework", "Python"] + (["-framework", "UIKit"] if uikit else [])
    sh(str(prefix / "bin" / "arm64-apple-ios-simulator-clang"),
       f"-I{prefix}/include/python3.13", f"-F{prefix}", *frameworks,
       "-Wl,-rpath," + str(prefix), "-o", str(out), str(source))
    build = sh("xcrun", "vtool", "-show-build-version", str(out)).stdout
    if "IOSSIMULATOR" not in build:
        sys.exit(f"launcher is not a simulator binary:\n{build}")
    return out


def spawn(udid: str, launcher: Path, prefix: Path, argv: list[str],
          env: dict[str, str]) -> str:
    """Run the launcher inside the simulator.

    `simctl spawn` fills in DYLD_ROOT_PATH -- a binary linked for the simulator
    platform refuses to run as a plain host process without it -- and forwards only
    variables prefixed `SIMCTL_CHILD_`.
    """
    environment = dict(os.environ)
    environment["SIMCTL_CHILD_PYTHONHOME"] = str(prefix)
    for key, value in env.items():
        environment["SIMCTL_CHILD_" + key] = value
    proc = subprocess.run(
        ["xcrun", "simctl", "spawn", udid, str(launcher), "-s", "-P", *argv],
        capture_output=True, text=True, env=environment, cwd=tempfile.gettempdir())
    return proc.stdout + proc.stderr


def read_marker(output: str, label: str) -> dict:
    line = next((ln for ln in output.splitlines() if ln.startswith("BW_JSON ")), None)
    if line is None:
        print(output.strip()[-2000:])
        sys.exit(f"the {label} run printed no result marker -- the interpreter did "
                 "not reach the end of the probe")
    return json.loads(line[len("BW_JSON "):])


def report(result: dict) -> None:
    if result["ok"]:
        print(f"  torch_file   {result['torch_file']}")
        print(f"  C_file       {result['C_file']}")
        print(f"  C_names      {result['C_names']}   aten_ops {result['aten_ops']}")
        print(f"  aten.mm      {result['mm']}")
        print(f"  x + x        {result['add']}")
        print(f"  nn.Linear    {result['linear_shape']} {result['linear_dtype']}")
        print(f"  metadata     torch {result['metadata_version_torch']}")
    else:
        print(f"  {result['error']}")
        for frame in result.get("traceback", []):
            print(f"    {frame}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--udid", default=os.environ.get("IOS_SIMULATOR_UDID"))
    ap.add_argument("--skip-stage", action="store_true",
                    help="reuse the scratch prefix already staged")
    args = ap.parse_args()

    if not args.wheel.exists():
        sys.exit(f"no such wheel: {args.wheel}")
    if "iphonesimulator" not in args.wheel.name:
        sys.exit(f"{args.wheel.name} is not an iOS-simulator wheel. The device wheel "
                 "is a different artefact and this harness cannot run it.")

    prefix = SCRATCH / "prefix"
    site = prefix / "lib" / "python3.13" / "site-packages"

    if not args.skip_stage:
        if not TARGET_PYTHON.exists():
            sys.exit(f"no simulator CPython at {TARGET_PYTHON}")
        print(f"+ copying {TARGET_PYTHON.name} -> {prefix}")
        if SCRATCH.exists():
            shutil.rmtree(SCRATCH)
        SCRATCH.mkdir(parents=True)
        shutil.copytree(TARGET_PYTHON, prefix, symlinks=True)

        print(f"+ unpacking {args.wheel.name} into site-packages")
        site.mkdir(parents=True, exist_ok=True)
        unpack(args.wheel, site)
        deps = stage_dependencies(site, args.wheel)
        print(f"  + {len(deps)} dependencies: {', '.join(deps)}")

        # Nothing host-native may reach the simulator. A Mach-O that arrived with a
        # dependency would be an x86_64/macOS object and its failure would look like a
        # wheel defect.
        allowed = {"_C.abi3.so", "libtorch_global_deps.so"}
        strays = [p for p in site.rglob("*")
                  if p.suffix in (".so", ".dylib", ".pyd") and p.name not in allowed]
        if strays:
            sys.exit(f"refusing to stage {len(strays)} host-native artefact(s): "
                     f"{[str(p) for p in strays[:5]]}")

    print("+ building a simulator python3.13 (Py_BytesMain + UIKit)")
    launcher = build_launcher(SCRATCH, uikit=True)

    udid = pick_device(args.udid)
    booted = json.loads(sh("xcrun", "simctl", "list", "devices", "booted",
                           "--json").stdout)
    already_up = any(d["udid"] == udid
                     for ds in booted["devices"].values() for d in ds)
    if not already_up:
        print(f"+ booting {udid}")
        sh("xcrun", "simctl", "boot", udid)

    try:
        results: dict[str, dict] = {}
        for label, mode, env in [("bare", "bare", {}),
                                 ("stubbed", "stubbed", {}),
                                 ("stubbed + TORCH_USE_RTLD_GLOBAL=1", "stubbed",
                                  {"TORCH_USE_RTLD_GLOBAL": "1"})]:
            print(f"\n+ {label}")
            probe = SCRATCH / "probe.py"
            probe.write_text(PROBE)
            results[label] = read_marker(
                spawn(udid, launcher, prefix, [str(probe), mode], env), label)
            report(results[label])

        print("\n+ sys.path")
        for entry in results["stubbed"]["sys_path"]:
            print(f"    {entry}")

        problems: list[str] = []
        plain = results["stubbed"]

        # The judgement. Everything else is context.
        if not plain["ok"]:
            problems.append(f"import torch failed in the simulator: {plain.get('error')}")
        else:
            if not plain["torch_file"].startswith(str(site) + "/"):
                problems.append(
                    f"torch.__file__ is {plain['torch_file']}, not under {site} -- the "
                    "simulator imported something that did not come from the wheel")
            if not plain["C_file"].startswith(str(site) + "/"):
                problems.append(f"torch._C came from {plain['C_file']}")
            if not plain["C_file"].endswith("_C.abi3.so"):
                problems.append(f"torch._C is {plain['C_file']}, not our extension")
            if plain["mm"] != [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]:
                problems.append(f"aten.mm.default gave {plain['mm']}")
            if plain["platform"] != "ios":
                problems.append(f"sys.platform is {plain['platform']!r} -- this did "
                                "not run on an iOS interpreter")
            # A tree from the repository on sys.path would make the whole run vacuous.
            for entry in plain["sys_path"]:
                if entry and Path(entry).is_relative_to(REPO):
                    problems.append(f"sys.path contains the development tree: {entry}")

        bare = results["bare"]
        print()
        print("  _multiprocessing stub  "
              + ("still required -- bare run: " + str(bare.get("error"))
                 if not bare["ok"] else
                 "NOT required any more -- the bare run imported torch"))
        print("  TORCH_USE_RTLD_GLOBAL  "
              + ("not needed: the wheel's torch/lib/libtorch_global_deps.so satisfies "
                 "_load_global_deps()" if plain["ok"] else
                 "still needed -- see the failure above"))
        print(f"  platform.system()      {plain.get('platform_system')}")
        print(f"  uname().version        {plain.get('uname_version')}")
        print("                         ^ the HOST kernel. The simulator is not an "
              "iPhone; see docs/platform/IOS.md §1")

        if problems:
            print()
            for problem in problems:
                print(f"FAIL: {problem}", file=sys.stderr)
            sys.exit(1)
        print()
        print(f"PASS -- {args.wheel.name} unpacks into an iOS CPython's site-packages "
              "and its torch computes in the simulator")
        print("        This says nothing about the device wheel "
              "(ios_*_arm64_iphoneos): different Mach-O platform, separate artefact.")
    finally:
        if not already_up:
            print(f"+ shutting down {udid}")
            sh("xcrun", "simctl", "shutdown", udid, check=False)


if __name__ == "__main__":
    main()
