#!/usr/bin/env python3
"""Install the Android wheel on a device and prove the torch inside it computes.

    ANDROID_SERIAL=emulator-5554 \
        python tools/wheel/verify_android.py dist/torchnative-*android*.whl

`tools/wheel/verify.py` judges the host wheel by installing it into a clean venv
and checking that `torch.__file__` comes out of that venv. Nothing about that
judgement needs a host; it needs an *interpreter the wheel is for*. On Android
there is one -- on the device -- so the same judgement is available, and this
script makes it rather than settling for `verify_cross.py`'s artefact
inspection.

What "installed" means here, precisely, because it is not `pip install`:

  * the archive is unpacked onto the device's CPython **site-packages**, at
    `<root>/lib/python3.13/site-packages`, which is where an installer would put
    it. Not onto a `PYTHONPATH` directory -- every earlier Android measurement
    in this repository (docs/DEVICE.md) used `PYTHONPATH=$ROOT/site`, and a
    `PYTHONPATH` entry is exactly the thing that can shadow a broken install.
  * `.data/purelib/` is relocated into site-packages, as an installer does with
    that PEP 427 directory. That is how upstream's `torch-<v>.dist-info` gets to
    where `importlib.metadata` can see it.
  * the pure-Python distributions the wheel's own `Requires-Dist` names are
    unpacked beside it, because that is what pip would resolve. They come from
    the host spike venv; every one of them is `py3-none-any` upstream.
  * the probe runs with `-s -P` and from a directory outside the tree, so the
    interpreter adds neither the user site nor the working directory to
    `sys.path`. (`-I` would be the closer analogue of verify.py, but it implies
    `-E`, which discards `PYTHONHOME` and leaves the staged runtime unable to
    find its own stdlib.)
  * `torch.__file__` is then checked to be inside site-packages anyway, rather
    than trusting either precaution.

Two things are measured that the host wheel cannot measure:

  1. **without `TORCH_USE_RTLD_GLOBAL=1`.** docs/DEVICE.md records that variable
     as *required* on Android -- but that was measured against a staged tree
     with no `torch/lib/libtorch_global_deps.so` in it. The wheel ships one, and
     shipping it is the entire argument of docs/WHEEL.md §3.2. Either the
     ordinary branch works now or the file is decoration.
  2. **with and without the `_multiprocessing` stub.** The Android CPython
     distribution builds neither `_multiprocessing` nor `_posixshmem`, and
     `torch/multiprocessing/__init__.py` imports the first at import time. That
     is a property of the interpreter, not of the wheel, so the run is done both
     ways and both results are reported instead of one of them being arranged
     away.

Nothing here installs an app or writes outside `/data/local/tmp/bw_wheel`.
`ANDROID_SERIAL` must be set explicitly; with two emulators routinely up on this
machine, guessing is how the wrong one gets written to.
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

ADB = os.environ.get("ADB") or str(
    Path.home() / "Library/Android/sdk/platform-tools/adb")
TARGET_PYTHON = Path(os.environ.get(
    "TARGET_PYTHON",
    "/Volumes/macMini/caches/target-python/aarch64-linux-android/prefix"))
SPIKE_SITE = Path(os.environ.get(
    "SPIKE_SITE",
    "/Volumes/macMini/caches/spike-venv/lib/python3.13/site-packages"))
DEVICE_ROOT = os.environ.get("DEVICE_ROOT", "/data/local/tmp/bw_wheel")

# Runs on the device. One JSON object on the last line; every failure is caught
# and reported inside it, because `adb shell`'s own exit code is not
# trustworthy (scripts/device_android.sh says the same, and prints a marker for
# the same reason).
PROBE = r'''
import json, os, sys, traceback, types

def install_stubs():
    """The two extension modules Android's CPython does not build.

    Copied in intent from scripts/device_parity.py: `_multiprocessing` stays
    empty because `resource_tracker.py:49` guards it with `hasattr(...,
    'sem_unlink')`, while `_posixshmem.shm_unlink` is read unguarded at line 54
    and so must exist -- wired to raise, so a real use fails loudly.
    """
    def unavailable(*a, **k):
        raise OSError("shared memory is unavailable on Android")
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
    try:
        import torchnative
        out["torchnative"] = torchnative.__file__
    except Exception as exc:
        out["torchnative"] = "<%s: %s>" % (type(exc).__name__, exc)
    out["ok"] = True
except BaseException as exc:
    out["ok"] = False
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
    out["traceback"] = traceback.format_exc().splitlines()[-6:]

print("BW_JSON " + json.dumps(out))
'''


def adb(*args: str, serial: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([ADB, "-s", serial, *args],
                          capture_output=True, text=True)
    if check and proc.returncode != 0:
        sys.exit(f"adb {' '.join(args)} failed:\n{proc.stdout}{proc.stderr}")
    return proc


def device_python(serial: str, argv: str, env: dict[str, str]) -> str:
    """Run the staged interpreter and return its combined output.

    `adb shell` has returned 0 for failing device commands, so nothing here
    reads its exit status; the probe's own marker line is the judgement.
    """
    assignments = " ".join(f"{k}={v}" for k, v in env.items())
    command = (
        f"cd {DEVICE_ROOT} && "
        f"unset PYTHONPATH; {assignments} "
        f"LD_LIBRARY_PATH={DEVICE_ROOT}/lib PYTHONHOME={DEVICE_ROOT} "
        f"./bin/python3.13 -s -P {argv} 2>&1"
    )
    return adb("shell", command, serial=serial).stdout


def unpack(wheel: Path, into: Path) -> None:
    """Unpack as an installer would: root to site-packages, `.data/purelib` too.

    Not `pip install`: there is no pip for this interpreter on this machine, and
    a device-side pip would need the network. This is the subset of PEP 427
    install behaviour the wheel actually uses -- extract, then relocate the one
    `.data/` scheme directory present.
    """
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(into)
    for data_dir in list(into.glob("*.data")):
        for scheme in list(data_dir.iterdir()):
            if scheme.name not in ("purelib", "platlib"):
                print(f"  ! ignoring unhandled .data scheme {scheme.name}")
                continue
            for entry in list(scheme.iterdir()):
                shutil.move(str(entry), str(into / entry.name))
        shutil.rmtree(data_dir)


def stage_dependencies(into: Path, wheel: Path) -> list[str]:
    """The pure-Python distributions pip would resolve for this wheel.

    Read from the wheel's own METADATA rather than from a list written here, so
    that adding a requirement to pyproject.toml cannot leave this staging behind
    -- the device would then import a torch missing one of its dependencies and
    the failure would look like a wheel defect.
    """
    with zipfile.ZipFile(wheel) as zf:
        meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        requires = [
            line.split(": ", 1)[1]
            for line in zf.read(meta).decode().splitlines()
            # Extras are not installed by `pip install <wheel>`, and staging one
            # here would be worse than useless: this distribution's `test` extra
            # is upstream **torch**, whose tree would land on top of the one the
            # wheel just delivered and make the whole run meaningless.
            if line.startswith("Requires-Dist: ") and "extra ==" not in line
        ]
    # Distribution name -> the top-level names it installs. Only the ones this
    # wheel requires, plus what those require in turn (`sympy` needs `mpmath`,
    # `jinja2` needs `markupsafe`) and `torchgen`'s `yaml`.
    modules = {
        "filelock": ["filelock"],
        "typing-extensions": ["typing_extensions.py"],
        "setuptools": ["setuptools", "pkg_resources"],
        "sympy": ["sympy", "mpmath"],
        "networkx": ["networkx"],
        "jinja2": ["jinja2", "markupsafe"],
        "fsspec": ["fsspec"],
    }
    staged: list[str] = []
    wanted = ["packaging", "yaml"]  # transitive: torchgen reads YAML
    for requirement in requires:
        name = requirement.split(";")[0].split("=")[0].split(">")[0]
        name = name.split("<")[0].split("[")[0].strip().lower()
        wanted += modules.get(name, [name])
    for entry in dict.fromkeys(wanted):
        source = SPIKE_SITE / entry
        if not source.exists():
            # Loudly, and fatally. This used to warn and carry on, and the run
            # then died much later inside `import torch` with a bare
            # ModuleNotFoundError -- which reads like a wheel defect and is
            # not one. The dependency list comes from the wheel's own METADATA,
            # so a name missing here means the staging source is wrong, not
            # that the requirement is optional.
            raise SystemExit(
                f"no {entry} under {SPIKE_SITE} -- it is required by the "
                f"wheel's METADATA. Set SPIKE_SITE to a site-packages "
                f"directory that has it.")
        target = into / entry
        if target.exists():
            # The wheel already provides this name. Overwriting it with the
            # host's copy is how a run stops being about the wheel.
            print(f"  ! {entry} already comes from the wheel -- not overwriting")
            continue
        if source.is_dir():
            shutil.copytree(source, target,
                            ignore=shutil.ignore_patterns(
                                "__pycache__", "*.so", "*.dylib", "*.pyd"))
        else:
            shutil.copy2(source, target)
        staged.append(entry)
    return staged


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--skip-stage", action="store_true",
                    help="reuse what is already on the device (the push is "
                         "slow; only safe if the wheel has not changed)")
    args = ap.parse_args()

    if not args.wheel.exists():
        sys.exit(f"no such wheel: {args.wheel}")
    if "android" not in args.wheel.name:
        sys.exit(f"{args.wheel.name} is not an Android wheel")

    serial = os.environ.get("ANDROID_SERIAL")
    if not serial:
        listed = subprocess.run([ADB, "devices"], capture_output=True, text=True)
        attached = [line.split()[0] for line in listed.stdout.splitlines()[1:]
                    if line.strip().endswith("device")]
        sys.exit(
            "set ANDROID_SERIAL explicitly -- this machine routinely has two "
            f"emulators up and this script will not choose. Attached: {attached}"
        )

    site = f"{DEVICE_ROOT}/lib/python3.13/site-packages"

    if not args.skip_stage:
        print(f"+ staging CPython runtime at {DEVICE_ROOT} on {serial}")
        adb("shell", f"rm -rf {site} && mkdir -p {DEVICE_ROOT}/lib "
                     f"{DEVICE_ROOT}/bin", serial=serial)
        adb("push", "--sync", str(TARGET_PYTHON / "bin" / "python3.13"),
            f"{DEVICE_ROOT}/bin/", serial=serial)
        adb("push", "--sync", str(TARGET_PYTHON / "lib" / "libpython3.13.so"),
            f"{DEVICE_ROOT}/lib/", serial=serial)
        adb("push", "--sync", str(TARGET_PYTHON / "lib" / "python3.13"),
            f"{DEVICE_ROOT}/lib/", serial=serial)

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "site-packages"
            staging.mkdir()
            unpack(args.wheel, staging)
            deps = stage_dependencies(staging, args.wheel)
            print(f"+ unpacked {args.wheel.name} + {len(deps)} dependencies: "
                  f"{', '.join(deps)}")

            # Nothing host-native may reach the device. The wheel's own
            # `_C.abi3.so` and global-deps library are the only binaries that
            # belong here, and verify_cross.py has already established what they
            # are; anything else is a Mach-O that came in with a dependency.
            allowed = {"_C.abi3.so", "libtorch_global_deps.so"}
            strays = [p for p in staging.rglob("*")
                      if p.suffix in (".so", ".dylib", ".pyd")
                      and p.name not in allowed]
            if strays:
                sys.exit(f"refusing to stage {len(strays)} host-native "
                         f"artefact(s): {[str(p) for p in strays[:5]]}")

            print(f"+ pushing to {site}")
            adb("push", "--sync", str(staging), f"{DEVICE_ROOT}/lib/python3.13/",
                serial=serial)

    listing = adb("shell", f"ls -l {site}/torch/_C.abi3.so "
                           f"{site}/torch/lib/libtorch_global_deps.so; "
                           f"du -sh {DEVICE_ROOT}", serial=serial).stdout
    print(listing.strip())

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(PROBE)
        probe_path = fh.name
    adb("push", probe_path, f"{DEVICE_ROOT}/bw_probe.py", serial=serial)
    os.unlink(probe_path)

    runs = [
        ("bare", "bare", {}),
        ("stubbed", "stubbed", {}),
        ("stubbed + TORCH_USE_RTLD_GLOBAL=1", "stubbed",
         {"TORCH_USE_RTLD_GLOBAL": "1"}),
    ]
    results: dict[str, dict] = {}
    for label, mode, env in runs:
        print(f"\n+ {label}")
        output = device_python(serial, f"bw_probe.py {mode}", env)
        line = next((ln for ln in output.splitlines()
                     if ln.startswith("BW_JSON ")), None)
        if line is None:
            print(output.strip()[-2000:])
            sys.exit(f"the {label} run printed no result marker at all -- the "
                     "interpreter did not reach the end of the probe")
        results[label] = json.loads(line[len("BW_JSON "):])
        r = results[label]
        if r["ok"]:
            print(f"  torch_file   {r['torch_file']}")
            print(f"  C_file       {r['C_file']}")
            print(f"  C_names      {r['C_names']}   aten_ops {r['aten_ops']}")
            print(f"  aten.mm      {r['mm']}")
            print(f"  x + x        {r['add']}")
            print(f"  nn.Linear    {r['linear_shape']} {r['linear_dtype']}")
            print(f"  metadata     torch {r['metadata_version_torch']}")
        else:
            print(f"  {r['error']}")
            for frame in r.get("traceback", []):
                print(f"    {frame}")

    print()
    problems: list[str] = []

    # The judgement. Everything else above is context.
    plain = results["stubbed"]
    if not plain["ok"]:
        problems.append(
            f"import torch failed on the device without TORCH_USE_RTLD_GLOBAL: "
            f"{plain.get('error')}")
    else:
        if not plain["torch_file"].startswith(site + "/"):
            problems.append(
                f"torch.__file__ is {plain['torch_file']}, not under {site} -- "
                "the device imported something that did not come from the wheel")
        if not plain["C_file"].startswith(site + "/"):
            problems.append(f"torch._C came from {plain['C_file']}")
        if not plain["C_file"].endswith("_C.abi3.so"):
            problems.append(f"torch._C is {plain['C_file']}, not our extension")
        if plain["mm"] != [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]:
            problems.append(f"aten.mm.default gave {plain['mm']}")
        if plain["platform"] != "android":
            problems.append(
                f"sys.platform is {plain['platform']!r} -- this did not run on "
                "an Android interpreter")

    # Not a failure, but the thing docs/DEVICE.md §4 will have to be corrected
    # for, so it has to be reported either way rather than inferred.
    bare = results["bare"]
    print("  _multiprocessing stub  "
          + ("still required -- bare run: " + str(bare.get("error"))
             if not bare["ok"] else
             "NOT required any more -- the bare run imported torch"))
    print("  TORCH_USE_RTLD_GLOBAL  "
          + ("not needed: the wheel's torch/lib/libtorch_global_deps.so "
             "satisfies _load_global_deps()"
             if plain["ok"] else "still needed -- see the failure above"))

    if problems:
        print()
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        sys.exit(1)
    print()
    print(f"PASS -- {args.wheel.name} unpacks into an Android CPython's "
          "site-packages and its torch computes on the device")


if __name__ == "__main__":
    main()
