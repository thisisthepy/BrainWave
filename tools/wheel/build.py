#!/usr/bin/env python3
"""Build a platform wheel that actually contains a torch you can import.

    python tools/wheel/build.py                            # this machine
    python tools/wheel/build.py --target android-arm64-v8a
    python tools/wheel/build.py --target android-x86_64    # refuses; see below
    python tools/wheel/build.py --target ios-arm64
    python tools/wheel/build.py --target ios-arm64-sim
    python tools/wheel/build.py --target linux-x86_64
    python tools/wheel/build.py --target linux-aarch64
    python tools/wheel/build.py --target windows-x86_64
    python tools/wheel/build.py --target windows-arm64

The distribution on PyPI as `torchnative 0.0.1a0` is `py3-none-any` and holds
only the `torchnative/` skeleton: no `_C`, no vendored tree. `pip install
torchnative` gets you something where `import torch` raises ImportError. This
script exists so that the *default* way to build produces the other thing.

What it does beyond `pip wheel .`:

  preflight   Refuse to build unless `vendor/vendor_torch.sh` has laid down the
              upstream Python tree and `vendor/install_shim.sh` has put our `_C`
              in the hole. Both are gitignored, so a fresh clone has neither,
              and setuptools would happily emit the empty shell again -- which
              is how the PyPI one came to exist. The failure has to be loud.

  freshness   Refuse to build unless the `_C` about to be packaged was built
              from the source on disk now, read off cargo's own dep-info. This
              script does not *build* the cross artefacts -- it picks them up
              from `CARGO_TARGET_DIR/<triple>/release/` -- and until 2026-08-29
              nothing looked at their age, so a five-day-old extension shipped
              with an exit code of 0 and passed every check downstream (§
              `artefact_verdict`, docs/WHEEL.md §11).

  repack      Two fixes that have no setuptools spelling, applied to the
              finished archive (§ `_repack`):
                - `torch-<v>.dist-info/` from upstream, so that
                  `importlib.metadata.version("torch")` answers. transformers
                  gates the entire torch integration on that call
                  (docs/VENDOR.md), so it is load-bearing, not decoration.
                - the Mach-O install name of `_C.abi3.so`, which cargo sets to
                  the absolute path of the build machine's target directory.

  verify      Diff the archive against the source tree file-by-file and fail on
              anything missing. A wheel that is merely *smaller* than it should
              be installs fine and breaks later, at some import nobody ran.

Cross targets (`--target`) reuse all of that and swap three things, which are
exactly the three that are platform-shaped (§ `TARGETS`):

  the extension   `torch/_C.abi3.so` comes from the cross build's target
                  directory instead of from the source tree. The source tree
                  keeps the host artefact, so the suites that read it keep
                  measuring the host.
  global deps     built by the target's compiler, and named `.so` rather than
                  `.dylib` on iOS as well as Android -- `_load_global_deps()`
                  picks the extension with `platform.system() == "Darwin"`,
                  which is False on iOS.
  the tag         `android_<api>_<abi>` / `ios_<major>_<minor>_<multiarch>`,
                  derived from the *target CPython distribution* rather than
                  written down here, and cross-checked against what the
                  artefact itself says it needs.

                  `manylinux_<major>_<minor>_<arch>` inverts that, and it is the
                  one case where the distribution is the wrong source: glibc
                  compatibility is not a property of the interpreter build, so
                  the Linux `_sysconfigdata_*.py` has no field for it. The floor
                  comes from the artefact's own `.gnu.version_r`, which is where
                  auditwheel reads it (§ `LinuxTarget`, docs/LINUX.md §5).

                  `win_amd64` / `win_arm64` are the odd pair: no floor exists to
                  compute, so the tag is a *name*. It is still not written down
                  here -- § `WindowsTarget` derives it from the target
                  distribution's own `sysconfig.get_platform()` branch, because
                  a name reached by analogy with the other one is a guess, and
                  `win_aarch64` is exactly as plausible as `win_arm64` to
                  everybody except pip.

Each non-Apple platform carries two architectures (docs/WHEELMATRIX.md). One of
the nine targets **refuses**: `--target android-x86_64`, because no
x86_64-linux-android CPython exists to derive a tag from and none can be
obtained here -- § `_ANDROID_X86_64_REFUSAL` has the whole reason. It is listed
rather than omitted so the refusal names what is missing, instead of the target
looking like one nobody thought about; `--target linux-x86_64` was in that state
until docs/LINUX.md §9 made cargo-zigbuild work, and is now built and executed.

Then check it for real -- building is not the proof:

    python tools/wheel/verify.py dist/<host wheel>        # installs it
    python tools/wheel/verify_cross.py dist/<cross wheel> # inspects it
    python tools/wheel/verify_android.py dist/<android wheel>    # imports it
    python tools/wheel/verify_ios_sim.py dist/<ios sim wheel>    # imports it
    python tools/wheel/verify_ios_device.py dist/<ios device wheel>  # links it

And check this script itself, which builds nothing and takes a second:

    python tools/wheel/build.py --self-test
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binfmt import (describe, elf_dynamic, elf_info, macho_arches,  # noqa: E402
                    pe_imports, pe_info,
                    macho_info, wasm_info)

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "torchnative" / "src" / "main"

# The two artefacts the vendor scripts produce. Their absence is the whole
# failure mode this script exists to catch, so they are named individually
# rather than checked as "does src/main look populated".
VENDORED_ROOT = SRC / "torch" / "__init__.py"
SHIM = SRC / "torch" / "_C.abi3.so"
STAMP = SRC / ".stamp"

SKIP_SUFFIXES = (".pyc", ".pyo")
SKIP_DIRS = {"__pycache__"}
#: Finder metadata. Excluded in pyproject too, but that only reaches files
#: *inside* a package -- the copy at the package root lands at the archive
#: root, which setuptools' package-data globs never see. Named here so the
#: preflight can refuse rather than ship one, as it did in 0.0.2a0's first
#: build.
SKIP_NAMES = {".DS_Store"}

#: setuptools' own build cache. `pip wheel` runs `setup.py build_py`, which
#: copies package files in here -- and `build_py` never removes a file from a
#: prior run that it is no longer asked to copy, so anything that lands here
#: by accident (Finder's `.DS_Store`, a `.pyc` from an interpreter nobody
#: targets any more, a stale `.so` a previous target left behind) is
#: re-copied into every wheel built from this checkout afterwards, silently.
#: This is invisible to the SKIP_NAMES scan above, because that scan reads
#: the *source* tree and this directory is a copy `pip` made of it on some
#: earlier run -- possibly before the file it now carries even existed in the
#: source. It is how `.DS_Store` shipped in all six wheels on 2026-08-30: an
#: early host build populated `build/lib.macosx-10.13-universal2-cpython-313/`
#: with one, and nothing after that build ever looked at this directory's age
#: or contents again.
BUILD_CACHE = REPO / "build"


def _fail(msg: str) -> None:
    sys.exit(f"tools/wheel/build.py: {msg}")


def check_build_cache() -> None:
    """Refuse to build while a setuptools build cache from a prior run exists.

    See `BUILD_CACHE`'s comment for why any pre-existing one is untrustworthy
    rather than merely worth a glance. Refused rather than cleared: this
    script has no business deciding on its own that a directory outside the
    vendored tree is safe to erase, and an automatic clear would make the
    *next* accidental file in there silent again in exactly the way this
    refusal exists to end. Each immediate entry is named so the fix stays
    narrow -- `rm -rf build/` wholesale is not what this asks for, and was
    blocked by this environment's own safety classifier the day this was
    found.

    A separate function, not inlined in `preflight`, so a self-test can drive
    it against a scratch directory instead of the real `build/` beside this
    checkout.
    """
    if BUILD_CACHE.is_dir() and any(BUILD_CACHE.iterdir()):
        stale = sorted(p.name for p in BUILD_CACHE.iterdir())
        _fail(
            f"a setuptools build cache exists at {BUILD_CACHE}\n"
            "  pip wheel's build_py copies package files in here and never\n"
            "  removes one from a prior run that it stops being asked to\n"
            "  copy -- so anything that landed here by accident (a Finder\n"
            "  .DS_Store, a stale .pyc, a .so a previous target left behind)\n"
            "  is silently re-copied into every wheel built from this\n"
            "  checkout afterwards. This is how .DS_Store shipped in all six\n"
            "  wheels on 2026-08-30.\n"
            f"  holds: {stale}\n"
            "  Fix (narrow -- only the stale cache, not the checkout):\n"
            + "".join(f"    rm -rf {BUILD_CACHE / name}\n" for name in stale)
        )


def preflight() -> dict[str, str]:
    """Refuse to build a wheel that would be missing its reason to exist."""
    if not VENDORED_ROOT.exists():
        _fail(
            f"no vendored torch tree at {VENDORED_ROOT.parent}\n"
            "  run: bash vendor/vendor_torch.sh\n"
            "  (the tree is gitignored -- a fresh clone never has it, and\n"
            "   building without it yields the py3-none-any shell on PyPI)"
        )
    if not SHIM.exists():
        _fail(
            f"no _C extension at {SHIM}\n"
            "  run: bash vendor/install_shim.sh\n"
            "  (without it the tree cannot import; that is by design --\n"
            "   vendor_torch.sh drops upstream's _C so the hole is visible)"
        )

    check_build_cache()

    # Finder metadata, refused rather than merely excluded. `.DS_Store` got into
    # 0.0.2a0's first build: the copy at the package root lands at the *archive*
    # root, where setuptools' package-data globs never look, and `.gitignore`
    # cannot see it because the vendored tree is not in git. Excluding it fixes
    # this build; refusing is what makes the next one say so instead of quietly
    # shipping whatever Finder left behind. It was caught by the iOS check
    # comparing two wheels member by member -- 2,566 shared, 1 differing -- which
    # is a long way round for a file that should never have been packageable.
    junk = sorted(
        str(p.relative_to(SRC)) for name in SKIP_NAMES
        for p in SRC.rglob(name)
    )
    if junk:
        _fail(
            "the source tree carries files that must not be packaged:\n"
            + "".join(f"   {j}\n" for j in junk)
            + "   remove them and re-run (find . -name .DS_Store -delete)"
        )

    stamp: dict[str, str] = {}
    if STAMP.exists():
        for line in STAMP.read_text().splitlines():
            key, _, value = line.partition("=")
            stamp[key] = value

    # `native_left=0` is vendor_torch.sh's own measurement that it removed every
    # compiled artefact from the tree. If it is not 0, the tree still carries
    # upstream binaries and the wheel would ship somebody else's `_C` alongside
    # ours -- with upstream's `.cpython-313-darwin.so` suffix winning the
    # importlib race, so ours would be dead weight and nothing would say so.
    if stamp.get("native_left") not in (None, "0"):
        _fail(
            f"vendored tree still holds {stamp['native_left']} native artefact(s)"
            " -- re-run vendor/vendor_torch.sh"
        )
    return stamp


def tree_files(root: Path) -> set[str]:
    """Every file the wheel is expected to carry, as archive-relative paths."""
    out: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(SKIP_SUFFIXES) or name == ".DS_Store":
                continue
            out.add(str((Path(dirpath) / name).relative_to(SRC)))
    return out


def run_pip_wheel(python: str, outdir: Path) -> Path:
    """Drive the PEP 517 build through pip.

    pip rather than `python -m build` because pip is in every virtualenv and
    `build` is not; `--no-build-isolation` because the build needs nothing that
    is not already installed, and isolation would download setuptools on every
    invocation. `--no-deps` because this distribution's dependency list is empty
    and resolving it would reach the network for no reason.
    """
    before = set(outdir.glob("*.whl")) if outdir.exists() else set()
    cmd = [
        python, "-m", "pip", "wheel", str(REPO),
        "--no-deps", "--no-build-isolation", "--wheel-dir", str(outdir),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    made = sorted(set(outdir.glob("*.whl")) - before)
    if len(made) != 1:
        # Rebuilding without a version bump overwrites in place, so an empty
        # diff is normal; more than one is not.
        made = sorted(outdir.glob("torchnative-*.whl"))
        if len(made) != 1:
            _fail(f"expected exactly one torchnative wheel in {outdir}, found {made}")
    return made[0]


def _record_line(arcname: str, data: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return arcname, "sha256=" + digest.decode("ascii"), str(len(data))


def _fix_install_name(data: bytes) -> bytes:
    """Strip the build machine's absolute path out of the Mach-O install name.

    cargo links a `cdylib` with `-install_name <CARGO_TARGET_DIR>/release/deps/
    lib_C.dylib`, so a wheel built here announces
    `/Volumes/macMini/caches/cargo-target-wheel/...` to anyone who runs `otool
    -L` on it. It is inert -- `dlopen()` of a module by path ignores
    `LC_ID_DYLIB`, which is why the extension imports anyway -- but a published
    artefact should not name a directory on the machine that built it.

    Doing it here rather than in `vendor/install_shim.sh` keeps the file that
    the test suites load byte-identical to the one cargo emitted, so a green
    suite is evidence about the artefact and not about this rewrite.

    Applies to the iOS artefacts too -- cargo does the same thing there. Not to
    the Android one: that is an ELF, and `install_name_tool` on an ELF is not a
    no-op but an error, so the format is checked rather than the host platform.
    """
    if macho_info(data) is None:
        return data
    if sys.platform != "darwin" or not shutil.which("install_name_tool"):
        return data
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "_C.abi3.so"
        path.write_bytes(data)
        subprocess.run(
            ["install_name_tool", "-id", "@rpath/_C.abi3.so", str(path)],
            check=True, capture_output=True,
        )
        # codesign: install_name_tool invalidates the ad-hoc signature that the
        # linker put on the arm64 slice, and macOS refuses to load an arm64
        # Mach-O with a *broken* signature (an unsigned one is fine, a stale one
        # is not). Re-sign ad-hoc. If codesign is missing, better to ship the
        # untouched original than an unloadable one.
        if shutil.which("codesign"):
            r = subprocess.run(
                ["codesign", "--force", "--sign", "-", str(path)],
                capture_output=True,
            )
            if r.returncode != 0:
                print("  ! codesign failed, keeping original install name",
                      r.stderr.decode(errors="replace").strip())
                return data
        else:
            print("  ! no codesign, keeping original install name")
            return data
        return path.read_bytes()


def honest_macos_plat(plat: str, so_data: bytes) -> str:
    """Correct the architecture half of a macOS platform tag.

    setuptools derives the platform tag from `sysconfig.get_platform()` and then
    lets `calculate_macosx_platform_tag` walk the archive -- but that walk only
    ever raises the *deployment target*; it never touches the architecture. So a
    wheel built by the python.org universal2 interpreter is tagged
    `macosx_11_0_universal2` no matter what is inside it. Ours holds an arm64
    Mach-O and nothing else.

    The difference is not cosmetic. `universal2` tells pip on an Intel Mac that
    this wheel is for it; the install succeeds and `import torch` then dies in
    `dlopen` with a mach-o architecture mismatch, at which point the user's
    evidence points at their machine rather than at this tag. Narrowing it means
    those users get "no matching distribution", which is the truth.
    """
    m = re.match(r"^(macosx_\d+_\d+)_(.+)$", plat)
    if not m:
        return plat
    arches = set(macho_arches(so_data))
    if not arches:
        return plat
    if arches == {"arm64"}:
        arch = "arm64"
    elif arches == {"x86_64"}:
        arch = "x86_64"
    elif arches == {"arm64", "x86_64"}:
        arch = "universal2"
    else:
        return plat  # something unusual; leave setuptools' answer alone
    return f"{m.group(1)}_{arch}"


def _retag(name: str, plat: str) -> str:
    """Swap the platform component of a wheel filename."""
    stem, _, _ = name.rpartition(".whl")
    parts = stem.split("-")
    parts[-1] = plat
    return "-".join(parts) + ".whl"


# ------------------------------------------------------------- cross targets
#
# Everything platform-shaped about a wheel is one of three things: which
# `_C.abi3.so` goes in it, which compiler builds the empty global-deps library
# beside it, and what the tag says. A target is those three answers.
#
# The tag is *derived*, not written down. Both PEP 738 and PEP 730 encode a
# minimum OS version, and the honest value for that is a property of the CPython
# distribution the extension is built against -- `ANDROID_API_LEVEL` and
# `IPHONEOS_DEPLOYMENT_TARGET` in its `_sysconfigdata_*.py`. Hardcoding either
# would mean the tag stays put when the distribution is replaced, which is the
# same class of lie as the `universal2` tag in §3.3 of docs/WHEEL.md.

TARGET_PYTHON_ROOT = Path(os.environ.get(
    "TORCHNATIVE_TARGET_PYTHON", "/Volumes/macMini/caches/target-python"))

# Where `cargo` put the cross artefacts. Cargo's own default is `<crate>/target`
# and every build wiring in this repository overrides it, so read the same
# variable rather than inventing a third convention.
CARGO_TARGET_DIR = Path(os.environ.get(
    "CARGO_TARGET_DIR", REPO / "rust" / "torch_c" / "target"))

#: Where the Pyodide distribution lives, and deliberately **not** a
#: subdirectory of `TARGET_PYTHON_ROOT`. Everything under that root is a
#: cross-compiled *CPython*, and every target above derives its tag by reading
#: one. `pyemscripten_<abi>_wasm32` cannot: the number in it is Pyodide's, and
#: the CPython inside Pyodide has never heard of it. Two roots because they are
#: two kinds of thing -- see `PyEmscriptenTarget.sysconfig`.
PYODIDE_ROOT = Path(os.environ.get(
    "TORCHNATIVE_PYODIDE", "/Volumes/macMini/caches/pyodide/pyodide"))

CRATE = REPO / "rust" / "torch_c"

# ------------------------------------------------------- artefact freshness
#
# The failure this exists to catch, measured 2026-08-29:
#
#     CARGO_TARGET_DIR/release/lib_C.dylib                 today
#     CARGO_TARGET_DIR/aarch64-apple-ios/release/...       4 days old
#     CARGO_TARGET_DIR/aarch64-apple-ios-sim/release/...   5 days old
#
# `build.py --target ios-arm64-sim` exited 0 and packaged the 5-day-old file.
# It does not build the cross artefact -- it picks one up from
# `CARGO_TARGET_DIR/<triple>/release/` -- and `preflight` looks only at the
# vendored tree and the host `_C`. So a wheel got built, verified and *passed*
# against source that had been superseded by a week of landed commits; the tell
# was an error message quoting a phrase no longer in the tree.
#
# Refuse rather than rebuild, for the reason `rust/torch_c/pytests/run.sh`
# refuses to install the shim it found stale:
#
#   * building here means writing down a second spelling of the cross build.
#     The device one needs PYO3_CONFIG_FILE (whose contents live in
#     docs/WHEEL.md §7.1, not in this repository), PYO3_CROSS_LIB_DIR and
#     TORCHNATIVE_PYTHON_FRAMEWORK_DIR; the Android one goes through
#     `scripts/device_android.sh build` and `cargo ndk --platform 21`. A second
#     spelling can drift from the first, and this repository's whole class of
#     recurring defect is a drift that is invisible until a device refuses the
#     file.
#   * this script already refuses when the artefact is *absent*, and points at
#     those same docs. Staleness is that question one notch further in;
#     answering it differently would mean "missing is your problem, stale is
#     mine".
#
# The criterion is cargo's own dep-info file (`lib_C.d`, next to the artefact),
# not a glob written here. That matters three ways:
#
#   * it lists what the build actually read, `include_str!` included -- this
#     crate pulls in `src/methods.json`, `src/overloads.json`, `src/surface.json`
#     and `src/bootstrap.py`, and a hand-written glob that missed one would be
#     silently blind to changes in it;
#   * it is per-target, sitting beside the artefact it describes, so the device
#     and simulator answers cannot be confused for each other;
#   * it records *absolute* paths, so an artefact built from a different
#     checkout is visible. A source glob cannot see that at all -- it would
#     compare this tree's mtimes against a binary built somewhere else and call
#     it fresh.
#
# `run.sh` compares bytes rather than mtimes, and deliberately: it *builds*, so
# it has a fresh reference, and a byte compare does not nag when a rebuild
# reproduces the previous output. Here there is nothing to compare against
# without building, which is what was just ruled out. mtime is what remains --
# and reading it off the dep-info makes it the same question cargo asks when it
# decides whether to rebuild, rather than a stricter one invented here.
#
# Three outcomes, kept apart. `run.sh` learned this from `cmp`, which
# distinguishes same (0), different (1) and "the comparison itself failed"
# (>1); folding the last into the middle once produced a staleness report about
# an artefact that was current. The same trap is here in the other direction and
# is the easier one to fall into: `max()` over an empty dependency list has no
# error to report, it just answers "nothing is newer than the artefact" -- so an
# unreadable, unparseable or foreign dep-info would read as *fresh*. Every path
# out of `artefact_verdict` therefore names which of the three it is, and only
# one of them is a staleness claim.

FRESH, STALE, UNKNOWN = "fresh", "stale", "unknown"


def _unescape_make(path: str) -> str:
    r"""Undo the `\ ` that cargo writes for a space inside a path."""
    return re.sub(r"\\(.)", r"\1", path)


def _make_rules(text: str) -> list[tuple[str, list[str]]]:
    r"""Split a Makefile-format dep-info into `(target, prerequisites)`.

    Only the shape cargo emits is handled: one rule per line, no line
    continuations, spaces inside paths escaped with a backslash. A line without
    an unescaped `:` is not a rule and is skipped -- cargo writes bare
    `<dep>:` lines in some configurations and those carry no information here.
    """
    rules: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # First unescaped colon that is not `C:\...`; cargo writes POSIX paths
        # on this platform, so the drive-letter case cannot arise, but the scan
        # is written to skip escaped colons regardless.
        idx = -1
        i = 0
        while i < len(line):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == ":":
                idx = i
                break
            i += 1
        if idx < 0:
            continue
        target = _unescape_make(line[:idx].strip())
        rest = line[idx + 1:]
        deps = [_unescape_make(tok)
                for tok in re.split(r"(?<!\\)\s+", rest.strip()) if tok]
        rules.append((target, deps))
    return rules


def artefact_verdict(artefact: Path) -> tuple[str, str]:
    """Was `artefact` built from the source that is on disk now?

    Returns `(FRESH | STALE | UNKNOWN, detail)`. `UNKNOWN` is never a staleness
    claim -- it means the question could not be answered, and the detail says
    why. See the block comment above for why the three are kept apart.
    """
    dep_file = artefact.with_suffix(".d")
    if not dep_file.exists():
        return UNKNOWN, (
            f"cargo wrote no dep-info beside it ({dep_file} does not exist), so "
            "what that build read is not recorded anywhere")
    try:
        text = dep_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return UNKNOWN, f"{dep_file} could not be read: {exc!r}"

    rules = _make_rules(text)
    if not rules:
        return UNKNOWN, f"{dep_file} holds no Makefile rule this can read"

    want = os.path.realpath(artefact)
    deps: list[str] | None = None
    for target, prerequisites in rules:
        if os.path.realpath(target) == want:
            deps = prerequisites
            break
    if deps is None:  # a target dir moved after the build; match on the name
        for target, prerequisites in rules:
            if os.path.basename(target) == artefact.name:
                deps = prerequisites
                break
    if deps is None:
        return UNKNOWN, (
            f"{dep_file} describes {[t for t, _ in rules][:3]}, none of which is "
            f"{artefact} -- it is some other build's record")
    if not deps:
        return UNKNOWN, (
            f"{dep_file} names {artefact.name} but lists no inputs, so there is "
            "nothing to compare it against")

    # Absolute paths from the machine that built it. If they are not inside this
    # crate then the artefact came out of a different checkout, and comparing it
    # against *this* tree's mtimes would answer a question nobody asked.
    crate = os.path.realpath(CRATE)
    foreign = [d for d in deps if not os.path.realpath(d).startswith(crate + os.sep)]
    if foreign:
        return UNKNOWN, (
            f"the build read {len(foreign)} input(s) from outside {CRATE}, e.g. "
            f"{foreign[0]} -- this artefact was built from a different checkout, "
            "so its age relative to this one says nothing")

    try:
        artefact_ns = artefact.stat().st_mtime_ns
    except OSError as exc:
        return UNKNOWN, f"{artefact} could not be stat'ed: {exc!r}"

    newest: tuple[int, str] | None = None
    for dep in deps:
        try:
            dep_ns = os.stat(dep).st_mtime_ns
        except OSError as exc:
            return UNKNOWN, (
                f"the build read {dep}, which cannot be stat'ed now: {exc!r}. "
                "Whether the artefact is current is therefore unknown")
        if newest is None or dep_ns > newest[0]:
            newest = (dep_ns, dep)

    assert newest is not None  # `deps` is non-empty and every stat succeeded
    if newest[0] > artefact_ns:
        gap = (newest[0] - artefact_ns) / 1e9
        return STALE, (
            f"{os.path.relpath(newest[1], REPO)} was modified {gap / 3600:.1f} h "
            f"({gap:.0f} s) after {artefact.name} was written, and it is one of "
            f"the {len(deps)} inputs that build read")
    gap = (artefact_ns - newest[0]) / 1e9
    return FRESH, (
        f"{len(deps)} recorded inputs, newest "
        f"{os.path.relpath(newest[1], REPO)}, {gap / 3600:.1f} h before it")


def _render_refusal(verdict: str, artefact: Path, detail: str,
                    rebuild_hint: str) -> str:
    """The text a reader gets. Separate from `require_current` so the self-test
    can check that a staleness report and a "cannot judge" read as different
    things, which is the whole reason the two verdicts exist."""
    if verdict == STALE:
        return (
            f"{artefact} is stale.\n"
            f"  {detail}.\n"
            "  Packaging it would put superseded machine code in the wheel, and\n"
            "  everything downstream -- the tag, the platform check, the file\n"
            "  list, verify_cross.py -- would pass, because none of them looks at\n"
            "  the age of the code. That has happened: a simulator wheel was built\n"
            "  and verified against source five days out of date, and the tell was\n"
            "  an error message quoting a phrase no longer in the tree.\n"
            f"  Fix: {rebuild_hint}"
        )
    return (
        f"cannot tell whether {artefact} is current.\n"
        f"  {detail}.\n"
        "  This is NOT a staleness report -- the artefact may well be fine. It is\n"
        "  the check failing to run, which is reported separately so that it is\n"
        "  never mistaken for a finding about the artefact.\n"
        f"  Fix: {rebuild_hint}, which writes the dep-info this reads."
    )


def require_current(artefact: Path, rebuild_hint: str) -> None:
    """`_fail` unless `artefact` was built from the source now on disk."""
    verdict, detail = artefact_verdict(artefact)
    if verdict == FRESH:
        print(f"  current: {detail}")
        return
    _fail(_render_refusal(verdict, artefact, detail, rebuild_hint))


HOST_REBUILD_HINT = "bash vendor/install_shim.sh"


def check_host_shim() -> None:
    """The host wheel's `_C` is a *copy*; check the thing it was copied from.

    `preflight` establishes that `torch/_C.abi3.so` exists. It cannot establish
    that it is current, because the file carries no record of what built it --
    `vendor/install_shim.sh` runs `cargo build --release` and then `cp`s the
    result into the vendored tree, and nothing ties the copy back to its source
    afterwards.

    So the question is asked in two halves, which is what
    `rust/torch_c/pytests/run.sh` already does for its own reasons: is the host
    cargo artefact current (dep-info), and is the shim that same artefact
    (bytes). Both have to hold. Bytes rather than mtime for the second half for
    run.sh's reason -- `cp` sets a fresh mtime on the copy, so mtime would say
    "newer" and mean nothing.

    The cross path does not need this second half: it reads the cargo artefact
    directly, so there is no copy in between.
    """
    host = None
    for name in ("lib_C.dylib", "lib_C.so"):
        candidate = CARGO_TARGET_DIR / "release" / name
        if candidate.exists():
            host = candidate
            break
    if host is None:
        _fail(
            "cannot tell whether the host _C.abi3.so is current.\n"
            f"  There is no lib_C.dylib or lib_C.so under {CARGO_TARGET_DIR}/"
            "release,\n"
            "  so the build it was copied from is not on this machine to compare\n"
            "  against. This is NOT a staleness report -- the shim may be fine.\n"
            "  CARGO_TARGET_DIR selects where to look; if the shim was installed\n"
            "  with a different one, set it to that.\n"
            f"  Fix: {HOST_REBUILD_HINT}"
        )
    require_current(host, HOST_REBUILD_HINT)

    try:
        installed = SHIM.read_bytes()
        built = host.read_bytes()
    except OSError as exc:
        _fail(
            f"cannot tell whether {SHIM} is the artefact {host} holds.\n"
            f"  Reading one of them failed: {exc!r}.\n"
            "  This is NOT a staleness report -- it is the comparison itself\n"
            "  failing, and the two are kept apart on purpose (see run.sh, where\n"
            "  a SIGKILLed `cmp` once got reported as a stale shim).\n"
            f"  Fix: retry; if it persists, {HOST_REBUILD_HINT}"
        )
    if installed != built:
        _fail(
            f"{SHIM} is not the artefact under {CARGO_TARGET_DIR}/release.\n"
            f"  {len(installed):,} B installed vs {len(built):,} B built.\n"
            "  install_shim.sh copies one to the other, so they differ only if\n"
            "  the crate has been rebuilt since -- meaning the wheel would carry\n"
            "  the older of the two, silently.\n"
            f"  Fix: {HOST_REBUILD_HINT}"
        )
    print(f"  current: torch/_C.abi3.so is byte-identical to {host.name}")


# NDK ABI names, keyed by the architecture half of CPython's `MULTIARCH`. This
# is the NDK's own mapping (developer.android.com/ndk/guides/abis), not a guess;
# `packaging.tags.android_platforms` normalises the hyphen to an underscore and
# that normalisation is applied here too, by `_normalise`.
_ANDROID_ABIS = {
    "aarch64": "arm64-v8a",
    "arm": "armeabi-v7a",
    "x86_64": "x86_64",
    "i686": "x86",
}


def _normalise(s: str) -> str:
    """`packaging.tags._normalize_string`: hyphens and periods become `_`."""
    return s.replace("-", "_").replace(".", "_")


def target_sysconfig(root: Path) -> dict[str, object]:
    """`build_time_vars` of a cross-compiled CPython, without importing it.

    The file is CPython's own generated `_sysconfigdata_<abi>_<multiarch>.py`,
    a single dict literal. Reading it is how the wheel tag gets to be a fact
    about the interpreter the extension will actually be loaded by, rather than
    a constant in this file.
    """
    found = sorted(root.glob("lib/python3.*/_sysconfigdata_*.py"))
    if not found:
        _fail(f"no _sysconfigdata_*.py under {root}/lib/python3.* -- "
              f"is {root} a cross-compiled CPython distribution?")
    namespace: dict[str, object] = {}
    exec(compile(found[0].read_text(), str(found[0]), "exec"), namespace)
    variables = namespace.get("build_time_vars")
    if not isinstance(variables, dict):
        _fail(f"{found[0]} defines no build_time_vars")
    return variables


class Target:
    """One cross target: an artefact, a compiler, and a derived tag."""

    def __init__(self, key: str, rust_target: str, python_root: Path,
                 artefact_name: str):
        self.key = key
        self.rust_target = rust_target
        self.python_root = python_root
        self.artefact = CARGO_TARGET_DIR / rust_target / "release" / artefact_name

    # The name `_load_global_deps()` will look for. It picks the extension with
    # `".dylib" if platform.system() == "Darwin" else ".so"` -- and
    # `platform.system()` is "iOS" on iOS and "Android" on Android, so both of
    # them want `.so` even though the iOS file is a Mach-O dylib. Getting this
    # wrong is silent: the `ctypes.CDLL` raises OSError, `_load_global_deps`
    # swallows it into `_preload_cuda_deps`, and the import fails somewhere else.
    # `None` means this platform loads no global-deps library at all and the
    # wheel must not carry one. Only Windows: `_load_global_deps()` opens with
    # `if platform.system() == "Windows": return`, while `_load_dll_libraries()`
    # globs `torch/lib/*.dll` and LoadLibrary's every hit -- so an empty stub
    # there would be loaded for nothing, and a failure to load it would be
    # raised rather than swallowed. See docs/WINDOWS.md §4.3.
    global_deps_name: str | None = "libtorch_global_deps.so"

    # Where the extension sits in the archive, which is not the same question as
    # what the file is called on the build machine. The name has to be one the
    # target interpreter's `_PyImport_DynLoadFiletab` contains, and for an abi3
    # extension that is `.abi3.so` everywhere CPython builds with POSIX dynload
    # -- but Windows has its own table (`_d.pyd`, `.cp313-win_amd64.pyd`,
    # `.pyd`) with no `.abi3.so` in it, so there the member is `_C.pyd`.
    extension_member = "torch/_C.abi3.so"

    # How to produce this artefact. Named rather than run -- see the freshness
    # block comment for why this script refuses instead of rebuilding.
    rebuild_hint = "rebuild the cross artefact"

    #: Why this target cannot be built *here*, or `None` if it can.
    #:
    #: Not the same thing as a missing artefact, and kept apart from it on
    #: purpose. "No cross-built extension at <path>, run <hint>" is advice: it
    #: says the build has not happened yet and names the command that would do
    #: it. A refusal says no such command exists on this machine, and running
    #: the hint cannot help -- a different sentence, and the one
    #: `--target linux-x86_64` needed before docs/LINUX.md §9 made
    #: cargo-zigbuild work.
    #:
    #: Checked in `main` *before* the artefact, so a refusing target answers
    #: with its reason instead of sending the reader off to run a rebuild hint
    #: that cannot succeed. A refusing target stays in `TARGETS`, and therefore
    #: in `--target`'s choices, so `--target <name>` refuses **by name**.
    #: Dropping it from the registry instead would make it look like a target
    #: nobody had thought about, which is exactly what the original
    #: `linux-x86_64` entry existed to avoid.
    refusal: str | None = None

    def sysconfig(self) -> dict[str, object]:
        return target_sysconfig(self.python_root)

    def platform_tag(self, artefact: bytes) -> str:      # pragma: no cover
        raise NotImplementedError

    def cc(self) -> list[str]:                           # pragma: no cover
        raise NotImplementedError

    def check_image(self, data: bytes, what: str) -> None:  # pragma: no cover
        """Is this machine code for this target? Raises through `_fail`."""
        raise NotImplementedError

    def check_artefact(self, data: bytes) -> None:
        self.check_image(data, str(self.artefact))

    def check_global_deps(self, data: bytes) -> None:
        self.check_image(data, f"the global-deps library from {self.cc()[0]}")


class AndroidTarget(Target):
    """PEP 738: `android_<api-level>_<abi>`.

    The API level in the tag is a *floor*: `packaging.tags.android_platforms`
    yields every level from the device's own down to 16, so a device at API 34
    accepts an `android_21_` wheel and a device at API 19 does not. The floor
    that is true for this build is the one CPython itself was configured with
    -- `scripts/device_android.sh build` passes `--platform 21` to `cargo ndk`
    for the same reason, and the two agreeing is checked below rather than
    assumed.
    """

    def __init__(self, key: str = "android-arm64-v8a",
                 rust_target: str = "aarch64-linux-android",
                 elf_machine: str = "aarch64",
                 refusal: str | None = None):
        super().__init__(
            key, rust_target,
            TARGET_PYTHON_ROOT / rust_target / "prefix", "lib_C.so",
        )
        #: The `elf_info` machine name `check_image` demands. Held rather than
        #: split off `rust_target` because the two spellings do not always
        #: agree -- the NDK triple for 32-bit ARM is `armv7a-linux-androideabi`
        #: and the ELF machine is `arm` -- so deriving one from the other would
        #: be right for the two ABIs we build and wrong for the two we do not.
        self.elf_machine = elf_machine
        #: The NDK's `<triple><api>-clang` prefix, which is the rust triple for
        #: every ABI except 32-bit ARM. Kept beside `elf_machine` for the same
        #: reason.
        self.clang_triple = rust_target
        self.refusal = refusal

    rebuild_hint = "scripts/device_android.sh build"

    def _api_and_abi(self) -> tuple[int, str]:
        variables = self.sysconfig()
        api = int(variables["ANDROID_API_LEVEL"])
        arch = str(variables["MULTIARCH"]).split("-")[0]
        if arch not in _ANDROID_ABIS:
            _fail(f"unknown Android architecture {arch!r} in MULTIARCH="
                  f"{variables['MULTIARCH']!r}")
        # The distribution has to be the one this target builds against, not
        # merely *an* Android CPython. Without this, pointing
        # TORCHNATIVE_TARGET_PYTHON at a tree whose `aarch64-linux-android/`
        # and `x86_64-linux-android/` differ in nothing but name would tag an
        # x86-64 extension `android_<api>_arm64_v8a` -- and `check_image`
        # would not catch it, because it checks the artefact against this
        # target and both would agree. Same check as `LinuxTarget._multiarch`
        # and for the same reason.
        if arch != self.elf_machine:
            _fail(
                f"{self.python_root} has MULTIARCH={variables['MULTIARCH']!r}, "
                f"whose architecture is {arch!r} and not {self.elf_machine!r} "
                f"-- it is not the\n  Android CPython distribution "
                f"--target {self.key} builds against"
            )
        return api, _normalise(_ANDROID_ABIS[arch])

    def platform_tag(self, artefact: bytes) -> str:
        api, abi = self._api_and_abi()
        tag = f"android_{api}_{abi}"
        _confirm_with_packaging(tag, "android", api_level=api, abi=abi)
        return tag

    def cc(self) -> list[str]:
        api, _ = self._api_and_abi()
        ndk = Path(os.environ.get(
            "ANDROID_NDK_HOME",
            Path.home() / "Library/Android/sdk/ndk/27.1.12297006"))
        found = sorted(ndk.glob(
            f"toolchains/llvm/prebuilt/*/bin/{self.clang_triple}{api}-clang"))
        if not found:
            _fail(
                f"no {self.clang_triple}{api}-clang under {ndk} -- set "
                "ANDROID_NDK_HOME. Without it the global-deps library would be "
                "built by the host cc, which puts a Mach-O inside an Android "
                "wheel and makes `import torch` fail on the device only"
            )
        return [str(found[0]), "-shared", "-fPIC"]

    def check_image(self, data: bytes, what: str) -> None:
        info = elf_info(data)
        if info is None:
            _fail(f"{what} is not an ELF image ({describe(data)}) -- "
                  "an Android wheel cannot carry it")
        want = (64, self.elf_machine, "dyn")
        if (info["bits"], info["machine"], info["type"]) != want:
            _fail(f"{what} is {describe(data)}, expected ELF 64-bit "
                  f"{self.elf_machine} dyn")


class IOSTarget(Target):
    """PEP 730: `ios_<major>_<minor>_<multiarch>`.

    Same floor semantics as Android -- `packaging.tags.ios_platforms` walks down
    to 12.0 -- so the version comes from `IPHONEOS_DEPLOYMENT_TARGET`, and the
    `multiarch` half straight from CPython's `MULTIARCH`, which is exactly the
    `sys.implementation._multiarch` that `ios_platforms` reads at run time.

    The device and the simulator are two targets, not one: their Mach-O images
    differ only in the platform field of `LC_BUILD_VERSION` (2 against 7), so a
    wheel carrying the wrong one is indistinguishable by size, architecture or
    symbol table and fails only when a real device tries to load it.
    """

    def __init__(self, key: str, rust_target: str, subdir: str, sdk: str,
                 platform_id: str):
        super().__init__(key, rust_target, TARGET_PYTHON_ROOT / subdir,
                         "lib_C.dylib")
        self.sdk = sdk
        self.platform_id = platform_id
        self.rebuild_hint = (
            "re-run the cross build for this target -- docs/WHEEL.md §7.1 has "
            f"the exact command (cargo build --release --target {rust_target}, "
            "with PYO3_CONFIG_FILE and PYO3_CROSS_LIB_DIR"
            + (", TORCHNATIVE_PYTHON_FRAMEWORK_DIR"
               if platform_id == "ios" else "")
            + ")")

    def _multiarch_and_min(self) -> tuple[str, tuple[int, int]]:
        variables = self.sysconfig()
        multiarch = _normalise(str(variables["MULTIARCH"]))
        major, _, minor = str(variables["IPHONEOS_DEPLOYMENT_TARGET"]).partition(".")
        return multiarch, (int(major), int(minor or 0))

    def platform_tag(self, artefact: bytes) -> str:
        multiarch, (major, minor) = self._multiarch_and_min()
        # The artefact has its own floor, from `-mios-version-min`. Rust's
        # default for these targets need not match CPython's, and if it is
        # *higher* then the interpreter's number is the wrong one to publish --
        # the wheel would claim to run on an iOS that cannot load its own
        # extension. Take the higher of the two and say which won.
        info = macho_info(artefact) or {}
        art_min = info.get("minos")
        if art_min and art_min[:2] > (major, minor):
            print(f"  tag floor from the artefact ({art_min[0]}.{art_min[1]}), "
                  f"not from CPython ({major}.{minor})")
            major, minor = art_min[0], art_min[1]
        tag = f"ios_{major}_{minor}_{multiarch}"
        _confirm_with_packaging(tag, "ios", version=(major, minor),
                                multiarch=multiarch)
        return tag

    def cc(self) -> list[str]:
        _, (major, minor) = self._multiarch_and_min()
        triple = f"arm64-apple-ios{major}.{minor}"
        if self.platform_id == "iossimulator":
            triple += "-simulator"
        return [
            "xcrun", "--sdk", self.sdk, "clang",
            "-target", triple, "-dynamiclib",
            "-install_name", f"@rpath/{self.global_deps_name}",
        ]

    def check_image(self, data: bytes, what: str) -> None:
        info = macho_info(data)
        if info is None:
            _fail(f"{what} is not a thin 64-bit Mach-O ({describe(data)})")
        if info["arch"] != "arm64":
            _fail(f"{what} is {info['arch']}, expected arm64")
        if info["platform"] != self.platform_id:
            _fail(
                f"{what} is built for {info['platform']!r}, not "
                f"{self.platform_id!r}. A wheel tagged for one and holding the "
                "other installs and then fails only on the real thing"
            )

    def check_artefact(self, data: bytes) -> None:
        super().check_artefact(data)
        # Device only. The simulator resolves CPython symbols the way macOS
        # does -- `.cargo/config.toml` gives it `-undefined dynamic_lookup`, so
        # it carries no Python dependency at all and demanding one here would
        # fail a correct artefact. On a physical device there is no libpython to
        # fall back on, so the framework has to be named in the load commands.
        info = macho_info(data) or {}
        if self.platform_id == "ios" and not any(
            "Python.framework" in d for d in info.get("dylibs", ())
        ):
            _fail(
                f"{self.artefact} does not link Python.framework "
                f"(LC_LOAD_DYLIB: {info.get('dylibs')}). See "
                "docs/RUST_CROSSBUILD.md §0.5, and check "
                "TORCHNATIVE_PYTHON_FRAMEWORK_DIR"
            )


class LinuxTarget(Target):
    """PEP 600: `manylinux_<glibcmajor>_<glibcminor>_<arch>`.

    The floor comes from a **different place than on Android and iOS**, and that
    is the whole of what is interesting here.

    Both of those read it off the target CPython: `ANDROID_API_LEVEL` and
    `IPHONEOS_DEPLOYMENT_TARGET` are fields in the distribution's
    `_sysconfigdata_*.py`, so the tag is a fact about the interpreter the
    extension will be loaded by. The Linux distribution has no such field --
    `ANDROID_API_LEVEL` is literally `0` in it -- because glibc compatibility is
    not a property of the interpreter build. It is a property of *this* artefact:
    the set of `GLIBC_x.y` symbol versions its own code references, which the
    linker recorded in `.gnu.version_r`. The highest of them is the oldest glibc
    that can load the file, and nothing else is.

    So `platform_tag` reads the artefact and ignores the interpreter for the
    floor. That is what `auditwheel` does, for the same reason. The interpreter
    is still consulted -- `MULTIARCH` has to say `x86_64-linux-gnu`, or this is
    not the distribution the extension was built against -- but it does not set
    the number.

    A wheel is manylinux only if it *also* depends on nothing outside the
    policy's library list, so `DT_NEEDED` is checked against it. Skipping that
    would let a wheel linking, say, `libopenblas.so` claim a tag whose entire
    promise is that it does not.

    The artefact is buildable on this machine as of docs/LINUX.md §9, with
    cargo-zigbuild; before that it was not, and this class still refuses by name
    when the file is missing rather than dropping out of `--target`'s choices.
    """

    #: PEP 599's external-library list for manylinux2014, which PEP 600 carries
    #: forward unchanged for `manylinux_2_17` and later.
    #:
    #: **This is the whole list and it is architecture-independent.** PEP 599
    #: varies its *architectures* (it names x86_64, i686, aarch64, ppc64,
    #: ppc64le and s390x) but not the table of libraries a conforming wheel may
    #: link, so nothing here is per-arch and adding an arch must not add an
    #: entry. The one thing that does vary is the dynamic loader, which is not
    #: in PEP 599's table at all -- it is never a `DT_NEEDED` in the usual
    #: sense -- yet appears as one on some architectures. Its name is
    #: arch-specific, so it lives in `LOADER` and is unioned in per target
    #: rather than being written into this frozenset, where an x86-64 name
    #: would silently be "allowed" for an aarch64 wheel.
    POLICY_LIBRARIES = frozenset({
        "libgcc_s.so.1", "libstdc++.so.6", "libm.so.6", "libdl.so.2",
        "librt.so.1", "libc.so.6", "libnsl.so.1", "libutil.so.1",
        "libpthread.so.0", "libresolv.so.2", "libX11.so.6", "libXext.so.6",
        "libXrender.so.1", "libICE.so.6", "libSM.so.6", "libGL.so.1",
        "libgobject-2.0.so.0", "libgthread-2.0.so.0", "libglib-2.0.so.0",
    })

    #: The dynamic loader's `DT_NEEDED` name per architecture -- glibc's
    #: `ld.so`, whose soname encodes the ABI. Not in `POLICY_LIBRARIES` (see
    #: there); only the entry for *this* target's architecture is allowed, so
    #: an aarch64 artefact that somehow named the x86-64 loader is refused
    #: rather than waved through.
    LOADER = {
        "x86_64": "ld-linux-x86-64.so.2",
        "aarch64": "ld-linux-aarch64.so.1",
    }

    #: `GLIBC_ABI_DT_RELR` is the one version name in glibc that is not
    #: `GLIBC_<numbers>`; it was added in 2.36 and means exactly that. Mapping it
    #: here rather than discarding it matters, because discarding it lowers the
    #: floor -- an artefact needing 2.36 would be tagged for whatever numeric
    #: version happened to be second-highest, and the wheel would install on a
    #: glibc that cannot load it. Any *other* non-numeric name is refused rather
    #: than guessed at, for the same reason.
    NAMED_VERSIONS = {"GLIBC_ABI_DT_RELR": (2, 36)}

    def __init__(self, key: str = "linux-x86_64",
                 rust_target: str = "x86_64-unknown-linux-gnu",
                 arch: str = "x86_64"):
        super().__init__(
            key, rust_target,
            # No `prefix/` subdirectory: this distribution is
            # python-build-standalone's `install_only` layout, like the iOS ones
            # and unlike the Android one. docs/LINUX.md §3 compares the four.
            TARGET_PYTHON_ROOT / rust_target, "lib_C.so",
        )
        #: The architecture as three different vocabularies spell it, which all
        #: happen to agree here and are still not the same word: the ELF
        #: `e_machine` name `elf_info` reports, the `<arch>` half of a PEP 600
        #: tag, and the architecture half of CPython's `MULTIARCH`. They agree
        #: for x86_64 and aarch64; they would not for 32-bit ARM (`arm` vs
        #: `armv7l`), so this is stored, not derived.
        self.arch = arch
        if arch not in self.LOADER:
            _fail(f"no dynamic-loader name known for {arch!r}; add it to "
                  f"{type(self).__name__}.LOADER")
        self.rebuild_hint = (
            f"PYO3_CROSS_LIB_DIR=<target-python>/lib cargo zigbuild --release "
            f"--target {rust_target}.{self.GLIBC_TARGET[0]}."
            f"{self.GLIBC_TARGET[1]}, from rust/torch_c "
            "(docs/LINUX.md §9.2 has the whole environment; §9.1 installs "
            "cargo-zigbuild and ziglang, which it needs)"
        )

    def _multiarch(self) -> str:
        variables = self.sysconfig()
        multiarch = str(variables["MULTIARCH"])
        want = f"{self.arch}-linux-gnu"
        if multiarch != want:
            _fail(f"{self.python_root} has MULTIARCH={multiarch!r}, not "
                  f"{want!r} -- it is not the {self.arch} Linux "
                  "distribution this target builds against")
        return multiarch

    def _policy_libraries(self) -> frozenset[str]:
        """PEP 599's list plus *this* architecture's loader, and no other's."""
        return self.POLICY_LIBRARIES | {self.LOADER[self.arch]}

    def _glibc_floor(self, artefact: bytes) -> tuple[int, int]:
        """The oldest glibc that can load this image, from its own version needs."""
        info = elf_dynamic(artefact)
        if info is None:
            _fail(f"{self.artefact} could not be read as a 64-bit ELF with "
                  "section headers, so its glibc requirements -- which are the "
                  "only source for the manylinux floor -- are unavailable")

        wanted: set[str] = set()
        for library, names in info["versions"].items():
            if library.startswith(("libc.so", "libm.so", "libpthread.so",
                                   "libdl.so", "librt.so", "libutil.so",
                                   "ld-linux")):
                wanted |= names
        if not wanted:
            # The §2.6 trap. An image linked against empty stand-in libraries has
            # no version requirements at all, and `max()` over that is not a low
            # floor -- it is no answer. Saying `manylinux_2_5` here would be the
            # check reporting its own failure as a finding about the artefact.
            _fail(
                f"{self.artefact} records no glibc symbol versions at all "
                f"(DT_NEEDED: {info['needed'] or 'none'}).\n"
                "  This is NOT a claim that it runs on an old glibc -- it is the\n"
                "  question having no answer, and the two must not be confused.\n"
                "  An ELF gets version requirements from the libc it was linked\n"
                "  against; an image linked with `-shared` against no libc at all\n"
                "  links cleanly and lands here (docs/LINUX.md §2.6).\n"
                f"  Fix: {self.rebuild_hint}"
            )

        floor = (2, 0)
        for name in sorted(wanted):
            if not name.startswith("GLIBC_"):
                continue
            rest = name[len("GLIBC_"):]
            parts = rest.split(".")
            if all(p.isdigit() for p in parts) and len(parts) >= 2:
                version = (int(parts[0]), int(parts[1]))
            elif name in self.NAMED_VERSIONS:
                version = self.NAMED_VERSIONS[name]
            else:
                _fail(
                    f"{self.artefact} requires glibc symbol version {name!r}, "
                    "which this does not know how to order.\n"
                    "  Refused rather than skipped: skipping an unknown version "
                    "can only lower the\n  floor, and a floor that is too low is "
                    "a wheel that installs where it cannot load.\n"
                    f"  Add it to {type(self).__name__}.NAMED_VERSIONS with the "
                    "glibc release that introduced it."
                )
            floor = max(floor, version)
        return floor

    def _check_policy(self, artefact: bytes) -> list[str]:
        info = elf_dynamic(artefact)
        if info is None:
            # The same distinction `_glibc_floor` two methods above makes, and
            # for the same reason: `elf_dynamic` returning `None` means the
            # section headers this reads DT_NEEDED from could not be found,
            # not that the image links nothing. Folding that into
            # `{"needed": []}` would make an unreadable artefact read as
            # policy-compliant -- `outside` empty, nothing refused -- which is
            # backwards: the check found nothing to say because it could not
            # look, not because there was nothing wrong. `platform_tag` below
            # happens to call `_glibc_floor` on the same bytes right after,
            # which does make this distinction and would catch the same
            # unreadable artefact -- but that is this method being right by
            # a caller's coincidence, not by its own contract, and any other
            # caller would not get that protection for free.
            _fail(
                f"{self.artefact} could not be read as a 64-bit ELF with "
                "section headers, so its DT_NEEDED libraries -- which the "
                "manylinux external-library policy (PEP 599) is checked "
                "against -- are unavailable")
        needed = list(info["needed"])
        outside = sorted(set(needed) - self._policy_libraries())
        if outside:
            _fail(
                f"{self.artefact} links {outside}, which manylinux's external-"
                "library policy does not\n  allow (PEP 599). The tag's entire "
                "promise is that the wheel needs nothing\n  beyond the listed "
                "libraries, so tagging this one manylinux would be false.\n"
                f"  It links: {needed}"
            )
        return needed

    def platform_tag(self, artefact: bytes) -> str:
        self._multiarch()
        needed = self._check_policy(artefact)
        major, minor = self._glibc_floor(artefact)
        tag = f"manylinux_{major}_{minor}_{self.arch}"
        print(f"  tag floor from the artefact's .gnu.version_r "
              f"(glibc {major}.{minor}), not from CPython -- the distribution "
              f"records no glibc minimum at all")
        print(f"  DT_NEEDED within the PEP 599 policy list: {needed}")
        _confirm_with_packaging(tag, "manylinux")
        _confirm_pep600_spelling(tag, self.arch)
        _confirm_manylinux_with_packaging(tag, self.arch)
        return tag

    #: The glibc `cc()` compiles the global-deps stub against, and the number
    #: docs/LINUX.md §9.2 passes to `cargo zigbuild --target
    #: x86_64-unknown-linux-gnu.<here>` for the Rust artefact. Kept in one place
    #: so the two members of the wheel are named the same version, but it does
    #: **not** set the tag: the tag is read off the artefact (see the class
    #: docstring), and lowering this constant would not lower the tag. The stub
    #: is empty, so it references no glibc symbol at all and records no version
    #: requirement whatever this says -- measured in docs/LINUX.md §9.3. That is
    #: why there is no cross-check between the two here: it could not fail.
    GLIBC_TARGET = (2, 17)

    @staticmethod
    def zig_command() -> list[str] | None:
        """`zig`, however it is installed -- the order cargo-zigbuild uses.

        zig ships two ways and only one of them is a binary on PATH. The
        `ziglang` wheel puts the compiler inside a Python package and is run as
        `<python> -m ziglang`; that is how cargo-zigbuild finds it when there is
        no `zig` executable, and this has to agree with cargo-zigbuild or the
        wheel's two native members get built by different toolchains.
        """
        binary = shutil.which("zig")
        if binary:
            return [binary]
        seen = set()
        for name in ("python3", "python", sys.executable):
            interpreter = shutil.which(name) if not os.path.isabs(name) else name
            if not interpreter or interpreter in seen:
                continue
            seen.add(interpreter)
            probe = subprocess.run([interpreter, "-m", "ziglang", "version"],
                                   capture_output=True)
            if probe.returncode == 0:
                return [interpreter, "-m", "ziglang"]
        return None

    def cc(self) -> list[str]:
        """A driver that compiles *and links* an ELF x86-64 shared object.

        Both halves in one command, because that is what `global_deps_stub`
        runs. This machine can do the two halves separately -- Apple clang emits
        ELF x86-64 objects and rustup's `rust-lld` links them, measured in
        docs/LINUX.md §2.7 -- and still cannot do them in one: driving lld
        through `clang --ld-path=` dies on `Library not loaded:
        @rpath/libLLVM.dylib`, because SIP strips `DYLD_LIBRARY_PATH` when
        `/usr/bin/clang` execs. So the separate-halves result is a fact about
        the machine, not a route this can take.
        """
        # cc-rs's spelling of the per-target override: the triple uppercased
        # with hyphens as underscores. Built from `rust_target` rather than
        # written out, so a second Linux target cannot silently pick up the
        # first one's compiler -- which is the bug this would otherwise have:
        # `CC_x86_64_unknown_linux_gnu` set for the x86-64 build would have
        # built the aarch64 wheel's global-deps stub as x86-64, and
        # `check_global_deps` is the only thing that would have caught it.
        env_name = f"CC_{self.rust_target.replace('-', '_')}"
        override = (os.environ.get(env_name) or os.environ.get("TARGET_CC"))
        if override:
            return [*override.split(), "-shared", "-fPIC"]
        zig = self.zig_command()
        if zig:
            major, minor = self.GLIBC_TARGET
            return [*zig, "cc", "-target",
                    f"{self.arch}-linux-gnu.{major}.{minor}", "-shared", "-fPIC"]
        _fail(
            f"no C compiler that targets {self.rust_target}.\n"
            f"  Tried, in order: ${env_name}, $TARGET_CC, `zig` "
            "on PATH, and\n"
            "  `<python> -m ziglang` -- the same order cargo-zigbuild uses, so "
            "that whatever\n"
            "  built the artefact also builds this stub.\n"
            "  Without one, the empty torch/lib/libtorch_global_deps.so would "
            "have to be built\n"
            "  by the host cc, which puts a Mach-O inside a Linux wheel and "
            "makes `import torch`\n"
            "  fail on Linux only (docs/VENDOR.md wall 1).\n"
            "  Fix: `pip install ziglang` into any interpreter on PATH "
            "(docs/LINUX.md §9.1),\n"
            "  or point CC_x86_64_unknown_linux_gnu at a cross gcc such as "
            "x86_64-linux-gnu-gcc."
        )

    def check_image(self, data: bytes, what: str) -> None:
        info = elf_info(data)
        if info is None:
            _fail(f"{what} is not an ELF image ({describe(data)}) -- "
                  "a Linux wheel cannot carry it")
        want = (64, self.arch, "dyn")
        if (info["bits"], info["machine"], info["type"]) != want:
            _fail(f"{what} is {describe(data)}, expected ELF 64-bit "
                  f"{self.arch} dyn")


#: The oldest glibc a manylinux tag may name, **per architecture**, and the one
#: place where "the same tag with a different arch" is not the same question.
#:
#: PEP 600's grammar allows any `manylinux_<major>_<minor>_<arch>`, but a tag no
#: installer will ever generate is not a tag. `packaging._manylinux.platform_tags`
#: -- the code pip runs -- floors the search at glibc 2.17 for every architecture
#: **except** x86-64 and i686, where it goes down to 2.5, because manylinux1 and
#: manylinux2010 (PEP 513, PEP 571) were x86-only and manylinux2014 (PEP 599) is
#: where aarch64 and the rest enter the scheme at all.
#:
#: So `manylinux_2_12_aarch64` is PEP 600-shaped and matches nothing: pip on an
#: aarch64 machine never yields that name, whatever its glibc. Checking only the
#: grammar -- which is what this file did while x86-64 was the only Linux target,
#: and which was correct for exactly that one architecture -- would have let such
#: a wheel out.
_MANYLINUX_ARCH_FLOOR = {"x86_64": (2, 5), "i686": (2, 5)}
_MANYLINUX_DEFAULT_FLOOR = (2, 17)


def _confirm_pep600_spelling(tag: str, arch: str) -> None:
    """The tag has to be a name PEP 600 defines, not merely a plausible one."""
    match = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", tag)
    if not match:
        _fail(f"{tag!r} is not PEP 600's manylinux_<major>_<minor>_<arch>")
    major, minor = int(match[1]), int(match[2])
    if match[3] != arch:
        _fail(f"{tag!r} names architecture {match[3]!r}, but this target builds "
              f"{arch!r}")
    floor = _MANYLINUX_ARCH_FLOOR.get(arch, _MANYLINUX_DEFAULT_FLOOR)
    if (major, minor) < floor:
        # Not one message with a number substituted into it: on x86-64 the
        # reason is PEP 600 defining the scheme no lower than manylinux1's 2.5,
        # and on every other architecture the reason is that the architecture
        # only entered the scheme with manylinux2014. Two different facts.
        why = ("below manylinux1's 2.5, and PEP 600 defines the scheme no lower"
               if arch in _MANYLINUX_ARCH_FLOOR else
               f"below manylinux2014's 2.17 -- {arch} is not in PEP 513 or PEP "
               "571, so it\n  enters the scheme at PEP 599 and pip generates no "
               "lower name for it")
        _fail(f"{tag!r} claims glibc {major}.{minor}, {why} -- "
              "no installer matches that")
    print(f"  tag {tag} is PEP 600-shaped (glibc {major}.{minor}, {arch}; "
          f"floor for {arch} is {floor[0]}.{floor[1]})")


def _confirm_manylinux_with_packaging(tag: str, arch: str) -> None:
    """Ask pip's own manylinux generator whether it would ever yield this name.

    `_confirm_with_packaging` cannot do this one. Its contract is to call
    `packaging.tags.<family>_platforms(**kwargs)`, and there is no
    `manylinux_platforms`: `packaging._manylinux.platform_tags` takes only an
    architecture list and reads the glibc of the **running** interpreter, which
    on this Mac is no glibc at all -- `_glibc_version_string()` returns None,
    `_have_compatible_abi` rejects a Mach-O `sys.executable`, and the generator
    yields nothing. That is why every Linux build so far printed "packaging has
    no manylinux_platforms -- tag spelling unchecked" and moved on.

    The missing piece is only the *host* half, and it is exactly the half a
    cross build is entitled to supply: the question is not "does this machine
    match the tag" but "is this a name pip would generate for a machine that
    does". So the two host probes are substituted -- the ABI check with True,
    the glibc version with the floor the tag itself claims -- and the generator
    is then run unmodified. Everything that decides the *name* (the per-arch
    floor, the legacy aliases, `_ALLOWED_ARCHS`) is still packaging's code and
    not a copy of it here.

    Substituting rather than reimplementing matters: the per-arch floor in
    `_MANYLINUX_ARCH_FLOOR` above is this file's own reading of PEP 599, and
    this function is what stops that reading from being self-confirming.
    """
    try:
        from packaging import _manylinux as pm
    except ImportError:
        print("  ! packaging._manylinux not importable -- manylinux spelling "
              "unchecked")
        return
    match = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", tag)
    if not match:                                     # pragma: no cover
        _fail(f"{tag!r} is not PEP 600's manylinux_<major>_<minor>_<arch>")
    claimed = (int(match[1]), int(match[2]))

    if arch not in getattr(pm, "_ALLOWED_ARCHS", {arch}):
        _fail(
            f"packaging._manylinux does not list {arch!r} among the "
            "architectures manylinux\n"
            f"  covers ({sorted(pm._ALLOWED_ARCHS)}), so pip would never "
            f"generate {tag!r}.")

    saved = {name: getattr(pm, name) for name in
             ("_have_compatible_abi", "_get_glibc_version")}
    try:
        # The host halves, and only those. `_get_glibc_version` is what the
        # generator counts *down* from, so handing it the tag's own floor asks
        # the narrowest possible question: on a glibc exactly this old, is this
        # name one of the answers? A higher number would also yield lower names
        # and could pass a tag whose floor pip would never emit.
        pm._have_compatible_abi = lambda executable, archs: True
        pm._get_glibc_version = lambda: pm._GLibCVersion(*claimed)
        accepted = list(pm.platform_tags([arch]))
    finally:
        for name, value in saved.items():
            setattr(pm, name, value)

    if tag not in accepted:
        _fail(
            f"packaging._manylinux.platform_tags([{arch!r}]) does not yield "
            f"{tag!r} even when told the\n"
            f"  host glibc is exactly {claimed[0]}.{claimed[1]}; it starts "
            f"{accepted[:3] or 'nothing at all'}.\n"
            "  pip generates the tags it matches against, so a name absent from "
            "that list matches\n  no installer on any machine.")
    legacy = [name for name in accepted if not name.startswith("manylinux_")]
    print(f"  tag {tag} yielded by packaging._manylinux.platform_tags("
          f"[{arch!r}]) at glibc {claimed[0]}.{claimed[1]}\n"
          f"      -- {len(accepted)} names in that list, of which the legacy "
          f"aliases are {legacy or 'none'}")


def _confirm_windows_normalisation(name: str) -> str:
    """`sysconfig.get_platform()`'s answer -> the tag, by packaging's own step.

    The last link of the chain in `WindowsTarget`'s docstring, and the only one
    that is packaging's code rather than CPython's: `_generic_platforms()` is
    `yield _normalize_string(sysconfig.get_platform())`, so the tag is that
    function applied to that string. Calling it here rather than writing
    `name.replace("-", "_")` is the difference between checking the spelling and
    asserting it -- `_normalize_string` also folds `.` and spaces, and if
    packaging ever changed what it folds, this would follow rather than drift.

    Falls back to this file's `_normalise`, loudly, when packaging is missing.
    That is the same shape of skip `_confirm_with_packaging` prints, and for the
    same reason: an unavailable check must read as unavailable, not as a pass.
    """
    try:
        from packaging.tags import _normalize_string
    except ImportError:
        print("  ! packaging not importable -- the win tag's normalisation is "
              "this file's own")
        return _normalise(name)
    tag = _normalize_string(name)
    print(f"  packaging.tags._normalize_string({name!r}) -> {tag} -- which is "
          "all of\n"
          "      packaging's Windows tag code, because `platform_tags()` falls "
          "through to\n"
          "      `_generic_platforms()` on Windows and there is no "
          "`windows_platforms` at all")
    return tag


class WindowsTarget(Target):
    """`win_amd64` and `win_arm64`, which are names and not derivations.

    Every other target here computes a floor from something: Android and iOS
    read a minimum OS out of the target CPython, Linux reads a glibc version out
    of the artefact. **Windows has no floor to compute.** The tag is one of
    four fixed strings (`win32`, `win_amd64`, `win_arm32`, `win_arm64`) and
    carries no version at all; PE records a `MajorSubsystemVersion`, but no
    installer looks at it and pip will hand a `win_amd64` wheel to any 64-bit
    Windows.

    **A name still has to be looked up, and that is what `_platform_name` does.**
    Adding the second Windows architecture is where "no floor to compute" stops
    meaning "nothing to get wrong": `win_arm64` is right, but *by analogy with
    `win_amd64`* it is a guess, and the same analogy produces `win_aarch64` and
    `win_arm64ec`, both of which are also plausible and neither of which any
    installer matches. So it is read out of the distribution instead.

    The chain is short and every link is on disk here:

      * `packaging.tags` has no `windows_platforms`. On Windows it falls through
        to `_generic_platforms()`, whose entire body is
        `yield _normalize_string(sysconfig.get_platform())`. So the tag *is*
        `sysconfig.get_platform()` with hyphens turned to underscores -- there
        is no Windows-specific tag code in packaging at all, which is also why
        `_confirm_with_packaging(tag, "windows")` has always printed a skip.
      * `sysconfig.get_platform()` on `os.name == "nt"` is four lines, and this
        distribution ships them (`Lib/sysconfig/__init__.py`). They test
        `sys.version.lower()` for `amd64`, then `(arm)`, then `(arm64)`, and
        fall back to `sys.platform`. **The order is load-bearing** -- `amd64`
        wins over everything, so the answer for a given distribution is decided
        by which of those substrings its own `sys.version` contains.
      * `sys.version` embeds `Py_GetCompiler()`, which on MSVC is the literal
        `[MSC v.<n> 64 bit (<arch>)]` compiled into `python3<minor>.dll`. That
        string is readable without running anything: `AMD64` in the x86-64
        distribution, `ARM64` in this one.

    So the tag is derived after all -- from the target distribution, like every
    other target in this file -- and `win_arm64` is the *result* rather than the
    assumption. `_platform_name` runs CPython's own branch order over the string
    it finds and refuses if the two distributions ever answer the same thing.

    What `check_image` checks instead of a floor is the two things Windows
    *does* make decidable: the image is a PE32+ DLL of this target's machine,
    and -- in `verify_windows.py` -- every symbol it imports is attributed to a
    named DLL by the import table. That second one has no Linux counterpart
    (docs/LINUX.md §6.1) and is as strong as the iOS device check.

    Two structural differences from every other target, both forced by upstream
    torch rather than chosen here:

    * the member is `torch/_C.pyd`, not `torch/_C.abi3.so` -- `Target.extension_member`
    * the wheel carries no global-deps library at all -- `Target.global_deps_name`
    """

    #: `python3.lib` is the abi3 import library: linking it is what makes the
    #: extension bind to `python3.dll` (the stable-ABI forwarder) rather than to
    #: `python313.dll`, and therefore what makes one file serve 3.13 and later.
    #: `PYO3_CROSS_LIB_DIR` must point at the directory holding it.
    ABI3_IMPORT_LIB = "python3.lib"

    global_deps_name = None
    extension_member = "torch/_C.pyd"

    #: The `[MSC v.<n> <bits> bit (<arch>)]` substring `sysconfig.get_platform()`
    #: branches on, lowercased, mapped to the name it returns for it. Keys and
    #: order are CPython's, not this file's: `Lib/sysconfig/__init__.py` tests
    #: `amd64` first and `(arm64)` last, and a dict preserves that. The ARM
    #: entries carry their parentheses because CPython's do -- a bare `arm64`
    #: would also match `[MSC v.1944 64 bit (ARM64EC)]`, which is a different
    #: architecture with no wheel tag of its own.
    #: Values are what `get_platform()` *returns* -- hyphenated, exactly as
    #: CPython spells them -- and not the tag. Turning `win-arm64` into
    #: `win_arm64` is packaging's step, not CPython's, and keeping the two
    #: apart is what lets `_confirm_windows_normalisation` check the join
    #: instead of this file asserting both halves at once.
    _PLATFORM_NAMES = {"amd64": "win-amd64", "(arm)": "win-arm32",
                       "(arm64)": "win-arm64"}

    def __init__(self, key: str = "windows-x86_64",
                 rust_target: str = "x86_64-pc-windows-msvc",
                 pe_machine: str = "x86_64",
                 expected_tag: str = "win_amd64"):
        super().__init__(
            key, rust_target,
            TARGET_PYTHON_ROOT / rust_target,
            # cargo names a `cdylib` after the crate with no `lib` prefix on
            # Windows, so this is `_C.dll` and not `lib_C.dll`.
            "_C.dll",
        )
        #: The `pe_info` machine name `check_image` demands, held rather than
        #: derived for the same reason `AndroidTarget.elf_machine` is: the COFF
        #: spelling and the rust triple's are not the same vocabulary.
        self.pe_machine = pe_machine
        #: What `_platform_name` is expected to *derive*, so that the derivation
        #: is checked against something rather than merely trusted. Not the
        #: source of the tag -- if the two disagree, `platform_tag` refuses and
        #: names both, because a distribution answering something other than
        #: this means the registry and the tree on disk have come apart.
        self.expected_tag = expected_tag
        self.rebuild_hint = (
            "PYO3_CROSS_LIB_DIR=<target-python>/libs "
            "PYO3_CROSS_PYTHON_VERSION=3.13 "
            f"cargo xwin build --release --target {rust_target}, from "
            "rust/torch_c (docs/WINDOWS.md §3 has the whole environment, "
            "including the four MSVC tool shims §3.2 installs, which "
            "cargo-xwin needs and this machine does not otherwise have)"
        )

    def sysconfig(self) -> dict[str, object]:
        """Windows CPython ships no `_sysconfigdata_*.py`, and that is correct.

        `sysconfig` only generates that file on POSIX; on Windows the values
        come from `_init_non_posix`, which computes them from `sys.prefix` at
        runtime. So the base class's reader cannot work here, and there is
        nothing equivalent to read -- which is also why `platform_tag` derives
        nothing from the interpreter.
        """
        _fail(
            f"{self.python_root} is a Windows CPython, which ships no "
            "_sysconfigdata_*.py --\n"
            "  sysconfig generates that only on POSIX. Nothing in the Windows "
            "tag needs it;\n"
            "  `_check_distribution` checks the distribution by its files "
            "instead."
        )

    def _check_distribution(self) -> None:
        """Is this the distribution the extension was linked against?

        Both files, not either: `python3.lib` is what the build linked and
        `python3.dll` is what `verify_windows.py` resolves imports against, and
        a distribution missing one of them fails in a different place.
        """
        for relative in (f"libs/{self.ABI3_IMPORT_LIB}", "python3.dll",
                         "python313.dll"):
            if not (self.python_root / relative).exists():
                _fail(
                    f"{self.python_root} has no {relative} -- it is not a "
                    f"{self.pe_machine} Windows CPython\n"
                    "  distribution of the shape this target builds against "
                    "(docs/WINDOWS.md §2)."
                )
        # And it has to be the *right* architecture's, not merely a Windows
        # one. `_check_distribution` above is satisfied by either of the two
        # trees under target-python/, which differ in no filename -- so without
        # this, pointing `--target windows-arm64` at the x86-64 distribution
        # would derive `win_amd64` from it and tag an ARM64 DLL with it.
        # `check_image` would not catch that: it checks the artefact against
        # this target, and the artefact is correct. Same trap as
        # `AndroidTarget._api_and_abi`'s MULTIARCH check.
        info = pe_info((self.python_root / "python313.dll").read_bytes())
        if info is None or info["machine"] != self.pe_machine:
            _fail(
                f"{self.python_root}/python313.dll is "
                f"{describe((self.python_root / 'python313.dll').read_bytes())},"
                f" not a PE {self.pe_machine} image\n"
                f"  -- it is not the Windows CPython distribution --target "
                f"{self.key} builds against."
            )

    def _platform_name(self) -> str:
        """`sysconfig.get_platform()`'s answer for this distribution, unrun.

        See the class docstring for why the tag is this and nothing else. The
        input is the `[MSC v.<n> <bits> bit (<arch>)]` string in the target's
        own `python313.dll`; the branch order below is CPython's, copied from
        the `os.name == "nt"` block of `Lib/sysconfig/__init__.py` **in this
        very distribution** rather than from memory.
        """
        dll = self.python_root / "python313.dll"
        pattern = re.compile(rb"\[MSC v\.\d+ \d+ bit \([A-Za-z0-9]+\)\]")
        found = sorted({m.decode() for m in pattern.findall(dll.read_bytes())})
        if len(found) != 1:
            _fail(
                f"{dll} carries {len(found)} distinct MSVC compiler strings "
                f"({found or 'none'}), and\n"
                "  the Windows wheel tag is `sysconfig.get_platform()`, which "
                "is a test on exactly\n"
                "  that string (see this class's docstring). One is needed, and "
                "one is what a\n"
                "  CPython built by MSVC has. Refused rather than guessed: the "
                "tag is a name, so a\n"
                "  wrong one is not a wrong number but a wheel pip never offers "
                "anybody."
            )
        compiler = found[0].lower()
        for needle, name in self._PLATFORM_NAMES.items():
            if needle in compiler:
                print(f"  tag is a NAME, derived not assumed: {dll.name} says "
                      f"{found[0]},\n"
                      f"      whose {needle!r} is the branch "
                      "`sysconfig.get_platform()` takes on this\n"
                      f"      distribution, so it answers {name!r}")
                return name
        _fail(
            f"{dll} says {found[0]}, which matches none of CPython's Windows "
            f"branches\n  ({list(self._PLATFORM_NAMES)}). "
            "`sysconfig.get_platform()` would fall through to `sys.platform` "
            "and\n  answer 'win32' on a 64-bit machine, which is not a tag this "
            "wheel may carry."
        )

    def platform_tag(self, artefact: bytes) -> str:
        self._check_distribution()
        tag = _confirm_windows_normalisation(self._platform_name())
        if tag != self.expected_tag:
            _fail(
                f"--target {self.key} expects the tag {self.expected_tag!r}, but "
                f"{self.python_root}\n  derives {tag!r}. The registry and the "
                "distribution on disk disagree about which\n  Windows this is; "
                "one of them is wrong and this cannot tell which."
            )
        imports = pe_imports(artefact) or {}
        if "python3.dll" not in imports:
            _fail(
                f"{self.artefact} does not import from python3.dll "
                f"(it imports from {sorted(imports) or 'nothing'}).\n"
                "  An abi3 extension has to bind the stable-ABI forwarder; "
                "binding python313.dll\n"
                "  instead makes the wheel serve 3.13 alone while its abi3 tag "
                "promises 3.13+.\n"
                f"  Fix: link {self.ABI3_IMPORT_LIB}, not python313.lib."
            )
        print(f"  imports {len(imports['python3.dll'])} names from python3.dll "
              "(the abi3 forwarder), so the\n"
              f"      abi3 tag is about the file and not only about the build "
              "flags")
        _confirm_with_packaging(tag, "windows")
        return tag

    def cc(self) -> list[str]:            # pragma: no cover - never reached
        _fail("the Windows wheel carries no global-deps library, so no C "
              "compiler is needed for it (docs/WINDOWS.md §4.3)")

    def check_image(self, data: bytes, what: str) -> None:
        info = pe_info(data)
        if info is None:
            _fail(f"{what} is not a PE image ({describe(data)}) -- "
                  "a Windows wheel cannot carry it")
        if (info["bits"], info["machine"], info["dll"]) != (64, self.pe_machine,
                                                            True):
            _fail(f"{what} is {describe(data)}, expected PE32+ "
                  f"{self.pe_machine} dll")


class PyEmscriptenTarget(Target):
    """`pyemscripten_<abi-version>_wasm32` -- Pyodide, and the one target whose
    tag does not come from a CPython.

    Everything else in this file rests on the principle stated in the header at
    line 55: the tag is *derived from the target CPython distribution* rather
    than written down here. `AndroidTarget` reads `ANDROID_API_LEVEL`,
    `IOSTarget` reads `IPHONEOS_DEPLOYMENT_TARGET`, `LinuxTarget` reads the
    artefact's own `.gnu.version_r`. **The principle survives here and its usual
    implementation does not**, and the gap between those two sentences is the
    whole of docs/WASM.md §9.4a:

        pyodide-lock.json  ->  info.abi_version = "2026_0"     <- the tag
                               info.platform    = "emscripten_5_0_3"
                               info.python      = "3.14.2"

    `2026_0` is Pyodide's `PYODIDE_ABI_VERSION`. CPython's build never sees it,
    so it is in no `_sysconfigdata_*.py`, and it cannot be: it is a property of
    the *runtime distribution*, not of the interpreter build. Ask the
    interpreter the way every other target does and it answers
    `emscripten-5.0.3-wasm32`, which normalises to `emscripten_5_0_3_wasm32` --
    a tag `packaging` accepts, that Pyodide's own `sys_tags()` yields, and that
    **nothing on PyPI is published under**. Wrong in the shape that works.

    That is not a warning, it is a trap: anyone adding a target by copying
    `AndroidTarget` reaches for `self.sysconfig()` first, gets a plausible
    answer, and ships it. So `sysconfig()` below **refuses** rather than
    answering, and names the alternative. A comment would have been read after
    the mistake; an exception is read instead of it. `WindowsTarget.sysconfig`
    refuses for a different reason (the file genuinely does not exist), and this
    is the same shape of guard for the opposite problem -- here the file exists,
    is readable, and is the wrong source.

    The rest is small, which is the finding of docs/WASM.md §9.3: `abi3` is
    inert on this target rather than harmful, so `extension_member` and
    `global_deps_name` are both inherited unchanged. `.abi3.so` really is in
    Pyodide's `EXTENSION_SUFFIXES` and `cp313-abi3-pyemscripten_2026_0_wasm32`
    really is in its `sys_tags()`; this class asserts neither, and
    `verify_cross.py` checks the first against the interpreter's own bytes.
    """

    #: Cargo does not apply the Emscripten `.so` convention to a `cdylib`; the
    #: file it writes is `_C.wasm`, with no `lib` prefix, and it is a wasm side
    #: module rather than an ELF shared object. It becomes `torch/_C.abi3.so`
    #: inside the archive like every other POSIX target's.
    ARTEFACT_NAME = "_C.wasm"

    def __init__(self):
        super().__init__("wasm32-emscripten", "wasm32-unknown-emscripten",
                         PYODIDE_ROOT, self.ARTEFACT_NAME)

    rebuild_hint = (
        "EM_CACHE=/tmp/em-cache-<scratch> "
        "PATH=<emsdk>/upstream/emscripten:$PATH "
        "cargo build --release --target wasm32-unknown-emscripten, from "
        "rust/torch_c (docs/WASM.md §9.6; never write to the shared emsdk "
        "cache -- set EM_CACHE first)"
    )

    def sysconfig(self) -> dict[str, object]:
        """Refuses. The tag is not in there, and the plausible answer is wrong.

        Structural on purpose (docs/WASM.md §9.4a). The file this would read --
        `_sysconfigdata__emscripten_wasm32-emscripten.py`, inside Pyodide's
        `python_stdlib.zip` -- exists and is readable, and every value in it is
        about CPython's build rather than about Pyodide. Its
        `sysconfig.get_platform()` answer is `emscripten-5.0.3-wasm32`. Using it
        would produce a wheel tagged for a platform that is real, accepted by
        `packaging`, and matched by nothing anyone installs.
        """
        _fail(
            "PyEmscriptenTarget.sysconfig() is refused by design.\n"
            "  The pyemscripten tag's version component is Pyodide's "
            "PYODIDE_ABI_VERSION, which\n"
            "  CPython's build never sees. The target's "
            "_sysconfigdata__emscripten_wasm32-emscripten.py\n"
            "  exists (inside python_stdlib.zip) and answers "
            "`emscripten_5_0_3_wasm32` -- a real tag,\n"
            "  accepted by packaging, published by nobody. Read\n"
            f"  {PYODIDE_ROOT / 'pyodide-lock.json'} instead; `platform_tag` "
            "does. docs/WASM.md §9.4a."
        )

    def _lock(self) -> dict:
        """`info` out of `pyodide-lock.json` -- the distribution, still."""
        lock = self.python_root / "pyodide-lock.json"
        if not lock.exists():
            _fail(
                f"no {lock} -- this is the only machine-readable place "
                "PYODIDE_ABI_VERSION exists\n"
                "  (docs/WASM.md §9.4a), so without it the tag cannot be "
                "derived at all. Set\n"
                "  TORCHNATIVE_PYODIDE to a Pyodide distribution."
            )
        try:
            info = json.loads(lock.read_text())["info"]
        except (ValueError, KeyError) as exc:
            _fail(f"{lock} has no readable info block: {exc}")
        for key in ("abi_version", "arch"):
            if not info.get(key):
                _fail(f"{lock} info block has no {key}")
        return info

    def platform_tag(self, artefact: bytes) -> str:
        info = self._lock()
        tag = _normalise(f"pyemscripten_{info['abi_version']}_{info['arch']}")
        # No `_confirm_with_packaging`: there is no `pyemscripten_platforms`
        # generator to ask. `packaging` on the *target* does yield this tag
        # (docs/WASM.md §9.3 read it off the real interpreter), but that path
        # reads Pyodide's own patched `sys.implementation`, which does not exist
        # on this machine. Said out loud rather than skipped silently, the same
        # way `_confirm_with_packaging` reports a `packaging` too old to know a
        # family.
        print(f"  tag {tag} derived from pyodide-lock.json info.abi_version\n"
              f"      (python {info.get('python')}, platform "
              f"{info.get('platform')})")
        print("  ! packaging has no pyemscripten_platforms, so the spelling is "
              "not confirmed\n"
              "      against pip's own generator -- unlike android and ios")
        return tag

    def cc(self) -> list[str]:
        """`emcc`, with the two flags that are the whole difference from Android.

        `-fwasm-exceptions` because docs/WASM.md §7.4a found it mandatory for
        *every* module in the process once any module uses it, side modules
        included. `-sSIDE_MODULE=2` because `ctypes.CDLL` on Emscripten is
        `dlopen`, and `dlopen` cannot take a main-module link -- which is the
        same fact `check_image` enforces for the extension.
        """
        emcc = shutil.which("emcc")
        if not emcc:
            _fail(
                "no emcc on PATH -- cannot build the global-deps side module.\n"
                "  Add <emsdk>/upstream/emscripten to PATH, and set EM_CACHE to "
                "a scratch\n"
                "  directory first: the shared emsdk cache must not be written "
                "to (docs/WASM.md §9.6)."
            )
        if not os.environ.get("EM_CACHE"):
            _fail(
                "EM_CACHE is unset. emcc writes sysroot artefacts into its "
                "cache on first use,\n"
                "  and the emsdk on this machine is shared and must not be "
                "modified. Set\n"
                "  EM_CACHE to a scratch directory (docs/WASM.md §9.6)."
            )
        return [emcc, "-shared", "-fPIC", "-fwasm-exceptions",
                "-sSIDE_MODULE=2"]

    def check_image(self, data: bytes, what: str) -> None:
        info = wasm_info(data)
        if info is None:
            _fail(f"{what} is not a WebAssembly module ({describe(data)}) -- "
                  "a pyemscripten wheel cannot carry it")
        if info["machine"] != "wasm32":
            _fail(f"{what} is {info['machine']}, expected wasm32")
        if not info["side_module"]:
            _fail(
                f"{what} is a main-module link, not a side module: it defines "
                "its own memory\n"
                "  instead of importing one, and Emscripten's dlopen cannot "
                "load it. Build it\n"
                "  with -sSIDE_MODULE (docs/WASM.md §9.2)."
            )


def _confirm_with_packaging(tag: str, family: str, **kwargs) -> None:
    """Ask `packaging` whether an installer would accept this tag.

    The point is not to *derive* the tag from packaging -- its generators need a
    running target interpreter's answers, which is precisely what a cross build
    does not have. It is to check the spelling against the code pip actually
    uses, so that a mistake here is a build failure rather than a wheel that no
    device will match. Skipped, loudly, if `packaging` is too old to know the
    family: that is the state in which the check is worthless, not the state in
    which it passes.
    """
    try:
        from packaging import tags as ptags
    except ImportError:
        print("  ! packaging not importable -- tag spelling unchecked")
        return
    generator = getattr(ptags, f"{family}_platforms", None)
    if generator is None:
        print(f"  ! packaging {getattr(__import__('packaging'), '__version__', '?')}"
              f" has no {family}_platforms -- tag spelling unchecked")
        return
    accepted = list(generator(**kwargs))
    if tag not in accepted:
        _fail(f"packaging.tags.{family}_platforms{kwargs} does not yield {tag!r};"
              f" it starts {accepted[:3]}")
    print(f"  tag {tag} accepted by packaging.tags.{family}_platforms")


#: Why `--target android-x86_64` refuses on this machine. Written out here
#: rather than inline in the registry because it is the one refusal in the file
#: with a *measurement* behind it rather than a missing package, and the
#: measurement is what makes it a refusal instead of a rebuild hint.
#:
#: The blocker is not the compiler. The NDK's `x86_64-linux-android21-clang`
#: is present and runs (its prebuilt directory is `darwin-x86_64`, and Rosetta
#: executes it), the rust target `x86_64-linux-android` is installed, and
#: `cargo ndk` is on PATH. What is missing is the **target CPython**: every
#: other target in this file reads its tag out of one, and
#: `target-python/x86_64-linux-android/` does not exist.
#:
#: It cannot be downloaded. python-build-standalone's 20260825 release -- the
#: source of the four fetchable distributions in docs/TARGET_PYTHON.md --
#: publishes 871 assets and no Android target among them (checked, not
#: assumed). CPython publishes no Android binaries either. The only route is a
#: local cross-build, which is what docs/TARGET_PYTHON.md §5 records for the
#: aarch64 one, and §5 also records that that build is the one distribution of
#: the five whose provenance could not be reconstructed.
#:
#: And the wheel could not be checked if it were built. docs/WHEELMATRIX.md
#: §3.3 has the measurement: this Mac's emulator ships only
#: `emulator/qemu/darwin-aarch64`, so it runs aarch64 guests and nothing else,
#: and all four installed system images are `arm64-v8a`. So `verify_android.py`
#: -- the one script in the repository that *executes* a cross wheel on the
#: target -- has no machine to execute an x86-64 one on, here or on any Apple
#: Silicon host.
_ANDROID_X86_64_REFUSAL = (
    "no x86_64-linux-android CPython distribution, and none can be obtained "
    "here.\n"
    "  The toolchain is present and is not the problem: the NDK's "
    "x86_64-linux-android21-clang\n"
    "  runs (darwin-x86_64 prebuilts, under Rosetta), the rust target "
    "x86_64-linux-android is\n"
    "  installed, and cargo-ndk is on PATH. The target CPython is the gap, and "
    "every tag in\n"
    "  this file is derived from one.\n"
    "  * It is not downloadable. python-build-standalone 20260825 publishes 871 "
    "assets and no\n"
    "    Android target; CPython publishes no Android binaries at all.\n"
    "  * A local cross-build is the only route, and docs/TARGET_PYTHON.md §5 "
    "records what that\n"
    "    bought last time: the aarch64 Android distribution is the one of five "
    "whose source\n"
    "    could not be reconstructed afterwards.\n"
    "  * The result could not be checked. This machine's emulator ships only\n"
    "    emulator/qemu/darwin-aarch64 and all four installed system images are "
    "arm64-v8a, so\n"
    "    verify_android.py -- the only script here that *runs* a cross wheel -- "
    "has no x86-64\n"
    "    Android to run it on, and no Apple Silicon host has one "
    "(docs/WHEELMATRIX.md §3.3).\n"
    "  Refused by name rather than dropped from the registry, so this reads as "
    "a target that\n"
    "  was thought about -- the reason `--target linux-x86_64` was listed while "
    "it refused."
)

#: Every target `--target` offers. The registry is the API: a key here is a
#: `--target` choice, and a target that refuses stays in it with its reason
#: (see `Target.refusal`) rather than disappearing.
TARGETS: dict[str, Target] = {
    t.key: t for t in (
        AndroidTarget(),
        AndroidTarget("android-x86_64", "x86_64-linux-android", "x86_64",
                      refusal=_ANDROID_X86_64_REFUSAL),
        IOSTarget("ios-arm64", "aarch64-apple-ios", "arm64-iphoneos",
                  "iphoneos", "ios"),
        IOSTarget("ios-arm64-sim", "aarch64-apple-ios-sim",
                  "arm64-iphonesimulator", "iphonesimulator", "iossimulator"),
        LinuxTarget(),
        LinuxTarget("linux-aarch64", "aarch64-unknown-linux-gnu", "aarch64"),
        WindowsTarget(),
        WindowsTarget("windows-arm64", "aarch64-pc-windows-msvc", "aarch64",
                      "win_arm64"),
        PyEmscriptenTarget(),
    )
}

#: The registry's keys, written down a second time on purpose.
#:
#: `TARGETS` is a dict comprehension keyed on `t.key`, so two targets that
#: disagree about their own key -- a copied constructor call with the key
#: argument not updated, which is exactly how the second Android, Linux and
#: Windows entries were written -- collapse into one silently. The dict would
#: have eight entries instead of nine, `--target` would offer eight choices,
#: and nothing would say which one had been eaten: the surviving entry is a
#: valid target that builds a valid wheel, just not the one that was asked for.
#:
#: Listed rather than counted, so the failure names the missing key instead of
#: reporting an arithmetic disagreement.
EXPECTED_TARGET_KEYS = (
    "android-arm64-v8a", "android-x86_64",
    "ios-arm64", "ios-arm64-sim",
    "linux-aarch64", "linux-x86_64",
    "wasm32-emscripten",
    "windows-arm64", "windows-x86_64",
)


def check_registry() -> None:
    """Refuse to run if the registry lost or gained an entry. See above."""
    have, want = sorted(TARGETS), sorted(EXPECTED_TARGET_KEYS)
    if have != want:
        lost = [k for k in want if k not in have]
        extra = [k for k in have if k not in want]
        _fail(
            "the target registry does not hold what it is declared to hold.\n"
            + (f"  missing: {lost}\n" if lost else "")
            + (f"  unexpected: {extra}\n" if extra else "")
            + "  A missing key is usually two TARGETS entries sharing one "
            "`key`, which the dict\n"
            "  comprehension silently collapses -- see EXPECTED_TARGET_KEYS. "
            "Fix the entry, or\n"
            "  update EXPECTED_TARGET_KEYS if the target was removed on "
            "purpose."
        )

#: Prefixes `verify` will accept as a cross tag. One entry per target family, so
#: that adding a family and forgetting this is a build failure rather than a
#: wheel tagged for one platform and checked as another.
CROSS_TAG_PREFIXES = ("android_", "ios_", "manylinux_", "win_",
                      "pyemscripten_")


def _repack(wheel: Path, extra: dict[str, bytes], dist_info: str,
            plat: str | None = None,
            overrides: dict[str, bytes] | None = None,
            renames: dict[str, str] | None = None) -> Path:
    """Rewrite the archive with `extra` added and RECORD regenerated.

    zipfile cannot delete or replace a member, so the whole archive is rebuilt.
    RECORD has to be regenerated anyway: it is a hash manifest, and pip verifies
    it on install, so appending files without touching it produces a wheel that
    fails at exactly the moment it looks like it worked.

    `plat` forces the platform tag (cross builds know theirs; the host derives
    it below from the extension). `overrides` replaces the content of members
    that are already there -- which is how a cross wheel gets the cross `_C`
    without the source tree ever holding it. `renames` moves a member, which
    only Windows needs: its interpreter has no `.abi3.so` in
    `_PyImport_DynLoadFiletab`, so the same bytes have to arrive as `_C.pyd`
    (`Target.extension_member`).
    """
    tmp = wheel.with_suffix(".whl.tmp")
    record_name = f"{dist_info}/RECORD"
    wheel_name = f"{dist_info}/WHEEL"
    rows: list[tuple[str, str, str]] = []
    overrides = overrides or {}
    renames = renames or {}
    old_plat = wheel.stem.split("-")[-1]
    new_plat: str | None = plat if plat and plat != old_plat else None

    # The architecture correction has to be known before WHEEL is written, and
    # it comes out of the extension, so read that first. Only for the host: a
    # cross target was told its tag and must not have it second-guessed by a
    # macOS-shaped rule.
    if plat is None:
        with zipfile.ZipFile(wheel) as src:
            so = next(
                (n for n in src.namelist() if n.endswith("torch/_C.abi3.so")), None
            )
            if so is not None and sys.platform == "darwin":
                so_data = src.read(so)
                honest = honest_macos_plat(old_plat, so_data)
                if honest != old_plat:
                    print(f"  retag: {old_plat} -> {honest} "
                          f"(extension is {'+'.join(macho_arches(so_data))})")
                    new_plat = honest
    elif new_plat:
        print(f"  retag: {old_plat} -> {new_plat}")

    unused = (set(overrides) | set(renames)) - set(zipfile.ZipFile(wheel).namelist())
    if unused:
        # An override that matches nothing is the wheel keeping its host
        # extension while everything else says it is a cross wheel -- silent,
        # and exactly the failure this whole path exists to avoid.
        _fail(f"override(s) name members the wheel does not have: {sorted(unused)}")

    with zipfile.ZipFile(wheel) as src, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.infolist():
            if item.filename == record_name:
                continue  # regenerated at the end
            data = overrides.get(item.filename) or src.read(item.filename)
            if item.filename.endswith("torch/_C.abi3.so"):
                data = _fix_install_name(data)
                name = renames.get(item.filename, item.filename)
                item = zipfile.ZipInfo(name, date_time=item.date_time)
                item.external_attr = 0o755 << 16
                item.compress_type = zipfile.ZIP_DEFLATED
            elif item.filename == wheel_name and new_plat:
                # `Tag:` in WHEEL and the filename have to agree; installers read
                # the filename, `wheel unpack` and auditors read this.
                data = b"".join(
                    (line.rsplit(b"-", 1)[0] + b"-" + new_plat.encode() + b"\n")
                    if line.startswith(b"Tag: ") else line + b"\n"
                    for line in data.splitlines()
                )
            dst.writestr(item, data)
            rows.append(_record_line(item.filename, data))

        for arcname, data in sorted(extra.items()):
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(info, data)
            rows.append(_record_line(arcname, data))

        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for row in sorted(rows):
            writer.writerow(row)
        writer.writerow([record_name, "", ""])
        dst.writestr(record_name, buf.getvalue())

    final = wheel if new_plat is None else wheel.with_name(
        _retag(wheel.name, new_plat)
    )
    tmp.replace(final)
    if final != wheel:
        wheel.unlink()
    return final


# A translation unit with one symbol, so that `nm` on the result says what it is
# instead of showing an empty library and leaving the next reader to guess.
_GLOBAL_DEPS_C = """
/* Generated by tools/wheel/build.py. See docs/WHEEL.md. */
const char torchnative_global_deps_note[] =
    "torchnative: this build has no global native dependencies; "
    "torch/__init__.py:_load_global_deps only needs this file to dlopen.";
"""


def global_deps_stub(target: "Target | None" = None) -> dict[str, bytes]:
    """An empty `torch/lib/libtorch_global_deps` so `import torch` needs no env.

    Wall 1 in docs/VENDOR.md: `torch/__init__.py:_load_global_deps()` does an
    unconditional `ctypes.CDLL(torch/lib/libtorch_global_deps.<ext>,
    RTLD_GLOBAL)`, and `vendor_torch.sh` drops `torch/lib/` entirely. Every
    invocation in this repository routes around it with
    `TORCH_USE_RTLD_GLOBAL=1`, which upstream provides for exactly this case
    ("a build environment where libtorch_global_deps is not available").

    That is fine for a source tree and unacceptable for a wheel. `pip install
    torchnative && python -c "import torch"` has to work, and an environment
    variable the user has never heard of is not a thing they will set. Worse,
    the wall-1 branch is not free: it also does
    `sys.setdlopenflags(RTLD_GLOBAL | RTLD_LAZY)` for the whole `from torch._C
    import *`, whose device-side effect VENDOR.md §7 item 4 lists as unverified.
    Shipping the file takes the ordinary branch instead.

    Upstream's copy exists to force MKL/OpenMP into the global namespace ahead
    of libtorch. This build links neither -- `_C.abi3.so` is self-contained
    against Accelerate -- so *empty* is not a stub standing in for something
    real, it is the correct content. The file has to be a loadable image
    though, not the zero-byte marker that `install_shim.sh` uses for
    `torch/bin/torch_shm_manager`; `CDLL` actually dlopens this one.

    Built here rather than in `install_shim.sh` so the source tree keeps
    behaving exactly as every existing doc describes -- wall 1 still stands
    there, and the suites that set the variable are still measuring what they
    say they measure.

    For a cross target the compiler is the target's, not the host's. Handing the
    host `cc` an Android wheel puts a Mach-O where the device expects an ELF;
    `CDLL` then raises OSError, `_load_global_deps` swallows it into the
    `_preload_cuda_deps` path, and the import dies somewhere with no mention of
    this file. The name changes too -- see `Target.global_deps_name`.
    """
    import tempfile

    if target is not None and target.global_deps_name is None:
        # Not "we could not build one" -- there is nothing for it to do. Said
        # out loud, because a wheel silently missing this file is the exact
        # failure this function exists to prevent on every other platform.
        print("  + no torch/lib/ global-deps library: _load_global_deps() "
              "returns immediately on\n"
              "      Windows, and _load_dll_libraries() would LoadLibrary an "
              "empty one for nothing\n"
              "      (docs/WINDOWS.md §4.3)")
        return {}

    if target is None:
        name = "libtorch_global_deps" + (
            ".dylib" if sys.platform == "darwin" else ".so")
        cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
        if not cc:
            _fail(
                "no C compiler (set CC) -- cannot build the empty "
                f"torch/lib/{name}, without which the installed "
                "wheel cannot `import torch` unless the user sets "
                "TORCH_USE_RTLD_GLOBAL=1 (docs/VENDOR.md wall 1)"
            )
        argv = [cc, "-shared", "-fPIC"]
        # Match the deployment target the wheel will be *tagged* with. Without
        # this the host `cc` stamps LC_BUILD_VERSION with the SDK it happens to
        # have -- measured at `macos 26.0+` inside a wheel tagged
        # `macosx_11_0_arm64` -- and dyld enforces that field, so the file the
        # tag promises would be unloadable on every macOS the tag claims. Same
        # class of mistake as the `universal2` tag in docs/WHEEL.md §3.3: the
        # archive contents and the tag disagreeing, in the direction that only
        # shows up on somebody else's machine.
        if sys.platform == "darwin":
            import sysconfig
            floor = sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")
            if floor:
                argv.append(f"-mmacosx-version-min={floor}")
    else:
        name = target.global_deps_name
        argv = target.cc()

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "global_deps.c"
        out = Path(tmp) / name
        src.write_text(_GLOBAL_DEPS_C)
        r = subprocess.run([*argv, "-o", str(out), str(src)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            _fail(f"{argv[0]} could not build the global-deps stub:\n{r.stderr}")
        data = out.read_bytes()
        # The compiler was asked for a target; check that it produced one. A
        # wrong `--sdk` or a host `cc` on $PATH is otherwise invisible until the
        # device refuses the file.
        if target is not None:
            target.check_global_deps(data)
        print(f"  + torch/lib/{name} "
              f"({len(data):,} B, empty by design -- VENDOR.md wall 1)"
              f"\n      {describe(data)}")
        return {f"torch/lib/{name}": data}


def upstream_dist_info(version: str) -> dict[str, bytes]:
    """Upstream's `torch-<v>.dist-info`, addressed into the wheel's purelib.

    Why it cannot simply sit at the archive root: pip resolves a wheel's own
    metadata directory by scanning top-level names ending in `.dist-info` and
    raising `UnsupportedWheel: multiple .dist-info directories found` if there
    is more than one. Routing it through `torchnative-<v>.data/purelib/` puts it
    in site-packages at install time while leaving exactly one at the root.

    Upstream's RECORD is deliberately not copied (vendor_torch.sh drops it too).
    Ours lists these files, so `pip uninstall torchnative` removes them; a
    RECORD of its own would additionally invite `pip uninstall torch` to delete
    the tree out from under us.
    """
    found = sorted(SRC.glob("torch-*.dist-info"))
    if not found:
        # Refused, not merely noted: `verify()`'s missing/unexpected checks
        # cannot catch this on their own. `expected` is a walk of *package*
        # directories (`tree_files(SRC / pkg)` in `main()`) and this
        # directory sits at `SRC`'s top level, so `expected` never names any
        # of its files; `extra` is exactly what this function returns, so an
        # empty `{}` here means `extra` never names them either. A wheel
        # missing this directory would therefore build and `verify()` clean
        # -- `importlib.metadata.version('torch')` would raise on it and
        # `transformers.is_torch_available()` would answer False, with
        # nothing above having refused to ship that.
        _fail(
            f"no torch-*.dist-info under {SRC}\n"
            "  vendor_torch.sh writes this as part of vendoring; without it "
            "the wheel would\n  ship without upstream's own dist-info -- "
            "importlib.metadata.version('torch') raises\n  and "
            "transformers' is_torch_available() answers False, silently, "
            "for anyone\n  who installs it. Re-run bash vendor/vendor_torch.sh."
        )
    root = found[0]
    prefix = f"torchnative-{version}.data/purelib/{root.name}"
    out: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "RECORD":
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out[f"{prefix}/{path.relative_to(root).as_posix()}"] = path.read_bytes()
    return out


def verify(wheel: Path, expected: set[str], target: "Target | None",
          extra: dict[str, bytes], dist_info: str) -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        if target is not None and target.extension_member != Target.extension_member:
            # The source tree always holds the host shim under the POSIX name;
            # this target's wheel holds the same slot under another one. Rewrite
            # the expectation rather than dropping it, so the member is still
            # required -- just under the name that will actually be searched for.
            expected = {target.extension_member if n == Target.extension_member
                        else n for n in expected}
        missing = sorted(n for n in expected if n not in names)
        if missing:
            head = "\n".join("    " + m for m in missing[:20])
            _fail(
                f"{len(missing)} file(s) present in {SRC} but absent from the "
                f"wheel:\n{head}\n"
                + ("    ...\n" if len(missing) > 20 else "")
                + "  a wheel that is merely smaller than the tree installs fine "
                "and\n  fails later, at whichever import first needs the "
                "missing file"
            )
        if "-none-any" in wheel.name:
            _fail(f"{wheel.name} is tagged py3-none-any -- setup.py's "
                  "BinaryDistribution did not take effect")
        if "-abi3-" not in wheel.name:
            _fail(f"{wheel.name} is not abi3-tagged -- the bdist_wheel "
                  "py_limited_api option did not take effect")

        # The other direction: a member nobody asked for. `missing` above is
        # "smaller than it should be"; this is "bigger than it should be", and
        # a wheel that ships thousands of legitimate vendored files makes an
        # allow-list of *names* the wrong shape for it -- there is nothing to
        # hand-maintain against. What is right-shaped is a comparison against
        # what the build actually intended to put there: `expected` (the same
        # source-tree walk `missing` was just checked against -- every
        # legitimate package file is already named by it) and `extra` (the
        # exact files *this build* generated: the global-deps stub,
        # upstream's relocated dist-info). The one thing genuinely open-ended
        # is this wheel's own `<name>-<version>.dist-info/`, which setuptools
        # writes and whose member list is not this script's to enumerate --
        # so that is the one exemption, granted by prefix rather than by name.
        #
        # This is what would have caught .DS_Store in all six wheels on
        # 2026-08-30: a stale `build/lib.../` setuptools cache held a copy
        # that was in neither the source tree (so `expected` never named it)
        # nor anything this build generated (so `extra` never named it either)
        # -- it was simply unaccounted for, and nothing before this check had
        # a notion of "unaccounted for" at all.
        known = set(expected) | set(extra)
        prefix = f"{dist_info}/"
        unexpected = sorted(
            n for n in names if n not in known and not n.startswith(prefix))
        if unexpected:
            head = "\n".join("    " + u for u in unexpected[:20])
            _fail(
                f"{len(unexpected)} file(s) in {wheel.name} that neither the "
                f"source tree nor this build's own additions account for:\n"
                f"{head}\n"
                + ("    ...\n" if len(unexpected) > 20 else "")
                + "  a wheel that gained content nobody asked for is exactly "
                "as unverified as\n  one that is missing content -- see the "
                ".DS_Store incident in docs/WHEEL.md"
            )

        if target is None:
            return

        # Read the finished archive rather than trusting that the override and
        # the extra went in. This is the only place that looks at what is
        # actually in the file that will be published.
        plat = wheel.stem.split("-")[-1]
        member = next(
            (n for n in names if n.endswith(target.extension_member)), None)
        if member is None:
            _fail(f"{wheel.name} has no {target.extension_member} -- the "
                  "extension is either missing or under a name this target's "
                  "interpreter does not search")
        target.check_image(zf.read(member), f"{wheel.name}::{member}")
        if target.extension_member != Target.extension_member:
            # The rename is the whole point on Windows, so a wheel carrying both
            # names is the rename half-applied and the interpreter finding the
            # wrong file first.
            stale = [n for n in names if n.endswith(Target.extension_member)]
            if stale:
                _fail(f"{wheel.name} carries {stale} as well as {member}")
        if target.global_deps_name is None:
            strays = sorted(
                n for n in names
                if n.startswith("torch/lib/libtorch_global_deps"))
            if strays:
                _fail(f"{wheel.name} carries {strays}, but this target loads no "
                      "global-deps library at all (docs/WINDOWS.md §4.3)")
        else:
            deps = f"torch/lib/{target.global_deps_name}"
            if deps not in names:
                _fail(f"{wheel.name} has no {deps}")
            target.check_image(zf.read(deps), f"{wheel.name}::{deps}")
            strays = sorted(
                n for n in names
                if n.startswith("torch/lib/libtorch_global_deps") and n != deps
            )
            if strays:
                # `_load_global_deps` looks for exactly one name. A second one is
                # a host artefact that came along for the ride.
                _fail(f"{wheel.name} carries {strays} beside {deps}")
        if not plat.startswith(CROSS_TAG_PREFIXES):
            _fail(f"{wheel.name} was built for {target.key} but is tagged "
                  f"{plat!r}, which is none of {CROSS_TAG_PREFIXES}")


def self_test() -> None:
    """Drive `artefact_verdict` through all three of its answers.

    A freshness check has one way to be useless and it is not being wrong: it
    is answering "fresh" for a reason that has nothing to do with the artefact.
    Six of the eight cases below are the check *failing to run*, and each has to
    come back `unknown` rather than `fresh` -- particularly `no inputs listed`,
    where `max()` over an empty list has no error to raise and would simply
    conclude that nothing is newer than the artefact.

    Costs no build: the artefact is a scratch file with a chosen mtime and the
    dep-info is written by hand. The prerequisites are real files under
    `rust/torch_c`, because the containment rule is one of the things under
    test.
    """
    import tempfile

    real = sorted(str(p) for p in (CRATE / "src").glob("*.rs"))
    if len(real) < 2:
        _fail(f"self-test needs source files under {CRATE / 'src'}; found {real}")
    hour = 3600 * 10**9

    cases: list[tuple[str, str, str]] = []   # (label, expected, fragment)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        def scenario(name: str, dep_text: str | None, *, artefact_age_h: float
                     ) -> tuple[str, str]:
            art = tmpdir / f"{name}.dylib"
            art.write_bytes(b"\xcf\xfa\xed\xfe" + b"\0" * 60)
            when = art.stat().st_mtime_ns - int(artefact_age_h * hour)
            os.utime(art, ns=(when, when))
            dep = art.with_suffix(".d")
            if dep_text is not None:
                dep.write_text(dep_text.replace("@ART@", str(art)))
            return artefact_verdict(art)

        checks: list[tuple[str, str, str, str | None, float]] = [
            # label, expected verdict, expected phrase, dep-info, artefact age
            ("current artefact", FRESH, "recorded inputs",
             "@ART@: " + " ".join(real[:2]), 0.0),
            ("a prerequisite modified after the build", STALE,
             "was modified", "@ART@: " + " ".join(real[:2]), 48.0),
            ("no dep-info at all", UNKNOWN, "does not exist", None, 0.0),
            ("dep-info with no rule in it", UNKNOWN, "no Makefile rule",
             "not a makefile\njust prose\n", 0.0),
            ("dep-info describing a different artefact", UNKNOWN,
             "some other build's record",
             "/elsewhere/lib_Other.dylib: " + real[0], 0.0),
            ("rule with no prerequisites", UNKNOWN, "lists no inputs",
             "@ART@:", 0.0),
            ("a prerequisite that is gone", UNKNOWN, "cannot be stat'ed now",
             f"@ART@: {CRATE}/src/deleted_by_the_test.rs", 0.0),
            ("prerequisites from another checkout", UNKNOWN,
             "different checkout",
             "@ART@: /some/other/worktree/rust/torch_c/src/lib.rs", 0.0),
        ]

        print(f"SELF-TEST of the artefact freshness check ({len(checks)} cases)")
        bad = 0
        for i, (label, expected, phrase, dep_text, age) in enumerate(checks):
            verdict, detail = scenario(f"case{i}", dep_text, artefact_age_h=age)
            ok = verdict == expected and phrase in detail
            bad += not ok
            print(f"  {'ok    ' if ok else 'WRONG '}{verdict:<8}{label}")
            if not ok:
                print(f"          expected {expected} containing {phrase!r}")
                print(f"          got      {verdict}: {detail}")
            cases.append((label, verdict, detail))

    # The verdict is internal; what reaches a reader is the refusal message. The
    # trap this shape exists to avoid is a check reporting its own failure as a
    # finding about the artefact, so assert on the rendered text of both kinds
    # -- a `stale` message has to make the staleness claim and an `unknown` one
    # has to disclaim it, in so many words.
    counts = {FRESH: 0, STALE: 0, UNKNOWN: 0}
    for _, verdict, _ in cases:
        counts[verdict] += 1
    if not (counts[STALE] and counts[UNKNOWN] and counts[FRESH]):
        _fail(f"SELF-TEST: the cases do not reach all three verdicts: {counts}")
    for verdict, phrase, must_not in (
        (STALE, "is stale.", "NOT a staleness report"),
        (UNKNOWN, "This is NOT a staleness report", "is stale."),
    ):
        for label, got, detail in cases:
            if got != verdict:
                continue
            rendered = _render_refusal(verdict, Path("/scratch/x.dylib"), detail,
                                       "REBUILD-HINT")
            if phrase not in rendered or must_not in rendered:
                _fail(f"SELF-TEST: the {verdict} message for {label!r} does not "
                      f"read as one -- it must contain {phrase!r} and must not "
                      f"contain {must_not!r}:\n{rendered}")
    print(f"\n  {counts[STALE]} staleness report(s), "
          f"{counts[UNKNOWN]} 'cannot judge', {counts[FRESH]} clean -- and the "
          "two refusal texts assert they are not each other")
    if bad:
        sys.exit(f"SELF-TEST: FAIL -- {bad}/{len(cases)} cases answered wrongly")
    print(f"SELF-TEST: PASS -- {len(cases)}/{len(cases)} cases answered as "
          "specified")


def _minimal_elf(machine: int = 0x3E, etype: int = 3,
                 with_sections: bool = True) -> bytes:
    """A 64-bit little-endian ELF with nothing in it. For the self-test.

    Hand-built rather than compiled, because the point of the cases that use it
    is what happens when an image carries *no* version requirements, and this
    machine cannot produce such a file through the normal route (docs/LINUX.md
    §2.7 builds one, but only with tools that a self-test may not assume).
    """
    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = 2                                     # ELFCLASS64
    ehdr[5] = 1                                     # ELFDATA2LSB
    ehdr[6] = 1                                     # EV_CURRENT
    struct.pack_into("<HH", ehdr, 16, etype, machine)
    if not with_sections:
        return bytes(ehdr)
    # Two section headers: the mandatory null one and a shstrtab holding "\0".
    shoff = 64
    struct.pack_into("<Q", ehdr, 0x28, shoff)
    struct.pack_into("<HHH", ehdr, 0x3A, 64, 2, 1)  # shentsize, shnum, shstrndx
    null = bytes(64)
    shstr = bytearray(64)
    struct.pack_into("<IIQQQQIIQQ", shstr, 0,
                     0, 3, 0, 0, shoff + 128, 1, 0, 0, 1, 0)
    return bytes(ehdr) + null + bytes(shstr) + b"\0"


def self_test_linux() -> int:
    """The Linux tag derivation, against real Linux ELF images.

    Separate from `self_test` and reported separately, because it answers a
    different question and can be *skipped* -- it reads files out of the target
    CPython distribution, which is not in the repository. A skip is printed as a
    skip; it never counts as a pass. `_confirm_with_packaging` has the same
    shape and the same reason.

    What is real here and what is not, stated plainly: the ELF parsing, the
    glibc floor and the policy check run against genuine Linux x86-64 shared
    objects that were built by somebody's real toolchain. Cases 1-6 take those
    out of the target CPython distribution, so they exercise the derivation and
    not the wheel -- and they are circular in one respect, since the thing they
    check is what this same parser read.

    Case 7 is the one that is not. It runs on this crate's own cross-built
    `lib_C.so` and asserts that the glibc version handed to `cargo zigbuild`
    comes back out of the derivation, which is an input to another tool compared
    against an output of this one. It only exists because the artefact became
    buildable (docs/LINUX.md §9.2); before that it is skipped, loudly.
    """
    target = TARGETS["linux-x86_64"]
    root = target.python_root
    fixtures = {
        # (path under the distribution, why it is here)
        "clean": root / "lib" / "thread3.0.6" / "libtcl9thread3.0.6.so",
        "outside_policy": root / "lib" / "libpython3.so",
    }

    print("\nLINUX SELF-TEST of the manylinux tag derivation")
    missing = sorted(str(p) for p in fixtures.values() if not p.exists())
    if missing:
        print("  ! skipped -- the target CPython distribution is not on this")
        print(f"    machine ({root} does not hold {len(missing)} fixture(s)).")
        print("    This is a SKIP, not a pass: the derivation is unexercised.")
        return 0

    def refusal(fn, *args) -> str:
        try:
            fn(*args)
        except SystemExit as exc:
            return str(exc)
        return ""

    checks: list[tuple[str, bool, str]] = []

    # 1. A real Linux x86-64 shared object, inside the policy list, gives a
    #    floor that comes from its own `.gnu.version_r`. libtcl9thread requires
    #    GLIBC_2.2.5, 2.3.4, 2.4, 2.7 and 2.14, and the answer is the highest.
    #
    #    2.14 rather than 2.7 is not a detail: sorting these names as *strings*
    #    puts "GLIBC_2.7" last, and this expectation said 2.7 until the test was
    #    run. A floor one that is too low is a wheel that installs on a glibc
    #    which cannot load it, so the ordering has to be numeric and it has to be
    #    checked on a fixture whose versions straddle the ten boundary.
    clean = fixtures["clean"].read_bytes()
    floor = target._glibc_floor(clean)
    checks.append((
        f"glibc floor read off a real ELF -> {floor[0]}.{floor[1]}",
        floor == (2, 14),
        f"expected (2, 14) from {fixtures['clean'].name} -- its highest "
        f"requirement, which is not its lexicographically last -- got {floor}",
    ))
    # ...and the interpreter is genuinely not the source. This is the premise of
    # the whole class, so it is asserted rather than assumed: if a future
    # distribution grows a glibc field, the derivation should be revisited and
    # this is what will say so. (`HAVE_GLIBC_MEMMOVE_BUG` is in there and is a
    # feature probe, not a floor -- hence matching on the *value* shape too.)
    variables = target.sysconfig()
    versioned = sorted(k for k, v in variables.items()
                       if isinstance(v, str) and v.startswith("GLIBC_"))
    checks.append((
        "the target CPython records no glibc minimum to derive from",
        not versioned
        and int(variables.get("ANDROID_API_LEVEL") or 0) == 0
        and not variables.get("IPHONEOS_DEPLOYMENT_TARGET"),
        f"expected none; GLIBC_-valued keys={versioned}, ANDROID_API_LEVEL="
        f"{variables.get('ANDROID_API_LEVEL')!r}, IPHONEOS_DEPLOYMENT_TARGET="
        f"{variables.get('IPHONEOS_DEPLOYMENT_TARGET')!r}",
    ))

    # 2. The image passes the PEP 599 external-library policy, and the tag it
    #    yields is one PEP 600 defines.
    needed = target._check_policy(clean)
    checks.append((
        f"DT_NEEDED {needed} accepted by the PEP 599 list",
        needed == ["libc.so.6"],
        f"expected ['libc.so.6'], got {needed}",
    ))
    tag = f"manylinux_{floor[0]}_{floor[1]}_x86_64"
    spelling = refusal(_confirm_pep600_spelling, tag, "x86_64")
    checks.append((
        f"{tag} accepted as PEP 600-shaped",
        spelling == "",
        f"refused: {spelling}",
    ))
    # ...and a floor below manylinux1's is refused, because no installer matches
    # it. Without this the spelling check only ever sees values that pass.
    spelling = refusal(_confirm_pep600_spelling, "manylinux_2_4_x86_64", "x86_64")
    checks.append((
        "a floor below manylinux1's glibc 2.5 is refused",
        "below manylinux1's 2.5" in spelling,
        f"got: {spelling[:200]!r}",
    ))
    # 2b. The floor is **per architecture**, and this is the case that says so.
    #     `manylinux_2_12_aarch64` is PEP 600-shaped, is not below manylinux1's
    #     2.5, and matches nothing: aarch64 enters the scheme at manylinux2014,
    #     so pip's generator floors it at 2.17. Checking only the grammar --
    #     which is all this did while x86-64 was the only Linux target, and was
    #     correct for exactly that one architecture -- would let it out.
    spelling = refusal(_confirm_pep600_spelling, "manylinux_2_12_aarch64",
                       "aarch64")
    checks.append((
        "manylinux_2_12_aarch64 is refused though 2.12 is fine on x86-64",
        "below manylinux2014's 2.17" in spelling,
        f"got: {spelling[:200]!r}",
    ))
    checks.append((
        "...and the same floor is accepted on x86_64, so it is the arch and "
        "not the number",
        refusal(_confirm_pep600_spelling, "manylinux_2_12_x86_64", "x86_64") == "",
        "manylinux_2_12_x86_64 was refused",
    ))
    # 2c. And a tag whose arch is not the one the target builds. The two Linux
    #     targets share this function, so a copied call passing the wrong arch
    #     is a real way to tag an aarch64 wheel `..._x86_64`.
    spelling = refusal(_confirm_pep600_spelling, "manylinux_2_17_x86_64",
                       "aarch64")
    checks.append((
        "a tag naming another architecture than the target's is refused",
        "names architecture 'x86_64'" in spelling,
        f"got: {spelling[:200]!r}",
    ))
    # 2d. pip's own generator, asked the cross question directly. This is what
    #     `_confirm_manylinux_with_packaging` exists for, and the two archs are
    #     both run so the per-arch floor above is confirmed by packaging's code
    #     rather than only by this file's reading of PEP 599.
    for probe, arch, want_ok in (("manylinux_2_17_aarch64", "aarch64", True),
                                 ("manylinux_2_17_x86_64", "x86_64", True),
                                 ("manylinux_2_12_aarch64", "aarch64", False)):
        said = refusal(_confirm_manylinux_with_packaging, probe, arch)
        checks.append((
            f"packaging._manylinux "
            f"{'yields' if want_ok else 'does not yield'} {probe}",
            (said == "") is want_ok,
            f"got: {said[:200]!r}",
        ))

    # 3. An image that links something off the list is refused *by name*. This
    #    one links `$ORIGIN/../lib/libpython3.13.so.1.0`, which is exactly the
    #    kind of private dependency the manylinux promise excludes.
    said = refusal(target._check_policy, fixtures["outside_policy"].read_bytes())
    checks.append((
        "an off-policy DT_NEEDED is refused, naming it",
        "libpython3.13.so.1.0" in said and "PEP 599" in said,
        f"got: {said[:200]!r}",
    ))

    # 4. The docs/LINUX.md §2.6 trap. An ELF with no version requirements is the
    #    check having no answer, and must not read as a low floor -- that is the
    #    same fold-two-outcomes-into-one mistake `artefact_verdict` exists to
    #    avoid, in the direction that silently passes.
    said = refusal(target._glibc_floor, _minimal_elf())
    checks.append((
        "no version requirements -> refused, and disclaimed as a finding",
        "records no glibc symbol versions" in said and "NOT a claim" in said,
        f"got: {said[:200]!r}",
    ))

    # 5. Bytes that are not a readable ELF are a third outcome again.
    said = refusal(target._glibc_floor, b"\xcf\xfa\xed\xfe" + b"\0" * 60)
    checks.append((
        "unreadable image -> refused as unavailable, not as a floor",
        "could not be read" in said,
        f"got: {said[:200]!r}",
    ))

    # 5b. `_check_policy` makes the same distinction on the same kind of
    #    input. Before this was enforced, unreadable bytes fed to
    #    `_check_policy` came back `needed == []` -- policy-compliant,
    #    silently -- because `elf_dynamic(artefact) or {"needed": []}`
    #    could not tell "no libraries" from "could not read the libraries".
    #    `platform_tag` happens to call `_glibc_floor` right after on the
    #    same bytes and that call refuses, so this was masked in the only
    #    call site that exists today -- exercised directly here so it is not
    #    left depending on that call order.
    said = refusal(target._check_policy, b"\xcf\xfa\xed\xfe" + b"\0" * 60)
    checks.append((
        "_check_policy on unreadable bytes refuses, not 'needed == []'",
        "could not be read" in said,
        f"got: {said[:200]!r}",
    ))

    # 6. `check_image` keeps the wrong machine out. An aarch64 ELF is the one
    #    that would otherwise sail through -- it is an ELF, it is 64-bit, and it
    #    is a shared object.
    said = refusal(target.check_image, _minimal_elf(machine=0xB7), "an aarch64 ELF")
    checks.append((
        "an aarch64 ELF is refused by the x86-64 target",
        "expected ELF 64-bit x86_64 dyn" in said,
        f"got: {said[:200]!r}",
    ))
    said = refusal(target.check_image, b"\xcf\xfa\xed\xfe" + b"\0" * 60, "a Mach-O")
    checks.append((
        "a Mach-O is refused by the Linux target",
        "not an ELF image" in said,
        f"got: {said[:200]!r}",
    ))

    # 7. On our own artefact, if it has been cross-built: the glibc version
    #    handed to `cargo zigbuild --target x86_64-unknown-linux-gnu.<v>` has to
    #    be the one that comes back out of the derivation. This is the only case
    #    here that is not circular -- every other one reads an ELF and checks
    #    what this same code read out of it, while this one compares the *input*
    #    given to a different tool against the *output* of the derivation. It
    #    fails if zig ignored the requested version, if a dependency pulled in a
    #    newer glibc symbol, or if GLIBC_TARGET drifted from the command in
    #    docs/LINUX.md §9.2.
    ours = target.artefact
    if not ours.exists():
        print("  ! case 7 skipped -- no cross-built lib_C.so "
              f"({ours}).\n"
              "    This is a SKIP, not a pass: the one non-circular case did "
              "not run.\n"
              f"    Build it with: {target.rebuild_hint}")
    else:
        want = f"manylinux_2_{target.GLIBC_TARGET[1]}_x86_64"
        floor = target._glibc_floor(ours.read_bytes())
        got = f"manylinux_{floor[0]}_{floor[1]}_x86_64"
        checks.append((
            f"our own lib_C.so derives {want}, the version zig was asked for",
            got == want,
            f"got {got}; either rebuild at {target.GLIBC_TARGET[0]}."
            f"{target.GLIBC_TARGET[1]} or move GLIBC_TARGET to match",
        ))

    bad = 0
    for label, ok, detail in checks:
        bad += not ok
        print(f"  {'ok    ' if ok else 'WRONG '}{label}")
        if not ok:
            print(f"          {detail}")
    if bad:
        print(f"LINUX SELF-TEST: FAIL -- {bad}/{len(checks)} wrong")
    else:
        print(f"LINUX SELF-TEST: PASS -- {len(checks)}/{len(checks)} cases on "
              "real Linux ELF"
              + (", including this crate's own artefact"
                 if ours.exists() else
                 "; this crate's own artefact was NOT among them"))
    return bad


def self_test_verify() -> int:
    """Break `verify`'s unexpected-member check and confirm it notices.

    Reproduces the .DS_Store incident (2026-08-30) without a real build: a
    scratch source tree and a hand-built zip stand in for the vendored tree
    and the finished wheel, the same substitution `self_test` above makes for
    the freshness check. `expected` and the wheel's legitimate members are
    both built from `tree_files`, the real function `main()` uses -- so this
    exercises the same source-of-truth `verify` is actually compared against,
    not a copy of it written here.

    Three cases: a clean wheel has to pass; a file that mirrors the .DS_Store
    incident exactly (something that landed in a package directory without
    ever being in the source tree) has to be refused, by name; and so does a
    stray extension for a target this wheel does not carry, which is the
    dangerous version of the same defect -- `.DS_Store` was silent because it
    was inert.
    """
    import tempfile

    global SRC
    original_src = SRC
    bad = 0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            src = tmpdir / "src"
            (src / "torch" / "lib").mkdir(parents=True)
            (src / "torch" / "__init__.py").write_text("# torch\n")
            (src / "torch" / "lib" / "thing.py").write_text("# thing\n")
            (src / "torchnative").mkdir()
            (src / "torchnative" / "__init__.py").write_text("# torchnative\n")

            # `tree_files` addresses everything relative to the module-global
            # SRC, not the `root` it is handed -- swapped here, restored in
            # `finally`, the same substitution `self_test_preflight_cache`
            # makes for `BUILD_CACHE`.
            SRC = src
            expected = tree_files(src / "torch") | tree_files(src / "torchnative")
            # The shape `main()` actually builds `extra` in: an empty
            # global-deps stub, and one file of upstream dist-info relocated
            # under `.data/`.
            extra = {
                "torch/lib/libtorch_global_deps.dylib": b"\0",
                "torchnative-9.9.9.data/purelib/torch-9.9.9.dist-info/METADATA":
                    b"Metadata-Version: 2.1\n",
            }
            dist_info = "torchnative-9.9.9.dist-info"
            wheel = tmpdir / "torchnative-9.9.9-cp313-abi3-macosx_11_0_arm64.whl"

            def make_wheel(junk: str | None, drop: str | None = None) -> None:
                if wheel.exists():
                    wheel.unlink()
                with zipfile.ZipFile(wheel, "w") as zf:
                    for name in sorted(expected):
                        if name == drop:
                            continue
                        zf.writestr(name, b"content")
                    for name, data in extra.items():
                        zf.writestr(name, data)
                    zf.writestr(f"{dist_info}/METADATA", b"Metadata-Version: 2.1\n")
                    zf.writestr(f"{dist_info}/WHEEL", b"Wheel-Version: 1.0\n")
                    zf.writestr(f"{dist_info}/RECORD", b"")
                    if junk:
                        zf.writestr(junk, b"whatever a stale cache left behind")

            def refusal(junk: str | None, drop: str | None = None) -> str:
                make_wheel(junk, drop)
                try:
                    verify(wheel, set(expected), None, extra, dist_info)
                except SystemExit as exc:
                    return str(exc)
                return ""

            cases = [
                ("a clean wheel", None, None, ""),
                ("a .DS_Store the source tree never had (2026-08-30's incident)",
                 "torch/.DS_Store", None, "torch/.DS_Store"),
                ("a stray extension from a previous target",
                 "torch/_C.cpython-311-darwin.so", None,
                 "torch/_C.cpython-311-darwin.so"),
                # `verify`'s ORIGINAL job, before the unexpected-member check
                # above existed: a wheel merely smaller than the source tree.
                # `self_test_verify` had no case for this direction at all --
                # every prior wheel here always carried every name in
                # `expected`, so a regression that broke or disabled the
                # `missing` check (verify()'s lines just above the
                # unexpected-member one) would have passed 3/3 unnoticed.
                ("a package file the source tree has but the wheel does not",
                 None, "torch/lib/thing.py", "torch/lib/thing.py"),
            ]
            for label, junk, drop, must_contain in cases:
                said = refusal(junk, drop)
                ok = (said == "") if junk is None and drop is None \
                    else (must_contain in said)
                bad += not ok
                print(f"  {'ok    ' if ok else 'WRONG '}{label}"
                      + (f" -- refused: {said.splitlines()[0]}" if said else ""))
                if not ok:
                    print(f"          expected "
                          f"{'a clean PASS' if junk is None and drop is None else 'refused, naming ' + repr(must_contain)}"
                          f"; got {said or '(passed)'}")
    finally:
        SRC = original_src
    if bad:
        print(f"VERIFY SELF-TEST: FAIL -- {bad}/{len(cases)} wrong")
    else:
        print(f"VERIFY SELF-TEST: PASS -- {len(cases)}/{len(cases)} cases -- "
              "verify() catching content nobody asked for, and\n"
              "  letting through a wheel that carries only what the source "
              "tree and this build's own additions put there")
    return bad


def self_test_preflight_cache() -> int:
    """Break `preflight`'s stale-build-cache refusal and confirm it notices.

    `BUILD_CACHE` is a module-level constant, not a parameter, because
    `preflight()` takes none -- it reads global state on purpose, the same as
    `VENDORED_ROOT` and `SHIM`. So this swaps the module global for the
    duration of the test rather than touching the real `build/` beside this
    checkout, and restores it in a `finally` even if an assertion raises.
    """
    import tempfile

    global BUILD_CACHE
    original = BUILD_CACHE
    bad = 0

    def refusal() -> str:
        try:
            check_build_cache()
        except SystemExit as exc:
            return str(exc)
        return ""

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            # 1. No cache at all -- must not be the thing that refuses.
            BUILD_CACHE = tmpdir / "no-such-build"
            said = refusal()
            ok = said == ""
            bad += not ok
            print(f"  {'ok    ' if ok else 'WRONG '}"
                  "no build/ at all -- not what preflight refuses on")

            # 2. An empty build/ -- pip has not run yet; must not refuse.
            empty = tmpdir / "empty-build"
            empty.mkdir()
            BUILD_CACHE = empty
            said = refusal()
            ok = said == ""
            bad += not ok
            print(f"  {'ok    ' if ok else 'WRONG '}"
                  "an empty build/ -- not refused either")

            # 3. A populated cache -- the 2026-08-30 shape: a lib.* directory
            #    left over from an earlier build. Must refuse and name it.
            stale = tmpdir / "stale-build"
            (stale / "lib.macosx-10.13-universal2-cpython-313").mkdir(
                parents=True)
            BUILD_CACHE = stale
            said = refusal()
            ok = ("lib.macosx-10.13-universal2-cpython-313" in said
                  and "rm -rf" in said)
            bad += not ok
            print(f"  {'ok    ' if ok else 'WRONG '}"
                  "a populated build/ -- refused, naming the stale entry "
                  "and a narrow rm")
            if not ok:
                print(f"          got: {said[:300]!r}")
    finally:
        BUILD_CACHE = original

    if bad:
        print(f"PREFLIGHT-CACHE SELF-TEST: FAIL -- {bad}/3 wrong")
    else:
        print("PREFLIGHT-CACHE SELF-TEST: PASS -- 3/3 cases")
    return bad


def self_test_upstream_dist_info() -> int:
    """Break `upstream_dist_info`'s refusal when the vendored tree lacks it.

    `SRC` is swapped for the duration, same as `self_test_verify` and
    `self_test_preflight_cache` do for their own module globals, restored in
    `finally`.

    Before this was enforced, a missing `torch-*.dist-info` produced only a
    `print()` and an empty `dict`, which `main()` folds straight into both
    `extra` (what actually goes in the wheel) and, through `extra`, what
    `verify()` is told to expect -- so nothing downstream ever learned the
    directory was supposed to exist. Case 1 is that absence, refused instead.
    Case 2 is the ordinary path: a real `torch-<v>.dist-info` still comes back
    relocated correctly, so the refusal is not bought by breaking the case
    that has always worked.
    """
    import tempfile

    global SRC
    original_src = SRC
    bad = 0

    def refusal() -> str:
        try:
            upstream_dist_info("9.9.9")
        except SystemExit as exc:
            return str(exc)
        return ""

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            # 1. No torch-*.dist-info at all -- must refuse, not print-and-empty.
            SRC = tmpdir / "no-dist-info"
            SRC.mkdir()
            said = refusal()
            ok = "no torch-*.dist-info" in said
            bad += not ok
            print(f"  {'ok    ' if ok else 'WRONG '}"
                  "no torch-*.dist-info in the vendored tree -- refused, not "
                  "silently empty")
            if not ok:
                print(f"          got: {said[:300]!r}")

            # 2. A real one is still read and relocated correctly -- the
            #    refusal above must not have cost this the normal path.
            SRC = tmpdir / "with-dist-info"
            info_dir = SRC / "torch-9.9.9.dist-info"
            info_dir.mkdir(parents=True)
            (info_dir / "METADATA").write_text("Metadata-Version: 2.1\n")
            (info_dir / "RECORD").write_text("should not be copied\n")
            out = upstream_dist_info("9.9.9")
            want_key = "torchnative-9.9.9.data/purelib/torch-9.9.9.dist-info/METADATA"
            ok = (out.get(want_key) == b"Metadata-Version: 2.1\n"
                 and not any(k.endswith("/RECORD") for k in out))
            bad += not ok
            print(f"  {'ok    ' if ok else 'WRONG '}"
                  "a real torch-*.dist-info is relocated under .data/purelib/, "
                  "RECORD excluded")
            if not ok:
                print(f"          got keys: {sorted(out)}")
    finally:
        SRC = original_src

    if bad:
        print(f"UPSTREAM-DIST-INFO SELF-TEST: FAIL -- {bad}/2 wrong")
    else:
        print("UPSTREAM-DIST-INFO SELF-TEST: PASS -- 2/2 cases")
    return bad


def self_test_pyemscripten() -> int:
    """`PyEmscriptenTarget`'s one structural claim, exercised rather than
    asserted in a comment.

    docs/WASM.md §9.4a: this is the only target whose tag is not derivable from
    a CPython, and the failure mode is that someone copies `AndroidTarget`,
    calls `self.sysconfig()`, gets `emscripten_5_0_3_wasm32` -- a tag
    `packaging` accepts and nothing on PyPI uses -- and ships it. A comment
    saying "do not do this" is read after the mistake. These four cases are the
    check that the refusal is real, that the number genuinely comes out of
    `pyodide-lock.json` (case 2 changes it and watches the tag follow), and that
    a main-module link is still rejected.
    """
    import json as _json
    import tempfile

    target = PyEmscriptenTarget()
    checks: list[tuple[str, bool, str]] = []

    # 1. sysconfig() refuses. Structural, not advisory.
    try:
        target.sysconfig()
        refused, detail = False, "it returned instead of failing"
    except SystemExit as exc:
        refused = "refused by design" in str(exc)
        detail = str(exc).splitlines()[0]
    checks.append((
        "PyEmscriptenTarget.sysconfig() refuses rather than answering",
        refused, detail))

    # 2. The tag really is read from pyodide-lock.json. Changing the number
    #    there has to change the tag -- otherwise it is written down somewhere
    #    and the derivation is a story about the code rather than the code.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyodide-lock.json").write_text(_json.dumps(
            {"info": {"abi_version": "1999_7", "arch": "wasm32",
                      "platform": "emscripten_0_0_0", "python": "3.14.2"}}))
        scratch = PyEmscriptenTarget()
        scratch.python_root = root
        derived = scratch.platform_tag(b"")
    checks.append((
        "the tag follows pyodide-lock.json's info.abi_version",
        derived == "pyemscripten_1999_7_wasm32",
        f"expected pyemscripten_1999_7_wasm32, got {derived!r}"))

    # 3. No lock file at all is a hard failure, not a guessed tag.
    with tempfile.TemporaryDirectory() as tmp:
        scratch = PyEmscriptenTarget()
        scratch.python_root = Path(tmp)
        try:
            scratch.platform_tag(b"")
            no_lock, detail3 = False, "it produced a tag with no distribution"
        except SystemExit as exc:
            no_lock = "pyodide-lock.json" in str(exc)
            detail3 = str(exc).splitlines()[0]
    checks.append((
        "a missing pyodide-lock.json fails rather than guessing",
        no_lock, detail3))

    # 4. check_image refuses a main-module link. Pyodide's own
    #    `pyodide.asm.wasm` is one, so this uses a real main module rather than
    #    a synthesised near-miss -- the same reason self_test_linux insists on
    #    real ELF.
    main = PYODIDE_ROOT / "pyodide.asm.wasm"
    if main.exists():
        try:
            target.check_image(main.read_bytes(), str(main))
            rejected, detail4 = False, "it accepted a main-module link"
        except SystemExit as exc:
            rejected = "main-module link" in str(exc)
            detail4 = str(exc).splitlines()[0]
        checks.append((
            "check_image refuses a main-module link (real pyodide.asm.wasm)",
            rejected, detail4))
    else:
        print(f"  ! main-module case skipped -- no {main}")

    print("\nPYEMSCRIPTEN SELF-TEST of the tag derivation and the "
          "sysconfig refusal")
    bad = 0
    for label, ok, detail in checks:
        bad += not ok
        print(f"  {'ok    ' if ok else 'WRONG '}{label}")
        if not ok:
            print(f"          {detail}")
    if bad:
        print(f"PYEMSCRIPTEN SELF-TEST: FAIL -- {bad}/{len(checks)} wrong")
    else:
        print(f"PYEMSCRIPTEN SELF-TEST: PASS -- {len(checks)}/{len(checks)} "
              "cases")
    return bad


def main() -> None:
    check_registry()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter whose pip/setuptools drive the build")
    ap.add_argument("--outdir", default=str(REPO / "dist"), type=Path)
    ap.add_argument("--target", default="host",
                    choices=["host", *sorted(TARGETS)],
                    help="cross target; default is this machine")
    ap.add_argument("--self-test", action="store_true",
                    help="exercise the artefact freshness check and exit; "
                         "builds nothing")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        failed = []
        if self_test_linux():
            failed.append("the Linux tag derivation")
        if self_test_verify():
            failed.append("verify()'s unexpected-member check")
        if self_test_preflight_cache():
            failed.append("preflight's stale-build-cache refusal")
        if self_test_upstream_dist_info():
            failed.append("upstream_dist_info's missing-tree refusal")
        if self_test_pyemscripten():
            failed.append("PyEmscriptenTarget's tag derivation")
        if failed:
            sys.exit("SELF-TEST: FAIL -- " + ", ".join(failed))
        return

    target = None if args.target == "host" else TARGETS[args.target]

    # Before anything else this target might need, including the vendored tree
    # and the host shim: a refusal is about the *machine*, and none of those
    # would change it. Answering with the reason here is what "refuses by name"
    # means -- `--target android-x86_64` says why, rather than either failing on
    # a missing artefact with a rebuild hint that cannot succeed, or not being
    # a choice at all. See `Target.refusal`.
    if target is not None and target.refusal:
        _fail(f"--target {target.key} cannot be built on this machine.\n"
              f"  {target.refusal}")

    # The host preflight runs for cross builds too, unchanged: `pip wheel` walks
    # the same source tree either way, so a missing vendored tree or a missing
    # host `_C` produces the same empty shell it always did. The cross artefact
    # is an *additional* requirement, never a substitute for those.
    stamp = preflight()
    print(f"vendored torch {stamp.get('version', '?')} "
          f"({stamp.get('py_modules', '?')} modules) + _C.abi3.so "
          f"({SHIM.stat().st_size:,} B)")

    # Every build packages the host shim -- a cross build overrides the member
    # afterwards, but `pip wheel` still walks the source tree to get there, and
    # `verify` compares the archive against that same tree. So the host artefact
    # is checked on both paths, exactly as `preflight` is.
    check_host_shim()

    overrides: dict[str, bytes] = {}
    plat: str | None = None
    if target is not None:
        if not target.artefact.exists():
            # `rebuild_hint` rather than a list of every target's build command:
            # the same three lines used to name Android's script and iOS's doc
            # for *any* missing artefact, so a Linux user was told to run
            # `scripts/device_android.sh build`. Each target already carries the
            # answer for itself, and it is the answer the staleness path quotes.
            _fail(
                f"no cross-built extension at {target.artefact}\n"
                f"  build it for {target.rust_target} first.\n"
                f"  Fix: {target.rebuild_hint}\n"
                "  CARGO_TARGET_DIR is currently "
                f"{CARGO_TARGET_DIR}"
            )
        # Existence was the only question this asked until 2026-08-29, and a
        # five-day-old artefact answers it just as well as a current one.
        require_current(target.artefact, target.rebuild_hint)
        cross = target.artefact.read_bytes()
        target.check_artefact(cross)
        print(f"target {target.key}: {target.artefact.name} "
              f"({len(cross):,} B)\n      {describe(cross)}")
        plat = target.platform_tag(cross)
        overrides["torch/_C.abi3.so"] = cross
    renames: dict[str, str] = {}
    if target is not None and target.extension_member != Target.extension_member:
        renames["torch/_C.abi3.so"] = target.extension_member
        print(f"  member: torch/_C.abi3.so -> {target.extension_member} "
              "(this interpreter's dynload table\n"
              "      has no .abi3.so in it)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    wheel = run_pip_wheel(args.python, args.outdir)

    version = wheel.name.split("-")[1]
    extra = {**upstream_dist_info(version), **global_deps_stub(target)}
    wheel = _repack(wheel, extra, f"torchnative-{version}.dist-info",
                    plat=plat, overrides=overrides, renames=renames)

    expected: set[str] = set()
    for pkg in ("torch", *stamp.get("packages", "").split(","), "torchnative"):
        if pkg and (SRC / pkg).is_dir():
            expected |= tree_files(SRC / pkg)
    verify(wheel, expected, target, extra, f"torchnative-{version}.dist-info")

    with zipfile.ZipFile(wheel) as zf:
        entries = zf.namelist()
        uncompressed = sum(i.file_size for i in zf.infolist())
    print()
    print(f"{wheel}")
    print(f"  {len(entries):,} entries, "
          f"{wheel.stat().st_size / 1e6:.1f} MB compressed, "
          f"{uncompressed / 1e6:.1f} MB installed")
    print(f"  tag: {'-'.join(wheel.stem.split('-')[2:])}")
    print()
    if target is None:
        print(f"next: python tools/wheel/verify.py {wheel}")
    elif target.key.startswith("android"):
        print(f"next: python tools/wheel/verify_cross.py {wheel}")
        print(f"      python tools/wheel/verify_android.py {wheel}")
    elif target.key.startswith("linux"):
        print(f"next: python tools/wheel/verify_cross.py {wheel}")
        print(f"      python tools/wheel/verify_linux.py {wheel}")
    else:
        print(f"next: python tools/wheel/verify_cross.py {wheel}")


if __name__ == "__main__":
    main()
