#!/usr/bin/env python3
"""What can be checked about the iOS **device** wheel without an iPhone.

    python tools/wheel/verify_ios_device.py dist/torchnative-*iphoneos*.whl

`tools/wheel/verify_ios_sim.py` proves the simulator wheel the whole way: it
unpacks it into a simulator CPython's site-packages, imports torch inside a
simulator process and computes with it. Nothing here can do that, and the reason
is not caution. It is dyld:

    ios_12_0_arm64_iphoneos's _C.abi3.so, in a simulator's site-packages
        ImportError: dlopen(...): tried: ... (incompatible platform (have 'iOS',
        need 'iOS-sim'))

    the same file, dlopen'd on this Mac
        OSError: dlopen(...): tried: ... (incompatible platform (have 'iOS',
        need 'macOS'))

So "run it and see" is unavailable, on both machines this repository has. That
left the device column of docs/platform/WHEEL.md §7.0 at *built* -- the artefact is a
device Mach-O, its tag is one an installer matches -- with everything after it
blank, and with a note saying a device is required.

**Part of that gap does not need a device.** The device and simulator artefacts
differ in exactly one thing that matters at load time, and it is checkable here:

  * the **simulator** extension resolves its CPython symbols by flat
    `-undefined dynamic_lookup` (docs/platform/WHEEL.md §7.1). All 118 of them are marked
    `dynamically looked up` in its symbol table; the file names no Python
    dependency at all.
  * the **device** extension has no libpython to fall back on, so it links
    `@rpath/Python.framework/Python` and every one of those 118 symbols is bound
    to that library by name, in the two-level namespace.

That binding is a claim the file makes about a library it does not contain, and
it is a claim that can be checked against the library. This script checks it:

  1. `nm -m` the wheel's Mach-O members. In a two-level-namespace image every
     undefined symbol carries the library it is bound to -- not a guess, the
     linker wrote it there.
  2. Resolve each bound library to something on disk: `Python.framework/Python`
     from the device CPython distribution, and the SDK's `.tbd` stub for each
     system library the load commands name.
  3. Ask, per library and not as a union, whether that library exports the
     symbols bound to it.

    torch/_C.abi3.so           222 undefined
      Python      118  <- Python.framework/Python (the device distribution)
      libSystem    88  <- iPhoneOS SDK stub
      Accelerate   16  <- iPhoneOS SDK stub
      unresolved    0

Per library rather than as a union on purpose: a `_Py*` symbol that happened to
exist in libSystem would pass a union check and still fail on a device, because
dyld does not look there for it.

It also checks the thing that makes the simulator result worth anything at all
to the device wheel: **outside the extension and the metadata, the two archives
are byte-identical.** Every one of the 2,685 vendored Python files that
`verify_ios_sim.py` imported is the same file in the device wheel.

What this does NOT say
----------------------

**It does not say the wheel works on a device.** It says the symbols resolve.
Specifically still open, and a device is the only way to close them:

  * **runtime load.** Symbols existing is necessary, not sufficient: dyld also
    has to *find* `@rpath/Python.framework/Python` at load time, which depends
    on the embedding app's `LC_RPATH` and on the framework being in the bundle.
    This script reads a framework off a shared cache directory; an app resolves
    `@rpath` against `@executable_path/Frameworks` (docs/platform/IOS.md §10).
  * **code signing.** Every Mach-O in an iOS app bundle must be signed with a
    profile the device trusts. Nothing here signs anything.
  * **`import torch` completing.** The simulator needed a `_multiprocessing`
    stub and a UIKit-loaded process to get through `torch/__init__.py`
    (docs/platform/IOS.md §4, §5). Whether a real app satisfies those is unmeasured.
  * **any kernel computing**, and any number about how fast.

The ladder, and where the device wheel stands on it, is printed at the end of
every run.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binfmt import macho_info  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# The device CPython distribution, whose framework the extension links. Same
# default and same environment variable as tools/wheel/build.py, so that the
# framework checked here is the one the artefact was built against.
TARGET_PYTHON_ROOT = Path(os.environ.get(
    "TORCHNATIVE_TARGET_PYTHON", "/Volumes/macMini/caches/target-python"))
DEVICE_PYTHON = TARGET_PYTHON_ROOT / "arm64-iphoneos"

# Members that are *expected* to differ between the device and simulator wheels.
# Everything else must not.
PLATFORM_MEMBERS = ("torch/_C.abi3.so", "torch/lib/libtorch_global_deps.so")


# ------------------------------------------------------------------ symbols

def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True)


# `nm -m` on a two-level-namespace image prints, for each import, the library it
# is bound to. `-undefined dynamic_lookup` symbols print `(dynamically looked
# up)` instead, which is a different fact and is kept separate below.
_UNDEF = re.compile(
    r"\(undefined\)\s+(?:weak\s+)?external\s+(\S+)\s+\((?:from\s+(\S+)|"
    r"(dynamically looked up))\)")


def undefined_symbols(path: Path) -> tuple[dict[str, set[str]], set[str], str]:
    """`({library: symbols}, dynamically-looked-up, error)` for one Mach-O.

    A non-empty `error` means the question was not answered -- which is not the
    same as "nothing is undefined", and the caller must not read it as one.
    """
    proc = _run("nm", "-m", "-arch", "arm64", str(path))
    if proc.returncode != 0:
        return {}, set(), f"nm -m failed ({proc.returncode}): {proc.stderr.strip()}"
    bound: dict[str, set[str]] = {}
    flat: set[str] = set()
    announced = parsed = 0
    for line in proc.stdout.splitlines():
        announced += "(undefined)" in line
        match = _UNDEF.search(line)
        if not match:
            continue
        parsed += 1
        symbol, library, dynamic = match.groups()
        if dynamic:
            flat.add(symbol)
        else:
            bound.setdefault(library, set()).add(symbol)
    if parsed != announced:
        # A library with no imports at all is legitimate -- the empty
        # `libtorch_global_deps.so` is one. What is not legitimate is nm saying
        # `(undefined)` on a line this could not read: then the count below is
        # short by however many were skipped, and "0 unresolved" would be the
        # check passing itself.
        return {}, set(), (
            f"nm -m printed {announced} '(undefined)' line(s) and only "
            f"{parsed} of them were understood, so the symbol list is "
            "incomplete")
    return bound, flat, ""


def exported_symbols(path: Path) -> tuple[set[str], str]:
    """External defined symbols of a Mach-O, via `nm -gU`."""
    proc = _run("nm", "-gU", "-arch", "arm64", str(path))
    if proc.returncode != 0:
        return set(), f"nm -gU failed ({proc.returncode}): {proc.stderr.strip()}"
    out = {line.split()[-1] for line in proc.stdout.splitlines() if line.strip()}
    if not out:
        return set(), f"{path} exports nothing that nm -gU could list"
    return out, ""


_SYMBOL_LIST = re.compile(r"\b(symbols|objc-classes):\s*\[")
_TOP_LEVEL_KEY = re.compile(r"^([a-z-]+):")


def tbd_symbols(path: Path) -> tuple[set[str], str]:
    """Symbols an SDK `.tbd` stub says its library exports.

    A `.tbd` is a text stub -- there is no arm64 dylib in an SDK to run `nm`
    over. The format is YAML, but PyYAML is not in every interpreter this
    repository uses, so the two list shapes that matter are scraped directly:
    `symbols:` and `objc-classes:` (which stand for `_OBJC_CLASS_$_<name>`).

    Only lists under `exports:` and `reexports:` count. tbd-v4 also allows an
    `undefineds:` section carrying a `symbols:` list of things the library
    *needs*, and counting those as exports would be this check answering with
    the wrong set.

    Umbrellas need no special handling: `libSystem.B.tbd` is a multi-document
    file that carries the stub of every library it re-exports inline, so
    scraping the whole file is scraping the closure.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return set(), f"{path} could not be read: {exc!r}"
    out: set[str] = set()
    section = ""
    position = 0
    for line in text.splitlines(keepends=True):
        key = _TOP_LEVEL_KEY.match(line)
        if key:
            section = key.group(1)
        elif line.startswith("---"):
            section = ""
        if section in ("exports", "reexports"):
            for match in _SYMBOL_LIST.finditer(line):
                kind = match.group(1)
                items, _ = _bracketed(text, position + match.end())
                for item in items:
                    if kind == "symbols":
                        out.add(item)
                    else:
                        out.add(f"_OBJC_CLASS_$_{item}")
                        out.add(f"_OBJC_METACLASS_$_{item}")
        position += len(line)
    if not out:
        return set(), f"{path} yielded no exported symbols -- it was not parsed"
    return out, ""


def _bracketed(text: str, start: int) -> tuple[list[str], int]:
    """Comma-separated items of a `[ ... ]` list beginning just after `start`."""
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    items = [tok.strip().strip("'\"") for tok in text[start:i].split(",")]
    return [tok for tok in items if tok], i


# ----------------------------------------------------------------- providers

def sdk_root(name: str) -> tuple[Path | None, str]:
    proc = _run("xcrun", "--sdk", name, "--show-sdk-path")
    if proc.returncode != 0:
        return None, f"xcrun --sdk {name} failed: {proc.stderr.strip()}"
    root = Path(proc.stdout.strip())
    if not root.is_dir():
        return None, f"xcrun named {root}, which is not a directory"
    return root, ""


def _short_names(install_name: str) -> set[str]:
    """The spellings `nm -m` might use for a library with this install name.

    `nm` prints the leaf, minus the `.dylib` and any version component:
    `/usr/lib/libSystem.B.dylib` prints as `libSystem`,
    `@rpath/Python.framework/Python` as `Python`.
    """
    leaf = install_name.rsplit("/", 1)[-1]
    stem = leaf[:-6] if leaf.endswith(".dylib") else leaf
    return {leaf, stem, stem.split(".")[0]}


class Provider:
    """Where a bound library's symbols are read from, and what it is."""

    def __init__(self, install_name: str, kind: str, path: Path, note: str):
        self.install_name = install_name
        self.kind = kind          # "framework" | "sdk stub"
        self.path = path
        self.note = note

    def symbols(self) -> tuple[set[str], str]:
        if self.kind == "framework":
            return exported_symbols(self.path)
        return tbd_symbols(self.path)


def resolve_provider(install_name: str, framework: Path,
                     sdk: Path) -> tuple[Provider | None, str]:
    """Map an `LC_LOAD_DYLIB` install name to a file that can be asked."""
    if "Python.framework/Python" in install_name:
        if not framework.is_file():
            return None, (
                f"{install_name} needs the device Python.framework, and there is "
                f"no file at {framework}. Set TORCHNATIVE_TARGET_PYTHON or pass "
                "--framework")
        return Provider(install_name, "framework", framework,
                        "the device CPython distribution"), ""
    if not install_name.startswith("/"):
        return None, (
            f"{install_name} is a relative install name this does not know how "
            "to resolve; only the Python framework and absolute system paths "
            "are handled")
    relative = install_name.lstrip("/")
    candidate = sdk / (
        relative[:-6] + ".tbd" if relative.endswith(".dylib")
        else relative + ".tbd")
    if not candidate.is_file():
        return None, (
            f"{install_name} has no stub in the SDK -- looked for {candidate}")
    return Provider(install_name, "sdk stub", candidate, f"{sdk.name}"), ""


# ------------------------------------------------------------------ the check

class Findings:
    """Two lists, kept apart on purpose.

    `problems` are statements about the wheel. `blind` are the check failing to
    run. Folding the second into the first is the recurring defect this
    repository keeps re-learning -- `rust/torch_c/pytests/run.sh` has the
    version of it where a SIGKILLed `cmp` got reported as a stale artefact. Both
    exit non-zero; only one of them is a finding.
    """

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.blind: list[str] = []

    def ok(self) -> bool:
        return not self.problems and not self.blind


def check_link_closure(name: str, data: bytes, framework: Path, sdk: Path,
                       findings: Findings, quiet: bool = False) -> None:
    """Every undefined symbol of one wheel member, against the library it names."""
    info = macho_info(data)
    if info is None:
        findings.blind.append(f"{name} is not a thin 64-bit Mach-O; not read")
        return
    if info["platform"] != "ios":
        findings.problems.append(
            f"{name} is built for {info['platform']!r}, not 'ios' -- this is not "
            "a device binary and its link closure is a different question")
        return

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(name).name
        path.write_bytes(data)
        bound, flat, error = undefined_symbols(path)
        if error:
            findings.blind.append(f"{name}: {error}")
            return

        total = sum(len(s) for s in bound.values()) + len(flat)
        if not quiet:
            print(f"  {name}   {total} undefined")

        if flat:
            # The simulator artefact looks exactly like this and it is correct
            # there. In a device image it means the extension expects a
            # flat-namespace fallback that iOS does not provide.
            findings.problems.append(
                f"{name} has {len(flat)} symbol(s) marked 'dynamically looked "
                f"up' (e.g. {sorted(flat)[:3]}). That is how the *simulator* "
                "artefact resolves CPython; a device has no libpython to fall "
                "back on, so this file is the wrong slice or was linked without "
                "PYO3_CONFIG_FILE + TORCHNATIVE_PYTHON_FRAMEWORK_DIR")

        unresolved = 0
        load_commands = list(info.get("dylibs") or ())
        for library in sorted(bound):
            symbols = bound[library]
            matches = [n for n in load_commands if library in _short_names(n)]
            if len(matches) != 1:
                findings.blind.append(
                    f"{name}: {len(symbols)} symbol(s) are bound to {library!r}, "
                    f"which matches {len(matches)} of the file's own load "
                    f"commands {load_commands}. Without exactly one match there "
                    "is no library to ask, so whether they resolve is unknown")
                continue
            provider, why = resolve_provider(matches[0], framework, sdk)
            if provider is None:
                findings.blind.append(f"{name}: {why}")
                continue
            available, error = provider.symbols()
            if error:
                findings.blind.append(f"{name}: {error}")
                continue
            missing = sorted(symbols - available)
            if not quiet:
                print(f"    {library:<12} {len(symbols):>4}  <- {provider.path.name}"
                      f"  ({provider.note})")
            if missing:
                unresolved += len(missing)
                findings.problems.append(
                    f"{name}: {len(missing)} of the {len(symbols)} symbol(s) "
                    f"bound to {library!r} are not exported by "
                    f"{provider.path}: {missing[:8]}"
                    + (" ..." if len(missing) > 8 else ""))
        if not quiet:
            print(f"    {'unresolved':<12} {unresolved:>4}"
                  + ("" if not flat else
                     f"   (+{len(flat)} bound to nothing at all)"))


def check_not_loadable_here(name: str, data: bytes, findings: Findings) -> None:
    """Show dyld refusing the file, rather than asserting that it would.

    This is the reason the script stops where it does. Printing dyld's own
    sentence puts it in front of whoever reads the output, so that "symbols
    resolve" is never read as "it ran".
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(name).name
        path.write_bytes(data)
        try:
            ctypes.CDLL(str(path))
        except OSError as exc:
            # dyld lists every path it tried, which is pages of the same
            # temporary directory. The sentence is the parenthesised reason.
            text = str(exc)
            reasons = sorted(set(re.findall(
                r"incompatible platform \(have '[^']*', need '[^']*'\)", text)))
            print(f"  {name}: "
                  + (" / ".join(reasons) if reasons
                     else text.strip().splitlines()[0][:200]))
            return
    findings.problems.append(
        f"{name} loaded on this macOS host. A device slice cannot -- if this "
        "one did, it is not the device slice")


def check_same_but_the_binary(device: Path, sibling: Path, findings: Findings,
                              quiet: bool = False) -> None:
    """Outside the extension and the metadata, the two wheels must be one wheel.

    This is what carries `verify_ios_sim.py`'s result across. That harness
    imported 2,372 vendored Python modules inside a simulator; if those files
    are the same bytes here, the device wheel's *Python* half is as verified as
    the simulator's. It is only the Mach-O half that is a separate artefact --
    which is the point docs/platform/WHEEL.md §7.4 makes, stated as a measurement rather
    than as a caveat.
    """
    with zipfile.ZipFile(device) as a, zipfile.ZipFile(sibling) as b:
        skip = set(PLATFORM_MEMBERS)
        names_a = {n for n in a.namelist()
                   if n not in skip and ".dist-info/" not in n}
        names_b = {n for n in b.namelist()
                   if n not in skip and ".dist-info/" not in n}
        only_a = sorted(names_a - names_b)
        only_b = sorted(names_b - names_a)
        if only_a or only_b:
            findings.problems.append(
                f"the two wheels do not carry the same files: "
                f"{len(only_a)} only in {device.name} ({only_a[:3]}), "
                f"{len(only_b)} only in {sibling.name} ({only_b[:3]})")
        differing = [n for n in sorted(names_a & names_b)
                     if a.read(n) != b.read(n)]
        if differing:
            findings.problems.append(
                f"{len(differing)} member(s) differ between the device and "
                f"simulator wheels outside {list(skip)}: {differing[:5]}")
        if not quiet:
            print(f"  {len(names_a & names_b):,} shared members, "
                  f"{len(differing)} differing "
                  f"(compared byte-for-byte; {', '.join(sorted(skip))} and the "
                  "dist-info are excluded, being the platform-shaped parts)")


def find_sibling(device: Path) -> tuple[Path | None, str]:
    """The simulator wheel that carries the SAME VERSION as `device`, or why
    there is none.

    `sorted(glob("*iphonesimulator*.whl"))[0]` -- what this used to be --
    picks whichever version sorts first in a directory that holds more than
    one, with no regard for which version `device` is. On 2026-08-30 that
    compared a 0.0.4a0 device wheel against a 0.0.2a0 simulator wheel and
    reported PASS; it was harmless only because the vendored Python tree had
    not changed between those two releases, which is luck, not a check.
    `check_same_but_the_binary` exists to carry `verify_ios_sim.py`'s import
    run across to the device wheel -- comparing against the wrong version
    is a claim about nothing.

    Returns `(None, reason)` rather than falling back to *a* wheel: an absent
    or version-mismatched sibling has to read as this script unable to
    compare, not as it having compared successfully.
    """
    version = device.name.split("-")[1]
    candidates = sorted(device.parent.glob("*iphonesimulator*.whl"))
    matches = [p for p in candidates if p.name.split("-")[1] == version]
    if not matches:
        if candidates:
            return None, (
                f"{len(candidates)} simulator wheel(s) in {device.parent} "
                f"({[p.name for p in candidates]}), none of them version "
                f"{version!r} -- comparing {device.name} against a different "
                "version is comparing against the wrong artefact, so this "
                "refuses rather than picking one"
            )
        return None, f"no *iphonesimulator*.whl in {device.parent}"
    if len(matches) > 1:
        return None, (
            f"{len(matches)} simulator wheels are version {version!r} "
            f"({[p.name for p in matches]}) -- ambiguous which one carries "
            "the import run this compares against; pass --against")
    return matches[0], ""


def ladder(link_findings: Findings, compare_findings: Findings,
          compared: bool) -> list[tuple[str, bool, str]]:
    """Print the ladder and return it as `(label, done, note)`.

    Each row is judged from the `Findings` that its own check populated --
    `link_findings` from `check_not_loadable_here`/`check_link_closure`,
    `compare_findings` from `check_same_but_the_binary` (or the sibling being
    unavailable). A blind spot or a problem in one has no way to reach the
    other's row: on 2026-08-30 the "symbols resolved" row read `[NO]` off a
    blind spot in the *sibling comparison*, with 0/222 symbols actually
    unresolved, because both checks were writing into one shared `Findings`
    and this used its `blind` list unfiltered. Two separate `Findings`
    objects -- one per rung -- make that impossible rather than merely fixed:
    there is no shared list left to leak through.
    """
    rows = [
        ("built", True,
         "device Mach-O, tag an installer matches -- verify_cross.py"),
        ("symbols resolved", link_findings.ok(),
         "every undefined symbol exported by the library it is bound to"),
        ("same tree as the simulator wheel", compared and compare_findings.ok(),
         "everything but the extension is byte-identical"),
        ("installed", False, "needs a device: no iOS filesystem here"),
        ("imported", False,
         "needs a device: dyld refuses this slice on macOS and in the simulator"),
        ("computed", False, "needs a device"),
    ]
    print("\n  the device wheel, rung by rung")
    for label, done, note in rows:
        mark = "yes" if done else "NO "
        print(f"    [{mark}] {label:<34} {note}")
    print("\n  Still open, and only a device closes them: dyld finding "
          "@rpath/Python.framework/Python\n"
          "  in a real app bundle, code signing, `import torch` completing, "
          "any kernel computing,\n"
          "  and every performance number. See docs/platform/IOS.md §10.")
    return rows


# ------------------------------------------------------------------ self-test

def self_test(device: Path, sibling: Path | None) -> int:
    """Break each judgement and check it notices.

    A resolver that finds everything is indistinguishable from a resolver that
    looks nowhere, so each fault is aimed at one judgement, and the two kinds
    of answer -- a finding about the wheel, and the check being unable to
    look -- are asserted to come back as themselves rather than as each
    other. Cases 6 and 7 are `find_sibling` itself, against a scratch
    multi-version directory shaped exactly like the real `dist/`. Case 8 is
    `ladder()` itself: that a blind spot in one rung's check cannot flip
    another rung's verdict.
    """
    sdk, error = sdk_root("iphoneos")
    if sdk is None:
        sys.exit(f"self-test cannot run: {error}")
    with zipfile.ZipFile(device) as zf:
        extension = zf.read("torch/_C.abi3.so")

    results: list[tuple[str, str, bool, str]] = []

    def record(label: str, want: str, findings: Findings, phrase: str) -> None:
        got = ("problem" if findings.problems else
               "blind" if findings.blind else "clean")
        text = " | ".join(findings.problems + findings.blind)
        results.append((label, want, got == want and phrase in text, text[:160]))

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        # 1. A framework that is a real Mach-O but not CPython. The 118 symbols
        #    bound to `Python` must come back missing -- if they do not, the
        #    provider is not being consulted at all.
        decoy = tmpdir / "Python"
        with zipfile.ZipFile(device) as zf:
            decoy.write_bytes(zf.read("torch/lib/libtorch_global_deps.so"))
        findings = Findings()
        check_link_closure("torch/_C.abi3.so", extension, decoy, sdk, findings,
                           quiet=True)
        record("a Python.framework that is not CPython", "problem", findings,
               "not exported by")

        # 2. A framework that is not there at all. That is the check unable to
        #    look, and it must NOT be reported as unresolved symbols.
        findings = Findings()
        check_link_closure("torch/_C.abi3.so", extension,
                           tmpdir / "no-such-framework", sdk, findings,
                           quiet=True)
        record("no device Python.framework on disk", "blind", findings,
               "no file at")

        # 3. An SDK with no stubs in it. Same distinction, on the other provider.
        empty_sdk = tmpdir / "Empty.sdk"
        empty_sdk.mkdir()
        findings = Findings()
        check_link_closure("torch/_C.abi3.so", extension,
                           DEVICE_PYTHON / "Python.framework" / "Python",
                           empty_sdk, findings, quiet=True)
        record("an SDK with no .tbd stubs", "blind", findings, "no stub in the SDK")

        # 4. The simulator slice in the device wheel's place. This is the
        #    substitution docs/platform/WHEEL.md §7.4 says nothing else distinguishes.
        if sibling is not None:
            with zipfile.ZipFile(sibling) as zf:
                sim = zf.read("torch/_C.abi3.so")
            findings = Findings()
            check_link_closure("torch/_C.abi3.so", sim,
                               DEVICE_PYTHON / "Python.framework" / "Python",
                               sdk, findings, quiet=True)
            record("the simulator extension passed off as the device one",
                   "problem", findings, "not 'ios'")

            # 5. One vendored Python file changed. The two-wheels comparison is
            #    what carries the simulator run across, so it has to be able to
            #    say that it does not apply.
            tampered = tmpdir / sibling.name
            with zipfile.ZipFile(sibling) as src, zipfile.ZipFile(
                    tampered, "w", zipfile.ZIP_DEFLATED) as dst:
                victim = next(n for n in src.namelist()
                              if n.endswith("torch/version.py"))
                for item in src.infolist():
                    body = src.read(item.filename)
                    if item.filename == victim:
                        body = body + b"\n# tampered by the self-test\n"
                    dst.writestr(item, body)
            findings = Findings()
            check_same_but_the_binary(device, tampered, findings, quiet=True)
            record("a vendored file differing between the two wheels", "problem",
                   findings, "differ between the device and simulator")

        # 6. `find_sibling` pairing by VERSION. This is the exact incident
        #    (2026-08-30): several versions of both wheels sitting beside each
        #    other, and `sorted(glob(...))[0]` picking whichever sorts first
        #    regardless of which version `device` is. No real device wheel
        #    needed -- the function only reads filenames.
        multiversion = tmpdir / "dist"
        multiversion.mkdir()
        for v in ("0.0.2a0", "0.0.3a0", "0.0.4a0"):
            (multiversion /
             f"torchnative-{v}-cp313-abi3-ios_12_0_arm64_iphoneos.whl").touch()
        for v in ("0.0.2a0", "0.0.3a0"):
            (multiversion / f"torchnative-{v}-cp313-abi3-"
             "ios_14_0_arm64_iphonesimulator.whl").touch()
        newest_device = (
            multiversion /
            "torchnative-0.0.4a0-cp313-abi3-ios_12_0_arm64_iphoneos.whl")

        got, err = find_sibling(newest_device)
        ok = got is None and "0.0.4a0" in err and "none of them version" in err
        results.append((
            "no 0.0.4a0 simulator wheel present -- refuses rather than "
            "falling back to 0.0.2a0/0.0.3a0 (the actual 2026-08-30 mistake)",
            "refused", ok, f"got sibling={got}, error={err!r}"[:200]))

        matching_sim = (
            multiversion / "torchnative-0.0.4a0-cp313-abi3-"
            "ios_14_0_arm64_iphonesimulator.whl")
        matching_sim.touch()
        got, err = find_sibling(newest_device)
        ok = got == matching_sim and not err
        results.append((
            "picks the same-version simulator wheel once it exists, not "
            "whichever version sorts first",
            "matched", ok,
            f"got {got.name if got else got!r}, error={err!r}"[:200]))

        # 8. `ladder()` itself: the exact 2026-08-30 shape. `check_link_closure`
        #    on our own real extension finds nothing wrong (0/222 unresolved --
        #    same as case 1's provider being consulted for real, just clean
        #    this time), and a *separate* phase (the sibling comparison) is
        #    blind. Before this file kept one `Findings` per rung, the ladder
        #    read "symbols resolved" off the whole shared `findings.blind`, so
        #    a blind spot that had nothing to do with symbols dragged that row
        #    to `[NO]` anyway -- confirmed live against a real device wheel
        #    with no simulator sibling beside it: 0 unresolved, row still NO.
        #    Two independent `Findings` make that impossible: there is no
        #    shared list left for one rung's blind spot to leak through into
        #    another's.
        clean_link = Findings()
        check_link_closure("torch/_C.abi3.so", extension,
                           DEVICE_PYTHON / "Python.framework" / "Python", sdk,
                           clean_link, quiet=True)
        blind_compare = Findings()
        blind_compare.blind.append(
            "no simulator wheel to compare against -- stands in for the "
            "sibling being absent or version-mismatched")
        rows = ladder(clean_link, blind_compare, compared=False)
        row = dict((label, done) for label, done, _ in rows)
        ok = (not clean_link.problems and not clean_link.blind
             and row["symbols resolved"] is True
             and row["same tree as the simulator wheel"] is False)
        results.append((
            "a blind sibling comparison does not downgrade 'symbols resolved' "
            "(the ladder is judged per rung, not off one shared findings list)",
            "yes/NO", ok,
            f"link findings clean={not clean_link.problems and not clean_link.blind}, "
            f"'symbols resolved'={row.get('symbols resolved')!r}, "
            f"'same tree...'={row.get('same tree as the simulator wheel')!r}"))

    print(f"SELF-TEST against {device.name}")
    caught = 0
    for label, want, ok, text in results:
        caught += ok
        print(f"  {'caught    ' if ok else 'MISSED    '}{label}")
        if not ok:
            print(f"      expected a '{want}' answer; got: {text or '(nothing)'}")
    if caught != len(results):
        print(f"\nSELF-TEST: FAIL -- {len(results) - caught} of {len(results)} "
              "fault modes went unnoticed", file=sys.stderr)
        return 1
    print(f"\nSELF-TEST: PASS -- {caught}/{caught} fault modes rejected, and "
          "each\n  as the right kind of answer (a finding about the wheel, or "
          "the check unable to look)")
    return 0


# ----------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--against", type=Path, default=None,
                    help="the simulator wheel to compare against; by default "
                         "the *iphonesimulator*.whl beside this one")
    ap.add_argument("--framework", type=Path,
                    default=DEVICE_PYTHON / "Python.framework" / "Python",
                    help="the device Python.framework binary the extension links")
    ap.add_argument("--sdk", default="iphoneos")
    ap.add_argument("--self-test", action="store_true",
                    help="break each judgement and check it notices")
    args = ap.parse_args()

    if not args.wheel.exists():
        sys.exit(f"no such wheel: {args.wheel}")
    if "iphoneos" not in args.wheel.name:
        sys.exit(f"{args.wheel.name} is not an iOS-device wheel. The simulator "
                 "wheel is a different artefact and tools/wheel/verify_ios_sim.py "
                 "runs that one for real.")
    sibling_error = ""
    if args.against is not None:
        sibling = args.against
    else:
        sibling, sibling_error = find_sibling(args.wheel)

    if args.self_test:
        sys.exit(self_test(args.wheel, sibling))

    if not shutil.which("nm"):
        sys.exit("no nm(1) on PATH -- this reads symbol tables and cannot run "
                 "without it")
    sdk, error = sdk_root(args.sdk)
    if sdk is None:
        sys.exit(f"cannot locate the {args.sdk} SDK: {error}")

    # Two `Findings`, kept apart the same way `problems` and `blind` are kept
    # apart within each: one per rung of the ladder, so a blind spot or a
    # problem in one phase has no shared list through which to reach the
    # other phase's verdict. `findings` below is only for the final combined
    # report (FAIL / CANNOT JUDGE) and the exit code, which do want both.
    link_findings = Findings()
    compare_findings = Findings()
    print(f"{args.wheel.name}")
    print(f"  framework  {args.framework}")
    print(f"  sdk        {sdk}")

    print("\n+ dyld will not load this here, which is why nothing below runs it")
    with zipfile.ZipFile(args.wheel) as zf:
        members = [n for n in zf.namelist() if n in PLATFORM_MEMBERS]
        if not members:
            sys.exit(f"{args.wheel.name} carries none of {PLATFORM_MEMBERS}")
        check_not_loadable_here(members[0], zf.read(members[0]), link_findings)

        print("\n+ every undefined symbol, against the library it is bound to")
        for name in members:
            check_link_closure(name, zf.read(name), args.framework, sdk,
                               link_findings)

    print("\n+ the rest of the archive, against the simulator wheel")
    if sibling is None or not sibling.exists():
        compare_findings.blind.append(
            (sibling_error or
             "no simulator wheel to compare against -- build one with "
             "`build.py --target ios-arm64-sim` or pass --against")
            + ". Without it, nothing here connects the device wheel to the "
            "run that verify_ios_sim.py did")
        print(f"  (none -- {sibling_error or 'not found'})")
    else:
        print(f"  {sibling.name}")
        check_same_but_the_binary(args.wheel, sibling, compare_findings)

    ladder(link_findings, compare_findings,
          compared=sibling is not None and sibling.exists())

    findings = Findings()
    findings.problems = link_findings.problems + compare_findings.problems
    findings.blind = link_findings.blind + compare_findings.blind

    sys.stdout.flush()
    if findings.problems:
        print()
        for problem in findings.problems:
            print(f"FAIL: {problem}", file=sys.stderr)
    if findings.blind:
        print()
        for gap in findings.blind:
            print(f"CANNOT JUDGE: {gap}", file=sys.stderr)
        print("  ^ these are not findings about the wheel. They are this script "
              "unable to\n    look, reported separately so the two are never "
              "read as each other.", file=sys.stderr)
    if not findings.ok():
        sys.exit(1)

    print(f"\nPASS -- every undefined symbol in {args.wheel.name} is exported by "
          "the library it\n       is bound to, and outside the extension the "
          "archive is the simulator wheel.")
    print("       This is NOT 'it works on a device'. It is 'the link resolves'. "
          "Load, code\n       signing, import and computation remain unmeasured "
          "-- see the ladder above.")


if __name__ == "__main__":
    main()
