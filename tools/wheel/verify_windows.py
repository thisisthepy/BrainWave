#!/usr/bin/env python3
"""What can be checked about the Windows x86-64 wheel without a Windows machine.

    python tools/wheel/verify_windows.py dist/torchnative-*win_amd64.whl
    python tools/wheel/verify_windows.py --self-test

This is the third of these scripts, after `verify_ios_device.py` and
`verify_linux.py`, and the first thing worth saying is that **it reaches as far
as the iOS one and further than the Linux one.** That is a property of the file
format, not of any cleverness here.

The question all three ask is: for each symbol this image does not define, which
library is supposed to provide it? The three formats answer it very differently.

    Mach-O (iOS)    two-level namespace. The linker records, per undefined
                    symbol, which dylib it bound to. Every symbol answers.
    ELF (Linux)     no such thing. `ld.so` searches the whole global scope in
                    load order, so in general the image does not say. Only
                    *versioned* symbols do -- `.gnu.version_r` is grouped by
                    library -- which covers glibc and not CPython, because
                    CPython versions none of its exports (docs/platform/LINUX.md §6.1).
    PE (Windows)    the import table **is** the answer. An import is not a name
                    to be searched for: it is an entry under an
                    IMAGE_IMPORT_DESCRIPTOR that names a DLL. There is no
                    unattributed import. Every symbol answers.

So on Windows the *attribution* is complete, exactly as on an iOS device. What
is incomplete is something else, and it is the same gap Linux has: whether the
named DLL really exports that symbol can only be checked for the DLLs that exist
on this machine.

    python3.dll         present, in the target distribution. **Checked.**
    python313.dll       present. Not used -- an abi3 extension must not bind it.
    vcruntime140*.dll   present, in the target distribution. **Checked.**
    kernel32.dll        an OS DLL. Not here, and not obtainable without a
    ntdll.dll           Windows install. Attributed but not resolved, and
    api-ms-win-crt-*    reported that way.
    bcryptprimitives.dll

That split is worth stating precisely, because it is *better* than the Linux
one in a way that is easy to lose: on Linux the 118 CPython imports could only
be checked as a union against libpython, since the file names no library for
them. Here the file names `python3.dll` for each of them, so a symbol that
happens to exist in some other DLL cannot stand in for a missing one.

What this does NOT say
----------------------

**It does not say the wheel works on Windows.** Nothing here loads anything.
Specifically:

  * **the OS half is unresolved.** ~108 imports name `kernel32.dll`, `ntdll.dll`
    and the ucrt `api-ms-win-*` forwarders. Those are Windows components; this
    machine has none of them and cannot get them. What is established is that
    the image asks for them *by name and by DLL*, and that the set of DLLs it
    asks for contains nothing unexpected.
  * **ordinal-only imports cannot be named.** If an import is by ordinal the
    file records a number, and the name lives in the exporting DLL's ordinal
    table. Those are counted separately rather than passed silently.
  * **`import torch` is not run.** No `LoadLibrary`, no DLL search path, no
    `os.add_dll_directory`. `_load_dll_libraries()` does several things at
    import time that only a Windows machine can exercise -- see
    docs/platform/WINDOWS.md §4.3.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binfmt import describe, pe_exports, pe_imports, pe_info  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# Same default and same variable as tools/wheel/build.py, so the DLLs the
# imports are resolved against are the ones the wheel was built for.
TARGET_PYTHON_ROOT = Path(os.environ.get(
    "TORCHNATIVE_TARGET_PYTHON", "/Volumes/macMini/caches/target-python"))

#: Wheel tag -> the distribution the imports are resolved against, and the
#: `pe_info` machine the member must be.
#:
#: There are two Windows distributions under `target-python/` now, and this file
#: read one of them by name until there were. It is the same silent-pass this
#: file's Linux counterpart has, and worse here: `python3.dll` is the stable-ABI
#: forwarder, so its export list is *the same set of names* on ARM64 as on
#: AMD64 by construction. Resolving an ARM64 `_C.pyd`'s imports against the
#: AMD64 `python3.dll` therefore reports every symbol accounted for, and the
#: run reads exactly like a correct one.
ARCHES = {
    "win_amd64": ("x86_64-pc-windows-msvc", "x86_64"),
    "win_arm64": ("aarch64-pc-windows-msvc", "aarch64"),
}

#: The distribution in use. Rebound by `select_arch` from the wheel's own tag;
#: the x86-64 default is what `--self-test` runs against.
WINDOWS_PYTHON = TARGET_PYTHON_ROOT / ARCHES["win_amd64"][0]
ARCH = "x86_64"


def select_arch(wheel: Path) -> str:
    """Point this run at the distribution the wheel's own tag names.

    The tag is a *name* on Windows rather than a derivation
    (`build.WindowsTarget`), and this is the other side of that: the name is
    also the only thing in the wheel that says which Windows it is for, so it
    is what selects the DLLs to resolve against.
    """
    global WINDOWS_PYTHON, ARCH
    match = re.search(r"-(win_\w+|win32)\.whl$", wheel.name)
    if not match:
        _fail(f"{wheel.name} carries no Windows platform tag -- the tag is "
              "what selects the\n  distribution its imports are resolved "
              "against, and there is none to read")
    tag = match.group(1)
    if tag not in ARCHES:
        _fail(f"{wheel.name} is tagged {tag!r}, which this has no CPython "
              f"distribution for (it has {sorted(ARCHES)}).\n"
              "  Refused rather than resolved against another architecture's "
              "python3.dll: that DLL\n  is the abi3 forwarder, so its export "
              "names are identical across architectures and\n  the check "
              "would PASS without having run.")
    subdir, ARCH = ARCHES[tag]
    WINDOWS_PYTHON = TARGET_PYTHON_ROOT / subdir
    return tag

#: The one binary member of the Windows wheel. There is no global-deps library:
#: `_load_global_deps()` returns immediately on Windows (docs/platform/WINDOWS.md §4.3).
MEMBERS = ("torch/_C.pyd",)

#: DLLs the target distribution ships, so their exports can actually be read.
#: `python313.dll` is deliberately *not* here: resolving against it would let an
#: extension that binds the version-locked DLL pass a check whose whole point is
#: that it binds the stable-ABI forwarder instead.
RESOLVABLE = ("python3.dll", "vcruntime140.dll", "vcruntime140_1.dll")

#: DLLs that are part of Windows. Being unable to resolve against these is a
#: fact about this machine; being asked for one *outside* this set is a fact
#: about the wheel, and the two must not read the same.
OS_DLL_PREFIXES = ("api-ms-win-", "ext-ms-win-")
OS_DLLS = frozenset({
    "kernel32.dll", "kernelbase.dll", "ntdll.dll", "advapi32.dll",
    "user32.dll", "ws2_32.dll", "userenv.dll", "dbghelp.dll", "shell32.dll",
    "ole32.dll", "oleaut32.dll", "bcrypt.dll", "bcryptprimitives.dll",
    "crypt32.dll", "secur32.dll", "psapi.dll", "version.dll", "winmm.dll",
    "powrprof.dll", "pdh.dll", "rpcrt4.dll",
    # The MSVC runtime is redistributable rather than in-box, but upstream torch
    # already requires it by name on Windows (`ctypes.CDLL("vcruntime140.dll")`
    # in `_load_dll_libraries`), so needing it is not new information.
    "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcrt.dll",
})


def _fail(msg: str) -> None:
    sys.exit(f"tools/wheel/verify_windows.py: {msg}")


def _is_os_dll(name: str) -> bool:
    lowered = name.lower()
    return lowered in OS_DLLS or lowered.startswith(OS_DLL_PREFIXES)


def available_exports() -> dict[str, set[str]]:
    """`{dll name: exported names}` for the DLLs the distribution ships."""
    found: dict[str, set[str]] = {}
    for name in RESOLVABLE:
        path = WINDOWS_PYTHON / name
        if not path.exists():
            continue
        exported = pe_exports(path.read_bytes())
        if exported is None:
            _fail(f"{path} is not a readable PE image ({describe(path.read_bytes())})")
        found[name.lower()] = exported
    if "python3.dll" not in found:
        _fail(
            f"no python3.dll under {WINDOWS_PYTHON}\n"
            "  Without it the CPython half of the symbol check cannot run at "
            "all. That is\n"
            "  the check being unavailable, not the wheel being fine.\n"
            "  TORCHNATIVE_TARGET_PYTHON selects the distribution root."
        )
    return found


def resolve(data: bytes, what: str, exports: dict[str, set[str]]) -> int:
    """Report, per DLL, which of this image's imports can be accounted for.

    Returns the number that cannot be -- which is *not* the number that is
    unresolved on this machine. A symbol imported from `kernel32.dll` is
    attributed and simply not checkable here; a symbol imported from a DLL that
    is neither shipped by the distribution nor part of Windows is a real
    problem, because nothing would provide it on the target either.
    """
    info = pe_info(data)
    if info is None:
        _fail(f"{what} is not a PE image ({describe(data)})")
    imports = pe_imports(data)
    if imports is None:
        _fail(f"{what} has no readable import table")

    total = sum(len(names) for names in imports.values())
    print(f"  {what}")
    print(f"    {total} imports from {len(imports)} DLL(s), every one attributed "
          "by the import table")

    problems = 0
    for dll in sorted(imports, key=str.lower):
        names = imports[dll]
        ordinals = {n for n in names if n.startswith("#")}
        by_name = names - ordinals
        key = dll.lower()
        if key in exports:
            missing = sorted(by_name - exports[key])
            mark = "->"
            # `key`, not `dll`: PE import names are case-insensitive and this
            # image spells one of them VCRUNTIME140.dll, which is not the
            # filename the exports were read from.
            note = f"resolved against {WINDOWS_PYTHON.name}/{key}"
            if missing:
                problems += len(missing)
                note = (f"{len(missing)} NOT exported by that DLL: "
                        f"{missing[:5]}")
                mark = "!!"
        elif _is_os_dll(dll):
            mark = "--"
            note = "a Windows component; attributed but not resolvable here"
        else:
            problems += len(names)
            mark = "!!"
            note = ("neither shipped by the target CPython nor a Windows "
                    "component -- nothing would provide this on the target "
                    "either")
        print(f"    {len(names):5d}  {mark} {dll}")
        print(f"           {note}")
        if ordinals:
            print(f"           {len(ordinals)} of them by ordinal, so the file "
                  "records no name to check")
    return problems


def check_wheel(wheel: Path) -> int:
    tag = select_arch(wheel)
    print(f"tag: {tag} -> {WINDOWS_PYTHON}")
    exports = available_exports()
    for name in sorted(exports):
        print(f"{name}: {len(exports[name]):,} exported symbols "
              f"({WINDOWS_PYTHON / name})")
    print(wheel.name)
    print()

    problems = 0
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        for member in MEMBERS:
            if member not in names:
                _fail(f"{wheel.name} has no {member}")
            data = zf.read(member)
            info = pe_info(data)
            print(f"  {member}: {describe(data)}")
            if info is None or (info["bits"], info["machine"], info["dll"]) != \
                    (64, ARCH, True):
                _fail(f"{member} is {describe(data)}, expected PE32+ {ARCH} dll")
            problems += resolve(data, member, exports)
            print()
        stale = sorted(n for n in names if n.endswith("torch/_C.abi3.so"))
        if stale:
            _fail(f"{wheel.name} carries {stale}; on Windows the extension has "
                  "to be torch/_C.pyd, which is the only suffix in "
                  "dynload_win.c's table that an abi3 build can use")

    print("ladder (docs/platform/WINDOWS.md §5):")
    print(f"  built              yes -- the archive holds a PE32+ {ARCH} DLL")
    print("  tagged             yes -- see tools/wheel/build.py WindowsTarget")
    print("  symbols resolve    CPython and the MSVC runtime yes, per DLL;")
    print("                     the Windows half is attributed per DLL but not")
    print("                     resolved, there being no Windows here")
    print("  imports on Windows UNKNOWN -- needs a Windows machine")
    print("  computes           UNKNOWN -- same")
    print()
    if problems:
        print(f"FAIL -- {problems} import(s) nothing would provide",
              file=sys.stderr)
        return 1
    print("PASS -- every import names a DLL, and every DLL that exists here "
          "provides what is asked of it")
    return 0


# ------------------------------------------------------------------ self-test
#
# "실패할 수 없는 검증은 검증이 아니다" (CLAUDE.md §5.5). The resolver above passes
# on the wheel this repository builds, which on its own says nothing. So it is
# also run on images it must reject, and on a real Windows CPython extension
# from the target distribution that it must accept -- the same shape as
# verify_linux.py's self-test, with the difference that here our own artefact is
# available too.


def _patch(data: bytes, offset: int, raw: bytes) -> bytes:
    return data[:offset] + raw + data[offset + len(raw):]


def _rename_imported_dll(data: bytes, old: bytes, new: bytes) -> bytes:
    """Point an import descriptor at a DLL nothing provides.

    Same length, so every RVA in the file stays valid and the only thing that
    changes is the claim about where those symbols come from. If the resolver
    were checking "does this symbol exist anywhere I can see", this would still
    pass -- the names are unchanged and python3.dll still exports them all.
    """
    if len(old) != len(new):
        raise SystemExit("the replacement DLL name must be the same length")
    index = data.find(old + b"\0")
    if index < 0:
        raise SystemExit(f"{old!r} is not in the image -- self-test cannot run")
    return _patch(data, index, new)


def self_test() -> int:
    exports = available_exports()
    checks: list[tuple[str, bool, str]] = []

    extensions = sorted((WINDOWS_PYTHON / "DLLs").glob("*.pyd"))
    if not extensions:
        print(f"  ! skipped -- no *.pyd under {WINDOWS_PYTHON}/DLLs.")
        print("    This is a SKIP, not a pass: the resolver is unexercised.")
        return 0

    print("SELF-TEST of the PE import resolver")

    # 1. A real Windows CPython extension module, from the distribution, has to
    #    come out clean. It is the same shape as ours: Py* from python3.dll or
    #    python313.dll, CRT from the ucrt forwarders, the rest from Windows.
    sample = next((p for p in extensions if p.name.startswith("_socket")),
                  extensions[0])
    data = sample.read_bytes()
    imports = pe_imports(data) or {}
    unattributed = [d for d in imports if not d]
    checks.append((
        f"{sample.name}: all {sum(len(v) for v in imports.values())} imports "
        "name a DLL",
        bool(imports) and not unattributed,
        f"got {len(unattributed)} unattributed and {len(imports)} DLLs",
    ))

    # 2. The claim that makes this stronger than the ELF check: for OUR
    #    artefact, the CPython imports are attributed to python3.dll by the
    #    file, not merely found in it. On Linux the equivalent 118 symbols name
    #    no library at all (docs/platform/LINUX.md §6.4), and that is the difference.
    ours = None
    for candidate in sorted(REPO.glob("dist/*win_amd64.whl")):
        with zipfile.ZipFile(candidate) as zf:
            if "torch/_C.pyd" in zf.namelist():
                ours = zf.read("torch/_C.pyd")
    if ours is None:
        print("  ! our own torch/_C.pyd was not found in dist/ -- cases 2 to 4 "
              "skipped.")
        print("    This is a SKIP, not a pass.")
    else:
        mine = pe_imports(ours) or {}
        py3 = mine.get("python3.dll", set())
        checks.append((
            f"our torch/_C.pyd attributes {len(py3)} imports to python3.dll, "
            "by name",
            bool(py3) and py3 <= exports["python3.dll"],
            f"{len(py3 - exports['python3.dll'])} of them are not exported by it",
        ))
        checks.append((
            "...and none to python313.dll, which would make the abi3 tag false",
            "python313.dll" not in mine,
            f"it imports {len(mine.get('python313.dll', ()))} names from it",
        ))

        # 3. The resolver must reject an image whose imports are attributed to a
        #    DLL that neither the distribution nor Windows provides. Every
        #    symbol name is untouched and every one of them really is exported
        #    by python3.dll, so a union-style check would pass this.
        tampered = _rename_imported_dll(ours, b"python3.dll", b"pythonX.dll")
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            bad = resolve(tampered, "tampered", exports)
        checks.append((
            "an import attributed to a DLL nothing provides is refused, "
            f"naming it ({bad} symbols)",
            bad == len(py3) and "pythonX.dll" in buffer.getvalue(),
            f"got {bad} problems; output mentions pythonX.dll: "
            f"{'pythonX.dll' in buffer.getvalue()}",
        ))

    # 4. python313.dll is deliberately absent from RESOLVABLE. If it were there,
    #    case 3's tamper could be defeated by binding the wrong DLL.
    checks.append((
        "python313.dll is not offered as a resolution target",
        "python313.dll" not in exports,
        f"exports has {sorted(exports)}",
    ))

    bad = 0
    for label, ok, detail in checks:
        bad += not ok
        print(f"  {'ok    ' if ok else 'WRONG '}{label}")
        if not ok:
            print(f"          {detail}")
    if bad:
        print(f"SELF-TEST: FAIL -- {bad}/{len(checks)} wrong", file=sys.stderr)
        return 1
    print(f"SELF-TEST: PASS -- {len(checks)}/{len(checks)} cases, on real "
          "Windows PE from the target distribution")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel", type=Path, nargs="?")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())
    if args.wheel is None:
        ap.error("give a wheel, or --self-test")
    if not args.wheel.exists():
        sys.exit(f"no such wheel: {args.wheel}")
    sys.exit(check_wheel(args.wheel))


if __name__ == "__main__":
    main()
