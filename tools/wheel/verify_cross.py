#!/usr/bin/env python3
"""Inspect a cross-built wheel. This is not the same judgement as verify.py.

    python tools/wheel/verify_cross.py dist/torchnative-*-android_21_arm64_v8a.whl
    python tools/wheel/verify_cross.py dist/torchnative-*-ios_12_0_arm64_iphoneos.whl
    python tools/wheel/verify_cross.py dist/torchnative-*-pyemscripten_2026_0_wasm32.whl

`tools/wheel/verify.py` proves a host wheel by installing it into a clean venv
and watching `torch.__file__` come out of that venv. Nothing here can do that:
there is no interpreter on this machine that an Android or iOS wheel is for.

So this checks the two things that *are* decidable from the artefact, and says
plainly that they are not the same claim:

  the tag is one an installer will match      PEP 738 / PEP 730 spelling, run
                                              through `packaging.tags`, which is
                                              the code pip itself uses
  the contents are for that platform          every Mach-O and ELF member read
                                              directly -- architecture, and for
                                              Apple the `LC_BUILD_VERSION`
                                              platform, which is the only thing
                                              distinguishing an iphoneos build
                                              from an iphonesimulator one
  the pieces `import torch` reaches for        `torch/_C.abi3.so` under a suffix
                                              this target's interpreter actually
                                              searches, the global-deps library
                                              under the name
                                              `_load_global_deps` computes, and
                                              the wall-4 marker
  the archive is internally consistent        RECORD hashes every member, which
                                              is what pip verifies on install

The four families do not all answer the same subset of those, and the report
says which. `pyemscripten_*_wasm32` is the widest gap and the newest: a
WebAssembly module has no symbol table, no version fields and no platform
record, so the "contents are for that platform" line above degrades to
"contents are wasm32, are `dlopen`-able, and export `PyInit__C`". The version
half of the tag -- the `2026_0` that pins Pyodide's ABI -- is checked against the
Pyodide distribution on this machine and *cannot* be checked against the
artefact, because nothing in a wasm module records it. See
`PyEmscriptenExpectation`, which lists the declined questions one by one rather
than answering nearby easier ones.

What it does NOT establish: that the extension loads, that `import torch`
completes, or that any kernel computes. For Android that gap is closed
separately by `tools/wheel/verify_android.py`, which runs on a device. For iOS
it is open -- see docs/WHEEL.md §7.

Exit 0 means every check above passed. Any failure prints `FAIL:` lines and
exits 1.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binfmt import (describe, elf_dynamic, elf_info,  # noqa: E402
                    macho_info, pe_imports, pe_info, wasm_exports,
                    wasm_import_records, wasm_info)
# The manylinux checks have to agree with the builder about what PEP 599 allows
# and how glibc version names order. Importing rather than restating is the
# difference between one rule and two that can drift apart silently.
import build as _build  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# Where the cross-compiled CPython distributions live, so that the checks can be
# made against the interpreter the wheel is *for* rather than against a table
# written down here. Same default as tools/wheel/build.py.
import os  # noqa: E402

TARGET_PYTHON_ROOT = Path(os.environ.get(
    "TORCHNATIVE_TARGET_PYTHON", "/Volumes/macMini/caches/target-python"))


class Expectation:
    """What a given platform tag implies about the archive behind it."""

    def __init__(self, plat: str):
        self.plat = plat
        self.family: str
        self.interpreters: list[Path] = []

    # -- construction --------------------------------------------------------

    @staticmethod
    def parse(plat: str) -> "Expectation":
        if plat.startswith("android_"):
            return AndroidExpectation(plat)
        if plat.startswith("ios_"):
            return IOSExpectation(plat)
        if plat.startswith("manylinux"):
            return LinuxExpectation(plat)
        if plat.startswith("win"):
            return WindowsExpectation(plat)
        if plat.startswith("pyemscripten_"):
            return PyEmscriptenExpectation(plat)
        raise SystemExit(
            f"{plat!r} is not a PEP 738 android_*, PEP 730 ios_*, PEP 600 "
            "manylinux_*, win* or pyemscripten_* tag. For a host wheel use "
            "tools/wheel/verify.py, which actually installs it."
        )

    # -- checks --------------------------------------------------------------

    def check_tag(self, problems: list[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def check_binary(self, name: str, data: bytes, problems: list[str]) -> None:
        raise NotImplementedError

    #: Where the extension sits in the archive. Windows overrides it: its
    #: dynload table has no `.abi3.so` in it. Mirrors
    #: `build.Target.extension_member`, which is what put the file there.
    extension_member = "torch/_C.abi3.so"

    #: `_manager_path()` checks this exists -- but only on non-Windows, where it
    #: returns `b""` before looking. docs/VENDOR.md wall 4.
    shm_manager_required = True

    def check_linkage(self, zf: zipfile.ZipFile, names: list[str],
                      problems: list[str]) -> None:
        """Anything the *whole archive's* binaries say about each other.

        A no-op for the four families that had no use for it: their per-member
        `check_binary` is the whole of what their format makes decidable without
        a device. wasm needs it because the question worth asking there --
        "would every Python C API symbol this module imports be resolved by the
        interpreter that loads it" -- is answered by comparing two files, not by
        reading one.
        """
        return

    def global_deps_name(self) -> str | None:
        # `_load_global_deps()` builds the filename with
        #     ".dylib" if platform.system() == "Darwin" else ".so"
        # and `platform.system()` is "Android" / "iOS" on these targets -- so
        # both want `.so`, including iOS where the file is a Mach-O dylib.
        # `None` means the platform loads no such library at all (Windows).
        return "libtorch_global_deps.so"


class AndroidExpectation(Expectation):
    family = "android"

    def __init__(self, plat: str):
        super().__init__(plat)
        m = re.match(r"^android_(\d+)_(.+)$", plat)
        if not m:
            raise SystemExit(f"malformed android tag {plat!r}")
        self.api = int(m.group(1))
        self.abi = m.group(2)
        self.interpreters = sorted(
            (TARGET_PYTHON_ROOT / "aarch64-linux-android" / "prefix" / "lib")
            .glob("libpython3.*.so"))

    def check_tag(self, problems: list[str]) -> None:
        _packaging_accepts(self, problems, api_level=self.api, abi=self.abi)
        if self.api < 16:
            problems.append(
                f"API level {self.api} is below 16, which packaging treats as "
                "the floor for a CPython-capable Android; no installer would "
                "generate a matching tag")

    def check_binary(self, name: str, data: bytes, problems: list[str]) -> None:
        info = elf_info(data)
        if info is None:
            problems.append(f"{name} is not an ELF image: {describe(data)}")
            return
        want = {"arm64_v8a": "aarch64", "armeabi_v7a": "arm",
                "x86_64": "x86_64", "x86": "i386"}.get(self.abi)
        if want is None:
            problems.append(f"no known machine for Android ABI {self.abi!r}")
        elif info["machine"] != want:
            problems.append(
                f"{name} is {info['machine']}, but the tag says {self.abi} "
                f"(= {want})")
        if info["type"] != "dyn":
            problems.append(f"{name} is ELF type {info['type']}, expected dyn")


class IOSExpectation(Expectation):
    family = "ios"

    def __init__(self, plat: str):
        super().__init__(plat)
        m = re.match(r"^ios_(\d+)_(\d+)_(.+)$", plat)
        if not m:
            raise SystemExit(f"malformed ios tag {plat!r}")
        self.version = (int(m.group(1)), int(m.group(2)))
        self.multiarch = m.group(3)
        arch, _, sdk = self.multiarch.partition("_")
        self.arch = arch
        # The Mach-O platform id the SDK half of the multiarch implies. This is
        # the pair that nothing else distinguishes: an `iphoneos` and an
        # `iphonesimulator` build are the same architecture, the same size, and
        # the same symbols.
        self.platform_id = {"iphoneos": "ios",
                            "iphonesimulator": "iossimulator"}.get(sdk)
        if self.platform_id is None:
            raise SystemExit(f"unknown iOS sdk {sdk!r} in tag {plat!r}")
        subdir = {"ios": "arm64-iphoneos",
                  "iossimulator": "arm64-iphonesimulator"}[self.platform_id]
        framework = TARGET_PYTHON_ROOT / subdir / "Python.framework" / "Python"
        self.interpreters = [framework] if framework.exists() else []

    def check_tag(self, problems: list[str]) -> None:
        _packaging_accepts(self, problems, version=self.version,
                           multiarch=self.multiarch)

    def check_binary(self, name: str, data: bytes, problems: list[str]) -> None:
        info = macho_info(data)
        if info is None:
            problems.append(
                f"{name} is not a thin 64-bit Mach-O: {describe(data)}")
            return
        if info["arch"] != self.arch:
            problems.append(
                f"{name} is {info['arch']}, but the tag says {self.arch}")
        if info["platform"] != self.platform_id:
            problems.append(
                f"{name} is built for {info['platform']!r}, but the tag says "
                f"{self.platform_id!r} -- these differ in nothing else, and the "
                "wrong one fails only on the real device")
        minos = (info["minos"] or (0, 0))[:2]
        if minos > self.version:
            problems.append(
                f"{name} needs iOS {minos[0]}.{minos[1]}, above the tag's "
                f"{self.version[0]}.{self.version[1]} -- the wheel claims an "
                "OS that cannot load it")
        if info["id"] and info["id"].startswith("/"):
            problems.append(
                f"{name} advertises the build machine's path as its install "
                f"name: {info['id']}")
        if name.endswith("_C.abi3.so") and self.platform_id == "ios":
            if not any("Python.framework" in d for d in info["dylibs"]):
                problems.append(
                    f"{name} does not link Python.framework; on a physical "
                    "device there is no libpython to resolve against "
                    f"(LC_LOAD_DYLIB: {info['dylibs']})")


class LinuxExpectation(Expectation):
    """PEP 600 `manylinux_<major>_<minor>_<arch>`.

    Two things here have no counterpart in the Android and iOS branches.

    First, `packaging` cannot check the spelling. It has `android_platforms` and
    `ios_platforms` but no `manylinux_platforms`, because matching a manylinux
    tag needs the glibc of the interpreter doing the matching -- it is not a
    function of arguments. So the tag is checked against PEP 600's grammar and
    the skip is printed rather than hidden. This branch is one step weaker than
    the other two and says so.

    Second, the floor in the tag is a claim about *content*, not about a target
    SDK, so it can be checked against the members: `.gnu.version_r` records the
    `GLIBC_x.y` each image needs, and a wheel tagged below the highest of those
    installs onto a glibc that cannot load it. iOS's `minos` check is the same
    shape; Android has no equivalent because an ELF records no API level.
    """

    family = "manylinux"

    #: PEP 599's external-library list, read from the builder rather than copied
    #: so the two cannot drift into disagreeing about what manylinux promises.
    POLICY_LIBRARIES = _build.LinuxTarget.POLICY_LIBRARIES
    NAMED_VERSIONS = _build.LinuxTarget.NAMED_VERSIONS

    #: Tag arch -> ELF machine, for the archs this repository has a CPython for.
    MACHINES = {"x86_64": "x86_64", "aarch64": "aarch64", "i686": "i386"}

    def __init__(self, plat: str):
        super().__init__(plat)
        m = re.match(r"^manylinux_(\d+)_(\d+)_(.+)$", plat)
        if not m:
            raise SystemExit(
                f"malformed manylinux tag {plat!r}. The legacy aliases "
                "(manylinux1/2010/2014) are deliberately not accepted -- this "
                "repository tags with PEP 600 only")
        self.glibc = (int(m.group(1)), int(m.group(2)))
        self.arch = m.group(3)
        if self.arch not in self.MACHINES:
            raise SystemExit(f"unknown manylinux arch {self.arch!r} in {plat!r}")
        triple = {"x86_64": "x86_64-unknown-linux-gnu"}.get(self.arch)
        lib = TARGET_PYTHON_ROOT / triple / "lib" if triple else None
        self.interpreters = sorted(lib.glob("libpython3.*.so")) if lib else []

    def check_tag(self, problems: list[str]) -> None:
        if self.glibc < (2, 5):
            problems.append(
                f"the tag claims glibc {self.glibc[0]}.{self.glibc[1]}, below "
                "manylinux1's 2.5 -- PEP 600 defines the scheme no lower and no "
                "installer matches it")
            return
        print(f"  tag                 {self.plat}  "
              f"(PEP 600-shaped: glibc {self.glibc[0]}.{self.glibc[1]}, "
              f"{self.arch})")
        print("  ! packaging has no manylinux_platforms, so unlike the android "
              "and ios tags\n"
              "    this spelling is not confirmed against pip's own generator")

    def check_binary(self, name: str, data: bytes, problems: list[str]) -> None:
        info = elf_info(data)
        if info is None:
            problems.append(f"{name} is not an ELF image: {describe(data)}")
            return
        want = self.MACHINES[self.arch]
        if info["machine"] != want:
            problems.append(
                f"{name} is {info['machine']}, but the tag says {self.arch}")
        if info["type"] != "dyn":
            problems.append(f"{name} is ELF type {info['type']}, expected dyn")

        dynamic = elf_dynamic(data)
        if dynamic is None:
            problems.append(
                f"{name} has no readable dynamic section, so neither its glibc "
                "requirement nor its library dependencies can be checked "
                "against the tag")
            return

        outside = sorted(set(dynamic["needed"]) - self.POLICY_LIBRARIES)
        if outside:
            problems.append(
                f"{name} links {outside}, which PEP 599's external-library "
                "policy does not allow -- the manylinux tag's whole promise is "
                "that it needs nothing beyond that list")

        needs = self._glibc_needed(name, dynamic, problems)
        if needs and needs > self.glibc:
            problems.append(
                f"{name} needs glibc {needs[0]}.{needs[1]}, above the tag's "
                f"{self.glibc[0]}.{self.glibc[1]} -- the wheel claims a glibc "
                "that cannot load it")
        elif needs:
            print(f"  glibc               {name} needs {needs[0]}.{needs[1]} "
                  f"<= the tag's {self.glibc[0]}.{self.glibc[1]}")
        else:
            # The empty global-deps stub lands here and is right to: it
            # references no libc symbol, so it constrains nothing. Saying "no
            # requirement" out loud keeps that from reading as a passed check.
            print(f"  glibc               {name} records no requirement "
                  "(it references no versioned symbol)")

    def _glibc_needed(self, name: str, dynamic: dict,
                      problems: list[str]) -> tuple[int, int] | None:
        floor: tuple[int, int] | None = None
        for library, versions in dynamic["versions"].items():
            for version in sorted(versions):
                if not version.startswith("GLIBC_"):
                    continue
                parts = version[len("GLIBC_"):].split(".")
                if all(p.isdigit() for p in parts) and len(parts) >= 2:
                    value = (int(parts[0]), int(parts[1]))
                elif version in self.NAMED_VERSIONS:
                    value = self.NAMED_VERSIONS[version]
                else:
                    problems.append(
                        f"{name} needs glibc symbol version {version!r} from "
                        f"{library}, which this does not know how to order; "
                        "refused rather than skipped, because skipping one can "
                        "only make the requirement look lower than it is")
                    continue
                floor = value if floor is None else max(floor, value)
        return floor


class WindowsExpectation(Expectation):
    """`win_amd64` and its two siblings, which are names rather than derivations.

    There is no version in a Windows wheel tag -- no API level, no deployment
    target, no glibc floor -- so unlike the other three families there is
    nothing here that can be *derived* wrongly. The whole of what the tag claims
    is an architecture, and that is checked against the PE header.

    The two interesting things are structural rather than about the tag, and
    both come from upstream torch and CPython rather than from a choice here:

      the member is `torch/_C.pyd`     `dynload_win.c` has no `.abi3.so` in its
                                       table, so the POSIX name would never be
                                       found; plain `.pyd` is the abi3 spelling
                                       (`.cp313-win_amd64.pyd` is the
                                       version-locked one)
      no global-deps library at all    `_load_global_deps()` returns before
                                       doing anything on Windows, and
                                       `_load_dll_libraries()` LoadLibrary's
                                       every `torch/lib/*.dll` -- so an empty
                                       one would be a load for nothing

    `torch/bin/torch_shm_manager` is likewise not required: `_manager_path()`
    returns `b""` on Windows before it checks.
    """

    family = "windows"

    MACHINES = {"win_amd64": "x86_64", "win32": "i386", "win_arm64": "aarch64"}

    extension_member = "torch/_C.pyd"
    shm_manager_required = False

    def __init__(self, plat: str):
        super().__init__(plat)
        if plat not in self.MACHINES:
            raise SystemExit(
                f"unknown Windows tag {plat!r}; the only ones pip generates are "
                f"{sorted(self.MACHINES)}")
        self.arch = self.MACHINES[plat]
        root = TARGET_PYTHON_ROOT / "x86_64-pc-windows-msvc"
        # python313.dll, not python3.dll: the dynload table is a set of string
        # constants compiled into the interpreter, and python3.dll is only a
        # forwarder with no code of its own.
        self.interpreters = sorted(root.glob("python3??.dll"))

    def global_deps_name(self) -> str | None:
        return None

    def check_tag(self, problems: list[str]) -> None:
        # `packaging.tags` has no windows_platforms: on Windows it derives the
        # tag from `sysconfig.get_platform()` of the *running* interpreter, so
        # there is no generator to hand arguments to. Same shape of gap as
        # manylinux, and reported the same way rather than hidden.
        print(f"  tag                 {self.plat}  "
              f"(a fixed name; {self.arch}, no version component)")
        print("  ! packaging has no windows_platforms, so unlike the android "
              "and ios tags\n"
              "    this spelling is not confirmed against pip's own generator")

    def check_binary(self, name: str, data: bytes, problems: list[str]) -> None:
        info = pe_info(data)
        if info is None:
            problems.append(f"{name} is not a PE image: {describe(data)}")
            return
        if info["machine"] != self.arch:
            problems.append(
                f"{name} is {info['machine']}, but the tag says {self.plat} "
                f"(= {self.arch})")
        if info["bits"] != (32 if self.plat == "win32" else 64):
            problems.append(f"{name} is PE{info['bits']}, wrong for {self.plat}")
        if not info["dll"]:
            problems.append(
                f"{name} is a PE executable, not a DLL -- an extension module "
                "has to be a DLL for LoadLibrary to take it")
        if name != self.extension_member:
            return
        # The abi3 promise, checked against the file rather than the build
        # flags: an extension bound to python313.dll serves 3.13 alone, however
        # the wheel is tagged. PE says which, because every import names its DLL.
        imports = pe_imports(data) or {}
        if "python3.dll" not in imports:
            problems.append(
                f"{name} imports from {sorted(imports) or 'nothing'} and not "
                "from python3.dll -- an abi3 extension binds the stable-ABI "
                "forwarder, and binding python313.dll makes the abi3 tag false")
        else:
            print(f"  abi3 binding        {len(imports['python3.dll'])} imports "
                  "from python3.dll (the stable-ABI forwarder)")


# Where the Pyodide distribution lives. Same shape as `TARGET_PYTHON_ROOT` above
# and deliberately a *second* variable rather than a subdirectory of it: the
# thing being read here is not a CPython distribution. It is a Pyodide one, and
# the difference is the whole of docs/WASM.md §9.4a -- the tag's version
# component exists only in Pyodide's `pyodide-lock.json`, and the CPython inside
# it answers a different, plausible, wrong tag.
PYODIDE_ROOT = Path(os.environ.get(
    "TORCHNATIVE_PYODIDE", "/Volumes/macMini/caches/pyodide/pyodide"))


class PyEmscriptenExpectation(Expectation):
    """`pyemscripten_<abi-version>_wasm32` -- and the family that can answer least.

    Every other family here is checked against a *symbol table*. WebAssembly has
    none. What it has is an import section and an export section, which name
    every symbol the host must supply and every symbol offered, as strings, in
    the module. So the linkage questions get easier and the platform questions
    become unanswerable, and those two facts have to be reported separately
    rather than netted off against each other.

    **Answered here, from the artefact:**

      wasm32, not wasm64      the memory type's limits flags. A side module
                              imports its memory, so this is read from the
                              import section (`binfmt._wasm_memory_bits`).
      loadable at all         a side module imports `env.memory` and
                              `env.__indirect_function_table`; a main-module
                              link defines and exports them, and Emscripten's
                              `dlopen` refuses it. This is the wasm spelling of
                              "is it a DLL and not an EXE", and docs/WASM.md
                              §9.2 hit it: `ctypes.CDLL` on the global-deps stub
                              is `dlopen`, so *both* wasm members have to be
                              side modules, not only the extension.
      `PyInit__C` is exported, and is a *function*. `dlsym` finds it and the
                              import system calls it; an export of that name
                              with kind `global` is a linker accident that fails
                              at import with nothing pointing at the module.
      nothing linked in that should not be -- `check_linkage` below.

    **Declined here, and each one is a question another family does answer:**

      the platform             Mach-O has `LC_BUILD_VERSION`; a wasm module
                               records no Emscripten version, no Pyodide ABI
                               version and no minimum anything. `2026_0` in the
                               tag is checked against `pyodide-lock.json`, which
                               is a check on *this machine's distribution*, not
                               on the wheel. If the wheel were built against a
                               different Pyodide, nothing in the wasm bytes
                               would say so.
      the abi3 binding         `WindowsExpectation` proves it by reading which
                               DLL each import is attributed to: `python3.dll`
                               and not `python313.dll`. Emscripten attributes
                               every undefined symbol to the single import
                               module `env`, so that distinction has no wasm
                               spelling. docs/WASM.md §9.3 makes this less
                               costly than it sounds -- the platform tag pins
                               CPython 3.14 and Emscripten 5.0.3 together, so
                               the abi3 field is inert here -- but inert is not
                               checked, and it is reported as unchecked.
      a manylinux-style floor  no counterpart, same as Mach-O and Android.
      whether every import resolves. Only the Python C API subset is decidable;
                               see `check_linkage`.
    """

    family = "pyemscripten"

    #: Supplied per-module by the dynamic loader when it places the module in
    #: memory, so they are undefined in the artefact by construction and their
    #: absence from the host's export table means nothing.
    LOADER_SUPPLIED = frozenset({"__memory_base", "__table_base"})

    def __init__(self, plat: str):
        super().__init__(plat)
        m = re.match(r"^pyemscripten_(\d+_\d+)_(wasm\d+)$", plat)
        if not m:
            raise SystemExit(
                f"malformed pyemscripten tag {plat!r}; the spelling Pyodide "
                "publishes is pyemscripten_<abi-version>_wasm32, e.g. "
                "pyemscripten_2026_0_wasm32")
        self.abi_version = m.group(1)
        self.arch = m.group(2)
        # `pyodide.asm.wasm` is the main module, and it is what
        # `check_suffix_is_searched` reads: CPython's `_PyImport_DynLoadFiletab`
        # is compiled into it as string constants exactly as it is into
        # `libpython3.x.so` on Android, so that check needs no wasm exception.
        main = PYODIDE_ROOT / "pyodide.asm.wasm"
        self.interpreters = [main] if main.exists() else []
        self.lock = PYODIDE_ROOT / "pyodide-lock.json"

    def check_tag(self, problems: list[str]) -> None:
        # `packaging.tags` has no `pyemscripten_platforms`. On the target itself
        # `sys_tags()` does yield this tag (docs/WASM.md §9.3 read 110 of them
        # off the real interpreter), but that generator reads
        # `sys.implementation._multiarch` and Pyodide's own patches, neither of
        # which exists here. Same shape of gap as manylinux and Windows, and
        # reported the same way rather than hidden.
        print(f"  tag                 {self.plat}  "
              f"(Pyodide ABI {self.abi_version}, {self.arch})")
        print("  ! packaging has no pyemscripten_platforms, so unlike the "
              "android and ios tags\n"
              "    this spelling is not confirmed against pip's own generator")
        # The one thing that *can* be confirmed, and the trap docs/WASM.md §9.4a
        # names: the version component is Pyodide's `PYODIDE_ABI_VERSION`, which
        # the CPython inside Pyodide has never heard of. Asking the interpreter
        # instead yields `emscripten_5_0_3_wasm32` -- accepted by `packaging`,
        # used by nothing. So it is checked against the distribution that owns
        # the number.
        before = len(problems)
        if not self.lock.exists():
            problems.append(
                f"no {self.lock} -- the {self.abi_version} in this tag is "
                "Pyodide's PYODIDE_ABI_VERSION and exists in no other "
                "machine-readable place (docs/WASM.md §9.4a). Without the "
                "distribution it cannot be checked at all, and it is the one "
                "component of this tag that is checkable; set "
                "TORCHNATIVE_PYODIDE")
            return
        try:
            info = json.loads(self.lock.read_text())["info"]
        except (ValueError, KeyError) as exc:
            problems.append(f"{self.lock} has no readable info block: {exc}")
            return
        if info.get("abi_version") != self.abi_version:
            problems.append(
                f"tag says Pyodide ABI {self.abi_version}, but {self.lock.name} "
                f"says {info.get('abi_version')!r} -- a wheel tagged for an ABI "
                "version this Pyodide does not have is one `micropip` will "
                "refuse by tag, before any of the contents matter")
        if info.get("arch") != self.arch:
            problems.append(
                f"tag says {self.arch}, but {self.lock.name} says "
                f"{info.get('arch')!r}")
        if len(problems) == before:
            print(f"  Pyodide             abi_version={info.get('abi_version')} "
                  f"platform={info.get('platform')} python={info.get('python')}")

    def check_binary(self, name: str, data: bytes, problems: list[str]) -> None:
        info = wasm_info(data)
        if info is None:
            problems.append(
                f"{name} is not a readable WebAssembly module: "
                f"{describe(data)}")
            return
        if info["machine"] != self.arch:
            problems.append(
                f"{name} is {info['machine']}, but the tag says {self.arch}")
        if not info["side_module"]:
            problems.append(
                f"{name} is a main-module link, not a side module -- it defines "
                "its own memory instead of importing one, and Emscripten's "
                "`dlopen` cannot load it. Both wasm members go through `dlopen` "
                "(the extension via the import system, the global-deps library "
                "via `ctypes.CDLL`), so this makes the wheel fail at import "
                "(docs/WASM.md §9.2)")
        exports = wasm_exports(data)
        if exports is None:
            problems.append(
                f"{name} has an unreadable export section -- reported rather "
                "than treated as exporting nothing")
            return
        if name != self.extension_member:
            return
        kind = exports.get("PyInit__C")
        if kind is None:
            near = sorted(e for e in exports if e.startswith("PyInit"))
            problems.append(
                f"{name} exports no PyInit__C -- the import system `dlsym`s "
                "exactly that name and raises ImportError when it is absent. "
                f"Its export section has {len(exports)} entries"
                + (f", including {near}" if near else
                   " and no PyInit_* at all"))
        elif kind != "func":
            problems.append(
                f"{name} exports PyInit__C as a {kind}, not a function -- "
                "`dlsym` finds it and the import system calls it")
        else:
            print(f"  wasm entry point    PyInit__C (func), "
                  f"{len(exports)} exports")

    def check_linkage(self, zf: zipfile.ZipFile, names: list[str],
                      problems: list[str]) -> None:
        """Would the interpreter resolve what the modules ask it for?

        Scoped to `Py*` / `_Py*` deliberately, and the scoping is the finding
        rather than a shortcut. Emscripten puts every undefined symbol under the
        import module `env`, and three different things end up there:

          * Python C API symbols, which **must** come out of the interpreter --
            they are compiled into `pyodide.asm.wasm`, and one that is missing is
            a module that cannot load;
          * C runtime symbols the Emscripten **JS** library supplies at load time
            (`exit`, `__assert_fail`), which are not in the main module's export
            section at all;
          * symbols another **side module in the same wheel** provides.

        Only the first is decidable from the artefacts. Measured against the six
        Pyodide wheels and one main module on this machine: all 2,736 `Py*`
        imports across those 32 modules resolve, while `kiwisolver` alone has 341
        unresolved non-`Py` ones and is a shipped, working package. Checking the
        whole `env` set would therefore reject correct wheels -- so the narrow
        check is the true one, and the breadth it does not have is stated here
        instead of being quietly assumed away.
        """
        if not self.interpreters:
            problems.append(
                f"no {PYODIDE_ROOT / 'pyodide.asm.wasm'} to resolve the wasm "
                "imports against; set TORCHNATIVE_PYODIDE")
            return
        host = wasm_exports(self.interpreters[0].read_bytes())
        if host is None:
            problems.append(
                f"{self.interpreters[0]} has no readable export section -- the "
                "interpreter side of the linkage check could not be read, "
                "which is not the same as everything resolving")
            return
        checked = 0
        for member in names:
            if member.endswith("/"):
                continue
            data = zf.read(member)
            if wasm_info(data) is None:
                continue
            records = wasm_import_records(data)
            if records is None:
                problems.append(
                    f"{member} has an unreadable import section -- reported "
                    "rather than treated as importing nothing")
                continue
            wanted = [f for module, f, _kind in records
                      if module == "env"
                      and (f.startswith("Py") or f.startswith("_Py"))
                      and f not in self.LOADER_SUPPLIED]
            missing = sorted(set(wanted) - set(host))
            checked += len(wanted)
            if missing:
                problems.append(
                    f"{member} imports {len(missing)} Python C API symbol(s) "
                    f"{self.interpreters[0].name} does not export, e.g. "
                    f"{missing[:5]} -- the module would fail to instantiate")
        print(f"  wasm linkage        {checked} Py* imports all resolved by "
              f"{self.interpreters[0].name}")
        print("  ! not checked: non-Py `env` imports. Emscripten's JS library "
              "supplies some\n"
              "    of them at load time and sibling side modules supply "
              "others, and neither\n"
              "    is visible in any artefact here (shipped Pyodide wheels "
              "have hundreds)")
        print("  ! not checked: the abi3 binding. Every wasm import is "
              "attributed to `env`,\n"
              "    so the python3.dll-vs-python313.dll test "
              "WindowsExpectation runs has no\n"
              "    wasm spelling (docs/WASM.md §9.3)")


def _packaging_accepts(exp: Expectation, problems: list[str], **kwargs) -> None:
    """Would pip's own tag generator produce this platform tag?

    `packaging.tags` is what pip uses to decide whether a wheel is for the
    machine it is running on, so asking it is the difference between checking
    the spelling against the specification and checking it against the
    implementation. It is not a substitute for the content checks: it looks at
    the string only.
    """
    try:
        from packaging import tags as ptags
    except ImportError:
        problems.append("packaging is not importable -- the tag was not checked "
                        "against the code pip uses")
        return
    generator = getattr(ptags, f"{exp.family}_platforms", None)
    if generator is None:
        problems.append(
            f"packaging in this environment has no {exp.family}_platforms; it "
            "is older than PEP 738/730 support and cannot check this tag")
        return
    accepted = list(generator(**kwargs))
    if exp.plat not in accepted:
        problems.append(
            f"packaging.tags.{exp.family}_platforms({kwargs}) does not yield "
            f"{exp.plat!r}; it starts {accepted[:3]}")
    else:
        print(f"  tag                 {exp.plat}  "
              f"(accepted by packaging.tags.{exp.family}_platforms)")


def _is_binary(data: bytes) -> bool:
    return (macho_info(data) is not None or elf_info(data) is not None
            or pe_info(data) is not None or wasm_info(data) is not None)


def check_record(zf: zipfile.ZipFile, dist_info: str,
                 problems: list[str]) -> None:
    """Every member hashed, every hash right.

    pip verifies RECORD on install, so a repack that adds a file without
    updating it produces a wheel that fails at the moment it looks like it
    worked. Checking it here means that failure mode cannot reach a device.
    """
    record_name = f"{dist_info}/RECORD"
    if record_name not in zf.namelist():
        problems.append(f"no {record_name}")
        return
    listed: dict[str, str] = {}
    for row in csv.reader(io.StringIO(zf.read(record_name).decode())):
        if row:
            listed[row[0]] = row[1] if len(row) > 1 else ""
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if name not in listed:
            problems.append(f"{name} is in the archive but not in RECORD")
            continue
        if name == record_name:
            continue
        want = listed[name]
        got = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(zf.read(name)).digest()).rstrip(b"=").decode()
        if want != got:
            problems.append(f"{name}: RECORD says {want}, content is {got}")
    for name in listed:
        if name not in zf.namelist():
            problems.append(f"{name} is in RECORD but not in the archive")


def check_suffix_is_searched(exp: Expectation, problems: list[str]) -> None:
    """Will the target interpreter even look for `_C.abi3.so`?

    CPython's `_PyImport_DynLoadFiletab` (dynload_shlib.c) is
    `{SOABI suffix, ".abi3.so", SHLIB_SUFFIX}`, and those are string constants
    compiled into the interpreter. Reading them out of the target binary is the
    strongest form this question takes without a device: if `.abi3.so` were not
    in that table the extension would simply never be found, and the failure
    would be a `ModuleNotFoundError` with nothing pointing here.

    (For Android this is also directly measured -- `scripts/device_android.sh`
    records the list from the device as
    `['.cpython-313-aarch64-linux-android.so', '.abi3.so', '.so']`.)

    Windows has a different table in a different file: `dynload_win.c` gives
    `{"_d.pyd", ".cp313-win_amd64.pyd", ".pyd"}` and no `.abi3.so` anywhere. So
    the suffix asked about comes from the expectation rather than being written
    down here -- which is the same fact that makes the member `_C.pyd`.
    """
    if not exp.interpreters:
        problems.append(
            "no target CPython found to check the extension suffix against "
            f"(looked under {TARGET_PYTHON_ROOT}); set TORCHNATIVE_TARGET_PYTHON")
        return
    suffix = "." + exp.extension_member.split("_C.", 1)[1]
    needle = suffix.encode() + b"\x00"
    for path in exp.interpreters:
        blob = path.read_bytes()
        if needle not in blob:
            problems.append(
                f"{path} does not contain the extension suffix {suffix!r} -- "
                f"this interpreter would never find {exp.extension_member}")
        else:
            print(f"  ext suffix          {suffix} present in {path.name}")


# ------------------------------------------------------------------ self-test
#
# "실패할 수 없는 검증은 검증이 아니다" (CLAUDE.md §5.5). Everything above passes
# on the wheels this repository builds, which says nothing on its own -- an empty
# function passes too. So each check is given a wheel it must reject, built by
# damaging a good one in exactly the way that check exists to notice.
#
# The damage is done by patching header fields rather than by substituting some
# other artefact, so the self-test needs nothing on disk beyond the wheel itself:
# every fault mode runs every time, and none of them can be quietly skipped.


def _patch(data: bytes, offset: int, raw: bytes) -> bytes:
    return data[:offset] + raw + data[offset + len(raw):]


def _wrong_pe_machine(data: bytes) -> bytes:
    """COFF `Machine` := the other 64-bit one. An aarch64 DLL under an amd64 tag."""
    import struct as _s
    from binfmt import _pe_headers
    coff = _pe_headers(data)[0]
    IMAGE_FILE_MACHINE_AMD64, IMAGE_FILE_MACHINE_ARM64 = 0x8664, 0xAA64
    (current,) = _s.unpack_from("<H", data, coff)
    other = (IMAGE_FILE_MACHINE_ARM64 if current == IMAGE_FILE_MACHINE_AMD64
             else IMAGE_FILE_MACHINE_AMD64)
    return _patch(data, coff, _s.pack("<H", other))


def _wrong_elf_machine(data: bytes) -> bytes:
    """Retarget the image at an architecture the tag does not claim.

    Flipped rather than assigned. Written as `e_machine := EM_X86_64` it damaged
    an aarch64 Android wheel and silently did nothing to an x86-64 manylinux
    one, so the fault mode passed by being absent -- exactly the shape CLAUDE.md
    §5.5 warns about, and it survived until a Linux wheel existed to run it on.
    """
    import struct as _s
    EM_X86_64, EM_AARCH64 = 0x3E, 0xB7
    current = _s.unpack_from("<H", data, 18)[0]
    other = EM_AARCH64 if current == EM_X86_64 else EM_X86_64
    if current == other:                                  # pragma: no cover
        raise SystemExit("cannot flip e_machine away from itself -- "
                         "self-test cannot run")
    return _patch(data, 18, _s.pack("<H", other))


def _wrong_macho_platform(data: bytes) -> bytes:
    """Retarget the image at a platform the tag does not claim.

    With `LC_BUILD_VERSION`, flip its platform field between ios (2) and
    iossimulator (7) -- the one difference between the device and the simulator
    artefact that no size, architecture or symbol check would see.

    The iOS *device* build has no such command: Rust's default deployment
    target for `aarch64-apple-ios` is 10.0, which predates it, so the image
    carries `LC_VERSION_MIN_IPHONEOS` instead. There the equivalent damage is to
    make it `LC_VERSION_MIN_MACOSX`, i.e. a macOS dylib wearing an iOS tag.
    """
    import struct as _s
    from binfmt import LC_BUILD_VERSION
    ncmds = _s.unpack_from("<I", data, 16)[0]
    off = 32
    for _ in range(ncmds):
        cmd, cmdsize = _s.unpack_from("<II", data, off)
        if cmd == LC_BUILD_VERSION:
            plat = _s.unpack_from("<I", data, off + 8)[0]
            return _patch(data, off + 8, _s.pack("<I", 7 if plat == 2 else 2))
        if cmd == 0x25:  # LC_VERSION_MIN_IPHONEOS -> LC_VERSION_MIN_MACOSX
            return _patch(data, off, _s.pack("<I", 0x24))
        off += cmdsize
    raise SystemExit("no version-min load command to corrupt -- "
                     "self-test cannot run")


def _wrong_wasm_memory_bits(data: bytes) -> bytes:
    """Flip the imported memory's index type between wasm32 and wasm64.

    The counterpart of `_wrong_elf_machine`, and the *only* header field a wasm
    module has that a platform claim can be checked against. Bit 0x04 of the
    memory type's limits flags is the memory64 bit, so this is a one-byte patch
    in the import section that leaves every other structure intact -- which is
    the point: nothing about the size, the exports or the code changes, exactly
    as with the iphoneos/iphonesimulator flip.
    """
    from binfmt import _uleb, _wasm_name
    # The absolute offset of the import section body, walked here rather than
    # taken from `_wasm_sections`, which hands back copies and not positions.
    i = 8
    while i < len(data):
        sid = data[i]
        i += 1
        size, i = _uleb(data, i)
        if sid == 2:
            break
        i += size
    else:                                                 # pragma: no cover
        raise SystemExit("no import section to corrupt -- self-test cannot run")
    body_at = i
    body = data[body_at:body_at + size]
    count, j = _uleb(body, 0)
    for _ in range(count):
        _module, j = _wasm_name(body, j)
        _field, j = _wasm_name(body, j)
        kind = body[j]
        j += 1
        if kind == 2:
            return _patch(data, body_at + j, bytes([body[j] ^ 0x04]))
        if kind == 0:
            _, j = _uleb(body, j)
        elif kind == 1:
            j += 1
            flags = body[j]
            j += 1
            _, j = _uleb(body, j)
            if flags & 0x01:
                _, j = _uleb(body, j)
        elif kind == 3:
            j += 2
        elif kind == 4:
            j += 1
            _, j = _uleb(body, j)
        else:                                             # pragma: no cover
            raise SystemExit(f"unknown import kind {kind}")
    raise SystemExit(                                     # pragma: no cover
        "the module defines its own memory rather than importing one -- "
        "self-test cannot run")


def _wasm_without_pyinit(data: bytes) -> bytes:
    """Make `PyInit__C` absent from the export section.

    Renamed to `PyInit__X` rather than deleted, on purpose and for two reasons.
    It is an in-place patch of the same length, so no section size, no LEB and
    no index has to be re-encoded and the damaged module stays well-formed --
    a truncated one would be rejected by `_wasm_sections` returning `None`,
    which is a *different* check firing and would make this fault mode a lie.
    And it leaves a near-miss behind, so the report has to name the symbol it
    could not find rather than only counting the exports it did.
    """
    needle = b"\x09PyInit__C"            # the export's length-prefixed name
    at = data.find(needle)
    if at < 0:                                            # pragma: no cover
        raise SystemExit("no PyInit__C export to corrupt -- "
                         "self-test cannot run")
    return _patch(data, at + len(needle) - 1, b"X")


def _rewrite(src: Path, dst: Path, *, drop=(), replace=None, add=None,
             rename=None, wheel_tag=None) -> Path:
    """Copy a wheel, dropping/replacing/adding members. RECORD is deliberately
    *not* regenerated -- for the faults that change content this is itself one
    of the things being tested."""
    replace = replace or {}
    out = dst / (rename or src.name)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(
            out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename in drop:
                continue
            data = zin.read(item.filename)
            if item.filename in replace:
                data = replace[item.filename](data)
            if wheel_tag and item.filename.endswith(".dist-info/WHEEL"):
                data = re.sub(rb"Tag: .*", b"Tag: " + wheel_tag.encode(), data)
            zout.writestr(item, data)
        for arcname, source in (add or {}).items():
            zout.writestr(arcname, zipfile.ZipFile(src).read(source))
    return out


def self_test(wheel: Path, reference: Path | None) -> int:
    import subprocess
    import tempfile

    stem = wheel.stem
    plat = stem.split("-")[-1]
    exp = Expectation.parse(plat)
    with zipfile.ZipFile(wheel) as zf:
        ext_data = zf.read(exp.extension_member)
    is_macho = macho_info(ext_data) is not None
    is_pe = pe_info(ext_data) is not None
    is_wasm = wasm_info(ext_data) is not None
    is_manylinux = plat.startswith("manylinux")
    name, version = stem.split("-")[0], stem.split("-")[1]
    dist_info = f"{name}-{version}.dist-info"
    deps_name = exp.global_deps_name()
    deps = f"torch/lib/{deps_name}" if deps_name else None

    if is_pe:
        damage = _wrong_pe_machine
        expect_platform = "but the tag says"
    elif is_macho:
        damage = _wrong_macho_platform
        expect_platform = "built for"
    elif is_wasm:
        damage = _wrong_wasm_memory_bits
        expect_platform = "but the tag says"
    else:
        damage = _wrong_elf_machine
        expect_platform = "but the tag says"

    # Each entry: what is broken, how, and the words the report must contain.
    # Matching on the message rather than only on the exit code is what keeps a
    # fault from being "caught" by an unrelated check -- the file-list
    # comparison would otherwise absorb almost all of these.
    faults: list[tuple[str, dict, str]] = [
        ("extension built for the wrong platform",
         {"replace": {exp.extension_member: damage}},
         expect_platform),
        ("a member edited without updating RECORD",
         {"replace": {"torch/version.py": lambda d: d + b"\n# tampered\n"}},
         "RECORD says"),
        ("part of the vendored tree dropped",
         {"drop": tuple(_some_tree_files(wheel))},
         "are missing here"),
        ("extension under the POSIX name this platform does not search"
         if is_pe else "extension missing",
         {"drop": (exp.extension_member,),
          **({"add": {"torch/_C.abi3.so": exp.extension_member}} if is_pe else {})},
         "beside torch/_C.pyd" if is_pe else f"no {exp.extension_member}"),
        ("WHEEL Tag: out of step with the filename",
         {"wheel_tag": f"cp313-abi3-{plat}-BOGUS"},
         "declares Tag:"),
        ("platform tag no installer would generate",
         {"rename": stem.replace(plat, _impossible(plat)) + ".whl",
          "wheel_tag": f"cp313-abi3-{_impossible(plat)}"},
         "below manylinux1's 2.5" if is_manylinux
         else "the only ones pip generates" if plat.startswith("win")
         else "says Pyodide ABI" if is_wasm
         else "does not yield"),
        ("abi tag downgraded from abi3",
         {"rename": stem.replace("-abi3-", "-cp313-") + ".whl"},
         "not 'abi3'"),
    ]

    if deps:
        faults[1:1] = [
            ("global-deps library missing",
             {"drop": (deps,)},
             "_load_global_deps"),
            ("global-deps library under the host's name",
             {"drop": (deps,),
              "add": {"torch/lib/libtorch_global_deps.dylib": deps}},
             "beside"),
        ]
    else:
        # The absence is by design here, so the fault mode is the *presence* of
        # one -- `_load_dll_libraries()` would LoadLibrary it for nothing, and
        # a failure there is raised rather than swallowed.
        faults.insert(1, (
            "a global-deps library on a platform that loads none",
            {"add": {"torch/lib/libtorch_global_deps.dll": exp.extension_member}},
            "returns before looking",
        ))

    if is_wasm:
        # The fault this whole family exists to make catchable, and the reason
        # docs/WASM.md §9.4b put the checker before the target: an extension
        # with no entry point is a wheel that installs, imports, and raises
        # ImportError -- no size, tag, architecture or RECORD check sees it.
        faults.append((
            "extension exports no PyInit__C",
            {"replace": {exp.extension_member: _wasm_without_pyinit}},
            "exports no PyInit__C",
        ))

    if exp.shm_manager_required:
        faults.insert(-2, ("wall-4 marker missing",
                           {"drop": ("torch/bin/torch_shm_manager",)},
                           "torch_shm_manager"))

    if is_manylinux:
        # Only this family can be retagged *within* its own valid range and
        # still be wrong: the floor is a claim about the members' own
        # `.gnu.version_r`, not about a target SDK. manylinux_2_5 is a tag
        # every installer on a modern distro happily matches, and the wheel
        # would then be handed to a glibc that cannot load `_C.abi3.so`.
        # Android has no counterpart at all; iOS's is the `minos` check.
        floor = "manylinux_2_5_" + plat.split("_", 3)[3]
        faults.append((
            "tag floor below the glibc the members actually need",
            {"rename": stem.replace(plat, floor) + ".whl",
             "wheel_tag": f"cp313-abi3-{floor}"},
            "above the tag's 2.5",
        ))

    print(f"SELF-TEST against {wheel.name}")
    print(f"  {len(faults)} fault modes; each must be reported, with the "
          "expected reason\n")
    caught = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for label, kwargs, expect in faults:
            broken = _rewrite(wheel, tmp, **kwargs)
            argv = [sys.executable, __file__, str(broken)]
            if reference:
                argv += ["--reference", str(reference)]
            proc = subprocess.run(argv, capture_output=True, text=True)
            report = proc.stdout + proc.stderr
            if proc.returncode == 0:
                print(f"  NOT CAUGHT  {label}")
            elif expect not in report:
                print(f"  WRONG REASON {label}")
                print(f"      wanted {expect!r}, got:")
                for line in report.splitlines():
                    if line.startswith("FAIL:"):
                        print(f"        {line}")
            else:
                caught += 1
                print(f"  caught      {label}")
            broken.unlink()

    # `_default_reference`'s own version-scoping, exercised directly rather
    # than through a tampered wheel -- the counterpart of `find_sibling`'s
    # cases 6/7 in verify_ios_device.py. `dist/` holds several released
    # versions side by side by design; an unscoped `sorted(glob(...))[-1]`
    # (what this function replaced) picks whichever sorts last regardless of
    # which version `wheel` is. That shape of bug, in `find_sibling`, compared
    # a 0.0.4a0 wheel against a 0.0.2a0 sibling on 2026-08-30 and reported
    # PASS -- and here it is worse than harmless: every one of the ~120
    # `{name}-{version}.data/...` paths differs by construction between any
    # two versions, so a version-mismatched reference guarantees "are missing
    # here" in the report whether or not a real drop happened, which is
    # exactly the phrase the "part of the vendored tree dropped" fault mode
    # above checks for -- an unscoped picker would make that fault mode catch
    # nothing, silently, by being permanently pre-satisfied.
    with tempfile.TemporaryDirectory() as tmp:
        multiversion = Path(tmp)
        for v in ("0.0.2a0", "0.0.3a0", "0.0.4a0"):
            (multiversion / f"{name}-{v}-cp313-abi3-macosx_11_0_arm64.whl").touch()
        old = multiversion / f"{name}-0.0.2a0-cp313-abi3-{plat}.whl"
        old.touch()
        want = multiversion / f"{name}-0.0.2a0-cp313-abi3-macosx_11_0_arm64.whl"
        picked = _default_reference(old)
        ref_ok = picked == want
    print(f"  {'caught      ' if ref_ok else 'NOT CAUGHT  '}"
          "_default_reference picks the SAME-VERSION macosx wheel beside a "
          "wheel, not whichever version sorts last")
    if not ref_ok:
        print(f"      wanted {want}, got {picked}")
    caught += ref_ok
    total = len(faults) + 1

    print()
    if caught != total:
        print(f"SELF-TEST: FAIL -- {total - caught} of {total} fault "
              "modes were not reported for the right reason", file=sys.stderr)
        return 1
    print(f"SELF-TEST: PASS -- {caught}/{total} fault modes rejected")
    return 0


def _some_tree_files(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return [n for n in zf.namelist()
                if n.startswith("torchgen/") and n.endswith(".py")][:20]


def _impossible(plat: str) -> str:
    """A same-shaped tag no installer would ever match.

    Below each family's own floor -- API 15 for Android, iOS 11, glibc 2.1 --
    so the rejection comes from the family's own defined lower bound and not
    from a spelling rule invented here. For Android and iOS that bound is
    enforced by `packaging`'s generator; for manylinux `packaging` has no
    generator, so it is PEP 600's own floor of 2.5 (manylinux1).
    """
    if plat.startswith("android_"):
        return re.sub(r"^android_\d+", "android_15", plat)
    if plat.startswith("manylinux"):
        return re.sub(r"^manylinux_\d+_\d+", "manylinux_2_1", plat)
    if plat.startswith("pyemscripten_"):
        # No floor to go below -- Pyodide's ABI versions are dates, not a range
        # with a bottom, and `packaging` has no generator to refuse one. So the
        # impossible tag is an ABI version the Pyodide distribution on this
        # machine does not have, which is the same source the real tag is
        # derived from rather than a rule invented here.
        return re.sub(r"^pyemscripten_\d+_\d+", "pyemscripten_1970_0", plat)
    if plat.startswith("win"):
        # No version to lower, so the impossible tag is an architecture pip has
        # no name for. `WindowsExpectation` refuses it by the same rule that
        # says there are only three.
        return "win_ia64"
    return re.sub(r"^ios_\d+_\d+", "ios_11_0", plat)


def _default_reference(wheel: Path) -> Path | None:
    """The `{name}-{version}-*macosx*.whl` beside `wheel`, if there is one.

    Version-scoped on purpose. `tools/wheel/verify_ios_device.py`'s
    `find_sibling` had the unscoped form of this picker --
    `sorted(glob(...))[0]` over every version in the directory -- and on
    2026-08-30 it compared a 0.0.4a0 device wheel against a 0.0.2a0 simulator
    wheel and reported PASS. It was harmless there only because the vendored
    tree had not changed between those two releases, which is luck, not a
    check. This repository's `dist/` holds several versions side by side by
    design (0.0.2a0 through 0.0.4a0 as of 2026-08-31), so an unscoped picker
    here is not a hypothetical: `sorted(glob("*macosx*.whl"))[-1]` -- what
    this function replaces below at the `--self-test` call site -- silently
    picked the newest macosx wheel in the directory regardless of which
    version `wheel` is, and every one of the ~120 `{name}-{version}.data/...`
    paths differs by construction between any two versions. That is enough
    spurious "missing"/"gained" noise on its own to make the self-test's
    "part of the vendored tree dropped" fault mode pass whether or not the
    deliberate drop it exists to catch actually happened -- the phrase it
    looks for, "are missing here", is already in the report before the tamper
    runs. Shared by both the `--self-test` default and the production default
    below so there is one picker, not two that can drift apart.
    """
    parts = wheel.stem.split("-")
    if len(parts) < 5:
        return None
    name, version = parts[0], parts[1]
    candidates = sorted(wheel.parent.glob(f"{name}-{version}-*macosx*.whl"))
    return candidates[-1] if candidates else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--reference", type=Path, default=None,
                    help="a wheel whose file list this one must match, to catch "
                         "a cross wheel that quietly lost part of the tree "
                         "(default: the newest macosx wheel of the SAME name "
                         "and version beside it)")
    ap.add_argument("--self-test", action="store_true",
                    help="damage this wheel nine ways and check that each is "
                         "reported, with the right reason")
    args = ap.parse_args()

    if not args.wheel.exists():
        sys.exit(f"no such wheel: {args.wheel}")

    if args.self_test:
        reference = args.reference
        if reference is None:
            reference = _default_reference(args.wheel)
        sys.exit(self_test(args.wheel, reference))

    parts = args.wheel.stem.split("-")
    if len(parts) < 5:
        sys.exit(f"{args.wheel.name} is not a PEP 427 wheel filename")
    name, version, pytag, abitag, plat = parts[0], parts[1], *parts[-3:]
    exp = Expectation.parse(plat)
    problems: list[str] = []

    print(f"{args.wheel.name}")
    print(f"  distribution        {name} {version}")
    print(f"  python / abi        {pytag} / {abitag}")
    if abitag != "abi3":
        problems.append(
            f"abi tag is {abitag!r}, not 'abi3' -- the wheel then serves one "
            "CPython version instead of 3.13 and every later one")
    exp.check_tag(problems)

    with zipfile.ZipFile(args.wheel) as zf:
        names = zf.namelist()
        dist_info = f"{name}-{version}.dist-info"

        # 1. WHEEL's Tag: and the filename have to agree. Installers read the
        #    filename; `wheel unpack`, auditors and `twine` read WHEEL.
        wheel_meta = zf.read(f"{dist_info}/WHEEL").decode()
        declared = [line.split(": ", 1)[1] for line in wheel_meta.splitlines()
                    if line.startswith("Tag: ")]
        want_tag = f"{pytag}-{abitag}-{plat}"
        if declared != [want_tag]:
            problems.append(
                f"{dist_info}/WHEEL declares Tag: {declared}, filename says "
                f"{want_tag}")
        else:
            print(f"  WHEEL Tag:          {want_tag}")
        if "Root-Is-Purelib: false" not in wheel_meta:
            problems.append(
                "WHEEL says Root-Is-Purelib: true -- a wheel carrying an "
                "extension unpacks into purelib and the extension lands in the "
                "wrong directory")

        # 2. Every binary in the archive is for the tagged platform. Walking all
        #    of them rather than a named list: the point of the check is to
        #    catch a member nobody thought about, and a list can only contain
        #    the ones somebody did.
        binaries = []
        for member in names:
            if member.endswith("/"):
                continue
            head = zf.read(member)
            if not _is_binary(head):
                continue
            binaries.append(member)
            exp.check_binary(member, head, problems)
        print(f"  binaries            {len(binaries)}")
        for member in binaries:
            print(f"                      {member}  "
                  f"{describe(zf.read(member))}")
        if not binaries:
            problems.append(
                "no binary members at all -- this is the py3-none-any shell "
                "again, in a platform-tagged wrapper")

        # 3. The files `import torch` reaches for by name. Which files those
        #    are is a property of the platform, not a constant: Windows renames
        #    one and does not load another at all.
        ext = exp.extension_member
        if ext not in names:
            problems.append(f"no {ext}")
        stale = [n for n in names
                 if n.endswith("torch/_C.abi3.so") and n != ext]
        if stale:
            problems.append(
                f"{stale} beside {ext} -- the extension is under two names and "
                "the interpreter would find whichever its table lists first")
        deps_name = exp.global_deps_name()
        deps = f"torch/lib/{deps_name}" if deps_name else None
        if deps and deps not in names:
            problems.append(
                f"no {deps} -- `_load_global_deps()` computes exactly this name "
                "on this platform, and without it `import torch` needs "
                "TORCH_USE_RTLD_GLOBAL=1 (docs/VENDOR.md wall 1)")
        strays = [n for n in names
                  if n.startswith("torch/lib/libtorch_global_deps") and n != deps]
        if strays and deps:
            problems.append(f"{strays} beside {deps}; only one is looked for")
        elif strays:
            problems.append(
                f"{strays}, but `_load_global_deps()` returns before looking on "
                "this platform -- `_load_dll_libraries()` would LoadLibrary it "
                "for nothing")
        if exp.shm_manager_required and "torch/bin/torch_shm_manager" not in names:
            problems.append(
                "no torch/bin/torch_shm_manager -- `_manager_path()` checks it "
                "exists on every non-Windows platform (docs/VENDOR.md wall 4)")

        # 4. The archive agrees with itself.
        check_record(zf, dist_info, problems)

        # 5. The interpreter this is for would look for this filename, and --
        #    where the format makes it decidable -- would resolve what the
        #    members ask of it. Only wasm implements the second half; see
        #    `Expectation.check_linkage`.
        check_suffix_is_searched(exp, problems)
        exp.check_linkage(zf, names, problems)

        # 6. Nothing was lost relative to a wheel that is known to work. The
        #    cross path swaps two members and adds one; anything else differing
        #    means the tree behind it changed between builds.
        reference = args.reference
        if reference is None:
            reference = _default_reference(args.wheel)
        if reference is None:
            problems.append(
                "no reference wheel to compare the file list against; build the "
                "host wheel first or pass --reference")
        else:
            with zipfile.ZipFile(reference) as ref:
                ref_names = set(ref.namelist())
            here = set(names)
            # Two members are expected to differ and both are excluded on both
            # sides rather than allowed one-directionally, which would also hide
            # a loss. The global-deps filename differs by design (.dylib vs .so,
            # or absent entirely on Windows); the extension's *name* differs
            # only where the target interpreter searches for another one, and
            # that it is present under the right name was already checked
            # against `exp.extension_member` above -- so folding both spellings
            # to one here weakens nothing.
            renamed = {"torch/_C.abi3.so", exp.extension_member}

            def strip(s):
                return {"torch/_C" if n in renamed else n for n in s
                        if not n.startswith("torch/lib/libtorch_global_deps")
                        and not n.endswith("/RECORD")}
            lost = sorted(strip(ref_names) - strip(here))
            gained = sorted(strip(here) - strip(ref_names))
            if lost:
                problems.append(
                    f"{len(lost)} file(s) in {reference.name} are missing here, "
                    f"e.g. {lost[:5]}")
            if gained:
                problems.append(
                    f"{len(gained)} file(s) here are not in {reference.name}, "
                    f"e.g. {gained[:5]}")
            if not lost and not gained:
                print(f"  file list           identical to {reference.name} "
                      f"({len(strip(here)):,} entries)")

    print()
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"PASS -- {args.wheel.name} is tagged for a platform an installer "
          "will match, and holds binaries for it")
    print("       NOT established here: that it loads, imports, or computes. "
          "See docs/WHEEL.md §7.")


if __name__ == "__main__":
    main()
