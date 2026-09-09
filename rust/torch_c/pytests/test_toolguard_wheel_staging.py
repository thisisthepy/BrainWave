"""A failed cross build must not leave a wheel in `--outdir`.

`tools/wheel/build.py --target <cross>` builds an intermediate wheel with
`pip wheel`, then patches it for the target platform. `global_deps_stub`
(compiling the empty `libtorch_global_deps` stub for the *target*, via
`zig`/`emcc`/`$TARGET_CC`) runs *after* that intermediate wheel exists, so a
missing cross toolchain used to fail there and leave the untouched,
host-tagged intermediate (`...-macosx_11_0_universal2.whl`) sitting in
`--outdir` -- a publishable-looking artefact with the wrong platform tag, from
a build that never finished.

`build.py` now stages the whole pipeline (pip wheel -> repack -> verify) in a
`tempfile.TemporaryDirectory` and moves the result into `--outdir` only after
every step has succeeded. This file reproduces the trap directly: it points
`--target linux-x86_64` at a hand-built, minimally-valid ELF stand-in for the
cross artefact (so the early "no cross-built extension" / staleness / tag
checks all pass) with no `zig`/`$TARGET_CC` available, so the build reaches
`global_deps_stub` and fails there -- exactly the failure this trap needs.

Building the fake ELF rather than a real cross artefact: this machine has no
`zig` and building one for real is exactly what LINUX.md §9 documents as
needing a toolchain this environment does not have. The fake only has to
satisfy `elf_info`/`elf_dynamic` (`ELF 64-bit x86_64 dyn`, one `DT_NEEDED` on
`libc.so.6`, one `GLIBC_2.17` version requirement) -- the same shape
`self_test_linux`'s `_minimal_elf` helper builds elsewhere in this repository,
extended with a `.dynamic`/`.gnu.version_r` pair because `platform_tag` reads
those too.
"""

import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BUILD_PY = REPO_ROOT / "tools" / "wheel" / "build.py"
VENDORED_TORCH = REPO_ROOT / "torchnative" / "src" / "main" / "torch"
HOST_SHIM = VENDORED_TORCH / "_C.abi3.so"

GOOD_PYTHON = "/Volumes/macMini/caches/spike-venv/bin/python"


def _minimal_linux_elf() -> bytes:
    """A tiny but structurally valid `ELF 64-bit x86_64 dyn` needing
    `GLIBC_2.17` off `libc.so.6` -- enough for `elf_info`/`elf_dynamic`
    (see `tools/wheel/binfmt.py`) to read a real answer, without a real
    compiler anywhere in the loop."""
    endian = "<"
    dynstr = b"\0" + b"libc.so.6\0" + b"GLIBC_2.17\0"
    dyn_entries = struct.pack(endian + "qQ", 1, 1)      # DT_NEEDED -> "libc.so.6"
    dyn_entries += struct.pack(endian + "qQ", 0, 0)      # DT_NULL
    verneed = struct.pack(endian + "HHIII", 1, 1, 1, 16, 0)
    glibc_name_off = 1 + len("libc.so.6\0")
    vernaux = struct.pack(endian + "IHHII", 0, 0, 1, glibc_name_off, 0)
    verneed_data = verneed + vernaux
    shstrtab = (b"\0" + b".dynstr\0" + b".dynamic\0" + b".gnu.version_r\0"
                + b".shstrtab\0")

    ehdr_size = 64
    dynstr_off = ehdr_size
    dynamic_off = dynstr_off + len(dynstr)
    verneed_off = dynamic_off + len(dyn_entries)
    shstrtab_off = verneed_off + len(verneed_data)
    shoff = shstrtab_off + len(shstrtab)

    def name_off(name: bytes) -> int:
        return shstrtab.index(name + b"\0")

    def shdr(name: int, stype: int, off: int, size: int, link: int = 0,
             entsize: int = 0) -> bytes:
        return struct.pack(endian + "IIQQQQIIQQ", name, stype, 0, 0, off,
                           size, link, 0, 1, entsize)

    SHT_NULL, SHT_STRTAB, SHT_DYNAMIC = 0, 3, 6
    SHT_GNU_VERNEED = 0x6FFFFFFE

    sections = (
        shdr(0, SHT_NULL, 0, 0)
        + shdr(name_off(b".dynstr"), SHT_STRTAB, dynstr_off, len(dynstr))
        + shdr(name_off(b".dynamic"), SHT_DYNAMIC, dynamic_off,
               len(dyn_entries), link=1, entsize=16)
        + shdr(name_off(b".gnu.version_r"), SHT_GNU_VERNEED, verneed_off,
               len(verneed_data), link=1)
        + shdr(name_off(b".shstrtab"), SHT_STRTAB, shstrtab_off, len(shstrtab))
    )

    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = 2   # ELFCLASS64
    ehdr[5] = 1   # little-endian
    ehdr[6] = 1
    struct.pack_into(endian + "HH", ehdr, 16, 3, 0x3E)     # ET_DYN, EM_X86_64
    struct.pack_into(endian + "Q", ehdr, 0x28, shoff)
    struct.pack_into(endian + "HHH", ehdr, 0x3A, 64, 5, 4)  # shentsize,shnum,shstrndx

    return bytes(ehdr) + dynstr + dyn_entries + verneed_data + shstrtab + sections


def test_a_failed_cross_build_leaves_no_wheel_in_outdir():
    if not VENDORED_TORCH.is_dir() or not HOST_SHIM.exists():
        print("   (skipped: no vendored torch tree / host shim -- run "
              "vendor/vendor_torch.sh && vendor/install_shim.sh first)")
        return
    if shutil.which("zig") or os.environ.get("TARGET_CC"):
        print("   (skipped: a real cross toolchain is on PATH/env -- this "
              "case needs global_deps_stub to fail)")
        return

    # CARGO_TARGET_DIR is deliberately the *real* one: `check_host_shim`
    # looks for the host `lib_C.dylib`/`lib_C.so` there regardless of
    # `--target`, and that is where `vendor/install_shim.sh` actually put it.
    # Only the linux-x86_64 subdirectory is faked, and it is removed in
    # `finally` below.
    cargo_target_dir_env = os.environ.get(
        "CARGO_TARGET_DIR", str(REPO_ROOT / "rust" / "torch_c" / "target"))
    rel_dir = (Path(cargo_target_dir_env) / "x86_64-unknown-linux-gnu"
               / "release")
    if rel_dir.exists():
        # SKIP, not fail. This was an assertion, on the belief that the
        # directory "does not exist on this host" -- it does on any checkout
        # where `build.py --target linux-x86_64` has been run, which is every
        # checkout that has cut a release. Failing there turns a green gate red
        # for a reason that is about the machine and not about the code, and a
        # gate that goes red after a wheel build is a gate people stop reading.
        # Refusing to overwrite a real artefact is still right; announcing it as
        # a defect is not.
        print(f"   (skipped: {rel_dir} already exists -- refusing to fake an "
              "artefact over a real cross build. Remove it to run this case.)")
        return

    with tempfile.TemporaryDirectory(prefix="toolguard-wheelstage-") as tmpdir:
        tmp = Path(tmpdir)
        outdir = tmp / "dist"
        try:
            rel_dir.mkdir(parents=True)
            artefact = rel_dir / "lib_C.so"
            artefact.write_bytes(_minimal_linux_elf())
            # A dep-info naming a real, unmodified source file as the only
            # input, so `require_current`'s freshness check (artefact newer
            # than every recorded input) passes honestly rather than being
            # bypassed.
            src_file = REPO_ROOT / "rust" / "torch_c" / "src" / "lib.rs"
            (rel_dir / "lib_C.d").write_text(f"{artefact}: {src_file}\n")

            env = dict(os.environ)
            env.pop("TARGET_CC", None)
            env.pop("CC_x86_64_unknown_linux_gnu", None)
            # Strip any `zig` this machine might have, so `global_deps_stub`
            # is guaranteed to reach the "no C compiler that targets ..."
            # refusal -- the exact failure that used to leave a stray wheel
            # behind. (`ziglang` as a pip package is not searched here; a
            # machine with it installed but no `zig` binary would need that
            # covered too, but this one does not have it.)
            env["PATH"] = os.pathsep.join(
                p for p in env.get("PATH", "").split(os.pathsep)
                if not (Path(p) / "zig").exists()
            )

            proc = subprocess.run(
                [GOOD_PYTHON, str(BUILD_PY), "--target", "linux-x86_64",
                 "--outdir", str(outdir)],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
            out = proc.stdout + proc.stderr

            assert proc.returncode != 0, (
                "expected the build to fail (no C compiler for the target) "
                f"-- got exit 0:\n{out}"
            )
            assert "no C compiler that targets" in out, out

            stray = sorted(outdir.glob("*.whl")) if outdir.exists() else []
            assert not stray, (
                f"a failed build left {stray} in {outdir} -- the whole "
                "point of staging is that nothing reaches --outdir until "
                "every step, including global_deps_stub, has succeeded. "
                f"Full output:\n{out}"
            )
        finally:
            shutil.rmtree(rel_dir.parent, ignore_errors=True)
            # `pip wheel`'s `setup.py build_py` populates the real, shared
            # `build/` cache (see `BUILD_CACHE` in build.py -- it is not
            # `--outdir`-relative), so this test's own runs must not leave it
            # behind for the next invocation's `check_build_cache` to trip
            # on.
            shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
