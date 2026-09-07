# WASM §10 — the checker, then the target

docs/platform/WASM.md §9 built a wasm wheel by hand and deliberately landed no
`PyEmscriptenTarget`, because `verify_cross.py` could not check one. This round
builds the checker and then the target. Both landed.

## 10.1 What a wasm reader can and cannot answer

`verify_cross.py`'s 1043 lines rest on ELF `.dynsym`/`.dynamic`, Mach-O
`LC_BUILD_VERSION`/`LC_LOAD_DYLIB`, and PE's import/export directories. A
WebAssembly module has none of those. It has an **import section** (id 2) and an
**export section** (id 7), which name every symbol the host must supply and
every symbol offered, as strings, in the module. Linkage therefore gets *easier*
to read and platform gets *impossible*, and those two facts are reported
separately rather than netted off.

Answered from the artefact (`binfmt.wasm_info` / `wasm_imports` /
`wasm_import_records` / `wasm_exports`, `PyEmscriptenExpectation`):

| question | how |
|---|---|
| wasm32, not wasm64 | the memory type's limits flags. A side module *imports* its memory, so this is read out of the import section, not section 5 |
| is it loadable at all | side module (imports `env.memory` + `env.__indirect_function_table`) vs main-module link (defines and exports them). Emscripten's `dlopen` refuses the latter. This is the wasm spelling of PE's `dll`-not-`exe` bit. **Not** the presence of `dylink.0` — `pyodide.asm.wasm` is a main module and carries one too |
| `PyInit__C` is exported, and is a **function** | export section, with the kind kept. An export of that name with kind `global` is a linker accident that fails at import with nothing pointing at the module |
| does the interpreter resolve what the module asks of it | `check_linkage`, scoped to `Py*`/`_Py*` under import module `env`, resolved against `pyodide.asm.wasm`'s export section |
| nothing linked in that should not be | every wasm member is checked, not only the extension — `ctypes.CDLL` on the global-deps stub is also `dlopen` |

**Declined, loudly.** Each of these is a question another family *does* answer,
and the report prints a `!` line for it rather than substituting a nearby easier
check:

- **The platform.** A wasm module records no Emscripten version, no Pyodide ABI
  version, no minimum anything. Two modules built a major release apart are
  byte-indistinguishable in their headers. The `2026_0` in the tag is checked
  against `pyodide-lock.json` — a check on *this machine's distribution*, not on
  the wheel. Build against a different Pyodide and nothing in the bytes says so.
- **The abi3 binding.** `WindowsExpectation` proves it by which DLL each import
  is attributed to (`python3.dll`, not `python313.dll`). Emscripten attributes
  every undefined symbol to the single import module `env`, so that distinction
  has **no wasm spelling at all**.
- **A manylinux-style floor.** No counterpart, same as Mach-O and Android.
- **Whether every import resolves.** Only the `Py*` subset is decidable. Non-`Py`
  `env` imports come from Emscripten's JS library at load time or from sibling
  side modules, neither visible in any artefact here — shipped, working Pyodide
  wheels have hundreds of them, so a whole-`env` check would reject correct
  wheels. The narrow check is the true one; the breadth it lacks is printed.
- **`packaging` confirmation.** There is no `pyemscripten_platforms` generator to
  ask, unlike android and ios. Printed as unconfirmed, the same way manylinux and
  win are.

## 10.2 The rejection, demonstrated

A checker nobody has seen reject is not a checker. `_wasm_without_pyinit`
renames the export in place (`PyInit__C` → `PyInit__X`) rather than deleting it,
so no section size or LEB has to be re-encoded and the module stays well-formed
— a truncated module would be rejected by `_wasm_sections` returning `None`,
which is a *different* check firing and would make the fault mode a lie.

    FAIL: torch/_C.abi3.so exports no PyInit__C -- the import system `dlsym`s
    exactly that name and raises ImportError when it is absent. Its export
    section has 130 entries, including ['PyInit__X']

Exit 1, naming the symbol and the near-miss. Full wasm self-test: **12/12 fault
modes rejected**, including the wasm32↔wasm64 memory-flags flip
(`_wrong_wasm_memory_bits`, the *only* header field a wasm module has that a
platform claim can be checked against).

This is only meaningful because the base wheel **passes** first. The §9
hand-built wheel is missing the 116-file `torch-2.13.0.dist-info` tree, so it
fails for that reason alone — and every fault mode would then have "caught"
trivially. A completed copy of it was made for the self-test.

## 10.3 `PyEmscriptenTarget`, and the trap made structural

The tag is `pyemscripten_2026_0_wasm32`. `2026_0` is Pyodide's
`PYODIDE_ABI_VERSION`, which CPython's build never sees. Ask the interpreter the
way every other target does and it answers `emscripten-5.0.3-wasm32` — a tag
`packaging` accepts and **nothing on PyPI uses**. Wrong in the shape that works.

The refusal is **structural, not a comment**:

- `PyEmscriptenTarget.sysconfig()` **is overridden to `_fail`**, naming
  `pyodide-lock.json` as the source. Anyone copying `AndroidTarget` reaches for
  `self.sysconfig()` first and gets an exception instead of a plausible answer.
  Every other `sysconfig()` caller in `build.py` is family-specific and
  inherited by none of them, so nothing is collaterally broken.
- `PYODIDE_ROOT` is a **second** root variable, deliberately not a subdirectory
  of `TARGET_PYTHON_ROOT`. Everything under that root is a cross-compiled
  CPython; a Pyodide distribution is a different kind of thing.
- `platform_tag` reads `info.abi_version` and `info.arch` out of the lock file,
  and fails hard if it is absent rather than guessing.

`self_test_pyemscripten()` in `build.py` exercises all of it — 4/4:

    ok    PyEmscriptenTarget.sysconfig() refuses rather than answering
    ok    the tag follows pyodide-lock.json's info.abi_version
    ok    a missing pyodide-lock.json fails rather than guessing
    ok    check_image refuses a main-module link (real pyodide.asm.wasm)

Case 2 writes a lock file saying `1999_7` and watches the derived tag become
`pyemscripten_1999_7_wasm32`. That is the difference between a derivation and a
story about one. Case 4 uses Pyodide's real `pyodide.asm.wasm` rather than a
synthesised near-miss.

## 10.4 §9's two inversions, verified rather than trusted

- **`.abi3.so` is in Pyodide's `EXTENSION_SUFFIXES`** — confirmed from the bytes:
  `pyodide.asm.wasm` contains the string `.abi3.so`, and
  `check_suffix_is_searched` (unchanged, no wasm exception) reports
  `ext suffix  .abi3.so present in pyodide.asm.wasm`. `Target.extension_member`
  needs no wasm case.
- **The abi3 tag is in `sys_tags()`** — *not* re-verified here. It requires the
  target interpreter, which is not runnable on this machine; §9.3 read it off
  the real one. `setup.py`'s `py_limited_api` is unchanged and no wasm exception
  was added; the claim stands on §9's evidence, not on this round's.

## 10.5 Out of scope

`_multiprocessing` is the one wall a wheel cannot carry. §9 assigns it to the
vendoring owner. Nothing here ships a top-level `_multiprocessing`, which would
shadow it on the other six platforms.

## 10.6 The other six platforms are unchanged

Byte-identical output, HEAD vs this tree, over the five cross families:

    ios_12_0_arm64_iphoneos            IDENTICAL
    ios_14_0_arm64_iphonesimulator     IDENTICAL
    android_21_arm64_v8a               IDENTICAL
    manylinux_2_17_x86_64              IDENTICAL
    win_amd64                          IDENTICAL

and `--self-test` identical for ios and manylinux. `build.py --self-test` is
identical modulo the repo path and tempdir names, plus the new
`PYEMSCRIPTEN SELF-TEST` block, which is additive.
