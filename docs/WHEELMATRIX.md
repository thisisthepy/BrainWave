# Two architectures per platform

Until this round the published wheel matrix had **exactly one CPU architecture
on every non-Apple platform**: Android was `arm64-v8a`, Linux was `x86_64`,
Windows was `amd64`. Not by any decision recorded anywhere — each platform was
opened by whichever cross-build happened to work first, and the second
architecture was never asked about. That is the same shape as the gaps
`docs/AUDIT.md` catalogues: not a wrong claim, an unexamined one.

This round adds the other architecture on each, and the interesting part is
that **the three do not fail or succeed for the same reasons**. One of them is
now the best-verified target in the repository; one builds and cannot be run
anywhere this project can reach; one refuses, and the thing missing is not what
it looks like.

| target | tag | built | verified |
|---|---|:--:|---|
| `linux-aarch64` | `manylinux_2_17_aarch64` | ✅ | **executed** — installed by pip on real Linux aarch64 and computed, twice, at two different glibcs; a real model ran |
| `windows-arm64` | `win_arm64` | ✅ | symbols only — no ARM64 Windows exists in this project's reach |
| `android-x86_64` | `android_21_x86_64` *(derivable, unemitted)* | ❌ **refuses by name** | — |

The existing six are unchanged and were rebuilt to prove it (§5).

<!-- DOCWATCH: symbol-in-file tools/wheel/build.py EXPECTED_TARGET_KEYS present -->
<!-- DOCWATCH: symbol-in-file tools/wheel/build.py check_registry present -->

---

## 1. The registry now holds nine targets, and says so twice

`tools/wheel/build.py`'s `TARGETS` is a dict comprehension keyed on `t.key`:

```python
TARGETS: dict[str, Target] = {t.key: t for t in ( ... )}
```

Every one of the three new entries was written by copying the existing one for
that platform and editing its arguments. **A copy whose `key` argument is not
edited does not raise anything.** The two entries collide, the later wins, and
the registry silently holds eight targets where nine were written — `--target`
offers eight choices, and the survivor is a perfectly good target that builds a
perfectly good wheel, just not the one that went missing.

So the keys are written down a second time, in `EXPECTED_TARGET_KEYS`, and
`check_registry()` runs first in `main()`. Listed rather than counted, so the
failure names the key instead of reporting `8 != 9`:

    android-arm64-v8a  android-x86_64
    ios-arm64          ios-arm64-sim
    linux-aarch64      linux-x86_64
    wasm32-emscripten
    windows-arm64      windows-x86_64

`rust/torch_c/pytests/test_wheelmatrix.py::test_the_registry_cannot_silently_lose_an_entry`
drives the collision rather than arguing it.

---

## 2. Each tag on its own terms

`build.py`'s header already distinguishes tags that are **derived** from tags
that are **names**. Adding a second architecture to each family is where that
distinction stops being a remark and starts deciding things.

### 2.1 `android_21_x86_64` — PEP 738, derived, and confirmable

PEP 738's `android_<api-level>_<abi>` is a floor: `packaging.tags.
android_platforms` yields every level from the device's own down to 16. The
level comes from `ANDROID_API_LEVEL` in the target CPython's
`_sysconfigdata_*.py`, and the ABI from the architecture half of its
`MULTIARCH`, through the NDK's own name table (`arm64-v8a`, `x86_64`, …).

`android_platforms` is the one generator in `packaging` that takes its inputs as
arguments instead of reading the running system, so a cross build can ask it
directly, and does:

```
>>> "android_21_x86_64" in packaging.tags.android_platforms(api_level=21, abi="x86_64")
True
```

The derivation is therefore complete and checked; what is missing is the
distribution to derive it *from* (§3).

`x86_64` is also the one ABI where CPython's `MULTIARCH` architecture and the
NDK's ABI name are the same string, which makes it the entry easiest to leave
out of the table and never notice — so the test asserts both spellings.

### 2.2 `manylinux_2_17_aarch64` — PEP 600, derived, and **not** by analogy

The floor is read off the artefact, not the interpreter: glibc compatibility is
a property of the compiled file (`.gnu.version_r`), which is where `auditwheel`
reads it. That part carries over from x86-64 unchanged.

Two things do **not** carry over, and both would have shipped a wheel nobody can
install.

**(a) The lowest legal floor is not 2.5 on aarch64.** `_confirm_pep600_spelling`
checked `(major, minor) >= (2, 5)` — PEP 600's grammatical floor, manylinux1 —
which is correct for x86-64 and i686 and for no other architecture.
`packaging._manylinux.platform_tags`, the code pip runs, says so in as many
words:

```python
# Oldest glibc to be supported regardless of architecture is (2, 17).
too_old_glibc2 = _GLibCVersion(2, 16)
if set(archs) & {"x86_64", "i686"}:
    # On x86/i686 also oldest glibc to be supported is (2, 5).
    too_old_glibc2 = _GLibCVersion(2, 4)
```

The reason is historical and not arbitrary: manylinux1 (PEP 513) and
manylinux2010 (PEP 571) were x86-only. aarch64 enters the scheme at PEP 599,
manylinux2014, glibc 2.17. So `manylinux_2_12_aarch64` is well-formed, is above
manylinux1's 2.5, and **matches no installer on any machine** — pip's generator
never emits it. A grammar check passes it; a wheel carrying it installs nowhere
and looks fine on the machine that built it.

`_MANYLINUX_ARCH_FLOOR` now holds that per-architecture floor, and the refusal
gives a different *reason* per architecture rather than one sentence with a
number substituted in, because they are two different facts.

**(b) The dynamic loader is per-architecture and is not in PEP 599's table.**
`POLICY_LIBRARIES` used to hold `ld-linux-x86-64.so.2` alongside the nineteen
libraries PEP 599 actually lists. It is not one of them — the loader is never a
`DT_NEEDED` in the usual sense, though it appears as one on x86-64. Left in that
frozenset it would have been "allowed" for the aarch64 wheel too, so an aarch64
image naming the x86-64 loader would have passed. It now lives in a per-arch
`LOADER` map and is unioned in for the target's own architecture only.

(As measured, the aarch64 artefact names no loader at all: its `DT_NEEDED` is
`['libm.so.6', 'libc.so.6', 'libpthread.so.0', 'libdl.so.2']`. The x86-64 one
does name it. Same crate, same flags, different linker convention.)

**(c) The spelling is now confirmed against pip's own code, which it never
was.** Every Linux build in this repository printed

    ! packaging 26.3 has no manylinux_platforms -- tag spelling unchecked

and moved on, because `packaging.tags` genuinely has no `manylinux_platforms`.
But `packaging._manylinux.platform_tags` exists and is what pip runs; the only
reason it cannot answer here is that it reads the **host** glibc, and this host
has none. That is exactly the half a cross build is entitled to substitute: the
question is not "does this machine match the tag" but "is this a name pip would
generate for a machine that does". `_confirm_manylinux_with_packaging`
substitutes the two host probes — the ABI check and the glibc version, the
latter with the tag's own floor, which is the narrowest form of the question —
and runs the generator unmodified. Everything deciding the *name* stays
packaging's code:

```
tag manylinux_2_17_aarch64 yielded by packaging._manylinux.platform_tags(['aarch64'])
    at glibc 2.17 -- 2 names in that list, of which the legacy aliases are
    ['manylinux2014_aarch64']
tag manylinux_2_17_x86_64  yielded by packaging._manylinux.platform_tags(['x86_64'])
    at glibc 2.17 -- 16 names in that list, of which the legacy aliases are
    ['manylinux2014_x86_64', 'manylinux2010_x86_64', 'manylinux1_x86_64']
```

**2 names against 16, at the same glibc, is the per-architecture floor made
visible** — and it is packaging saying it, not this file. That is what stops
`_MANYLINUX_ARCH_FLOOR` above from being self-confirming.

`tools/wheel/verify_cross.py` now runs the same confirmation, so its manylinux
branch is no longer a step weaker than its android and ios branches.

<!-- DOCWATCH: symbol-in-file tools/wheel/build.py _confirm_manylinux_with_packaging present -->
<!-- DOCWATCH: symbol-in-file tools/wheel/build.py _MANYLINUX_ARCH_FLOOR present -->

### 2.3 `win_arm64` — a NAME, and looked up rather than assumed

Windows wheel tags carry no version at all. There is no floor to compute, which
reads like "nothing to get wrong" and is the opposite: when a tag is a name, the
only way to be wrong is to *invent* it, and `win_aarch64` and `win_arm64ec` are
exactly as plausible spellings of "64-bit ARM Windows" as `win_arm64` to
everybody except pip. Reaching `win_arm64` by analogy with `win_amd64` is a
guess that happens to be right, and a guess that happens to be right is
indistinguishable from one that does not until somebody tries to install the
wheel.

So it is read out of the target distribution, and the chain is short enough that
every link is checkable here:

1. **`packaging` has no Windows tag code at all.** `platform_tags()` falls
   through to `_generic_platforms()`, whose entire body is
   `yield _normalize_string(sysconfig.get_platform())`. This is also why
   `_confirm_with_packaging(tag, "windows")` has always printed a skip — there
   is no `windows_platforms` to call, and there never was.
2. **`sysconfig.get_platform()` on `os.name == "nt"` is four lines**, and the
   target distribution ships them in its own `Lib/sysconfig/__init__.py`:

   ```python
   if os.name == 'nt':
       if 'amd64' in sys.version.lower():  return 'win-amd64'
       if '(arm)'  in sys.version.lower():  return 'win-arm32'
       if '(arm64)' in sys.version.lower(): return 'win-arm64'
       return sys.platform
   ```

   **The order is load-bearing**: `amd64` wins over everything, so which branch
   a distribution takes is decided entirely by which substring its own
   `sys.version` contains.
3. **`sys.version` embeds `Py_GetCompiler()`**, which on MSVC is the literal
   `[MSC v.<n> 64 bit (<arch>)]` compiled into `python3<minor>.dll`. Readable
   without running anything:

       target-python/x86_64-pc-windows-msvc/python313.dll   [MSC v.1944 64 bit (AMD64)]
       target-python/aarch64-pc-windows-msvc/python313.dll  [MSC v.1944 64 bit (ARM64)]

`WindowsTarget._platform_name()` runs CPython's own branch order over that
string, and `_confirm_windows_normalisation()` applies packaging's
`_normalize_string` to the result — so the hyphen-to-underscore step is
packaging's function rather than a `.replace()` here. The build prints the whole
chain:

```
tag is a NAME, derived not assumed: python313.dll says [MSC v.1944 64 bit (ARM64)],
    whose '(arm64)' is the branch `sysconfig.get_platform()` takes on this
    distribution, so it answers 'win-arm64'
packaging.tags._normalize_string('win-arm64') -> win_arm64
```

The registry still records the expected name per target, and `platform_tag`
refuses if the derivation and the registry disagree — that means the tree on
disk and the target asked for have come apart, and neither is authoritative
enough to override the other.

**One more check exists because the two distributions differ in no filename.**
`_check_distribution` was satisfied by either of them: both have
`libs/python3.lib`, `python3.dll`, `python313.dll`. So `--target windows-arm64`
pointed at the x86-64 tree would derive `win_amd64` from it and ship an ARM64
DLL under it — and `check_image` would *not* catch that, because it checks the
artefact against the target and the artefact is correct. `python313.dll`'s PE
machine is now checked too. `AndroidTarget._api_and_abi` has the identical trap
and the identical check, on `MULTIARCH`.

<!-- DOCWATCH: symbol-in-file tools/wheel/build.py _confirm_windows_normalisation present -->

---

## 3. What each one actually did

### 3.1 `linux-aarch64` — built, installed by pip, and computed

Toolchain: nothing new. `cargo-zigbuild` and the `ziglang` wheel
(docs/LINUX.md §9.1) already on this machine, `rustup target add
aarch64-unknown-linux-gnu`, and the target CPython downloaded from
python-build-standalone's `20260825` release — the same release
docs/TARGET_PYTHON.md §3 and §4 pin for the other two:

    cpython-3.13.15+20260825-aarch64-unknown-linux-gnu-install_only.tar.gz
    90,153,645 bytes, kept in target-python/_download/ per §"Rule going forward"

Build, and the tag falling out of the artefact rather than out of the command:

```
target linux-aarch64: lib_C.so (6,558,912 B)  ELF 64-bit little-endian aarch64 dyn
  tag floor from the artefact's .gnu.version_r (glibc 2.17), not from CPython
  DT_NEEDED within the PEP 599 policy list:
      ['libm.so.6', 'libc.so.6', 'libpthread.so.0', 'libdl.so.2']
  tag manylinux_2_17_aarch64 is PEP 600-shaped (glibc 2.17, aarch64; floor for aarch64 is 2.17)
  tag manylinux_2_17_aarch64 yielded by packaging._manylinux.platform_tags(['aarch64'])
  + torch/lib/libtorch_global_deps.so (2,928 B)  ELF 64-bit little-endian aarch64 dyn

dist/torchnative-0.0.13a0-cp313-abi3-manylinux_2_17_aarch64.whl
  2,696 entries, 14.6 MB compressed, 60.3 MB installed
```

`verify_cross.py` and `verify_linux.py` both PASS, the latter resolving 124
unversioned symbols against the **aarch64** libpython (§4).

**And then it ran, which is the part that is not an argument.** Docker on this
machine runs an aarch64 Linux VM, so a `linux/arm64` container is native — not
QEMU — and both a container and the wheel are the real thing.

**Run A, at exactly the glibc the tag names.** `quay.io/pypa/manylinux2014_aarch64`
is CentOS 7 AltArch, `ldd (GNU libc) 2.17`, `uname -m` = `aarch64`. The wheel was
installed with `--no-deps` and `tools/ci/verify_published.py` — the script the
Linux and Windows CI legs run — was executed against it:

```
== machine: aarch64  glibc: ldd (GNU libc) 2.17
Linux-7.0.12-linuxkit-aarch64-with-glibc2.17
torch          : 2.13.0
PASS  this is the shim, not upstream     True
aten ops       : 302
PASS  mm sum                             24.0
PASS  nn.Linear sum                      12.0
PASS  sub(int64,float16) value           [2047.0]
… 31 checks …
PASS  lstm returned h_n and c_n          ((1, 1, 4), (1, 1, 4))
RESULT: ALL PASS
```

Every expected value in that script is hardcoded from a macOS arm64 run of the
same source, so a disagreement localises to the platform. **The tag's promise
and the run are the same number**: the wheel says it needs glibc 2.17 and it was
loaded by glibc 2.17.

**Run B, on a modern aarch64 Linux, installed by pip's own matcher.** On
`python:3.13-slim` (Debian, glibc 2.4x) the wheel was installed as

    pip install --no-deps --no-index --find-links dist/ torchnative

— a *bare distribution name*, so pip had to match the `manylinux_2_17_aarch64`
tag against the machine itself to find any candidate at all. That is the tag
being accepted by the installer rather than by `packaging` on this Mac. The same
31 checks passed, and then the model leg CI runs:

```
generated : 'On-device inference is a very powerful technique for learning from
             data. It is a powerful technique for learning from data because it
             is a very fast'
PASS  generated text matches macOS arm64 True
PASS  QuantizedLinear count              210
PASS  Linear left dense (lm_head)        1
RESULT: ALL PASS
```

SmolLM2-135M through real `transformers` 5.x, text character-identical to macOS
arm64, and the `q8_0` module-replacement quantiser landing 210 leaves.

The 4400 ms / 5.5 tok/s in that log is **not a performance measurement** and is
not carried anywhere: it is a container on a VM sharing a laptop with a compile.

### 3.2 `windows-arm64` — built, symbols resolve, and that is the whole claim

Toolchain: nothing new either. `cargo-xwin` and the four MSVC tool shims
(docs/WINDOWS.md §3.2) already exist; `rustup target add aarch64-pc-windows-msvc`
and the PBS `20260825` Windows ARM64 distribution
(`cpython-3.13.15+20260825-aarch64-pc-windows-msvc-install_only.tar.gz`,
43,939,670 bytes, kept in `_download/`).

```
target windows-arm64: _C.dll (7,282,688 B)  PE32+ aarch64 dll
  [tag derivation as §2.3]
  imports 125 names from python3.dll (the abi3 forwarder)
  member: torch/_C.abi3.so -> torch/_C.pyd

dist/torchnative-0.0.13a0-cp313-abi3-win_arm64.whl
  2,695 entries, 14.7 MB compressed, 61.0 MB installed
```

`verify_windows.py` PASSes with all 241 imports attributed to a named DLL by the
import table: 125 resolved against the **ARM64** `python3.dll`, 8 against the
ARM64 `vcruntime140.dll`, 108 attributed to Windows components that do not exist
here.

**Nothing has executed it and nothing here can.** There is no ARM64 Windows on
this machine, Docker's VM is Linux, and the existing CI job runs
`windows-latest`, which is x86-64. The honest ceiling is the same rung
`win_amd64` sat on before its CI leg existed, and the README column says exactly
that. GitHub's `windows-11-arm` runner images would close it the same way
`ubuntu-latest` closed Linux x86-64 — that is a change to
`.github/workflows/verify-published-wheel.yml` and a published `win_arm64`
wheel to install, neither of which is this round's to make.

### 3.3 `android-x86_64` — refuses, and the missing piece is not the toolchain

This one is listed in `TARGETS` and refuses by name, the way `--target
linux-x86_64` did before docs/LINUX.md §9 made `cargo-zigbuild` work.

**The toolchain is present and is not the problem**, which is worth stating
plainly because it is what the refusal would normally mean:

| piece | state |
|---|---|
| rust target `x86_64-linux-android` | installed |
| NDK `x86_64-linux-android21-clang` | present, and **runs** — the NDK's only prebuilt directory is `darwin-x86_64`, and Rosetta executes it (`Target: x86_64-unknown-linux-android21`) |
| `cargo-ndk` | on `PATH` |

What is missing is the **target CPython**, and every tag in `build.py` is derived
from one.

* **It is not downloadable.** python-build-standalone's `20260825` release —
  the source of the four fetchable distributions in docs/TARGET_PYTHON.md —
  publishes 871 assets across 26 triples, and not one of them is Android
  (checked against the release API, not assumed). CPython publishes no Android
  binaries at all.
* **A local cross-build is the only route, and docs/TARGET_PYTHON.md §5 records
  what that bought last time.** The existing `aarch64-linux-android` tree is a
  local build from a `heads/3.13-dirty` working tree on a machine that is not
  this one; it is the one distribution of the five whose *source* cannot be
  reconstructed, and §6 lists it first among what is still unrecoverable.
  Adding a second unreproducible Android CPython is not a neutral cost.
* **The result could not be checked.** This is the decisive one and it is a
  measurement, not a judgement. `verify_android.py` is the only script in this
  repository that *executes* a cross wheel on its target, and this machine's
  emulator cannot run an x86-64 guest:

      ~/Library/Android/sdk/emulator/qemu/     ->  darwin-aarch64        (only)
      ~/Library/Android/sdk/system-images/*/*/ ->  arm64-v8a, arm64-v8a  (only)

  Google ships no x86-64 emulation backend for Apple Silicon, so this is a
  property of the host architecture rather than of what happens to be installed.
  An `android_21_x86_64` wheel built here could be checked exactly as far as
  `verify_cross.py` reaches — symbols — and no further.

**What would unblock it**, so this is falsifiable rather than a closed door:
either an x86-64 Android CPython with recorded provenance (a clean CPython
`Android/` cross-build at a named commit, kept with its download and a
docs/TARGET_PYTHON.md section, per that file's own rule), or an x86-64 host —
CI or otherwise — that can boot an x86-64 emulator to verify the result. The
first alone yields a wheel verified only by symbols; the pair yields the same
claim `android-arm64-v8a` already has.

The refusal is checked in `main()` **before** the artefact, so `--target
android-x86_64` answers with this reason instead of "no cross-built extension
at …, run `scripts/device_android.sh build`" — advice that cannot succeed.

---

## 4. The verifiers were the silent-failure risk, not the builder

Three scripts selected a target CPython by hardcoded triple:
`verify_cross.py` (all three families), `verify_linux.py`, `verify_windows.py`.
That was correct while there was one distribution per platform.

The failure mode with two is not an error. **`libpython3.13.so` exports the same
names on aarch64 as on x86-64, and `python3.dll` the same on ARM64 as on AMD64 —
the latter by construction, since it is the stable-ABI forwarder.** So resolving
an aarch64 wheel's undefined symbols against the x86-64 interpreter *succeeds*
and prints `0 unresolved`. An export table is a list of names with no machine in
it, and nothing downstream would have caught it: the check does not fail, it
stops being a check.

All three now select by the wheel's own tag, and refuse rather than fall back
when there is no distribution for that architecture — because falling back is
the thing that passes.

    verify_linux.py:   tag arch: aarch64 -> target-python/aarch64-unknown-linux-gnu
    verify_windows.py: tag: win_arm64    -> target-python/aarch64-pc-windows-msvc

`verify_cross.py` reads the roots out of `build.TARGETS` directly, so a target
added to the registry cannot be verified against a distribution nobody chose.

**One regression this caused, and how it surfaced.** `verify_cross.py` reads
`POLICY_LIBRARIES` *from the builder* rather than copying it, "so the two cannot
drift into disagreeing about what manylinux promises". When the loader moved out
of that frozenset (§2.2b), `verify_cross.py` immediately started refusing the
**x86-64** wheel it had passed for weeks, naming `ld-linux-x86-64.so.2` as an
off-policy dependency. That is the coupling working: a copy would have diverged
in silence. Fixed by unioning the loader in per architecture there too.

---

## 5. The six existing targets are unchanged, and that was measured

Parameterising three classes is a change to the existing targets whether or not
it was meant to be, so both x86-64 cross artefacts were rebuilt from scratch and
their wheels re-derived:

```
target linux-x86_64:   lib_C.so 7,796,224 B  -> manylinux_2_17_x86_64
   DT_NEEDED ['libm.so.6', 'libc.so.6', 'ld-linux-x86-64.so.2',
              'libpthread.so.0', 'libdl.so.2']
target windows-x86_64: _C.dll   8,668,672 B  -> win_amd64
   python313.dll says [MSC v.1944 64 bit (AMD64)] -> 'win-amd64' -> win_amd64
```

Same tags as before, now reached by derivation rather than by literal in the
Windows case. `verify_cross.py`, `verify_linux.py` and `verify_windows.py` PASS
on all four cross wheels, and both verifier self-tests pass (6/6 and 5/5).
`build.py --self-test` is 17/17 on the Linux derivation, up from 11/11.

---

## 6. What is claimed, and what is not

| | built | tag confirmed against pip's code | symbols | **executed** |
|---|:--:|:--:|:--:|---|
| `linux-x86_64` | ✅ | ✅ `packaging._manylinux` | ✅ | ✅ CI, `ubuntu-latest` (run 34038982934) |
| `linux-aarch64` | ✅ | ✅ `packaging._manylinux` | ✅ | ✅ **here**, glibc 2.17 and modern, + a real model |
| `windows-x86_64` | ✅ | — *(no generator exists)* | ✅ | ✅ CI, `windows-latest` (run 34038982934) |
| `windows-arm64` | ✅ | — *(no generator exists)* | ✅ | ❌ no ARM64 Windows in reach |
| `android-arm64-v8a` | ✅ | ✅ `packaging.tags` | ✅ | ✅ device/emulator, `verify_android.py` |
| `android-x86_64` | ❌ refuses | ✅ *(the derivation works; there is nothing to derive from)* | — | ❌ no x86-64 Android emulator on Apple Silicon |

"Executed" means a `torch` from that wheel produced a number, on that
architecture. It is the only row where ✅ is not about a file.

Two things this round deliberately did not do:

* **Publish anything.** The wheels are in `dist/` and nothing was uploaded; the
  README's "on PyPI" row is untouched and still describes `0.0.12a0`.
* **Time anything.** The Linux aarch64 runs are correctness only. Docker's
  aarch64 VM is native and would not be dishonest to time, but it shares a
  laptop with the build that produced the wheel, and `linux/amd64` under QEMU —
  the other container this machine can run — must never be timed at all.
