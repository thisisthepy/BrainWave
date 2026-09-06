# The five CPython distributions in `target-python/`

`/Volumes/macMini/caches/target-python/` holds the CPython builds that this
repository's cross-platform verification reads. Nothing in the repository said
where any of them came from; `docs/IOS_CI.md` §"So the blocker is provenance,
not configuration" records the cost of that — this machine is the only witness
for iOS at any level. This document is the answer to that: what each one is,
by evidence read out of the artefacts, and whether it can be fetched again.

Method, so a later reader can redo it: `lib/python3.13/_sysconfigdata_*.py`
carries `CONFIG_ARGS`, `abs_srcdir` and the build host's paths verbatim;
`Py_GetBuildInfo`'s branch/commit/date triple is a set of adjacent strings in
the interpreter binary (`strings -a … | grep -B4 'heads/'`); `include/…/
patchlevel.h` carries `PY_VERSION`; `vtool -show-build` gives the Mach-O
platform and SDK. Every identification below is from those, not from which
upstream project seemed likely.

Summary:

| directory | what it is | fetchable again |
|---|---|---|
| `arm64-iphonesimulator` | local CPython `iOS/` build, 3.13 branch @ `d894d467a61` | **rebuildable, not downloadable** |
| `arm64-iphoneos` | local CPython `iOS/` build, same commit | **rebuildable, not downloadable** |
| `aarch64-linux-android` | local CPython `Android/` cross-build, 3.13 @ `b4c504d76ff`-**dirty** | **not reproducible** — see §5 |
| `x86_64-unknown-linux-gnu` | python-build-standalone `20260825` | yes, checksum matched |
| `x86_64-pc-windows-msvc` | python-build-standalone `20260825` | yes, checksum matched |

---

## 1. `arm64-iphonesimulator` — the one blocking iOS CI

**What it is.** Not `python-build-standalone`, not BeeWare's
`Python-Apple-support`, and not any published artefact. It is a **local build
of CPython's own in-tree iOS support**, made on a Mac at
`/Users/ibrew/Desktop/cypthon/armsim64` on **18 Oct 2024, 02:15**.

Evidence, in order of how conclusive it is:

- `lib/python3.13/_sysconfigdata__ios_arm64-iphonesimulator.py`:

      abs_srcdir  = /Users/ibrew/Desktop/cypthon/armsim64
      prefix      = iOS/Frameworks/arm64-iphonesimulator
      CONFIG_ARGS = … '--host=arm64-apple-ios12.0-simulator'
                    '--build=arm64-apple-darwin' '--enable-framework'
                    '--with-build-python=…/cross-build/macOS/bin/python3.13'
                    '--with-openssl=…/cross-build/iphonesimulator.arm64/openssl'
                    LIBLZMA/BZIP2/LIBFFI flags pointing at
                    `cross-build/iphonesimulator.arm64/{xz,bzip2,libffi}`

  `prefix = iOS/Frameworks/arm64-iphonesimulator` is CPython's `iOS/Makefile`
  layout, and `cross-build/iphonesimulator.arm64/…` is what CPython's
  `iOS/README.md` walkthrough produces. `Python-Apple-support` would show
  `merge/`/`support/` paths and ship a `Python.xcframework`; PBS would show
  `/build/Python-3.13.x` and `/tools/deps`. Neither appears.
- `bin/` contains CPython's `iOS/Resources/bin/` compiler shims verbatim —
  `arm64-apple-ios-simulator-clang` is a two-line
  `xcrun --sdk iphonesimulator${IOS_SDK_VERSION} clang -target arm64-apple-ios-simulator $@`.
  These exist in no redistributed distribution.
- `include/python3.13/patchlevel.h`: `PY_VERSION "3.13.0+"` — the trailing `+`
  means a git checkout past the 3.13.0 tag, not a release tarball.
- Build-info strings in `Python.framework/Python`:
  `Oct 18 2024` / `02:15:03` / `d894d467a61` / `heads/3.13`, and
  `[Clang 16.0.0 (clang-1600.0.26.3)]` (Xcode 16.0).
- `vtool -show-build`: `platform IOSSIMULATOR`, `minos 14.0`, `sdk 18.0`.
  (`CFLAGS` says `-mios-version-min=12.0` and `IPHONEOS_DEPLOYMENT_TARGET` in
  sysconfig is `12.0`; the linker floors arm64-simulator at 14.0. Both numbers
  are true of different things, and `tools/wheel/build.py` reads the sysconfig
  one for the wheel tag.)
- Single-slice `arm64` Mach-O, `Python.framework/Python`
  sha256 `52a87e2c8575312f2ce048f79d4e83b329748e34693c6988f6ffe1a849ad5761`.

**Where to get it again.** There is **no URL**. The source commit is fully
identified, though, so the artefact is *rebuildable*:

    python/cpython @ d894d467a615ada8150e9e6987f1969f0a5e2cd9   (branch 3.13)
    committed 2024-10-17T17:04:02Z
    "[3.13] gh-113570: reprlib.repr does not use builtin __repr__ …"

(confirmed to exist upstream via the GitHub API; the 11-char prefix in the
binary resolves to that full sha.) Rebuilding is CPython's documented
`iOS/README.md` procedure: build a host `python3.13`, build the third-party
deps into `cross-build/iphonesimulator.arm64/{xz,bzip2,libffi,openssl}`, then
`configure --host=arm64-apple-ios12.0-simulator --build=arm64-apple-darwin
--enable-framework` with the `*_CFLAGS`/`*_LIBS` above.

**A rebuild will not be byte-identical.** The sha256 above pins *this* file,
not the recipe: the OpenSSL/xz/bzip2/libffi versions are not recorded anywhere
(only their install prefixes are), and the Xcode is pinned only to
`clang-1600.0.26.3` / SDK 18.0. What is reproducible is a functionally
equivalent distribution, which is what the verification needs.

**What reads it.** `tools/wheel/verify_ios_sim.py` — and it is the most
layout-sensitive consumer in the repository. It `copytree`s the whole
directory to a scratch prefix and then requires, by exact path:

    prefix/bin/arm64-apple-ios-simulator-clang       the compiler shim, used to
                                                     build the launcher
    prefix/include/python3.13/Python.h               `-I`
    prefix/Python.framework                          `-F` and `-Wl,-rpath`
    prefix/lib/python3.13/site-packages              where the wheel is staged
    SIMCTL_CHILD_PYTHONHOME=prefix                   stdlib at prefix/lib/python3.13

A same-project rebuild is fine. A *different* project's distribution would
break it: `Python-Apple-support` ships an `.xcframework` (no top-level
`Python.framework`, no `bin/` shims), so both the `-F` and the compiler
invocation fail; `python-build-standalone` publishes no iOS target at all.
`tools/wheel/verify_cross.py` additionally globs
`<root>/<subdir>/Python.framework/Python` for the framework identity check.

**What the iOS CI job would then need.** With the provenance closed, the
remaining gap is not knowledge but a runner. A job would have to:

1. Build the distribution from the recipe above — a macOS runner with Xcode
   16-or-later and roughly 30–60 min for host-python + four deps + two CPython
   configurations. This is the expensive part, and it is a *cache*
   candidate, keyed on the CPython commit plus the Xcode version.
2. Boot a simulator (`xcrun simctl`) — `verify_ios_sim.py` drives the runtime
   via `simctl spawn` and `SIMCTL_CHILD_*`, so a hosted macOS runner suffices;
   no device, no signing.
3. Point `TARGET_PYTHON_IOS_SIM` at the built prefix. The script already reads
   that variable, so nothing in the harness changes.

Step 1 is the whole cost, and it is what "this machine is the only witness"
was actually purchasing. Note that the *device* half (`verify_ios_device.py`)
is symbol resolution only and needs `arm64-iphoneos` on disk but no hardware,
so it can ride the same cached build.

---

## 2. `arm64-iphoneos`

**What it is.** The same build, same day, one configuration over: the device
slice. `/Users/ibrew/Desktop/cypthon/arm64`, built **18 Oct 2024, 02:14** —
one minute before the simulator one, from the same tree.

    _sysconfigdata__ios_arm64-iphoneos.py
      abs_srcdir  = /Users/ibrew/Desktop/cypthon/arm64
      prefix      = iOS/Frameworks/arm64-iphoneos
      CONFIG_ARGS = … '--host=arm64-apple-ios12.0' '--build=arm64-apple-darwin'
                    '--enable-framework'
      SOABI       = cpython-313-iphoneos
    build info    Oct 18 2024 / 02:14:24 / d894d467a61 / heads/3.13
    vtool         platform IOS, minos 12.0, sdk 18.0
    Python.framework/Python  sha256
      d861e0db8b2a3a88a47382d6e6ed235e9ebb57e67c20f5ffb71dc7950ca35dfe

Same source commit, same "rebuildable, not downloadable" verdict as §1.

**It was hand-carried.** `target-python/__MACOSX/arm64-iphoneos/…` sits beside
it — the resource-fork sidecar macOS's Archive Utility writes when a zip made
on one Mac is expanded on another. So this directory arrived as a zip from
elsewhere, which is consistent with it being a personal build and inconsistent
with any download.

**Two directories in it are not part of the original build** and must not be
mistaken for upstream layout:

    linkstub.disabled/libpython3.13.dylib -> …/Python.framework/Python

created 23 Aug 2026 by this repository's own cross-build work (see
`docs/ABI3.md` §"조사 중(15:55)" and `docs/RUST_CROSSBUILD.md`). It is a
symlink giving `-lpython3.13` something to resolve against; `docs/ABI3.md`
line 261 records the aarch64-apple-ios abi3 build succeeding with
`linkstub.disabled` passed as `-L`. A fresh rebuild will not have it.

**What reads it.** `tools/wheel/verify_ios_device.py` (via
`TORCHNATIVE_TARGET_PYTHON`, then `/arm64-iphoneos`), which resolves
`@rpath/Python.framework/Python` for the 118 Python symbols an on-device
`_C.abi3.so` binds, and reports "no device Python.framework on disk" as a
*blind* verdict rather than a pass when the directory is absent — so losing
this directory degrades the check honestly instead of silently.
`tools/wheel/verify_cross.py` uses the same framework path.

---

## 3. `x86_64-unknown-linux-gnu` — python-build-standalone 20260825

**What it is.** `astral-sh/python-build-standalone`, CPython **3.13.15**,
release tag **20260825**, `install_only` variant. Unambiguous from
`_sysconfigdata__linux_x86_64-linux-gnu.py`:

    abs_srcdir  = /build/Python-3.13.15
    prefix      = /install
    CONFIG_ARGS = … '--with-openssl=/tools/deps' '--with-system-libmpdec'
                  '--enable-optimizations' '--with-lto' '--enable-bolt'
                  '--with-mimalloc' '--enable-experimental-jit=yes-off'
                  'MODULE_BUILDTYPE=static'
                  '--enable-static-libpython-for-interpreter'
                  'BOLT_APPLY_FLAGS=…' 'PROFILE_TASK=-m test --pgo …'

`/build`, `/tools/deps`, `--enable-bolt` and the `BOLT_*` skip-lists are PBS's
container conventions; no other project builds CPython this way.
Bundled `pip` is 26.2.1.

**Where to get it again.**

    https://github.com/astral-sh/python-build-standalone/releases/download/20260825/cpython-3.13.15+20260825-x86_64-unknown-linux-gnu-install_only.tar.gz
    sha256 8a70011ae25276a9925f89304cdc086466cd269ee6cfe68a9506694ca5ff4f9c
    119,852,836 bytes

**Verified, without re-downloading 120 MB.** The tarball was kept, at
`_download/linux.tar.gz`; its local sha256 is the value above, and that value
appears against exactly that filename in the release's published `SHA256SUMS`
asset (fetched, 20260825). Size matches too, and no other 3.13.15 release
(20260807, 20260814, 20260901) matches either size — so the tag is pinned by
two independent facts.

The archive unpacks to a `python/` root; the directory on disk is that root's
contents renamed to the triple. That renaming is local convention, not
upstream layout.

**What reads it.** `tools/wheel/verify_linux.py` — `LINUX_PYTHON.glob(
"lib/libpython3.*.so.*")` for the export set the wheel's undefined symbols are
unioned against, plus `lib/python3.13/lib-dynload/_dbm…so` and `_tkinter…so`
and `lib/libtcl9*.so` as its positive controls. `tools/wheel/verify_cross.py`
globs `<triple>/lib/libpython3.*.so`. A same-project rebuild is safe. A
distribution without `lib-dynload/_dbm` or without Tcl/Tk would lose the
controls (the script's own §"positive controls" reasoning), and one that ships
only a static libpython would fail the glob outright — which the script
reports with an explicit "TORCHNATIVE_TARGET_PYTHON selects the distribution
root" message rather than a traceback.

    lib/libpython3.13.so.1.0  sha256
      06ae3c82efa4a938c7144574377994db2016a012bb5dca91acc4aff20674fa1c

---

## 4. `x86_64-pc-windows-msvc` — python-build-standalone 20260825

**What it is.** The Windows artefact of the *same* PBS release. There is no
`_sysconfigdata` on Windows, so the evidence is the layout and the version:
`include/patchlevel.h` gives `PY_VERSION "3.13.15"`, `python313.dll` carries
the string `3.13.15`, and the root is PBS's Windows shape — `python.exe`,
`python3.dll`, `python313.dll`, `DLLs/`, `Lib/`, `libs/`, `tcl/`,
`vcruntime140{,_1}.dll`, and **`.pdb` next to every binary**, which the
python.org installer does not ship. Bundled `pip` is 26.2.1, matching §3.

**Where to get it again.**

    https://github.com/astral-sh/python-build-standalone/releases/download/20260825/cpython-3.13.15+20260825-x86_64-pc-windows-msvc-install_only.tar.gz
    sha256 82a792c25550a421b29f381eaeafa6dccd1ffcbd97a1b1507b202f5df877cecf
    47,207,576 bytes

Verified the same way: `_download/windows.tar.gz` hashes to that value, and
that value appears against that filename in the 20260825 `SHA256SUMS`.

**What reads it.** `tools/wheel/verify_windows.py` reads exports out of
`python3.dll`, `vcruntime140.dll` and `vcruntime140_1.dll` (its `RESOLVABLE`
tuple), and deliberately refuses to resolve against `python313.dll` so an
abi3 extension cannot be let off for binding the version-specific DLL.
`tools/wheel/verify_cross.py` globs `python3??.dll` for the interpreter's
dynload table. **This is the layout-sensitive part**: the three names in
`RESOLVABLE` are hardcoded, so a distribution that omits the vcruntime DLLs
(they are redistributables, and some layouts assume a system install) would
turn those checks from "checked" into "unresolved" without any error — a
weaker verdict that still prints as a run.

    python3.dll    d577f725e8f4d99ec448b33465356bf924f3029050f5c7ea648e6db919e8d78a
    python313.dll  994063df8b0dfb72d49f1c49fb8747571cb398951b23b87de568011c7cb3bde6

---

## 5. `aarch64-linux-android` — a local build from a modified tree

**What it is.** CPython's own `Android/` cross-build support, run on a **Linux
machine that is not this one**: hostname `BOOK4U-G72AG`, user `brew24`, in
`/home/brew24/cpython/cross-build/aarch64-linux-android`. Built **13 Oct 2024**.

    _sysconfigdata__android_aarch64-linux-android.py
      CONFIG_ARGS = '--host=aarch64-linux-android' '--build=x86_64-pc-linux-gnu'
                    '--with-build-python=/home/brew24/cpython/cross-build/build/python'
                    '--without-ensurepip' '--enable-shared'
                    '--with-openssl=…/aarch64-linux-android/prefix'
                    'CC=…/ndk/26.2.11394342/toolchains/llvm/prebuilt/
                          linux-x86_64/bin/aarch64-linux-android21-clang'
      ANDROID_API_LEVEL = 21
      SOABI             = cpython-313-aarch64-linux-android
    patchlevel.h  PY_VERSION "3.13.0+"
    compiler      Clang 17.0.2 (NDK 26.2.11394342)
    lib/libpython3.13.so  sha256
      9783fd7f29b17e4434348b2e8518c3b5dbe66e72f8263d97d72120c3fbf59549

The full `build/` tree came along with the install `prefix/` — `config.log`,
`config.status`, `Makefile`, object directories, 468 MB in total. That is why
the provenance was recoverable at all here: `config.log`'s first lines record
the `configure` invocation and the build host verbatim.

**This one is not reproducible, and the reason is in the binary.** The build
info string is:

    Oct 13 2024 / b4c504d76ff / heads/3.13-dirty

`b4c504d76ff` resolves upstream to `b4c504d76ff3aa42943854571fa7610db5407e80`
("[3.13] gh-124309: fix staggered race on eager tasks", 2024-10-12) — but
**`-dirty` means the working tree carried uncommitted modifications**, and
nothing on disk records what they were. Checking out that commit and
rebuilding gives a *near* equivalent, not the same thing. `--with-openssl`
also points at a prefix whose OpenSSL version is unrecorded, same gap as §1.

Whether the modifications mattered is unknown and, from the artefacts alone,
unknowable. In 2024 CPython's Android support was new and out-of-tree patching
was ordinary, so the prior should be "they mattered somewhat".

**What reads it.**

- `scripts/device_android.sh` — `TARGET_PYTHON` defaults to
  `…/aarch64-linux-android/prefix`.
- `tools/wheel/verify_android.py` — pushes exactly three paths to the device:
  `prefix/bin/python3.13`, `prefix/lib/libpython3.13.so`, and the whole
  `prefix/lib/python3.13` stdlib, then runs
  `LD_LIBRARY_PATH=… PYTHONHOME=… ./bin/python3.13`. This is the one place a
  target distribution is *executed* rather than inspected, so it is the most
  sensitive to a swap: a static-only or `bin/`-less distribution has nothing
  to push, and a different API level changes what the device will load.
- `tools/wheel/verify_cross.py` globs
  `aarch64-linux-android/prefix/lib/libpython3.*.so`.
- `tools/wheel/build.py` reads `ANDROID_API_LEVEL` out of the sysconfig for
  the wheel's platform tag, deliberately rather than hardcoding it — so
  replacing this distribution correctly changes the tag, by design.

Note the extra `prefix/` level: this is the only one of the five where the
distribution root is a subdirectory, and every consumer hardcodes it.

---

## 6. What is still unrecoverable

Stated plainly, because a guess here would be worse than the gap.

1. **The Android tree's local modifications (§5).** `heads/3.13-dirty` says
   they existed; nothing says what they were. The build host `BOOK4U-G72AG`
   is not this machine and is not otherwise referenced in this repository.
   This is the only distribution of the five whose *source* cannot be
   reconstructed.
2. **The exact third-party dependency versions** in the two iOS builds and the
   Android build. `CONFIG_ARGS` records `--with-openssl=<prefix>` and the
   xz/bzip2/libffi include and lib paths, but a prefix is not a version, and
   the prefixes no longer exist. A rebuild picks whatever the recipe's scripts
   fetch today.
3. **Byte-identical reproduction of anything locally built.** Even with the
   commit pinned, `Py_GetBuildInfo` embeds a build timestamp and the objects
   embed absolute source paths, so §1, §2 and §5 can be *equivalent* but never
   *identical*. Their sha256s above are therefore identity records for the
   files now on disk — useful for detecting drift or corruption, not for
   validating a rebuild.
4. **The original source trees.** `/Users/ibrew/Desktop/cypthon/{arm64,armsim64}`
   no longer exists on this machine (checked). The Android `build/` tree
   survived only because it was copied wholesale alongside the prefix.

What is *no longer* unrecoverable, and was before this round: the Linux and
Windows distributions are now pinned to a published URL and a checksum
verified against the vendor's own `SHA256SUMS`, and both iOS distributions are
pinned to an upstream CPython commit that exists and a documented build
procedure. That is enough for the iOS CI job in §1 to be specified, which was
the thread this started from.

## Rule going forward

Anything added to `target-python/` gets its download kept in `_download/` and
a section here. The two that kept their tarballs took minutes to identify; the
two that did not took reading disassembled build strings, and the one that was
neither downloaded nor clean is the one that stayed lost.
