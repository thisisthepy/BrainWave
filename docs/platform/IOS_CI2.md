# Putting the iOS simulator check in CI — the build-it-yourself round

`docs/platform/IOS_CI.md` answered "where does the runner get an `arm64-iphonesimulator`
CPython" with "nowhere published — but the source is pinned, so build one."
This document is that job, and an honest account of how much of it has been
run.

## First, the sizing had to be checked, not inherited

Before writing a single step, two claims from `docs/platform/TARGET_PYTHON.md` needed
verifying against the actual harness rather than taken on faith:

- **Does `verify_ios_sim.py` key off `TARGET_PYTHON_IOS_SIM`, or a hardcoded
  path?** Read from the source (`tools/wheel/verify_ios_sim.py:86-88`):

      TARGET_PYTHON = Path(os.environ.get(
          "TARGET_PYTHON_IOS_SIM",
          "/Volumes/macMini/caches/target-python/arm64-iphonesimulator"))

  It is the environment variable, with that path only as the **default when
  the variable is unset**. On a hosted runner the variable will always be set
  (the workflow sets it), so the hardcoded fallback never executes there. The
  sizing holds: nothing in `verify_ios_sim.py` needs to change.
- **Does the wheel-selection glob match what `pip download` actually
  produces?** `docs/platform/IOS.md` records a real filename,
  `torchnative-0.0.1a0-cp313-abi3-ios_14_0_arm64_iphonesimulator.whl`, and the
  script's own check is `if "iphonesimulator" not in args.wheel.name`. The
  workflow's glob (`dist/torchnative-*-ios_*_iphonesimulator.whl`) matches that
  shape.
- **Does a previously-written, previously-reverted version of this job already
  answer the download question?** Yes — `git show ff831ff:.github/workflows/…`
  (the revert commit's parent) contains a `Download the simulator wheel` step
  using `pip download --platform ios_14_0_arm64_iphonesimulator
  --python-version 3.13 --abi abi3 --no-deps --only-binary :all:`, which is
  independently what this round arrived at before finding that commit. That
  prior attempt's failure was never the download step — it never got that
  far, because there was no simulator CPython to unpack into. This round
  restores that step unchanged and adds the part that was missing: building
  the CPython first.

## What the new job does

`.github/workflows/verify-published-wheel.yml` gained one matrix leg,
`ios-simulator-arm64` on `macos-15`, in the same job as the existing
`linux-x86_64` / `windows-amd64` legs (same shape the reverted attempt used:
one job, `if: matrix.label == '...'` / `!= '...'` guards per step, rather than
a separate job). Its steps, guarded to run only for that label:

1. **Xcode in use** — records `xcodebuild -version` for the cache key. No
   Xcode is installed explicitly; whatever `macos-15` ships is what gets used
   and pinned into the key, so a runner-image Xcode bump invalidates the cache
   instead of silently building against a different one.
2. **Cache the iOS-simulator CPython** (`actions/cache@v4`) — keyed on the
   pinned CPython commit (`d894d467a615ada8150e9e6987f1969f0a5e2cd9`, from
   `docs/platform/TARGET_PYTHON.md` §1) plus the Xcode version string. This is the step
   that makes a 30-60 minute build tolerable on every run: cache hit skips the
   next step entirely.
3. **Build the arm64-iphonesimulator CPython** (only on cache miss) — follows
   CPython's own `iOS/README.rst` procedure at the pinned commit (fetched and
   read while writing this, not assumed):
   - Downloads the four dependency archives (XZ, BZip2, libFFI, OpenSSL) as
     prebuilt `iphonesimulator.arm64` tarballs from
     `beeware/cpython-apple-source-deps` releases — the same source
     `iOS/README.rst` names for these. Tags are pinned in the job's `env:`
     (`IOS_DEPS_TAG_*`), taken from that repository's release list at the time
     of writing (`XZ-5.6.4-2`, `BZip2-1.0.8-2`, `libFFI-3.4.7-2`,
     `OpenSSL-3.5.8-1`).
   - Checks out the pinned CPython commit **twice**, into `cpython-host` and
     `cpython-ios` — the cross-build needs an already-built same-version host
     interpreter to run its own build-time Python scripts, and a single
     source tree cannot be configured for both a host and a cross build at
     once.
   - Builds `cpython-host` as a plain macOS `./configure && make && make
     install`.
   - Cross-configures `cpython-ios` with `iOS/Resources/bin` prepended to
     `PATH` (the compiler-shim wrappers `iOS/README.rst` requires),
     `--host=arm64-apple-ios-simulator --build=arm64-apple-darwin
     --enable-framework=$RUNNER_TEMP/ios-sim-python
     --with-build-python=.../cpython-host/bin/python3.13
     --with-openssl=$RUNNER_TEMP/deps/openssl`, and the `LIBLZMA_*`/`BZIP2_*`/
     `LIBFFI_*` `_CFLAGS`/`_LIBS` pairs pointed at the downloaded archives'
     `include/`/`lib/` (confirmed flat by extracting the libFFI tarball during
     this round — `include/ffi.h`, `lib/libffi.a` directly under the
     extraction root, no nested version directory).
   - `make && make install`. `--enable-framework=DIR` with an absolute `DIR`
     assembles `bin/`, `include/`, `lib/`, `Python.framework/` directly under
     `DIR` — the same top-level shape `docs/platform/TARGET_PYTHON.md` describes for
     the machine's own copy — so `DIR` itself becomes `TARGET_PYTHON_IOS_SIM`.
4. **Point the harness at the built CPython** — exports
   `TARGET_PYTHON_IOS_SIM=$RUNNER_TEMP/ios-sim-python` via `GITHUB_ENV`, cache
   hit or not.
5. **Download the simulator wheel** — `pip download` from PyPI by platform
   tag, matching the existing Linux/Windows legs' "install what a user gets"
   framing and reusing the exact invocation the reverted attempt already had.
6. **Verify it computes in a simulator** — `python
   tools/wheel/verify_ios_sim.py dist/torchnative-*-ios_*_iphonesimulator.whl`,
   unchanged from the harness that already runs on every release build here.

The five existing Linux/Windows steps (`Install the published wheel` through
`Run a real model`) are guarded `if: matrix.label != 'ios-simulator-arm64'`,
matching the reverted attempt's shape exactly — so this leg skips them
cleanly, and they skip the new steps cleanly by the mirror-image guard.

## What has been executed, and what has not

**Executed, in this round:**

- Read `tools/wheel/verify_ios_sim.py` end to end to confirm the
  `TARGET_PYTHON_IOS_SIM` / hardcoded-default question above.
- Fetched and read `iOS/README.rst` from `python/cpython` at the pinned
  commit over the network, to write the configure invocation from the actual
  documented procedure rather than from memory.
- Queried the GitHub API for the `iOS/` directory at the pinned commit (only
  `README.rst`, `Resources/`, `testbed/` — no `Makefile` automation exists at
  this commit, so the workflow has to script the steps `iOS/README.rst`
  describes by hand rather than calling one provided target).
- Queried `beeware/cpython-apple-source-deps`'s release list over the network
  and confirmed asset names and the most recent tag per dependency.
- Downloaded one dependency archive (`libFFI-3.4.7-2-iphonesimulator.arm64.tar.gz`)
  and extracted it, to confirm the `include/`/`lib/` layout the `_CFLAGS`/
  `_LIBS` flags assume.
- Diffed this round's workflow against the previously-reverted commit
  (`git show ff831ff:… | diff -`) and confirmed the download step matches
  independently, and that the guard shape (`if: matrix.label == /
  != 'ios-simulator-arm64'`) is inherited rather than invented here.
- Parsed the edited YAML with `yaml.safe_load` — it loads, and the matrix and
  step count are as expected.

**Not executed, and not claimed to work:**

- **The workflow has never run on a GitHub-hosted runner.** The branch
  (`work/ioscib`) is not pushed to `origin` — confirmed by `git branch -a`
  showing no `origin/work/ioscib` — so `gh workflow run` has nothing to target
  and was not invoked, per this round's instructions. Nothing below the "What
  we are standing on" step for the `ios-simulator-arm64` leg has ever executed
  anywhere.
- **The `./configure`/`make`/`make install` invocations for both the host and
  the cross build are unverified.** They follow the documented procedure, but
  this Mac's own `arm64-iphonesimulator` build predates this repository and
  was made by hand (`docs/platform/TARGET_PYTHON.md`), not from a script that can be
  diffed against. A first real run is likely to need at least one correction —
  a missing `PYTHONFORBUILD`-shaped environment variable, an OpenSSL/libFFI
  minor-version mismatch with what CPython 3.13 at this commit expects, or a
  path assumption specific to this machine.
- **The YAML parse check is syntactic only.** It confirms the file is valid
  YAML and the `if:`/matrix structure is shaped as intended; it does not run
  `actionlint` (not installed in this environment) and cannot catch a
  semantically wrong but syntactically valid expression — which is exactly
  the class of bug the original revert was not caught by, per this round's
  brief.
- **Whether `macos-15` ships an Xcode new enough.** `docs/platform/TARGET_PYTHON.md`
  says "Xcode 16+"; this round did not install a pinned Xcode version and did
  not verify what `macos-15`'s default is at the time a workflow actually
  runs — the "Xcode in use" step records it into the cache key so a mismatch
  is at least visible, but does not guarantee sufficiency.
- **Whether GitHub's hosted-runner licence for Xcode covers this.** Building
  CPython from source and linking system frameworks (`Python.framework`,
  `UIKit`) does not touch code signing or the App Store, and `verify_ios_sim.py`
  already does the same on this machine without any signing identity. Nothing
  found suggests a licence or entitlement problem, but nothing here can
  substitute for a real run confirming it.

## What this still cannot do

Unchanged from `docs/platform/IOS_CI.md`: **the device rung is untouched.** A runner
has no phone, and the simulator runs on the host kernel — its probe output is
the runner's own `uname().version`, not an iPhone's. The README's platform
table keeps separate simulator and device columns; this round does not touch
the device column and nothing here should be read as closing it.

<!-- DOCWATCH: symbol-in-file tools/wheel/verify_ios_sim.py TARGET_PYTHON_IOS_SIM present -->
