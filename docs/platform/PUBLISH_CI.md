# Publishing from a tag: `.github/workflows/publish-pypi.yml`

**This workflow has never run.** It was written and validated without a runner
— YAML parsed, job graph read, tag/version logic exercised locally, and every
guarantee nullified and confirmed red (§7). Written, reached and agreed are
three different claims, and this is the first. The first real use should be a
`workflow_dispatch` with `dry_run: true`, which builds and verifies nine
wheels and uploads nothing.

Tests: **38** in `rust/torch_c/pytests/test_cipub.py`.

---

## 0. What it replaces, and why that matters more than the automation

The manual release was: one person, one Mac, nine `python tools/wheel/build.py
--target ...` invocations, `twine upload`. The problem was not the typing. It
was that the toolchain lived in that machine's scratch directories — a `zig`
shim at `/tmp/zigbin`, an emsdk under `/Volumes/macMini/caches`, a
hand-written `PYO3_CONFIG_FILE` for iOS — so the recipe existed only as
whatever was on PATH that afternoon. Restoring it cost a full round.

Second, on 2026-09-11 eight cross artefacts were **44 hours stale** against the
Rust sources, and only `build.py`'s freshness guard (`require_current`)
stopped superseded machine code shipping. A runner starts empty every time, so
that failure mode stops existing rather than being caught.

Moving the wiring into a file is the point. The automation is a side effect.

---

## 1. Can `vendor/vendor_torch.sh` run on a CI runner?

**Yes, and it produces a byte-identical tree — with one pin that is not
optional.**

The script reads a torch installation and copies its Python tree, dropping
every compiled artefact. Its default source is a local venv
(`/Volumes/macMini/caches/spike-venv/...`), but the path is
`TORCHNATIVE_TORCH_SRC`, so anything shaped like site-packages works. Needs:
`rsync`, `sed`, `find` (all present on both hosted runner images) and a
torch wheel. Takes about a minute after the download.

### 1.1 The pin

`vendor_torch.sh` copies upstream's `torch-2.13.0.dist-info/` **verbatim**,
and `build.py`'s `upstream_dist_info()` puts it into
`torchnative-<v>.data/purelib/` in every one of the nine wheels. That
directory's `WHEEL` file reads:

    Tag: cp313-cp313-macosx_14_0_arm64

So the *platform of the torch wheel that was vendored from* is shipped inside
all nine of our wheels, including the Windows and wasm ones. Vendor from a
Linux torch wheel on a Linux runner and that member changes — and so does
`METADATA`, whose `Requires-Dist` differs between the CUDA and macOS builds
(the local tree's METADATA mentions `nvidia` 15 times; a CPU-only build's
would not match either).

The fix is a pure download, and it works from any host because pip resolves for
a platform it is not running on:

    pip download torch==2.13.0 --only-binary=:all: --no-deps \
        --platform macosx_14_0_arm64 --python-version 3.13 -d _download

Measured here (2026-09-12): 111,213,743 bytes, ~40 s.

### 1.2 The measurement

Vendored from exactly that wheel into a scratch tree and compared, member by
member, against the released `torchnative-0.1.0b3-cp313-abi3-macosx_11_0_arm64.whl`
in `dist/`:

    wheel members (vendored roots)   2671
    disk files    (vendored roots)   2666
    shared                           2666
    identical                        2666
    differing                        0
    only on disk                     0
    only in wheel                    5

`.stamp` also agrees: `py_modules=2372`, `native_left=0`, `add_hooks=1`
— the same numbers the real tree carries.

The five are each accounted for, and none of them comes from vendoring:

| member | why it is only in the wheel |
|---|---|
| `torch/_C.abi3.so` | our shim, put there by `vendor/install_shim.sh` |
| `torch/bin/torch_shm_manager` | the zero-byte wall-4 marker `install_shim.sh` places |
| `torch/lib/libtorch_global_deps.dylib` | `build.py`'s `global_deps_stub()`, per target |
| `torch-2.13.0.dist-info/INSTALLER` | see §2.2 |
| `torch-2.13.0.dist-info/REQUESTED` | see §2.2 |

**Caching.** `actions/cache` on the unpacked wheel, keyed on version +
platform. It never changes within a release line, so the 111 MB download
happens once per cache eviction.

---

## 2. Does a CI-built wheel match a locally-built one?

### 2.1 What was determined, and what was not

**Determined, by measurement:** the ~2,666 vendored members — 98% of each
15 MB wheel — come out byte-identical from a wheel-only vendoring source, with
no macOS and no spike-venv involved. That is the part that was in doubt and it
is now not.

**Not determinable without a runner:** the compiled `_C` in each wheel. It
cannot be byte-identical and should not be expected to be. `build.py` already
rewrites the Mach-O install name because cargo embeds `CARGO_TARGET_DIR`
into it (`_fix_install_name`); rustc additionally embeds absolute source and
`~/.cargo/registry` paths in panic strings and debug info. A different
toolchain version changes codegen outright. What CI can promise about `_C` is
what `tools/wheel/verify_cross.py` already checks — architecture, Mach-O
`LC_BUILD_VERSION` platform / ELF machine / PE machine, the `PyInit__C`
export, the DT_NEEDED set, the glibc floor derived from `.gnu.version_r` —
not a hash.

**Zip metadata** (member mtimes, the zip's own ordering) differs by
construction and is not part of any check.

### 2.2 The one thing CI would otherwise change, and the workflow's answer

`torch-2.13.0.dist-info/INSTALLER` and `REQUESTED` are **not in upstream's
wheel**. They are written into site-packages by the installer, and the released
0.1.0b3 wheels carry them because the vendoring source was an installed venv:

    INSTALLER   b'uv'
    REQUESTED   b''

The workflow recreates both after unzipping, so the CI wheel matches the
verified one. **They should probably be dropped instead** — they record which
tool installed torch on somebody's laptop, they say `uv` inside a wheel
installed by pip, and nothing reads them. That is a content change to the
published artefact, so it is left as a recommendation rather than made here.

### 2.3 The machinery, reused rather than rewritten

`tools/wheel/verify_cross.py --reference <wheel>` is the member-list
comparison (its `_default_reference` picks the same-version macOS wheel; the
docstring records the 2026-08-30 incident where an unscoped picker compared
0.0.4a0 against 0.0.2a0 and passed). The `collect` job runs it for each of the
eight cross wheels against the macOS one. No third comparison tool was written.

---

## 3. The three target CPythons CI cannot obtain

Every target's platform tag is **derived from a target CPython distribution**
rather than written down — `ANDROID_API_LEVEL`, `IPHONEOS_DEPLOYMENT_TARGET`,
`MULTIARCH`. So a missing distribution is not a build inconvenience, it is a
missing tag.

`docs/platform/TARGET_PYTHON.md` identified all of them. Restated as a CI
question:

| distribution | on a fresh runner |
|---|---|
| `x86_64-unknown-linux-gnu` | **download**, python-build-standalone 20260825, sha256 pinned |
| `aarch64-unknown-linux-gnu` | **download**, same release |
| `x86_64-pc-windows-msvc` | **download**, same release |
| `aarch64-pc-windows-msvc` | **download**, same release |
| Pyodide 314.0.6 core | **download**, pyodide release asset |
| `arm64-iphoneos` | rebuildable (CPython @ `d894d467a61`, 30–60 min) — **not downloadable** |
| `arm64-iphonesimulator` | same |
| `aarch64-linux-android` | **not reproducible at all** — TARGET_PYTHON.md §5: cross-built on a machine that is not this one, from a `heads/3.13-dirty` tree, and nothing records what the modifications were |

The last one is the hard stop, and it is why "nine wheels in CI" is not
achievable today by building alone.

### 3.1 The chosen answer: one checksummed bundle

The three are fetched as a single tar, pinned by URL and sha256 in two
repository variables:

    TARGET_PYTHON_BUNDLE_URL       a release asset or other stable URL
    TARGET_PYTHON_BUNDLE_SHA256    checked with shasum -a 256 -c before use

    tar -czf target-python-bundle.tar.gz -C /Volumes/macMini/caches/target-python \
        arm64-iphoneos arm64-iphonesimulator aarch64-linux-android
    # ~700 MB unpacked; publish once, then set both variables

**This is a one-time human step and it has not been done.** `preflight`
refuses in two seconds with that instruction if the URL variable is unset, and
the build job refuses if the checksum variable is.

Why a bundle rather than rebuilding the two iOS ones in CI, which is possible:

* It makes iOS *identical* to the local distributions rather than
  "functionally equivalent". TARGET_PYTHON.md §6.3 is explicit that a rebuild
  can never be byte-identical — build timestamps and absolute source paths are
  embedded — so a rebuild would silently change what the iOS wheels were built
  against.
* It is one mechanism for three problems instead of two mechanisms for two,
  and the Android one has no second option.
* 30–60 min of macOS runner time per cache miss, avoided.

What it costs: the bundle is the one input to the release that nothing else can
check. The sha256 is the whole of its integrity, which is why the build job
refuses rather than warns when it is unset.

---

## 4. The matrix, and why each target is where it is

| runner | targets | why |
|---|---|---|
| `macos-14` | `host`, `ios-arm64`, `ios-arm64-sim` | not a preference. The macOS wheel's tag is read out of the Mach-O the host build produced (`honest_macos_plat`), and both iOS builds need `xcrun` and an iOS SDK — the iOS distribution's own compiler shims in `bin/arm64-apple-ios-simulator-clang` are two-line `xcrun` wrappers. There is no cross-from-Linux route for either. |
| `ubuntu-24.04` | `linux-x86_64`, `linux-aarch64`, `windows-x86_64`, `windows-arm64`, `android-arm64-v8a`, `wasm32-emscripten` | all six are already cross builds today, through toolchains that do not care what host they run on: `cargo zigbuild`, `cargo xwin`, `cargo ndk`, emscripten. On Linux the NDK also stops going through the `darwin-x86_64` prebuilts under Rosetta. A Linux runner is a tenth the cost. |

`fail-fast: false`: one broken target reports next to the other eight rather
than hiding them behind a cancellation.

Every job builds the host shim first (`vendor/install_shim.sh`), because
`build.py`'s `preflight` requires `torch/_C.abi3.so` to exist even when the
wheel is for another platform — a tree with the hole still open is the
`py3-none-any` shell.

---

## 5. The refusals

### 5.1 The tag must be the version

`tools/ci/check_tag_version.py`, run in `preflight`, which every other job
depends on. Nothing checked this before: `build.py` reads the version out of
the metadata and never sees the tag; the release process read the tag and never
saw `pyproject.toml`. `git tag v0.1.0b4 && git push --tags` would have
published **0.1.0b3** under that name — nine internally consistent wheels, all
the wrong version — and PyPI never allows a filename to be reused, even after
deletion, so the real 0.1.0b4 would have been burnt.

The comparison is on the **literal text** as well as the parsed version.
`v0.1.0-beta3` and `0.1.0b3` normalise to the same PEP 440 release and are
refused, because the tag text is what appears on the release page and what a
reader copies into `pip install torchnative==...`.

Run it locally:

    python tools/ci/check_tag_version.py v0.1.0b3     # exit 0
    python tools/ci/check_tag_version.py v0.1.0b4     # exit 1, names both

### 5.2 Nine wheels or no upload

`collect` downloads all nine artefacts and checks the **filenames by name**,
not by count — a count passes when two jobs both produce a macOS wheel and the
Windows one is missing, and cannot say which. Same reasoning as `build.py`'s
`EXPECTED_TARGET_KEYS`, which exists because a dict comprehension collapsed
two entries and a count could not tell.

Then `twine check --strict` on every wheel, then `verify_cross.py
--reference` on each of the eight against the macOS one. `publish` needs
`collect`, so a partial matrix never reaches an upload step.

### 5.3 The exposure that remains

PyPI has no atomic multi-file upload. `gh-action-pypi-publish` hands twine one
directory, but each file is a separate request, so a network failure at file 4
leaves 3 live. `skip-existing: true` lets a re-run complete the set instead of
dying on the three already there.

What is left is a window — seconds to minutes, until the re-run — in which PyPI
serves a partial platform set, and `pip install torchnative` on a platform
whose wheel has not landed resolves to the previous release or to nothing.
There is no way to close that from the client side. It is smaller than today's
exposure, where the same window exists and is bounded by how fast a human
notices.

### 5.4 The gate: run, not required

The workflow **runs** `rust/torch_c/pytests/run.sh` on the tagged tree rather
than requiring a green status on the commit.

Requiring is faster and trusts two things: that the gate ran on *this* tree — a
tag can be moved, and cutting a hotfix tag on an older commit is ordinary — and
that the status API's answer is about the same suite. A release is a handful per
month. An hour of runner time is the cheaper side of that trade.

It runs on `macos-14` rather than Linux because the recorded baseline (1438
ok, DOCWATCH 1122/1122, cargo 33/33) was measured on macOS arm64 and nobody has
measured the suite on Linux. A gate whose number cannot be compared to the
recorded one cannot notice a regression.

---

## 6. `workflow_dispatch` and dry run

`dry_run` defaults to **true**, so the safe answer is also the default answer.
A dispatch with `dry_run: true` runs preflight, vendor, gate, all nine builds
and all of `collect`; `publish` is skipped by its `if:`. A tag push always
sets `dry_run=false`.

---

## 7. What was nullified, and what went red

Each guarantee was broken on purpose and the test confirmed to fail.

| nullification | test that went red |
|---|---|
| `environment: pypi` -> `environment: testpypi` | `test_the_publish_job_runs_in_the_pypi_environment` |
| `id-token: write` removed from `publish` | `test_the_publish_job_asks_for_an_oidc_token` |
| `password: ${{ secrets.PYPI_API_TOKEN }}` added to the upload step | `test_the_workflow_carries_no_pypi_token_of_any_kind` |
| the `check_tag_version.py` invocation replaced with `echo` | `test_the_tag_version_check_actually_runs`, `test_the_tag_version_check_runs_before_anything_is_built` |
| `needs: [preflight, collect]` -> `needs: [preflight]` on `publish` | `test_the_publish_job_cannot_start_without_the_nine_wheel_check` |
| `windows-arm64` removed from the matrix | `test_the_matrix_builds_exactly_the_nine_targets` |
| `--strict` dropped from `twine check` | `test_every_wheel_is_twine_checked_before_the_upload_job` |
| `|| true` appended to the gate step | `test_nothing_in_the_workflow_disables_its_own_assertions` |
| `--platform macosx_14_0_arm64` dropped from the torch download | `test_the_vendoring_source_is_pinned_to_a_platform` |
| the reference wheel replaced with `/dev/null` | `test_the_reference_comparison_runs_with_a_real_reference` |
| a `vendor_torch.sh` step added back into the build matrix | `test_the_vendored_tree_is_built_once_and_shared` |
| `dry_run` default flipped to `false` | `test_the_dry_run_input_exists_and_defaults_to_not_uploading` |
| the bundle's sha256 check removed | `test_the_unobtainable_target_pythons_are_refused_early_and_by_name` |
| `compare()`'s literal-text comparison disabled | `test_a_disagreeing_version_is_reported_and_names_both`, `test_a_tag_that_normalises_equal_but_reads_differently_is_refused` |

Fourteen nullifications, fourteen red. Restored: 38 ok, 0 FAIL.

Note what the sixth row does **not** also break. `test_the_matrix_matches_build_pys_own_registry` compares this file's `NINE_TARGETS` constant against `build.py`'s `EXPECTED_TARGET_KEYS`, not against the matrix, so removing a matrix entry does not move it; removing a *registry* entry does. The two tests are a chain -- registry == constant, constant == matrix -- and it is the chain, not either link, that makes "the matrix follows the registry" checkable.

---

## 8. What this round did not do

* Nothing was published, to PyPI or TestPyPI.
* No token was added anywhere.
* `dist/` was not touched.
* No guard in `build.py` was weakened. None was in the way: `require_current`
  and `check_build_cache` both pass trivially on a runner whose disk was empty
  five minutes earlier, which is the point.
* The target-python bundle (§3.1) has **not** been published and the two
  repository variables are **not** set. Until they are, the workflow refuses in
  `preflight`.
