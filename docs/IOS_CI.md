# Why the iOS simulator is not checked in CI, and what it would take

The obvious job works on paper. A hosted `macos-latest` runner has Xcode and
simulators, the simulator wheel is on PyPI, and `tools/wheel/verify_ios_sim.py`
is the same harness that passes here on every release build. It was added, run,
and failed in sixteen seconds:

```
no simulator CPython at /Volumes/macMini/caches/target-python/arm64-iphonesimulator
```

## Xcode is not the missing piece

`verify_ios_sim.py` does not run the wheel against the runner's own CPython —
it cannot, the Mach-O platform differs. It unpacks the wheel into an **iOS
CPython's `site-packages`** and drives that interpreter inside a booted
simulator. So the job needs a CPython built for `arm64-iphonesimulator`, and a
GitHub runner has no such thing.

This machine does, at `/Volumes/macMini/caches/target-python/arm64-iphonesimulator`:

```
205 MB, created 18 October 2024
bin/  include/  lib/  Python.framework/
```

Its siblings for Android, Linux and Windows sit beside it, and two of those have
their download tarballs preserved in `_download/` (`linux.tar.gz`,
`windows.tar.gz`). **The two Apple ones do not, and no document in this
repository records where any of them came from.**

## So the blocker is provenance, not configuration

> **Answered — see [`docs/TARGET_PYTHON.md`](TARGET_PYTHON.md).** It is neither
> candidate guessed at below. It is a **local build of CPython's own in-tree
> `iOS/` support**, made on a Mac on 18 October 2024 at commit `d894d467a61`
> (branch `3.13`, which resolves upstream). There is no URL, because there was
> never a published artefact — but there is a pinned source and a documented
> build procedure, which is enough to reconstruct an equivalent one. Never a
> byte-identical one: the build embeds its own timestamp and paths, and the
> versions of OpenSSL, xz, bzip2 and libffi it linked were never recorded.
>
> So the CI job is now sized rather than blocked: a macOS runner with Xcode 16+,
> roughly 30–60 minutes to build host-python, four dependencies and the iOS
> configuration — cacheable on commit plus Xcode version — and
> `TARGET_PYTHON_IOS_SIM` pointed at the prefix. `verify_ios_sim.py` already
> reads that variable, so the harness itself needs no change. The device half
> rides the same build and needs no hardware.

Putting this in CI meant answering "where does an `arm64-iphonesimulator`
CPython 3.13 come from, reproducibly" — a published artefact with a URL and a
checksum, fetched by the workflow. Candidates existed (CPython 3.13 supports iOS
as a tier-3 target; BeeWare's `Python-Apple-support` publishes simulator
XCFrameworks) and **neither of them is what is in that directory**, which is why
guessing would have been worse than looking.

## What is true today

- The simulator check **runs and passes on every release build here**, most
  recently against the published `0.0.11a0` wheel: `platform.system()` answers
  `iOS`, and `aten.mm`, `x + x` and an `nn.Linear` forward all return upstream's
  values. That is why the README's simulator column is ✅.
- **This machine is the only witness.** If it were lost, iOS verification could
  not be reconstructed from the repository — that is the real cost of the
  undocumented distribution, and it is larger than the missing CI job.
- The **device** rung is untouched by any of this. A runner has no phone, and the
  simulator runs on the host kernel — its own output prints this Mac's
  `uname().version`. CI cannot close it at any level of effort.

## The next round, if it is taken

1. Identify what the four `target-python/` distributions are, by inspecting them
   rather than guessing, and record it — that is worth doing whether or not CI
   follows.
2. Find a published, checksummed source for the simulator one.
3. Only then add the job, with the fetch as its first step.

<!-- DOCWATCH: symbol-in-file tools/wheel/verify_ios_sim.py arm64-iphonesimulator present -->
