# 0.1.0b3 — release notes

**The first release shaped by a real NPU.** A user ran `0.1.0b2` on a
Windows Intel NPU laptop. `to(device.npu)` worked — 252 of Qwen3-4B's
Linears lowered and OpenVINO reported `EXECUTION_DEVICES=['NPU']`, which
is the first evidence in this project's history that anything reached an
NPU. Then `generate()` died, and finding out why produced most of this
release.

Everything below was found by running on the device. None of it was
visible from here.

Read §3 before deciding what this does for you. `0.1.0b2` **cannot
generate with Qwen3-4B at all**; if you have that version, this replaces it.

---

## 1. Features added

- **`to(device.npu)` compiles the decode shape up front**, with a
  `progress(done, total, name)` callback. `_NPULinear` compiles a
  static-shape IR per batch, and `generate()` uses two shapes — the
  prompt length, then 1 per token with a KV cache — so 504 driver
  compiles used to happen lazily, *inside the first generated token*,
  with no way to tell a stall from a hang. They now happen where the
  caller asked for them and can say what they are doing. `eager=False`
  declines.

- **The OpenVINO compile cache persists.** `ov_core_compile_model` was
  called with a property count of **zero**, so `ov::cache_dir` was never
  set and nothing was cached: every process restart recompiled all 504.
  This also sets the repository's first cache-path convention, for the
  QNN and CoreML caches to follow:

  | platform | directory |
  |---|---|
  | Windows | `%LOCALAPPDATA%\torchnative\Cache\openvino` |
  | macOS | `~/Library/Caches/torchnative/openvino` |
  | Linux | `$XDG_CACHE_HOME/torchnative/openvino` (default `~/.cache`) |
  | Android | `$HOME/.cache/torchnative/openvino` — app-private |
  | iOS | `~/Library/Caches/torchnative/openvino` — sandbox container |
  | wasm | none; no persistent filesystem |

  Windows is `%LOCALAPPDATA%` and never `%APPDATA%`: a driver- and
  device-keyed blob must not roam to a machine it was not compiled for.
  `TORCHNATIVE_CACHE_DIR` moves the root, `TORCHNATIVE_OPENVINO_CACHE_DIR`
  moves this backend only, and either disables caching with `0`, `off`,
  `none`, `false` or empty.

  **Not under the Hugging Face cache**, though the case for it was real:
  `huggingface_hub` owns and *prunes* that layout, and it is undefined
  for inputs that never came from the Hub. See [`../devices/NPUCACHE.md`](../devices/NPUCACHE.md).

- **`torch._C._shim_f16_bytes`** — a tensor to f16 bytes without building
  Python objects.

## 2. Defects fixed

- **A `MemoryError`, and then a Rust panic, on any real LLM.**
  `_NPULinear._weights_blob` did
  `pack_f16(weight.detach().flatten().tolist())`. Qwen3's `down_proj` is
  9728 × 2560 = 24,903,680 elements, so `.tolist()` built **24.9 million
  `PyFloat` objects — about 800 MB of CPython heap** to produce a 50 MB
  blob. Measured here on that exact shape: the new route produces the
  same 47.5 MB in 0.03 s with a Python heap peak of 47.5 MB, which *is*
  the returned bytes. At 2²⁰ elements, side by side: **24.0× the blob
  before, 1.00× now**, byte-identical output.

  The activation path had the same defect on **every call, both
  directions**; it now passes f16 bytes in and rebuilds through
  `torch.frombuffer`. 24.0× and 20.0× become 1.00×.

- **The shim panicked instead of raising.** When CPython could not
  allocate, `tolist` crossed the FFI boundary as a Rust panic rather than
  a `MemoryError`. `flat_objects` built scalars with `into_py_any`, which
  for `f64` reaches `PyFloat::new` → `PyFloat_FromDouble(val).assume_owned(py)`,
  and pyo3 documents `assume_owned` as *"panics on NULL"*. pyo3 offers no
  fallible `PyFloat::new`, so the float and int arms now call
  `Bound::from_owned_ptr_or_err` — pyo3's own fallible sibling, which
  fetches the `MemoryError` CPython already set. One panic point survives
  in `nest`'s `PyList::new` and is recorded rather than silenced.

- **252 `ov::Core` objects for one model.** Each leaf built its own —
  252 dlopens of the plugin registry, 252 device enumerations, 252 cache
  resolutions. Now one, shared. A leaf built alone by `from_torch` still
  makes its own, so sharing is an optimisation, not a requirement.

## 3. Measured but not implemented

- **Parallel compilation is refused, with reasons** —
  [`../devices/NPUPAR.md`](../devices/NPUPAR.md). Two of four preconditions are
  UNVERIFIED: whether `ov::Core::compile_model` is thread-safe (the word
  appears in no OpenVINO header, guide or API doc; openvino#27366 asks
  exactly this and was closed unanswered), and whether the driver-resident
  compiler serialises internally (the path ends in the closed NPU UMD).
  A third bounds the gain: `shim_f16_bytes` never calls `allow_threads`,
  so the conversion runs GIL-held. The fourth is a hazard —
  `FileStorageCacheManager::write_cache_entry` opens the final
  `<hash>.blob` directly, with no temp file, no rename and no
  cross-process guard, and running two scripts at once is normal.
  Shipping threads on that would risk a corrupted cache entry that later
  loads as a valid-looking compiled model. A tripwire test keeps the
  decision honest.

- **Still no NPU speed number, of any kind.** Nothing here was timed on
  the device. `252 → 1` is a count of constructions against a fake.

- **Utilisation will stay low, and that is structural.** Each `Linear` is
  an independent infer request, so one token is 252 separate NPU calls
  stitched together by Python. The per-call marshalling is fixed; the
  call structure is not.

- **`MAX_DIM = 2**17` is an unsourced constant**, copied from the
  archived `intel_npu_acceleration_library`, which gives no reason
  either. Qwen3-4B's `lm_head` exceeds it, so 252 leaves lower and one
  does not (`fraction_moved` 0.9033). [`../devices/NPUDIM.md`](../devices/NPUDIM.md)
  now records where it is *not* from: `131072` occurs nowhere in the
  OpenVINO NPU plugin, the NPU compiler, the Level Zero graph extension
  or the shipped NPU binaries, and the one per-dimension limit the
  compiler names is `VPU_DIMENSION_LIMIT = 8192`, which it tiles past
  rather than refusing. **The real ceiling is still unmeasured** — no
  dimension above 8192 has been compiled for `NPU` here — so the
  constant is unchanged and `tools/devices/intelnpu_dimsweep.py` is the
  experiment that would settle it.

- **Qualcomm and Apple remain refusals.** `torch.compile` remains a
  permanent one, for the structural reason in
  [`../graph/COMPILE.md`](../graph/COMPILE.md). And of the 297
  architectures that forward, **82** are still numerically unjudged —
  [`../architectures/ARCH100.md`](../architectures/ARCH100.md) measured
  reachability, and a forward is not a match.

## 4. Documentation corrected

New: [`../devices/NPUCACHE.md`](../devices/NPUCACHE.md), [`../devices/NPUPAR.md`](../devices/NPUPAR.md).
[`../devices/INTELNPU.md`](../devices/INTELNPU.md) gains the surviving panic point and the
`tolist` reasoning.

---

## 5. Gate at this head

```
1387 ok / 0 FAIL          (0.1.0b2 shipped at 1348)
DOCWATCH   1118 / 1118
cargo test 33 passed / 0 failed
TREE_UNCHANGED_DURING_GATE=yes
```

Two rounds in this release had a nullification come back **green** and
said so rather than moving on. One was a vacuous test — a storage-versus-view
check whose float32 fixture had already been materialised before Rust saw
it — and was widened until it failed. The other was dead code: a per-core
memo no caller could reach, which was deleted rather than tested.

## 6. Platform status

The standing table is [`RELEASE_0_1_0b0.md`](RELEASE_0_1_0b0.md) §6 and it stays as written: it is
that release's record and rewriting it would erase what was true then. This
section is where a reader of **0.1.0b3** finds out what has run, and it
supersedes b0 §6 wherever the two disagree. Everything below was measured
against the **published** artefacts in `dist/torchnative-0.1.0b3-*` — not a
fresh build — because the question is what a user gets.

Read the grades strictly. *builds* means the artefact exists and
`tools/wheel/verify_cross.py` accepts its tag and contents; that is a claim
about tags, binaries and symbol resolution and nothing else. *reaches* means
an interpreter **for that platform** unpacked the wheel into its own
site-packages, imported torch, and `torch.__file__` came back out of that
site-packages — the judgement every runtime harness in `tools/wheel/` makes,
and the one that stops a run from silently measuring some other torch.
*agrees* would mean outputs compared against a reference, and **no row below
earns it**: the runtime harnesses check `aten.mm.default` against a fixed
`[[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]` and `nn.Linear`'s shape and dtype, which
is arithmetic actually executing, not agreement with upstream torch.

| platform | grade | what ran, and what the harness judged |
|---|---|---|
| macOS arm64 | **reaches** | `verify.py` PASS. The published `macosx_11_0_arm64` wheel `pip install`ed into a throwaway venv (deps resolved from the network: filelock, fsspec, Jinja2, MarkupSafe, mpmath, networkx, setuptools, sympy, typing_extensions) and `torch.__file__` came out of that venv. 1299 `_C` names, 896 aten ops, `aten.mm.default` = 3×2 of 4.0 (float32) |
| iOS simulator arm64 | **reaches** | `verify_ios_sim.py` PASS — **first run of this release's wheel**; b0 §6's result was against a wheel built while the release was numbered `0.0.13a0`. iPhone 16 Pro, iOS 18.0 simulator, booted and shut down by the harness. `torch.__file__` under the scratch prefix's site-packages, `torch._C` = that tree's `_C.abi3.so`, 1299 names / 896 ops, `aten.mm.default` correct, `nn.Linear(4,3)` → `[2, 3] float32`, `sys.platform == 'ios'`, and no repository path on `sys.path`. `TORCH_USE_RTLD_GLOBAL` is **not** needed — the wheel's `torch/lib/libtorch_global_deps.so` satisfies `_load_global_deps()`. The `_multiprocessing` stub **is** still needed; the bare run dies in `torch/multiprocessing/__init__.py`, which is a property of the iOS CPython distribution, not of the wheel |
| Android arm64 | **reaches** | `verify_android.py` PASS — **first run of this release's wheel**. AVD `pmp_api26`, API 26, `arm64-v8a`, started for this run on port 5560 and killed after it; `/data/local/tmp/bw_wheel` removed. API 26 was chosen over `pmp_api36` deliberately: the wheel is tagged `android_21`, so the lowest available API is the one that tests the tag's own floor claim rather than a comfortable ceiling — `ro.build.version.sdk` was read off the device as `26` rather than assumed from the AVD name. Same judgement as iOS: `torch.__file__` under `/data/local/tmp/bw_wheel/lib/python3.13/site-packages`, 1299 names / 896 ops, `aten.mm.default` correct, `nn.Linear` → `[2, 3] float32`, `sys.platform == 'android'`. `TORCH_USE_RTLD_GLOBAL` not needed; `_multiprocessing` stub still needed |
| WASM (Pyodide) | **reaches** | `verify_wasm_browser.py` PASS — **the first time any Pyodide interpreter has imported a wheel from this project.** b0 §6's "nothing has imported *this* wheel under Pyodide" is now false for 0.1.0b3. `torch.__file__` = `/lib/python3.14/site-packages/torch/__init__.py`, `torch._C` = that tree's `_C.abi3.so`, 1299 names / 896 ops, `aten.mm.default` = 3×2 of 4.0 with `dtype == torch.float32`, `nn.Linear` → `[2, 3] float32`, `sys.platform == 'emscripten'`. The `_multiprocessing` stub is needed here too. See §6.1 for how, given that `node` is not installed |
| Linux x86_64 · Linux aarch64 | **builds** | `verify_linux.py` PASS on both published wheels. Symbol-level only, and the script is explicit about the ladder: ELF has no two-level namespace, so only *versioned* symbols name their library — glibc does, CPython does not, so the `Py*` imports are checked as a union against the target distribution's `libpython3.13.so`. Nothing executed: this machine is arm64 macOS and has no Linux userspace |
| Windows amd64 · Windows arm64 | **builds** | `verify_windows.py` PASS on both published wheels. Attribution here is *complete* — a PE import table names the DLL per symbol — but only the DLLs present on this machine (`python3.dll`, `vcruntime140*.dll` from the target distribution) can be checked to export what is asked of them. Nothing executed |
| iOS device arm64 | **builds** | `verify_ios_device.py` PASS. Still **never executed, on any release** — unchanged from b0 §6. No device is attached (`xcrun devicectl list devices`: *No devices found*), and the artefact cannot be run anywhere else: dyld rejects an `iOS` Mach-O in a simulator (*have 'iOS', need 'iOS-sim'*) and on macOS (*need 'macOS'*). What the script does prove without a device is the part that differs from the simulator wheel: the device extension binds its CPython symbols two-level to `Python.framework`, where the simulator's are flat `dynamic_lookup` |

### 6.1 How WASM was reached without `node`

Pyodide's non-browser runner is node, and this machine has none — `node`,
`npm`, `deno` and `bun` are all absent (checked 2026-09-13). That is not a
dead end, and the specific reason it is not is worth writing down, because
"WASM is unreachable here" was the standing answer:

* a bare wasm runtime (`wasmtime`, `wasmer`) cannot run Pyodide at all. A
  Pyodide build is Emscripten output — `pyodide.asm.wasm` plus JS glue — so a
  JavaScript host is not optional.
* macOS ships `jsc`, but it has none of the host functions the glue calls.
* macOS also ships **Safari**, and a browser is the environment Pyodide is
  primarily built for.

So `tools/wheel/verify_wasm_browser.py` stages the wheel on the host — reusing
`verify_android.py`'s `unpack` and `stage_dependencies` unchanged, so the
definition of "installed" cannot drift between the three runtime harnesses —
tars the staged tree, serves it over loopback with the local Pyodide
distribution, and opens Safari on a page that unpacks the tree into the Pyodide
filesystem's site-packages, runs the probe, and POSTs one JSON object back.
`safaridriver` is **not** used: enabling Safari's remote automation is an
interactive, administrator-authorised step, and none of this needs it.

Two things this arrangement does not carry, stated rather than papered over:

* the probe's two modes share **one** interpreter, where the device harnesses
  use two processes. The probe therefore clears `torch*`, `_multiprocessing`
  and `_posixshmem` out of `sys.modules` before each mode, or the second answer
  would be about the first run.
* the result crosses JSON through JavaScript, where `4.0` and `4` are the same
  number. `mm` alone would therefore compare equal against a float expectation
  it no longer is, so the dtype is carried separately and checked.
* the local distribution is `pyodide-core` 314.0.6, i.e. **CPython 3.14**,
  while the wheel is `cp313-abi3`. What ran is the abi3 forward-compatibility
  path — which is what a Pyodide user on this ABI gets, but is not the same as
  a cp313 Pyodide. The tag's `2026_0` half is still only checked against the
  distribution on this machine; no wasm module records it.

### 6.2 Reproducing each row

```bash
D=dist; P=/Volumes/macMini/caches/spike-venv/bin/python

# macOS arm64 -- reaches network for the dependency resolve
$P tools/wheel/verify.py $D/torchnative-0.1.0b3-cp313-abi3-macosx_11_0_arm64.whl

# iOS simulator -- boots and shuts down a simulator itself
$P tools/wheel/verify_ios_sim.py     $D/torchnative-0.1.0b3-cp313-abi3-ios_14_0_arm64_iphonesimulator.whl

# Android -- start the AVD yourself, on a port nothing else is using
~/Library/Android/sdk/emulator/emulator -avd pmp_api26 -port 5560     -no-window -no-audio -no-snapshot -gpu swiftshader_indirect &
ANDROID_SERIAL=emulator-5560 $P tools/wheel/verify_android.py     $D/torchnative-0.1.0b3-cp313-abi3-android_21_arm64_v8a.whl
adb -s emulator-5560 shell rm -rf /data/local/tmp/bw_wheel
adb -s emulator-5560 emu kill     # only because this command started it

# WASM -- opens a Safari tab; needs a logged-in GUI session
$P tools/wheel/verify_wasm_browser.py     $D/torchnative-0.1.0b3-cp313-abi3-pyemscripten_2026_0_wasm32.whl

# artefact-only, no interpreter for the target on this machine
$P tools/wheel/verify_linux.py   $D/torchnative-0.1.0b3-cp313-abi3-manylinux_2_17_x86_64.whl
$P tools/wheel/verify_linux.py   $D/torchnative-0.1.0b3-cp313-abi3-manylinux_2_17_aarch64.whl
$P tools/wheel/verify_windows.py $D/torchnative-0.1.0b3-cp313-abi3-win_amd64.whl
$P tools/wheel/verify_windows.py $D/torchnative-0.1.0b3-cp313-abi3-win_arm64.whl
$P tools/wheel/verify_ios_device.py     $D/torchnative-0.1.0b3-cp313-abi3-ios_12_0_arm64_iphoneos.whl
```

`verify_cross.py` was also run against all nine; it passes on eight and
refuses `macosx_11_0_arm64` by design — that is a host tag, and it says to use
`verify.py`, which is the row above.

### 6.3 What is still unreached, and the specific missing piece

| | missing piece |
|---|---|
| iOS device | a physical iPhone attached to this machine. Nothing else substitutes: dyld refuses the `iOS` Mach-O in both other places this repository can run code |
| Linux, either arch | a Linux userspace. No container runtime is installed here, and this host is arm64 macOS, so even `manylinux_2_17_aarch64` has nowhere to execute |
| Windows, either arch | a Windows machine or an emulated one. `wine` is not installed, and an arm64 macOS host cannot run `win_amd64` regardless |
| Android on real hardware | a device; `adb devices` was empty for this round. What ran was an emulator, which is the same ABI and API level but not the same silicon |
| WASM under a cp313 Pyodide | a Pyodide distribution built on CPython 3.13. The one here is 3.14, so the abi3 path is what was exercised |
| WASM's non-`Py*` `env` imports, and the abi3 binding | nothing records them in any artefact; `verify_cross.py` declines both by name and neither is closed by the run above — a successful import is evidence the imports resolved, but it does not enumerate them |

### 6.4 One defect found in the harnesses, not in the wheels

`verify_android.py`'s dependency staging demanded `pkg_resources` alongside
`setuptools` unconditionally. setuptools removed `pkg_resources` in 82 and the
spike venv now has 84, so the staging raised `SystemExit` before any device or
simulator was touched — and the message named the wheel's METADATA, so a
staging-source fact read as a wheel fact. It now stages `pkg_resources` when
the source has one. **No wheel was at fault**, and both device harnesses were
blocked by it, since `verify_ios_sim.py` imports the same function.

### 6.5 Not a platform claim

**An Intel NPU on Windows accepted a compiled model from this project and
reported `EXECUTION_DEVICES=['NPU']`.** One leaf, checked at compile time. Not
a generated token, not a benchmark. Carried forward from the note this section
replaced.
