# WASM — feasibility, layer by layer

Status: **complete for all four layers, and §8 closes the last one — a real downloaded Pyodide
interpreter actually loading this project's extension and running `import torch`.** §7 closed
layer 3 against a synthetic stand-in host this crate generated itself; §8 repeats every one of
its load-bearing checks against Pyodide 314.0.6 (real CPython 3.14.2) and goes further, reaching
`import torch` and a real `aten.mm`/`nn.Linear` forward pass. It was written as the investigation
ran, one entry per step, so that it would survive an interrupted session. Anything not answered
still says so explicitly.

**Read §8 before §7, and §7 before §2d/§3c/§5.** §8 supersedes §7 where they differ: §7's "the
extension loads" was proven against a hand-written stub host; §8 proves the same claim against a
real interpreter and adds `import torch` plus a real computation, which §7 explicitly left open
(§7.6). §7 in turn supersedes §2d/§3c/§5 the same way it always did: §2d said emscripten was not
attempted; it has now been built and run under Node. §3c's conclusion — that Emscripten voids
`abi3` — is **not** overturned by §7 or §8 and is restated in §7.7 with the measurement that
confirms it.

Question this document answers: the user names six supported platforms, but **WASM is
absent from the `docs/design/DESIGN.md` §722 matrix.** Is the matrix stale, or is WASM a
different kind of thing? This is a feasibility determination, **not an implementation.**

Vocabulary used throughout, kept strict:

| word | meaning |
|---|---|
| **works** | built/ran here, exit 0, output inspected |
| **blocked** | built/ran here, failed, and the failing thing is named |
| **not attempted** | no execution — reasoning or reading only |

## 0. Environment as found

- `rustc 1.98.0 (88d9e12ae 2026-08-18)`
- wasm targets **already installed**, nothing added:
  `wasm32-unknown-unknown`, `wasm32-wasip1`, `wasm32-unknown-emscripten`
- No `node`, `wasmtime` or `wasmer` **on `PATH`** — but §6a corrects this: a complete emsdk
  5.0.3 with `emcc` and a bundled Node 24 is already on this machine. Nothing was installed
  and the emsdk was not used; only inspected.
- `CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wasm2`
- All probing is done in a **separate experiment crate**, following the `rust/vk_probe`
  precedent. No dependency is added to the shipping crate.

## Layer 1 — does candle build for `wasm32`? **Yes, and better than expected.**

**Verdict: works.** The candle surface `rust/torch_c/src/` actually calls compiles for
both `wasm32-wasip1` and `wasm32-unknown-unknown`, including the entire quantised path.
One qualification, in §1d: SIMD is broken upstream, so it compiles *scalar-only*.

### 1a. The probe

`rust/wasm_probe/` — a separate crate, **not** a dependency of `rust/torch_c` and not in a
workspace with it, following the `rust/vk_probe` precedent for exactly the stated reason:
putting a wasm target's constraints on the shipping crate before the question is answered
risks the three platforms that currently build, for nothing.

Its `src/lib.rs` names every `candle_core` item the shipping crate imports. The list was
not invented — it is the output of

```
grep -rhoE "use candle_core::\{?[^;]*" rust/torch_c/src/
```

which is:

```
CpuStorage  DType  Layout  Shape  Tensor  Device  Module
quantized::{GgmlDType, QMatMul, QStorage, QTensor}
Error::{Msg, MatMulUnexpectedStriding, WithBacktrace}
```

plus the twelve `GgmlDType` variants `quant.rs` enumerates and the `half::{f16, bf16}`
constructors `reduced.rs` uses. `candle-core` is pinned to the same version and the same
`default-features = false` as `rust/torch_c/Cargo.toml:36`.

It is a library and not a `#[test]`, deliberately: **there is no wasm runtime on this
machine** (§0), so the only question answerable here is compilation. Every reference is
written so that a missing item is a *compile* error.

**Control:** the probe builds for the host (`aarch64-apple-darwin`) first, exit 0. Without
that, "it compiles for wasm32" would not distinguish a working target from a probe that
had quietly stopped referring to anything.

### 1b. Results — measured

`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wasm2`, `rustc 1.98.0`.

| target | command | result |
|---|---|---|
| `aarch64-apple-darwin` (control) | `cargo build --release` | **exit 0**, 0 errors |
| `wasm32-wasip1` | `cargo build --release --target wasm32-wasip1` | **exit 0**, 0 errors, `libwasm_probe.rlib` produced |
| `wasm32-unknown-unknown` | bare | **exit 101** — `getrandom`, see §1c |
| `wasm32-unknown-unknown` | with the §1c pin | **exit 0**, 0 errors |
| `wasm32-wasip1` **+ `-C target-feature=+simd128`** | | **exit 101, 38 errors** — see §1d |

The prompt asked specifically whether `Tensor`, `DType`, matmul and `QTensor` survive.
**All four survive, on both wasm targets.** `QTensor::quantize`, `QTensor::dequantize`,
`QTensor::data`, `QStorage::from_data`, `QMatMul::from_arc`, `QMatMul::forward` and all
twelve `GgmlDType` variants compile. The GGUF/k-quant machinery is *not* behind the
`wasm32` gate.

### 1c. What `wasm32` drops — the `tokenizers` problem solves itself

`docs/design/CANDLE_DEPS.md` is **still accurate**; re-verified against the same
`candle-core 0.11.0` source. `Cargo.toml.orig:37` still reads

```
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
tokenizers = { workspace = true, features = ["onig"] }
```

and `src/quantized/mod.rs:17` still gates `pub mod tokenizer;` on
`#[cfg(not(target_arch = "wasm32"))]`. Neither has a feature gate; both key on the target
architecture alone.

The consequence is that **on wasm32 the entire `tokenizers`/`onig` subtree drops out for
free** — no patch, no fork, no `[patch.crates-io]`. Measured with `cargo tree` on the probe's
own graph:

| target | crates |
|---|---|
| `aarch64-apple-darwin` | 129 |
| `wasm32-wasip1` | **80** (−49) |
| `wasm32-unknown-unknown` | **84** (−45) |

The 49 that vanish are exactly the subtree `docs/design/CANDLE_DEPS.md` §3a costed: `tokenizers`,
`onig`, `onig_sys`, `regex`/`regex-automata`/`regex-syntax`, `compact_str`,
`derive_builder`(+core/macro), `darling`(+core/macro), `esaxx-rs`, `spm_precompiled`,
`unicode-normalization-alignments`, `monostate`, `cc`, `pkg-config`, `find-msvc-tools` and
friends. (49 here versus that document's −44 is not a discrepancy: this probe's graph is
not `torch_c`'s — it has no PyO3 and names `half` directly.)

**So CANDLE_DEPS.md §6's last row — "WASM target: irrelevant, candle already filters it" —
is correct as a statement about relevance but understates the direction.** On wasm32 the
dependency problem that document spends 400 lines on does not exist. If WASM ever became a
target, it is the one platform that needs none of §8's vendored patch.

`wasm32-unknown-unknown` needs four crates `wasip1` does not, and they name the environment:
`wasm-bindgen`(+macro/support/shared) and `bumpalo`. That is the browser. The cause is
`getrandom`, which refuses `wasm32-unknown-unknown` outright:

```
error: The wasm32-unknown-unknown targets are not supported by default; you may need to
enable the "wasm_js" configuration flag.
```

Getting past it needs **both** `--cfg getrandom_backend="wasm_js"` in `RUSTFLAGS` **and** an
explicit `getrandom` dependency with `features = ["wasm_js"]` — the cfg alone gives a second,
different error. Two major versions sit in candle's graph simultaneously (`rand` 0.8 pulls
`getrandom` 0.3, `rand` 0.9 pulls 0.4), so both must be pinned. `rust/wasm_probe/Cargo.toml`
carries that block with a comment saying it is a finding and not a wanted dependency.

**This is a real, if small, structural fact: `wasm32-unknown-unknown` forces a browser
dependency (Web Crypto via `js-sys`) into the build, and `wasm32-wasip1` needs nothing.**
Anything server- or CLI-side should target `wasip1`.

### 1d. The one thing that is blocked — WASM SIMD does not compile

This is the load-bearing negative result of layer 1.

candle **has** hand-written WASM SIMD kernels: `src/cpu/simd128.rs` and
`src/quantized/simd128.rs`, both `use core::arch::wasm32::*`. `k_quants.rs` dispatches to
them at six sites (`vec_dot_q4_0_q8_0`, `q8_0_q8_0`, `q2k_q8k`, `q4k_q8k`, `q6k_q8k`,
`q8k_q8k`). All of it is gated on `#[cfg(target_feature = "simd128")]`, which is **off by
default** on both wasm targets.

Turning it on does not compile:

```
RUSTFLAGS='-C target-feature=+simd128' cargo build --release --target wasm32-wasip1
  -> exit 101, 38 errors, all E0433, all in candle-core-0.11.0/src/cpu/mod.rs
     19x cannot find type `CurrentCpuF16` in this scope
     19x cannot find type `CurrentCpuBF16` in this scope
```

The cause is an upstream inconsistency, confirmed by reading the three CPU backends:

| module | defines |
|---|---|
| `src/cpu/avx.rs` | `CurrentCpu`, `CurrentCpuF16`, `CurrentCpuBF16` |
| `src/cpu/neon.rs` | `CurrentCpu`, `CurrentCpuF16`, `CurrentCpuBF16` |
| `src/cpu/simd128.rs` | **`CurrentCpu` only** |

and `src/cpu/mod.rs` gates the reduced-precision helpers (`vec_add_f16` and the `bf16`
equivalents) on `#[cfg(any(target_feature = "neon", target_feature = "avx2",
target_feature = "simd128"))]` — a three-way list — while the scalar fallbacks beneath them
are gated on `#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]`, a
**two-way** list that forgets `simd128`. So enabling `simd128` selects the SIMD branch of a
helper whose SIMD types were never written for this backend.

Named precisely: **`simd128` is a maintained-in-name-only backend in candle 0.11.0. Nobody
compiles it with `+simd128`, because it does not build.**

What this costs, concretely: WASM would run candle's **scalar** kernels. The
comparison is not "WASM is a bit slower" but "WASM gets none of the vectorised
quantised dot products that `docs/perf/PERF.md`/`docs/perf/PERF_ANDROID.md` treat as the
baseline on NEON". It is a fixable bug — the fix is to write the two missing types in
`simd128.rs`, or to add `simd128` to the fallback's `not(any(...))` list, which is a
one-line change that would at least make `+simd128` build and pick up the six quantised
kernels. **Not attempted here**: this is a feasibility study and the fix belongs upstream
or in a vendored patch, neither of which is this task's area.

### 1e. Also survives, but unverified at runtime

`rayon`, `memmap2` and `num_cpus` all *compile* for `wasm32-wasip1` — they are in the 80.
candle uses `rayon` in `cpu_backend/mod.rs`, `sort.rs`, `conv2d.rs` and `utils.rs`, and
`utils.rs:317` calls `num_cpus::get_physical()` to size its thread pool. **Whether any of
that behaves at runtime is not attempted** — wasm32 without the threads proposal has one
thread, and `std::thread::Builder::new().spawn()` on `wasip1` is a runtime error rather than
a compile error. This is the most likely place for a "compiles but does not run" surprise,
and it cannot be checked without a runtime (§ execution requirements).

## Layer 2 — does PyO3 work on `wasm32`? **`abi3-py313` + `extension-module` builds and links.**

**Verdict: works, for `wasm32-wasip1`, with one added linker flag.** The result is a real
extension module in shape. Whether anything can *load* it is layer 3, and that is where the
answer turns.

### 2a. Getting the control right first

The first attempt at this layer produced a **false negative that would have ended the
investigation**, and it is worth recording because it is the §5.5 failure mode: a check that
cannot pass is not a check.

- The probe was an `rlib`. `cargo build --features pyo3-route --target wasm32-wasip1` gave
  **exit 0** — but an `rlib` build never invokes the linker, so that exit 0 said nothing.
- Changing it to `cdylib` (which is what `rust/torch_c/Cargo.toml:10` is) made **the host
  build fail too**, with undefined `_Py*` symbols. Had that not been checked, "wasm fails to
  link" would have been reported as a wasm finding when it was a miswired probe.

The cause: `rust/torch_c/.cargo/config.toml` supplies `-C link-arg=-undefined -C
link-arg=dynamic_lookup` on Apple targets, and the probe had no such file.
`rust/wasm_probe/.cargo/config.toml` now mirrors it. **Control: host `cdylib` with
`extension-module` + `abi3-py313` links, exit 0, 1.5 MB `libwasm_probe.dylib`.**

### 2b. Results — measured

| target | crate type | flags | result |
|---|---|---|---|
| `aarch64-apple-darwin` (control) | cdylib | `-undefined dynamic_lookup` | **exit 0**, 1,502,176 B dylib |
| `wasm32-wasip1` | cdylib | none | **exit 101** — `rust-lld: undefined symbol: _Py_DecRef`, `_Py_IncRef`, `PyErr_GetRaisedException`, `PyList_Type`, `PyType_GetFlags`, … |
| `wasm32-wasip1` | cdylib | `-C link-arg=--allow-undefined` | **exit 0**, 1,073,660 B `wasm_probe.wasm` |

PyO3's build script raised **no objection at all** to being cross-compiled to wasm with
`abi3-py313`. That is the abi3 forward-compatibility path: with `extension-module` on, no
`libpython` is linked and no interpreter needs to be found at build time, so there is
nothing for the build script to fail on. The failure, when it came, was purely at link.

`--allow-undefined` tells `wasm-ld` to turn unresolved symbols into **module imports**
rather than errors. That is the wasm analogue of the note already in
`rust/torch_c/.cargo/config.toml`: *"Android needs no extra flags: ELF shared libraries may
carry undefined symbols and the interpreter resolves them at load time."* wasm can express
the same thing; it just will not do it by default.

**Second false-positive caught.** The first `--allow-undefined` build produced a 128 KB
`.wasm`, which is far too small to contain candle. `--gc-sections` had discarded all of it,
because nothing reachable from `PyInit_wasm_probe` called the layer-1 functions. The probe
now exposes `probe_all()`, which calls every one of them, and the artefact is **1.07 MB** —
candle and PyO3 in one wasm module together.

### 2c. The artefact has the right shape — verified, not assumed

The `.wasm` was parsed directly (import and export sections, `/tmp/wimp.py`, a ~30-line
reader written for this):

```
total imports: 50
  Py*/_Py* imports: 45      all from module "env"
  wasi_snapshot_preview1:  5   environ_get environ_sizes_get fd_write proc_exit sched_yield
exports containing PyInit: ['PyInit_wasm_probe']
```

This is exactly the shape of a CPython extension module: it **exports `PyInit_<name>`** and
**imports the 45 `Py*` symbols it needs from its host**. Nothing about `abi3-py313` +
`extension-module` is rejected by the wasm target.

**But read the import list again, because it is the whole of layer 3.** Those 45 symbols are
imports of a *core wasm module*, resolved by whoever instantiates it. For this to load into
a Python interpreter, something has to play the role `dlopen` plays on Linux and Android —
and `wasm32-wasip1` core modules have no dynamic linking. See §3.

### 2d. Emscripten — **not attempted**, and the reason is not technical

`wasm32-unknown-emscripten` is the target PyO3 actually supports on purpose:
`pyo3-build-config-0.29.2/src/lib.rs:73` special-cases it to emit `-sSIDE_MODULE=2
-sWASM_BIGINT` for rustc < 1.95, and `:304` exempts it from the "wasm has no rpath" rule.
It is the Pyodide target. The rust target is installed here.

The toolchain is present too — `/Volumes/macMini/caches/emsdk` is a complete emsdk **5.0.3**
with `upstream/emscripten/emcc` and a bundled `node 24.19.0_64bit`.

**It was not used.** The brief said to verify that directory and not touch it, and
verification turned up a specific reason to obey: `find` shows **22 files under it modified
today**, and it holds a live `upstream/emscripten/cache/cache.lock`. It is in active use by
another project. `emcc` writes to that cache on every invocation, so building through it
risks another workstream's build for a datapoint that layer 3 makes secondary anyway.

So: **emscripten is "not attempted", not "does not work".** What is known about it is read
from PyO3's source, not measured here.

## Layer 3 — what shape does CPython take on WASM? **This is where it splits in two.**

**Verdict: `wasm32-wasip1` is blocked. `wasm32-unknown-emscripten` is open but costs the
project's central ABI decision.** Everything in this section is read from CPython's own
policy documents and PEPs, **not measured here** — there is no target CPython for either
wasm platform on this machine.

The layers above quietly assumed one thing: that the `.wasm` produced in §2c, which exports
`PyInit_wasm_probe` and imports 45 `Py*` symbols, can be *loaded by an interpreter*. That is
the same assumption `rust/torch_c/.cargo/config.toml` states for Android — *"ELF shared
libraries may carry undefined symbols and the interpreter resolves them at load time."*
**Whether that sentence has a wasm translation is the whole of layer 3, and the answer
differs between the two wasm platforms.**

### 3a. The two are not variants of one platform

| | `wasm32-wasip1` | `wasm32-unknown-emscripten` |
|---|---|---|
| CPython support tier | **tier 2** (PEP 11) | **tier 3** (PEP 776, from Python 3.14) |
| what it is | server/CLI, POSIX-ish capability sandbox | browser / Node — Pyodide, PyScript, JupyterLite |
| `dlopen`/`dlsym` | **absent** | **provided by Emscripten** |
| our extension shape (`_C.abi3.so` loaded at import) | **cannot work** | works — Pyodide loads side modules this way |
| PyO3 support | incidental | **explicit** (`pyo3-build-config` special-cases the triple) |
| wheel platform tag | **none exists** | `pyemscripten_*_wasm32` (PEP 783) |
| threads | n/a here | **forbidden** — PEP 783: *"libraries cannot use `-pthread`"* |

Note the inversion: **WASI has the higher CPython support tier and is the one that cannot
take our artefact.** Tier is about whether CPython itself builds and passes tests, not about
whether third-party extension modules can be loaded. Reading the tier alone would give the
wrong answer.

### 3b. WASI — blocked, and the blocker has a name

The blocker is **`dlopen` does not exist in WASI preview 1**. It is not a gap in CPython;
it is absent from the platform. CPython's own WASI build has to skip building the test
modules that need it (`_testimportmultiple`, `_testmultiphase`, `_testsinglephase`,
`xxlimited`, `xxlimited_35`), and the documented approach for extension modules on WASI is
to compile them **statically into the interpreter** as builtins.

This is exactly the sentence the brief asked for — *what does our embedding shape become
there*. The answer:

> On WASI, `torch._C` cannot be a wheel. It would have to be a **CPython fork built with
> `torch._C` as a builtin module**, shipped as a whole interpreter binary.

That is a different product from the one `tools/wheel/build.py` makes. Note also that the
§2c artefact *already showed this*: it imports its `Py*` symbols from a module literally
named `env`, meaning the instantiating host must supply all 45 — which is not something a
Python interpreter does for a module it imports.

`--enable-wasm-dynamic-linking` exists in CPython's configure, but it is the Emscripten
path, not a WASI one.

### 3c. Emscripten — open, but it invalidates `abi3-py313`

Emscripten implements `dlopen`/`dlsym`, which is why it is *the* wasm target that supports
native extension modules, why Pyodide works, and why PyO3 special-cases it. Our shape
survives.

**The cost lands squarely on this project's most load-bearing decision.**
`rust/torch_c/Cargo.toml:13-23` spends ten lines justifying `abi3-py313`, and `docs/design/ABI3.md`
§7 recommends it, on the grounds that one artefact loads into many interpreter versions and
that a version-pinned `.so` is a silent failure mode. **On Emscripten that argument does not
hold**:

- Emscripten **makes no ABI stability guarantee between its own versions** (PEP 783 says so
  outright). The `pyemscripten` platform therefore pins a specific Emscripten compiler
  version, the set of statically linked libraries, and specific linker flags.
- Pyodide's own position, from its maintainers' discussion, is that abi3 *works* but is
  *useless* until the underlying ABI is stabilised, and that building with the limited API
  should be disabled for Pyodide.

So on Emscripten the wheel is pinned to a Python feature release **and** an Emscripten
version anyway. abi3 buys nothing, and the one thing it does buy elsewhere — a single
artefact across interpreter versions — is unavailable. **This does not block WASM. It means
WASM would be the one platform where the project's ABI strategy does not apply**, and that
is a design fact worth knowing before anyone commits.

### 3d. `-pthread` is forbidden — and layer 1 §1e is where that bites

PEP 783: *"libraries cannot use `-pthread`."* §1e recorded that `rayon`, `num_cpus` and
candle's thread-pool code in `utils.rs` all compile for wasm. **They compile; they cannot be
used as intended.** `utils.rs:317` sizes a pool from `num_cpus::get_physical()` and
`utils.rs:127` spawns with `std::thread::Builder`.

Combined with §1d — SIMD does not compile — the honest performance statement for WASM is:
**scalar kernels, single thread.** That is not "somewhat slower than Android"; it is the
slowest configuration candle has. No number is offered here because none was measured.

## Layer 4 — is there a distribution path? **Yes, for Emscripten. None for WASI.**

The brief's framing — *"Pyodide uses its own index; can it go to PyPI?"* — was true until
recently and is **now out of date.**

### 4a. PEP 783 exists and PyPI accepts the tag

**PEP 783 (Emscripten Packaging)** defines the platform tag series

```
pyemscripten_${YEAR}_${PATCH}_wasm32        e.g. pyemscripten_2026_0_wasm32   (Python 3.14)
```

and states that package indexes **SHOULD accept any wheel whose platform tag matches
`pyemscripten_[0-9]+_[0-9]+_wasm32`**. PyPI supports these uploads; packages built for
Pyodide can be published to PyPI directly and installed at runtime, rather than living only
in Pyodide's own index. The tag replaced the earlier `pyodide_${YEAR}_${PATCH}_wasm32`, and
before that the form was `emscripten_3_1_45_wasm32`-style, versioned on the compiler.

`maturin` already emits the tag. Our wheel builder does not — `tools/wheel/build.py` has
`AndroidTarget` (PEP 738, `android_<api>_<abi>`), `IOSTarget` (`ios_<major>_<minor>_...`) and
the macOS path, and nothing for wasm. Adding a `PyEmscriptenTarget` is the same shape of work
as the two that exist. **That file is another workstream's area and was not touched.**

So layer 4 is the *easiest* of the four, and it is the one the original question expected to
be hardest.

### 4b. WASI has no tag at all

There is no platform tag for `wasm32-wasi` in any PEP, which is consistent with §3b: a
platform that cannot load extension-module wheels has no need of a tag for them.

## Recommendation on `DESIGN.md` §722

**Recommendation: do not add WASM to the §722 matrix. Add a WASM row to the README platform
table instead, and reconcile the two lists inside `DESIGN.md` that currently disagree.**

### 5a. The matrix is not stale — but `DESIGN.md` contradicts itself, and that is the real finding

The brief asked whether the matrix is stale or WASM is a different kind of thing. It is
neither, exactly. **The six platforms and the five in the matrix come from two different
lists in the same document.**

- `DESIGN.md` §861 (§10, repository layout) says the build tool `pypackpack` targets
  **Android · iOS · macOS · Linux · Windows · WASM** — that is the six.
- `DESIGN.md` §722 lists **five** platforms across six rows (Android appears twice, CPU and
  GPU). No WASM.

They are not the same kind of table. §722 is a ***`kernels` backend* matrix** — its columns
are "which `kernels` backend", "which actual kernel", "resolved when". It answers where
optimised kernels come from. §861 is a **build-target list**.

**On its own terms, §722 is correct to omit WASM.** `kernels`' backend enumeration is
`cpu` · `cuda` · `metal` · `rocm` · `xpu` (+`cann`, `neuron`). It has no `wasm` backend — the
same gap §722 already documents at length for `vulkan` on Android GPU. And per §1d/§3d there
would be no vectorised kernel to point the entry at even if a backend existed. A WASM row
in §722 would read `cpu` / scalar / build-time, which is what "no entry" already means.

**What is genuinely wrong is that §861 promises a platform the rest of the document never
mentions again.** That is the mismatch the user noticed. The fix is not to add a row to
§722; it is to say in §861, or beside it, that WASM is a pypackpack capability that
torchnative has not adopted, with a pointer here.

### 5b. What to do with the README

`README.md:264` currently says WASM's absence is *"an open question rather than an
oversight"* and gives two reasons: candle excludes `wasm32` from parts of itself, and CPython
on WASM is a different embedding shape. **This investigation resolves both, and one of them
was wrong:**

| README's reason | status after this investigation |
|---|---|
| "candle excludes `wasm32` from parts of itself" | **Not a blocker — it is a benefit.** What candle excludes is `quantized::tokenizer`, which `torch_c` never used, and excluding it drops 49 crates for free (§1c). Everything `torch_c` calls survives, `QTensor` included (§1b). |
| "CPython on WASM is a different embedding shape" | **Correct, and sharper than stated.** It is two different shapes: WASI cannot load our artefact at all (§3b), Emscripten can but voids `abi3-py313` (§3c). |

Suggested replacement rows for the README platform table, using its own legend
(⚠️ = built, never executed):

```
| in the target matrix | ... | ❌ *not listed* — see docs/platform/WASM.md, correctly so |
| rust target installed | ... | ✅ (wasip1, unknown-unknown, emscripten all present) |
| target CPython        | ... | 🔲 |
| extension builds      | ... | ⚠️ wasip1 only; emscripten not attempted |
```

The row `can be run *here*` should stay ❌ but its footnote at `README.md:206` needs a
correction: it lists `node` among the runtimes this machine lacks, and that is **no longer
true** — see below.

### 5c. If someone does pursue it, the order is fixed

1. **Emscripten, not WASI.** WASI needs a CPython fork with `torch._C` as a builtin; that is
   a different product.
2. **Fix candle's `simd128` first** (§1d), or accept scalar kernels. This is a small upstream
   patch and it gates whether the result is worth shipping.
3. **Drop `abi3` for that target only** (§3c) and expect a wheel per Pyodide release.
4. **Then** add a `PyEmscriptenTarget` to `tools/wheel/build.py` (§4a).

Steps 1-3 are all cheaper than step 4 is misleading: the packaging works, which makes it
tempting to start there.

## What would be needed to actually verify by execution

Nothing below was run. This section exists so the gap is a list rather than an adjective.

### 6a. What is actually on this machine — one correction

`node`, `wasmtime` and `wasmer` are **not** on `PATH`; confirmed. But the claim in
`README.md:206` that this machine has no `node` is **wrong**, and the same would be said of
this document if it were not checked:

```
/Volumes/macMini/caches/emsdk/node/24.19.0_64bit/bin/node      exists
/Volumes/macMini/caches/emsdk/upstream/emscripten/emcc         exists (emsdk 5.0.3)
/Volumes/macMini/caches/emsdk/python/3.13.3_64bit/bin/python3  exists
```

**A full Emscripten toolchain and a Node 24 are already here.** Nothing was installed and
nothing was used: the brief said to verify that directory and not touch it, and verification
gave an independent reason to obey — 22 files under it were modified today and it holds a
live `upstream/emscripten/cache/cache.lock`, so another workstream is using it. `emcc` writes
to that cache on every run.

So the honest status of layer 2's emscripten half is **"not attempted for a scheduling
reason, not a capability reason."** If that directory is free, or `EM_CACHE` is redirected to
a scratch path, `cargo build --target wasm32-unknown-emscripten --features pyo3-route` is
runnable **today**.

### 6b. To execute, in increasing order of cost

| to answer | what is needed | have it? |
|---|---|---|
| does the wasip1 `.wasm` instantiate | `wasmtime` or `wasmer` | **no** — and §3b says the answer would be "no host supplies those 45 imports", so this is low value |
| does the emscripten build link | `emcc` + `EM_CACHE` pointed away from the shared emsdk | **yes**, gated on not disturbing another workstream |
| does `import torch` work in Pyodide | a Pyodide distribution matching the Emscripten version, plus Node (present) | **no Pyodide** — it is a download, not a build |
| do the golden tests pass on WASM | Pyodide + `numpy`/`torch` reference wheels for `pyemscripten`, and a harness that does not assume a local CPython | **no** — `tools/golden/compare.py` runs against a host interpreter |
| is it fast enough to matter | all of the above **plus** the `simd128` fix (§1d) | **no** — and without §1d the measurement would only restate "scalar is slow" |

The cheapest meaningful next step is the second row, and it needs no installation.

### 6c. Reproducing what *was* run

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wasm2
cd rust/wasm_probe

# layer 1 -- candle only
cargo build --release                                          # host control, exit 0
cargo build --release --target wasm32-wasip1                    # exit 0
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
  cargo build --release --target wasm32-unknown-unknown         # exit 0
RUSTFLAGS='-C target-feature=+simd128' \
  cargo build --release --target wasm32-wasip1                  # exit 101, 38 errors (§1d)

# layer 2 -- with PyO3
cargo build --release --features pyo3-route                     # host control, exit 0
RUSTFLAGS='-C link-arg=--allow-undefined' \
  cargo build --release --features pyo3-route --target wasm32-wasip1   # exit 0, 1.07 MB

# crate-count comparison (§1c)
cargo tree --target <triple> --prefix none | grep -oE '^[a-z0-9_-]+ v[0-9.]+' | sort -u | wc -l
```

`rust/wasm_probe/` is an investigation crate. It is **not** a dependency of `rust/torch_c`
and shares no workspace with it; `rust/torch_c/` and `tools/wheel/` were not modified.

## Summary table

**This table is the first-pass snapshot and is superseded by §7.10 and §8.4 — see those for what
was actually executed.** Rows 2d and 3(Emscripten) below were written before any Emscripten
execution happened; §8.4 in particular reaches `import torch` and a real forward pass, which
this table still describes as unattempted.

| layer | question | verdict |
|---|---|---|
| 1 | candle on `wasm32` | **works** — `Tensor` `DType` matmul `QTensor` all survive, both targets; 49 fewer crates |
| 1d | candle WASM SIMD | **blocked** — `+simd128` fails to build, `simd128.rs` lacks `CurrentCpuF16`/`CurrentCpuBF16` |
| 2 | PyO3 `abi3-py313` + `extension-module` | **works** on wasip1 with `--allow-undefined`; exports `PyInit_*`, imports 45 `Py*` |
| 2d | same, emscripten | **not attempted** — shared emsdk in use by another workstream |
| 3 | CPython on WASI | **blocked** — no `dlopen`; would need `torch._C` as a CPython builtin |
| 3 | CPython on Emscripten | **open**, but `abi3-py313` buys nothing there; no `-pthread` |
| 4 | distribution | **works** — PEP 783 `pyemscripten_*_wasm32`, accepted by PyPI |
| — | `DESIGN.md` §722 | **leave WASM out** — it is a `kernels` backend matrix and there is no wasm backend. Fix the §861/§722 contradiction instead |

Sources for the layer 3/4 claims, none of which were measured here:
[PEP 776](https://peps.python.org/pep-0776/) ·
[PEP 783](https://peps.python.org/pep-0783/) ·
[Pyodide PyEmscripten ABI](https://pyodide.org/en/stable/development/abi.html) ·
[abi3-on-Pyodide discussion](https://github.com/pyodide/pyodide/discussions/4377) ·
[CPython WASI dlopen issue](https://github.com/python/cpython/issues/115983) ·
[tier promotion](https://discuss.python.org/t/wasm32-emscripten-and-wasm32-wasi-have-been-promoted-to-tier-3-platforms-for-cpython/17590)

---

# 7. Emscripten, actually executed

Everything in §1–§6 above was a *compile* result. §2d and §6a left one thing open for a
scheduling reason: `emcc` was never run. This section runs it. **New vocabulary rule for this
section: "runs" means a `.wasm` was executed by Node on this machine and its stdout was read.**

## 7.0 Setup — what was used, and what was protected

```sh
emsdk   /Volumes/macMini/caches/emsdk                                    (5.0.3, shared)
emcc    /Volumes/macMini/caches/emsdk/upstream/emscripten/emcc
node    /Volumes/macMini/caches/emsdk/node/24.19.0_64bit/bin/node        v24.19.0
export EM_CACHE=/Volumes/macMini/caches/emcc-scratch      # NOT the shared emsdk cache
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-emcc
```

The shared emsdk is redirected away from with `EM_CACHE`, which is the whole of §6a's concern:
`emcc` writes its sysroot and port cache on every invocation, and that is the only part of the
emsdk it writes to. Baseline before starting: **24431 files under the emsdk, 22 modified in the
last 24h** — the same 22 §2d saw. The count is re-checked at the end of this section.

`emcc` did populate the scratch cache on first use, as expected:

```
cache:INFO: generating system headers: sysroot_install.stamp...
   (this will be cached in "/Volumes/macMini/caches/emcc-scratch/sysroot_install.stamp")
```

Nothing was installed. `wasm32-unknown-emscripten` was already an installed rustc target (§0).

## 7.1 Layer 1 on Emscripten — candle compiles, and the crate graph is identical to WASI

| target | command | result |
|---|---|---|
| `wasm32-unknown-emscripten` | `cargo build --release` | **exit 0**, 0 errors |

Crate counts, re-measured on this branch (numbers differ by +2 from §1c because the lockfile
has moved since; the *comparison* is what matters):

| target | crates |
|---|---|
| `aarch64-apple-darwin` | 131 |
| `wasm32-wasip1` | 82 |
| `wasm32-unknown-emscripten` | **82** |

`comm` on the two sorted crate lists is **empty in both directions**: the emscripten and wasip1
dependency graphs are *the same set of 82 crates*. So §1c's finding — the `tokenizers`/`onig`
subtree drops out for free on wasm32 — holds on Emscripten too, and Emscripten costs nothing
extra. It also does **not** drag in the `wasm-bindgen`/`bumpalo` browser subtree that
`wasm32-unknown-unknown` forces (§1c); the `getrandom` `wasm_js` pin in the probe's
`Cargo.toml` is inert here.

**One thing this table does not say, and §5.5 requires saying it.** The layer-1 `cdylib` link
also exits 0, and it produces a **65-byte `.wasm`**:

```
0061 736d 01000000  000f 08 "dylink.0" ... 0715 01 11 "__wasm_call_ctors"
```

That is a side module containing one empty function. `--gc-sections` discarded all of candle
because, with no PyO3 feature on, nothing is reachable from an export. **A layer-1 emscripten
`cdylib` exit 0 proves nothing on its own** — it is the same false positive §2b caught on
wasip1, in a new place. The load-bearing artefact is §7.2's, and it is 857 KB.

Note also what the 65-byte header already tells us: rustc's `wasm32-unknown-emscripten` cdylib
link emits a **`dylink.0` section**, i.e. `-sSIDE_MODULE`. That section is the difference from
wasip1 in one word, and §7.3 is about what it buys.

## 7.2 Layer 2 on Emscripten — `abi3-py313` + `extension-module` links with **no extra flag**

```sh
cargo build --release --features pyo3-route --target wasm32-unknown-emscripten
```

**exit 0. 857,161-byte `wasm_probe.wasm`.** Candle and PyO3 in one module — the size is the
control that §2b established: 128 KB would mean candle had been stripped.

**This is the sharpest difference from WASI, and it is not a matter of degree.**

| | `wasm32-wasip1` | `wasm32-unknown-emscripten` |
|---|---|---|
| link with no extra flags | **exit 101** — `rust-lld: undefined symbol: _Py_DecRef`, … | **exit 0** |
| what makes it link | `-C link-arg=--allow-undefined`, added by hand | nothing — `dylink.0`/`SIDE_MODULE` is the target default |
| unresolved `Py*` become | imports of module `"env"` on a **core** module | imports of `"env"` **and `GOT.mem`** on a **side** module |

The second row is the answer to "how does Emscripten resolve CPython symbols — same as
wasip1's `--allow-undefined` or not?" **Not the same.** On wasip1 the flag is a way of telling
`wasm-ld` to stop complaining, and it produces a core module whose imports must be supplied by
whoever *instantiates* it — which, as §3b says, a Python interpreter never does for a module it
imports. On Emscripten the same unresolved symbols are recorded in a `dylink.0` section, which
is a **defined interchange format for dynamic linking**: the loader is expected to resolve them
against an already-loaded main module. That is `dlopen`'s job, and it is the thing WASI has no
equivalent of.

### 7.2a The artefact's shape, read from the binary

Disassembled with the emsdk's own `wasm-dis` (binaryen) and counted:

```
sections:  dylink.0  TYPE  IMPORT  FUNCTION  GLOBAL  EXPORT  START  ELEM  DATACOUNT  CODE  DATA
exports:   PyInit_wasm_probe            (llvm-nm: 00003735 T PyInit_wasm_probe)
imports:   198 total   =  88 "env"  +  70 "GOT.func"  +  40 "GOT.mem"
```

Of those, **54 are CPython**: 45 functions from `"env"` and **9 data symbols from `"GOT.mem"`**:

```
GOT.mem:  PyExc_AttributeError  PyExc_BaseException  PyExc_SystemError  PyExc_TypeError
          PyList_Type  PyModule_Type  PyTuple_Type  PyType_Type  PyUnicode_Type
```

The `GOT.mem` half is new information relative to §2c, which counted 45 and stopped. Those nine
are **data**, not functions — type objects and exception singletons — and on wasip1 they were
folded into the same undifferentiated `"env"` import list. Emscripten separates them because
data relocation is a distinct problem from function relocation, and it is one more reason the
two platforms are not variants of each other.

### 7.2b The build imports **20 pthread symbols** — §1e/§3d, now measured rather than predicted

The non-CPython `"env"` imports include:

```
pthread_create  pthread_detach  pthread_attr_init  pthread_attr_setstacksize
pthread_mutex_{init,lock,trylock,unlock,destroy}  pthread_mutexattr_{init,settype,destroy}
pthread_cond_{init,wait,signal,broadcast,destroy}  pthread_condattr_{init,setclock,destroy}
sched_yield  sysconf
```

§1e said "`rayon`, `memmap2` and `num_cpus` compile; whether they *behave* is not attempted",
and §3d said PEP 783 forbids `-pthread`. **This import list is the two statements meeting.**
The artefact was built **without** `-pthread`, and it still names `pthread_create`, because
candle's thread-pool code is linked in and only the *linker* knows it is unreachable at
runtime. Emscripten without `-pthread` supplies a `pthread_create` that fails rather than one
that spawns. So the shape is: **it links, it loads, and any code path that actually reaches
`pthread_create` fails at runtime rather than at build time.** That is the "compiles but does
not run" surprise §1e predicted, and it is now located precisely.

## 7.3 Layer 3, part one — **candle executes on Emscripten under Node**

This is the result the first pass could not reach at all, and it is the one thing about WASM
that nobody in this project knew: **our layer-1 code runs.**

`rust/wasm_probe/src/main.rs` is a new `[[bin]]` in the same probe crate. It is deliberately
*not* an "ok / not ok" harness — it prints computed **values** and compares each to a number
worked out by hand, because an exit code alone cannot tell "candle ran" apart from "the runtime
started and the code had been gc'd away". That is the same false positive as §2b and §7.1, and
it is the specific way a runtime probe lies.

**Control first**, on the host, since a probe that passes everywhere proves nothing:

```
$ cargo run --release --bin wasm_probe                       # aarch64-apple-darwin
target_arch=aarch64 target_os=macos
tensor  sum(ones(2,3)) = 6  expect 6  PASS
matmul  dims=[2, 4] sum=48  expect [2,4] 48  PASS
reduced f16sum=24 bf16sum=24  expect 24 24  PASS
q4_0    bytes=1152 maxerr=0 matmulsum=511.96875  expect 1152 ~0 512  PASS
simd128 enabled = false
== failures = 0 ==                                            exit 0
```

Then the same source, compiled to `wasm32-unknown-emscripten` (998,294-byte `.wasm` plus a
55,653-byte `wasm_probe.js` loader, both emitted by rustc via `emcc`) and executed by the
emsdk's bundled Node:

```
$ node wasm_probe.js                                          # node v24.19.0
target_arch=wasm32 target_os=emscripten
tensor  sum(ones(2,3)) = 6  expect 6  PASS
matmul  dims=[2, 4] sum=48  expect [2,4] 48  PASS
reduced f16sum=24 bf16sum=24  expect 24 24  PASS
q4_0    bytes=1152 maxerr=0 matmulsum=511.96875  expect 1152 ~0 512  PASS
simd128 enabled = false
== failures = 0 ==                                            exit 0
```

**Verdict: runs.** Read the two blocks side by side — they are *identical below the header
line*, including the last digit of the quantised matmul.

What that costs to say precisely, item by item:

| checked | host | emscripten/node |
|---|---|---|
| `Tensor::zeros` + broadcast add + `sum_all` | 6 | 6 |
| `matmul`, shape and value | `[2,4]`, 48 | `[2,4]`, 48 |
| `f16` and `bf16` round trip through `to_dtype` | 24, 24 | 24, 24 |
| `QTensor::quantize(Q4_0)` block packing | 1152 B | 1152 B |
| `QTensor::dequantize` max abs error | 0 | 0 |
| `QMatMul::forward` | **511.96875** | **511.96875** |
| `cfg!(target_feature = "simd128")` | false | **false** |

The Q4_0 row is the load-bearing one. 2048 elements at 32 weights per 18-byte block is 64
blocks is 1152 bytes — the *block layout* is identical, so this is not "some quantiser ran",
it is candle's GGML-compatible one. And `511.96875` rather than 512 is the quantisation error
itself: it is the *same wrong answer* on both platforms, which is a much stronger statement
than agreement on a round number would have been.

**The `simd128 = false` row is the honest half.** §1d predicted this and it is now confirmed at
runtime rather than inferred: what executed above is candle's **scalar** kernel path. §1d's
upstream bug (`simd128.rs` lacks `CurrentCpuF16`/`CurrentCpuBF16`) is unchanged and was not
fixed here.

**No timing is reported, and that is deliberate.** Between §1d (scalar only) and §3d/§7.2b
(single-threaded, `-pthread` forbidden by PEP 783) this is the slowest configuration candle
has. A number here would be read as "WASM performance" when it is "WASM performance with both
accelerators off", and the interesting question — what it costs *after* the simd128 fix — is
not answerable on this machine.

## 7.4 Layer 3, part two — **`dlopen` works on our actual `cdylib`**

§3a's whole argument turns on one row of its table: WASI has no `dlopen`, Emscripten does. That
row was **read from CPython and Pyodide policy documents, not measured** — §3 says so. It is
also the row that decides whether `torch._C` can ever be a wheel on WASM, so it is worth more
than a citation.

`rust/wasm_probe/dlopen_host.c` is a 60-line C program: `dlopen` a path, `dlsym`
`wasm_probe_run`, call it, compare the returned bitfield against 31. It loads the
**candle-only** side module, not the PyO3 one, on purpose — that separates "does the dynamic
loader work" from "can 54 CPython symbols be resolved" (§7.5), so a failure can only be blamed
on one of them.

`wasm_probe_run` is a new `#[no_mangle] pub extern "C"` in the probe's `lib.rs`. It exists
because of §7.1: without an export reachable from outside, `--gc-sections` reduces the
emscripten cdylib to 65 bytes. With it the candle-only side module is **865,573 bytes**.

**Host control**, `cc dlopen_host.c` against `libwasm_probe.dylib`:

```
dlopen  PASS (handle=0x745d8a40)
dlsym   PASS (wasm_probe_run=0x106afb6a0)
call    wasm_probe_run() = 31  expect 31  PASS      exit 0
```

**Emscripten**, `emcc -sMAIN_MODULE=1`, side module loaded from disk via `-sNODERAWFS=1`, run
under Node 24:

```
dlopen  PASS (handle=0x7fab0)
dlsym   PASS (wasm_probe_run=0x14aa)
call    wasm_probe_run() = 31  expect 31  PASS      exit 0
```

**Verdict: runs.** Same bitfield, 31, on both — all five candle checks including the quantised
`QMatMul::forward`, executing from inside a module that was loaded **at runtime, by name, out of
a file**, sharing the main module's linear memory and heap. That is the exact mechanism CPython
uses to import a native extension, and it is exactly what §3b says WASI cannot do.

**This is the single most load-bearing new fact in this document.** Everything §3c and §5c say
about "Emscripten is the one to pursue" rested on a citation; it now rests on an execution.

### 7.4a The first attempt failed, and the failure is a real constraint

The `-sMAIN_MODULE=1` build linked fine and then died at load:

```
dlopen FAIL: could not load dynamic lib: side_nopyo3.wasm
LinkError: WebAssembly.Instance(): Import #136 "env" "__cpp_exception":
           tag import requires a WebAssembly.Tag
```

rustc's `wasm32-unknown-emscripten` side module uses **native wasm exception handling**, so it
imports `__cpp_exception` as a wasm **tag**. A main module compiled with emcc's default
exception mode exports no such tag, and the two cannot be linked. Adding `-fwasm-exceptions` to
the *main* module fixed it, and that is the only change between the failing and passing runs
above.

Recorded because it is not a quirk of this probe: **the main module and every side module must
agree on the exception ABI, and rustc picks native wasm EH for you.** Any real host — a Pyodide
distribution, a custom CPython build — has to have been built with `-fwasm-exceptions` for a
Rust extension to load into it. It is one more thing pinned by the platform, alongside the
Emscripten version and linker flags PEP 783 already pins (§3c), and it belongs on that list.

## 7.5 Layer 3, part three — **the PyO3 extension loads, and `PyInit_` runs**

§7.4 loaded the candle-only module. This loads the real shape: the `cdylib` built with
`abi3-py313` + `extension-module`, 893,771 bytes, exporting **both** `PyInit_wasm_probe` and
`wasm_probe_run`, and importing the 54 CPython symbols of §7.2a.

There is no CPython for this target on this machine, so the host supplies those 54 itself.
`rust/wasm_probe/gen_pystubs.py` generates them **from the side module's own import table**
(`wasm-dis` output), not from CPython headers — there are none to take, and guessing is not an
option: a WebAssembly import whose type does not match the exporting module is a `LinkError` at
*instantiation*, so an arity wrong by one gives "will not load" rather than "misbehaves".
45 function stubs + 9 `char[512]` data symbols, one special case: `PyModuleDef_Init` returns
its argument, because PyO3's multi-phase init is literally `return PyModuleDef_Init(&MODULE_DEF)`
and that is the smallest thing that lets `PyInit_` run to completion.

`rust/wasm_probe/pyinit_host.c`, `emcc -fwasm-exceptions -sMAIN_MODULE=1`, Node 24:

```
loading: side_pyo3.wasm
dlopen  PASS -- module instantiated
candle  wasm_probe_run() = 31  expect 31  PASS
dlsym   PASS PyInit_wasm_probe=0x14fc
PyInit  returned 0x160900, 1 stub call(s), last=PyModuleDef_Init
PyInit  PASS -- PyModuleDef.m_name == "wasm_probe" at +20
== failures = 0 ==                                              exit 0
```

**Verdict: runs.** Four things, each worth naming separately:

1. **The extension instantiated** with all 54 CPython imports bound to a host module — including
   the nine `GOT.mem` *data* symbols, which are the half wasip1 has no mechanism for.
2. **candle still returns 31 from inside the PyO3 build.** PyO3's presence did not change the
   numerical result or let `--gc-sections` take anything.
3. **`PyInit_wasm_probe` was found by `dlsym` and executed.** It made exactly **one** call into
   the host, and that call was `PyModuleDef_Init` — which independently confirms PyO3 0.29 uses
   *multi-phase* module initialisation, since single-phase would have called `PyModule_Create2`
   and a dozen others.
4. **It returned the right module definition.** The returned pointer was dereferenced from the
   *main* module and its `m_name` reads `"wasm_probe"`, at offset **+20** — which also measures
   `sizeof(PyModuleDef_Base)` on wasm32 as 20 bytes (8-byte `PyObject` header + `m_init` +
   `m_index` + `m_copy`, all 4-byte). That a pointer stored in the side module's data segment is
   dereferenceable from the main module is the proof that **data relocation into the shared
   linear memory worked**, not just function relocation.

This is as far as it is possible to go without a target CPython, and it is further than §3
assumed was reachable.

### 7.5a Three negative controls — including one that made this section weaker

§5.5 of CLAUDE.md: a check that cannot fail is not a check. Three were run.

| control | change | result |
|---|---|---|
| **A** | `PyModuleDef_Init` stub returns `0` instead of its argument | `PyInit returned 0` → **FAIL, exit 1** |
| **B** | delete `PyType_IsSubtype` from the host entirely | `dlopen` **still succeeds**, run still exits 0 |
| **C** | delete `PyModuleDef_Init` — the one symbol `PyInit_` calls | loads, then **aborts** on the call |

**A** proves the `PyInit PASS` line is not vacuous. **C** proves resolution is real for symbols
that are actually called.

**B is the one that matters, because it falsified something this section originally claimed.**
The first draft of `pyinit_host.c` printed *"all 54 CPython imports resolved against this
host"*. **That was false.** Emscripten's dynamic loader does not refuse a side module with an
unresolvable import — it substitutes a stub that aborts if reached, exactly as **C** then
demonstrated:

```
thread '<unnamed>' panicked at .../panicking.rs: panic in a function that cannot unwind
Aborted()   RuntimeError: unreachable
```

So **on Emscripten, "the extension loaded" does not mean "the interpreter has every symbol it
needs"; a missing one surfaces as a runtime abort at first use.** The probe's output was
corrected to say so.

That is not a detail — it is the same failure mode as the 20 `pthread` imports in §7.2b, and it
generalises into a deployment property: **an ABI mismatch between a Rust extension and its
Emscripten CPython host presents as an abort deep in a call, not as an import error.** On Linux
or Android the loader rejects the `.so` up front. This is a *worse* diagnostic than the
"silent failure mode" `docs/design/ABI3.md` §7 warns about for version-pinned `.so` files, and it is
one more argument on the same side as §3c/§7.6.

## 7.6 Layer 4 — what a real verdict still needs, and how close it actually is

§7.5 is the ceiling without a target CPython. The remaining question is what it would take to
run `import torch` for real, and the answer turned out to be **much closer than §6b estimated**.

`docs/platform/WASM.md` §6b listed "a Pyodide distribution matching the Emscripten version" as a blocker
and left it at that, as if the matching were the hard part. It is not. Pyodide's own
`Makefile.envs` on `main`:

```
export PYODIDE_EMSCRIPTEN_VERSION ?= 5.0.3
export PYVERSION                  ?= 3.14.2
export PYODIDE_ABI_VERSION        ?= 2026_0
```

**The emsdk on this machine is 5.0.3 — the exact version current Pyodide pins.** And
`docs/development/abi/314.md` requires *"Rust version 1.93.0 or later"*; this machine has 1.98.0
(§0). The toolchain half of the gap is already closed, by coincidence rather than by anyone's
plan.

What is missing is only the artefact:

| to answer | needed | status |
|---|---|---|
| does the extension load into a real CPython | a built Pyodide distribution (`pyodide-core`, a download, not a build) | **absent** — nothing Pyodide-shaped anywhere on this machine |
| do the golden tests pass | the above **plus** `numpy`/reference wheels for `pyemscripten_2026_0`, and a harness not assuming a host interpreter | absent; `tools/golden/compare.py` runs against a host CPython |
| is it worth shipping | all of the above **plus** the §1d `simd128` fix | absent, and §1d is upstream |

**The next step is a download and is not attempted here.** Fetching and unpacking a Pyodide
distribution is outside the immediate request (CLAUDE.md §5.7) and the brief said to install
nothing. It is recorded as the one remaining step rather than done.

Two things that would otherwise be found the hard way, if someone does take that step:

- **`-fwasm-exceptions` is mandatory, at compile *and* link time**, for the main module as well
  as the side module. §7.4a discovered this empirically — the first `dlopen` died with
  `tag import requires a WebAssembly.Tag` — and Pyodide's ABI spec lists "the stack unwinding
  ABI" as one of the five things a PyEmscripten platform pins. The empirical finding and the
  spec agree.
- **`-sSUPPORT_LONGJMP=wasm` is also part of the platform**, and the probe's host did *not* pass
  it. It got away with that only because nothing in the side module uses `setjmp`. A build that
  does would fail; the emcc invocations in §7.0 are a probe, not a conforming build recipe.

## 7.7 What this does **not** change: `abi3` is still void on Emscripten

§3c concluded that Emscripten invalidates `abi3-py313` on the strength of a PEP 783 sentence
and a Pyodide discussion thread. **Nothing in §7 overturns that, and the platform's own
documentation now makes it sharper than §3c stated.** This is the part not to over-read:
`abi3-py313` *built* (§7.2), *linked* (§7.2), *loaded* (§7.5) and its `PyInit_` *ran* (§7.5).
None of that buys what `abi3` exists to buy.

From Pyodide's ABI documentation:

> The Emscripten compiler makes no ABI stability guarantees between versions, and several
> linker flags can adjust the ABI. […] Pyodide adopts a **new PyEmscripten platform for each
> feature release of Python.**

And the platforms, read off their own specifications:

| platform tag | CPython | **Emscripten** |
|---|---|---|
| `pyemscripten_2025_0` | 3.13 | **4.0.9** |
| `pyemscripten_2026_0` | 3.14 | **5.0.3** |
| `pyemscripten_2026_5` | 3.15 | **6.0.5** |

**Read the right-hand column.** Three CPython feature releases, three *different compilers*. The
entire premise of `abi3` — `rust/torch_c/Cargo.toml:13-23` spends ten lines on it, and
`docs/design/ABI3.md` §7 recommends it — is that one artefact serves many interpreter versions. On
Emscripten, supporting 3.13, 3.14 and 3.15 means **building three times with three different
Emscripten toolchains**, and the `abi3` tag changes nothing about that count. The limited API
would still be *used*; it would simply buy nothing.

Restated as a deployment consequence, since that is what the question is for:

> **On every other platform this project ships, "one binary per platform" is achieved by
> `abi3`. On WASM it is not achievable at all.** WASM would be one binary per *PyEmscripten
> platform*, i.e. one per CPython feature release, each built with the Emscripten compiler that
> platform names. That is the same cardinality as dropping `abi3` everywhere, and it is a
> different distribution model from the other five platforms — not a variation on it.

Three further consequences, each measured above rather than argued:

1. **Mismatches abort, they do not fail to load** (§7.5a). A side module whose host lacks a
   symbol still `dlopen`s, and aborts at first use. So the wrong-platform wheel is not caught
   by the loader. That is *worse* than the version-pinned-`.so` failure mode `docs/design/ABI3.md` §7
   already calls silent.
2. **Scalar, single-threaded, and that is structural.** §1d (candle's `simd128` does not
   compile) and §7.2b/§3d (PEP 783 and Pyodide both forbid `-pthread`; the artefact imports 20
   `pthread` symbols that can only abort) are not tuning problems.
3. **The ABI pins the unwinding mode too** (§7.4a, §7.6), so an extension is tied to its host's
   exception ABI on top of everything else.

**None of this says WASM is impossible — §7.3/§7.4/§7.5 say the opposite, by execution.** It
says WASM is a platform whose distribution model this project does not currently have, and that
the cost is a wheel per CPython feature release rather than the one-artefact story the other
five platforms get.

## 7.8 Corrections to earlier sections of this document

| section | said | now |
|---|---|---|
| header | "complete for all four layers" | true only for *compilation*; nothing had been executed |
| §0, §1a, §6a | "there is no wasm runtime on this machine" | **wrong** — the emsdk's Node 24 is a runtime, and §7.3 used it |
| §2d, §6a | emscripten "not attempted", shared emsdk in use | **attempted and passed**; `EM_CACHE` redirection was sufficient (§7.9) |
| §3a table, row `dlopen` | "provided by Emscripten" (read from PEPs) | **measured** — §7.4, on our own `cdylib` |
| §3a table, row "our extension shape" | "works — Pyodide loads side modules this way" (inferred) | **measured** — §7.5, `PyInit_` runs and returns the right `PyModuleDef` |
| §1e | rayon/threads "compiles but unverified at runtime" | **located**: 20 `pthread` imports present in the artefact, unusable at runtime (§7.2b) |
| §6b row 2 | "does the emscripten build link — gated on not disturbing another workstream" | **done**, exit 0, and the emsdk was not disturbed (§7.9) |
| §6b row 3 | "no Pyodide — it is a download, not a build" | still true, but the toolchain gap is **closed**: Pyodide pins emscripten 5.0.3, which is what is here (§7.6) |
| §5b README rows | "extension builds: ⚠️ wasip1 only; emscripten not attempted" | should read **✅ emscripten builds, loads and runs under Node**; `can be run here` flips from ❌ to ✅ |

## 7.9 Reproducing §7, and the emsdk was left as found

```sh
export PATH="/Volumes/macMini/caches/emsdk/upstream/emscripten:\
/Volumes/macMini/caches/emsdk/node/24.19.0_64bit/bin:$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-emcc
export EM_CACHE=/Volumes/macMini/caches/emcc-scratch          # not the shared emsdk cache
cd rust/wasm_probe

# 7.1 / 7.2 -- compile and link
cargo build --release --target wasm32-unknown-emscripten                        # exit 0 (65 B .wasm!)
cargo build --release --features pyo3-route --target wasm32-unknown-emscripten  # exit 0, 857 KB

# 7.3 -- candle executes
cargo run   --release --bin wasm_probe                                          # host control
cargo build --release --bin wasm_probe --target wasm32-unknown-emscripten
node $CARGO_TARGET_DIR/wasm32-unknown-emscripten/release/wasm_probe.js          # exit 0

# 7.4 -- dlopen a side module
cargo build --release --lib --target wasm32-unknown-emscripten
cc dlopen_host.c -o host_dl && ./host_dl $CARGO_TARGET_DIR/release/libwasm_probe.dylib   # control
emcc dlopen_host.c -O1 -fwasm-exceptions -sMAIN_MODULE=1 -sALLOW_MEMORY_GROWTH=1 \
     -sNODERAWFS=1 -o dl.js
node dl.js wasm_probe.wasm                                                      # exit 0
#   NB: without -fwasm-exceptions this fails at load, see 7.4a

# 7.5 -- the PyO3 module, with generated CPython stubs
cargo build --release --lib --features pyo3-route --target wasm32-unknown-emscripten
wasm-dis wasm_probe.wasm -o side.wat && python3 gen_pystubs.py side.wat > pystubs_gen.h
emcc pyinit_host.c -O1 -fwasm-exceptions -sMAIN_MODULE=1 -sALLOW_MEMORY_GROWTH=1 \
     -sNODERAWFS=1 -o py.js
node py.js wasm_probe.wasm                                                      # exit 0
```

New files, all inside `rust/wasm_probe/`: `src/main.rs`, `dlopen_host.c`, `pyinit_host.c`,
`gen_pystubs.py`, plus a `wasm_probe_run` export added to `src/lib.rs`. `rust/torch_c/` and
`tools/wheel/` were not touched.

**The shared emsdk.** §2d declined to run `emcc` because 22 files under
`/Volumes/macMini/caches/emsdk` had been modified that day and a `cache.lock` was present.
`EM_CACHE=/Volumes/macMini/caches/emcc-scratch` was sufficient: every cache write `emcc` made
went there (`sysroot_install.stamp`, `libc.a`, `libc++-noexcept.a`, `libcompiler_rt.a`,
`pic/*`, …). Rechecked afterwards:

| | before | after |
|---|---|---|
| files under `/Volumes/macMini/caches/emsdk` | 24431 | **24431** |
| of those, modified in the last 24 h | 22 | **22** |
| modified in the last 70 min (i.e. by this session) | — | **0** |
| files under `EM_CACHE=/…/emcc-scratch` | 0 | 1605 (55 MB) |

The 55 MB of system libraries `emcc` generated all landed in the scratch directory; the shared
emsdk was not written to at all. **§2d's caution was right and its conclusion was too strong**:
the correct response to a shared toolchain is to redirect its cache, not to skip the experiment.

### 7.9a Regressions, re-run after these changes

The changes are confined to `rust/wasm_probe/` and this file, but the shipping crate's suites
were re-run anyway, because "I only touched X" is a claim and not a check:

| check | result |
|---|---|
| `PYTHON=$PY sh rust/torch_c/pytests/run.sh` | **exit 0** — 197 ok, 0 not ok |
| `$PY tools/golden/compare.py` | **exit 0** — 2811/2811 cases, 0 failed, ops covered = 119 |

## 7.10 Summary of §7 — the four layers, executed

| layer | question | first pass | §7 |
|---|---|---|---|
| 1 | candle on emscripten | not attempted | **works** — exit 0, same 82 crates as wasip1, identical set |
| 1 | candle *runs* | impossible, "no runtime" | **runs** — Node 24, all 5 checks, `511.96875` identical to host |
| 1d | wasm SIMD | blocked upstream | **unchanged** — confirmed off at runtime (`simd128 = false`) |
| 2 | PyO3 `abi3` + `extension-module` link | wasip1 only, needs `--allow-undefined` | **works on emscripten with no extra flag** — side module, 857 KB |
| 3 | `dlopen` our `cdylib` | read from PEPs | **runs** — dlopen + dlsym + call, bitfield 31 = host |
| 3 | `PyInit_` on the real extension | read from PEPs | **runs** — returns `PyModuleDef` with `m_name == "wasm_probe"` |
| 3 | `import torch` in a real interpreter | not attempted | **still not attempted** — needs a Pyodide download (§7.6) |
| 4 | distribution | PEP 783 works | **works, and costs `abi3`** — one wheel per CPython release, per §7.7 |

**The headline, stated so it cannot be over-read:** *our extension shape builds, loads and
executes on Emscripten under Node — and doing so costs the one-binary-per-platform property
that `abi3` gives this project everywhere else.* Both halves are measured. Neither cancels the
other.

---

# 8. A real CPython, at last

§7 closed layer 3 against a host this crate itself synthesised (`gen_pystubs.py`'s stub table).
§7.6 named the remaining gap precisely: *"does the extension load into a real CPython"* — needs
a downloaded Pyodide distribution, and that download was out of scope for that session
(CLAUDE.md §5.7). This section does the download and answers the question it was blocking.

**New vocabulary for this section, stricter than §7's:** "real interpreter" means Pyodide's own
`pyodide.asm.wasm`/`pyodide.asm.mjs`, loaded through its own JS loader, running our `.wasm` as a
side module through Pyodide's actual import machinery — not `pyinit_host.c`, not
`gen_pystubs.py`. Nothing here is a stand-in host.

## 8.0 Pyodide obtained, and the CPython-3.13 question answered

The brief asked for Pyodide matching CPython 3.13 specifically, because `rust/torch_c` pins
`abi3-py313`. **It exists, and it does not pair with the toolchain on this machine:**

| Pyodide series | CPython | Emscripten |
|---|---|---|
| `0.28.x` (e.g. `0.28.3`) | **3.13.2** | **4.0.9** |
| `314.x` (e.g. `314.0.6`) | 3.14.2 | **5.0.3** |

The emsdk at `/Volumes/macMini/caches/emsdk` — the only one this session is permitted to use
(brief: no modification, no installer, no version change) — is **5.0.3**. That is not a
coincidence with `314.x`: §7.6 already found "the emsdk on this machine is 5.0.3 — the exact
version current Pyodide pins" and read it from Pyodide's `Makefile.envs` on `main`. This session
confirms it against a tagged release rather than `main`, and confirms the corollary the brief
asked for: **the CPython-3.13-pinned Pyodide release needs Emscripten 4.0.9, which is not on
this machine and installing a second emsdk is outside this task's permitted actions ("use the
existing emsdk").** So the honest statement is:

> **A Pyodide matching this project's `abi3-py313` floor exists (`0.28.x`, CPython 3.13.2), but
> is not reachable with the toolchain this task was scoped to. The only Pyodide buildable here
> is `314.x`, CPython 3.14.2.** `abi3-py313` is a *floor*, not a pin — 3.14 is forward of 3.13 —
> so this does not by itself invalidate the exercise, but it means this section verifies against
> 3.14.2, not 3.13, and that substitution is a scope constraint, not a preference.

Fetched: `pyodide-core-314.0.6.tar.bz2` (6.7 MB) from the GitHub release, into
`/Volumes/macMini/caches/pyodide/pyodide/`. `disk-avail` before: 115 GB; the download and
extract cost under 30 MB. Not the full `pyodide-314.0.6.tar.bz2` (350 MB, includes the whole
package index) — `pyodide-core` is the runtime alone, which is all loading one hand-built
extension needs.

Version read from the interpreter itself, not the tarball name, using the emsdk's own Node
(`node --experimental-wasm-stack-switching`, required by Pyodide's own `python` launcher for
Node 20–24) and Pyodide's `python` CLI entry point:

```
$ ./python -c "import sys; print(sys.version)"
3.14.2 (main, Aug 25 2026, 05:50:55) [Clang 23.0.0git ...]
$ ./python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
.cpython-314-wasm32-emscripten.so
$ ./python -c "import importlib.machinery as m; print(m.EXTENSION_SUFFIXES)"
['.cpython-314-wasm32-emscripten.so', '.abi3.so', '.so']
```

**Rung 1: done.** `.abi3.so` is a recognised extension suffix on this interpreter — confirming
from the interpreter's own import machinery, not inference, that an `abi3` artefact is even
eligible to be found by `import`.

## 8.1 Rung 2 — built against this interpreter, with a named caveat

`rust/wasm_probe` was rebuilt exactly as §7.9 reproduces it — `cargo build --release --lib
--features pyo3-route --target wasm32-unknown-emscripten`, same emsdk, same `EM_CACHE`
redirection — **exit 0, 893,771-byte `wasm_probe.wasm`**, byte-identical in size to §7.5's
artefact. That number is not incidental: it means this session's rebuild reproduced §7 exactly
before adding anything new.

**The caveat the brief asked for:** this was *not* built against CPython 3.14's headers, because
`abi3-py313` + `extension-module` never consults target headers at all (§2a: "no `libpython` is
linked and no interpreter needs to be found at build time"). The limited-API surface named in
the artefact's import table is fixed by the `abi3-py313` choice, not by which CPython is
installed where the build runs. So "built against THAT interpreter's ABI" is true only in the
sense that follows from abi3 working as designed: the same artefact that linked against no
interpreter now has to be *loaded* by 3.14 for real, which is exactly what §7.7 already said
`abi3` does not exempt this platform from doing per-Emscripten-version anyway. Rung 2 is real,
but it is a statement about the limited API, not about 3.14 headers.

## 8.2 Rung 3 — loaded by the real interpreter, and proven unstubbed

`rust/wasm_probe/src/lib.rs`'s `pyo3_route::probe_all()` was already exposed for exactly this.
Loaded into Pyodide 314.0.6 via its own `pyodide.mjs` loader (`loadPyodide()`), the `.wasm`
written into Pyodide's virtual FS as `/probe_modules/wasm_probe.abi3.so`, `sys.path` extended,
then plain `import wasm_probe`:

```
=== sys.version ===
3.14.2 (main, Aug 25 2026, 05:50:55) [Clang 23.0.0git ...]
=== attempting import ===
MODULE OBJECT: <module 'wasm_probe' from '/probe_modules/wasm_probe.abi3.so'>
MODULE FILE: /probe_modules/wasm_probe.abi3.so
IMPORT_OK
=== calling probe_all() ===
probe_all() -> 101  (expect: match host)
```

**"It imported" is exactly the evidence the brief said not to trust.** So `probe_all()` was
compared against the host control, built the same way as §7.3 controls everything:

```
$ python3 -c "import sys; sys.path.insert(0,'/tmp/wasm4_hostmod'); import wasm_probe; \
              print(wasm_probe.probe_all())"
101
```

**Identical: 101 on `aarch64-apple-darwin`, 101 inside Pyodide 3.14.2.** This is a stronger
control than §7.3's, and worth being precise about why. `probe_all()` chains five Tensor
operations, a `QMatMul::forward`, and marshals the result through PyO3's `PyResult<usize>` —
which means it round-trips through `Py_BuildValue`-equivalent int construction, exception
machinery, and reference counting *supplied by Pyodide's own CPython*, not a hand-generated stub
table. A wrong or leaked reference count would not necessarily change `101`, but a genuinely
missing or mis-typed symbol among the 45 functions / 9 data symbols (§7.2a) would abort before
returning anything — see §8.2a for the demonstration that this is not a vacuous claim.

**Rung 3: done, against a real interpreter, not the §7 stand-in.**

### 8.2a The aborting-stub control, repeated against the real host

§7.5a ran three negative controls against `pyinit_host.c` — a host this crate wrote. The brief
is explicit that Emscripten's aborting-stub substitution is the reason "it loaded" is not
evidence, so that control needed repeating against Pyodide itself, not just cited from §7.

Added `rust/wasm_probe/src/lib.rs::pyo3_route::probe_bogus_symbol()`, gated behind a new
`bogus-symbol-test` feature (off by default — this artefact is never meant to be loaded by
anything else). It declares one `extern "C"` import, `Wasm4Probe_DoesNotExistInAnyCPython`, a
name that cannot collide with any real CPython symbol past or future, and calls it. Built with
`--features pyo3-route,bogus-symbol-test`:

```
$ cargo build --release --lib --features pyo3-route,bogus-symbol-test \
    --target wasm32-unknown-emscripten     # exit 0, 894,592 B
```

**It still links.** Then, against real Pyodide 3.14.2:

```
=== import (should SUCCEED despite unresolved symbol) ===
IMPORT_OK
=== probe_all() -- real functionality still works ===
probe_all() -> 101 (expect 101, matching host)
=== probe_bogus_symbol() -- should ABORT at first use ===
thread '<unnamed>' panicked ... panic in a function that cannot unwind
Aborted()   RuntimeError: unreachable
    at wasm://.../pyodide.asm.mjs ...
ABORTED_AS_EXPECTED
```

**Three facts, in one run, against the production interpreter rather than a stand-in:**

1. A module with a genuinely unresolvable import still **imports successfully** — Pyodide's
   dynamic loader substitutes an aborting stub for it, exactly as §7.5a found against the
   synthetic host, now confirmed on the real one.
2. **Real, resolved symbols keep working in the same module** — `probe_all()` still returns 101.
   Import succeeding is not "everything is stubbed"; it is "only the unresolvable one is."
3. **The unresolved symbol aborts the whole runtime at first use** — not a Python exception, a
   `RuntimeError: unreachable` that Pyodide itself reports as a *fatal error*, unrecoverable
   within that instance. This is worse than an `ImportError`, and it is the same shape §7.5a
   already named: on Emscripten, an ABI mismatch surfaces as a crash deep in a call, not a
   load-time rejection.

**This is the proof the brief asked for.** `IMPORT_OK` alone, from §8.2, would have been exactly
the false signal the brief warned about. Paired with this control, `probe_all() == 101` is now
known to mean the 45 functions and 9 data symbols `probe_all()` actually touches were genuinely
resolved against Pyodide's own CPython — not that the loader waved everything through.

## 8.3 Rung 4 and rung 5 — `rust/torch_c` builds for this target, and a real forward pass ran

The first draft of this section, written before actually running the command, guessed that
`rust/torch_c`'s larger dependency graph would fail to cross-compile and stopped there. **That
guess was wrong and is corrected here rather than left.** The command was then run for real:

```sh
cd rust/torch_c
CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-emcc-torchc \
  cargo build --release --target wasm32-unknown-emscripten
```

**exit 0. `_C.wasm`, 3,261,949 bytes**, no edits to `rust/torch_c` — its `aten.rs`, `bootstrap.py`
and `overloads.json` (this task's forbidden files, owned by other agents this session) were read,
not touched. Disassembled the same way as §7.2a/§8.2a: exports `PyInit__C`; imports 157 `env` +
96 `GOT.func` + 57 `GOT.mem`, of which 94 are `Py*`/`_Py*` functions and ~21 are `Py*Type`/`PyExc_*`
data symbols (the rest of `GOT.mem` is this crate's own Rust statics — thread registries, gemm
thresholds — which is expected PIC data for a Rust `dylink.0` side module, not a host import).
That is a much larger real surface than `wasm_probe`'s 45+9, and it linked on the first attempt
with the same `abi3-py313` + `extension-module` pairing §7/§8 already established works.

### 8.3a `import torch` — reached, against the real interpreter

`vendor/vendor_torch.sh` was run (unmodified, as documented) against
`TORCHNATIVE_TORCH_SRC=/Volumes/macMini/caches/spike-venv/...`, producing torch 2.13.0, 2372
Python modules, in `torchnative/src/main/torch(gen)?/functorch/` — gitignored, not committed.
That tree, plus the `wasm32-unknown-emscripten` `_C.wasm` renamed to `_C.abi3.so` (the same
naming `vendor/install_shim.sh` uses for the host build), was copied whole into `/tmp` (scratch,
not the repo) and mounted into Pyodide via `pyodide.FS.mount(NODEFS, ...)`.

First attempt hit a wall already named in `docs/platform/VENDOR.md` ("wall 1"): `torch/__init__.py`
unconditionally tries `ctypes.CDLL(.../libtorch_global_deps.dylib)`, a native artefact
`vendor_torch.sh` deliberately excludes (§ its own header comment: "the vendored tree has exactly
one hole"). `VENDOR.md` already documents upstream's own off-switch — `TORCH_USE_RTLD_GLOBAL` —
and setting it before `import torch` is what this section does too, on *both* sides of the
comparison below, so it is not a wasm-specific workaround.

Past that, a **second wall not previously documented anywhere in this repo**: `torch/__init__.py`
unconditionally imports `torch.multiprocessing`, which imports
`multiprocessing.resource_tracker`, which imports two CPython stdlib extension modules —
`_multiprocessing` and `_posixshmem` — that **Pyodide's own distribution does not ship**, by
design: `ModuleNotFoundError ... "removed from the Python standard library in the Pyodide
distribution due to browser limitations"`. This is not a candle, PyO3, or `torch_c` problem; it
is CPython's multiprocessing subsystem requiring POSIX shared memory that a browser/Node sandbox
does not expose, and it would block `import torch` on Emscripten **regardless of what `_C`
contains**, plain-Python `torch/multiprocessing/__init__.py` included.

Diagnosed, not fixed: two empty `sys.modules` stub entries were injected in the **driving Node
script only** (never in the vendored tree, never in the repo) purely to find out whether this was
the *last* wall before `_C` itself got exercised:

```python
import sys, types
for _name in ("_multiprocessing", "_posixshmem"):
    sys.modules[_name] = types.ModuleType(_name)
sys.modules["_posixshmem"].shm_unlink = lambda *a, **k: None   # resource_tracker registers this at import time
```

With both stubs and `TORCH_USE_RTLD_GLOBAL=1` set, `import torch` completed:

```
=== import torch ===
IMPORT_TORCH_OK
torch.__version__ = 2.13.0
=== provenance of torch._C ===
torch.__file__      = /torch_tree/torch/__init__.py
torch._C.__file__   = /torch_tree/torch/_C.abi3.so
_C.abi3.so on-disk size = 3261949
```

The size (3,261,949) matches the artefact `cargo` produced exactly — `torch._C` is the module
this session built for `wasm32-unknown-emscripten`, not some other path.

### 8.3b Rung 5 — `aten.mm` and `nn.Linear.forward`, both computed and compared to a host control

```python
a = torch.ones(2, 3); b = torch.ones(3, 4)
c = a @ b                                            # aten::mm
lin = torch.nn.Linear(3, 4, bias=False)
with torch.no_grad():
    lin.weight.fill_(1.0)
out = lin(torch.ones(1, 3))                          # aten::linear forward
```

| | Pyodide 3.14.2 / Emscripten 5.0.3 / Node | host `aarch64-apple-darwin`, same crate, same commit |
|---|---|---|
| `(a@b).shape, .sum()` | `(2, 4), 24` | `(2, 4), 24.0` |
| `Linear(3,4,bias=False)(ones(1,3)).shape, .sum()` | `(1, 4), 12` | `(1, 4), 12.0` |

The host control was built fresh (`cargo build --release` for `rust/torch_c`, same worktree, same
commit) and run through the same vendoring + `TORCH_USE_RTLD_GLOBAL=1` steps, via the venv that
`vendor_torch.sh` itself reads from (`/Volumes/macMini/caches/spike-venv`, CPython 3.13.0 — the
floor `abi3-py313` names). **Identical results on both sides**, and they are not trivially
guessable numbers: `24 = 2·4·3` is `ones(2,3) @ ones(3,4)` with the contraction dimension folded
in, and `12 = 4·3` is `Linear` with unit weights over a 3-wide unit input — both would read
differently if either the matmul kernel or the bias/weight application in `_C` were wrong on this
target. This is `aten.mm` and an `nn.Linear` forward, computed for real by the extension this
session built for Emscripten, agreeing with the same code on the host.

**Rung 4: done. Rung 5: done.** Both reached with the real `torch._C` (not `wasm_probe`), the
real vendored `torch` Python tree, and the real Pyodide interpreter identified in §8.0 — with the
one remaining honest caveat named precisely rather than hidden underneath a ✅: getting there
needed two `sys.modules` stubs for a stdlib subsystem (`multiprocessing`) that structurally cannot
exist under Emscripten's browser/Node sandbox, independent of anything this project's own code
does. That is a **new wall**, not documented before this session, and §8.3c names it as one.

### 8.3c A new wall: `torch.multiprocessing` cannot exist on Emscripten, and `torch/__init__.py` imports it unconditionally

Named precisely, because CLAUDE.md §5.5 asks not to leave "it needed a workaround" vague:

- `_multiprocessing` and `_posixshmem` are C extension modules **Pyodide does not ship, on
  purpose** — multiprocessing needs POSIX shared memory and process forking, neither of which a
  single Emscripten instance in a browser/Node sandbox has. This is not a gap that will close by
  upgrading Pyodide; it is the same category of fact as §3d/§7.2b's `-pthread` prohibition, one
  level up the stack.
- `torch/__init__.py:2298` imports `torch.multiprocessing` unconditionally as part of the top-level
  `from torch import (...)` block — there is no lazy-import or feature flag guarding it upstream.
  So **any** Emscripten `import torch`, regardless of what `_C` contains or how it was built, hits
  this unless something intervenes before that line runs.
- What was done here is a **diagnostic stub in the calling harness**, deliberately not landed
  anywhere in the repo (not the vendored tree, which is regenerated and gitignored anyway; not
  `torch_c`; not `tools/wheel/`). It answers "is this the last wall" (yes, for the two-op
  computation in §8.3b) without claiming a real fix exists. A real fix is a design decision that
  belongs with whoever owns the vendoring/wall-tracking work in `docs/platform/VENDOR.md` — candidates
  visible from here, not chosen or landed: patch `torch/__init__.py` to make the
  `torch.multiprocessing` import conditional (crosses into "modifying vendored upstream source",
  which `vendor_torch.sh`'s own design has avoided everywhere else), or provide real
  `_multiprocessing`/`_posixshmem` shims as an installed package (more consistent with how
  `docs/platform/VENDOR.md`'s existing walls are handled, e.g. the `torch_shm_manager` marker file). Both
  are out of this task's scope to decide.

## 8.4 Ladder, stated plainly

| rung | question | result |
|---|---|---|
| 1 | Pyodide obtained, CPython identified | **done** — 314.0.6, CPython 3.14.2, Emscripten 5.0.3 (exactly this machine's emsdk); CPython 3.13 exists as Pyodide `0.28.x` but needs Emscripten 4.0.9, not present |
| 2 | extension built against that interpreter's ABI | **done, with caveat** — `abi3-py313` never consults target headers; that is what "built against the limited API" means here, for both `wasm_probe` and the real `torch_c` |
| 3 | loaded by that interpreter, imports proven resolved not stubbed | **done** — `import wasm_probe` succeeds, `probe_all()==101` matches the host control exactly, and a deliberately-unresolvable symbol was shown to import-fine-but-abort-on-call against the *real* interpreter (§8.2a) |
| 4 | `import torch` reached | **done** — real `rust/torch_c` built for `wasm32-unknown-emscripten` unmodified (3.26 MB, `PyInit__C` exported), real vendored torch 2.13.0 tree, loaded into real Pyodide 3.14.2; needed the documented `TORCH_USE_RTLD_GLOBAL` off-switch plus a new, previously-undocumented stdlib stub for `multiprocessing` (§8.3c) |
| 5 | something computed (`aten.mm`, `nn.Linear`) | **done** — both computed inside Pyodide, both bit-for-bit matching a fresh host build of the same commit |

**The headline: every rung the brief asked for was reached, by execution, against a real
downloaded Pyodide interpreter — not the synthetic stand-in host §7 was limited to.** The one
qualifier that keeps this honest: reaching `import torch` needed a stub for a stdlib subsystem
that cannot exist on this platform at all, independent of anything in this project's control, and
that stub was diagnostic only, never landed in the repo.

## 8.5 Reproducing §8

```sh
export PATH="/Volumes/macMini/caches/emsdk/upstream/emscripten:\
/Volumes/macMini/caches/emsdk/node/24.19.0_64bit/bin:$HOME/.cargo/bin:$PATH"
export EM_CACHE=/Volumes/macMini/caches/emcc-scratch                 # not the shared emsdk cache
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-emcc-bogus

# fetch Pyodide (once) -- CPython 3.14.2 / Emscripten 5.0.3, matching this emsdk
mkdir -p /Volumes/macMini/caches/pyodide && cd /Volumes/macMini/caches/pyodide
curl -sL -o pyodide-core-314.0.6.tar.bz2 \
  https://github.com/pyodide/pyodide/releases/download/314.0.6/pyodide-core-314.0.6.tar.bz2
tar xjf pyodide-core-314.0.6.tar.bz2                                  # -> ./pyodide/

# rung 2/3: build wasm_probe, real functionality
cd /path/to/repo/rust/wasm_probe
cargo build --release --lib --features pyo3-route --target wasm32-unknown-emscripten
node --experimental-wasm-stack-switching /tmp/wasm4_load_probe.mjs \
  "$CARGO_TARGET_DIR/wasm32-unknown-emscripten/release/wasm_probe.wasm" wasm_probe

# rung 3 negative control: unresolved symbol imports fine, aborts on call
cargo build --release --lib --features pyo3-route,bogus-symbol-test \
  --target wasm32-unknown-emscripten
node --experimental-wasm-stack-switching /tmp/wasm4_bogus_test.mjs \
  "$CARGO_TARGET_DIR/wasm32-unknown-emscripten/release/wasm_probe.wasm"

# rung 4/5: build the real torch_c, unmodified, for this target
cd /path/to/repo/rust/torch_c
CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-emcc-torchc \
  cargo build --release --target wasm32-unknown-emscripten            # exit 0, _C.wasm 3,261,949 B

# vendor the real torch Python tree (unmodified script; not committed, gitignored)
cd /path/to/repo && sh vendor/vendor_torch.sh                          # torch 2.13.0, 2372 modules

# assemble a scratch tree -- NOT the repo -- with the wasm-built _C dropped in
mkdir -p /tmp/wasm4_torch_test
cp -R torchnative/src/main/torch      /tmp/wasm4_torch_test/torch
cp -R torchnative/src/main/torchgen   /tmp/wasm4_torch_test/torchgen
cp -R torchnative/src/main/functorch  /tmp/wasm4_torch_test/functorch
cp -R torchnative/src/main/torch-2.13.0.dist-info /tmp/wasm4_torch_test/
cp "$CARGO_TARGET_DIR/wasm32-unknown-emscripten/release/_C.wasm" \
   /tmp/wasm4_torch_test/torch/_C.abi3.so
mkdir -p /tmp/wasm4_torch_test/torch/bin && : > /tmp/wasm4_torch_test/torch/bin/torch_shm_manager

node --experimental-wasm-stack-switching /tmp/wasm4_import_torch2.mjs /tmp/wasm4_torch_test
```

`/tmp/wasm4_load_probe.mjs`, `/tmp/wasm4_bogus_test.mjs` and `/tmp/wasm4_import_torch2.mjs` are
scratch Node/ESM scripts using `loadPyodide()` from `pyodide/pyodide.mjs` — not committed,
reproduced by the shape above: `loadPyodide({indexURL})`, mount or write files into `pyodide.FS`
(`.abi3.so` is a suffix `importlib.machinery.EXTENSION_SUFFIXES` already recognises, §8.0), set
`os.environ["TORCH_USE_RTLD_GLOBAL"]="1"` and the two `sys.modules` stubs from §8.3c, extend
`sys.path`, then plain `import torch`.

**New files, all inside `rust/wasm_probe/`:** the `bogus-symbol-test` feature and
`probe_bogus_symbol()` in `src/lib.rs`, and the corresponding `Cargo.toml` feature entry.
`rust/torch_c/` was read and built against (not modified — its `aten.rs`, `bootstrap.py`,
`overloads.json` were untouched); `tools/wheel/` was not touched. `torchnative/src/main/torch/`
was vendored (gitignored, not committed) via the existing `vendor/vendor_torch.sh`, unmodified,
and only ever *copied* (never edited) into `/tmp` scratch space for the Pyodide runs.

The emsdk was checked before and after this session's use, the same way §7.9 did — this time
against a 2-hour window covering the whole session rather than a 24-hour one, since the session
itself was shorter than a day:

```
files under /Volumes/macMini/caches/emsdk modified in the last 2h: 0
EM_CACHE=/Volumes/macMini/caches/emcc-scratch and cargo-target-emcc-torchc absorbed all writes
```

### 8.6 Corrections to §7 (and to this section's own first draft)

| section | said | now |
|---|---|---|
| §7.6 | "the next step is a download and is not attempted here" | **done** — Pyodide 314.0.6 fetched, CPython 3.14.2 confirmed from the interpreter itself (§8.0) |
| §7.5/§7.5a | "the imports resolved" proven against `pyinit_host.c`, a host this crate wrote | **repeated against Pyodide itself** (§8.2, §8.2a) — same conclusion, stronger host |
| §7's implicit scope | only `rust/wasm_probe` was ever run under emscripten | `rust/torch_c` itself now builds for `wasm32-unknown-emscripten` unmodified (§8.3) — a question §7 never asked |
| this section's own §8.3, first draft | "blocked at dependency resolution" for `rust/torch_c` | **wrong, corrected in place** (§8.3) — it built on the first real attempt; the guess was never run before being written, which is exactly the CLAUDE.md §5.5 mistake this document otherwise tries to avoid |
| §5b's suggested README rows | "extension builds: emscripten builds, loads and runs under Node" | should now read **loads into a real CPython, and reaches `import torch` with a real forward pass** — a materially stronger claim than "runs under Node" |

---

# 9. A wheel — built, installed, and imported

§8 ended with `import torch` and two computed ops inside real Pyodide, and with the sentence
"there is still no WASM wheel". That sentence is now wrong. **A `pyemscripten` wheel of this
distribution was built, installed into a real Pyodide 3.14.2, and `import torch` came up out of
it and computed.** The artefact under test was `rust/torch_c` built from this worktree's HEAD, not
§8's month-old one.

The question this section was opened to answer was "which loader — Pyodide or a statically linked
wasi CPython". §8 had already answered *loader*. What was left was **distribution**, and the
surprise is how little of it was missing.

## 9.1 The result, stated first

```
wheel   torchnative-0.0.12a0-cp313-abi3-pyemscripten_2026_0_wasm32.whl      13,161,912 B
host    Pyodide 314.0.6 · CPython 3.14.2 · Emscripten 5.0.3 · Node 24.19.0
run     TORCH 2.13.0
        MM      [[3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]]        torch.ones(2,3) @ torch.ones(3,4)
        LINEAR  (1, 2)  [[-0.18123117089271545, -0.034667909145355225]]     manual_seed(0)
        FILE    /lib/python3.14/site-packages/torch/_C.abi3.so
```

The `LINEAR` value is bit-identical to the one produced by the **8-30** `_C.wasm` of §8 through the
same seed, so two independently built wasm artefacts a week apart agree. `_C.wasm` grew
3,261,949 → 4,724,414 B over that week; the crate did, not the target.

## 9.2 The four things the wheel had to carry, and only one of them was new

Assembled member by member against the vendored tree, each wall found by running into it:

| # | wall | what fixed it | size |
|---|---|---|---|
| 1 | `ModuleNotFoundError: typing_extensions` | the six `pyproject.toml` dependencies; five are in Pyodide's index | none — `micropip` resolves them |
| 2 | `OSError: could not load dynamic lib .../torch/lib/libtorch_global_deps.so` | an **empty side module built by `emcc`** | one `cc()` — see §9.4 |
| 3 | `RuntimeError: Unable to find torch_shm_manager` | the empty marker `vendor/install_shim.sh:52` already places | none |
| 4 | `ModuleNotFoundError: _multiprocessing` | §8.3c's stub, still diagnostic, still not landed | **unsolved — §9.5** |

Wall 2 is the one that is genuinely target-specific and the one a wheel *must* solve, because
`_load_global_deps()` runs at `torch/__init__.py:444` before anything else. §8 got past it a
different way — `TORCH_USE_RTLD_GLOBAL=1`, an environment variable the caller sets. **A wheel does
not get to set an environment variable in its consumer's process.** Carrying a real
`torch/lib/libtorch_global_deps.so` removes that off-switch from the recipe entirely; the §9.1 run
sets no `TORCH_USE_RTLD_GLOBAL` at all. That is a strict improvement on §8's recipe and it is the
reason the wheel shape is worth more than the scratch-tree shape it replaces.

The stub that worked:

```sh
export EM_CACHE=/tmp/em-cache-wasm      # never the shared emsdk cache
echo 'int torchnative_global_deps_stub(void){return 0;}' > empty.c
emcc -shared -fPIC -fwasm-exceptions -sSIDE_MODULE=2 -o libtorch_global_deps.so empty.c
```

293 bytes. `-fwasm-exceptions` because §7.4a made it mandatory for *every* module in the process,
side modules included; `-sSIDE_MODULE=2` because `ctypes.CDLL` on Emscripten is `dlopen` on a side
module and a main-module link is not loadable.

## 9.3 The finding that inverts §3c and §7.7: **`abi3` needs no change at all**

§3c and §7.7 concluded that Emscripten voids `abi3-py313`, and the natural reading of that — the
one this section started with — is that a wasm wheel must be tagged `cp314-cp314` and must rename
its member to Pyodide's `EXT_SUFFIX`. **Both halves are false, and `packaging` on the target says
so.** Read off the real interpreter:

```
sysconfig.get_platform()        emscripten-5.0.3-wasm32
EXT_SUFFIX                      .cpython-314-wasm32-emscripten.so
importlib EXTENSION_SUFFIXES    ['.cpython-314-wasm32-emscripten.so', '.abi3.so', '.so']
packaging.tags.sys_tags()       110 tags, including
                                  cp314-cp314-pyemscripten_2026_0_wasm32
                                  cp314-abi3-pyemscripten_2026_0_wasm32
                                  cp313-abi3-pyemscripten_2026_0_wasm32   <-- ours
```

- `.abi3.so` **is** in the target's suffix table, so `torch/_C.abi3.so` — `Target.extension_member`
  exactly as it stands — is found. No `extension_member` override, unlike Windows.
- `cp313-abi3-pyemscripten_2026_0_wasm32` **is** a tag this interpreter accepts, so `setup.py`'s
  `py_limited_api = "cp313"` needs no per-target exception.

Both were verified by building the wheel that way and importing it (§9.1), after first building
the `cp314-cp314` + `EXT_SUFFIX`-named variant and finding it works too. Two spellings, one result.

**§7.7 is still right about what it actually claimed** and this does not overturn it: the platform
tag `pyemscripten_2026_0` pins CPython 3.14 *and* Emscripten 5.0.3 together, so the wheel is
single-platform whatever the ABI field says. The correction is narrower and worth having: `abi3`
here is **inert, not harmful**. It buys nothing, it costs nothing, and — this is the part that
matters for `tools/wheel/` — it means the wasm wheel is the *only* one of the seven targets that
needs no ABI-tag machinery, because the pin it would have needed is already carried by the
platform tag beside it.

## 9.4 What `tools/wheel/build.py` would need — sized

Nothing was landed there. §4a guessed "the same shape of work as the two that exist"; having built
the wheel by hand, that guess is right, and here is the itemised version. A `PyEmscriptenTarget`
needs the three answers `Target` asks for and no fourth:

| `Target` member | for wasm | size |
|---|---|---|
| `artefact` | `wasm32-unknown-emscripten/release/_C.wasm` — note `.wasm`, not `.so`; cargo does not apply the Emscripten convention | one string |
| `extension_member` | **inherit** `torch/_C.abi3.so` (§9.3) | zero |
| `global_deps_name` | inherit `libtorch_global_deps.so` | zero |
| `cc()` | `[emcc, "-shared", "-fPIC", "-fwasm-exceptions", "-sSIDE_MODULE=2"]` — the two extra flags are the whole difference from `AndroidTarget.cc()` | ~10 lines |
| `platform_tag()` | **`pyemscripten_{abi_version}_wasm32`** — see the trap below | ~15 lines |
| `check_image()` | wasm magic `b"\0asm"` + version word. `tools/wheel/binfmt.py` has Mach-O, ELF and PE readers and **no wasm reader**; this is the only genuinely new code | ~20 lines in `binfmt.py`, ~10 here |
| `sysconfig()` / `python_root` | see the trap below | ~5 lines, plus an unzip step |

### 9.4a The trap: the tag is *not* derivable from the target's `_sysconfigdata`

This is the dependency that is not the one anyone names, and it is a design-premise collision
rather than a missing file. `build.py`'s stated principle, in its own header at line 55, is that

> the tag [is] derived from the *target CPython distribution* rather than written down here

and `Target.sysconfig()` implements that by `exec`-ing the target's `_sysconfigdata_*.py`. For
Emscripten that file exists — it is `_sysconfigdata__emscripten_wasm32-emscripten.py`, inside
Pyodide's `python_stdlib.zip` rather than loose on disk, so `target_sysconfig()`'s
`root.glob("lib/python3.*/_sysconfigdata_*.py")` finds nothing until it is extracted. That part is
a five-line fix.

The part that is not a five-line fix: **the file does not contain the tag, and cannot.**
`sysconfig.get_platform()` on the target answers `emscripten-5.0.3-wasm32`, which normalises to
`emscripten_5_0_3_wasm32` — a *real* accepted tag (it is in `sys_tags()`, above) but the wrong one
to publish, because it pins the compiler rather than the platform and nothing on PyPI is tagged
that way. The tag to publish is `pyemscripten_2026_0_wasm32`, and `2026_0` is **Pyodide's**
`PYODIDE_ABI_VERSION`, a number CPython's build never sees. It lives in exactly one machine-readable
place in a distribution:

```
pyodide-lock.json  ->  info.abi_version = "2026_0"
                       info.platform    = "emscripten_5_0_3"
                       info.python      = "3.14.2"
```

So `PyEmscriptenTarget.platform_tag()` has to read `pyodide-lock.json`, not `_sysconfigdata`. It is
still derived-from-the-distribution rather than written-down — the principle survives — but the
distribution it derives from is Pyodide's, not CPython's, and that is a new *kind* of input to a
file whose every existing target reads a CPython. Anyone who sets out to add this target by copying
`AndroidTarget` will reach for `self.sysconfig()` first, find `emscripten_5_0_3_wasm32`, and it will
work well enough to be believed.

### 9.4b `verify_cross.py` is the larger half

`build.py` is ~60 lines of new code. `tools/wheel/verify_cross.py` is 1043 lines built on ELF/Mach-O
symbol tables (`elf_dynamic`, `elf_symbols`, `macho_info`) and its mutation suite at line 725
drops members and asserts the checker notices. **A wasm module has none of those structures** — its
imports and exports live in the wasm import/export sections, a different format from all three
readers in `binfmt.py`. Sizing this honestly: a wasm import/export section reader is the piece of
work, and it is bigger than the target itself. Until it exists a wasm wheel can be *built* by this
repo but not *checked* by it, which is the condition every other target was deliberately not
shipped in.

## 9.5 What still blocks shipping, in order

1. **`_multiprocessing` — unchanged from §8.3c, and a wheel makes it sharper, not softer.** Walls
   1–3 all turned out to be things a wheel can carry. This one is not: the fix has to run *before*
   `torch/__init__.py:2298`, so a wheel can only supply it by shipping a top-level `_multiprocessing`
   module (which would shadow the real one on the other six platforms) or by patching vendored
   upstream source. Both are decisions for whoever owns `docs/platform/VENDOR.md`; neither was taken here,
   and §9.1's run still used the harness stub. **This is the one thing standing between "a wheel
   that works when the caller cooperates" and "a wheel".**
2. `filelock` — one of the seven `pyproject.toml` dependencies, and the only one **absent from
   Pyodide's package index entirely** (checked against all of `pyodide-lock.json`, not just the
   `pyodide-core` subset). It is a pure-Python `py3-none-any` wheel, so `micropip` can take it from
   PyPI; it just means a wasm install is not satisfiable from the Pyodide index alone. Small, but it
   is the kind of thing found at install time by a user rather than here.
3. `verify_cross.py` cannot inspect a wasm module (§9.4b).
4. §1d's `simd128` remains off and upstream.

Nothing on that list is `dlopen`, and nothing on it is CPython-shaped. The README's platform table
still reads `wheel builds ❌ *WASI has no dlopen*`, which was true of the WASI column and was never
true of the Emscripten one; that cell is now measurably wrong and belongs to whoever owns the README.

## 9.6 Reproducing §9, and what was left where

```sh
export PATH="/Volumes/macMini/caches/emsdk/upstream/emscripten:\
/Volumes/macMini/caches/emsdk/node/24.19.0_64bit/bin:$HOME/.cargo/bin:$PATH"
export EM_CACHE=/tmp/em-cache-wasm                      # not the shared emsdk cache
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wasm-wheel

cd rust/torch_c && cargo build --release --target wasm32-unknown-emscripten   # _C.wasm, 4,724,414 B
# global-deps stub, as in §9.2
# assemble the wheel: the four package roots from torchnative/src/main, plus
#   torch/_C.abi3.so                       <- _C.wasm
#   torch/lib/libtorch_global_deps.so      <- the emcc stub
#   torch/bin/torch_shm_manager            <- empty
#   dist-info WHEEL Tag: cp313-abi3-pyemscripten_2026_0_wasm32
# then in Node: loadPyodide({indexURL}), loadPackage the five index deps,
#   zipfile.extractall into /lib/python3.14/site-packages, install the §8.3c
#   sys.modules stubs, import torch.
```

Scratch only, as §8 was: the wheel and the stub live in `/tmp`, the Node drivers are not committed,
and nothing under `tools/wheel/` was modified — §9.4 sizes that work rather than starting it, because
a `PyEmscriptenTarget` that `verify_cross.py` cannot check would read as progress it is not.

`torchnative/src/main/torch/` was only ever *read* and copied. The shared emsdk was not written to:
`EM_CACHE` absorbed the one `emcc` invocation, and the sysroot stamp it generated landed in
`/tmp/em-cache-wasm`.

## 9.7 Correction to §8.4

The ladder's five rungs are unchanged, but rung 4's parenthetical — "needed the documented
`TORCH_USE_RTLD_GLOBAL` off-switch" — is now avoidable rather than required. A wheel that carries
`torch/lib/libtorch_global_deps.so` satisfies `_load_global_deps()` directly, and §9.1 ran with the
variable unset. The stdlib stub for `multiprocessing` is still needed and is still the last wall.
