# `cuda` — one `resolve()` arm, five named refusals, and a premise that was wrong

**Up front: this round wired CUDA and could not run it. The machine it was
written on is an arm64 Mac with no NVIDIA GPU and no `nvcc`, so not one line of
the CUDA-gated code was compiled here, let alone executed. §8 states exactly
what "CUDA support" does and does not mean as a result, and nothing above §8
claims more than §8 allows.**

| question | answer |
|---|---|
| Is `cuda` in Metal's category or Vulkan's | **Metal's.** `Device::Cuda(CudaDevice)` is already a variant of candle's closed enum, so a `cuda` tensor is an ordinary `candle_core::Tensor`: no `tensor::Repr` arm, no dispatcher arm, no kernel of ours (§1) |
| How big is the device-side change | **One `resolve()` arm**, plus a cache, plus a target-scoped `Cargo.toml` entry — the `mps` shape exactly (§1) |
| Does `cudarc`'s `dynamic-loading` make this a one-wheel-per-platform story | **No, and this was the premise that turned out false.** candle 0.11.0 pins cudarc to `dynamic-linking`, and cudarc's own build script `panic!`s when both are on. The CUDA build is a separate wheel (§2, §3) |
| What does the user get when CUDA is unavailable | A `NotImplementedError` naming **which** of `not_built` / `no_driver` / `no_device` / `wrong_arch` / `unclassified` it was (§4) |
| Did anything run on a GPU | **No.** §6 is the procedure for a machine that has one; §5 is the instrument it reads |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs classify_cuda_refusal present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs cuda_host_readback_gate present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs CUDA_HOST_READBACK_OPS present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs cuda_arch_mismatch present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs note_cuda_dispatch present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_cuda.py test_the_refusals_this_host_cannot_enter_are_still_named_one_by_one present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_cuda.py test_the_cuda_readback_list_is_the_mps_one_and_is_re_derived_from_aten_rs present -->

---

## 0. The four premises this round was given, and how each held up

The round was handed a design and told to implement it rather than relitigate
it — but also to verify each premise and report any that turned out false. One
did, and it is the one that decides the shipping story.

| premise | verdict | where |
|---|---|---|
| **(a)** CUDA is in Metal's category, not Vulkan's: no `Repr` arm, no dispatcher arm, no kernels | **TRUE, and stronger than stated** | §1 |
| **(b)** The whole device-side change is `resolve()` returning `Device::new_cuda(index)` plus a target-scoped feature | **TRUE for the device.** Understated for the *safety*: `mps` needed a gate for the same structural reason and `cuda` needs it more, not less | §1.2, §5 |
| **(c)** cudarc's `dynamic-loading` needs no CUDA at build time, so abi3's one-binary-per-platform property is preserved | **FALSE** — three separate ways | §2, §3 |
| **(d)** Feature gating must be target-scoped so CUDA can never reach Android, iOS or wasm | **TRUE, and implemented that way** | §2.3 |

### 0.1 One premise that was not on the list and was also false

`rust/torch_c/CLAUDE.md` and a repository-root `CLAUDE.md` were named as
required reading. **Neither exists in this checkout** — `find . -iname
'CLAUDE.md'` outside `dist/` and `build/` returns nothing, and there is none in
`~/.claude/` either. Everything else on the reading list was there and was
read. Recorded here rather than passed over, because a document that is
believed to exist is a worse state than one known to be missing.

---

## 1. Why this is the `mps` shape and not the `vulkan` shape

`docs/devices/VULKAN2.md` §5.1 is the sizing round that found the asymmetry, and it
found it while looking at this exact enum:

```rust
pub enum Device { Cpu, Cuda(CudaDevice), Metal(MetalDevice) }   // closed
```

Its conclusion was about Vulkan — there is **nowhere to put** a Vulkan handle,
so a Vulkan tensor cannot be a `candle::Tensor`, so `PyTensorBase.inner: Repr`
needs a fourth arm, so all 396 `tensor()` call sites become relevant. The same
sentence says the opposite about `Cuda`: **the seat is already there.**

So the two devices are not the same size of work, and `docs/devices/VULKAN3.md` §1's
table for `mps` transfers to `cuda` line for line:

| | `cuda` | `vulkan` |
|---|---|---|
| candle backend | **exists** (`cuda` feature, off) | none; zero occurrences of `vulkan` in candle |
| tensor representation | ordinary `Repr::Dense(candle::Tensor)` | `Repr::Vulkan(VkTensor)` — a fourth arm |
| kernels to teach | **none, not one** | one per op, by name, with a SPIR-V shader each |
| this round's change | one `resolve()` arm + a cache + `Cargo.toml` | 827 lines, shaders, a `Repr` arm |

**Adding a `Repr::Cuda` arm would have bought the Vulkan cost and nothing
else**, because candle's own ops already run on the device. The instruction was
explicit about not adding one; the measurement above is why that instruction is
right rather than merely followed.

### 1.1 The arm carries no `#[cfg]`, and that is candle's doing rather than a decision

The `mps` arm is `#[cfg(target_vendor = "apple")]`. The `cuda` arm is not
conditional at all, and the reason is worth stating because it is what made
this round testable:

```rust
// candle-core 0.11.0, src/lib.rs
#[cfg(not(feature = "cuda"))]
pub use dummy_cuda_backend as cuda;
pub use cuda::{CudaDevice, CudaStorage};
```

With the feature off, `Device::new_cuda(ordinal)` still exists and returns
`Error::NotCompiledWithCudaSupport`. So the arm **compiles on every target this
crate builds for — Android, iOS, wasm included — and runs there**, and what it
does there is refuse by name with `reason: not_built`. The refusal is a live
code path on a machine with no GPU rather than a `cfg`'d-out one, which is why
§4 could be tested here at all.

### 1.2 What (b) understated

Premise (b) said the whole change on the device side is `resolve()`. That is
true of *reaching* the device. It is not true of being safe on it, and
`docs/devices/MPS.md` had already established why: an accelerator tensor that is an
ordinary `Repr::Dense` has **no structural protection** against this crate's own
kernels reading it back to the host and computing there. `tensor()` cannot
refuse it, because it is a real candle tensor. `mps` needed a gate for that
reason and `cuda` inherits the same exposure — see §5, where it turns out to be
*worse* on CUDA.

---

## 2. The build cost — premise (c), and the three ways it is false

Premise (c) was: *cudarc has a `dynamic-loading` feature that requires NO CUDA
libraries at build time and dlopens libcuda/libcudart/cuBLAS at runtime.*

**The feature exists and does what the premise says. It is unreachable from
here.** All three findings below are reads of the actual crate sources at the
versions this project resolves (`candle-core` 0.11.0, `cudarc` 0.19.8 as required and **0.19.9 as resolved** in
`Cargo.lock` — every fact below was re-checked against 0.19.9 and holds unchanged,
`candle-kernels` 0.11.0, `cudaforge` 0.1.2), not recollection.

### 2.1 candle pins the opposite feature, and the two are mutually exclusive

```toml
# candle-core 0.11.0, Cargo.toml
[dependencies.cudarc]
version = "0.19.8"
features = ["std", "cublas", "cublaslt", "curand", "driver", "nvrtc",
            "f16", "f8", "cuda-version-from-build-system", "dynamic-linking"]
default-features = false
```

Two of those decide everything.

**`dynamic-linking`.** cudarc's build script turns it into
`cargo:rustc-link-lib=dylib=cuda` and the same for `cudart`, `cublas`,
`cublasLt`, `curand` and `nvrtc`. Those are *link-time* dependencies and they
become ELF `DT_NEEDED` entries.

**And Cargo cannot union its way out of it.** The obvious move — add a `cudarc`
entry of our own with `features = ["dynamic-loading"]` and let Cargo union the
feature sets — produces a build that panics before it compiles anything:

```rust
// cudarc 0.19.8, build.rs
#[cfg(all(feature = "dynamic-loading", feature = "dynamic-linking"))]
panic!("Both `dynamic-loading` and `dynamic-linking` features are active, this is a bug");
```

Feature unification is what makes this repository's Accelerate and Metal
entries compose; here it is what makes the escape impossible. Reaching
`dynamic-loading` would mean patching or forking candle-core, which is a
different and much larger decision than the one this round was making.
`test_cuda_cannot_reach_the_android_ios_or_wasm_builds` asserts that no entry in
`Cargo.toml` names `dynamic-loading`, so the trap cannot be walked into later.

**`cuda-version-from-build-system`.** cudarc's build script then does this:

```rust
let output = std::process::Command::new("nvcc").arg("--version").output();
// ... on failure, without `fallback-latest`:
panic!("`nvcc --version` failed.\n{output_result:?}");
```

So a CUDA build **requires the CUDA toolkit on the build machine**. This is also
why `cargo check` cannot be used here as a cheap syntax gate for the CUDA-gated
code: the dependency graph cannot be resolved without `nvcc`.

### 2.2 `candle-kernels` needs nvcc too, and links `cudart` on its own account

Independently of cudarc:

```rust
// candle-kernels 0.11.0, build.rs
let bindings = KernelBuilder::new().source_dir("src") ... .build_ptx()?;
moe_builder.build_lib(out_dir.join("libmoe.a"))?;
println!("cargo:rustc-link-lib=dylib=cudart");
```

`build_ptx` shells out to `nvcc --ptx`, and `build_lib` compiles `moe_*.cu` and
`mmq_*.cu` into a static archive. Both go through `cudaforge`, whose
architecture selection is:

```rust
pub fn detect_compute_cap() -> Result<GpuArch> {
    if let Ok(cap_str) = std::env::var("CUDA_COMPUTE_CAP") { return GpuArch::parse(&cap_str); }
    detect_from_nvidia_smi()   // `nvidia-smi --query-gpu=compute_cap`
}
```

**Two consequences.**

1. On a machine with no GPU, `CUDA_COMPUTE_CAP` is not optional — `nvidia-smi`
   is absent and the build script errors. The CI job in §7 sets it, and says in
   its own comments that it must.
2. **Premise (c)'s remaining-axis claim is half wrong.** It said the compute-
   capability axis is answered by "PTX plus driver JIT, not more wheels". That
   is true of `build_ptx`'s output — nvcc emits PTX for `compute_N` and the
   driver JITs it onto any `sm_M` with `M >= N`. It is **not** true of
   `build_lib`'s output: `-gencode=arch=compute_N,code=sm_N` compiled to an
   object is cubin with no embedded PTX, so those kernels have **no JIT path at
   all** and need `M == N`.

`_C._cuda_probe()` reports `arch_exact` for this, and `cuda_arch_mismatch`
deliberately refuses only on `M < N` — refusing every `M != N` would refuse the
ordinary, correct case of running sm_80 PTX on an sm_89 card.

### 2.3 The gating — premise (d), which held

The entry is scoped on target **and** on a cfg key, and the two halves do
different jobs:

```toml
[target.'cfg(all(any(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")),
                     all(target_os = "windows", target_arch = "x86_64")),
                 torch_c_cuda))'.dependencies]
candle-core = { version = "0.11.0", default-features = false, features = ["cuda"] }
```

* The **cfg key** is the switch, and it is a cfg rather than a Cargo feature for
  the reason `Cargo.toml`'s Accelerate entry already gives at length: a
  dependency entry's `features` list is static, so a switchable exception means
  a switchable *entry*, and a renamed `candle-core` beside the plain one is
  rejected outright. The cfg key switches the entry without duplicating it and
  leaves `cargo build --release` spelled as it is everywhere else.
* The **target list** is the half that cannot be turned off. Android is
  `target_os = "android"`, iOS is `"ios"`, wasm is `"emscripten"` or
  `"unknown"` — none of them match `linux` or `windows`, so this entry is
  unreachable from those three **even with `--cfg torch_c_cuda` set**. That is
  the same care the Metal entry takes with `target_vendor = "apple"`, applied
  from the other side.
* It is a **third entry** rather than a clause on either existing one, for the
  reason `docs/devices/VULKAN3.md` §1.1 records: the Accelerate entry is gated on
  `not(torch_c_no_accelerate)`, which the Android parity build sets, and folding
  anything into it makes the parity build silently lose that device — a gate
  that stops testing without saying so.

`aarch64` is admitted for Linux because Jetson and Grace-Hopper are real CUDA
targets and are not Android. Windows is x86_64 only; there is no CUDA toolkit
for `aarch64-pc-windows-msvc`.

**Measured, not argued.** `cargo tree -e features` was run per target **with
`--cfg torch_c_cuda` forced on**, which is the interesting direction: the
question is not whether the default build is clean but whether the switch can be
thrown by mistake on a target that must never see CUDA.

```
$ cargo tree -e features --target <T> \
      --config 'target.<T>.rustflags = ["--cfg","torch_c_cuda"]' \
  | grep -cE "cudarc|candle-kernels"
```

| target | cuda crates in the graph, with the cfg **on** |
|---|---|
| `x86_64-unknown-linux-gnu` | **36** — and `candle-core feature "cuda"` present |
| `x86_64-pc-windows-msvc` | **36** |
| `aarch64-linux-android` | **0** |
| `aarch64-apple-ios` | **0** |
| `aarch64-apple-darwin` | **0** |
| `wasm32-unknown-emscripten` | **0** |
| `x86_64-unknown-linux-gnu`, cfg **off** | **0** |

That is premise (d) settled empirically rather than by reading Cargo's matching
rules. Two things it also settles: the host build's graph is unchanged (which is
why golden did not move), and `ug-cuda` — which `candle-core`'s `cuda` feature
mentions as `candle-ug?/cuda` and which therefore appears in `Cargo.lock` — is
**not** in any built graph, because `candle-ug` is optional and off.

**The cross *builds* were not re-run**, only the graphs. A target whose graph is
byte-for-byte the crates it had before cannot have been changed by this entry.
And the claim that the CUDA entry *compiles* rests entirely on the CI job in §7,
which has not run. Both statements are in §8.

---

## 3. The wheel — why one more platform wheel is not the answer, and one more *variant* is

This is where premise (c)'s conclusion falls, and it falls harder than the
feature detail.

Upstream torch splits into `cu121` / `cu124` / `cu126` wheels. Premise (c)'s
reasoning was that upstream does this because it **bundles** cuBLAS, cuDNN and
NCCL — and that a project which does not bundle them keeps abi3's
one-binary-per-platform property.

**The bundling half is right. The conclusion does not follow, because the split
is not caused only by bundling.**

A CUDA-enabled `_C.abi3.so` built as §2 describes carries `libcuda.so.1`,
`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`, `libcurand.so.10`,
`libnvrtc.so.12` and `libstdc++.so.6` as **load-time** dependencies. On a host
without them, `dlopen` fails — and because this distribution *is* `torch`, the
failure is `import torch` raising `ImportError`. Not "cuda is unavailable":
**the whole package stops working**, on every machine without a driver, which is
most machines.

**This is exactly the contrast `Cargo.toml`'s `ash` entry is written around**,
and it is worth putting the two side by side because they look similar and are
opposite:

| | `vulkan` (`ash`, `loaded`) | `cuda` (`cudarc`, `dynamic-linking`) |
|---|---|---|
| when the driver is missing | `dlopen` fails **inside a function**; `_vulkan_probe()` returns `available: False` | the **extension module** fails to load |
| blast radius | one device | `import torch` |
| can it ship in the default artefact | **yes** — it does | **no** |

So: **the CUDA build is a separate wheel.** The `abi3` property is not lost —
one binary still serves CPython 3.13 and every later version, which is what
`docs/design/ABI3.md` §5 argues for and what a user's Python 3.14 already demonstrated
for the CPU wheels. What is lost is *one wheel per platform*: there is now a
CPU `manylinux_x86_64` and a CUDA `manylinux_x86_64`, and a user picks.

**And the cu12/cu13 axis does not fully disappear either**, though it does
shrink. `libcudart.so.12` and `libcudart.so.13` are different sonames, and
cudarc's bindings are generated per CUDA version. Not bundling means we do not
ship four gigabytes of cuBLAS per variant; it does not mean one artefact
satisfies both major versions. What premise (c) got right is the *magnitude*:
one variant per CUDA major, not one per minor, and no cuDNN or NCCL axis at all.

**None of this has been built.** The `readelf -d` claim above is what the CI job
in §7 checks, and that job has never run.

---

## 4. The refusal — five names, and which of them anyone can reach

The instruction asked for a named refusal distinguishing no driver, no device
and wrong arch. There are five names, because two more are real:

| reason | what it is | can it be entered on the project's Mac |
|---|---|---|
| `not_built` | the artefact has no CUDA in it — every wheel published so far, and this checkout | **yes, live** |
| `no_driver` | libraries loaded, driver did not answer: no kernel module, a container without `--gpus`, a driver older than the runtime | no |
| `no_device` | a driver, and no such device — `cuda:3` on a two-GPU box, or `CUDA_VISIBLE_DEVICES=` | no |
| `wrong_arch` | a GPU older than the kernels in the build | no |
| `unclassified` | the driver said something the table does not know | no |

The message puts the reason **near the front as a token**, so a caller can match
on it without parsing prose, and carries the driver's own words through
unedited afterwards:

```
NotImplementedError: cuda:0 is not available -- reason: not_built. This artefact
has no CUDA in it. The CUDA build is a separate wheel, not a flag on this one:
candle pins cudarc to `dynamic-linking`, so a CUDA-enabled `_C.so` names
libcuda/libcudart/libcublas as load-time dependencies and would fail to import
at all on a machine without them -- taking `import torch` with it
(docs/devices/CUDA.md §3). The underlying error was: the candle crate has not been
built with cuda support
```

### 4.1 What the classification matches on, and why that spelling

`cudarc::driver::DriverError`'s `Debug` — which is also its `Display`, and
therefore what candle wraps and this crate sees — is

```
DriverError(CUresult::CUDA_ERROR_NO_DEVICE, "<cuGetErrorString text>")
```

so the **`CUresult` enumerator name is in every message**. The table matches on
that and not on the sentence: the enumerator is ABI, while the sentence comes
from `cuGetErrorString` in whichever driver is installed and is free to be
reworded. Each token in `CUDA_REFUSAL_TOKENS` is a variant of
`cudarc::driver::sys::CUresult`, read out of the crate rather than recalled, and
present in both 0.19.8 (which candle requires) and 0.19.9 (which this lock
resolves) — all twelve, checked in each.

| token | reason |
|---|---|
| `CUDA_ERROR_NOT_INITIALIZED`, `CUDA_ERROR_STUB_LIBRARY`, `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`, `CUDA_ERROR_SYSTEM_NOT_READY`, `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE` | `no_driver` |
| `CUDA_ERROR_NO_DEVICE`, `CUDA_ERROR_INVALID_DEVICE`, `CUDA_ERROR_DEVICE_UNAVAILABLE`, `CUDA_ERROR_DEVICE_NOT_LICENSED` | `no_device` |
| `CUDA_ERROR_NO_BINARY_FOR_GPU`, `CUDA_ERROR_INVALID_PTX`, `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` | `wrong_arch` |

**`unclassified` exists so the taxonomy cannot lie by exhausting.** A classifier
that mapped every input onto four names would be least trustworthy exactly when
it mattered — an unanticipated driver state reported confidently as the wrong
one. `test_an_unrecognised_driver_error_is_not_forced_into_one_of_the_four`
holds that arm open.

### 4.2 `wrong_arch` is checked at `resolve()`, not at the first kernel

All three `wrong_arch` tokens arrive at **module load or kernel launch**, which
means without further work the user's first sight of "your GPU is too old" would
be attached to whichever op happened to run first, several frames inside candle.

So `cuda_device()` reads the device's compute capability through
`CudaContext::compute_capability()` and compares it to `CUDA_COMPUTE_CAP` baked
in at compile time, before the device is cached and before any kernel runs. The
comparison is `>=` and the asymmetry is the point (§2.2): PTX JITs upwards and
never downwards.

### 4.3 What was actually tested here, stated narrowly

`test_the_refusals_this_host_cannot_enter_are_still_named_one_by_one` drives all
five arms through `_C._shim_cuda_classify_refusal` with driver-shaped fixtures,
and `test_every_published_reason_has_a_fixture_and_every_fixture_a_reason`
closes the vocabulary at both ends so a sixth reason cannot be added silently.

**That tests the classifier, not the machine.** It establishes that *if* a driver
returns `CUDA_ERROR_NO_DEVICE` the user is told `no_device`. It does not
establish that any driver ever did. Only `not_built` was entered live.

---

## 5. The readback gate and the counters

### 5.1 One derived list, two devices

`MPS_HOST_READBACK_OPS` is not a statement about Metal. It is the set of kernels
in `aten.rs` that pull a tensor's bytes to the host and do the arithmetic in
Rust — a property of **this crate's kernel**, derived by scanning that file
(`docs/devices/MPS.md` §3.1). Two accelerators cannot legitimately disagree about it,
and a second hand-written list is exactly how they would come to.

So `CUDA_HOST_READBACK_OPS` is an **alias**, not a copy, and
`test_the_cuda_readback_list_is_the_mps_one_and_is_re_derived_from_aten_rs`
asserts both halves against the loaded artefact: that the two tables are equal,
and that the table equals the set re-derived from `aten.rs` at test time.

The derivation's limit is inherited unchanged and is not papered over.
`docs/architectures/VOICE3.md` §6 records that it follows helper calls **one level, by name**,
and that `var`/`std` reached the list through nothing — a `read_flat` inside
`var_reduce` was invisible because `var_reduce` was not one of the six helper
names. That family would have been the one in that file silently computing on
the CPU under an accelerator label, with no test saying so. The blind spot is
covered by `test_every_host_readback_in_aten_is_classified`, which this round
did not weaken and which now guards two devices instead of one.

### 5.2 The gate is *more* necessary on CUDA than it was on Metal

This is the one finding in this round that is about CUDA specifically rather
than about accelerators in general.

`docs/devices/MPS.md` §2 pulled the `mps` guard and measured what came back: fourteen
probed ops, **thirteen of them silently correct-on-the-CPU**, and the fourteenth
loud. The loud one explains the rest — `read_flat` widens through `f64` on the
float path, Metal has no `F32 -> F64`, so the float readbacks raised before they
could be silent. The quiet thirteen were all on the integer path.

**CUDA implements `f64`.** The same kernels that were noisy on Metal succeed
quietly on CUDA. The gate therefore covers strictly more there, and a build that
skipped it because "mps already proved most of these are loud" would be reading
a Metal property as a general one — the exact mistake `docs/devices/MPSATTN.md` §1
records `docs/devices/MPSFWD.md` making about `_softmax`.

### 5.3 The counters, and why they are not the kind of evidence that can be moved out of the way

`docs/devices/MPSATTN.md` §3.1 records, against its own round, that moving a `read_flat`
one call deeper — into a helper named something the scan does not know — passes
**both** of the `mps` derivation tests while keeping the readback. Every check
that greps the source can be defeated by moving the thing it greps for. So
`_C._cuda_counters()` is built the way `_vulkan_counters()` is: as a statement
about what the process did.

```
resolves           a cuda Device handed out by resolve()
dispatches         ops that reached a kernel with cuda tensors, past the gate
readback_refusals  times the gate fired
refusals           times resolve() refused, for any of the five reasons
ops                {op name: count} -- which ops, not just how many
device_free_bytes  read from the driver, at the moment you ask
device_total_bytes  "
```

`_cuda_counters()` **opens nothing and perturbs nothing.** It reads the device
cache without resolving, so `device_free_bytes` stays `None` until something has
actually put a tensor on the GPU — which is the correct answer rather than a
gap. `_cuda_probe()` is the opposite and is meant to be: it resolves, and
therefore increments `resolves`. The first draft routed the counters through
`resolve()` too, which would have made every reading of the instrument bump one
of the numbers it reports and allocate a CUDA context as a side effect of being
asked whether a CUDA context existed.

Three of those are counted at **doors**: `cuda_device()`, the single `is_cuda`
arm of `aten_dispatch`, and `cuda_host_readback_gate`. There is one of each, and
a kernel cannot acquire a CUDA tensor without coming through the dispatcher's —
`test_the_dispatcher_gates_cuda_before_it_runs_a_kernel` asserts there is
exactly one such arm and that the gate precedes the count precedes the kernel.

The last two are **not in this repository at all**. They come from
`CudaContext::mem_get_info()`, which is the GPU reporting its own memory.
Nothing here can move that out of the way, and no amount of host computation can
make it fall. That is the property §3.1's warning is about.

**What the pair does and does not establish, so it is not over-read.**
`dispatches` says an op ran with CUDA tensors and was not refused.
`device_free_bytes` says the bytes are on the GPU. Together with candle's CUDA
backend having no silent CPU fallback — which is a *read of candle's source*,
not a measurement — that is the argument. `docs/devices/MPSATTN.md` §5 makes exactly this
narrowing for `mps` and it applies here word for word. §6 closes the remaining
gap from outside the process.

**None of these counters has been exercised.** On this machine every one of them
is zero except `refusals`, and `test_a_counter_moves_on_the_one_runtime_event_this_host_can_produce`
asserts a delta on that one — which proves the counters are runtime instruments
rather than constants, and proves nothing about a GPU.

### 5.4 Disabled one at a time, and every one of them went red

A verification that cannot fail is not a verification, so each thing this round
added was reverted, rebuilt where a rebuild was needed, and the suite actually
watched to go red. Eight defeats, eight catches.

| what was disabled | caught by | rebuild |
|---|---|---|
| the CUDA entry loses its target list and is gated on `torch_c_cuda` alone | `test_cuda_cannot_reach_the_android_ios_or_wasm_builds` | no |
| a `dynamic-loading` feature is added to the cudarc entry | same | no |
| the workflow is scrubbed of every "no GPU" / "build claim" disclaimer | `test_the_cuda_workflow_claims_a_build_and_not_a_computation` | no |
| the workflow starts allocating on `device="cuda"` | same | no |
| the workflow drops `CUDA_COMPUTE_CAP` | same | no |
| the dispatcher counts *before* it gates | `test_the_dispatcher_gates_cuda_before_it_runs_a_kernel` | no |
| `mps_host_readback_gate` gets its own copy of the message instead of delegating | `test_the_two_devices_share_one_gate_body` | no |
| `CUDA_HOST_READBACK_OPS` becomes a hand-written two-entry list | `test_the_cuda_readback_list_is_the_mps_one_and_is_re_derived_from_aten_rs` | **yes** |
| `classify_cuda_refusal` collapses every input onto `no_driver` | three tests, including the live `not_built` one | **yes** |
| `CUDA_REFUSALS` stops being incremented | `test_a_counter_moves_on_the_one_runtime_event_this_host_can_produce` | **yes** |

**What this does not cover, and it is the same shape as everything else here:**
these are the checks that can be defeated *on this machine*. The
compute-capability refusal, the driver-error arms and the memory counter cannot
be disabled-and-watched here, because they cannot be exercised here at all.

---

## 6. The procedure, for someone with an NVIDIA GPU

**Two hosts this is written for.** A Windows machine with an NVIDIA GPU (use
WSL2 — the toolkit, `nvidia-smi` and the build all work there, and the Windows
build needs `cargo-xwin` wiring this round did not touch), or Google Colab with
a T4 (`Runtime > Change runtime type > T4 GPU`; free tier is enough).

The question the procedure answers is **not** "is the answer right". A host
readback produces a right answer too — that is the entire content of
`docs/devices/MPS.md` §1.2, where `tril` was accused of computing on the CPU on the
evidence that it succeeded and the value was correct, and the accusation was
wrong because right answers do not distinguish the two. The question is **where
the arithmetic happened.**

### Step 1 — establish what the GPU is, before anything is built

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
```

Write the `compute_cap` down without the dot: `7.5` → `75`. A T4 is `75`.

### Step 2 — the toolkit

```bash
nvcc --version | sed -n '4p'     # cudarc's build script parses exactly this line
```

If that fails, the build will fail with `` `nvcc --version` failed `` (§2.1) and
nothing below applies. On Colab, nvcc is already installed.

### Step 3 — build the extension with CUDA on

```bash
git clone <this repo> && cd torchnative
mkdir -p "$HOME/.cargo"
cat >> "$HOME/.cargo/config.toml" <<'EOF'
[target.x86_64-unknown-linux-gnu]
rustflags = ["--cfg", "torch_c_cuda"]
EOF
export CUDA_COMPUTE_CAP=75          # the number from step 1
cd rust/torch_c && cargo build --release && cd ../..
```

The config file rather than `RUSTFLAGS` for the reason `Cargo.toml` records:
`RUSTFLAGS` *replaces* the rustflags from `.cargo/config.toml` instead of adding
to them. The config file also reaches `vendor/install_shim.sh`'s own
`cargo build`, so the shim that gets vendored is this artefact.

### Step 4 — confirm the artefact really is a CUDA one

```bash
readelf -d rust/torch_c/target/release/lib_C.so | grep -E "libcuda|libcublas|libcudart"
```

Nothing here means the cfg did not reach the dependency table, and everything
below would then be measuring a CPU build wearing a CUDA name. `cargo tree -e
features | grep 'candle-core feature "cuda"'` is the second opinion.

### Step 5 — the probe, before any tensor exists

```bash
mkdir -p /tmp/stage && cp rust/torch_c/target/release/lib_C.so /tmp/stage/_C.abi3.so
PYTHONPATH=/tmp/stage python -c "import _C, json; print(json.dumps(_C._cuda_probe(), indent=1))"
```

Expected on a working host: `built: true`, `available: true`, `reason: null`, a
`name`, a `compute_cap` matching step 1, and `free_bytes` below `total_bytes`.

If `arch_exact` is `false`, the PTX kernels will JIT fine and the statically
compiled `moe`/`mmq` kernels have no binary for this device (§2.2). Rebuild with
`CUDA_COMPUTE_CAP` set to the device's own capability if anything reaches them.

### Step 6 — the measurement that separates "ran on the GPU" from "the answer is right"

<!-- CUDA_PROCEDURE_PYTHON -->
```python
# PYTHONPATH=/tmp/stage python this_file.py
import _C

cuda = _C.device("cuda")

# --- 6a. does the tensor's storage live on the GPU? -----------------------
# Free device memory read from the driver, not from any bookkeeping of ours.
# 64 MiB of float32 is large enough to be unambiguous against allocator noise
# and small enough for any card.
before = _C._cuda_counters()
big = _C._aten_dispatch("aten.ones.default", [4096, 4096], device=cuda)
after = _C._cuda_counters()

dropped = before["device_free_bytes"] - after["device_free_bytes"]
print("free bytes dropped by", dropped, "-- the tensor is 67108864")
assert dropped >= 67108864, (
    "device free memory did not fall by at least the tensor's size, so the "
    "storage is not on the GPU")

# --- 6b. did an op dispatch on cuda tensors, past the readback gate? ------
b = _C._cuda_counters()
prod = _C._aten_dispatch("aten.mm.default", big, big)
a = _C._cuda_counters()

print("dispatches +", a["dispatches"] - b["dispatches"])
print("ops seen:", {k: v for k, v in a["ops"].items() if k not in b["ops"]})
assert a["dispatches"] > b["dispatches"], "aten.mm.default never reached the cuda door"
assert a["readback_refusals"] == b["readback_refusals"], (
    "the readback gate fired -- something on this path computes on the host")

# --- 6c. the negative control: an op that is REFUSED, by name -------------
# If this succeeds, the gate is not wired and 6b's silence means nothing.
refused = _C._shim_cuda_host_readback_ops()[0]
print("expecting a refusal for", refused)
try:
    _C._aten_dispatch(refused, big)
except NotImplementedError as e:
    assert "not implemented for the cuda device" in str(e), str(e)
    print("refused correctly:", str(e)[:90])
else:
    raise AssertionError(f"{refused} was not refused on cuda -- the gate is not wired")

# --- 6d. and the answer is also right ------------------------------------
# Last, deliberately. This is the check that CANNOT distinguish GPU from host,
# which is why it is not the evidence and is here only to catch a wrong kernel.
back = _C._aten_dispatch("aten._to_copy.default", prod, device=_C.device("cpu"))
print("mm(ones) corner:", back.tolist()[0][0], "expected 4096.0")
assert back.tolist()[0][0] == 4096.0
print("counters:", _C._cuda_counters())
```

### Step 7 — the external witness, which is the one thing the process cannot fake

Everything in step 6 is measured by the process under test. Two checks from
outside it close that:

```bash
# 7a. Does the OS agree that this PID holds GPU memory?
#     Run step 6 with a sleep at the end, then in another shell:
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 7b. Did the SMs actually execute?  Run a loop of large matmuls for ~30s and
#     watch utilisation. A host fallback shows 0%; real kernels do not.
nvidia-smi dmon -s u -c 30
```

`sm%` staying at 0 through a 4096×4096 matmul loop is the signature of an answer
that was computed somewhere else. `sm%` rising is the evidence step 6 cannot
produce from inside.

**A wall-clock sanity check belongs here too and is not a substitute.** A T4 does
a 4096³ `float32` matmul in roughly a hundredth of the time a runner CPU does.
An order of magnitude is suggestive; `dmon` is dispositive.

### Step 8 — the refusals, deliberately provoked

Three of the five reasons in §4 can be entered on purpose on a machine that has
a GPU, and each should name itself:

```bash
# no_device -- a working driver and nothing visible
CUDA_VISIBLE_DEVICES= PYTHONPATH=/tmp/stage python -c \
  "import _C; print(_C._cuda_probe()['reason'])"          # expect: no_device

# no_device again, by index
PYTHONPATH=/tmp/stage python -c \
  "import _C; print(_C._cuda_probe(99)['reason'])"        # expect: no_device

# wrong_arch -- build for a newer card than you have, then probe.
#   CUDA_COMPUTE_CAP=90 cargo build --release            (on a T4, sm_75)
PYTHONPATH=/tmp/stage python -c \
  "import _C; print(_C._cuda_probe()['reason'])"          # expect: wrong_arch
```

`no_driver` needs a host with the libraries and no working driver — a container
started without `--gpus` is the easy one. If any of these answers
`unclassified`, the driver produced a token §4.1's table does not know; the
`error` field holds its exact words and that is the report to file.

---

## 7. Gates

Run on the round's own worktree, `work/cuda` on develop `2a3d1d5`.

| | baseline (develop) | this round |
|---|---|---|
| `pytests/run.sh` | 1101 ok, 0 FAIL | **1119 ok, 0 FAIL** — the 18 in `test_cuda.py` |
| `cargo test` | 30 passed | **33 passed** — the three in `device.rs::cuda_tests` |
| DOCWATCH | 1006/1006 | **1016/1016** |
| golden `compare.py` | 11420/11420, ops=302, failed=0 | **11420/11420, ops=302, failed=0 — unmoved** |
<!-- DOCWATCH: count golden_cases_passed ge 11420 --> <!-- DOCWATCH: count golden_ops_covered ge 302 -->
<!-- DOCWATCH: count golden_cases_failed eq 0 -->

The suite total is prose rather than a `count` marker: DOCWATCH's `smoke_ok`
reads `test_shim.py`'s own `ok` lines (480), not `run.sh`'s across every file,
and a marker claiming otherwise fails while looking like a regression in the
suite. `golden_cases_failed` is `eq` because a change in either direction is
news; the two above it are `ge` because another round is free to raise them —
docs/devices/MPS.md §6 records what happened the one time that was written `eq`.

**golden is the negative control and it must not move.** Nothing in this round
touches a CPU kernel: the dispatcher gained one arm that no CPU dispatch enters,
and `device.rs` gained a section that a CPU dispatch never reaches.

`.github/workflows/build-cuda-wheel.yml` is the CUDA build gate and **has not
run.** It cannot be run from here — this project's machine cannot execute GitHub
Actions — and it is the only compiler that will ever see the
`#[cfg(torch_c_cuda)]` island in `device.rs`. Its first run should be treated as
part of this change rather than as a later regression.

---

## 8. What "CUDA support" does and does not mean at the end of this round

**Does mean:**

* `torch.device("cuda")` constructs, as it always did, and `resolve()` now has a
  real arm for it instead of falling through to "device not available".
* A build with `--cfg torch_c_cuda` on Linux or Windows x86_64 (or Linux
  aarch64) pulls candle's `cuda` feature and cudarc into the graph, and every
  kernel in this crate that already goes through candle would run on the GPU,
  because candle's own op does. **No kernel of ours was written or needed.**
* When CUDA is unavailable the user is told **which** of five reasons it is, in
  a message carrying the driver's own words, and `_C._cuda_probe()` reports the
  same as a value rather than an exception.
* The ops whose kernels in this crate would compute on the host under a `cuda`
  label are refused at the door, by name, from a list **derived from `aten.rs`**
  and re-derived by a test against the loaded artefact.
* `_C._cuda_counters()` exists, is wired at the three doors, and reads free
  device memory from the driver.
* CUDA is structurally unable to reach the Android, iOS or wasm builds.

**Does NOT mean:**

* **Nothing has run on a GPU. Not one kernel, not one tensor, not once.**
* **Nothing has been compiled with CUDA on, anywhere.** This machine has no
  `nvcc`, and without `nvcc` even `cargo check` cannot resolve the dependency
  graph. The `#[cfg(torch_c_cuda)]` code in `device.rs` — the compute-capability
  read, the ordinal read, the `mem_get_info` call, the probe's device half — has
  been written against cudarc 0.19.8's and candle 0.11.0's actual sources and
  **has never been through a compiler.** The CI job in §7 is where that first
  happens, and it has not run either.
* No CUDA wheel exists. §3 argues one must be separate; nothing has built one.
* Four of the five refusal reasons have never been produced by a driver. Their
  classifier is tested; the states are not.
* The counters have never counted anything but refusals.
* The `wrong_arch` check has never compared a real capability to a real build.
* No performance claim of any kind. None was attempted.
* The cross builds were not re-run. §2.3's argument is that they cannot be
  affected; that is a reading of Cargo's target matching, not a build.

**The README's `computes` cell for `cuda` stays ⚠️, and it must stay ⚠️ until
step 6 and step 7 of §6 have both been run and the `sm%` was not zero.** The
temptation this round was most exposed to is the one `docs/devices/MPS.md` §1.2 names:
concluding from a right answer that the right device produced it. It does not
follow, and until somebody runs §6 nobody knows.
