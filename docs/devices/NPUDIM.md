# NPUDIM — where `MAX_DIM = 2**17` comes from, and what OpenVINO actually limits

## 0. The question, and the verdict

`torchnative/src/main/torchnative/export/intelnpu.py:212` says

    MAX_DIM = 2 ** 17     # 131072

Any `torch.nn.Linear` with a weight dimension above it is refused by name and
left on the CPU. That bites every real LLM: Qwen3-4B's `lm_head` is
151936 x 2560, and it is usually the single largest weight in the model
(`docs/platform/RELEASE_0_1_0b1.md` records `fraction_moved` 0.9033 with 252
leaves lowering and that one not).

The constant was copied from the archived `intel_npu_acceleration_library`
(`nn/linear.py:66`), which gives no reason. The user asked which kind of limit
it is:

  (a) hardware — a fixed field width or silicon shape constraint;
  (b) driver/compiler — a limit in the NPU plugin, the driver-resident
      compiler, or Level Zero, which can move with versions;
  (c) a safety margin in the archived library with no backing in OpenVINO;
  (d) something else.

**Verdict: (c), for the number itself.** `131072` / `2**17` occurs nowhere in
the OpenVINO NPU plugin source, nowhere in the NPU compiler source, nowhere in
the Level Zero graph extension header, and nowhere in the shipped NPU binaries
of either wheel examined. The single strongest piece of evidence is §2.1: the
only per-dimension limit the NPU compiler names is
`VPU_DIMENSION_LIMIT = 8192` (`nce_invariant.hpp:40`) — a *different* number,
sixteen times smaller — and the compiler's response to exceeding it is to
**tile the operation**, not to refuse it (§2.2). "Refuse above N" is not the
shape of any limit in this stack.

**But (c) does not tell us the real ceiling, and this document does not claim
one.** What the evidence establishes is that `2**17` is unsourced, not that
`151936` compiles. §5 says what would settle that, and
`tools/devices/intelnpu_dimsweep.py` is the experiment. **`MAX_DIM` is
therefore unchanged by this round.** Raising it on the strength of "OpenVINO
does not contain that number" would be exactly the inference the rules of
evidence in `docs/devices/QNNOPS.md` §5 forbid: concluding a permissive fact
from the absence of a restrictive one.

## 1. What was examined

| Artefact | Where | Version / ref |
|---|---|---|
| OpenVINO wheel, Windows | `/tmp/ovprobe/openvino-2026.3.1-22476-cp313-cp313-win_amd64.whl` | 2026.3.1-22476 |
| OpenVINO wheel, Linux | `/tmp/ovprobe/openvino-2025.4.1-20426-cp313-cp313-manylinux2014_x86_64.whl` | 2025.4.1-20426 |
| OpenVINO NPU plugin source | `github.com/openvinotoolkit/openvino`, `src/plugins/intel_npu/` | `master`, sparse checkout |
| NPU compiler source | `github.com/openvinotoolkit/npu_compiler` | `develop` @ `0b38f7d42113ff329ac2bdd33583d123de4ccf2f`, 2026-09-02 |
| Level Zero NPU graph extension | `github.com/intel/level-zero-npu-extensions`, `ze_graph_ext.h` | `master` |
| Archived library | `github.com/intel/intel-npu-acceleration-library` | full history, 72 commits |

The Windows wheel matters more than the Linux one here: it is the only one of
the two that ships the compiler itself
(`openvino/libs/openvino_intel_npu_compiler.dll`, 79 MB). The Linux wheel ships
only `libopenvino_intel_npu_plugin.so` and relies on the driver-resident
compiler, which is a closed binary not present on this machine (§4).

## 2. Findings — what OpenVINO does limit

### 2.1 The one named per-dimension limit is 8192, and it is not 2**17

    constexpr int64_t VPU_DIMENSION_LIMIT = 8192;

— `npu_compiler`, `src/vpux_compiler/include/vpux/compiler/dialect/VPU/utils/nce_invariant.hpp:40`,
in `namespace vpux::VPU::NCEInvariant`, alongside `VPU_CHANNEL_ALIGNMENT = 16`,
`MAX_STRIDE = 8` and `SUPPORTED_BATCH_SIZE = 1`. It is enforced at IR
verification time:

    if (vpux::VPU::NCEInvariant::hasDimensionExceedingVPULimit(operandShape)) {
        return errorAt(op, "Op dimensions exceed VPU_DIMENSION_LIMIT: {0}", operandShape);
    }

— `src/vpux_compiler/src/dialect/VPUIP/IR/ops/nce_cluster_task.cpp:644-652`.

That this is the compiler actually shipped, and not a source tree with no
relation to the binary, is checkable without running anything: the format
string appears verbatim in the wheel.

    $ strings -a openvino_intel_npu_compiler.dll | grep VPU_DIMENSION_LIMIT
    Op dimensions exceed VPU_DIMENSION_LIMIT: {0}

(1 occurrence in `openvino_intel_npu_compiler.dll`, 2026.3.1; 0 in
`openvino_intel_npu_plugin.dll`; 0 in the 2025.4.1 Linux
`libopenvino_intel_npu_plugin.so` — the limit lives in the compiler, which is
why the Linux wheel has no trace of it.)

### 2.2 Exceeding it causes tiling, not refusal

`VPU_DIMENSION_LIMIT` is a *per-workload* constraint the compiler is
responsible for meeting, in a pass named for the job:

    SmallVector<int64_t> outputDimThresholds(outputShape.size(), VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
    ...
    while (!isSupportedTileSize(nTilesOnDim, dimToTile, outputDimThresholds) &&
           nTilesOnDim[dimToTile] <= outputShape[dimToTile]) {
        ++nTilesOnDim[dimToTile];
    }

— `src/vpux_compiler/src/dialect/VPU/transforms/passes/ensure_nce_ops_size_requirements.cpp:255-267`
(pass `EnsureNCEOpSizeRequirements`, same file :179-190). The loop increases the
tile count until every tile is within 8192, bounded only by the dimension
itself. A 151936-wide output is 19 tiles, not an error.

The same shape appears at the DMA limit, and there it is documented:

    // Hardware single-task limit is 65535 indices, but empirical testing shows that exceeding
    // 32768 with non-constant indices causes device hang or performance regression,
    // depending on platform. Using half the hardware limit as a safe upper bound.
    // See E#149660 for details.
    constexpr int64_t MAX_SAFE_GATHER_DMA_INDICES = 32768;

— `src/vpux_compiler/src/dialect/IE/transforms/passes/convert_gather_elements_to_gather.cpp:30-34`.
That is what a real, sourced hardware limit looks like in this codebase: a
number, the hardware number it is derived from, the observed failure mode, and
a ticket. `2**17` has none of those.

### 2.3 The Level Zero descriptor field is 32 bits wide, not 17

The graph-argument descriptor the plugin fills in across the driver boundary:

    #define ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE 5
    ...
    uint32_t dims[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];   ///< [out] tensor dimensions upto 5D

— `intel/level-zero-npu-extensions`, `ze_graph_ext.h:258,275` (also `:422`,
`:458` for the `_2_t`/`_3_t` variants; `ze_graph_tensor_ref_t` at `:407` uses
`uint64_t shape[8]`). The constrained quantity in this descriptor is the
**rank** (5, or 8 in the newer `ze_graph_argument_properties_4_t`,
`ze_graph_ext.h:264`), never the extent. `grep -c 131072 ze_graph_ext.h` → 0.

This is the closest thing to hypothesis (a) as literally stated — "a fixed
width in a tensor descriptor" — and at the Level Zero boundary it is not there.

### 2.4 Nothing reports a maximum dimension, so it cannot be discovered at runtime

`openvino/runtime/intel_npu/properties.hpp` — read from both wheels — declares
every NPU-specific property. The read-only ones are
`NPU_DEVICE_ALLOC_MEM_SIZE`, `NPU_DEVICE_TOTAL_MEM_SIZE`, `NPU_DRIVER_VERSION`,
`NPU_COMPILER_VERSION` and `NPU_MAX_TILES`. **There is no property reporting a
maximum tensor dimension, extent, or element count.** The Level Zero side
agrees: `ze_device_graph_properties_t` (`ze_graph_ext.h:113-122`) carries
`graphExtensionVersion`, `compilerVersion`, `graphFormatsSupported` and
`maxOVOpsetVersionSupported` — and nothing about shapes.

Consequence for deliverable 2 of this round: **runtime discovery by property
query is not available.** The only runtime discovery possible is trial
compilation, which is what §5's experiment does.

### 2.5 The 2025.4.1 → 2026.3.1 diff moves no dimension limit

The `intel_npu/properties.hpp` diff between the two wheels adds
`NPU_COMPILER_TYPE` + `CompilerType`, `NPU_PLATFORM`,
`NPU_DISABLE_IDLE_MEMORY_PRUNING`, `NPU_ENABLE_STRIDES_FOR`, and makes
`NPU_MAX_TILES` read-only. No shape or dimension property appears or changes.
So this comparison neither rules (a) out (as the round's brief hoped it might)
nor supports it: there was no dimension limit in either version to move.

### 2.6 The archived library gives no reason, and its history does not either

    if any(dim > 2**17 for dim in layer.weight.shape):
        return layer

— `intel-npu-acceleration-library`, `intel_npu_acceleration_library/nn/linear.py:66`.
`git log -S "2**17"` over the whole repository returns exactly one commit,
`bb4a4f84fb5266125a6f207839e0f12c1c57a486` (2024-02-28), whose entire message
is **"Intel NPU acceleration Library release"** — the squashed initial import.
There is no earlier commit, no linked issue, no comment. `2**17` and `131072`
occur exactly once each in the whole repository, at that line; neither the
README nor `docs/` mentions a dimension limit. **This lead is exhausted: the
reason is not recoverable from that repository.**

## 3. The trap: `131072` *is* in the shipped plugin, and it is not a limit

    $ strings -a libopenvino_intel_npu_plugin.so | grep 131072
    N9intel_npu8MetadataILj131072EEE
    N9intel_npu8MetadataILj131073EEE
    N9intel_npu8MetadataILj131074EEE
    N9intel_npu8MetadataILj131075EEE

A `strings | grep 131072` on the NPU plugin returns a hit, and it looks like
confirmation. It is not. These are Itanium-mangled names for
`intel_npu::Metadata<131072u>` … `<131075u>`, and the template parameter is a
**blob metadata version**:

    static constexpr uint32_t make_version(uint16_t major, uint16_t minor) {
        return major << 16 | (minor & 0x0000ffff);
    }
    ...
    constexpr uint32_t METADATA_VERSION_2_0{MetadataBase::make_version(2, 0)};

— `openvino`, `src/plugins/intel_npu/src/plugin/include/metadata.hpp:93-95,157-164`.
`2 << 16 == 131072`. The four hits are versions 2.0, 2.1, 2.2 and 2.3, and they
are consecutive integers, which is the tell. Recorded here because it is the
one piece of evidence in this investigation that would have produced a
confident wrong answer.

Outside those mangled names, `131072` appears **0 times** in
`openvino_intel_npu_compiler.dll`, `openvino_intel_npu_plugin.dll` and
`openvino_intel_npu_vm_runtime.dll` (2026.3.1), and **0 times** in the whole of
`src/plugins/intel_npu/` in the OpenVINO source tree.

## 4. UNVERIFIED

Named, with what would settle each. Nothing below is used to support §0.

* **The real ceiling for a `Linear` on an Intel NPU.** This document shows
  `2**17` is not sourced to OpenVINO. It does **not** show that
  `out_features=151936` compiles, or that anything between 8193 and 151936
  does. No dimension above 8192 was compiled for `NPU` in this round, on any
  machine. Settled by: `tools/devices/intelnpu_dimsweep.py` Stage B on the
  Windows NPU laptop (§5).
* **The driver-resident compiler (`NPU_COMPILER_TYPE=DRIVER`).** OpenVINO can
  hand compilation to a compiler inside the NPU driver rather than to the
  shipped `openvino_intel_npu_compiler.dll`. That driver binary is closed, is
  not in either wheel, and was not inspected. It is the one place a limit
  invisible to this document could live, and it is also the flow the archived
  library used in 2024. Settled by: running §5's sweep twice on the laptop,
  once with `NPU_COMPILER_TYPE=PLUGIN` and once with `DRIVER`, and comparing
  where each stops.
* **Whether `npu_compiler` @ `develop` is the source of the 2026.3.1 DLL.**
  §2.1 shows one format string matching verbatim, which establishes a common
  ancestry, not identical revisions. The wheel ships no compiler source or
  revision stamp we read. Settled by: `NPU_COMPILER_VERSION` off a real
  device, checked against the repository's version file.
* **Whether the compiler's tiling (§2.2) actually fires for a MatMul, as
  opposed to a convolution.** The pass is on `VPU::TilingBuilderOpInterface`,
  and this round did not trace which ops implement it on which platform, nor
  how `MatMul` is lowered to NCE. A 151936-wide MatMul could still be rejected
  earlier, by an op-conversion pass that never gets as far as tiling. Settled
  by: the same Stage B compile, whose failure message would name the pass.
* **Why the archived library chose `2**17`.** §2.6 exhausts its repository.
  The reason may exist in an internal Intel ticket. Not settleable from open
  sources; noted so nobody re-runs that search.
* **Per-platform variation (NPU 3720 / Meteor Lake vs 4000 / Lunar Lake vs
  50XX).** `npu_compiler` carries `NPU37XX`/`NPU40XX`/`50XX` trees and
  `VPU_DIMENSION_LIMIT` is a single shared constant, but this round did not
  audit whether a per-platform override exists. Settled by: reading
  `src/vpux_compiler/src/NPU*XX/` for a shadowing definition, or by running §5
  on two different NPU generations.

## 5. The experiment that settles it

`tools/devices/intelnpu_dimsweep.py`, shaped after
`tools/devices/intelnpu_verify.py`: **Stage A is SELECTION** (which dimensions
`linear_ir` will emit, pure Python, no OpenVINO, no NPU — a green Stage A is
not a hardware result and the tool says so), **Stage B is EXECUTION** (compile
each dimension for `NPU` and read `EXECUTION_DEVICES` back).

Stage B sweeps `out_features` over
`4096, 8192, 8193, 16384, 32768, 65536, 131072, 131073, 151936` — chosen to
bracket every candidate boundary this document found or ruled out: `8192` is
§2.1's real named limit, `8193` is the first value requiring §2.2's tiling,
`65535/65536` is the DMA-family limit of §2.2, `131072/131073` is the constant
under investigation, and `151936` is Qwen3-4B's `lm_head`.

It reports, per dimension, one of `compiled+NPU`, `compiled+other-device`, or
the OpenVINO error text. Three distinguishable outcomes:

* every dimension compiles and executes on NPU → `MAX_DIM` is a pure safety
  margin and can be removed;
* the sweep stops at exactly 131072 → the number is real after all, sourced to
  the driver-resident compiler (§4 item 2), and the answer is (b);
* the sweep stops somewhere else → that value, not `2**17`, is the constant,
  and its error message names which layer imposed it.

Until one of those runs, `MAX_DIM` stays at `2**17` and every oversized leaf
keeps being refused **by name, with its shape and the limit** — the behaviour
`docs/graph/NPU2.md` exists for, and the one thing this round did not put at
risk.
