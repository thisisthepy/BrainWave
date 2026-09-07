# Intel NPU (Core Ultra / AI Boost) — how the archived library reaches the NPU, and what torchnative does about it

This round was asked to work out how `intel_npu_acceleration_library` actually reaches
the Intel NPU internally, and to use that understanding to add Intel NPU support to
torchnative. Half this document's value is the source investigation, so it is recorded
in full, with line citations, before any design is chosen.

The user's constraints, stated up front so the design can be read against them:

- **No OpenVINO *export* path.** Writing a model out to a vendor IR file and handing that
  to a vendor tool is not what is wanted.
- `intel_npu_acceleration_library` is known to be end-of-life. Intel archived the
  repository on 2025-04-24; last release `v1.4.0`, Windows-only wheels.
- The wanted shape is the PyTorch-facing one:
  `NPUModelForCausalLM.from_pretrained(model_id, config=CompilerConfig(dtype=int4))`
  then `model.generate(...)`. Not "emit a graph to a file".

Everything in §1 cites the archived source at commit
`073ad6a3a1eb20fdd1ba00d72c7241586372ebee` ("archive repository", Thu Apr 24 12:05:02
2025 -0700), read from a clone rather than from the README. OpenVINO citations are
against its C headers at `master`, fetched 2026-09-07, and against the
`openvino 2026.3.1` macOS arm64 wheel, which is what §3 measures with.

---

## 1. The source investigation

### 1.1 The deciding question: does the compiled extension link libtorch?

**No. It does not link libtorch, includes no PyTorch header, names no ATen or `c10`
symbol, and is not a Python C-API extension module at all.** This is the most
consequential fact in this document. torchnative replaces `torch._C` with a Rust
extension and ships no libtorch; an extension that needed libtorch symbols could not be
hosted here under any design.

How it was determined — three independent checks, each one a command with an answer:

1. **The link line names exactly one library, and it is not libtorch.**
   `CMakeLists.txt:111-114`:

   ```cmake
   add_library(intel_npu_acceleration_library SHARED src/bindings.cpp)

   # Link the OpenVINO libraries
   target_link_libraries(intel_npu_acceleration_library PRIVATE openvino::runtime)
   ```

   There is no `find_package(Torch)`, no `TORCH_LIBRARIES` and no `torch/extension.h`
   anywhere in that file. The only `FetchContent_Declare` is `openvino`
   (`CMakeLists.txt:81-85`), pulling the binary archive pinned at
   `OV_VERSION_SHORT "2024.4"` (`CMakeLists.txt:40-41`).

2. **`grep -rE 'torch|ATen|\bc10\b|at::Tensor' src/ include/` returns zero hits.**
   Not "a few false positives" — none. The complete include list across `src/` and
   `include/` is OpenVINO headers (`openvino/openvino.hpp`, `opset1`, `opset4`–`opset9`,
   `opset13`, `runtime/intel_npu/properties.hpp`), the library's own four headers, and
   C++ standard headers. `src/bindings.cpp:6` includes exactly one thing:
   `intel_npu_acceleration_library/nn_factory.h`.

3. **The whole surface is `extern "C"` and loaded by `ctypes`, not by the import
   system.** `grep -rE 'PyInit|Python\.h|pybind11'` over `src/`, `include/`,
   `CMakeLists.txt` and `setup.py` also returns zero hits. `src/bindings.cpp:8` opens a
   single `extern "C" {` block that closes on the last line of the 580-line file,
   covering all **92** `intel_npu_acceleration_library_DLL_API` entry points. The Python
   side loads the result as a plain shared library —
   `intel_npu_acceleration_library/backend/bindings.py:40-59`:

   ```python
   if sys.platform == "win32":
       ...
       lib = ctypes.WinDLL(os.path.join(dll_path, "intel_npu_acceleration_library.dll"))
   elif sys.platform == "linux":
       # In Linux it is required to explicitly load openvino lib
       _ = ctypes.CDLL(os.path.join(dll_path, "libopenvino.so"))
       lib = ctypes.CDLL(os.path.join(dll_path, "libintel_npu_acceleration_library.so"))
   else:
       raise RuntimeError(
           f"Platform {sys.platform} is not supported for intel-npu-acceleration-library library"
       )
   ```

   `setup.py:30-37` confirms the shape from the packaging side: the "extension" is a
   `CMakeExtension` whose sources are `CMakeLists.txt` plus `src/*.cpp`, built by
   shelling out to `cmake` (`setup.py:70-73`). There is no `Extension` with Python
   sources and no `build_ext` compiling against `Python.h`.

**So the PyTorch coupling is entirely on the Python side, above the FFI boundary.**
That is the good news. §1.5 is the bad news about what the boundary is made of.

### 1.2 What it needs from PyTorch at runtime

A census over all 33 `.py` files in the package:

| Need | Where | On the required path? | Hostable by torchnative? |
|---|---|---|---|
| `nn.Module` subclassing, `named_children`, `add_module`, `nn.Parameter` | `compiler.py:103-141`, `nn/linear.py:16,28-29` | **Yes — this is the entire mechanism** | Yes |
| `torch.no_grad`, `.eval()`, dtype casts | `compiler.py:61,81`, `backend/runtime.py:25` | Yes | Yes |
| `.numpy()` / `torch.from_numpy` | `backend/runtime.py:64,76,97,134,183-184` | **Yes, on every dispatch** | **No — see §1.5** |
| `torch._dynamo.register_backend` | `compiler.py:14` (module-level import), `compiler.py:270` | **No** — registers an *optional* `torch.compile` backend named `npu`; nothing in `compile()` calls it | No, but it is one import to satisfy, not an architecture |
| `torch.fx` | `optimizations.py:7,94`, `compiler.py:272-274` type hints | Only the dynamo backend and horizontal fusion's `symbolic_trace` | Partially |
| `torch.autograd.Function` | `nn/autograd.py:11` (`AutogradMatMul`), used at `nn/linear.py:33,45` | Only when `self.training`; inference takes `run_matmul` at `nn/linear.py:47` | Not needed for inference |
| `torch.overrides.TorchFunctionMode` | `device.py:7,13` | Only the opt-in `config.use_to=True` path (`compiler.py:63-65`) | Not needed for the default path |
| `torch.profiler.record_function` | `backend/runtime.py:9,115,203` | Yes, wraps every dispatch — but it is a no-op context manager | Easily stubbed |
| **`neural_compressor`** | `quantization.py:6,9`, `compiler.py:9` | **Yes, for `dtype=int4`/`int8`** | **No** — see §1.4 |
| `transformers` `LlamaMLP`/`LlamaAttention`/`GemmaMLP`/`GemmaAttention` | `compiler.py:7-8` (module-level) | Yes, for the LLM fast paths | Yes, if transformers imports |

**The important negative result: `torch.compile` is not on the required path.**
`compile(model, config)` at `compiler.py:42-81` is pure `nn.Module` subtree replacement.
It walks the tree recursively (`module_optimization` at `compiler.py:103-141`, which is
`named_children()` + `add_module()` and nothing else) and swaps `torch.nn.Linear` →
`nn.Linear` and `torch.nn.Conv2d` → `nn.Conv2d` (`lower_linear`, `compiler.py:144-173`).
No tracing, no graph capture, no `symbolic_trace`, no dynamo. The
`@register_backend def npu(...)` at `compiler.py:270-289` is an *additional* entry point
for users who want `torch.compile(model, backend="npu")`; `NPUModelForCausalLM` never
touches it.

This is a genuine relief. `torch.compile` is a permanent refusal in torchnative (abi3
blocks PEP 523 — docs/graph/COMPILE.md), so had the mechanism been routed through it, this
would have been a hard wall. It is not.

Also measured: **no `torch.ops` / `torch.library` custom operator registrations anywhere
in the package**, and no `rename_privateuse1_backend` —
`grep -rE 'torch\.library|torch\.ops|custom_op|rename_privateuse1'` returns zero hits.
The `"npu"` device string in `device.py` is not a real PyTorch device:
`parse_to_arguments` (`device.py:97-125`) intercepts `device="npu"` inside a
`TorchFunctionMode` and **rewrites it to `"cpu"`** (`device.py:113-115`). There is no
dispatch-key backend here at all.

### 1.3 How the NPU is actually reached

**Mechanism: the OpenVINO runtime, with its NPU plugin, which in turn talks to the Level
Zero NPU driver. Not Level Zero directly, and not a bundled proprietary blob.**

The chain, file by file:

1. **Python builds a graph by name.** `NNFactory.__init__` (`backend/factory.py:25-51`)
   calls `backend_lib.createNNFactory(b"NPU", profile)` and then reflectively installs
   one Python method per entry in `get_supported_ops()` (`backend/ops.py:28`). That
   table is **61 entries** — `matmul`, `eltwise_add`, `softmax`, `gelu`, `reshape`,
   `scaled_dot_product_attention`, `adaptive_avg_pool`, … `linear` and `convolution` are
   *not* in it; they are separately bound DLL entry points (`src/bindings.cpp:443` and
   `src/bindings.cpp:472`). 61 reflected ops out of 92 exported functions.

2. **Those names cross the `extern "C"` boundary and become OpenVINO graph nodes.**
   `src/bindings.cpp` forwards each to `intel_npu_acceleration_library::ModelFactory`
   (`include/intel_npu_acceleration_library/nn_factory.h:19-34`), a thin wrapper over
   `ov::opset` node constructors — e.g. `parameter` is
   `std::make_shared<ov::opset8::Parameter>` (`nn_factory.h:43-47`) and `constant` is
   `std::make_shared<ov::opset1::Constant>` (`nn_factory.h:57-60`) — accumulating a
   `std::shared_ptr<ov::Model>` (`inference.h:67`).

3. **The model is compiled for device `"NPU"`.** `OVInferenceModel::compile_model`
   (`inference.h:76-101`):

   ```cpp
   void compile_model(std::string device) {
       if (!_isNPUAvailable(core)) {
           // Fallback to auto in case there is no NPU device. Handle this situation at python level
           device = "CPU";
       }
       // set letency hint
       core.set_property(ov::cache_dir("cache"));
       core.set_property(device, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
       if (device == "NPU") {
           core.set_property(device, intel_npu_acceleration_library::npu_compiler_type("DRIVER"));
           core.set_property(device, ov::intel_npu::turbo(true));
           ...
       }
       compiled_model = core.compile_model(model, device);
       infer_request = compiled_model.create_infer_request();
       infer_request.infer();
   }
   ```

   **Lines 77-79 are the silent fallback this whole round is organised around.** The
   device is rewritten in C++, the model compiles, the answers come out correct, and the
   only signal is a Python-level `warnings.warn` elsewhere (`backend/utils.py:56-60`).

4. **`"NPU"` presence is nothing more than an OpenVINO device-list query.**
   `include/intel_npu_acceleration_library/common.h:36-39`:

   ```cpp
   bool _isNPUAvailable(ov::Core& core) {
       std::vector<std::string> availableDevices = core.get_available_devices();
       return std::find(availableDevices.begin(), availableDevices.end(), "NPU") != availableDevices.end();
   }
   ```

5. **`NPU_COMPILER_TYPE = "DRIVER"` (`inference.h:86`, key declared at `common.h:26`) is
   the detail that closes the door on bypassing OpenVINO.** It selects the compiler that
   lives *inside the Intel NPU driver* rather than OpenVINO's in-process compiler. That
   driver-resident compiler's input format is OpenVINO IR. So even the "talk to the
   driver" path is an OV-IR-ingesting path.

So: **there is no non-OpenVINO route to the Intel NPU.** Level Zero is real and is
underneath, but what you can hand it is a driver-compiled blob whose source format is OV
IR. This constrains every option in §2.

`ov::cache_dir("cache")` at `inference.h:82` is why the library's first run is slow and
later runs fast; `modelling.py:95-97,112` adds a second, coarser cache on top (a
`torch.save` pickle of the whole compiled module tree).

### 1.4 `NPUModelForCausalLM.from_pretrained`, and how int4/int8 is applied

`NPUModelForCausalLM` is thin. `modelling.py:128-137`:

```python
class NPUModelForCausalLM:
    from_pretrained = partialmethod(
        NPUModel.from_pretrained, transformers_class=AutoModelForCausalLM
    )
```

`NPUModel.from_pretrained` (`modelling.py:63-113`) does four things:

1. Mangle a cache key from the model name + `config.dtype` + `config.training` + kwargs
   (`get_mangled_model_name`, `modelling.py:24-39`) and, if a pickle exists,
   `torch.load` it and return (`modelling.py:95-97`).
2. Otherwise call the **real** `AutoModelForCausalLM.from_pretrained(...)`
   (`modelling.py:101-103`). Ordinary transformers, ordinary weights.
3. `model = npu_lib.compile(model, config)` (`modelling.py:104`).
4. Optionally `torch.save` the result (`modelling.py:112`), refusing when
   `trust_remote_code=True` (`modelling.py:106-109`).

So the whole "NPU model" is `AutoModelForCausalLM` **plus a module-tree rewrite**.
`generate()` afterwards is stock transformers `generate()` — it never learns anything
about the NPU; it just calls `forward` on modules whose `forward` now dispatches over
ctypes. **This is the single most useful structural fact in the document for
torchnative**, and §3 is built on it.

**Quantization** (`compiler.py:67-75`) has two stages, easy to conflate:

```python
if config.dtype in (int8, int4):
    model = quantize_model(model, config.dtype)   # neural_compressor RTN
    weights_quantization(model)                   # rebind forward
if not config.use_to:
    create_npu_kernels(model)                     # swap in NPU modules
```

- `quantize_model` (`quantization.py:152-174`) delegates to Intel **neural-compressor**:
  `PostTrainingQuantConfig(approach="weight_only", ... "group_size": -1, "scheme": "sym",
  "algorithm": "RTN")` then `fit(...)` (`quantization.py:90-109`), then
  `export_compressed_model(...)` producing `WeightOnlyLinear` modules
  (`quantization.py:112-149`). int4 packs with `compression_dtype=torch.int8`
  (`quantization.py:143-149`).
- `lower_linear` (`compiler.py:158-173`) then converts each `WeightOnlyLinear` into
  `nn.QuantizedLinear`, taking `layer.qweight` as `uint8` for 4-bit and `int8` for
  8-bit, with `layer.scales`, and refusing anything else:
  `raise RuntimeError(f"Unsupported quantization bits: {layer.bits}")` (`compiler.py:172`).
- There is also a **self-contained** symmetric quantizer that does *not* need
  neural-compressor: `quantize_tensor` (`quantization.py:15-51`), per-output-channel
  symmetric, scale `max(|W|)/127` in fp16, plus `compress_to_i4` nibble packing
  (`quantization.py:54-64`, which calls the DLL's `compressToI4`, `src/bindings.cpp:22-24`).
  It is reached from `Linear.fromTensor` (`nn/linear.py:70-104`) — the `dtype`-driven
  path, not the neural-compressor path.

That second, dependency-free quantizer is what docs/graph/QUANT2.md §3 was pointing at when it
cited this library as the precedent for torchnative's own `quantize_`: the scheme is
per-row symmetric int8/int4 with an fp16 scale, and it is ~35 lines of elementwise
torch. **torchnative does not need neural-compressor to reproduce it.**

### 1.5 The finding that actually decides hostability, and it is not libtorch

§1.1 answered the question that was asked and answered it favourably. Then the question
was asked one level lower, and the answer changes the verdict on Option A.

**The archived library's entire FFI boundary is numpy, and torchnative's shim has no
numpy bridge.**

Its `argtypes` are declared as `np.ctypeslib.ndpointer` (`backend/bindings.py:14-18`):

```python
c_fp16_array = np.ctypeslib.ndpointer(dtype=np.float16, ndim=2, flags="C_CONTIGUOUS")
c_fp32_array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
c_i8_array   = np.ctypeslib.ndpointer(dtype=np.int8,    ndim=2, flags="C_CONTIGUOUS")
```

and every call site converts with `.numpy()` (`backend/runtime.py:64,76,97,183-184`) and
reads results back with `torch.from_numpy` (`backend/runtime.py:134`).

Measured on this shim, in the vendored tree, with
`hasattr(torch._C, "_aten_implemented")` asserted first:

| | |
|---|---|
| `tensor.to(torch.float16).numpy()` | `NotImplementedError: not implemented in torch._C shim: TensorBase.numpy` |
| `torch.from_numpy(...)` | `NotImplementedError: ... overload resolution has no table entry for this op` |

`test_the_shim_has_no_numpy_bridge_which_is_why_this_packs_bytes` re-measures both on
every run, by behaviour rather than by version, so the day a numpy bridge lands this
goes red and the conclusion is revisited instead of inherited.

**So the deciding question had a second half.** libtorch would have been fatal and is
absent; numpy is not fatal but is load-bearing on every single dispatch, and satisfying
it means either implementing the numpy bridge or rewriting `backend/runtime.py`. That is
why §3 crosses the boundary in `struct`-packed bytes instead.

---

## 2. The options, and the one chosen

Reading §1.3 back: **every route to the Intel NPU passes through the OpenVINO runtime,
because the NPU driver's own compiler ingests OpenVINO IR.** What the archived library
demonstrates is not a way *around* OpenVINO — it is a way around the OpenVINO **export
workflow**: no `.xml`/`.bin` is ever written, no `ovc` or Model Optimizer is ever run,
and the `openvino` Python package is never imported on the required path. The graph is
built in-process and compiled from memory.

That distinction is the whole design space, and it is the one thing the user has to sign
off on: **torchnative cannot reach the Intel NPU without OpenVINO runtime binaries on
the machine.** It *can* avoid an OpenVINO export path in the sense meant — no IR files
on disk, no vendor conversion tool, no `import openvino`. If "no OpenVINO" means "not
even the runtime DLL", then Intel NPU support is not achievable and this round's answer
is a refusal. §5 item 5 keeps that question open rather than deciding it here.

### Option A — host the archived library on torchnative's shim

- *Cost.* Provide `torch._dynamo.register_backend` (a no-op suffices),
  `torch.profiler.record_function`, `torch.overrides.TorchFunctionMode`,
  `torch.autograd.Function` — **and the numpy bridge of §1.5**, which is not a stub but
  a real tensor-interop implementation on every dispatch. Then replace
  `neural_compressor` entirely, because it reaches deep into PyTorch internals and is
  not plausibly hostable.
- *Risk.* High and mostly not technical. Archived, Windows-only wheels, pinned to
  OpenVINO 2024.4 (`CMakeLists.txt:40-41`), and it copies the entire OpenVINO Python
  package into the wheel (`CMakeLists.txt:101-104`). Every bug from here on is ours.
- *Verdict.* **Rejected.** §1.5 is why this is a clean rejection rather than a close
  call. Its value to us was §1.1–§1.4, which we now have.

### Option B — build OV IR in memory from torchnative and compile it through the OpenVINO **C** API over `ctypes`

- *Cost.* An IR emitter for the subset we care about, plus a `ctypes` binding of about
  twenty C entry points. The same kind of work `coreml.py` and `nnapi.py` already do,
  against a documented, stable format.
- *Why the C API.* `openvino_c.dll` / `libopenvino_c.so` is a stable `extern "C"`
  surface loadable by `ctypes` — the same property that made the archived library's own
  DLL hostable (§1.1). torchnative ships no C++ of its own for this, keeps its abi3
  single-wheel discipline, and adds no build-time OpenVINO SDK.
- *Risk.* The user said "no OpenVINO export path". Nothing is exported to a file here,
  but OV IR is still the interchange format. This must be surfaced, not buried — it is,
  above and in §5. Secondary risk, now retired: the C API has **no graph-construction
  surface**, so `ov_model_t` can only be obtained by reading IR
  (`ov_core_read_model_from_memory_buffer`, `ov_core.h:190-194`). That is precisely why
  Intel wrote `src/bindings.cpp` at all. §3.3 shows a hand-written IR document being
  accepted, so this is a cost rather than a blocker.
- *Verdict.* **Chosen.**

### Option C — our own native shim linking `openvino::runtime`, constructing `ov::Model` programmatically

What Intel did, reimplemented in Rust or C++ inside torchnative.

- *Cost.* Highest. Needs an OpenVINO SDK at build time and a per-platform native
  artefact, cutting against the single abi3 wheel.
- *Benefit over B.* Avoids OV IR as an interchange format entirely — the only design
  that satisfies a strict reading of "no OpenVINO export path".
- *Verdict.* **Not now, kept on the table.** If the user's objection to B is
  specifically the IR format rather than the OpenVINO dependency, C is the answer, and
  B's device layer, refusals and verdict logic carry over unchanged.

### Option D — talk to the Level Zero NPU driver directly

- *Verdict.* **Refused, by name.** Per §1.3 step 5 the NPU driver's own compiler ingests
  OpenVINO IR. Going "direct to Level Zero" would still mean producing OV IR, and would
  additionally mean reimplementing the driver handshake. It buys nothing.

### What was chosen, and what the first step is

**Option B**, and the first step is the one §1.4 makes available: **module replacement**,
not graph lowering.

`compile_model(model, device="NPU")` walks a module tree and swaps every
`torch.nn.Linear` for an `NPULinear` whose `forward` runs on the device. That is not an
approximation of the archived library's mechanism — it *is* the mechanism
(`compiler.py:103-141` + `compiler.py:144-160` + `nn/linear.py:35-51`), and it is the
whole of what makes `from_pretrained(...)` + `generate(...)` run on an NPU.

Graph lowering (`compile_module`) is the *other* door and it refuses by name. The two
are not interchangeable: leaf replacement cannot reach fusions or inter-module structure,
and redirecting one to the other would tell a caller a model was offloaded when most of
it was not.

---

## 3. What was built, and where each claim was measured

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/intelnpu.py verdict_execution_devices present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/intelnpu.py assert_execution_device present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/intelnpu.py linear_ir present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/intelnpu.py NPULinear present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/intelnpu.py compile_model present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/intelnpu.py evidence present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/intelnpu.py judge present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/intelnpu.py IntelNPUUnsupported present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/intelnpu.py dynamo_backend present -->

### 3.1 The verification trap, and how it is closed

The results being correct proves nothing. `inference.h:77-79` will happily give correct
answers on the CPU. An inference from timing proves nothing either — a small model on a
fast CPU beats an NPU round trip.

The assertion used is OpenVINO's own per-compiled-model answer to "where will this run":
the `EXECUTION_DEVICES` property, read through
`ov_compiled_model_get_property(compiled, "EXECUTION_DEVICES", &value)`
(`ov_compiled_model.h`). This is the direct analogue of `MLComputePlan`'s per-operation
answer in docs/graph/NPU2.md: the runtime's own statement of fact, queried *after*
compilation, not inferred from the output.

**The claim this design is allowed to make is exactly:** for a model compiled through
`torchnative.export.intelnpu`, `EXECUTION_DEVICES` reads back `NPU` and nothing else. If
it reads `CPU`, or `NPU` alongside anything else, the module raises rather than
returning a correct answer quietly.

**What it does not prove**, stated so nobody reads more into it: `EXECUTION_DEVICES` is
per compiled model, not per operation. CoreML's `MLComputePlan` is finer-grained — it
answers per operation, which is how docs/graph/NPU2.md caught a partially-offloaded model.
OpenVINO's NPU plugin compiles a model whole or refuses it, so per-model is the right
granularity here; but a heterogeneous `HETERO:NPU,CPU` device would report both and is
treated as failure. `AUTO`, `HETERO` and `MULTI` are refused as target devices for the
same reason.

**And there is a second, coarser trap this design has that CoreML's did not.** A model
is not one Linear. `compile_model` returns a report naming every leaf module type left
behind, and a `fully_offloaded` flag which is `False` for any model containing so much
as a `ReLU`. The archived library has no equivalent: `lower_linear` returns `None` for
anything it does not recognise (`compiler.py:173`) and the caller gets a model it
believes is offloaded.

### 3.2 The evidence is four things, not one

`probe()` gathers a bundle (`evidence`) and judges it (`judge`). The split is so the
refusals can be tested against bundles no machine here can produce.

| # | Question | Refusal if wrong |
|---|---|---|
| 1 | Does `EXECUTION_DEVICES` say `NPU`, alone? | `IntelNPUExecutionError`, quoting the string |
| 2 | Does the same document compiled for `CPU` say `CPU`? | `IntelNPUExecutionError`: the property is not tracking the request, so reading 1 is evidence of nothing |
| 3 | Does a Linear that actually **ran** agree with the reference to f16? | `IntelNPUExecutionError`: right device, wrong arithmetic |
| 4 | Does a **different input** move the answer far more than the agreement? | `IntelNPUExecutionError`: a device returning a cached or constant answer looks exactly like a correct one |

Row 4 is docs/graph/NPU2.md §3.5's control, and it nearly shipped broken here in the same
shape that document warns about. `evidence`'s weights and inputs are quarter-integers,
exactly representable in f16, so a correct device gives an agreement of **0.0**. A pure
ratio check — `control > agreement * 100` — is then `control > 0`, which passes for any
two answers that are not bit-identical. **A relative margin against zero is not a
margin.** `judge` uses `max(agreement * 100, tolerance)`, and
`test_the_numeric_control_has_a_floor_and_not_just_a_ratio` fails if the floor is
removed.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_intelnpu.py test_verdict_refuses_partial_offload present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_intelnpu.py test_the_numeric_control_has_a_floor_and_not_just_a_ratio present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_intelnpu.py test_dynamo_backend_refuses_permanently present -->

### 3.3 What was measured on an arm64 Mac — which is much more than expected

The previous shape of this document said the OpenVINO calls were unexercised and that
the most likely first failure was the hand-written IR. **That turned out to be avoidable.
OpenVINO publishes macOS arm64 wheels**, and the wheel's `libopenvino_c.2631.dylib`
exports the same C API a Windows install does. So the whole path except the device
string was run here, against OpenVINO **2026.3.1**, for device `"CPU"`.

Measured (the wheel was *downloaded and unzipped*, never installed — nothing was added
to any interpreter's site-packages, and the `openvino` Python package is not imported by
anything in this design):

| Claim | Status |
|---|---|
| All 12 originally-bound C symbols exist in `libopenvino_c` | **verified**, `nm -gU` |
| `OV_STATUS`'s 18 entries match `ov_common.h:135-163` | **verified** against the fetched header, including upstream's `UNKNOW_EXCEPTION` (-17) beside the correctly spelled `UNKNOWN_C_ERROR` (-15) |
| `minimal_ir()` is accepted by a real OpenVINO | **verified** — reads and compiles |
| `linear_ir()` is accepted, with a weights blob | **verified** |
| `EXECUTION_DEVICES` reads back off a compiled model | **verified** — `['CPU']` |
| Inference runs and returns f16 bytes | **verified** |
| A lowered `nn.Linear` agrees with the shim's eager answer | **verified** — 1.6e-04 on a 2-layer MLP, f16 |
| A different input moves the answer | **verified** — 4.3e-01, ~2700× the agreement |
| The shim has no `.numpy()` / `torch.from_numpy` (§1.5) | **verified** |
| `EXECUTION_DEVICES` reads back `NPU` | **NOT verified — this is what the Windows run is for** |

The `linear_ir` shape was not written from the schema either. It was read off a
reference document that OpenVINO's own `ov.save_model` emitted for exactly this graph,
then checked back through `ov_core_read_model_from_memory_buffer`.

### 3.4 The negative controls on the tests themselves

CLAUDE.md §5.5 — a verification that cannot fail is not a verification. Four faults were
injected and each was required to turn the suite red:

| Injected fault | Caught by |
|---|---|
| `judge`'s numeric floor replaced by the bare ratio | `test_the_numeric_control_has_a_floor_and_not_just_a_ratio` |
| `verdict_execution_devices` relaxed to "NPU in devices" | `test_verdict_refuses_partial_offload` |
| `transpose_b="false"` everywhere | four tests, via an OpenVINO load failure |
| `transpose_b="false"` **only when `in_features == out_features`** | `test_a_square_linear_pins_the_transpose_that_would_otherwise_be_silent`, and only that test |

The fourth injection is the one worth the trouble. On a non-square Linear a flipped
transpose is a shape error and OpenVINO refuses the IR, so the third injection was
caught structurally — which looks like coverage and is not. On a **square** Linear it
loads and computes `x @ W` instead of `x @ W.T`: right shape, plausible magnitudes,
wrong function. That is the only case where the fault is silent, and before the square
test existed nothing here would have caught it. The test also asserts that the right and
wrong answers are far apart (1.10) before asserting agreement with the right one, so
agreeing with one really does rule out the other.

### 3.5 Refusals

Every unsupported thing refuses by name with a reason, and each has a test asserting the
refusal **on its message text**, not merely its type — a refusal that stops naming the
mechanism is a weakened refusal.

| Condition | Refusal |
|---|---|
| Platform is not Windows or Linux | `IntelNPUUnavailable`, naming the platform and why the plugin does not exist there |
| `openvino_c` not found | `IntelNPUUnavailable`, naming the filenames tried and the env var to override |
| OpenVINO present but `NPU` absent | `IntelNPUUnavailable`, naming the devices that *were* found — the analogue of `common.h:36-39`, but raising where the archived library only warns |
| `EXECUTION_DEVICES` is not exactly `["NPU"]` | `IntelNPUExecutionError`. **The anti-silent-fallback assertion.** |
| Device is `AUTO` / `HETERO` / `MULTI` | `IntelNPUUnsupported` — each may place part of the graph elsewhere |
| A model with no `torch.nn.Linear` | `IntelNPUUnsupported` — returning it unchanged and calling it compiled *is* the silent fallback |
| A weight dimension above `MAX_DIM` (2¹⁷) | `IntelNPUUnsupported`. The archived library draws the same line at `nn/linear.py:66` but **returns the torch layer unchanged**, leaving it on the CPU inside a model the caller believes is offloaded |
| Non-floating-point weight | `IntelNPUUnsupported`, pointing at the quantized path |
| `compile_module` (captured-graph lowering) | `IntelNPUUnsupported`, and explicitly *not* redirected to `compile_model` |
| `neural_compressor` quantization | `IntelNPUUnsupported`, pointing at `torchnative.quant.quantize_` and §1.4 |
| `torch.compile` / dynamo | `IntelNPUUnsupported`, pointing at docs/graph/COMPILE.md. Permanent. Unlike the archived library, torchnative does not offer the optional dynamo backend at all |

`compile_model(device="NPU")` compiles the first swapped layer **eagerly**, before
returning. An earlier version compiled lazily at first forward, which meant that on a
machine with no NPU it handed back a model that looked offloaded and only disclosed
otherwise several layers into a `generate()` loop — by which point the answers are
correct and nothing draws attention. That is the failure this module exists to prevent,
reintroduced by its own convenience.

---

## 4. What to run on the Windows Intel NPU machine

Numbered and copy-pasteable, from a PowerShell prompt in the repository root.

**1. Install the Intel NPU driver.** On a Core Ultra, "Intel(R) AI Boost" must appear in
Device Manager under *Neural processors*. If it does not, nothing below can work.
<https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>
— this is the URL the archived library itself points at (`backend/utils.py:31`). It
required driver `>= 2408` (`backend/utils.py:11`); newer is fine.

**2. Get an OpenVINO runtime with the NPU plugin.**

```powershell
python -m pip install openvino
```

The pip package ships `openvino_c.dll` inside `openvino/libs`. **Nothing imports the
`openvino` Python package** — only its DLL is loaded, by `ctypes`. (An archive from
<https://storage.openvinotoolkit.org/repositories/openvino/packages/> works equally; the
archived library pinned 2024.4, any release with an NPU plugin will do.)

**3. Point torchnative at it.**

```powershell
$ov = python -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs'))"
$env:TORCHNATIVE_OPENVINO_C = "$ov\openvino_c.dll"
$env:PATH = "$ov;$env:PATH"
$env:PYTHONPATH = "torchnative\src\main"
```

(If you installed the archive instead, `$ov` is `...\runtime\bin\intel64\Release`.)

**4. Run the probe.** This is the whole evidence bundle and needs no torch — pure Python
floats over `ctypes`, so it works even if the shim is not built.

```powershell
python -m torchnative.export.intelnpu
```

**5. Run the test file.**

```powershell
python rust\torch_c\pytests\test_intelnpu.py
```

**6. Optional — the same assertion under the shim, on a real `nn.Module` tree.** Needs
the vendored shim installed (`bash vendor/install_shim.sh` under WSL or Git Bash).

```powershell
python -c @"
import torch
assert hasattr(torch._C, '_aten_implemented'), 'not the shim'
from torchnative.export import intelnpu as I
m = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))
x = torch.randn(4, 64); eager = m(x)
m, report = I.compile_model(m, device='NPU')
print('report:', report)
print('max abs diff vs eager:', float((m(x) - eager).abs().max()))
"@
```

### What success looks like

Step 4 exits **0** and ends with a line beginning `PROVEN:`. The JSON above it must
contain these fields with these shapes:

```json
{
  "devices": ["CPU", "GPU", "NPU"],
  "npu_available": true,
  "FULL_DEVICE_NAME": "Intel(R) AI Boost",
  "execution_devices": ["NPU"],
  "execution_devices_control": ["CPU"],
  "linear_execution_devices": ["NPU"],
  "linear_max_abs_diff": 0.0,
  "linear_control_diff": 3.1875,
  "verdict": "npu",
  "assert_device": "NPU",
  "control_moved": true,
  "numeric_control_moved": true
}
```

Read it in this order, because the fields are not equally important:

- **`"execution_devices": ["NPU"]`** — OpenVINO's own statement, read off the compiled
  model, that this model runs on the NPU. **This is the claim.** A list of length one.
- **`"execution_devices_control": ["CPU"]`** — the same document compiled for CPU. The
  reading **moved** when the request moved, so it is tracking the request rather than
  being a constant. Without this the field above proves nothing.
- **`"linear_execution_devices": ["NPU"]`** — the model that actually *ran* also landed
  on the NPU. The probe IR and the Linear are compiled separately and both must land.
- **`linear_control_diff` ≫ `linear_max_abs_diff`** — a different input moved the
  answer. Without this, a device returning a cached or constant answer would look
  identical to a correct one.

Step 5 should print the same `ok` lines seen on the development machine plus this one:

```
ok intelnpu: a model compiled for NPU reports EXECUTION_DEVICES=['NPU'] on 'Intel(R) AI Boost', the CPU control reports ['CPU'], ...
```

Step 6 should print a report with `'execution_devices': ['NPU']` and a `max abs diff`
around 1e-3 or smaller (f16), and `'left_on_cpu': {'ReLU': 1}` — which is honest, not a
failure: the ReLU really is on the CPU.

### What silent CPU fallback looks like

The point of this round is that these are *distinguishable*, and **none of them is "the
numbers came out wrong"** — in every case below the arithmetic is perfectly correct.

| Output | What it means |
|---|---|
| `REFUSED: IntelNPUExecutionError: ... asked OpenVINO to compile for 'NPU', but the compiled model reports EXECUTION_DEVICES = ['CPU']` | **The silent fallback, caught.** OpenVINO accepted the compile and redirected it. This is `inference.h:77-79`'s behaviour surfacing. Check Device Manager and the driver version. |
| `"verdict": "no-npu"`, `"missing": "NPU"`, exit 2 | OpenVINO loaded but sees no NPU: driver missing, or an OpenVINO build without the NPU plugin. Check `devices` in the JSON. |
| `REFUSED: ... EXECUTION_DEVICES = ['NPU', 'CPU']` | Partial offload — part of the graph runs on the CPU. Refused. This is the case docs/graph/NPU2.md records going unnoticed on CoreML. |
| `REFUSED: ... the device control failed. Compiling for 'NPU' and compiling for 'CPU' both report ...` | The property is not tracking the request, so the NPU reading is not evidence. Refused. |
| `REFUSED: ... the numeric control failed` | Two different inputs produced near-identical answers. A device returning a cached or constant result looks like this. Refused. |
| `REFUSED: ... ov_core_compile_model(device=NPU) failed with ov_status_e GENERAL_ERROR -- <text>` | The NPU plugin refused the graph. The message carries OpenVINO's own `ov_get_last_err_msg()` text; send that back. |
| Step 6 prints a good diff and `'execution_devices': ['CPU']` | Would be impossible — `compile_model` raises before returning. If you ever see it, the eager assertion in `compile_model` has been removed. |

**A run that prints correct numbers and nothing else is not evidence.** That is the
entire lesson of docs/graph/NPU2.md, and it is why this module raises where
`intel_npu_acceleration_library` warns.

---

## 5. Left undone

Stated plainly rather than left to be discovered.

1. **Nothing here has touched a real NPU.** Every claim in §3.3's last row is open. The
   most likely remaining failure is no longer the IR (that is now measured) but the NPU
   plugin declining a graph the CPU plugin accepts — the NPU plugin is stricter about
   shapes, layouts and dynamic dimensions.
2. **Only `torch.nn.Linear` is lowered.** No `Conv2d`, no attention, no normalisation.
   Every other leaf stays on the CPU and the report says so. The archived library lowers
   `Conv2d` too (`compiler.py:161-162`) and has fast paths for `LlamaMLP` /
   `LlamaAttention` (`compiler.py:176-191`); neither is here.
3. **Static shapes.** The batch dimension is baked into the compiled IR, so a
   differently-shaped input recompiles. For `generate()` that is a recompile per
   sequence length, which is a real performance problem and not a correctness one. The
   fix is dynamic dimensions in the IR, and it wants its own round.
4. **The crossing is `tolist()` + `struct`, one element at a time.** §1.5 explains why
   it is not numpy, but the cost is real: this is fine for a test and far too slow for a
   7B model. The right fix is a bytes-level accessor on the shim, which is a `torch._C`
   change and not an export-layer one.
5. **No quantization.** `quantize_` refuses. §1.4 shows the dependency-free scheme is
   ~35 lines and that docs/graph/QUANT2.md §3 already wants it; `int4`'s nibble packing would
   also need an OV `Constant` of element type `i4`, which this IR emitter does not do.
6. **`NPUModelForCausalLM.from_pretrained` does not exist.** It is now genuinely close —
   §1.4 shows it is `AutoModelForCausalLM.from_pretrained` plus `compile_model`, both of
   which exist — but items 2, 3 and 4 above mean it would produce something that works
   and is unusably slow, and shipping that under the name the user asked for would
   misrepresent it. That is the next round's job, and it should begin with item 3.
7. **`compile_module` (captured-graph lowering) is unwritten.** It refuses. It would
   serialise a `decompose` → `refold` trace to OV IR, reusing `target.TargetSet` and
   `nnapi.fold_constants` the way `coreml.py` does, and it is what would reach the
   fusions leaf replacement cannot.
8. **The "no OpenVINO" question is the user's to answer.** §2 establishes that the
   OpenVINO *runtime* cannot be avoided while the OpenVINO *export workflow* can. If the
   objection is to IR as an interchange format rather than to the dependency, the design
   moves from Option B to Option C and this module's device layer, refusals and verdict
   logic carry over unchanged.

---

## 6. Reproducing

The pure and refusal tests need nothing:

```sh
PYTHON=/Volumes/macMini/caches/spike-venv/bin/python
$PYTHON rust/torch_c/pytests/test_intelnpu.py      # 26 ok, 7 skips naming what is absent
```

The real-OpenVINO half needs an `openvino_c`, which on macOS means downloading the wheel
and unzipping it — **not installing it**, since nothing imports the Python package:

```sh
mkdir -p /tmp/ov && cd /tmp/ov
$PYTHON -m pip download openvino --no-deps -d .
unzip -q openvino-*.whl
export TORCHNATIVE_OPENVINO_C=/tmp/ov/openvino/libs/libopenvino_c.*.dylib
cd - && $PYTHON rust/torch_c/pytests/test_intelnpu.py   # 32 ok, 1 skip
```

The seven OpenVINO tests skip **by name** without that variable, saying it is unset;
`test_probe_on_real_hardware` skips saying the platform has no NPU plugin.
docs/devices/VULKAN3.md §6.1 is why every skip line names the missing thing: a skip with a false
reason is counted as a pass.

---

## The granularity fix — one oversized leaf no longer refuses a whole model

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/intelnpu.py plan_lowering present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_intelnpu.py test_an_oversized_leaf_is_left_behind_and_named_not_fatal present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_intelnpu.py test_how_much_moved_is_a_value_and_not_only_prose present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_intelnpu.py test_the_predicate_matches_quantize_s_signature_and_narrows_selection present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_intelnpu.py test_the_plan_and_the_real_lowering_cannot_drift_apart present -->

**The report from the field.** A user ran `Qwen/Qwen3-4B-Instruct-2507` on an
actual Intel NPU and the whole model was refused for one layer:

    IntelNPUUnsupported: out_features=151936 exceeds MAX_DIM=131072,
    so this Linear is not lowered.

That is `lm_head`, the vocabulary projection. **The refusal was right about the
silence and wrong about the granularity.** Every model with a large vocabulary
has that layer, it is usually the single largest weight, and refusing the model
because of it makes every real LLM unreachable.

`MAX_DIM = 2**17` is the same line the archived library draws
(`nn/linear.py:66`). The difference was never the limit — it is what happens at
it. The archived library **silently returns the torch layer unchanged**, which
leaves an unannounced CPU layer inside a model the caller believes is on the
NPU. That is `../graph/NPU2.md`'s failure exactly.

**So: the same outcome, the opposite epistemics.** The layer stays on the CPU
and is **named in the report** with its shape and the limit it exceeded, and
`fully_offloaded` goes `False`. Nothing stopped being checked; the check stopped
being fatal and started being announced.

### What changed

* **`predicate(name, module) -> bool`** on `_compile_model` and
  `plan_lowering`. Deliberately the **same signature** as
  `torchnative.quant.quantize_`, and for the reason that function's docstring
  already gives: `lm_head` is both the largest weight and the layer whose error
  lands on the logits with nothing after it to attenuate. Two true facts pulling
  opposite ways, so the choice is the caller's. One idea, one spelling.
* **An oversized leaf is caught per leaf**, recorded as `(name, reason)` in
  `report["skipped"]` — the same shape `quantize_`'s report uses — and the walk
  continues.
* **`fraction_moved` is a value**, `parameters_moved / parameters_total`, so a
  caller who skims the report cannot mistake a partial offload for a whole one.
  `fully_offloaded` is `False` if *anything* stayed behind: a non-Linear leaf, a
  predicate exclusion, or an oversized Linear.
* **`plan_lowering(model, predicate=None)`** answers "what would be lowered"
  **without OpenVINO and without an NPU**. It is not evidence that anything ran
  — `probe()` and `assert_execution_device()` remain the only functions that
  answer that — but it makes the *selection* testable on a machine with neither,
  which is where this defect lived and why it survived. It calls `linear_ir`,
  the same pure function `_NPULinear.__init__` calls, so the plan cannot drift
  from the rule; `test_the_plan_and_the_real_lowering_cannot_drift_apart`
  asserts they agree on both sides of the limit.

### What a Qwen3-4B lowering now says

Measured on this arm64 Mac against a Qwen3 of the real widths (vocab 151936,
hidden 2560, intermediate 9728) cut to **2 layers**, because the full 36-layer
model is ~4 B parameters and does not fit here in float32:

| | |
|---|---|
| Linears lowered | **14 of 15** |
| skipped | `lm_head` — `Linear(out_features=151936, in_features=2560) stays on the CPU: … exceeds MAX_DIM=131072` |
| `left_on_cpu` | `Embedding: 1, Qwen3RMSNorm: 9, Qwen3RotaryEmbedding: 1, SiLUActivation: 2` |
| `fully_offloaded` | **False** |
| `fraction_moved` | **0.206** (201 850 880 / 979 776 512) |

The 0.206 is dominated by the two-layer cut. For the **real 36 layers** the
transformer-block Linears are 36 × 100 925 440 ≈ 3.63 B of about 4.41 B total,
so roughly **82 %** of parameters would move, with `lm_head` (≈ 9 %) and the
embedding (≈ 9 %) staying behind. **That row is arithmetic from the published
widths, not a measurement** — no 36-layer model was instantiated here, and no
Intel NPU was contacted by any of this.

### What is still true

The withdrawn `compile_model` stays withdrawn; this behaviour lives in the
private `_compile_model` and in `plan_lowering`. `IntelNPUUnsupported` is
unchanged for what is genuinely unsupportable — a non-2-D weight, an integer
weight, an unsupported device string — and `verdict_execution_devices` still
refuses a partial offload by name. The rule this module exists for is intact:
**an unannounced CPU layer is the failure; the fix was to announce it, not to
stop checking.**
