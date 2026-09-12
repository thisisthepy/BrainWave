# NPU2 — which unit actually ran it, and the blob on a real NNAPI runtime

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/nnapi_device.py verify_on_device present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/nnapi_device.py run_on_device present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/nnapi_device.py devices present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/nnapi_device.py build_runner present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/nnapi_device.py DEVICE_DIR present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/nnapi_runner.c ANeuralNetworksCompilation_createForDevices present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/nnapi_runner.c ANeuralNetworksModel_getSupportedOperationsForDevices present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_npu2.py test_the_coreml_models_docs_npu_executed_ran_on_the_cpu present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_npu2.py test_pinning_float32_is_what_puts_the_neural_engine_out_of_reach present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_npu2.py test_a_graph_executes_on_the_neural_engine_and_agrees_with_replay present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_npu2.py test_a_conv_relu_blob_executes_on_nnapi_and_agrees_with_replay present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_npu2.py test_the_whole_model_executes_on_nnapi_and_the_control_is_orders_larger present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_npu2.py test_a_driver_that_does_not_claim_the_operations_refuses_by_name present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_npu2.py test_executing_on_a_device_widened_nothing present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_npu2.py test_the_device_module_refuses_to_guess_which_emulator_to_use present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/coreml.py plan_lowering present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/coreml.py compute_plan present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/coreml.py _CoreMLLinear present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/coreml.py _compile_model present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/_module_to.py _lower_for_coreml present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_anepath.py test_the_neural_engine_is_supported_at_float16_and_absent_at_float32 present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_anepath.py test_a_lowered_leaf_records_which_unit_coreml_actually_preferred present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_anepath.py test_the_float32_spelling_agrees_and_the_float16_one_only_nearly_does present -->

## 1. The headline: the CoreML models docs/graph/NPU.md executed ran on the **CPU**

`docs/graph/NPU.md` §2 recorded the `.mlpackage` as **executed** — compiled by macOS,
run through `MLModel.predict`, agreeing with `DecomposedTrace.replay` at
2–3e-08. Every word of that is true. It is also not the sentence
"ran on the NPU", and this round's first job was to find out which one it was.

`MLModel` chooses among three units, and `MLComputePlan` is CoreML's own answer
to which one it picked. Read for all three of docs/graph/NPU.md's float32 graphs
(`MLComputePlan.load_from_path`, `ComputeUnit.ALL`, per operation):

| graph | operations | preferred | supported |
|---|---|---|---|
| `mlp_gelu_softmax` | `linear`, `gelu`, `linear`, `softmax` | **CPU** | CPU, GPU |
| `cnn_conv_pool_relu` | `conv`, `reduce_mean`, `relu` | **CPU** | CPU, GPU |
| `sigmoid` | `sigmoid` | **CPU** | CPU, GPU |

**The Neural Engine is not in the supported column at all.** Not "available but
not preferred" — CoreML does not offer the unit for these programs, so no
`compute_units` setting could have reached it. The machine has one:
`MLComputeDevice.get_all_compute_devices()` returns Neural Engine, GPU and CPU.

### 1.1 And the reason is the flag docs/graph/NPU.md was right to set

`compile_model(float32=True)` is not a detail there either — §6 of that
document measured coremltools' float16 default disagreeing with replay by
2.3e-04 against 3.0e-08, and pinned float32 so a numerical claim would mean
what it says.

That pin *is* what puts the Neural Engine out of reach. The same CNN, both
precisions, one `MLComputePlan` read each:

| | `conv` supported devices | `conv` preferred |
|---|---|---|
| `float32=True` | CPU, GPU | CPU |
| `float32=False` (float16) | CPU, GPU, **NeuralEngine** | CPU |

The Neural Engine is float16 hardware. So the two claims docs/graph/NPU.md wanted are
**mutually exclusive on one run**: float32 buys a 3e-08 agreement and forfeits
the NPU; float16 reaches the NPU and the agreement becomes a float16 agreement.
`test_pinning_float32_is_what_puts_the_neural_engine_out_of_reach` measures both
on one model, so the comparison cannot be confounded by graph shape.

None of this is a defect in docs/graph/NPU.md. It is a distinction that document did
not draw, and drawing it is what this round is for.

## 2. A graph that does execute on the Neural Engine

Three convolutions and a pool at 64×64, at float16, compiled with
`ComputeUnit.CPU_AND_NE` so the GPU is not an option:

```
ios16.cast          CPU           (boundary conversion)
ios16.conv          NeuralEngine
ios16.relu          NeuralEngine
ios16.conv          NeuralEngine
ios16.relu          NeuralEngine
ios16.conv          NeuralEngine
ios16.relu          NeuralEngine
ios16.reduce_mean   NeuralEngine
ios16.relu          NeuralEngine
ios16.cast          CPU
```

Every compute operation is on the Neural Engine; only the two `cast` operations
at the boundary stay on the CPU. It then **runs**, 128 outputs, compared
element-wise against `DecomposedTrace.replay`:

| run | max abs diff vs `replay` |
|---|---|
| `CPU_AND_NE` | **2.0e-04** |
| `CPU_ONLY`, same package | 7.7e-05 |
| the two against **each other** | **1.2e-04** |

2.0e-04 is float16, and it is the same order docs/graph/NPU.md measured for the
float16 default (2.3e-04). That is the honest tolerance for an NPU claim here,
not a regression.

**The third row is the one that makes this evidence rather than a plan.** A
compute plan is a statement of intent; identical outputs from the two runs
would be exactly what it would look like if the Neural Engine plan were being
ignored. They are not identical, and the size of the difference is the size
half precision produces. The test asserts that difference is non-zero and says
in its message what a zero would mean.

### 2.1 Why the small models stayed on the CPU even at float16

The CNN of §1.1 has the Neural Engine in its *supported* set at float16 and
still gets `preferred: CPU`. The wide model does not. Nothing was changed
between them except size — 3→8 channels at 16×16 against 3→64→128→128 at 64×64.
The planner is weighing dispatch cost against work, and below some amount of
work the CPU wins. That is CoreML's decision and this document reports it
rather than arguing with it; the consequence for a reader is only that
"reaches the Neural Engine" is a property of a *model*, not of this lowering.

**No timing is reported here.** Four agents were running on this machine while
these numbers were taken, and docs/devices/MPS.md and docs/devices/VULKAN3.md both record why a
throughput number measured like that is worse than no number.

### 2.2 And now `to(torchnative.device.npu)` lowers for it

Everything above is a measurement of artefacts built by hand in a test. The
device namespace could see the hardware and not use it: on this machine
`device.npu.availability()` returned `available=True, kind=measured` and
`resolve()` named the Apple Neural Engine through the `coreml` backend, and
then `model.to(device.npu)` raised `NotImplementedError` — only the `openvino`
arm had been wired. `torchnative.export.coreml` now has the equivalent of
`intelnpu.plan_lowering` and `_compile_model`, and `device/_module_to.py` has
the dispatch arm.

**There is no float32 road to the unit, and that was checked rather than
assumed.** coremltools' `compute_precision` accepts exactly three things
(`converters/_converters_entry.py`): `precision.FLOAT32` (no transform),
`precision.FLOAT16` (cast everything), and
`transform.FP16ComputePrecision(op_selector=...)` — which is a *subset selector
for the float16 cast*, not a third precision and not a float32 route. Nothing
in the API asks for float32 on the Neural Engine, because the Neural Engine is
float16 hardware. Measured here for `ios16.linear` at three sizes, one
`MLComputePlan` read each:

| program | precision | preferred | supported |
|---|---|---|---|
| `linear` (1, 1024→1024) | float32 | CPU | CPU, GPU |
| `linear` (1, 4096→4096) | float32 | CPU | CPU, GPU |
| `linear` (128, 1024→1024) | float32 | GPU | CPU, GPU |
| `linear` (1, 1024→1024) | float16 | CPU | CPU, GPU, **NeuralEngine** |
| `linear` (1, 4096→4096) | float16 | CPU | CPU, GPU, **NeuralEngine** |
| `linear` (128, 1024→1024) | float16 | **NeuralEngine** | CPU, GPU, **NeuralEngine** |

The float32 rows are the §1.1 finding again on a different operator: the unit is
absent from the *supported* column at every size, so this is a property of the
precision and not of the size. The float16 rows add §2.1's: the unit is
supported at every size and *preferred* only once there is enough work, which
is why batch 1 goes to the CPU and batch 128 does not.

**So they are two products, and therefore two spellings.**

```python
model.to(torchnative.device.npu)                        # float16
model.to(torchnative.device.npu, precision="float32")   # float32
```

With the grades stated rather than averaged. Both through `coreml.verify`,
which runs the compiled model and compares against `DecomposedTrace.replay`,
on one `Linear(1024, 1024)` at batch 128 with `ComputeUnit.CPU_AND_NE`:

| spelling | max abs diff vs `replay` | grade | unit |
|---|---|---|---|
| `precision="float32"` | **2.7e-06** | **agrees** (≤ 2e-05) | CPU / GPU |
| `precision="float16"` | **1.5e-03** | agrees *at float16* | **Neural Engine** |

2e-05 is `verify`'s own default and is where docs/graph/NPU.md set it; it is the
bar the word *agrees* means in this project, and the float16 path does not meet
it. That is reported and not papered over by widening one tolerance to cover
both — 1.5e-03 is larger than §2's 2.0e-04 because a `Linear(1024, 1024)`
accumulates over 1024 terms where that CNN did not, and it is half precision
behaving exactly as half precision does.

**Nothing succeeds without saying what ran.** `MLComputePlan` is read at every
compile, not optionally, and the per-operation rows land on
`model.torchnative_offload["plans"]` keyed by the shape that produced them —
because §2.1 means "which unit" is not answerable until there is a real shape.
Two things warn, and only these two, so that silence stays informative:

* at `to()`, when the chosen precision puts the unit out of the supported
  column entirely (`precision="float32"`), since that is decided by the
  precision and not the shape;
* at the forward that compiles a new shape, when the unit *is* supported and
  CoreML preferred something else anyway.

The eager batch-1 compile does not warn. It is a probe this code chose the
shape for, and warning that CoreML preferred the CPU for a shape nobody asked
for is noise; its plan is still recorded, marked `probe: True`.

**Linear only, and conv is named rather than half-done.** `supported_ops()` has
a MIL lowering for `aten.convolution.default` and conv leaves are still left on
the CPU and reported, because a conv leaf's program cannot be built without its
input's spatial dimensions and those are not knowable at `to()` time. A
Linear's can: batch is the only free dimension. Widening this needs a shape
source, not a bigger table.

Zero leaves lowered raises, as on the Intel arm. `rust/torch_c/pytests/test_anepath.py`
holds all of it, and each guarantee was nullified individually and seen to go
red.

## 3. NNAPI, executed — and by which driver

docs/graph/NPU.md §2 put the NNAPI blob under **structurally validated** and said
plainly why: "There is no NNAPI runtime on a Mac." That is still true of the
Mac. It is not true of the Android emulators already on this machine.

### 3.1 What was missing was one program

`nnapi.py` ends at `parse_model`, which decodes the blob back through the
layout `serialize_model` wrote. That proves the *layout* and says nothing about
arithmetic — a `_SIGNATURES` entry in the wrong position decodes perfectly and
computes something else, which is why `verify_shapes` exists.

`nnapi_runner.c` is the other half. It reads the same layout and replays it
into `ANeuralNetworksModel`: every operand, every immediate, every weight
buffer and every opcode comes out of the blob. It is a **replayer, not a
converter** — there is no second lowering on the device that could agree with
the first by sharing a mistake, the same reason `verify_shapes` compares
against capture rather than against a recomputation.

`nnapi_device.py` builds it with the NDK, pushes it, runs it, pulls the output
bytes back and compares them against `DecomposedTrace.replay`.

Two layout details are not incidental, because getting either wrong produces a
disagreement that *looks* arithmetic:

* **Weights are not in the blob.** A `NUMBERED_BUFFER` value carries
  `(buf_num, offset, size)` and the bytes live in `used_weights[buf_num]`,
  already permuted to NHWC where the operand is CHANNELS_LAST. They travel in a
  side file, each buffer length-prefixed.
* **Shapes in the blob are NNAPI's, not PyTorch's** — upstream ran `fix_shape`
  over them. So an input operand marked CHANNELS_LAST is fed NHWC, and an
  output operand marked CHANNELS_LAST has the *reference* permuted to match
  rather than the device's answer reshaped. Reshaping would make a layout error
  agree on the first element and look like noise afterwards.

### 3.2 Which driver answered

Read from the device with `ANeuralNetworks_getDeviceCount`, not assumed:

| name | type | version | feature level |
|---|---|---|---|
| `nnapi-sample_all` | 2 | `JUST_AN_EXAMPLE` | 1000008 |
| `nnapi-sample_quant` | 2 | `JUST_AN_EXAMPLE` | 1000008 |
| `nnapi-sample_sl_shim` | 2 | `JUST_AN_EXAMPLE` | 1000008 |
| `nnapi-reference` | 2 | 13818094 | 1000008 |

**These are all software.** `nnapi-reference` is the runtime's own CPU
reference implementation; the three `nnapi-sample_*` are the sample drivers the
emulator image ships, and they report their version as `JUST_AN_EXAMPLE`. No
hardware accelerator is present on an emulator, and this document does not
claim one. What it claims is that the blob upstream's serialiser wrote is
accepted by a real NNAPI runtime and computes the right numbers — which is the
thing that was untested, and which a driver swap does not change.

The driver is *chosen*, with `ANeuralNetworksCompilation_createForDevices`,
rather than left to the runtime. So "which driver ran this" is a decision this
side made and can report, not an observation it has to infer.

### 3.3 `Conv2d → ReLU`

436-byte blob, 9 operands, 2 operations, 256 output elements.

| driver | operations claimed | max abs diff vs `replay` |
|---|---|---|
| `nnapi-reference` | 2/2 | **1.2e-07** |
| `nnapi-sample_all` | 2/2 | 1.2e-07 |
| `nnapi-sample_sl_shim` | 2/2 | 1.2e-07 |

All three drivers are run, not one, so a result that depended on a particular
software implementation would show up as a disagreement between them.

### 3.4 The whole model — docs/graph/REFOLD.md §4's deliverable, executed

`Conv → BatchNorm → ReLU → Conv → ReLU6 → AdaptiveAvgPool → Linear → Softmax`,
folded and constant-folded exactly as that document describes, is the same
artefact it was:

```
pairs folded  1
bytes         1156
opcodes       CONV_2D(3) RELU(19) CONV_2D(3) RELU6(21)
              AVERAGE_POOL_2D(1) RESHAPE(22) FULLY_CONNECTED(9) SOFTMAX(25)
outside nnapi.supported_ops()   0
```

and it now runs. `nnapi-reference`, 8/8 operations claimed, 5 outputs:

| | |
|---|---|
| device vs `replay`, same input | **~3e-08** (1.5e-08 and 3.0e-08 on two runs) |
| device vs `replay`, **different** input | **7.2e-02** |

> **docs/graph/REFOLD.md §4's line moves.** That section carries a blockquote saying
> "Still structurally validated, not executed. There is no NNAPI runtime on a
> Mac; docs/graph/NPU.md §2 draws that line and nothing here moves it." This does.

### 3.5 The control, and the version of it that would have been worthless

The first attempt at that second row gave **7.3e-04**, not 7.2e-02, and it
would have been a bad check. A softmax over a randomly-initialised `Linear(4,5)`
is nearly uniform — every output sits near 0.2 whatever the input is — so
feeding the device the *wrong picture entirely* moved the answer by less than a
thousandth. A 1e-4 tolerance would then have been passing on the model's
flatness rather than on the device's arithmetic, with only a factor of seven
between "right" and "completely wrong".

Widening the last layer's initialisation spreads the output (`0.106, 0.018,
0.099, 0.281, 0.495` instead of five numbers near 0.2) and the control moves to
7.2e-02 — **six orders of magnitude above the agreement.** The test requires
four. CLAUDE.md §5.5: this is the same shape as the `padding=1, stride=1`
convolution in docs/graph/NPU.md §4 whose fault injection was the identity.

### 3.6 The negative control for the driver selection itself

`nnapi-sample_quant` is a quantised-only sample driver.
`ANeuralNetworksModel_getSupportedOperationsForDevices` reports it claiming
**0 of 8** operations and the compilation fails, so `verify_on_device` refuses
by name with that line in the message. If it ever succeeded, the device name
would not be deciding anything and every "this driver ran it" sentence above
would be unfounded.

That is also why the runner reports `supports N/M` before compiling: a driver
with partial support would otherwise fall back invisibly and the answer would
be attributed to the wrong device.

## 4. What this round did **not** widen

docs/graph/REFOLD.md left a standing warning — it measured `mobilenet_v2` getting
*worse* under a bigger table, 203 nodes to 1,191, and told the next person that
more ops lowering is not automatically progress.

**Nothing here widens anything, and that is checked rather than asserted.**
`nnapi.supported_ops()` is the same **25** overloads it was, pinned by value in
`test_executing_on_a_device_widened_nothing`, and the whole model still has
zero ops outside it. No entry was added to `_SIGNATURES`; no decomposition or
refold rule changed; no Rust changed, so the golden harness is untouched at
11385/11385, ops=300.

Split the way CLAUDE.md §5.3 asks:

| | |
|---|---|
| **feature added** | `nnapi_device.py` + `nnapi_runner.c` — execution of an NNAPI blob on a device |
| **claim corrected** | docs/graph/NPU.md's executed CoreML claim is a CPU claim; docs/graph/REFOLD.md §4's "not executed" no longer holds |
| **coverage added** | **none** — 25 serialisable overloads before and after |
| **tests added** | 9, in `rust/torch_c/pytests/test_npu2.py` |

## 5. What is still missing, and how big it is

* **No hardware NNAPI accelerator was reached.** Everything in §3 is a software
  driver on an emulator. Reaching a real NPU needs a physical Android device
  with a vendor driver, which is a hardware acquisition and not a code change.
  Nothing else about the path would differ: the same blob, the same runner, the
  same `createForDevices` call with a different name.
* **NNAPI is deprecated.** The runtime is present and complete on API 36 (via
  `/apex/com.android.neuralnetworks/lib64/libneuralnetworks.so`) but Android 15
  deprecated it for new development. Whatever succeeds this path on Android,
  the serialiser work is not wasted — but a future round should not assume the
  API keeps growing.
* **The API-26 emulator cannot be used for this.** NNAPI arrives at API 27;
  `libneuralnetworks.so` is simply absent on 26. That is why §6 says API 36.
* **One CoreML claim cannot be made at once.** §1.1: float32 agreement and
  Neural Engine execution exclude each other. A round that wants both needs
  either a float32 accuracy claim on the CPU *and* a separate float16 claim on
  the Neural Engine — which is what §1 and §2 are — or a way to bound the
  float16 error against the float32 answer, which is a numerical-analysis
  question and not an export question.
* **The Neural Engine only takes large enough models** (§2.1). A future round
  that wants "our lowering runs on the NPU" for a *small* model will find the
  planner declining, and that is not something the lowering can fix.

## 6. Reproducing

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-npu2
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export TORCH_C_STAGE=/tmp/stage-npu2
PY=/Volumes/macMini/caches/spike-venv/bin/python

cd rust/torch_c && cargo build --release && cd -
bash vendor/install_shim.sh

# The NNAPI half needs a device with API >= 27. `pmp_api26` cannot run it.
emulator -avd pmp_api36 -port 5556 -no-window -no-audio -no-snapshot-save &
export ANDROID_SERIAL=emulator-5556        # required; never inferred
export PATH="$PATH:$HOME/Library/Android/sdk/platform-tools"
adb wait-for-device

PYTHON=$PY sh rust/torch_c/pytests/run.sh
```

Without `ANDROID_SERIAL` the NNAPI tests **skip by name**, saying that this
module will not guess which shared emulator to use; without `coremltools` the
CoreML tests skip saying so. docs/devices/VULKAN3.md §6.1 is why both skip lines name
the missing thing: a skip with a false reason is counted as a pass.

The emulators here are shared with other projects. This round wrote only inside
`/data/local/tmp/bw_device`, removed every file it pushed, and **installed no
app** — docs/devices/VULKAN3.md §4's precedent for using what is already on disk
without modifying it.
