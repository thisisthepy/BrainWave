# QNN — ExecuTorch's Qualcomm backend behind an `nn.Module`, and the claim it does not make

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/npu.py DelegateModule present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/npu.py delegate_ present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/npu.py NpuModelForCausalLM present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/npu.py delegated_paths present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn.py qnn_aot_refusal present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn.py soc_targets present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn.py lower_cpu_reference present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn.py delegation_report present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn.py read_artefact present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn.py match_device present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn.py runtime_backends present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn_device.py device_soc present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn_device.py htp_stub_for present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn_device.py DEVICE_DIR present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn_device.py fastrpc_nodes present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/qnn_device.py SOC_TABLE_UNAVAILABLE present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_qnn.py test_a_real_checkpoint_still_generates_with_a_submodule_delegated present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_qnn.py test_the_qnn_module_refuses_the_very_file_the_generic_one_runs present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_qnn.py test_the_delegated_submodule_agrees_with_upstream_at_a_derived_tolerance present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_qnn.py test_no_claim_is_made_that_anything_ran_on_an_npu present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_qnn.py test_an_artefact_built_for_the_wrong_silicon_is_refused_before_it_is_pushed present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_qnn.py test_an_absent_soc_table_is_not_reported_as_an_unrecognised_chipset present -->

## 0. At a glance

| | |
|---|---|
| Can the QNN ahead-of-time lowering run on this arm64 Mac? | **No**, and §1 is the measurement rather than the doc quote |
| Where does the upstream/torchnative boundary fall? | **At the `.pte` file.** §2 |
| Does the front end survive? `from_pretrained` / `generate`? | **Yes** — SmolLM2-135M, one submodule delegated, six tokens generated (§4.1) |
| Does an artefact load and run and agree with upstream? | **Yes** — the *XNNPACK control*, at a derived tolerance, ratio 0.16 (§7) |
| Did anything run on a Hexagon NPU? | **No, and this round cannot claim it did.** §6.4 |
| Was a Snapdragon device attached? | **Not at first.** One appeared mid-round and steps 1-3 of §5 were run against it, read-only: **SM8550, HTP v73** (§5.1) |
| New tests | 19, `rust/torch_c/pytests/test_qnn.py` |
| Nullifications attempted / uncaught | **7 / 1**, and the uncaught one found a real hole (§8.3) |
| Defects found | **1** — an absent SoC table reported as an unrecognised chipset, found by running against real silicon (§5.1) |
| Rust changed | **none**, so golden is required to be exactly unmoved (§9) |

---

## 1. Step one — what ExecuTorch's QNN backend actually requires

The instruction was to answer this *before* writing code, with citations. It
was, and the answer changed the design: the lowering was always going to be
offline, and §1.3 is why it has no choice.

### 1.1 The four numbers

| | |
|---|---|
| **ExecuTorch version** | `1.4.1` — the current PyPI release, installed and used here. The QNN backend has shipped in-tree since 0.2 |
| **QNN SDK version** | **2.37.0.250724**. Read out of the installed package, not transcribed — `executorch/backends/qualcomm/scripts/download_qnn_sdk.py` resolves `QNN_VERSION` from `qnn_config.sh` and falls back to that literal. ExecuTorch's own documentation says "Although newer versions are available, we have verified and recommend using **QNN 2.37.0** for stability" |
| **Host OS for the lowering** | "The QNN Backend is currently verified on the following Linux host operating systems: **Ubuntu 22.04 LTS (x64)**, **CentOS Stream 9**, **Windows Subsystem for Linux (WSL)** with Ubuntu 22.04." macOS/Darwin appears **nowhere** in that document |
| **Android NDK** | "This example is verified with **NDK 26c**", `ANDROID_ABI=arm64-v8a`, `ANDROID_PLATFORM=android-30` (the last two from `backends/qualcomm/scripts/build.sh`) |

Plus: `g++` **13 or higher** for the AOT part, and the chipsets ExecuTorch's own
docs name as verified — "This example is verified with **SM8550 and SM8450**".

### 1.2 What must be on the device

Four kinds of file, and they go in one directory that is on both
`LD_LIBRARY_PATH` and `ADSP_LIBRARY_PATH`:

| | |
|---|---|
| the model | the `.pte`, with the QNN context binary embedded in its delegate segment |
| the ExecuTorch runtime | `qnn_executor_runner` and `libqnn_executorch_backend.so`, built for `arm64-v8a` with `-DEXECUTORCH_BUILD_QNN=ON` |
| the QNN backend | `libQnnHtp.so`, `libQnnSystem.so` — from `$QNN_SDK_ROOT/lib/aarch64-android/` |
| the HTP **stub and skeleton** for this silicon | `libQnnHtpV<arch>Stub.so` and `libQnnHtpV<arch>Skel.so` |

The last row is the one worth understanding. The **stub** runs on the
application processor; the **skeleton** is loaded *onto the Hexagon DSP itself*
over FastRPC. `ADSP_LIBRARY_PATH` exists so the DSP-side loader can find the
skel. `torchnative.export.qnn_device.htp_stub_for(arch)` derives the pair from
the architecture number rather than carrying a list, because the list grows
every silicon generation and a stale copy would silently omit the newest part.

### 1.3 Can the lowering run on an arm64 Mac? — **No**, and here is the measurement

The doc's OS list is evidence and not proof: "not verified" is a weaker claim
than "cannot". So this was measured, in a venv on the external SSD, on the
actual wheel.

```text
executorch 1.4.1, macosx_14_0_arm64 wheel, CPython 3.13

executorch/backends/qualcomm/                       present  (Python tree)
executorch/backends/qualcomm/python/                ABSENT
  -> import executorch.backends.qualcomm.python.PyQnnManagerAdaptor
     ModuleNotFoundError: No module named 'executorch.backends.qualcomm.python'

op builders that import it at module scope           112 of 112
  builders/node_visitor.py:51
      torch.int8: PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8

so, transitively:
  executorch.backends.qualcomm.partition.qnn_partitioner       ModuleNotFoundError
  executorch.backends.qualcomm.utils.utils                     ModuleNotFoundError
  to_edge_transform_and_lower_to_qnn                           unreachable
  generate_qnn_executorch_compiler_spec                        unreachable
```

`PyQnnManagerAdaptor` is a pybind11 extension **built against `$QNN_SDK_ROOT`**,
and the host libraries it links exist for one target. ExecuTorch's own
installer says so in code, not in prose:

```python
# executorch/backends/qualcomm/__init__.py
if env_flag not in ("1", "true", "yes"):
    if qnn_sdk_root_flag:  ...
    elif is_linux_x86():
        ok = install_qnn_sdk()
```

```python
# executorch/backends/qualcomm/scripts/download_qnn_sdk.py:91
def is_linux_x86() -> bool:
    return platform.system().lower() == "linux" and platform.machine().lower() in (
        "x86_64", "amd64", "i386", "i686",
    )
```

and every host-side path in that package points at one directory:

```python
sdk_lib_dir = str(qnn_sdk_dir / "lib" / "x86_64-linux-clang")          # :689
qnn_lib     = qnn_sdk_dir / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"  # :696
```

The same is true of the whole profiling toolchain — ExecuTorch's own QNN
debugger shells out to `$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator`
and `.../qnn-profile-viewer` (`backends/qualcomm/debugger/utils.py:244,323`).

**So this is not "macOS is untested". The host library that has to be dlopened
to compile a QNN context binary is an x86-64 ELF object and there is no arm64
Mach-O build of it.** A `libQnnHtp.so` cannot be loaded by a Darwin process on
Apple silicon under any configuration.

### 1.4 What *can* run here, then

This is the half that made the round deliverable rather than a report:

| | on macOS arm64 |
|---|---|
| `torch.export.export` (upstream torch 2.14.0) | **works** |
| `executorch.exir.to_edge_transform_and_lower` | **works** |
| the XNNPACK partitioner | **works** |
| `.to_executorch()` → `.pte` bytes | **works** |
| `executorch.runtime.Runtime.load_program` / `execute` | **works** |
| the `.pte` flatbuffer schema (`_deserialize_pte_binary`) | **works** |
| **the QNN option flatbuffer** (`qc_schema`, `option_to_flatbuffer`, `flatbuffer_to_option`) | **works** — pure Python, no SDK |
| `QcomChipset` / `_soc_info_table` | **works** — plain `IntEnum`s |
| anything that touches `PyQnnManagerAdaptor` | **cannot** |

The last three rows are why the artefact reader in §4.3 is fully tested here
even though nothing can produce a real QNN artefact here. The *schema* is
inspectable; only the *compiler* is not.

### 1.5 Sources

* [Qualcomm AI Engine Backend — ExecuTorch 1.3/stable](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)
  and its [markdown source](https://docs.pytorch.org/executorch/stable/_sources/backends-qualcomm.md.txt) — host OS list, QNN 2.37.0, NDK 26c, g++ 13, SM8550/SM8450, the device push list and the `LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH` run line
* [`backends/qualcomm/README.md`](https://github.com/pytorch/executorch/blob/main/backends/qualcomm/README.md) — `QcomChipset` is the authority on supported SoCs
* [`backends/qualcomm/serialization/qc_schema.py`](https://github.com/pytorch/executorch/blob/main/backends/qualcomm/serialization/qc_schema.py) — `QcomChipset`, `HtpArch`, `QnnExecuTorchBackendType`, `QnnExecuTorchHtpPrecision`, `_soc_info_table`
* [`backends/qualcomm/scripts/build.sh`](https://github.com/pytorch/executorch/blob/main/backends/qualcomm/scripts/build.sh) — `EXECUTORCH_BUILD_QNN=ON`, `ANDROID_ABI=arm64-v8a`, `ANDROID_PLATFORM=android-30`
* [pytorch/executorch#16465](https://github.com/pytorch/executorch/issues/16465) — the verbatim HTP-init failure quoted in §6.3, and `ro.soc.model` as the property QNN's SoC detection reads
* the installed wheel itself, for everything in §1.3

---

## 2. Where the upstream/torchnative boundary falls

**At the `.pte` file, and at nothing else.**

```
OFFLINE — Linux x86-64, upstream torch, upstream executorch, QNN SDK
────────────────────────────────────────────────────────────────────
  transformers model
    → torch.export.export                        UPSTREAM torch
    → to_edge_transform_and_lower_to_qnn         UPSTREAM executorch
      → QnnPartitioner + 112 op builders         UPSTREAM
      → PyQnnManagerAdaptor → libQnnHtp.so       QUALCOMM SDK
    → .to_executorch()                           UPSTREAM
    → layer0_mlp.pte  ◄── the boundary ──────────────────────────────
────────────────────────────────────────────────────────────────────
ON DEVICE — torchnative
  QnnModule(".../layer0_mlp.pte")                torchnative
    reads the delegate table, refuses if it is not QnnBackend
    ExecuTorch runtime loads it and calls forward()
  npu.delegate_(model, {"model.layers.0.mlp": that})   torchnative
  model.generate(...)                            UPSTREAM transformers
```

### 2.1 Why the boundary is there and not one step earlier

`docs/graph/EXPORT5.md` §0 is the reason, and it is a measurement rather than a
preference:

> Does `torch.export.export()` return an `ExportedProgram`? **Yes** … On how
> many modules? **4 hand-written.** On 40 `transformers` architectures:
> upstream 10, this shim **0**.

torchnative's own `torch.export` is the weakest thing in this tree. Putting it
on the critical path to a vendor NPU would make the whole Android story depend
on it, and it would fail on the first real architecture. `docs/graph/DYNAMO.md` §18
reached the same conclusion from the other side for `torch.compile`, and
`docs/graph/CAPTURE.md` §9 lists trace serialisation as *absent* — a captured trace
cannot leave the process, so there is no torchnative-native path to a `.pte`
even in principle today.

So the lowering is upstream's, it runs on a machine that is not a phone, and
what ships is a file.

### 2.2 What torchnative still owns

The boundary is narrow but it is not empty. On the torchnative side:

* **reading the artefact and deciding whether it is what it claims to be**
  (§4.3) — the delegate table, the compile spec, the SoC and HTP architecture;
* **refusing** an artefact that would run on the CPU while looking like
  success (§6.2, §8);
* **the front end** — putting the artefact behind an `nn.Module` inside a real
  `transformers` model without touching `from_pretrained` or `generate` (§3).

That third item is the part shared with the Apple and Windows back ends, and
it is deliberately in its own file (§10).

---

## 3. The front end — `torchnative/export/npu.py`

Vendor-neutral, 250 lines, and it contains no string that names Qualcomm.

```python
from torchnative.export.npu import NpuModelForCausalLM
from torchnative.export.qnn import QnnModule

model = NpuModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    plan={"model.layers.0.mlp": QnnModule("layer0_mlp.pte")},
)
model.generate(input_ids, max_new_tokens=6)
```

### 3.1 `from_pretrained` returns the model, not a wrapper

This is the one design decision in the file. `generate()` is
`GenerationMixin.generate` — several thousand lines that read `self.config`,
`self.device`, `self.can_generate()`, the cache classes,
`_prepare_generation_config`. **A wrapper has to forward all of it, and every
forward is a place to be wrong.** Returning the model means there is nothing to
forward.

`docs/graph/QUANT2.md` §3 already settled this shape for `quantize_`, and cited the
archived `intel_npu_acceleration_library.compile(model, dtype=torch.int8)` as
the precedent from the other side. This is the same move with a different
payload behind the leaf.

The test asserts it **by identity**, not by name:

```python
model.generate.__func__ is GenerationMixin.generate     # True
isinstance(model, LlamaForCausalLM)                     # True
```

A wrapper that defined its own `generate` would pass a `hasattr` check and fail
that one, which is precisely why it is written that way.

### 3.2 A plan is validated whole before anything moves

`delegate_` resolves every path in the plan before it replaces the first one.
A half-applied plan leaves a model that runs, is neither the eager model nor
the delegated one, and whose output does not say which. That is the same shape
as every failure `docs/graph/NPU2.md` records, one layer up.

`test_every_bad_plan_is_refused_by_name_and_leaves_the_model_alone` submits a
plan with one good entry and one bad one and requires
`delegated_paths(model)` to be unchanged afterwards. Nullification 1 (§8.3)
deleted the pre-validation pass and that assertion went red.

---

## 4. What was built, and which half of it ran

`docs/graph/NPU.md` §2's distinction, kept: **executed** and **structurally
validated** are different claims and this section does not merge them.

| artefact | claim | how |
|---|---|---|
| the front-end swap on SmolLM2-135M | **executed** | under the shim, real checkpoint, `generate()` produced six tokens |
| an XNNPACK-delegated `.pte` from a hand-written MLP | **executed** | `Runtime.load_program` → `execute`, compared to eager |
| an XNNPACK-delegated `.pte` from **SmolLM2's layer-0 MLP** | **executed** | same, and §7 is the numerical comparison |
| the device's SoC and FastRPC node | **read from the device** | §5.1 — `getprop`, read-only, SM8550/v73 |
| a **QNN**-delegated `.pte` | **not produced** | §1.3 — the host cannot |
| anything on an HTP | **not executed** | §6.4 — nothing to push |

### 4.1 The front end, on a real checkpoint

Under the shim (`torch 2.13.0`, `_C.abi3.so`), with SmolLM2-135M loaded from a
local HF cache and never from the network:

```text
type                       LlamaForCausalLM
isinstance LlamaForCausalLM True
generate is GenerationMixin.generate (by identity)   True
delegated                  ['model.layers.0.mlp']
delegate wraps             LlamaMLP   (576 -> 1536 -> 576)
generate(max_new_tokens=6) "The capital of France is the capital of the country."
delegate calls             6           -- once per decode step
state_dict still carries   model.layers.0.mlp.inner.*
```

The delegate here forwards to the module it replaced, because this subprocess
runs under the shim where there is no ExecuTorch runtime. **That is the point
of splitting the claims**: this row is about the *replacement*, not about what
the replacement computes, and §7 is the row about what it computes.

Six calls for six new tokens is the assertion with teeth — a delegate that had
been swapped in and then bypassed would report zero and every other line above
would be identical.

### 4.2 The lowering pipeline, executed on the control

`lower_cpu_reference` runs the **identical** pipeline the QNN path runs
(`torch.export.export` → `to_edge_transform_and_lower` → `.to_executorch()`)
with the XNNPACK partitioner in place of `QnnPartitioner`. On a two-linear MLP:

```text
delegation      1 subgraph, 4 delegated nodes / 1 not (0.80)
backends        ['XnnpackBackend']
is_qnn          False
loaded and run  yes, output [2, 8]
max abs diff vs eager   8.9e-08
```

It is **not a fallback** and the code says so twice. It exists so that a
failure in the shared machinery shows up on a machine with no Qualcomm
anything, and so that §4.3's `QnnModule` has something real to refuse.

### 4.3 The artefact reader

Everything is read through **upstream's own** schema —
`executorch.exir._serialize._deserialize_pte_binary` for the program and
`qc_schema_serialize.flatbuffer_to_option` for the QNN compile spec. There is
no second parser here that could agree with the first by sharing a mistake,
which is the reason `docs/graph/NPU2.md` §3.1 gives for `nnapi_runner.c` being a
replayer rather than a converter.

```python
art = read_artefact("layer0_mlp.pte")
art.backend_ids        # ('QnnBackend',)
art.is_qnn             # True
art.htp_plan()         # {'backend_type': kHtpBackend, 'soc_model': 57,
                       #  'htp_arch': 75, 'precision': kHtpFp16}
match_device(art, "SM8650")   # (True,  'HTP v75, backend_type kHtpBackend, ...')
match_device(art, "SM8550")   # (False, 'built for HTP v75 ... SM8550 is HTP v73')
```

**`is_qnn` is "at least one segment", not "all".** A real model legitimately
mixes — QNN for what the partitioner claimed, portable kernels for the rest —
and a reader that demanded purity would call every real model `False`. What it
must not do is call a program with **zero** QNN segments `True`, which is the
entire silent-fallback shape.

`match_device` compares the **HTP architecture**, not the part number. SM8550,
SA8255, SC8380XP, SSG2115P, SSG2125P, SXR1230P and QCS9100 are all v73 in
ExecuTorch's own table, and a context binary built for one runs on another of
that generation. Refusing a combination that works teaches the reader to stop
believing refusals.

#### The fixture, named for what it is

The reader's *positive* branch is tested against a **QNN-shaped** program, and
the test says so in its own docstring. It takes the XNNPACK control, rewrites
the delegate's `id` to `QnnBackend`, attaches a genuine `QnnExecuTorchOptions`
flatbuffer built by ExecuTorch's own serialiser, and writes it back through
ExecuTorch's own serialiser. Every byte of schema is upstream's; the payload
behind the delegate is still XNNPACK's and a real QNN backend would reject it
instantly.

It is **not a QNN artefact**. It exists because a reader tested only against
non-QNN files passes while returning `False` unconditionally — nullification 2
(§8.3) made `is_qnn` return `True` always and nine tests went red, which is the
check that the fixture is doing work in both directions.

---

## 5. The device procedure

`adb devices -l` listed **nothing** when this round started, so the steps below
were written to be run when a Snapdragon device is attached, with every one of
them reading the device rather than assuming it. **A device then appeared
mid-round**, and steps 1–3 — which are read-only — were run against it. §5.1 is
what they returned, and what running them found.

Copy-pasteable. Steps 1–3 run on the Mac, 4–8 on a **Linux x86-64** host,
9–14 back on whichever host holds the device.

```sh
# ═══ 1. Name the device. Never inferred: the Android devices here are shared.
export PATH="$PATH:$HOME/Library/Android/sdk/platform-tools"
adb devices -l                      # pick the serial for YOUR device
export ANDROID_SERIAL=<serial>      # required by every step below

# ═══ 2. Ask the device what it is. Do not assume a chipset.
adb -s "$ANDROID_SERIAL" shell getprop ro.product.cpu.abi     # must be arm64-v8a
adb -s "$ANDROID_SERIAL" shell getprop ro.soc.model           # e.g. SM8650
adb -s "$ANDROID_SERIAL" shell getprop ro.soc.manufacturer    # QTI / Qualcomm
adb -s "$ANDROID_SERIAL" shell getprop ro.board.platform      # fallback

# ═══ 3. Map it through ExecuTorch's own table, and check the DSP is there.
#     Prints the chipset name, soc_model, htp_arch and the stub/skel pair.
PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
  "$PYTHON" -c '
from torchnative.export import qnn_device as D
import json; print(json.dumps(D.device_report(), indent=2))'
#     And the FastRPC nodes the HTP is reached through. If these are absent,
#     nothing below will work and it is not a software problem:
adb -s "$ANDROID_SERIAL" shell 'ls -l /dev/adsprpc-smd /dev/cdsprpc-smd 2>&1'

# ═══ 4. On a LINUX x86-64 host: the SDK. (§1.3 -- not a Mac, not arm64.)
#     ExecuTorch pins 2.37.0.250724; `pip install executorch` fetches it
#     automatically on linux-x86 (backends/qualcomm/__init__.py), or:
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.37.0.250724
source "$QNN_SDK_ROOT/bin/envsetup.sh"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"

# ═══ 5. On the same host: ExecuTorch with the QNN backend, AOT + runtime.
export EXECUTORCH_ROOT=$PWD/executorch
export ANDROID_NDK_ROOT=$HOME/android-ndk-r26c        # NDK 26c
cd "$EXECUTORCH_ROOT" && ./backends/qualcomm/scripts/build.sh
#     -> build-x86/    the AOT half (PyQnnManagerAdaptor)
#     -> build-android/ qnn_executor_runner + libqnn_executorch_backend.so
#        (-DEXECUTORCH_BUILD_QNN=ON -DANDROID_ABI=arm64-v8a
#         -DANDROID_PLATFORM=android-30 -DEXECUTORCH_ENABLE_EVENT_TRACER=ON)
export PYTHONPATH="$EXECUTORCH_ROOT/..:$PYTHONPATH"

# ═══ 6. On the same host: lower ONE submodule. SOC_MODEL from step 2/3.
#     This is upstream torch and upstream executorch. torchnative is not in
#     this process except as the thin driver in torchnative.export.qnn.
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM
from torchnative.export import qnn

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M").eval()
mlp   = model.model.layers[0].mlp
x     = torch.randn(1, 5, model.config.hidden_size)

path, report = qnn.lower(mlp, (x,), "SM8650", "layer0_mlp.pte", fp16=True)
print("delegation:", report)            # <-- READ THIS. §6.2.
assert report["delegated_fraction"] > 0.9, report
art = qnn.read_artefact(path)
print(art.htp_plan())                   # backend_type must be kHtpBackend
print(qnn.match_device(art, "SM8650"))  # must be (True, ...)
PY

# ═══ 7. Sanity: the artefact must name QnnBackend and match the silicon.
python -c '
from torchnative.export import qnn
a = qnn.read_artefact("layer0_mlp.pte")
assert a.is_qnn, a.backend_ids
ok, why = qnn.match_device(a, "SM8650")
assert ok, why
print("artefact ok:", why)'

# ═══ 8. Copy layer0_mlp.pte, qnn_executor_runner and
#     libqnn_executorch_backend.so back to the host that holds the device.

# ═══ 9. Stage. ONE directory, and this project owns it by name.
D=/data/local/tmp/bw_qnn
ARCH=75                                   # from step 3's htp_arch. NOT a guess.
adb -s "$ANDROID_SERIAL" shell "mkdir -p $D"
adb -s "$ANDROID_SERIAL" push layer0_mlp.pte                     $D
adb -s "$ANDROID_SERIAL" push qnn_executor_runner                $D
adb -s "$ANDROID_SERIAL" push libqnn_executorch_backend.so       $D
for f in libQnnHtp.so libQnnSystem.so \
         libQnnHtpV${ARCH}Stub.so libQnnHtpV${ARCH}Skel.so; do
  adb -s "$ANDROID_SERIAL" push "$QNN_SDK_ROOT/lib/aarch64-android/$f" $D
done

# ═══ 10. Run it. ADSP_LIBRARY_PATH is how the DSP-side loader finds the skel.
adb -s "$ANDROID_SERIAL" shell "cd $D && \
  export LD_LIBRARY_PATH=$D && export ADSP_LIBRARY_PATH=$D && \
  ./qnn_executor_runner --model_path ./layer0_mlp.pte 2>&1"
#     ^ READ THE LOG, not the exit code. §6.3 is what a failure looks like
#       and §6.1 is what a success has to contain.

# ═══ 11. THE NEGATIVE CONTROL. Move the skeleton away and run it again.
#     It MUST fail. If it still succeeds, the DSP was never involved and
#     every sentence about the NPU above is unfounded. §6.1(d).
adb -s "$ANDROID_SERIAL" shell "mv $D/libQnnHtpV${ARCH}Skel.so $D/skel.hidden"
adb -s "$ANDROID_SERIAL" shell "cd $D && \
  export LD_LIBRARY_PATH=$D && export ADSP_LIBRARY_PATH=$D && \
  ./qnn_executor_runner --model_path ./layer0_mlp.pte 2>&1 | tail -20"
adb -s "$ANDROID_SERIAL" shell "mv $D/skel.hidden $D/libQnnHtpV${ARCH}Skel.so"

# ═══ 12. The per-operation answer -- QNN's MLComputePlan (§6.1(b)).
#     Needs the runner built with -DEXECUTORCH_ENABLE_EVENT_TRACER=ON and the
#     artefact lowered with profile_level=QnnExecuTorchProfileLevel.kProfileOptrace.
adb -s "$ANDROID_SERIAL" shell "cd $D && \
  export LD_LIBRARY_PATH=$D && export ADSP_LIBRARY_PATH=$D && \
  ./qnn_executor_runner --model_path ./layer0_mlp.pte --etdump_path etdump.etdp"
adb -s "$ANDROID_SERIAL" pull $D/etdump.etdp .
adb -s "$ANDROID_SERIAL" pull $D/output/qnn-profiling-data_0.log .
#     Back on the LINUX host, decode it with the HTP-specific reader:
"$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer" \
  --reader "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so" \
  --input_log qnn-profiling-data_0.log --output optrace_0.json
#     -> optrace_0_qnn_htp_analysis_summary.json  is HTP hardware counters.
#        If the graph did not execute on the HTP, that reader has nothing to read.

# ═══ 13. Numbers. Compare element-wise against upstream torch on CPU, at the
#     tolerance derived in §7 -- and read §7.2 first if fp16/quantised.

# ═══ 14. LEAVE IT AS YOU FOUND IT. Shared device.
adb -s "$ANDROID_SERIAL" shell "rm -rf $D"
adb -s "$ANDROID_SERIAL" shell "ls -d $D 2>&1"   # must say No such file
#     Nothing was installed, so there is nothing to uninstall.
```

Steps 1, 3, 9 and 14 are what `torchnative.export.qnn_device` automates;
`device_report()` is step 2+3 and `cleanup()` is step 14.

### 5.1 Steps 1–3, executed — and the defect they found

A Samsung Galaxy Tab S9 Ultra attached itself to this machine partway through
the round. It is **shared**, so only the read-only steps were run against it:
`getprop`, and `ls -d` on named paths. **Nothing was pushed, nothing was
installed, and `/data/local/tmp/bw_qnn` was verified absent before and after.**

```text
ro.product.cpu.abi        arm64-v8a
ro.soc.model              SM8550          <- Snapdragon 8 Gen 2
ro.soc.manufacturer       QTI
ro.board.platform         kalama          <- SM8550's codename
ro.hardware               qcom
ro.product.model          SM-X910
ro.build.version.release  16   (sdk 36)

/dev/adsprpc-smd          crw-rw-r-- system system 490, 0    <- present
/dev/cdsprpc-smd          No such file or directory
/data/local/tmp/bw_qnn    No such file or directory          <- clean, before and after
```

Through `device_report()`, on an interpreter that has ExecuTorch:

```text
soc_status      known
chipset         SM8550
soc_model       43
htp_arch        73
htp_libraries   libQnnHtp.so  libQnnSystem.so
                libQnnHtpV73Stub.so  libQnnHtpV73Skel.so
fastrpc         ('/dev/adsprpc-smd',)
staged          ()          staged_complete  False
```

**So the target this round would lower for is real, identified, and not
guessed**: SM8550, HTP **v73**. That is one generation below the SM8650/v75 the
§4.3 fixture uses, which is why `match_device` refusing an SM8650 artefact for
an SM8550 device is not a hypothetical — it is the exact mismatch a careless
copy of the tutorial's `--soc_model SM8650` would produce against *this* device.

#### The defect

Run from the shim interpreter — the one with no ExecuTorch — `device_report()`
returned **`chipset: null`** for a device whose `ro.soc.model` is `SM8550`, a
chipset ExecuTorch knows perfectly well. `resolve_soc` needs ExecuTorch to read
`QcomChipset`, could not import it, raised, and `device_soc` caught that and
rendered it as "the table does not have this part".

**A refusal about the host wearing a verdict about the device.** The two lead a
reader to opposite actions: one says `pip install executorch`, the other says
this silicon will never work — which is what pytorch/executorch#16465 *actually*
describes, a device reporting `CQ8750S` on which QNN's own detection later
fails with "No Snapdragon SOC detected".

Fixed by making the outcome three-valued — `SOC_KNOWN`, `SOC_NOT_IN_TABLE`,
`SOC_TABLE_UNAVAILABLE` — and deciding between the last two by asking whether
`soc_targets()` can be read at all, rather than by parsing the message. The
raw string is carried in every case, because it is the only thing in the report
that came from the device.

`test_an_absent_soc_table_is_not_reported_as_an_unrecognised_chipset` asks
**both** interpreters the same question about the same string and requires the
answers to differ. Neither half catches it alone, which is exactly why the
first version passed everything (§8.3, nullification 7).

#### And one smaller thing

`ls /dev/ | grep rpc` returns an **empty listing** to the `shell` user on
Android 16 / API 36, on a device where `ls -l /dev/adsprpc-smd` succeeds. A
FastRPC check built on the directory listing would report "no DSP" on a device
that has one — a false negative about the only piece of hardware this round is
about. `fastrpc_nodes()` stats named paths instead.

---

## 6. The verification trap

This is the section the round exists for. `docs/graph/NPU2.md` is this project's
record of paying for the mistake twice — the CoreML models `docs/graph/NPU.md`
called "executed" had run on the **CPU**, and every NNAPI driver on the
emulator was software. **The results were correct either way.**

So: what output proves HTP execution, and what does a silent fallback look
like instead?

### 6.1 What would prove it

Four things, and the first two are necessary but not sufficient on their own.

**(a) AOT — the partition, before anything is pushed.**
`delegation_report(edge)` gives `delegated_nodes / total_nodes`. This is the
*ceiling* on how much could have run on the NPU. It is not evidence that
anything did; it is evidence about what could not have.

**(b) AOT — the artefact, decoded.** `read_artefact(path).htp_plan()` must
report `backend_type == kHtpBackend` and an `htp_arch` matching the device's.
An HTP context binary is compiled *for one HTP generation*, and QNN's CPU
reference implementation cannot execute one. So this is a strong structural
claim — and still a claim about a file.

**(c) RUNTIME — the optrace / QHAS summary.** This is the closest QNN analogue
of `MLComputePlan`, and it is the one that answers "which unit". Lower with
`profile_level=kProfileOptrace`, run, pull `qnn-profiling-data_0.log`, and
decode it with **`libQnnHtpOptraceProfilingReader.so`**. The output,
`optrace_0_qnn_htp_analysis_summary.json`, is a decode of **HTP hardware
counters**. A graph that ran anywhere else produces nothing for that reader to
read. Step 12 above is the procedure.

**(d) RUNTIME — the negative control, and this is the one that makes it
evidence rather than a plan.** Move `libQnnHtpV<arch>Skel.so` out of
`ADSP_LIBRARY_PATH` and run again. The skeleton is the code that executes *on
the DSP*; without it the backend cannot initialise and the run **must fail**.
If it still succeeds and returns the same numbers, the DSP was never involved.

This is exactly the argument `docs/graph/NPU2.md` §2 makes with its third row — two
runs that are *identical* are what it looks like when the accelerator plan is
being ignored — transplanted to a backend where the tell is a hard failure
rather than a difference.

### 6.2 What the silent fallback looks like

QNN's fallbacks are not where NNAPI's and CoreML's are, and getting this
backwards would send a reader to the wrong log.

| layer | silent? | what it looks like |
|---|---|---|
| **AOT partition** | **YES — this is the dangerous one** | `QnnPartitioner` tags what it can. Everything it declined stays in the program as **portable CPU kernels**. The `.pte` loads, runs, and returns the right answer. A model reporting 2 delegated nodes of 900 ran on the CPU and **nothing in its output says so.** |
| **AOT backend type** | **semi-silent** | a compile spec with `kCpuBackend` or `kGpuBackend` still produces a `QnnBackend` delegate. It still delegates. It still gets the right answer. It never goes near the NPU. This is the fallback that looks most like success, which is why `match_device` refuses it by name and why nullification 3 (§8.3) was run against exactly that check. |
| **AOT precision** | semi-silent | `kHtpQuantized` is the *default*. A float32 comparison against a quantised artefact is meaningless — §7.2. |
| **runtime HTP init** | **NO — loud, and fail-closed** | §6.3 |

**So the trap for QNN is at compile time, not at run time.** The number to
read is `delegated_fraction`, and step 6 of §5 asserts on it rather than
printing it.

### 6.3 The runtime does not fall back, and that is citable

Verbatim from [pytorch/executorch#16465](https://github.com/pytorch/executorch/issues/16465),
a device whose `ro.soc.model` QNN did not recognise:

```text
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> No Snapdragon SOC detected
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Failed to init router
[ERROR] [Qnn ExecuTorch]: Failed to create backend_handle for Backend ID 6, error=4000
F executorch:qnn_executor_runner.cpp:312]
  In function main(), assert failed (method.ok()):
  Loading of method forward failed with status 0x1
Aborted
```

**The method load fails and the process aborts. There is no CPU fallback.**
The same check catches a mismatched architecture and says so by number —
`QnnDsp Arch 68 set by custom config is different from arch associated with
SoC 57` — which is what `match_device` exists to catch one step earlier.

This is a genuinely better property than NNAPI's, where `nnapi-reference` is
a perfectly good software driver that answers correctly, and than CoreML's,
where the planner simply prefers the CPU and says nothing.

### 6.4 **This round cannot make the claim**

Three of the four things in §6.1 were not observed, and the fourth was
observed only for a non-QNN artefact:

* **no QNN artefact was produced** — §1.3, the host cannot compile one;
* **no QNN artefact reached a device.** A Snapdragon device did appear
  (§5.1, SM8550/v73) and was identified read-only, but there was nothing to
  push to it and nothing was pushed;
* **`QnnBackend` is not registered** in any ExecuTorch runtime reachable from
  here: `Runtime.get().backend_registry.registered_backend_names` is
  `['CoreMLBackend', 'VgfBackend', 'MLXBackend', 'XnnpackBackend']`.

**So nothing in this round ran on a Hexagon NPU, and no sentence here says
otherwise.** `test_no_claim_is_made_that_anything_ran_on_an_npu` asserts the
absence rather than leaving it to prose, and it goes red the day the AOT half
becomes available on the host — so the sentence gets rewritten deliberately
instead of inherited, which is the failure mode `docs/verification/DOCWATCH.md` exists for.

What is *not* missing is the path: §5 is the procedure, and every piece of it
that does not need Qualcomm hardware is implemented and tested.

---

## 7. The numbers

### 7.1 The tolerance is derived, not chosen

`docs/numerics/AGREE.md` §2's rule, applied to one module instead of 263
architectures: upstream runs the **same submodule** in float64, the float32
answer is scored against it, and the tolerance is the **p90 of that error,
floored at 8 ulp**. Its defence is that anything tighter would have to call
upstream wrong on a tenth of its own samples.

SmolLM2-135M, `model.layers.0.mlp` (`LlamaMLP`, 576 → 1536 → 576), lowered
through `torch.export` → `to_edge_transform_and_lower` → `.to_executorch()`,
loaded by `Runtime.load_program` and executed, on five inputs the export never
saw:

```text
delegation      1 subgraph, 6 delegated nodes / 1 not
artefact        10,619,776 bytes, backends ['XnnpackBackend']

              rel(artefact vs upstream f32)   upstream f32-vs-f64   ratio
  sample 0            1.329e-07                    8.361e-07        0.16
  sample 1            1.723e-07                    1.197e-06        0.14
  sample 2            1.184e-07                    7.846e-07        0.15
  sample 3            1.980e-07                    1.245e-06        0.16
  sample 4            1.190e-07                    9.768e-07        0.12

float32 eps  1.192e-07      8 ulp floor  9.537e-07
tolerance    p90 of the oracle column, floored at 8 ulp   =  1.23e-06
worst rel    1.980e-07                                    -- 6x inside it
worst ratio  0.16                                         -- AGREE.md's rule allows 4.0
```

The ratio column is the interesting one and it is the same shape
`docs/numerics/AGREE.md` §3 found across 285 architectures: **below 1 means the
artefact's float32 answer is nearer the float64 truth than upstream's own
float32 path is.** It is here, on all five samples, by roughly six-fold. That
is what two independent float32 truncation orders look like; it is not a
claim that the artefact is "better".

The test also asserts `min(oracle) > 0` — an upstream error of exactly zero
would drop the tolerance to the 8-ulp floor and make the derivation
decorative, which is CLAUDE-file §5.5's check-that-cannot-fail wearing the
tolerance's clothes.

### 7.2 What this comparison can and cannot mean

**It cannot mean anything about QNN.** It is the XNNPACK artefact, on this
Mac's CPU, in float32. What it establishes is that the shared
export/lower/serialise/load/execute pipeline **preserves the function** —
which is a real and necessary result, and is not the result the round was
after.

**And the QNN comparison, when somebody runs §5, will not be a float32
comparison at all.** ExecuTorch's own schema names the two HTP modes:

```python
class QnnExecuTorchHtpPrecision(IntEnum):
    kHtpQuantized = 0      # <- the default
    kHtpFp16      = 1
```

The Hexagon Tensor Processor is not float32 hardware. So:

* with `kHtpFp16`, the honest tolerance is a **float16** tolerance. `docs/graph/NPU2.md`
  §2 measured the analogous CoreML number at 2.0e-04 against 3.0e-08 for
  float32 — four orders of magnitude — and called it "the honest tolerance for
  an NPU claim here, not a regression". The same applies.
* with `kHtpQuantized` — the default — the numbers are **quantised**, the
  artefact needs a calibrated quantizer that this round did not build, and a
  float32 element-wise comparison is not merely loose but *category-wrong*.
  `docs/graph/QUANT2.md` §2.1 is the precedent: quantisation cannot go on this
  repository's bit-exactness axis, and a separate axis has to be built for it.

**So a float32 numerical claim and an HTP execution claim exclude each other
on one run,** exactly as `docs/graph/NPU2.md` §1.1 found for the Neural Engine, and
for the same physical reason. A round that wants both needs two runs and has
to say which is which.

`compiler_spec(..., fp16=True)` is the default here for that reason: it is the
mode where a numerical claim is *bounded* rather than *absent*.

---

## 8. Refusals, and the one that got away

### 8.1 Everything unsupported refuses by name

Sixteen refusals, each carrying the name of the thing that is wrong. A refusal
that says only "unsupported" sends the reader to the wrong place.

| refusal | names |
|---|---|
| the QNN AOT half is missing | `executorch.backends.qualcomm.python.PyQnnManagerAdaptor`, `x86_64-linux-clang`, `is_linux_x86()` |
| `lower()` on such a host | the above, **before** touching the module |
| unknown SoC name | the name, and the 20 it would have accepted |
| a SoC with no `_soc_info_table` entry | the chipset, and that its HTP arch is unknown |
| artefact is not an ExecuTorch program | the file, and the decode error |
| artefact has no `QnnBackend` delegate | the backends it *does* have |
| `QnnBackend` segment with no compile spec | `qnn_compile_spec`, and that the QNN partitioner did not write this |
| artefact built for the wrong HTP arch | both architectures, and that QNN does not fall back |
| artefact whose backend type is not HTP | `kGpuBackend` / `kCpuBackend` by name |
| `QnnBackend` not registered in this runtime | the backends that are |
| `require_runtime=False` handle asked to execute | that it is an inspection handle |
| submodule path not found | what *is* there at the point it stopped |
| plan entry that is not a `DelegateModule` | the type it got |
| empty plan / empty path / non-dict plan | each by name |
| a `DelegateModule` with no `backend_name` | the class |
| `adb` with no `ANDROID_SERIAL` | the variable, and that the devices are shared |
| `htp_stub_for(None)` | that there is no default safe to invent |
| no SoC property set on the device | all three property names, and that guessing produces an artefact QNN refuses at init |

**No existing refusal was weakened.** `nnapi.py`, `nnapi_device.py`,
`coreml.py`, `target.py`, `refold.py`, `fuse.py` and `decompose.py` are
byte-identical to develop; `git status --short` in §9 shows the only tracked
file this round modified is `export/__init__.py`, and that change is a
docstring paragraph.

### 8.2 In particular, NNAPI is untouched

`nnapi.py` works, it is tested, and Android 10–14 devices are real. Nothing
here replaces it. The two coexist by design: NNAPI reaches whatever driver the
platform offers on an ordinary Android device, and QNN reaches the Hexagon NPU
on Qualcomm silicon at the cost of an offline Linux build. `docs/graph/NPU2.md` §5
notes NNAPI is deprecated for *new development*; deprecated is not gone, and
this round adds a second road rather than closing the first.

### 8.3 Nullification: 7 attempted, 1 uncaught

The repository's standing rule — a verification that cannot fail is not a
verification — applied by breaking the code on purpose. Sources were backed up
with `cp` first and restored after; `git status --short` confirmed the restore.

| # | fault injected | tests that went red |
|---|---|---|
| 1 | `delegate_` stops pre-validating the whole plan | 1 — `..._leaves_the_model_alone` |
| 2 | `PteArtefact.is_qnn` returns `True` always | **9** |
| 3 | `match_device` stops comparing the HTP arch | 1 — `..._wrong_silicon...` |
| 4 | `htp_plan` reads `htp_arch` from `soc_model` | 2 |
| 5 | `qnn_aot_refusal` stops checking the adaptor | **0 — NOT CAUGHT** |
| 6 | `QnnModule` stops checking the runtime registry | 1 |
| 7 | `device_soc` collapses `SOC_TABLE_UNAVAILABLE` into `SOC_NOT_IN_TABLE` | 1 — and this is the fault §5.1 found in the wild, so the test was written *from* it |

**Nullification 5 is the useful one.** Deleting the check that names
`PyQnnManagerAdaptor` changed nothing, because the function then fell through
to the *next* import check (`qnn_partitioner`), which also fails — so
`qnn_aot_available()` stayed `False`, `lower()` still refused, and every
assertion still passed. The sentence §1.3 leads with had simply stopped being
produced and no test noticed.

The reason it was invisible is worth writing down: the test that asserts the
adaptor is named reads the **shim** fixture, and the shim interpreter has no
executorch at all, so the refusal there is about the *package* and the branch
that names the adaptor is unreachable from it. The fix reads
`qnn_aot_refusal()` on the interpreter that *does* have executorch — the only
one where the distinction is observable — and requires `PyQnnManagerAdaptor`,
`x86_64-linux-clang` and `is_linux_x86` to all appear. Re-run with the same
fault: **caught**.

This is `docs/graph/EXPORT5.md` §11's finding repeating — the uncaught nullification
is worth more than the five caught ones, because the five confirmed tests that
were already working and the one found a test that was not.

---

## 9. Gates

```text
rust/torch_c/pytests/run.sh    1120 ok, 0 FAIL, exit 0     (1101 on develop; +19)
cargo test --release           30 passed, 0 failed
DOCWATCH                       PASS -- 1028/1028 evaluated marker(s) hold
                                                           (1006 on develop; +22)
tools/golden/compare.py        11420/11420 cases passed, 0 failed,
                               ops covered=302, pending case builders=0
```

**Golden is exactly unmoved**, which is the correct result and the check that
this round did what it says: no Rust changed, no op was added to
`_aten_implemented()`, and no case builder was needed in `tools/golden/cases.py`.
The two counts that moved are the two this round added to — 19 tests and 22
DOCWATCH markers — and both moved by exactly the amount added.

The gate was run with `TORCHNATIVE_QNN_PYTHON` set, so the ExecuTorch tests ran
rather than skipping. Measured without it: **eight skip entirely and two skip
their ExecuTorch half**, all ten naming `TORCHNATIVE_QNN_PYTHON`. The `ok`
count is unchanged at 19, because a skip is a pass here — which is exactly why
each skip line has to name the missing thing (docs/devices/VULKAN3.md §6.1: a skip
with a false reason is counted as a pass).

`git status --short`:

```text
 M torchnative/src/main/torchnative/export/__init__.py
?? docs/devices/QNN.md
?? rust/torch_c/pytests/test_qnn.py
?? torchnative/src/main/torchnative/export/npu.py
?? torchnative/src/main/torchnative/export/qnn.py
?? torchnative/src/main/torchnative/export/qnn_device.py
```

One tracked file modified, `+10` lines, all of them a docstring paragraph
(`git diff --stat`: `1 file changed, 10 insertions(+)`). No Rust, no
`bootstrap.py`, no `overloads.json`, no `tools/golden/`, and nothing under
`torchnative/src/main/torch/` touched by hand.

---

## 10. What this expects to share with the Intel NPU round

`/Volumes/macMini/worktrees/bw-intelnpu` is building the same front-end shape
for Intel's NPU concurrently. That worktree was **read, never written**. What
follows is what this round expects the coordinating session to reconcile.

**The one file that should be shared is `torchnative/export/npu.py`.** It is
250 lines, names no vendor, and holds exactly four things: `DelegateModule`
(an `nn.Module` whose forward is somebody else's artefact, with a `refuse` that
always names the backend), `delegate_` (whole-plan validation, then in-place
replacement, returns the same object), `delegated_paths`, and
`NpuModelForCausalLM.from_pretrained` (twelve lines, returns the real model).

The Intel round's public surface as read on 2026-09-07 has `compile_module`,
`quantize_`, `probe`, `supported_ops`, `available_devices`,
`assert_execution_device` and `npu_available` in `export/intelnpu.py`. Three of
those line up with names here and the alignment is deliberate:

| this round | Intel round | same question |
|---|---|---|
| `qnn.probe()` | `intelnpu.probe()` | what can this host do, measured |
| `qnn.match_device(art, soc)` + §6 | `intelnpu.assert_execution_device(..., expect="NPU")` | **where would/did it run** |
| `qnn.QnnModule` | `intelnpu.compile_module` | the vendor's leaf |
| `npu.delegate_` / `NpuModelForCausalLM` | `intelnpu.quantize_` / `compile_module` | the front-end swap |

**The concrete suggestion**: have `intelnpu.compile_module` return a
`DelegateModule` subclass and let `npu.delegate_` do the swapping, rather than
each back end growing its own replacement walk. The two rounds would then share
one front end and one refusal vocabulary, and the difference between them would
be exactly the thing that should differ — which vendor artefact sits behind the
leaf.

**Expected conflict, and it is small**: both rounds edited the module docstring
of `torchnative/export/__init__.py`. The Intel round inserted a paragraph
*before* the `nnapi_device` one and renumbered "third" to "fourth"; this round
inserted *after* it and used no ordinal, precisely so the hunks are disjoint and
the count does not have to be renumbered a third time. Nothing else in this
round touches a tracked file.

---

## 11. What is not done

* **No QNN artefact exists.** §1.3. Everything from `to_edge_transform_and_lower_to_qnn`
  onward is untested against a real partitioner, so the numbers in §5 step 6
  (`delegated_fraction` on a real `LlamaMLP` under `QnnPartitioner`) are
  **unknown**. They may be poor: QNN's 112 op builders are not the same set as
  XNNPACK's, and `docs/graph/REFOLD.md` §1.1 is this project's record of assuming a
  bigger table meant better coverage and being wrong three times.
* **Nothing ran on an NPU.** §6.4. The device that appeared is the right one —
  SM8550, HTP v73, `/dev/adsprpc-smd` present — and it is not what was missing.
  What was missing is the artefact, and producing one needs a Linux x86-64 host
  this round did not have.
* **The quantised path was not built.** `kHtpQuantized` is QNN's default and
  the interesting one for a phone; it needs a calibrated quantizer
  (`executorch.backends.qualcomm.quantizer`) and, per §7.2, its own numerical
  axis. `docs/graph/QUANT2.md` §2 is what that axis has to look like.
* **`lower()` has never executed.** It refuses on every host reachable from
  here, so its body — the four-line `torch.export` →
  `to_edge_transform_and_lower_to_qnn` → `to_executorch` sequence — is
  **wired, not run**. That is stated plainly rather than implied by the tests
  passing.
* **`qnn_device` reads but never writes.** `device_report()` and `fastrpc_nodes()`
  ran against a real SM8550 (§5.1) and found a defect doing it. `cleanup()` —
  a `rm -rf` of one directory — has **not** run, because nothing was ever
  staged: the device is shared, this round had no QNN artefact to push, and
  `/data/local/tmp/bw_qnn` was verified absent rather than created and removed.
* **No whole model, only one submodule.** `model.layers.0.mlp` is the smallest
  thing that makes the claim real; a decoder layer, an attention block or the
  whole stack are each a larger question, and attention in particular runs into
  the KV cache, which `docs/graph/CAPTURE.md` §9 already names as the wall mutation
  hits first.
* **No timing, anywhere.** Three other rounds were building on this machine
  and load average was above 8 on 8 cores throughout. `docs/devices/MPS.md` and
  `docs/devices/VULKAN3.md` both record why a throughput number measured like that is
  worse than no number.

---

## 12. Reproducing

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-qnn
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export TORCH_C_STAGE=/tmp/stage-qnn
PY=/Volumes/macMini/caches/spike-venv/bin/python

sh vendor/vendor_torch.sh            # fresh worktree only
PYTHON=$PY bash vendor/install_shim.sh

# The ExecuTorch half needs its own interpreter. It must NOT be spike-venv:
# executorch pulls torch 2.14.0 and spike-venv's torch 2.13.0 is the upstream
# oracle every numerical claim in this repository is measured against.
$PY -m venv /Volumes/macMini/caches/qnn-venv
/Volumes/macMini/caches/qnn-venv/bin/pip install executorch py-cpuinfo transformers
export TORCHNATIVE_QNN_PYTHON=/Volumes/macMini/caches/qnn-venv/bin/python

PYTHON=$PY sh rust/torch_c/pytests/run.sh
```

Without `TORCHNATIVE_QNN_PYTHON`, eight tests **skip by name** and two skip
their ExecuTorch half, all ten saying that the variable is unset; without a
local SmolLM2-135M the checkpoint tests skip saying so. `docs/devices/VULKAN3.md` §6.1 is why every skip line
names the missing thing: a skip with a false reason is counted as a pass.

The step-1 measurement of §1.3, on its own:

```sh
$TORCHNATIVE_QNN_PYTHON -c '
import platform, importlib
print(platform.system(), platform.machine())
for m in ("executorch",
          "executorch.backends.qualcomm",
          "executorch.backends.qualcomm.python.PyQnnManagerAdaptor",
          "executorch.backends.qualcomm.partition.qnn_partitioner"):
    try:
        importlib.import_module(m); print("ok    ", m)
    except Exception as e:
        print("REFUSE", m, "--", type(e).__name__, e)'
```
