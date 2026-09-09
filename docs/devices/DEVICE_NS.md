# `torchnative.device` — a namespace this project owns, and what each name resolves to

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/__init__.py Availability present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/__init__.py NpuResolution present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/__init__.py EagerDevice present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/__init__.py CompiledDevice present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/__init__.py EagerUseRefused present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/__init__.py NpuUnresolved present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/__init__.py NPU_CANDIDATES present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/_module_to.py make present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/device/_module_to.py install present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_devicens.py test_mps_availability_is_measured_not_declared present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_devicens.py test_npu_resolves_differently_per_host present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_devicens.py test_npu_never_resolves_to_the_cpu present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_devicens.py test_to_is_unchanged_for_every_ordinary_argument_form present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_devicens.py test_ordinary_calls_reach_upstream_with_identical_arguments present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_devicens.py test_cuda_keeps_its_five_named_reasons present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_devicens.py test_npu_refuses_to_be_a_tensor_destination present -->

## 0. What this is

    import torchnative
    torchnative.device.cpu · .mps · .vulkan · .cuda · .npu

A device namespace **this project owns**, so that it can carry meaning this
project can honour. It is not a PyTorch device type and does not try to become
one.

The README's roadmap argued the shape out; this document records what was
built, what each name resolves to on which host, which probe answers each
availability question, and — in §6 — the one place the implementation
deliberately disagrees with a function it was told to reuse, because that
function is wrong.

## 1. Why not `torch.device("npu")`

PyTorch has no `npu` device type. Making it appear to have one would be a claim
**about PyTorch** that is not true, and the label would then be accepted by
every spelling that takes a device — `torch.empty`, `Tensor.to`, a
`Generator` — without any of them being able to honour it.

The two mechanisms that would do it are both deliberately untaken:

* `torch.utils.rename_privateuse1_backend` → `_C._rename_privateuse1_backend`.
  [`DEVICE_ABS.md`](DEVICE_ABS.md) §7.3 records it as unimplemented, with "no
  measured demand" as the reason. This namespace is why that stays true: the
  demand it would have served is served here instead, honestly.
* Registering a PrivateUse1 backend. Same objection, plus it would put an
  `npu` label into `DEVICE_TYPES` where the eager dispatcher would then have to
  refuse it op by op — an NPU cannot take single ops at all (§3).

The namespace is ours, and a name in it means what we say it means.

## 2. The members, and what each resolves to per host

| name | kind | resolves to | availability probe |
|---|---|---|---|
| `cpu` | eager | the interpreter's own CPU | unconditional |
| `mps` | eager | Apple Metal, via candle's Metal backend | **an allocation** (§6) |
| `vulkan` | eager | the Vulkan device the loader offers | `torch._C._vulkan_probe` |
| `cuda` | eager | candle's CUDA backend | `torch._C._cuda_probe` |
| `npu` | **compiled** | **per host — see below** | resolution, §4 |

`npu` is the only name whose meaning depends on the machine:

| host | backend | unit | probe |
|---|---|---|---|
| macOS (`darwin`) | `coreml` | **Apple Neural Engine** | `coremltools.models.compute_device.MLComputeDevice.get_all_compute_devices` |
| Windows | `openvino` | **Intel NPU** | `torchnative.export.intelnpu.npu_available` / `available_devices` |
| Android | `qnn` | **Qualcomm Hexagon NPU** | `torchnative.export.qnn_device.device_report` |
| anything else | — | — | refuses by name |

Measured on this host (arm64 macOS): `npu` resolves to the **Apple Neural
Engine**, and `MLComputeDevice.get_all_compute_devices()` lists
`MLNeuralEngineComputeDevice`, `MLGPUComputeDevice`, `MLCPUComputeDevice`.

### 2.1 `npu` must say which one, and must never mean the CPU

[`../graph/NPU2.md`](../graph/NPU2.md) §1 is the reason this is a rule rather
than a nicety. Three CoreML graphs were recorded as **executed** — compiled,
run, agreeing with replay to 2–3e-08 — and every word of that was true. It was
also not the sentence "ran on the NPU": `MLComputePlan` showed all three
running on the **CPU**, with the Neural Engine not even in the supported
column.

So `NpuResolution` cannot be constructed without a `backend`, a `unit` **and** a
`source`, and no resolver may return a CPU unit. `resolve()` raises
`NpuUnresolved` — by name, naming the unit it was looking for — rather than
falling back. `test_npu_never_resolves_to_the_cpu` and
`test_npu_refuses_by_name_on_a_host_with_no_npu_path` hold that down.

**`host()` is overridable** through `TORCHNATIVE_DEVICE_HOST`. That exists so
the per-host table can be exercised for hosts this machine is not: without it,
a resolution that ignored the host entirely would answer "Apple Neural Engine"
here and never be asked anything else, and would pass every other test in the
file. `test_npu_resolves_differently_per_host` is the one that fails, and it was
verified by hardcoding the host and watching it go red (§7, N2).

## 3. Eager and compiled are different types, not a flag

`cpu`, `mps`, `vulkan` and `cuda` are `EagerDevice`: they dispatch operator by
operator and they are **tensor destinations**. `npu` is a `CompiledDevice`: an
NPU is handed a whole subgraph ahead of time and cannot dispatch a single
operator (CLAUDE.md §8), so it is not a destination at all.

The difference is expressed as **the absence of an attribute**:

    torchnative.device.mps.torch_device   -> torch.device('mps')
    torchnative.device.npu.torch_device   -> EagerUseRefused

`EagerUseRefused` names the device and says to use `model.to(...)` instead.
`torch.empty(2, 2, device=torchnative.device.npu)` also refuses, from the shim's
own overload resolution, because an `NpuDevice` is not a `torch.device`.

**Why the objects are not `torch.device` subclasses.** Measured: `torch._C.device`
is not an acceptable base type, and `torch.empty` does not stringify unknown
objects — a custom object with `__str__` returning `"mps"` is refused, and
`__torch_function__` is not consulted for the device argument. So an eager
torchnative device converts through `.torch_device`, and `nn.Module.to` takes
the object directly because §5 teaches it to.

## 4. Availability is measured, and every answer says what measured it

`Availability` carries `available`, a **named** `reason` when it is not, the
`source` that produced the answer as a string the reader can go and call, and
`kind` ∈ {`measured`, `declared`}. It refuses to be constructed unavailable
without a reason, or available carrying one.

`cuda` passes `_cuda_probe`'s `reason` through **unedited**, so the five names
— `not_built`, `no_driver`, `no_device`, `wrong_arch`, `unclassified` — survive
into the namespace. Nothing here collapses them; `test_cuda_keeps_its_five_named_reasons`
compares against `_shim_cuda_refusal_reasons()` and against the probe's own
answer. `vulkan` keeps refusing without a loader (`no_loader`, carrying the
loader's own error text), and the Intel path keeps refusing without the
OpenVINO runtime. **No existing refusal was weakened to make this namespace
work.**

Measured on this host:

| device | available | reason | source |
|---|---|---|---|
| `cpu` | yes | — | unconditional |
| `mps` | **yes** | — | `torch.empty(2, 2, device="mps")` |
| `vulkan` | no (without the loader) | `no_loader` | `torch._C._vulkan_probe` |
| `cuda` | no | `not_built` | `torch._C._cuda_probe` |
| `npu` | yes | — | CoreML compute-device list |

## 5. Where `nn.Module.to` is intercepted, and the proof upstream is unchanged

### 5.1 The interception point, and why it can be nowhere else

Upstream's `to` (`torch/nn/modules/module.py`, the `to` at line 1254) begins:

    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
    ...
    def convert(t): ... return t.to(device, ...)
    return self._apply(convert)

Two measured consequences:

* `_parse_to` is the **first statement** and knows only `torch.device`
  spellings. A torchnative device reaching it is a `TypeError` about argument
  combinations — a refusal, but not an interception.
* `_apply(convert)` descends to **tensors**. A compiled target is not a tensor
  destination, so there is nothing for `convert` to do with one.

So the interception is a wrapper **around** `Module.to`, ahead of `_parse_to`,
installed by `torchnative/device/_module_to.py`.

### 5.2 Why it is installed from `torchnative.device` and not from `bootstrap.py`

`bootstrap.py` is baked into `torch._C` and runs while `torch/__init__.py` is on
its first lines — **`torch.nn` does not exist yet**, so there is nothing to
patch. `_module_to.install()` is called at the bottom of
`torchnative/device/__init__.py` instead.

That placement is sufficient, not merely convenient: a torchnative device object
**cannot exist** without importing `torchnative.device`, so there is no call
site that can reach `to(a torchnative device)` with the patch not installed.
That is a stronger guarantee than a `sys.meta_path` hook, and it needs no entry
in it.

### 5.3 What `to()` does with each kind

| argument | behaviour |
|---|---|
| anything with no torchnative device | `return original(self, *args, **kwargs)` — untouched |
| an **eager** torchnative device | swap the label for `.torch_device`, then upstream's own code path: parameters move, `_apply` descends, `self` is returned |
| the **compiled** target | resolve the NPU first, then refuse by name (§5.5) |
| two torchnative devices | `TypeError` — a module has one device |
| `to(npu, dtype)` | `TypeError` — a compiled target is not a conversion |

**`to()` always returns `self`.** Never a wrapper. `optimum` returns an
inference object and that is why it cannot backprop; this project ships its own
`torch`, so it does not have to. If `_module_to` ever returns something that is
not `self`, the only thing distinguishing this project from `optimum` is gone —
`test_to_an_eager_torchnative_device_moves_the_parameters` asserts identity.

### 5.4 The proof that upstream semantics are unchanged

Two tests, because one of them was not enough and nullification proved it.

**Differential, on results.** `test_to_is_unchanged_for_every_ordinary_argument_form`
runs each ordinary form through the patched `nn.Module.to` **and** through the
original captured at install time (`to._torchnative_original`), on two
identically seeded modules, and requires the resulting parameter and buffer
devices, dtypes and shapes to agree:

    to("cpu")            to(torch.device("cpu"))   to(torch.float16)
    to(torch.float64)    to(device, dtype)         to(device=…, dtype=…)
    to(other_tensor)     to("cpu", non_blocking=True)
    to(dtype=…, non_blocking=…)                    to(memory_format=…)
    to("mps")            (when Metal is available)

`test_to_raises_the_same_errors_upstream_does` extends it to error behaviour:
an integer dtype must still be refused, with the same exception type.

**Byte-identical passthrough, on arguments.** The differential test was **not
sufficient**, and this is the most useful thing this round found about its own
verification. A wrapper that did `kwargs.pop("non_blocking")` before delegating
**passed it** — `non_blocking` has no observable effect on a CPU-to-CPU copy, so
an argument that does not change the result is invisible to a test that compares
results. Recorded as N5 in §7.

`_module_to.make(original)` is factored out so a test can wrap a **spy** and
assert the arguments upstream actually received are identical to those passed.
`test_ordinary_calls_reach_upstream_with_identical_arguments` covers eleven
forms including `non_blocking=True` and `non_blocking=False`, and it is what
catches N5.

### 5.5 `to(npu)` resolves first, then lowers or refuses — per backend

`to(torchnative.device.npu)` **always resolves first**, whichever branch
follows, so the caller learns which NPU this host actually has rather than
being handed a flat verdict about their machine. The dispatch key is then
`resolution.backend`, **not** `host()`: the host chooses the backend
(`NPU_CANDIDATES`, an ordered list per host since docs/devices/NPUVENDOR.md)
and the backend is what has or has not been wired, so keying
on the host would be reading the wrong fact one step early.
`test_the_openvino_branch_is_chosen_by_the_resolution_and_not_by_the_platform`
is the test that says so.

**`openvino` (Intel NPU) is wired.** It calls
`torchnative.export.intelnpu._compile_model`, which walks `named_children()`
and swaps each eligible `torch.nn.Linear` for a leaf whose forward runs on the
OpenVINO device. That call is in place and returns the same object, so `to()`
returns `self`: still an `nn.Module`, with its `parameters()`, `state_dict()`
and `named_children()` intact, so `generate()` keeps working and never learns
anything about the NPU. No library path is passed — the OpenVINO runtime is
discovered from the pip package (`pip install torchnative[npu]`).

**The partial-offload report is delivered twice, and that is the point.**
`_compile_model` names every leaf left behind, because "the model is on the
NPU" is false for any model with a `LayerNorm` in it. So:

* `model.torchnative_offload` — the report as a dict, for a caller who asks.
  An attribute rather than a return value because the return value is fixed by
  upstream's contract; a value rather than prose because `fraction_moved` is
  what makes "90% offloaded" impossible to mistake for "offloaded".
* a `UserWarning`, **only** when `fully_offloaded` is False, for a caller who
  does not ask. This is the load-bearing half:
  [`../graph/NPU2.md`](../graph/NPU2.md) is an entire document about a partial
  offload that went unnoticed *because the answers were right*, and an
  attribute nobody reads reproduces it exactly. A full offload is silent, which
  is what keeps the warning informative when it fires.

**Zero leaves lowered is a refusal.** `_compile_model`'s `IntelNPUUnsupported`
propagates unchanged and no report is attached — returning an untouched model
with a success message is the silent CPU fallback this path exists to prevent,
and a report on a model that was never lowered is the same lie with a receipt.

**`coreml` and `qnn` still refuse**, at the same quality of message and after
the same real resolution: `NotImplementedError` naming the resolved unit, the
backend, the probe, what exists (the capture layer; the per-vendor
execution-device evidence: `intelnpu.probe`, `assert_execution_device`,
`verdict_execution_devices`) and what is missing (the equivalent leaf). They
are **not** stubbed into a fake success. Returning the model unchanged would be
an argument accepted and dropped: the caller would hold a model they believe is
on the Neural Engine and which is in fact on the CPU —
[`../graph/NPU2.md`](../graph/NPU2.md) §1 exactly, and CLAUDE.md §6 on promised
refusals that never happen.
`test_to_the_compiled_target_refuses_and_names_what_it_resolved_to` fails if
`self` comes back on this host (N7).

**What the Intel branch is and is not evidence of.** No machine in this
repository has an Intel NPU, and `library_candidates` refuses on darwin by
design. `rust/torch_c/pytests/test_npuwire.py` therefore fakes exactly two
boundaries and nothing above them — the probe (`TORCHNATIVE_DEVICE_HOST=windows`
plus `intelnpu.npu_available` / `available_devices`, the shape §8 established)
and the OpenVINO runtime (`intelnpu.OpenVINO`, four methods). The resolver,
`_compile_model`, `linear_ir`, `verdict_execution_devices`, the report and the
`_module_to` wrapper are all real. Every test in that file says so in its name
and its docstring: it is evidence about **dispatch**, not about hardware.
Nothing there shows a number was computed on an Intel NPU.

## 6. The one deliberate disagreement: `_mps_is_available` is not a probe

`bootstrap.py` installs it as a constant:

    module._mps_is_available = _constant_function("torch._C._mps_is_available", False)

with a comment justifying the `False`: *"candle's `metal` feature is off in
Cargo.toml, so there is no Metal backend linked in"*.

**That comment is stale, and the constant is a false negative.** Measured on
this host, through the shim (`torch._C._aten_implemented` present):

| | |
|---|---|
| `torch._C._mps_is_available()` | `False` |
| `torch.empty(2, 2, device="mps").device` | `mps:0` |
| `(a + a).cpu().sum()` for `a = torch.ones(3, 3, device="mps")` | `18.0` |
| `Cargo.toml` Apple-target entry | `features = ["metal"]` |

`candle-metal-kernels` and `objc2-metal` compile into this artefact. Metal is
on and computing; the constant says it is not.

So `mps.availability()` reports `kind="measured"` from **an actual allocation**,
and carries the constant alongside as `detail["declared"]` with
`detail["declared_disagrees"]`. The constant is reused — it is just not allowed
to be the answer. Reporting it as availability would tell a user there is no
Metal on a machine that is computing on Metal, which is the same class of error
as [`../graph/NPU2.md`](../graph/NPU2.md)'s, pointing the other way.

**This is a defect in `bootstrap.py`, not in this namespace, and it is not
fixed here.** Fixing it means deciding what `torch.backends.mps.is_available()`
should return, which changes behaviour for every existing caller and reaches
past this round's request (CLAUDE.md §5.7). It is recorded, measured, and left.

## 7. Nullification — what was broken, and whether the tests noticed

Every row was applied to the source, the suite was run, and the source restored.

| # | nullification | caught by |
|---|---|---|
| N1 | `mps` availability read off the build-time constant | `test_mps_availability_is_measured_not_declared` |
| N2 | `npu` resolution ignores the host (always `coreml`) | `test_npu_resolves_differently_per_host`, `test_npu_refuses_by_name_on_a_host_with_no_npu_path` |
| N3 | `npu` falls back to a CPU resolution instead of refusing | `test_npu_never_resolves_to_the_cpu` |
| N4 | `cuda`'s five reasons collapsed to `unclassified` | `test_cuda_keeps_its_five_named_reasons` |
| N5 | `to()` drops `non_blocking=` on ordinary calls | **initially NOT caught** — see below |
| N6 | `to()` wraps the module instead of returning `self` | `test_to_an_eager_torchnative_device_moves_the_parameters` |
| N7 | `to(npu)` returns the model unchanged | `test_to_the_compiled_target_refuses_and_names_what_it_resolved_to` |
| N13 | `_resolve_openvino`'s **success** branch returns the wrong unit | **initially NOT caught** — see below |
| N14 | `_resolve_qnn`'s **success** branch returns the wrong unit | **initially NOT caught** — see below |

**N5 escaped the first time, and that is the finding.** The differential test
compares observable module state, and `non_blocking` has no observable effect on
a CPU-to-CPU copy, so a wrapper that silently dropped it produced byte-identical
parameters and a green suite. This is CLAUDE.md §5.5's shape: a verification
that could not fail for that class of change.

The fix is `test_ordinary_calls_reach_upstream_with_identical_arguments`, which
checks the **arguments upstream received** rather than their effect. With it in
place N5 is caught. The lesson generalises to any patch of a core method:
comparing results cannot see an argument that does not change the result.

**N13 and N14 escaped for a different reason: those branches never run here.**
The Windows and Android resolvers only ever reached their *refusal* paths on
this Mac, so changing what they return on success was invisible. The fix is
`test_the_windows_and_android_success_branches_resolve_correctly`, which
temporarily replaces `intelnpu.npu_available` / `available_devices` and
`qnn_device.device_report` with functions answering as a machine with the
hardware would, and requires the resolver to name the Intel NPU and the Hexagon
NPU respectively. The probes are not reimplemented and the resolver is
unmodified.

**That test is not evidence that any NPU was reached** — §8 still holds. It is
evidence that the resolver names the right unit *when the probe says yes*,
which is the part that was previously unchecked. It also asserts the real
probes refuse again once restored, so the fake cannot leak into the other
tests.

## 8. What this round did not verify here

* **Windows and Android `npu` resolution never ran against hardware.** Both
  paths refuse on this Mac, by name. Their *refusal* paths are tested directly
  and their *success* paths only against a simulated probe (§7, N13/N14) — no
  Intel NPU and no Hexagon device was contacted. The
  Intel path needs a Windows machine with the OpenVINO runtime; the Hexagon path
  needs a reachable device with an HTP architecture.
* **The Neural Engine was never executed on through this namespace**, because
  `to(npu)` refuses at the compile step. What is verified is that the unit is
  *present and named*, from CoreML's own compute-device list — which is
  reachability, not execution, and [`../graph/NPU2.md`](../graph/NPU2.md) is
  emphatic that those are different claims.
* **`vulkan` was measured both ways** (available with the emulator's loader
  supplied, `no_loader` without) but no tensor was computed on it through this
  namespace.
* **`cuda` has never been built or run**, here or anywhere in this project.
