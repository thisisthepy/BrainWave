# 0.1.0b2 — release notes

**`model.to(device.npu)` lowers instead of refusing.** That is the whole
point of this release, and it is the first one where the four lines the
README documents do something on an accelerator path rather than
resolving it and stopping.

```python
from torchnative.transformers import AutoModelForCausalLM
from torchnative import device

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
model.to(device.npu)
```

**Read §3 before deciding this works for you.** What is proven here is
*dispatch*. There is no Intel NPU and no OpenVINO on the machine this was
built on, so nothing in this release is evidence that a graph executed on
Qualcomm, Apple or Intel silicon. That claim needs hardware and this
release does not make it.

Read [`RELEASE_0_1_0b0.md`](RELEASE_0_1_0b0.md) §5 for what the project still does not
do; none of it changed.

---

## 1. Features added

- **`to(device.npu)` lowers on the `openvino` backend.** `_to_compiled`
  dispatches on `resolution.backend` rather than on `host()` — the host
  chooses the backend, the backend decides what can be done — and calls
  `intelnpu._compile_model`, which walks `named_children()` and replaces
  each `nn.Linear` with a leaf whose forward runs on the OpenVINO device.
  The model stays a real `nn.Module`, so `generate()` still works.

  `coreml` and `qnn` still refuse, after the same real resolution and at
  the same quality of message. They are not stubbed into success.

- **The offload report reaches the caller.** `model.torchnative_offload`
  carries the dict for a caller who asks — an attribute rather than a
  return value, because upstream fixes what `to()` returns — and a
  `UserWarning` fires when `fully_offloaded` is False, for a caller who
  does not. **A full offload is silent**, which is what keeps the warning
  worth reading. A partial offload reported as a full one is the defect
  [`../graph/NPU2.md`](../graph/NPU2.md) is entirely about.

  Qwen3-4B *will* warn: `lm_head` is 151936 wide and `MAX_DIM` is 131072,
  so 252 leaves lower and one does not.

- **Zero leaves lowered is a refusal, not a success.**

- **`pip install "torchnative[npu]"` is now the whole setup for Intel NPU.**
  No system-wide OpenVINO, no PATH edits, no DLL path. `pip install openvino`
  ships the entire runtime — `openvino_c.dll` and
  `openvino_intel_npu_plugin.dll`, or `libopenvino_c.so.NNNN` and
  `libopenvino_intel_npu_plugin.so` — inside the Python package, verified by
  downloading the real wheels. `load_openvino_c` discovers that directory;
  on Windows it goes through `os.add_dll_directory` so the sibling DLLs
  resolve.

- **`torchnative.export.qnn_plan`** — the per-leaf QNN lowering plan, the
  Qualcomm counterpart to `intelnpu.plan_lowering`, so `to(device.npu)` on
  Android has something to say beyond naming a unit.

## 2. Defects fixed

- **The OpenVINO runtime could not be found where users actually have it.**
  The last resort was `ctypes.CDLL("openvino_c.dll")`, which consults the OS
  loader path only, so a `pip install openvino` was invisible by
  construction. `library_candidates` also globs the discovered directory now
  rather than matching a hardcoded list — that list offered
  `libopenvino_c.so.2025` and `.so.2024` while the shipped file is
  `.so.2541`, so it had already rotted.

- **`qnn_plan` and `qnn_ops` did not compose: every `Linear` was declined.**
  Built in parallel, each green, each against its own idea of one contract —
  `qnn_ops.check_leaf` takes an **ATen operator name**, `qnn_plan` was
  passing the **module path**. The table correctly answered "no node visitor
  is registered for `fc1`" and a model of nothing but `nn.Linear` planned
  zero eligible leaves. Every test stayed green because every test injected a
  stand-in keyed the same wrong way. A test that drives the real table with
  no stand-in now pins the handshake.

- **The gate ran on whatever `python3` was on PATH.** `run.sh` takes
  `${PYTHON:-python3}`; the system interpreter here has neither `numpy` nor
  `typing_extensions`, and one run reported **473 failures that were not
  regressions**. Both `run.sh` and `publish_main.sh` now refuse to start,
  naming the interpreter and every missing import. The required set
  deliberately excludes `torch`, `torchgen`, `functorch` and `torchnative`:
  those are this repo's own build products, so their absence is a normal
  pre-build state and not a wrong interpreter. The first version of the guard
  required `torch` and therefore refused the one interpreter the gate exists
  to use.

- **A failed cross build left a publishable wheel in `dist/`.** `build.py`
  builds an intermediate with pip before patching it for the target; when a
  later step failed, that intermediate stayed. Twice during the 0.1.0b1 build
  a `macosx_11_0_universal2` wheel appeared this way. Uploaded, it could have
  served macOS users a wheel nobody verified. pip, repack and verify now run
  in a temporary directory and the wheel moves into `dist/` only after
  `verify()` succeeds.

## 3. Measured but not implemented

- **Nothing in this release has executed on an NPU.** The tests fake exactly
  two boundaries — the device probe and `intelnpu.OpenVINO` — and nothing
  above them. Real and under test: the resolver, `_compile_model`,
  `linear_ir`/`MAX_DIM`, `verdict_execution_devices`, the report, and the
  `_module_to` wrapper. Every test name says "dispatch evidence; the NPU is
  faked".

  **Unproven, and only hardware can settle it:** that the pip-installed
  runtime is found on Windows, that a real compile succeeds with
  `EXECUTION_DEVICES` reading `NPU`, the driver-compiler latency, and
  `from_pretrained` through `generate()` end to end.

- **Qualcomm remains a refusal.** `device.npu` resolves on Android and then
  declines, because `/sys/class/fastrpc` on the device tested registers only
  the audio DSP — there is no compute-DSP FastRPC endpoint reachable from an
  adb shell. `qnn_plan` and `qnn_ops` describe what *would* lower; neither has
  run on Hexagon.

- **Apple's Neural Engine remains a refusal** on the `coreml` backend.

- **Two operators are missing from the shim**, found by this release's own
  tests: `torch.allclose` and `torch.equal` have no overload-table entry.

- **`torch.compile` is still a permanent refusal**, for the structural reason
  in [`../graph/COMPILE.md`](../graph/COMPILE.md).

- **Architecture coverage is unchanged**: 297 of 297 forward, and the **82**
  that [`../architectures/ARCH100.md`](../architectures/ARCH100.md) could not judge numerically are
  still unjudged. Reachability is not agreement.

- **No performance figure is part of this release.** `../perf/PERF.md` was
  measured at 96 operators against a tree now at 302.

## 4. Documentation corrected

- [`../devices/DEVICE_NS.md`](../devices/DEVICE_NS.md) §5.5 said `to(npu)` refuses outright, and the
  README's roadmap row said recompiling was not implemented. Both were true
  when written and are not now.
- `PROJECT.md` said the three backend extras are empty because no backend has
  a Python-side runtime dependency. `npu` does, and both the file and the
  prose say so.
- A docstring in `export/qnn_device.py` justified stat-by-name with "`ls /dev`
  returns nothing for the `shell` user". That is false on the device tested.

---

## 5. Gate at this head

```
1348 ok / 0 FAIL          (0.1.0b1 shipped at 1311)
DOCWATCH   1110 / 1110
cargo test 33 passed / 0 failed
TREE_UNCHANGED_DURING_GATE=yes
```

Three suites in this release passed on the environment rather than on the
code, and all three were caught by re-verifying after the merge rather than
by the round that wrote them: a file reaching the vendored `torch` without
`TORCH_USE_RTLD_GLOBAL`, a plan tested only against a stand-in that shared
its own misunderstanding, and a suite calling two operators the shim does not
implement — green in a worktree with no vendored tree, where `import torch`
fell through to an upstream install. Worth stating plainly, because the
counts above are only worth what the verification behind them is.

## 6. Platform status

Unchanged from [`RELEASE_0_1_0b0.md`](RELEASE_0_1_0b0.md) §6, which should be read as this
release's platform table. Nothing new ran on a platform for this release.
