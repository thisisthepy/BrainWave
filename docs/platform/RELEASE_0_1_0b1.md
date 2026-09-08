# 0.1.0b1 — release notes

**A patch beta, and it exists because `0.1.0b0` shipped without the API its
own README documents.** The published `0.1.0b0` wheels carry
`torchnative/export/` and nothing else under `torchnative/`: no
`torchnative/device/`, no `torchnative/transformers/`, no
`plan_lowering`, and the five withdrawn Intel-NPU entry points
(`compile_model`, `compile_module`, `quantize_`, `dynamo_backend`,
`NPULinear`) still exported. The commit that added the first set and
withdrew the second landed three hours **after** those wheels were built.

That was found by unzipping the published artefact rather than by reading
the changelog, which is the only way it could have been found:

```
dist/torchnative-0.1.0b0-cp313-abi3-win_amd64.whl
  torchnative/export/intelnpu.py
    plan_lowering    0      <- absent
    compile_model   15      <- withdrawn upstream, still shipped
```

So every example in `README.md` — `from torchnative import device`,
`from torchnative.transformers import AutoModelForCausalLM`,
`model.to(device.npu)` — fails on the published beta. This release is
that gap closed, plus one accelerator claim that was not earned.

Read §3 and §5 of [`RELEASE_0_1_0b0.md`](RELEASE_0_1_0b0.md) first if you
are deciding whether to upgrade: nothing in this release changes what the
project does *not* do, and that is still the larger part.

---

## 1. Features added

- **`torchnative/device/` and `torchnative/transformers/` are actually in
  the wheel.** They were written before `0.1.0b0` and missed its build.
  `packages.find`'s `torchnative*` already matched them; only the timing
  was wrong. This is packaging reaching the code, not new code.
- **`torchnative.export.qnn_ops`** — the QNN/HTP supported-operator table:
  `supported_ops()` (127 targets), `not_supported_ops()` (3),
  `to_be_implemented_ops()` (6), `constraints()`, and `check_leaf()`,
  which returns a reason string and never a bare `False`.

  Every entry is sourced from the `target = [...]` lists in the 115
  `op_*.py` node-visitor builders of the installed `executorch 1.4.1`
  wheel — the registrations the partitioner consults, not a description of
  them. Nothing was guessed in. Max tensor rank, max dimension, the
  static-shape requirement and per-op numeric constraints are named as
  **UNVERIFIED** in [`../devices/QNNOPS.md`](../devices/QNNOPS.md) §5, with what
  would settle them: the QNN SDK headers, which are not on this machine.

  This table is not an execution claim. QNN's partitioner declines nodes
  **silently at compile time** and the program still runs and still
  computes correctly, because declined nodes fall back to the CPU — the
  opposite of NNAPI and CoreML, which fail closed at runtime. An
  optimistic table would therefore be worse than none.

- **`tools/devices/intelnpu_verify.py`** — the script to run on an Intel
  NPU machine. It keeps **selection** (Stage A, `plan_lowering`, pure
  Python, no OpenVINO, no hardware) apart from **execution** (Stage B,
  `probe`, which needs both). Reporting the two together is how a
  selection result gets read as an execution claim. Stage A builds the
  model on the `meta` device, so a 4B checkpoint costs no download and no
  RAM.

## 2. Defects fixed

- **`device.npu` claimed a Qualcomm Hexagon NPU from an SoC name string.**
  On the attached Galaxy Tab S9 Ultra (SM8550, a real 8 Gen 2),
  `device.npu.resolve()` returned "Qualcomm Hexagon NPU", and every input
  to that verdict was a name: `_resolve_qnn` read only
  `report["htp_arch"]`, which is `getprop ro.soc.model` looked up in a
  table. **A device whose NPU does not work produces the same answer.**

  The measurement that settles it: `/sys/class/fastrpc` on that device
  registers `adsprpc-smd` and `adsprpc-smd-secure` and nothing else —
  the **audio** DSP. The HTP skeleton loads onto the compute DSP over
  `cdsprpc`/`fastrpc-cdsp`, and neither node exists. `device_report` had
  already computed `fastrpc` and `staged_complete`; the resolver used
  neither.

  `_resolve_qnn` now raises `NpuUnresolved`, naming that `htp_arch` is a
  part number rather than a probe. **It refuses; it does not fall back to
  the CPU.** This proves the *claim* was unearned, not that the silicon is
  dead — `remoteproc-cdsp-md` and `rdbg_cdsp` exist, so a compute DSP is
  present, but this image exposes no FastRPC endpoint reachable from an
  adb shell.

  Same shape as `_mps_is_available()` hardcoded `False` on a host
  computing on Metal, and the NNAPI path claiming vendor acceleration
  while the device enumerated only `nnapi-reference`. Here it ran the
  other way — claiming hardware that was not reachable.

- **A suite that passed on the environment rather than on the code.**
  `test_qnnprobe.py` reaches `torch`, whose `_load_global_deps()` dlopens
  `torch/lib/libtorch_global_deps.*` — a file `tools/wheel/build.py`
  creates for a wheel and which does not exist in the source tree. It
  passed standalone because `TORCH_USE_RTLD_GLOBAL` happened to be set in
  the running shell, and failed under `run.sh`, which does not set it.
  Both affected suites are now re-verified under
  `env -u TORCH_USE_RTLD_GLOBAL`.

- **A fake probe built to the old criterion.** `test_devicens.py`'s
  android success-branch test fed `_resolve_qnn` a report carrying
  `htp_arch` alone. The resolver now also requires `htp_reachable`, so it
  refused — correctly. The fake supplies both. This is not a lowered gate:
  the test exists to show the resolver names the right unit **when the
  probe says yes**, and "yes" now means both facts.

## 3. Measured but not implemented

- **Nothing has executed on any NPU through `to(device.npu)`.** It
  resolves and then refuses, by design — returning `self` would hand back
  a model the caller believes is on the accelerator and which is running
  on the CPU. The missing step is the one that turns a captured graph into
  a leaf a module can carry ([`../devices/DEVICE_NS.md`](../devices/DEVICE_NS.md) §5).
- **Intel NPU selection is measured; Intel NPU execution is not.**
  `plan_lowering` on a real-width Qwen3-4B: **252 leaves eligible, 1
  declined** (`lm_head`, `out_features=151936` exceeds
  `MAX_DIM=131072`), `fraction_moved` 0.9033. On SmolLM2-135M: 211
  eligible, 0 declined, 0.9997. Both are statements about *selection*,
  computed from shapes on the `meta` device. Stage B has not been run by
  this project on Intel hardware.
- **The QNN op table has never been exercised on real silicon.** Neither
  has the AOT-lowering path in [`../devices/QNN.md`](../devices/QNN.md).
- **`torch.compile` is still a permanent refusal**, for the structural
  reason in [`../graph/COMPILE.md`](../graph/COMPILE.md); nothing here changes it.
- **Architecture coverage is unchanged**: 297 of 297 forward, and the
  **82** of them that `../architectures/ARCH100.md` could not judge
  numerically are still unjudged. Reachability is not agreement.
- **No performance figure is part of this release.** `../perf/PERF.md` was
  measured at 96 operators against a tree now at 302.

## 4. Documentation corrected

- Every example in `README.md` and [`../api/TRANSFORMERS.md`](../api/TRANSFORMERS.md)
  now uses `from torchnative import device` and `model.to(device.npu)`,
  rather than `import torchnative` with a fully-qualified call. The import
  line and the call site had disagreed, so the snippets did not run as
  written.
- [`../devices/QNNOPS.md`](../devices/QNNOPS.md) records the source of every table entry,
  and separates what is sourced from what is not.
- A docstring in `export/qnn_device.py` justified stat-by-name with "`ls
  /dev` returns nothing for the `shell` user". That is false on this
  device; `ls -l /dev/` lists fine. The real justification is robustness.

---

## 5. Gate at this head

```
1311 ok / 0 FAIL          (previous baseline 1297)
DOCWATCH   1110 / 1110
cargo test 33 passed / 0 failed
TREE_UNCHANGED_DURING_GATE=yes
```

One caution about that number, since this release is partly about numbers
that were not what they looked like. An earlier run of the same gate
reported **473 FAIL** — not because anything was broken, but because
`run.sh` takes `${PYTHON:-python3}` and the system `python3` has neither
`typing_extensions` nor `numpy`. The identical trap had already cost a
`publish_main.sh` run (`setuptools` missing). It failed loudly this time;
the same mistake in the other direction passes silently, and `run.sh`
should refuse to start on an interpreter that cannot import what the
suites need. That is not in this release.

## 6. Platform status

Unchanged from [`RELEASE_0_1_0b0.md`](RELEASE_0_1_0b0.md) §6, which should be read as this
release's platform table. Nothing new ran on a platform for this release:
the Android work here **removed** a claim rather than adding one.
