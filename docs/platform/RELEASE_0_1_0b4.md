# 0.1.0b4 — release notes

**An NPU ran this project's arithmetic for the first time, and a model
that `to(device.npu)` had accepted stopped killing the process on its
first forward.** Both happened on macOS, on the one machine here with
the hardware; the Intel and Qualcomm arms are unchanged.

Read §3 before upgrading for the accelerator: what is proven is narrower
than "it runs on the Neural Engine", and §3 says exactly how narrow.

---

## 1. Features added

- **`to(device.npu)` reaches the Apple Neural Engine.** `MLComputePlan`
  — CoreML's own answer to which unit it picked — reports
  `preferred: NeuralEngine` for `Linear` and `Conv2d` at float16. That
  is not a label this project applies; it is read back per operation
  and lands on `model.torchnative_offload["plans"]`.

- **Two spellings, because they are two products:**

  ```python
  model.to(device.npu)                        # float16 -> Neural Engine
  model.to(device.npu, precision="float32")   # float32 -> CPU/GPU
  ```

  There is no float32 road to the ANE, and that was settled from
  coremltools' source rather than assumed: `compute_precision` accepts
  FLOAT32, FLOAT16 and a *subset selector* for the float16 cast, and
  the ANE is float16 hardware. Rather than one spelling with a silent
  mode, the weaker numerical claim gets its own name.

- **Real checkpoints lower.** `dtype="auto"` on a modern Hugging Face
  model is bfloat16, which the CoreML path refused outright. bf16 and
  f16 are now widened to float32 for marshalling — exact in both cases,
  so the compiled program is bit-identical to the one from an f32
  checkpoint. SmolLM2-135M lowers 211 of 211 Linears and runs.

- **`adapt` stage 0** — re-estimate normalisation statistics, no
  backward. Stage 1 (`Tent`) existed; the cheaper half, which is the
  one that matters on a device, did not. It is built on
  `delta.BufferSnapshot` rather than `Delta`, because a re-estimated
  statistic is a fresh measurement and not an offset from a base, and
  averaging two of them across a federated round is not the operation
  averaging two deltas is.

- **`torch.equal` and `torch.allclose`**, which the shim did not
  implement. Agreement checked against upstream across dtypes, NaN,
  both infinities, empty and 0-d tensors, and the tolerance boundary.

- **A tag-triggered PyPI workflow** (`publish-pypi.yml`), by OIDC with
  no token. **It has never run**; `docs/platform/PUBLISH_CI.md` says so
  in its first line and a test enforces that sentence.

## 2. Defects fixed

- **`model(x)` ended the process, silently.** After a successful
  `to(device.npu)`, the first forward compiled 217 leaves and the
  interpreter vanished — no traceback, no "Segmentation fault", nothing
  to grep. The cause is not ours: CoreML binds each numpy input into a
  *lingering* execution stream and a libdispatch worker drops
  libcoremlpython's Python reference milliseconds later **without the
  GIL**, corrupting CPython's heap. `coremltools.convert` ending in
  `gc.collect()` is why it first looked like a conversion problem. One
  reused input buffer per compiled shape avoids it.

- **The shim computed on Metal and reported it could not.**
  `torch.backends.mps.is_available()` and `is_built()` were hardcoded
  `False` on a host where `(a @ b)` really ran on the GPU. Those two
  functions are the standard gate, so transformers and accelerate would
  never have selected MPS. They are different questions and now get
  different answers.

- **Eight of ten subpackages were unreachable as attributes.**
  `torchnative.adapt` raised `AttributeError`, and so did `api`,
  `delta`, `distributed`, `export`, `kernels`, `nn` and `quant`. Nobody
  saw it because `from torchnative import adapt` works regardless and
  every document spells it that way.

- **`float64` on `mps` used to *succeed*** and hand back a tensor that
  could only be cloned. It refuses by name now.

- **The NPU vendor was resolved from the operating system.** A Ryzen AI
  laptop was told it had no Intel NPU — on a machine that has an NPU —
  and every `win_arm64` machine was told it had an Intel one, through a
  runtime PyPI does not even publish for that platform.

- **`compute_plan` leaked a compiled bundle on every call**: 8442
  directories, 11.6 GB. A gate run now returns disk instead of
  consuming it.

- **An empty compute plan read as silence.** A model whose plan CoreML
  declines to produce got no warning at all — the exact shape
  [`../graph/NPU2.md`](../graph/NPU2.md) §1 exists to prevent. "CoreML told us nothing" and
  "CoreML told us CPU" are now different sentences.

- **Every published wheel said `INSTALLER = uv`**, because the vendored
  tree is assembled out of an installed venv. Excluded from the
  dist-info this release ships.

## 3. Measured but not implemented

- **A decode step reaches the Neural Engine on none of its Linears.**
  Per shape, batch 1 against batch 128: 576→192, 576→576, 576→1536 and
  1536→576 are all **CPU** at batch 1 and NeuralEngine at 128; `lm_head`
  is CPU at both. The unit is *supported* for all of them and CoreML
  prefers the CPU at batch 1 — and a decode step is batch 1 by
  definition. **This arm is a prefill story, not a decode story.**

- **float16 does not meet the agreement bar** that float32 meets.
  float32 is 2.7e-06 (linear) and 4.5e-06 (conv) against `verify`'s own
  2e-05; float16 is 1.5e-03 and 1.26e-03 under its own, weaker, named
  grade. The tolerance was not widened to cover both.

- **Only `Linear` and `Conv2d` lower, and the rest were rejected on
  measurement.** `layer_norm`, `relu`, `gelu`, `softmax`, `max_pool`
  and `batch_norm` are ANE-*supported* at float16 and never
  ANE-*preferred*, at every size tried. Swapping such a leaf buys a
  tensor round trip and does not reach the unit. `gather` is not in the
  supported column at all; `matmul` is, but attention is not a leaf.

- **Intel NPU and Qualcomm are unchanged.** A user reported
  `to(device.npu)` ending the process on a Windows Intel NPU laptop on
  0.1.0b3; that is undiagnosed and may still occur. Qualcomm refuses,
  correctly: the device tested exposes no compute-DSP FastRPC endpoint.

- **`MAX_DIM = 2**17` has no backing in OpenVINO.** The only
  per-dimension limit the Intel NPU compiler names is 8192, sixteen
  times smaller, and it *tiles* rather than refusing. The constant is
  unchanged anyway, because raising it would infer a permissive fact
  from the absence of a restrictive one.

- **The decode lead is gone.** 46.6 tok/s against upstream's 44.4 has
  become 45.3 against 45.9 — about 1.5% behind, re-measured at 304
  operators under controlled load.

- **`torch.compile` is still a permanent refusal**, for the structural
  reason in [`../graph/COMPILE.md`](../graph/COMPILE.md); nothing here
  changes it.

- **82 of 297 architectures** were never numerically judged — that count
  is [`../architectures/ARCH100.md`](../architectures/ARCH100.md)'s, and
  it measured reachability, which is not agreement. Of the ones that can
  be judged, 288 of 290 agree, and those counts are now pinned by
  DOCWATCH so they cannot rot again.

## 4. Documentation corrected

`GAPS.md` §3.3, README and `DEVICE_ABS.md` described the MPS defect in
the present tense after it was closed. `NPU2.md` §8.3 recorded a
diagnosis that later measurement disproved, and now opens by quoting it.
`ARCH300.md` carried a stale forward count with no supersede banner.

---

## 5. Gate at this head

```
1589 ok / 0 FAIL       (0.1.0b3 shipped at 1387)
DOCWATCH   1174 / 1174
cargo test 33 passed / 0 failed
```

Three diagnoses were reversed during this release, each by a control
rather than by argument: a crash blamed on conversion-inside-forward was
neither; a retention mechanism dismissed as decoration was load-bearing;
and a cold-cache hypothesis was discarded when a never-compiled shape
planned fully on its first attempt. Worth recording, because the counts
above are only worth what the verification behind them is.

## 6. Platform status

Unchanged from [`RELEASE_0_1_0b3.md`](RELEASE_0_1_0b3.md) §6 — the published wheels ran on
the iOS simulator, an Android emulator and Pyodide for the first time in
that release, and nothing new ran on a platform for this one. Linux,
Windows and the iOS device remain at *builds*.
