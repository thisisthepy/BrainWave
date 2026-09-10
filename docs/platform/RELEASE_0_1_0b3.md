# 0.1.0b3 — release notes

**The first release shaped by a real NPU.** A user ran `0.1.0b2` on a
Windows Intel NPU laptop. `to(device.npu)` worked — 252 of Qwen3-4B's
Linears lowered and OpenVINO reported `EXECUTION_DEVICES=['NPU']`, which
is the first evidence in this project's history that anything reached an
NPU. Then `generate()` died, and finding out why produced most of this
release.

Everything below was found by running on the device. None of it was
visible from here.

Read §3 before deciding what this does for you. `0.1.0b2` **cannot
generate with Qwen3-4B at all**; if you have that version, this replaces it.

---

## 1. Features added

- **`to(device.npu)` compiles the decode shape up front**, with a
  `progress(done, total, name)` callback. `_NPULinear` compiles a
  static-shape IR per batch, and `generate()` uses two shapes — the
  prompt length, then 1 per token with a KV cache — so 504 driver
  compiles used to happen lazily, *inside the first generated token*,
  with no way to tell a stall from a hang. They now happen where the
  caller asked for them and can say what they are doing. `eager=False`
  declines.

- **The OpenVINO compile cache persists.** `ov_core_compile_model` was
  called with a property count of **zero**, so `ov::cache_dir` was never
  set and nothing was cached: every process restart recompiled all 504.
  This also sets the repository's first cache-path convention, for the
  QNN and CoreML caches to follow:

  | platform | directory |
  |---|---|
  | Windows | `%LOCALAPPDATA%\torchnative\Cache\openvino` |
  | macOS | `~/Library/Caches/torchnative/openvino` |
  | Linux | `$XDG_CACHE_HOME/torchnative/openvino` (default `~/.cache`) |
  | Android | `$HOME/.cache/torchnative/openvino` — app-private |
  | iOS | `~/Library/Caches/torchnative/openvino` — sandbox container |
  | wasm | none; no persistent filesystem |

  Windows is `%LOCALAPPDATA%` and never `%APPDATA%`: a driver- and
  device-keyed blob must not roam to a machine it was not compiled for.
  `TORCHNATIVE_CACHE_DIR` moves the root, `TORCHNATIVE_OPENVINO_CACHE_DIR`
  moves this backend only, and either disables caching with `0`, `off`,
  `none`, `false` or empty.

  **Not under the Hugging Face cache**, though the case for it was real:
  `huggingface_hub` owns and *prunes* that layout, and it is undefined
  for inputs that never came from the Hub. See [`../devices/NPUCACHE.md`](../devices/NPUCACHE.md).

- **`torch._C._shim_f16_bytes`** — a tensor to f16 bytes without building
  Python objects.

## 2. Defects fixed

- **A `MemoryError`, and then a Rust panic, on any real LLM.**
  `_NPULinear._weights_blob` did
  `pack_f16(weight.detach().flatten().tolist())`. Qwen3's `down_proj` is
  9728 × 2560 = 24,903,680 elements, so `.tolist()` built **24.9 million
  `PyFloat` objects — about 800 MB of CPython heap** to produce a 50 MB
  blob. Measured here on that exact shape: the new route produces the
  same 47.5 MB in 0.03 s with a Python heap peak of 47.5 MB, which *is*
  the returned bytes. At 2²⁰ elements, side by side: **24.0× the blob
  before, 1.00× now**, byte-identical output.

  The activation path had the same defect on **every call, both
  directions**; it now passes f16 bytes in and rebuilds through
  `torch.frombuffer`. 24.0× and 20.0× become 1.00×.

- **The shim panicked instead of raising.** When CPython could not
  allocate, `tolist` crossed the FFI boundary as a Rust panic rather than
  a `MemoryError`. `flat_objects` built scalars with `into_py_any`, which
  for `f64` reaches `PyFloat::new` → `PyFloat_FromDouble(val).assume_owned(py)`,
  and pyo3 documents `assume_owned` as *"panics on NULL"*. pyo3 offers no
  fallible `PyFloat::new`, so the float and int arms now call
  `Bound::from_owned_ptr_or_err` — pyo3's own fallible sibling, which
  fetches the `MemoryError` CPython already set. One panic point survives
  in `nest`'s `PyList::new` and is recorded rather than silenced.

- **252 `ov::Core` objects for one model.** Each leaf built its own —
  252 dlopens of the plugin registry, 252 device enumerations, 252 cache
  resolutions. Now one, shared. A leaf built alone by `from_torch` still
  makes its own, so sharing is an optimisation, not a requirement.

## 3. Measured but not implemented

- **Parallel compilation is refused, with reasons** —
  [`../devices/NPUPAR.md`](../devices/NPUPAR.md). Two of four preconditions are
  UNVERIFIED: whether `ov::Core::compile_model` is thread-safe (the word
  appears in no OpenVINO header, guide or API doc; openvino#27366 asks
  exactly this and was closed unanswered), and whether the driver-resident
  compiler serialises internally (the path ends in the closed NPU UMD).
  A third bounds the gain: `shim_f16_bytes` never calls `allow_threads`,
  so the conversion runs GIL-held. The fourth is a hazard —
  `FileStorageCacheManager::write_cache_entry` opens the final
  `<hash>.blob` directly, with no temp file, no rename and no
  cross-process guard, and running two scripts at once is normal.
  Shipping threads on that would risk a corrupted cache entry that later
  loads as a valid-looking compiled model. A tripwire test keeps the
  decision honest.

- **Still no NPU speed number, of any kind.** Nothing here was timed on
  the device. `252 → 1` is a count of constructions against a fake.

- **Utilisation will stay low, and that is structural.** Each `Linear` is
  an independent infer request, so one token is 252 separate NPU calls
  stitched together by Python. The per-call marshalling is fixed; the
  call structure is not.

- **`MAX_DIM = 2**17` is an unsourced constant**, copied from the
  archived `intel_npu_acceleration_library`, which gives no reason
  either. Qwen3-4B's `lm_head` exceeds it, so 252 leaves lower and one
  does not (`fraction_moved` 0.9033). [`../devices/NPUDIM.md`](../devices/NPUDIM.md)
  now records where it is *not* from: `131072` occurs nowhere in the
  OpenVINO NPU plugin, the NPU compiler, the Level Zero graph extension
  or the shipped NPU binaries, and the one per-dimension limit the
  compiler names is `VPU_DIMENSION_LIMIT = 8192`, which it tiles past
  rather than refusing. **The real ceiling is still unmeasured** — no
  dimension above 8192 has been compiled for `NPU` here — so the
  constant is unchanged and `tools/devices/intelnpu_dimsweep.py` is the
  experiment that would settle it.

- **Qualcomm and Apple remain refusals.** `torch.compile` remains a
  permanent one, for the structural reason in
  [`../graph/COMPILE.md`](../graph/COMPILE.md). And of the 297
  architectures that forward, **82** are still numerically unjudged —
  [`../architectures/ARCH100.md`](../architectures/ARCH100.md) measured
  reachability, and a forward is not a match.

## 4. Documentation corrected

New: [`../devices/NPUCACHE.md`](../devices/NPUCACHE.md), [`../devices/NPUPAR.md`](../devices/NPUPAR.md).
[`../devices/INTELNPU.md`](../devices/INTELNPU.md) gains the surviving panic point and the
`tolist` reasoning.

---

## 5. Gate at this head

```
1387 ok / 0 FAIL          (0.1.0b2 shipped at 1348)
DOCWATCH   1118 / 1118
cargo test 33 passed / 0 failed
TREE_UNCHANGED_DURING_GATE=yes
```

Two rounds in this release had a nullification come back **green** and
said so rather than moving on. One was a vacuous test — a storage-versus-view
check whose float32 fixture had already been materialised before Rust saw
it — and was widened until it failed. The other was dead code: a per-core
memo no caller could reach, which was deleted rather than tested.

## 6. Platform status

Unchanged from [`RELEASE_0_1_0b0.md`](RELEASE_0_1_0b0.md) §6, with one addition that is not a
platform claim but is the first of its kind: **an Intel NPU on Windows
accepted a compiled model from this project and reported
`EXECUTION_DEVICES=['NPU']`.** One leaf, checked at compile time. Not a
generated token, not a benchmark.
