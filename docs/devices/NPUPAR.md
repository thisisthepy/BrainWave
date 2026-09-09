# One `ov::Core`, and why the 252 compiles are still serial

A user running Qwen3-4B on a Windows Intel NPU asked a fair question: `to(device.npu)`
lowers 252 `nn.Linear` leaves, each compiles a static-shape IR per batch size, `generate()`
uses two batch sizes (the prompt length once, then 1 per token with a KV cache), so **504
driver compiles happen before the second token appears** — one at a time, each lazily on
its first forward. Why is that not parallel?

Two answers, and they are different in kind.

* **The `ov::Core` count was a defect and it is fixed.** 252 → 1. §1.
* **Parallel compilation is a question, and this round answers it "no, not on this
  evidence".** §2 lists the four things that would have to hold and what was found for
  each. Two are UNVERIFIED. What ships instead is **eager** compilation — the same serial
  work, moved to where the user asked for it. §3.

Nothing here was measured on an Intel NPU. This repository's development host is an arm64
Mac; `library_candidates` refuses on darwin by design, and there is no OpenVINO on it.
Every number below is either a count of Python-level constructions (which `test_ovpar.py`
asserts) or a reading of OpenVINO's source. §4 says exactly what is unverified.

---

## 1. The Core count: 252 → 1

`_NPULinear._compile_for` was:

```python
if self._ov is None:
    self._ov = OpenVINO(self.library)
    found = self._ov.devices()
    ...
```

Per leaf. So a 252-leaf model built 252 `ov::Core` objects, and each one:

* ran `ov_core_create`, which dlopens and initialises OpenVINO's plugin registry;
* ran `ov_core_get_available_devices`, which **enumerates and initialises every plugin it
  finds** — the NPU plugin (which talks to the Level Zero driver) and the GPU plugin
  included, whether or not the model wants them;
* resolved and `makedirs`-ed the compile-cache directory (`docs/devices/NPUCACHE.md`).

That is waste, but the sharper problem is the third-party one. OpenVINO's model-cache
serialisation is a **per-hash mutex living inside one `CoreImpl`**:

```cpp
const auto lock = m_cache_guard.get_hash_lock(cache_content.m_blob_id);
```
— `src/inference/src/dev/core_impl.cpp`, at all four `compile_model` overloads.

252 separate cores do not share `m_cache_guard` with one another *at all*. So the
one-core-per-leaf shape had already opted out of the only concurrency protection OpenVINO
offers, before any thread existed. **Parallel compilation across 252 independent cores
would have been worse than the serial version, not better**, and that is why the shared
core had to land first.

### Where the shared core lives, and how its lifetime is guaranteed

`_compile_model` builds it — that is the function that knows a model is one unit — and
hands it to every `_NPULinear.from_torch` call. It is **not** a module-level singleton: a
process-wide core would outlive every model, could never be closed, and would freeze the
cache-directory decision that `docs/devices/NPUCACHE.md` deliberately leaves per-core and
overridable.

`_NPULinear` still works alone. `from_torch(layer)` with no `core=` gets `None` and builds
its own at first compile, exactly as before — sharing did not make a core mandatory.
(`test_a_standalone_npulinear_still_builds_its_own_core`.)

**Lifetime.** `OpenVINO.close()` calls `ov_core_free`, and a compiled model whose core has
been freed is a use-after-free that on this path presents as plausible wrong numbers rather
than as a crash. Two strong references make the accidental version unreachable:

1. every lowered leaf holds the core in `self._ov`;
2. every handle `compile_ir` returns holds it in `compiled._torchnative_core`.

So the core is reachable for exactly as long as anything made from it is, and Python's
refcount is the guarantee — nothing has to be remembered. `_compile_model` never calls
`close()`; a shared core has no single owner who could know when.

This is **not** a claim that an explicit `close()` is safe. `probe()` closes deliberately at
the end of a `with` block, after the compiled models inside it are gone, and that stays the
caller's call. What is ruled out is the core being collected because nobody kept a name for
it — which is precisely what a core created inside `_compile_model` and stored nowhere else
would have been.

The device-presence assertion moved into `open_core` with it, so a model asks OpenVINO
which devices exist **once**, not 252 times. That is a separate cost from constructing the
core — `ov_core_get_available_devices` initialises every plugin it finds — and it is
asserted separately.

There is deliberately no per-core "already verified" memo. The first version had one, and
deleting it did not turn any test red: a leaf that has been handed a core never calls
`open_core`, so the memo was unreachable. A guarantee nothing can check is not a guarantee,
so it was removed rather than kept.

---

## 2. Parallel compilation: four questions, two UNVERIFIED

### 2.1 Does the GIL actually get released? — Partly, and the part that does not bounds the win

`load_openvino_c` uses `ctypes.CDLL`, not `ctypes.PyDLL`. CPython's documentation is
explicit about what that means:

> *"Instances of this class represent loaded shared libraries. Functions in these libraries
> use the standard C calling convention, and are assumed to return `int`."* — and, for
> `PyDLL`: *"Instances of this class behave like `CDLL` instances, except that the Python
> GIL is **not** released during the function call, and after the function execution the
> Python error flag is checked."*
> — `Doc/library/ctypes.rst`, classes `ctypes.CDLL` and `ctypes.PyDLL`

So `CDLL` **does** release the GIL for the duration of the foreign call. That covers
`ov_core_compile_model`, which is where the driver compile happens. Good.

But the compile *path* is not all foreign, and the two Python-side pieces are not small:

* **The IR text is built in Python.** `linear_ir` produces the XML document as a Python
  string. GIL held.
* **The weights go through `torch._C._shim_f16_bytes`, and it does not release the GIL.**
  `rust/torch_c/src/tensor.rs`, `shim_f16_bytes`, takes `py: Python<'py>` and **never calls
  `py.allow_threads`**. Holding a `Python<'py>` token *is* holding the GIL in pyo3; the
  whole body — `flatten_all`, `contiguous`, `reduced::to_dtype` to F16, `to_vec1`, and the
  element-by-element `to_bits().to_le_bytes()` loop — runs with it held. For a Qwen3
  `down_proj` that loop is 24,903,680 elements, per leaf, per compile.

**This is the honest bound on any speedup.** Whatever fraction of per-leaf wall time is
weights-blob construction is serial no matter how many threads compile. It is not known
what that fraction is on the user's machine — nobody has measured a real NPU compile split
between blob-building and driver time — so the ceiling is unknown, not merely low. Adding
`allow_threads` to `shim_f16_bytes` is a plausible follow-up and would need its own round;
it is not free, because the pyo3 borrow of the tensor has to be shown safe across it.

### 2.2 Is `ov::Core::compile_model` thread-safe for concurrent calls on one Core? — UNVERIFIED

**No vendor statement was found.** Searched: the shipped headers in the 2026.3.1 Windows
and 2025.4.1 Linux wheels (`openvino/include/openvino/runtime/core.hpp`,
`compiled_model.hpp`, `openvino/c/ov_core.h` — the string "thread-safe" does not appear in
any header under `openvino/include/`), the plugin-developer guide RST in the repository,
and the published API docs for `ov::Core`. None of them says.

Somebody asked exactly this question — [openvinotoolkit/openvino#27366](https://github.com/openvinotoolkit/openvino/issues/27366),
"Is Core threading safe?", 2024-11-01: *"whether one `ov::Core` is needed per model"* and
*"if a single instance is sufficient, whether all its methods are thread-safe"*. It was
**closed as stale with no maintainer answer.**

What the *implementation* shows is defensive locking — `std::lock_guard<std::mutex>
lock(get_mutex(device_name))` around plugin acquisition, a global mutex whenever the plugin
registry is iterated, and the per-hash cache guard quoted in §1. That is evidence that
concurrent use was anticipated. **It is not a guarantee**, it is an implementation detail of
one version, and this project ships against whichever OpenVINO a user has installed. Adding
threads on the strength of reading somebody's mutexes is exactly the shape of reasoning
this repository refuses elsewhere.

### 2.3 Does the NPU's driver-resident compiler serialise internally anyway? — UNVERIFIED

The NPU plugin's compile path is `driver_compiler_adapter.cpp` →
`_zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, ...)` → the Level Zero graph
extension in the **NPU user-mode driver**, which is closed source. There is no mutex in the
adapter itself; whether the driver's compiler-in-driver serialises behind that call is not
observable from OpenVINO's source and is not documented.

One piece of weak, indirect evidence points the wrong way for threads: OpenVINO's own bulk
model-compilation tooling uses a **`ProcessPoolExecutor`**, not a thread pool. Processes
sidestep both the GIL and any in-process core sharing; choosing them is consistent with
threads not having helped, but it is equally consistent with the GIL alone being the reason.
It does not settle anything.

**What would settle it:** on the user's machine, compile N distinct leaves serially and then
with 2, 4 and 8 threads against one shared core, with the compile cache disabled
(`TORCHNATIVE_OPENVINO_CACHE_DIR=0`, so the second run is not measuring cache hits), and
report wall time. If the 8-thread time is within noise of the serial time, the driver
serialises and threads buy nothing. That measurement needs the hardware and cannot be
faked; it is the one experiment that turns this section from a refusal into a decision.

### 2.4 Is the CACHE_DIR path safe under concurrency? — Safe in-process, unguarded across processes

**Within one process, and only within one `Core`:** yes, by construction. `CacheGuard` is
documented in its own header as

> *"This class represents RAII guard class to protect multiple threads to modify the same
> cached network. Use `CacheGuard::getHashLock(hash)` to acquire lock for specific cache
> entry identified by its 'hash'."*
> — `src/inference/src/cache_guard.hpp`

Two threads compiling *different* models take different hash locks and proceed
concurrently, writing to different `<hash>.blob` files. Two threads compiling the *same*
model serialise. That is exactly the right shape, and it is another reason §1 had to land
first — 252 cores have 252 unrelated guards.

**Across processes there is no guard at all**, and the write is not atomic:

```cpp
void write_cache_entry(const std::string& id, StreamWriter writer) override {
    const auto blob_path = get_blob_file(id);           // <cache_dir>/<hash>.blob
    ...
    std::ofstream stream;
    stream.open(blob_path, std::ios_base::binary);      // the FINAL path, directly
    writer(stream);
    stream.close();
```
— `src/inference/src/cache_manager.hpp`, `FileStorageCacheManager`

No temporary file, no rename. A user running two scripts at once — the normal case — can
have two processes writing the same `<hash>.blob` simultaneously, and a crash or a kill
mid-write leaves a truncated file at the name a later run treats as a cache entry.

The read side does *some* validation and it is the mitigation that keeps this from being
catastrophic today: `load_model_from_cache` parses a `CompiledBlobHeader`, checks the
recorded file info and runtime properties, and on **any** exception — header or import —
calls `remove_cache_entry(...)` and recompiles from scratch (`OPENVINO_WARN("Could not load
model from cache.")`). So a truncated blob is normally detected and discarded rather than
loaded as a valid-looking compiled model.

**But there is no checksum over the body.** A blob whose header survives and whose payload
is corrupt is validated only to the extent the plugin's `import_model` happens to notice. A
cache entry that loads as a plausible compiled model and computes the wrong function is the
worst outcome this module can produce, and nothing above rules it out.

That risk exists **today, unchanged by this round** — it is a cross-process property of
OpenVINO's cache manager, and adding threads inside one process would not create it. But it
is the reason not to treat "the cache is fine under concurrency" as settled, and it is why
`docs/devices/NPUCACHE.md`'s per-user cache directory matters: a *shared* cache directory
across users would widen exactly this window.

---

## 3. What shipped instead: eager, serial, reportable

`_compile_model(..., eager=True)` — the default — compiles the **batch=1 decode shape** for
every lowered leaf before returning, and `progress(done, total, name)` reports each one.

This is not a speed-up. The same 252 compiles happen; it changes **where**. `generate()`
compiles the prompt-length shape on the first token and batch=1 on every token after, so
lazily the 252 decode-shape compiles land inside the *second* generated token — the model
emits one word and then appears to hang. Eagerly they happen inside `to(device.npu)`, which
the user typed and can watch.

batch=1 only. The prompt-length shape is not knowable until there is a prompt.

Three properties, each with a test:

* **A leaf that will not compile is named, not fatal.** It goes into `report["eager_failed"]`
  as `(name, reason)`, `fully_offloaded` drops to False, and the leaf keeps its lazy path —
  so it still compiles at first forward if the refusal was transient. Eager compilation must
  not turn a working lazy model into a hard failure.
* **The first leaf is still fatal.** It is the assertion that the device is real. Absorbing
  it by name would turn "there is no NPU on this machine" into 252 named warnings attached
  to a model the caller believes is offloaded — the silent CPU fallback with a report
  stapled to it, which is `docs/graph/NPU2.md`'s failure exactly.
* **`eager=False` restores the old behaviour** for a caller who wants to lower and inspect a
  model without paying minutes of compile.

An oversized leaf never reaches this loop: it was refused at `from_torch` and is in
`report["skipped"]` with its shape and `MAX_DIM`, unchanged. Reporting it as an *eager*
failure would have read as "OpenVINO would not compile it" and lost the limit.

---

## 4. What is unverified

* **Any speed number.** No compile was timed. There is no Intel NPU and no OpenVINO on this
  host, and `library_candidates` refuses on darwin by design. The claim "252 → 1 cores" is a
  count of Python constructions against a fake, not a measurement of dlopen time.
* **§2.2** — whether concurrent `compile_model` on one core is safe. No vendor statement
  exists; the implementation's locks are evidence, not a contract.
* **§2.3** — whether the NPU compiler-in-driver serialises. Closed source, undocumented.
  §2.3 names the measurement that would settle it.
* **§2.1's ceiling** — the split between GIL-held blob construction and GIL-released driver
  compile is unknown, so how much a thread pool *could* win is unknown even if 2.2 and 2.3
  came back green.
* **§2.4's body corruption** — that a header-intact, body-corrupt blob is caught by
  `import_model` is not established; only header-level validation was read.

The standing check that no thread pool arrived without this document being revisited is
`test_the_compile_path_still_spawns_no_threads` in `rust/torch_c/pytests/test_ovpar.py`.

---

## Sources

* [`src/inference/src/dev/core_impl.cpp`](https://github.com/openvinotoolkit/openvino/blob/master/src/inference/src/dev/core_impl.cpp) — `m_cache_guard.get_hash_lock`, `get_mutex(device_name)`, `load_model_from_cache`
* [`src/inference/src/cache_guard.hpp`](https://github.com/openvinotoolkit/openvino/blob/master/src/inference/src/cache_guard.hpp) — the per-hash guard and its documented scope
* [`src/inference/src/cache_manager.hpp`](https://github.com/openvinotoolkit/openvino/blob/master/src/inference/src/cache_manager.hpp) — `FileStorageCacheManager::write_cache_entry`, non-atomic
* [`src/plugins/intel_npu/src/compiler_adapter/src/driver_compiler_adapter.cpp`](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/src/compiler_adapter/src/driver_compiler_adapter.cpp) — the hand-off to the closed driver compiler
* [openvinotoolkit/openvino#27366](https://github.com/openvinotoolkit/openvino/issues/27366) — "Is Core threading safe?", closed stale, unanswered
* [CPython `ctypes` documentation](https://docs.python.org/3/library/ctypes.html) — `CDLL` releases the GIL; `PyDLL` does not
* `openvino/include/openvino/runtime/core.hpp` and `openvino/c/ov_core.h`, from the 2026.3.1 win_amd64 and 2025.4.1 manylinux wheels — no thread-safety statement
* `rust/torch_c/src/tensor.rs`, `shim_f16_bytes` — takes `Python<'py>`, no `allow_threads`
