# The compiled-model cache, and where torchnative is allowed to write

This document records one change and one house rule. The change is that
`torchnative.export.intelnpu` now sets OpenVINO's `ov::cache_dir`, so a compiled
model survives the process. The house rule is *where* — this repository had no
cache-path convention before this round, so whatever was chosen here becomes the
answer the QNN and CoreML caches inherit.

Evidence: `rust/torch_c/pytests/test_ovcache.py`. Implementation:
`torchnative/src/main/torchnative/_cachedir.py` (the path) and
`torchnative/src/main/torchnative/export/intelnpu.py` (the property).

## 1. What was wrong

`OpenVINO.compile_ir` called

```c
ov_core_compile_model(core, model, device, 0, &out)
```

The `0` is `property_args_size`. With no properties, `ov::cache_dir` is never
set and OpenVINO caches nothing. Two costs, both real and both measured by a
user running Qwen3-4B on a Windows Intel NPU:

* every process restart recompiles every leaf from scratch;
* `_NPULinear` compiles a **static-shape** IR per batch, and `generate()` uses
  two shapes — the prompt length, then 1 per token once the KV cache is warm. A
  36-layer Qwen3-4B lowers to 252 leaves, so that is 252 x 2 = **504 driver
  compiles before the second generated token**, none of which survives.

The module's own docstring already noted that the archived
`intel_npu_acceleration_library` cached at two levels — `ov::cache_dir` at
`inference.h:82` and a pickle at `modelling.py:95-97,112`. We did neither.

## 2. Where the cache lives, and why not under the Hugging Face cache

| platform | root |
|---|---|
| every platform with a filesystem | `$XDG_CACHE_HOME/torchnative/openvino`, default `~/.cache/torchnative/openvino` |
| `emscripten`, `wasi` | **nothing.** No cache. |

**One rule, not one per OS.** An earlier version used each platform's native
convention — `%LOCALAPPDATA%\torchnative\Cache` on Windows, `~/Library/Caches`
on macOS and iOS, XDG elsewhere. That is what a desktop application should do
and it is wrong here, because it makes torchnative the only thing in a user's ML
toolchain that does it. The two that matter ignore the OS:

    torch/hub.py                  DEFAULT_CACHE_DIR = "~/.cache"
    huggingface_hub/constants.py  ~/.cache, XDG_CACHE_HOME consulted first

Neither reads `sys.platform`. On Windows a user's checkpoints are already in
`C:\Users\<u>\.cache\huggingface` and torch's hub artefacts in
`C:\Users\<u>\.cache\torch`; sending only the NPU blobs to `%LOCALAPPDATA%`
splits one thing across two places for a tidiness nobody asked for. This
distribution ships that torch, so its convention is not a foreign one.

Roaming was the argument for `%LOCALAPPDATA%`, since a compiled NPU blob is
keyed to this machine's driver. It is not a real risk: `~/.cache` resolves under
`%USERPROFILE%`, which is not the roaming `%APPDATA%` tree, and
`huggingface_hub` already keeps multi-gigabyte checkpoints there.

**iOS and Android keep the same rule, and give something up to do it.** The
correct directory on each is one only the host application knows —
`Context.getCacheDir()` (`/data/data/<pkg>/cache`) and the sandbox container's
`Library/Caches`. Both need a JNI Context or an Objective-C container URL, and a
pure-Python process can compute neither. `$HOME/.cache` on those platforms is
app-*private* but it is app *data*: Android will not reclaim it under storage
pressure and "Clear cache" will not touch it; on iOS it is not OS-purgeable.

The alternative was to cache nothing on mobile until an embedder says where.
That is worse by a wide margin — it means recompiling every leaf on every
launch, 504 driver compiles for a Qwen3-4B, on exactly the devices where compute
is scarcest. **A cache in a less-reclaimable directory beats no cache.**

So `TORCHNATIVE_CACHE_DIR` is the one thing a mobile host should configure. An
embedder that sets it gets the correct directory; an embedder that does not
still gets a working cache.

**Hugging Face was the other candidate and was rejected.** These artefacts are
derived from HF checkpoints, so sitting under `HF_HOME`/`HF_HUB_CACHE` has a
real argument: it keeps a model's derived files next to it and inherits a
disk-location choice the user has already made. Against it:

* `huggingface_hub` owns that tree's layout and *prunes* it —
  `huggingface-cli delete-cache` walks `models--*/blobs` and `snapshots`. A
  directory we add there is somewhere between at-risk and someone else's.
* It is undefined for the inputs that never came from the Hub: a local
  `safetensors` file, a `state_dict`, a `torch.nn.Module` built in the script.
  That is a real fraction of what this library lowers, so an HF-anchored rule
  would need a non-HF rule beside it anyway.
* PROJECT.md's position on not writing into another distribution's directory
  layout applies to a cache tree as much as to a package tree.

What is kept from the argument in favour is the **inheritance**: a user who has
moved their caches is still honoured, one level up, through `XDG_CACHE_HOME` /
`%LOCALAPPDATA%` and through the two overrides in §3. Setting `HF_HOME` does not
move the torchnative root, and a test asserts that.

## 3. Override and disable

| variable | effect |
|---|---|
| `TORCHNATIVE_CACHE_DIR` | moves **every** torchnative cache root |
| `TORCHNATIVE_OPENVINO_CACHE_DIR` | moves (or disables) just this backend; used verbatim, wins over the above |

Both accept a disable spelling instead of a path: `0`, `off`, `no`, `none`,
`false`, `disable`, `disabled`, or empty, case-insensitive after stripping. The
empty string counts because `VAR= python ...` is how a shell user most often
means "unset this", and treating it as a path would put the cache at `/`.

Disabling is not a corner case. A shared or networked home directory, CI, a
read-only container image and a machine where the user simply does not want
gigabytes of blobs all need it.

The names follow `TORCHNATIVE_OPENVINO_C`, the naming precedent already in
`export/intelnpu.py`: `TORCHNATIVE_` + the thing + what it is.

"Off" is spelled `None` internally — the same value the wasm platforms return —
so a caller has one no-cache state to handle and not two, and the no-cache
compile is byte-for-byte the `property_args_size=0` call that shipped before
this round rather than a third code path.

## 4. When the directory cannot be written

`ensure_cache_dir` creates the directory and then **writes and removes a probe
file**. Not `os.access`: that answers with the real uid's permission bits and
gets network filesystems, read-only mounts, ACLs, full disks and container
overlays wrong in both directions, and the failure it misses would surface later
from inside the OpenVINO plugin, which is the worst place for it.

On failure the compile still happens, without a cache — degraded, not broken.
Refusing to compile because a directory is read-only would be worse than the
problem.

But not silently. `docs/graph/NPU2.md`'s position, arrived at the hard way on
CoreML, is that the silently degraded path *is* the defect, so this raises an
`IntelNPUCacheWarning` naming the directory, the OS error, the cost of not
caching, and `TORCHNATIVE_OPENVINO_CACHE_DIR` as the fix.

**Once.** With 504 compiles in a `generate()`, a warning per compile would be
504 identical lines, which is the same as silence with extra steps. Two things
hold the number down: the directory is resolved and probed once per `OpenVINO`
instance, at construction, not per `compile_ir`; and a module-level set of
already-announced directories covers the case of several `OpenVINO` instances in
one process. A test asks twenty times and asserts it heard exactly one warning.

## 5. Cache-key correctness, and what is NOT verified

The dangerous failure of a cache is not a miss, it is a wrong hit. A blob that
returned another layer's weights would be far worse than no cache: the answers
would be plausible and wrong, which is the exact shape this module's
`EXECUTION_DEVICES` assertion exists to refuse elsewhere.

OpenVINO keys a cached blob on the **model**, the **device** and the **compile
config**. This module varies exactly two things between compiles:

* the **batch dimension**, which `linear_ir` writes into the IR text as
  `<dim>N</dim>`, so two shapes of the same layer are two different model
  documents;
* the **weights**, which are the `Const` payload handed to `read_model` as the
  weights tensor, so two layers of the same shape are two different models.

Both are inside the model, so a wrong hit would require a hash **collision**,
not a key that omits what we vary.

**That last sentence is OpenVINO's contract, and it is not verified here.**
There is no Intel NPU and no OpenVINO runtime on the machine this was written on
— `library_candidates` refuses on `darwin` by design and `import openvino`
fails. `test_ovcache.py` verifies the *premise* (that the IR text and the weight
blob really do differ along those two axes) and stops there, saying so in the
test's own docstring. If OpenVINO's hash did not cover the constant data, every
same-shaped `Linear` in a Qwen3-4B would collide and the model would emit
garbage from the first token — loud rather than subtle, and `TORCHNATIVE_OPENVINO_CACHE_DIR=0`
is the switch to confirm it. That check needs the hardware and has not been run.

Also unverified for the same reason: that a cache hit actually occurs on the
second process, that it is faster, and that the blob OpenVINO writes decodes to
the same compiled model. Everything in §2, §3 and §4 *is* verified, because the
path logic is pure and takes its platform and environment as arguments.

## 6. The C call

```c
ov_core_compile_model(core, model, device, 2, &out, "CACHE_DIR", "<dir>")
```

`ov_core.h:204` defines `property_args_size` as "How many properties args will
be passed, each property contains 2 args: key and value" — it is the **arg**
count, not the pair count, and the C side rejects an odd one. One property is
therefore `2`.

Still the variadic `ov_core_compile_model`, and still **not**
`ov_core_compile_model_props`. The reasoning already recorded in
`load_openvino_c` is that the non-variadic entry point is on OpenVINO master but
not in every release a user has installed; passing properties does not change
that, so it survives this round intact and only the count went from 0 to 2.

The key itself is read from the runtime when the runtime surfaces it:
`ov_property_key_cache_dir` is declared `OPENVINO_C_VAR(const char*)` at
`openvino/c/ov_property.h:97-98`, i.e. an exported **variable**, so it is read
with `ctypes.c_char_p.in_dll` rather than called. The literal `"CACHE_DIR"` is
the fallback for a runtime that does not surface data symbols; both spellings
are the same string, which is the only reason taking the fallback silently is
safe.

## 7. Done in a later round: eager compilation, and one Core

This section used to read "Not done": `to(device.npu)` *could* pre-compile the
batch=1 decode shape for every leaf, so those 252 compiles happen during `to()`
rather than stalling the first generated token. That landed, together with the
larger defect underneath it -- `_NPULinear` was building **one `ov::Core` per
leaf**, so a 252-leaf model constructed 252 of them, each resolving and creating
the cache directory this document is about. It now constructs one.

`docs/devices/NPUPAR.md` is the record, including why the compiles are still
**serial**.

## 8. Concurrency, and what this cache directory does and does not promise

Nothing above changed, but the parallel-compilation question forced the cache
path to be read under concurrency, and two facts belong here rather than only in
NPUPAR.md:

* **Within one process and one `ov::Core`, concurrent writes are guarded.**
  `CacheGuard` (`src/inference/src/cache_guard.hpp`) is a per-hash mutex --
  "protect multiple threads to modify the same cached network". Different models
  hash differently, take different locks, and write different `<hash>.blob`
  files. This is another reason the one-Core-per-leaf shape was wrong: 252 cores
  have 252 unrelated guards.
* **Across processes there is no guard, and the write is not atomic.**
  `FileStorageCacheManager::write_cache_entry`
  (`src/inference/src/cache_manager.hpp`) opens the final `<hash>.blob` path
  directly -- no temporary file, no rename. Two scripts running at once, or a
  kill mid-write, can leave a truncated file where a cache entry should be. The
  read side parses a header and discards the entry on any exception, so
  truncation is normally *detected*; there is no checksum over the body.

This is why the directory is **per user** and not shared. A shared cache
directory would widen exactly that window, across users who cannot see each
other's runs.
