# `torch.compile` and abi3 — the fork, and which way to take it

**The conflict is real for this build, but it is not the conflict that was
recorded.** What abi3 forbids is not "Dynamo"; it is one thing inside Dynamo —
CPython's PEP 523 frame-evaluation hook. Everything else `torch.compile` needs,
including the largest single file in `torch/csrc/dynamo/`, is ordinary
pybind11 that a Limited-API extension may have. That narrowing does not save
the feature, because the frame hook is the part you cannot do without, but it
does change what the alternatives are — and it turns up a *different* path to a
graph, `torch.export`, whose blockers contain no CPython internals at all.

And there is something to fix before any of that is decided. Today
`torch.compile` fails loudly, but only by accident: it dies on a missing
**data** module (27 string constants) that has nothing to do with compilation.
`bootstrap.py` already ships `set_eval_frame` as a state cell that installs no
hook. **Close that unrelated data gap — an obviously-correct fix somebody will
make — and `torch.compile` silently starts returning the eager function.**
§5 reproduces that in five lines on today's tree.

Measured 2026-09-06, `darwin/arm64`, CPython 3.13, against
`rust/torch_c` at `work/compile`. Reproduction in §8.

---

## 0. At a glance

| | |
|---|---|
| Is the conflict real? | **Yes, but confined to PEP 523 frame evaluation** (§1) |
| `Py_BUILD_CORE` in `torch/csrc/dynamo/` | 6 of 28 files — **5 are the frame hook; the 6th uses it for two iterator struct layouts and carries its own fallback** (§1.2) |
| What `torch.compile` does today | Raises `AttributeError: '_Unimplemented' object has no attribute 'AOTINDUCTOR_DIR'` — a real refusal, from an unrelated cause, naming nothing (§2) |
| Is it a silent no-op today? | **No — but it is one gap away from being one** (§5) |
| `docs/FASTPATH.md`'s `_compile_fast_path` | **Unrelated.** Positional-argument dispatch, no connection to `torch.compile` (§2.1) |
| Is a narrower subset reachable? | **Yes: `torch.export`.** 18 missing `torch._C` symbols, **zero** in a `Py_BUILD_CORE` file, and the chain never reaches an eval-frame symbol (§3) |
| Cost of two wheel flavours | Not 7 → 14. **7 + 6 per CPython minor, forever** — and it buys only the *possibility* (§4) |
| Recommendation | **Ship abi3 only. Refuse `torch.compile` by name, now, before §5 lands. Spend the effort on `torch.export`.** (§7) |

---

## 1. Is the conflict real at the level everyone assumes?

The recorded fact is compile-time and about upstream's C sources: `Py_BUILD_CORE`
appears in 6 of `torch/csrc/dynamo/`'s 28 files. This project does not build
those sources — it replaces `torch._C` with Rust. So the fact does not transfer
by itself. The question is what `torch.compile` needs *from us*.

### 1.1 What it needs from us, measured

`spike/compile_depth.py` calls `torch.compile(f, backend="eager")(x)`, and on
each failure patches only the object named in the failing line and goes again.
The chain, on today's tree:

```
1. torch._C._export.pt2_archive_constants          data, 27 strings, no behaviour
2. torch._C._dynamo.eval_frame.set_skip_guard_eval_unsafe    bookkeeping flag
3. torch._C._dispatch_tls_local_include_set        dispatcher TLS
4. torch._C._dispatch_tls_local_exclude_set        dispatcher TLS
5. torch._C._ForceDispatchKeyGuard                 context manager
```

None of those five is abi3-impossible. But filling them does not produce a
compiler — it produces §5's silent eager fallback, because the thing that would
have to work, `set_eval_frame`, is a Python-level cell that remembers a callback
and installs no hook. `docs/DYNAMO.md` §13–14 established that with counters
(`python-level calls to f: 3`, `dynamo frame counters: {}`) and §5 below
reproduces it on today's tree.

So the answer to "does it need those 6 files, or a set of symbols that happen to
live in them" is: **it needs one capability that lives in those files, and no
amount of symbol-filling substitutes for it.** The capability is making CPython
re-enter our callback on every bytecode frame.

### 1.2 The asymmetry with autograd, pulled on

`docs/AUTOGRAD.md` found `torch/csrc/autograd/` at 0 of 129 files. Dynamo is 6
of 28. That looked like "advanced features are core-coupled", but reading the 6
files says otherwise — **the `Py_BUILD_CORE` in each is scoped
(`#define` … `#undef`) and you can see exactly what it buys:**

| file | lines | internal headers it opens | what for |
|---|---:|---|---|
| `cpython_includes.h` | 78 | `pycore_frame`, `pycore_interpframe`, `pycore_interpframe_structs`, `pycore_ceval`, `pycore_pystate`, `pycore_code`, `pycore_genobject`, `pycore_stackref` | the frame-hook header itself |
| `eval_frame.c` | 877 | `pycore_interpframe`, `pycore_code`, `pycore_stackref` | `_PyInterpreterState_SetEvalFrameFunc` |
| `cpython_defs.c` | 407 | `pycore_frame`, `pycore_interpframe`, `pycore_opcode`, `pycore_opcode_metadata`, `pycore_genobject`, `pycore_stackref` | hand-copied CPython frame internals |
| `framelocals_mapping.cpp` | 216 | `pycore_code` | reading frame locals |
| `stackref_bridge.c` | 24 | `pycore_stackref` | 3.14 stackref shim |
| **`guards.cpp`** | **8877** | **`pycore_tuple`, `pycore_range`** | **`_PyTupleIterObject` / `_PyRangeIterObject` layouts, for one guard fast path** |

The last row is the interesting one. `guards.cpp` is **larger than the other
five put together, five times over**, and its entire core dependency is two
struct layouts for iterating tuples and ranges quickly — for which **upstream
already ships its own fallback**, a hand-copied struct definition used on
CPython < 3.12:

```c
#if IS_PYTHON_3_12_PLUS
#define Py_BUILD_CORE
#include <internal/pycore_range.h>   // _PyRangeIterObject
#include <internal/pycore_tuple.h>   // _PyTupleIterObject
#undef Py_BUILD_CORE
#else
typedef struct { PyObject_HEAD Py_ssize_t it_index; PyTupleObject* it_seq; }
    _PyTupleIterObject;   // "Manually create ... Copied from CPython"
#endif
```

So the honest count is **5 of 28 files, ~1600 lines, all of them the frame
hook** — and `set_is_in_mode_without_ignore_compile_internals`, one of the
`torch._C._dynamo.guards` symbols the export path wants (§3), is a two-line bool
setter at `guards.cpp:134` that touches none of it.

**Verdict: the conflict is real and it is exactly one capability wide.** It is
not "dynamo is core-coupled". It is "PEP 523 is core-coupled", which is a
tighter and more defensible statement, and it is the one that should replace the
recorded conflict.

Why PEP 523 is not reachable under `Py_LIMITED_API`, restated from
`docs/DYNAMO.md` §15 rather than re-derived: `_PyInterpreterState_SetEvalFrameFunc`
is private by name, `_PyInterpreterFrame` is a layout that changes shape between
CPython minors (`cpython_includes.h` carries four sets of conditionals for
3.11/3.12/3.13/3.14), and abi3's whole promise is that one binary loads into all
of them. Upstream solves this by recompiling per CPython version. That is the
opposite of what an abi3 wheel is.

---

## 2. What `torch.compile` does today

```
>>> torch.compile(f)
AttributeError: '_Unimplemented' object has no attribute 'AOTINDUCTOR_DIR'
  at torch/export/pt2_archive/constants.py:5
```

Identical for `backend="eager"`, `"aot_eager"` and `"inductor"` — the failure is
in `get_compiler_fn`'s unconditional import chain, before the backend matters.

Of the three possibilities in the brief — refuse, silently no-op, partially work
— **this is "refuse", which is the good one.** Two qualifications:

- It refuses **at construction**, not at call. `torch.compile(f)` itself raises;
  the user never gets a callable back. Nothing partially works.
- It names nothing. `AOTINDUCTOR_DIR` is a path constant inside `.pt2` archive
  handling. A user reading that traceback learns that a string is missing from a
  module they have never heard of. They do not learn that `torch.compile` is
  unavailable in this build, or why, or that it will stay that way.

**And it is refusing for the wrong reason** — which is §5's problem.

### 2.1 `docs/FASTPATH.md`'s `_compile_fast_path` is not this

The brief expected `_compile_fast_path` in `bootstrap.py` to be a `torch.compile`
path. It is not, and the name collision is worth writing down so the next reader
does not spend the same half hour. `_compile_fast_path`
(`rust/torch_c/src/bootstrap.py`) `exec`-compiles a per-operator Python closure
that calls `dispatch(key, arg, arg, ...)` positionally instead of building a
`**kwargs` dict. It is a dispatch optimisation for *every* `torch.*` call. It
has no relationship to Dynamo, PEP 523, graphs or backends. **Nothing in this
repository has ever implemented any part of `torch.compile`.**

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _compile_fast_path present -->

---

## 3. The narrower thing that *is* reachable: `torch.export`

`docs/DYNAMO.md` §17 left this unmeasured as item 2 — "`torch.export` (a
different path from Dynamo, FakeTensor-based, does not use the eval-frame
hook): how far does it get in this shim?" Measured now, and it is the most
useful result in this document.

`spike/export_depth3.py` runs `torch.export.export(M(), (x,))`, no-opping one
missing symbol at a time and recording each. **19 rounds, and the eval-frame
hook never appears:**

```
 1  torch._C._unset_dispatch_mode                   autograd/init.cpp
 2  torch._C._only_lift_cpu_tensors                 utils/python_dispatch.cpp
 3  torch._C._set_only_lift_cpu_tensors             utils/python_dispatch.cpp
 4  torch._C._ensureCUDADeviceGuardSet              Module.cpp
 5  torch._C._dynamo.guards.set_is_in_mode_without_ignore_compile_internals
                                                    dynamo/guards.cpp:134
 6  torch._C._push_on_torch_dispatch_stack          autograd/init.cpp
 7  torch._C._pop_torch_dispatch_stack              autograd/init.cpp
 8  TensorBase._is_view
 9  TensorBase.is_mkldnn
10  torch._C._functorch.is_batchedtensor            functorch/init.cpp
11  torch._C._functorch.is_legacy_batchedtensor     functorch/init.cpp
12  torch._C._functorch.is_gradtrackingtensor       functorch/init.cpp
13  TensorBase.is_inference
14  TensorBase.is_conj
15  torch._C._profiler.gather_traceback             profiler/python/init.cpp
16  torch._C._dispatch_tls_is_dispatch_key_included utils/python_dispatch.cpp
17  torch._C._functionalization_reapply_views_tls   utils/python_dispatch.cpp
18  torch._C._dispatch_tls_local_exclude_set        utils/python_dispatch.cpp
19  STOP: 'NoneType' has no attribute 'has'  @ torch/_subclasses/meta_utils.py:1061
```

Round 19 is **an artefact of the crude no-op**, not a wall: `_dispatch_tls_local_exclude_set`
returning `None` instead of a `DispatchKeySet` is what `meta_utils.py` then
calls `.has()` on. The census stops there because a no-op stops being
informative once the caller needs a real value, not because something structural
appeared.

**What matters is what is *not* in that list.** Every entry lives in
`autograd/init.cpp`, `utils/python_dispatch.cpp`, `Module.cpp`,
`functorch/init.cpp`, `profiler/python/init.cpp` or `TensorBase` — and
`grep -rl Py_BUILD_CORE` over `autograd/`, `profiler/` and `functorch/` returns
**zero files**. The one `dynamo/guards.cpp` entry is the bool setter from §1.2.
These are dispatcher TLS reads, tensor predicates and a traceback grab: ordinary
pybind11 work, the same kind already done hundreds of times in this crate.

**Two honesty limits on this result**, so it is not read as more than it is:

1. A no-op is not an implementation. This is a **census of blockers**, not a
   claim that export works or a size estimate for making it work. Several of the
   18 need real return values, and what lies past round 19 is unmeasured.
2. It was measured on one trivial `nn.Module`. A real model will want more.

What it *does* establish, and this is enough to decide with: **the road to a
graph that does not pass through PEP 523 exists, is already partly open, and
none of its known obstacles are abi3 obstacles.**

This is also the same door `docs/CAPTURE.md`'s `_aten_dispatch` capture already
walks through — see §6.

---

## 4. The options, and what each costs

### Option A — two wheel flavours per platform

Ship the abi3 wheel for portability and a version-specific wheel that could
carry a frame hook.

**The build-matrix cost is worse than "7 → 14".** The two flavours do not scale
the same way. That is the entire point of abi3:

| | abi3 flavour | version-specific flavour |
|---|---|---|
| wheels per CPython release | **0 more** | **+6** (macOS, Android, iOS device, iOS sim, Linux, Windows) |
| today (3.13 + 3.14 in the wild) | 6 | 12 |
| after 3.15 | 6 | 18 |
| after 3.16 | 6 | 24 |

`docs/WHEEL.md` §2.3 measured a 3.13-built binary loading and computing under
CPython 3.14.7 with no rebuild. A version-specific flavour gives that up
permanently, and the growth is unbounded because CPython does not stop shipping.

**CI cost** is worse than the wheel count suggests, because the wheels are not
the expensive part. Each row of the matrix needs a target CPython distribution
on the build host, a cross toolchain (`cargo-zigbuild` for Linux, `cargo-xwin`
for Windows, NDK for Android, two Apple SDKs for iOS), a `verify_<platform>.py`
run, and — for Android and iOS — a device or emulator, of which this machine
can run **one at a time** (CLAUDE.md). Multiply the emulator-serialised part by
the number of live CPython minors.

**Code cost, which dominates everything above.** A non-abi3 wheel does not give
you `torch.compile`; it gives you *permission to write it*. What would then have
to be written, in Rust, none of it existing today:

- a PEP 523 hook against `_PyInterpreterFrame`, with a separate struct-layout
  path per CPython minor (upstream's `cpython_includes.h` carries four);
- the guard tree — `guards.cpp` is 8877 lines;
- frame-locals extraction, bytecode transformation, and the rest of
  `torch/csrc/dynamo/`;
- a **backend**. `backend="eager"` is a debugging backend; the default is
  TorchInductor, which generates code and invokes a compiler at runtime.
  `docs/DESIGN.md` §5 already records that iOS W^X forbids runtime code
  generation — so on the platform this project exists for, the fully-paid
  version of this option still does not deliver the headline feature.

`docs/ABI3.md` and `rust/torch_c/Cargo.toml` also record that the asymmetry runs
the wrong way for reversibility, and that is worth restating here because it is
the one part of this option that is *cheap*: Limited API is a subset, so
abi3 → version-pinned needs no source change, while the reverse means hunting
down every private API already in use. **Nothing about staying on abi3 forecloses
Option A later.** It can be taken the day someone is actually willing to write a
frame hook.

<!-- DOCWATCH: symbol-in-file rust/torch_c/Cargo.toml abi3-py313 present -->

### Option B — abi3 only, refuse `torch.compile` by name

**Wheel and CI cost: zero.** No matrix change, no new toolchain, no new target
CPython.

**User-visible behaviour**: `torch.compile(f)` raises immediately with a message
that says it is unavailable in this build, why (PEP 523 is outside the stable
ABI), and what to use instead. The user finds out at the call, not at some
import three frames down that mentions a `.pt2` archive constant.

**What it costs**: the README can no longer imply that a headline PyTorch
feature is present. That is a cost in claims, not in capability — the capability
is already absent, and §5 shows it is currently absent in a way that is about to
get worse rather than better.

### Option C — a narrower subset of `torch.compile`

Asked precisely: is any part of `torch.compile` reachable without
`Py_BUILD_CORE`?

**No.** Every entry point into Dynamo goes through
`eval_frame.py:compile_wrapper`, whose first statement is `set_eval_frame(...)`,
before it looks at what it was handed. `docs/DYNAMO.md` §11 measured three
models — a lambda, an `nn.Linear`, a `TransformerEncoderLayer` — dying at the
identical line. There is no partial mode, no fallback flag, and no subset of
guards or backends that reaches a graph without the hook first.

**But the question was the right one, and it has an answer one door over.**
`torch.export` (§3) gets to a graph without PEP 523 at all, and `docs/CAPTURE.md`'s
`_aten_dispatch` capture already does so today, inside abi3. **Option C is real;
it is just not spelled `torch.compile`.** See §6.

---

## 5. The thing to fix regardless of which option is chosen

`bootstrap.py` ships this (`if "_dynamo" in roots:`, comments abridged):

```python
_eval_frame_cell = [None]

def set_eval_frame(callback):
    prior = _eval_frame_cell[0]
    _eval_frame_cell[0] = callback
    return prior
```

Its own comment is honest about what it is — "a place to remember what the
(never-consulted) hook was last set to" — and the reasoning behind it is sound:
`torch/_dynamo/__init__.py:133` rebinds `torch.manual_seed` through
`_disable_dynamo` at import time, so merely `import transformers` calls into
this. The cell exists to let an import succeed.

**The problem is what it does when `torch.compile` is actually called.** It
returns cleanly. It does not raise. Dynamo proceeds, finds no hook installed,
and the wrapped function runs in Python.

Today the user is protected from that by pure accident — the unrelated
`pt2_archive_constants` `AttributeError` from §2 fires first. `spike/silent_eager.py`
removes the accident, filling that data module and the four dispatcher blockers
from §1.1, and asks the only question that matters:

```
torch.compile(f)(x) returned: tensor([3., 3., 3.])
python-level calls to f: 3         (3 == the function ran eagerly, three times)
dynamo frames counter: {}          (nothing was ever traced)
dynamo stats counter : {}

VERDICT: SILENT EAGER FALLBACK
```

**Five stubs, none of them about compilation, and `torch.compile` becomes a lie.**
And `torch._C._export.pt2_archive_constants` is 27 string constants with no
behaviour whatsoever — the single most obviously-correct fix in the vicinity,
the kind of thing that gets closed in passing while doing something else. When
it is closed, this build starts telling users their model was compiled.

This is the worst of the three outcomes the brief named, and it is one commit
away in either direction. **The refusal should go in before the data gap
closes, not after.**

### 5.1 The fix, precisely — and why it is not in this change

The refusal belongs in `bootstrap.py`, which is another agent's file this round,
so it is written out here rather than applied. It goes in the same
`if "_dynamo" in roots:` block, and the shape is:

```python
def set_eval_frame(callback):
    # The import-time path must still work: `torch/_dynamo/__init__.py:133`
    # calls through here via `_disable_dynamo` on a plain `import transformers`,
    # and `callback is None` is the "uninstall" spelling that path uses.
    if callback is not None:
        raise NotImplementedError(
            "torch.compile is not available in this build. It requires "
            "CPython's PEP 523 frame-evaluation hook "
            "(_PyInterpreterState_SetEvalFrameFunc), which is outside the "
            "stable ABI this extension is built against (abi3-py313); see "
            "docs/COMPILE.md. Use torch.export or the capture API "
            "(docs/CAPTURE.md) for a graph."
        )
    prior = _eval_frame_cell[0]
    _eval_frame_cell[0] = callback
    return prior
```

Three notes for whoever applies it:

- **`callback is None` must stay a no-op.** That is the uninstall spelling, and
  it is the one the import-time path uses. Refusing it breaks
  `import transformers`. A refusal that fires only on a non-`None` callback
  fires only when something is genuinely trying to install a hook.
- **The test must be able to fail.** Assert on `torch.compile(f)(x)` raising
  `NotImplementedError` *and* on `import transformers` still succeeding — the
  second is what pins the `None` carve-out, and without it the guard could be
  tightened to always-raise and nothing would notice until an unrelated suite
  broke.
- **It survives the data gap closing.** Unlike today's accidental refusal, this
  one does not depend on `pt2_archive_constants` staying unimplemented, which is
  the entire point.

`torch._C._dynamo.eval_frame.set_eval_frame` is the right place rather than
`torch.compile` itself: `torch/__init__.py` is vendored upstream source that
this project does not modify (`docs/DESIGN.md` §1), and the cell is ours.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py set_eval_frame present -->

### 5.2 A second silent no-op, found while measuring

Not asked for, and reported rather than acted on. `torch.jit.trace` **returns
the module unchanged and traces nothing:**

```
jit.trace returned: Counted | is it the module itself? True
  calls to python forward during traced call: 1   (0 would mean a trace ran)
  has .graph? False
```

The cause is `bootstrap.py:107`, `os.environ.setdefault("PYTORCH_JIT", "0")`,
added by `docs/TORCHSCRIPT.md` for a good reason (it is what stops
`@torch.jit.script_method` at `torch/utils/mkldnn.py` import time). Upstream's
`torch/jit/_script.py` honours that flag by returning the function untouched —
so the behaviour is upstream-faithful. **What is not upstream-faithful is that
this build turns the flag on by default**, making a silent no-op the default
where upstream makes it opt-in. A user who sets nothing gets a `torch.jit.trace`
that quietly does nothing.

This is the same class of problem as §5 and probably wants the same treatment,
but it belongs to `docs/TORCHSCRIPT.md`'s decision, not this one. Flagged,
unmeasured beyond the above, and left.

---

## 6. What is on the other side, if the effort goes there instead

Two paths to a graph already exist inside abi3, and they are not in competition
with each other:

| | `torch.compile` | `torch.export` (§3) | capture (`docs/CAPTURE.md`) |
|---|---|---|---|
| intercepts at | CPython bytecode frames (PEP 523) | FakeTensor + dispatch | `_aten_dispatch`, one hook |
| reachable under abi3 | **no** (§1) | **nothing known says no** — 18 ordinary symbols | **already works** |
| region selection | automatic, with graph breaks | whole callable | **manual** `_capture_begin`/`_capture_end` |
| status here | 0% | blockers censused, not implemented | working, bit-exact replay |
| what a delegate gets | — | `ExportedProgram` | `CaptureTrace`, Core ATen lowered |

`docs/DYNAMO.md` §16 already made the case that capture and `torch.compile` are
**different products** rather than one being a subset of the other: capture is
NPU-delegate infrastructure, `torch.compile` is a transparent accelerator for
arbitrary Python. That framing survives this document. What this document adds
is that **`torch.export` sits between them** — more automatic than capture (no
manual region markers), and unlike `torch.compile`, not structurally barred.

---

## 7. Recommendation

**Take Option B, and treat §3 as where the released effort goes.**

1. **Ship abi3 only.** The build matrix stays at 6 platform wheels and does not
   grow with CPython. The alternative grows by 6 per CPython minor forever, and
   pays that before writing the first line of a frame hook — and even fully
   paid, iOS W^X still blocks the default backend (§4A).
2. **Land the §5.1 refusal before the `pt2_archive_constants` gap closes.**
   This is the time-sensitive part. Today's loud failure is an accident, and the
   accident is one obvious commit from ending.
3. **Say it in the README by name**, next to the platform table: `torch.compile`
   is not available, because PEP 523 is outside the stable ABI that makes one
   wheel serve every CPython 3.13+. That is a comprehensible trade, and stating
   it is better than a user discovering it through `AOTINDUCTOR_DIR`.
4. **Put the graph story on `torch.export` and capture.** §3's 18 blockers are
   the same kind of work this crate already does well, and none of them is a
   CPython internal.
5. **Leave Option A open and cheap.** Nothing here forecloses it; `Cargo.toml`
   already records that abi3 → version-pinned needs no source change. Revisit if
   someone is genuinely prepared to write a per-CPython-minor frame hook — and
   note that the day that happens, `guards.cpp`'s 8877 lines turn out to be
   abi3-compatible anyway (§1.2), so the frame hook is the whole of what a
   second flavour would be *for*.

**What would change this recommendation:** a Rust PEP 523 binding appearing that
someone else maintains across CPython minors, or a decision that the desktop
platforms matter enough on their own to justify a desktop-only second flavour
(3 rows, not 6). Neither is true today.

---

## 8. Reproduction

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-compile
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
(cd rust/torch_c && cargo build --release) && bash vendor/install_shim.sh

PY=/Volumes/macMini/caches/spike-venv/bin/python
export PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1

$PY spike/compile_depth.py eager   # §1.1  the 5-step chain
$PY spike/export_depth.py          # §3    three front doors compared
$PY spike/export_depth2.py         # §3,§5.2  export past the data gap; jit.trace
$PY spike/export_depth3.py         # §3    the 18-symbol census
$PY spike/silent_eager.py          # §5    SILENT EAGER FALLBACK
```

The `spike/` scripts are **not part of the crate** and nothing that ships
imports them; each says so in its docstring. They monkey-patch `torch._C` at
runtime and write nothing.

§1.2's file table comes from the upstream source tree outside this repository:

```sh
P=/Volumes/macMini/caches/pytorch-spike/pytorch/torch/csrc
grep -rl Py_BUILD_CORE $P/dynamo/                      # 6 of 28
grep -rl Py_BUILD_CORE $P/autograd/ $P/profiler/ $P/functorch/   # 0
sed -n '95,115p' $P/dynamo/guards.cpp                  # the scoped #define
```

## 9. Unverified

| # | item | state |
|---|---|---|
| 1 | What lies past round 19 of the `torch.export` census (§3) once the no-ops are replaced by real values | **not measured.** The census is a blocker list, not a size estimate |
| 2 | Whether a real model's `torch.export` needs symbols beyond the 18 | **not measured** — one trivial `nn.Module` only |
| 3 | Whether a desktop-only second flavour (3 rows) is worth costing separately | not costed. §7 names it as the thing that would change the recommendation |
| 4 | Whether the §5.1 refusal breaks any transformers path that reaches `set_eval_frame` with a non-`None` callback without the user asking for `torch.compile` | **not measured.** `docs/DYNAMO.md` §8 item 6 left the same question open. Whoever applies §5.1 should run the transformers suites, not just `import transformers` |
| 5 | §5.2 `torch.jit.trace` — whether the silent no-op has other reachable spellings, and what the right fix is | flagged only; belongs to `docs/TORCHSCRIPT.md` |
