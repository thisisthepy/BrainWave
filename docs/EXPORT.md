# `torch.export` — the census re-derived, and the wall behind it

`docs/COMPILE.md` recommended refusing `torch.compile` permanently and spending
the effort on `torch.export` instead, on the strength of a census: 19 rounds,
18 ordinary missing `torch._C` symbols, **zero** in a `Py_BUILD_CORE` file, and
the eval-frame hook never reached. That census is **correct and reproduces
exactly** — round for round, name for name, on this tree.

It was also measured with crude no-ops, and COMPILE.md said so twice: round 19
was "an artefact of my crude no-op returning `None` where a `DispatchKeySet` was
wanted, not a wall", and the whole thing was labelled "a **census of blockers**,
not a claim that export works". This document is what happened when the no-ops
were replaced by implementations.

**The 18 are real. All 18. None was an artefact.** Round 19 was an artefact,
exactly as COMPILE.md said, and it opens onto 11 more blockers of the same
ordinary kind. Then it opens onto two that are not that kind at all, and they
are the result:

* **`torch.fx.Graph()` cannot be constructed in this shim.** Four lines, no
  export involved. `torch._C._NodeBase` is missing 12 of its 19 members and
  stubs 4 of the remaining 7, so `fx.Node.__init__` raises on the *root node*
  of an empty graph. Every graph front end upstream has is built on this.
* **No `TorchDispatchMode` ever sees an operator.** `_aten_dispatch` does not
  consult the mode stack; `aten.rs`'s capture hook runs *after* the kernel and
  records a result it cannot replace. `torch.export`, `make_fx` and Dynamo all
  build their graph from `__torch_dispatch__` callbacks on a mode, and in this
  shim there are none.

So the honest answer to "get `torch.export.export()` to produce a graph for a
small real module" is **no, and the reason is not the 29 names.** They are real
work, they are ordinary binding surface, and closing all of them leaves the two
walls above untouched. §4 sizes both. §6 says what to do.

Measured 2026-09-06, `darwin/arm64`, CPython 3.13, `work/export`.
Reproduction in §7. Gates unmoved: suite **602 ok** (587 + the 15 new tests in
`rust/torch_c/pytests/test_export.py`), `DOCWATCH: PASS`, golden **9691/9691
ops=255** — no Rust changed.

---

## 0. At a glance

| | |
|---|---|
| COMPILE.md's 18, reproduced? | **Yes, identically** — same names, same order, same round 19 stop (§1) |
| How many were artefacts of the crude no-op? | **Zero.** All 18 are raising stubs today (§2) |
| Was round 19 an artefact? | **Yes**, as COMPILE.md said. A real `DispatchKeySet` walks past it (§2.1) |
| Blockers past round 19, same kind | **+11** — 9 more `torch._C` names, an `_InferenceMode`, a 10-member RAII guard family (§2.2) |
| Blockers past round 19, *different* kind | **+3** — a missing kernel, a per-tensor bit, meta storage (§3) |
| Does `torch.export` produce a graph? | **No** (§4) |
| Why not, in one line | **`torch.fx.Graph()` does not build, and no mode sees an op** (§4.1, §4.2) |
| Is that abi3? | **No.** `torch/csrc/fx/node.cpp` is plain C; the mode wall is `aten.rs`'s (§4.3) |
| What *does* record this module | `capture.rs`: 3 nodes, the right 3 operators, **2 of 3 under a different overload** than upstream (§5) |
| Recommendation | Fix the dispatcher entrance first, `_NodeBase` second, names last (§6) |

---

## 1. The census reproduces exactly

`spike/export_depth3.py`, unmodified, on this tree:

```
 0  torch._C._unset_dispatch_mode                    12  TensorBase.is_inference
 1  torch._C._only_lift_cpu_tensors                  13  TensorBase.is_conj
 2  torch._C._set_only_lift_cpu_tensors              14  torch._C._profiler.gather_traceback
 3  torch._C._ensureCUDADeviceGuardSet               15  torch._C._dispatch_tls_is_dispatch_key_included
 4  torch._C._dynamo.guards.set_is_in_mode_...       16  torch._C._functionalization_reapply_views_tls
 5  torch._C._push_on_torch_dispatch_stack           17  torch._C._dispatch_tls_local_exclude_set
 6  torch._C._pop_torch_dispatch_stack               18  STOP: 'NoneType' has no attribute 'has'
 7  TensorBase._is_view                                      @ torch/_subclasses/meta_utils.py:1061
 8  TensorBase.is_mkldnn
 9  torch._C._functorch.is_batchedtensor             reached PEP 523 eval-frame: False
10  torch._C._functorch.is_legacy_batchedtensor
11  torch._C._functorch.is_gradtrackingtensor
```

Eighteen names, one stop, the eval-frame hook never reached. Nothing in
COMPILE.md §3 needs correcting on its own terms. The question this document
asks is the one it explicitly deferred: **what happens when the stand-ins are
replaced by behaviour.**

---

## 2. The re-derived census

`torchnative/src/main/torchnative/export/upstream.py` implements them. It is a
**staging area, not the final home** — every function in it belongs in
`rust/torch_c/src/bootstrap.py` beside `_install_dispatch_keys`, and §8 carries
the hand-off. It monkey-patches at runtime only because it runs after
`import torch`.

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/upstream.py install present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/upstream.py installed_names present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/upstream.py _is_definitely_a_view present -->

### 2.1 All 18 are real; round 19 was the artefact

Every one of the 18 is a **raising stub today**, not a name that quietly works.
`install()` reports 29 names it replaced and the test
`test_the_census_names_were_placeholders_not_implementations` holds the 18
against that list, so this cannot drift into prose.

Two shapes of placeholder, both from `bootstrap.py`, and the difference matters
for anyone checking this:

* `_Unimplemented` — what is left when the stubs say nothing about a name.
  `torch._C._dynamo.guards.set_is_in_mode_without_ignore_compile_internals` is
  the only one of the 18 in this shape.
* a `_make_function` / `_make_property` product — an ordinary Python function or
  property whose body raises `NotImplementedError`. The other 17.

**Round 19 dissolves, exactly as predicted.** `_dispatch_tls_local_exclude_set`
returning a real `DispatchKeySet` — the class `_install_dispatch_keys` already
builds — makes `meta_utils.py:1061`'s `.has(DispatchKey.ADInplaceOrView)` answer
`False`, which is the truth for a shim that has entered no guard. The census
walks past it without further comment.

### 2.2 What is past round 19, of the same kind

Eleven more, in the order they appear, each needing behaviour rather than a
name:

| | name | what it needed |
|---|---|---|
| 19 | `_dispatch_tls_local_include_set` | the include half of the pair |
| 20 | `_dispatch_tls_is_dispatch_key_excluded` | reads the set |
| 21 | `_meta_in_tls_dispatch_include` / `_set_...` | a flag pair |
| 22 | `_get_dispatch_mode`, `_set_dispatch_mode` | the infra-mode slots |
| 23 | `_len_torch_dispatch_stack` | **not a stub** — see below |
| 24 | `_get_dispatch_stack_at` | reads the stack |
| 25 | `_ForceDispatchKeyGuard`, `_ExcludeDispatchKeyGuard`, `_IncludeDispatchKeyGuard` | synthesised types with no `__enter__` |
| 26 | `_InferenceMode` | same |
| 27 | the RAII guard family, 10 of them | same (§2.4) |
| 28 | `_profiler.symbolize_tracebacks` | the other half of `gather_traceback` |
| 29 | `_functorch.is_functorch_wrapped_tensor` | already implemented; left alone |

**Number 23 is the one worth stopping on**, because it is the failure mode this
repository keeps meeting and it was not a missing name at all.
`_len_torch_dispatch_stack` is an entry in `bootstrap.py`'s
`_DISCOVERED_RETURNS` table with the value `0`, and a comment saying "Nothing
pushes onto it here, so it is empty and disabled." That was true when it was
written. Under `torch.export` something *does* push — `FakeTensorMode` and
`ProxyTorchDispatchMode` both do — and a constant zero would have made
`with FakeTensorMode():` a block that entered, reported itself absent, and
changed nothing. That is `docs/COMPILE.md` §5's silent eager fallback wearing a
different hat, and it would not have been caught by any check for missing names,
because the name is not missing.

`install()` reports such names separately (`overridden`, as against `replaced`)
for exactly this reason: "how many placeholders did you fill" and "how many
answers did you change" are different numbers and merging them hides the second.

### 2.3 The three predicates that are `False` as a fact

`is_mkldnn`, `is_inference`, `is_conj`. None is a stand-in:

* `is_mkldnn` — `bootstrap.py`'s build-flag table already answers `_has_mkldnn`
  `False`, and a tensor cannot be in a layout the build does not have.
* `is_inference` — inference mode is autograd TLS state, and there is none here.
* `is_conj` — the conjugate bit is a `TensorImpl` dispatch-key bit; candle has
  no such bit.

### 2.4 The RAII guard family — one failure, ten times

Ten classes (`_DisableTorchDispatch`, `_DisableFuncTorch`, `_DisableAutocast`,
`_AutoDispatchBelowAutograd`, `_RestorePythonTLSSnapshot`,
`_DisablePythonDispatcher`, `_EnablePythonDispatcher`, `_EnablePreDispatch`,
`_PreserveDispatchKeyGuard`, `_SetExcludeDispatchKeyGuard`) are synthesised as
*types* by `bootstrap.py`'s `_ShimMeta`, so they construct and then fail at the
`with`:

```
TypeError: '_DisableTorchDispatch' object does not support the context manager protocol
```

A message that names the class and not the shim, raised at the `with` rather
than at the thing that is missing. They are one family and are fixed as one.

`_DisableTorchDispatch` deserves a note, because "entering a counter is a
correct implementation" is true here **only because of §4.2**. Upstream's
`no_dispatch()` suppresses the mode stack so `fake_tensor.py:502` can build a
meta tensor without re-entering the fake mode. Nothing here consults the mode
stack, so there is nothing to suppress. The counter is recorded rather than
discarded so that the day `_aten_dispatch` learns to consult the stack, this
guard is a body to fill and not a hole to find.

### 2.5 The one place the shim has to say something it cannot fully know

`TensorBase._is_view` and `TensorBase._base`.

Views here are **real** — `docs/VIEWS.md` §6 made every in-place kernel write
through a `Layout` into shared storage — so `False` is not free. Measured side
by side against upstream on the same three tensors:

| | `storage_offset` | `stride` | `numel` | `storage.nbytes()` |
|---|---|---|---|---|
| `x = arange(12).reshape(3,4)` | 0 / 0 | (4,1) / (4,1) | 12 / 12 | 48 / 48 |
| `x[1:, 1:]` | 5 / 5 | (4,1) / (4,1) | 6 / 6 | 48 / 48 |
| `x.t()` | 0 / 0 | (1,4) / (1,4) | 12 / 12 | 48 / 48 |

(shim / upstream — identical in all twelve cells.)

So `_is_definitely_a_view` is a **sound positive** detector built from signals
the shim genuinely has: a non-zero storage offset, non-contiguous strides, or a
footprint smaller than the storage each *prove* a view.

**What it misses, said plainly:** a view that covers the whole storage
contiguously — `x.view(12)`, `x[:]`, `x.reshape(3,4)` on a contiguous `x` — is
indistinguishable from its base under every signal this shim exposes. Upstream
answers `True` there because `TensorImpl` carries a base pointer;
`PyTensorBase` does not. That is one wrong answer, it is in
`rust/torch_c/src/tensor.rs`, and it is recorded here rather than papered over.

`_base` **refuses by name** when `_is_view()` said `True`. There is no base
object to return, and `None` there means "not a view" to the caller —
`meta_utils.py:2246` reads it exactly that way, one line after `_is_view()` told
it the opposite. A module exported with a sliced input therefore fails loudly at
the tensor that caused it, rather than producing a graph whose inputs quietly
lost their aliasing.

### 2.6 The profiler pair, and a contract found the hard way

`gather_traceback` was no-opped to `None` in COMPILE.md's census and the no-op
survived, because nothing in that run symbolised the result. With the later
rounds real, `torch/_logging/_internal.py:1510` does. The contract, read out of
`torch/utils/_traceback.py:180` and `:259`:

```
gather_traceback(python, script, cpp)  -> opaque handle
symbolize_tracebacks([handle, ...])    -> [[{filename, line, name}, ...], ...]
```

`line` is a line *number* — it is `FrameSummary`'s second positional argument —
and the frames come back **innermost first**, because `_extract_symbolized_tb`
reverses them and applies `skip` from the front to elide
`CapturedTraceback.extract`'s own frame. Getting either wrong is a `TypeError`
or a reversed stack several frames from the shim, which is how both were found.

This is a small illustration of the general point: **a no-op census cannot see a
contract, only a name.** Six of the 29 names above needed a specific return
shape, and none of those shapes is visible until something consumes the value.

---

## 3. Three blockers that are not binding surface

Past the 29, in order:

### 3.1 `torch.empty_strided` — no kernel, no table row

```
NotImplementedError: not implemented in torch._C shim: torch.empty_strided(...) --
overload resolution has no table entry for this op (rust/torch_c/src/overloads.json)
```

`overloads.json` has `empty` and `empty_like` and no `empty_strided`; `aten.rs`
has no kernel with that name. It is on the critical path because it is *the*
constructor `meta_utils.py:2008` uses to make the meta tensor behind every fake
tensor, so nothing downstream is reachable without it.

<!-- DOCWATCH: op-not-implemented aten.empty_strided.default -->

### 3.2 `torch._C._set_throw_on_mutable_data_ptr` — a per-tensor bit

`fake_tensor.py:943` marks a freshly built `FakeTensor` so that `.data_ptr()`
raises on it. There is nowhere in `PyTensorBase` to put that bit, and a Python
side-table keyed by identity would be a different guarantee (a `FakeTensor` is
not necessarily weak-referenceable, and the bit has to survive `_make_subclass`).
Rust.

### 3.3 A meta tensor has no storage to memoise

```
NotImplementedError: Cannot copy out of meta tensor; no data!
   @ torch/_subclasses/meta_utils.py:2066   ->  r.untyped_storage()
```

`set_storage_memo(s, r.untyped_storage())` asks a meta tensor for its storage so
that two views of the same base map to the same fake storage. Upstream's meta
tensor has a zero-sized storage; this shim's `tensor.rs::storage_snapshot`
refuses on `Repr::Meta` by design, and that refusal is right — it exists so no
kernel reads bytes that are not there. What is missing is a *storage handle*
that carries identity and size without bytes. Rust, and a design question rather
than a line.

---

## 4. The wall

Past all of that is the result, and it is not a count.

### 4.1 `torch.fx.Graph()` does not build

Four lines, no export, no fake tensors, no modes:

```python
import torch.fx
g = torch.fx.Graph()          #  <-- raises here
```
```
NotImplementedError: not implemented in torch._C shim: _NodeBase._update_args_kwargs
   torch/fx/graph.py:1369   self._root: Node = Node(self, "", "root", "", (), {})
   torch/fx/node.py:356     self._update_args_kwargs(args, kwargs)
```

An empty `Graph` constructs a sentinel root `Node`, so the failure is at
construction and not at the first operator. `torch._C._NodeBase` — upstream's
`torch/csrc/fx/node.cpp`, a C struct plus accessors — is:

| | count | names |
|---|---:|---|
| absent entirely | **12** | `_args`, `_kwargs`, `_input_nodes`, `_repr_fn`, `_sort_key`, `graph`, `meta`, `name`, `op`, `target`, `type`, `users` |
| present, raising stub | **4** | `_prepend`, `_remove_from_list`, `_replace_input_with`, `_update_args_kwargs` |
| present and real | 3 | `_erased`, `_next`, `_prev` |

`torch._C._fx_map_arg` and `torch._C._fx_map_aggregate` are raising stubs too,
and `torch._C._NodeIter` is an empty synthesised type.

**COMPILE.md's census could not have found this**, and that is not a criticism
of it — the census stopped at round 19, and `fx` is reached long after. It is
the reason a blocker census is not a size estimate, which COMPILE.md also said.

### 4.2 No `TorchDispatchMode` sees an operator

With the mode stack working (§2.2), a mode still sees nothing:

```python
with LoggingMode():
    y = (x * 2 + 1).relu()

shim:      SEEN: []
upstream:  SEEN: ['aten.mul.Tensor', 'aten.add.Tensor', 'aten.relu.default']
```

The mode enters, `_len_torch_dispatch_stack()` reports 1 inside the block, and
`__torch_dispatch__` is never called. `_aten_dispatch` does not consult the
stack. `aten.rs`'s hook is *post-hoc*:

```rust
let out = crate::tensor::promote(py, out)?;      // kernel has already run
...
if crate::capture::is_active() { crate::capture::record(py, op, args, kwargs, &out); }
```

That is the right place for a recorder and the wrong place for a mode. A
`__torch_dispatch__` handler must run **instead of** the kernel and **return the
result** — `FakeTensorMode` has no data to compute with, and
`ProxyTorchDispatchMode` returns a `Proxy`, not a tensor. Recording after the
fact cannot do either.

**This is why the two walls are one wall.** Suppose `_NodeBase` were filled in
tomorrow. `torch.export` would then run, build an `fx.Graph`, install a
`ProxyTorchDispatchMode`, call the module, see no `__torch_dispatch__` fire, and
return a graph with a placeholder, an output, and **no operators**. It would be
an `ExportedProgram`. It would print. It would serialise. That is precisely the
half-working graph this round was told to watch for, and it is one plausible,
well-intentioned commit away.

`rust/torch_c/pytests/test_export.py::test_a_graph_front_end_is_not_offered_while_modes_are_not_consulted`
is that guard, and it is written so closing either half makes it demand the
other rather than going quiet.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export.py test_a_graph_front_end_is_not_offered_while_modes_are_not_consulted present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export.py test_capture_is_the_only_working_front_end_and_records_the_module_it_ran present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export.py test_the_dispatch_mode_stack_counts_instead_of_answering_zero present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export.py test_base_refuses_for_a_detected_view_rather_than_answering_none present -->

### 4.3 None of it is abi3

Worth stating because it is the one thing COMPILE.md's recommendation turned on,
and it survives intact. `torch/csrc/fx/node.cpp` is a plain C extension type
with no CPython internals; the mode-stack entrance is this project's own
`aten.rs`. **The road to a graph still does not pass through PEP 523.** It is
longer than the census suggested, and its remaining length is in `rust/`, not in
`bootstrap.py`.

---

## 5. What `capture.rs` records for the same module — and where it disagrees

`capture.rs` is the front end this project already has, and it works:

```python
class M(nn.Module):
    def forward(self, x): return (x * 2 + 1).relu()
```

| | ops recorded |
|---|---|
| `capture.rs` (`_capture_begin` / `_capture_end`) | `aten.mul.Scalar`, `aten.add.Scalar`, `aten.relu.default` |
| upstream `make_fx` on the same module | `aten.mul.Tensor`, `aten.add.Tensor`, `aten.relu.default` |

Three nodes each, one input, one output, same order, same three *operators*.
**Two of the three disagree on the overload.**

That is not cosmetic. Core ATen and ExecuTorch's Edge dialect are defined per
**overload**, and `torchnative.export.decompose` decides what to decompose by
overload key. `aten.mul.Tensor` with a scalar second argument is what upstream's
dispatcher produces for `tensor * 2` — the `Number` is wrapped — while
`capture.rs` records the `.Scalar` overload the shim's own resolution picked.
A lowering table keyed on one spelling silently misses the other.

The comparison the round was asked for — "two front ends onto the same
operators should agree on the ops, and where they disagree that is worth
knowing" — therefore has an answer even though `torch.export` does not run:
**they agree on the operators and disagree on the overloads, on the smallest
possible module, in two of three nodes.** That disagreement is pinned by
`test_capture_is_the_only_working_front_end_and_records_the_module_it_ran` so it
cannot drift, and it is a thing to settle *before* a second front end arrives,
not after.

What this comparison cannot say: upstream's export runs decompositions that
capture does not, and that difference is unmeasured here because export does not
run. It is the interesting half and it stays open.

---

## 6. What to do, in order

The order is not the order of size. It is the order in which each step stops
being able to produce a plausible-looking wrong answer.

1. **The dispatcher entrance.** Make `_aten_dispatch` consult the torch-dispatch
   mode stack *before* `aten_dispatch_inner`, and return the mode's result. This
   is the only item that is a design change rather than a fill, it is the one
   that makes every later item mean something, and until it lands **nothing
   should make a graph front end reachable**. `rust/torch_c/src/aten.rs`.
2. **`_NodeBase`.** 12 members and 4 methods, plus `_fx_map_arg` /
   `_fx_map_aggregate` / `_NodeIter`. Mechanical, testable in isolation
   (`torch.fx.Graph()` either builds or it does not), and worth having on its own
   — `torch.fx` is more than export's substrate.
3. **The three of §3.** `aten.empty_strided`, the mutable-data-ptr bit, and a
   meta storage handle.
4. **The 29 names.** Last, because they are the part that is already written
   (§8) and the part that changes nothing on its own.

And one thing to *not* do: do not fill 2, 3 and 4 and leave 1. That ordering
produces an `ExportedProgram` with no operators in it, and §4.2 is the argument
that it would not look wrong.

---

## 7. Reproduction

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-export
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export TORCH_C_STAGE=/tmp/stage-export
PY=/Volumes/macMini/caches/spike-venv/bin/python

cd rust/torch_c && cargo build --release && cd ../..
bash vendor/install_shim.sh
PYTHON=$PY sh rust/torch_c/pytests/run.sh          # 602 ok, DOCWATCH: PASS

# COMPILE.md's census, unmodified
PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY spike/export_depth3.py

# §4.1 in four lines
PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY -c '
import torch, torch.fx; torch.fx.Graph()'

# §4.2 side by side -- the same script against the shim and against upstream
cat > /tmp/mode_probe.py <<'PY'
import torch
print("shim" if hasattr(torch._C, "_aten_implemented") else "upstream")
if hasattr(torch._C, "_aten_implemented"):
    from torchnative.export import upstream; upstream.install()
from torch.utils._python_dispatch import TorchDispatchMode
seen = []
class Log(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        seen.append(str(func)); return func(*args, **(kwargs or {}))
with Log():
    (torch.ones(3) * 2 + 1).relu()
print("SEEN:", seen)
PY
PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY /tmp/mode_probe.py
env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL $PY /tmp/mode_probe.py
```

`test_export.py` runs its own subprocess with the vendored tree on
`PYTHONPATH`, so it needs `vendor/install_shim.sh` to have run — the same silent
skip as the decompose-road tests, for the same reason.

---

## 8. Hand-off: the `bootstrap.py` patch

`torchnative/src/main/torchnative/export/upstream.py` is where this work lives
today and it is the wrong place. It monkey-patches `torch._C` after
`import torch`, which forces a `rebind()` pass over `sys.modules` to re-point
roughly forty `from torch._C import ...` bindings — including aliases like
`torch/utils/_mode_utils.py:15`'s `no_dispatch = torch._C._DisableTorchDispatch`,
which is why that pass matches by object identity rather than by name. **All of
that disappears in `bootstrap.py`**, which runs before any of those imports.

The patch, in the shape `docs/GLU.md` §1.1 and `docs/SETITEM.md` §2 used:

**1. Move the bodies.** Copy these functions from `upstream.py` into
`bootstrap.py`, immediately after `_install_dispatch_keys` (which ends at line
6961 on this tree), dropping `install()`, `rebind()`, `InstallReport`,
`installed_names()`, `_is_stub`, `_is_ours`, `_mark_ours` and `_MARK` — those
exist to describe and undo a runtime patch and have no meaning in the
bootstrap:

```
_Tls  /  _TLS  /  _included  /  _excluded
_install_tls
_install_mode_stack
_install_tensor_predicates   ( + _is_definitely_a_view )
_install_functorch
_install_profiler            ( + _GatheredFrames )
_install_dynamo_bool
_install_inference_mode
_install_raii_guards         ( + _RAII_GUARDS, _is_guard_active )
```

Each takes `(module, put)` in `upstream.py`; in `bootstrap.py` they take
`module` alone and use plain `setattr`/`module.X = ...` like their neighbours,
since there is nothing to displace.

**2. Call them.** At `bootstrap.py:9779`:

```python
     _install_dispatch_keys(module)
+    _install_dispatcher_tls(module)      # was _install_tls
+    _install_dispatch_mode_stack(module) # was _install_mode_stack
+    _install_tensor_view_predicates(module)
+    _install_functorch_predicates(module)
+    _install_profiler_tracebacks(module)
+    _install_dynamo_mode_flag(module)
+    _install_inference_mode(module)
+    _install_raii_guards(module)
```

(The renames are only to keep `bootstrap.py`'s `_install_*` names describing
what they install rather than where they came from.)

**3. Delete the table row that is now wrong.** `_DISCOVERED_RETURNS` at
`bootstrap.py:6731` has

```python
    "_len_torch_dispatch_stack": 0,
```

and the paragraph above it explaining that nothing pushes onto the stack. Both
go — §2.2 is why. Leaving the row would have the table's constant win over the
real function depending on install order, which is exactly the failure that row
now represents.

**4. Do not touch `set_eval_frame`.** Item 6 above installs a bool setter that
lives under `torch._C._dynamo.guards`, and that is not an opening.
`set_eval_frame`'s refusal is deliberate and tested; `docs/COMPILE.md` §5.1 is
why it must outlive any symbol-filling on this path. Nothing in this patch goes
near it.

After the patch, `upstream.py` should be deleted and `test_export.py`'s
subprocess should stop importing it — the tests below `test_install_*` are about
the installed behaviour and read `torch._C` directly, so only the three
`install()`-report tests change.

---

## 9. What this document does not claim

Split the way `docs/COMPILE.md` §5.3 asks for, because "a round landed" is not a
number:

| | |
|---|---|
| **feature added** | none reaching `_aten_implemented()`; no Rust changed |
| **binding surface implemented** | 29 `torch._C` names, in a staging module, behind a hand-off (§8) |
| **defect found** | `_len_torch_dispatch_stack` answering a constant `0` (§2.2); `_base`/`_is_view` able to disagree with each other (§2.5) |
| **tests added** | 15, in `rust/torch_c/pytests/test_export.py` |
| **measurement** | the re-derived census (§2), the storage-model comparison (§2.5), the two walls (§4), the capture/upstream overload disagreement (§5) |
| **documentation corrected** | none — `docs/COMPILE.md` §3 is accurate as written and §1 says so |

And the negative, stated as a negative: **`torch.export.export()` does not
produce a graph in this shim, and the reason is two walls that a blocker census
is structurally unable to see.** COMPILE.md's recommendation — abi3 only, refuse
`torch.compile`, spend the effort here — is not overturned by that. The effort
is still better spent here than on PEP 523, because the remaining obstacles are
ordinary engineering in this project's own Rust. It is just a longer road than
18 names, and §6 is the order to walk it in.
