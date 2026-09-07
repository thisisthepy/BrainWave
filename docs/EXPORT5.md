# `torch.export` — it returns a graph, the graph runs, and the numbers are upstream's

`docs/EXPORT4.md` §3 listed four claims and answered **"not reached"** to three
of them. This round answers all four **yes**, and the scope of that yes is four
hand-written modules. On forty real `transformers` architectures, upstream
exports, replays and agrees on ten, and this shim on **zero**. Both halves are
the result and neither is the headline on its own.

Two things stood between EXPORT4 and here, and this round took both:

* **Wall 8, the meta storage handle** (`docs/EXPORT.md` §3.3), left open by
  EXPORT4 as a storage-model design question. §2 is the design, including the
  table of which of upstream's expectations it meets and which it **refuses by
  name**.
* **`docs/EXPORT.md` §8's hand-off.** The 29 census names lived in a staging
  module and every export measurement in this repository had been taken under a
  runtime monkey-patch. §7 moved them. It is a separate, delimited step (§12
  says which gates covered which), and it turned out to be a **prerequisite for
  measuring anything**: before it, `export_sweep.py` against the shim stopped at
  census name #0 on all forty architectures.

Behind wall 8 were seven more, and one of them was not a gap but the failure
`docs/EXPORT.md` §4.2 predicted in prose, arriving in the flesh: for part of
this round `torch.export.export()` **succeeded**, returned an `ExportedProgram`,
printed, serialised — and its graph contained **no operators**. §6 is what that
was and how it was found.

Measured 2026-09-07, `darwin/arm64`, CPython 3.13, `work/export5`, on develop
`c6e4a3a`.

---

## 0. At a glance

| | |
|---|---|
| Does `torch.export.export()` return an `ExportedProgram`? | **Yes** (§8) |
| Does the graph it holds run? | **Yes** (§8) |
| Does it produce upstream's numbers, element-wise? | **Yes — bit-identical**, at a tolerance derived from upstream's own f32-vs-f64 error (§8) |
| On how many modules? | **4 hand-written.** On 40 `transformers` architectures: upstream 10, this shim **0** (§10) |
| Wall 8, the meta storage handle | **closed** (§2) |
| `docs/EXPORT.md` §8's hand-off | **paid** — 32 names now in `bootstrap.py`, no monkey-patch (§7) |
| Defects found | **3** — a guard that restored nothing (§5), `to(memory_format=)` silently dropping the request (§3), `_functionality_to_backend_keys` answering `[]` for a non-functionality key (§11) |
| Of those, found by a *nullification* rather than by a test | **1** (§11) |
| Nullifications attempted / uncaught | **17 / 1** (§11) — and the uncaught one is the most useful finding here |
| Walls remaining | 3 named, all on real architectures (§10) |

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export5.py test_the_exported_graph_is_not_empty_which_is_the_failure_that_looks_right present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export5.py test_agreed_the_replay_matches_upstream_element_wise_at_a_derived_tolerance present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export5.py test_the_meta_storage_identity_is_shared_by_a_view_and_not_by_a_stranger present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export5.py test_preserve_dispatch_key_guard_actually_restores_what_it_saved present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export5.py test_the_census_names_are_present_with_no_monkey_patch_at_all present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export5.py test_functionality_to_backend_keys_matches_upstream_key_for_key present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/storage.rs meta_has_no_bytes present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs pre_dispatch_mode present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _refuse_unrepresentable_memory_format present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_dispatch_key_set present -->
<!-- DOCWATCH: count golden_ops_covered ge 302 -->
<!-- DOCWATCH: count golden_cases_passed ge 11405 -->

---

## 1. The wall was re-measured before anything was planned

`docs/EXPORT4.md` §1 lost its first hours to a worktree whose vendored tree had
no `torch/__init__.py`, so every probe silently imported upstream torch 2.13.0
and reported that export already worked. **The tree was unbuilt again at the
start of this round** — `torchnative/src/main/torch/` held only `nn/` and a
README — so the first action was `vendor/vendor_torch.sh` and
`vendor/install_shim.sh`, and the first assertion after it was:

```
file        /Volumes/.../torchnative/src/main/torch/__init__.py
aten_impl   True
nops        302
```

Every probe in this round prints which side it is on, and `test_export5.py`
asserts `is_shim` before it asserts anything else. Then the real answer:

```
torch.export.export(M(), (x,))
  -> NotImplementedError: Cannot copy out of meta tensor; no data!
     @ torch/_subclasses/meta_utils.py:2071   self.set_storage_memo(s, r.untyped_storage())
```

`meta_utils.py:2071`, exactly where `docs/EXPORT4.md` §7 left it. The prediction
held.

---

## 2. Wall 8 — what a meta tensor's storage **is** in this shim

`docs/EXPORT.md` §3.3 stated the requirement precisely and it turned out to be
the whole specification: *"a storage handle that carries identity and size
without bytes"*.

### 2.1 What upstream actually asks of the handle

Measured on 2.13.0 rather than reasoned about, because two of the answers are
not what the names suggest:

```
torch.empty((3,4), device="meta").untyped_storage()
    nbytes()      48          <- the real footprint, not zero
    device        meta
    data_ptr()    0           <- EVERY meta storage, base or view, any size
    _cdata        <address>   <- distinct per storage; a VIEW shares its base's
    element_size  1
    storage[0]    NotImplementedError: Not available for 'meta' device type
    resize_(8)    works
```

**`data_ptr()` carries no identity on a meta storage.** That single measurement
decided the design. The obvious implementation — hand back the identity token
from `data_ptr()` — would have given out a number that looks like an address and
is a counter, and `torch/serialization.py` reads `data_ptr()` precisely to tell
storages apart. Identity is asked for through `_cdata`, and that is the only
place it is answered.

And what `meta_utils.py` does with the result is narrower still: it keys a
`WeakValueDictionary` on the *source* tensor's storage id and stores the handle,
so that two views of one base map to one fake storage. **It never reads through
it.**

### 2.2 The design

A meta tensor's storage in this shim is a `PyStorageBase` with:

| field | value | why |
|---|---|---|
| `device` | `"meta"` | the field existed and refused every non-cpu value at construction; this is a second accepted value, not a loosened check |
| `len` | `numel × itemsize` | what the elements *would* occupy. Real, and it is what `meta_storage()` consults |
| `buf` | empty | there are no bytes |
| `filled` | **`false`, permanently** | and this is load-bearing — see below |
| `origin` | `Repr::Meta`'s `storage_id` | identity, answered through `_cdata` |

`filled = false` is the part that makes the rest safe. `storage.rs`'s module
invariant is *"only something that actually delivered bytes may set this"*, and
`TensorBase.set_` refuses on an unfilled storage — the guard `docs/CKPT.md` §4
put there after a legacy checkpoint loaded to a complete state dict in which
every weight was `0.0`. A meta storage therefore **cannot be laundered into a
real tensor's bytes** by the one path that would produce silent zeros. Nothing
had to be added for that; the existing invariant covers the new arm because the
new arm is honest about having no bytes.

### 2.3 Identity, and the one thing that would have been invisible

`storage_id` is a `usize` from a process-wide counter, stored on
`Repr::Meta { shape, storage_id }`. A counter and not an address, because a meta
storage has no address — upstream says so with its `data_ptr() == 0`.

**`aten.view.default`'s meta kernel propagates it.** Upstream's `view` shares
storage, so a meta tensor and its own view must answer the same `_cdata`;
measured, they do. A fresh id there would have been **completely invisible**:
`meta_utils.py`'s memo would simply have held two entries, export would still
have run, and the aliasing the memo exists to preserve would have been silently
lost. `test_the_meta_storage_identity_is_shared_by_a_view_and_not_by_a_stranger`
is that case, and nullifying the propagation (N3) is caught by it and by nothing
else.

`Repr::Meta` has exactly **one** construction site in the whole crate
(`tensor.rs`), which is why adding a field to it was cheap.

### 2.4 What this handle meets, and what it refuses **by name**

The table the brief asked for. "Refuses" means raises with a message naming the
op and the reason; none of these is a silent wrong answer.

| upstream expects | here | |
|---|---|---|
| `nbytes()` / `size()` / `len()` = real footprint | `48` for a 3×4 float32 | **met** |
| `device` is `meta` | yes | **met** |
| `data_ptr()` is `0` | `0` | **met** |
| `_cdata` a stable identity; a view shares its base's | yes, via `Repr::Meta`'s token, propagated by `view` | **met** |
| `element_size()` is 1 | yes | **met** |
| `storage[i]` raises `Not available for 'meta' device type` | upstream's own message, verbatim | **met** |
| `resize_()` **works** | **refused** | `len` here is derived from the tensor's shape and dtype; a resize would leave storage and tensor disagreeing about a number the tensor owns |
| writing (`__setitem__`, `copy_`) | **refused** | no bytes exist to write |
| `_shim_bytes()` | **refused** | returning `b""` would say "this storage is empty"; the truth is "it has 48 bytes and none of them exist" — different claims |
| aliasing between two *separately constructed* meta tensors that upstream would consider views of one storage | **not modelled** | see below |

The last row is the honest limit. This shim can only express meta aliasing where
it can *propagate* the token, i.e. through `view`. `slice` and `t` have no meta
kernel and refuse; `_base` already refuses by name for a detected view
(`docs/EXPORT.md` §2.5). So the unmodelled case is bounded by existing refusals
plus the one hole §2.5 already records — a whole-storage contiguous view that
`_is_view` cannot detect.

`test_the_dense_storage_path_is_untouched_by_the_meta_handle` exists because
this had to be a **new arm** and not a loosening: a dense tensor's storage still
reports `cpu`, still says its bytes arrived, and still answers a non-zero
address. Without that test, making the meta branch work by weakening `snapshot`
would have passed everything else.

---

## 3. `is_contiguous(memory_format=...)`, and a defect the *premise* check found

`fake_tensor.py:1295` calls it on every tensor it hashes, so export reached it
once per cached dispatch and `TensorBase.is_contiguous() takes no keyword
arguments` stopped it.

Upstream's semantics, measured — including the signature, which is keyword-only
(`is_contiguous() takes 0 positional arguments but 1 was given`):

| asked | upstream | here |
|---|---|---|
| `contiguous_format` | the ordinary answer | the ordinary answer |
| `preserve_format` | the ordinary answer (**`False`** on a permuted tensor — not an unconditional `True`, which was the tempting guess) | the ordinary answer |
| `channels_last` / `channels_last_3d` | `True` only for a tensor in that layout | **`False`, as a fact** |
| anything else | — | **refused by name** rather than treated as `contiguous_format` |

The `False` is a fact in the same sense as `is_mkldnn` (`docs/EXPORT.md` §2.3):
there is no channels-last representation in this build at all.

### 3.1 The defect: `to(memory_format=channels_last)` was dropping the request

**A `False` that rests on an invariant is only as good as the invariant, so the
test checks the invariant and not the answer.**
`test_channels_last_is_false_as_a_fact_because_the_build_cannot_make_one` tries
to *construct* a channels-last tensor and requires that to fail.

It did not fail. `bootstrap.py`'s `Tensor.to` had
`kwargs.pop("memory_format", None)` — the request was **silently discarded** and
a contiguous tensor handed back. A caller doing
`x = x.to(memory_format=torch.channels_last)` held something it believed was
channels-last, and every later `is_contiguous(memory_format=channels_last)`
disagreed with what it had asked for. Nothing raised, because dropping an
argument raises nothing.

`_refuse_unrepresentable_memory_format` now accepts `contiguous_format` and
`preserve_format` — both of which ask for what the result already is — and
refuses the channels-last pair by name, on `to` and on `type`.

This is the finding worth generalising: **a test written against the answer
would have passed; the test written against the answer's premise failed.**

---

## 4. The backend-flag family, and `_dispatch_key_set`

Nine `_get_*`/`_set_*` pairs behind `torch.backends.<name>.flags()`, which
export enters before it traces anything.

**Every getter is derived from `_BUILD_FLAGS`, not transcribed from upstream,
and on two of them those differ.** `_get_mkldnn_enabled()` is `True` on upstream
2.13.0 *on this machine*, where `torch.backends.mkldnn.is_available()` is
`False` — upstream's flag is a user **preference** that outlives the backend
being absent. Here it is `False`, because `_has_mkldnn` is `False` and there is
nothing to prefer. `_get_cudnn_allow_tf32()` is the same story.
`_get_onednn_allow_tf32()` answers `None` — "not applicable" — and that *is*
upstream's own answer on a build without oneDNN.

**Every setter accepts the value its getter already reports and refuses every
other value by name.** The asymmetry is the implementation. A setter that
accepted `True` would let `torch.backends.mkldnn.flags(_enabled=True)` return
having changed nothing, which is `_len_torch_dispatch_stack`'s constant `0`
wearing a different hat. The accept-the-current-value half is not a loophole —
`set_flags` reads the getters on entry and writes them back on exit, so the exit
write must not raise.

**`_dispatch_key_set(tensor)` is a `str`, not a `DispatchKeySet`** — measured,
and worth stating because the name says otherwise and the obvious guess is
wrong. Its only reader compares two of them for equality
(`fake_tensor.py:2083`). Ten tensors varying dtype, rank, `requires_grad` and
view-ness produced exactly two answers, and the only input that mattered was the
**device**:

```
CPU   ->  DispatchKeySet(CPU, ADInplaceOrView, AutogradCPU, AutocastCPU)
meta  ->  DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
```

`requires_grad` does not change it, and **meta carries no Autocast key while CPU
does** — the asymmetry a hand-written table would most likely have smoothed
over. A version keyed on `requires_grad` would have looked more thorough and
failed as a silent **cache miss** rather than an error.

Also here: `aten.zeros_like` on a meta input, which was one word — the dense
kernel read `input.tensor()?.dims()` where `input.dims()` suffices, so a
question about a *shape* was answered with a refusal about *bytes*, the same
mismatch `stride()` had in `docs/EXPORT4.md` §6.5. Fixing it in the dense kernel
rather than writing a second meta arm means `zeros_like`'s dtype rule still
exists once. `empty_strided` moved from refusing `layout`'s mere *presence* to
refusing every layout except `torch.strided`, which is the one it returns.

<!-- DOCWATCH: op-implemented aten.zeros_like.default -->

---

## 5. The defect: a guard whose own description said what it did not do

This is the item worth reading if only one is.

`_PreserveDispatchKeyGuard`'s one-line entry in `_RAII_GUARDS` has always read:

> `"_PreserveDispatchKeyGuard": "saves and restores the whole TLS key state"`

It saved and restored **nothing**. It was a counter, like the other nine in the
family, and `docs/EXPORT.md` §2.4 had been explicit that the family were
counters. What made this one different is that **upstream delegates a restore to
it**:

```python
with torch._C._PreserveDispatchKeyGuard():
    torch._C._set_meta_in_tls_dispatch_include(True)
    try:
        yield
    finally:
        fake_mode.in_kernel_invocation = prev_in_kernel
        # torch._C._set_meta_in_tls_dispatch_include(prev_in_kernel)   <-- COMMENTED OUT
```

`fake_tensor.py::in_kernel_invocation_manager` sets the flag inside the guard
and leaves the matching reset commented out, naming the guard as the thing that
undoes it. With the guard inert the flag **latched `True`** after the first
kernel invocation, and the *next* entry died on that same function's own
assertion:

```
AssertionError: True, False
   @ torch/_subclasses/fake_tensor.py:663   in_kernel_invocation_manager
```

Same shape as `docs/EXPORT4.md` §5's `no_dispatch()`: no name was missing, no
stub was raising, every existing test passed, and the message names neither the
guard nor the shim.

The fix gives the five **key-state** guards a real save/restore of
`(included, excluded, meta_in_tls_dispatch_include)`. The snapshot is by
reference because `DispatchKeySet` is immutable here.

**Two tests, because the obvious one passes for the wrong reason.** The sequence
`[False, True, False]` is asserted rather than the final value — a guard that
reset the flag to a constant `False` on exit passes a final-value check. And a
**nested** case, `[True, False, True, True]`, because reset-to-constant passes
the un-nested test and fails this one.

Named and not fixed: the four argument-taking guards (`_Force…`, `_Exclude…`,
`_Include…`, `_SetExclude…`) now **restore** and still **set nothing**. Restoring
without setting is strictly safer than the counter it replaced and is what
`in_kernel_invocation_manager` depends on, but a caller relying on one of them
to actually exclude a key still gets no exclusion. `_KEY_STATE_GUARDS`' comment
says so.

### 5.1 Setters that refuse rather than lie to their own getters

Four more of the same shape, all reached by export:

* **`_set_conj` / `_set_neg`** — `meta_utils.py:2173` calls both on every meta
  tensor. `is_conj` answers `False` as a *fact* (`docs/EXPORT.md` §2.3), so
  `False` is a no-op and **`True` refuses**: accepting it would leave `is_conj`
  answering `False` immediately afterwards, a setter invisible to its own
  getter.
* **`grad_dtype`** — the getter is the tensor's own dtype, because that is where
  `tape.rs` accumulates. Upstream's setter is real and changes the backward
  pass's precision; here nothing would honour it, so a different dtype refuses.
* **`_has_symbolic_sizes_strides`** — `False`, and a fact: shapes here are
  `Vec<usize>` and `Repr::Meta`'s `shape`, concrete by construction, with no
  representation for anything else.

---

## 6. The wall that was not a wall: an `ExportedProgram` with no operators

With wall 8 and the six behind it closed, `torch.export.export()` **succeeded**.
It returned an `ExportedProgram`. It printed. It had a graph signature.

```
graph():
    %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
    %x : [num_users=0] = placeholder[target=x]
    return (c_lifted_tensor_0,)

constants: {'lifted_tensor_0': FakeTensor of size (3,)}
ops: []
```

For `x.relu()`. **The input is unused** and the answer is a lifted constant.
This is `docs/EXPORT.md` §4.2, written in 2026-09-06 as a prediction, arriving
verbatim:

> `torch.export` would then run, build an `fx.Graph`, install a
> `ProxyTorchDispatchMode`, call the module, see no `__torch_dispatch__` fire,
> and return a graph with a placeholder, an output, and **no operators**. It
> would be an `ExportedProgram`. It would print. It would serialise.

### 6.1 Why, and it is a structural finding

`torch/utils/_python_dispatch.py::_push_mode` branches on **`mode._dispatch_key`**,
not on `mode._mode_key`:

```python
def _push_mode(mode):
    k = mode._dispatch_key if hasattr(mode, "_dispatch_key") else None
    if k is None:
        _push_on_torch_dispatch_stack(mode)
        return
    ...
    _set_mode_pre_dispatch(mode)
```

`torch.export`'s tracer constructs its `ProxyTorchDispatchMode` with
`DispatchKey.PreDispatch`. So the proxy mode **never reaches
`_push_on_torch_dispatch_stack` or `_set_dispatch_mode`** — it goes to
`torch._ops._set_mode_pre_dispatch`, which keeps it in an ordinary Python object
in that module. `aten.rs`'s door read `_len_torch_dispatch_stack` and the infra
slots, and neither can see it.

Instrumented at the moment `relu` ran:

```
stack depth: 0   infra: [FAKE: FakeTensorMode, PROXY: None, FUNCTIONAL: None]
FAKE SAW aten.relu.default
```

Only `FakeTensorMode` ever saw an operator. The fake result had no proxy
attached, so the tracer treated it as a constant and lifted it.

`aten.rs::pre_dispatch_mode` now consults
`torch._ops._get_current_dispatch_mode_pre_dispatch()` **before** the user stack
and the infra slots — upstream's `PreDispatch` key sits above the `Python` key,
so a pre-dispatch mode runs first and re-dispatches into those below it.
Precedence *within* the pre-dispatch stack (FUNCTIONAL over PROXY) is not
reimplemented; that helper already encodes it and is called for exactly that
reason. The pop/restore pair is `torch._ops`' own
`_pop_mode_from_pre_dispatch` / `_set_mode_pre_dispatch`. A failure to reach
`torch._ops` is `None` and not an error, because the standalone `_C` that
`tools/golden/loader.py` imports has no `torch` package around it.

With that, the same module:

```
graph():
    %x : [num_users=1] = placeholder[target=x]
    %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%x,), kwargs = {})
    return (relu,)
constants: {}
```

### 6.2 What this cost the test suite, permanently

`test_the_exported_graph_is_not_empty_which_is_the_failure_that_looks_right`
exists because of this, and it is written to be hard to delete: it counts
`call_function` nodes **and** asserts the constants list is empty, since a fake
tensor in the constants list is how the empty graph announced itself. An
assertion that only checked "did it export" passed on the graph above.

---

## 7. The hand-off — `docs/EXPORT.md` §8, paid

Thirty-two `torch._C` names moved from
`torchnative/src/main/torchnative/export/upstream.py` into
`rust/torch_c/src/bootstrap.py`.

**Done as a separate step, after §2–§6 had landed and gated**, so a bisect
across the two is possible; §12 says which gates covered which.

### 7.1 Why it was a prerequisite and not a tidy-up

`docs/EXPORT4.md` §1.1 recorded that every export measurement in this repository
had been taken with `upstream.install()` applied, and that without it export
stopped at census name #0. That was abstract until this round ran the sweep:

```
export_sweep.py against the shim, before the hand-off:
    40/40 architectures stopped at  torch._C._unset_dispatch_mode
```

Every architecture, the same name, census #0. The sweep's subprocess never calls
`install()`, so it measured nothing at all. **The hand-off is what makes
`export_sweep.py` a measurement.**

### 7.2 How it was done, and the one thing that was not verbatim

The `_install_*` functions were copied **verbatim**, keeping their `(C, put)`
signature, with a four-line `_put` adapter in `bootstrap.py`. `docs/EXPORT.md`
§8 proposed re-signaturing them to take `module` alone; the verbatim copy was
preferred because this is a move of nine hundred lines of working, tested
behaviour and **a move that changes no line cannot change behaviour**. Renaming
is a separate reviewable change and should not ride inside a bisect boundary
that means "the same code, in its final home".

Dropped, as §8 specified: `install`, `rebind`, `InstallReport`,
`installed_names`, `_is_ours`, `_mark_ours`, `_MARK` — all of which existed to
describe and undo a runtime patch. One signature did change: `_install_tensor_
predicates` lost a `torch_module` parameter that its body never read.

`_DISCOVERED_RETURNS`' `"_len_torch_dispatch_stack": 0` row is deleted (§8 item
3), and the paragraph above it that said "nothing pushes onto it here" is
replaced by one saying why it was removed. `set_eval_frame` is untouched (§8
item 4).

### 7.3 Two install-order traps, both silent

**Position is load-bearing and the first attempt was wrong.** Placed beside
`_install_dispatch_keys` — which is where §8 proposed them —
`_install_dynamo_bool` was silently undone: the name was in
`torch._C._dynamo.guards.__dict__` afterwards and its value was the
`_Unimplemented` the stub pass had written over it. Nothing raised; export
simply stopped on that name again. The block now runs **last** in `install()`.

**And then it was still wrong, for a second reason.** `C._dynamo.guards` at
bootstrap time is not the module the name has to land on: nothing has imported
`torch._C._dynamo.guards` yet, so the attribute access falls through
`_attach_module_catchall`'s PEP 562 `__getattr__`, which — because `guards`
starts lowercase — synthesises an `_Unimplemented` and caches it. The install
succeeded onto a throwaway object. It now builds and registers the submodule in
`sys.modules` the same way `eval_frame` is, so the later real import finds it.

In `upstream.py` neither trap existed, because that module ran after
`import torch` had already created everything. **Both are invisible failures**:
an attribute was set, no exception was raised, and the name stayed a stub.

### 7.4 What is left of `upstream.py`

A forwarder that **implements nothing**. `install()` is a no-op returning a
report whose counts are all zero and whose repr says `moved-to-bootstrap`;
`rebind()` returns `0`; `__getattr__` forwards `_RAII_GUARDS`, `_is_stub` and
friends to the bootstrap. It is kept rather than deleted because three test
files reach for those names, and pointing them at the bootstrap's copies is what
makes them start testing the **final home**. Deleting the file would have
deleted those tests with it.

---

## 8. The three verdicts, never collapsed

`rust/torch_c/pytests/export_sweep.py` keeps `exported` / `replayed` / `agreed`
apart and headlines the third. So does `test_export5.py`, as three separate
tests plus a fourth for the empty-graph case §6 made necessary.

### 8.1 The tolerance is derived, not chosen

`docs/AGREE.md`'s method, re-run here rather than transcribed: each module is run
on **upstream** in float32 and float64, and the tolerance is the p90 of upstream's
own relative error, floored at 8 ulp of float32.

```
upstream's OWN float32-vs-float64 relative error, 4 modules:
    add_two       4.475e-08   (0.38 ulp)
    relu          3.968e-08   (0.33 ulp)
    scalar_chain  3.801e-08   (0.32 ulp)
    chain         9.284e-09   (0.08 ulp)
  p90                         4.475e-08
  floor = 8 ulp               9.537e-07
  TOLERANCE USED              9.537e-07   = 8.00 ulp
```

**The floor is doing the work here and that is worth saying rather than hiding.**
These four modules are numerically easy, so the p90 lands well below a single
ulp; `docs/AGREE.md` puts the floor there for exactly this case — "so that a
population which happened to be numerically easy could not drive the tolerance
below a few ulp". The tolerance is still not chosen: both the p90 and the floor
are computed, and which one wins is a measurement.

### 8.2 The result

Comparison is the shim's **exported-and-replayed** output against **upstream's
eager** output, element for element — not against the shim's own eager output,
which would only prove export and eager agree with each other.

| module | exported | replayed | ops | rel. vs upstream eager | verdict |
|---|---|---|---:|---|---|
| `relu` | yes | yes | 1 | `0.000e+00` | **agreed, bit-identical** |
| `add_two` | yes | yes | 2 | `0.000e+00` | **agreed, bit-identical** |
| `scalar_chain` | yes | yes | 3 | `0.000e+00` | **agreed, bit-identical** |
| `chain` | yes | yes | 5 | `0.000e+00` | **agreed, bit-identical** |

Bit-identity is asserted **separately** from within-tolerance, because
`docs/EXPORT4.md` §9 is right that a boolean cannot distinguish the two and the
difference is the whole question once decompositions arrive. There are no
decompositions here yet, and that is why the answer is exact.

---

## 9. The overload disagreement now reaches the exported graph

`docs/EXPORT.md` §5 found `capture.rs` recording `aten.mul.Scalar` where
upstream dispatches `aten.mul.Tensor`. `docs/EXPORT4.md` §5 found the same
difference reaching a second front end, a `TorchDispatchMode`. It now reaches
the **third and the one that matters most**:

```
shim     exported graph:  aten.mul.Scalar, aten.add.Scalar, aten.relu.default
upstream exported graph:  aten.mul.Tensor, aten.add.Tensor, aten.relu.default
```

Same operators, two of three overloads different, in an `ExportedProgram`. Core
ATen and ExecuTorch's Edge dialect are defined **per overload**, so a lowering
table keyed on one spelling silently misses the other.

**It is unclosed**, and `test_the_shim_and_upstream_agree_on_which_operators_the_graph_holds`
asserts the operators agree while *requiring the overload disagreement to still
be present* — so closing it fails the test and has to be a decision rather than a
drift. It is deliberately not spelled `assert ops == ops`, which would fail for a
known documented reason and tempt someone to delete it.

Note it does **not** affect §8's numbers: the graph replays bit-identically,
because the shim's own dispatcher resolves the same overloads on replay that it
recorded. The disagreement is with upstream's *spelling*, not with its
arithmetic — which is precisely why it needs a test rather than a tolerance.

---

## 10. The real sweep, and the three walls that remain

`export_sweep.py`, 40 `transformers` architectures, both sides, after the
hand-off:

```
architectures swept                              : 40
upstream exported+replayed+agreed                : 10
of those 10, this shim exported+replayed+agreed  :  0     (0.0%)
```

**Zero.** Where the shim's 40 stop:

| stage | n | |
|---|---:|---|
| `export` | 26 | the three walls below |
| `forward` | 10 | eager gaps — the model does not run at all, so export is not the blocker |
| `construction` | 4 | `transformers` config issues, both sides |

The three export walls, by frequency:

| n | wall | |
|---:|---|---|
| 14 | `AttributeError: '_SchemaType' object has no attribute 'annotation_str'` | schema surface |
| 11 | `TensorBase.set_: the storage has never been filled` | §2's `filled` invariant meeting `meta_utils.py`'s "crazy town" branch — see below |
| 1 | `AssertionError: Could not find common device for aten.empty_strided.default` | `docs/EXPORT4.md` §5's shape, in a path the guard fix does not cover |

**The 11 deserve a note, because they are wall 8's own consequence.**
`meta_utils.py`'s fallback branch calls `meta_storage()` and then `set_` to
attach it to a meta tensor — which is pure metadata, no bytes involved. §2's
`filled = false` refuses it, correctly for the dense case the invariant was
written for (`docs/CKPT.md` §4's silent zeros) and too broadly for this one. The
narrowing — allow `set_` when *both* the tensor and the storage are meta, refuse
otherwise — is a storage-model decision of the same class as §2 and is **not
attempted here**, for `docs/EXPORT4.md` §7's reason: landing a design change
without room to verify it is the failure the brief warns about.

Also left, and named:

* **`aten.t` / `aten.slice` on a meta tensor.** `nn.Linear` needs the first, so
  no module with a linear layer exports. It is **not** a missing kernel in the
  ordinary sense: `t` produces a *non-contiguous* meta tensor, and
  `docs/EXPORT4.md` §6.5's meta `stride()` answer rests explicitly on the
  invariant that no meta tensor here can be non-contiguous — with a test that
  fails the day one becomes constructible. Adding the kernel therefore requires
  adding a stride to `Repr::Meta` **in the same change**, which is the same
  layout-model question as wall 8. `Repr::Meta`'s own comment has said so since
  it was written.
* **`empty_strided` with a genuinely non-contiguous stride** — refused by name,
  unchanged, same question.
* **The `.Scalar`/`.Tensor` overload disagreement** (§9).
* **The setting half of the four argument-taking key-state guards** (§5).
* **`_dispatch_has_computed_kernel_for_dispatch_key` for keys other than
  Meta/CPU** — refused by name, deliberately, unchanged.
* **The shim's `DispatchKey` enum is narrower than upstream's** (§11).

---

## 11. Nullification: 17 attempted, **1 uncaught** — and that one is the finding

Every landing was broken deliberately, rebuilt (`bootstrap.py` is
`include_str!`'d, so editing without rebuilding retests the old binary),
re-run against five suites, and reverted.

| | nullification | caught by |
|---|---|---|
| N1 | meta `untyped_storage()` refuses again | 12 tests |
| N2 | meta `data_ptr()` answers the identity token instead of `0` | 1 |
| N3 | meta `view` mints a fresh storage id instead of inheriting | **1** |
| N4 | `_PreserveDispatchKeyGuard` back to a bare counter | 9 |
| N5 | the door stops consulting the pre-dispatch stack (**the empty-graph failure**) | 7 |
| N6 | `_set_conj`/`_set_neg` accept `True` silently | 1 |
| N7 | `is_contiguous(channels_last)` falls through to the ordinary answer | 1 |
| N8 | `_dispatch_key_set` answers the same string for cpu and meta | 1 |
| N9 | the hand-off drops `_install_tls` | 36 |
| N10 | `to(memory_format=channels_last)` silently drops the request again | 1 |
| N11 | the meta storage claims `filled = true` | 1 |
| N12 | a meta storage hands back empty bytes instead of refusing | 1 |
| N13 | `_has_symbolic_sizes_strides` answers `True` | 8 |
| N14 | `grad_dtype` setter accepts any dtype silently | 1 |
| N15 | `zeros_like` asks `tensor()` for a shape again | 7 |
| N16 | **`_functionality_to_backend_keys` always answers `[]`** | **NOTHING** |
| N17 | `empty_strided` back to refusing `layout`'s mere presence | 7 |

### 11.1 N16, the one that went uncaught

Replacing the whole body with `return []` broke **nothing**. Every suite stayed
green and `torch.export.export()` still worked.

The reason is instructive. Its only caller is `_push_mode`, which uses the
result to *uncache* per-dispatch-key handlers:

```python
ks = torch._C._functionality_to_backend_keys(k)
for op in get_cached_ops():
    for key in ks:
        op._uncache_dispatch(key)
```

With `[]`, nothing is uncached. Today `get_cached_ops()` is empty in this shim,
so the loop has no body either way — which is exactly why it was invisible. It
is a **cache invalidation that quietly invalidates nothing**, the same "entered
and changed nothing" shape as `_len_torch_dispatch_stack`'s constant `0`, and it
would surface as a stale handler rather than as an error.

### 11.2 Writing the test for it found two more bugs — in my own implementation

The gap is closed by
`test_functionality_to_backend_keys_matches_upstream_key_for_key`, which asks
**both sides** the same question and diffs, rather than checking the shim
against a list. That immediately failed twice:

1. **The fallback was wrong.** The implementation returned `[]` for a key that is
   not a functionality key, with a docstring confidently asserting that was
   upstream's answer. It is not: upstream answers `[key]` — `CPU -> [CPU]`,
   `Undefined -> [Undefined]`, without exception. **This is on the export path**:
   `_push_mode` calls it with `DispatchKey.PreDispatch`, a plain key, so the
   empty list made §6's very machinery skip its cache invalidation.
2. **The shim's `DispatchKey` enum is incomplete.** It has the bare `MAIA`
   member but none of its per-functionality spellings (`AutogradMAIA`,
   `SparseMAIA`, `QuantizedMAIA`, `NestedTensorMAIA`), while `MTIA` has all of
   them. It also lacks `Quantized` entirely. That is pre-existing and not this
   round's, so it is **pinned** in the test rather than smoothed over — the
   difference must not grow.

A transcribed table would have listed `AutogradMAIA` and answered a key the enum
does not have. Deriving from the enum and diffing against upstream is what turned
that into a visible, bounded gap.

N16 re-run after the test existed: **caught by 1 test.**

### 11.3 What the nullification pass could not test

Stated because a limits section that is missing is itself a finding:

* The pass ran five suites, not the full gate, for time. A nullification caught
  only by `test_train.py` or the golden harness would read as uncaught here. The
  full gate was run before and after, green both times.
* N5 kills export outright, so it proves the pre-dispatch consult is
  load-bearing without proving *which* of the seven tests depends on which half
  of it.
* Nothing nullified the §10 refusals, because they are refusals — there is
  nothing to break that a test would notice as different from the wall.

---

## 12. Which gates covered which step

The brief asked for the two steps to stay bisectable, so this is the record.

| | after §2–§6 (wall 8 + the walls behind it) | after §7 (the hand-off) | final |
|---|---|---|---|
| suite | 1004 ok, **2 FAIL** | 1004 ok, **2 FAIL** | **1027 ok, 0 FAIL** |
| DOCWATCH | 903/903 | 903/903 | **903/903** |
| golden | 11405/11405 ops=302 | 11405/11405 ops=302 | **11405/11405 ops=302** |
| cargo test | 30/30 | 30/30 | **30/30** |

The two failures at each stage were **different failures**, which is the point
of running the gate twice:

* after §2–§6: `test_export_still_stops_and_it_stops_at_the_storage_handle`
  (EXPORT4's two-way guard, firing in the "it started working" direction it was
  built for) and
  `test_save_refuses_the_legacy_container_and_every_write_into_a_snapshot`
  (asserted a meta tensor refuses `untyped_storage()`);
* after §7: the same guard, plus
  `test_the_census_names_were_placeholders_not_implementations` — which could
  only fail once the names stopped being placeholders, i.e. only because the
  hand-off worked.

Neither step's failures overlap, so a bisect lands on the right one.

---

## 13. Existing assertions that had to move, and why none is a weakening

Three. Recorded individually because "a test needed updating" is where a
weakening hides.

| test | was | now | why it is not a weakening |
|---|---|---|---|
| `test_export_still_stops_and_it_stops_at_the_storage_handle` | asserted export stops at `untyped_storage()`, failing in **both** directions | `test_export_no_longer_stops_at_the_storage_handle_and_returns_a_real_graph` | The old test's own failure message specified the rewrite: *"confirm the graph REPLAYS to the same numbers as eager, element-wise against upstream, then rewrite EXPORT4.md §3."* That was done (§8). The replacement **still fails in two directions** — if export stops again, and if the graph comes back empty — so it is strictly more than the old one checked |
| `test_save_refuses_..._every_write_into_a_snapshot` | one assertion: a meta tensor refuses `untyped_storage()` | four properties of the handle plus **five** door-refusals on it | Replaces one refusal with nine assertions. A handle that answered plausibly to everything would have satisfied the old test's replacement and fails this |
| `test_the_census_names_were_placeholders_not_implementations` | census ⊆ `install()`'s `replaced` list | `test_the_census_names_are_implemented_by_the_bootstrap_with_no_install_call` | **Inverted, and stronger.** The old claim could be satisfied by a name that was a stub and then got patched; the new one cannot be satisfied by a stub at all. It also asserts `replaced == []` and `rebound == 0`, so a regression to the monkey-patch fails |

---

## 14. What this round claims, split the way `docs/COMPILE.md` §5.3 asks

| | |
|---|---|
| **feature added** | the meta storage handle (§2); `is_contiguous(memory_format=)` (§3); nine backend flag pairs, `_dispatch_key_set`, `_functionality_to_backend_keys` (§4); `aten.zeros_like` on meta (§4); `_set_conj`/`_set_neg`, `grad_dtype`, `_has_symbolic_sizes_strides`, `_dispatch_tls_set_dispatch_key_included` (§5.1); **the pre-dispatch stage of the dispatcher door** (§6) |
| **defect fixed** | **3** — `_PreserveDispatchKeyGuard` restoring nothing (§5); `to(memory_format=)` silently dropping the request (§3.1); `_functionality_to_backend_keys` answering `[]` for a non-functionality key (§11.2) |
| **moved** | 32 census names, staging module → `bootstrap.py` (§7). No behaviour change intended; the functions are byte-identical apart from one dropped dead parameter |
| **tests added** | **21**, in `rust/torch_c/pytests/test_export5.py` |
| **tests rewritten** | 3 (§13) |
| **measurement** | the three verdicts with a derived tolerance (§8); the 40-architecture sweep, both sides (§10); the wall sequence (§10); 17 nullifications (§11) |
| **documentation corrected** | `docs/EXPORT.md` §3.3 (wall 8 closed); `docs/EXPORT4.md` §3 and §7 (the four claims, and the wall) |
| **deleted** | nothing. `upstream.py` was emptied, not removed (§7.4) |

Counted how: "feature added" is a name or behaviour that was a refusal before
and answers now; it is **not** the number of tests, and §11's uncaught
nullification is why that distinction is not academic.

And the negative, stated as a negative: **`torch.export.export()` works on four
modules I wrote and on none of the ten real architectures upstream can export.**
The distance from here is three named walls (§10), of which one is wall 8's own
`filled` invariant meeting a legitimate metadata-only `set_`, and one is the
`aten.t` meta kernel that cannot land without the stride field `Repr::Meta` has
been documented as needing since it was created. Neither is a missing name.
