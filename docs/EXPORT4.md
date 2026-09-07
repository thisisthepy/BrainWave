# `torch.export` — six walls closed, and the one that is a design question

`docs/EXPORT.md` §6 gave an ordering and a prediction: steps 1 and 2 first (the
dispatcher consults the mode stack; `_NodeBase` builds), and then the wall would
be `aten.empty_strided`. Both steps had landed when this round started. **The
prediction held** — export stopped at `empty_strided`, exactly there — and this
round closed it and five more behind it.

It did **not** reach a working `torch.export.export()`. §3 is what "export works"
would mean and how far short of it this is, stated before any of the good news,
because `docs/EXPORT.md` §4.2 is about precisely the temptation to report the
distance travelled instead of the distance remaining.

**One of the six was a defect, not a gap, and `docs/EXPORT.md` had predicted it
in prose.** §2.4 said `no_dispatch()` was a bare counter that was correct "only
because this shim never consults the mode stack in the first place", and that
when the door learned to consult it the guard would become load-bearing. The
door learned in step 1. The guard did not follow. §5 is what that cost.

Measured 2026-09-07, `darwin/arm64`, CPython 3.13, `work/export4`, on
develop `976a01b`.

---

## 0. At a glance

| | |
|---|---|
| Was `empty_strided` still the wall? | **Yes** — §6's prediction held, unlike REPEAT.md's moving first wall (§1) |
| Walls closed this round | **6** (§4–§6) |
| Of those, defects rather than gaps | **1** — `no_dispatch()` suppressed nothing (§5) |
| Does `torch.export.export()` return an `ExportedProgram`? | **No** (§3) |
| Where does it stop now? | `meta_utils.py:2071`, `r.untyped_storage()` — `docs/EXPORT.md` §3.3 (§7) |
| Is the remaining wall a missing name? | **No.** A storage handle with identity and size but no bytes — a design question (§7) |
| Does a `TorchDispatchMode` see operators? | **Yes**, and now stops seeing them under `no_dispatch()` (§5) |
| Tests added | 18, in `rust/torch_c/pytests/test_export4.py` |
| Nullifications attempted / caught | **9 / 9** (§8) |
| New measurement tool | `rust/torch_c/pytests/export_sweep.py` (§9) |

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export4.py test_export_still_stops_and_it_stops_at_the_storage_handle present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export4.py test_no_dispatch_actually_suppresses_now_that_the_door_reads_the_stack present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_export4.py test_empty_strided_refuses_a_non_contiguous_stride_by_name present -->

---

## 1. The wall was re-measured before anything was planned, and it had not moved

This repository has had a "first wall" move three times (`docs/REPEAT.md`), so
`docs/EXPORT.md` §6's list was treated as a prediction and not as a plan. The
first thing this round did was run export.

**A measurement error came first, and it is worth recording because it would
have invalidated the entire round.** The initial probe reported that
`torch.export.export()` already worked — an `ExportedProgram` with the right
three operators, `.Tensor` overloads matching upstream, replaying bit-exactly
against upstream across twelve real modules including a transformer block. All
of it was true and none of it was about this shim: the worktree's vendored tree
contained `torch/_C.abi3.so` and **no `torch/__init__.py`**, so `PYTHONPATH` had
nothing to shadow `site-packages` with and every probe had imported upstream
torch 2.13.0.

The tell was `side=upstream` printed by a sweep that had been given the shim's
`PYTHONPATH`. This is `docs/EXPORT.md`'s own genre of failure — a verification
that could not fail — and the fix is the cheap one: **every probe in this round
prints which side it is on, and `test_export4.py` asserts `is_shim` before it
asserts anything else.**

    assert r["is_shim"], "subprocess did not get the shim-backed torch"

After `vendor/vendor_torch.sh` populated the tree, the real answer:

```
torch.export.export(M(), (x,))
  -> NotImplementedError: not implemented in torch._C shim: torch.empty_strided(...)
     @ torch/_subclasses/meta_utils.py:2009
```

`empty_strided`, at `meta_utils.py:2009`, which is the line `docs/EXPORT.md`
§3.1 named. The prediction held.

### 1.1 One caveat that governs every number below

That wall is reached **only with `torchnative.export.upstream.install()`
applied.** Without it export stops much earlier, at census name #0
(`torch._C._unset_dispatch_mode`), because **`docs/EXPORT.md` §8's hand-off never
happened** — the 29 names are still in the staging module and not in
`bootstrap.py`.

So "export reaches `untyped_storage()`" means "reaches it under the staging
monkey-patch". That is the configuration `docs/EXPORT.md` measured in too, and
§8 already owns the debt; this round did not pay it (§10).

---

## 2. What was landed, split the way `docs/COMPILE.md` §5.3 asks

"A round landed" is not a number, and `docs/EXPORT.md` §5.3's warning is that
implementation, defect-fixing and documentation get counted in the same column.
Separated:

| | |
|---|---|
| **feature added** | `aten.empty_strided.default` (§4); `torch.is_inference_mode_enabled`; `torch._C._should_allow_numbers_as_tensors`; `torch._C._dispatch_has_computed_kernel_for_dispatch_key`; `torch._C._set_throw_on_mutable_data_ptr` and the per-tensor bit behind it; `torch._C._set_warn_deprecated_on_mutable_data_ptr`, its softer sibling; `stride()`/`storage_offset()` on a meta tensor |
| **defect fixed** | **`no_dispatch()` suppressed nothing** once the door began consulting the mode stack (§5) — predicted by `docs/EXPORT.md` §2.4 and not caught by any check for a missing name, because no name was missing |
| **tests added** | 18, `rust/torch_c/pytests/test_export4.py`; 4 existing count/named-list assertions updated in `test_shim.py` and `test_dispatch.py` (§12) |
| **measurement** | the re-derived wall sequence (§1, §7); the two predicate tables derived from upstream rather than transcribed (§6); `export_sweep.py` (§9) |
| **documentation corrected** | `docs/EXPORT.md` §2.4's "entering a counter is a correct implementation" is no longer true, and §3.1/§3.2 are now closed; §13 records all three |
| **deleted** | none |

Six walls is not six features: two of the six are one-line predicates, one is a
defect fix, and none of them is `torch.export` working.

---

## 3. What "export works" would mean, and how far short this is

Three claims, and they are not the same claim. `docs/EXPORT.md` §4.2 is the
argument that the first can be true while the others are false and **it would not
look wrong**:

| claim | status |
|---|---|
| `torch.export.export()` returns an `ExportedProgram` | **No** (§7) |
| the graph it holds contains the module's operators | not reached |
| the graph REPLAYS to the same numbers as eager | not reached |
| the replay agrees with **upstream** element-wise | not reached |

**Nothing in this round is evidence for any of the four.** The six walls are on
the road to the first, and the first is the weakest of them — an
`ExportedProgram` containing a placeholder, an output and no operators would
satisfy it, print, and serialise.

`export_sweep.py` (§9) exists so that when the first does become true, the other
three are measured in the same breath rather than inferred from it. It keeps
`exported`, `replayed` and `agreed` as three separate verdicts and reports
`agreed` as the headline, because reachability is not agreement and this project
has been burned by treating it as such.

The honest summary is: **export got materially further, and it does not work.**

---

## 4. `aten.empty_strided` — and a refusal that is forced, not chosen

`rust/torch_c/src/aten.rs`, `rust/torch_c/src/overloads.json`.

<!-- DOCWATCH: op-implemented aten.empty_strided.default -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json empty_strided present -->

It is the constructor behind every fake tensor (`meta_utils.py:2009`), so
`torch.export` reaches it before anything interesting. **It serves the
contiguous case and refuses every other stride by name.** That split is forced
by the storage model, and both halves of the force are in the types:

* `Repr::Meta { shape }` stores a shape and **no stride**. `docs/META.md` §6
  records that as a deliberate narrowing — upstream's meta *does* carry stride.
* A dense tensor cannot be given an arbitrary caller-supplied stride either.
  `Tensor::from_storage` always allocates contiguous strides and the constructor
  that would pair a custom `candle_core::Layout` with a storage is not public.
  `as_strided` works around this by **materialising** — gathering the requested
  elements out of an existing base into fresh contiguous storage — and that
  trick is unavailable here, because `empty_strided` **has no base to gather
  from**.

So a non-contiguous request has no representation on either path, and the two
available answers are refusing it or returning a contiguous tensor while
claiming it is strided. The second is the failure this repository keeps meeting:
the caller asked for a layout, got a different one silently, and every later
`.stride()` disagrees with what it asked for. `FakeTensor` exists precisely to
reason about layout, so a fake tensor with a quietly wrong stride would make
export's own metadata wrong rather than merely incomplete.

The refusal names the op, the stride asked for, and the only stride that was
available:

```
NotImplementedError: aten.empty_strided.default: a non-contiguous stride is not
representable in this shim -- asked for size=[2, 3] stride=[1, 2], and the only
stride this shim can build for that size is the contiguous [3, 1]. ...
```

**A size-1 axis does not decide the refusal.** No two distinct index tuples
differ in a length-1 axis, so its stride is unobservable and upstream is
indifferent to it. Refusing there would reject shapes that are in fact perfectly
contiguous — which is how a narrow-but-honest refusal turns into a wrong one.
`test_empty_strided_does_not_let_a_size_one_axis_decide_the_refusal` is that
case.

**It returns zeros**, not uninitialised memory, for `zeros_or_ones`' reason: the
contract permits any bytes and deterministic ones are the safe direction. That
is a divergence from upstream and it is asserted, so it stays a known one.

The contiguous case is not narrow: a fake tensor built from a contiguous real
tensor asks for exactly the contiguous stride, which is why closing only this
half moves export at all.

---

## 5. The defect: `no_dispatch()` had stopped suppressing anything

This is the item worth reading if only one is read.

`docs/EXPORT.md` §2.4 wrote the specification as a prediction:

> Entering a counter is a correct implementation **only because this shim never
> consults the mode stack in the first place** … When `_aten_dispatch` learns to
> consult the stack, this guard stops being a counter and starts being
> load-bearing, and the counter is here so that change is a body to fill rather
> than a hole to find.

`_aten_dispatch` learned to consult the stack in §6 step 1 (commit `c3ff53f`).
The guard did not follow. From that commit until this round, **`no_dispatch()`
suppressed nothing.**

Nothing reported it, and nothing could have: no name was missing, no stub was
raising, `_len_torch_dispatch_stack` counted correctly, and every existing test
passed. The symptom surfaced four walls later and several frames away:

```
AssertionError: Could not find common device for aten.empty_strided.default
   @ torch/_subclasses/fake_tensor.py:1207 _find_common_device
```

`meta_utils.py:2009` builds the meta tensor behind every fake tensor *under*
`no_dispatch()`. With the suppression inert, `aten.empty_strided.default` was
dispatched **into the `FakeTensorMode` that was trying to build it**, and died
because a factory has no tensor arguments to take a device from. The message
names `empty_strided`, which had just been implemented, and says nothing about
the guard that should have prevented the re-entry — a good half-hour of looking
in the wrong place.

**The fix puts the state where both halves can reach it.**
`bootstrap.py::_install_dispatch_suppression` owns a thread-local depth;
`_DisableTorchDispatch.__enter__`/`__exit__` write it; `aten.rs`'s
`any_dispatch_mode_active` reads it and returns `false` while it is held. Keeping
the counter in the guard's own module — which is what had been done — leaves the
door unable to see it, and that is the arrangement that failed.

Three properties, each with its own test, because the obvious single assertion
passes for the wrong reason:

* a mode **does** see operators outside the guard. Asserting only the
  suppression would pass on a shim where modes see nothing at all, which is the
  state this whole road started from;
* it sees **none** inside;
* the depth is a **count**, not a flag. `fake_tensor.py` enters `no_dispatch()`
  inside code already inside it, so a boolean would be cleared by the inner
  exit, and the guard must actually release rather than latch off — a latched
  mode stack would make every later test in the file pass for the wrong reason.

Measured, before and after:

| | mode sees, outside guard | mode sees, inside `no_dispatch()` |
|---|---|---|
| before | `mul.Scalar`, `add.Scalar`, `relu.default` | **the same three** |
| after | `mul.Scalar`, `add.Scalar`, `relu.default` | *(nothing)* |

The `.Scalar`/`.Tensor` overload disagreement `docs/EXPORT.md` §5 pinned for
`capture.rs` **is visible here too** — the mode sees `mul.Scalar` where upstream
dispatches `mul.Tensor`. It is the same resolution difference reaching a second
front end, and it is unclosed; §10.

---

## 6. The predicates and bits behind the walls, derived rather than assumed

Each was a raising stub reached once per fake dispatch, so each stopped export
entirely.

### 6.1 `torch.is_inference_mode_enabled` — not a constant

`fake_tensor.py:1801` calls it on every cached dispatch. Upstream spells it
**only at torch level** — there is no `torch._C._is_inference_mode_enabled` on
2.13.0 — and `torch/__init__.py` harvests it off `_VariableFunctions`, so it
lives beside `is_grad_enabled` in `bootstrap.py::_install_grad_mode` and is set
on `_VariableFunctions` as well as on the module.

**It is state, not a constant, and that is the whole point.** `docs/EXPORT.md`
§2.2 is about `_len_torch_dispatch_stack` answering a constant `0` while
something really was pushing — a block that entered, reported itself absent, and
changed nothing. A constant `False` here is that failure with the operands
swapped, and no check for a missing name would catch it, because the name is not
missing. `_InferenceMode.__enter__` writes through to the same flag, and the test
asserts the **sequence** `[False, True, False]`, not the default.

Nullification N3 — replacing the body with `return False` — is caught by exactly
that test and nothing else (§8).

### 6.2 `_should_allow_numbers_as_tensors` — a table of 31 names

Upstream's is a `static std::unordered_set<std::string>` in
`torch/csrc/utils/python_arg_parser.cpp`. **The shim's table was derived by
enumerating upstream, not transcribed from the C++**: every name in
`dir(torch.ops.aten)`, plus its `_` and `_out` spellings, was passed to the real
predicate on torch 2.13.0, and the 31 that answer `True` are the table.

`test_should_allow_numbers_as_tensors_matches_upstream_name_for_name` re-runs
that derivation against upstream and diffs both directions, so a torch upgrade
that adds a name fails here. A transcription would have gone stale silently.

The shape is worth stating because "binary op" is the wrong rule:
`add`/`sub`/`mul`/`div` and their long spellings (`subtract`, `multiply`,
`divide`, `true_divide`, `floor_divide`), each in three forms, plus `to`,
`copy`, `copy_` and `_to_copy`. **`rsub` and `pow` are not in it.**

### 6.3 `_dispatch_has_computed_kernel_for_dispatch_key` — and a refusal kept

`fake_impls.py:1568`'s `has_meta` asks this once per fake dispatch, and
`fake_tensor.py:3077` calls it an *optimization*. The risk is asymmetric and
worth writing down:

* a wrong `True` costs a raised-and-caught `NotImplementedError`;
* a wrong `False` **skips a kernel that exists** and sends the op to
  `maybe_run_unsafe_fallback`, which raises outright when sizes are symbolic —
  i.e. under `torch.export`, always.

**The rule was measured.** All **1783** aten overloads on 2.13.0 answer `True`
for `"Meta"`, with no exceptions, and `prims` does too. So the rule is the
namespace, and the test re-derives it rather than trusting this paragraph — if a
future torch introduces an aten op with no computed Meta kernel, it fails and the
rule needs revisiting.

**It answers for `Meta` and `CPU` and refuses every other key by name.** Those
two are the keys this shim's single dispatcher door serves. For `CUDA`,
`Autograd`, `SparseCPU` and the rest it has no kernels and no registry, and both
available answers are claims it cannot support: `False` would assert that
upstream's dispatcher has no computed CUDA kernel either, and `True` would
promise a kernel that does not exist. **This refusal was not weakened to get
further** — the wall was elsewhere — and
`test_has_computed_kernel_refuses_a_dispatch_key_it_cannot_speak_for` asserts it
by name.

### 6.4 `_set_throw_on_mutable_data_ptr` — the per-tensor bit

`docs/EXPORT.md` §3.2 called this "Rust" and was right about the reason. It is a
field on `PyTensorBase`, an `AtomicBool`, and `data_ptr()` checks it before
reading storage, with upstream's own message.

A Python side-table keyed by identity would be a **different guarantee**: a
`FakeTensor` is not reliably weak-referenceable, and the bit has to survive
`Tensor._make_subclass`, which builds a new Python object around the same
`PyTensorBase`. As a field it survives both by construction.

It takes the tensor by `PyRef` rather than by value, because upstream's mutates
the object the caller passed — `fake_tensor.py:943` calls it on `self` inside
`FakeTensor.__new__` and keeps that object. **A version that set the bit on a
clone would leave every real `FakeTensor` answering an address and nothing would
fail**, which is why the test asserts the bit through a reader
(`_shim_throws_on_mutable_data_ptr`) and not only through the raise. It is
one-way, like upstream's: an un-setter would let a fake tensor be laundered into
one whose `data_ptr()` answers.

### 6.5 A meta tensor's `stride()` — derived, not invented

`meta_utils.py:2066` calls `r.stride()` on the meta tensor it just built. The
refusal it got was `Cannot copy out of meta tensor; no data!` — a message about
**bytes**, and a stride is not bytes. It was the wrong refusal for the question.

`Repr::Meta` stores no stride, so the obvious reading is that this cannot be
answered. It can, because of a fact the type already asserts: **`is_contiguous()`
returns `true` for every `Repr::Meta`**, since no kernel in this tree can produce
a non-contiguous one and `empty_strided` now refuses by name rather than build
one (§4). A contiguous tensor's stride is a function of its shape alone, so the
answer is the same value the dense path would compute, arrived at without a
storage to read it off. `storage_offset()` is `0` for the same reason — there is
no storage to be offset into.

The invariant is asserted rather than assumed:
`test_a_meta_tensor_is_contiguous_so_its_stride_is_derivable` fails the day a
non-contiguous meta tensor becomes constructible, which is the day this answer
would start lying. The dense path is untouched, and a transposed view still
reports its real `(1, 4)` rather than a contiguous stride computed from its
shape.

---

## 7. Where export stops now

```
NotImplementedError: Cannot copy out of meta tensor; no data!
   @ torch/_subclasses/meta_utils.py:2071
        self.set_storage_memo(s, r.untyped_storage())
```

This is **`docs/EXPORT.md` §3.3, unchanged and unclosed**, and it is the reason
this round stopped here rather than pressing on. It is not a missing name and
not a kernel:

`set_storage_memo` asks a meta tensor for its storage so that two views of the
same base map to the same fake storage. Upstream's meta tensor has a zero-sized
storage; this shim's `tensor.rs::storage_snapshot` refuses on `Repr::Meta` **by
design**, and that refusal is right — it exists so no kernel reads bytes that are
not there. What is missing is a **storage handle that carries identity and size
without bytes**, which is a change to the storage model rather than a line.

Attempting it inside the remaining budget of this round would have meant landing
a design change without room to verify it, which is the failure the brief warns
about. §10 leaves it open with its reason.

`test_export_still_stops_and_it_stops_at_the_storage_handle` pins it, and fails
in **both** directions — if export regresses to an earlier wall, and if it starts
succeeding. The second is deliberate: an `ExportedProgram` appearing there must
be met with an element-wise replay comparison (§3) before anyone calls it
working, and the test says so in its failure message rather than inviting the
assertion to be deleted.

### 7.1 The wall sequence, in the order it was walked

Each closed, then re-measured. This is the deliverable the round was asked for.

| # | stopped at | kind | closed |
|---:|---|---|---|
| 1 | `torch.empty_strided` — no table entry, no kernel | missing op | **yes** (§4) |
| 2 | `torch.is_inference_mode_enabled` | missing predicate | **yes** (§6.1) |
| 3 | `torch._C._should_allow_numbers_as_tensors` | missing table | **yes** (§6.2) |
| 4 | `torch._C._dispatch_has_computed_kernel_for_dispatch_key` | missing predicate | **yes** (§6.3) |
| 5 | `Could not find common device for aten.empty_strided.default` | **defect** — `no_dispatch()` inert | **yes** (§5) |
| 6 | `torch._C._set_throw_on_mutable_data_ptr` | missing per-tensor bit | **yes** (§6.4) |
| 7 | `r.stride()` on a meta tensor | wrong refusal | **yes** (§6.5) |
| 8 | `r.untyped_storage()` on a meta tensor | **storage model** | **no** (§7) |

Seven refusals moved; six distinct things were built (walls 6 and 7 are two of
them, wall 5 was a fix). Wall 8 is where it stands.

---

## 8. Nullification: 9 attempted, 9 caught

Every landing was broken deliberately and the suite re-run, because a test that
cannot fail is not a test (`docs/EXPORT.md`'s genre, and this round's own §1).
Each nullification was applied to the source, **rebuilt** — `bootstrap.py` is
`include_str!`'d into the extension at Rust build time, so editing it without
rebuilding retests the old binary — reinstalled, and reverted.

| | nullification | caught by |
|---|---|---|
| N1 | `empty_strided` match arm removed | 6 tests |
| N2 | non-contiguous refusal weakened to accept | whole probe dies — red |
| N3 | `is_inference_mode_enabled` becomes constant `False` | `test_inference_mode_round_trips_...` **only** |
| N4 | `_should_allow_numbers_as_tensors` answers `True` for everything | `test_should_allow_..._matches_upstream_name_for_name` |
| N5 | `has_computed_kernel` answers instead of refusing `CUDA` | `test_has_computed_kernel_refuses_a_dispatch_key_it_cannot_speak_for` |
| N6 | `data_ptr()` ignores the throw bit | `test_set_throw_on_mutable_data_ptr_...` |
| N7 | meta `stride()` answers an empty tuple | 4 tests |
| N8 | the door stops honouring `no_dispatch()` | `test_no_dispatch_actually_suppresses_...`, `..._nests_and_is_released_...` |
| N9 | `_set_warn_deprecated_on_mutable_data_ptr` collapsed into the thrower | `test_the_warn_deprecated_sibling_warns_and_still_answers` |

**No nullification went uncaught.** N3 is the one worth noting: it is caught by
exactly one test, and that test is the one asserting the *sequence* rather than
the default. Had the test asserted only `is_inference_mode_enabled() is False`,
N3 would have passed — which is §6.1's point made mechanically.

N2's bluntness is a limitation and is recorded as one: it kills the probe
subprocess rather than failing a named assertion, so it proves the refusal is
load-bearing without proving *which* behaviour depends on it.

---

## 9. `export_sweep.py` — the tool for the claim this round could not make

`rust/torch_c/pytests/export_sweep.py`, new. `arch_sweep.py` asks "does a forward
pass run" across every `transformers` architecture; this asks the harder
question, and keeps **three** verdicts apart rather than merging them:

```
exported   -- torch.export.export() returned an ExportedProgram
replayed   -- ep.module()(*inputs) ran without raising
agreed     -- the replayed outputs match eager element-wise
```

`exported` is never reported as a success on its own; the headline is `agreed`.
An `ExportedProgram` holding zero `call_function` nodes is failed as its own
stage (`export_empty`) rather than allowed into replay, where it would happily
return the placeholder and "agree" on an identity module — `docs/EXPORT.md` §4.2
is why that check exists before it can ever be needed.

It reuses `arch_sweep.py`'s config shrinking, input synthesis and failure
classification by import rather than by copy, so the two sweeps build the same
model from the same config and their results line up. It records the **worst**
deviation rather than a boolean, because a boolean cannot distinguish
bit-identical from just-inside-tolerance and that difference is the whole
question once decompositions are involved. The comparison mode attributes nothing
that upstream also refuses.

It is **not wired into `run.sh`**, for `arch_sweep.py`'s reasons: minutes to run,
hundreds of models, and a measurement rather than an invariant.

**It has not been run against the shim**, because export does not get far enough
for its result to mean anything (§3). It is landed now so that the round which
closes §7 measures agreement in the same breath as reachability instead of
inferring one from the other.

---

## 10. What was left undone, and why

* **`docs/EXPORT.md` §8's hand-off.** The 29 names are still in
  `torchnative/src/main/torchnative/export/upstream.py`, so every measurement
  here is under a runtime monkey-patch (§1.1). Moving them is mechanical and
  large, it touches the `rebind()` pass over ~40 `from torch._C import ...`
  bindings, and doing it in the same round as six behavioural changes would have
  made a bisect impossible. It is the right next mechanical task.
* **`meta_utils.py:2071`, the meta storage handle** (§7). A design question about
  the storage model, deliberately not attempted without room to verify it.
* **The `.Scalar`/`.Tensor` overload disagreement** (§5, `docs/EXPORT.md` §5).
  Now confirmed to reach a *second* front end — a `TorchDispatchMode` sees
  `mul.Scalar` where upstream dispatches `mul.Tensor`. Untouched. It should be
  settled before a graph front end starts consuming those keys, not after, since
  Core ATen and the Edge dialect are defined per overload.
* **`empty_strided` with a genuinely non-contiguous stride** (§4). Refused by
  name. Closing it needs either a stride on `Repr::Meta` or a dense tensor that
  can carry a caller-supplied layout, and both are the same storage-model
  question as §7.
* **`_dispatch_has_computed_kernel_for_dispatch_key` for keys other than
  `Meta`/`CPU`** (§6.3). Refused by name, deliberately.

---

## 11. Gates

```
suite            984 ok, 0 FAIL      (baseline 966 ok -- +18, all in test_export4.py)
DOCWATCH         882/882             (baseline 877/877 -- +5 markers, this doc and EXPORT.md)
golden           11405/11405 ops=302 (baseline 11405/11405 ops=301 -- +1, aten.empty_strided)
cargo test       30/30
```

The golden `ops` count moving by exactly one, with `cases_passed` unchanged, is
the check that this round added one operator and altered no existing kernel's
numbers.

## 12. Existing assertions that had to move, and why each is not a weakening

Four, all of them count-or-name assertions that this round's single new aten op
legitimately shifts. Each is recorded because "a test needed updating" is where
a weakening hides.

| test | was | now | why |
|---|---|---|---|
| `test_a_packet_reports_the_overloads_the_file_declares` | `registry == 1008` | `1009` | the fifth instance of a mechanism that file already documents four times: `overloads.json` carries `aten::empty_strided.out`, the yaml produces it only via `autogen:`. `registry_default` is **unchanged at 461**, which is the check that it is that mechanism |
| `test_core_ops_and_op_tags_agree` | `tag_core_count == 133` | `134` | `empty_strided` carries `tags: core` in `native_functions.yaml`, read off the file rather than asserted because it felt fundamental |
| `test_schema_text_survives_the_round_trip...` | `len(keys) == 364`, 9-entry `from_tables` | `366`, 10 entries | +2 distinct identities (`empty_strided` and `.out`); only the `.out` half joins `from_tables`, and **that split across the two halves of one op is the check** — a transcription slip would have put both there |
| `test_fake_tensor_mode_is_reached_and_names_what_stops_it_returning_a_fake` | accepted `empty_strided` / `is_inference_mode` | also accepts the meta-storage refusal | the wall it names moved because this round closed the old one. The accepted reasons stay a **list, not a wildcard**, so the next move must also be looked at |

None of the four relaxes a refusal. The one that comes closest is the last, and
it was kept as an explicit list for exactly that reason.

## 13. Corrections to `docs/EXPORT.md`

One, and it is a correction the document itself asked for.

**§2.4 is no longer true as written.** It says of `_DisableTorchDispatch`:

> "entering a counter is a correct implementation" is true here **only because of
> §4.2** … Nothing here consults the mode stack, so there is nothing to suppress.

Both halves have since become false: `_aten_dispatch` consults the mode stack
(§6 step 1), and a counter is therefore no longer a correct implementation. The
sentence was conditional and its condition expired; §5 is what it cost between
the condition expiring and this round noticing.

§3.1, §3.2 and §3.3 are accurate as written. §3.1 and §3.2 are now closed; §3.3
is not, and §7 above is it, unchanged.
