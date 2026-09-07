# BIND5 — one argument form, three architectures; and `torch.export`'s step two

Worktree `work/bind5` on develop `b33e2ee`, vendored tree assembled fresh. torch 2.13.0
upstream (`/Volumes/macMini/caches/spike-venv/bin/python`). Territory:
`rust/torch_c/src/bootstrap.py`, `tools/golden/reach_allow.json`, and a new
`rust/torch_c/pytests/test_bind5.py`. `aten.rs`, `tensor.rs`, `dtype.rs`, `device.rs`,
`capture.rs` and `tape.rs` were not touched; `test_shim.py` was not touched at all. The
only `reach_allow.json` edit is a **deletion**: `multinomial`'s "unexercised spelling"
entry, which §7 closes and whose own text said to remove it when a test spelled the
name.

**Leading with the answer this round was asked for, because it is a table of accepts
and refuses and not a count.** A single-element integral Tensor sitting inside a
`SymInt[]` size list, measured against real torch 2.13.0 in a separate process, one case
at a time:

| what is in the size list | upstream | this build, before | this build, after |
|---|---|---|---|
| `tensor(3)` — 0-dim, int64 | **accepts** → `(2, 3)` | `zeros` yes, `new_zeros` **no** | **accepts, both** |
| `tensor([3])` / `tensor([[3]])` — one element, 1-D and 2-D | **accepts** | as above | **accepts, both** |
| `tensor(3, uint8/int16/int32/int64)` | **accepts** | as above | **accepts, both** |
| `tensor(3.5)` — float | **refuses**, `TypeError` | **accepted it, gave `(2, 3)`** | **refuses**, `TypeError` |
| `tensor(3.0)` — whole-valued float | **refuses**, `TypeError` | **accepted it** | **refuses**, `TypeError` |
| `tensor(True)` — bool | **refuses**, `RuntimeError` | **accepted it, gave `(2, 1)`** | **refuses**, `RuntimeError` |
| `tensor([3, 4])` — two elements | **refuses**, `TypeError` | refused (`RuntimeError`) | **refuses**, `TypeError` |
| `tensor(-1)` — negative | **refuses**, `RuntimeError` | refuses, `OverflowError` | refuses, `OverflowError` (§1.4) |

Three of the eight rows are the deliverable: **this build was more permissive than the
thing it replaces**, in exactly the shape `docs/bindings/ARGFORM.md` forbids, and nothing could
see it because nothing asserted it. §1.3.

**A second argument form arrived mid-round and lands `vilt`** (§7): a single-element
integral Tensor in a **scalar** `int`/`SymInt` position, which upstream's parser accepts
and this one did not. Same three-line shape, same table:

| in a scalar `int`/`SymInt` position | upstream | this build, after |
|---|---|---|
| `tensor(2)`, `tensor([2])`, `tensor([[2]])`, any integral dtype | **accepts** | **accepts** |
| `tensor(2.0)` — float | **refuses**, `TypeError` | **refuses**, `TypeError` |
| `tensor([2, 3])` — two elements, and an empty tensor | **refuses**, `TypeError` | **refuses**, `TypeError` |
| `tensor(True)` — bool | **refuses**, `RuntimeError` | **refuses**, `RuntimeError` |

Suite **895 ok**, 0 FAIL, `DOCWATCH: PASS` 799/799, `EXIT=0`. Golden **11336/11336,
ops=299 — exactly unmoved**. No kernel added anywhere in this round.

This worktree branched before the `arch300` and `last7` rounds landed on develop, so its
numbers are its own: develop is at 886 ok / DOCWATCH 792 / golden 11385 ops=300, and the
coordinator re-measures the pinned counts at the merge.

---

## 0. At a glance

| item | verdict | architectures |
|---|---|---|
| 1. `Tensor.new_zeros((…, 0-dim int Tensor, …))` | upstream **accepts**; landed | `led`, `longformer`: **move**, onto `TensorBase.where` (§4.1) |
| 1b. float / bool / multi-element Tensor in a size list | upstream **refuses**; this build now refuses too | — (the fix is a *narrowing* of docs/bindings/BIND4.md §3) |
| 2. `torch.zeros((tuple))` for `fastspeech2_conformer` | already landed by docs/bindings/BIND4.md §3; **nothing left to pass** | stops one wall later at `repeat_interleave`, `aten.rs` (§4.2) |
| 3. `torch.embedding(Parameter, None, …)` | upstream **refuses identically**; **not a gap** | `sam3_lite_text_text_model`: same wall, correctly (§4.3) |
| 4. `torch._C._NodeBase` + `_NodeIter` + `_fx_map_arg`/`_fx_map_aggregate` | landed | `torch.fx.Graph()` **builds** (§2) |
| 5. where the `export` wall moves | **`torch.empty_strided`**, `docs/graph/EXPORT.md` §3.1 — item **three** (§3) |
| 6. `docs/bindings/BIND3.md` §7's objection to the §8 hand-off | **re-measured and gone** (§3.2) | |
| 7. single-element integral Tensor in a scalar `int`/`SymInt` | upstream **accepts**; landed in `_TypeChecker` | `vilt`: **forward runs** (§7) |

---

## 1. The size-list rule — `led`, `longformer`, and a defect in `zeros`

### 1.1 What `led` and `longformer` actually pass

`LongformerSelfAttention._sliding_chunks_query_key_matmul`, borrowed verbatim by
`modeling_led.py:440` and `modeling_longformer.py:796`:

```python
diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
    (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
)
```

`docs/kernels/STRIDED.md` §6 called this "an argument form, not a missing op", and it is — but
the tuple is not the form. **A plain `x.new_zeros((2, 3))` already worked here.** What
stops it is `chunks_count`, and the only way to know what `chunks_count` is is to look
at the real call rather than at the source: instrumented with a `new_zeros` spy on the
live model,

```text
NEW_ZEROS TUPLE ELEMENT TYPES: ['int', 'Tensor=tensor(2)', 'int', 'int']
   nonint: <class 'torch.Tensor'> tensor(2) torch.Size([]) torch.int64
```

So it is the **same** form `docs/bindings/BIND4.md` §3 landed for `torch.zeros`, arriving at a
method. Three architectures, one rule — which is what this round was told to check
rather than assume.

### 1.2 What upstream accepts, measured for the family and not only for the caller

Measured in a separate process, `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`:

```text
x.new_zeros((2, tensor(3), 4))            -> (2, 3, 4)
x.new_zeros([2, tensor(3), 4])            -> (2, 3, 4)
x.new_zeros((tensor(3),))                 -> (3,)
x.new_zeros((2, tensor(3)), dtype=, device=)  -> (2, 3), dtype honoured
x.new_ones / x.new_empty / x.new_full     -> all accept it
torch.ones / torch.empty / Tensor.view    -> all accept it   (docs/bindings/BIND4.md §3 re-confirmed)
```

**It is installed for `new_zeros` alone**, plus the `zeros` that was already there. That
is `docs/bindings/ARGFORM.md` §2's scoping and `docs/bindings/BIND4.md` §3's own precedent: upstream's rule
is general, but a rule installed table-wide accepts a Tensor in *every* `SymInt[]`
position with no matching measurement behind each of them.
`test_the_size_list_rule_is_installed_at_exactly_two_call_sites` pins the count as source
structure, so widening it is a deliberate edit and not a quiet one.

### 1.3 The defect this round found — `docs/bindings/BIND4.md` §3 was too permissive

`_coerce_symint_size_tensors` was `int(item) if isinstance(item, TensorBase) else item`.
Applied to anything Tensor-shaped, that is not upstream's rule; it is `__int__`'s. Three
of upstream's refusals were being answered with a tensor:

```text
                                shim (before)     upstream
torch.zeros((2, tensor(3.5)))   (2, 3)            TypeError  ... "type must be tuple of
torch.zeros((2, tensor(3.0)))   (2, 3)                        ints,but got Tensor"
torch.zeros((2, tensor(True)))  (2, 1)            RuntimeError  Expected scalar.isIntegral(…)
```

`docs/bindings/BIND4.md` §3's own comment said multi-element Tensors were "left to fail on their
own `__int__`", and float and bool were not considered at all. Nothing failed, because
nothing asked. This is the shape `docs/bindings/ARGFORM.md` names: **a build that accepts what the
thing it replaces refuses**, arriving through a fix that was otherwise right.

The rule is now written out: **one element (by `numel()`, not by `dim()` — `tensor([3])`
and `tensor([[3]])` are both accepted upstream) and an integral non-`bool` dtype.**
Everything else raises, with upstream's message text, and `bool` raises upstream's
*different exception class* because upstream reaches it one layer further in — past the
unpack, at the scalar conversion. `test_bind5.py` compares the class, so a single blanket
`TypeError` fails.

`int8` is absent from the accepted-dtype tests on purpose and not by oversight: this shim
cannot **construct** an `int8` tensor at all (`torch.tensor: dtype not storable by the
candle backend`), so the probe would be measuring `torch.tensor` rather than the
size-list rule. Upstream accepts `int8` there; recorded here rather than left as a hole.

### 1.4 The one row where the classes still differ, and why it is left

A negative size refuses on both sides and with different classes — upstream
`RuntimeError: zeros: Dimension size must be non-negative`, here `OverflowError: can't
convert negative int to unsigned` from the kernel below. The value is *let through* the
parser deliberately: the non-negativity rule belongs to the kernel, which already
enforces it, and reproducing it in the argument parser would be a second surface claiming
to own it. `test_a_negative_size_tensor_refuses_but_not_with_upstreams_class` asserts
both classes by name, so the divergence is pinned rather than described.

### 1.5 The `DeviceContext` trap, asked of both wrappers

`docs/bindings/BIND4.md` §3.1 is the live trap in this file: `DeviceContext.__torch_function__`
decides whether to inject `device=` by testing `func in _device_constructors()`, which
reads `torch.zeros` **fresh off the module by object identity**, so a wrapper that skips
to the inner table-driven closure silently detaches the context manager. `zeros`
reproduces the `_MODE_STACK` guard for that reason and still does.

`Tensor.new_zeros` **does not need it, and that is measured rather than reasoned about**:
a bound tensor method is not among `_device_constructors()`'s 36 names, so the answer
inside `with torch.device("meta")` is the receiver's device on both sides.
`test_the_wrappers_are_transparent_to_a_device_context` asserts both, in both columns —
the factory one is the regression check, the method one is the claim.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _coerce_symint_size_tensors present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_tensor_size_list_tensor_forms present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_new_zeros_led_and_longformer_spelling_now_computes present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_a_float_tensor_in_a_size_list_is_refused_as_upstream_refuses_it present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_a_bool_tensor_is_refused_with_upstreams_own_exception_type present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_the_size_list_rule_is_installed_at_exactly_two_call_sites present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_the_wrappers_are_transparent_to_a_device_context present -->

---

## 2. `torch.fx.Graph()` builds — `docs/graph/EXPORT.md` §6 item 2

Taken **only because item 1 landed first.** `docs/graph/EXPORT.md` §6 forbids filling this in
while the dispatcher ignores modes, and the reason is stated as a consequence: export
would run, install a proxy mode, see nothing fire, and return an `ExportedProgram` with
no operators in it. `docs/design/DISPATCH3.md` landed the entrance in `aten.rs`; modes see
operators in upstream's order; so item 2 is next and is not a shortcut past item 1.

### 2.1 What was there

A synthesised type with three real-looking members and four raising stubs, so an empty
`Graph` died on its own sentinel root node (`graph.py:1369`,
`Node(self, "", "root", "", (), {})`).

**`docs/graph/EXPORT.md` §4.1's census undercounted by three, and the three matter.** It listed
`_erased`, `_next` and `_prev` as "present and real". They were present and **raising** —
class-level getters whose body raised `NotImplementedError`. That is not a cosmetic
correction: `torch/fx/node.py:885`'s `__setattr__` calls `hasattr(self, name)` on the
very first assignment, and a getter that raises `NotImplementedError` rather than
`AttributeError` makes `hasattr` **propagate** instead of answering `False`. So they had
to be deleted from the class before anything could hold state, and an implementation that
filled only the twelve absent names and the four stubs would still not have built a
graph.

### 2.2 The part a plausible implementation gets wrong

`_sort_key`. Nodes carry a tuple that must order them the way the linked list does, and
it is maintained by `_prepend` rather than derived from a walk. Inserting `x` between `p`
and `n`:

```text
len(p) > len(n)   ->  p[:-1] + (p[-1] + 1,)
len(p) < len(n)   ->  n[:-1] + (n[-1] - 1,)
equal             ->  p + (0,)
```

Derived from measurement, not from `torch/csrc/fx/node.cpp` (this tree does not carry
it). On three placeholders at `(0,)`/`(1,)`/`(2,)`: inserting before the middle gives
`(0, 0)`, before *that* gives `(0, -1)`, appending gives `(3,)`, inserting before the
first gives `(-1,)`.

**A monotonically increasing counter reproduces the right list order and the right key
for an append-only graph**, and diverges the moment anything is inserted in the middle —
which is what every `fx` pass does. The check was falsified before being trusted:
replacing the three branches with the equal-branch alone turns
`test_the_sort_key_survives_insertion_in_the_middle` and
`test_a_graph_of_nodes_round_trips_exactly_as_upstream_builds_it` red, and nothing else
in the 889 moves. A test that cannot fail is not a test (CLAUDE.md §5.5), so it was made
to fail on purpose once.

### 2.3 Three more rules that were measured rather than chosen

* **The container conversion is asymmetric.** After `_update_args_kwargs`, a top-level
  `args` tuple is still a plain `tuple`, a nested list is an `immutable_list`, and the
  top-level `kwargs` dict is an `immutable_dict`. That falls out of `map_aggregate`
  mapping a tuple to a tuple and a list/dict to its immutable twin, applied to both — and
  the import of `torch.fx.immutable_collections` is **lazy**, because `torch.fx` does not
  exist when `bootstrap.py` runs.
* **`_NodeIter` skips erased nodes without unlinking them.** `graph.py:1619` sets
  `_erased` *after* `_remove_from_list()` and says why ("iterators may retain handles to
  erased nodes"); measured by flagging a node that is still linked, which then vanishes
  from `g.nodes`.
* **`_remove_from_list` does not self-link.** After it, the removed node still points at
  its former neighbours (measured). Resetting them to `self` would look tidier and would
  break the retained-handle case above.

`__eq__` and `__hash__` are deliberately **not** touched: `users` and `_input_nodes` are
dicts keyed by node, so identity hashing is what makes two structurally identical nodes
two different users. Only `__lt__`/`__gt__`/`__le__`/`__ge__` are installed, over
`_sort_key`.

### 2.4 The round trip, against upstream, in a separate process

Same script both sides. Node names, opcodes, sort keys, `users`, `_input_nodes`, the
printed graph, the graph after `replace_all_uses_with` and `erase_node`, `g.lint()`, the
reversed iteration order, and `GraphModule.code` — **identical**:

```text
graph():
    %a : [num_users=1] = placeholder[target=a]
    %b : [num_users=0] = placeholder[target=b]
    %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%a, 2), kwargs = {})
    return mul_tensor
```

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_fx_node_base present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_fx_graph_constructs_and_its_root_node_matches_upstreams present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_the_sort_key_survives_insertion_in_the_middle present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_a_node_added_to_a_graph_round_trips_through_every_member present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_the_node_iterator_skips_erased_nodes_without_unlinking_them present -->

---

## 3. Where the wall moves — verbatim

`torch.export.export(M(), (torch.ones(3),))` for `M.forward = (t * 2 + 1).relu()`, with
`torchnative.export.upstream.install()` applied (the 29 staged names of
`docs/graph/EXPORT.md` §2, which are still item **4** and still staged):

```text
  File ".../torch/_subclasses/meta_utils.py", line 2008, in meta_tensor
    r = callback(
        lambda: torch.empty_strided(
  File ".../torch/_subclasses/fake_tensor.py", line 506, in mk_fake_tensor
    make_meta_t(),
  File ".../torch/_subclasses/meta_utils.py", line 2009, in <lambda>
    lambda: torch.empty_strided(
  File "torch_c_bootstrap.py", line 4272, in fn
NotImplementedError: not implemented in torch._C shim: torch.empty_strided(...) --
overload resolution has no table entry for this op (rust/torch_c/src/overloads.json);
call torch.ops.aten.empty_strided.<overload>, which carries the overload and reaches the
same dispatcher
```

**That is `docs/graph/EXPORT.md` §3.1 exactly, item three, at the line §3.1 names.** It is not
item 2 and it is not a graph front end returning an empty graph — `export` does not
return. Which is the prediction this round was given and it held.

Without `install()`, `export` stops earlier and at a name rather than a kernel:

```text
  File ".../torch/_subclasses/fake_tensor.py", line 209, in unset_fake_temporarily
    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
NotImplementedError: not implemented in torch._C shim: torch._C._unset_dispatch_mode
```

So the two columns are `torch._C._unset_dispatch_mode` (item 4, the names) and
`torch.empty_strided` (item 3, the kernel), and nothing in between is `_NodeBase`'s any
more.

### 3.1 `fx.Graph` is reached with or without the names

`torch.fx.Graph()` builds in **both** columns — it does not depend on the 29 at all. That
is worth saying because it makes item 2 independently useful, which is the argument
`docs/graph/EXPORT.md` §6 gives for doing it on its own terms ("`torch.fx` is more than
export's substrate").

### 3.2 `docs/bindings/BIND3.md` §7's objection, re-measured — it is gone

That round refused to move any of the 29 names into `bootstrap.py`, and the reason was
specific and measurable: with the dispatcher ignoring modes, installing them turned a
refusal into a **silent eager fallback** — `with FakeTensorMode():` entered, reported
itself active, handed back real tensors and said nothing. One level below the empty
graph.

Re-measured on this tree, with the entrance landed:

| | before `install()` | after `install()` | upstream |
|---|---|---|---|
| `with TorchDispatchMode():` | refuses by name (`set_is_in_mode_without_ignore_compile_internals`) | **SEEN** `ones`, `mul.Scalar`, `add.Scalar`, `relu.default` | `ones`, `mul.Tensor`, `add.Tensor`, `relu.default` |
| `with FakeTensorMode(): ones(3) * 2` | refuses by name (`_only_lift_cpu_tensors`) | **raises** `torch.is_inference_mode_enabled` | returns a `FakeTensor` |
| `torch.fx.Graph()` | **builds** | **builds** | builds |

**`FakeTensorMode` no longer returns an eager tensor silently — it raises.** The
condition `docs/bindings/BIND3.md` §7 held the hand-off on is not satisfiable any more, so
`docs/graph/EXPORT.md` §8's patch is unblocked. It was **not** taken here: it is item 4, it is
a different file (`torchnative/src/main/torchnative/export/upstream.py`) and it changes
three tests in `test_export.py`, none of which is this round's territory. Recorded so the
round that does it does not have to re-derive the verdict.

The `.Scalar`/`.Tensor` disagreement in the first row is `docs/design/DISPATCH3.md` §5's, above
the dispatcher in the argument parser, and is unchanged by this round.

### 3.3 `test_export.py` was not touched, and it still holds

Its load-bearing test is written as *"if no mode sees an operator, no graph front end may
return a graph"*. With modes working it takes the `modes_work` branch, which demands that
a returned graph **contain operators** — and `export` does not return, so there is nothing
to check and nothing weakened. The stricter half is now the live half, which is what that
test was built for.

---

## 4. The architectures — `arch_sweep.py --one`, one at a time

### 4.1 `led` and `longformer` — **move**

Both stopped at `new_zeros`. Both now run the whole chunked-attention construction and
stop, together, one call later, in the same borrowed function
(`modeling_led.py:396` / `modeling_longformer.py:752`,
`_mask_invalid_locations`):

```text
    ).where(beginning_mask.bool(), beginning_input)
NotImplementedError: not implemented in torch._C shim: TensorBase.where
```

**Named as actionable, and not by this round.** `torch.where` works here and every
`where` overload is implemented (`aten.where.self`, `.default`, `.Scalar`,
`.ScalarSelf`, `.ScalarOther`); what is missing is the **method** — `methods.json` has no
`where` row, so `TensorBase.where` falls through to the surface stub. `methods.json` is
not this round's file, and a Python-level `Tensor.where` in `bootstrap.py` would be a new
binding rather than the argument form this round was scoped to, so it is recorded here as
the next item rather than taken.

### 4.2 `fastspeech2_conformer` — **nothing left to pass to `zeros`**

The `torch.zeros((tuple))` gap named for this architecture in `docs/kernels/TAIL4.md` §8.2 was
**already closed** by `docs/bindings/BIND4.md` §3 and is closed on this tree; re-measured before
anything was written, rather than assumed open because a work item said so. Its wall is
`docs/bindings/BIND4.md` §3.2's, unchanged:

```text
  File ".../modeling_fastspeech2_conformer.py", line 123, in length_regulator
    repeated = torch.repeat_interleave(encoded_embedding, target_duration, dim=0)
NotImplementedError: not implemented in torch._C shim: torch.repeat_interleave with a
tensor `repeats` ...
```

**One correction, in this file's own refusal text.** It said "this shim has neither
kernel". `aten.index_select.default` **is** implemented here — checked rather than
inherited. Only `aten::repeat_interleave.Tensor` is missing, and building the index
without it would mean reading `repeats` back to the host, which is a decision and not a
transcription. The message now says that instead of a false symmetry, so the next round
is not told to add a kernel that is already there.

### 4.3 `sam3_lite_text_text_model` — **not a gap, re-confirmed**

`docs/bindings/ARGFORM.md` §3 measured that this is upstream's own
refusal. Re-measured on this tree, both sides, same call:

```text
shim      TypeError: torch.embedding(): no matching overload in torch._C shim for
                     (Parameter, NoneType, int, bool, bool)
upstream  TypeError: embedding(): argument 'indices' (position 2) must be Tensor,
                     not NoneType
```

Same class, same rule, and **it was checked before implementing** rather than after. The
model is producing an `indices` that evaluates to `None` under the sweep's shrunk
random-weight config; that is a harness question and there is no argument form to add.

### 4.4 `vilt` — **forwards**

Not one of the three that shared §1's gap; picked up mid-round, and its wall is §7's.
`arch_sweep.py --one vilt`: **ok**, forward, `all_modalities`.

---

## 7. `torch.multinomial(Tensor, Tensor)` — `vilt`, and a third coercion site

Added mid-round. `vilt` stopped at `torch.multinomial(probs, num_samples)` with a Tensor
`num_samples`. **Not a missing kernel and not a schema gap** — `multinomial.default`
exists here and its schema matches upstream's, which `verify_schemas.py` checks. It is
upstream's argument parser, and here that is `_TypeChecker`.

### 7.1 What upstream's rule actually is, and why `bool` is the interesting one

Measured on upstream 2.13.0 in a separate process, one case at a time:

```text
multinomial(w, tensor(2))            -> (2,)          select(0, tensor(1))    -> ok
multinomial(w, tensor([2]))          -> (2,)          select(0, tensor([[1]])) -> ok
multinomial(w, tensor(2, int32))     -> (2,)          select(0, tensor(-1))   -> ok
multinomial(w, tensor(2.0))          -> TypeError   ... must be int, not Tensor
multinomial(w, tensor([2, 3]))       -> TypeError   ... must be int, not Tensor
multinomial(w, tensor(True))         -> RuntimeError  scalar.isIntegral( false)
                                                      INTERNAL ASSERT FAILED
```

Same three lines as §1's list rule, and the same asymmetry: **one element by `numel()`
(any ndim), an integral dtype, and `bool` refused with a different exception class.**
The class difference is structural rather than cosmetic, and reproducing it decided how
this is written: upstream's `ParameterType::INT64` check accepts an integral tensor
*including* `bool`, and the unpack (`torch/csrc/utils/pybind.cpp`) then asserts
`scalar.isIntegral(false)`. So the **predicate says yes and the coercion says no**, in
that order. A predicate that simply excluded `bool` would give `TypeError` and would look
right; `test_a_bool_tensor_in_a_scalar_int_position_raises_upstreams_class` is what
separates the two.

**It is upstream's parser and not one binding**, checked before being written table-wide
rather than after: `select`, `transpose`, `unsqueeze`, `narrow`, `sum(dim=)` and
`repeat_interleave` all take it upstream, unprompted. That is what makes `_TypeChecker`
the right place and a `multinomial` wrapper the wrong one — the opposite conclusion from
§1, and it is the *measurement* that differs, not the taste: §1's rule is about
`SymInt[]` **elements**, this one is about scalar positions.

**The `overloads.json` shortcut was not available and is worth recording as closed.** An
invented row for a `(Tensor, Tensor)` signature fails `verify_schemas.py`, which checks
every schema string against upstream.

### 7.2 The defect the change exposed: a THIRD coercion site

`_Overloads.resolve` has two coercion sites (positional and keyword). There is a third —
`_compile_fast_path` generates one, and it reproduced only the sized-int-list coercion.
So with the predicates widened, the fast path handed the **raw Tensor** to the dispatcher
while the slow path handed an int.

That did not fail. `aten.rs` unpacks a single-element tensor anyway, **including a `bool`
one**, so `x.select(0, tensor(True))` came back as a shape-`(4,)` tensor rather than as
upstream's `RuntimeError`. A divergence from upstream returning a plausible answer, by
spelling — positional went one way and keyword the other. The fast path now emits the
same coercion, and `test_the_positional_and_keyword_spellings_of_it_agree` holds the two
spellings against each other rather than each against upstream separately, which is what
makes it a check on the *pair*.

### 7.3 What was deliberately not opened

A Tensor as an **element of a sized int list** (`grid.sum(dim=tensor(0))`, an
`int[1]? dim`). Upstream accepts it; this build refuses it. That position reaches the
same parser check upstream, so opening it is one line — and it is the table-wide widening
§1.2 declined, with no measured caller behind it. **This build is narrower than upstream
here, on purpose**, and `test_a_tensor_inside_a_sized_int_LIST_is_still_refused_here`
asserts the absence in both directions so a later round inverts it rather than closing it
silently.

### 7.4 `reach_allow.json`

`multinomial`'s `shape3_unexercised_spelling` entry is **deleted**. Its own text said the
name was allowlisted because "a test asserting a value against upstream on a random op is
exactly the shallow-coverage failure mode this file warns against" — true, and the reason
it is closable now is that the test does not depend on the RNG at all: a **one-hot**
weight vector with `replacement=True` has exactly one legal answer, so the values are
compared and not only the shape. The suite fails the moment the entry is stale
(`REACH ... 'multinomial' is allowlisted as unexercised, but a test now spells it`),
which is how this was found rather than remembered.

`arch_sweep.py --one vilt`: **ok**, forward runs on `all_modalities`.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _symint_from_tensor present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _fast_symint_coerce present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_multinomial_takes_a_tensor_num_samples_which_is_vilts_wall present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_a_bool_tensor_in_a_scalar_int_position_raises_upstreams_class present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_the_positional_and_keyword_spellings_of_it_agree present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_bind5.py test_a_tensor_inside_a_sized_int_LIST_is_still_refused_here present -->

---

## 8. What this round changed, split the way CLAUDE.md §5.3 asks

| | |
|---|---|
| **feature added** | two argument forms — `Tensor.new_zeros` with a single-element integral Tensor in `size` (§1), and a single-element integral Tensor in any scalar `int`/`SymInt` position (§7); `torch._C._NodeBase`, `_NodeIter`, `_fx_map_arg`, `_fx_map_aggregate` — enough that `torch.fx.Graph()` constructs (§2) |
| **defect fixed** | `docs/bindings/BIND4.md` §3's coercion accepted float, whole-float and bool Tensors that upstream refuses (§1.3); the generated fast path reproduced only one of `resolve`'s two coercions, so positional and keyword spellings could disagree (§7.2); the `repeat_interleave` refusal claimed a missing kernel that exists (§4.2) |
| **tests added** | 27, in `rust/torch_c/pytests/test_bind5.py`. No test was modified, inverted or deleted anywhere |
| **documentation corrected** | `docs/graph/EXPORT.md` §4.1's `_NodeBase` census listed `_erased`/`_next`/`_prev` as "present and real"; all three were raising getters (§2.1). `docs/bindings/BIND3.md` §7's verdict on the §8 hand-off is superseded by measurement (§3.2) |
| **deleted** | `reach_allow.json`'s `multinomial` entry, which §7 closes (§7.4) |
| **kernels added** | **none.** Golden 11336/11336 ops=299, exactly unmoved |

"No unimplemented left" is **not** claimed for anything here. `led`/`longformer` moved to
a named next item, `fastspeech2_conformer` did not move, `sam3_lite_text_text_model`
cannot move from this file, and `torch.export` does not produce a graph — §3 says exactly
where it stops.

---

## 9. Reproduction

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-bind5
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export TORCH_C_STAGE=/tmp/stage-bind5
PY=/Volumes/macMini/caches/spike-venv/bin/python

cd rust/torch_c && cargo build --release && cd ../..
bash vendor/install_shim.sh                       # bootstrap.py is include_str!'d
PYTHON=$PY sh rust/torch_c/pytests/run.sh         # 895 ok, DOCWATCH: PASS 799/799
$PY tools/golden/compare.py                       # 11336/11336, ops=299

# §2, in four lines
PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY -c '
import torch, torch.fx
g = torch.fx.Graph(); a = g.placeholder("a")
g.output(g.call_function(torch.ops.aten.mul.Tensor, (a, 2))); print(g)'

# §3, both columns
PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY -c '
import torch, torch.nn as nn
from torchnative.export import upstream; upstream.install()
class M(nn.Module):
    def forward(self, t): return (t * 2 + 1).relu()
torch.export.export(M(), (torch.ones(3),))'

# §4, one at a time -- --one takes a single name
for m in led longformer fastspeech2_conformer sam3_lite_text_text_model vilt; do
  PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
    $PY rust/torch_c/pytests/arch_sweep.py --one $m
done
```

`test_bind5.py` runs its own upstream subprocess with `PYTHONPATH` and
`TORCH_USE_RTLD_GLOBAL` stripped, and skips silently on the shim side when
`vendor/install_shim.sh` has not run — the same guard `test_bind4.py` and
`test_bind3.py` carry, for the same reason.

<!-- DOCWATCH: count golden_cases_total ge 11336 -->
<!-- DOCWATCH: count golden_cases_passed ge 11336 -->
<!-- DOCWATCH: count golden_ops_covered ge 299 -->
<!-- DOCWATCH: count golden_pending eq 0 -->
