# Which overload a scalar reaches, and what `.shape` hands back

**One-line result.** Upstream wraps a Python scalar into a tensor for exactly
**five** of the operators that have a `.Scalar` overload — `add`, `sub`, `mul`,
`div` and `floor_divide` — and dispatches `.Scalar` for **all the rest**:
`rsub`, `remainder`, `fmod`, `pow` (both directions) and every one of the six
comparisons. This build dispatches `.Scalar` for all of them.

**The second result is that the divergence is invisible in the answers.** Over
487 measured `(dtype × scalar × operator)` combinations spanning nine storable
dtypes and six numeric scalar shapes, the values and result dtypes agree with
upstream in every case that this build's dtype support reaches. The four ops
that dispatch a *different overload* do not produce a different number or a
different dtype in any of them. What differs is the **recorded op key** — which
`capture`, the tape and `adapt` read — and the `grad_fn` name a user prints.

So the dispatch divergence is **recorded, not closed**, and §3 says what closing
it would have cost. `Tensor.shape` is a separate matter and *was* closed: §5.

---

## 0. What this document is, and how each row got here

- Upstream is `torch` 2.13.0, the macOS arm64 wheel in
  `/Volumes/macMini/caches/spike-venv` — the same build every other measurement
  in this repository uses.
- **Which overload fires** was read from upstream with a `TorchDispatchMode`
  logger and from this build with `torch._C._capture_begin` /
  `_capture_end`, whose `nodes` carry the resolved aten key. Both report the
  op the build actually ran, neither infers it from a table.
- `docs/numerics/SCALAR.md` established that upstream's scalar *precision* rules are per
  kernel with no principle behind them. That was the reason not to infer a
  dispatch rule either, and the enumeration below was run rather than derived.
  It happens to come out tidier than SCALAR.md's did — but it was not assumed
  to.
- The **in-place** rows were measured upstream only. This build refuses to
  capture a region that writes in place (`aten.add_.Scalar writes in place;
  capture …`), so the shim column for those rows is read from the refusal
  message, which names the key the resolver picked. That is the same fact by a
  different road, and it is flagged as such below rather than presented as a
  capture result.

---

## 1. The enumeration

Left column is what a user writes. `s` is a Python `int` or `float`; **both were
run and neither changed any row**, which is itself worth recording — the choice
of overload does not depend on the scalar's Python type on either side.

| expression | upstream | this build | same? |
|---|---|---|---|
| `x + s`, `s + x`, `torch.add(x, s)`, `x.add(s)` | `add.Tensor` | `add.Scalar` | **no** |
| `x - s`, `torch.sub(x, s)`, `x.sub(s)` | `sub.Tensor` | `sub.Scalar` | **no** |
| `x * s`, `s * x`, `torch.mul(x, s)`, `x.mul(s)` | `mul.Tensor` | `mul.Scalar` | **no** |
| `x / s`, `torch.div(x, s)`, `x.div(s)` | `div.Tensor` | `div.Scalar` | **no** |
| `x // s`, `torch.floor_divide(x, s)` | `floor_divide.default` | `floor_divide.Scalar` | **no** |
| `s - x` | `rsub.Scalar` | `rsub.Scalar` | yes |
| `s / x` | `reciprocal.default` → `mul.Tensor` | `reciprocal.default` → `mul.Scalar` | decomposes the same way, then diverges on the multiply |
| `x % s`, `torch.remainder(x, s)` | `remainder.Scalar` | `remainder.Scalar` | yes |
| `torch.fmod(x, s)` | `fmod.Scalar` | *(no table entry — refuses)* | n/a |
| `x ** s`, `torch.pow(x, s)`, `x.pow(s)` | `pow.Tensor_Scalar` | `pow.Tensor_Scalar` | yes |
| `s ** x` | `pow.Scalar` | `pow.Scalar` | yes |
| `x == s` | `eq.Scalar` | `eq.Scalar` | yes |
| `x != s` | `ne.Scalar` | `ne.Scalar` | yes |
| `x < s` | `lt.Scalar` | `lt.Scalar` | yes |
| `x <= s` | `le.Scalar` | `le.Scalar` | yes |
| `x > s` | `gt.Scalar` | `gt.Scalar` | yes |
| `x >= s` | `ge.Scalar` | `ge.Scalar` | yes |
| `x.clamp(s)` | `clamp.default` | `clamp.default` | yes |
| `x.clamp_min(s)` | `clamp_min.default` | `clamp_min.default` | yes |
| `x.add(x, alpha=s)` | `add.Tensor` | `add.Tensor` | yes — the scalar is an `alpha`, not the operand |
| `torch.full((2,), s)` | `full.default` | `full.default` | yes |
| `x += s`, `x.add_(s)` | `add_.Tensor` | `add_.Scalar` † | **no** |
| `x -= s` | `sub_.Tensor` | `sub_.Scalar` † | **no** |
| `x *= s`, `x.mul_(s)` | `mul_.Tensor` | `mul_.Scalar` † | **no** |
| `x /= s` | `div_.Tensor` | `div_.Scalar` † | **no** |

† read from this build's capture *refusal*, which names the key — see §0.

**Five families diverge and eleven agree.** The five are `add`, `sub`, `mul`,
`div` and `floor_divide`, in both their out-of-place and in-place spellings.
`docs/training/BACKWARD4.md` §8 found `mul` and recorded it as one operator's problem;
it is four operators plus `floor_divide`, and it is all of them or none —
nothing in the enumeration singles `mul` out.

### 1.1 `mul_.Scalar` is the one row that was already known and is not this

`docs/numerics/SCALAR.md` §2.2 recorded that `x *= 0.3` reaches `aten.mul_.Tensor`
upstream while this build's parser reports `mul_.Scalar`, and that the two land
on kernels with **different scalar precision rules** — `mul_.Scalar` is the only
spelling of a scalar multiply in upstream that narrows. That row is the same
divergence this section enumerates, seen from the precision side. It is the one
place where the overload choice is known to reach the numbers, and it reaches
them only at `float16`/`bfloat16` with a scalar not representable in the tensor's
dtype. §2 measures the out-of-place ops and finds no such reach.

---

## 2. What the divergence costs today: nothing in the values

Nine storable dtypes × six numeric scalars (`2`, `0`, `-3`, `300`, `2.5`, `0.3`)
× nine operators, values and result dtype compared against upstream. **487
comparable cases.** Only nine disagree, and none of the nine is one of the ops
that dispatches a different overload:

| what disagrees | cases | why it is not this |
|---|---|---|
| `x ** 0.3` at `float32`, and at every integral dtype (which promotes to `float32`) | 5 | last-ulp in the `pow` kernel: `1.792790055` here, `1.792789936` upstream. **Same aten key on both sides** (`pow.Tensor_Scalar`) — a kernel accuracy question, not a dispatch one. Not in this round's territory (`aten.rs`). |
| `uint8_tensor ** 300` | 1 | upstream refuses (`value cannot be converted to type uint8_t without overflow`); this build wraps and answers. Same key both sides — a missing range check. |
| `uint8_tensor // -3` and `uint8_tensor < -3` | 2 | upstream **wraps the negative scalar into the tensor's unsigned dtype** (`-3` becomes `253`, so `3 // 253 == 0` and `3 < 253` is true); this build keeps it signed. `lt.Scalar` is the *same* key on both sides, so this is a scalar-conversion defect in the `.Scalar` path and not the overload choice. Recorded here; see §6. |
| `int8` anywhere | 54 | `torch.tensor: dtype not storable by the candle backend` — a pre-existing backend gap, unrelated. |

`add.Scalar` and `add.Tensor` promote by different rules (docs/numerics/PROMOTE.md), which
is exactly why this was worth measuring rather than reasoning about. The measured
answer is that **for every scalar a Python program can write, the two rules agree
on this build's supported dtypes.** They would only separate where upstream's
"wrapped number" concept bites — a wrapped scalar does not participate in
promotion the way a real 0-dim tensor does — and `.Scalar` reproduces that
behaviour by construction, because a `Scalar` argument is not a promotion
participant either. The two roads meet, and that is not luck: it is what the
`.Scalar` overload is *for*.

**So changing the dispatch buys no value or dtype that is currently wrong.**

---

## 3. The decision: record, do not change. And what recording costs

The change would be to wrap the scalar in a 0-dim tensor in the resolver and
dispatch `.Tensor`. It was not made, for four measured reasons and one
structural one.

1. **It buys nothing in the answers** (§2). Every value and dtype it would
   affect is already right.
2. **It would have to reimplement wrapped numbers to stay right.** Upstream's
   `add.Tensor(x, wrap(2.5))` is not the same as `add.Tensor(x, tensor(2.5))` —
   the wrapped one is flagged `is_wrapped_number` and is *excluded* from type
   promotion, which is how `int64_tensor * 2.5` comes out `float32` and not
   `float64`. This build has no such flag. Wrapping without it regresses the 487
   cases that currently pass; wrapping with it is a new concept in the resolver
   whose only purpose is to make the recorded key read differently.
3. **It would strand the `.Scalar` kernels.** `docs/numerics/SCALAR.md` §2's fix — build
   the scalar operand at `opmath_in(storage)` for `Mul` and `Div` — lives in
   `arith_scalar`, on the `.Scalar` path. After wrapping, no Python spelling
   reaches it, and the golden cases pinning it (`_scalar_rule_cases`, three
   separating scalars × two reduced dtypes) would be exercising a path no user
   takes. That is the §5.5 shape: a gate that can no longer fail.
4. **It costs a tensor allocation on the hottest path in the library.** SDPA
   scales by `mul.Scalar` and dropout by `div_.Scalar`; both are per-token.
5. **It moves the recorded op key under `capture`, `adapt` and the tape**, all
   three of which read op keys, while an eager recorder is being built on top of
   them this round.

**What recording costs, stated plainly.** Three things, and the first is the one
a user meets:

- **`grad_fn` names.** `x * 2` prints `MulBackward1` here and `MulBackward0`
  upstream. See §4 — the situation is stranger than BACKWARD4 §8 knew.
- **A recorded trace does not read like upstream's.** Anything that inspects
  `capture` nodes, or an exported graph, sees `aten.add.Scalar` where upstream's
  exporter would have written `aten.add.Tensor`. Consumers that match on the key
  need both spellings.
- **`torch.ops.aten.mul.Tensor(x, torch.tensor(2.0))` and `x * 2` are the same
  op upstream and two different ops here.** Any equivalence a caller assumes
  between those two spellings is an assumption this build breaks.

---

## 4. The `grad_fn` names, and the thing BACKWARD4 §8 could not have known

`docs/training/BACKWARD4.md` §8 declined to map `mul.Scalar` to `MulBackward0`, on the
grounds that the label would then disagree with the op in the trace and a user
reading `torch.ops.aten.mul.Scalar` directly would see a name upstream does not
give it. That reasoning is sound. What it did not check is **what the other three
ops in the same family already do.** Measured, at the raw key, on both sides:

| aten key | upstream | this build |
|---|---|---|
| `aten.add.Scalar` | `AddBackward1` | `AddBackward0` |
| `aten.add.Tensor` | `AddBackward0` | `AddBackward0` |
| `aten.sub.Scalar` | `SubBackward1` | `SubBackward0` |
| `aten.sub.Tensor` | `SubBackward0` | `SubBackward0` |
| `aten.mul.Scalar` | `MulBackward1` | `MulBackward1` |
| `aten.mul.Tensor` | `MulBackward0` | `MulBackward0` |
| `aten.div.Scalar` | `DivBackward1` | `DivBackward0` |
| `aten.div.Tensor` | `DivBackward0` | `DivBackward0` |

**`mul` is the only one of the four that is faithful to its key.** `add`, `sub`
and `div` all fall through the naive rule (CamelCase the base name, append
`Backward0`), which happens to produce the name upstream gives the *user-level
expression* — so `x + 2` prints `AddBackward0` and looks right, at the cost of
`torch.ops.aten.add.Scalar(x, 2)` printing a name upstream does not give it.
That is precisely the trade §8 declined for `mul`, taken three times by accident.

The two coherent end states are:

- make all four faithful to the key (`add.Scalar → AddBackward1`, …), which makes
  `x + 2`, `x - 2` and `x / 2` print names upstream does not print — a
  regression on the spelling users actually write; or
- make all four match the user-level expression (`mul.Scalar → MulBackward0`),
  which makes all four raw-key spellings wrong — consistently.

Neither is right. The right answer is the dispatch change §3 rejects, which makes
both columns right at once. **So this is left as it is**, inconsistent, and the
inconsistency is written down here rather than smoothed over in either direction.
A round that revisits scalar dispatch inherits this table.

### 4.1 Three rows that are *not* this, and two of them were fixed

Measured in the same run, and separated out because the aten key **agrees** on
both sides — so these are naming defects with no trade-off attached:

| aten key | upstream | was | now |
|---|---|---|---|
| `aten.rsub.Scalar` (`2 - x`) | `RsubBackward1` | `RsubBackward0` | **fixed** |
| `aten.pow.Scalar` (`2 ** x`) | `PowBackward2` | `PowBackward0` | **fixed** |
| `aten.floor_divide` (`x // 2`) | `NotImplemented` | `FloorDivideBackward0` | **recorded, not fixed** |

The first two are overload indices the naive rule cannot derive — the same class
as the `mul.Scalar`, `squeeze.dim` and `max.default` rows already in
`_GRAD_FN_NAMES`, simply never measured. Both are reached by an ordinary Python
operator, both are now in the table, and both are now in the fixture
`test_grad_fn_names_and_the_grad_mode_gate_agree_with_upstream` compares against
upstream, so they cannot rot back into a guess.

`floor_divide` is different and is deliberately left alone. Upstream's node class
is literally named `NotImplemented` — it is upstream's marker for an op with no
derivative, and the node exists in order to raise if a backward reaches it.
Naming this build's hollow node `NotImplemented` would make the *string* match
while making a claim about differentiability that nothing here backs. That is the
`_GradFnNode` boundary docs/training/BACKWARD4.md §1.3 drew, on the other side.

---

## 5. `Tensor.shape` now answers with `torch.Size` — closed

`docs/architectures/DEMAND.md` recorded this unranked because no model asked for it:
`isinstance(t.shape, torch.Size)` was `False`. It is now `True`, and the whole
`torch.Size` surface matches upstream field for field.

**What made it cheap is that the class already existed.** `bootstrap.py` has
built a real `class Size(tuple)` with a `numel()` since the type table was
written; `torch.Size((2, 3))` already worked and already answered `numel()`.
The only gap was that `TensorBase.shape` never constructed it. So this is not a
new type — it is a registration.

`tensor.rs` gets a `SIZE_CLASS` `OnceLock` and a `_set_size_class` function, the
same shape as the `TENSOR_CLASS` / `_set_tensor_class` pair beside it and for the
same reason: `_C` cannot build a Python `tuple` subclass for itself and **must
not import `torch` to find one**, because `tools/golden/loader.py` imports `_C`
standalone with no `torch` package around it. Nothing registers there, `shape`
stays the plain tuple it has always been, and the harness — which compares shapes
as sequences — cannot tell. `ops covered` is unchanged at 203 and the golden
count is unchanged at 8509/8509, which is what "structurally blind to it" means
here.

Six things were measured against upstream, and two were surprises:

| | upstream | before | now |
|---|---|---|---|
| `isinstance(t.shape, torch.Size)` | `True` | `False` | `True` |
| `isinstance(t.size(), torch.Size)` | `True` | `False` | `True` |
| `repr(t.shape)` | `torch.Size([2, 3])` | `(2, 3)` | `torch.Size([2, 3])` |
| `torch.Size([2,3,4])[1:]` | `Size` | `tuple` | `Size` |
| `S + (5,)`, `(5,) + S`, `S * 2` | `Size` | `tuple` | `Size` |
| `pickle.loads(pickle.dumps(t.shape))` | `Size` | `PicklingError` | `Size` |
| `repr(torch.Size)` | `<class 'torch.Size'>` | `<class 'torch.install.<locals>.Size'>` | `<class 'torch.Size'>` |
| `type(t.stride())` | `tuple` | `tuple` | `tuple` — **unchanged on purpose** |

- **The `repr` is not cosmetic.** It is `torch.Size([2, 3])` — square brackets
  *inside* the call, which a `tuple` subclass does not produce. Thirty-odd
  `transformers` docstrings print exactly that string, and so does every
  `print(x.shape)` a user writes.
- **The type is closed under slicing, `+`, reflected `+` and `*`.** A subclass of
  `tuple` gets none of that for free — `tuple`'s own slots return plain tuples —
  so each is written out. `(5,) + size` returning a `Size` needs an explicit
  `__radd__`; that one is easy to miss.
- **`__qualname__` had to be set.** Without it the class is
  `install.<locals>.Size`, which pickle cannot resolve, so `torch.Size` was
  unpicklable. Upstream's pickles.
- **`stride()` was left alone.** Upstream returns a *plain* tuple from
  `stride()`, measured. Wrapping every dimension list would have been a new
  divergence rather than the removal of one, and the test asserts that it is
  still a tuple.

### 5.1 What depends on either behaviour

Surveyed across the vendored tree and the installed `transformers`:

- **Nothing depends on `shape` being a plain tuple.** One `type(...)` call in
  `torch/_utils.py:59` is on a dtype, not a shape.
- **Thirteen sites in the vendored tree read `isinstance(…, torch.Size)`**, and
  the load-bearing pair is `torch/distributions/distribution.py:276` and
  `torch/distributions/categorical.py:145`, both of which say
  `if not isinstance(sample_shape, torch.Size): sample_shape = torch.Size(...)`
  and then call `.numel()`. Those worked before by upgrading, and work now by
  not needing to.
- **Fourteen sites call `.shape.numel()` or `.size().numel()`** in the vendored
  tree (`fsdp`, `fx`, `distributions`, `masked`). Every one of them was an
  `AttributeError` waiting on a code path this build does not yet run.
- `transformers/exporters/exporter_dynamo.py:463` branches on
  `isinstance(obj, torch.Size)` when serialising, and would have serialised a
  shape as a bare tuple.

The change is safe because `Size` **is** a `tuple` and compares and hashes equal
to one — `t.shape == (2, 3)` and `hash(t.shape) == hash((2, 3))` both hold, on
both sides. Those two are asserted as controls in the test, so if the change ever
grows teeth it was argued not to have, that is where it shows.

### 5.2 The check that can fail

`test_shape_and_size_answer_with_torch_size_and_it_behaves_like_upstreams`
compares the whole surface above field for field against upstream in a
subprocess, rather than against transcribed literals. Emptying `SIZE_CLASS` (or
dropping the `_set_size_class` call) makes it fail and nothing else fail —
verified by doing it: the fixture dies on `'tuple' object has no attribute
'numel'`, which is the exact symptom docs/architectures/DEMAND.md named. A new public path that
nothing breaks on is a path nobody uses.

### 5.3 A pointer that pointed nowhere

The comment on `shape` in `tensor.rs` said the difference was "recorded in
docs/design/TORCH_C.md". It was not; TORCH_C.md has never mentioned it. The real record
was in docs/architectures/DEMAND.md, three documents away. Noted because a stale cross-
reference is worse than none — it sends a reader to a file that will honestly
tell them nothing is there.

---

## 6. Left open

- **`uint8_tensor // -3` and `uint8_tensor < -3`** (§2). Upstream converts a
  negative scalar into the tensor's unsigned dtype before comparing or dividing
  (`-3` → `253`); this build compares it signed. Two cases, same aten key on
  both sides, so it is a conversion defect on the `.Scalar` path rather than a
  dispatch one. Not touched here because it is `aten.rs`'s scalar conversion and
  this round does not own that file.
- **`uint8_tensor ** 300`** answers where upstream refuses for overflow. Same
  shape of gap, same file.
- **`x ** 0.3` is one ulp out at `float32`** for every dtype that promotes there.
  `pow.Tensor_Scalar` on both sides; a kernel question.
- **`torch.fmod(x, s)`** has no overload-table entry and refuses, where upstream
  dispatches `fmod.Scalar`. The kernel is not the missing piece — the table row
  is.
- **The `grad_fn` inconsistency of §4** — three of the four scalar arithmetic ops
  name the user-level expression and one names the key. Closing it properly means
  closing §3.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs set_size_class present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_shape_and_size_answer_with_torch_size_and_it_behaves_like_upstreams present -->
<!-- DOCWATCH: op-implemented aten.mul.Scalar -->
<!-- DOCWATCH: op-implemented aten.add.Scalar -->
<!-- DOCWATCH: op-implemented aten.rsub.Scalar -->
