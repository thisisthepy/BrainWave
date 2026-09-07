# CTOR — the legacy `torch.Tensor(...)` constructor

`docs/architectures/DEMAND.md` §0.1 rank 1. The brief for this round said to test the framing before
inheriting it, so §1 is the framing test and it comes first, because it changes the answer.

---

## 1. The MRO, measured — and why "structural" was the wrong word

DEMAND.md §0.1 rank 1 calls this gap **structural**, and gives the reason:

> the refusal fires inside PyO3's `#[new] fn py_new` in `tensor.rs`, and the class that could
> carry a Python-level override — `torch.Tensor(TensorBase)` — lives in the vendored tree,
> which must not be modified.

The second half of that is a non sequitur, and measuring it is what this round started with.
Run on the shim (`PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1`):

```text
shim
torch.Tensor        : <class 'torch.Tensor'>  module 'torch'
MRO                 : ['torch.Tensor', 'torch._C.TensorBase', 'builtins.object']
is TensorBase?      : False
__new__  in Tensor.__dict__     : False        resolved owner: TensorBase
__init__ in Tensor.__dict__     : False        resolved owner: object
setattr on TensorBase           : settable
setattr on torch.Tensor         : settable
```

So:

- **`torch.Tensor` is a vendored-tree Python class** (`torch/_tensor.py:102`,
  `class Tensor(torch._C.TensorBase)`), re-exported by `torch/__init__.py:1931`. It is not
  `TensorBase` itself.
- **It does not define `__new__` or `__init__`.** It inherits `__new__` from `TensorBase` and
  `__init__` from `object`.
- **It is a heap type and `setattr` on it works.**

A class that *inherits* `__new__` and accepts `setattr` can be given a working `__new__` from
outside the file that declares it. Living in the vendored tree makes a class unavailable to
**editing**; it does not make it unavailable to **patching**. `bootstrap.py` installs onto
classes it does not own everywhere already — that is most of what it does.

The one real obstacle is **ordering**, not ownership: `bootstrap.py` runs while `torch._C` is
being imported, and `torch.Tensor` does not exist yet (`bootstrap.py:446` says so in as many
words, about a different problem). But the shim already has a hook for exactly that moment, and
already uses it for exactly this class:

```python
# bootstrap.py, _initExtension
tensor_cls = getattr(torch_module, "Tensor", None)
if tensor_cls is not None:
    module._set_tensor_class(tensor_cls)
```

with its own comment: *"This is the right moment for it, and it is upstream's own moment:
`torch/__init__.py:1931` runs `from torch._tensor import Tensor` and `:2189` calls this
function, so the class exists and no tensor has been made yet."*

**Verdict: not structural.** It is a `bootstrap.py` change at an existing hook. `tensor.rs` is
not touched by this round and the vendored tree is not touched by this round.

### 1.1 Why the override goes on `Tensor` and not on `TensorBase`

`TensorBase` is settable too, and putting `__new__` there would need no late hook at all. It is
still wrong, and the reason is CPython's `tp_new_wrapper` guard.

Assigning a Python `__new__` to a type replaces its `tp_new` with `slot_tp_new`. The native PyO3
constructor is then reachable only through the *saved* wrapper object, and calling it trips
CPython's safety check:

```text
TypeError: torch._C.TensorBase.__new__(Tensor) is not safe, use object.__new__()
```

— measured, by doing exactly that. The check walks up from the subtype past every `slot_tp_new`
and then compares against the wrapper's own type's `tp_new`; once `TensorBase.tp_new` is itself
`slot_tp_new`, nothing matches and the native allocator becomes **permanently unreachable from
Python**. That would take `TensorBase(existing)` — the re-wrap form every internal caller uses,
including `_make_subclass`, which is how a `Parameter` is born — down with it.

Patching only `torch.Tensor` leaves `TensorBase.tp_new` native, so the same walk succeeds:
`Tensor` is `slot_tp_new` → skip → `TensorBase` is the PyO3 `tp_new` → matches. `TensorBase.
__new__(cls, data)` keeps working, which is what `_make_subclass` already relies on and what
this round's `__new__` delegates to.

---

## 2. What upstream's `torch.Tensor(...)` actually accepts

`torch 2.13.0`, `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`, every script printing its side.
`D` below is `torch.get_default_dtype()` — measured to be the answer, not assumed: the whole
table was re-run under `set_default_dtype(torch.float64)` and the float32 entries moved with it.

| call | result | dtype |
|---|---|---|
| `Tensor()` | `(0,)` | `D` |
| `Tensor(5)` | `(5,)` uninitialised | `D` |
| `Tensor(2, 3)` | `(2, 3)` uninitialised | `D` |
| `Tensor(0)` | `(0,)` | `D` |
| `Tensor(torch.Size([2,3]))` | `(2, 3)` uninitialised — **a size, not data** | `D` |
| `Tensor(torch.Size([]))` | `()` — a **scalar**, unlike `Tensor()` | `D` |
| `Tensor([1,2,3])` | `(3,)` values `1.,2.,3.` | `D` |
| `Tensor((1,2,3))` | `(3,)` values — a plain tuple is **data** | `D` |
| `Tensor([[1,2],[3,4]])` | `(2,2)` values | `D` |
| `Tensor([])` | `(0,)` | `D` |
| `Tensor([[]])` | `(1, 0)` | `D` |
| `Tensor(range(3))` | `(3,)` values `0.,1.,2.` | `D` |
| `Tensor([True, False])` | `(2,)` values `1., 0.` | `D` |
| `Tensor(ndarray)` | values, **any** source dtype (`f16 f32 f64 i8 i16 i32 i64 u8 bool`) | `D` |
| `Tensor(0-d ndarray)` | `()` scalar | `D` |
| `Tensor(tensor)` | re-wrap, shares memory, `is` is **False** | **the source's**, not `D` |
| `Tensor(5, device='cpu')` | `(5,)` | `D` |
| `Tensor([1,2], device='cpu')` | `(2,)` values | `D` |

and the refusals, whose wording matters because a caller may be catching them:

| call | upstream |
|---|---|
| `Tensor(-1)` | `RuntimeError: Trying to create tensor with negative dimension -1: [-1]` |
| `Tensor(1.5)` | `TypeError: new(): data must be a sequence (got float)` |
| `Tensor(None)` | `TypeError: new(): data must be a sequence (got NoneType)` |
| `Tensor(np.float32(3.5))` | `TypeError: new(): data must be a sequence (got numpy.float32)` |
| `Tensor('x')` | `TypeError: new(): invalid data type 'str'` |
| `Tensor(2, 3.0)` | `TypeError: new(): argument 'size' failed to unpack the object at pos 2 …` |
| `Tensor([1,2], [3,4])` | `TypeError: new() received an invalid combination of arguments - got (list, list)` |
| `Tensor(t, t)` | `TypeError: … got (Tensor, Tensor)` |
| `Tensor(5, dtype=torch.float64)` | `TypeError: … got (int, dtype=torch.dtype)` — **`dtype=` is not accepted** |
| `Tensor(5, requires_grad=True)` | `TypeError: … got (int, requires_grad=bool)` |

**The three rules that decide the implementation**, each of which an obvious implementation
gets wrong:

1. **`torch.Size` is a size, `tuple` is data** — and `torch.Size` *is* a `tuple` subclass. A
   `isinstance(x, (list, tuple))`-first branch answers two numbers where upstream answers a
   `(2, 3)` empty tensor. It has to be checked by exact type, before the sequence branch.
2. **Data always lands on the default dtype**, never on the source's. `Tensor(int64 ndarray)` is
   `float32`. This is what separates `torch.Tensor(x)` from `torch.tensor(x)`, which infers.
3. **A tensor argument is the exception to rule 2** — `Tensor(int64 tensor)` stays `int64`. The
   re-wrap form does not cast.

### 2.1 The typed constructors

`torch.FloatTensor` and its nine siblings, which `bootstrap.py::_install_legacy_tensor_types`
already builds. Upstream, measured:

| call | result |
|---|---|
| `FloatTensor(3)` | `(3,)` float32 |
| `FloatTensor([1,2,3])` | `(3,)` float32 values |
| `FloatTensor(ndarray f64)` | `(3,)` **float32** — coerced |
| `LongTensor(ndarray f64 [1.7, 2.9])` | `(2,)` int64 `[1, 2]` — coerced, **truncating** |
| `DoubleTensor(ndarray f32)` | `(2,)` float64 |
| `BoolTensor(ndarray i64 [1,0])` | `(2,)` bool `[True, False]` |
| `FloatTensor(Size([2,3]))` | `(2,3)` float32 — **a size**, same rule as `Tensor` |
| `LongTensor(Size([2,3]))` | `(2,3)` int64 |
| `FloatTensor([1,2])` under `set_default_dtype(float64)` | still **float32** — the class's dtype wins, not `D` |
| `FloatTensor(float64 tensor)` | **`TypeError: expected TensorOptions(dtype=float, …)`** |
| `LongTensor(float32 tensor)` | **`TypeError: expected TensorOptions(dtype=long long, …)`** |

`type(...)` of every one of these is `torch.Tensor`, never `torch.FloatTensor`.

---

## 3. What this round changed

One function in `bootstrap.py`, installed at the `_initExtension` hook of §1. No `tensor.rs`
change; no vendored-tree change; no new kernel — every form routes to
`aten.lift_fresh.default`, which is the primitive `torch.tensor`, `new_tensor` and `as_tensor`
already use, or to `TensorBase`'s existing native size/re-wrap constructor.
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _make_tensor_class_new present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _array_like_data present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _sized_tensor present -->
<!-- DOCWATCH: op-implemented aten.lift_fresh.default -->

This round added **no kernel**: `ops covered` was 185 before it and 185 after. That is also why the
golden harness cannot see any of this — it dispatches by op key, and no key changed — and why the
coverage is six vendored-tree road tests instead (`test_ctor_*` in `pytests/test_shim.py`).
docs/verification/GOLDEN.md's blind spot, the same shape `as_tensor` and `meshgrid` had in docs/architectures/DEMAND1.md.

That claim was first written as `count golden_ops_covered eq 185`, which is the one thing a marker
must not do: `ops covered` is a **shared global**, so an `eq` on it fails on somebody else's commit.
It did, within the hour — docs/architectures/DEMAND2.md landed four kernels in a round running beside this one and
took it to 189. "This round added no kernel" is a fact about a diff and cannot be expressed as an
assertion about a global total; `ge` is the honest marker, and the sentence above carries the rest.
<!-- DOCWATCH: count golden_ops_covered ge 185 -->
<!-- DOCWATCH: count golden_cases_passed ge 8126 -->
<!-- DOCWATCH: count smoke_ok ge 354 -->

### 3.1 Implemented

| form | how |
|---|---|
| `Tensor()` / `Tensor(5)` / `Tensor(2,3)` / `Tensor(0)` / `Tensor(-1)` | already native (`tensor.rs::py_new`); delegated to unchanged |
| `Tensor(existing)` | already native; delegated to unchanged |
| `Tensor(torch.Size([...]))` | **new** — unwrapped to the size form, by exact type, before the sequence branch |
| `Tensor([...])`, `Tensor((...))`, `Tensor(range(...))`, nested | **new** — `lift_fresh(_tensor_new_from_data(data, D, device))` |
| `Tensor(ndarray)`, any source dtype | **new** — same path, `dtype=D` forcing the cast |
| `Tensor(..., device=...)` | **new** — carried into both branches |
| `FloatTensor(ndarray)` … `BoolTensor(ndarray)` | **new** — the ten typed classes gained the ndarray branch |
| `FloatTensor(Size([...]))` … | **new** — the `torch.Size` rule, which they had backwards |

### 3.2 Refused by name

| form | refusal |
|---|---|
| `Tensor(5, dtype=...)`, `Tensor(5, requires_grad=...)` | `TypeError` naming the keyword — matches upstream, which also rejects these |
| `Tensor(1.5)`, `Tensor(None)`, `Tensor(np.float32(3.5))` | `TypeError: new(): data must be a sequence (got <type>)` — upstream's wording |
| `Tensor('x')` | `TypeError: new(): invalid data type 'str'` — upstream's wording |
| `Tensor([1,2], [3,4])`, `Tensor(t, t)`, `Tensor(2, 3.0)` | `TypeError` naming the combination — upstream's shape |
| `FloatTensor(float64 tensor)` and every other typed/dtype mismatch | **left as it was, and recorded in §4** |

### 3.3 Divergences recorded, not papered over

<!-- DOCWATCH: hasattr as_tensor true -->

1. **`torch.Tensor(ndarray)` copies where upstream may alias.** Upstream shares the array's
   memory when the array's dtype already equals the target dtype (`Tensor(float32 ndarray)`:
   mutating the array afterwards shows through — measured `True`) and copies when a cast is
   needed (`Tensor(float64 ndarray)`: measured `False`). This shim always copies, because its
   tensors do not wrap foreign buffers. **This is the same divergence `docs/architectures/DEMAND1.md` §4
   already recorded for `torch.as_tensor`, for the same reason, and it is recorded here in the
   same words rather than being closed.**

2. **The size form's bytes are zeros where upstream's are uninitialised.** Pre-existing, from
   `tensor.rs::py_new`; `docs/kernels/KERNELS26.md` §12.2 already carries it. Reading a `torch.Tensor(n)`
   before writing it is undefined upstream, so this narrows undefined behaviour rather than
   disagreeing about a defined one — and both measured callers (`sew_d`'s
   `torch.Tensor(hidden_size).uniform_()`, `pegasus`'s `create_weight`) write it immediately.

3. **`FloatTensor(tensor of another dtype)` casts here and is a `TypeError` upstream.**
   Pre-existing behaviour of `_install_legacy_tensor_types`, found while measuring §2.1 and left
   alone this round: changing it would turn a currently-working call into a raise, and no
   measured caller reaches it. Named here so the next round does not have to re-measure it.
