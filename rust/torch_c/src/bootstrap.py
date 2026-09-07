"""Build the `torch._C` *name* surface, at `_C` import time, inside the `.so`.

Read docs/models/IMPORT_TORCH.md first; this is the file that document describes.

Why this file exists
--------------------
VENDOR.md measured what the vendored Python tree demands of `_C` before
`import torch` finishes: 989 names, 27-32 C submodules that are themselves
packages, a 694-member `TensorBase`, a 985-member `_VariableFunctions`, and
74 enum instances written into the `torch` module rather than into `_C`.
None of that is computation. It is names, types, and metaclasses -- built
dynamically, once, at import.

Doing it here rather than in Rust is a mechanical choice, not an
architectural one: the operations are identical either way (create a heap
type, set an attribute, insert into `sys.modules`), Rust would spell them
through `Bound<'_, PyAny>` at several times the length, and it would still
run at exactly this moment. `_C` remains one native extension module -- this
file is compiled into it with `include_str!` and executed by
`run_bootstrap` in `lib.rs`.

Everything with *behaviour* stays in Rust: dtype, device, TensorBase, and
the aten dispatcher. Nothing here computes. Every callable it creates either
routes into `_C._aten_dispatch` -- the single door DESIGN.md §6 depends on --
or raises `NotImplementedError` naming itself.

The three rules
---------------
1. **No module-level `__getattr__` on `_C`.** The vendored tree turns whole
   subsystems off by asking `hasattr(torch._C, "_c10d_init")`
   (VENDOR.md wall 11). A catch-all would answer yes to every one of those
   and switch on distributed, RPC, CUDA, XPU, MTIA and MPS at once. A name is
   present here only if the surface table has it and the tree does not merely
   *probe* for it.

2. **Types and submodules do get a catch-all**, because the stubs are
   incomplete for them (`_special`, `_fft` and `_linalg` have no stub at all)
   and because nothing gates on a *member* of a `_C` type the way it gates on
   a name in `_C`. The catch-all hands back an object that raises when used.

3. **Anything not implemented names itself.** DESIGN.md §6 -- the shim is its
   own instrument, so running the tree produces the work queue.

Where the names come from
-------------------------
`vendor/gen_surface.py`, from `torchnative/src/main/torch/_C/*.pyi` -- the vendored tree's
own stubs. Not from an installed upstream `_C.so`. The distinction matters:
the stubs are the tree's statement of what it expects, they ship under the
same BSD licence as the rest of the vendored tree, and using them keeps the
build from needing real torch present.
"""

from __future__ import annotations

import builtins
import contextlib
import datetime
import enum
import functools
import importlib.util
import inspect
import json
import math
# `numbers` for the ABCs a `numpy` scalar registers with -- see the `Scalar`
# predicate in `_TypeChecker`. Importing numpy itself here is not an option:
# the shim has no numpy dependency and `_C` is imported before it would be
# available.
import numbers as _numbers
import os
import re
import string
import sys
# `threading` for `_install_thread_local_store` only. Upstream's
# `_stash_obj_in_tls` is a C++ thread-local, and a module-level dict would be a
# *different* semantics that happens to agree in a single-threaded process --
# docs/training/BACKWARD2.md §4.3.
import threading
import traceback as _traceback
import types

# ---------------------------------------------------------------------------
# Deliberate default: scripting reports itself unavailable
# ---------------------------------------------------------------------------
#
# There is no TorchScript frontend here (docs/graph/TORCHSCRIPT.md has the size of
# what is missing -- upstream's `torch/csrc/jit/` is ~213k lines of C++, and
# the first symbol the frontend needs, `SourceRangeFactory.make_range`, is
# where this shim has always stopped, naming itself). That is not going to
# change from inside `bootstrap.py`.
#
# Upstream already has an off switch for exactly this: `PYTORCH_JIT=0` (read
# once by `torch/jit/_state.py:EnabledProxy.__init__`, unmodified in the
# vendored tree). With it set, `torch.jit.script(obj)` and
# `torch.jit.script_method(fn)` both take their own `if not _enabled: return
# obj` branch and hand back the original Python function, unscripted --
# upstream's own documented fallback, not a behaviour invented here. A
# `@torch.jit.script` at module scope (`transformers/models/gpt_bigcode/
# modeling_gpt_bigcode.py:54`, and five other modeling files -- see
# docs/graph/TORCHSCRIPT.md §5) then imports as a plain function instead of running
# a compiler frontend that is not built.
#
# `setdefault`, not an unconditional set: a caller who explicitly asks for
# `PYTORCH_JIT=1` still gets the real (and here, `NotImplementedError`-naming)
# path -- this only changes the answer nobody supplied. Must run before
# `torch.jit._state` is first imported, which happens during `import torch`
# itself (`torch/__init__.py` imports `torch.jit` after `from torch._C import
# *`); `_C`'s own import is what runs this file, so the ordering holds without
# needing a hook anywhere else.
os.environ.setdefault("PYTORCH_JIT", "0")

# ---------------------------------------------------------------------------
# Deliberate omissions
# ---------------------------------------------------------------------------
#
# VENDOR.md wall 11, generalised. Each of these is a name the vendored tree
# probes with `hasattr(torch._C, ...)` in order to decide whether a subsystem
# exists. Leaving the name out is the off-switch upstream itself provides, and
# it is the only one it provides -- there is no module-level flag. The list was
# produced by grepping the vendored tree for those probes, so every entry is a
# switch that is really wired, not a guess about what might be optional.
#
# This is not the same as "not implemented". A name that is merely unbuilt
# still appears, and raises when used. A name in this set must *not* appear,
# because its absence is the answer to a question the tree asks.
# `vendor/gen_surface.py` derives the real list by scanning the tree for
# `hasattr(torch._C, "...")` and `getattr(torch._C, "...", ...)`; it arrives in
# the surface under "probes", and `install` reads it from there. Deriving it
# beats hand-listing: the hand-written version this replaced was already
# missing `_cuda_isInBadFork`, `_mps_is_in_bad_fork` and `_xpu_isInBadFork`.
#
# These are the ones a scan cannot see.
EXTRA_OFF_SWITCHES = frozenset(
    {
        # `torch/cuda/streams.py:11` and `torch/xpu/streams.py:11` guard on the
        # `*StreamBase` sibling and then use the event type unconditionally
        # inside that guard, so the event type has to go when the stream does.
        "_CudaEventBase",
        "_XpuEventBase",
    }
)

# Probes the scan finds but that this shim answers YES to.
#
# `gen_surface.py` derives the off-switch list by scanning the tree for
# `hasattr(torch._C, "...")`, so every subsystem the tree can be asked about
# lands in "probes" whether or not it is built here. That is the right default
# -- absence is how upstream says "not built" -- but it means a subsystem that
# *does* get built has to be taken back out by name, here, rather than by the
# surface generator guessing.
#
# `_c10d_init` is the switch for `torch.distributed`
# (`torch/distributed/__init__.py:28`). docs/distributed/DISTRIBUTED.md is what it costs and
# what it buys; `_install_distributed_c10d` is the implementation. Answering it
# is not free: with it off, `torch.distributed` is a five-line stub, and with it
# on the tree walks straight into `_C._distributed_c10d` while `import torch`
# is still running (`torch/utils/data/dataloader.py:26` pulls it in), so the
# subsystem has to be complete enough to finish that import before this line
# can be true.
ANSWERED_PROBES = frozenset({"_c10d_init"})

# Dunders `TensorBase` must *not* be given, even though the stub declares them.
#
# The rest it must: `torch/_tensor.py:1115` is `__itruediv__ =
# _C.TensorBase.__idiv__`, so the operator set is demanded by name in the class
# body, the same way the ordinary methods are. But a raising stand-in for any
# of these breaks the object rather than making it loud --
#
#   `__getattribute__`/`__setattr__`/`__getattr__`  intercept everything
#   `__del__`                                       raises during collection
#   `__init__`/`__new__`                            stop construction
#   `__hash__`/`__eq__`                             break dict and set use
#   `__repr__`/`__str__`/`__format__`/`__dir__`     break every traceback
#   `__reduce__`/`__reduce_ex__`/`__sizeof__`       break copy and pickle
#
# -- and several of them are consulted by the interpreter itself, where an
# exception has nowhere to go. They are left absent so the failure, if one
# comes, is an `AttributeError` naming the member rather than a `TensorBase`
# that cannot be printed.
UNSAFE_DUNDERS = frozenset(
    {
        "__init__",
        "__new__",
        "__init_subclass__",
        "__subclasshook__",
        "__class__",
        "__getattribute__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__del__",
        "__dir__",
        "__doc__",
        "__module__",
        "__dict__",
        "__slots__",
        "__repr__",
        "__str__",
        "__format__",
        "__hash__",
        "__eq__",
        "__ne__",
        "__sizeof__",
        "__reduce__",
        "__reduce_ex__",
    }
)

_BUILTIN_EXCEPTION_BASES = {
    name
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type)
    and issubclass(getattr(builtins, name), BaseException)
}


class _Unimplemented:
    """A name that exists and refuses.

    DESIGN.md §6: the message is the work queue, so it always carries the
    fully-qualified name. It is deliberately *not* a chameleon -- the record
    mode of `vendor/probe.py` uses one of those, which is right for an
    instrument and wrong here, because a placeholder that answers every
    question quietly produces wrong behaviour instead of a work item.

    `__bool__` is part of that, and it was missing. Without it every instance
    is truthy, so `if torch._C._has_cudnn:` answered *yes* -- the class doing
    precisely what its own docstring says it must not. One site was measured to
    reach that branch on the `import torch` path
    (`torch/backends/cudnn/__init__.py:231`, a class body: `benchmark_limit`
    became a pair of placeholders where upstream without cuDNN leaves it
    `None`). It raises now.

    Note which question this refuses. Names the stubs declare as functions come
    from `_make_function` and are ordinary Python functions -- truthy, as
    upstream's builtins are. An `_Unimplemented` is what is left when the stubs
    say *nothing* about a name, so "is it there?" is a question the shim has no
    grounds to answer either way, and saying so is the honest reply.
    """

    __slots__ = ("_qualname", "__dict__")

    def __init__(self, qualname: str) -> None:
        self._qualname = qualname

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"not implemented in torch._C shim: {self._qualname}"
        )

    def __bool__(self):
        raise NotImplementedError(
            "torch._C shim cannot answer whether this exists: "
            f"{self._qualname}. It is a placeholder, and a truth test on one "
            "used to silently mean yes."
        )

    def __repr__(self) -> str:
        return f"<torch._C shim: {self._qualname} (not implemented)>"


def _make_function(qualname: str, name: str):
    """A real Python function, not a `_Unimplemented`.

    Two places need the difference. `torch/__init__.py:2212` assigns
    `__module__` on every harvested `_VariableFunctions` member, and
    `_C._add_docstr` assigns `__doc__` on hundreds of `TensorBase` members --
    both are writable on a function and neither is on a builtin.
    """

    def fn(*args, **kwargs):
        raise NotImplementedError(f"not implemented in torch._C shim: {qualname}")

    fn.__name__ = name
    fn.__qualname__ = qualname
    return fn


def _make_property(qualname: str):
    """`TensorBase.is_sparse` and friends.

    `torch/_prims_common/__init__.py:90` pulls the *descriptor* out of the
    class (`torch.Tensor.is_sparse.__get__`), so these cannot be plain values
    (VENDOR.md wall 10).
    """

    def getter(self):
        raise NotImplementedError(f"not implemented in torch._C shim: {qualname}")

    return property(getter)


class _ShimMeta(type):
    """Metatype for synthesised `_C` types.

    Two unrelated jobs, both measured in VENDOR.md.

    *Members.* `torch/_torch_docs.py` reaches into `_C` types at import time
    (`_add_docstr(torch.Stream.query, ...)`), and the stubs do not list every
    member. Unknown ones are answered here rather than left to fail, because a
    missing member of a type is a hole, not a switch (rule 2 in the module
    docstring).

    *Identity.* `torch/_awaits/__init__.py:12` builds
    `class _PyAwaitMeta(type(torch._C._Await), type(Generic))`, which fails
    with `duplicate base class` if `type(_Await)` is plain `type`
    (VENDOR.md wall 12). Upstream's answer is pybind11's metatype; this is
    ours. The mirror case -- `torch/autograd/variable.py:14` needs
    `type(_C._LegacyVariableBase)` to *be* `type` -- is handled by
    `PLAIN_TYPE_METACLASS` below.
    """

    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        value = _Unimplemented(f"{cls.__module__}.{cls.__name__}.{name}")
        setattr(cls, name, value)
        return value


# Types whose metatype must be exactly `type`. `torch/autograd/variable.py:14`
# is `class Variable(_C._LegacyVariableBase, metaclass=VariableMeta)` where
# `VariableMeta(type)`; a richer metatype on the base is a metaclass conflict.
# and `torch/autograd/function.py:365` is `class _SingleLevelFunction(
# _C._FunctionBase, ..., metaclass=FunctionMeta)` with `FunctionMeta(type)`.
#
# VENDOR.md wall 12 measured upstream's distribution -- 135 `pybind11_type`,
# 51 `type`, 2 `_TensorMeta`, 1 `OpaqueBaseMeta` -- and concluded that no
# blanket rule works because the tree depends on the difference in *both*
# directions. This is the other half of that: the default is `_ShimMeta`
# (needed by `_Await`, which must not be plain `type`), and the exceptions are
# found by running the tree, because a metaclass conflict is only visible when
# the class statement executes.
PLAIN_TYPE_METACLASS = frozenset({"_LegacyVariableBase", "_FunctionBase"})

# Submodules the stubs do not declare but the tree imports anyway. Upstream
# registers 32 from C; the stub directory covers 27. The rest are found by
# running the tree, which is the only way they can be found -- see
# VENDOR.md §6 item 2 on requirements that no amount of reading reveals.
EXTRA_SUBMODULES = (
    "_special",
    "_fft",
    "_linalg",
    "_sparse",
    "_nested",
    "_nn_functional",
    "_VariableFunctions",
)


class _SubmoduleFinder:
    """Serves `torch._C.<known>` and anything nested under it.

    `from torch._C._jit_tree_views import SourceRangeFactory` is an *import
    statement*, so an attribute on `_C` does not satisfy it -- `_C` is a
    compiled module and has no `__path__` for the machinery to search
    (VENDOR.md wall 8). Upstream registers its submodules into `sys.modules`
    from C; this does the same, and adds a finder for the nested case
    (`torch/utils/_python_dispatch.py:22` imports `torch._C._dynamo.guards`,
    so the submodules are themselves packages).

    The finder answers only for names below a submodule it already created.
    An unknown `torch._C.<name>` still fails, so `try: import ... except
    ImportError` stays a working question.

    `closed` names it for the submodules where that last sentence has to hold
    *inside* them too. `from torch._C._distributed_c10d import ProcessGroupGloo`
    tries the attribute first and the import second, so a module `__getattr__`
    that raises `AttributeError` is not enough on its own -- this finder was
    then handing back an empty module, `_GLOO_AVAILABLE` came out True, and
    `init_process_group` reached for a backend that is not there. A closed
    submodule keeps its own catch-all; what it loses is the second chance to
    answer as a *module*.
    """

    def __init__(self, prefix: str, roots: set[str], closed=()) -> None:
        self._prefix = prefix + "."
        self._depth = prefix.count(".") + 1
        self._roots = roots
        self._closed = frozenset(closed)

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._prefix):
            return None
        tail = fullname[len(self._prefix) :]
        head = tail.split(".")[0]
        if head not in self._roots:
            return None
        if head in self._closed and "." in tail:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        _attach_module_catchall(module)
        return module

    def exec_module(self, module):
        return None


def _attach_module_catchall(module) -> None:
    """PEP 562 `__getattr__`, with a cache so identity holds.

    `torch/_ops.py:132` does `isinstance(k, TransformType)` where both the
    value and the class came out of `torch._C._functorch`. Handing out a fresh
    object per lookup breaks that for a reason that has nothing to do with
    torch, so every synthesised member is remembered.
    """
    name = module.__name__

    def __getattr__(attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        qualname = f"{name}.{attr}"
        if attr[:1].isupper() or attr.lstrip("_")[:1].isupper():
            value = _ShimMeta(attr, (), {"__module__": name, "__init__": _permissive_init})
        else:
            value = _Unimplemented(qualname)
        setattr(module, attr, value)
        return value

    module.__getattr__ = __getattr__


def _permissive_init(self, *args, **kwargs):
    """Synthesised `_C` types have to be *constructible*, with arguments.

    `torch/_sources.py:87` is `class SourceContext(SourceRangeFactory)` and its
    `__init__` calls `super().__init__(source, filename, file_lineno,
    leading_whitespace_len)` -- while `torch/nn/functional.py` is still
    importing, from inside the TorchScript source parser that `@_overload` runs
    at decoration time (VENDOR.md wall 13). So existing is not enough.
    """
    return None


def _build_type(name, spec, module_name, resolved):
    """One synthesised `_C` type."""
    bases = []
    for base in spec.get("bases", ()):
        if base in resolved:
            bases.append(resolved[base])
        elif base in _BUILTIN_EXCEPTION_BASES:
            bases.append(getattr(builtins, base))
        # `Tensor`, `Protocol`, `Generic[...]` and the like are dropped: they
        # are annotation-only bases in the stub, and `Tensor` in particular is
        # `torch.Tensor`, which does not exist while `_C` is being imported.

    if "Enum" in spec.get("bases", ()) or "IntEnum" in spec.get("bases", ()):
        # A real `enum.Enum`, because the tree type-checks against these:
        # `torch/_ops.py:139` asserts `isinstance(k, DispatchKey)`
        # (VENDOR.md wall 14). A stand-in class with class attributes would
        # pass the name lookup and fail the assertion.
        members = [m for m in spec.get("attrs", ()) if not m.startswith("__")]
        base = enum.IntEnum if "IntEnum" in spec.get("bases", ()) else enum.Enum
        cls = base(name, members, module=module_name)
        return cls

    body = {
        "__module__": module_name,
        "__init__": _permissive_init,
        "__doc__": f"torch._C shim placeholder for {module_name}.{name}",
    }
    for member in spec.get("methods", ()):
        if member.startswith("__") and member.endswith("__"):
            continue
        body[member] = _make_function(f"{name}.{member}", member)

    if "get" in spec.get("methods", ()) and "JitType" in spec.get("bases", ()):
        # The 14 TorchScript type singletons -- `_C.IntType.get()`,
        # `_C.TensorType.get()` and so on. `torch/_higher_order_ops/schema.py:56`
        # builds a dict of them at import, so `get` must return an object, and
        # the same object each time: the tree uses them as dict *keys* and as
        # identity comparisons. A fresh instance per call would be a subtle,
        # silent wrong answer rather than a loud one.
        body["get"] = classmethod(_singleton_getter())
    for member in spec.get("attrs", ()):
        if member.startswith("__") and member.endswith("__"):
            continue
        body.setdefault(member, _make_property(f"{name}.{member}"))

    meta = type if name in PLAIN_TYPE_METACLASS else _ShimMeta
    if bases:
        # An exception type must really derive from the exception it claims to
        # (`torch/distributed/__init__.py:41` catches `_C._DistError`), and a
        # metatype cannot be imposed on a base that has a stricter one.
        meta = type(bases[0]) if isinstance(bases[0], type) and issubclass(
            bases[0], BaseException
        ) else meta
    return meta(name, tuple(bases), body)


def _singleton_getter():
    cache: dict = {}

    def get(cls):
        if cls not in cache:
            cache[cls] = cls()
        return cache[cls]

    return get


def _order_types(types_spec):
    """Bases before subclasses. 26 `_C` types derive from `JitType`."""
    remaining = dict(types_spec)
    ordered = []
    made = set()
    while remaining:
        progressed = False
        for name in list(remaining):
            spec = remaining[name]
            deps = [b for b in spec.get("bases", ()) if b in types_spec]
            if all(d in made for d in deps):
                ordered.append((name, spec))
                made.add(name)
                del remaining[name]
                progressed = True
        if not progressed:
            # A base cycle in the stub. Emit the rest without their bases
            # rather than looping; if this ever fires it is a real finding.
            for name, spec in remaining.items():
                spec = dict(spec, bases=[])
                ordered.append((name, spec))
            break
    return ordered


# ---------------------------------------------------------------------------
# The op registry -- `torch.ops.aten.<op>.<overload>`
# ---------------------------------------------------------------------------


class _SchemaType:
    """A type inside a schema, kept as its source spelling.

    Not comparable to the TorchScript type singletons (`_C.TensorType.get()`
    and friends): `torch/_library/utils.py:163` decides "is this a tensor
    argument" by comparing against those objects, and answering `True` from a
    string match would be claiming a correspondence the shim has not built.
    Answering `False` makes the callers take their conservative branch, which
    is the safe direction.
    """

    __slots__ = ("_spelling",)

    def __init__(self, spelling: str) -> None:
        self._spelling = spelling

    def __str__(self) -> str:
        return self._spelling

    def __repr__(self) -> str:
        return self._spelling

    def __eq__(self, other) -> bool:
        return isinstance(other, _SchemaType) and self._spelling == other._spelling

    def __hash__(self) -> int:
        return hash(self._spelling)

    # The TorchScript type singletons carry no spelling of their own, but the
    # class they are instances of is named for the type -- `TensorType.get()`
    # is a `TensorType`. That is a reading, not a correspondence: it does not
    # make a `_SchemaType` *be* one of those objects, which is what the
    # docstring above declines to claim.
    _SINGLETON_SPELLINGS = {
        "TensorType": "Tensor",
        "IntType": "int",
        "SymIntType": "SymInt",
        "FloatType": "float",
        "ComplexType": "complex",
        "BoolType": "bool",
        "StringType": "str",
        "NumberType": "Scalar",
        "NoneType": "None",
        "DeviceObjType": "Device",
        "GeneratorType": "Generator",
        "AnyType": "Any",
    }

    @classmethod
    def _spelling_of(cls, other) -> str:
        if isinstance(other, _SchemaType):
            return other._spelling
        spelling = cls._SINGLETON_SPELLINGS.get(type(other).__name__)
        if spelling is None:
            raise NotImplementedError(
                "torch._C shim: FunctionSchema types can be compared against "
                "the scalar TorchScript type singletons only, and "
                f"{type(other).__name__} is not one of them. A parametrised "
                "type (ListType.ofInts() and friends) would need a real type "
                "lattice, which this shim does not have"
            )
        return spelling

    def isSubtypeOf(self, other) -> bool:
        """Equality on the normalised spelling, plus `Any` on top.

        `torch/_subclasses/fake_impls.py:145` asks this of every argument of
        every prim (`torch/_prims/__init__.py:368`), and it used not to be here
        at all -- the question never arrived, because `_get_schema` answered
        with an empty argument list and `any(...)` over nothing is False. Once
        the schemas became real the question became real too.

        Only the two relations the callers need are answered. `T` is a subtype
        of `T` and of `Any`; `T?` and `T[]` are *not* subtypes of `T`, which is
        upstream's rule and is why `contains_tensor_types` recurses through
        `containedTypes()` rather than relying on this. Anything wider would be
        a type lattice, and inventing one would let a wrong answer through
        quietly.
        """
        other_spelling = self._spelling_of(other)
        if other_spelling == "Any":
            return True
        mine = _decompose_type(self._spelling)
        theirs = _decompose_type(other_spelling)
        return mine[:3] == theirs[:3]

    def containedTypes(self) -> list:
        """The element type of a container, one layer at a time.

        `Tensor[]?` yields `Tensor[]`, which yields `Tensor` -- so
        `contains_tensor_types`'s recursion terminates and finds the tensor
        inside an optional list.
        """
        text = self._spelling.strip()
        if text.endswith("?"):
            return [_SchemaType(text[:-1].strip())]
        if text.endswith("]"):
            base, _, _, _ = _decompose_type(text)
            return [_SchemaType(base)]
        return []


#: `(op spelling, predicate)` for every question answered from a schema with no
#: text behind it. Read through `_C._shim_unanswered_predicates()`; see
#: `_Schema._answer_without_text` for what is and is not in here.
_UNANSWERED_PREDICATES: set = set()


class _AliasInfo:
    __slots__ = ("is_write", "before_set", "after_set")

    def __init__(self, is_write: bool, symbols) -> None:
        self.is_write = is_write
        self.before_set = set(symbols)
        self.after_set = set(symbols)


class _Argument:
    __slots__ = ("name", "type", "kwarg_only", "default_value", "alias_info", "N")

    def __init__(self, name, typ, kwarg_only, default_value, alias_info, N=None):
        self.name = name
        self.type = typ
        self.kwarg_only = kwarg_only
        self.default_value = default_value
        self.alias_info = alias_info
        self.N = N

    def has_default_value(self):
        return self.default_value is not None

    def __repr__(self):
        return f"{self.type} {self.name}"


def _split_top_level(text: str) -> list:
    """Split on commas that are not inside (), [] or ''.

    Backslash escapes are honoured inside a quoted run, because upstream's own
    schemas contain one: `aten::_test_string_default(Tensor dummy,
    str a='\\"\\'\\\\', ...)` closes its single-quoted default with an *escaped*
    quote, and a scanner that stops at the first bare `'` splits that argument
    in half.
    """
    out, depth, quote, start = [], 0, "", 0
    escaped = False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if quote:
            if ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    tail = text[start:]
    if tail.strip():
        out.append(tail)
    return [chunk.strip() for chunk in out if chunk.strip()]


class _Schema:
    """`torch._C.FunctionSchema`, parsed rather than stubbed.

    `torch._C.parse_schema` is called at import from seven places
    (`torch/library.py:80` and `:151`, `torch/_prims/__init__.py:322`,
    `torch/_library/custom_ops.py:742`, ...), and what those callers do with
    the result -- count arguments, look for `kwarg_only` tensors, look for
    `alias_info.is_write` -- needs the schema to have really been read. A stub
    with an empty argument list would answer "no mutable arguments" to every
    question, which is a wrong answer rather than a missing one.

    The grammar handled is torch's own:
    `ns::name.overload(Type(alias) name=default, *, ...) -> (R1, R2)`.
    """

    __slots__ = ("name", "overload_name", "arguments", "returns", "_source",
                 "_placeholder")

    def __init__(self, qualname: str, overload: str, source: str = "",
                 placeholder: bool = False) -> None:
        self.name = qualname
        self.overload_name = overload
        self.arguments = []
        self.returns = []
        self._source = source
        self._placeholder = placeholder

    @property
    def is_placeholder(self) -> bool:
        """Whether this schema is a stand-in with no text behind it.

        Not upstream's -- upstream has no such thing, because upstream always
        has the text. It is here because this shim sometimes does not, and the
        alternative to saying so is an object that answers every question with
        the answer an empty argument list implies. Read
        `_C._shim_placeholder_schemas()` for the ones handed out so far.
        """
        return self._placeholder

    def _spelling(self) -> str:
        suffix = f".{self.overload_name}" if self.overload_name else ""
        return f"{self.name}{suffix}"

    def _answer_without_text(self, question: str, answer):
        """Record a predicate answered from an empty schema, and answer it.

        Raising here was tried and is wrong, and the measurement is why. With
        the refusal in place, a full run (import, the transformers road, FSDP,
        the decomposition pass) hits it for 102 distinct `(op, predicate)`
        pairs -- and **84 of those ops do not exist upstream at all.** They are
        names the tree *synthesises* and probes: `torch/distributed/tensor/_ops/
        autogen.py` builds `<base>_` and `<base>_functional`, and
        `torch/_ops.py` asks every packet for a `default` overload that
        `aten::add` and `aten::mul` (which are `add.Tensor`/`add.Scalar`
        upstream) do not have. Upstream answers all 84 with AttributeError at
        the packet lookup, and every caller's guard for that -- `packet is
        None`, `except AttributeError` -- reaches the same branch as
        `is_mutable == False`. Refusing turns a question upstream answers into
        an import failure.

        Of the 18 that do exist upstream, 17 are `is_mutable == False` and one
        is not (`aten::native_dropout_backward.out`). All 18 are transcribed in
        `_GENERATED_ATEN_SCHEMA_TEXT`, so no op upstream has is answered from an
        empty schema -- which is the claim `verify_schemas.py --unanswered`
        re-checks against a real torch.

        What is left is therefore a lie only about ops that do not exist, and
        it is not a silent one: every pair that gets here is listed by
        `_C._shim_unanswered_predicates()`, so the set can be diffed rather
        than rediscovered.
        """
        _UNANSWERED_PREDICATES.add((self._spelling(), question))
        return answer

    @classmethod
    def parse(cls, text: str) -> "_Schema":
        text = text.strip()
        open_paren = text.index("(")
        head = text[:open_paren].strip()
        # The *matching* close paren, found by depth. `rindex(")")` looked
        # right and was wrong: in
        # `aten::add_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)`
        # the last `)` belongs to the return type's alias annotation, so the
        # argument list swallowed the arrow and produced two arguments, the
        # second of them named `Tensor(a!`.
        depth = 0
        close_paren = -1
        for i in range(open_paren, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    close_paren = i
                    break
        if close_paren < 0:
            raise RuntimeError(f"torch._C shim: unbalanced schema: {text}")
        body = text[open_paren + 1 : close_paren]
        tail = text[close_paren + 1 :].strip()

        namespace, _, rest = head.rpartition("::")
        name, _, overload = rest.partition(".")
        qualname = f"{namespace}::{name}" if namespace else name

        schema = cls(qualname, overload, text)
        kwarg_only = False
        for chunk in _split_top_level(body):
            if chunk == "*":
                kwarg_only = True
                continue
            schema.arguments.append(_parse_argument(chunk, kwarg_only))

        if tail.startswith("->"):
            returns = tail[2:].strip()
            if returns.startswith("(") and returns.endswith(")"):
                returns = returns[1:-1]
            for chunk in _split_top_level(returns):
                schema.returns.append(_parse_argument(chunk, False))
        return schema

    @property
    def is_mutable(self) -> bool:
        """A *property*, because that is what upstream's is.

        `torch/_library/utils.py:104` reads `if schema.is_mutable:` and fifteen
        more sites in the vendored tree read `._schema.is_mutable` the same
        way. This was a method, and a bound method is truthy -- so every schema
        answered "mutable", `is_functional_schema` was False everywhere, and
        `torch.library.register_autograd` refused every op it was given
        (`torch/distributed/_functional_collectives.py:637` is where that
        surfaced). The always-true predicate, again.

        It hid because the only test covering it used the shim's own spelling,
        `is_mutable()`, which reads correctly whichever it is. `_is_view_op`
        below stays a method: `torch/distributed/tensor/_dispatch.py:569` calls
        it with parentheses.

        Being a property was only half of it. The value stayed constant -- now
        always *False* -- because the argument list it reads was empty for every
        aten op: `_get_schema` handed out a placeholder and `any([])` is False
        (docs/distributed/DISTRIBUTED.md §8.1). The schemas are real now, so the `any` below
        reads something. The placeholder branch is written out rather than left
        to `any([])` because it is a *different* statement -- "answered from no
        text" rather than "read the arguments and found no writer" -- and
        `_answer_without_text` is what keeps the two countable apart.
        """
        if self._placeholder:
            return self._answer_without_text("is_mutable", False)
        return any(
            a.alias_info is not None and a.alias_info.is_write for a in self.arguments
        )

    def _is_view_op(self) -> bool:
        """A view op aliases an argument into a return without writing to it.

        `torch/_library/custom_ops.py:794` asks this during
        `@torch.library.custom_op` registration, which runs at import.
        Replicated from `MathBitsFallback.h`, the same rule `OpOverload`
        applies to its own schema (`torch/_ops.py:838`).

        Counted separately on a placeholder for the same reason `is_mutable` is:
        with no arguments the `if not writes` below is taken and the answer is
        False for every op, which is a claim rather than an absence.
        """
        if self._placeholder:
            return self._answer_without_text("_is_view_op()", False)
        writes = [
            a.alias_info.is_write for a in self.arguments if a.alias_info is not None
        ]
        if not writes:
            return False
        return not any(writes) and any(r.alias_info is not None for r in self.returns)

    def __str__(self) -> str:
        if self._source:
            return self._source
        return f"{self._spelling()}(...) -> ..."

    def __repr__(self) -> str:
        return str(self)


def _split_default(chunk: str):
    """`Tensor(a!) self=[0, 0]` -> `("Tensor(a!) self", "[0, 0]")`.

    The `=` that separates a default is the one at depth zero and outside a
    quoted run. `str a='='` is not in aten today, but the scanner costs the
    same either way and getting it wrong would silently truncate a type.
    """
    depth, quote, escaped = 0, "", False
    for i, ch in enumerate(chunk):
        if escaped:
            escaped = False
            continue
        if quote:
            if ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "=" and depth == 0:
            return chunk[:i].strip(), chunk[i + 1 :].strip()
    return chunk.strip(), None


def _parse_argument(chunk: str, kwarg_only: bool) -> _Argument:
    chunk, default = _split_default(chunk)

    spelling, _, name = chunk.rpartition(" ")
    if not spelling:
        spelling, name = chunk, ""

    alias_info = None
    if "(" in spelling and spelling.endswith(")"):
        inner = spelling[spelling.index("(") + 1 : -1]
        # `Tensor(a!)` mutates; `Tensor(a)` only aliases.
        alias_info = _AliasInfo("!" in inner, inner.replace("!", "").split("|"))
    return _Argument(name.strip(), _SchemaType(spelling.strip()), kwarg_only, default,
                     alias_info)


# ---------------------------------------------------------------------------
# Re-printing a `native_functions.yaml` entry the way upstream prints a schema
# ---------------------------------------------------------------------------
#
# The vendored tree carries `torchgen/packaged/ATen/native/native_functions.yaml`
# -- 2584 `- func:` lines, each an aten schema, shipped in the wheel as a data
# file (pyproject.toml) and already read at runtime for the Core ATen tag set
# (`torchnative/export/decompose.py`, docs/graph/DECOMP.md §2). It is the source of
# the schema text: it is upstream's own file rather than a transcription, and it
# needs no upstream torch, which a wheel does not have.
#
# It is not *quite* what upstream prints, though. `str(FunctionSchema)` is a C++
# printer working from parsed IValues, and it renormalises the defaults on the
# way out. Measured over all 2584 entries against torch 2.13.0's
# `_jit_get_all_schemas()`, 165 differ, in exactly five ways -- and after the
# five rules below the residual is 0/2584. Each rule states what it reproduces:
#
#   1. `DeviceIndex` is spelled `int` in the printed schema.
#   2. `float`-typed defaults go through C++'s double printer (rule 4), always;
#      `Scalar`-typed ones only when the literal is written as a float, since a
#      `Scalar` default of `1` is an int IValue and prints as `1`.
#   3. String defaults print double-quoted with `'`, `"` and `\` escaped.
#   4. Sized-list defaults broadcast: `SymInt[2] stride=1` -> `[1, 1]`. `int[N]`
#      is the exception and it is upstream's own: its printer re-collapses a
#      uniform `int` list of length > 1 back to the scalar, with the comment
#      "we want to faithfully replicate the schema string". So `int[2]
#      padding=0` stays `0` while `int[1] padding=0` (length 1, so the collapse
#      does not apply) prints `[0]`. Both spellings occur; 101 arguments across
#      the file split along exactly that line.
#   5. Enum-valued defaults print as their integer.
#
# What this is not: it is not a schema *parser* for the tree to use. The parsing
# is `_Schema.parse`'s, unchanged. This only fixes the spelling of defaults so
# that `str(op._schema)` is upstream's string, which is what `verify_schemas.py`
# now diffs.


def _print_double(value: float) -> str:
    """C++'s `operator<<(ostream&, double)` as `torch::jit` configures it.

    Two branches, both visible in aten's schemas: a finite value below 1e10
    that is integral prints as the integer plus a bare `.` (`1.`, `0.`, and
    `-0.` for negative zero), and everything else prints at
    `max_digits10 == 17` significant digits -- which is why upstream spells
    `1/3` as `0.33333333333333331` where the yaml writes `0.3333333333333333`.
    """
    if math.isfinite(value) and abs(value) < 1e10:
        whole = int(value)
        if float(whole) == value:
            negative_zero = value == 0.0 and math.copysign(1.0, value) < 0
            return f"{whole}{'-.' if negative_zero else '.'}"
    return f"{value:.17g}"


#: `printQuotedString` in `function_schema.cpp`. Both quote characters are
#: escaped even though the output is double-quoted, which is why upstream's
#: `_test_string_default` reads `str a="\"\'\\"`.
_SCHEMA_STRING_ESCAPES = {
    "\\": "\\\\", "'": "\\'", '"': '\\"',
    "\a": "\\a", "\b": "\\b", "\f": "\\f",
    "\n": "\\n", "\r": "\\r", "\t": "\\t", "\v": "\\v",
}

#: Defaults written as an enumerator in the yaml and as its integer in the
#: printed schema. These three are the whole list -- the file's only non-`None`,
#: non-boolean word-shaped defaults are `Mean` (33 uses), `long` (8) and
#: `contiguous_format` (4). The values are the C++ enumerator values:
#: `at::Reduction::Mean == 1` (None/Mean/Sum), `c10::ScalarType::Long == 4`
#: (Byte, Char, Short, Int, Long -- the order `torchgen/model.py`'s `ScalarType`
#: declares), and `c10::MemoryFormat::Contiguous == 0`.
_SCHEMA_ENUM_DEFAULTS = {"Mean": "1", "long": "4", "contiguous_format": "0"}

_SIZED_LIST_TYPE = re.compile(r"^(.*)\[(\d*)\]$")


def _unquote_schema_string(literal: str) -> str:
    out, escaped = [], False
    for ch in literal[1:-1]:
        if escaped:
            out.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        else:
            out.append(ch)
    return "".join(out)


def _quote_schema_string(text: str) -> str:
    return '"' + "".join(_SCHEMA_STRING_ESCAPES.get(c, c) for c in text) + '"'


def _element_type_of(spelling: str):
    """`(element type, list length or None, is a list)` for a type spelling."""
    base = re.sub(r"\([^)]*\)", "", spelling).strip()
    if base.endswith("?"):
        base = base[:-1].strip()
    match = _SIZED_LIST_TYPE.match(base)
    if match is None:
        return base, None, False
    size = match.group(2)
    return match.group(1), (int(size) if size else None), True


#: A `Scalar` default written as a float. `Scalar alpha=1` is an int IValue and
#: prints back as `1`; `Scalar alpha=1.0` is a double and prints as `1.`. So the
#: test is on the spelling, not on whether `float()` accepts it.
_FLOAT_LITERAL = re.compile(
    r"^[-+]?((\d+\.\d*|\.\d+)([eE][-+]?\d+)?|\d+[eE][-+]?\d+)$"
)


def _normalise_scalar_default(element: str, literal: str) -> str:
    if literal in _SCHEMA_ENUM_DEFAULTS:
        return _SCHEMA_ENUM_DEFAULTS[literal]
    if len(literal) >= 2 and literal[0] in "\"'" and literal[-1] == literal[0]:
        return _quote_schema_string(_unquote_schema_string(literal))
    if element == "float":
        try:
            return _print_double(float(literal))
        except ValueError:
            return literal
    if element == "Scalar" and _FLOAT_LITERAL.match(literal):
        return _print_double(float(literal))
    return literal


def _normalise_default(spelling: str, literal: str) -> str:
    element, size, is_list = _element_type_of(spelling)
    if not is_list:
        return _normalise_scalar_default(element, literal)
    if literal == "None":
        return literal
    if literal.startswith("[") and literal.endswith("]"):
        items = [
            _normalise_scalar_default(element, item)
            for item in _split_top_level(literal[1:-1])
        ]
    elif size:
        items = [_normalise_scalar_default(element, literal)] * size
    else:
        return _normalise_scalar_default(element, literal)
    if element == "int" and len(items) > 1 and len(set(items)) == 1:
        return items[0]
    return "[" + ", ".join(items) + "]"


def _normalise_argument(chunk: str) -> str:
    chunk, default = _split_default(chunk)
    spelling, _, name = chunk.rpartition(" ")
    if not spelling:
        spelling, name = chunk, ""
    spelling = spelling.replace("DeviceIndex", "int")
    head = f"{spelling} {name}".strip()
    if default is None:
        return head
    return f"{head}={_normalise_default(spelling, default)}"


def _normalise_schema_text(text: str) -> str:
    """A `native_functions.yaml` entry, spelled the way upstream prints it."""
    open_paren = text.index("(")
    depth, close_paren = 0, -1
    for i in range(open_paren, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                close_paren = i
                break
    if close_paren < 0:
        raise RuntimeError(f"torch._C shim: unbalanced schema: {text}")
    head = text[:open_paren]
    body = text[open_paren + 1 : close_paren]
    tail = text[close_paren + 1 :].replace("DeviceIndex", "int")
    arguments = [
        chunk if chunk == "*" else _normalise_argument(chunk)
        for chunk in _split_top_level(body)
    ]
    return f"{head}({', '.join(arguments)}){tail}"


# ---------------------------------------------------------------------------
# Where the aten schema text comes from
# ---------------------------------------------------------------------------

_NATIVE_FUNCTIONS_RELPATH = os.path.join(
    "torchgen", "packaged", "ATen", "native", "native_functions.yaml"
)

#: `[answer, roots it was derived from]`. The roots are kept so that a *failed*
#: lookup is retried once the tree it was looking for arrives -- `_get_schema`
#: can be reached from `torch/_ops.py` while `import torch` is still running,
#: and caching "not found" from that moment would make it permanent. Read
#: through `_C._shim_schema_source()`.
_SCHEMA_SOURCE_CELL: list = []


def _native_functions_roots() -> list:
    roots = []
    for name in ("torch", "torch._C", "_C"):
        candidate = getattr(sys.modules.get(name), "__file__", None)
        if candidate:
            root = os.path.dirname(os.path.dirname(os.path.abspath(candidate)))
            if root not in roots:
                roots.append(root)
    roots.extend(path for path in sys.path if path)
    return roots


def _native_functions_source() -> str:
    """The vendored `native_functions.yaml`, or why there is not one.

    Located relative to `torch.__file__` first, as
    `torchnative/export/decompose.py` does and for its reason: `torch/` and
    `torchgen/` are siblings both in an installed wheel and in the source tree,
    and if two trees are on the path the answer has to come from the one whose
    ops are being asked about. `_C.__file__` is the fallback for the same
    layout seen from inside the extension, and a `sys.path` sweep is the last
    resort.

    Never raises. A build without the data file is a build whose schemas are
    placeholders, and a placeholder says so when asked a question it cannot
    answer -- that is a better failure than `import torch` stopping here.
    """
    roots = _native_functions_roots()
    if _SCHEMA_SOURCE_CELL and (os.path.isabs(_SCHEMA_SOURCE_CELL[0])
                                or _SCHEMA_SOURCE_CELL[1] == roots):
        return _SCHEMA_SOURCE_CELL[0]
    found = None
    for root in roots:
        path = os.path.join(root, _NATIVE_FUNCTIONS_RELPATH)
        if os.path.isfile(path):
            found = path
            break
    answer = found or (
        "no native_functions.yaml on this path: looked for "
        f"{_NATIVE_FUNCTIONS_RELPATH} beside torch/ and along sys.path. Every "
        "aten schema is a placeholder in this process (docs/bindings/SCHEMA.md)"
    )
    del _SCHEMA_SOURCE_CELL[:]
    _SCHEMA_SOURCE_CELL.extend((answer, roots))
    return answer


#: `(qualname, overload_name)` -> the raw `- func:` text, built on first use.
_ATEN_SCHEMA_INDEX: dict = {}
#: The same key -> the parsed `_Schema`, built one op at a time.
_ATEN_SCHEMA_CACHE: dict = {}


def _aten_schema_index() -> dict:
    """A line scan of the `- func:` entries, not a YAML parse.

    Same constraint as `decompose._scan_core_tags`: `pyyaml` is not a declared
    dependency of this distribution, so importing `yaml` here would make a
    correctly-installed wheel fail at `import torch`. The format relied on is
    one line per entry, beginning at column 0 with `- func:` -- true for all
    2584 entries of 2.13.0's file, and `verify_schemas.py` diffs the result
    against upstream's registry so a format change is a failure rather than a
    quietly shorter table.

    Indexing is one pass over 15789 lines and does no parsing; the normalise +
    parse happens per op in `_aten_schema`, so a process that asks for twelve
    schemas pays for twelve.
    """
    if _ATEN_SCHEMA_INDEX:
        return _ATEN_SCHEMA_INDEX
    source = _native_functions_source()
    if not os.path.isabs(source or ""):
        return _ATEN_SCHEMA_INDEX
    try:
        with open(source, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:  # a data file that is present but unreadable
        return _ATEN_SCHEMA_INDEX
    for line in lines:
        if not line.startswith("- func:"):
            continue
        text = line[len("- func:") :].strip()
        head = text.split("(", 1)[0]
        name, _, overload = head.partition(".")
        _ATEN_SCHEMA_INDEX[(f"aten::{name}", overload)] = f"aten::{text}"
        _ATEN_SCHEMA_NAMES.add(name)
        # File order, not sorted: `OpOverloadPacket.overloads()` is a list and
        # upstream's is registration order, which for aten is the order of this
        # file. Nothing here depends on the order, and reproducing it costs
        # nothing.
        _ATEN_OVERLOADS_BY_NAME.setdefault(f"aten::{name}", []).append(overload)
    return _ATEN_SCHEMA_INDEX


#: Every aten op *name* the file declares, ignoring overloads. Built with the
#: index, and empty when there is no file.
_ATEN_SCHEMA_NAMES: set = set()

#: `aten::<name>` -> the overload names the file declares for it, `""` for the
#: unnamed one (upstream's spelling; `OpOverloadPacket.overloads()` turns it
#: into `"default"`). Built with the index, and the reason it exists is
#: docs/graph/DECOMP.md §3: `_jit_get_operation` used to answer `["default"]` for
#: every packet, and `aten::transpose` has no `default` overload.
_ATEN_OVERLOADS_BY_NAME: dict = {}


def _aten_overload_names(qualname: str) -> list:
    """What overloads `native_functions.yaml` declares for one packet.

    Empty when the file does not declare the name at all -- which is a real
    outcome and not an error: 176 of upstream's 1730 aten names are absent from
    it (`_is_absent_inplace_variant` measures the same gap from the other
    side), and every non-aten namespace is absent by construction.
    """
    _aten_schema_index()
    return list(_ATEN_OVERLOADS_BY_NAME.get(qualname, ()))


#: `(qualname, overload)` -> the tag *names* the file writes on that entry.
_ATEN_TAGS: dict = {}
#: Tag names the file used that `_C.Tag` has no member for. Read through
#: `_C._shim_unknown_tags()`; an empty list is the claim that nothing was
#: dropped, and a non-empty one names what was.
_UNKNOWN_TAGS: set = set()


def _scan_aten_tags() -> dict:
    """The `tags:` line of every `- func:` entry.

    `torchnative/export/decompose.py:_scan_core_tags` reads the same line for
    `core` alone and documents the format: a flow scalar or a one-line flow
    list, two spaces in, with zero block sequences across all 2584 entries of
    2.13.0's file. This reads all of them rather than one, because the tag
    upstream's export path asks about first is not `core` --
    `torch/_decomp/__init__.py:57` reads `maybe_aliasing_or_mutating` to decide
    whether an op may be preserved, and `_get_operation_overload` answering
    `[]` made that question always false.

    **Three tags are not on the line and are not optional.**
    `torchgen/model.py:756-765` adds them while parsing, and an entry's real
    tag set is the written ones plus these:

        pt2_compliant_tag   every aten entry, unconditionally
        out                 entries with an `out` argument
        inplace             entries whose name ends in `_`

    Measured: without them 0 of 118 implemented ops match upstream's
    `OpOverload.tags`, because `pt2_compliant_tag` alone is on all of them.
    """
    if _ATEN_TAGS:
        return _ATEN_TAGS
    source = _native_functions_source()
    if not os.path.isabs(source or ""):
        return _ATEN_TAGS
    try:
        with open(source, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return _ATEN_TAGS
    key = None
    for line in lines:
        if line.startswith("- func:"):
            signature = line[len("- func:"):].strip()
            head = signature.split("(", 1)[0]
            name, _, overload = head.partition(".")
            key = (f"aten::{name}", overload)
            # torchgen/model.py:756-765, in its order.
            implicit = ["pt2_compliant_tag"]
            if _has_out_argument(signature):
                implicit.append("out")
            if _is_inplace_name(name):
                implicit.append("inplace")
            _ATEN_TAGS[key] = implicit
        elif key is not None and line.startswith("  tags:"):
            text = line[len("  tags:"):].split("#", 1)[0].strip().strip("[]")
            written = [tag.strip() for tag in text.split(",") if tag.strip()]
            _ATEN_TAGS[key] = written + _ATEN_TAGS[key]
        elif line.startswith("#"):
            # A comment at column 0 does not end the entry -- YAML does not see
            # it at all. Treating it as a terminator is not hypothetical: the
            # file puts a two-line comment between `rsub.Scalar` and its own
            # `tags: pointwise`, and between the same pair for `sinh.out` and
            # `tanh_backward.grad_input`. Those three were exactly the residual
            # against upstream's `OpOverload.tags` before this line existed.
            continue
        elif line and not line.startswith(" "):
            key = None
    return _ATEN_TAGS


#: `torchgen/model.py:2663`. The dunder in-place spellings are `__i<name>__`
#: for exactly these, and nothing else -- `__int__` would be a false positive
#: for a rule that only looked for a leading `i`, which is why torchgen keeps
#: the list rather than the prefix.
_AUGMENTED_ASSIGNMENT_NAMES = frozenset(
    {"add", "sub", "mul", "div", "mod", "pow", "lshift", "rshift", "and",
     "xor", "or"}
)


def _is_inplace_name(name: str) -> bool:
    """`BaseOperatorName.parse(...).inplace` (`torchgen/model.py:2716`).

    Two shapes, and the second is why this is not `name.endswith("_")`:
    `aten::__iand__` is in place and does not end in a single underscore. Ten
    of the file's entries are that shape.
    """
    match = re.match(r"^__([^_]+)__$", name)
    if match is not None:
        inner = match.group(1)
        return inner.startswith("i") and inner[1:] in _AUGMENTED_ASSIGNMENT_NAMES
    return name.endswith("_")


def _has_out_argument(signature: str) -> bool:
    """`FunctionSchema.is_out_fn()` (`torchgen/model.py:1688`).

    Upstream's definition is quoted there and is *not* "an argument called
    `out`": it is **any keyword-only argument that is mutable**. The names
    differ -- `binary_cross_entropy_backward.grad_input` is an out function
    whose out argument is called `grad_input` -- so a rule keyed on the name
    misses those.

    Read off the text rather than off a parsed `_Schema`, because this runs
    while the index is being built and parsing 2584 schemas to answer one
    question about each is the cost `_aten_schema_index` exists to avoid. The
    close paren is found by depth, not by `rfind`: the return type of an out
    function ends in `-> Tensor(a!)`, so the last `)` on the line is not the
    one that closes the argument list.
    """
    open_paren = signature.find("(")
    if open_paren < 0:
        return False
    depth, close_paren = 0, -1
    for index in range(open_paren, len(signature)):
        if signature[index] == "(":
            depth += 1
        elif signature[index] == ")":
            depth -= 1
            if depth == 0:
                close_paren = index
                break
    if close_paren < 0:
        return False
    seen_star = False
    for chunk in _split_top_level(signature[open_paren + 1 : close_paren]):
        chunk = chunk.strip()
        if chunk == "*":
            seen_star = True
            continue
        if seen_star and "!)" in chunk.split("=", 1)[0]:
            return True
    return False


def _aten_tags(module, qualname: str, overload: str) -> list:
    """`OpOverload.tags`, as `_C.Tag` members, from the file.

    This answered `[]` for every op and docs/graph/DECOMP.md §2 measured what that
    cost: `torch.Tag.core in op.tags` was False for all 120 implemented ops, so
    a classifier built on it would call nothing Core ATen and refuse whole
    programs. `decompose.core_ops()` went around it by reading the same file
    directly, which was right for that module and did nothing for the tree's
    own readers -- `_should_decompose_because_unsafe_op` among them.

    A tag name `_C.Tag` has no member for is dropped and recorded rather than
    raised on: `Tag`'s members come from the vendored `.pyi`, so a mismatch
    means the two halves of one upstream release disagree, and stopping
    `import torch` over it would be out of proportion to a tag nobody reads.
    `_C._shim_unknown_tags()` is how the drop is countable instead of silent.
    """
    tag_type = getattr(module, "Tag", None)
    if tag_type is None:
        return []
    key = (qualname, "" if overload == "default" else overload)
    tags = []
    for name in _scan_aten_tags().get(key, ()):
        member = getattr(tag_type, name, None)
        if member is None:
            _UNKNOWN_TAGS.add(name)
            continue
        tags.append(member)
    return tags


#: `<dispatch key>` -> `["aten::<name>[.<overload>]", ...]`, built on first use.
_DISPATCH_REGISTRATIONS: dict = {}

#: The keys `native_functions.yaml` is authoritative about. These four are
#: *alias* keys: they say how an op is put together, not which backend runs it,
#: so the file declaring them is a fact about the operator and not a claim
#: about this build. Backend keys (`CPU`, `CUDA`, `MPS`, ...) are the opposite
#: -- the file lists what upstream's C++ build registers, and answering with
#: that here would claim 1500 kernels this shim does not have. Those are
#: refused by name; see `_dispatch_registrations`.
_FILE_DECLARED_DISPATCH_KEYS = (
    "CompositeImplicitAutograd",
    "CompositeImplicitAutogradNestedTensor",
    "CompositeExplicitAutograd",
    "CompositeExplicitAutogradNonFunctional",
)


def _scan_dispatch_registrations() -> dict:
    """Which ops the file declares under each alias dispatch key.

    A line scan, for `_aten_schema_index`'s reason (`pyyaml` is not a declared
    dependency of this distribution). The format relied on is one more level
    deep than the schema scan: entries begin at column 0 with `- func:`, their
    properties are indented two spaces, and a `dispatch:` block's keys are
    indented four. A key line may name several keys at once
    (`CPU, CUDA: foo`).

    **`CompositeImplicitAutograd` is not only what the file writes down.**
    `torchgen/model.py:872` gives it to every entry that has no `dispatch:`
    block at all and is neither `structured` nor a `structured_delegate` --
    that default is where most CIA ops come from, and a scan that only looked
    for the literal word would find a third of them.

    Measured against upstream 2.13.0's dispatcher: this yields **743** aten
    names and `_dispatch_get_registrations_for_dispatch_key` answers **744**,
    with **zero** names on this side that are not on upstream's. The one
    upstream has and the file does not is `aten::get_gradients`, a TorchScript
    builtin -- the same 176-name gap `_is_absent_inplace_variant` documents,
    seen from another side.
    """
    if _DISPATCH_REGISTRATIONS:
        return _DISPATCH_REGISTRATIONS
    for key in _FILE_DECLARED_DISPATCH_KEYS:
        _DISPATCH_REGISTRATIONS[key] = []
    source = _native_functions_source()
    if not os.path.isabs(source or ""):
        return _DISPATCH_REGISTRATIONS
    try:
        with open(source, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return _DISPATCH_REGISTRATIONS

    state = {"name": None, "keys": set(), "dispatch": False,
             "structured": False, "delegate": False, "in_block": False}

    def flush():
        name = state["name"]
        if name is None:
            return
        keys = set(state["keys"])
        if not state["dispatch"] and not state["structured"] and not state["delegate"]:
            keys.add("CompositeImplicitAutograd")
        for key in keys:
            if key in _DISPATCH_REGISTRATIONS:
                _DISPATCH_REGISTRATIONS[key].append(f"aten::{name}")

    for line in lines:
        if line.startswith("- func:"):
            flush()
            state.update(
                name=line[len("- func:"):].strip().split("(", 1)[0],
                keys=set(), dispatch=False, structured=False, delegate=False,
                in_block=False,
            )
            continue
        if state["name"] is None:
            continue
        if line.startswith("#"):
            continue  # a column-0 comment does not end an entry -- see
            # `_scan_aten_tags`, where treating it as one lost three tags
        if line and not line.startswith(" "):
            flush()
            state["name"] = None
            continue
        stripped = line.strip()
        if line.startswith("  ") and not line.startswith("   "):
            state["in_block"] = False
            property_name = stripped.split(":", 1)[0]
            if property_name == "dispatch":
                state["dispatch"] = True
                state["in_block"] = True
            elif property_name == "structured":
                state["structured"] = stripped.split(":", 1)[1].strip() == "True"
            elif property_name == "structured_delegate":
                state["delegate"] = True
            continue
        if state["in_block"] and stripped and not stripped.startswith("#"):
            for key in stripped.split(":", 1)[0].split(","):
                state["keys"].add(key.strip())
    flush()
    return _DISPATCH_REGISTRATIONS


def _dispatch_registrations(key: str) -> list:
    """`torch._C._dispatch_get_registrations_for_dispatch_key`, from the file.

    docs/graph/DECOMP.md §3 named this function as what costs the decomposition table
    its CompositeImplicitAutograd half: `CustomDecompTable.__init__` enumerates
    every CIA registration through it, so `core_aten_decompositions()` raised
    here and the pass fell back to the post-autograd table.

    The list this returns is a statement about *upstream's operators*, not
    about this build's kernels, and the caller uses it that way:
    `torch/_export/utils.py:1298 _materialize_cpp_cia_ops` only walks the names
    to force `torch.ops.aten.<name>.<overload>` into existence, and the filter
    that decides whether an op is really CIA *here* is `_is_cia_op`, which asks
    `op.py_kernels` -- a Python-side registration this tree does make. So a
    generous materialisation list cannot smuggle anything in; it can only fail
    to mention something.

    Backend keys are refused rather than answered. See
    `_FILE_DECLARED_DISPATCH_KEYS`.
    """
    name = getattr(key, "name", key)
    registrations = _scan_dispatch_registrations()
    if name not in registrations:
        raise NotImplementedError(
            f"not implemented in torch._C shim: "
            f"_dispatch_get_registrations_for_dispatch_key({name!r}). "
            f"native_functions.yaml is authoritative for the alias keys "
            f"{', '.join(_FILE_DECLARED_DISPATCH_KEYS)}, and answering for a "
            f"backend key from the same file would claim kernels this build "
            f"does not have"
        )
    return list(registrations[name])


def _is_absent_inplace_variant(qualname: str) -> bool:
    """`aten::<base>_` where the file declares `<base>` and not `<base>_`.

    The registry is otherwise open on purpose: `_jit_get_operation` hands back a
    callable for any name and `_aten_dispatch` refuses at call time, which is
    the discovery mechanism DESIGN.md §6 asks for. That is fine for a name
    somebody typed and wrong for a name somebody *synthesised* --
    `torch/distributed/tensor/_ops/autogen.py:244` builds `<base>_` and asks the
    resulting packet `is_mutable`, to find out whether an in-place variant
    exists at all. Upstream answers with AttributeError (there is no
    `aten::convolution_`); this shim answered with an operator whose schema was
    empty, and the empty schema then answered the question. False before this
    work and a refusal after it -- neither is upstream's answer, and the
    difference is one level above the schema.

    The rule is exactly this shape and no wider, because
    `native_functions.yaml` is *not* a complete list of aten operators: 176 of
    upstream's 1730 aten names are absent from it (`quantized_lstm`, which
    `torch/__init__.py:2395` reads unconditionally at import, `zero`, `resize`,
    the TorchScript numeric builtins), and refusing on "absent from the file"
    stops `import torch` on line 2395. What the file *is* complete for is the
    in-place variant of an op it declares: `add_` sits beside `add`, `relu_`
    beside `relu`. Measured on 2.13.0 -- of the 1348 names of the form
    `<yaml base>_` that the file does not declare, upstream has zero, and of
    the upstream in-place names the file lacks, none has its base in the file.
    So this refuses 1348 names and none of them is an operator.
    """
    namespace, _, name = qualname.partition("::")
    if namespace != "aten" or not name.endswith("_") or name.endswith("__"):
        return False
    _aten_schema_index()
    if not _ATEN_SCHEMA_NAMES:
        return False  # no file, so nothing is known to be absent
    return name not in _ATEN_SCHEMA_NAMES and name[:-1] in _ATEN_SCHEMA_NAMES


def _aten_schema(qualname: str, overload: str):
    """The parsed schema for an aten op, or None if the file has no entry.

    None is a real outcome: upstream's registry has 3754 aten schemas and this
    file has 2584, the difference being the `.out`/functional variants
    `torchgen`'s `native_function_generation.py` synthesises at build time
    rather than writing down. Four of them (`embedding.out`, `empty_like.out`,
    `div.Scalar_out`, `div.Scalar_mode_out`) are in the transcribed tables and
    are answered from there; the rest are placeholders and say so.
    """
    # `""` and `"default"` name the same overload and both spellings arrive.
    # `torch/_ops.py:1245` converts one to the other on the way in
    # (`use_key = "" if key == "default" else key`) and
    # `torch/_library/effects.py:55` does not, so an index keyed on only one of
    # them answers half the callers with a placeholder -- which is how
    # `aten::_adaptive_avg_pool2d` came back empty from a file that has it.
    key = (qualname, "" if overload == "default" else overload)
    if key in _ATEN_SCHEMA_CACHE:
        return _ATEN_SCHEMA_CACHE[key]
    raw = _aten_schema_index().get(key)
    parsed = None
    if raw is not None:
        try:
            parsed = _Schema.parse(_normalise_schema_text(raw))
        except Exception:  # noqa: BLE001 -- an entry this cannot read is a
            # placeholder, which announces itself; raising here would stop
            # `import torch` on a schema nobody asked a question about.
            parsed = None
    _ATEN_SCHEMA_CACHE[key] = parsed
    return parsed


# aten really does have dunder-named operators, and the tree registers
# decompositions for all ten at import (`torch/_decomp/decompositions.py:6239`
# is `register_inplace(aten.__iand__, aten.__and__)`). Everything else spelled
# with dunders that reaches the op registry is Python asking a question about
# the *namespace object* -- `__origin__`, `__deepcopy__`, `__wrapped__` -- and
# must be refused so that `getattr(..., default)` and `copy` keep working.
# The list is the tree's own: `grep -o 'aten\.__[a-z_]*__'`.
ATEN_DUNDER_OPS = frozenset(
    {
        "__and__",
        "__iand__",
        "__ilshift__",
        "__ior__",
        "__irshift__",
        "__ixor__",
        "__lshift__",
        "__or__",
        "__rshift__",
        "__xor__",
    }
)


def _is_refused_op_name(name: str) -> bool:
    return (
        name.startswith("__")
        and name.endswith("__")
        and name not in ATEN_DUNDER_OPS
    )


class _RecordFunction:
    """The handle `profiler::_record_function_enter_new` returns.

    Upstream's is a `torch.ScriptObject` of TorchScript class
    `__torch__.torch.classes.profiler._RecordFunction`, wrapping a C++
    `at::RecordFunction` that the profiler's callbacks observe. **This build has
    no profiler that could observe one** — see `_PROFILER_MARKERS` — so this is
    an opaque token that carries the region's name and nothing else.

    It carries the name rather than being a bare sentinel so that a caller
    holding one can be debugged, and so `record_function.__exit__`'s
    `if record is None: raise AssertionError` distinguishes "never entered"
    from "entered".
    """

    __slots__ = ("name", "args")
    __module__ = "torch.classes.profiler"

    def __init__(self, name, args=None):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"ScriptObject <__torch__.torch.classes.profiler._RecordFunction {self.name!r}>"


# `torch.ops.<ns>.<op>.<overload>` entries that are answered in Python instead
# of at the one door in `aten.rs`.
#
# **There are two of them and they are both profiler markers**, which is the
# whole justification. `torch.optim` wraps every `step()` and every
# `zero_grad()` in `with torch.autograd.profiler.record_function(...)`, so these
# two names gate **every optimiser in `torch.optim`, SGD included** — and
# neither is arithmetic. `docs/training/AUTOGRAD.md` §7 measured that:
#
#     optimiser.zero_grad()   FAIL  profiler._record_function_enter_new.default
#
# A no-op is the honest answer here and it is checkable rather than asserted:
# nothing in this build can observe a record. Measured on this artefact —
#
#     torch.profiler.profile()          NotImplementedError: _supported_activities
#     torch.autograd.profiler.profile() NotImplementedError: _ExperimentalConfig.trace_only
#     torch.autograd._profiler_enabled()NotImplementedError: _profiler_enabled
#
# — so there is no profiler to start, no way to ask whether one is running, and
# therefore no callback that a marker could reach. That is the difference
# between this and a silent stub: a stub for `bernoulli_` would lose a draw
# somebody wanted, and a marker with no listener loses nothing that exists.
# The day a profiler lands here, these two are where it hooks in.
#
# They are answered above `_aten_dispatch` rather than inside it because they
# are not kernels: they take a `str`, return an object with no storage, and
# have no dtype, device or shape. Putting them in `aten.rs`'s table would make
# `_aten_implemented()` — which means "has a kernel *and* golden compares it
# against upstream" — list something golden cannot compare.
def _record_function_enter_new(name, args=None):
    return _RecordFunction(name, args)


def _record_function_exit(record):
    # Upstream returns `()`, i.e. None at the Python level. The handle is
    # accepted and dropped; `record_function.__exit__` ignores the result.
    return None


_PROFILER_MARKERS = {
    "profiler._record_function_enter_new.default": _record_function_enter_new,
    # `record_function.__exit__` calls the `._RecordFunction` overload **by
    # name**, not `.default` -- `.default` takes a `Tensor` handle and is the
    # older spelling. Both are answered, because `torch.jit` scripting takes
    # the other branch and reaches `.default`.
    "profiler._record_function_exit._RecordFunction": _record_function_exit,
    "profiler._record_function_exit.default": _record_function_exit,
}


def _op_callable(dispatch, qualname: str, overload: str):
    """The one door, wearing the shape `torch/_ops.py` expects.

    `aten::add` + `Tensor` becomes the dispatch key `aten.add.Tensor`, which is
    exactly what `_C._aten_dispatch` takes and exactly the spelling
    `_C._aten_implemented()` reports. Overload is part of the key on purpose
    (docs/design/TORCH_C.md §1): folding `add.Tensor` and `add.Scalar` together would
    make one implementation look like two.

    `_PROFILER_MARKERS` is consulted first, and it is two entries. See the note
    on that table for why those two do not belong behind the door.
    """
    namespace, _, name = qualname.partition("::")
    key = f"{namespace}.{name}.{overload or 'default'}"

    marker = _PROFILER_MARKERS.get(key)
    if marker is not None:
        marker._shim_aten_key = key
        return marker

    def op(*args, **kwargs):
        return dispatch(key, *args, **kwargs)

    op.__name__ = f"{name}.{overload or 'default'}"
    op.__qualname__ = op.__name__
    op._shim_aten_key = key
    return op


# ---------------------------------------------------------------------------
# Overload resolution -- `torch.<op>(...)` -> `aten.<op>.<overload>`
# ---------------------------------------------------------------------------
#
# `torch.full` is not an aten op. Upstream it is a C binding that reads the
# actual arguments, picks one of several native functions, and calls it; the
# aten overload name is a property of the *native function it picked*, not of
# the name the caller typed. So a shim that stops at `torch.ops.aten.<op>.<ov>`
# leaves the whole user-facing API dark, which is where this project was:
#
#     torch.ops.aten.full.default([2], True).dtype   ->  torch.bool
#     torch.full((2,), True)                         ->  NotImplementedError
#
# and the vendored tree plus transformers call the second spelling.
#
# What is reproduced here is `PythonArgParser::raw_parse`: a signature list per
# name, tried in order, first one that binds wins. That is deliberately the
# same algorithm rather than a lookup table of special cases, because the
# ordering *is* the semantics -- see the note in `overloads.json` about
# `arange.start` versus `arange.start_step`.
#
# The single door is untouched. Resolution decides *which key* to hand
# `_C._aten_dispatch`; it never computes, never reaches a kernel by another
# route, and an op it resolves to but that has no kernel raises the same
# `aten op not implemented in torch._C shim: <key>` as before. A name with no
# table entry keeps the old refusal. So the instrument gains resolution without
# gaining a second entrance.

_SCHEMA_BASE_TYPES = frozenset(
    {
        "Tensor",
        "Scalar",
        "SymInt",
        "int",
        "float",
        "bool",
        "str",
        "ScalarType",
        "Layout",
        "Device",
        "MemoryFormat",
        "Generator",
    }
)


@functools.lru_cache(maxsize=4096)
def _decompose_type(spelling: str):
    """`SymInt[]?` -> `("SymInt", True, True, 0)`, `Tensor(a!)` -> `("Tensor", False, False, None)`.

    Returns `(base, is_list, is_optional, list_size)`. The order of the strips
    matters: `?` binds outermost (`int[]?` is an optional list, not a list of
    optional ints), and an alias annotation is attached to the base.

    Memoised because it is a pure function of its string: the answer depends on
    nothing but the characters, there is no context in which `"int[1]?"` means
    two different things, and the returned tuple is immutable so a caller
    cannot corrupt the entry for the next one. The bound is there because
    `parse_schema` will happily accept text from outside the tables; the tables
    themselves contain a few hundred distinct spellings.

    `list_size` is the number inside the brackets -- `int[1]` gives `1`, a bare
    `int[]` gives `0`, and a non-list gives `None`. It is not decoration:
    `FunctionParameter::check` accepts a *bare int* wherever the declared list
    has a size, which is the whole reason `x.sum(0)` binds
    `sum.dim_IntList(Tensor self, int[1]? dim, ...)`. Dropping the number and
    demanding a real sequence would make that call report "no matching
    overload" -- a wrong answer in the shape of a right one.
    """
    text = spelling.strip()
    optional = text.endswith("?")
    if optional:
        text = text[:-1].strip()
    is_list = False
    list_size = None
    if text.endswith("]"):
        is_list = True
        open_bracket = text.rindex("[")
        inside = text[open_bracket + 1 : -1].strip()
        list_size = int(inside) if inside.isdigit() else 0
        text = text[:open_bracket].strip()
    # `Tensor(a!)`, `Tensor(a)` -- the alias annotation is `_AliasInfo`'s
    # business (it is what `is_mutable()` reads); for type checking it is not
    # part of the type.
    if text.endswith(")") and "(" in text:
        text = text[: text.index("(")].strip()
    return text, is_list, optional, list_size


class _TypeChecker:
    """"Does this Python value satisfy this schema type?"

    `FunctionParameter::check` in `python_arg_parser.cpp`, restricted to the
    spellings the table contains -- `install` refuses to start if the table
    grows one this does not know, so the restriction is enforced rather than
    assumed.

    Three of these rules are torch's and would be got wrong by intuition:

      * `bool` does **not** satisfy `int`. `bool` subclasses `int` in Python,
        but torch's `THPUtils_checkLong` excludes it explicitly, and it has to
        -- `torch.full((2,), True)` must reach `Scalar` as a bool so the result
        is `torch.bool` and not `int64`.
      * `int` **does** satisfy `float`, one way only.
      * a zero-dim tensor satisfies `Scalar`. torch's `DOUBLE`/`SCALAR` case
        falls through to accepting a 0-dim `THPVariable`, which is why
        `torch.pow(t, other_0d_tensor)` picks `pow.Tensor_Tensor` -- declared
        first -- rather than `pow.Tensor_Scalar`.
    """

    def __init__(self, module) -> None:
        self._module = module
        self._tensor = module.TensorBase
        self._dtype = module.dtype
        self._layout = getattr(module, "layout", None)
        self._memory_format = getattr(module, "memory_format", None)
        self._device = module.device
        self._generator = getattr(module, "Generator", None)

    def check(self, spelling, value) -> bool:
        base, is_list, optional, list_size = _decompose_type(str(spelling))
        if optional and value is None:
            return True
        if is_list:
            if isinstance(value, (list, tuple)):
                if base in ("int", "SymInt"):
                    # `_base`'s scalar-Tensor arm is deliberately NOT open for
                    # list ELEMENTS: that would accept a Tensor in every
                    # `SymInt[]` position table-wide, which is the scoping
                    # docs/bindings/ARGFORM.md §2 forbids and which
                    # `_coerce_symint_size_tensors` handles for the two
                    # spellings that were measured (docs/bindings/BIND5.md §1.2).
                    # Mirrors `predicate_for`'s `int_list_ok`, which is the
                    # arm that actually runs.
                    return all(
                        isinstance(item, int) and not isinstance(item, bool)
                        for item in value
                    )
                return all(self._base(base, item) for item in value)
            # torch: "if a size is specified (e.g. IntArrayRef[2]) we also
            # allow passing a single int". `x.sum(0)` is that rule.
            return (
                list_size is not None
                and list_size > 0
                and base in ("int", "SymInt")
                and isinstance(value, int)
                and not isinstance(value, bool)
            )
        return self._base(base, value)

    # -- the same rules, compiled once per schema argument ------------------
    #
    # `check` above re-derives everything on every call: it re-parses the
    # spelling, then walks `_base`'s twelve-way string chain. Both answers are
    # fixed the moment the schema is parsed, and the schema outlives the
    # process. `predicate_for` returns a closure that answers exactly what
    # `check` answers for the same argument, with the parsing and the chain
    # already done. `check` is kept, unchanged, as the readable statement of the
    # rule; `coerce`'s one rule is inlined at the call site in `_bind`, guarded
    # by `_ArgPlan.sized_int_list`, which is its precondition precomputed.
    #
    # `check` and `coerce` were also tried *fused* into a single
    # value-or-sentinel closure, which is one fewer call and one fewer attribute
    # load per bound argument. It measured within noise of this (view 1.84 vs
    # 1.80 us, transpose 1.52 vs 1.51) and it needed a second copy of the twelve
    # `_base` rules with a different return shape, so it was dropped: two
    # spellings of the zero-dim-tensor-satisfies-Scalar rule is a real hazard
    # bought with no measurable time. See docs/bindings/BIND.md.
    #
    # Built lazily, on the first call that needs it, for the same reason the
    # `_TypeChecker` itself is: `layout` and `memory_format` do not exist when
    # the tables are parsed. Once built they are fixed, which is already true
    # of the attributes `check` reads -- `__init__` snapshots them.

    def _base_predicate(self, base: str):
        """`self._base(base, ·)` with `base` already decided."""
        tensor = self._tensor
        if base == "Tensor":
            return lambda value: isinstance(value, tensor)
        if base == "Scalar":

            def scalar(value):
                if isinstance(value, (bool, int, float, complex)):
                    return True
                # A number that is not one of Python's own. `numpy` scalars
                # register with the `numbers` ABCs, so this reaches them
                # without importing numpy -- and `vits` needs it:
                # `modeling_vits.py:1379` multiplies a tensor by
                # `np.prod(self.config.upsample_rates)`, an `np.int64`.
                # Upstream binds it (measured: the call fires
                # `aten.mul.Tensor` and keeps `int64`); refusing it here makes
                # Python fall back to `np.int64.__rmul__`, which asks the
                # tensor for `__array__` and stops on `TensorBase.numpy`.
                if isinstance(value, _numbers.Number) and not isinstance(value, tensor):
                    return True
                return isinstance(value, tensor) and value.dim() == 0

            return scalar
        if base in ("int", "SymInt"):
            # `type(value) is int` implies both halves of the second test and
            # is true of nearly every value that reaches here, so it is tried
            # first; the rest decides everything else. `bool` is excluded
            # deliberately -- see the class docstring.
            #
            # The Tensor arm is upstream's `ParameterType::INT64` check and is
            # what `vilt` needs: `torch.multinomial(probs, num_samples)` with a
            # Tensor `num_samples` (docs/bindings/BIND5.md §7). It is NOT multinomial-
            # specific -- measured on `select`, `narrow`, `transpose`,
            # `unsqueeze`, `sum(dim=)` and `repeat_interleave` as well, which
            # is why it is here rather than in a per-op wrapper. `bool` is
            # ACCEPTED here on purpose and refused one layer later by
            # `_symint_from_tensor`, reproducing upstream's two classes.
            accepted = tuple(
                getattr(self._module, d) for d in _SIZE_LIST_TENSOR_DTYPES
            ) + (self._module.bool,)

            def int_ok(value):
                if type(value) is int:
                    return True
                if isinstance(value, int):
                    return not isinstance(value, bool)
                return (
                    isinstance(value, tensor)
                    and value.numel() == 1
                    and any(value.dtype == d for d in accepted)
                )

            return int_ok
        if base == "float":
            return lambda value: isinstance(value, (int, float)) and not isinstance(
                value, bool
            )
        if base == "bool":
            return lambda value: isinstance(value, bool)
        if base == "str":
            return lambda value: isinstance(value, str)
        if base == "ScalarType":
            dtype = self._dtype
            return lambda value: isinstance(value, dtype)
        if base == "Layout":
            layout = self._layout
            if layout is None:
                return lambda value: False
            return lambda value: isinstance(value, layout)
        if base == "MemoryFormat":
            memory_format = self._memory_format
            if memory_format is None:
                return lambda value: False
            return lambda value: isinstance(value, memory_format)
        if base == "Device":
            device = self._device
            return lambda value: isinstance(value, (device, str))
        if base == "Generator":
            generator = self._generator
            if generator is None:
                return lambda value: False
            return lambda value: isinstance(value, generator)
        raise RuntimeError(f"torch._C shim: unhandled schema type: {base}")

    def predicate_for(self, base: str, is_list: bool, optional: bool, sized_int_list: bool):
        """`lambda value: self.check(<that spelling>, value)`, precomputed."""
        base_ok = self._base_predicate(base)
        if is_list:
            # An explicit loop rather than `all(base_ok(item) for item in
            # value)`: the generator costs a frame per element plus one to stop,
            # and shape lists are the hottest thing this checks. Short-circuits
            # in the same place `all` does.
            #
            # The `return` at the bottom of each is torch's "if a size is
            # specified (e.g. IntArrayRef[2]) we also allow passing a single
            # int" -- `x.sum(0)` is that rule. `sized_int_list` is false unless
            # the base is `int`/`SymInt`, so the non-int loop below can only
            # ever answer False there.
            if base in ("int", "SymInt"):
                # `SymInt[]` is every shape argument -- `view`, `transpose`,
                # `expand`, `sum(dim=...)`. Worth not paying a call per element.
                # A single-element integral Tensor sitting **inside** the list
                # is upstream's `SymInt` rule applied per element, and it is
                # the same rule docs/bindings/BIND5.md §7 landed one position over for
                # a *scalar* `SymInt`. Measured on torch 2.13.0
                # (docs/kernels/REPEAT.md §4):
                #
                #     x.as_strided(size=(2, tensor(2)),   ...)  -> works
                #     x.as_strided(size=(2, tensor([2])), ...)  -> works
                #     x.as_strided(size=(2, tensor(2.0)), ...)  -> TypeError
                #     x.as_strided(size=(2, tensor(True)),...)  -> RuntimeError
                #     x.as_strided(size=(2, tensor([2,3])),...) -> TypeError
                #
                # `bool` is accepted *here* and refused by the coercion, so
                # upstream's two exception classes fall out rather than being
                # chosen -- exactly as `_symint_from_tensor` does for the
                # scalar position. This predicate is on the hot path for every
                # shape argument, so the Tensor test is reached only after the
                # int tests have already declined.
                def int_list_ok(value):
                    if isinstance(value, (list, tuple)):
                        for item in value:
                            if type(item) is int:
                                continue
                            if isinstance(item, int) and not isinstance(item, bool):
                                continue
                            if not _symint_tensor_element_ok(item):
                                return False
                        return True
                    return (
                        sized_int_list
                        and isinstance(value, int)
                        and not isinstance(value, bool)
                    )

                inner = int_list_ok
            else:

                def list_ok(value):
                    if isinstance(value, (list, tuple)):
                        for item in value:
                            if not base_ok(item):
                                return False
                        return True
                    return False

                inner = list_ok
        else:
            inner = base_ok
        if optional:
            return lambda value: value is None or inner(value)
        return inner

    @staticmethod
    def coerce(spelling, value):
        """The value to actually bind, once `check` has said yes.

        Two rules. A bare int that satisfied a sized int list is widened to a
        one-element tuple, so the kernel behind the key always sees a list
        where the schema says list -- torch does the same thing one layer down
        (`IntArrayRef` of length one). And a single-element Tensor that
        satisfied a scalar `int`/`SymInt` is unpacked, which is the layer that
        refuses `bool` (`_symint_from_tensor`).
        """
        base, is_list, _, list_size = _decompose_type(str(spelling))
        if (
            is_list
            and list_size
            and base in ("int", "SymInt")
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return (value,)
        if (
            not is_list
            and base in ("int", "SymInt")
            and not isinstance(value, int)
        ):
            return _symint_from_tensor(value)
        return value

    def _base(self, base: str, value) -> bool:
        if base == "Tensor":
            return isinstance(value, self._tensor)
        if base == "Scalar":
            if isinstance(value, (bool, int, float, complex)):
                return True
            # See `_base_predicate`'s `scalar` above: `numbers.Number` is what
            # reaches a `numpy` scalar without importing numpy.
            if isinstance(value, _numbers.Number) and not isinstance(value, self._tensor):
                return True
            return isinstance(value, self._tensor) and value.dim() == 0
        if base in ("int", "SymInt"):
            if isinstance(value, int):
                return not isinstance(value, bool)
            # See `_base_predicate`'s `int_ok`: one element, any ndim, an
            # integral dtype -- and `bool`, which `_symint_from_tensor`
            # refuses afterwards rather than this predicate.
            accepted = tuple(
                getattr(self._module, d) for d in _SIZE_LIST_TENSOR_DTYPES
            ) + (self._module.bool,)
            return (
                isinstance(value, self._tensor)
                and value.numel() == 1
                and any(value.dtype == d for d in accepted)
            )
        if base == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if base == "bool":
            return isinstance(value, bool)
        if base == "str":
            return isinstance(value, str)
        if base == "ScalarType":
            return isinstance(value, self._dtype)
        if base == "Layout":
            return self._layout is not None and isinstance(value, self._layout)
        if base == "MemoryFormat":
            return self._memory_format is not None and isinstance(
                value, self._memory_format
            )
        if base == "Device":
            # torch takes a bare string wherever a device is taken, and
            # `device_arg` in aten.rs already accepts one.
            return isinstance(value, (self._device, str))
        if base == "Generator":
            return self._generator is not None and isinstance(value, self._generator)
        raise RuntimeError(f"torch._C shim: unhandled schema type: {base}")


def _is_schema_default(value, default_source) -> bool:
    """Is this the value the schema would have used anyway?

    Arguments equal to their own default carry no information, and dropping
    them is what keeps `torch.ones(2, pin_memory=False)` from arriving at a
    kernel that refuses every non-`None` `pin_memory`. It also means a kernel
    only ever sees arguments a caller actually asked for.

    Comparison is restricted to literal defaults against scalar values on
    purpose: `==` against a `TensorBase` would go through a synthesised
    `__eq__` that raises.
    """
    if default_source is None:
        return False
    if default_source == "None":
        return value is None
    if default_source == "True":
        return value is True
    if default_source == "False":
        return value is False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    for parse in (int, float):
        try:
            return value == parse(default_source)
        except (TypeError, ValueError):
            continue
    return False


class _ArgPlan:
    """One schema argument with everything that is fixed already computed.

    `_bind` used to re-derive all of this on every call -- `_decompose_type`
    was the third-hottest function in a SmolLM2 forward pass at 35030 calls,
    re-parsing a few dozen distinct strings. The spelling of an argument's type
    is settled when the schema is parsed and never changes afterwards
    (`_Schema.parse` fills `arguments` once and nothing writes to them), so
    there is no context in which the same argument decomposes two ways.

    `predicate` is the exception: it needs a `_TypeChecker`, which cannot exist
    until `install` has synthesised `layout` and `memory_format`. It is filled
    in by `_SchemaPlan.arm` on the first call.
    """

    __slots__ = (
        "name",
        "base",
        "is_list",
        "optional",
        "sized_int_list",
        "scalar_int",
        "int_list",
        "default_source",
        "has_default",
        "predicate",
    )

    def __init__(self, argument) -> None:
        base, is_list, optional, list_size = _decompose_type(str(argument.type))
        self.name = argument.name
        self.base = base
        self.is_list = is_list
        self.optional = optional
        # `check`'s "a sized int list also takes a bare int" precondition and
        # `coerce`'s "widen it to a one-tuple" precondition are the same
        # predicate on the type; `list_size` is never None when `is_list`.
        self.sized_int_list = bool(is_list and list_size and base in ("int", "SymInt"))
        # `_symint_from_tensor`'s precondition, precomputed the same way
        # `sized_int_list` is: a SCALAR `int`/`SymInt` position, where the
        # predicate may now have said yes to a single-element Tensor.
        self.scalar_int = bool(not is_list and base in ("int", "SymInt"))
        # The list twin of `scalar_int`: a `SymInt[]`/`int[]` position whose
        # elements may each need `_symint_from_tensor`'s unpack.
        self.int_list = bool(is_list and base in ("int", "SymInt"))
        # `_Argument.has_default_value()` is exactly this test.
        self.default_source = argument.default_value
        self.has_default = argument.default_value is not None
        self.predicate = None


class _SchemaPlan:
    """One schema's binding invariants, computed once instead of per call.

    Everything here was a list comprehension, a dict comprehension, or a
    `_decompose_type` inside `_Overloads._bind`. None of it depends on the
    arguments of the call.
    """

    __slots__ = (
        "arguments",
        "positional",
        "by_name",
        "varargs_intlist",
        "required",
        "any_defaults",
        "n_arguments",
        "n_positional",
        "armed",
    )

    def __init__(self, schema, self_bound: bool) -> None:
        plans = [_ArgPlan(argument) for argument in schema.arguments]
        self.arguments = tuple(plans)
        self.positional = tuple(
            plan
            for plan, argument in zip(plans, schema.arguments)
            if not argument.kwarg_only
        )
        # Last spelling wins, as the dict comprehension it replaces did.
        self.by_name = {plan.name: plan for plan in plans}
        # The varargs int-list rule's *static* half. Its precondition is "the
        # signature has exactly one positional argument (past `self`) and that
        # argument is an int list"; only "and the caller passed an int there"
        # is left for call time. When the count matches, the index is in range
        # by construction: `len(positional) - skip == 1` means
        # `len(positional) == skip + 1`.
        skip = 1 if self_bound else 0
        self.varargs_intlist = len(self.positional) - skip == 1 and (
            self.positional[skip].is_list
            and self.positional[skip].base in ("int", "SymInt")
        )
        # "every argument without a default was bound" -- the arguments *with*
        # one can never fail that test, so they need not be walked.
        self.required = tuple(plan.name for plan in plans if not plan.has_default)
        # Whether the "drop arguments equal to their own default" pass has
        # anything at all to look at. `view(Tensor self, SymInt[] size)` and
        # every other pure-shape schema has no defaults, so the pass is a
        # second dict built to hold exactly what the first one held.
        self.any_defaults = any(plan.default_source is not None for plan in plans)
        self.n_arguments = len(plans)
        self.n_positional = len(self.positional)
        self.armed = False

    def arm(self, checker) -> None:
        """Attach the type predicates. Called once, on the first bind."""
        for plan in self.arguments:
            plan.predicate = checker.predicate_for(
                plan.base, plan.is_list, plan.optional, plan.sized_int_list
            )
        self.armed = True


class _Overloads:
    """The ordered signature list for one `torch.<name>` or one `Tensor.<name>`.

    `self_bound` is the only difference between the two. A method's receiver is
    already chosen before any signature is considered -- upstream binds it
    outside `PythonArgParser` entirely -- so it is passed in as `args[0]` and
    the parts of the algorithm that count positional arguments skip over it.
    Concretely that is the varargs int-list rule: `x.view(2, 3)` has to mean
    `view([2, 3])`, and the precondition for that rule is "the signature has
    exactly one positional argument", which is only true of `view` once `self`
    is out of the count.
    """

    __slots__ = (
        "name",
        "schemas",
        "keys",
        "self_bound",
        "_checker_source",
        "_candidates",
        "_skip",
        "_armed",
    )

    def __init__(self, name: str, schemas, checker_source, self_bound: bool = False) -> None:
        self.name = name
        self.self_bound = self_bound
        self._skip = 1 if self_bound else 0
        # Every plan of one entry is armed together, on the first `resolve`.
        # A `_SchemaPlan` is constructed here and nowhere else and is only ever
        # reachable through this entry, so arming per entry is the same work as
        # arming per plan -- one flag test per *call* instead of one per
        # *candidate*, and one `_TypeChecker` built instead of one per plan.
        self._armed = False
        self.schemas = [_Schema.parse(text) for text in schemas]
        # A callable rather than a `_TypeChecker`: `layout` and `memory_format`
        # are synthesised later in `install` than this table is parsed, and the
        # first *call* is long after both. Parsing and validating the table
        # still happens now -- see the `_SCHEMA_BASE_TYPES` check below.
        self._checker_source = checker_source
        self.keys = []
        for schema in self.schemas:
            namespace, _, op = schema.name.partition("::")
            self.keys.append(f"{namespace}.{op}.{schema.overload_name or 'default'}")
            if self_bound and not schema.arguments:
                raise RuntimeError(
                    f"torch._C shim: methods.json entry {name!r} has a schema with "
                    f"no `self` to bind the receiver into: {schema}"
                )
            for argument in schema.arguments:
                base, _, _, _ = _decompose_type(str(argument.type))
                if base not in _SCHEMA_BASE_TYPES:
                    # At install time, not at call time. A spelling nobody
                    # taught `_TypeChecker` would otherwise make every call to
                    # that overload silently fail to match, which reads as "no
                    # overload matched" -- a wrong answer wearing the shape of
                    # a right one.
                    raise RuntimeError(
                        f"torch._C shim: overloads.json entry {name!r} uses schema "
                        f"type {base!r}, which _TypeChecker does not handle: "
                        f"{schema}"
                    )
        # `(plan, key)` in declaration order -- the order overload resolution
        # walks, which is part of the answer (`pow.Tensor_Tensor` before
        # `pow.Tensor_Scalar`). Zipped once here rather than on every call.
        self._candidates = tuple(
            (_SchemaPlan(schema, self_bound), key)
            for schema, key in zip(self.schemas, self.keys)
        )

    def _bind_keywords(self, plan, positional, given, kwargs, bound):
        """The keyword half of `FunctionSignature::parse`. Mutates `bound`.

        Answers False if this schema does not accept the keywords.

        This is split out of `resolve` rather than inlined into it because
        keywords are rare: over a SmolLM2-135M prefill, 183 of 1188 `resolve`
        calls pass any, and 463 of 1651 candidate attempts are refusals that
        never reach here. Keeping it out leaves the loop every call runs free
        of a branch that almost never fires, and it is not a second statement
        of anything -- the positional half of the rules is in `resolve`, the
        keyword half is here, each written once.

        `given` is the number of positional arguments the call supplied, after
        the varargs int-list rewrite.
        """
        # "given twice". In the pre-merge spelling this ran *before* the
        # positional type checks rather than after; both orders answer False
        # for the same calls and neither has a side effect, so which of the two
        # reasons is found first is not observable.
        for parameter in positional[:given]:
            if parameter.name in kwargs:
                return False
        by_name = plan.by_name
        for name, value in kwargs.items():
            parameter = by_name.get(name)
            if parameter is None or name in bound:
                return False
            if not parameter.predicate(value):
                return False
            # `coerce`'s rules, with their preconditions precomputed.
            if (
                parameter.sized_int_list
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                value = (value,)
            elif (
                parameter.scalar_int
                and type(value) is not int
                and isinstance(value, _C_TENSORBASE[0])
            ):
                value = _symint_from_tensor(value)
            elif parameter.int_list and isinstance(value, (list, tuple)):
                value = _coerce_symint_list(value)
            bound[name] = value
        return True

    def resolve(self, args, kwargs):
        """Choose the overload and bind the call.

        This is torch's `FunctionSignature::parse` -- once per candidate, in
        declaration order, first acceptance wins -- with everything that does
        not depend on `args`/`kwargs` already computed into the `_SchemaPlan`s
        when the table was parsed. What is left here only reads the call.

        The per-candidate parse used to be a separate `_bind` method. It had
        exactly one caller, this loop, so folding it in removes a Python frame
        per candidate -- 1651 of them per SmolLM2-135M forward pass -- and
        duplicates nothing. docs/bindings/BIND.md §6 priced "merging the frames" against
        "a second copy of the resolution loop"; that price is real for `fn` and
        `method`, which are two call sites into `resolve`, and it was not real
        here.

        `args` is always a tuple: `_torch_function`'s `fn` and `_tensor_method`'s
        `method` are the only callers and both build one.
        """
        if not self._armed:
            checker = self._checker_source()
            for plan, _key in self._candidates:
                if not plan.armed:
                    plan.arm(checker)
            self._armed = True

        # `self` is bound before any signature is looked at, so it is not part
        # of what the parser counts (see the class docstring).
        skip = self._skip
        n_args = len(args)

        for plan, key in self._candidates:
            call = args
            given = n_args

            # The varargs int-list rule, with torch's exact precondition: it
            # applies only when the signature has a *single* positional
            # argument and that argument is an int list, which is what makes
            # `torch.ones(2, 3)` mean `ones([2, 3])` while `torch.full(2, 3)`
            # stays an error rather than becoming `full([2], 3)`. The half of
            # the precondition that reads the signature is
            # `plan.varargs_intlist`.
            if plan.varargs_intlist and given > skip:
                head = call[skip]
                if isinstance(head, int) and not isinstance(head, bool):
                    # Slicing a tuple yields a tuple, so the `tuple()` calls
                    # this line used to make were copies of something already
                    # of the right type.
                    call = call[:skip] + (call[skip:],)
                    given = skip + 1

            if given > plan.n_positional:
                continue

            positional = plan.positional
            bound = {}
            for value, parameter in zip(call, positional):
                if not parameter.predicate(value):
                    bound = None
                    break
                # `coerce`'s rules, with their preconditions precomputed.
                if (
                    parameter.sized_int_list
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ):
                    value = (value,)
                elif (
                    parameter.scalar_int
                    and type(value) is not int
                    and isinstance(value, _C_TENSORBASE[0])
                ):
                    # The predicate let a single-element Tensor through; this
                    # is upstream's unpack, and the one place a `bool` tensor
                    # becomes a RuntimeError. An `int` SUBCLASS that is not
                    # `bool` (numpy's integers) also lands here and `int()` is
                    # the right answer for it too.
                    value = _symint_from_tensor(value)
                elif parameter.int_list and isinstance(value, (list, tuple)):
                    # The same unpack, per element. `_coerce_symint_list`
                    # returns `value` unchanged when no element is a Tensor,
                    # which is every shape argument in a model forward.
                    value = _coerce_symint_list(value)
                bound[parameter.name] = value
            if bound is None:
                continue

            if kwargs and not self._bind_keywords(
                plan, positional, given, kwargs, bound
            ):
                continue

            # If every argument got a value the "was everything required
            # bound?" test cannot fail, so it need not be walked. `bound`'s
            # keys are always argument names, so equal counts means equal sets.
            # (A schema with a repeated argument name has fewer distinct names
            # than arguments, so the counts cannot match and the walk still
            # happens.)
            if len(bound) != plan.n_arguments:
                for name in plan.required:
                    if name not in bound:
                        bound = None
                        break
                if bound is None:
                    continue

            if not plan.any_defaults:
                return key, bound
            by_name = plan.by_name
            result = {}
            for name, value in bound.items():
                source = by_name[name].default_source
                if source is None or not _is_schema_default(value, source):
                    result[name] = value
            return key, result

        owner = "Tensor." if self.self_bound else "torch."
        shown = args[1:] if self.self_bound else args
        raise TypeError(
            f"{owner}{self.name}(): no matching overload in torch._C shim for "
            f"({_describe_call(shown, kwargs)}). Candidates tried, in order:\n"
            + "\n".join(f"  {schema}" for schema in self.schemas)
        )


def _describe_call(args, kwargs) -> str:
    parts = [type(a).__name__ for a in args]
    parts += [f"{k}={type(v).__name__}" for k, v in kwargs.items()]
    return ", ".join(parts)


# Python-level keyword arguments torch's factory functions accept that no aten
# schema mentions. `requires_grad` is the autograd knob, and this shim has no
# autograd (DESIGN.md §3 stage 0) -- so the flag is *carried*, exactly as
# `TensorBase.requires_grad_` has always carried it, and `backward()` is where
# the absence is reported.
#
# **It used to be refused here, and docs/training/BACKWARD2.md §4.1 is why it is not.**
# The refusal's stated ground was that "returning a tensor that quietly records
# nothing would be worse than refusing", and that ground does not survive being
# measured, for one reason:
#
#     torch.ones(2, 2, requires_grad=True)      <- refused
#     torch.ones(2, 2).requires_grad_(True)     <- succeeds, and always has
#
# Both produce the same object. So the refusal protected nobody: it redirected
# a caller to a spelling that hands over the identical tensor **with no warning
# at all**, which is strictly worse than either accepting or refusing both. The
# boundary has not moved -- `Tensor.backward()` still refuses, by name, and now
# it is the *first* wall a `.backward()` reaches rather than the third.
#
# `torchnative.adapt`'s stage-2 refusal is the load-bearing consumer and it gets
# more accurate, not less: its probe is
# `torch.ones(1, requires_grad=True).sum().backward()`, which before this change
# refused at the *factory* -- no evidence about autograd at all, since
# `.requires_grad_(True)` walks past it -- and now refuses at `.backward()`,
# which is the thing the sentence claims to test.
# `install()` fills this in with the live `_C` module once dtype singletons
# exist (`module.bool`, `module.int64`, ...), the same one-slot-list pattern
# `_C_TENSORBASE` above uses -- a global rebound later so `_strip_python_only_
# kwargs` reads a container that exists at definition time.
_PY_DTYPE_MODULE = [None]

# docs/bindings/ARGFORM.md. `dtype=bool`/`dtype=int`/`dtype=float` (the *Python* types,
# not `torch.bool`/`torch.int64`/`torch.float32`) are upstream's own overload
# resolver accepting a `type` wherever a schema takes `ScalarType` -- measured
# on 2.13.0 with `torch.ones(2, dtype=bool)`, `.to(dtype=int)`, and
# `torch.tensor([1], dtype=float)`, all of which upstream accepts and maps to
# exactly these three dtypes (bool -> `torch.bool`, int -> `torch.int64`,
# float -> `torch.float64` for `torch.tensor`/`.to()` but -- measured
# separately -- `torch.float64` for `torch.ones(dtype=float)` too). `complex`
# is deliberately left out: it maps to `torch.complex128` upstream, but
# `_NUMPY_DTYPE_TO_TORCH`'s own note records that complex dtypes have no
# candle storage behind them here, so mapping `dtype=complex` to a name this
# shim cannot back would trade one refusal for a worse one further in.
_PY_TYPE_TO_DTYPE_NAME = {
    bool: "bool",
    int: "int64",
    float: "float64",
}

# docs/bindings/ARGFORM.md. `axis=`/`keepdims=` are numpy's spellings of `dim=`/
# `keepdim=`, and upstream accepts them on some reductions and not others --
# docs/numerics/SCALAR2.md found the same "per-op, no principle" shape for scalar
# dispatch. Each key here was measured against torch 2.13.0 directly
# (`x.mean(axis=1, keepdim=True)`, `torch.mean(x, axis=1, keepdim=True)`) and
# accepted; nothing is added on the strength of "this is probably fine
# elsewhere too". `keepdims` is included for `mean` because it was measured
# alongside `axis` on the same call and both are accepted -- it was not
# separately requested by any of the six architectures, so it rides on
# `mean`'s own measurement rather than starting one of its own.
_NUMPY_KEYWORD_ALIASES = {
    "mean": {"axis": "dim", "keepdims": "keepdim"},
}


def _strip_python_only_kwargs(name: str, kwargs: dict):
    """Returns `(kwargs_for_the_schema, requires_grad)`.

    The flag comes back rather than being applied here because this runs
    *before* the dispatch that makes the tensor. `_apply_requires_grad` is the
    other half, and both call sites use it.
    """
    # Nothing to strip out of nothing, and the overwhelming majority of calls
    # arrive with no keyword arguments at all. A fresh dict rather than the
    # caller's, so the result is still something the caller may keep.
    if not kwargs:
        return {}, False
    kwargs = dict(kwargs)
    requires_grad = bool(kwargs.pop("requires_grad", False))
    # An explicit `out=None` is how the vendored tree spells "no out tensor".
    # Left in place it would fail to bind the `.out` overload (`out` is not
    # optional there) *and* fail to bind the plain one (`out` is not an
    # argument of it), so the call would report no matching overload.
    if "out" in kwargs and kwargs["out"] is None:
        del kwargs["out"]
    # numpy-spelling aliases -- see `_NUMPY_KEYWORD_ALIASES` above. Renaming
    # rather than adding: if the caller already gave the canonical name too,
    # this must not silently pick one. A plain `kwargs[canonical] = kwargs.
    # pop(alias)` would overwrite the caller's `dim` with `axis` and never
    # notice -- a dict assignment does not raise on a key that already
    # exists -- so the collision is refused by name here instead, matching
    # upstream's own refusal (measured: `mean(dim=1, axis=1)` raises
    # `TypeError: mean() received multiple values for argument 'dim'`).
    aliases = _NUMPY_KEYWORD_ALIASES.get(name)
    if aliases:
        for alias, canonical in aliases.items():
            if alias in kwargs:
                if canonical in kwargs:
                    raise TypeError(
                        f"{name}() received multiple values for argument "
                        f"{canonical!r} ({alias!r} is its numpy-spelling alias)"
                    )
                kwargs[canonical] = kwargs.pop(alias)
    # `dtype=bool`/`dtype=int`/`dtype=float` -- see `_PY_TYPE_TO_DTYPE_NAME`
    # above. Applied wherever a `dtype=` keyword shows up rather than per-op,
    # because the measurement this rests on (`.to()`, `torch.tensor()`,
    # `torch.ones()`) is upstream's *overload resolver* accepting a `type` for
    # `ScalarType`, not a rule any one op adds on top of it.
    dtype = kwargs.get("dtype")
    if dtype in _PY_TYPE_TO_DTYPE_NAME:
        dtype_module = _PY_DTYPE_MODULE[0]
        if dtype_module is not None:
            kwargs["dtype"] = getattr(dtype_module, _PY_TYPE_TO_DTYPE_NAME[dtype])
    return kwargs, requires_grad


# Upstream's factory wording, transcribed from torch 2.13.0 by running the
# failing case. It is a *third* spelling of one rule -- `TensorBase.requires_grad
# = True` says "only Tensors of ...", `requires_grad_(True)` says "only Tensors
# of floating point dtype ...", and this one capitalises. docs/training/BACKWARD2.md §4.2
# records all three, measured side by side; reproducing each at the door that
# produces it is what lets a caller grep for the text upstream gave them.
_FACTORY_REQUIRES_GRAD_REFUSAL = (
    "Only Tensors of floating point and complex dtype can require gradients"
)


def _refuse_non_floating_requires_grad(tensor, mode, message):
    """Upstream's one rule: only a float or complex tensor may require grad.

    Stated at three doors because upstream states it at three doors with three
    different wordings (docs/training/BACKWARD2.md §4.2). `mode` is checked because
    upstream only objects to `True` -- `requires_grad = False` goes through on
    an integer tensor, and `nn.Module._apply` writes exactly that over integer
    buffers.

    A tensor whose `dtype` cannot answer either question is let through rather
    than refused: this is a rule about which tensors may carry a flag, not a
    place to invent a second dtype taxonomy.
    """
    if not mode:
        return
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        return
    if getattr(dtype, "is_floating_point", False) or getattr(dtype, "is_complex", False):
        return
    raise RuntimeError(message)


def _apply_requires_grad(result, message=_FACTORY_REQUIRES_GRAD_REFUSAL):
    """Carry `requires_grad=True` onto whatever a factory just built.

    A call given `requires_grad=True` may return something that is not a tensor
    -- `_torch_level_function` wraps every op and not only factories, and an op
    with a tuple result reaches here the same way. A non-tensor cannot carry the
    flag, and that is a `TypeError` rather than a silent drop, because a
    silently dropped `requires_grad` is precisely the thing the old refusal was
    written to prevent and the one part of its argument that was right.
    """
    tensorbase = _C_TENSORBASE[0]
    if tensorbase is None or not isinstance(result, tensorbase):
        raise TypeError(
            "torch._C shim: requires_grad=True was given to a call that did not "
            f"return a tensor ({type(result).__name__}), so there is nothing to "
            "carry the flag"
        )
    _refuse_non_floating_requires_grad(result, True, message)
    result.requires_grad = True
    return result


# Filled in by `install()` once `module.TensorBase` is reachable. A module-level
# list rather than a global rebound later, so `_apply_requires_grad` reads a
# container that exists at definition time; if `install()` ever forgot to fill
# it, every `requires_grad=True` would fail loudly on the first call rather than
# dropping the flag.
_C_TENSORBASE = [None]


# The dtypes upstream's `SymInt[]` argument parser will take a Tensor element
# in. Measured against real torch 2.13.0 in a separate process, one dtype at a
# time (docs/bindings/BIND5.md §1), NOT inferred from `zeros` alone: every integral
# dtype is accepted and `bool` is not, which is why `bool` is absent from a
# list that would otherwise be spelled "integral".
_SIZE_LIST_TENSOR_DTYPES = ("uint8", "int8", "int16", "int32", "int64")

# `(accepted_integral_dtypes, bool_dtype)`, filled by `install()` beside
# `_C_TENSORBASE`. `_TypeChecker` can read them off its own `module`, but the
# COERCION happens in `_Overloads`, which has no module -- see
# `_symint_from_tensor`.
_C_SYMINT_TENSOR_DTYPES = [None]


def _symint_from_tensor(value):
    """Upstream's unpack for a Tensor bound to a scalar `int`/`SymInt`.

    Two layers, and the split is upstream's own -- it is what makes a `bool`
    tensor a `RuntimeError` here while a `float` one is a `TypeError` from the
    parser. `THPUtils_checkIndex`-side accepts any single-element tensor of an
    integral dtype **including bool**; the unpack
    (`torch/csrc/utils/pybind.cpp`) then asserts `scalar.isIntegral(false)`,
    which bool fails. So the predicate says yes and this says no, in that
    order, and the two exception classes fall out of it rather than being
    chosen. docs/bindings/BIND5.md §7 has the measurement for both.
    """
    dtypes = _C_SYMINT_TENSOR_DTYPES[0]
    if dtypes is not None and value.dtype == dtypes[1]:
        raise RuntimeError(
            "scalar.isIntegral( false) INTERNAL ASSERT FAILED -- a bool tensor "
            "cannot be bound to an int/SymInt argument; upstream asserts the "
            "same thing in torch/csrc/utils/pybind.cpp"
        )
    return int(value)



def _symint_tensor_element_ok(item) -> bool:
    """Is `item` a Tensor upstream would accept inside an int list?

    One element (any ndim), an integral dtype **or `bool`**. Bool is accepted
    here and refused by `_coerce_symint_list`, which is where upstream refuses
    it too -- see `_symint_from_tensor` for why the split produces upstream's
    two different exception classes.

    Returns False rather than raising for anything else, because this is a
    *predicate*: a `float` tensor here means "this overload does not bind",
    and the caller may have another one to try. The message comes later, from
    whoever runs out of candidates.
    """
    tensorbase = _C_TENSORBASE[0]
    if tensorbase is None or not isinstance(item, tensorbase):
        return False
    dtypes = _C_SYMINT_TENSOR_DTYPES[0]
    if dtypes is None:
        return False
    integral, boolean = dtypes
    return item.numel() == 1 and (
        any(item.dtype == d for d in integral) or item.dtype == boolean
    )


def _coerce_symint_list(values):
    """`_symint_from_tensor`, applied to each element of a bound int list.

    The predicate has already said yes, so the only thing that can raise here
    is the `bool` tensor -- and it must, because upstream does, one layer
    further in than the parser. `values` is returned unchanged when it holds
    no Tensor at all, which is the case for every shape argument in a model
    forward, so this costs one `isinstance` per element on the hot path and
    allocates nothing.

    `longformer` and `led` are the callers that made this general rather than
    a per-composite helper: `_sliding_chunks_matmul_attn_probs_value` spells
    `padded_value.as_strided(size=(..., seq_len // window, ...), ...)` where
    one element arrives as a 0-dim tensor. docs/kernels/REPEAT.md §4.
    """
    tensorbase = _C_TENSORBASE[0]
    for item in values:
        if isinstance(item, tensorbase):
            break
    else:
        return values
    dtypes = _C_SYMINT_TENSOR_DTYPES[0]
    out = []
    for item in values:
        if not isinstance(item, tensorbase):
            out.append(item)
        elif dtypes is not None and item.dtype == dtypes[1]:
            # Upstream's wording, and its exception TYPE, both measured.
            raise RuntimeError(
                "Expected scalar.isIntegral( false) to be true, but got false.  "
                "(Could this error message be improved?  If so, please report an "
                "enhancement request to PyTorch.)"
            )
        else:
            out.append(int(item))
    return out


def _coerce_symint_size_tensors(module, name, size):
    """Upstream's `SymInt[]` size-list rule, for one call's `size` argument.

    A single-element integral Tensor sitting in a size list is taken for its
    value, the way a bare Python int is. This is upstream's own argument
    parser rule and not a `zeros` accident -- docs/bindings/BIND4.md §3 measured it for
    `zeros`, `ones`, `empty` and `view`, and docs/bindings/BIND5.md §1 measured the
    `new_*` family and the REFUSALS, which is the half that was missing.

    The refusals are the point. Upstream draws three lines here, and the
    first version of this rule (docs/bindings/BIND4.md §3, `int(item)` applied to
    anything that was a Tensor) crossed two of them, making this build **more
    permissive than the thing it replaces** -- exactly what docs/bindings/ARGFORM.md
    forbids. Measured upstream, `torch.zeros((2, X))`:

        X = tensor(3)          -> (2, 3)
        X = tensor([3])        -> (2, 3)     one element, any ndim
        X = tensor(3, int8)    -> (2, 3)     every integral dtype
        X = tensor(3.0)        -> TypeError  ... "type must be tuple of ints,
                                             but got Tensor"   -- even whole-valued
        X = tensor(3.5)        -> TypeError  the same
        X = tensor(True)       -> RuntimeError  Expected scalar.isIntegral(...)
        X = tensor([3, 4])     -> TypeError  the same unpack failure

    `int(item)` accepted the first three of those refusals silently
    (`tensor(3.5)` became 3, `tensor(True)` became 1), which no test could see
    because nothing asserted them. They raise here, with upstream's own
    message text, and `bool` gets upstream's *different* exception TYPE
    because upstream reaches it one layer further in -- past the unpack, at
    the scalar conversion.

    A negative size is left to the kernel below, which already refuses it
    (`OverflowError`, where upstream says `RuntimeError: ... Dimension size
    must be non-negative`): both refuse, and inventing a message here would
    be a second surface claiming to know a rule the kernel owns.
    """
    if not isinstance(size, (list, tuple)):
        return size
    tensorbase = module.TensorBase
    if not any(isinstance(item, tensorbase) for item in size):
        return size
    accepted = tuple(getattr(module, d) for d in _SIZE_LIST_TENSOR_DTYPES)
    out = []
    for pos, item in enumerate(size, start=1):
        if not isinstance(item, tensorbase):
            out.append(item)
            continue
        if item.numel() == 1 and any(item.dtype == d for d in accepted):
            out.append(int(item))
            continue
        if item.numel() == 1 and item.dtype == module.bool:
            raise RuntimeError(
                "Expected scalar.isIntegral( false) to be true, but got false.  "
                "(Could this error message be improved?  If so, please report an "
                "enhancement request to PyTorch.)"
            )
        raise TypeError(
            "%s(): argument 'size' failed to unpack the object at pos %d with "
            'error "type must be tuple of ints,but got Tensor"' % (name, pos)
        )
    return out


def install(module, surface_json: str, overloads_json: str, methods_json: str) -> None:
    surface = json.loads(surface_json)
    dispatch = module._aten_dispatch
    real = set(vars(module))
    off = (frozenset(surface.get("probes", ())) | EXTRA_OFF_SWITCHES) - ANSWERED_PROBES
    # Readable from Python so the decision can be inspected rather than
    # reverse-engineered from what is absent.
    module._shim_off_switches = sorted(off)

    # -- overload resolution ----------------------------------------------
    #
    # Parsed and validated now, so a malformed table stops `import _C` at the
    # line that is wrong. The `_TypeChecker` it will use is built on first
    # call, because `layout` and `memory_format` do not exist yet.
    checker_cell = []

    def _checker_source():
        if not checker_cell:
            checker_cell.append(_TypeChecker(module))
        return checker_cell[0]

    overloads = {
        name: _Overloads(name, schemas, _checker_source)
        # `_README`, not `_`. The predicate used to be `not
        # name.startswith("_")` while its comment said README, and the
        # difference was not cosmetic: it made a `torch.<op>` binding
        # impossible for every aten op whose name begins with an underscore.
        # `torch._grouped_mm` -- which is what `torch.nn.functional.grouped_mm`
        # calls and therefore what `transformers`' MoE layer calls -- refused
        # with "overload resolution has no table entry" while its entry sat in
        # the table. docs/kernels/GROUPED_MM.md §6.1. The sibling comprehension below
        # always spelled the intent correctly; this one now matches it.
        for name, schemas in json.loads(overloads_json).items()
        if not name.startswith("_README")
    }
    # Inspectable for the same reason as `_shim_off_switches`: which aten key a
    # `torch.<op>` call can reach should be answerable by asking, not by
    # reading the table back out of the artefact.
    module._shim_overloads = {
        name: list(entry.keys) for name, entry in sorted(overloads.items())
    }

    # The same table, for `tensor.<method>(...)`. Separate file and separate
    # dict because they are separate bindings upstream (see methods.json's
    # `_README`), and separate here so `_shim_overloads` keeps meaning exactly
    # "what `torch.<op>` can reach".
    methods = {
        name: _Overloads(name, schemas, _checker_source, self_bound=True)
        for name, schemas in json.loads(methods_json).items()
        if not name.startswith("_README")
    }
    module._shim_methods = {
        name: list(entry.keys) for name, entry in sorted(methods.items())
    }

    # `_apply_requires_grad`'s type test. Set before any factory can run, which
    # is why it is here and not beside the `TensorBase` member installation
    # further down.
    _C_TENSORBASE[0] = module.TensorBase
    # `_symint_from_tensor`'s dtype rule, snapshotted for the same reason:
    # the coercion runs inside `_Overloads`, which never sees `module`.
    _C_SYMINT_TENSOR_DTYPES[0] = (
        tuple(getattr(module, d) for d in _SIZE_LIST_TENSOR_DTYPES),
        module.bool,
    )
    # See `_PY_DTYPE_MODULE`'s own comment above `_strip_python_only_kwargs`:
    # filled in here so `dtype=bool`/`dtype=int`/`dtype=float` can be resolved
    # to `module.bool`/`module.int64`/`module.float64` at call time. Safe to
    # set this early -- it is only ever read after `install()` has returned,
    # by which point every dtype singleton below exists regardless of whether
    # it came from Rust (`real`) or the types loop just below.
    _PY_DTYPE_MODULE[0] = module

    # -- types ------------------------------------------------------------
    resolved = {}
    for name, spec in _order_types(surface["types"]):
        if name in real or name in off:
            continue
        resolved[name] = _build_type(name, spec, "torch._C", resolved)
        setattr(module, name, resolved[name])

    # The two walls a `Tensor.backward()` reaches, in the order it reaches
    # them. Both are synthesised above as table-less stubs, and both are
    # replaced here -- one because it is not autograd at all, the other because
    # it is. The second stopped being a wall in docs/training/BACKWARD9.md: it is the
    # engine now, translating upstream's call into `_eager_backward` and back.
    # docs/training/BACKWARD2.md §4.3.
    _install_thread_local_store(module)
    _install_engine(module)

    # `torch.Size` is a real tuple subclass upstream and the tree relies on it
    # being one (`isinstance(x.shape, tuple)`, unpacking, slicing).
    #
    # Every member below was **measured** against torch 2.13.0 rather than
    # assumed, and two of the measurements were surprises worth writing down:
    #
    #   * `repr` is `torch.Size([2, 3])` -- square brackets inside the call,
    #     not the tuple's own `(2, 3)`. It is not cosmetic. Thirty-odd
    #     `transformers` docstrings print exactly that string, and so does
    #     every `print(x.shape)` a user writes.
    #   * the type is **closed** under slicing, `+`, reflected `+` and `*`:
    #     `torch.Size([2,3])[1:]` is a `Size`, and `(5,) + size` is a `Size`
    #     too. A subclass of `tuple` gets none of that for free -- `tuple`'s
    #     own slots return plain tuples -- so each one is written out.
    #
    # `numel()` of the empty size is `1`, not `0`: it is the product over no
    # dimensions, which is what a zero-dim tensor's element count is.
    class Size(tuple):
        __module__ = "torch"
        # Without this the qualname is `install.<locals>.Size`, which shows up
        # in `repr(torch.Size)` and -- the part that is not cosmetic -- makes
        # the class unpicklable, because pickle resolves a class by looking up
        # `__module__`.`__qualname__`. `torch.Size` pickles upstream.
        __qualname__ = "Size"

        def numel(self):
            n = 1
            for d in self:
                n *= d
            return n

        def __repr__(self):
            return "torch.Size([" + ", ".join(repr(d) for d in self) + "])"

        __str__ = __repr__

        def __getitem__(self, index):
            item = tuple.__getitem__(self, index)
            return Size(item) if isinstance(index, slice) else item

        def __add__(self, other):
            result = tuple.__add__(self, other)
            return result if result is NotImplemented else Size(result)

        def __radd__(self, other):
            return Size(tuple(other) + tuple(self))

        def __mul__(self, other):
            result = tuple.__mul__(self, other)
            return result if result is NotImplemented else Size(result)

        __rmul__ = __mul__

    module.Size = Size
    resolved["Size"] = Size

    # `TensorBase.shape` (and `size()` with no `dim`, which goes through it)
    # answers with this class from here on. Registered rather than imported,
    # because `_C` runs without a `torch` package under the golden loader --
    # see `SIZE_CLASS` in `tensor.rs`.
    module._set_size_class(Size)

    # -- TensorBase members ----------------------------------------------
    #
    # VENDOR.md wall 3: `class Tensor(torch._C.TensorBase)` in
    # `torch/_tensor.py` runs `_C._add_docstr(_C.TensorBase.<name>, ...)`
    # hundreds of times in its *class body*, so the whole member set is
    # demanded before `import torch` returns. PyO3's type objects are ordinary
    # heap types and accept `setattr`, which is why `TensorBase` stays the
    # native class rather than becoming a Python subclass -- a subclass would
    # have broken `isinstance(op_result, TensorBase)`, since results come out
    # of Rust as the native type.
    tensorbase = module.TensorBase
    existing = set(vars(tensorbase))

    def _wanted(member):
        return member not in existing and member not in UNSAFE_DUNDERS

    for member in surface["tensorbase"]["methods"]:
        if _wanted(member):
            setattr(tensorbase, member, _make_function(f"TensorBase.{member}", member))
    for member in surface["tensorbase"]["attrs"]:
        if _wanted(member):
            setattr(tensorbase, member, _make_property(f"TensorBase.{member}"))

    # ...and then the ones that are not stubs. Installed *after* the stub loop
    # so a real implementation always wins over the placeholder that the stub
    # surface would otherwise have left in place, and installed unconditionally
    # rather than through `_wanted` -- several of these (`__eq__`, `__ne__`)
    # are in `UNSAFE_DUNDERS`, which is a rule about *raising* stand-ins, not
    # about working implementations. A raising `__eq__` breaks dict and set
    # use; one that returns a mask is what upstream has.
    _install_tensor_methods(module, tensorbase, dispatch, methods)

    # `type(TensorBase)`. Upstream's is a distinct pybind11-adjacent metatype;
    # ours is plain `type`, because a PyO3 type's metatype cannot be replaced
    # after creation (`__class__` assignment refuses: `type` is not a heap
    # type). `torch/nn/parameter.py:19` does `class _ParameterMeta(_TensorMeta)`
    # and then uses it as the metaclass of a `torch.Tensor` subclass, which
    # works either way. Recorded in docs/models/IMPORT_TORCH.md as a difference,
    # because `isinstance(X, _TensorMeta)` is now true of every class.
    module._TensorMeta = type

    # -- submodules -------------------------------------------------------
    # Keyed on the module's *real* name, not the literal "torch._C".
    #
    # The golden harness (`tools/golden/loader.py`) loads this artefact
    # standalone, as a module called `_C`, and then imports real upstream torch
    # in the same interpreter to compare against. Registering `torch._C._onnx`
    # in `sys.modules` from a shim that is not `torch._C` made real torch's own
    # `_C` initialisation fail -- `generic_type: cannot initialize type
    # "TensorProtoDataType": an object with that name is already defined`. The
    # shim was reaching into a namespace it does not own.
    prefix = module.__name__
    roots = set(surface["submodules"]) | set(EXTRA_SUBMODULES)
    # `_distributed_c10d` is closed: `_install_distributed_c10d` builds it whole
    # and its absent names have to stay absent (see `absent_backends` there).
    sys.meta_path.append(_SubmoduleFinder(prefix, roots, closed={"_distributed_c10d"}))
    for name in sorted(roots):
        spec = surface["submodules"].get(name, {})
        sub = types.ModuleType(f"{prefix}.{name}")
        sub.__path__ = []
        sub_types = {}
        for tname, tspec in _order_types(spec.get("types", {})):
            sub_types[tname] = _build_type(tname, tspec, sub.__name__, sub_types)
            setattr(sub, tname, sub_types[tname])
        for fname in spec.get("functions", ()):
            setattr(sub, fname, _make_function(f"torch._C.{name}.{fname}", fname))
        for vname in spec.get("values", ()):
            setattr(sub, vname, _Unimplemented(f"torch._C.{name}.{vname}"))
        _attach_module_catchall(sub)
        setattr(module, name, sub)
        sys.modules[f"{prefix}.{name}"] = sub

    # `torch._C._linalg.linalg_vector_norm` -- `torch/nn/functional.py:6100`'s
    # `normalize` calls this spelling directly, not `torch.linalg_vector_norm`
    # (which upstream does not expose either -- `torch.linalg.vector_norm` is
    # the public name, a different, Python-level function). `_linalg` has no
    # stub data (`EXTRA_SUBMODULES`, above), so the generic loop just ran left
    # every name on it as the catch-all `_Unimplemented`. docs/architectures/DEMAND.md §0.1
    # rank 3: this replaces the one name it asks for with a real resolving
    # function, sharing the exact overload machinery `torch.<op>` uses --
    # `overloads["linalg_vector_norm"]` is the same table
    # `_torch_level_function` reads for the top-level names, just not
    # harvested onto `torch.<name>` because upstream does not put it there
    # either.
    module._linalg.linalg_vector_norm = _torch_level_function(
        "linalg_vector_norm", dispatch, overloads
    )

    # `torch._C._fft.fft_fftn` -- `fnet`'s wall (docs/architectures/ARCH200.md,
    # docs/bindings/BIND3.md §6, docs/kernels/COMPLEX3.md §6). `torch/fft/__init__.py` is
    # `fftn = _add_docstr(_fft.fft_fftn, ...)`, so upstream needs this exact
    # spelling to be a real function; `_fft` has no stub data (`_linalg`'s
    # reason directly above), so the generic submodule loop had left every
    # name on it as the catch-all `_Unimplemented`. There is no decomposition
    # engine on this path -- `aten::fft_fftn` being `CompositeImplicitAutograd`
    # does not help a name upstream never routes through `torch.ops.aten`.
    #
    # The decomposition below is upstream's own, read off a `TorchDispatchMode`
    # logger (docs/kernels/COMPLEX3.md §6.1): widen to complex, optionally slice/pad
    # each transformed axis to match `s`, then `_fft_c2c`. Verified from
    # outside this file, in a probe process supplying these same lines: a full
    # two-layer `FNetModel` forward agrees with upstream element-wise, max
    # absolute difference 7.15e-07 over 256 outputs. That verification needs
    # `_to_copy(dtype=complex64)` reachable, which is a `bootstrap.py`-level
    # dtype gate this round did not touch -- so on THIS worktree, cut before
    # that gate opened, the numeric path still refuses; what is checked here
    # is that the name resolves to this function and no longer to the
    # catch-all, which does not need the gate open.
    #
    # `norm` maps to `_fft_c2c`'s integer `normalization` argument as
    # `{backward: 0, ortho: 1, forward: 2}`, read off the same trace, not off
    # the C++.
    _FFT_NORM = {None: 0, "backward": 0, "ortho": 1, "forward": 2}

    def fft_fftn(self, s=None, dim=None, norm=None, *, out=None):
        rank = self.dim()
        if dim is None:
            dim = list(range(rank)) if s is None else list(range(rank - len(s), rank))
        dim = [d + rank if d < 0 else d for d in dim]
        c = dispatch(
            "aten._to_copy.default", self,
            dtype=module.complex128 if self.dtype == module.float64
            else module.complex64,
        )
        if s is not None:
            for d, want in zip(dim, s):
                have = c.shape[d]
                if want < have:
                    c = dispatch("aten.slice.Tensor", c, d, 0, want)
                elif want > have:
                    pad = [0] * (2 * (c.dim() - d))
                    pad[-1] = want - have
                    c = dispatch("aten.constant_pad_nd.default", c, pad)
        return dispatch("aten._fft_c2c.default", c, dim, _FFT_NORM[norm], True)

    fft_fftn.__name__ = fft_fftn.__qualname__ = "fft_fftn"
    module._fft.fft_fftn = fft_fftn

    # `torch._C._linalg.linalg_qr` -- `rwkv`'s *construction* wall.
    #
    # `torch/linalg/__init__.py:2823` is `qr = _add_docstr(_linalg.linalg_qr,
    # ...)`, so `torch.linalg.qr` **is** this binding rather than a wrapper
    # around it, and `torch.nn.init.orthogonal_` (which `rwkv`'s
    # `_init_weights` calls) reaches it on the first line of the model's
    # constructor. There is no `torch.linalg_qr` and no `Tensor.linalg_qr` on
    # 2.13.0 -- checked both directions -- so this cannot be an
    # `overloads.json` row without inventing a door upstream does not have
    # (docs/bindings/SPELLINGS.md). It is a composite here for the same reason
    # `_install_nn`'s `upsample_bilinear2d` is one.
    #
    # The kernel already exists and is golden-compared (`aten.linalg_qr.default`,
    # docs/kernels/TAIL1.md §5); it returns upstream's `linalg_qr(Q=..., R=...)`
    # namedtuple, including the `mode="r"` one-dimensional empty `Q`, so
    # nothing is reshaped or renamed on the way through.
    #
    # `out=` is refused by name rather than ignored: upstream writes into the
    # supplied tensors and returns them, and silently returning fresh ones
    # would leave a caller holding unwritten buffers it believes are results.
    def linalg_qr(A, mode="reduced", *, out=None):
        if out is not None:
            raise NotImplementedError(
                "not implemented in torch._C shim: torch._C._linalg.linalg_qr("
                "out=...) -- upstream writes the factorisation into the tensors "
                "given and returns them; this shim only allocates its own"
            )
        return dispatch("aten.linalg_qr.default", A, mode)

    linalg_qr.__name__ = linalg_qr.__qualname__ = "linalg_qr"
    linalg_qr.__module__ = "torch._C._linalg"
    module._linalg.linalg_qr = linalg_qr

    # `torch._C._linalg.linalg_norm` -- `owlv2`'s and `owlvit`'s wall
    # (docs/kernels/COMPLEX.md §6). `torch/linalg/__init__.py:1353` is
    # `norm = _add_docstr(_linalg.linalg_norm, ...)`, the same shape as `qr`
    # above, and again with no `torch.linalg_norm` / `Tensor.linalg_norm` to
    # hang a table row on.
    #
    # **`ord` selects between different computations, so this forwards the ones
    # it can and refuses the rest by name.** `linalg.norm` is a vector norm or
    # a matrix norm depending on `ord` *and* `dim` together:
    #
    #     dim is an int (or a 1-list)     -> vector norm, any numeric `ord`
    #     dim is None, ord is None        -> flatten to 1-D, 2-norm
    #     dim is a 2-tuple                -> MATRIX norm
    #     dim is None, ord is not None    -> MATRIX norm when `A` is 2-D
    #     ord is 'fro' or 'nuc'           -> MATRIX norm
    #
    # Only the vector cases go through `aten.linalg_vector_norm.default`, which
    # is the kernel this shim has. The matrix cases are `linalg_matrix_norm`
    # upstream -- `nuc` and `ord=±2` need singular values, and `fro`/`±1`/`±inf`
    # are row/column reductions, none of which `vector_norm` computes -- so they
    # raise naming themselves rather than quietly answering the flattened
    # vector norm, which has the right shape and the wrong number.
    #
    # What the two blocked architectures actually spell, read out of
    # `transformers` rather than guessed:
    #
    #     modeling_owlv2.py:975    torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    #     modeling_owlv2.py:1048   torch.linalg.norm(x, dim=-1, keepdim=True)
    #     modeling_owlvit.py:957   (identical)
    #     modeling_owlvit.py:1028  (identical)
    #
    # -- both vector norms over an `int` `dim`, i.e. inside the branch below.
    def linalg_norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None):
        if out is not None:
            raise NotImplementedError(
                "not implemented in torch._C shim: torch._C._linalg.linalg_norm("
                "out=...)"
            )
        if isinstance(ord, str):
            raise NotImplementedError(
                f"not implemented in torch._C shim: torch._C._linalg.linalg_norm("
                f"ord={ord!r}) -- a matrix norm. Upstream routes 'fro' and 'nuc' to "
                "linalg_matrix_norm; 'nuc' needs singular values and 'fro' is a "
                "different reduction from the flattened vector 2-norm. This shim has "
                "aten.linalg_vector_norm.default and no matrix-norm kernel"
            )
        if dim is not None and not isinstance(dim, int):
            axes = list(dim)
            if len(axes) > 1:
                raise NotImplementedError(
                    f"not implemented in torch._C shim: torch._C._linalg.linalg_norm("
                    f"dim={tuple(axes)!r}) -- a {len(axes)}-tuple `dim` selects "
                    "upstream's matrix norm (linalg_matrix_norm), which this shim has "
                    "no kernel for"
                )
            dim = axes
        elif dim is not None:
            dim = [dim]
        if dim is None and ord is not None and A.dim() != 1:
            raise NotImplementedError(
                f"not implemented in torch._C shim: torch._C._linalg.linalg_norm("
                f"ord={ord!r}, dim=None) on a {A.dim()}-dimensional input -- with "
                "`dim=None` and an explicit `ord`, upstream requires a 1-D or 2-D "
                "input and takes the MATRIX norm when it is 2-D. Only the 1-D case "
                "is a vector norm, and only that one is implemented here. Pass "
                "`dim=-1` for the per-row vector norm, which is what the callers "
                "this was written for spell"
            )
        p = 2 if ord is None else ord
        if dtype is None:
            return dispatch(
                "aten.linalg_vector_norm.default", A, p, dim, keepdim
            )
        return dispatch(
            "aten.linalg_vector_norm.default", A, p, dim, keepdim, dtype=dtype
        )

    linalg_norm.__name__ = linalg_norm.__qualname__ = "linalg_norm"
    linalg_norm.__module__ = "torch._C._linalg"
    module._linalg.linalg_norm = linalg_norm

    # -- `_monitor._WaitCounter` -- a block that can be entered -------------
    #
    # `torch/distributed/c10d_logger.py:96` wraps every public collective in
    # `with _WaitCounter(f"pytorch.wait_counter.c10d.{name}").guard():`, and
    # `distributed_c10d.py:3403` does the same around object serialisation, so
    # this is on the path of every c10d call rather than a diagnostic corner.
    # Upstream's counts elapsed time into a process-global registry that
    # nothing in this project reads. Nothing here measures anything; the whole
    # requirement is that `guard()` return something a `with` can enter.
    class _WaitCounter:
        __module__ = "torch._C._monitor"

        def __init__(self, key):
            self.key = key

        def guard(self):
            return contextlib.nullcontext()

    module._monitor._WaitCounter = _WaitCounter

    # -- `_distributed_c10d`, which needs structure rather than names -------
    #
    # Replaces what the loop above built for it. Must run before the
    # `surface["module"]` loop below, which skips any name already set --
    # `_c10d_init` is installed from in here.
    _install_distributed_c10d(
        module, surface["submodules"].get("_distributed_c10d", {}))

    # -- `_dynamo.eval_frame` real no-ops ---------------------------------
    #
    # DYNAMO.md: `torch._dynamo` is an unconditional import (`transformers`
    # pulls it in through `masking_utils.py:42` on `torch >= 2.6`, no
    # `hasattr` gate exists to skip it -- DYNAMO.md §6). It reaches 52 names
    # under `_C._dynamo`, but only two were ever *called* rather than merely
    # referenced at that point: `eval_frame.set_guard_error_hook` and
    # `eval_frame.set_code_exec_strategy`, both at module scope in
    # `torch/_dynamo/guards.py:5457` and `torch/_dynamo/decorators.py:125`.
    # Both calls discard the return value -- they register a hook / tag a
    # code object for an eval-frame hooking mechanism that only
    # `torch.compile` installs, which this project never calls (DYNAMO.md
    # §3.2). So the only requirement is that the call not raise.
    #
    # docs/architectures/ARCH26.md added a third and fourth: `set_eval_frame` and
    # `set_eval_frame_isolate_recompiles_id`. `torch/_dynamo/__init__.py:133`
    # rebinds `torch.manual_seed = torch._disable_dynamo(torch.manual_seed)`
    # **unconditionally at `torch._dynamo` import time** -- so merely
    # `import transformers` (which imports `torch._dynamo` for the reason
    # above) is enough to make every later `torch.manual_seed(...)` call route
    # through `torch/_dynamo/eval_frame.py`'s `_fn`, which calls
    # `prior = set_eval_frame(None)` before the wrapped call and
    # `set_eval_frame(prior)` after (also `set_eval_frame_isolate_recompiles_id`
    # a few lines away, on the same call shape). Neither is a hook
    # registration like the first two -- both are a **get-and-set**, and the
    # caller relies on the return value to restore the prior state, so unlike
    # the two above these cannot be unconditional `None`-returning no-ops:
    # a nested `_fn` call has to see what the outer one set. `deberta`'s toy
    # forward (docs/architectures/ARCH26.md) is what surfaced this -- none of the twenty
    # architectures in docs/architectures/ARCH20.md called `torch.manual_seed` after
    # `transformers` had already pulled in `torch._dynamo`.
    #
    # `torch.compile` is never invoked here, so there is no real eval-frame
    # hook to install -- the state cell is honest about being exactly that: a
    # place to remember what the (never-consulted) hook was last set to.
    #
    # `_SubmoduleFinder` above already answers `torch._C._dynamo.<anything>`
    # generically -- including two levels deep, since it only checks that the
    # first path segment (`_dynamo`) is a known root -- so every other name
    # under `eval_frame`, `guards`, `utils` and `compiled_autograd` already
    # exists via the lazily-created catch-all module (DYNAMO.md §3.3, §7
    # item 4: `utils`/`compiled_autograd` are 0-access in this path and stay
    # empty). Only these four need a real body instead of the catch-all's
    # `_Unimplemented`, which raises on call. Pre-registering the module here
    # (rather than teaching the finder to special-case four names) means the
    # finder never runs for this particular submodule: Python's import system
    # checks `sys.modules` before consulting `sys.meta_path`.
    # `torch._C._export.pt2_archive_constants` is a *module* of 27 archive
    # path strings, with no behaviour at all -- but the generic stub makes it an
    # `_Unimplemented`, so every attribute raises. That is why `torch.compile`
    # died on `AOTINDUCTOR_DIR` rather than on anything about compiling
    # (docs/graph/COMPILE.md §2), and it stands in front of `torch.export` too.
    #
    # Transcribed from upstream rather than invented: these are wire-format
    # paths inside a `.pt2` archive, so a guessed value would produce an archive
    # nothing else can read. Closing it is only safe *after* `set_eval_frame`
    # refuses by name -- before that, it turned a loud failure into a silent
    # eager fallback, which COMPILE.md measured.
    if "_export" in roots:
        _pt2c = types.ModuleType("torch._C._export.pt2_archive_constants")
        for _k, _v in {
            'AOTINDUCTOR_DIR': 'data/aotinductor/',
            'ARCHIVE_FORMAT_PATH': 'archive_format',
            'ARCHIVE_FORMAT_VALUE': 'pt2',
            'ARCHIVE_ROOT_NAME': 'package',
            'ARCHIVE_VERSION_PATH': 'archive_version',
            'ARCHIVE_VERSION_VALUE': '0',
            'CONSTANTS_CONFIG_FILENAME_FORMAT': 'data/constants/{}_constants_config.json',
            'CONSTANTS_DIR': 'data/constants/',
            'CONSTANTS_PARAM_CONFIG_FORMAT': 'data/constants/{}_model_constants_config.json',
            'CUSTOM_OBJ_FILENAME_PREFIX': 'custom_obj_',
            'EXECUTORCH_DIR': 'data/executorch/',
            'EXTRA_DIR': 'extra/',
            'MODELS_DIR': 'models/',
            'MODELS_FILENAME_FORMAT': 'models/{}.json',
            'MODULE_INFO_PATH': 'extra/module_info.json',
            'MTIA_DIR': 'data/mtia',
            'OPAQUE_OBJ_FILENAME_PREFIX': 'opaque_obj_',
            'SAMPLE_INPUTS_DIR': 'data/sample_inputs/',
            'SAMPLE_INPUTS_FILENAME_FORMAT': 'data/sample_inputs/{}.pt',
            'TENSOR_CONSTANT_FILENAME_PREFIX': 'tensor_',
            'TS_SAMPLE_INPUTS_FILENAME_FORMAT': 'extra/{}.forward.sample_input.pt',
            'WEIGHTS_CONFIG_FILENAME_FORMAT': 'data/weights/{}_weights_config.json',
            'WEIGHTS_DIR': 'data/weights/',
            'WEIGHTS_PARAM_CONFIG_FORMAT': 'data/weights/{}_model_param_config.json',
            'WEIGHT_FILENAME_PREFIX': 'weight_',
            'XL_MODEL_WEIGHTS_DIR': 'xl_model_weights/',
            'XL_MODEL_WEIGHTS_PARAM_CONFIG_PATH': 'xl_model_weights/model_param_config',
        }.items():
            setattr(_pt2c, _k, _v)
        setattr(module._export, "pt2_archive_constants", _pt2c)
        sys.modules["torch._C._export.pt2_archive_constants"] = _pt2c

    if "_dynamo" in roots:
        eval_frame = types.ModuleType(f"{prefix}._dynamo.eval_frame")
        eval_frame.__path__ = []
        _attach_module_catchall(eval_frame)

        def set_guard_error_hook(hook):
            return None

        def set_code_exec_strategy(code, strategy):
            return None

        # Two independent cells: upstream's `set_eval_frame` and
        # `set_eval_frame_isolate_recompiles_id` are separate C functions with
        # separate storage (`torch/csrc/dynamo/eval_frame.c`), and nothing here
        # ever reads one while writing the other, but conflating them would be
        # a guess this shim has no call site to check.
        _eval_frame_cell = [None]
        _eval_frame_isolate_cell = [None]

        def set_eval_frame(callback):
            # Refuse to *install* a hook, and say why. Without this the cell
            # accepts the callback, never installs anything, and `torch.compile`
            # quietly returns the eager function -- docs/graph/COMPILE.md measured
            # exactly that by stubbing five unrelated symbols. Today's failure
            # is an accident of a missing data module, and closing that module
            # would turn a loud error into a lie.
            #
            # `callback is None` must stay a no-op: it is the *uninstall*
            # spelling, and `torch/_dynamo/__init__.py` takes that path on a
            # plain `import transformers`. Refusing it would break that import.
            if callback is not None:
                raise NotImplementedError(
                    "torch.compile is not available in this build. It needs "
                    "CPython's PEP 523 frame-evaluation hook "
                    "(_PyInterpreterState_SetEvalFrameFunc), which is outside "
                    "the stable ABI this extension is built against "
                    "(abi3-py313); see docs/graph/COMPILE.md. Use torch.export or "
                    "the capture API (docs/graph/CAPTURE.md) for a graph."
                )
            prior = _eval_frame_cell[0]
            _eval_frame_cell[0] = callback
            return prior

        def set_eval_frame_isolate_recompiles_id(recompiles_id):
            prior = _eval_frame_isolate_cell[0]
            _eval_frame_isolate_cell[0] = recompiles_id
            return prior

        set_guard_error_hook.__name__ = set_guard_error_hook.__qualname__ = (
            "set_guard_error_hook"
        )
        set_code_exec_strategy.__name__ = set_code_exec_strategy.__qualname__ = (
            "set_code_exec_strategy"
        )
        set_eval_frame.__name__ = set_eval_frame.__qualname__ = "set_eval_frame"
        set_eval_frame_isolate_recompiles_id.__name__ = (
            set_eval_frame_isolate_recompiles_id.__qualname__
        ) = "set_eval_frame_isolate_recompiles_id"
        eval_frame.set_guard_error_hook = set_guard_error_hook
        eval_frame.set_code_exec_strategy = set_code_exec_strategy
        eval_frame.set_eval_frame = set_eval_frame
        eval_frame.set_eval_frame_isolate_recompiles_id = (
            set_eval_frame_isolate_recompiles_id
        )

        dynamo_sub = getattr(module, "_dynamo")
        setattr(dynamo_sub, "eval_frame", eval_frame)
        sys.modules[f"{prefix}._dynamo.eval_frame"] = eval_frame

    # -- _VariableFunctions ----------------------------------------------
    #
    # VENDOR.md wall 5. `torch.add`, `torch.mm`, `torch.full` and ~620 more
    # public names are written nowhere in the Python tree; `torch/__init__.py`
    # harvests them off this one object at import time and assigns
    # `__module__` on each. Instance attributes, not class attributes: a
    # function reached through a class becomes a bound method and
    # `method.__module__` is read-only.
    class _VariableFunctionsHolder:
        """Enumerable, non-binding, and open at the edges.

        *Enumerable* because `torch/__init__.py:2212` harvests `dir()` of this
        object to create `torch.add` and 620 more. *Non-binding* because every
        function lives in the instance `__dict__`, not the class: reached
        through a class a function becomes a bound method and
        `method.__module__` is read-only, which that same loop assigns.
        *Open* because the stub lists 976 names and upstream has 985 --
        `torch/jit/_builtins.py:117` wants `torch._VF.stft`, which is not one
        of the 976. A member of this object is a hole, never a switch, so the
        catch-all costs nothing (rule 2).
        """

        __module__ = "torch._C"

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            fn = _torch_level_function(name, dispatch, overloads)
            setattr(self, name, fn)
            return fn

    varfns = _VariableFunctionsHolder()
    for name in surface["varfns"]:
        if name.startswith("__"):
            continue
        setattr(varfns, name, _torch_level_function(name, dispatch, overloads))
    # `torch.div(int, int, rounding_mode=...)` -- docs/bindings/ARGFORM.md,
    # `longformer`'s wall. Measured with a `TorchDispatchMode` logger on
    # 2.13.0: `torch.div(7, 2, rounding_mode='trunc')` reaches
    # `aten.div.Tensor_mode` -- upstream wraps a bare Python number into a
    # 0-dim tensor wherever a schema position wants `Tensor` and gets a
    # `Scalar` instead ("wrapped number" promotion), which every overload
    # table entry here already assumes never happens for `self` (the table's
    # `Tensor self` position has no `Scalar self` sibling, because nothing
    # else needed one).
    #
    # `div` needs it because `torch.div(a, b, ...)` has no requirement that
    # `a` be a Tensor upstream, unlike (measured) `torch.add`/`torch.mean`/
    # most everything else that takes a leading `Tensor self` -- `torch.mean(5)`
    # refuses upstream with "argument 'input' ... must be Tensor, not int",
    # so this wrapping is written for `div` alone rather than generalised into
    # the table-driven resolver, which would make every `Tensor self` position
    # accept a bare number regardless of whether upstream does.
    #
    # The dtype a wrapped number gets is upstream's own rule too (measured):
    # a Python `bool` wraps to `torch.bool`, an `int` to `torch.int64`, a
    # `float` to `torch.float32` -- and from there the real `aten.div.*`
    # kernel does upstream's own type promotion (true division promotes an
    # int/int divide to float32 even with no `rounding_mode`), so nothing
    # about *how* the divide computes is decided here.
    def _wrap_number_self(value):
        if isinstance(value, module.TensorBase):
            return value
        if isinstance(value, bool):
            return dispatch("aten.scalar_tensor.default", value, dtype=module.bool)
        if isinstance(value, int):
            return dispatch("aten.scalar_tensor.default", value, dtype=module.int64)
        if isinstance(value, float):
            return dispatch("aten.scalar_tensor.default", value, dtype=module.float32)
        return value

    _table_div = varfns.div

    def div(input, other, *args, **kwargs):
        if not isinstance(input, module.TensorBase) and isinstance(
            input, (bool, int, float)
        ):
            input = _wrap_number_self(input)
            if not isinstance(other, module.TensorBase) and isinstance(
                other, (bool, int, float)
            ):
                other = _wrap_number_self(other)
        return _table_div(input, other, *args, **kwargs)

    div.__name__ = div.__qualname__ = "div"
    varfns.div = div

    # `torch.zeros((..., 0-dim int Tensor, ...), dtype=..., device=...)` --
    # `fastspeech2_conformer`'s wall (docs/kernels/TAIL4.md §8.2, docs/bindings/BIND4.md).
    # `FeatureProjection`/the length-regulator forward builds `max_len =
    # torch.sum(duration_labels, dim=1).max()`, a 0-dim Tensor, and then
    # writes `torch.zeros((batch, max_len, dim), dtype=..., device=...)`
    # straight from it -- upstream never narrows it to a Python int first.
    #
    # Measured against real torch 2.13.0, in a separate process: upstream's
    # `SymInt[]` argument parser accepts a 0-dim integral Tensor as a list
    # element and takes its value, exactly the way it already accepts a bare
    # Python int there. This is NOT a `zeros`-specific accident -- the same
    # call shape was measured to work for `torch.ones`, `torch.empty` and
    # `Tensor.view`, which all take the same `SymInt[]`/`SymInt[]` position --
    # so it is upstream's general size-list parsing rule. It is still written
    # only for `zeros`, the same way `div`'s wrapped-number rule directly
    # above is scoped to `div` alone rather than folded into `_TypeChecker`:
    # that would accept a Tensor in *every* `SymInt[]` position table-wide
    # without a matching measurement for each of them, which is exactly the
    # silent-divergence trap docs/bindings/ARGFORM.md §2 names.
    #
    # The refusals were the half this comment used to hand-wave -- a
    # multi-element Tensor was "left to fail on its own `__int__`", and a
    # FLOAT or BOOL Tensor was not considered at all, so `torch.zeros((2,
    # tensor(3.5)))` quietly gave a (2, 3) tensor where upstream raises.
    # `_coerce_symint_size_tensors` now carries upstream's three lines and its
    # message text; docs/bindings/BIND5.md §1 has the measurement.
    _table_zeros = varfns.zeros

    def zeros(size, *args, **kwargs):
        # `_torch_level_function`'s own guard, reproduced rather than
        # inherited: `torch.zeros` is one of the 36 names
        # `DeviceContext.__torch_function__` matches by object identity
        # (`func in _device_constructors()`, `torch/utils/_device.py`), read
        # fresh off the `torch` module at call time -- which is now THIS
        # closure, not the table-driven one `_table_zeros` closes over. If
        # `zeros` skipped straight to `_table_zeros`, that inner closure's own
        # `_MODE_STACK` check would fire with *itself* as `func`, which is no
        # longer the object `_device_constructors()` finds at `torch.zeros`,
        # and `with torch.device("meta"): torch.zeros(2)` would silently go
        # back to returning a CPU tensor -- docs/devices/DEVICE_ABS.md §7.2's exact
        # failure, caught by `test_meta_road_through_the_vendored_tree` the
        # first time this wrapper was written without the guard.
        if _MODE_STACK:
            return _through_torch_function_modes(zeros, (size,) + args, kwargs)
        return _table_zeros(
            _coerce_symint_size_tensors(module, "zeros", size), *args, **kwargs
        )

    zeros.__name__ = zeros.__qualname__ = "zeros"
    varfns.zeros = zeros

    # `torch.tensor` is the one name on this object that is not an overload set
    # -- see `_tensor_factory`.
    varfns.tensor = _tensor_factory(module, dispatch)
    # `torch.frombuffer` is the second, and for the same reason: upstream has no
    # `aten::frombuffer` at all (`torch.ops.aten.frombuffer` raises
    # `AttributeError` on 2.13.0), only the `_C` binding. The loop above had
    # installed a refusal that pointed the caller at
    # `torch.ops.aten.frombuffer.<overload>` -- a work item nobody could ever
    # close, because there is nothing there to reach. Point it at the real
    # implementation instead. It is the entire cost of the safetensors load
    # path; see `_frombuffer` in lib.rs and docs/models/CKPT.md.
    varfns.frombuffer = module._frombuffer
    # `torch.asarray` is the third, and it is `frombuffer`'s sibling in every
    # respect: no `aten::asarray` exists either (`torch.ops.aten.asarray` raises
    # `AttributeError` on 2.13.0), so the refusal the loop installed named a
    # work item that could never be closed. It is what safetensors' *default*
    # backend calls -- `frombuffer` serves the `pread` and bytes backends -- so
    # between the two, every safetensors route reaches a real reader. See
    # `_asarray` in lib.rs and docs/models/CKPT2.md §4.
    varfns.asarray = module._asarray
    # `torch.get_default_dtype` is the fourth. It is not an aten op at all --
    # upstream binds `THPModule_getDefaultDtype` straight onto `_C`
    # (`torch/_C/__init__.pyi:1399`) *and* lists it among the variable
    # functions, so overload resolution was never going to find a table entry
    # and the refusal it produced named a `torch.ops.aten.get_default_dtype`
    # that does not exist. `torch/distributed/_shard/sharded_tensor/
    # metadata.py:20` calls it in a dataclass field default, at import time.
    #
    # The value is not a free choice here: it has to be the same dtype
    # `dtype.rs`'s `default_float()` gives factory functions, or a caller
    # reading this to decide what it will get would be told the wrong thing.
    # That used to be arranged by installing a *constant* function returning
    # `module.float32`, because the value lived in a Rust `const` and
    # `set_default_dtype` refused by name for want of anywhere to write.
    #
    # It is a process-global now (`_set_default_dtype`, which
    # `torch/__init__.py:1385` forwards to and `transformers` calls on the way
    # into `from_pretrained`), so both names come from Rust and read the same
    # cell. A constant function here would have re-introduced the divergence
    # from the other side: `torch.get_default_dtype()` still saying float32
    # after the factories had moved.
    varfns.get_default_dtype = module.get_default_dtype
    module._VariableFunctions = varfns
    module._VariableFunctionsClass = type(varfns)
    module._TensorBase = module.TensorBase
    _install_grad_mode(module, varfns)

    # -- `_C._nn`, and the two composites that are not overload sets -------
    _install_nn(module, dispatch)
    # Bound once here rather than looked up per call: it is read on the hot
    # path in `_torch_level_function` and `_tensor_method`, and it is a Rust
    # builtin that nothing replaces.
    global _CAPTURE_ACTIVE
    _CAPTURE_ACTIVE = module._capture_active
    _install_composites(module, varfns, dispatch)

    # -- the enum instances `_initExtension` writes into `torch` -----------
    _install_namespace_types(module, surface["namespace"])

    # -- module-level names ------------------------------------------------
    for name, kind in surface["module"].items():
        if name in off or hasattr(module, name):
            continue
        if kind == "function":
            setattr(module, name, _make_function(f"torch._C.{name}", name))
        elif kind == "bool":
            # A build-configuration flag. `_BUILD_FLAGS` installs the value
            # later (after `_install_behaviour`); what happens here is the
            # check that there *is* one. Falling through to a placeholder
            # would put a truthy object in a branch, which is the bug this
            # kind exists to prevent -- so the surface refuses to build
            # instead, naming the flag that needs deciding.
            if name not in _BUILD_FLAGS:
                raise RuntimeError(
                    f"torch._C shim: the stubs declare {name} as a build "
                    "flag (_bool) and _BUILD_FLAGS has no answer for it. A "
                    "flag has two answers and both change behaviour; decide "
                    "one in bootstrap.py rather than leaving a placeholder, "
                    "which is truthy."
                )
        elif name.lstrip("_")[:1].isupper():
            # Capitalised means a type, and it has to really be one: the names
            # the stubs miss turn up in *annotations*, which Python evaluates
            # at function definition time. `torch/nn/functional.py:7170` is
            # `scale_recipe_a: ScalingType | list[ScalingType]`, and `|` on a
            # non-type is a TypeError -- a placeholder object cannot stand in.
            setattr(module, name, _ShimMeta(
                name, (), {"__module__": "torch._C", "__init__": _permissive_init}))
        else:
            setattr(module, name, _Unimplemented(f"torch._C.{name}"))

    _install_legacy_tensor_types(module, dispatch)

    # The overload tables, re-keyed by aten `(qualname, overload)` so that
    # `_get_schema` can answer from them. They are already parsed -- one
    # `_Schema` per entry, built at the top of this function -- and their text
    # is upstream's own `str(op._schema)`, so nothing is re-derived here.
    #
    # Keyed off the parsed schema rather than off the table key, which is the
    # only thing that works for both: `overloads.json`'s key is the op name but
    # `methods.json`'s is the Python method name, and the two differ (`item` is
    # `_local_scalar_dense`, `__mul__` is `mul`). Same rule as
    # `verify_schemas.py:_aten_name`.
    transcribed = {}
    for entry in list(overloads.values()) + list(methods.values()):
        for schema in entry.schemas:
            transcribed.setdefault((schema.name, schema.overload_name), schema)

    _install_behaviour(module, dispatch, transcribed)
    _install_torch_function_modes(module)
    _install_device(module, varfns, module.TensorBase)
    _install_repr_surface(module, varfns, module.TensorBase)
    _install_serialization(module)

    # ---- docs/graph/EXPORT.md §8's hand-off ------------------------------------
    #
    # **Last, and the position is load-bearing.** These eight install the 32
    # `torch.export` census names over the placeholders that the stub tables
    # and the submodule synthesiser leave behind, so they have to run after
    # every one of those. The first attempt put them beside
    # `_install_dispatch_keys` -- which is where docs/graph/EXPORT.md §8 proposed
    # them -- and `_install_dynamo_bool` was silently undone: the name was in
    # `torch._C._dynamo.guards.__dict__` afterwards and its value was the
    # `_Unimplemented` the submodule pass had written over it. Nothing raised;
    # `torch.export` simply stopped at that name again. That is the same
    # install-order hazard as the `_len_torch_dispatch_stack` table row §8 item
    # 3 deletes, arriving from the other side, and it is why this block is
    # here rather than three thousand lines earlier.
    _install_tls(module, _put)
    _install_mode_stack(module, _put)
    _install_tensor_predicates(module, _put)
    _install_functorch(module, _put)
    _install_profiler(module, _put)
    _install_dynamo_bool(module, _put)
    _install_inference_mode(module, _put)
    _install_raii_guards(module, _put)

    # PyO3 emits `__all__` on `#[pymodule]` modules, so `from torch._C import *`
    # -- which is how most of the `torch` namespace comes into being
    # (`torch/__init__.py:445`) -- copies only what is listed. Setting an
    # attribute is not enough (VENDOR.md wall 6).
    module.__all__ = sorted(
        n for n in vars(module) if not n.startswith("_") and n not in off
    )


# ---------------------------------------------------------------------------
# Reading a checkpoint: the `torch.save` zip container
# ---------------------------------------------------------------------------
#
# This is the one place in this file that parses a format, and the module
# docstring's "nothing here computes" deserves an answer rather than an
# exception.
#
# The rule is about *tensor* computation: no arithmetic may happen outside the
# one door in `aten.rs`, because that door is DESIGN.md §6's instrument. Nothing
# below touches a tensor -- it locates byte ranges in a container and hands them
# to `StorageBase`, which is in Rust with the invariant that matters (see
# storage.rs). What is left here is container parsing, and it is here because
# the container is an ordinary zip archive and CPython ships a correct zip
# reader. The alternatives were a hand-written zip parser in Rust, or a new
# crate dependency on all three cross-compiled targets; both are worse trades
# for a format the standard library already reads, and neither buys anything
# `zipfile` does not already give.
#
# What is *not* here is as deliberate as what is. `get_record_offset_no_read`
# and the mmap path are left refusing, because closing them means reproducing
# torch's record-alignment arithmetic, and a wrong offset does not raise -- it
# reads the neighbouring tensor's bytes. docs/models/CKPT.md §6.


class _RecordHolder:
    """What `PyTorchFileReader.get_storage_from_record` hands back.

    Upstream returns a **Tensor** -- `at::empty({...}).set_(storage)` -- and
    `torch/serialization.py:2128` immediately unwraps it again with
    `._typed_storage()._untyped_storage`. That round trip only makes sense
    because upstream's tensor is a *view* of the storage, so wrapping and
    unwrapping is free and loses nothing.

    Here it would not be free: this shim's tensors copy (storage.rs), so
    returning a real Tensor would copy every byte of the checkpoint into a
    tensor that exists only to be thrown away, and -- worse -- the storage that
    came back out of it would have to be a second copy to be honest about not
    aliasing. So the two calls the caller actually makes are answered directly,
    and the object says what it is rather than pretending to be a Tensor.
    """

    __slots__ = ("_typed",)

    def __init__(self, typed):
        self._typed = typed

    def _typed_storage(self):
        return self._typed


class _ZipRecords:
    """`torch._C.PyTorchFileReader` over CPython's `zipfile`.

    Upstream this is miniz behind `caffe2/serialize/inline_container.cc`. The
    names below are the ones `torch/serialization.py` actually calls, measured
    by running `torch.load` against a reader that logged every attribute it was
    asked for; everything else keeps the type catch-all's refusal.
    """

    def __init__(self, name_or_buffer):
        import zipfile

        self._zf = zipfile.ZipFile(name_or_buffer)
        names = self._zf.namelist()
        if not names:
            raise RuntimeError("torch._C shim: empty archive, no records")
        # Every record lives under one top-level directory named for the
        # archive; upstream derives it the same way, from the first entry.
        head = names[0].split("/")[0]
        self._prefix = f"{head}/" if f"{head}/" == names[0][: len(head) + 1] else ""

    def _full(self, name):
        return self._prefix + name

    def has_record(self, name):
        try:
            self._zf.getinfo(self._full(name))
        except KeyError:
            return False
        return True

    def get_record(self, name):
        return self._zf.read(self._full(name))

    def get_all_records(self):
        n = len(self._prefix)
        return [entry[n:] for entry in self._zf.namelist()]

    def get_record_offset(self, name):
        """Where this record's payload starts in the file.

        Read out of the local file header on disk rather than recomputed with
        `ZipInfo.FileHeader()`: `torch.save` pads the header's *extra* field to
        align payloads (`.storage_alignment`, 64 bytes by default), and a
        regenerated header does not carry that padding. The two answers differ
        by the padding, silently, and the number is used to slice storages.
        """
        info = self._zf.getinfo(self._full(name))
        fp = self._zf.fp
        here = fp.tell()
        try:
            fp.seek(info.header_offset)
            header = fp.read(30)
            if header[:4] != b"PK\x03\x04":
                raise RuntimeError(
                    f"torch._C shim: no local file header for record {name!r} "
                    f"at offset {info.header_offset}"
                )
            name_len = int.from_bytes(header[26:28], "little")
            extra_len = int.from_bytes(header[28:30], "little")
        finally:
            fp.seek(here)
        return info.header_offset + 30 + name_len + extra_len

    def get_record_header_offset(self, name):
        return self._zf.getinfo(self._full(name)).header_offset

    def get_storage_from_record(self, name, nbytes, cls):
        """The payload of one record, as a storage.

        Note the order this establishes, which `TensorBase.set_` depends on and
        storage.rs explains: the bytes are in the storage *before* the storage
        is handed back, so by the time `_rebuild_tensor` calls `set_` there is
        something to copy. The legacy container does it the other way round,
        which is why this shim reads one format and refuses the other.
        """
        import torch

        data = self._zf.read(self._full(name))
        if len(data) != nbytes:
            raise RuntimeError(
                f"torch._C shim: record {name!r} is {len(data)} bytes, but the "
                f"checkpoint's pickle says {nbytes}"
            )
        storage = cls(nbytes)
        storage._shim_fill(data)
        return _RecordHolder(
            torch.storage.TypedStorage(
                wrap_storage=storage, dtype=torch.uint8, _internal=True
            )
        )

    def serialization_id(self):
        """Upstream writes this record so two checkpoints can be compared
        without reading them. It is optional, and `torch/serialization.py` uses
        it only for a telemetry callback, so an absent one is an empty string
        rather than an error."""
        try:
            return self._zf.read(self._full(".data/serialization_id")).decode()
        except KeyError:
            return ""


# The zip container's *extra field* header id torch pads local file headers
# with. Measured off a 2.13.0 archive: every record's extra field is
# `46 42 <len-4 LE> 5a 5a ...` -- id 0x4246, then that many `Z` bytes -- and
# every payload consequently starts on a 64-byte boundary. It is the id
# torch's miniz fork uses for exactly this, and it is written here so an
# archive from this shim has the same shape as one from upstream rather than
# merely the same contents.
_ZIP_ALIGN_EXTRA_ID = 0x4246

# What `zipfile` adds to a local file header when the entry needs zip64, and
# the condition it adds it under (CPython 3.13 `ZipFile._open_to_write`:
# `zip64 = force_zip64 or (zinfo.file_size * 1.05 > ZIP64_LIMIT)`, and
# `ZipInfo.FileHeader(zip64)` then prepends a 20-byte extra field). It is
# reproduced rather than ignored because the alignment arithmetic below has to
# know the header's real length -- getting it wrong would not raise, it would
# put every payload of a >2 GB checkpoint 20 bytes off the boundary the
# `.storage_alignment` record claims.
_ZIP64_EXTRA_LEN = 20
_ZIP64_LIMIT = (1 << 31) - 1


class _ZipWriter:
    """`torch._C.PyTorchFileWriter` over CPython's `zipfile`.

    The writing half of `_ZipRecords`, and it is here for the same reason that
    one is (see the section comment above): the container is an ordinary zip
    archive, nothing in it computes, and the standard library already has a
    correct zip writer. What this class owns is the two things `zipfile` does
    not know about -- torch's record *naming* and torch's payload *alignment*.

    **`write_end_of_file` is the method the traceback names when a save
    fails, and it is almost never the method at fault.**
    `torch/serialization.py:1002` puts the whole of `_save` inside
    `with _open_zipfile_writer(f) as z:`, and `__exit__` (:854) calls this
    unconditionally. An exception from inside the block runs `__exit__`, which
    raises its own, and the second one is what propagates -- so before this
    class existed, *every* failure anywhere on the save path was reported as
    `PyTorchFileWriter.write_end_of_file`. docs/models/SAVE.md §1.

    The three records `_save` does not write are written here, matching where
    upstream's C++ writer writes them:

    | record | who | value |
    |---|---|---|
    | `version` | `writeEndOfFile()` | `3\\n`, upstream's, and the C++ *reader* validates it |
    | `.data/serialization_id` | `writeEndOfFile()` | **not written** -- see `serialization_id` |
    | everything else | `_save` | unchanged |
    """

    def __init__(self, name_or_buffer, compute_crc32=True, storage_alignment=64):
        import os
        import zipfile

        self._own_stream = None
        if isinstance(name_or_buffer, (str, bytes, os.PathLike)):
            path = os.fsdecode(name_or_buffer)
            self._own_stream = open(path, "wb")  # noqa: SIM115 -- closed in write_end_of_file
            stream = self._own_stream
            # Upstream names the archive after the file, extension stripped:
            # `x.pt` -> `x/data.pkl`, `a.b.c.pt` -> `a.b.c/data.pkl`,
            # `no_ext` -> `no_ext/data.pkl` (all measured on 2.13.0). Nothing
            # reads the name back -- both readers derive the prefix from the
            # first entry -- but a file that says `archive/` where upstream
            # says `x/` is a gratuitous difference in a format other tools
            # inspect.
            self._archive = os.path.splitext(os.path.basename(path))[0]
        else:
            stream = name_or_buffer
            self._archive = "archive"

        self._align = max(1, int(storage_alignment))
        # Accepted and not honoured *downward*: `zipfile` always computes the
        # CRC, so `compute_crc32=False` gets a correct checksum where upstream
        # would write zero. That is a superset -- no reader can distinguish a
        # correct CRC from a required one -- and the alternative would be
        # hand-writing the local headers to put a deliberate zero in them.
        self._compute_crc32 = bool(compute_crc32)
        self._zf = zipfile.ZipFile(stream, "w", zipfile.ZIP_STORED, allowZip64=True)
        self._written = []
        # `(payload offset, payload length)` per record written, in order. See
        # `_next_header_offset` -- the alignment arithmetic needs it and
        # `ZipInfo.__slots__` will not hold it.
        self._layout = []
        self._ended = False

    def _bytes_of(self, name, data, size):
        """The payload, from any of the three shapes `_save` passes."""
        if isinstance(data, str):
            payload = data.encode("utf-8")
        elif isinstance(data, (bytes, bytearray, memoryview)):
            payload = bytes(data)
        else:
            # A storage. `torch/serialization.py:1312` hands the object itself
            # (upstream's writer reads through `data_ptr()`); here the bytes
            # come out of the storage directly, which is also why `data_ptr()`
            # is free to be an identity rather than a readable address --
            # nothing on this path reads through it (storage.rs).
            reader = getattr(data, "_shim_bytes", None)
            if reader is None:
                raise NotImplementedError(
                    "torch._C shim: PyTorchFileWriter.write_record was handed a "
                    f"{type(data).__name__} for record {name!r}. It writes str, "
                    "bytes, and this shim's storages; upstream additionally "
                    "accepts anything exposing a data pointer, which cannot be "
                    "read from Python"
                )
            payload = bytes(reader())
        if size is not None and len(payload) != size:
            # Not a formality. `_save` computes `size` from `storage.nbytes()`
            # and the reader trusts the *pickle's* number, so a disagreement
            # here is a record whose payload does not match what will be asked
            # of it -- which reads back as the neighbouring tensor's bytes
            # rather than as an error (docs/models/CKPT.md §6).
            raise RuntimeError(
                f"torch._C shim: PyTorchFileWriter.write_record({name!r}) was "
                f"told {size} bytes and given {len(payload)}"
            )
        return payload

    def write_record(self, name, data, size=None, compress=False):
        import struct
        import zipfile

        if compress:
            raise NotImplementedError(
                "torch._C shim: PyTorchFileWriter.write_record(compress=True) "
                "-- torch writes checkpoints uncompressed and both readers "
                "here assume ZIP_STORED"
            )
        payload = self._bytes_of(name, data, size)
        full = f"{self._archive}/{name}"
        # A fixed timestamp, where upstream writes the wall clock. It makes a
        # save byte-reproducible: the same object saved twice gives the same
        # file, which is what lets a federated delta be addressed by its hash
        # (README, docs/design/DESIGN.md §10). 1980-01-01 is the zip epoch, the
        # earliest value the format can carry.
        info = zipfile.ZipInfo(full, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o600 << 16

        # Pad the *local* header's extra field so the payload lands on the
        # alignment `_save` advertised in the `.storage_alignment` record. A
        # header is 30 bytes plus the name plus the extra, and the extra field
        # itself has a 4-byte head -- so a padding of 1..3 is not expressible
        # and has to become one whole alignment more.
        zip64 = _ZIP64_EXTRA_LEN if len(payload) * 1.05 > _ZIP64_LIMIT else 0
        fixed = 30 + len(full.encode("utf-8")) + zip64
        pad = (-(self._next_header_offset() + fixed)) % self._align
        if pad and pad < 4:
            pad += self._align
        if pad:
            info.extra = struct.pack("<HH", _ZIP_ALIGN_EXTRA_ID, pad - 4) + b"Z" * (pad - 4)

        self._zf.writestr(info, payload)

        # Upstream's central directory carries no padding -- only the local
        # header does (measured: `cd_extra=0`, `local_extra=18` and up). The
        # entry is on disk by now, so clearing it here affects only what
        # `close()` will put in the directory.
        info.extra = b""

        # Recorded, not recomputed: this is what the *next* record's alignment
        # is measured from, and `ZipInfo` has `__slots__` so it cannot carry it.
        payload_at = info.header_offset + fixed + pad
        self._layout.append((payload_at, len(payload)))
        if payload_at % self._align != 0:
            # Not reachable for any input this file can construct, and checked
            # anyway because the failure is silent: a `.storage_alignment`
            # record that says 64 over payloads that are not aligned is a file
            # that lies about itself, and no reader would complain.
            raise RuntimeError(
                "torch._C shim: PyTorchFileWriter could not align record "
                f"{name!r} -- its payload starts at {payload_at}, which is not "
                f"a multiple of {self._align}"
            )
        self._written.append(name)

    def _next_header_offset(self):
        """Where `zipfile` will put the next local file header.

        Computed from the last record's own payload position and length rather
        than read off `ZipFile.start_dir`, which is private. A stored entry on a
        seekable stream is header + payload and nothing else -- no data
        descriptor -- so the next header begins where the last payload ends.
        """
        if not self._layout:
            return self._zf.fp.tell()
        payload_at, payload_len = self._layout[-1]
        return payload_at + payload_len

    def write_record_metadata(self, name, numbytes):
        """`torch.serialization.skip_data` -- a record's header with no bytes.

        Refused rather than filled with zeros. The whole point of the context
        manager is to write a checkpoint whose payloads are supplied later by
        something else; a shim that quietly wrote `numbytes` zeros would produce
        exactly the file docs/models/CKPT.md §4 spent a section on -- one that loads,
        matches every key, and is all zeros.
        """
        raise NotImplementedError(
            "torch._C shim: PyTorchFileWriter.write_record_metadata "
            f"({name!r}, {numbytes} bytes) -- this is the "
            "torch.serialization.skip_data path, which writes a record header "
            "with no payload for a caller to fill in afterwards. It is refused "
            "rather than zero-filled (docs/models/SAVE.md §6)"
        )

    def write_end_of_file(self):
        """Close the archive. Idempotent -- `_opener.__exit__` calls it on the
        way out of a `with` block that may already have failed."""
        if self._ended:
            return
        self._ended = True
        # Upstream's C++ `writeEndOfFile()` writes this, not `_save`, and its
        # C++ *reader* validates the number on the way in
        # (`kMinSupportedFileFormatVersion`). Measured: a 2.13.0 archive holds
        # exactly `b"3\n"`.
        if "version" not in self._written:
            self.write_record("version", "3\n", 2)
        self._zf.close()
        if self._own_stream is not None:
            self._own_stream.close()

    def get_all_written_records(self):
        return list(self._written)

    def archive_name(self):
        return self._archive

    def serialization_id(self):
        """**Not written, and this is the one place where writing something
        would be worse than writing nothing.**

        Upstream's `writeEndOfFile()` emits a `.data/serialization_id` record:
        forty digits, two twenty-digit halves, a combined hash of the record
        names and a combined CRC of their contents. The hash half is
        `std::hash<std::string>`, which is implementation-defined -- there is
        no portable way to compute the same number, so any id produced here
        would differ from upstream's *for identical content*. Its only consumer
        is a telemetry callback (`torch/serialization.py:2226`) that compares
        ids to decide whether two checkpoints are the same file, and a
        deterministic wrong answer to that question is worse than no answer:
        absent, both readers report `""` (see `_ZipRecords.serialization_id`).
        """
        return ""

    def __getattr__(self, name):
        """The rest of upstream's writer surface, refusing by name.

        This class replaces a synthesised `_C` type, and those answer an
        unknown member through `_ShimMeta` rather than with an `AttributeError`
        (rule 2 in this file's docstring: a missing member of a type is a hole,
        not a switch). Keeping that here means a caller reaching for
        `set_min_version` or `write_record_metadata`'s neighbours gets a work
        item instead of a name error.

        Dunders are excluded and get the ordinary `AttributeError`: a
        `NotImplementedError` from a `__deepcopy__` or `__getstate__` probe
        would turn a question into a failure.
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise NotImplementedError(
            f"not implemented in torch._C shim: PyTorchFileWriter.{name}"
        )


def _install_serialization(module) -> None:
    module.PyTorchFileReader = _ZipRecords
    module.PyTorchFileWriter = _ZipWriter

    # `torch/_utils.py:290 _validate_loaded_sparse_tensors` runs at the end of
    # every `torch.load`, and asks this before it looks at what was loaded.
    # `False` is upstream's default state (the checks are opt-in, via
    # `torch.sparse.check_sparse_tensor_invariants`), and it is the honest
    # answer here for a second reason: this shim has no sparse tensors, so the
    # list those checks would run over is always empty.
    module._check_sparse_tensor_invariants = lambda: False

    def _get_tensor_metadata(tensor):
        """`torch/_utils.py:get_tensor_metadata`, read once per tensor by
        `_reduce_ex_internal` (`torch/_tensor.py:531`).

        Upstream's answer for an ordinary tensor is `{}` (measured on 2.13.0),
        and `_save` appends the metadata to the record only `if metadata:` --
        so an empty dict means "this tensor has none", which is true of every
        tensor this shim can make. It is not a stub standing in for something:
        the metadata upstream can put here is set by
        `torch._C._set_tensor_metadata`, which stays refused, so there is
        nothing that could have put a key in it.
        """
        if not isinstance(tensor, module.TensorBase):
            raise TypeError(
                "torch._C shim: _get_tensor_metadata expected a tensor, got "
                f"{type(tensor).__name__}"
            )
        return {}

    module._get_tensor_metadata = _get_tensor_metadata


# ---------------------------------------------------------------------------
# The torch-function mode stack
# ---------------------------------------------------------------------------
#
# `with torch.device("meta"):` is not a device feature. Upstream implements it
# as a `TorchFunctionMode` -- `torch/utils/_device.py:DeviceContext` -- pushed
# onto a stack that every factory function consults, and `torch.device.__enter__`
# in `torch/csrc/Device.cpp` does nothing but push it. So the context manager
# and `torch.set_default_device` are the same mechanism wearing two names, and
# neither can be built without the stack.
#
# **The stack alone would have been worse than nothing**, and that is the whole
# reason this file grew a mode dispatch as well. docs/devices/DEVICE_ABS.md §7.2 said
# it in advance: push a `DeviceContext` and never consult it and
# `with torch.device("meta"): torch.zeros(2)` returns a *CPU* tensor, silently,
# with the block appearing to work. A `NotImplementedError` is better than a
# wrong answer, so either both halves land or neither does.
#
# Everything above `_C` is already vendored and needs nothing written here:
# `torch/overrides.py` has `TorchFunctionMode`, `_push_mode`, `_pop_mode` and
# `_get_current_function_mode_stack`; `torch/utils/_device.py` has
# `DeviceContext`; `torch/__init__.py` has `set_default_device` and
# `get_default_device`. All five `_C` names they bottom out in are below.
#
# **The stack is process-wide where upstream's is thread-local.**
# `PythonTorchFunctionTLS` is per-thread; this is one list. Recorded rather than
# fixed: a mode entered on one thread would be seen by another, which upstream
# would not do. Nothing in this shim's measured paths is multi-threaded, and the
# fix (a `threading.local`) is a two-line change in this block if that stops
# being true.
_MODE_STACK: list = []


def _through_torch_function_modes(func, args, kwargs):
    """Run `func` under the innermost mode, upstream's way.

    The top mode is **popped for the duration**, which is not bookkeeping but
    the thing that makes modes work at all: every `__torch_function__`
    implementation ends by calling `func(*args, **kwargs)` again, and without
    the pop that call would find the same mode still on top and recurse
    forever. Upstream does the same with a `StashTorchFunctionModeGuard`.

    `types` is passed empty. Upstream fills it with the argument types that
    override `__torch_function__`, and this shim answers `False` to every
    `_has_torch_function*` predicate for the reason recorded above
    `_DISCOVERED_RETURNS` -- no type in the vendored tree overrides it. So the
    honest tuple is the empty one, and `DeviceContext.__torch_function__` (the
    only mode that reaches here today) does not read it.
    """
    mode = _MODE_STACK.pop()
    try:
        return mode.__torch_function__(func, (), args, kwargs)
    finally:
        _MODE_STACK.append(mode)


def _install_torch_function_modes(module) -> None:
    """The five `_C` names `torch/overrides.py` bottoms out in."""

    def _push_on_torch_function_stack(mode):
        _MODE_STACK.append(mode)

    def _pop_torch_function_stack():
        if not _MODE_STACK:
            # Upstream raises here too; `DeviceContext.__exit__` relies on the
            # stack being non-empty and would otherwise unwind past its own
            # entry silently.
            raise RuntimeError(
                "torch._C shim: pop from an empty torch function mode stack"
            )
        return _MODE_STACK.pop()

    def _len_torch_function_stack():
        return len(_MODE_STACK)

    def _get_function_stack_at(index):
        return _MODE_STACK[index]

    def _is_torch_function_mode_enabled():
        return bool(_MODE_STACK)

    # `_is_torch_function_enabled` is a *different* question from the one above
    # and keeps its old answer: upstream returns whether the torch-function
    # protocol is enabled for subclasses at all, which is what gates the
    # `has_torch_function*` predicates. No type here overrides
    # `__torch_function__`, so `False` stays the fact -- see the note above
    # `_DISCOVERED_RETURNS`. Only the *mode* half became real.
    def _is_torch_function_enabled():
        return False

    for fn in (
        _push_on_torch_function_stack,
        _pop_torch_function_stack,
        _len_torch_function_stack,
        _get_function_stack_at,
        _is_torch_function_mode_enabled,
        _is_torch_function_enabled,
    ):
        fn.__qualname__ = fn.__name__
        fn.__module__ = "torch._C"
        setattr(module, fn.__name__, fn)

    # `DisableTorchFunctionSubclass`, the sixth, and a context manager rather
    # than a function -- `surface.json` harvested it from the `.pyi` as
    # `"function"`, so it became a raising stub, and
    # `torch.autograd.profiler.record_function.__exit__` opens with
    # `with torch._C.DisableTorchFunctionSubclass():` (docs/training/LOSS.md §6). So it
    # sits on the road to `optimizer.zero_grad()` twice over: once for the
    # marker and once for this.
    #
    # **A no-op is the correct body, not a stand-in**, and the line above says
    # why: what it disables is subclass `__torch_function__` dispatch, and
    # `_is_torch_function_enabled()` already returns `False` here because no
    # type in the vendored tree overrides the protocol. Entering it turns off
    # something already off. It is deliberately NOT wired to `_MODE_STACK`:
    # that is the *mode* stack, which `DisableTorchFunction` (the other name)
    # governs upstream, and clearing it here would silently drop a
    # `with torch.device(...)` block that happened to span a `record_function`.
    class _DisableTorchFunctionSubclass:
        __module__ = "torch._C"
        __qualname__ = "DisableTorchFunctionSubclass"
        __slots__ = ()

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    _DisableTorchFunctionSubclass.__name__ = "DisableTorchFunctionSubclass"
    module.DisableTorchFunctionSubclass = _DisableTorchFunctionSubclass

    # Readable for the same reason as `_shim_overloads` and
    # `_shim_registrations`: which `torch.ops` keys this build answers *above*
    # the one door should be answerable by asking, not by reading bootstrap.py.
    # There are two ops and three keys, and the list being short is the point.
    module._shim_profiler_markers = sorted(_PROFILER_MARKERS)



def _fast_symint_coerce(value):
    """`resolve`'s scalar `int`/`SymInt` coercion, for the generated fast path.

    Anything that is not a Tensor is passed through untouched -- an `int`
    subclass that is not `bool` reaches here only because `type(value) is int`
    was false, and `int()` on it would be a no-op the dispatcher already does.
    """
    tensorbase = _C_TENSORBASE[0]
    if tensorbase is not None and isinstance(value, tensorbase):
        return _symint_from_tensor(value)
    return value


def _fast_symint_list_coerce(value):
    """`resolve`'s per-element int-list coercion, for the generated fast path.

    The list twin of `_fast_symint_coerce`, and it exists for the reason that
    one does: **the fast path has to reproduce `resolve`'s coercions, not only
    its predicates.** docs/bindings/BIND5.md §7.2 records what happened the last time
    it did not -- the predicate started admitting a single-element Tensor at a
    scalar `SymInt`, this path handed the raw Tensor down, and the Rust side
    unpacked it anyway *including a `bool` one*, so the divergence came back as
    a plausible answer rather than an error. The same trap re-fired here one
    position over: with only the slow path taught, `x.as_strided((2, tensor
    (True)), (4, 1))` answered shape `[2, 1]` positionally and raised
    `RuntimeError` by keyword. Measured, not reasoned about.
    """
    if isinstance(value, (list, tuple)):
        return _coerce_symint_list(value)
    return value


def _compile_fast_path(fn, name, entry, dispatch, is_method):
    skip = 1 if is_method else 0
    lines = [
        "def _fast(args):",
        "    dispatch = getattr(_sys.modules.get('_C'), '_aten_dispatch', None) or getattr(_sys.modules.get('torch._C'), '_aten_dispatch', None)",
        "    if dispatch is None: return NotImplemented",
        "    n_args = len(args)",
    ]
    if is_method:
        lines.append("    self = args[0]")
        lines.append("    n_args -= 1")
        

    if not entry._armed:
        checker = entry._checker_source()
        for plan, _key in entry._candidates:
            if not plan.armed:
                plan.arm(checker)
        entry._armed = True

    import sys
    g = {
        "_sys": sys,
        "NotImplemented": NotImplemented,
        # The fast path reproduces `resolve`'s COERCIONS as well as its
        # predicates. It used to reproduce only `sized_int_list`'s, which was
        # invisible while the predicates admitted nothing that needed the
        # other one -- the moment a scalar `int`/`SymInt` started accepting a
        # single-element Tensor, this path handed the raw Tensor to the
        # dispatcher and the slow path did not. Two spellings of one rule is
        # exactly the hazard `_TypeChecker`'s docstring names, and here it had
        # teeth: the Rust side unpacked the Tensor anyway, including a `bool`
        # one, so a divergence from upstream came back as a plausible answer
        # rather than as an error (docs/bindings/BIND5.md §7.2).
        "_symint_coerce": _fast_symint_coerce,
        "_symint_list_coerce": _fast_symint_list_coerce,
    }
    
    for c_idx, (plan, key) in enumerate(entry._candidates):
        required_kwarg_only = any(
            not arg.has_default for arg in plan.arguments[plan.n_positional:]
        )
        if required_kwarg_only:
            continue
            
        if plan.varargs_intlist:
            lines.append(f"    # Candidate {c_idx} varargs_intlist")
            lines.append(f"    if n_args > {skip}:")
            target_idx = 1 if is_method else 0
            lines.append(f"        if type(args[{target_idx}]) is int:")
            if is_method:
                lines.append(f"            return dispatch('{key}', self, args[1:])")
            else:
                lines.append(f"            return dispatch('{key}', args)")
                
        for L in range(plan.n_positional - skip, -1, -1):
            can_match = True
            for i in range(skip + L, plan.n_positional):
                if not plan.positional[i].has_default:
                    can_match = False
                    break
            if not can_match:
                continue
                
            lines.append(f"    if n_args == {L}:")
            conds = []
            if is_method:
                p_name = f"p_{c_idx}_0"
                g[p_name] = plan.positional[0].predicate
                conds.append(f"{p_name}(self)")
            
            for i in range(L):
                p_name = f"p_{c_idx}_{skip + i}"
                g[p_name] = plan.positional[skip + i].predicate
                arg_idx = i + 1 if is_method else i
                conds.append(f"{p_name}(args[{arg_idx}])")
                
            if conds:
                lines.append(f"        if {' and '.join(conds)}:")
                indent = "            "
            else:
                indent = "        "
                
            call_args = []
            if is_method:
                call_args.append("self")
            for i in range(L):
                arg_idx = i + 1 if is_method else i
                if plan.positional[skip + i].sized_int_list:
                    call_args.append(
                        f"(args[{arg_idx}],) if type(args[{arg_idx}]) is int"
                        f" else _symint_list_coerce(args[{arg_idx}])"
                    )
                elif plan.positional[skip + i].int_list:
                    call_args.append(f"_symint_list_coerce(args[{arg_idx}])")
                elif plan.positional[skip + i].scalar_int:
                    call_args.append(
                        f"args[{arg_idx}] if type(args[{arg_idx}]) is int"
                        f" else _symint_coerce(args[{arg_idx}])"
                    )
                else:
                    call_args.append(f"args[{arg_idx}]")
                    
            call_str = ", ".join(call_args)
            if call_str:
                lines.append(f"{indent}return dispatch('{key}', {call_str})")
            else:
                lines.append(f"{indent}return dispatch('{key}')")
                
    lines.append("    return NotImplemented")
    source = "\n".join(lines)
    exec(source, g)
    fn._fast = g["_fast"]
    fn._compiled = True


# Replaced with `module._capture_active` at install time (see `_install_*`).
# Until then nothing has captured yet, so answering False is correct.
def _CAPTURE_ACTIVE() -> bool:
    return False


def _torch_level_function(name: str, dispatch, overloads):
    """A `torch.<name>` harvested from `_VariableFunctions`.

    With a table entry it resolves: the arguments choose an aten overload and
    the call goes through `_C._aten_dispatch` with the resolved key. Without
    one it still refuses, and still will not guess -- `torch.add(...)` picks
    among `add.Tensor`, `add.Scalar` and `add.out` by argument type, and
    sending every call to `.default` would name a key that mostly does not
    exist, poisoning the work queue DESIGN.md §6 is built on.

    Note what stays true either way: resolution decides *which key*, and the
    key still goes to the one door. An op that resolves but has no kernel
    raises `aten op not implemented in torch._C shim: <key>` from `aten.rs` --
    which is a strictly better work item than the old blanket refusal, because
    it names the overload the caller actually needed rather than the whole
    family.
    """
    entry = overloads.get(name)

    # The mode check is written out in both branches rather than wrapped around
    # them, and that is a cost decision. A decorator would add a Python frame to
    # every one of ~985 functions on every call; a module-global truthiness test
    # is a `LOAD_GLOBAL` and a jump. It costs nothing measurable when no mode is
    # entered, which is every call this shim has ever made outside a
    # `with torch.device(...)` block.
    #
    # `fn` passes *itself* as `func`. That is required, not stylistic:
    # `DeviceContext.__torch_function__` tests `func in _device_constructors()`,
    # a set built by reading `torch.zeros`, `torch.empty` and 34 more names off
    # the `torch` module -- which are these exact closure objects.
    if entry is None:

        def fn(*args, **kwargs):
            if _MODE_STACK:
                return _through_torch_function_modes(fn, args, kwargs)
            raise NotImplementedError(
                f"not implemented in torch._C shim: torch.{name}(...) -- overload "
                f"resolution has no table entry for this op "
                f"(rust/torch_c/src/overloads.json); call "
                f"torch.ops.aten.{name}.<overload>, which carries the overload "
                f"and reaches the same dispatcher"
            )

    else:

        def fn(*args, **kwargs):
            if _MODE_STACK:
                return _through_torch_function_modes(fn, args, kwargs)
            if not getattr(fn, "_compiled", False):
                _compile_fast_path(fn, name, entry, dispatch, False)
            # **The fast path is off while a capture records.** A recording
            # made through it carries no argument names -- `dispatch` is
            # called positionally, which is the whole saving -- so the same
            # call would be recorded two different ways depending on whether
            # the caller spelled its arguments out:
            #
            #     t.transpose(0, 1)            kwargs []
            #     t.transpose(dim0=0, dim1=1)  kwargs ['dim0', 'dim1', 'self']
            #
            # A trace's contents must not depend on that. The decomposition
            # pass reads operands by the schema's names (the third wall in
            # `test_decompose_lowers_the_op_capture_md_named`), so the nameless
            # shape reaches it as an empty map rather than as an error.
            #
            # `_capture_active()` is one call per op and this is the hot path,
            # so it is guarded by `_fast` being reached at all -- and a capture
            # is not open during decode, which is where the saving was measured.
            if not kwargs and not _CAPTURE_ACTIVE():
                res = fn._fast(args)
                if res is not NotImplemented:
                    return res
            # `_strip_python_only_kwargs({})` is `({}, False)`, and `**kwargs`
            # already handed us a fresh dict, so the call is skippable when it
            # would have nothing to strip. Most calls are in that case.
            if kwargs:
                stripped, requires_grad = _strip_python_only_kwargs(name, kwargs)
            else:
                stripped, requires_grad = kwargs, False
            key, bound = entry.resolve(args, stripped)
            out = dispatch(key, **bound)
            return _apply_requires_grad(out) if requires_grad else out

    fn.__name__ = name
    fn.__qualname__ = name
    return fn


# ---------------------------------------------------------------------------
# `TensorBase` members that do something
# ---------------------------------------------------------------------------
#
# docs/design/C_SURFACE.md §4 measured a small Llama forward plus greedy `generate`
# and found 50 of `TensorBase`'s 694 members actually used. Everything here
# serves that list, and nothing here is a second entrance: a method resolves an
# overload and calls `_C._aten_dispatch`, exactly like `torch.<op>` does.
#
# Three groups, and the difference between them is worth keeping visible:
#
#   1. table-driven (`methods.json`)  -- the overload machine, `self` bound in.
#   2. Python-level                   -- `to`, `item`, `__bool__`, `__getitem__`.
#      Upstream's binding for each of these is not a plain overload set either:
#      `THPVariable_to` reads its arguments and picks between several aten
#      calls, `THPVariable_getitem` walks the index and emits a *sequence* of
#      them. So these are written out, and each one still ends at the one door.
#   3. autograd-shaped               -- `requires_grad`, `grad_fn`, `data`.
#      These are the honest edge of the shim; see `_install_autograd_shape`.


def _tensor_method(name: str, dispatch, entry):
    """One `TensorBase.<name>`, resolving against `entry` with `self` bound.

    Operator dunders answer `NotImplemented` when nothing binds, rather than
    raising. That is upstream's behaviour and it is load-bearing rather than
    polite: the vendored tree compares tensors against strings and against
    `None` in several places, and Python only gets to fall back to its own
    identity comparison if `__eq__` declines. A `TypeError` out of `__eq__`
    would turn `x == "cpu"` into a crash where upstream gives `False`.
    """
    is_dunder = name.startswith("__") and name.endswith("__")

    def method(self, *args, **kwargs):
        if not getattr(method, "_compiled", False):
            _compile_fast_path(method, name, entry, dispatch, True)
        # See the `fn` branch above: a recording must not depend on whether the
        # caller spelled its arguments out.
        if not kwargs and not _CAPTURE_ACTIVE():
            res = method._fast((self,) + args)
            if res is not NotImplemented:
                return res
        if kwargs:
            stripped, requires_grad = _strip_python_only_kwargs(name, kwargs)
        else:
            stripped, requires_grad = kwargs, False
        try:
            # See `_torch_function` above for why the empty case skips the call.
            key, bound = entry.resolve((self,) + args, stripped)
        except TypeError:
            if is_dunder:
                return NotImplemented
            raise
        out = dispatch(key, **bound)
        return _apply_requires_grad(out) if requires_grad else out

    method.__name__ = name
    method.__qualname__ = f"TensorBase.{name}"
    return method


def _install_tensor_T(tensorbase) -> None:
    """`Tensor.T`, upstream's dimension-reversing transpose alias.

    No aten op named `T` exists upstream: a `TorchDispatchMode` logger around
    `x.T` on 2.13.0 fires exactly one record, `aten.permute.default`, with
    dims reversed (`x.permute(*range(x.dim() - 1, -1, -1))`) -- confirmed for
    both a 2-D tensor (the ordinary transpose) and a 1-D one (upstream still
    dispatches `permute(0)` rather than returning `self`, even though the
    values are unchanged; measured with `is`, which came back `False`). So
    this is Python-level surface over the `permute` member `methods.json`
    already carries, not a second kernel.

    `falcon`'s attention module is what asked for this (docs/models/COMPAT.md,
    transformers 4.x compatibility sweep).
    """

    def getter(self):
        return self.permute(*range(self.dim() - 1, -1, -1))

    tensorbase.T = property(getter)


def _install_tensor_complex_parts(tensorbase, dispatch) -> None:
    """`Tensor.real` / `Tensor.imag` -- `fnet`'s other wall (docs/kernels/COMPLEX3.md
    §6.2), and the reason it is not in `methods.json` alongside everything
    else `_install_tensor_methods` harvests: upstream exposes both as
    **properties** (`torch._C.TensorBase.real`, a `getset_descriptor`), not as
    callables, so a table-driven entry could never carry them -- `methods.json`
    is keyed on names called with `()`. Before this, both fell through to the
    raising stub the surface list installs for every name it does not
    recognise.

    Both aten keys (`aten.real.default`, `aten.imag.default`) are implemented
    and proven element-wise against upstream in `pytests/test_complex.py`; this
    adds nothing to their arithmetic, only the property that reaches them --
    checked before writing it, the way docs/bindings/BINDINGS.md's `mish` was not.
    """
    tensorbase.real = property(lambda self: dispatch("aten.real.default", self))
    tensorbase.imag = property(lambda self: dispatch("aten.imag.default", self))


def _install_tensor_size_list_tensor_forms(module, tensorbase) -> None:
    """`Tensor.new_zeros((..., 0-dim int Tensor, ...))` -- `led` and
    `longformer`'s shared wall (docs/kernels/STRIDED.md §6, docs/bindings/BIND5.md §1).

    `LongformerSelfAttention._sliding_chunks_query_key_matmul`, borrowed
    verbatim by `modeling_led.py:440` and `modeling_longformer.py:796`, writes

        diagonal_chunked_attention_scores.new_zeros(
            (batch_size * num_heads, chunks_count + 1,
             window_overlap, window_overlap * 2 + 1)
        )

    and `chunks_count` there is a **0-dim int64 Tensor**, not a Python int --
    confirmed by instrumenting the real call rather than read off the source
    (`['int', 'Tensor=tensor(2)', 'int', 'int']`). It is the same argument
    form docs/bindings/BIND4.md §3 landed for `torch.zeros`, arriving at a *method*
    this time, and measured to be upstream's rule for the method too before
    being written: `x.new_zeros((2, tensor(3), 4))` gives `(2, 3, 4)` on real
    torch 2.13.0.

    This wraps `new_zeros` ALONE and not the rest of the `new_*` family, for
    the reason docs/bindings/ARGFORM.md §2 gives and docs/bindings/BIND4.md §3 followed for
    `zeros`: upstream accepts the form for `new_ones`, `new_empty` and
    `new_full` as well (measured, docs/bindings/BIND5.md §1), but nothing measured
    calls them that way, and a rule installed table-wide would accept a
    Tensor in every `SymInt[]` position with no matching measurement behind
    each one.

    Unlike `torch.zeros`, there is no `_MODE_STACK` guard to reproduce here.
    `DeviceContext.__torch_function__` matches its 36 constructors by object
    identity off the `torch` MODULE (`_device_constructors()`,
    `torch/utils/_device.py`), and a bound tensor method is not among them --
    which is why docs/bindings/BIND4.md §3.1's trap does not have a sibling on this
    path. The wrapped method's behaviour is otherwise the table-driven one,
    byte for byte: it forwards `*args` and `**kwargs` untouched.
    """
    table_new_zeros = tensorbase.new_zeros

    def new_zeros(self, *args, **kwargs):
        # `*args` rather than a named `size` parameter, so that the arity
        # refusal for `x.new_zeros()` stays the table's own message about the
        # aten schema instead of becoming a Python signature error about a
        # parameter upstream does not name.
        if args:
            args = (
                _coerce_symint_size_tensors(module, "new_zeros", args[0]),
            ) + args[1:]
        return table_new_zeros(self, *args, **kwargs)

    new_zeros.__name__ = "new_zeros"
    new_zeros.__qualname__ = "TensorBase.new_zeros"
    tensorbase.new_zeros = new_zeros


def _install_tensor_methods(module, tensorbase, dispatch, methods) -> None:
    for name, entry in methods.items():
        setattr(tensorbase, name, _tensor_method(name, dispatch, entry))

    _install_tensor_size_list_tensor_forms(module, tensorbase)
    _install_tensor_T(tensorbase)
    _install_tensor_complex_parts(tensorbase, dispatch)
    _install_tensor_conversions(module, tensorbase, dispatch)
    _install_tensor_scalars(tensorbase, dispatch)
    _install_tensor_indexing(module, tensorbase, dispatch)
    _install_tensor_softmax(tensorbase, dispatch)
    _install_tensor_where(tensorbase, dispatch)
    _install_tensor_chunk(tensorbase, dispatch)
    _install_tensor_index_put_(tensorbase, dispatch)
    _install_autograd_shape(tensorbase)


# `x.float()` and friends. Upstream has no `aten::float`: `THPVariable_float`
# is `self.to(ScalarType::Float)`, and a `TorchDispatchMode` logger confirms it
# -- `f.float()` on a tensor that is already float32 produces *no* aten record
# at all, while `i.float()` produces `aten._to_copy.default`. Both halves are
# reproduced below: the no-change case returns `self` without dispatching, and
# the change case names `_to_copy`, which is the key upstream's dispatcher
# really sees. (The parser-level key would be `aten.to.dtype`; that overload is
# composite and never reaches a kernel. Naming the one that does keeps the work
# queue of DESIGN.md §6 pointing at something implementable.)
_DTYPE_METHODS = {
    "float": "float32",
    "double": "float64",
    "half": "float16",
    "bfloat16": "bfloat16",
    "long": "int64",
    "int": "int32",
    "short": "int16",
    "char": "int8",
    "byte": "uint8",
    "bool": "bool",
}


def _install_tensor_conversions(module, tensorbase, dispatch) -> None:
    def _to_copy(self, dtype=None, device=None, copy=False):
        if isinstance(device, str):
            device = module.device(device)
        # `!=`, not `is not`. This used to be load-bearing for a bad reason --
        # `self.dtype` built a fresh `PyDtype` on every read, so identity was
        # never true even for the same dtype. That is fixed (docs/bindings/BIND.md §9;
        # `t.dtype is torch.float32` holds now, and `aten.baddbmm.default`'s
        # decomposition stopped promoting float32 to float64 with it). It
        # stays `!=` anyway, because `device` on the next line has no such
        # contract upstream and a mixed pair of predicates here would read as
        # a distinction that means something. Both types define `__eq__`.
        wants_dtype = dtype is not None and dtype != self.dtype
        wants_device = device is not None and device != self.device
        if not (wants_dtype or wants_device or copy):
            return self
        return dispatch(
            "aten._to_copy.default",
            self,
            dtype=dtype if wants_dtype else None,
            device=device if wants_device else None,
        )

    for name, dtype_name in _DTYPE_METHODS.items():
        def convert(self, _dtype_name=dtype_name):
            return _to_copy(self, dtype=getattr(module, _dtype_name))

        convert.__name__ = name
        convert.__qualname__ = f"TensorBase.{name}"
        setattr(tensorbase, name, convert)

    def to(self, *args, **kwargs):
        """`Tensor.to`, in upstream's own argument shapes.

        This is the clearest case of docs/bindings/OVERLOAD.md §9 item 7 -- a Python
        binding whose signatures do not line up with any aten schema. torch's
        parser takes `to(Device device=None, ScalarType dtype=None, ...)`,
        `to(ScalarType dtype, ...)` and `to(Tensor other, ...)`; the aten
        overloads are `to.dtype`, `to.device` (which requires *both*),
        `to.other` and `to.dtype_layout` (all-keyword). `x.to('cpu')` binds the
        first parser signature and none of the aten ones, so a table entry
        would report "no matching overload" for a call real torch accepts.
        """
        kwargs = dict(kwargs)
        copy = bool(kwargs.pop("copy", False))
        kwargs.pop("non_blocking", None)
        _refuse_unrepresentable_memory_format("TensorBase.to", kwargs)
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)
        other = kwargs.pop("other", None)
        if kwargs:
            raise TypeError(
                f"Tensor.to(): unexpected keyword argument(s) "
                f"{sorted(kwargs)} in torch._C shim"
            )
        # The two trailing positional booleans are `non_blocking` then `copy`,
        # in that order, and only the second one means anything here. Counting
        # them rather than folding both into `copy` (which is what this did)
        # matters as soon as anything passes `non_blocking` positionally -- and
        # `nn.Module.to` does, on every call: `module.py:1369` is
        # `t.to(device, dtype, non_blocking)`. With both bools read as `copy`,
        # `model.to("cpu", non_blocking=True)` would have copied every
        # parameter it was asked to leave alone.
        bools_seen = 0
        for value in args:
            if isinstance(value, bool):
                bools_seen += 1
                if bools_seen == 2:
                    copy = copy or value
                elif bools_seen > 2:
                    raise TypeError(
                        "Tensor.to(): too many boolean positional arguments in "
                        "torch._C shim"
                    )
                # `non_blocking` is accepted and ignored: there is no async
                # copy engine here, so every transfer is already synchronous,
                # which is what upstream does on a CPU-only build too.
            elif isinstance(value, module.dtype):
                dtype = value
            elif isinstance(value, (module.device, str)):
                device = value
            elif isinstance(value, tensorbase):
                other = value
            elif value is None:
                # `Module.to` passes its parsed device through even when it is
                # `None` ("dtype only"), so this is a normal call shape.
                continue
            else:
                raise TypeError(
                    f"Tensor.to(): torch._C shim does not understand argument "
                    f"{value!r} of type {type(value).__name__}"
                )
        if other is not None:
            dtype, device = other.dtype, other.device
        return _to_copy(self, dtype=dtype, device=device, copy=copy)

    to.__name__ = "to"
    to.__qualname__ = "TensorBase.to"
    setattr(tensorbase, "to", to)
    setattr(tensorbase, "type_as", lambda self, other: _to_copy(self, dtype=other.dtype))

    # `Tensor.type` -- **three behaviours in one name**, and the one nobody
    # expects is the one with no argument.
    #
    # `THPVariable_type` in `torch/csrc/autograd/python_variable_methods.cpp`
    # branches on what it was handed, and all three were measured on 2.13.0
    # rather than read:
    #
    #     x.type()                    'torch.FloatTensor'   -- a *string*
    #     x.type(torch.float64)       a cast, `.to(dtype)`
    #     x.type('torch.DoubleTensor')the same cast, by legacy NAME
    #     x.type(torch.DoubleTensor)  the same again, by legacy CLASS
    #
    # A shim that implemented only the cast would answer a tensor where a
    # caller asked for a name, and `if t.type() == 'torch.cuda.FloatTensor'`
    # is a real idiom -- so the no-argument form is not an afterthought.
    #
    # GPT-2's **eager** attention is what asked: `eager_attention_forward` in
    # `transformers/models/gpt2/modeling_gpt2.py:66` is
    # `attn_weights = attn_weights.type(value.dtype)`, the dtype form, and the
    # whole path stopped there. (`attn_implementation="sdpa"` is the default
    # and does not reach it, which is why this survived every sweep.)
    #
    # Two details are upstream's and are kept:
    #
    #   * **the cast is `.to`, so it keeps `.to`'s identity rule** --
    #     `x.type(torch.float32) is x` is `True` upstream for a float32 `x`,
    #     measured, and `_to_copy` above already answers that way. That is why
    #     this routes through `_to_copy` rather than dispatching directly.
    #   * **a bad argument and a bad name raise different exceptions**:
    #     `x.type(3)` is a `TypeError("dtype must be a type, str, or dtype
    #     object")` and `x.type('torch.Nonsense')` is a
    #     `ValueError("invalid type: 'torch.Nonsense'")`. Both measured.
    #
    # The name table is `_LEGACY_TENSOR_DTYPES`, the same one
    # `_install_legacy_tensor_types` builds `torch.FloatTensor` and its nine
    # siblings from. Deliberately the same object and not a copy: a second
    # list of dtype spellings is the failure docs/verification/AUDIT.md found six times.
    # It covers every dtype this build can *hold* -- `uint16`, `complex64` and
    # the rest have no candle storage, so no tensor can carry them and no
    # `x.type()` can be asked about them.
    def _legacy_type_name(dtype):
        for legacy, dtype_name in _LEGACY_TENSOR_DTYPES.items():
            if getattr(module, dtype_name, None) == dtype:
                return "torch." + legacy
        raise NotImplementedError(
            "not implemented in torch._C shim: TensorBase.type() has no legacy "
            f"tensor-type name for {dtype} -- the ten this build can hold are "
            + ", ".join("torch." + n for n in sorted(_LEGACY_TENSOR_DTYPES))
        )

    def _dtype_from_legacy_name(name):
        """Upstream's `torch::utils::options_from_string`, narrowed to what
        this build can express."""
        parts = name.split(".")
        if len(parts) < 2 or parts[0] != "torch":
            raise ValueError(f"invalid type: '{name}'")
        stem = parts[-1]
        if len(parts) == 3:
            # `torch.cuda.FloatTensor` and friends. Upstream reads the middle
            # segment as a device and fails on the backend rather than on the
            # name; routing it through `module.device` keeps that decision in
            # the one place that knows which backends are linked, the same
            # reasoning `cuda()` above is written from.
            if parts[1] == "sparse":
                raise NotImplementedError(
                    "not implemented in torch._C shim: "
                    f"TensorBase.type('{name}') asks for the sparse layout, and "
                    "this build has only `torch.strided`"
                )
            device = module.device(parts[1])
        elif len(parts) == 2:
            device = None
        else:
            raise ValueError(f"invalid type: '{name}'")
        if stem == "Tensor":
            return module.get_default_dtype(), device
        if stem not in _LEGACY_TENSOR_DTYPES:
            if stem.endswith("Tensor"):
                # A name upstream may well accept that this build has no dtype
                # for. Saying "invalid" would be a lie about upstream; saying
                # "not implemented here" is the truth about this shim.
                raise NotImplementedError(
                    "not implemented in torch._C shim: "
                    f"TensorBase.type('{name}') -- the ten legacy tensor types "
                    "this build can hold are "
                    + ", ".join("torch." + n for n in sorted(_LEGACY_TENSOR_DTYPES))
                )
            raise ValueError(f"invalid type: '{name}'")
        return getattr(module, _LEGACY_TENSOR_DTYPES[stem]), device

    def type_(self, dtype=None, non_blocking=False, **kwargs):
        # `non_blocking` is accepted and ignored for `to`'s reason: there is no
        # async copy engine here. `memory_format` is accepted only for the two
        # formats that ask for what the result already is; the channels-last
        # pair refuses rather than being dropped (docs/graph/EXPORT5.md §3).
        _refuse_unrepresentable_memory_format("TensorBase.type", kwargs)
        kwargs.pop("async", None)
        if kwargs:
            raise TypeError(
                "TensorBase.type(): unexpected keyword argument(s) "
                f"{sorted(kwargs)} in torch._C shim"
            )
        if dtype is None:
            return _legacy_type_name(self.dtype)
        if isinstance(dtype, module.dtype):
            return _to_copy(self, dtype=dtype)
        if isinstance(dtype, str):
            name = dtype
        elif isinstance(dtype, type):
            # Upstream reads the type object's `tp_name`, which for its own
            # C-defined `torch.DoubleTensor` is `'torch.DoubleTensor'` and for
            # a builtin like `int` is a bare `'int'`. `__module__` reproduces
            # that split: the legacy classes here carry `__module__ = "torch"`
            # (`_install_legacy_tensor_types`), builtins carry `"builtins"`.
            where = getattr(dtype, "__module__", None)
            name = dtype.__name__ if where in (None, "builtins") else (
                f"{where}.{dtype.__name__}")
        else:
            raise TypeError("dtype must be a type, str, or dtype object")
        want, device = _dtype_from_legacy_name(name)
        return _to_copy(self, dtype=want, device=device)

    type_.__name__ = "type"
    type_.__qualname__ = "TensorBase.type"
    setattr(tensorbase, "type", type_)

    # `Tensor.expand_as` -- `zoedepth`'s wall once 2-D convolution existed
    # (docs/kernels/KERNELS26.md §7). A *name* gap, not a kernel gap: measured on
    # upstream 2.13.0, `aten::expand_as` is `CompositeImplicitAutograd`
    # (`_dispatch_has_kernel_for_dispatch_key` is True) and a
    # `TorchDispatchMode` trace of `x.expand_as(y)` fires exactly one op --
    # `aten.expand.default(x, y.shape)` -- which this shim already implements
    # and already golden-compares. So this is `torch.conv2d`'s shape of fix
    # (ARCH26.md §7), not a new computation path.
    #
    # It goes through `.expand` rather than calling `dispatch` directly so that
    # it inherits `expand`'s own `-1` handling and rank rules, and so that it
    # stays a **view**: `x.expand_as(y)` shares storage with `x` upstream
    # (measured: `data_ptr()` equal), and `expand.default` is in this shim's
    # aliasing table for the same reason.
    def expand_as(self, other):
        return self.expand(list(other.shape))

    expand_as.__name__ = "expand_as"
    expand_as.__qualname__ = "TensorBase.expand_as"
    setattr(tensorbase, "expand_as", expand_as)

    # `Tensor.tile` -- `sam3_video`'s wall once `outer` and `ones_like` were
    # behind it. A *name* gap over `repeat`, and the same kind of check:
    # `aten::tile` is `CompositeImplicitAutograd` and a `TorchDispatchMode`
    # trace of `x.tile((2,))` fires exactly one op, `aten.repeat.default`.
    #
    # **`tile` and `repeat` differ in what they do with too FEW dims**, which is
    # the whole reason this is not an alias:
    #
    #     x is (2, 3)
    #     x.repeat(2)      REFUSES  -- repeat needs at least as many as the rank
    #     x.tile(2)        (2, 6)   -- dims are left-padded with 1s to the rank
    #
    # Too MANY is the same on both (the extra dims become new leading axes), so
    # a case built only from `len(dims) >= rank` cannot tell them apart. Padding
    # on the LEFT rather than the right is the other half: `x.tile(2)` on a
    # `(2,3)` is `(2, 6)`, not `(4, 3)`. Both measured.
    def tile(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        dims = [int(d) for d in dims]
        rank = len(self.shape)
        if len(dims) < rank:
            dims = [1] * (rank - len(dims)) + dims
        return self.repeat(dims)

    tile.__name__ = "tile"
    tile.__qualname__ = "TensorBase.tile"
    setattr(tensorbase, "tile", tile)

    # `Tensor.outer` -- upstream has the member as well as the free function
    # (`hasattr(torch.Tensor, "outer")` is True, measured), and `sam3_video`
    # reaches the free one. Both are installed because a name that exists
    # upstream and refuses here is the failure mode this file keeps hitting;
    # the member delegates so the rank check lives in one place.
    def outer(self, vec2):
        return _outer_impl(self, vec2)

    outer.__name__ = "outer"
    outer.__qualname__ = "TensorBase.outer"
    setattr(tensorbase, "outer", outer)

    # `Tensor.new_tensor` -- docs/architectures/ARCH26.md, `zoedepth`'s wall (through
    # `Dinov2`'s `_init_weights`, `transformers/initialization.py`'s
    # `trunc_normal_`, `torch/nn/init.py`'s `_no_grad_trunc_normal_`, which
    # reads a scalar bound back with `tensor.new_tensor(a, device="cpu").item()`
    # -- not one of the twenty in docs/architectures/ARCH20.md, none of which triggered
    # `trunc_normal_`'s truncated-bounds path during construction).
    #
    # A `TorchDispatchMode` logger on 2.13.0 shows `x.new_tensor(5.0,
    # device="cpu")` firing exactly one record, `aten.lift_fresh.default` --
    # the same single op `torch.tensor(...)` makes (`_tensor_factory`'s own
    # docstring measured that already, for the top-level function). So this is
    # the same construction, minus the mode-stack detour: `torch.tensor` is
    # one of `DeviceContext`'s 36 names that has to consult
    # `with torch.device(...):`, but `Tensor.new_tensor` is a method, not one
    # of those 36, and defaults `dtype`/`device` from the receiver rather than
    # from nothing -- upstream's doc is explicit that the defaults come from
    # `self`, not from a global default, and that `requires_grad` defaults to
    # `False` regardless of `self.requires_grad`.
    #
    # `_tensor_factory` builds `varfns.tensor` (harvested into `torch.tensor`
    # by `torch/__init__.py` after this module returns), not `module.tensor`
    # (`module` here is `_C`, which has no such attribute) -- so this cannot
    # simply call that function's build result. It calls the two primitives
    # `_tensor_factory`'s body calls, in the same order, with the same
    # `requires_grad`/`pin_memory` refusals, instead.
    def new_tensor(self, data, *, dtype=None, device=None, requires_grad=False,
                    pin_memory=False):
        # `requires_grad` is carried rather than refused -- see
        # `_strip_python_only_kwargs` for the whole argument. `pin_memory` still
        # refuses, and the two are not the same case: a pinned allocation is a
        # capability this shim does not have, while the flag is one it has had
        # all along under a different spelling.
        if pin_memory:
            raise NotImplementedError(
                "not implemented in torch._C shim: TensorBase.new_tensor("
                "pin_memory=True)"
            )
        dtype = self.dtype if dtype is None else dtype
        device = self.device if device is None else device
        if isinstance(device, str):
            device = module.device(device)
        out = dispatch(
            "aten.lift_fresh.default",
            _new_from_data(module, data, dtype, device),
        )
        return _apply_requires_grad(out) if requires_grad else out

    new_tensor.__name__ = "new_tensor"
    new_tensor.__qualname__ = "TensorBase.new_tensor"
    setattr(tensorbase, "new_tensor", new_tensor)

    # -- the device spellings that are `.to()` in disguise -------------------
    #
    # `x.cpu()` and `x.cuda()` are what upstream's `TensorBase` exposes next to
    # `to`, and they are the spellings model code actually writes. Both are
    # routed through the same `_to_copy` rather than given their own path, so
    # the "already there, return self" short-circuit and the refusal for an
    # unavailable backend are decided in one place. Measured against torch
    # 2.13.0: `x.cpu() is x` is `True` for a CPU tensor and the call records no
    # dispatcher traffic at all.
    def cpu(self, *args, **kwargs):
        return _to_copy(self, device=module.device("cpu"))

    def cuda(self, device=None, non_blocking=False, memory_format=None):
        # Upstream on a build without CUDA raises `AssertionError: Torch not
        # compiled with CUDA enabled` from `torch/cuda/__init__.py`. This
        # instead reaches `PyDevice::resolve`, which refuses the label by name
        # -- the same shape of answer, from the one place that knows which
        # backends are linked, rather than a second hardcoded claim.
        return _to_copy(self, device=module.device("cuda") if device is None
                        else module.device(device))

    # `-1` is not an error code: it is how torch spells "this device kind is
    # not indexed" (measured, `torch.zeros(2).get_device()` is `-1` while an
    # mps tensor's is `0`), and `torch/_utils.py:_get_device_index` branches on
    # exactly that. Lives here rather than in tensor.rs only because PyO3
    # derives the same slot name for a `get_device` method and the `device`
    # getter, and the crate is built without `multiple-pymethods`.
    def get_device(self):
        index = self.device.index
        return -1 if index is None else index

    # Dtype predicates rather than device ones, but on the same road and dead
    # for the same reason: `nn.Module.to`'s `convert(t)` calls
    # `t.is_floating_point()` on every parameter (`module.py:1365`), so
    # `.float()` and `.double()` stopped here. They read the dtype the tensor
    # already carries -- `dtype.rs` owns both flags -- and record no dispatcher
    # traffic upstream, so they are metadata reads, not a bypass of the door.
    for name, flag in (("is_floating_point", "is_floating_point"),
                       ("is_complex", "is_complex")):
        def predicate(self, _flag=flag):
            return getattr(self.dtype, _flag)

        predicate.__name__ = name
        predicate.__qualname__ = f"TensorBase.{name}"
        setattr(tensorbase, name, predicate)

    for fn, name in ((cpu, "cpu"), (cuda, "cuda"), (get_device, "get_device")):
        fn.__name__ = name
        fn.__qualname__ = f"TensorBase.{name}"
        setattr(tensorbase, name, fn)


def _install_tensor_softmax(tensorbase, dispatch) -> None:
    """`Tensor.softmax(dim, dtype=None)` and `Tensor.log_softmax(dim, dtype=None)`.

    The two are one function apart and are installed together so they cannot
    drift on `dtype=` or on `half_to_float`. `log_softmax` is here for the
    identical reason -- `aten::log_softmax.int` is `CompositeImplicitAutograd`
    and never reaches a kernel, while `aten::_log_softmax` is the dispatched
    leaf (docs/training/LOSS.md). Note that `torch._log_softmax`, the *private* free
    spelling, is a genuine `overloads.json` entry, because it names the leaf;
    `torch.log_softmax` is bound to this member instead. That asymmetry is
    written out in `overloads.json`'s own `_README`.

    `F.log_softmax(x, dim)` reaches `Tensor.log_softmax` -- measured: the
    vendored `torch/nn/functional.py` spells it `input.log_softmax(dim)`, so
    without this member `F.log_softmax` refuses at `TensorBase.log_softmax`
    even with the kernel present. That is the shape of gap golden is
    structurally blind to, since golden compares by dispatch key.

    `methods.json`'s `_README` explains why this is not a table entry: the
    parser-level key for `Tensor.softmax` is `aten::softmax.int`, which is
    `CompositeImplicitAutograd` and never reaches a kernel, while the key
    upstream's dispatcher actually sees is `aten._softmax.default` (measured
    with a `TorchDispatchMode` logger on torch 2.13.0, docs/bindings/NN_SURFACE.md §6):

        x.softmax(dim)                      -> aten._softmax.default(x, dim, False)
        x.softmax(dim, dtype=torch.float64)  -> aten._to_copy.default(x, dtype=float64)
                                                 then aten._softmax.default(_, dim, False)
        x.softmax(dim, dtype=x.dtype)        -> aten._softmax.default(x, dim, False)
                                                 (no `_to_copy` -- measured: a no-op dtype
                                                 emits no conversion call either)

    `half_to_float` stays `False` unconditionally: it is an autocast signal
    (true only when the dispatcher itself decides to keep a fp16 input in
    fp32 accumulation), this shim has no autocast, and the trace above shows
    upstream's own eager path never sets it for a plain `dtype=None` call or
    an explicit-`dtype` one either. Same reasoning, same shape as `to.dtype`
    versus `_to_copy` in `_install_tensor_conversions`.
    """

    def softmax(self, dim, dtype=None):
        source = self
        if dtype is not None and dtype != self.dtype:
            source = dispatch("aten._to_copy.default", self, dtype=dtype)
        return dispatch("aten._softmax.default", source, dim, False)

    softmax.__name__ = "softmax"
    softmax.__qualname__ = "TensorBase.softmax"
    setattr(tensorbase, "softmax", softmax)

    def log_softmax(self, dim, dtype=None):
        source = self
        if dtype is not None and dtype != self.dtype:
            source = dispatch("aten._to_copy.default", self, dtype=dtype)
        return dispatch("aten._log_softmax.default", source, dim, False)

    log_softmax.__name__ = "log_softmax"
    log_softmax.__qualname__ = "TensorBase.log_softmax"
    setattr(tensorbase, "log_softmax", log_softmax)


def _install_tensor_where(tensorbase, dispatch) -> None:
    """`Tensor.where(condition, other)` -- `led` and `longformer`'s wall.

    **The receiver is the `x` branch, not the condition, and that is the whole
    reason this is not a `methods.json` row.** Measured on torch 2.13.0 in a
    separate process:

        x.where(c, y)      is      torch.where(c, x, y)

    with `x = [[1, 2], [3, 4]]`, `y = [[10, 20], [30, 40]]` and
    `c = [[True, False], [False, True]]` both answering `[[1, 20], [30, 4]]`.
    So the receiver binds into the aten schema's **second** slot:

        aten::where.self(Tensor condition, Tensor self, Tensor other)
                          ^ arg 0          ^ the receiver

    `methods.json`'s machine binds the receiver into argument **0** and only
    argument 0 -- `_Overloads(self_bound=True)` passes it as `args[0]` and
    every positional count skips exactly one. A `where` row in that table would
    therefore compute `torch.where(x, c, y)`: the *right shape*, the right
    dtype, and the two branches swapped. That failure is invisible to a shape
    check and to any test whose two branches broadcast the same, which is why
    it is written out here instead. docs/kernels/REPEAT.md §1.

    `docs/bindings/BIND5.md` §4.1 recorded this as "`methods.json` simply has no row";
    re-measured here rather than trusted, and the row would have been wrong.
    The kernels it named were right -- all five `where` overloads are
    implemented and unchanged by this.

    Two overloads, chosen the way upstream's parser chooses (measured: a
    Tensor `other` takes `where.self`, a `Number` other takes
    `where.ScalarOther`, and `x.where(c, True)` is accepted with the bool
    treated as a wrapped number). `.ScalarSelf` and `.Scalar` are unreachable
    through this door by construction -- the receiver is always a Tensor -- and
    stay reachable through `torch.where`.
    """

    def where(self, condition, other):
        if isinstance(other, tensorbase):
            return dispatch("aten.where.self", condition, self, other)
        if isinstance(other, (bool, int, float, complex)):
            return dispatch("aten.where.ScalarOther", condition, self, other)
        # Upstream's shape of message, which names both overloads rather than
        # only the one that came closest -- a caller who passed `None` needs to
        # see that neither door takes it.
        raise TypeError(
            "where() received an invalid combination of arguments - got "
            f"(Tensor, {type(other).__name__}), but expected one of:\n"
            " * (Tensor condition, Tensor other)\n"
            " * (Tensor condition, Number other)\n"
        )

    where.__name__ = "where"
    where.__qualname__ = "TensorBase.where"
    setattr(tensorbase, "where", where)


def _install_tensor_index_put_(tensorbase, dispatch) -> None:
    """`Tensor.index_put_(indices, values, accumulate=False)`.

    Same wall as the `index_put_` composite in `_install_composites`:
    `aten::index_put_`'s `indices: Tensor?[]` argument (a list of optional
    Tensors, one per axis) is a type `_decompose_type` cannot represent --
    see that function's docstring and the free-function `index_put_`'s note
    above -- so this cannot be a `methods.json` entry either, for the same
    install-time `_SCHEMA_BASE_TYPES` failure. The body is the method-door
    twin of that composite: same dispatch, `self` bound instead of taken as
    the first positional argument.
    """

    def index_put_(self, indices, values, accumulate=False):
        return dispatch(
            "aten.index_put_.default", self, list(indices), values, accumulate
        )

    index_put_.__name__ = "index_put_"
    index_put_.__qualname__ = "TensorBase.index_put_"
    setattr(tensorbase, "index_put_", index_put_)


def _install_tensor_chunk(tensorbase, dispatch) -> None:
    """`Tensor.chunk(chunks, dim=0)`, transcribed from upstream's composite.

    `aten::chunk` is `CompositeImplicitAutograd`: it never reaches a kernel,
    it picks between two ops that do. So it is Python-level here for the same
    reason `softmax` is -- a `methods.json` entry would name
    `aten.chunk.default`, a key no dispatcher ever sees. The body is
    `at::native::chunk` (`aten/src/ATen/native/TensorShape.cpp`) line for
    line, including the zero-extent branch, whose comment upstream says what
    it is for: `split` with a split size of 0 would discard the requested
    number of chunks, because an arbitrary number of empty chunks sums to
    zero all the same.

    Measured with a `TorchDispatchMode` logger on torch 2.13.0:

        arange(10).chunk(3)  -> aten.split.Tensor          shapes 4,4,2
        arange(10).chunk(4)  -> aten.split.Tensor          shapes 3,3,3,1
        arange(10).chunk(5)  -> aten.split.Tensor          shapes 2,2,2,2,2
        arange(3).chunk(7)   -> aten.split.Tensor          shapes 1,1,1  (three, not seven)
        empty(0).chunk(3)    -> aten.split_with_sizes      shapes 0,0,0  (three, not one)

    The last two are the reason this is the real arithmetic rather than an
    even division: `chunks` is an upper bound on how many pieces come back,
    not a promise, and only the zero-extent case is allowed to return exactly
    `chunks` of them.

    Returns a `tuple`, which is what upstream's `THPVariable_chunk` returns;
    `_aten_dispatch` hands back a `list` from `split`, so the conversion is
    here and is not cosmetic (`a, b = t.chunk(2)` works either way, but
    `t.chunk(2) + (x,)` does not).
    """

    def chunk(self, chunks, dim=0):
        if self.dim() == 0:
            raise RuntimeError("chunk expects at least a 1-dimensional tensor")
        if chunks <= 0:
            raise RuntimeError(
                f"chunk expects `chunks` to be greater than 0, got: {chunks}"
            )
        dim_size = self.shape[dim]
        split_size = (dim_size + chunks - 1) // chunks
        if split_size == 0 and dim_size == 0:
            sizes = [split_size] * chunks
            sizes[chunks - 1] = split_size - (split_size * chunks - dim_size)
            return tuple(
                dispatch("aten.split_with_sizes.default", self, sizes, dim)
            )
        return tuple(dispatch("aten.split.Tensor", self, split_size, dim))

    chunk.__name__ = "chunk"
    chunk.__qualname__ = "TensorBase.chunk"
    setattr(tensorbase, "chunk", chunk)

    def flatten(self, start_dim=0, end_dim=-1):
        """`Tensor.flatten`, `cohere`'s wall (docs/architectures/ARCH20.md §5).

        `aten::flatten.using_ints` is `CompositeImplicitAutograd` and a
        `TorchDispatchMode` logger on 2.13.0 shows it firing *nothing* of its
        own: `x.flatten()`, `x.flatten(1)`, `x.flatten(1, 2)` and
        `x.flatten(0, -2)` each produce exactly one record,
        `aten.view.default`. So a `methods.json` entry would name
        `aten.flatten.using_ints`, a key no dispatcher ever sees -- the same
        complaint that keeps `softmax` and `chunk` out of that table.

        The body is `at::native::flatten`, transcribed:

            wrap both dims; start_dim <= end_dim or raise
            0-d          -> reshape to [1]      (NOT a no-op: shape (1,), measured)
            start == end -> self                (no op at all; the view is shared)
            otherwise    -> reshape with the run collapsed to its product

        `reshape`, not `view`, because that is what upstream's body calls --
        the single `view.default` in the trace is `reshape` lowering to it for
        a contiguous input, and a non-contiguous one would take `reshape`'s
        copying arm. Spelling `view` here would refuse where upstream copies.

        Two refusals, both measured and both upstream's words: `x.flatten(2, 1)`
        gives "flatten() has invalid args: start_dim cannot come after
        end_dim", and an out-of-range dim gives the ordinary
        `Dimension out of range` IndexError -- which `_wrap_dim` below raises
        with upstream's exact range, because "expected to be in range of
        [-3, 2]" is the part that tells the caller what to pass instead.
        """
        rank = self.dim()
        extent = max(rank, 1)

        def _wrap_dim(dim):
            wrapped = dim + extent if dim < 0 else dim
            if wrapped < 0 or wrapped >= extent:
                raise IndexError(
                    f"Dimension out of range (expected to be in range of "
                    f"[{-extent}, {extent - 1}], but got {dim})"
                )
            return wrapped

        start = _wrap_dim(start_dim)
        end = _wrap_dim(end_dim)
        if start > end:
            raise RuntimeError(
                "flatten() has invalid args: start_dim cannot come after end_dim"
            )
        if rank == 0:
            return dispatch("aten.reshape.default", self, [1])
        if start == end:
            return self
        sizes = list(self.shape)
        collapsed = 1
        for extent_at in sizes[start : end + 1]:
            collapsed *= extent_at
        shape = sizes[:start] + [collapsed] + sizes[end + 1 :]
        return dispatch("aten.reshape.default", self, shape)

    flatten.__name__ = "flatten"
    flatten.__qualname__ = "TensorBase.flatten"
    setattr(tensorbase, "flatten", flatten)


def _install_tensor_scalars(tensorbase, dispatch) -> None:
    """`item()` and `__bool__`, both of which leave the tensor world.

    `aten::item` exists, but it is not what upstream reaches: a
    `TorchDispatchMode` logger over `t.item()` records exactly
    `aten._local_scalar_dense.default`, and over `bool(t)` the same one. So
    that is the key, and the numel check that upstream does before it stays on
    this side, where it can carry torch's own message.
    """

    def item(self):
        if self.numel() != 1:
            raise RuntimeError(
                f"a Tensor with {self.numel()} elements cannot be converted to Scalar"
            )
        return dispatch("aten._local_scalar_dense.default", self)

    def __bool__(self):
        if self.numel() == 0:
            raise RuntimeError("Boolean value of Tensor with no values is ambiguous")
        if self.numel() != 1:
            raise RuntimeError(
                "Boolean value of Tensor with more than one value is ambiguous"
            )
        return bool(dispatch("aten._local_scalar_dense.default", self))

    def __float__(self):
        return float(item(self))

    def __int__(self):
        return int(item(self))

    def __index__(self):
        value = item(self)
        if isinstance(value, float):
            raise TypeError(
                "only integer tensors of a single element can be converted to an index"
            )
        return int(value)

    for name, fn in (
        ("item", item),
        ("__bool__", __bool__),
        ("__float__", __float__),
        ("__int__", __int__),
        ("__index__", __index__),
    ):
        fn.__name__ = name
        fn.__qualname__ = f"TensorBase.{name}"
        setattr(tensorbase, name, fn)


def _install_tensor_indexing(module, tensorbase, dispatch) -> None:
    """`x[...]` and `x[...] = v`, decomposed the way upstream decomposes them.

    Upstream's indexing is not one aten call; it is a walk over the index that
    emits `select.int` for an integer, `slice.Tensor` for a slice, `unsqueeze`
    for a `None`, and one `index.Tensor` at the end if any index was a tensor.
    Measured, on torch 2.13.0:

        f[0]        -> [select.int]
        f[0, 1]     -> [select.int, select.int]
        f[:, 1]     -> [select.int]          (the full slice emits nothing)
        f[0:1]      -> [slice.Tensor]
        f[None]     -> [unsqueeze]
        f[bool_t]   -> [index.Tensor]
        f[[0, 1]]   -> [lift_fresh, index.Tensor]
        f[..., [-2], :] -> [lift_fresh, index.Tensor]

    So this reproduces the walk rather than inventing a single `getitem` op,
    and every step goes through `_aten_dispatch`. What it does *not* do is
    mixed basic-and-advanced indexing: an index containing both a tensor and a
    non-trivial slice is refused by name instead of being approximated.

    `__setitem__` is the same walk read backwards, and the measurements that
    shape it are in its own docstring.
    """

    # Located with `is`, never with `==` or `.index()`: an index tuple may
    # hold a tensor, and `TensorBase.__eq__` now answers with a mask.
    # `tuple.index(Ellipsis)` would compare its way there elementwise.
    def _expand_ellipsis(self, index):
        ellipses = [k for k, item in enumerate(index) if item is Ellipsis]
        if not ellipses:
            return index
        if len(ellipses) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        consumed = sum(
            1 for item in index if item is not None and item is not Ellipsis
        )
        at = ellipses[0]
        fill = (slice(None),) * max(self.dim() - consumed, 0)
        return index[:at] + fill + index[at + 1 :]

    def _is_full_slice(item):
        return (
            isinstance(item, slice)
            and item.start is None
            and item.stop is None
            and item.step is None
        )

    # -- sequence indices (docs/architectures/ARCH20.md §7, the `falcon` wall) -------------
    #
    # `fused_qkv[..., [-2], :]` (`modeling_falcon.py:283`, `_split_heads`) is a
    # *list* in an index tuple, and it used to hit the "index of type list"
    # refusal at the bottom of the walk. Upstream lifts the list into an index
    # tensor and takes the advanced-indexing path -- measured on 2.13.0:
    #
    #     x[..., [-2], :]     -> [lift_fresh.default, index.Tensor]
    #     x[..., (0, 1), :]   -> [lift_fresh.default, index.Tensor]   (tuple too)
    #     x[[0, 1]]           -> [lift_fresh.default, index.Tensor]
    #     x[0, [1, 2]]        -> [select.int, lift_fresh.default, index.Tensor]
    #
    # and `torch.equal(x[..., [-2], :], x.index_select(1, tensor([1])))` is
    # True, so the lifted tensor is an ordinary index tensor with upstream's
    # ordinary negative-index wrapping. A list of `bool` becomes a *mask*
    # rather than a positional index (`x[[True, False]]` also lowers to
    # `lift_fresh` + `index.Tensor`), which is `_tensor_new_from_data`'s own
    # dtype inference and not something this file decides.
    def _is_sequence_index(item):
        return isinstance(item, (list, tuple))

    def _lift_sequence_index(item):
        """The `lift_fresh(_tensor_new_from_data(...))` pair upstream emits.

        Deliberately the same two steps `torch.tensor` takes (`_tensor_factory`
        above), because upstream's list index and upstream's `torch.tensor` are
        the same C++ path (`internal_new_from_data`) and produce the same
        single aten record."""
        return dispatch(
            "aten.lift_fresh.default", module._tensor_new_from_data(item, None, None)
        )

    def _index_tuple(index):
        """`treatSequenceAsTuple`, transcribed from
        `python_variable_indexing.cpp`.

        A **tuple** index is always a tuple of indices. A **list** index is the
        ambiguous one, and upstream resolves it by looking inside: a short list
        that contains a slice, an `Ellipsis`, a `None`, a tensor or another
        sequence is read as a *tuple of indices*; anything else (a plain list
        of numbers, or any list of 32 or more items) is read as one index
        tensor. Measured, both arms:

            x[[slice(None)]]  -> [alias.default]              tuple arm
            x[[[0, 1]]]       -> [lift_fresh, index.Tensor]   tuple arm, inner list lifts
            x[[0, 1]]         -> [lift_fresh, index.Tensor]   tensor arm

        Upstream also emits a `UserWarning` on the tuple arm ("Using a
        non-tuple sequence for multidimensional indexing is deprecated"). That
        warning is *not* reproduced here: it is a deprecation notice about
        Python-level spelling, it carries no information this shim can act on,
        and emitting warnings from the indexing hot path is a cost with no
        caller asking for it. The *behaviour* the warning describes is
        reproduced exactly.
        """
        if isinstance(index, tuple):
            return index
        if isinstance(index, list) and len(index) < 32:
            for item in index:
                if (
                    item is Ellipsis
                    or item is None
                    or isinstance(item, (slice, list, tuple, str))
                    or isinstance(item, tensorbase)
                ):
                    return tuple(index)
        return (index,)

    def __getitem__(self, index):
        index = _index_tuple(index)
        index = _expand_ellipsis(self, index)

        if any(
            isinstance(item, tensorbase) or _is_sequence_index(item) for item in index
        ):
            if any(
                not (
                    item is None
                    or isinstance(item, tensorbase)
                    or _is_sequence_index(item)
                    or _is_full_slice(item)
                )
                for item in index
            ):
                raise NotImplementedError(
                    "not implemented in torch._C shim: TensorBase.__getitem__ mixing "
                    "a tensor index with integer or slice indices -- upstream applies "
                    "basic indexing first and then aten.index.Tensor, and this shim "
                    "does not reproduce that composition yet"
                )
            result = self
            dim = 0
            for item in index:
                if item is None:
                    result = dispatch("aten.unsqueeze.default", result, dim)
                dim += 1
            indices = [
                _lift_sequence_index(item)
                if _is_sequence_index(item)
                else (item if isinstance(item, tensorbase) else None)
                for item in index
            ]
            return dispatch("aten.index.Tensor", result, indices)

        result = self
        dim = 0
        for item in index:
            if item is None:
                result = dispatch("aten.unsqueeze.default", result, dim)
                dim += 1
            elif isinstance(item, bool):
                # torch treats a plain `True`/`False` as a zero-dim mask, which
                # adds a dimension of length 1 or 0. Not measured as used, and
                # guessing it is the kind of thing this shim refuses.
                raise NotImplementedError(
                    "not implemented in torch._C shim: TensorBase.__getitem__ with a "
                    "Python bool index"
                )
            elif isinstance(item, int):
                result = dispatch("aten.select.int", result, dim, item)
            elif isinstance(item, slice):
                if _is_full_slice(item):
                    dim += 1
                    continue
                result = dispatch(
                    "aten.slice.Tensor",
                    result,
                    dim,
                    item.start,
                    item.stop,
                    1 if item.step is None else item.step,
                )
                dim += 1
            else:
                raise NotImplementedError(
                    f"not implemented in torch._C shim: TensorBase.__getitem__ with "
                    f"an index of type {type(item).__name__}"
                )
        return result

    __getitem__.__name__ = "__getitem__"
    __getitem__.__qualname__ = "TensorBase.__getitem__"
    setattr(tensorbase, "__getitem__", __getitem__)

    # The dtype upstream's `lift_fresh` produces for a bare Python number on
    # the right of a subscript assignment. **It is the receiver's, not the
    # number's** -- measured with a `TorchDispatchMode` logger on torch
    # 2.13.0, and it is not what `torch.tensor(v)` would infer:
    #
    #     float32 x;  x[t] = 5      ->  lift_fresh(float32()),  then index_put_
    #     int64   x;  x[t] = 5.0    ->  lift_fresh(int64()),    then index_put_
    #     float32 x;  x[t] = True   ->  lift_fresh(float32()),  then index_put_
    #     int64   x;  x[:] = 3.0    ->  lift_fresh(int64()),    then fill_
    #
    # This used to infer from the Python type instead, which agreed with
    # upstream whenever the two happened to line up and diverged otherwise.
    # It survived because `index_put_` only accepted a 1-D receiver, so the
    # one call that reached it (`inv_perm[perm] = arange(...)`, int64 on both
    # sides) was in the agreeing half. Both halves are reachable now, and
    # `index_put_` requires the dtypes to match exactly -- upstream does too
    # -- so an inferred `int64` against a `float32` receiver would refuse a
    # write upstream performs.
    def _lift(value, like):
        if isinstance(value, tensorbase):
            return value
        return dispatch("aten.scalar_tensor.default", value, dtype=like.dtype)

    def __setitem__(self, index, value):
        """`x[...] = v` -- the `__getitem__` walk, then a write through it.

        Measured with a `TorchDispatchMode` logger on torch 2.13.0, with the
        `zeros`/`ones` that built the operands stripped out:

            x[t] = tensor      -> [lift_fresh, lift_fresh, index_put_.default]
            x[t] = 5.0         -> [lift_fresh, lift_fresh, index_put_.default]
            x[boolmask] = 1.0  -> [lift_fresh, lift_fresh, index_put_.default]
            x[:, t] = tensor   -> [lift_fresh, index_put_.default]  (indices [None, t])
            x[:] = tensor      -> [copy_.default]
            x[...] = tensor    -> [copy_.default]
            x[:] = 3.0         -> [lift_fresh, fill_.Tensor]
            x[0] = 3.0         -> [lift_fresh, select.int, fill_.Tensor]
            x[0, 1] = 9.0      -> [lift_fresh, select.int, select.int, copy_.default]
            x[1:3] = tensor    -> [slice.Tensor, copy_.default]
            x[1:3] = 5.0       -> [lift_fresh, slice.Tensor, fill_.Tensor]
            x[None] = 0.0      -> [lift_fresh, unsqueeze, fill_.Tensor]
            0-d x[...] = 3.0   -> [lift_fresh, copy_.default]

        **The last five rows are why the choice between `copy_` and `fill_` is
        not "tensor on the right or number on the right".** An earlier reading
        of this table had `x[0] = 3.0` lowering to `copy_`, which is the shape
        the *number/tensor* rule predicts and not what torch does. Upstream's
        rule is in `python_variable_indexing.cpp::copy_to` and it is about
        shapes, so it is reproduced as such below:

            sizes equal            -> copy_
            else source is 0-d     -> fill_
            else                   -> broadcast, then copy_

        `x[0] = 3.0` takes the middle arm ((3,) destination, 0-d source) and a
        0-d destination takes the first, which is why `0-d x[...] = 3.0` is a
        `copy_` while `x[:] = 3.0` on a matrix is a `fill_`.

        **The basic-index write works now, and what changed is not here.**
        `aten.select.int` and `aten.slice.Tensor` (step 1) always aliased
        their input -- candle's `narrow`/`squeeze` clone the storage `Arc` and
        rebuild only the `Layout`. What was missing was the *write*: every
        in-place op swapped the wrapper's tensor for a freshly computed one
        instead of writing into the buffer that wrapper pointed at, so the
        upstream sequence ran to completion and changed nothing. That is now
        `PyTensorBase::write_into` (docs/kernels/VIEWS.md §6), and this branch is the
        caller it was built for.

        **Two things still refuse, and neither is a view problem.**

          * A slice with `step != 1`. `slice.Tensor` handles it by
            `index_select`, which *copies* -- measured in docs/kernels/VIEWS.md §4,
            `x` and `slice.Tensor(x, 0, 5, 2)` hold independent buffers. So
            the walk would produce something that is not a view of `self`, and
            writing into it would be exactly the silent no-op this branch used
            to exist to prevent.
          * Mixed basic and advanced indexing, unchanged and for the same
            reason as in `__getitem__`.

        A **sequence** index (`x[[0, 2]] = v`) takes the advanced arm, lifted
        by the same `_lift_sequence_index` the read side uses -- the two walks
        have to agree on what an index *is*, or `x[i] = x[i]` would take two
        different routes.
        """
        index = _index_tuple(index)
        index = _expand_ellipsis(self, index)

        # A step != 1 slice on the *write* side, lowered to `index_put_`.
        #
        # `aten.slice.Tensor` reaches a step above 1 through `index_select`,
        # which materialises (docs/kernels/VIEWS.md §6.4), so the basic walk below
        # would narrow to a tensor that does not share storage with the
        # receiver and the write would be silently lost. The positions the
        # slice names are handed to `index_put_` as an integer index instead,
        # and `index_put_` writes through the receiver's own buffer.
        #
        # Restricted, on purpose, to **one** stepped slice with every other
        # item a full slice: `index_put_` implements a single index group, and
        # a second one has to be refused rather than approximated. That covers
        # every call site the 297-architecture sweep found (docs/bindings/SETITEM.md
        # §1) -- `pe[:, 0::2] = v` and `freqs_t[..., k::3] = v`.
        #
        # The dtype cast is not incidental. Upstream reaches this through
        # `copy_`, which casts (measured: `int64 x; x[0::2] = [1.7, 2.7, 3.7]`
        # gives `[1, 0, 2, 0, 3, 0]`), while `index_put_` requires the dtypes
        # to match exactly. Without the cast this path would refuse a write
        # upstream performs.
        _stepped = [
            k
            for k, item in enumerate(index)
            if isinstance(item, slice) and item.step is not None and item.step != 1
        ]
        if _stepped:
            if len(_stepped) > 1 or any(
                not (
                    _is_full_slice(item)
                    or (isinstance(item, slice) and item.step not in (None, 1))
                )
                for item in index
            ):
                raise NotImplementedError(
                    "not implemented in torch._C shim: TensorBase.__setitem__ with a "
                    "stepped slice alongside another non-trivial index -- the write "
                    "lowers to aten.index_put_, which implements a single index "
                    "group, and a second group is refused rather than approximated "
                    "(docs/bindings/SETITEM.md §3)"
                )
            axis = _stepped[0]
            item = index[axis]
            if item.step <= 0:
                # torch's own wording, from aten.slice.Tensor.
                raise ValueError(f"step must be greater than zero, got {item.step}")
            positions = list(range(*item.indices(self.shape[axis])))
            source = _lift(value, self)
            if source.dtype != self.dtype:
                source = dispatch("aten._to_copy.default", source, dtype=self.dtype)
            if not positions:
                # Upstream writes nothing here but still checks the broadcast,
                # and raises with `copy_`'s wording rather than
                # `index_put_`'s. Not reproduced: an empty stepped write is
                # not a form any swept architecture uses (docs/bindings/SETITEM.md §4).
                return
            dispatch(
                "aten.index_put_.default",
                self,
                [None] * axis + [_lift_sequence_index(positions)],
                source,
                False,
            )
            return

        if any(
            isinstance(item, tensorbase) or _is_sequence_index(item) for item in index
        ):
            if any(
                not (
                    item is None
                    or isinstance(item, tensorbase)
                    or _is_sequence_index(item)
                    or _is_full_slice(item)
                )
                for item in index
            ):
                raise NotImplementedError(
                    "not implemented in torch._C shim: TensorBase.__setitem__ mixing "
                    "a tensor index with integer or slice indices -- upstream applies "
                    "basic indexing first and then aten.index_put_, and this shim "
                    "does not reproduce that composition yet"
                )
            indices = [
                _lift_sequence_index(item)
                if _is_sequence_index(item)
                else (item if isinstance(item, tensorbase) else None)
                for item in index
            ]
            dispatch("aten.index_put_.default", self, indices, _lift(value, self), False)
            return

        # The `__getitem__` walk, verbatim in its op choices -- an index that
        # reads as `x[0, 1:3]` must narrow to the same view whether it is being
        # read or written, and the only way to guarantee that is to emit the
        # same keys with the same arguments.
        view = self
        dim = 0
        for item in index:
            if item is None:
                view = dispatch("aten.unsqueeze.default", view, dim)
                dim += 1
            elif isinstance(item, bool):
                raise NotImplementedError(
                    "not implemented in torch._C shim: TensorBase.__setitem__ with a "
                    "Python bool index"
                )
            elif isinstance(item, int):
                view = dispatch("aten.select.int", view, dim, item)
            elif isinstance(item, slice):
                if _is_full_slice(item):
                    dim += 1
                    continue
                step = 1 if item.step is None else item.step
                if step != 1:
                    raise NotImplementedError(
                        "not implemented in torch._C shim: TensorBase.__setitem__ with "
                        f"a step-{step} slice -- aten.slice.Tensor reaches a step above "
                        "1 through index_select, which copies rather than narrowing, so "
                        "the write would land in a tensor that does not share storage "
                        "with the receiver and would be silently lost. Reading "
                        "`x[::2]` is fine; writing to it is what is missing "
                        "(docs/kernels/VIEWS.md §6.4)"
                    )
                view = dispatch(
                    "aten.slice.Tensor", view, dim, item.start, item.stop, step
                )
                dim += 1
            else:
                raise NotImplementedError(
                    f"not implemented in torch._C shim: TensorBase.__setitem__ with "
                    f"an index of type {type(item).__name__}"
                )

        # `copy_to`, upstream's own three arms. `_lift` is a no-op for a tensor
        # on the right, so `source` is whatever upstream would have handed to
        # `copy_to` after `lift_fresh`.
        source = _lift(value, self)
        if tuple(view.shape) == tuple(source.shape):
            dispatch("aten.copy_.default", view, source)
        elif source.dim() == 0:
            dispatch("aten.fill_.Tensor", view, source)
        else:
            # Upstream expands first; `copy_` here broadcasts the source into
            # the destination's shape itself, and refuses the same pairs.
            dispatch("aten.copy_.default", view, source)

    __setitem__.__name__ = "__setitem__"
    __setitem__.__qualname__ = "TensorBase.__setitem__"
    setattr(tensorbase, "__setitem__", __setitem__)

    def __len__(self):
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    setattr(tensorbase, "__len__", __len__)


# The class name upstream gives the node an op produces, where the naive rule
# does not get it right.
#
# **Measured, not derived.** `docs/training/BACKWARD4.md` §3.1 ran 48 ops through real
# torch 2.13.0 and read `type(y.grad_fn).__name__` off each; the naive rule
# below -- CamelCase the aten base name and append `Backward0` -- agreed on 41
# of them. These are the seven that disagreed, and each one disagrees for a
# reason worth keeping visible:
#
#   mul.Scalar     MulBackward1     the trailing digit is an *overload* index,
#   squeeze.dim    SqueezeBackward1  not always 0, and it is not derivable
#   max.default    MaxBackward1      from the aten name at all
#   reshape        ViewBackward0    upstream decomposes before it records
#   linear         AddmmBackward0   likewise
#   to.dtype       ToCopyBackward0  the recorded op is the copy, not the cast
#   contiguous     (no node)        a no-op returns its own input; the door's
#                                    identity test already handles this one
#
# A name outside this table falls back to the rule, and a fallback that is
# wrong is a *smaller* divergence than the one it replaces: before this round
# the shim printed no `grad_fn=` field at all for any intermediate. That is the
# whole claim being made for the fallback, and
# `test_grad_fn_names_agree_with_upstream` checks the table against real torch
# so it cannot rot into a guess.
_GRAD_FN_NAMES = {
    "aten.mul.Scalar": "MulBackward1",
    "aten.squeeze.dim": "SqueezeBackward1",
    "aten.max.default": "MaxBackward1",
    "aten.reshape.default": "ViewBackward0",
    # Measured, not derived, and both were wrong in the first pass:
    #   aten.matmul.default -> MmBackward0     upstream decomposes matmul
    #     before autograd records, so there is no `MatmulBackward0` node
    #     anywhere in upstream -- same class as `reshape` and `linear`
    #     above, and it was simply missed.
    #   aten.clamp.default  -> ClampBackward1  the trailing digit is an
    #     overload index and this one is 1 for every clamp form measured
    #     (both bounds, min only, and the `Tensor.clamp` spelling).
    "aten.matmul.default": "MmBackward0",
    "aten.clamp.default": "ClampBackward1",
    # docs/numerics/SCALAR2.md §4, measured the same way. Both are overload indices the
    # rule cannot derive, and both are reached by an ordinary Python operator:
    #   aten.rsub.Scalar -> RsubBackward1   `2 - x`
    #   aten.pow.Scalar  -> PowBackward2    `2 ** x`
    # Neither is a dispatch divergence -- upstream runs the same aten key for
    # these two spellings that this build does -- which is what separates them
    # from the add/sub/mul/div rows §2 records and does not change.
    "aten.rsub.Scalar": "RsubBackward1",
    "aten.pow.Scalar": "PowBackward2",
    "aten.linear.default": "AddmmBackward0",
    "aten.to.dtype": "ToCopyBackward0",
    "aten._to_copy.default": "ToCopyBackward0",
    "aten._softmax.default": "SoftmaxBackward0",
    "aten._log_softmax.default": "LogSoftmaxBackward0",
    "aten._unsafe_view.default": "UnsafeViewBackward0",
}

_GRAD_FN_CLASSES = {}


def _grad_fn_name(op: str) -> str:
    """`aten.native_layer_norm.default` -> `NativeLayerNormBackward0`."""
    known = _GRAD_FN_NAMES.get(op)
    if known is not None:
        return known
    base = op.split(".")[1] if "." in op else op
    parts = [p for p in base.strip("_").split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) + "Backward0"


class _GradFnNode:
    """What `grad_fn` returns for a non-leaf, and it is deliberately hollow.

    `docs/training/BACKWARD4.md` §1.3 asked which caller is the first to reach past
    nullness and answered it by counting: on a real `from_pretrained` +
    `.train()` + forward, `grad_fn` is read 546 times and **every one of them is
    a nullness or a truthiness test**. The 547th, and the only one that reaches
    further on any path this shim runs, is `torch/_tensor_str.py:646`:

        grad_fn_name = type(grad_fn).__name__

    So the class *name* is the one attribute that has to be faithful, and it is
    -- see `_GRAD_FN_NAMES`. Everything else upstream's node has
    (`next_functions`, `name()`, `metadata`, `register_hook`, `_saved_*`, and
    an `apply` that computes) is **absent rather than stubbed**, so a caller
    that needs a graph gets an `AttributeError` naming exactly what it wanted
    instead of an empty tuple that reads as "this node has no inputs".

    That distinction is the whole boundary between this round and the
    structural group. `next_functions = ()` would be a graph of one node with
    no edges, which is a false statement about a real graph; no attribute at
    all is a true statement about a shim that has none.
    """

    __slots__ = ("_shim_op",)

    def __init__(self, op):
        self._shim_op = op

    def __repr__(self):
        return f"<{type(self).__name__} object at {id(self):#x}>"

    def __getattr__(self, name):
        """Everything a real node has, refused by name.

        A bare `AttributeError` would be the truth -- there is no attribute --
        but it is the *shape* of truth a caller misreads. `getattr(node,
        "next_functions", ())` and every `hasattr` guard in
        `torch/nn/modules/module.py` would take it as "this node happens not to
        have inputs" and carry on, which is the silent-wrong-answer failure
        `docs/training/BACKWARD.md` §5.2 is about. So the refusal says what is absent and
        that it is absent by construction.

        Dunders are excluded and still raise `AttributeError`, because that is
        the protocol: `copy`, `pickle` and `repr` all probe for optional
        `__reduce__`/`__deepcopy__` hooks and read a raise as "not provided".
        Turning those probes into `NotImplementedError` would break callers that
        never asked about autograd at all.
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise NotImplementedError(
            f"not implemented in torch._C shim: grad_fn.{name} -- "
            f"{type(self).__name__} records that {self._shim_op} produced this "
            "tensor and nothing else. There is no graph behind it: no "
            "next_functions, no saved operands, no apply. "
            "Tensor.backward() does not read this and works anyway: it walks "
            "the eager tape, which records the same ops from the other side "
            "(docs/training/BACKWARD9.md). See docs/training/BACKWARD4.md §1.3, docs/training/BACKWARD2.md §2."
        )


def _grad_fn(self):
    """`t.grad_fn` -- `None` for a leaf, an opaque node for anything else.

    Built on read rather than at the door. The door stores one string
    (`tensor.rs`'s `from_op`); allocating a Python object per op instead would
    have put an allocation on the hot path for a value that a real
    `.train()` forward reads **zero** times (docs/training/BACKWARD4.md §1.1).
    """
    op = self._shim_from_op
    if op is None:
        return None
    cls = _GRAD_FN_CLASSES.get(op)
    if cls is None:
        cls = type(_grad_fn_name(op), (_GradFnNode,), {"__slots__": ()})
        _GRAD_FN_CLASSES[op] = cls
    return cls(op)


def _install_autograd_shape(tensorbase) -> None:
    """`requires_grad`, `grad_fn`, `is_leaf`, `data` -- the papered-over part.

    **This is the one group here that is not an implementation.** There is no
    autograd behind this shim (DESIGN.md §3 stage 0), and `from_config` reaches
    `TensorBase.requires_grad_` before it reaches anything else interesting, so
    the choice was between stopping there and carrying an inert flag.

    The flag is carried, and the boundary is drawn where it can be seen:

      * `requires_grad` stores and reports what was set. Nothing reads it --
        but only a floating-point or complex tensor may carry it, which is
        upstream's rule and was missing here until docs/training/BACKWARD2.md §1.4
        measured it. `TensorBase.set_requires_grad` in `tensor.rs` is where it
        is stated; `_refuse_non_floating_requires_grad` below is the same rule
        under `requires_grad_`'s own upstream wording.
      * `grad_fn` and `grad` are always `None`, which is the truth -- no graph
        node was ever created and no gradient was ever accumulated.
      * `is_leaf` is always `True`, and **this one is not the truth** -- it is
        the one claim in this group that upstream would contradict, so it is
        the one to know about before extending anything here.

        Upstream does not store `is_leaf`; it *is* `grad_fn is None`, and it is
        the predicate that separates "a tensor a user made" from "a tensor an
        op made". `torch/optim/optimizer.py:1153` is the load-bearing reader --

            if not self.defaults.get("differentiable", None) and not (
                param.is_leaf or param.retains_grad
            ):
                raise ValueError("can't optimize a non-leaf Tensor")

        -- so with `True` here, `torch.optim.SGD([x * 2])` is **accepted** and
        upstream raises. Measured both ways, docs/training/BACKWARD3.md §1.1, along with
        the two other guards that key on the same predicate
        (`requires_grad_` and `setattr` on a non-leaf, which upstream refuses
        with "you can only change requires_grad flags of leaf variables").

        It is nevertheless the honest report *given* the rest of this group: no
        op propagates `requires_grad` (docs/training/BACKWARD2.md §1, W4), so a tensor an
        op made here really does behave as a constant -- no gradient reaches it,
        nothing accumulates into it, and `backward()` refuses. `is_leaf` cannot
        be made to say otherwise without `grad_fn` becoming a node; there is no
        version of it that is a smaller change, which is docs/training/BACKWARD3.md §3
        and the reason W4/W6/W7 did not land as a group.

        Note also that `retains_grad` is *unimplemented* and is unreachable
        only because Python short-circuits the `or` above. Whatever gives
        `is_leaf` a second answer has to give `retains_grad` a first one, or
        upstream's `ValueError` becomes a shim refusal naming the wrong thing.
      * `backward()` stays a raising stub, so code that actually depends on
        the flag meaning something fails by name rather than silently getting
        zeros. That refusal is the *whole* of the boundary now: since
        docs/training/BACKWARD2.md §4.1 the factory keyword carries the flag instead of
        refusing it, because `requires_grad_` had always carried it and the two
        spellings disagreeing protected nobody.

    `data` returns `self`, not a detached alias. Upstream's `.data` shares
    storage with the original, so `p.data.normal_()` writes through to `p`;
    returning `self` gives that same write-through with the same object, and
    differs in that our `.data` still reports the original `requires_grad`.
    Recorded in docs/bindings/TENSORBASE.md rather than hidden.
    """

    def _make_subclass(cls, data, require_grad=False, dispatch_strides=False,
                       dispatch_device=False, device_for_backend_keys=None):
        """`torch.Tensor._make_subclass`, which is how a `Parameter` is born.

        `torch/nn/parameter.py:57` is `torch.Tensor._make_subclass(cls, data,
        requires_grad)` and two more sites do the same. Upstream re-wraps the
        same `TensorImpl` in a new Python object of class `cls`; here `cls(data)`
        reaches `TensorBase`'s `#[new]`, which PyO3 allocates with the subtype
        it was called on -- so the result really is a `Parameter`, and
        `nn.Module.__setattr__`'s `isinstance(value, Parameter)` is satisfied.

        The trailing three arguments are dispatch-key plumbing for subclasses
        that override strides or device; there is no dispatcher key set here,
        so they are accepted and ignored rather than refused -- refusing would
        stop a call that upstream makes with their defaults.
        """
        # `TensorBase.__new__(cls, ...)`, not `cls(...)`. `Parameter.__new__`
        # is the caller here, and calling `cls(data)` would re-enter it --
        # measured, as a `RecursionError` five frames deep. Upstream allocates
        # directly too; `_make_subclass` is below `__new__`, not above it.
        made = tensorbase.__new__(cls, data)
        made.requires_grad = bool(require_grad)
        return made

    # Upstream's is a C builtin, and `inspect.signature` raises `ValueError` on
    # it. `torch/_dynamo/decorators.py:966` catches exactly that and skips its
    # signature comparison, so upstream never compares. A Python function *has*
    # a readable signature, so the comparison ran here -- and rejected it,
    # because `torch/_dynamo/polyfills/tensor.py:12` spells the flag
    # `requires_grad` while the keyword upstream actually accepts is
    # `require_grad` (measured on 2.13.0: `requires_grad=` is a `TypeError`
    # there), which is also what `torch/_C/__init__.pyi:2389` declares.
    #
    # Renaming to match the polyfill would make this shim accept a keyword
    # upstream rejects and reject the one it accepts. So the signature is
    # withheld instead, which is the same amount of information upstream gives
    # -- and `sig_ident(...) != sig_ident(wildcard_sig)` in that same function
    # is dynamo's own allowance for a callable that declares nothing. The `def`
    # above and the stub remain the statement of what is accepted.
    _make_subclass.__signature__ = inspect.signature(lambda *args, **kwargs: None)

    # A *static* method, not a class method: upstream's is called as
    # `torch.Tensor._make_subclass(cls, data, requires_grad)` with the target
    # class passed explicitly, so binding the receiver would shift every
    # argument one place left and `Parameter` would arrive as the data.
    tensorbase._make_subclass = staticmethod(_make_subclass)

    def requires_grad_(self, mode=True):
        # Upstream refuses this on a non-leaf before it looks at the dtype, with
        # a longer message than the attribute setter's -- both were transcribed
        # from 2.13.0 by running the failing case, and the difference between
        # them is upstream's, not this shim's. docs/training/BACKWARD3.md §1.1 listed
        # this as a live divergence that `is_leaf` being hardcoded made
        # unreachable; docs/training/BACKWARD4.md §4.1 is it closed.
        if self._shim_from_op is not None:
            raise RuntimeError(
                "you can only change requires_grad flags of leaf variables. If "
                "you want to use a computed variable in a subgraph that doesn't "
                "require differentiation use var_no_grad = var.detach()."
            )
        # Upstream's *third* wording of the one dtype rule, transcribed from
        # 2.13.0 by running the failing case. `requires_grad_` says "floating
        # point dtype" and nothing about complex -- and then accepts a
        # `complex64` tensor anyway, which is upstream's message being loose
        # rather than upstream's behaviour differing. The behaviour is the same
        # rule `TensorBase.set_requires_grad` enforces; only the text differs,
        # and the text is what a caller greps for. docs/training/BACKWARD2.md §4.2.
        _refuse_non_floating_requires_grad(
            self, bool(mode),
            "only Tensors of floating point dtype can require gradients",
        )
        self.requires_grad = bool(mode)
        return self

    requires_grad_.__name__ = "requires_grad_"
    requires_grad_.__qualname__ = "TensorBase.requires_grad_"
    setattr(tensorbase, "requires_grad_", requires_grad_)
    setattr(tensorbase, "grad_fn", property(_grad_fn))

    def _set_grad(self, value):
        """`p.grad = ...`, which is a real slot now. docs/training/BACKWARD.md.

        The docstring above says `grad` is always `None`, "which is the truth
        -- no gradient was ever accumulated", and docs/training/AUTOGRAD.md §7 argued
        explicitly for keeping it that way *while nothing writes to it*. The
        tape writes to it, so that antecedent is gone: what would be dishonest
        now is a `torch.optim` step that silently skipped every parameter
        because the slot it reads cannot be filled.

        **Something fills it implicitly now** (docs/training/BACKWARD9.md):
        `Tensor.backward()` reaches `_ImperativeEngine.run_backward`, which
        accumulates into this slot the way upstream's `AccumulateGrad` does --
        a dense copy on the first backward and an in-place `+=` after.
        `CaptureTrace.backward()` still *returns* gradients for the caller to
        assign, the shape `torch.optim.sgd.sgd` already had (docs/training/LOSS.md §6.4),
        and the two coexist because the setter does not care who writes.
        `None` is accepted because it is what `zero_grad(set_to_none=True)`,
        the default, writes.
        """
        if value is not None and not isinstance(value, tensorbase):
            raise TypeError(
                "torch._C shim: Tensor.grad can only be set to a TensorBase or "
                f"None, got {type(value).__name__}"
            )
        self._shim_grad = value

    setattr(tensorbase, "grad", property(lambda self: self._shim_grad, _set_grad))
    # Always `True`. The one claim in this group upstream would contradict, and
    # the docstring above says which upstream guard reads it and what that
    # costs -- `test_the_backward_seed_is_absent_and_nothing_guesses_a_one`
    # pins the consequence so it cannot go stale in prose.
    # Derived, not asserted. Upstream does not store `is_leaf` -- it *is*
    # `grad_fn is None`, which docs/training/BACKWARD3.md §3 is entirely about, and
    # writing it as anything else here would re-create the divergence this
    # round exists to close.
    setattr(tensorbase, "is_leaf", property(lambda self: self._shim_from_op is None))
    setattr(
        tensorbase,
        "retains_grad",
        property(lambda self: bool(self._shim_retains_grad)),
    )
    # The getter is `self` (docs/bindings/TENSORBASE.md records why it is not a detached
    # view). The *setter* is what `nn.Module._apply` needs -- see
    # `_shim_set_data` in tensor.rs for what it costs and what it agrees with.
    def _set_data(self, value):
        if not isinstance(value, tensorbase):
            raise TypeError(
                "torch._C shim: Tensor.data can only be set to a TensorBase, got "
                f"{type(value).__name__}"
            )
        self._shim_set_data(value)

    setattr(tensorbase, "data", property(lambda self: self, _set_data))
    def retain_grad(self):
        """`t.retain_grad()`, which was `lambda self: None` until this round.

        Upstream's contract has two halves and this implements the half that is
        true here. `retains_grad` starts reporting `True`, which is what
        `torch/optim/optimizer.py:1153` reads -- and reading it is only
        *possible* now that `is_leaf` stopped short-circuiting the `or`
        (docs/training/BACKWARD3.md §1.2). What does not happen is the other half:
        **no gradient is populated into a non-leaf's `.grad`.** That is no
        longer because `Tensor.backward()` refuses -- it answers since
        docs/training/BACKWARD9.md, and fills every *leaf*'s `.grad` -- but because the
        engine accumulates onto the tape's constants, and a non-leaf is a node
        result rather than a constant. The tape holds its value; nothing
        surfaces it. docs/training/BACKWARD9.md §6 names it as unbuilt.

        On a leaf it is a no-op with no flag set, which is upstream's behaviour
        too -- upstream returns early for a tensor that is already an
        accumulating leaf, and `x.retain_grad(); x.retains_grad` is `False`
        there as well (measured on 2.13.0, docs/training/BACKWARD4.md §3.3).
        """
        if not self.requires_grad:
            raise RuntimeError(
                "can't retain_grad on Tensor that has requires_grad=False"
            )
        if self._shim_from_op is not None:
            self._shim_set_retains_grad(True)
        return None

    retain_grad.__name__ = "retain_grad"
    retain_grad.__qualname__ = "TensorBase.retain_grad"
    setattr(tensorbase, "retain_grad", retain_grad)


def _install_grad_mode(module, varfns) -> None:
    """The grad-mode flags `torch.no_grad()` turns on and off.

    docs/models/FROM_CONFIG.md §2.2 measured `_set_grad_enabled` at **84 calls** during
    `AutoModelForCausalLM.from_config` -- the single most-called name in the
    whole trace, because every `@torch.no_grad()`-decorated initialiser flips it
    twice. So this is not an edge case that can be left refusing.

    **What is implemented is the flag, not what the flag means.** The state
    round-trips exactly: `no_grad()` reads the previous value, sets `False`,
    restores it, and gets back what it stored. That is the whole of the
    observable contract at this layer, and `torch/autograd/grad_mode.py` needs
    nothing else. What does *not* exist is the thing the flag would govern --
    there is no graph, so turning recording "on" records nothing either. It is
    the same boundary `TensorBase.requires_grad` draws (see
    `_install_autograd_shape`), and it is drawn in the same place: `backward()`
    stays a raising stub.

    `is_grad_enabled` is overwritten on `_VariableFunctions` as well as on the
    module. `torch/__init__.py` harvests `torch.is_grad_enabled` off that
    object, so leaving the harvested copy as the table-less refusal would make
    `torch.is_grad_enabled()` and `torch._C.is_grad_enabled()` disagree.
    """
    state = {
        "grad": True,
        # `torch/autograd/grad_mode.py:340` and `:393`. Both are real
        # backend-configuration switches (thread pool, layout enforcement) with
        # nothing behind them here, and both are context managers that restore
        # what they read -- so, like `grad`, they have to round-trip.
        "multithreading": True,
        "layout_enforcement": False,
        # `torch.is_inference_mode_enabled()`. Upstream spells this ONLY at
        # torch level -- there is no `torch._C._is_inference_mode_enabled` on
        # 2.13.0 -- and it is harvested off `_VariableFunctions`, so leaving it
        # as the table-less refusal made `fake_tensor.py:1801` stop the whole of
        # `torch.export` on a predicate that has an obvious answer.
        #
        # It lives HERE, in the same dict as `grad`, rather than as a constant,
        # because docs/graph/EXPORT.md §2.2 is about exactly the other choice: a
        # constant `False` would make `with torch.inference_mode():` a block
        # that enters, reports itself absent, and changes nothing. The setter
        # below is what `_InferenceMode.__enter__` writes through, so the guard
        # and the predicate cannot drift apart.
        "inference": False,
    }

    def is_grad_enabled():
        return state["grad"]

    def _set_grad_enabled(mode):
        # Two writes, one source of truth. `state` stays the value
        # `torch.is_grad_enabled()` reports; the mirror is what the *door* reads
        # once per op, because W5 has to be gated by grad mode and a dict lookup
        # per dispatch is not a cost this shim can add for callers who never
        # differentiate. See `tensor.rs`'s `GRAD_ENABLED`, and
        # `test_no_grad_gates_grad_fn` for the two being checked against each
        # other rather than assumed equal.
        state["grad"] = bool(mode)
        module._shim_set_grad_enabled_flag(bool(mode))

    def is_inference_mode_enabled():
        return state["inference"]

    def _set_inference_mode_enabled(mode):
        # Not an upstream spelling. It exists so the `_InferenceMode` guard has
        # somewhere to write that the predicate can read, instead of each
        # keeping its own flag and agreeing only by luck.
        state["inference"] = bool(mode)

    def _is_multithreading_enabled():
        return state["multithreading"]

    def _set_multithreading_enabled(mode):
        state["multithreading"] = bool(mode)

    def _is_grad_layout_enforcement_enabled():
        return state["layout_enforcement"]

    def _set_grad_layout_enforcement_enabled(mode):
        state["layout_enforcement"] = bool(mode)

    for name, fn in (
        ("is_grad_enabled", is_grad_enabled),
        ("_set_grad_enabled", _set_grad_enabled),
        ("is_inference_mode_enabled", is_inference_mode_enabled),
        ("_set_inference_mode_enabled", _set_inference_mode_enabled),
        ("_is_multithreading_enabled", _is_multithreading_enabled),
        ("_set_multithreading_enabled", _set_multithreading_enabled),
        ("_is_grad_layout_enforcement_enabled", _is_grad_layout_enforcement_enabled),
        ("_set_grad_layout_enforcement_enabled", _set_grad_layout_enforcement_enabled),
    ):
        fn.__name__ = name
        fn.__qualname__ = f"torch._C.{name}"
        setattr(module, name, fn)
    varfns.is_grad_enabled = is_grad_enabled
    # Same reason as `is_grad_enabled` above: `torch/__init__.py` harvests this
    # name off `_VariableFunctions`, and the harvested copy is the one
    # `torch.is_inference_mode_enabled` ends up being.
    varfns.is_inference_mode_enabled = is_inference_mode_enabled
    # Readable so the state can be inspected rather than inferred.
    module._shim_grad_state = state


def _install_thread_local_store(module) -> None:
    """`_stash_obj_in_tls` and its three siblings -- thread plumbing, not autograd.

    `torch/autograd/graph.py:989` `_engine_run_backward` stashes a
    `contextvars.Context` under the key `"context"` so that the engine's
    **device threads** can read the compiler config, and removes it in a
    `finally:`. That is the whole observable contract at this layer, and a
    thread-local key/value store is a complete implementation of it -- the same
    judgement `_install_grad_mode` records for `no_grad()`'s flag, where what is
    implemented is the flag and not what the flag would govern.

    **The reason to implement it is that it was standing in front of the wall
    that matters.** docs/training/BACKWARD2.md §1.2 measured what a user got before:

        >>> torch.ones(2,2).requires_grad_(True).sum().backward()
        NotImplementedError: not implemented in torch._C shim: torch._C._stash_obj_in_tls

    which says nothing about autograd, because `_stash_obj_in_tls` is not
    autograd. Worse, stubbing only that one does not reveal the engine either:
    `_engine_run_backward`'s `finally:` calls `_remove_obj_from_tls`, which then
    raises **while the engine's own refusal is unwinding** and replaces it.
    docs/training/BACKWARD.md §14.1 met the same masking shape inside `torch.save` and
    called a refusal a `finally:` overwrites "a refusal nobody can size".

    `threading.local()` rather than a module dict, deliberately. A dict would
    agree with upstream in every single-threaded test that could be written here
    and be a different semantics -- and the *reason* the key exists at all is
    that upstream has more than one thread.
    """
    store = threading.local()

    def _slots():
        # `threading.local` gives each thread a fresh, empty instance, so the
        # map has to be created on first touch per thread rather than once.
        try:
            return store.slots
        except AttributeError:
            store.slots = {}
            return store.slots

    def _stash_obj_in_tls(key, value):
        _slots()[key] = value

    def _get_obj_in_tls(key):
        # Upstream raises if the key is absent rather than returning None, and
        # `torch/_dynamo` reads this key only under an `_is_key_in_tls` guard.
        # A `KeyError` here is the honest report; a `None` would be a value.
        return _slots()[key]

    def _is_key_in_tls(key):
        return key in _slots()

    def _remove_obj_from_tls(key):
        _slots().pop(key, None)

    for name, fn in (
        ("_stash_obj_in_tls", _stash_obj_in_tls),
        ("_get_obj_in_tls", _get_obj_in_tls),
        ("_is_key_in_tls", _is_key_in_tls),
        ("_remove_obj_from_tls", _remove_obj_from_tls),
    ):
        fn.__name__ = name
        fn.__qualname__ = f"torch._C.{name}"
        setattr(module, name, fn)


def _install_engine(module) -> None:
    """`_ImperativeEngine.run_backward` -- the wall, **opened**. docs/training/BACKWARD9.md.

    This is where both `Tensor.backward()` and `torch.autograd.grad()` land
    (docs/training/BACKWARD2.md §1.3 measured that it is one wall and not two), and for
    eleven rounds it refused by name. `docs/training/BACKWARD7.md` §6 listed what stood
    between the refusal and an engine; `docs/training/BACKWARD8.md` closed the two that
    were defects; this closes the rest that a training loop needs.

    **It is a translation, not a second engine.** Everything below turns
    upstream's call into `torch._C._eager_backward(output, grad_output, wrt,
    retain_graph)` and turns the answer back into upstream's shape. No
    derivative rule, no traversal and no lifetime rule lives here -- those are
    `tape.rs` and `capture.rs`, and putting any of them here would be a second
    place for the two to disagree.

    Upstream's signature, which is C++ and positional:

        run_backward(tensors, grad_tensors, keep_graph, create_graph, inputs,
                     allow_unreachable=False, accumulate_grad=False)

    `torch.autograd.backward` passes `allow_unreachable=True,
    accumulate_grad=True` and ignores the return; `torch.autograd.grad` passes
    **`allow_unused` in the `allow_unreachable` slot**, `accumulate_grad=False`,
    and returns the tuple. So the two entry points differ here by exactly two
    flags, which is why one function serves both and why `allow_unused` is not
    a separate parameter.

    Three things are refused by name rather than approximated, and each is
    refused because answering would be a *wrong* answer rather than a slow one:
    `create_graph=True` (the backward runs under `NoGradGuard`, so its own ops
    are not on the tape and a second backward would silently see nothing),
    more than one root tensor (the eager tape has one output and summing seeds
    across roots is not what upstream does), and `GradientEdge` inputs.

    **`.grad` accumulation is the piece with design content**, and
    `_accumulate_into_grad` below is where it is.
    """
    engine = getattr(module, "_ImperativeEngine", None)
    if engine is None:
        return

    def _no_grad_call(fn, *args):
        """Run `fn` with grad mode off and restore it, whatever happens.

        Accumulation is not differentiable and must not be recorded: an
        `add_` into `p.grad` under grad mode would be a write the *next*
        tape could see. It is also the write docs/training/BACKWARD8.md §2.3's guard is
        about, so doing it off the tape is what keeps the two independent --
        `forgive_own_write` forgives an op its own write, and this is not one.
        """
        was = module.is_grad_enabled()
        module._set_grad_enabled(False)
        try:
            return fn(*args)
        finally:
            module._set_grad_enabled(was)

    def _dense_copy_of(gradient):
        """A gradient the caller can write to, which `clone()` alone is not.

        `sum()`'s gradient is the seed *expanded* to the operand's shape --
        `stride() == (0, 0)`, one element of storage read by every position --
        and `clone()` preserves that layout. Storing it as `p.grad` gives a
        tensor whose first in-place write refuses:

            RuntimeError: unsupported operation: more than one element of the
            written-to tensor refers to a single memory location.

        So `p.grad.zero_()`, `p.grad.add_(...)` and every `foreach` optimizer
        would fail on the second step of a loop whose loss ends in `.sum()`.
        Measured, not reasoned about (docs/training/BACKWARD9.md §2).

        `contiguous()` materialises when the layout is not dense and returns
        the same storage when it already is, so the `data_ptr` test is what
        makes this exactly one copy in both cases rather than two in one.
        """
        dense = gradient.contiguous()
        if dense.data_ptr() == gradient.data_ptr():
            dense = dense.clone()
        return dense

    def _accumulate_into_grad(leaf, gradient):
        """`p.grad += g`, with the one thing that cannot be skipped: the clone.

        Upstream's `AccumulateGrad` assigns on the first accumulation and adds
        in place after, and the in-place half is load-bearing for optimizers --
        `foreach` SGD and `zero_grad(set_to_none=False)` both assume `p.grad`
        keeps its identity across steps.

        **The first store is a clone, and it is not defensive.** The tape hands
        back one gradient tensor per constant, and for some programs two
        constants get *the same object*: `z = x + y; z.sum().backward()`
        returns the seed itself for both, `grads[0] is grads[1]`, measured
        (docs/training/BACKWARD9.md §2). Storing it directly would make `x.grad` and
        `y.grad` one tensor, and the very next step's `x.grad.add_(...)` would
        double `y.grad` silently -- a wrong number, not an error.
        `test_two_leaves_of_one_add_do_not_share_one_gradient_tensor` is that
        difference, and removing the clone turns it red.

        The clone is also what makes `.grad` the caller's to mutate:
        `zero_grad(set_to_none=False)` calls `p.grad.zero_()`, and a `.grad`
        aliasing a value the tape had returned would zero that too.

        And it is `_dense_copy_of` rather than `clone()` for a second reason
        found the same way -- see that function.
        """
        existing = leaf.grad
        if existing is None:
            leaf.grad = _no_grad_call(_dense_copy_of, gradient)
        else:
            _no_grad_call(lambda: existing.add_(gradient))

    def run_backward(
        self,
        tensors,
        grad_tensors=(),
        keep_graph=False,
        create_graph=False,
        inputs=(),
        allow_unreachable=False,
        accumulate_grad=False,
    ):
        if create_graph:
            raise NotImplementedError(
                "not implemented in torch._C shim: create_graph=True -- the eager "
                "backward runs under a NoGradGuard, so the ops it performs are not "
                "themselves recorded and a second backward through them would find "
                "an empty graph rather than fail. Double backward, "
                "torch.autograd.grad(..., create_graph=True) and any "
                "gradient-penalty term need it; first-order training does not. "
                "docs/training/BACKWARD9.md §6"
            )
        tensors = tuple(tensors)
        if len(tensors) != 1:
            raise NotImplementedError(
                "not implemented in torch._C shim: "
                f"_ImperativeEngine.run_backward over {len(tensors)} root tensors "
                "-- the eager tape has one output, and differentiating several "
                "roots at once means seeding each and summing into one traversal. "
                "Call backward() once per root, or sum them into a scalar first. "
                "docs/training/BACKWARD9.md §6"
            )
        root = tensors[0]
        if not isinstance(root, module.TensorBase):
            raise NotImplementedError(
                "not implemented in torch._C shim: backward from a GradientEdge -- "
                "the eager tape is addressed by tensor identity and has no name for "
                "an edge. docs/training/BACKWARD9.md §6"
            )
        seed = grad_tensors[0] if grad_tensors else None
        # `wrt=None`: ask the tape for every leaf that requires grad and select
        # here. Passing `inputs` through would hand the selection to
        # `_eager_backward`, which *raises* for a tensor no recorded op read --
        # and that is precisely the case `allow_unused=True` exists to answer
        # with `None`. Selecting in Python is what makes both defaults reachable.
        answer = module._eager_backward(root, seed, None, bool(keep_graph))
        leaves = answer["tensors"]
        grads = answer["grads"]
        found = {}
        for leaf, gradient in zip(leaves, grads):
            if gradient is not None:
                found[id(leaf)] = gradient

        inputs = tuple(inputs or ())
        if accumulate_grad:
            # `Tensor.backward()`. Upstream accumulates into every leaf that
            # requires grad, or into `inputs` alone when it is given.
            targets = inputs if inputs else [t for t in leaves if id(t) in found]
            for leaf in targets:
                gradient = found.get(id(leaf))
                if gradient is None:
                    if not allow_unreachable:
                        raise RuntimeError(
                            "One of the differentiated Tensors appears to not have "
                            "been used in the graph. Set allow_unused=True if this "
                            "is the desired behavior."
                        )
                    continue
                _accumulate_into_grad(leaf, gradient)
            return ()

        # `torch.autograd.grad()`. Nothing is written to `.grad`; the answer is
        # returned in `inputs` order, and `allow_unreachable` is `allow_unused`.
        out = []
        for leaf in inputs:
            gradient = found.get(id(leaf))
            if gradient is None and not allow_unreachable:
                raise RuntimeError(
                    "One of the differentiated Tensors appears to not have been "
                    "used in the graph. Set allow_unused=True if this is the "
                    "desired behavior."
                )
            out.append(gradient)
        return tuple(out)

    run_backward.__name__ = "run_backward"
    run_backward.__qualname__ = "_ImperativeEngine.run_backward"
    engine.run_backward = run_backward


# numpy's dtype names, mapped to this shim's. Keyed on `arr.dtype.name`, which
# is a plain `str`, so nothing here imports numpy -- the same constraint the
# `_numbers` import at the top of this file is written around, and for the same
# reason: `_C` is imported before numpy would be available, and the shim does
# not depend on it.
#
# The names left out are left out deliberately. `complex64`/`complex128`,
# `float128`, `uint16`/`uint32`/`uint64` and the datetime kinds have no candle
# storage behind them; they refuse by *name* in `_array_like_data` below rather
# than being mapped to something close, because a silent widening of `uint32`
# to `int64` is the shape of wrongness docs/verification/GOLDEN.md exists to stop.
_NUMPY_DTYPE_TO_TORCH = {
    "bool": "bool",
    "uint8": "uint8",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
}


def _array_like_data(module, data):
    """`(nested_list, dtype)` for a numpy array, or `None` for anything else.

    Every construction path in this file bottoms out in
    `module._tensor_new_from_data`, which walks *nested sequences of Python
    bool/int/float* and refuses anything else by name. An `ndarray` is not one:
    measured on this shim before this function existed, `torch.tensor(ndarray)`,
    `torch.as_tensor(ndarray)`, `Tensor.new_tensor(ndarray)` **and**
    `torch.FloatTensor(ndarray)` all raised

        TypeError: torch._C shim: torch.tensor() takes nested sequences of
        Python bool/int/float, and got ndarray

    -- including `as_tensor`, whose own docstring claimed the ndarray path
    worked and merely copied. It did not work at all. That claim is corrected
    where it stands.

    `.tolist()` is the bridge, and it is a **copy**, which is the recorded
    divergence: upstream aliases the array's buffer when no cast is needed
    (docs/architectures/DEMAND1.md §4, docs/bindings/CTOR.md §3.3), and this shim's tensors do not wrap
    foreign buffers, so there is no aliasing answer to give.

    Returning the numpy dtype alongside the values is what keeps
    `torch.tensor(ndarray)` honest: upstream **preserves** the array's dtype
    there (`torch.tensor(float64 array)` is `float64`), while nested-list
    inference would answer the default float. The callers that want the default
    instead -- the legacy `torch.Tensor(...)` constructor and its ten typed
    siblings, which cast unconditionally -- pass their own `dtype` and ignore
    this one.

    **What is deliberately not array-like here**: a numpy *scalar*
    (`np.float32(3.5)`) has `__array__` and `tolist` too, and upstream's legacy
    constructor rejects it (`new(): data must be a sequence (got numpy.float32)`)
    while `torch.tensor` accepts it. It is filtered out by `_numbers.Number` --
    numpy scalars register with those ABCs and `ndarray` does not (measured) --
    so it falls through to the *scalar* handling each caller already had, which
    is the behaviour both of those callers want. A `TensorBase` is filtered for
    the same reason in reverse: it has `__array__`, and every caller checks for
    it first anyway.
    """
    if isinstance(data, (module.TensorBase, str, bytes, bytearray)):
        return None
    if isinstance(data, _numbers.Number):
        return None
    if not (hasattr(data, "__array__") and hasattr(data, "tolist")
            and hasattr(data, "shape")):
        return None
    name = getattr(getattr(data, "dtype", None), "name", None)
    mapped = _NUMPY_DTYPE_TO_TORCH.get(name)
    if mapped is None:
        raise NotImplementedError(
            f"not implemented in torch._C shim: building a tensor from an array "
            f"of dtype {name!r} -- this shim reads an array through .tolist(), "
            f"and only {sorted(_NUMPY_DTYPE_TO_TORCH)} have a storage behind "
            f"them here. See bootstrap.py::_array_like_data"
        )
    return data.tolist(), getattr(module, mapped)


def _new_from_data(module, data, dtype, device):
    """`module._tensor_new_from_data`, with the array-like and iterator cases.

    The one door every `lift_fresh` construction goes through, so that
    `torch.tensor`, `torch.as_tensor`, `Tensor.new_tensor` and the legacy
    `torch.Tensor(...)` constructor agree about what "data" means instead of
    each growing its own answer.

    Two conversions, both of which the Rust walker cannot do and neither of
    which changes an already-working call:

    - **array-likes**, per `_array_like_data` above.
    - **arbitrary sequences**, because upstream's `torch.Tensor(range(3))` is a
      `(3,)` tensor of `0., 1., 2.` (measured) and the walker takes only `list`
      and `tuple`. `str`/`bytes` are excluded: they are sequences, and upstream
      answers `TypeError: new(): invalid data type 'str'`, so letting them
      through would turn a refusal into three tensors of character codes.
    """
    arrayed = _array_like_data(module, data)
    if arrayed is not None:
        values, array_dtype = arrayed
        return module._tensor_new_from_data(
            values, array_dtype if dtype is None else dtype, device
        )
    if (not isinstance(data, (list, tuple, str, bytes, bytearray, module.TensorBase))
            and not isinstance(data, _numbers.Number)
            and hasattr(data, "__iter__")):
        data = list(data)
    return module._tensor_new_from_data(data, dtype, device)


def _tensor_factory(module, dispatch):
    """`torch.tensor(...)`, which has no overload set to resolve against.

    Every other name in `overloads.json` maps to aten overloads. This one does
    not: upstream's `torch.tensor` is `THPVariable_tensor` ->
    `internal_new_from_data`, a `_C` function, and a real `torch.tensor([1, 2])`
    produces exactly one aten record -- `aten.lift_fresh.default` (measured).
    `aten::tensor` does exist, but it is a TorchScript builtin that this path
    never reaches, so routing here would have named a key upstream never names.

    So the split is upstream's own: `_C` builds the data, and the single aten
    call upstream really makes is the single aten call made here. The door is
    still one door -- `lift_fresh` goes through `_aten_dispatch` like anything
    else.
    """

    def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
        # `torch.tensor` is one of the 36 names `DeviceContext` treats as a
        # device constructor, so it has to consult the mode stack like every
        # `_torch_level_function` does. It is spelled out here rather than
        # shared because this function is not built by that factory -- see the
        # docstring.
        if _MODE_STACK:
            kwargs = {"dtype": dtype, "device": device,
                      "requires_grad": requires_grad, "pin_memory": pin_memory}
            return _through_torch_function_modes(tensor, (data,), kwargs)
        # `requires_grad` is carried, not refused -- `_strip_python_only_kwargs`
        # carries the argument. `pin_memory` is a different case and still
        # refuses: it names an allocation this shim cannot make.
        if pin_memory:
            raise NotImplementedError(
                "not implemented in torch._C shim: torch.tensor(pin_memory=True)"
            )
        if isinstance(device, str):
            device = module.device(device)
        out = dispatch(
            "aten.lift_fresh.default",
            _new_from_data(module, data, dtype, device),
        )
        return _apply_requires_grad(out) if requires_grad else out

    tensor.__name__ = "tensor"
    tensor.__qualname__ = "tensor"
    return tensor


# The legacy per-dtype tensor classes -- `torch.IntTensor` and its nine
# siblings. `vits` is what needs them (docs/kernels/KERNELS26.md §24):
# `modeling_vits.py:349` builds `torch.IntTensor([self.hidden_size])` and then
# subscripts it, and until now they were `_ShimMeta` placeholders, so the model
# stopped on `'IntTensor' object is not subscriptable` -- a TypeError from a
# type that was never a tensor class at all.
#
# **They must stay real types**, not factory functions, for the reason the
# placeholder loop above gives: these names turn up in *annotations*, which
# Python evaluates at import time. `transformers/modeling_flash_attention_
# utils.py:602` is `max_seqlen_q: int | torch.IntTensor | None = None`, and
# `int | <function>` is a TypeError -- measured, it stops `import transformers`
# dead. So each is a class whose `__new__` returns a `TensorBase`, which is
# also what upstream does: `type(torch.IntTensor([1]))` is `torch.Tensor`, not
# `torch.IntTensor`.
#
# All three of the legacy constructor's forms, and the ambiguity is the whole
# reason this is worth writing carefully (docs/kernels/KERNELS26.md §12.1 ground 3):
#
#     IntTensor(2, 3)      a SIZE -> a (2, 3) tensor of int32
#     IntTensor([2, 3])    DATA   -> a (2,) tensor holding 2 and 3
#     IntTensor(existing)  a re-wrap, cast to the class's dtype
#
# `[2, 3]` looks exactly like a size list and is not one. An implementation
# that took a sequence as a shape answers `(2, 3)` zeros where upstream answers
# two numbers, and nothing downstream raises -- so the data branch is checked
# first and by type, never by shape.
_LEGACY_TENSOR_DTYPES = {
    "ByteTensor": "uint8",
    "CharTensor": "int8",
    "ShortTensor": "int16",
    "IntTensor": "int32",
    "LongTensor": "int64",
    "BoolTensor": "bool",
    "HalfTensor": "float16",
    "BFloat16Tensor": "bfloat16",
    "FloatTensor": "float32",
    "DoubleTensor": "float64",
}


def _sized_tensor(module, base, sizes, device, empty_is_scalar=False):
    """The legacy constructors' *size* form -- `Tensor(2, 3)` and its relatives.

    `TensorBase(*sizes)` is the native one (`tensor.rs::py_new`) and stays the
    path for the ordinary case, because it is the path that is already measured
    and golden-compared, and routing it through a kernel would put aten records
    into a trace that upstream's own trace does not have.

    Two cases it cannot answer, both introduced by callers that reach this
    function and did not reach `py_new` directly:

    - **the empty size**, from `torch.Tensor(torch.Size([]))`, which upstream
      answers with a **scalar** `()` tensor. `TensorBase()` is `(0,)` -- the
      one-dimensional empty tensor, measured and deliberate -- so the two
      genuinely differ, and which one is right depends on *how the caller
      spelled it*: `torch.Tensor()` is `(0,)` and `torch.Tensor(torch.Size([]))`
      is `()`, both measured upstream. That is what `empty_is_scalar`
      distinguishes, and it is the only reason this function needs a flag. The
      first version of it did not have one and silently turned `torch.Tensor()`
      into a scalar -- caught by diffing the whole shape table against upstream
      rather than by any single case.
    - **a non-CPU device**, from `torch.Tensor(5, device=...)`. `py_new` takes
      no keywords at all (its signature is `(*args)`), so this is the only
      place the keyword upstream accepts can be honoured.

    Both fall through to `empty`, which is a real kernel with a device argument
    and the same "no correct value to diff" contract the size form has.
    """
    sizes = tuple(sizes)
    on_cpu = device is None or getattr(device, "type", "cpu") == "cpu"
    if on_cpu and (sizes or not empty_is_scalar):
        return base(*sizes)
    return module._VariableFunctions.empty(list(sizes), device=device)


def _install_legacy_tensor_types(module, dispatch) -> None:
    base = module.TensorBase

    def make(name, dtype_name):
        dtype = getattr(module, dtype_name)

        def __new__(cls, *args, **kwargs):
            # The re-wrap form first, as `TensorBase.__new__` does, so the
            # single-tensor path stays one check.
            if len(args) == 1 and isinstance(args[0], base):
                return dispatch("aten._to_copy.default", args[0], dtype=dtype)
            # `torch.Size` is a SIZE, and it has to be checked before the
            # sequence branch below because it *is* a `tuple` subclass -- so
            # the `isinstance(..., (list, tuple))` test catches it and answers
            # data. Measured upstream: `FloatTensor(torch.Size([2, 3]))` is a
            # `(2, 3)` float32 tensor and `LongTensor(torch.Size([2, 3]))` a
            # `(2, 3)` int64 one, i.e. the same size rule `torch.Tensor` has
            # (docs/bindings/CTOR.md §2.1). Before this check they answered a `(2,)`
            # tensor holding 2 and 3 -- the exact confusion the comment above
            # this function warns about, one type further out than it looked.
            if len(args) == 1 and isinstance(args[0], module.Size):
                return dispatch("aten._to_copy.default",
                                _sized_tensor(module, base, tuple(args[0]),
                                              kwargs.get("device"),
                                              empty_is_scalar=True),
                                dtype=dtype)
            # DATA, decided by type and never by shape: a sequence is values.
            # Through `_VariableFunctions.tensor`, which is where
            # `_tensor_factory` installs it -- `torch.tensor` is hoisted from
            # there by `torch/__init__.py`, and `_C` itself has no `tensor`.
            #
            # `_array_like_data` extends this to an `ndarray`, which is what
            # `pegasus` reaches: `PegasusSinusoidalPositionalEmbedding.
            # create_weight` is `torch.FloatTensor(numpy_array)`. Upstream
            # **coerces** it to the class's dtype rather than preserving the
            # array's (`LongTensor(array([1.7, 2.9]))` is `[1, 2]`, truncating
            # -- measured), which is what passing `dtype=dtype` here does.
            if len(args) == 1:
                arrayed = _array_like_data(module, args[0])
                if arrayed is not None:
                    return module._VariableFunctions.tensor(arrayed[0], dtype=dtype)
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                return module._VariableFunctions.tensor(args[0], dtype=dtype)
            # ...and everything else is the size form, which `TensorBase`
            # already implements (including upstream's negative-dimension
            # wording and the `()` -> `(0,)` rule). It builds at the default
            # float, so the cast is what makes it this class's dtype -- and
            # the values are zeros where upstream's are uninitialised, which
            # is docs/kernels/KERNELS26.md §12.2's recorded property of the size form
            # and not a new one.
            return dispatch("aten._to_copy.default",
                            _sized_tensor(module, base, args, kwargs.get("device")),
                            dtype=dtype)

        return _ShimMeta(name, (), {
            "__module__": "torch",
            "__new__": __new__,
            "__doc__": f"torch._C shim: the legacy {name} constructor "
                       f"({dtype_name}). Returns a TensorBase, as upstream "
                       f"returns a torch.Tensor.",
        })

    installed = []
    for name, dtype_name in _LEGACY_TENSOR_DTYPES.items():
        if not hasattr(module, dtype_name):
            continue
        setattr(module, name, make(name, dtype_name))
        installed.append(name)
    # Readable for the same reason `_shim_nn_implemented` is: which of these
    # ten do something should be answerable by asking rather than by trying.
    # All ten are installed. `CharTensor` then refuses from one layer down --
    # candle has no `int8` storage, so the refusal comes from
    # `_tensor_from_flat` naming the *dtype*, which is a better message than a
    # missing name would be. `ShortTensor` computes: `int16` is storable here,
    # checked rather than assumed from `int8` being absent.
    module._shim_legacy_tensor_types = sorted(installed)
    module._shim_tensor_new = _make_tensor_class_new(module, dispatch)


def _make_tensor_class_new(module, dispatch):
    """`torch.Tensor.__new__` -- docs/architectures/DEMAND.md §0.1 rank 1, and it is not
    structural.

    DEMAND.md called this gap structural on the ground that the refusal lives in
    PyO3's `#[new] fn py_new` and the only class that could carry a Python-level
    override, `torch.Tensor`, is in the vendored tree. Measured (docs/bindings/CTOR.md
    §1), `torch.Tensor` is `class Tensor(torch._C.TensorBase)` at
    `torch/_tensor.py:102`; its MRO is `(Tensor, TensorBase, object)`; it
    defines **neither** `__new__` nor `__init__`, inheriting the first from
    `TensorBase`; and it is a settable heap type. A class that inherits
    `__new__` and accepts `setattr` can be given one from outside the file that
    declares it. Being in the vendored tree makes a class unavailable to
    *editing*, not to *patching*, and patching classes it does not own is most
    of what this file does.

    The real obstacle was ordering, and the hook for it already existed:
    `_initExtension` installs this, at the moment its own comment already
    names -- `torch/__init__.py:1931` imports `Tensor` and `:2189` calls
    `_initExtension`, so the class exists and no tensor has been made yet.

    **It goes on `Tensor`, not on `TensorBase`, and that is not a preference.**
    Assigning a Python `__new__` replaces a type's `tp_new` with `slot_tp_new`,
    after which the native constructor is reachable only through the saved
    wrapper -- and CPython's `tp_new_wrapper` guard then refuses it:

        TypeError: torch._C.TensorBase.__new__(Tensor) is not safe,
                   use object.__new__()

    measured, by doing exactly that. The guard walks up from the subtype past
    every `slot_tp_new` and compares with the wrapper's own type's `tp_new`;
    once `TensorBase`'s is itself `slot_tp_new` nothing matches and the native
    allocator is permanently unreachable from Python -- taking
    `TensorBase(existing)` with it, which is the re-wrap form every internal
    caller uses and the one `_make_subclass` builds a `Parameter` out of.
    Patching only `Tensor` leaves `TensorBase.tp_new` native, so the walk
    succeeds (`Tensor` is a slot -> skip -> `TensorBase` matches) and
    `base.__new__(cls, ...)` keeps working. Verified: after the patch,
    `torch.Tensor(5)` is a `Tensor`, `nn.Parameter(...)` is a `Parameter`, and
    `nn.Linear(2, 2)` still has two parameters.

    The dispatch order below is upstream's parser, and each branch is a rule
    that a plausible implementation gets wrong -- see docs/bindings/CTOR.md §2:

        Tensor(existing)        re-wrap; keeps the SOURCE dtype, not the default
        Tensor(torch.Size(...)) a SIZE -- and `torch.Size` is a `tuple` subclass,
                                so this must precede the sequence branch
        Tensor(), Tensor(2, 3)  a SIZE, native
        Tensor([...]), (...),   DATA, always at the DEFAULT dtype, whatever the
        Tensor(ndarray),        source dtype is -- which is the whole difference
        Tensor(range(...))      between this and `torch.tensor`, which infers
    """
    base = module.TensorBase

    def __new__(cls, *args, **kwargs):
        device = kwargs.pop("device", None)
        if kwargs:
            # Upstream's own answer, in shape if not word for word: it reports
            # the combination it got. `dtype=` and `requires_grad=` are the two
            # that get tried, and upstream rejects **both** -- the legacy
            # constructor takes `device` and nothing else (measured:
            # `Tensor(5, dtype=torch.float64)` is a TypeError there too). So
            # this is not a shim limitation and is not spelled as one.
            got = ", ".join(
                [_upstream_type_name(a) for a in args]
                + [f"{k}={_upstream_type_name(v)}" for k, v in kwargs.items()])
            raise TypeError(
                f"new() received an invalid combination of arguments - got "
                f"({got}), but expected one of:\n * (*, torch.device device)"
                f"\n * (torch.Storage storage)\n * (Tensor other)"
                f"\n * (tuple of ints size, *, torch.device device)"
                f"\n * (object data, *, torch.device device)"
            )
        if isinstance(device, str):
            device = module.device(device)

        if len(args) == 1:
            only = args[0]
            # Re-wrap. Native, and it keeps the source dtype -- `Tensor(int64
            # tensor)` is `int64` upstream, the one case where this constructor
            # does not land on the default dtype.
            if isinstance(only, base):
                return base.__new__(cls, only)
            # A SIZE, before the sequence branch, because `torch.Size` is a
            # `tuple` subclass and the branch below would read it as data --
            # answering a `(2,)` tensor holding 2 and 3 where upstream answers
            # a `(2, 3)` empty one, with nothing downstream raising.
            if isinstance(only, module.Size):
                return base.__new__(
                    cls, _sized_tensor(module, base, tuple(only), device,
                                       empty_is_scalar=True))
            if isinstance(only, bool) or isinstance(only, int):
                return base.__new__(cls, _sized_tensor(module, base, (only,), device))
            if isinstance(only, (str, bytes, bytearray)):
                raise TypeError(
                    f"new(): invalid data type '{type(only).__name__}'")
            if isinstance(only, _numbers.Number) or only is None:
                # `Tensor(1.5)`, `Tensor(None)`, `Tensor(np.float32(3.5))`.
                # Upstream's wording, transcribed. A float is not a size (it
                # cannot be one) and not a sequence, so it is neither form.
                raise TypeError(
                    f"new(): data must be a sequence (got "
                    f"{_upstream_type_name(only)})")
            # DATA. `dtype` is the default dtype and is passed explicitly:
            # nested-list inference would answer `int64` for `Tensor([1, 2])`
            # and the array's own dtype for `Tensor(int64 ndarray)`, and
            # upstream answers the default float for both.
            out = dispatch(
                "aten.lift_fresh.default",
                _new_from_data(module, only, module.get_default_dtype(), device),
            )
            return base.__new__(cls, out)

        # Zero arguments, or several. Several must all be sizes: upstream has
        # no multi-argument data form, so `Tensor([1, 2], [3, 4])` and
        # `Tensor(t, t)` are both refusals there and are refusals here.
        #
        # **Which** refusal depends on the first argument, and upstream really
        # does use two different messages -- measured:
        #
        #     Tensor([1,2], [3,4])  invalid combination of arguments - got (list, list)
        #     Tensor(t, t)          invalid combination of arguments - got (Tensor, Tensor)
        #     Tensor(2, 3.0)        argument 'size' failed to unpack ... at pos 2
        #
        # i.e. once the first argument reads as a size, the whole call is the
        # size overload and a later bad element is an unpack failure *within*
        # it; if the first argument does not, no overload matched at all. A
        # caller catching one of these by text gets the one upstream gives.
        if args and not isinstance(args[0], (int, bool)):
            got = ", ".join(_upstream_type_name(a) for a in args)
            raise TypeError(
                f"new() received an invalid combination of arguments - got "
                f"({got}), but expected one of:\n * (*, torch.device device)"
                f"\n * (torch.Storage storage)\n * (Tensor other)"
                f"\n * (tuple of ints size, *, torch.device device)"
                f"\n * (object data, *, torch.device device)"
            )
        for i, arg in enumerate(args):
            if isinstance(arg, (int, bool)):
                continue
            # Transcribed from upstream 2.13.0 including the missing space
            # after the comma -- it is upstream's, and a caller matching on
            # this text would be matching upstream's spelling of it.
            raise TypeError(
                f"new(): argument 'size' failed to unpack the object at "
                f"pos {i + 1} with error \"type must be tuple of ints,but "
                f"got {_upstream_type_name(arg)}\"")
        return base.__new__(cls, _sized_tensor(module, base, args, device))

    __new__.__name__ = "__new__"
    __new__.__qualname__ = "Tensor.__new__"
    # The same wall `_make_subclass` hit, and the same answer -- see its
    # comment for the whole argument. `torch/_dynamo/decorators.py:1006`
    # compares this function's signature against the polyfill
    # `torch/_dynamo/polyfills/tensor.py:42::tensor_pynew` and raises
    # `TypeError: Signature mismatch` unless one of the two is the wildcard
    # `(*args, **kwargs)`. Upstream's `Tensor.__new__` is a C builtin, so
    # `inspect.signature` raises `ValueError` on it and the comparison is
    # skipped entirely; a Python function has a readable signature, so the
    # comparison ran -- and `(cls, *args, **kwargs)` is not the wildcard,
    # because `cls` is an ordinary positional parameter.
    #
    # Measured: without this line, `PegasusSinusoidalPositionalEmbedding`
    # stops at `decorators.py:1011` before reaching any tensor at all, and
    # every later model in the same process then fails differently
    # (`Duplicate dispatch rule for ... _make_subclass`) because the first
    # failure left the polyfill registration half-done. Withholding the
    # signature is the same amount of information upstream gives.
    __new__.__signature__ = inspect.signature(lambda *args, **kwargs: None)
    __new__.__doc__ = (
        "torch._C shim: the legacy torch.Tensor(...) constructor. Accepts the "
        "re-wrap form Tensor(existing), the size forms Tensor(), Tensor(2, 3) "
        "and Tensor(torch.Size(...)), and the data forms Tensor(sequence) and "
        "Tensor(ndarray) -- the last two at the default dtype, as upstream "
        "does. See docs/bindings/CTOR.md."
    )
    return staticmethod(__new__)


def _upstream_type_name(value):
    """Upstream names a rejected argument's type the way `repr` of its class
    does, module-qualified for anything outside builtins: `float`, `NoneType`,
    `numpy.float32`. Transcribed rather than guessed -- the numpy-scalar
    spelling is the one that carries a module, and it is the one a caller
    catching this message would be matching on.

    A tensor is the exception and is bare `Tensor`, not `torch.Tensor`:
    upstream's `Tensor(t, t)` reports `got (Tensor, Tensor)` (measured), and
    the qualified spelling was what this function produced before the diff
    against upstream caught it."""
    cls = type(value)
    for candidate in ("TensorBase", "Tensor"):
        parent = getattr(sys.modules.get("torch._C"), candidate, None)
        if isinstance(parent, type) and isinstance(value, parent):
            return "Tensor"
    if cls.__module__ in ("builtins", None):
        return cls.__name__
    return f"{cls.__module__}.{cls.__name__}"


def _install_namespace_types(module, namespace) -> None:
    """`torch.strided`, `torch.contiguous_format`, `torch.per_tensor_affine`.

    `dtype` is not here: `_C` owns that type in Rust and registers every
    instance itself (see `dtype.rs`), because a dtype means something. These
    three are pure labels -- there is no candle concept behind `torch.strided`
    -- so they are built here, in the same spirit as `device`: a label, and the
    thing that uses it is what fails.
    """
    for kind in ("layout", "memory_format", "qscheme"):
        cls = getattr(module, kind, None)
        if cls is None or not isinstance(cls, type):
            cls = _ShimMeta(kind, (), {"__module__": "torch", "__init__": _permissive_init})
            setattr(module, kind, cls)
        for name, value_kind in namespace.items():
            if value_kind != kind or hasattr(module, name):
                continue
            instance = cls.__new__(cls)
            instance._shim_name = name
            setattr(module, name, instance)
            cls.__repr__ = lambda self: f"torch.{self._shim_name}"
            cls.__str__ = cls.__repr__


# Operator schemas that live in C++ upstream and in no `.pyi`.
#
# `_get_schema` answered every op with a `_Schema` carrying no arguments and no
# returns. For `aten::` that is only a missing answer -- `_aten_dispatch` is
# what runs an op, and it reads `overloads.json`, not this. For these three
# namespaces it is a *wrong* answer: `torch/distributed/_functional_collectives.py`
# registers autograd formulas at import time and
# `torch/library.py:1417` refuses unless `is_functional_schema(schema)`, which
# needs arguments and returns to have really been read. An empty schema has no
# returns, so it is never functional, so the import stops.
#
# These are registered from `torch/csrc/distributed/c10d/Functional.cpp` and
# friends -- i.e. from the half of torch this shim replaces -- so there is
# nowhere else they could come from. The text is transcribed from upstream
# 2.13.0's own registry (`torch._C._jit_get_all_schemas()` filtered by
# namespace, after importing `torch.distributed._functional_collectives`), not
# written by hand; `pytests/verify_schemas.py` re-derives it the same way.
#
# Whether an op *runs* is a separate question from whether its schema is known,
# and this table answers only the second. Calling one still goes through
# `_op_callable` to the one door in `aten.rs`, which has no kernel for the
# `_c10d_functional` namespace and refuses by name.
_NON_ATEN_SCHEMA_TEXT = (
    "_c10d_functional::_wrap_tensor_autograd(Tensor input) -> Tensor",
    "_c10d_functional::all_gather_into_tensor(Tensor input, int group_size, Any group_name) -> Tensor",
    "_c10d_functional::all_gather_into_tensor_coalesced(Tensor[] inputs, int group_size, Any group_name) -> Tensor[]",
    "_c10d_functional::all_gather_into_tensor_out(Tensor input, int group_size, Any group_name, *, Tensor(a!) out) -> Tensor(a!)",
    "_c10d_functional::all_reduce(Tensor input, str reduce_op, Any group_name) -> Tensor",
    "_c10d_functional::all_reduce_(Tensor(a!) input, str reduce_op, Any group_name) -> Tensor(a!)",
    "_c10d_functional::all_reduce_coalesced(Tensor[] inputs, str reduce_op, Any group_name) -> Tensor[]",
    "_c10d_functional::all_reduce_coalesced_(Tensor[](a!) inputs, str reduce_op, Any group_name) -> Tensor[](a!)",
    "_c10d_functional::all_to_all_single(Tensor input, SymInt[] output_split_sizes, SymInt[] input_split_sizes, Any group_name) -> Tensor",
    "_c10d_functional::batch_p2p_ops(str[] op_list, int[] peer_list, int[] tag_list, Tensor[] tensors, str group_name) -> Tensor[]",
    "_c10d_functional::broadcast(Tensor input, int src, Any group_name) -> Tensor",
    "_c10d_functional::broadcast_(Tensor(a!) input, int src, Any group_name) -> Tensor(a!)",
    "_c10d_functional::irecv(Tensor tensor, int src, int tag, str group_name) -> Tensor",
    "_c10d_functional::isend(Tensor tensor, int dst, int tag, str group_name) -> Tensor",
    "_c10d_functional::reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, Any group_name) -> Tensor",
    "_c10d_functional::reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduce_op, int group_size, Any group_name) -> Tensor[]",
    "_c10d_functional::reduce_scatter_tensor_out(Tensor input, str reduce_op, int group_size, Any group_name, *, Tensor(a!) out) -> Tensor(a!)",
    "_c10d_functional::wait_tensor(Tensor tensor) -> Tensor",
    "_c10d_functional_autograd::all_gather_into_tensor(Tensor input, int group_size, Any group_name) -> Tensor",
    "_c10d_functional_autograd::all_to_all_single(Tensor input, SymInt[] output_split_sizes, SymInt[] input_split_sizes, Any group_name) -> Tensor",
    "_c10d_functional_autograd::reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, Any group_name) -> Tensor",
    "_dtensor::shard_dim_alltoall(Tensor input, int gather_dim, int shard_dim, Any group_name) -> Tensor",
)


# aten schemas that are *generated* rather than declared, and so are in no data
# file this tree carries.
#
# `native_functions.yaml` has 2584 entries and upstream's registry has 3754 aten
# schemas. The difference is `torchgen/native_function_generation.py`, which
# synthesises `.out`, functional and mutable variants at build time from the
# declared ones. That code is vendored, but running it means parsing the YAML
# with `pyyaml`, which is not a dependency of this distribution (the same wall
# `decompose._scan_core_tags` is written around), so the generated half cannot
# be re-derived here.
#
# This is not the whole generated half -- it is the part the tree asks a
# *question* about, measured rather than guessed. With placeholders instrumented,
# a full run (import, the transformers road, FSDP, the decomposition pass) reads
# `is_mutable` or `_is_view_op()` on 102 ops with no text; 84 of those do not
# exist upstream either, and these 18 do. Transcribing them is what makes
# "no operator upstream has is answered from an empty schema" true, and
# `verify_schemas.py` re-derives them from a real torch the same way it does
# `_NON_ATEN_SCHEMA_TEXT`.
#
# One of the 18 is the reason the list is not optional:
# `native_dropout_backward.out` is mutable upstream and a placeholder answers
# False. The other 17 agree with the placeholder's answer today, and are here so
# that "agrees today" is not what the property rests on.
_GENERATED_ATEN_SCHEMA_TEXT = (
    "aten::_batch_norm_with_update_functional(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor running_mean_out, Tensor running_var_out)",
    "aten::_fused_adam(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] exp_avgs_out, Tensor[] exp_avg_sqs_out, Tensor[] max_exp_avg_sqs_out)",
    "aten::_fused_adam.tensor_lr(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] exp_avgs_out, Tensor[] exp_avg_sqs_out, Tensor[] max_exp_avg_sqs_out)",
    "aten::_fused_adamw(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] exp_avgs_out, Tensor[] exp_avg_sqs_out, Tensor[] max_exp_avg_sqs_out)",
    "aten::_fused_adamw.tensor_lr(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] exp_avgs_out, Tensor[] exp_avg_sqs_out, Tensor[] max_exp_avg_sqs_out)",
    "aten::_index_put_impl(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor",
    "aten::_native_batch_norm_legit_functional(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor running_mean_out, Tensor running_var_out)",
    # The four TorchScript numeric builtins the tree probes as `.default`.
    # `aten::add` really is `(Scalar, Scalar) -> Scalar` upstream -- the tensor
    # overloads are `add.Tensor`/`add.Scalar` -- which is why
    # `overloads.json`'s README refuses to put these in the *resolution* table.
    # Knowing an op's schema is not making `torch.add` reach it.
    "aten::add(Scalar a, Scalar b) -> Scalar",
    "aten::copysign(Scalar a, Scalar b) -> float",
    "aten::div(Scalar a, Scalar b) -> float",
    "aten::mul(Scalar a, Scalar b) -> Scalar",
    "aten::sub(Scalar a, Scalar b) -> Scalar",
    "aten::exponential(Tensor self, float lambd=1., *, Generator? generator=None) -> Tensor",
    "aten::geometric(Tensor self, float p, *, Generator? generator=None) -> Tensor",
    "aten::log_normal(Tensor self, float mean=1., float std=2., *, Generator? generator=None) -> Tensor",
    "aten::native_dropout_backward.out(Tensor grad_output, Tensor mask, float scale, *, Tensor(a!) out) -> Tensor(a!)",
    "aten::rrelu_with_noise_functional(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> (Tensor, Tensor noise_out)",
    "aten::uniform(Tensor self, float from=0., float to=1., *, Generator? generator=None) -> Tensor",
)


def _build_transcribed_schemas():
    out = {}
    for text in _NON_ATEN_SCHEMA_TEXT + _GENERATED_ATEN_SCHEMA_TEXT:
        parsed = _Schema.parse(text)
        out[(parsed.name, parsed.overload_name)] = parsed
    return out


_TRANSCRIBED_SCHEMAS = _build_transcribed_schemas()


def _constant_function(qualname: str, value):
    def fn(*args, **kwargs):
        return value

    fn.__name__ = qualname.rsplit(".", 1)[-1]
    fn.__qualname__ = qualname
    return fn


# `_C` functions whose *return value* the import path consumes, with the site
# that consumes it. A `NotImplementedError` here is not a work item -- it is a
# stop, because the tree has no fallback. Each entry states what upstream
# returns and what the shim returns instead.
_DISCOVERED_RETURNS = {
    # `torch/_tensor.py:105` -- `__dlpack_c_exchange_api__: object =
    # torch._C._dlpack_exchange_api()`, in the `class Tensor` body. Upstream
    # hands back a PyCapsule of C function pointers for zero-copy tensor
    # exchange. The shim has no DLPack bridge, and `None` is what the attribute
    # would hold on a build without one.
    "_dlpack_exchange_api": None,
    # `torch/_library/fake_class_registry.py:420`, from the `@register_fake_class`
    # decorator that `torch/_library/opaque_object.py:72` applies at import.
    # Upstream returns the registered `torch::class_` wrapper and the caller
    # discards it -- the call is there to assert the class exists. There are no
    # custom classes in this shim, so nothing to assert and nothing to return.
    "_get_custom_class_python_wrapper": None,
    # `torch/_library/opaque_object.py:268`. Upstream records the name in the
    # C++ type registry so a custom op schema can mention it; the return value is
    # unused. The shim has no schema registry, so this is a no-op -- recorded
    # rather than hidden, because if opaque types ever have to round-trip through
    # a schema this is where that starts.
    "_register_opaque_type": None,
    # `torch/jit/_trace.py:616`, at module scope. Upstream flips a global in the
    # TorchScript tracer so that tracing a Python-only op warns instead of
    # silently producing a constant. There is no tracer here, so there is nothing
    # to flip; if TorchScript tracing is ever wanted this is one of its entry
    # points.
    "_tracer_warn_use_python": None,
    # `torch/library.py:1767`, reached from `torch/_native/registry.py:894` while
    # that module registers its overrides at import. Upstream returns an opaque
    # handle to the kernel already registered at (op, dispatch_key), which the
    # caller keeps so it can re-enter the dispatcher without recursing through its
    # own override. There is no dispatcher here and therefore no prior kernel;
    # `None` says so. The override machinery still records itself, and still has
    # nothing to override.
    "_dispatch_get_computed_kernel_for_dispatch_key": None,
    # `torch/xpu/__init__.py:278`, reached from `torch.xpu.is_available()`,
    # reached from `transformers/masking_utils.py:39` -- at *import* of
    # `transformers.generation.utils`, which is the lazy import
    # docs/models/IMPORT_TORCH.md §11 item 3 recorded `from_config` as dying in.
    #
    # This one is a real answer, not a stand-in: upstream returns the number of
    # XPU devices, and a build with no XPU support has none. The name has to
    # exist -- `torch.xpu` is an ordinary Python package that is always
    # importable, so the absence trick (VENDOR.md wall 11) does not apply here;
    # `hasattr` never gets a chance to be the question.
    "_xpu_getDeviceCount": 0,
    # `torch/mtia/__init__.py:152`, reached from
    # `torch/_dynamo/device_interface.py:297` at module scope, reached from
    # `torch/_inductor/utils.py:115`, reached from the same transformers lazy
    # import as the entry above. Upstream returns whether the build has MTIA
    # support compiled in; this one does not. Same shape of answer, same
    # reason the name cannot simply be absent.
    "_mtia_isBuilt": False,
    # `torch/mtia/__init__.py:81`, reached from `torch/random.py:82` -- that
    # is, from `torch.manual_seed`, which walks every accelerator's seeding
    # entry point before it reaches the CPU generator it was actually asked
    # about. Upstream answers "was this process forked after MTIA was
    # initialised?"; MTIA is never initialised here, so `False` is true and
    # `manual_seed_all` then queues its callback instead of running it.
    "_mtia_isInBadFork": False,
    # `torch/random.py:127`, the last stop in `torch.manual_seed` before the
    # CPU generator. Upstream's default is the literal string
    # `"privateuseone"`, and the caller then does `hasattr(torch, name)` --
    # nothing is registered under that name here, so the branch is skipped
    # exactly as it is upstream on a stock build. Returning the real default
    # rather than `None` matters: `hasattr(torch, None)` is a TypeError.
    "_get_privateuse1_backend_name": "privateuseone",
    # `torch/nn/modules/module.py:530` -- the first line of
    # `nn.Module.__init__`, so *every* module ever constructed calls it, which
    # makes it the very next wall after `import torch` on the road to
    # `from_config`. Upstream increments an internal usage counter and returns
    # nothing; there is no counter here and nothing reads the result. This is
    # the cleanest member of docs/design/C_SURFACE.md §7's second tier -- a name whose
    # existence is the whole requirement.
    "_log_api_usage_once": None,
    # `torch/nn/modules/module.py`, the same shape: upstream records that a
    # module of this class was instantiated. Returns nothing, read by nobody.
    "_log_api_usage_metadata": None,
    # The `__torch_function__` fast-path predicates. `torch/nn/init.py:597`
    # is the first of hundreds of sites: *every* function in `torch/nn/init.py`,
    # `torch/nn/functional.py` and `torch/_tensor.py` opens with
    # `if torch.overrides.has_torch_function_variadic(...)`, and upstream's C
    # implementation answers "does any argument's type override
    # `__torch_function__`?".
    #
    # `False` is the true answer here, not a stand-in: the protocol is a real
    # dispatch mechanism this shim does not implement, and there is no type in
    # the vendored tree's inference path that overrides it -- `Parameter` and
    # `Tensor` inherit the default. If a subclass ever does override it, this
    # is the single place that has to learn to say so, and until then a `True`
    # would send every one of those call sites into a handler that cannot run.
    # `torch/_tensor.py:1186`, in `Tensor.__len__`, and a dozen more places
    # that warn only while TorchScript tracing. Upstream returns the active
    # `TracingState` or `None`; there is no tracer here, so `None` is the true
    # answer -- and it has to be *callable and answering*, not merely present,
    # because `len(tensor)` is on the path. (docs/design/C_SURFACE.md §1-3 noticed
    # this name being looked up during `import torch`; it is called later.)
    "_get_tracing_state": None,
    "_has_torch_function": False,
    "_has_torch_function_unary": False,
    "_has_torch_function_variadic": False,
    # The *dispatch*-mode stack, which `torch/utils/_python_dispatch.py`
    # consults. **Its `_len_torch_dispatch_stack: 0` row used to be here and is
    # gone** -- docs/graph/EXPORT.md §8 item 3. The comment that stood here said
    # "nothing pushes onto it here, so it is empty and disabled", which was true
    # when it was written and stopped being true the moment `torch.export`
    # arrived: `FakeTensorMode` and `ProxyTorchDispatchMode` both push.
    # `_install_mode_stack` below installs the real stack, and leaving the row
    # would have let a constant `0` win over it depending on install order --
    # making `with FakeTensorMode():` a block that entered, reported itself
    # absent, and changed nothing (docs/graph/EXPORT.md §2.2).
    #
    # Its torch-*function* sibling used to be beside it as another pair of
    # constants (`_len_torch_function_stack: 0`,
    # `_is_torch_function_mode_enabled: False`). It is a real stack now --
    # `_install_torch_function_modes` below -- because `with torch.device(...)`
    # is a torch-function mode and nothing else, and a stack that always
    # answered zero would have made `with torch.device("meta"):` a block that
    # succeeded and changed nothing. docs/devices/META.md §8.
    "_is_torch_function_all_disabled": False,
}

# Build-configuration flags: plain `bool` *values* upstream, not callables, so
# they cannot go in `_DISCOVERED_RETURNS` above (which wraps everything in a
# function).
#
# These were being answered by `_Unimplemented`, which is **truthy**. That is
# the one shape of placeholder that is worse than absence for a name spelled
# `_has_<backend>`: every `if torch._C._has_mps:` in the tree took the branch
# that assumes a backend exists, and did so silently. It cost a wall to
# notice -- `torch.manual_seed(0)` reaches `torch/mps/__init__.py:67`, whose
# whole guard is `if not torch._C._has_mps: return`, and went straight past it
# into `_mps_get_default_generator`.
#
# `False` is not a stand-in here, it is the fact: this shim's tensor engine is
# candle on the CPU (DESIGN.md §4) and there is no CUDA, MPS, XPU or MKLDNN
# behind any of these names. `torch.cuda.is_available()` already answered
# `False` by a different route; this makes the rest of the family agree with
# it instead of each guarding site getting a different answer depending on
# which spelling it happened to use.
#
# The table is no longer a hand list of the ones that happened to cost a wall.
# `install` requires an entry for **every** module-level name the stubs
# annotate `_bool` (`surface.json` kind `"bool"`), because a build flag has
# exactly two answers and both change behaviour -- there is no third,
# placeholder-shaped answer to fall back on. Adding a `_bool` to the stubs and
# not deciding here is now an import-time error naming the flag, rather than a
# truthy object waiting in a branch.
#
# Every value below is `False`, and that is a fact about this build rather than
# a default: candle on the CPU links none of these libraries. Two entries are
# worth their own note --
#
#   `_has_kleidiai`  Upstream on the host that measured this answers *True*
#                    (KleidiAI is ARM's kernel library and an arm64 mac build
#                    picks it up). That is a fact about upstream's build, not
#                    about the API, and copying it would be the same error as
#                    the truthy placeholder in a politer form.
#
#   `_GLIBCXX_USE_CXX11_ABI`
#                    The only one where the question does not really apply:
#                    there is no libstdc++ under this shim at all, so neither
#                    answer describes it. `False` is chosen because it is the
#                    one that cannot make a caller assume a GNU C++ ABI is
#                    present. `torch/__init__.py:2354`
#                    (`compiled_with_cxx11_abi`) is the only reader and nothing
#                    on the `import torch` path calls it. Before this it was
#                    not even a value -- the install loop reads a leading
#                    capital as a type name, so `_GLIBCXX_USE_CXX11_ABI` was
#                    being answered with a *class*.
_BUILD_FLAGS = {
    # Reached through `torch.backends.<name>.is_available()`, one per line.
    "_has_mps": False,
    "_has_cuda": False,
    "_has_xpu": False,
    "_has_mkldnn": False,
    # `torch/backends/cudnn/__init__.py:126`, and :231 in a class body -- the
    # one flag measured to change a branch during `import torch`.
    "_has_cudnn": False,
    # `torch/backends/cusparselt/__init__.py:51`.
    "_has_cusparselt": False,
    # `torch/backends/kleidiai/__init__.py:7`. See the note above.
    "_has_kleidiai": False,
    # `torch/cuda/__init__.py:155` -- `has_magma: bool = torch._C._has_magma`,
    # a module-level assignment, so the placeholder was being published as
    # `torch.cuda.has_magma` for anyone who read it.
    "_has_magma": False,
    # `torch/backends/mkldnn/__init__.py:25`. ACL is a sub-capability of
    # MKLDNN, so `True` here would contradict `_has_mkldnn` above.
    "_has_mkldnn_acl": False,
    # `torch/backends/mkl/__init__.py:7`, and `_meta_registrations.py:2864`,
    # which registers a meta kernel for `torch.ops.mkl._mkl_linear` when true.
    # This build has no such op, so the registration was for nothing.
    "has_mkl": False,
    # No LAPACK is linked; `torch.linalg` has no kernels here.
    "has_lapack": False,
    # candle parallelises with rayon, not OpenMP.
    "has_openmp": False,
    # FFT support. No `aten.fft*` kernel exists in this build.
    "has_spectral": False,
    # See the note above: the question does not apply, and this is the answer
    # that claims the least.
    "_GLIBCXX_USE_CXX11_ABI": False,
}

# The same, for members of synthesised `_C` types: `"<Type>.<member>": value`.
# Installed after the types are built.
_DISCOVERED_TYPE_RETURNS = {
    # `torch/_sources.py:122` -- `get_source_lines_and_file(fn,
    # ErrorReport.call_stack())`, reached from `parse_def`, reached from
    # `_check_overload_body`, which `@torch.jit._overload` runs *at decoration
    # time* while `torch/nn/functional.py` is importing (VENDOR.md wall 13).
    # Upstream returns TorchScript's compilation call stack as a string, used
    # only to decorate an error message. There is no such stack here, and the
    # empty string is what upstream returns when the stack is empty.
    "ErrorReport.call_stack": "",
}


def _install_fx_node_base(module) -> None:
    """`torch._C._NodeBase`, `_NodeIter`, `_fx_map_arg`, `_fx_map_aggregate`.

    docs/graph/EXPORT.md §6 **item 2**, and only item 2. Item 1 -- the dispatcher
    entrance -- landed in `aten.rs` first (docs/design/DISPATCH3.md), which is the
    precondition §6 states in the negative: with modes not consulted, filling
    this in would let `export` run, install a proxy mode, see nothing fire and
    return an `ExportedProgram` with no operators in it. Modes fire now, so
    this is the next item and not a shortcut past one.

    What was here before: a synthesised type carrying `_erased`, `_next` and
    `_prev` and four raising stubs, so `torch.fx.Graph()` died on its own
    sentinel root node (`graph.py:1369`, `Node(self, "", "root", "", (), {})`)
    -- four lines, no export involved (docs/graph/EXPORT.md §4.1).

    Every rule below was measured against real torch 2.13.0 in a separate
    process rather than read off `torch/csrc/fx/node.cpp`, which this tree
    does not carry; docs/bindings/BIND5.md §2 has the transcripts.

    **The sort key is the part that a plausible implementation gets wrong.**
    Nodes carry a `_sort_key` tuple that must order them the same way the
    linked list does, including after an insertion *between* two neighbours,
    and it is maintained by `_prepend` rather than by the list walk. Inserting
    `x` between `p` and `n`:

        len(p) > len(n)   ->  p[:-1] + (p[-1] + 1,)
        len(p) < len(n)   ->  n[:-1] + (n[-1] - 1,)
        equal             ->  p + (0,)

    Measured, on a graph of three placeholders `a`/`b`/`c` at `(0,)`/`(1,)`/
    `(2,)`: inserting before `b` gives `(0, 0)`, then before *that* gives
    `(0, -1)`, appending at the end gives `(3,)` and inserting before `a`
    gives `(-1,)`. A monotonically increasing counter -- the obvious wrong
    answer -- produces the right list order and the wrong `_sort_key` order
    the moment anything is inserted in the middle, and nothing in `fx` would
    say so until a pass sorted nodes.

    **`_update_args_kwargs` converts, and the conversion is asymmetric.**
    Measured: a top-level `args` tuple stays a plain `tuple`, a nested list
    becomes `immutable_list`, and the top-level `kwargs` dict becomes
    `immutable_dict`. That falls out of `map_aggregate` mapping a tuple to a
    tuple and a list/dict to its immutable twin, applied to both. The import
    is lazy because `torch.fx` does not exist yet when this file runs.

    **`_NodeIter` skips erased nodes.** `graph.py:1619` sets `_erased = True`
    *after* `_remove_from_list()` and says why in a comment ("iterators may
    retain handles to erased nodes"); measured directly by setting `_erased`
    on a still-linked node, which then vanishes from `g.nodes` without being
    unlinked.

    **`_remove_from_list` does NOT self-link.** After it, the removed node's
    `_prev`/`_next` still point at its former neighbours (measured). A version
    that reset them to `self` would look tidier and would break the retained-
    handle case above.
    """
    node_base = module._NodeBase
    # The synthesised type carried `_erased`/`_next`/`_prev` as CLASS-LEVEL
    # raising getters. They have to go before anything can hold state: those
    # three names are read on the very first `Node.__setattr__`
    # (`torch/fx/node.py:885` calls `hasattr(self, name)`), and a getter that
    # raises `NotImplementedError` rather than `AttributeError` makes
    # `hasattr` propagate instead of answering `False`. So the wall was not
    # only the four stubs docs/graph/EXPORT.md §4.1 counted -- the three members it
    # listed as "present and real" were raising too, which is why they are
    # deleted rather than left alone here.
    for _stub in (
        "_args", "_erased", "_input_nodes", "_kwargs", "_next", "_prev",
        "_repr_fn", "_sort_key", "graph", "meta", "name", "op", "target",
        "type", "users",
    ):
        if _stub in vars(node_base):
            delattr(node_base, _stub)
    _immutable = []

    def _immutable_collections():
        # Lazy: `torch.fx` is not importable at bootstrap time. Upstream's
        # node.cpp does the same lazy import for the same reason.
        if not _immutable:
            from torch.fx.immutable_collections import immutable_dict, immutable_list

            _immutable.append((immutable_list, immutable_dict))
        return _immutable[0]

    def _fx_map_aggregate(a, fn):
        """`fn` over every LEAF of an argument aggregate, structure preserved."""
        if isinstance(a, tuple):
            mapped = tuple(_fx_map_aggregate(e, fn) for e in a)
            # namedtuples rebuild positionally, not from an iterable
            return type(a)(*mapped) if hasattr(a, "_fields") else mapped
        if isinstance(a, list):
            immutable_list, _ = _immutable_collections()
            return immutable_list(_fx_map_aggregate(e, fn) for e in a)
        if isinstance(a, dict):
            _, immutable_dict = _immutable_collections()
            return immutable_dict(
                (k, _fx_map_aggregate(v, fn)) for k, v in a.items()
            )
        if isinstance(a, slice):
            return slice(
                _fx_map_aggregate(a.start, fn),
                _fx_map_aggregate(a.stop, fn),
                _fx_map_aggregate(a.step, fn),
            )
        return fn(a)

    def _fx_map_arg(a, fn):
        """`fn` over every NODE in an argument aggregate, everything else
        passed through. A `set` is a leaf here, not an aggregate -- measured
        upstream, where `_fx_map_arg({a}, ...)` gives `{a}` back untouched."""
        return _fx_map_aggregate(a, lambda x: fn(x) if isinstance(x, node_base) else x)

    def __init__(self, graph, name, op, target, return_type=None):
        self.graph = graph
        self.name = name
        self.op = op
        self.target = target
        self.type = return_type
        self._args = ()
        self._kwargs = {}
        self._input_nodes = {}
        self.users = {}
        self.meta = {}
        self._repr_fn = None
        self._sort_key = ()
        self._erased = False
        # A fresh node is its own neighbour, so `_prepend` can unlink it
        # unconditionally before splicing it in.
        self._prev = self
        self._next = self

    def _update_args_kwargs(self, args, kwargs):
        for old in self._input_nodes:
            old.users.pop(self, None)
        identity = lambda x: x  # noqa: E731
        self._args = _fx_map_aggregate(args, identity)
        self._kwargs = _fx_map_aggregate(kwargs, identity)
        found = {}

        def collect(n):
            found.setdefault(n)
            return n

        _fx_map_arg(self._args, collect)
        _fx_map_arg(self._kwargs, collect)
        self._input_nodes = found
        for n in found:
            n.users.setdefault(self)

    def _remove_from_list(self):
        prev, nxt = self._prev, self._next
        if prev is not None:
            prev._next = nxt
        if nxt is not None:
            nxt._prev = prev

    def _prepend(self, x):
        if x is self:
            return
        x._remove_from_list()
        prev = self._prev
        prev._next = x
        x._prev = prev
        x._next = self
        self._prev = x
        psk, nsk = prev._sort_key, self._sort_key
        if len(psk) > len(nsk):
            x._sort_key = psk[:-1] + (psk[-1] + 1,)
        elif len(psk) < len(nsk):
            x._sort_key = nsk[:-1] + (nsk[-1] - 1,)
        else:
            x._sort_key = psk + (0,)

    def _replace_input_with(self, old_input, new_input):
        def swap(n):
            return new_input if n is old_input else n

        self._update_args_kwargs(
            _fx_map_arg(self._args, swap), _fx_map_arg(self._kwargs, swap)
        )

    node_base.__init__ = __init__
    node_base._update_args_kwargs = _update_args_kwargs
    node_base._remove_from_list = _remove_from_list
    node_base._prepend = _prepend
    node_base._replace_input_with = _replace_input_with
    # Ordering is by `_sort_key`, not by a walk of the list: upstream defines
    # all four comparisons on the C type and `fx` passes nodes to `sorted()`.
    # `__eq__`/`__hash__` are deliberately NOT touched -- `users` and
    # `_input_nodes` are dicts KEYED BY NODE, so identity hashing is what makes
    # two structurally identical nodes two different users.
    node_base.__lt__ = lambda self, other: self._sort_key < other._sort_key
    node_base.__gt__ = lambda self, other: self._sort_key > other._sort_key
    node_base.__le__ = lambda self, other: self._sort_key <= other._sort_key
    node_base.__ge__ = lambda self, other: self._sort_key >= other._sort_key

    class _NodeIter:
        """`torch._C._NodeIter(root, reversed)` -- `graph.py:302`'s iterator.

        `reversed` picks the direction; the root sentinel terminates and is
        never yielded, and erased nodes are skipped rather than yielded.
        """

        def __init__(self, root, reversed=False):
            self._root = root
            self._cur = root
            self._reversed = bool(reversed)

        def __iter__(self):
            return self

        def __next__(self):
            cur = self._cur
            while True:
                cur = cur._prev if self._reversed else cur._next
                self._cur = cur
                if cur is self._root:
                    raise StopIteration
                if not cur._erased:
                    return cur

    _NodeIter.__module__ = "torch._C"
    _NodeIter.__qualname__ = _NodeIter.__name__ = "_NodeIter"
    module._NodeIter = _NodeIter
    module._fx_map_arg = _fx_map_arg
    module._fx_map_aggregate = _fx_map_aggregate


def _install_dispatch_suppression(module) -> None:
    """The counter behind `no_dispatch()`, read by the dispatcher door itself.

    docs/graph/EXPORT.md §2.4 wrote this function's specification as a prediction:

        "Entering a counter is a correct implementation **only because this shim
        never consults the mode stack in the first place** ... When
        `_aten_dispatch` learns to consult the stack, this guard stops being a
        counter and starts being load-bearing, and the counter is here so that
        change is a body to fill rather than a hole to find."

    `_aten_dispatch` has since learned to consult the stack (EXPORT.md §6 step
    1), so that day arrived, and the counter had indeed stopped being correct.
    The symptom was not a missing name: `torch/_subclasses/meta_utils.py:2009`
    builds the meta tensor behind every fake tensor under `no_dispatch()`, that
    suppression did nothing, and so `aten.empty_strided.default` was dispatched
    **into** the `FakeTensorMode` that called it.  It died in
    `_find_common_device` -- a factory has no tensor arguments to take a device
    from -- several frames from the guard that should have prevented the
    re-entry.  docs/graph/EXPORT4.md §5.

    So the state lives here, where both halves can reach it: the guard writes it
    and `aten.rs`'s `any_dispatch_mode_active` reads it.  Keeping it in the
    guard's own module instead would leave the door unable to see it, which is
    the arrangement that just failed.

    Thread-local, because the mode stack is: suppressing dispatch on one thread
    must not suppress it on another.  The depth is a *count*, not a flag,
    because `no_dispatch()` nests -- `fake_tensor.py` enters it inside code that
    is already inside it, and a boolean would be cleared by the inner exit.
    """
    import threading

    _local = threading.local()

    def _depth():
        return getattr(_local, "depth", 0)

    def _shim_dispatch_suppressed():
        return _depth() > 0

    def _shim_push_dispatch_suppression():
        _local.depth = _depth() + 1
        return _local.depth

    def _shim_pop_dispatch_suppression():
        d = _depth()
        if d <= 0:
            # Not defensive padding: an unbalanced pop means some guard's
            # `__exit__` ran without its `__enter__`, and silently clamping to
            # zero would leave dispatch suppressed or unsuppressed at random for
            # the rest of the process.  Better to say so at the pop.
            raise RuntimeError(
                "torch._C._shim_pop_dispatch_suppression: popped with depth 0 -- "
                "a dispatch-suppression guard exited without entering"
            )
        _local.depth = d - 1
        return _local.depth

    for name, fn_ in (
        ("_shim_dispatch_suppressed", _shim_dispatch_suppressed),
        ("_shim_push_dispatch_suppression", _shim_push_dispatch_suppression),
        ("_shim_pop_dispatch_suppression", _shim_pop_dispatch_suppression),
    ):
        fn_.__name__ = name
        fn_.__qualname__ = f"torch._C.{name}"
        setattr(module, name, fn_)


def _install_dispatcher_kernel_predicates(module) -> None:
    """`torch._C._dispatch_has_computed_kernel_for_dispatch_key(name, key)`.

    `fake_impls.py:1568`'s `has_meta` asks this once per fake dispatch, and
    `fake_tensor.py:3077` calls it an *optimization*: a `True` answer means "run
    the meta kernel, and if it raises `NotImplementedError` fall back", which is
    what would happen anyway; a `False` answer means "skip trying".  So the
    direction of the risk is not symmetric, and it is worth saying which way:

    * a wrong `True` costs a raised-and-caught `NotImplementedError`;
    * a wrong `False` **skips a kernel that exists** and sends the op down
      `maybe_run_unsafe_fallback`, which raises `UnsupportedOperatorException`
      outright when sizes are symbolic -- i.e. under `torch.export`, always.

    **The rule was measured, not assumed.**  Every one of the 1783 `aten`
    overloads on torch 2.13.0 answers `True` for `"Meta"`, with no exceptions,
    and `prims` does too; the predicate distinguishes builtin operators from
    custom-library ones that never registered a meta kernel.  So the rule is the
    namespace, and `test_export4.py::test_has_computed_kernel_matches_upstream_across_the_whole_aten_surface`
    re-derives it against upstream rather than trusting this paragraph.

    **It answers for `Meta` and `CPU` and refuses every other key by name.**
    Those two are the keys this shim can speak for: its kernels are reached
    through one door (`_aten_dispatch`) that serves both, and an operator it
    lacks raises `NotImplementedError` there, which is precisely the condition
    the `True` answer promises the caller will handle.  For `CUDA`, `Autograd`,
    `SparseCPU` and the rest, this shim has no kernels and no registry to
    consult, and the two available lies are both bad -- `False` would say "no
    such kernel exists" about upstream's dispatcher, and `True` would promise a
    CUDA kernel that does not exist.  So it refuses, naming the key, rather than
    guessing.  docs/graph/EXPORT4.md §5.
    """
    answerable = ("Meta", "CPU")
    builtin = ("aten::", "prims::", "prim::")

    def _dispatch_has_computed_kernel_for_dispatch_key(name, dispatch_key):
        key = str(dispatch_key)
        if key not in answerable:
            raise NotImplementedError(
                "not implemented in torch._C shim: "
                "_dispatch_has_computed_kernel_for_dispatch_key("
                f"{name!r}, {key!r}) -- this shim can answer for "
                f"{' and '.join(answerable)} only, because those are the keys "
                "its single dispatcher door serves. It keeps no kernel registry "
                "for other keys, so both available answers would be a claim it "
                "cannot support. docs/graph/EXPORT4.md §5"
            )
        return str(name).startswith(builtin)

    _dispatch_has_computed_kernel_for_dispatch_key.__name__ = (
        "_dispatch_has_computed_kernel_for_dispatch_key"
    )
    _dispatch_has_computed_kernel_for_dispatch_key.__qualname__ = (
        "torch._C._dispatch_has_computed_kernel_for_dispatch_key"
    )
    module._dispatch_has_computed_kernel_for_dispatch_key = (
        _dispatch_has_computed_kernel_for_dispatch_key
    )
    module._shim_has_computed_kernel_keys = answerable


def _install_arg_parser_predicates(module) -> None:
    """`torch._C._should_allow_numbers_as_tensors` -- a fixed table, not a policy.

    `fake_tensor.py:2648` asks this on **every** dispatch that reaches a fake
    tensor, so with it left as a synthesised `_Unimplemented` the whole of
    `torch.export` stopped here.  It is the third wall docs/graph/EXPORT4.md measured
    and the cheapest of the three, because it is pure data.

    Upstream's is a `static std::unordered_set<std::string>` in
    `torch/csrc/utils/python_arg_parser.cpp`, consulted by
    `should_allow_numbers_as_tensors(name)`.  It answers one question: for this
    operator, may a Python number bind to a `Tensor` parameter?  That is true of
    the arithmetic ops that have a `Scalar` overload and of the conversion ops,
    and false of everything else -- `torch.relu(2)` is not a thing, while
    `torch.add(t, 2)` is.

    **The table below was derived by enumerating upstream, not transcribed from
    the C++.**  Every name in `dir(torch.ops.aten)`, plus its `_` and `_out`
    spellings, was passed to the real
    `torch._C._should_allow_numbers_as_tensors` on torch 2.13.0 and the ones
    answering `True` are exactly these 31.  The derivation is rerun by
    `test_export4.py::test_should_allow_numbers_as_tensors_matches_upstream_name_for_name`,
    which compares this table against upstream directly and so fails if a torch
    upgrade adds a name -- a transcription would go stale silently instead.

    Note the shape of the set: `add`/`sub`/`mul`/`div` and their long spellings
    (`subtract`, `multiply`, `divide`, `true_divide`, `floor_divide`), each in
    three forms (plain, in-place, `_out`), plus `to`, `copy`, `copy_` and
    `_to_copy`.  `rsub` and `pow` are NOT in it, which is worth knowing before
    anyone assumes "binary op" is the rule.
    """
    allowed = frozenset({
        "_to_copy",
        "add", "add_", "add_out",
        "copy", "copy_",
        "div", "div_", "div_out",
        "divide", "divide_", "divide_out",
        "floor_divide", "floor_divide_", "floor_divide_out",
        "mul", "mul_", "mul_out",
        "multiply", "multiply_", "multiply_out",
        "sub", "sub_", "sub_out",
        "subtract", "subtract_", "subtract_out",
        "to",
        "true_divide", "true_divide_", "true_divide_out",
    })

    def _should_allow_numbers_as_tensors(name):
        return name in allowed

    _should_allow_numbers_as_tensors.__name__ = "_should_allow_numbers_as_tensors"
    _should_allow_numbers_as_tensors.__qualname__ = (
        "torch._C._should_allow_numbers_as_tensors"
    )
    module._should_allow_numbers_as_tensors = _should_allow_numbers_as_tensors
    # Readable so a test can diff the table against upstream rather than
    # probing it one name at a time.
    module._shim_numbers_as_tensors_names = tuple(sorted(allowed))


def _install_dispatch_keys(module) -> None:
    """`DispatchKey` and `DispatchKeySet`, for real.

    VENDOR.md wall 14 -- `torch/_ops.py:139` asserts `isinstance(k,
    DispatchKey)` and IMPORT_WALLS reached the same line from the opposite
    direction, which is what makes it structure rather than a detour. It is
    also, unusually for this file, *cheap* structure: `DispatchKey` is an
    enumeration and `DispatchKeySet` is a set of them, and both fall out of
    `enum` and `frozenset` with no dispatcher behind them.

    Every `HigherOrderOperator.__init__` runs `_dispatch_keyset_full()` and
    then removes keys from it (`torch/_ops.py:304`), so this executes during
    `import torch`, not merely on use. What is *not* here is anything that
    consults the set to route a call -- `_aten_dispatch` remains the one door.
    """
    DispatchKey = module.DispatchKey

    class DispatchKeySet:
        __module__ = "torch._C"
        __slots__ = ("_keys",)

        def __init__(self, key=None):
            if isinstance(key, DispatchKeySet):
                self._keys = key._keys
            elif key is None:
                self._keys = frozenset()
            else:
                self._keys = frozenset({key})

        @classmethod
        def _of(cls, keys):
            made = cls.__new__(cls)
            made._keys = frozenset(keys)
            return made

        def has(self, key):
            return key in self._keys

        def add(self, key):
            return self._of(self._keys | {key})

        def remove(self, key):
            return self._of(self._keys - {key})

        def __or__(self, other):
            return self._of(self._keys | DispatchKeySet(other)._keys)

        def __and__(self, other):
            return self._of(self._keys & DispatchKeySet(other)._keys)

        def __sub__(self, other):
            return self._of(self._keys - DispatchKeySet(other)._keys)

        def __iter__(self):
            return iter(sorted(self._keys, key=lambda k: k.name))

        def __contains__(self, key):
            return key in self._keys

        def __len__(self):
            return len(self._keys)

        def __bool__(self):
            return bool(self._keys)

        def __eq__(self, other):
            return isinstance(other, DispatchKeySet) and self._keys == other._keys

        def __hash__(self):
            return hash(self._keys)

        def highestPriorityTypeId(self):
            # Priority is the enum's declaration order in the stub, which is
            # upstream's own order. Not a claim that the shim dispatches by it.
            return max(self._keys, key=lambda k: list(DispatchKey).index(k))

        def raw_repr(self):
            return sum(1 << list(DispatchKey).index(k) for k in self._keys)

        def __repr__(self):
            return "DispatchKeySet(" + ", ".join(k.name for k in self) + ")"

    module.DispatchKeySet = DispatchKeySet
    every = DispatchKeySet._of(DispatchKey)
    module._dispatch_keyset_full = lambda: every
    module._dispatch_keyset_full_after = lambda key: DispatchKeySet._of(
        list(DispatchKey)[list(DispatchKey).index(key) + 1 :]
    )
    module._dispatch_keyset_to_string = lambda keyset: repr(keyset)
    # "Does the dispatcher know this op at all?" `torch/_decomp/__init__.py:90`
    # uses it to filter TorchScript's junk overloads (`aten.add.float_int`) out
    # of the decomposition registry.
    #
    # PAPERED OVER, and the direction was chosen rather than defaulted. `False`
    # would empty the decomposition table, which is the mechanism DESIGN.md §2
    # relies on for everything outside Core ATen -- so the table would be gone
    # in exchange for tidiness. `True` keeps the table and lets a handful of
    # junk overloads into it, where they are inert. The cost of the lie is a
    # few extra registry entries; the cost of the truth would be the feature.
    module._dispatch_has_kernel = lambda *a, **k: True
    # The narrower question -- "for *this* backend key?" -- is answered
    # conservatively, because a `True` there would claim a specific kernel.
    module._dispatch_has_kernel_for_dispatch_key = lambda *a, **k: False
    module._dispatch_has_kernel_for_any_dispatch_key = lambda *a, **k: False
    # "Which ops are registered under this key?" Answered from
    # `native_functions.yaml` for the four alias keys it declares, and refused
    # by name for backend keys. docs/graph/DECOMP.md §3 -- this is what
    # `core_aten_decompositions()` stopped at.
    module._dispatch_get_registrations_for_dispatch_key = _dispatch_registrations

    # A `DispatchKeySet` *value*, not a function.
    # `torch/_subclasses/functional_tensor.py:146` does
    # `torch._C._additional_keys_to_prop_for_wrapper_tensors.add(...)` in a
    # class body. Upstream seeds it with the keys a wrapper subclass should
    # inherit from its wrapped tensor; empty is the honest starting point for a
    # shim with no key propagation, and `.add` still works because the set is
    # real.
    module._additional_keys_to_prop_for_wrapper_tensors = DispatchKeySet()

    def _parse_dispatch_key(name):
        """`torch/library.py:915` -- "is this string a dispatch key?".

        Upstream returns the `DispatchKey` or `None`, and the caller branches
        on `None` to decide whether the string was a device type instead. The
        enum comes from the vendored stub, so this is a real lookup, not a
        guess.
        """
        return getattr(DispatchKey, name, None)

    module._parse_dispatch_key = _parse_dispatch_key
    module._dispatch_key_parse = _parse_dispatch_key


def _install_library(module, schemas) -> None:
    """`torch._C._dispatch_library` -- the operator registry, as a recorder.

    `torch/library.py:244` builds one of these per `Library(...)`, and
    `torch/_meta_registrations.py`, `torch/_decomp/` and several `torch/_refs`
    modules construct libraries and call `define`/`impl` *at import time*, so
    this is import-blocking rather than lazy.

    **This is the largest thing in this file that is papered over, and it is
    worth being precise about what it costs.** Upstream's object writes into
    the C++ dispatcher, so a kernel registered from Python is afterwards
    reachable through `torch.ops`. Here the registrations are recorded and
    dropped: `_aten_dispatch` is the only thing that answers a call, and it
    knows nothing about them. Concretely, `torch.library.impl(...)` appears to
    succeed and then has no effect.

    Recording rather than discarding, and exposing it as
    `_C._shim_registrations`, is what keeps that from being invisible -- the
    size of that list is the size of the gap.
    """
    registrations: list = []
    module._shim_registrations = registrations

    class _DispatchLibrary:
        __module__ = "torch._C"
        __slots__ = ("kind", "ns", "dispatch_key")

        def __init__(self, kind, ns, dispatch_key, filename="", lineno=0):
            self.kind = kind
            self.ns = ns
            self.dispatch_key = dispatch_key

        def define(self, schema, alias_analysis="", tags=()):
            # `torch/library.py:313` keeps the result and then recomputes the
            # name itself from the schema, so what matters is only that this is
            # the op name -- upstream returns exactly that.
            name = schema.split("(")[0].strip()
            registrations.append(("define", self.ns, name, schema))
            # Keep the text, not just the fact that it arrived. Upstream's
            # dispatcher parses it here and `torch.ops.<ns>.<op>._schema`
            # reads it back later; the shim used to drop it and then answer
            # that read with an empty schema, which is a *wrong* answer rather
            # than a missing one -- an empty schema has no returns, so
            # `is_functional_schema` is False and
            # `torch.library.register_autograd` refuses every custom op.
            # `transformers/integrations/moe.py:253` is where that showed up.
            #
            # This is the one thing `define` does that is not merely recorded,
            # and it does not narrow the gap the docstring above describes:
            # knowing an op's schema is not knowing how to run it, and
            # `_aten_dispatch` still has no kernel for any of these.
            try:
                parsed = _Schema.parse(f"{self.ns}::{schema}")
            except Exception:  # noqa: BLE001 -- a schema this cannot parse is
                # not worth failing a registration over; the caller gets the
                # same empty schema it got before.
                pass
            else:
                schemas[(parsed.name, parsed.overload_name)] = parsed
            return name

        def impl(self, name, dispatch_key, fn, with_keyset=False):
            registrations.append(("impl", self.ns, name, dispatch_key))

        def impl_with_aoti_compile(self, ns, name, dispatch_key):
            registrations.append(("impl_with_aoti_compile", ns, name, dispatch_key))

        def fallback(self, dispatch_key, fn, with_keyset=False):
            registrations.append(("fallback", self.ns, dispatch_key, None))

        def reset(self):
            return None

        def __getattr__(self, name):
            # The registry has more entry points than the five `torch/library.py`
            # uses (`register_ad_inplace_or_view_fallback`, and others reached
            # only from `torch/_library/`). They all record and return nothing,
            # so they are answered here rather than transcribed one at a time --
            # and every one of them lands in `_shim_registrations`, so the list
            # still measures the gap.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)

            def recorder(*args, **kwargs):
                registrations.append((name, self.ns, args and args[0], None))

            return recorder

    module._dispatch_library = _DispatchLibrary


# ---------------------------------------------------------------------------
# `torch._C._nn` -- the three names a model path actually calls
# ---------------------------------------------------------------------------
#
# The submodule loop above gives `_nn` all 70 names the stubs declare, every
# one of them a raising stub. docs/bindings/NN_SURFACE.md measured which of them a
# 2-layer Llama forward plus greedy `generate` calls, on upstream torch 2.13.0,
# by wrapping each builtin rather than by reading the tree:
#
#     _C._nn.linear                          150 calls
#     _C._nn.scaled_dot_product_attention      10
#     _C._nn.silu                              20
#     ...and nothing else. 3 of 96.
#
# So this is not a port of `_C._nn`; it is the three names, and the other 67
# stay raising stubs on purpose (rule 2 -- a hole, never a switch).
#
# WHY THESE ARE PYTHON-LEVEL COMPOSITIONS AND NOT NEW KERNELS. `aten::linear`
# and `aten::dropout` are `CompositeImplicitAutograd` upstream: they have no
# kernel at all, and what a `TorchDispatchMode` logger sees under
# `F.linear(...)` is `t` + `addmm`/`mm` + `view`, never `aten.linear.default`.
# Reproducing that decomposition here is therefore *following* upstream rather
# than routing around it, and it keeps DESIGN.md §6's single door: every step
# below goes through `_C._aten_dispatch` with a key the golden harness already
# compares against upstream. Nothing here computes.
#
# `scaled_dot_product_attention` is the opposite shape -- upstream picks a
# backend and calls one fused op -- so it is a selection, not a decomposition.


def _install_nn(module, dispatch) -> None:
    nn = module._nn
    reachable = frozenset(module._aten_all_implemented())

    def _t(weight):
        return dispatch("aten.t.default", weight)

    # `aten::linear`, transcribed from what upstream *does*, measured branch by
    # branch against torch 2.13.0 (docs/bindings/NN_SURFACE.md §4):
    #
    #     2-D  + bias      t, addmm
    #     N-D  + bias      view, t, addmm, view          (contiguous input)
    #     any  + bias      t, ..., matmul, ..., add      (non-contiguous input)
    #     2-D, no bias     t, mm
    #     N-D, no bias     t, view, mm, _unsafe_view     (i.e. plain matmul)
    #
    # The no-bias case is exactly `matmul(input, weight.t())` in every rank, so
    # it is spelled that way -- `aten.matmul.default` has a kernel here and it
    # is the op upstream's own `at::native::linear` calls.
    #
    # PATCHED, AND SAY SO: the bias branches want `aten.addmm.default`, which
    # this shim has no kernel for. Until it does, bias goes through
    # `matmul` + `add.Tensor` -- which is a real upstream path (it is the
    # non-contiguous branch above), just taken under a condition upstream would
    # not take it under. The numbers agree to float rounding; the accumulation
    # does not, because `addmm` is a fused GEMM and this is a GEMM followed by
    # a separate broadcast add. The `if` below is written against
    # `_aten_all_implemented()` rather than hardcoded, so the day an `addmm`
    # kernel lands this takes upstream's path with no edit here.
    _HAS_ADDMM = "aten.addmm.default" in reachable

    def linear(input, weight, bias=None):
        wt = _t(weight)
        if bias is None:
            return dispatch("aten.matmul.default", input, wt)
        if _HAS_ADDMM and input.dim() == 2:
            return dispatch("aten.addmm.default", bias, input, wt)
        if _HAS_ADDMM and input.dim() > 2 and input.is_contiguous():
            sizes = tuple(input.shape)
            flat = dispatch("aten.view.default", input, [-1, sizes[-1]])
            out = dispatch("aten.addmm.default", bias, flat, wt)
            return dispatch(
                "aten.view.default", out, list(sizes[:-1]) + [tuple(out.shape)[-1]]
            )
        return dispatch(
            "aten.add.Tensor", dispatch("aten.matmul.default", input, wt), bias
        )

    def silu(input):
        return dispatch("aten.silu.default", input)

    # `torch/nn/functional.py:2054` is `gelu = _add_docstr(torch._C._nn.gelu,
    # ...)` -- `F.gelu` *is* this binding, not a wrapper around it, so it is a
    # `_nn` composite rather than a table entry for the same reason `linear`
    # and `silu` are: nothing looks up `_C._nn.gelu` through `_Overloads`.
    #
    # `approximate` has to be keyword-only *at this Python signature*, not
    # just in what gets forwarded to `dispatch`. The schema marks it
    # keyword-only (`*` before it) and upstream enforces that at the parser
    # (measured: `torch._C._nn.gelu(x, "tanh")` raises "takes 1 positional
    # argument but 2 were given" on torch 2.13.0). `gelu_default` in aten.rs
    # reproduces that rejection too, but only by checking the length of the
    # positional tuple it is *given* -- and this composite always calls
    # `dispatch` with exactly one positional (`input`) and `approximate` as a
    # kwarg regardless of how its own caller passed it. Without the bare `*`
    # below, `gelu(x, "tanh")` would bind fine at this level and silently
    # reach the tanh branch where upstream raises -- caught by testing the
    # positional form directly, not reasoned out in advance.
    def gelu(input, *, approximate="none"):
        return dispatch("aten.gelu.default", input, approximate=approximate)

    # Upstream's `scaled_dot_product_attention` is a *backend selection*, and
    # the selection was measured rather than assumed (docs/bindings/NN_SURFACE.md §5).
    # On CPU, 4-D float inputs with `dropout_p == 0` go to
    # `aten._scaled_dot_product_flash_attention_for_cpu` -- for float32,
    # float64, float16 and bfloat16 alike, with or without a mask, with or
    # without `is_causal`, and (measured) with both at once. Everything else
    # falls back to the math backend, which is a different op sequence
    # (`mul.Scalar`, `expand`, `view`, `bmm`, `_safe_softmax`, ...).
    #
    # Only the flash path is wired. The math fallback is refused by name rather
    # than approximated, because silently substituting a plain softmax for
    # `_safe_softmax` would differ from upstream exactly on the fully-masked
    # rows that `_safe_softmax` exists for.
    #
    # **The reason given here used to be "`aten._safe_softmax.default` has no
    # kernel", and that stopped being true.** It has been in `IMPLEMENTED` and
    # golden-compared since docs/kernels/SDPA.md; so have `mul.Scalar`, `expand`,
    # `view` and `bmm`. Re-checked against the built artefact rather than
    # against this comment (docs/kernels/TRIL.md §2): every kernel the math backend
    # needs for the 3-D and non-4-D cases is present, and what is missing is
    # the *composite* -- nobody has transcribed upstream's math-backend op
    # sequence, its scale handling or its mask expansion. That is a real
    # reason to refuse and it is a different one. The refusals below say so.
    #
    # This is the second time a refusal in this function went stale (the
    # bool-mask one below is the first), and both times the text named kernels
    # that had since landed. A refusal is code that never runs on the happy
    # path, so nothing re-reads it; the rule this repository keeps re-learning
    # is that touching one means re-deriving its claim, not just its wording.
    #
    # The aten op returns `(output, logsumexp)`; upstream's binding returns the
    # first. `logsumexp` is dropped here for the same reason upstream drops it:
    # it is a backward-pass residual and this shim has no backward.
    _FLASH = "aten._scaled_dot_product_flash_attention_for_cpu.default"

    def _boolean_attn_mask(mask, dtype):
        """`convert_boolean_attn_mask`, shared by the flash and math paths.

        Argument order in the `where` is the half a plausible reading gets
        backwards, so it was read off the *values*, not the shapes:
        `where(tensor([[True, False]]), 0.0, -inf)` gives `[[0.0, -inf]]`. A
        `True` in a boolean attention mask means *attend*, so it selects the
        zero; the `-inf` is what a `False` selects.
        """
        neg_inf = dispatch("aten.scalar_tensor.default", float("-inf"), dtype=dtype)
        zero = dispatch("aten.scalar_tensor.default", 0.0, dtype=dtype)
        return dispatch("aten.where.self", mask, zero, neg_inf)

    def _sdpa_math(query, key, value, attn_mask, dropout_p, is_causal, scale,
                   enable_gqa):
        """`at::native::_scaled_dot_product_attention_math`, transcribed.

        **Why this exists at all**: a non-zero `dropout_p` takes the whole call
        off the fused kernel. `_scaled_dot_product_flash_attention_for_cpu`
        does not implement dropout, so upstream falls back here -- which means
        `gpt2`, `bert` and `gpt_bigcode` run ONE op in `.eval()` and twenty in
        `.train()` (docs/training/TRAIN.md §3). Until this was written, attention
        dropout was the wall behind `nn.Dropout` for all three.

        The sequence is upstream's, measured with a `TorchDispatchMode` logger
        on torch 2.13.0 and reproduced op for op and in order.

        Two details are measured rather than reasoned:

          * **The scale is applied as its square root, to BOTH operands.**
            Upstream multiplies `query` by `sqrt(scale)` and `key.transpose(
            -2, -1)` by `sqrt(scale)` rather than scaling the product once --
            it is a numerical-stability choice, and it is *observable*: for
            `E=8` with no explicit scale the factor is 0.5946035575013605,
            which is `sqrt(1/sqrt(8))`, and a shim that multiplied the logits
            by `1/sqrt(8)` afterwards would differ in the last bits of every
            attention weight.
          * **A negative explicit `scale` is not just passed through.**
            Upstream takes `abs(scale)`, roots it, and negates the *query*
            multiplier, so the sign survives exactly once instead of being
            lost to the square root.
        """
        if is_causal and attn_mask is not None:
            raise RuntimeError(
                "_scaled_dot_product_attention: Explicit attn_mask should not "
                "be set when is_causal=True"
            )
        if attn_mask is not None and attn_mask.dtype == module.bool:
            attn_mask = _boolean_attn_mask(attn_mask, query.dtype)

        raw_scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
        factor = math.sqrt(abs(raw_scale))
        q = dispatch(
            "aten.mul.Scalar", query, -factor if raw_scale < 0.0 else factor
        )

        if is_causal:
            L, S = query.shape[-2], key.shape[-2]
            causal = dispatch("aten.ones.default", [L, S], dtype=module.bool)
            causal = dispatch("aten.tril.default", causal)
            attn_mask = _boolean_attn_mask(causal, query.dtype)

        if enable_gqa:
            # `repeat_interleave` along the head dim, spelled the way upstream
            # spells it: unsqueeze, expand, contiguous clone, view. Not a
            # `repeat`, which would interleave the wrong way round.
            n_rep = query.shape[1] // key.shape[1]
            if n_rep != 1:
                def _repeat_heads(t):
                    b, h, s, e = t.shape
                    x = dispatch("aten.unsqueeze.default", t, 2)
                    x = dispatch("aten.expand.default", x, [b, h, n_rep, s, e])
                    x = dispatch("aten.clone.default", x)
                    return dispatch("aten.view.default", x, [b, h * n_rep, s, e])

                key, value = _repeat_heads(key), _repeat_heads(value)

        k_t = dispatch("aten.transpose.int", key, -2, -1)
        k_t = dispatch("aten.mul.Scalar", k_t, factor)
        attn = dispatch("aten.matmul.default", q, k_t)
        if attn_mask is not None:
            attn = dispatch("aten.add.Tensor", attn, attn_mask)
        # `_safe_softmax`, not `softmax`: a fully-masked row is all `-inf`, and
        # the two disagree there by exactly the amount that matters (zeros
        # versus nan). It is the reason this composite could not be
        # approximated while that kernel was missing.
        attn = dispatch("aten._safe_softmax.default", attn, -1)
        if dropout_p > 0.0:
            attn = module._VariableFunctions.dropout(attn, dropout_p, True)
        return dispatch("aten.matmul.default", attn, value)

    def scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        *,
        scale=None,
        enable_gqa=False,
    ):
        if query.dim() != 4:
            raise NotImplementedError(
                "not implemented in torch._C shim: "
                f"scaled_dot_product_attention on a {query.dim()}-D query -- upstream "
                "drops to the math backend for anything but 4-D {B, H, T, K}. Every "
                "kernel that backend needs is implemented here "
                "(_safe_softmax, mul.Scalar, expand, view, bmm); what is missing is "
                "the composite that sequences them, which nobody has transcribed"
            )
        if attn_mask is not None and attn_mask.dtype == module.bool:
            # `convert_boolean_attn_mask`, and it is now built rather than
            # refused -- `falcon` is what asked (docs/architectures/ARCH20.md §7), which
            # passes a bool mask straight into SDPA.
            #
            # **The refusal that used to be here had gone stale, and that is
            # the interesting part.** Its own text named the two kernels it was
            # waiting on -- `aten.scalar_tensor.default` and
            # `aten.where.self` -- and both have been in `IMPLEMENTED`, and
            # golden-compared, since docs/architectures/ARCH.md. Nothing re-read the refusal
            # when they landed, so an architecture stayed blocked on a wall
            # that had already been removed. A refusal that names its
            # dependencies is only better than one that does not if somebody
            # re-checks them.
            #
            # The sequence is upstream's, measured with a TorchDispatchMode
            # logger on torch 2.13.0 and reproduced op for op and *in order*:
            #
            #     scalar_tensor(-inf)                -> the masked-out fill
            #     scalar_tensor(0.0)                 -> the attend fill
            #     where.self(mask, zero, neg_inf)    -> the additive mask
            #     _scaled_dot_product_flash_attention_for_cpu(q, k, v, mask)
            #
            # Argument order in the `where` is the half a plausible reading
            # gets backwards, so it was read off the *values*, not the shapes:
            # `where(tensor([[True, False]]), 0.0, -inf)` gives `[[0.0,
            # -inf]]`. A `True` in a boolean attention mask means *attend*, so
            # it selects the zero; the `-inf` is what a `False` selects. Both
            # fills carry the query's dtype, not the default float, which is
            # what keeps a float16 forward in float16.
            attn_mask = _boolean_attn_mask(attn_mask, query.dtype)
        # Grouped-query attention. The repetition itself is NOT here -- it is
        # in the aten kernel, because that is where upstream does it: a
        # TorchDispatchMode over `enable_gqa=True` reports one op, with the
        # key and value still at their original head count (see
        # `repeat_kv_heads` in aten.rs for the measurement). What lives at
        # this level is the two checks upstream does at this level, both
        # measured on 2.13.0:
        #
        #   enable_gqa=False, mismatched heads
        #       "The size of tensor a (9) must match the size of tensor b (3)
        #        at non-singleton dimension 1"
        #   enable_gqa=True, head counts that do not divide
        #       "Number of heads in key and value must divide the number of
        #        heads in "
        #
        # The second one really does end mid-sentence upstream; it is
        # reproduced verbatim rather than tidied, for the reason docs/models/CKPT2.md
        # §4 gives about `view.dtype`'s messages -- a message that differs
        # from upstream's only in wording is useless exactly where it is
        # needed.
        #
        # A single key/value head is NOT a mismatch: it is an ordinary
        # singleton broadcast and upstream accepts it with `enable_gqa=False`
        # (measured). Refusing it here would have refused multi-query
        # attention, which several architectures use.
        q_heads, kv_heads = query.shape[1], key.shape[1]
        if enable_gqa:
            if kv_heads == 0 or q_heads % kv_heads != 0:
                raise RuntimeError(
                    "Number of heads in key and value must divide the number of heads in "
                )
        elif kv_heads != q_heads and kv_heads != 1:
            raise RuntimeError(
                f"The size of tensor a ({q_heads}) must match the size of tensor b "
                f"({kv_heads}) at non-singleton dimension 1"
            )
        # **The backend selection, and it is upstream's, not a convenience.**
        # `_scaled_dot_product_flash_attention_for_cpu` refuses `dropout > 0`
        # -- upstream's own kernel does, with "Currently do not support dropout
        # > 0" -- so a non-zero `dropout_p` is not a slower road to the same
        # answer, it is a *different* backend with a different op sequence
        # (docs/training/TRAIN.md §4). Measured on 2.13.0: `dropout_p=0.0` fires exactly
        # one op and `dropout_p=0.1` fires twenty.
        if dropout_p != 0.0:
            return _sdpa_math(
                query, key, value, attn_mask, dropout_p, is_causal, scale,
                enable_gqa,
            )
        return dispatch(
            _FLASH,
            query,
            key,
            value,
            dropout_p,
            is_causal,
            attn_mask=attn_mask,
            scale=scale,
        )[0]


    def adaptive_avg_pool2d(input, output_size):
        """`torch._C._nn.adaptive_avg_pool2d` -- docs/kernels/FIXES.md §3.

        Measured: `F.adaptive_avg_pool2d`'s Python wrapper
        (`torch/nn/functional.py`) calls `_list_with_default(output_size,
        input.size())` and then hands the result straight to
        `torch._C._nn.adaptive_avg_pool2d` -- which is *this* binding, not
        `torch.ops.aten.adaptive_avg_pool2d.default`. `_list_with_default`
        (measured, `torch/nn/modules/utils.py`) returns a bare int
        **unchanged** when given one, so the int-to-pair normalisation upstream
        visibly performs is not in the Python layer either -- it is inside
        this C++ binding's own argument parser (`BroadcastingList2[int]`),
        which is a step the raw aten op's schema (`SymInt[2] output_size`,
        no default expansion) does not have.

        docs/kernels/FIXES.md §3 tried the expansion at the aten level instead and
        `tools/golden/compare.py`, which calls the raw aten op directly,
        caught it as a SILENT DIVERGENCE: upstream's raw aten op refuses a
        bare int and this shim's would not have. So the expansion belongs
        here, one door up, where upstream itself puts it -- the raw aten call
        this dispatches to is untouched and still refuses a scalar.

        `None` entries mean "keep this axis's current size", the same
        broadcasting rule `_list_with_default` applies to the tuple form
        (measured: `F.adaptive_avg_pool2d(x, (None, 4))` keeps the height).
        """
        if isinstance(output_size, int):
            output_size = [output_size, output_size]
        elif isinstance(output_size, (list, tuple)):
            sizes = input.shape[-2:]
            output_size = [
                v if v is not None else sizes[i] for i, v in enumerate(output_size)
            ]
        return dispatch("aten.adaptive_avg_pool2d.default", input, output_size)

    def hardtanh(input, min_val=-1.0, max_val=1.0):
        return dispatch("aten.hardtanh.default", input, min_val, max_val)

    def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
                   count_include_pad=True, divisor_override=None):
        """`torch._C._nn.avg_pool2d` -- `efficientnet`'s wall (docs/kernels/TAIL3.md
        §7, docs/bindings/BIND2.md item 1).

        `torch/nn/modules/pooling.py:779` (`nn.AvgPool2d.forward`) calls
        `F.avg_pool2d`, which binds straight to this name with the leaf
        schema's own seven arguments -- unlike `upsample_bilinear2d`/
        `upsample_bicubic2d` a few lines up, there is no separate `.vec`
        overload here to pick between (docs/bindings/BINDINGS.md's `upsample_nearest2d`
        warning: measure the signature against upstream, not against a
        neighbour). Measured: `torch._C._nn.avg_pool2d.__doc__` on 2.13.0
        gives exactly `avg_pool2d(input, kernel_size, stride=None, padding=0,
        ceil_mode=False, count_include_pad=True, divisor_override=None)`,
        which is `aten::avg_pool2d`'s own seven parameters in order.

        `aten.avg_pool2d.default` has been implemented and golden-compared
        since `sew_d` (docs/kernels/TAIL3.md §7) -- confirmed again here in
        `_aten_implemented()` before writing this binding, the check
        docs/bindings/BINDINGS.md's `mish` miss says to make. It already treats a
        `None`/empty `stride` as "the kernel size" itself (measured with a
        `TorchDispatchMode` logger: `F.avg_pool2d(x, 2)` fires
        `aten.avg_pool2d.default(x, [2, 2])` with no third argument at all),
        so `stride=None` is forwarded through rather than defaulted here.
        """
        return dispatch(
            "aten.avg_pool2d.default", input, kernel_size, stride, padding,
            ceil_mode, count_include_pad, divisor_override,
        )



    def one_hot(tensor, num_classes=-1):
        return dispatch("aten.one_hot.default", tensor, num_classes)


    def pad(input, pad, mode="constant", value=None):
        """`torch._C._nn.pad`, which `F.pad` calls after its own dispatch.

        `bert` is the caller (docs/architectures/ARCH20.md §2), through
        `modeling_utils.py:2701 _adjust_bias` -- so this runs while the model
        is being *built*, not during a forward.

        `torch/nn/functional.py:5823` hands all four arguments through, and
        upstream's binding then picks an aten op by `mode`. Measured with a
        `TorchDispatchMode` logger, `F.pad(x, (0, 3), "constant", 0)` produces
        exactly one record, `aten.constant_pad_nd.default`; `F.pad(x, (0, 3),
        "reflect")` and `"replicate"` likewise produce exactly one record
        each, `aten.reflection_pad{n}d.default`/`aten.replication_pad{n}d
        .default` with `n = len(pad) // 2` -- not derived from the input's
        rank, because upstream itself refuses combinations that do not line
        up with the rank *inside* the kernel, and deriving `n` from the rank
        here would accept combinations upstream rejects (docs/kernels/PAD.md §5).
        `circular` is a genuinely different kernel, a `new_empty`/`slice`/
        `copy_` composite built out of `cat` rather than one of these leaves,
        and is refused by name rather than approximated with one of the
        above -- a wrong padding is the kind of divergence that shows up as a
        slightly wrong number rather than as an error.

        `value=None` means zero, and that is upstream's own default rather
        than a choice here: `constant_pad_nd`'s schema is `Scalar value=0`,
        and `F.pad(x, (1, 1))` with no value pads with zeros (measured).
        """
        if mode != "constant":
            n = len(pad) // 2
            # Literal string constants, not an f-string: `tools/golden/
            # reach.py`'s `composite_keys` finds a composite's reach by
            # scanning `bootstrap.py` for `aten.<op>.<overload>` as an
            # `ast.Constant` -- an f-string is a `JoinedStr` and is invisible
            # to that scan, which would make these six look unreached the
            # moment this landed (measured: it did, before this rewrite).
            if mode == "reflect":
                op = {
                    1: "aten.reflection_pad1d.default",
                    2: "aten.reflection_pad2d.default",
                    3: "aten.reflection_pad3d.default",
                }.get(n)
            elif mode == "replicate":
                op = {
                    1: "aten.replication_pad1d.default",
                    2: "aten.replication_pad2d.default",
                    3: "aten.replication_pad3d.default",
                }.get(n)
            else:
                op = None
            if op is not None:
                return dispatch(op, input, list(pad))
            if mode in ("reflect", "replicate"):
                raise NotImplementedError(
                    f"not implemented in torch._C shim: torch._C._nn.pad(mode={mode!r}) "
                    f"with {len(pad)} pad values (n={n}) -- only 1D/2D/3D "
                    f"({{1,2,3}} pairs) have a kernel here"
                )
            raise NotImplementedError(
                f"not implemented in torch._C shim: torch._C._nn.pad(mode={mode!r}) "
                f"-- reflect and replicate are implemented; circular is a "
                f"new_empty/slice/copy_ composite upstream (docs/kernels/PAD.md §3)"
            )
        return dispatch(
            "aten.constant_pad_nd.default",
            input,
            list(pad),
            0 if value is None else value,
        )

    def softplus(input, beta=1, threshold=20):
        """`torch._C._nn.softplus`, which `F.softplus` binds directly to.

        `mamba`'s discretisation runs `F.softplus(dt)` on every step
        (docs/architectures/ARCH20.md §4). The kernel -- `aten.softplus.default` -- has been
        here and golden-compared since docs/kernels/OPS4.md; only the name was
        missing, which is the same shape of gap as `torch.stack`,
        `torch.exp` and `torch.conv1d` in this round.

        One record, measured, in all three spellings: `F.softplus(x)`,
        `F.softplus(x, 2.0, 10.0)` and `torch._C._nn.softplus(x, 1, 20)` each
        fire `aten.softplus.default` and nothing else. The defaults are
        upstream's schema defaults (`Scalar beta=1, Scalar threshold=20`) and
        are integers there, which matters to `arith`-style wrapped-number
        rules elsewhere and is why they are not written as `1.0`/`20.0`.
        """
        return dispatch("aten.softplus.default", input, beta, threshold)

    def upsample_bilinear2d(input, output_size, align_corners, scale_factors=None):
        """`torch._C._nn.upsample_bilinear2d` -- `zoedepth`'s wall.

        `torch/nn/functional.py:5248` (`F.interpolate`, `mode="bilinear"`)
        calls this with exactly this four-argument shape, which is the
        **`.vec`** schema: `output_size` may be `None`, and `scale_factors` is
        a list rather than the leaf's two separate `float?` arguments.

        `.vec` is `CompositeImplicitAutograd`. Measured with a
        `TorchDispatchMode` logger on 2.13.0, `F.interpolate(x,
        scale_factor=2, mode="bilinear")` on a `(1,1,2,3)` input emits exactly
        one record:

            aten.upsample_bilinear2d.default((1,1,2,3), [4, 6], False, 2.0, 2.0)

        -- a *concrete* output size, with the scale factors passed through
        beside it. So `.vec` belongs here, not in a table, for the reason
        `layer_norm` and `group_norm` give.

        **The scale factors are forwarded, not just used to size the output**,
        and that is the part a shortcut loses. `1/scale` and `in/out` coincide
        whenever `out == in * scale` exactly -- which is every case a
        `scale_factor=2` test produces -- and diverge as soon as the product is
        not integral. Dropping them here would make `scale_factor=1.5` sample a
        different grid, silently, with the right output shape.

        Upstream's `compute_output_size`: **exactly one** of `output_size` and
        `scale_factors` may be given -- not "output_size wins", which is what
        this was written as first and what a golden case caught. Upstream
        raises `Must specify exactly one of output_size and scale_factors`, and
        so does this. When it is `scale_factors`, each output extent is
        `floor(input.size(i+2) * factor)`.
        """
        sizes = list(input.shape)
        if output_size is not None and scale_factors is not None:
            raise RuntimeError(
                "Must specify exactly one of output_size and scale_factors"
            )
        if output_size is not None:
            osize = [int(v) for v in output_size]
            scale_h = scale_w = None
        else:
            if scale_factors is None:
                raise RuntimeError(
                    "Must specify exactly one of output_size and scale_factors"
                )
            factors = list(scale_factors)
            osize = [int(sizes[i + 2] * factors[i]) for i in range(len(factors))]
            scale_h = float(factors[0]) if len(factors) > 0 else None
            scale_w = float(factors[1]) if len(factors) > 1 else None
        return dispatch(
            "aten.upsample_bilinear2d.default", input, osize, align_corners,
            scale_h, scale_w,
        )

    _LEAF_SENTINEL = object()

    def upsample_bicubic2d(
        input, output_size, align_corners, scale_factors=None,
        scales_w=_LEAF_SENTINEL,
    ):
        """`torch._C._nn.upsample_bicubic2d` -- `yolos`' wall.

        `torch/nn/functional.py:5286` (`F.interpolate`, `mode="bicubic"`,
        `antialias=False`) calls this with exactly this four-argument shape,
        which is the **`.vec`** schema: `output_size` may be `None` and
        `scale_factors` is a list, where the leaf takes two separate `float?`
        arguments. `.vec` is `CompositeImplicitAutograd`, so it belongs here
        rather than in a table, for the reason `upsample_bilinear2d` gives
        directly above -- and the scale factors are **forwarded**, not merely
        used to size the output, because `1/scale` and `in/out` are different
        grids as soon as the product is not integral.

        `antialias=True` reaches `_upsample_bicubic2d_aa`, a different op with
        no kernel here; it stays a raising stub naming itself rather than being
        silently aliased to this one, which would return a plausible image at
        the wrong sharpness.

        The **five-argument leaf spelling** (`scales_h`, `scales_w` as two
        separate floats) is accepted too, because upstream's binding is
        overloaded and accepts it -- measured. A fifth argument is the only
        thing that distinguishes the two shapes, so it is the discriminator.
        """
        _leaf = scales_w is not _LEAF_SENTINEL
        if _leaf and output_size is None:
            raise RuntimeError(
                "It is expected output_size equals to 2, but got size 0"
            )
        sizes = list(input.shape)
        if _leaf:
            # The five-argument leaf spelling. Upstream's binding is
            # overloaded and accepts both; measured, `torch._C._nn.
            # upsample_bicubic2d(x, [1, 4], False, None, 1.5)` is a real call
            # and answers on the `1/scale` grid, so refusing it here would be
            # a surface this shim invented a hole in.
            return dispatch(
                "aten.upsample_bicubic2d.default", input,
                [int(v) for v in output_size], align_corners,
                scale_factors, scales_w,
            )
        if output_size is not None and scale_factors is not None:
            raise RuntimeError(
                "Must specify exactly one of output_size and scale_factors"
            )
        if output_size is not None:
            osize = [int(v) for v in output_size]
            scale_h = scale_w = None
        else:
            if scale_factors is None:
                raise RuntimeError(
                    "Must specify exactly one of output_size and scale_factors"
                )
            factors = list(scale_factors)
            osize = [int(sizes[i + 2] * factors[i]) for i in range(len(factors))]
            scale_h = float(factors[0]) if len(factors) > 0 else None
            scale_w = float(factors[1]) if len(factors) > 1 else None
        return dispatch(
            "aten.upsample_bicubic2d.default", input, osize, align_corners,
            scale_h, scale_w,
        )

    def upsample_nearest2d(
        input, output_size, scale_factors=None, scales_w=_LEAF_SENTINEL,
    ):
        """`torch._C._nn.upsample_nearest2d` -- `vilt`'s wall (docs/kernels/TAIL1.md §4).

        `torch/nn/functional.py:5188` (`F.interpolate`, `mode="nearest"`) calls
        this with **three** arguments -- `(input, output_size, scale_factors)`,
        the `.vec` schema -- where bilinear's `.vec` has four because nearest
        has no `align_corners` at all. The leaf schema is
        `(self, output_size, scales_h, scales_w)`, so **a fourth argument is
        the only thing that distinguishes the two shapes** and it is the
        discriminator, the same way it is for `upsample_bicubic2d` one
        argument further along.

        Measured on torch 2.13.0 rather than inferred from bilinear, because
        the argument counts differ:

            F.interpolate((1,1,3,5), scale_factor=1.5, mode="nearest")
                -> aten.upsample_nearest2d.default((1,1,3,5), [4, 7], 1.5, 1.5)
            F.interpolate((1,1,3,5), size=(7,11), mode="nearest")
                -> aten.upsample_nearest2d.default((1,1,3,5), [7, 11])
            torch._C._nn.upsample_nearest2d(x, [4,7], [1.5,1.5])
                -> RuntimeError: Must specify exactly one of output_size and
                   scale_factors

        -- so three positional arguments is `.vec` (that last line is upstream
        refusing to be given both), and `output_size` and `scale_factors` are
        mutually exclusive there exactly as they are for bilinear.

        **The scale factors are forwarded, not merely used to size the
        output.** `1/scale` and `in/out` coincide whenever the product is
        integral, which is every case a `scale_factor=2` test produces, and
        diverge as soon as it is not -- dropping them would sample a different
        grid, silently, at the right output shape. They happen to agree on the
        `3x5 -> 4x7` case above (checked), which is precisely why a test built
        only from that case would not have noticed.
        """
        _leaf = scales_w is not _LEAF_SENTINEL
        sizes = list(input.shape)
        if _leaf:
            if output_size is None:
                raise RuntimeError(
                    "It is expected output_size equals to 2, but got size 0"
                )
            return dispatch(
                "aten.upsample_nearest2d.default", input,
                [int(v) for v in output_size], scale_factors, scales_w,
            )
        if output_size is not None and scale_factors is not None:
            raise RuntimeError(
                "Must specify exactly one of output_size and scale_factors"
            )
        if output_size is not None:
            osize = [int(v) for v in output_size]
            scale_h = scale_w = None
        else:
            if scale_factors is None:
                raise RuntimeError(
                    "Must specify exactly one of output_size and scale_factors"
                )
            factors = list(scale_factors)
            osize = [int(sizes[i + 2] * factors[i]) for i in range(len(factors))]
            scale_h = float(factors[0]) if len(factors) > 0 else None
            scale_w = float(factors[1]) if len(factors) > 1 else None
        return dispatch(
            "aten.upsample_nearest2d.default", input, osize, scale_h, scale_w,
        )

    def _int_pair(value):
        """`int[2]` the way upstream's argument parser takes it.

        `F.unfold(x, kernel_size=2)` sends `_pair(2)` -- already a tuple -- but
        `torch._C._nn.im2col(x, 2, 1, 0, 1)` with bare ints computes upstream
        (measured on 2.13.0), because an `int[N]` schema broadcasts a scalar.
        The Rust `pair_arg` broadcasts a length-1 list but reads its argument
        through `shape_arg`, which wants a sequence, so the scalar case is
        normalised here rather than there -- `aten.rs` is not this round's file
        and the fix belongs on the side that knows it is a Python calling
        convention.
        """
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        return [int(value), int(value)]

    def im2col(input, kernel_size, dilation, padding, stride):
        """`torch._C._nn.im2col` -- **`llama4`'s vision tower** (docs/architectures/VOICE3.md,
        docs/kernels/COMPLEX2.md §8), and `F.unfold` *is* this binding.

        `torch/nn/functional.py`'s `unfold` ends in

            return torch._C._nn.im2col(
                input, _pair(kernel_size), _pair(dilation),
                _pair(padding), _pair(stride))

        -- a straight forward with no reshaping and no branch, so there is no
        `.vec`/leaf discrimination to do here and nothing this binding can get
        wrong that the kernel does not already own. All four `int[2]` arguments
        are **required**: `torch._C._nn.im2col(x, [2,2])` is
        `TypeError: im2col() missing 3 required positional argument` upstream,
        so no defaults are invented here either.

        NOT an `overloads.json` or `methods.json` row: `torch.im2col` and
        `Tensor.im2col` do not exist on 2.13.0 (checked, and `reach_allow.json`
        staked the claim on it), so a table row would invent a door upstream
        lacks -- `upsample_nearest2d`'s and `reflection_pad1d`'s reasoning.
        """
        return dispatch(
            "aten.im2col.default", input, _int_pair(kernel_size),
            _int_pair(dilation), _int_pair(padding), _int_pair(stride),
        )

    def col2im(input, output_size, kernel_size, dilation, padding, stride):
        """`torch._C._nn.col2im` -- **`f5-tts`'s wall** (docs/architectures/VOICE.md rank 15),
        and `F.fold` *is* this binding, `im2col`'s shape one argument longer.

        `torch/nn/functional.py`'s `fold` forwards
        `(input, _pair(output_size), _pair(kernel_size), _pair(dilation),
        _pair(padding), _pair(stride))`, and the same three arguments are
        required-with-no-default that `im2col`'s are.

        **It is not `im2col`'s inverse** -- overlapping windows are *summed*
        here -- but that is the kernel's property, restated in `aten.rs`, and
        the round trip is a case in `test_voice3.py` for exactly that reason.
        The binding adds nothing to it; what it adds is that `F.fold` reaches
        it at all.
        """
        return dispatch(
            "aten.col2im.default", input, _int_pair(output_size),
            _int_pair(kernel_size), _int_pair(dilation), _int_pair(padding),
            _int_pair(stride),
        )

    def upsample_nearest1d(input, output_size, scale_factors=None):
        """`torch._C._nn.upsample_nearest1d` -- `bigvgan`/`voice`
        (docs/architectures/VOICE.md rank 14), reached as `F.interpolate(x, ..., mode=
        "nearest")` on a **3-D** input.

        `torch/nn/functional.py`'s `interpolate` calls
        `torch._C._nn.upsample_nearest1d(input, output_size, scale_factors)`,
        the same three-argument `.vec` shape the 2-D op has one line below it.

        **The discriminator is the third argument's TYPE, not the arity**, and
        this is where the 1-D op differs from `upsample_nearest2d` above. The
        2-D leaf schema is `(self, output_size, scales_h, scales_w)`, so a
        *fourth* argument tells leaf from `.vec` there. The 1-D leaf schema is
        `(self, output_size, scales)` -- **three arguments, the same count as
        `.vec`** -- so a sentinel on a fourth parameter cannot work. Measured on
        2.13.0, all four accepted upstream:

            _nn.upsample_nearest1d(x, [7], None)    -> .default(x, [7])
            _nn.upsample_nearest1d(x, None, [1.5])  -> .default(x, [7], 1.5)
            _nn.upsample_nearest1d(x, [7], 1.5)     -> .default(x, [7], 1.5)
            _nn.upsample_nearest1d(x, [7])          -> .default(x, [7])

        A *sequence* third argument is `.vec`'s `scale_factors`; a float is the
        leaf's `scales`. Taking arity as the discriminator would read the third
        line as `.vec` and hand a bare float to `[0]`.

        `output_size` and `scale_factors` are mutually exclusive in the `.vec`
        shape and **both** wrong ways refuse with the same words upstream uses
        -- giving both, and giving neither (measured; the second is not
        guessable from the first).

        **The scale factor is forwarded, not merely used to size the output**,
        for `upsample_nearest2d`'s reason: `1/scale` and `in/out` coincide only
        when the product is integral. `(1,2,5)` at `scale=1.5` gives `[7]`, and
        `5/7 != 1/1.5`, so the two readings pick different source columns --
        which is what the `un1d_scale_*` cases and `un1d_via_F_interpolate`
        compare element-wise against a live upstream.
        """
        if isinstance(scale_factors, (list, tuple)) or scale_factors is None:
            if (output_size is None) == (scale_factors is None):
                raise RuntimeError(
                    "Must specify exactly one of output_size and scale_factors"
                )
            if output_size is not None:
                return dispatch(
                    "aten.upsample_nearest1d.default", input,
                    [int(v) for v in output_size], None,
                )
            factor = float(scale_factors[0])
            osize = [int(list(input.shape)[2] * factor)]
            return dispatch(
                "aten.upsample_nearest1d.default", input, osize, factor,
            )
        if output_size is None:
            raise RuntimeError(
                "It is expected output_size equals to 1, but got size 0"
            )
        return dispatch(
            "aten.upsample_nearest1d.default", input,
            [int(v) for v in output_size], float(scale_factors),
        )

    def upsample_linear1d(input, output_size, align_corners, scale_factors=None):
        """`torch._C._nn.upsample_linear1d` -- `sam_vision_model` /
        `sam_hq_vision_model`'s wall (docs/architectures/ARCH200.md §2, docs/kernels/RNN.md §3).
        `F.interpolate(x_3d, mode="linear")` calls this and only this; the
        kernel (`aten.upsample_linear1d.default`) has been implemented,
        golden-compared and bit-compared against upstream since docs/kernels/RNN.md,
        which is what this binding was checked against before being written
        (docs/bindings/BINDINGS.md's `mish` is what happens when that check is
        skipped).

        `torch/nn/functional.py`'s `interpolate` calls this with exactly
        `(input, output_size, align_corners, scale_factors)`, which is the
        **`.vec`** schema (`aten::upsample_linear1d.vec(Tensor input,
        SymInt[]? output_size, bool align_corners, float[]? scale_factors)`).
        `.vec` is `CompositeImplicitAutograd` and lowers to the leaf,
        `aten::upsample_linear1d(Tensor self, SymInt[1] output_size, bool
        align_corners, float? scales=None)`.

        **The discriminator is the fourth argument's TYPE, not the arity**,
        docs/bindings/BIND3.md §3.1's exact trap for `upsample_nearest1d` one line
        above this in `torch._C._nn`: both schemas take four arguments here
        (`.vec`'s `output_size` may be `None`; the leaf's may not), so a
        sequence fourth argument is `.vec`'s `scale_factors` and a float is
        the leaf's `scales`. Measured on 2.13.0:

            _nn.upsample_linear1d(x, [9], False, None)     -> .default(x, [9], False)
            _nn.upsample_linear1d(x, None, True, [1.5])    -> .default(x, [9], True, 1.5)
            _nn.upsample_linear1d(x, [9], True, 1.5)       -> .default(x, [9], True, 1.5)
            _nn.upsample_linear1d(x, [9], False)           -> .default(x, [9], False)

        `output_size` and `scale_factors` are mutually exclusive in the
        `.vec` shape; giving both or neither refuses with upstream's own
        message, `Must specify exactly one of output_size and
        scale_factors`, measured directly rather than assumed from
        `upsample_nearest1d`/`upsample_bilinear2d`'s copy of it.

        **The scale factor is forwarded, not merely used to size the
        output**, for the same reason `upsample_bilinear2d` and
        `upsample_nearest1d` forward theirs: `1/scale` and `in/out` coincide
        only when the product is integral.
        """
        if isinstance(scale_factors, (list, tuple)) or scale_factors is None:
            if (output_size is None) == (scale_factors is None):
                raise RuntimeError(
                    "Must specify exactly one of output_size and scale_factors"
                )
            if output_size is not None:
                return dispatch(
                    "aten.upsample_linear1d.default", input,
                    [int(v) for v in output_size], align_corners,
                )
            factor = float(scale_factors[0])
            osize = [int(list(input.shape)[2] * factor)]
            return dispatch(
                "aten.upsample_linear1d.default", input, osize,
                align_corners, factor,
            )
        if output_size is None:
            raise RuntimeError(
                "It is expected output_size equals to 1, but got size 0"
            )
        return dispatch(
            "aten.upsample_linear1d.default", input,
            [int(v) for v in output_size], align_corners, float(scale_factors),
        )

    def leaky_relu(input, negative_slope=0.01):
        """`torch._C._nn.leaky_relu` -- `vits`' wall after the `IntTensor`
        constructor.

        `torch/nn/functional.py:1969` is `torch._C._nn.leaky_relu(input,
        negative_slope)`, so `F.leaky_relu` *is* this binding rather than a
        wrapper around it -- the same relationship `gelu` has, and the reason
        it is a `_nn` name rather than an `overloads.json` entry. There is no
        `torch.leaky_relu` and no `Tensor.leaky_relu` upstream
        (`hasattr` is False for both on 2.13.0), so adding either would invent
        a surface.

        The in-place sibling `leaky_relu_` is deliberately absent: `F.leaky_relu`
        reaches it only when called with `inplace=True`, `vits` does not, and
        `aten.leaky_relu_.default` has no kernel here -- so the name stays a
        raising stub that says which op it needed.
        """
        return dispatch("aten.leaky_relu.default", input, negative_slope)

    def glu(input, dim=-1):
        """`torch._C._nn.glu` -- `F.glu(x, dim)` *is* this binding, not a
        wrapper around it, the same relationship `gelu`/`leaky_relu` have to
        `_nn` (`torch/nn/functional.py` binds `F.glu` straight to
        `torch._C._nn.glu`). There is no `torch.glu` or `Tensor.glu` upstream
        (`hasattr` is False for both on 2.13.0), so no `overloads.json` or
        `methods.json` entry is wanted -- same reasoning as `silu`/`gelu`.

        docs/architectures/ARCH100.md: blocks seven ASR encoders (`parakeet` x3, `lasr` x2,
        `cohere_asr`, `parakeet_tdt`), all of which gate the trailing channel
        axis after a conv or linear -- which is why the default below is
        `dim=-1` and not `dim=0`: measured against upstream 2.13.0, a bare
        `F.glu(x)` with no `dim` halves the *last* axis, not the first.

        Kernel behaviour (measured, not re-derived here): an odd size at
        `dim` raises `RuntimeError("Halving dimension must be even, but "
        "dimension {d} is size {n}")` rather than floor-dividing, and it is
        float-only -- `NotImplementedError('"glu_cpu" not implemented for '
        "'Long'")` on an integer input, the same refusal shape `silu` has
        rather than `sigmoid`'s promotion. Both are the kernel's own rules
        (`aten.glu.default`); this binding only supplies the default `dim`
        and the pass-through.
        """
        return dispatch("aten.glu.default", input, dim)

    # -- the cross-entropy composites (docs/training/LOSS.md) ---------------------
    #
    # Three names, all `CompositeImplicitAutograd`, and **none of them appears
    # in a dispatch trace** -- which is why docs/training/AUTOGRAD.md §5.3's op scan
    # named two missing kernels and the path needed six more names. A
    # `TorchDispatchMode` sits below the composite layer, so it records the
    # leaves a call lands on and never the names a caller uses to get there.
    #
    # `transformers` reaches a loss through the last of them:
    # `ForCausalLMLoss` -> `fixed_cross_entropy` -> `F.cross_entropy` ->
    # `torch._C._nn.cross_entropy_loss`.
    #
    # `nll_loss_forward` is deliberately NOT installed here: upstream has no
    # `torch._C._nn.nll_loss_forward` (`hasattr` is False on 2.13.0), and
    # inventing it would be a surface upstream does not have. The kernel behind
    # it is reachable as `aten.nll_loss_forward.default`, which is where the
    # golden cases compare it.

    def nll_loss(self, target, weight=None, reduction=1, ignore_index=-100):
        """`at::native::nll_loss` -- element 0 of the forward's pair.

        The pair's *second* element, `total_weight`, is dropped here exactly as
        upstream drops it. It is not dead: `nll_loss_backward` takes it as an
        argument, which is why the kernel returns it and why the golden cases
        check it (docs/training/LOSS.md §3.1).
        """
        return dispatch(
            "aten.nll_loss_forward.default", self, target, weight, reduction, ignore_index
        )[0]

    def nll_loss_nd(self, target, weight=None, reduction=1, ignore_index=-100):
        """`at::native::nll_loss_nd_symint`, transcribed.

        Only the 1-D/2-D arm is written out. Upstream's 4-D arm and its
        `dim == 3 or dim > 4` reshape both land on `nll_loss2d`, which this
        shim has no kernel for; they refuse by that name rather than being
        approximated, because `nll_loss2d` is a different reduction (it sums
        over a spatial extent) and substituting `nll_loss` would be silently
        wrong rather than slow.
        """
        if self.dim() < 1:
            raise ValueError(f"Expected 1 or more dimensions (got {self.dim()})")
        if self.dim() != 1 and self.shape[0] != target.shape[0]:
            raise ValueError(
                f"Expected input batch_size ({self.shape[0]}) to match "
                f"target batch_size ({target.shape[0]})."
            )
        if self.dim() in (1, 2):
            return nll_loss(self, target, weight, reduction, ignore_index)
        raise NotImplementedError(
            f"not implemented in torch._C shim: torch._C._nn.nll_loss_nd(input.dim()="
            f"{self.dim()}) -- upstream routes every rank but 1 and 2 through "
            "aten.nll_loss2d_forward.default, which has no kernel here. That op "
            "reduces over a spatial extent, so nll_loss is not a slower road to "
            "the same answer"
        )

    def cross_entropy_loss(
        self, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
    ):
        """`at::native::cross_entropy_loss_symint`, whose third branch is the
        one a causal-LM loss takes:

            nll_loss_nd(log_softmax(self, class_dim, self.dtype), target, ...)

        `class_dim` is `0` for a 1-D input and `1` otherwise -- so for the
        `(N, V)` shape `ForCausalLMLoss` flattens to, the log-softmax is over
        the vocabulary, which is dim 1 and also the trailing axis. That matters
        more than it looks: it means a real cross-entropy takes `_log_softmax`'s
        **last-dim** kernel, the narrowing one (docs/training/LOSS.md §2.1).

        The `dtype` argument to `log_softmax` is `self.scalar_type()`, i.e. a
        no-op cast. It is passed anyway, because that is upstream's call and
        `Tensor.log_softmax` is measured to emit no `_to_copy` for a no-op
        dtype -- so following upstream costs nothing here.

        The other two branches refuse, by name and with what each would need:
        upstream picks them on `self.sizes() == target.sizes()` (soft/probability
        targets) and on `label_smoothing > 0`.
        """
        if tuple(self.shape) == tuple(target.shape):
            raise NotImplementedError(
                "not implemented in torch._C shim: torch._C._nn.cross_entropy_loss("
                "probability targets) -- input and target have the same shape, which "
                "is how upstream selects cross_entropy_loss_prob_target. That branch "
                "is a different formula (-(log_softmax(self) * target).sum()), not a "
                "different spelling of this one"
            )
        if label_smoothing > 0.0:
            raise NotImplementedError(
                f"not implemented in torch._C shim: torch._C._nn.cross_entropy_loss("
                f"label_smoothing={label_smoothing}) -- upstream selects "
                "cross_entropy_loss_label_smoothing, which blends a uniform term into "
                "the target and is not expressible as nll_loss over a log_softmax. "
                "label_smoothing=0.0, the default and what F.cross_entropy passes "
                "unless asked otherwise, takes the branch this shim implements"
            )
        class_dim = 0 if self.dim() == 1 else 1
        return nll_loss_nd(
            module.TensorBase.log_softmax(self, class_dim, self.dtype),
            target, weight, reduction, ignore_index,
        )

    for fn, name in (
        (linear, "linear"),
        (silu, "silu"),
        (gelu, "gelu"),
        (scaled_dot_product_attention, "scaled_dot_product_attention"),
        (pad, "pad"),
        (softplus, "softplus"),
        (upsample_bilinear2d, "upsample_bilinear2d"),
        (upsample_bicubic2d, "upsample_bicubic2d"),
        (upsample_nearest2d, "upsample_nearest2d"),
        (upsample_nearest1d, "upsample_nearest1d"),
        (upsample_linear1d, "upsample_linear1d"),
        (im2col, "im2col"),
        (col2im, "col2im"),
        (leaky_relu, "leaky_relu"),
        (adaptive_avg_pool2d, "adaptive_avg_pool2d"),
        (avg_pool2d, "avg_pool2d"),
        (hardtanh, "hardtanh"),
        (one_hot, "one_hot"),
        (nll_loss, "nll_loss"),
        (nll_loss_nd, "nll_loss_nd"),
        (cross_entropy_loss, "cross_entropy_loss"),
        (glu, "glu"),
    ):
        fn.__name__ = fn.__qualname__ = name
        fn.__module__ = "torch._C._nn"
        setattr(nn, name, fn)

    # Readable for the same reason as `_shim_overloads`: which of `_nn`'s 70
    # names does something should be answerable by asking.
    module._shim_nn_implemented = [
        "adaptive_avg_pool2d", "avg_pool2d", "col2im", "cross_entropy_loss",
        "gelu", "glu", "hardtanh", "im2col", "leaky_relu", "linear", "nll_loss",
        "nll_loss_nd", "one_hot", "pad", "scaled_dot_product_attention", "silu",
        "softplus", "upsample_bicubic2d", "upsample_bilinear2d",
        "upsample_linear1d", "upsample_nearest1d", "upsample_nearest2d",
    ]


def _outer_impl(input, vec2):
    """`aten::outer(Tensor self, Tensor vec2) -> Tensor`, shared by the free
    function and the `TensorBase` member so the rank check exists once.

    A **spelling, not a kernel**, and checked as such: `aten::outer` is
    `CompositeImplicitAutograd` and a `TorchDispatchMode` trace of
    `torch.outer(a, b)` fires exactly two ops -- `aten.view.default` and
    `aten.mul.Tensor` -- both of which this shim already implements and already
    golden-compares. `expand_as`'s shape of fix (docs/kernels/KERNELS26.md §6.3).

    Written through `.reshape` and `*` rather than through `dispatch` directly,
    so it inherits the broadcasting and type promotion the `mul` path already
    has: `outer(int64, int64)` is `int64` and `outer(int64, float32)` is
    `float32`, both measured, neither restated here.

    **Both arguments must be 1-D.** Upstream refuses a 0-D or 2-D argument and
    names which one it was; without the check a 2-D `self` would broadcast into
    a silently wrong shape instead of raising.
    """
    for name, t in (("self", input), ("vec2", vec2)):
        rank = len(t.shape)
        if rank != 1:
            raise RuntimeError(
                f"outer: Expected 1-D argument {name}, but got {rank}-D"
            )
    return input.reshape([input.shape[0], 1]) * vec2


def _install_composites(module, varfns, dispatch) -> None:
    """`torch.dropout`, the other `CompositeImplicitAutograd` on the path.

    `torch/nn/functional.py:1491` is `_VF.dropout(input, p, training)`, so
    every `nn.Dropout` in a model reaches `torch._C._VariableFunctions.dropout`
    -- 10 times in the measured Llama `generate` (docs/bindings/NN_SURFACE.md §3).

    It is not in `overloads.json` because it is not an overload set in the
    sense that table means. `aten::dropout`'s body short-circuits before the
    dispatcher: `if (p == 0 || !train || input.numel() == 0) return input;`.
    Measured, and the measurement is the point -- `F.dropout(x, 0.0, False)`
    produces *no* aten record at all on upstream torch. An eval-mode model
    therefore needs no dropout kernel, and routing this name through the
    overload table would have invented a requirement upstream does not have.

    `train=True` is the training half, and docs/training/TRAIN.md is where it was
    measured. There is **no `aten.dropout.default` kernel to write**: with a
    `TorchDispatchMode` logger on torch 2.13.0, a training-mode `nn.Dropout`
    forward fires

        empty_like.default, bernoulli_.float, div_.Scalar, mul.Tensor

    and `aten.dropout.default` never appears -- it is
    `CompositeImplicitAutograd`, the same shape as `aten::linear` and
    `aten::layer_norm`, which `_install_nn` and `layer_norm` below answer by
    decomposition for the same reason. It is also *not* `native_dropout`;
    that is the functionalised spelling, and eager CPU never reaches it
    (`is_fused_kernel_acceptable` wants CUDA/XPU/lazy).

    So the body below is `at::native::_dropout_impl` transcribed, and three of
    its details are measured rather than assumed (docs/training/TRAIN.md §1):

      * the range check runs **before** the short-circuit, so
        `torch.dropout(x, 1.5, False)` raises rather than returning `x`;
      * the identity return is the identity *object* -- `torch.dropout(x, 0.,
        False) is x` is True upstream;
      * `p == 1` is `input * zeros({})`, **not** `zeros_like(input)`. Signed
        zero survives it and `±inf` becomes `nan`, neither of which a
        `zeros_like` gives, and no `(y == 0).all()` test can tell the two
        apart.
    """

    tensorbase = module.TensorBase

    def _dropout_impl(input, p, train, inplace):
        # `TORCH_CHECK(p >= 0 && p <= 1, ...)`, spelled so that `nan` is
        # refused rather than falling through both comparisons.
        if not (p >= 0 and p <= 1):
            raise RuntimeError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        if p == 0 or not train or input.numel() == 0:
            return input
        mul = "aten.mul_.Tensor" if inplace else "aten.mul.Tensor"
        if p == 1:
            zero = dispatch(
                "aten.zeros.default", [], dtype=input.dtype, device=input.device
            )
            return dispatch(mul, input, zero)
        if not inplace and module._capture_active():
            # **The functionalised route, taken only while capture records.**
            #
            # `bernoulli_` and `div_` both write in place, and capture refuses
            # mutation so that a trace stays single-assignment
            # (docs/graph/CAPTURE.md). So the eager decomposition below cannot be
            # recorded at all, and `.train()` dropout was uncapturable --
            # `gpt2`, `bert`, `opt` and `gpt_bigcode`, docs/training/TRAIN.md's own four
            # (docs/training/AUTOGRAD.md §6.5 named it as one of two reasons to want
            # this op). `aten.native_dropout.default` is upstream's own
            # out-of-place spelling of the same thing, is one node rather than
            # four, and hands back the mask a backward will need.
            #
            # **This is following upstream, not routing around it.**
            # `native_dropout` is precisely what functionalisation rewrites
            # eager dropout to; taking it inside a capture region is the same
            # substitution upstream makes for the same reason. Outside one,
            # eager keeps the eager path, because `is_fused_kernel_acceptable`
            # wants CUDA/XPU/lazy and CPU eager never reaches `native_dropout`
            # upstream either.
            #
            # **It is not bit-identical to the branch below, and it must not be
            # claimed to be** (docs/training/LOSS.md §7): the two spellings put the
            # scale in different places -- `_dropout_impl` divides the *mask*
            # by `1 - p`, `native_dropout` multiplies the *output* by a
            # `1/(1-p)` narrowed to the tensor's dtype. The masks agree draw
            # for draw (both consume `numel` 64-bit draws through
            # `bernoulli_`'s stream, measured across all four dtypes) and the
            # survivors can differ by an ULP in `bfloat16`/`float16`. That is
            # upstream's difference, reproduced, not this shim's.
            return dispatch("aten.native_dropout.default", input, p, train)[0]
        noise = dispatch("aten.empty_like.default", input)
        # `1 - p` is the *keep* probability, and it is computed once and used
        # twice: upstream draws with it and then divides by it, so any rounding
        # in it cancels between the mask and the scale. Computing `1 - p` twice
        # would be the same value here; computing the scale as `1 / (1 - p)`
        # and multiplying would not be -- `div_` rounds in the tensor's dtype.
        keep = 1 - p
        dispatch("aten.bernoulli_.float", noise, keep)
        dispatch("aten.div_.Scalar", noise, keep)
        return dispatch(mul, input, noise)

    def dropout(input, p, train):
        return _dropout_impl(input, p, train, inplace=False)

    def dropout_(input, p, train):
        return _dropout_impl(input, p, train, inplace=True)

    def native_dropout(input, p, train):
        """`torch.native_dropout` -- `hasattr(torch, 'native_dropout')` is True
        on 2.13.0, so the name exists upstream and is spelled here.

        Not an `overloads.json` entry even though `aten::native_dropout` is a
        dispatched leaf: the free function takes exactly the leaf's arguments
        and there is nothing to resolve, so a one-signature table entry would
        add a parser round trip and no information. `torch._log_softmax` is in
        the table because its *name* differs from the public one; this name
        does not.
        """
        return dispatch("aten.native_dropout.default", input, p, train)

    for fn, name in ((dropout, "dropout"), (dropout_, "dropout_"),
                     (native_dropout, "native_dropout")):
        fn.__name__ = fn.__qualname__ = name
        fn.__module__ = "torch._C"
        setattr(varfns, name, fn)

    def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5,
                    cudnn_enable=True):
        """`torch.layer_norm` / `F.layer_norm` / `nn.LayerNorm`'s forward.

        `aten::layer_norm` is `CompositeImplicitAutograd` -- measured with a
        `TorchDispatchMode` logger on torch 2.13.0, `F.layer_norm(...)`,
        `torch.layer_norm(...)` and a `nn.LayerNorm` forward all bottom out at
        `aten.native_layer_norm.default` only; `aten.layer_norm.default` never
        fires. Upstream's C++ body is exactly `std::get<0>(native_layer_norm(
        input, normalized_shape, weight, bias, eps))` -- the mean/rstd that
        `native_layer_norm` also returns are what backward would need, and
        there is no backward here, so they are computed and discarded.

        `cudnn_enable` is accepted and ignored, matching what happens on a CPU
        tensor upstream (`native_layer_norm`'s CPU kernel does not consult
        it). Overloads.json does not carry `layer_norm` for the reason
        `softmax` is not in `methods.json`: the parser-level key would be
        `aten.layer_norm.default`, which is never the key upstream's own
        dispatcher answers to, so the table would name a work item that is
        not the real one.
        """
        result = dispatch(
            "aten.native_layer_norm.default", input, normalized_shape,
            weight, bias, eps,
        )
        return result[0]

    layer_norm.__name__ = layer_norm.__qualname__ = "layer_norm"
    layer_norm.__module__ = "torch._C"
    setattr(varfns, "layer_norm", layer_norm)

    def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5,
                    cudnn_enabled=True):
        """`torch.group_norm` / `F.group_norm` / `nn.GroupNorm`'s forward.

        `sew_d`'s wall (docs/kernels/KERNELS26.md §19), and `layer_norm`'s shape
        exactly: `aten::group_norm` is `CompositeImplicitAutograd`, so a
        `TorchDispatchMode` logger on torch 2.13.0 sees all three of
        `torch.group_norm(...)`, `F.group_norm(...)` and an `nn.GroupNorm`
        forward emit `aten.native_group_norm.default` and nothing else --
        `aten.group_norm.default` never fires. So this is a composite here and
        not an `overloads.json` entry, for the reason `layer_norm`'s note above
        gives: the table would name a key upstream's own dispatcher never
        answers to.

        **The three arguments `native_group_norm` needs and this signature does
        not carry are derived here**, which is most of the body:

            N    = input.shape[0]
            C    = input.shape[1]
            HxW  = the product of everything after that

        `N * C * HxW == input.numel()` by construction, so the element-count
        check the kernel makes cannot fire through this door -- it is there for
        a caller that spells the leaf op directly, which upstream's schema
        allows too.

        Rank is checked here rather than in the kernel because it is *this*
        function's rule: upstream's `F.group_norm` raises "Expected at least 2
        dimensions" before calling anything, and a rank-1 input would otherwise
        reach the leaf with `HxW = 1` and normalise something.

        `cudnn_enabled` is accepted and ignored, exactly as `layer_norm`'s
        `cudnn_enable` is: `native_group_norm`'s CPU kernel does not consult
        it. The mean and rstd the leaf also returns are what backward would
        need; there is no backward here, so they are computed and discarded --
        which is precisely why they need their own cases in
        `tools/golden/cases.py` rather than being covered by a forward.
        """
        shape = list(input.shape)
        if len(shape) < 2:
            raise RuntimeError(
                "Expected at least 2 dimensions for input tensor but received "
                f"{len(shape)}"
            )
        hxw = 1
        for extent in shape[2:]:
            hxw *= extent
        result = dispatch(
            "aten.native_group_norm.default", input, weight, bias,
            shape[0], shape[1], hxw, num_groups, eps,
        )
        return result[0]

    group_norm.__name__ = group_norm.__qualname__ = "group_norm"
    group_norm.__module__ = "torch._C"
    setattr(varfns, "group_norm", group_norm)

    def isfinite(input):
        """`torch.isfinite` -- the third `CompositeImplicitAutograd` here.

        `torch/_tensor_str.py:155` is the caller: the formatter selects the
        finite non-zero elements before it decides column width, precision and
        whether to switch to scientific notation, so `print(tensor)` cannot
        take a step without this.

        It is a composite rather than an `overloads.json` entry for exactly
        the reason the note above `layer_norm` gives, and this time the
        measurement is unambiguous. A `TorchDispatchMode` logger on torch
        2.13.0 shows `torch.isfinite` bottoming out as:

            floating input:  eq.Tensor -> abs.default -> ne.Scalar -> mul.Tensor
            integral input:  ones_like.default

        `aten.isfinite.default` never fires. Naming it in the overload table
        would have added a seventh kernel to the `repr` work item and it would
        have been a kernel upstream does not have either.

        The two branches are upstream's own C++ body, transcribed:
        `(self == self) * (self.abs() != inf)` for floats -- `self == self` is
        the NaN test and the `abs() != inf` is the infinity test -- and an
        all-true tensor for anything integral, because no integral value is
        infinite or NaN.

        The `*` really is a multiply on two `bool` tensors, not a rename of
        `&`. `mul.Tensor` is what upstream dispatches and what this calls, so
        a gap would name the key upstream names. Complex dtypes take
        upstream's third branch, which this shim cannot reach: there is no
        complex storage in candle, so `is_complex` never holds here.
        """
        if not input.dtype.is_floating_point:
            return dispatch(
                "aten.full.default", list(input.shape), True, dtype=module.bool
            )
        finite_mask = dispatch(
            "aten.ne.Scalar", dispatch("aten.abs.default", input), float("inf")
        )
        not_nan = dispatch("aten.eq.Tensor", input, input)
        return dispatch("aten.mul.Tensor", not_nan, finite_mask)

    isfinite.__name__ = isfinite.__qualname__ = "isfinite"
    isfinite.__module__ = "torch._C"
    setattr(varfns, "isfinite", isfinite)

    def index_put_(input, indices, values, accumulate=False):
        """`torch.index_put_(input, indices, values, accumulate=False)`.

        Not an `overloads.json` entry: `aten::index_put_`'s schema is

            aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)

        and `indices: Tensor?[]` -- a list whose *elements* are optional
        Tensors, one per axis, `None` meaning "take this whole axis" -- does
        not round-trip through `_decompose_type`. That function's own
        docstring says `?` binds outermost (`int[]?` is an *optional list*),
        which is exactly the opposite shape: stripping the trailing `]`
        first leaves `"Tensor?"` as the base type, and that fails the
        `_SCHEMA_BASE_TYPES` membership check at install time -- measured:

            RuntimeError: torch._C shim: overloads.json entry 'index_put_'
            uses schema type 'Tensor?', which _TypeChecker does not handle:
            aten::index_put_(...)

        So this is not a missing table entry; it is a schema shape the type
        checker has no notation for, and the table cannot express it at all.

        This lands at the same door `layer_norm`/`isfinite` above use for a
        `CompositeImplicitAutograd` mismatch, except the mismatch here is in
        the type checker's vocabulary rather than in which dispatch key
        fires. There is exactly one candidate schema
        (`aten.index_put_.default`), so there is nothing to resolve -- the
        call goes straight to `dispatch` with no argument validation of its
        own, the same contract every other composite here keeps: validation
        is `aten.rs`'s job, after the call.

        `Tensor.index_put_` is the method-door twin, `_install_tensor_index_put_`
        below -- the two are kept in separate functions rather than shared
        because one binds `self` and the other does not, the same split
        `_install_tensor_indexing`'s note draws for `__getitem__`.
        """
        return dispatch(
            "aten.index_put_.default", input, list(indices), values, accumulate
        )

    index_put_.__name__ = index_put_.__qualname__ = "index_put_"
    index_put_.__module__ = "torch._C"
    setattr(varfns, "index_put_", index_put_)

    def square(input, *, out=None):
        """`torch.square` -- `persimmon`'s wall, and a composite, not an op.

        `transformers/activations.py:213` (`ReLUSquaredActivation`) is the
        caller: `squared = torch.square(relu_applied)`, once per MLP per layer.

        **The measurement is what decides this is not an `overloads.json`
        entry.** A `TorchDispatchMode` logger on torch 2.13.0 shows
        `torch.square(x)` firing exactly one record -- `aten.pow.Tensor_Scalar`
        -- for every dtype tried (float32/64/16, bfloat16, int8..int64, uint8,
        bool). `aten.square.default` never fires: `aten::square` is
        `CompositeImplicitAutograd` and its C++ body is `self.pow(2)`. So an
        overload entry would name a kernel upstream does not have either, which
        is the complaint `layer_norm`'s note above makes.

        The exponent is the *integer* 2, not `2.0`, and that is load-bearing
        rather than cosmetic: `pow`'s wrapped-number rule keeps an integral
        tensor integral under an integer exponent, so `square(int64([2,3]))` is
        `int64([4,9])` -- measured -- while a `2.0` here would return
        `float32`. The same rule is why `square(float16)` stays `float16`.

        `out=` is refused by name rather than forwarded. Upstream accepts it
        (`torch.square(x, out=o)` works), but the route would be
        `aten::pow.Tensor_Scalar_out`, which this shim has no kernel for; every
        other `out=` in this file refuses for the same reason.

        `square(bool_t)` is `int64` upstream and refuses here -- see
        `pow_result_tag`'s `Bool` arm in `aten.rs` for the measurement and for
        why the exponent fast-path ladder behind it is not reproduced.
        """
        if out is not None:
            raise NotImplementedError(
                "not implemented in torch._C shim: torch.square(out=...) -- "
                "upstream routes it to aten::pow.Tensor_Scalar_out, which this "
                "shim has no kernel for; square into a fresh tensor instead"
            )
        return dispatch("aten.pow.Tensor_Scalar", input, 2)

    square.__name__ = square.__qualname__ = "square"
    square.__module__ = "torch._C"
    setattr(varfns, "square", square)

    def concat(tensors, dim=0, *, out=None):
        """`torch.concat` -- `zoedepth`'s wall after `relu_`, and an alias.

        `aten::concat` is `CompositeImplicitAutograd` and its body is
        `at::cat(tensors, dim)`. Measured with a `TorchDispatchMode` logger on
        torch 2.13.0: `torch.concat([a, b], dim=0)` and `torch.concat([a, b],
        1)` each fire exactly `aten.cat.default` and nothing else --
        `aten.concat.default` never fires. So this is a composite for the
        reason `square`'s note above gives, rather than an `overloads.json`
        entry that would name a key upstream's dispatcher never answers to.

        There is deliberately no `Tensor.concat` beside it: upstream has none
        either (`hasattr(torch.Tensor, "concat")` is False on 2.13.0), so
        adding one would invent a surface.

        `out=` is refused by name rather than forwarded, as every other `out=`
        in this file is: the route would be `aten::cat.out`, which this shim
        has no kernel for.
        """
        if out is not None:
            raise NotImplementedError(
                "not implemented in torch._C shim: torch.concat(..., out=...) "
                "-- would need aten.cat.out, which has no kernel here"
            )
        return dispatch("aten.cat.default", list(tensors), dim)

    concat.__name__ = concat.__qualname__ = "concat"
    concat.__module__ = "torch._C"
    setattr(varfns, "concat", concat)

    def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
                   count_include_pad=True):
        """`torch.avg_pool1d` -- `sew_d`'s wall after `sign`.

        `nn.AvgPool1d.forward` -> `F.avg_pool1d` -> here. `aten::avg_pool1d` is
        `CompositeImplicitAutograd`; measured with a `TorchDispatchMode` logger
        on torch 2.13.0, `torch.avg_pool1d(x, 3, 2)` fires exactly

            aten.unsqueeze.default(-2)
            aten.avg_pool2d.default([1, 3], [1, 2])
            aten.squeeze.dim(-2)

        and `aten.avg_pool1d.default` never fires. So this is a composite for
        the reason `layer_norm`'s note above gives, and the **height** axis is
        the degenerate one -- `[1, k]`, not `[k, 1]`. Getting that round the
        wrong way pools along a length-1 axis, which is the identity, and then
        the *output shape* is wrong rather than the values, so it fails loudly
        rather than quietly. It is transcribed from the trace regardless.

        `stride=None` means "the kernel size", which is upstream's `int[1]
        stride=[]` default written in Python. It is not 1, and the difference
        is the whole output length.
        """
        length = [kernel_size] if isinstance(kernel_size, int) else list(kernel_size)
        if stride is None or (not isinstance(stride, int) and not list(stride)):
            step = length
        else:
            step = [stride] if isinstance(stride, int) else list(stride)
        pad = [padding] if isinstance(padding, int) else list(padding)
        widened = dispatch("aten.unsqueeze.default", input, -2)
        pooled = dispatch(
            "aten.avg_pool2d.default", widened, [1, length[0]], [1, step[0]],
            [0, pad[0]], ceil_mode, count_include_pad,
        )
        return dispatch("aten.squeeze.dim", pooled, -2)

    avg_pool1d.__name__ = avg_pool1d.__qualname__ = "avg_pool1d"
    avg_pool1d.__module__ = "torch._C"
    setattr(varfns, "avg_pool1d", avg_pool1d)

    def einsum(equation, *operands, path=None):
        """`torch._C._VariableFunctions.einsum` -- `sam3_video`'s wall after
        `div`'s promotion.

        `modeling_sam3.py:2113` is `torch.einsum("bqc,bchw->bqhw",
        mask_embeddings, instance_embeds)`, once per forward, and
        `torch/functional.py:372` sends it here as `_VF.einsum(equation,
        operands)` -- a list, not varargs.

        **`aten::einsum` is `CompositeImplicitAutograd`, and this reproduces
        its decomposition rather than inventing one.** Measured with a
        `TorchDispatchMode` logger on 2.13.0, `einsum("bqc,bkc->bqk", ...)`
        fires exactly `unsqueeze`, `permute`, `view`, **`bmm`**, `view`,
        `permute`, `view` -- every one of which this shim already has, and
        `aten.einsum.default` never fires. So this is a composite for the
        reason `layer_norm`'s note above gives, and -- the part that matters
        numerically -- **the contraction really is a `bmm`**, so the
        accumulation order is upstream's rather than a hand-rolled sum's.

        The algorithm, for one contraction of two operands:

            batch      labels in BOTH inputs and in the output
            summed     labels in BOTH inputs and NOT in the output
            free_a     labels in a only (and in the output)
            free_b     labels in b only (and in the output)

            a -> (batch, free_a, summed) -> (B, M, K)
            b -> (batch, summed, free_b) -> (B, K, N)
            bmm                             (B, M, N)
            -> batch + free_a + free_b   -> permuted to the output order

        A label that appears in one operand only and *not* in the output is
        summed out of that operand first, before the pairing -- otherwise it
        would have to become a spurious free axis.

        More than two operands fold left, which is what `torch/functional.py`
        does too when `opt_einsum` is unavailable.

        **The ellipsis (`...`)** stands for the leading axes an operand has
        beyond its explicit labels -- `longt5` (`modeling_longt5.py:662`,
        docs/kernels/TAIL3.md §7, docs/bindings/BIND2.md item 2) passes exactly one equation
        form, `'...qhd,...khd->...hqk'`. It is expanded to real, fresh labels
        (drawn from letters that appear nowhere else in the equation) before
        the rest of this function ever sees it -- so the contraction
        machinery above never has to know an ellipsis existed, and this stays
        "one branch that maps `...` to explicit labels", not a general
        `numpy`-style broadcasting planner. Every operand that carries `...`
        must have the *same* number of leading axes once its explicit labels
        are subtracted from its rank; upstream's own ellipsis broadcasts
        mismatched ranks against each other, and that broadcasting -- not the
        expansion itself -- is refused by name here as out of scope, because
        `longt5`'s call has no such mismatch and inventing the broadcast rule
        untested would be exactly `docs/bindings/BINDINGS.md`'s `upsample_nearest2d`
        trap the other direction: matching a *summary* ("ellipsis support")
        rather than the measured call.

        **Refused by name, not approximated:**

          * a label repeated *within* one operand (`ii->i`) -- that is a
            diagonal, not a contraction, and there is no `diagonal` kernel
            here;
          * the sublist form (`einsum(a, [0, 1], b, [1, 2])`), which is a
            different overload;
          * a non-`None` `path`, which only `opt_einsum` supplies.

        The implicit-output form (no `->`) IS supported, with upstream's rule:
        the output is every label appearing exactly once across all operands,
        in alphabetical order.
        """
        if path is not None:
            raise NotImplementedError(
                "not implemented in torch._C shim: torch.einsum(..., path=...) "
                "-- only opt_einsum supplies a path and it is not wired here"
            )
        # Both calling conventions, because upstream has both:
        # `torch/functional.py:362` unpacks "the old interface of passing the
        # operands as one list argument", and line 372 then calls
        # `_VF.einsum(equation, operands)` -- with a list. So this name is
        # reached with a list from the vendored tree and with varargs from a
        # caller who writes `torch.einsum("ij,jk->ik", a, b)` directly.
        if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
            operands = tuple(operands[0])
        tensors = list(operands)
        if not tensors:
            raise RuntimeError("einsum(): must provide at least one operand")
        text = equation.replace(" ", "")
        if "." in text and "..." not in text:
            raise NotImplementedError(
                "not implemented in torch._C shim: torch.einsum with a "
                f"malformed ellipsis ({equation!r})"
            )
        if "->" in text:
            lhs_text, out_labels = text.split("->", 1)
        else:
            lhs_text, out_labels = text, None
        terms = lhs_text.split(",")
        if len(terms) != len(tensors):
            raise RuntimeError(
                f"einsum(): the equation has {len(terms)} operand(s) but "
                f"{len(tensors)} were given"
            )
        if "..." in text:
            # Fresh labels for the axes `...` stands for, drawn from letters
            # this equation does not already use -- `used` has to look past
            # `...` itself, which is not a label.
            used = set(lhs_text.replace("...", ""))
            if out_labels is not None:
                used |= set(out_labels.replace("...", ""))
            pool = [c for c in string.ascii_uppercase + string.ascii_lowercase
                    if c not in used]
            ranks = []
            for term, tensor in zip(terms, tensors):
                if "..." in term:
                    explicit = len(term) - 3
                    erank = len(tensor.shape) - explicit
                    if erank < 0:
                        raise RuntimeError(
                            f"einsum(): the subscript {term!r} has more labels "
                            f"than the operand has dimensions"
                        )
                    ranks.append(erank)
            if len(set(ranks)) > 1:
                raise NotImplementedError(
                    "not implemented in torch._C shim: torch.einsum(...) with "
                    f"an ellipsis standing for a different number of axes per "
                    f"operand ({equation!r}) -- that is upstream's broadcasting "
                    "rule, which is not reproduced here"
                )
            erank = ranks[0] if ranks else 0
            if erank > len(pool):
                raise NotImplementedError(
                    "not implemented in torch._C shim: torch.einsum(...) needs "
                    f"more free labels than are available ({equation!r})"
                )
            ellipsis_labels = "".join(pool[:erank])
            terms = [t.replace("...", ellipsis_labels) if "..." in t else t
                     for t in terms]
            if out_labels is not None and "..." in out_labels:
                out_labels = out_labels.replace("...", ellipsis_labels)
        for term in terms:
            if len(set(term)) != len(term):
                raise NotImplementedError(
                    "not implemented in torch._C shim: torch.einsum with a label "
                    f"repeated inside one operand ({term!r}) -- that is a diagonal, "
                    "not a contraction, and aten.diagonal has no kernel here"
                )
        for term, tensor in zip(terms, tensors):
            if len(term) != len(tensor.shape):
                raise RuntimeError(
                    f"einsum(): the subscript {term!r} has {len(term)} label(s) "
                    f"but the operand has {len(tensor.shape)} dimension(s)"
                )
        if out_labels is None:
            counts = {}
            for term in terms:
                for label in term:
                    counts[label] = counts.get(label, 0) + 1
            out_labels = "".join(sorted(l for l, n in counts.items() if n == 1))

        def sum_over(term, tensor, keep):
            """Drop the axes of `tensor` whose label is not in `keep`."""
            drop = [i for i, label in enumerate(term) if label not in keep]
            if not drop:
                return term, tensor
            reduced = dispatch("aten.sum.dim_IntList", tensor, drop, False)
            return "".join(l for l in term if l in keep), reduced

        def align(term, tensor, order):
            """Permute `tensor` so its labels come out in `order`."""
            perm = [term.index(label) for label in order]
            if perm == list(range(len(perm))):
                return tensor
            return dispatch("aten.permute.default", tensor, perm)

        term, acc = terms[0], tensors[0]
        for index in range(1, len(terms)):
            other_term, other = terms[index], tensors[index]
            # Labels still needed after this contraction: the output's, plus
            # everything the operands still to come will pair against.
            later = set(out_labels)
            for rest in terms[index + 1:]:
                later |= set(rest)
            term, acc = sum_over(term, acc, set(other_term) | later)
            other_term, other = sum_over(other_term, other, set(term) | later)

            shared = [l for l in term if l in other_term]
            batch = [l for l in shared if l in later]
            summed = [l for l in shared if l not in later]
            free_a = [l for l in term if l not in other_term]
            free_b = [l for l in other_term if l not in term]

            left = align(term, acc, batch + free_a + summed)
            right = align(other_term, other, batch + summed + free_b)
            sizes = {}
            for label, extent in zip(batch + free_a + summed, left.shape):
                sizes[label] = int(extent)
            for label, extent in zip(batch + summed + free_b, right.shape):
                sizes[label] = int(extent)

            def product(labels):
                total = 1
                for label in labels:
                    total *= sizes[label]
                return total

            b_size, m_size = product(batch), product(free_a)
            k_size, n_size = product(summed), product(free_b)
            left = dispatch("aten.reshape.default", left, [b_size, m_size, k_size])
            right = dispatch("aten.reshape.default", right, [b_size, k_size, n_size])
            product_out = dispatch("aten.bmm.default", left, right)
            term = "".join(batch + free_a + free_b)
            acc = dispatch(
                "aten.reshape.default", product_out, [sizes[l] for l in term] or [1])
            if not term:
                acc = dispatch("aten.reshape.default", acc, [])

        # A single operand, or a leftover label to reduce after the last pair.
        term, acc = sum_over(term, acc, set(out_labels))
        return align(term, acc, list(out_labels))

    einsum.__name__ = einsum.__qualname__ = "einsum"
    einsum.__module__ = "torch._C"
    setattr(varfns, "einsum", einsum)

    # `repeats` is genuinely optional upstream, because the one-argument
    # overload's single argument is the *repeats* vector, not an input. A
    # `None` default would collide with nothing here, but a sentinel says
    # "not given" rather than "given as None", and upstream distinguishes the
    # two: `repeat_interleave(x, None)` is a combination error there.
    _RI_UNSET = object()

    def repeat_interleave(input, repeats=_RI_UNSET, dim=None, *, output_size=None):
        """`torch.repeat_interleave` -- `cohere`'s wall, also a composite.

        `models/cohere/modeling_cohere.py:115` is the caller, and its own
        comment says why it is not `cat`: *"diff from Llama: we interleave()
        instead of cat()"*. So this is on cohere's rotary path, once per
        forward.

        **Two of upstream's overloads, and only one of them is here.** A
        `TorchDispatchMode` logger on torch 2.13.0:

            repeat_interleave(x, 2, dim=-1)   unsqueeze, expand, clone, view
            repeat_interleave(x, 2)           view, unsqueeze, expand, clone, view
            repeat_interleave(x, tensor, ...) repeat_interleave.Tensor, index_select

        The integer-`repeats` overload (`aten::repeat_interleave.self_int`) is
        `CompositeImplicitAutograd` and emits *no* record of its own -- it is
        the four-op expansion above, every one of which this shim already has.
        The tensor-`repeats` overload is a genuine kernel plus `index_select`;
        **both exist here now** (docs/kernels/REPEAT.md), where docs/kernels/LAST7.md §5.2 left
        the first of the two missing and this function refusing by name. The
        refusal it left behind is gone and the two callers it named --
        `fastspeech2_conformer`'s length regulator and the one-argument
        spelling -- reach the kernel.

        The expansion, transcribed from the trace rather than invented:

            dim is None:  flatten to 1-D first, then dim = 0
            unsqueeze at dim+1        -> (..., n_dim, 1, ...)
            expand that axis to repeats
            clone (expand is a view; the copy is what materialises the repeat)
            view back with sizes[dim] *= repeats

        Checked against upstream's answers, not just its op sequence:
        `repeat_interleave([[0,1,2],[3,4,5]], 2, dim=1)` is
        `[[0,0,1,1,2,2],[3,3,4,4,5,5]]` and `dim=0` is
        `[[0,1,2],[0,1,2],[3,4,5],[3,4,5]]` -- the two differ, so a wrong
        unsqueeze axis cannot pass both.

        Three refusals copied from upstream's own messages, all measured:
        a negative `repeats`, an out-of-range `dim`, and an `output_size` that
        disagrees with the computed one. `repeats=0` is *not* an error --
        it produces a zero-length axis, and `repeats=1` is the identity.
        """
        if repeats is _RI_UNSET or isinstance(input, (list, tuple)) or not isinstance(
            input, tensorbase
        ):
            # `aten::repeat_interleave.Tensor(Tensor repeats, ...)` -- the
            # one-argument spelling, where the only argument *is* the repeats
            # vector and the answer is the index vector. `torch.repeat_interleave
            # (tensor([2, 0, 3]))` is `tensor([0, 0, 2, 2, 2])` upstream; with
            # the kernel present this is now that call and nothing else.
            # docs/kernels/REPEAT.md §2.
            if repeats is not _RI_UNSET or dim is not None or not isinstance(
                input, tensorbase
            ):
                # Upstream answers a combination error here, and the shape of
                # it is the point: the overload that takes `dim` is the one
                # that takes an `input` *and* a `repeats`, so there is no
                # overload left for a `dim` beside a lone repeats vector.
                raise TypeError(
                    "repeat_interleave() received an invalid combination of "
                    "arguments - expected one of:\n"
                    " * (Tensor input, Tensor repeats, int dim = None, *, "
                    "int output_size = None)\n"
                    " * (Tensor input, int repeats, int dim = None, *, "
                    "int output_size = None)\n"
                    " * (Tensor repeats, *, int output_size = None)"
                )
            return dispatch(
                "aten.repeat_interleave.Tensor", input, output_size=output_size
            )
        if isinstance(repeats, tensorbase):
            # Upstream's lowering, transcribed from a `TorchDispatchMode`
            # logger on torch 2.13.0 rather than invented (docs/kernels/REPEAT.md §2):
            #
            #   dim=0, repeats (3,)     repeat_interleave.Tensor, index_select
            #   dim=0, repeats (1,)     view [1], expand [n], then the two above
            #   dim=None                view [-1] first, then dim = 0
            #
            # The broadcast of a one-element `repeats` happens *here*, above
            # the kernel, which is where upstream does it -- the kernel itself
            # refuses a length that disagrees with the axis, and would refuse
            # a legitimate `repeat_interleave(x, tensor([2]), 0)` if this were
            # pushed down into it.
            if dim is None:
                input = dispatch("aten.view.default", input, [-1])
                axis = 0
            else:
                rank = input.dim()
                axis = dim if dim >= 0 else dim + rank
                if axis < 0 or axis >= rank:
                    raise IndexError(
                        f"Dimension out of range (expected to be in range of "
                        f"[{-rank}, {rank - 1}], but got {dim})"
                    )
            extent = list(input.shape)[axis]
            if repeats.dim() == 1 and repeats.numel() == 1 and extent != 1:
                repeats = dispatch("aten.view.default", repeats, [1])
                repeats = dispatch("aten.expand.default", repeats, [extent])
            # Upstream's check, and its wording, measured on
            # `repeat_interleave(x, tensor([2, 3]), 0)` with `x.size(0) == 3`.
            # It fires before the kernel, so a `repeats` of the wrong length
            # never becomes an index vector that `index_select` would then
            # reject for a different reason.
            if repeats.dim() == 1 and repeats.numel() != extent:
                raise RuntimeError(
                    f"repeats must have the same size as input along dim, but "
                    f"got repeats.size(0) = {repeats.numel()} and "
                    f"input.size(0) = {extent}"
                )
            index = dispatch(
                "aten.repeat_interleave.Tensor", repeats, output_size=output_size
            )
            return dispatch("aten.index_select.default", input, axis, index)
        repeats = int(repeats)
        if repeats < 0:
            raise RuntimeError("Repeats must be non-negative")

        if dim is None:
            # Upstream's `self.flatten()`; the trace's leading `view.default`.
            input = dispatch("aten.view.default", input, [-1])
            axis = 0
        else:
            rank = input.dim()
            axis = dim if dim >= 0 else dim + rank
            if axis < 0 or axis >= rank:
                # Upstream's wording, verbatim, including the range it prints.
                raise IndexError(
                    f"Dimension out of range (expected to be in range of "
                    f"[{-rank}, {rank - 1}], but got {dim})"
                )

        sizes = list(input.shape)
        expanded = sizes[: axis + 1] + [repeats] + sizes[axis + 1 :]
        final = list(sizes)
        final[axis] = sizes[axis] * repeats
        if output_size is not None and int(output_size) != final[axis]:
            raise RuntimeError(
                f"repeat_interleave: Invalid output_size, expected "
                f"{final[axis]} but got {int(output_size)}"
            )

        widened = dispatch("aten.unsqueeze.default", input, axis + 1)
        widened = dispatch("aten.expand.default", widened, expanded)
        # `expand` is a view with a zero stride on the new axis; the `clone` is
        # what turns the repeat into real elements, and upstream does the same.
        widened = dispatch("aten.clone.default", widened)
        return dispatch("aten.view.default", widened, final)

    repeat_interleave.__name__ = repeat_interleave.__qualname__ = "repeat_interleave"
    repeat_interleave.__module__ = "torch._C"
    setattr(varfns, "repeat_interleave", repeat_interleave)

    # Both of the above are reachable as `TensorBase` members too, and a name
    # with no case is a name nobody checks (docs/kernels/GROUPED_MM.md §6.4): the
    # kernel-level cases passed for weeks while `clamp_`/`chunk`/`__setitem__`
    # raised `NotImplementedError` through the member. `Tensor.square()` and
    # `Tensor.repeat_interleave(...)` bind to the same closures, with `self`
    # in the first slot -- they cannot drift from `torch.square` /
    # `torch.repeat_interleave` because they *are* those functions.
    setattr(tensorbase, "square", square)
    setattr(tensorbase, "repeat_interleave", repeat_interleave)

    # ...and `torch.flatten`, which is the mirror image: the *member* is what
    # `cohere` called and it is installed in `_install_tensor_chunk`, but
    # `torch.flatten(x, 1)` is the same composite and would otherwise be a
    # refusal pointing at `torch.ops.aten.flatten.<overload>` -- a work item
    # nobody could close, since `aten.flatten.using_ints` is composite and
    # never reaches a kernel. Bound to the member so the two cannot drift.
    def flatten(input, start_dim=0, end_dim=-1):
        return module.TensorBase.flatten(input, start_dim, end_dim)

    flatten.__name__ = flatten.__qualname__ = "flatten"
    flatten.__module__ = "torch._C"
    setattr(varfns, "flatten", flatten)

    # ...and `torch.softmax`, the same shape of gap as `flatten` above, found
    # by running `torch.softmax(x, dim=1)` rather than by reading a list.
    #
    # `Tensor.softmax` has worked since docs/bindings/NN_SURFACE.md §6 (installed by
    # `_install_tensor_softmax`), and `aten._softmax.default` has been
    # implemented and golden-compared far longer than that -- but the free
    # function refused with "overload resolution has no table entry", pointing
    # the caller at `torch.ops.aten.softmax.<overload>`, a work item that
    # cannot be closed.
    #
    # **An `overloads.json` entry would be the wrong fix, and the near-miss is
    # close enough to be tempting.** The parser-level key for `torch.softmax`
    # is `aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None)` --
    # a real ATen op with a real schema, so the table would validate and
    # `verify_schemas.py` would pass -- but it is `CompositeImplicitAutograd`
    # and never reaches a kernel. Re-measured for this change with a
    # `TorchDispatchMode` logger on torch 2.13.0:
    #
    #     torch.softmax(x, dim=1)                      -> aten._softmax.default
    #     torch.softmax(x, dim=1, dtype=torch.float64) -> aten._to_copy.default
    #                                                     then aten._softmax.default
    #     x.softmax(1)                                 -> aten._softmax.default
    #     F.softmax(x, dim=1)                          -> aten._softmax.default
    #
    # `aten.softmax.int` never fires, for any of the four. Naming it would move
    # the refusal from a generic message to a *specific wrong* one -- the
    # `layer_norm` complaint above, and the reason `methods.json`'s README
    # keeps `softmax` out of that table too.
    #
    # Bound to the member for the same reason `flatten` is: they are one
    # function, so the free spelling and the member cannot disagree about
    # `dtype=` handling or about `half_to_float`.
    def softmax(input, dim, dtype=None):
        return module.TensorBase.softmax(input, dim, dtype)

    softmax.__name__ = softmax.__qualname__ = "softmax"
    softmax.__module__ = "torch._C"
    setattr(varfns, "softmax", softmax)

    # `torch.log_softmax`, the same shape as `softmax` directly above and for
    # the same reason: `aten::log_softmax.int` is `CompositeImplicitAutograd`,
    # so an `overloads.json` entry would validate against upstream's schema and
    # still name a key no dispatcher ever sees. Bound to the member so the free
    # spelling and `Tensor.log_softmax` are one function (docs/training/LOSS.md).
    #
    # `torch._log_softmax` -- one underscore away -- IS an `overloads.json`
    # entry, because that name is the dispatched leaf.
    def log_softmax(input, dim, dtype=None):
        return module.TensorBase.log_softmax(input, dim, dtype)

    log_softmax.__name__ = log_softmax.__qualname__ = "log_softmax"
    log_softmax.__module__ = "torch._C"
    setattr(varfns, "log_softmax", log_softmax)

    def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        """`torch.conv1d` -- `mamba`'s depthwise causal convolution.

        `nn.Conv1d.forward` reaches `F.conv1d`, which `torch/nn/functional.py`
        binds straight to `torch.conv1d` (`_add_docstr(torch.conv1d, ...)`), so
        this is the name a `nn.Conv1d` actually calls.

        **The kernel was already here.** `aten.convolution.default` has been
        implemented and golden-compared since docs/kernels/OPS4.md, and
        `torch.conv1d(...)` still refused -- the third instance in this round
        of a kernel with no spelling (docs/architectures/ARCH20.md §5, §9). `aten::conv1d` is
        `CompositeImplicitAutograd`; measured with a `TorchDispatchMode` logger
        on 2.13.0, every form of the call fires exactly one record:

            conv1d(x, w, b, 1, 2, 1, 4)  -> convolution(x, w, b, [1], [2], [1],
                                                        False, [0], 4)

        so the whole of the composite is filling in `transposed=False` and
        `output_padding=[0]`, the two arguments `conv1d` does not have and
        `convolution` requires.

        Scalars widen to one-element lists because `convolution`'s schema takes
        `SymInt[]` for all three, and the trace shows upstream passing `[1]`
        where the caller wrote `1`.

        `padding` accepts upstream's two string spellings as well as a number.
        `"valid"` is zero. `"same"` is `dilation * (kernel - 1) // 2`, which is
        only the whole answer when that product is *even* -- when it is odd,
        upstream pads the input asymmetrically (one extra zero on the right)
        with `aten::constant_pad_nd` before convolving symmetrically with
        `total // 2` (docs/kernels/RNN.md §1.2). Measured: `padding="same"` with a
        3-tap kernel and dilation 1 reaches `convolution(..., [1], ...)`,
        which is the even case. Non-unit stride with `"same"` is upstream's
        own refusal.
        """
        def _as_list(value):
            return list(value) if isinstance(value, (list, tuple)) else [value]

        stride = _as_list(stride)
        dilation = _as_list(dilation)
        if isinstance(padding, str):
            if padding == "valid":
                padding = [0]
            elif padding == "same":
                if any(s != 1 for s in stride):
                    raise RuntimeError(
                        "padding='same' is not supported for strided convolutions"
                    )
                kernel = weight.shape[-1]
                total = dilation[0] * (kernel - 1)
                if total % 2 != 0:
                    # docs/kernels/RNN.md §1.2, measured with a `TorchDispatchMode`
                    # logger on upstream 2.13.0:
                    #   aten.constant_pad_nd.default(x, [0, 1])
                    #   aten.convolution.default(..., padding=[total // 2], ...)
                    # i.e. one extra element of zero padding on the RIGHT,
                    # then the ordinary symmetric convolution. Padding the
                    # left instead gives a different answer -- pinned by
                    # test_bind4.py against upstream, not asserted in prose.
                    input = dispatch(
                        "aten.constant_pad_nd.default", input, [0, 1], 0.0
                    )
                padding = [total // 2]
            else:
                raise ValueError(
                    f"conv1d: padding must be 'valid', 'same', or an int, got {padding!r}"
                )
        else:
            padding = _as_list(padding)

        return dispatch(
            "aten.convolution.default",
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            [0],
            groups,
        )

    conv1d.__name__ = conv1d.__qualname__ = "conv1d"
    conv1d.__module__ = "torch._C"
    setattr(varfns, "conv1d", conv1d)

    def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        """`torch.conv2d` -- docs/architectures/ARCH26.md, `zoedepth`'s wall (`Dinov2`'s
        patch-embedding `nn.Conv2d`, through `F.conv2d`, which
        `torch/nn/functional.py` binds straight to `torch.conv2d` the same way
        it binds `F.conv1d` to `torch.conv1d` above).

        Same composite as `conv1d` immediately above -- `aten::conv2d` is
        `CompositeImplicitAutograd` and a `TorchDispatchMode` logger on 2.13.0
        shows every form of the call firing exactly one record:

            conv2d(x, w, b, stride=2, padding=1, dilation=1, groups=1)
              -> convolution(x, w, b, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        The only difference from `conv1d` is arity: two spatial dims instead
        of one, so a scalar `stride`/`padding`/`dilation` widens to **two**
        elements, not one -- measured in the trace above, where `stride=2`
        reaches `convolution` as `[2, 2]`. `nn.Conv2d.__init__` already
        normalises its own `stride`/`padding`/`dilation` attributes to 2-tuples
        via `_pair` before `_conv_forward` calls here, so in practice the
        widening below mostly serves a caller that spells `F.conv2d` directly
        with a bare int -- but it is upstream's own rule either way, not a
        convenience added here.

        `padding="same"`/`"valid"` follow `conv1d`'s rule component-wise: each
        of the two `dilation[i] * (kernel[i] - 1)` totals must be even, and
        either being odd is refused by name for the reason `conv1d` refuses
        it -- upstream pads that axis asymmetrically, and picking a half would
        shift the output by one sample on that axis.
        """
        def _as_list(value, n=2):
            if isinstance(value, (list, tuple)):
                return list(value)
            return [value] * n

        stride = _as_list(stride)
        dilation = _as_list(dilation)
        if isinstance(padding, str):
            if padding == "valid":
                padding = [0, 0]
            elif padding == "same":
                if any(s != 1 for s in stride):
                    raise RuntimeError(
                        "padding='same' is not supported for strided convolutions"
                    )
                kernel = list(weight.shape[-2:])
                totals = [dilation[i] * (kernel[i] - 1) for i in range(2)]
                if any(total % 2 != 0 for total in totals):
                    raise NotImplementedError(
                        "not implemented in torch._C shim: torch.conv2d("
                        "padding='same') where dilation*(kernel-1) is odd on "
                        "some axis -- upstream pads that axis asymmetrically "
                        "with aten::constant_pad_nd before convolving, and "
                        "picking either half of the split here would shift "
                        "the output by one sample on that axis"
                    )
                padding = [total // 2 for total in totals]
            else:
                raise ValueError(
                    f"conv2d: padding must be 'valid', 'same', or an int/pair, got {padding!r}"
                )
        else:
            padding = _as_list(padding)

        return dispatch(
            "aten.convolution.default",
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            [0, 0],
            groups,
        )

    conv2d.__name__ = conv2d.__qualname__ = "conv2d"
    conv2d.__module__ = "torch._C"
    setattr(varfns, "conv2d", conv2d)

    def conv_transpose2d(
        input, weight, bias=None, stride=1, padding=0, output_padding=0,
        groups=1, dilation=1,
    ):
        """`torch.conv_transpose2d` -- `zoedepth`'s wall once `conv2d` and
        `expand_as` were behind it (docs/kernels/KERNELS26.md §7).

        `ZoeDepthUpsample` is `nn.ConvTranspose2d(channels, channels,
        kernel_size=factor, stride=factor, padding=0)`, reached through
        `F.conv_transpose2d`, which -- like `F.conv1d` and `F.conv2d` above --
        **is** `torch.conv_transpose2d` (asserted, not assumed: `F.conv_transpose2d
        is torch.conv_transpose2d` is True on 2.13.0). `aten::conv_transpose2d`
        is `CompositeImplicitAutograd`, and a `TorchDispatchMode` logger shows
        every form of the call firing exactly one record:

            conv_transpose2d(x, w, b, stride=2, padding=1, output_padding=1,
                             dilation=2)
              -> convolution(x, w, b, [2,2], [1,1], [2,2], True, [1,1], 1)

        **The signature is NOT `conv2d`'s with an extra argument.** Upstream's
        is `(input, weight, bias, stride, padding, output_padding, groups,
        dilation)` -- `groups` comes *before* `dilation`, where `conv2d` has
        `dilation` before `groups`. Read off `torch.conv_transpose2d.__doc__`
        and confirmed by calling it positionally, because transcribing
        `conv2d`'s order here would silently swap the two for every positional
        caller, and both are small integers that usually produce a
        plausible-looking tensor rather than an error.

        `padding` here is not `conv2d`'s: there is no `'same'`/`'valid'` form
        (upstream's transposed signature takes no string), so a string is
        refused rather than being given `conv2d`'s meaning.
        """
        def _as_list(value, n=2):
            if isinstance(value, (list, tuple)):
                return list(value)
            return [value] * n

        if isinstance(padding, str):
            raise ValueError(
                "conv_transpose2d: padding must be an int or a pair, got "
                f"{padding!r} -- 'same'/'valid' are conv2d's forms and upstream's "
                "transposed signature does not accept them"
            )
        return dispatch(
            "aten.convolution.default",
            input,
            weight,
            bias,
            _as_list(stride),
            _as_list(padding),
            _as_list(dilation),
            True,
            _as_list(output_padding),
            groups,
        )

    conv_transpose2d.__name__ = conv_transpose2d.__qualname__ = "conv_transpose2d"
    conv_transpose2d.__module__ = "torch._C"
    setattr(varfns, "conv_transpose2d", conv_transpose2d)

    def conv_transpose1d(
        input, weight, bias=None, stride=1, padding=0, output_padding=0,
        groups=1, dilation=1,
    ):
        """`torch.conv_transpose1d` -- `vits`' last wall (docs/kernels/KERNELS26.md §24).

        `modeling_vits.py`'s HiFi-GAN decoder is
        `nn.ConvTranspose1d(channels, channels // 2, kernel, stride=rate,
        padding=(kernel - rate) // 2)`, once per entry in `upsample_rates`,
        reached through `F.conv_transpose1d` -- which **is**
        `torch.conv_transpose1d` (asserted, not assumed, exactly as
        `conv_transpose2d` above asserts its own).

        Same shape as its 2-D sibling in every respect, including the argument
        order that is not `conv1d`'s: **`groups` comes before `dilation`**.
        The one difference is downstream rather than here -- `aten.rs` can
        honour `groups` for the 1-D transposed case and not for the 2-D one,
        because candle's `conv_transpose1d` takes a `groups` argument and its
        `ParamsConvTranspose2D` has no field for one.

        The `_as_list` width is 1, not 2. Passing 2 would hand
        `aten.convolution.default` a two-element `stride` for a rank-3 input,
        which is where a copy-paste of the 2-D body goes wrong.
        """
        def _as_list(value, n=1):
            if isinstance(value, (list, tuple)):
                return list(value)
            return [value] * n

        if isinstance(padding, str):
            raise ValueError(
                "conv_transpose1d: padding must be an int or a 1-tuple, got "
                f"{padding!r} -- 'same'/'valid' are conv1d's forms and upstream's "
                "transposed signature does not accept them"
            )
        return dispatch(
            "aten.convolution.default",
            input,
            weight,
            bias,
            _as_list(stride),
            _as_list(padding),
            _as_list(dilation),
            True,
            _as_list(output_padding),
            groups,
        )

    conv_transpose1d.__name__ = conv_transpose1d.__qualname__ = "conv_transpose1d"
    conv_transpose1d.__module__ = "torch._C"
    setattr(varfns, "conv_transpose1d", conv_transpose1d)

    def norm_except_dim(v, pow=2, dim=0):
        """`torch.norm_except_dim` -- the piece of `weight_norm` that a traced
        sweep cannot see (docs/kernels/KERNELS26.md §8.3).

        `aten::norm_except_dim` is `CompositeImplicitAutograd`, and it is called
        from `_WeightNorm.right_inverse`, which `ParametrizationList.__init__`
        runs at **construction** time. ARCH26.md §6's trace ran on a forward, so
        it found `_weight_norm_interface` and missed this one entirely.

        A `TorchDispatchMode` on 2.13.0 shows what it decomposes to, and it is
        not one shape:

            dim=0  or  dim=v.dim()-1  ->  view, norm.ScalarOpt_dim, view
            a middle dim              ->  transpose, clone, view, norm..., view, transpose
            dim=-1                    ->  norm.Scalar   (the whole-tensor norm)

        Those are three routes to one answer: **keep axis `dim`, reduce every
        other axis, keepdim** -- checked against
        `v.pow(2).sum(other_dims, keepdim=True).sqrt()` for dim 0, 1 and 2 of a
        3-D tensor, and against `v.norm()` for `dim=-1`. So this is written as
        that one statement rather than as upstream's three, since
        `aten.norm.ScalarOpt_dim` accepts a multi-axis `dim` list directly
        (measured: `norm(x, 2, [0, 1])` reduces both).

        `dim=-1` is upstream's "no axis is exempt" spelling and gives a 0-d
        result, not a `v.dim()-1` reduction -- it is the one value of `dim` that
        does NOT mean an axis index here, which is why it is a branch rather
        than a normalisation.
        """
        rank = len(v.shape)
        if dim == -1:
            dims = list(range(rank))
            keepdim = False
        else:
            axis = dim if dim >= 0 else dim + rank
            if not 0 <= axis < rank:
                raise IndexError(
                    f"norm_except_dim: dimension out of range (expected to be in "
                    f"range of [{-rank}, {rank - 1}], but got {dim})"
                )
            dims = [d for d in range(rank) if d != axis]
            keepdim = True
        return dispatch("aten.norm.ScalarOpt_dim", v, pow, dims, keepdim)

    norm_except_dim.__name__ = norm_except_dim.__qualname__ = "norm_except_dim"
    norm_except_dim.__module__ = "torch._C"
    setattr(varfns, "norm_except_dim", norm_except_dim)

    def _weight_norm(v, g, dim=0):
        """`torch._weight_norm` -- what `_WeightNorm.forward` calls.

        `aten::_weight_norm` is `CompositeImplicitAutograd` and a
        `TorchDispatchMode` shows it firing exactly one record,
        `aten._weight_norm_interface.default`, whose second result (the norms)
        it discards. So this is that call and a `[0]`.

        `_weight_norm_interface` only accepts `dim == 0` or `dim == v.dim()-1`
        (upstream trips an internal assertion otherwise). Both measured callers
        are inside that -- `vits` uses the default `dim=0`, `sew_d` passes
        `dim=2` on a 3-D `Conv1d` weight -- and a middle dim is refused by the
        kernel with a message that says so.
        """
        return dispatch("aten._weight_norm_interface.default", v, g, dim)[0]

    _weight_norm.__name__ = _weight_norm.__qualname__ = "_weight_norm"
    _weight_norm.__module__ = "torch._C"
    setattr(varfns, "_weight_norm", _weight_norm)

    def outer(input, vec2):
        """`torch.outer` -- `sam3_video`'s wall once `div.Tensor_mode` was behind
        it, and the third line of the same `Sam3ViTRotaryEmbedding.__init__`
        that `remainder` and `div` were the first two of.

        Upstream has the member as well (`hasattr(torch.Tensor, "outer")` is
        True), so both are installed and both go through `_outer_impl`, whose
        docstring carries the measurement.
        """
        return _outer_impl(input, vec2)

    outer.__name__ = outer.__qualname__ = "outer"
    outer.__module__ = "torch._C"
    setattr(varfns, "outer", outer)

    def tile(input, *dims):
        """`torch.tile` -- the free-function spelling of `TensorBase.tile`.

        Upstream has both (`hasattr(torch, "tile")` is True, measured), and both
        accept the dims either as one sequence or as varargs. Delegates to the
        member so the left-padding rule lives in exactly one place -- see the
        comment there for why `tile` is not an alias for `repeat`.
        """
        return input.tile(*dims)

    tile.__name__ = tile.__qualname__ = "tile"
    tile.__module__ = "torch._C"
    setattr(varfns, "tile", tile)


    def nonzero(input, *, out=None, as_tuple=False):
        _refuse_out("nonzero", out)
        if as_tuple:
            return module._aten_dispatch("aten.where.default", input)
        return module._aten_dispatch("aten.nonzero.default", input)

    nonzero.__module__ = "torch._C"
    setattr(varfns, "nonzero", nonzero)

    def tensor_nonzero(self, *, as_tuple=False):
        if as_tuple:
            return module._aten_dispatch("aten.where.default", self)
        return module._aten_dispatch("aten.nonzero.default", self)

    tensor_nonzero.__module__ = "torch._C"
    tensor_nonzero.__name__ = tensor_nonzero.__qualname__ = "nonzero"
    setattr(module.TensorBase, "nonzero", tensor_nonzero)

    # -- `torch.randn` / `torch.rand` and their `_like`/`normal` siblings ----
    #
    # docs/kernels/RANDOM.md. There is no `aten::randn`/`aten::rand` kernel in
    # `aten.rs` -- only `aten.empty.memory_format` and
    # `aten.uniform_.default`/`aten.normal_.default` -- so these cannot be
    # `overloads.json` entries (a schema entry would name a kernel this shim
    # does not have, the same complaint the table's own README makes about
    # `layer_norm`). Real torch's *own* C++ body for these factories is the
    # same composition: measured against torch 2.13.0, seeded
    # `torch.randn(4, 4)` is bit-identical to seeded
    # `torch.empty(4, 4).normal_(0., 1.)`, and the same holds for
    # `rand`/`uniform_`, `rand_like`/`randn_like`, and every one of
    # `torch.normal`'s four overloads against `mean + std * randn(shape)`
    # (`shape` being the broadcast of `mean`'s and `std`'s shapes when both
    # are tensors) -- so composing here, in Python, reproduces upstream's
    # generator stream exactly rather than approximating it.
    #
    # `out=` is refused by name on all of them rather than silently ignored:
    # upstream *resizes* the given tensor to the requested shape
    # (`torch.randn(4, 4, out=torch.empty(2, 2))` returns a 4x4 tensor), and
    # there is neither `aten::empty.out` nor a generic `resize_` in `aten.rs`
    # to do that with. `dtype`/`layout`/`device`/`pin_memory`/`generator`/
    # `requires_grad` are not reimplemented here either -- they are forwarded
    # to the already-installed, already-validated `varfns.empty`/
    # `varfns.empty_like` and `TensorBase.uniform_`/`normal_`, which refuse
    # exactly what those refuse (an integer `dtype`, a foreign `generator`,
    # `requires_grad=True`) for exactly the reasons documented at their own
    # definitions.

    def _factory_size(args):
        # `torch.randn(4, 4)` and `torch.randn((4, 4))` both have to reach
        # `empty` with the same `[4, 4]` list -- upstream accepts a size
        # given either as varargs or as one sequence argument.
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return list(args[0])
        return list(args)

    def _refuse_out(name, out):
        if out is not None:
            raise NotImplementedError(
                f"not implemented in torch._C shim: torch.{name}(out=...) -- "
                f"upstream resizes the out tensor to the requested shape, "
                f"which needs a kernel this shim does not have (no "
                f"aten::empty.out, no generic resize_); construct into a "
                f"fresh tensor instead"
            )

    def _empty_kwargs(dtype, layout, device, pin_memory, requires_grad):
        # `dtype`/`layout`/`device` are forwarded only when given, so an
        # omitted one reaches `aten.rs` as "absent" (the schema default,
        # `None`) rather than as an explicit value -- `empty.memory_format`'s
        # `reject_unsupported([(2, "layout"), (4, "pin_memory"), ...])` refuses
        # an explicit `layout`/`pin_memory` *by position*, regardless of the
        # value, so a caller-omitted `pin_memory=False` forwarded verbatim
        # would be refused for asking for something nobody asked for. Passing
        # it only when truthy keeps the two cases apart: silently getting an
        # ordinary (unpinned) tensor when the caller said nothing, and
        # honestly refusing when the caller said `pin_memory=True`.
        kwargs = {"requires_grad": requires_grad}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if layout is not None:
            kwargs["layout"] = layout
        if device is not None:
            kwargs["device"] = device
        if pin_memory:
            kwargs["pin_memory"] = pin_memory
        return kwargs

    def _empty_like_kwargs(dtype, layout, device, memory_format, requires_grad):
        kwargs = {"requires_grad": requires_grad}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if layout is not None:
            kwargs["layout"] = layout
        if device is not None:
            kwargs["device"] = device
        if memory_format is not None:
            kwargs["memory_format"] = memory_format
        return kwargs

    def randn(*args, generator=None, out=None, dtype=None, layout=None,
               device=None, requires_grad=False, pin_memory=False):
        _refuse_out("randn", out)
        t = varfns.empty(
            _factory_size(args),
            **_empty_kwargs(dtype, layout, device, pin_memory, requires_grad),
        )
        if generator is not None:
            return t.normal_(0.0, 1.0, generator=generator)
        return t.normal_(0.0, 1.0)

    def rand(*args, generator=None, out=None, dtype=None, layout=None,
             device=None, requires_grad=False, pin_memory=False):
        _refuse_out("rand", out)
        t = varfns.empty(
            _factory_size(args),
            **_empty_kwargs(dtype, layout, device, pin_memory, requires_grad),
        )
        if generator is not None:
            return t.uniform_(0.0, 1.0, generator=generator)
        return t.uniform_(0.0, 1.0)

    def rand_like(input, *, generator=None, dtype=None, layout=None,
                  device=None, requires_grad=False, memory_format=None):
        t = varfns.empty_like(
            input,
            **_empty_like_kwargs(dtype, layout, device, memory_format, requires_grad),
        )
        if generator is not None:
            return t.uniform_(0.0, 1.0, generator=generator)
        return t.uniform_(0.0, 1.0)

    def randn_like(input, *, generator=None, dtype=None, layout=None,
                   device=None, requires_grad=False, memory_format=None):
        t = varfns.empty_like(
            input,
            **_empty_like_kwargs(dtype, layout, device, memory_format, requires_grad),
        )
        if generator is not None:
            return t.normal_(0.0, 1.0, generator=generator)
        return t.normal_(0.0, 1.0)

    def _broadcast_shape(a, b):
        # Plain shape arithmetic, no aten call -- needed *before* the
        # standard-normal draw below can be sized, since `mean`/`std` are not
        # necessarily the same shape (`torch.normal(mean.shape=(3,1),
        # std.shape=(1,4))` draws at `(3, 4)`, measured against torch 2.13.0).
        a, b = list(a), list(b)
        n = max(len(a), len(b))
        out = []
        for i in range(1, n + 1):
            da = a[-i] if i <= len(a) else 1
            db = b[-i] if i <= len(b) else 1
            if da == db or da == 1 or db == 1:
                out.append(max(da, db))
            else:
                raise RuntimeError(
                    f"The size of tensor a ({da}) must match the size of "
                    f"tensor b ({db}) at non-singleton dimension {n - i}"
                )
        return list(reversed(out))

    def normal(mean=0.0, std=1.0, size=None, *, generator=None, out=None,
               dtype=None, layout=None, device=None, pin_memory=None):
        if out is not None:
            raise NotImplementedError(
                "not implemented in torch._C shim: torch.normal(out=...) -- "
                "upstream resizes the out tensor to the requested shape, "
                "which needs a kernel this shim does not have (no "
                "aten::normal.*_out, no generic resize_); construct into a "
                "fresh tensor instead"
            )
        mean_is_tensor = isinstance(mean, tensorbase)
        std_is_tensor = isinstance(std, tensorbase)
        gen_kwargs = {"generator": generator} if generator is not None else {}

        if size is not None:
            # The `float_float` overload -- the one shaped exactly like
            # `randn`/`rand` above, plain numbers plus an explicit size.
            if mean_is_tensor or std_is_tensor:
                raise TypeError(
                    "torch.normal(): size= is only valid when both mean and "
                    "std are plain numbers, not Tensors"
                )
            t = varfns.empty(
                list(size),
                **_empty_kwargs(dtype, layout, device, pin_memory, False),
            )
            return t.normal_(float(mean), float(std), **gen_kwargs)

        if dtype is not None or layout is not None or device is not None \
                or pin_memory is not None:
            raise NotImplementedError(
                "not implemented in torch._C shim: torch.normal(dtype=/"
                "layout=/device=/pin_memory=...) without size= -- upstream's "
                "own arg parser only accepts these together with size= (the "
                "float_float overload); with a Tensor mean or std the "
                "result's dtype/device/layout come from that tensor instead"
            )
        if not mean_is_tensor and not std_is_tensor:
            raise TypeError(
                "torch.normal(): missing required argument 'size' (mean and "
                "std are both plain numbers)"
            )

        if mean_is_tensor and std_is_tensor:
            shape = _broadcast_shape(list(mean.shape), list(std.shape))
            draw_dtype = mean.dtype
        elif mean_is_tensor:
            shape = list(mean.shape)
            draw_dtype = mean.dtype
        else:
            shape = list(std.shape)
            draw_dtype = std.dtype

        # Upstream's own composition, transcribed (measured bit-identical
        # against torch 2.13.0 for all three tensor-carrying overloads,
        # including the broadcasting case): draw standard normal at the
        # output shape, then affine-transform it by `std` and `mean`.
        n = varfns.empty(shape, dtype=draw_dtype)
        n = n.normal_(0.0, 1.0, **gen_kwargs)
        if mean_is_tensor and std_is_tensor:
            scaled = dispatch("aten.mul.Tensor", std, n)
            return dispatch("aten.add.Tensor", mean, scaled)
        if mean_is_tensor:  # std is a plain float
            scaled = dispatch("aten.mul.Scalar", n, std)
            return dispatch("aten.add.Tensor", mean, scaled)
        # mean is a plain float, std is a Tensor
        scaled = dispatch("aten.mul.Tensor", std, n)
        return dispatch("aten.add.Scalar", scaled, mean)

    for fn, name in (
        (randn, "randn"), (rand, "rand"), (rand_like, "rand_like"),
        (randn_like, "randn_like"), (normal, "normal"),
    ):
        fn.__name__ = fn.__qualname__ = name
        fn.__module__ = "torch._C"
        setattr(varfns, name, fn)

    def batch_norm(input, weight, bias, running_mean, running_var, training,
                    momentum, eps, cudnn_enabled):
        """`torch.batch_norm` / `F.batch_norm` / every `nn.BatchNorm*d` forward.

        docs/architectures/DEMAND.md's rank 1, and `layer_norm`/`group_norm`'s shape exactly:
        `aten::batch_norm` is `CompositeImplicitAutograd`, so a
        `TorchDispatchMode` logger on torch 2.13.0 sees `torch.batch_norm(...)`,
        `F.batch_norm(...)` and an `nn.BatchNorm2d` forward all emit
        `aten.native_batch_norm.default` (behind an `empty.memory_format` the
        output allocation makes) and never `aten.batch_norm.default`. So this
        is a composite and not an `overloads.json` entry, for the reason
        `layer_norm`'s note above gives.

        `cudnn_enabled` is accepted and ignored, as `layer_norm`'s
        `cudnn_enable` and `group_norm`'s `cudnn_enabled` are: the CPU kernel
        does not consult it.

        **The four length checks are this function's, not the kernel's**, and
        that is why they are here. Upstream's leaf does not check: measured
        with three channels, a length-2 `weight` and a length-4 `weight` both
        return, and both return bit-identical numbers to the correct length-3
        call -- the short one having read a float past the end of its buffer
        (docs/architectures/DEMAND1.md §1.6). Upstream's *composite* is what refuses, and
        this is upstream's composite. The order below is measured, and it
        shows: a call with a wrong `weight` **and** a wrong `running_mean`
        reports the `running_mean` one.

            num_features = input.size(1) if input.dim() >= 2 else 0
              (rank 0 and rank 1 both report 0, measured -- both then fail the
               running_mean check below with "should contain 0 elements")

            1. running_mean  length, else "running_mean must be defined in evaluation mode"
            2. running_var   length, else "running_var must be defined in evaluation mode"
            3. weight        length
            4. bias          length

        **`input.numel() == 0` short-circuits before all four**, also measured
        rather than assumed: `N=0` with a deliberately wrong `weight`
        succeeds. Upstream's body for it is `out = input.clone()`, then
        `out * weight[0]` and `out + bias[0]` if those are defined -- pinned by
        `C=0`, which raises `IndexError: select(): index 0 out of range for
        tensor of size [0] at dimension 0` from inside that `weight[0]`. The
        running statistics are not touched on this path. Transcribed rather
        than replaced by "return an empty tensor" because upstream's own
        comment says why it clones: an empty tensor would break the gradient
        chain, and the `weight[0]`/`bias[0]` keep the result attached to the
        parameters.

        `F.batch_norm`'s own two checks -- `_verify_batch_size` ("Expected more
        than 1 value per channel when training") and the `eps` sign rules --
        are above this, in the vendored tree, and are upstream's own Python.
        Nothing is added for them here.
        """
        del cudnn_enabled
        shape = list(input.shape)
        num_features = shape[1] if len(shape) >= 2 else 0

        numel = 1
        for extent in shape:
            numel *= extent
        if numel == 0:
            out = dispatch("aten.clone.default", input)
            if weight is not None:
                out = dispatch("aten.mul.Tensor", out, dispatch("aten.select.int", weight, 0, 0))
            if bias is not None:
                out = dispatch("aten.add.Tensor", out, dispatch("aten.select.int", bias, 0, 0))
            return out

        for name, value, needed_in_eval in (
            ("running_mean", running_mean, True),
            ("running_var", running_var, True),
            ("weight", weight, False),
            ("bias", bias, False),
        ):
            if value is None:
                if needed_in_eval and not training:
                    raise RuntimeError(f"{name} must be defined in evaluation mode")
                continue
            count = 1
            for extent in value.shape:
                count *= extent
            if count != num_features:
                raise RuntimeError(
                    f"{name} should contain {num_features} elements not {count}"
                )

        result = dispatch(
            "aten.native_batch_norm.default", input, weight, bias,
            running_mean, running_var, training, momentum, eps,
        )
        return result[0]

    batch_norm.__name__ = batch_norm.__qualname__ = "batch_norm"
    batch_norm.__module__ = "torch._C"
    setattr(varfns, "batch_norm", batch_norm)

    def as_tensor(data, dtype=None, device=None):
        """`torch.as_tensor` -- docs/architectures/DEMAND.md rank 4, a **spelling**, not a kernel.

        `whisper`'s `.generate()` is the measured hit
        (`generation_whisper.py:1608`, `_retrieve_init_tokens`), but
        `transformers/generation/utils.py` calls it from several shared
        helpers, so it is a wall a queue of already-running models is behind
        rather than one model's.

        Not an `overloads.json` entry, and the reason is `torch.tensor`'s
        (`_tensor_factory`): `aten::as_tensor` exists as a schema, but the
        Python binding is `THPVariable_as_tensor` -> `internal_new_from_data`,
        and a `TorchDispatchMode` logger on 2.13.0 shows the real calls firing
        something else entirely:

            torch.as_tensor([1, 2, 3])              ->  aten.lift_fresh.default
            torch.as_tensor(t, dtype=torch.float64) ->  aten._to_copy.default
            torch.as_tensor(t)                      ->  NO record at all

        Both of those primitives already have kernels and golden cases, so the
        branch between them is the whole of the fix.

        **The no-record row is the one worth writing down.** `torch.as_tensor(t)`
        and `torch.as_tensor(t, dtype=t.dtype)` return the *same object* --
        measured, `is` is True, not merely equal. That is the entire difference
        between `as_tensor` and `tensor`, and returning a copy here would be
        right on values and wrong on identity, which nothing downstream raises
        about.

        **What is NOT reproduced, said rather than papered over**:
        `torch.as_tensor(ndarray)` upstream *shares memory* with the array
        (measured -- mutating the array afterwards shows through the tensor).
        This shim's tensors do not wrap foreign buffers, so an ndarray goes
        through `lift_fresh` like a list does and is **copied**. Every measured
        caller in `transformers` builds a fresh array or list and never touches
        it again, so the difference has no observed consequence; it is still a
        difference, and docs/architectures/DEMAND1.md §4 records it.

        **A correction to the paragraph above, made in docs/bindings/CTOR.md's round.**
        Until that round the ndarray path here did not copy either -- it
        *raised*, because `_tensor_new_from_data` walks nested sequences of
        Python scalars and refuses an `ndarray` by name. "Copies rather than
        aliases" was the intended divergence and was written down as though it
        were the observed one; the observed one was a `TypeError`. It now really
        does copy, through `_new_from_data`, and the dtype is the array's --
        which is what upstream answers here (`as_tensor(float64 array)` is
        `float64`), and is the difference between this path and the legacy
        `torch.Tensor(ndarray)` constructor, which casts to the default dtype.
        """
        if _MODE_STACK:
            return _through_torch_function_modes(
                as_tensor, (data,), {"dtype": dtype, "device": device}
            )
        if isinstance(device, str):
            device = module.device(device)
        if isinstance(data, module.TensorBase):
            # Already a tensor: "no conversion" is not a copy, it is identity.
            if dtype is None or dtype == data.dtype:
                return data
            return dispatch("aten._to_copy.default", data, dtype=dtype)
        return dispatch(
            "aten.lift_fresh.default",
            _new_from_data(module, data, dtype, device),
        )

    as_tensor.__name__ = as_tensor.__qualname__ = "as_tensor"
    as_tensor.__module__ = "torch._C"
    setattr(varfns, "as_tensor", as_tensor)

    def meshgrid(tensors, indexing=None):
        """`torch._VF.meshgrid` -- `swin`'s wall, and a spelling over two ops
        that have both been implemented for a long time.

        `torch/functional.py:504` is `_VF.meshgrid(tensors, **kwargs)`, so this
        is the name the vendored tree reaches. `aten::meshgrid` is
        `CompositeImplicitAutograd`, and a `TorchDispatchMode` logger on 2.13.0
        records, for two 1-D inputs and for **both** `indexing` values:

            view.default, expand.default, view.default, expand.default

        and nothing else. No transpose, no permute -- which is the measurement
        that pins how `"xy"` is done, because the obvious implementation of
        `"xy"` is a transpose of the `"ij"` result and a transpose would have
        appeared in that trace.

        `"xy"` is `"ij"` with the **first two inputs swapped and the first two
        outputs swapped back**. Measured: `meshgrid(arange(3), arange(4),
        indexing="xy")` gives two tensors of shape `(4, 3)`, which is what the
        swap produces -- and also what a transpose would produce. The two agree
        on shapes and on values; only the op trace tells them apart, so
        following the trace rather than the shapes is what keeps this composite
        honest about the ops a caller's capture will see.

        A **0-d** input counts as extent 1, measured
        (`meshgrid(tensor(1), arange(4))` gives two `(1, 4)`), which falls out
        of the shape being empty rather than needing a case.

        `indexing=None` means `"ij"`, with a `UserWarning` upstream that this
        will become required. The warning is upstream's C++ and is not
        reproduced; the default is.
        """
        if isinstance(tensors, module.TensorBase):
            tensors = [tensors]
        else:
            tensors = list(tensors)
            # `torch.meshgrid(a, b)` reaches `torch/functional.py` as a tuple
            # of tensors and `torch.meshgrid([a, b])` as a one-element tuple
            # holding the list; upstream unwraps the second form, so this does.
            if len(tensors) == 1 and not isinstance(tensors[0], module.TensorBase):
                tensors = list(tensors[0])
        if indexing is None:
            indexing = "ij"
        if indexing not in ("ij", "xy"):
            raise RuntimeError(
                'torch.meshgrid: indexing must be one of "xy" or "ij", '
                f"but received: {indexing}"
            )

        order = list(range(len(tensors)))
        if indexing == "xy" and len(tensors) >= 2:
            order[0], order[1] = order[1], order[0]
        swapped = [tensors[i] for i in order]

        sizes = []
        for t in swapped:
            shape = list(t.shape)
            if len(shape) > 1:
                raise RuntimeError(
                    "torch.meshgrid: Expected 0D or 1D tensor in the tensor "
                    f"list but got: {t}"
                )
            sizes.append(shape[0] if shape else 1)

        grids = []
        for axis, t in enumerate(swapped):
            view_shape = [1] * len(swapped)
            view_shape[axis] = sizes[axis]
            grids.append(
                dispatch(
                    "aten.expand.default",
                    dispatch("aten.view.default", t, view_shape),
                    list(sizes),
                )
            )
        if indexing == "xy" and len(grids) >= 2:
            grids[0], grids[1] = grids[1], grids[0]
        return tuple(grids)

    meshgrid.__name__ = meshgrid.__qualname__ = "meshgrid"
    meshgrid.__module__ = "torch._C"
    setattr(varfns, "meshgrid", meshgrid)


def _install_behaviour(module, dispatch, transcribed) -> None:
    """The names that have to *do* something for the import to finish.

    `transcribed` is `(qualname, overload) -> _Schema` for every entry of
    `overloads.json` and `methods.json` -- see `_get_schema` below for why the
    overload tables are also a schema source.
    """

    # `torch/backends/cudnn/__init__.py:223` builds `enabled =
    # ContextProp(torch._C._get_cudnn_enabled, torch._C._set_cudnn_enabled)`
    # at import time, and `torch/nn/functional.py:2987`'s `F.layer_norm` reads
    # `torch.backends.cudnn.enabled` on every call to pass it through as
    # `aten::layer_norm`'s `cudnn_enable` argument (which this shim's
    # `layer_norm` composite, `_install_composites`, accepts and ignores --
    # there is no cudnn backend here regardless of the flag). `_has_cudnn` is
    # `False` (see `_BUILD_FLAGS` below), but that only answers
    # `is_available()`; the getter/setter pair is a plain module-level boolean
    # state cell and has to work whether or not a backend exists, or
    # `F.layer_norm` never gets past reading the flag. Measured wall,
    # docs/models/GPT2.md.
    _cudnn_enabled_cell = [True]  # upstream's own default

    def _get_cudnn_enabled():
        return _cudnn_enabled_cell[0]

    def _set_cudnn_enabled(value):
        _cudnn_enabled_cell[0] = bool(value)

    module._get_cudnn_enabled = _get_cudnn_enabled
    module._set_cudnn_enabled = _set_cudnn_enabled

    # The determinism flags -- the same shape of state cell as `cudnn_enabled`
    # above, and the wall `bert` stopped on (docs/architectures/ARCH20.md §2).
    #
    # `F.pad` reads `torch.are_deterministic_algorithms_enabled()` on **every
    # call**, before it does anything else (`torch/nn/functional.py:5806`), and
    # that bottoms out at `_C._get_deterministic_algorithms()`. `bert`'s
    # `tie_weights` pads the output-embedding bias when the head's vocabulary
    # is wider than the tied embedding's, so the model cannot be *constructed*
    # without this name -- the failure was in `from_config`, not in the
    # forward.
    #
    # The defaults are upstream's, read off torch 2.13.0 rather than guessed:
    #
    #     _get_deterministic_algorithms()                    False
    #     _get_deterministic_algorithms_warn_only()          False
    #     _get_deterministic_fill_uninitialized_memory()     True
    #     _get_cudnn_deterministic()                         False
    #     _get_mkldnn_deterministic()                        False
    #
    # Note `fill_uninitialized_memory` is the one that defaults *True*, which
    # is the cell a blanket "all determinism flags start off" would have got
    # wrong. It is only consulted when determinism is on, so it changes nothing
    # here, but the getter is what `torch.utils.deterministic` reads and a
    # wrong constant there is a wrong answer to a question the tree asks.
    #
    # These are plain state cells, not claims. Setting
    # `use_deterministic_algorithms(True)` makes this shim *report* determinism
    # without any kernel changing behaviour -- upstream's flag makes individual
    # kernels select deterministic implementations and raise on the ones with
    # none, and there is no such selection here. Refusing the setter would be
    # worse: `torch/__init__.py:1585` calls it unconditionally from
    # `set_deterministic_debug_mode`, and every kernel in this shim is a single
    # implementation with no nondeterministic sibling to pick instead, so the
    # flag is honest about the only thing it can be honest about -- what it was
    # last set to.
    _deterministic_cells = {
        "algorithms": False,
        "warn_only": False,
        "fill_uninitialized_memory": True,
        "cudnn": False,
        "mkldnn": False,
    }

    def _get_deterministic_algorithms():
        return _deterministic_cells["algorithms"]

    def _set_deterministic_algorithms(mode, warn_only=False):
        # `torch/__init__.py:1534` passes `warn_only` by keyword and
        # `:1585`/`:1589` omit it entirely, so both spellings have to bind.
        _deterministic_cells["algorithms"] = bool(mode)
        _deterministic_cells["warn_only"] = bool(warn_only)

    def _get_deterministic_algorithms_warn_only():
        return _deterministic_cells["warn_only"]

    def _get_deterministic_fill_uninitialized_memory():
        return _deterministic_cells["fill_uninitialized_memory"]

    def _set_deterministic_fill_uninitialized_memory(mode):
        _deterministic_cells["fill_uninitialized_memory"] = bool(mode)

    def _get_cudnn_deterministic():
        return _deterministic_cells["cudnn"]

    def _set_cudnn_deterministic(value):
        _deterministic_cells["cudnn"] = bool(value)

    def _get_mkldnn_deterministic():
        return _deterministic_cells["mkldnn"]

    def _set_mkldnn_deterministic(value):
        _deterministic_cells["mkldnn"] = bool(value)

    module._get_deterministic_algorithms = _get_deterministic_algorithms
    module._set_deterministic_algorithms = _set_deterministic_algorithms
    module._get_deterministic_algorithms_warn_only = (
        _get_deterministic_algorithms_warn_only
    )
    module._get_deterministic_fill_uninitialized_memory = (
        _get_deterministic_fill_uninitialized_memory
    )
    module._set_deterministic_fill_uninitialized_memory = (
        _set_deterministic_fill_uninitialized_memory
    )
    module._get_cudnn_deterministic = _get_cudnn_deterministic
    module._set_cudnn_deterministic = _set_cudnn_deterministic
    module._get_mkldnn_deterministic = _get_mkldnn_deterministic
    module._set_mkldnn_deterministic = _set_mkldnn_deterministic

    def _add_docstr(obj, doc):
        # `torch/_tensor.py` and `torch/_torch_docs.py` call this thousands of
        # times at import. Upstream returns the object; the callers rely on it.
        try:
            obj.__doc__ = doc
        except (AttributeError, TypeError):
            pass
        return obj

    module._add_docstr = _add_docstr

    def _initExtension(shm_manager_path):
        """VENDOR.md wall 2 -- the wall strict mode used to stop at.

        Upstream's does two things worth reproducing: it writes the dtype,
        layout, memory_format and qscheme instances into the *`torch`* module
        (wall 7 -- they are not attributes of `_C` upstream at all), and it
        takes the `torch_shm_manager` path that `_manager_path()` insisted on
        (wall 4) without running it.
        """
        torch_module = sys.modules.get("torch")
        if torch_module is None:
            return
        # Wall 20, and the one that decides whether a model can have
        # parameters at all. Upstream's `_C` never returns a bare `TensorBase`
        # -- `THPVariable_Wrap` instantiates `THPVariableClass`, which is
        # `torch._tensor.Tensor` -- and `torch/nn/parameter.py:54` branches on
        # `type(data) is torch.Tensor`. Get that wrong and `Parameter(...)`
        # takes its custom-tensor path, returns something that is not a
        # `Parameter`, and `nn.Module.__setattr__` files it as a plain
        # attribute: the model builds and has no parameters.
        #
        # This is the right moment for it, and it is upstream's own moment:
        # `torch/__init__.py:1931` runs `from torch._tensor import Tensor` and
        # `:2189` calls this function, so the class exists and no tensor has
        # been made yet.
        tensor_cls = getattr(torch_module, "Tensor", None)
        if tensor_cls is not None:
            module._set_tensor_class(tensor_cls)
            # docs/architectures/DEMAND.md §0.1 rank 1, and the reason this hook is the one
            # that closes it: `torch.Tensor` inherits `__new__` from
            # `TensorBase` and defines none of its own (measured, docs/bindings/CTOR.md
            # §1), so it can be given one here without editing the vendored
            # file that declares it. The guard is on `__new__` being *inherited*
            # rather than on the class being untouched -- if a future vendored
            # tree grows its own `__new__`, that one is upstream's and wins.
            new_fn = getattr(module, "_shim_tensor_new", None)
            if new_fn is not None and "__new__" not in vars(tensor_cls):
                tensor_cls.__new__ = new_fn
                module._shim_tensor_constructor = True
        # The same registration, one type over, and needed for the same kind of
        # `isinstance` check: `torch/storage.py:836` refuses a bare
        # `torch._C.StorageBase` in `TypedStorage(wrap_storage=...)`, and every
        # tensor of every `torch.save` goes through that constructor
        # (`torch/_tensor.py:311`). `torch/__init__.py:1940` imports
        # `UntypedStorage`, well before :2189 calls this.
        storage_cls = getattr(torch_module, "UntypedStorage", None)
        if storage_cls is not None:
            module._set_storage_class(storage_cls)
        kinds = (module.dtype, module.layout, module.memory_format, module.qscheme)
        for name, value in list(vars(module).items()):
            # Filtered on the *type*, not on the leading underscore. Skipping
            # underscored names looked harmless and dropped `torch._mkldnn`,
            # which is a real layout that `torch/_export/serde/serialize.py:171`
            # reads at import of that module -- past the point this shim was
            # being tested, so it would have surfaced much later.
            if isinstance(value, kinds) and not hasattr(torch_module, name):
                setattr(torch_module, name, value)

    module._initExtension = _initExtension

    # VENDOR.md walls 15 and 16: neither of these is behind a `hasattr` gate.
    # `torch/autograd/__init__.py:653` raises if `_autograd_init()` is falsy
    # and `torch/jit/__init__.py:315` does the same for `_jit_init()`, so
    # "inference only, no backward" (DESIGN.md §3 stage 0) does not extend to
    # "do not import autograd". Returning True claims only that initialisation
    # succeeded, which for a shim with no autograd state is true.
    module._autograd_init = lambda: True
    module._jit_init = lambda: True
    module._init_names = lambda *a, **k: None

    # `torch/autograd/function.py:622`, the *first* line of `Function.apply` --
    # so every `torch.autograd.Function` subclass on a model's forward reaches
    # it, whether or not a backward is ever wanted. `bloom`'s
    # `GeLUFunction.apply(x)` is the measured caller (docs/architectures/ARCH20.md §6), once
    # per MLP per layer.
    #
    # `False` is upstream's answer outside a transform (measured on 2.13.0),
    # and it is the only answer this shim can give truthfully: functorch's
    # transforms are `vmap`/`grad`/`jvp`, none of which exists here, so nothing
    # is ever pushed onto the interpreter stack this predicate reports on.
    # `False` is also upstream's *ordinary* branch -- it runs `forward`
    # directly instead of routing through `_functorch.autograd_function`.
    module._are_functorch_transforms_active = lambda: False

    # `_C._FunctionBase.apply` -- what `torch.autograd.Function.apply`
    # delegates to on its ordinary branch (`torch/autograd/function.py:625`,
    # `return super().apply(*args, **kwargs)`).
    #
    # `bloom`'s wall (docs/architectures/ARCH20.md §6). Its `BloomGelu` calls
    # `GeLUFunction.apply(x)` in every MLP, so an inference-only shim reaches
    # `autograd.Function` on a *forward*, not through anything gradient-shaped.
    # `_FunctionBase` is one of the synthesised placeholder types, so it was
    # usable as a base class -- the tree's `class BackwardCFunction(
    # _C._FunctionBase, ...)` imports fine -- and had no `apply`, which
    # surfaced as `AttributeError: 'super' object has no attribute 'apply'`
    # rather than as this shim's own refusal. A `super()` lookup does not go
    # through `_ShimMeta.__getattr__`, so a stub was never going to appear
    # here; it has to be a real entry in the class dict.
    #
    # **What upstream's version does that this does not, stated rather than
    # skipped.** `THPFunction_apply` allocates a graph node, records the input
    # metadata, marks the outputs' `grad_fn`, and handles dirty/
    # non-differentiable marking. All of that is autograd bookkeeping
    # (DESIGN.md §3 stage 0: there is none here), and none of it changes the
    # *value* `forward` returns -- which is the only thing a forward-only shim
    # can observe. So this runs the user's `forward` with a real ctx and
    # returns its result.
    #
    # Both of upstream's two `forward` shapes are honoured, because a model may
    # use either and picking one would silently mis-call the other:
    #
    #     combined:  forward(ctx, *args)                    -- bloom's shape
    #     separate:  forward(*args) + setup_context(ctx, inputs, output)
    #
    # The `setup_context` test is the tree's own `_is_setup_context_defined`,
    # read out of `sys.modules` at call time rather than reimplemented -- same
    # late-binding shape as `_set_generator_metaclass` above, and for the same
    # reason: the predicate belongs to the tree and would drift if copied.
    def _function_base_apply(cls, *args, **kwargs):
        backward_cls = getattr(cls, "_backward_cls", None)
        if backward_cls is None:
            raise NotImplementedError(
                "not implemented in torch._C shim: _FunctionBase.apply on a "
                f"class with no _backward_cls ({cls!r}) -- upstream's version "
                "allocates a graph node, and this shim only reproduces the "
                "forward call that autograd.Function's metaclass sets up"
            )
        ctx = backward_cls()
        # Upstream fills this from the inputs' `requires_grad`. Nothing here
        # requires grad -- `requires_grad=True` is refused at construction --
        # so it is all-False, and it is provided rather than left missing
        # because a `forward` is allowed to read it.
        #
        # It goes into a private slot and not onto `ctx` directly: upstream's
        # `needs_input_grad` is a **read-only getset** on the C node
        # (`type(torch._C._FunctionBase.needs_input_grad)` is
        # `getset_descriptor`), and the shim's placeholder surface reproduces
        # that shape as a `property` -- so a plain assignment raises
        # "property ... has no setter". The property installed below reads this
        # slot, which keeps the attribute read-only from the model's side, as
        # upstream's is.
        ctx._shim_needs_input_grad = tuple(False for _ in args)

        fn_module = sys.modules.get("torch.autograd.function")
        setup_context = getattr(cls, "setup_context", None)
        separate = (
            fn_module is not None
            and setup_context is not None
            and getattr(fn_module, "_is_setup_context_defined", None) is not None
            and fn_module._is_setup_context_defined(setup_context)
        )
        if separate:
            output = cls.forward(*args, **kwargs)
            # Upstream calls this so the ctx a later backward would read is
            # populated. There is no backward here, but the user's
            # `setup_context` may also be where `mark_dirty`/`mark_
            # non_differentiable` are called, and skipping it would make this
            # shim run a *different* forward from upstream's.
            cls.setup_context(ctx, args, output)
            return output
        return cls.forward(ctx, *args, **kwargs)

    _function_base_apply.__name__ = "apply"
    _function_base_apply.__qualname__ = "_FunctionBase.apply"
    _function_base_apply.__module__ = "torch._C"
    module._FunctionBase.apply = classmethod(_function_base_apply)
    module._FunctionBase.needs_input_grad = property(
        lambda self: getattr(self, "_shim_needs_input_grad", ())
    )

    def _multiprocessing_init():
        """VENDOR.md wall 17 -- C writing into a *Python* package's namespace.

        `torch/multiprocessing/__init__.py:37` calls this and `spawn.py:14`
        then imports `_prctl_pr_set_pdeathsig` from the package. That name is
        defined nowhere in the vendored source; only running the tree reveals
        the requirement.
        """
        mp = sys.modules.get("torch.multiprocessing")
        if mp is not None and not hasattr(mp, "_prctl_pr_set_pdeathsig"):
            mp._prctl_pr_set_pdeathsig = _Unimplemented(
                "torch.multiprocessing._prctl_pr_set_pdeathsig"
            )

    module._multiprocessing_init = _multiprocessing_init

    def _set_generator_metaclass(meta):
        """VENDOR.md wall 19, which turns out to be a supported hook.

        The wall was read as a bidirectional coupling -- `_C` having to import
        a metaclass out of the vendored Python tree. It is that, but upstream
        arranged it deliberately: `torch/_prims/rng_prims.py:411` says so in a
        comment ("Late-bind OpaqueBaseMeta as Generator's metaclass ... to
        avoid making torch._C depend on torch._opaque_base at init time") and
        calls this setter. So there is nothing to invert: `_C` exposes the
        setter and the tree hands the metaclass in.

        `Generator` is synthesised with `_ShimMeta`, a heap type, so the
        `__class__` assignment is legal. It would not be if `Generator` were a
        PyO3 type -- `type` is a static type and cannot be reassigned away
        from (measured).
        """
        module.Generator.__class__ = meta

    module._set_generator_metaclass = _set_generator_metaclass

    _install_dispatch_keys(module)
    _install_arg_parser_predicates(module)
    _install_dispatcher_kernel_predicates(module)
    _install_dispatch_suppression(module)
    _install_fx_node_base(module)
    # Seeded with the schemas that exist only in C++ upstream, or only in
    # torchgen's build-time generation -- this tree carries neither -- then
    # added to by every `define()` the tree makes; see
    # `_DispatchLibrary.define`.
    schemas = dict(_TRANSCRIBED_SCHEMAS)
    _install_library(module, schemas)

    # -- op registry ------------------------------------------------------
    def _overload_names(qualname):
        """The overload names of one packet, from every table that knows any.

        **This used to be `["default"]` for every op, and docs/graph/DECOMP.md §3 is
        what that cost.** `torch/_decomp/__init__.py:82` expands a
        packet-level `@register_decomposition(aten.transpose)` by calling
        `packet.op_overloads()`, which walks exactly this list -- so with
        `["default"]` in it, upstream's rule for `transpose` landed on
        `aten.transpose.default`, an overload that does not exist. The rule was
        in the tree, under a name nothing would ever look up. Measured: 592
        registry entries here against upstream's 1097, and 525 of ours ended in
        `.default` against upstream's 456.

        Three sources, unioned, because no one of them is complete:

        1. `native_functions.yaml`, which is authoritative for the 1554 aten
           names it declares and says nothing about the other 176;
        2. `schemas` -- `_TRANSCRIBED_SCHEMAS` plus every `Library.define()`
           the tree has made so far, which is where `prims::` and
           `_c10d_functional::` overloads come from and where the generated
           `.out` variants the file does not carry come from;
        3. `transcribed` -- `overloads.json` and `methods.json`.

        Falling back to `["default"]` when all three are silent keeps the
        registry open (docs/bindings/SCHEMA.md §12): a name nobody has declared still
        yields a callable, and `_aten_dispatch` refuses it at call time with
        the op named. What is *not* done is the reverse -- adding `default` to
        a packet whose overloads are known. That is the bug being fixed, and
        re-adding it "for safety" would put every packet-level rule back on the
        non-existent key.
        """
        known: list = []
        for overload in _aten_overload_names(qualname):
            if overload not in known:
                known.append(overload)
        for table in (schemas, transcribed):
            for entry_name, entry_overload in table:
                if entry_name == qualname and entry_overload not in known:
                    known.append(entry_overload)
        return known or ["default"]

    def _jit_get_operation(qualname):
        if "::" not in qualname:
            raise RuntimeError(f"torch._C shim: not a qualified op name: {qualname}")
        name = qualname.split("::", 1)[1]
        if _is_refused_op_name(name) or _is_absent_op(qualname):
            raise RuntimeError(f"torch._C shim: no operator {qualname}")
        return _op_callable(dispatch, qualname, ""), _overload_names(qualname)

    module._shim_overload_names = _overload_names

    def _get_operation_overload(qualname, overload):
        name = qualname.split("::", 1)[-1]
        if _is_refused_op_name(name) or _is_absent_op(qualname):
            return None
        op = _op_callable(dispatch, qualname, overload)

        def op_dk(dispatch_key, *args, **kwargs):
            return op(*args, **kwargs)

        return op, op_dk, _aten_tags(module, qualname, overload)

    module._shim_unknown_tags = lambda: sorted(_UNKNOWN_TAGS)

    # Handed out when there is no text. Kept so that the ones given away can be
    # listed -- `_shim_placeholder_schemas()` is to schema text what
    # `_shim_registrations` is to `Library.impl`: the size of the gap, readable
    # from Python instead of inferred from what is absent.
    placeholders: dict = {}

    def _is_absent_op(qualname):
        """`_is_absent_inplace_variant`, minus anything actually registered.

        A `Library.define()` for `<base>_` -- or a transcribed table entry --
        is evidence the op exists that the file does not have, and it wins.
        """
        if not _is_absent_inplace_variant(qualname):
            return False
        return not any(name == qualname
                       for table in (schemas, transcribed)
                       for name, _ in table)

    def _schema_route(qualname, overload):
        """`(source name, schema)` -- four sources, in this order.

        1. `registered` -- the `_c10d_functional` family and the generated aten
           schemas, which exist only in C++ or in torchgen upstream, plus every
           `Library.define()` the tree made at import. None of these is in
           `native_functions.yaml`; `verify_schemas.py` checks that, because an
           entry the file *does* carry would silently shadow the file.
        2. `native_functions.yaml`, re-printed (`_normalise_schema_text`).
        3. `tables` -- `overloads.json` and `methods.json`, whose strings are
           `str(op._schema)` copied from upstream 2.13.0. Four of their 173
           overloads are `.out` variants torchgen synthesises, which the file
           does not carry; the other 169 are also in the file, and reaching them
           here would mean the file failed to answer.

           **The order of 2 and 3 is load-bearing and was the other way round.**
           The tables are the oracle `test_schema_text_survives_the_round_trip_
           through_the_transcribed_tables` re-prints against, and while they
           were consulted first they *answered* those 173 lookups -- so the test
           compared the tables with themselves and passed with the float printer
           deleted (measured). A check that cannot fail is not a check.
        4. A placeholder, which answers `arguments`/`returns` with nothing and
           records every predicate it is asked.

        Before this there was only 1 and 4, so every one of the 117 implemented
        aten ops took route 4 -- and docs/distributed/DISTRIBUTED.md §8.1 is what that cost.
        """
        # `""` is upstream's spelling of "the default overload" and is what
        # `_Schema.parse` puts in `overload_name`; `"default"` is what
        # `torch/_library/effects.py:55` passes. One key for both.
        key = (qualname, "" if overload == "default" else overload)
        known = schemas.get(key)
        if known is not None:
            return "registered", known
        known = _aten_schema(*key)
        if known is not None:
            return "native_functions.yaml", known
        known = transcribed.get(key)
        if known is not None:
            return "tables", known
        if key not in placeholders:
            placeholders[key] = _Schema(*key, placeholder=True)
        return "placeholder", placeholders[key]

    def _get_schema(qualname, overload):
        return _schema_route(qualname, overload)[1]

    def _shim_schema_source():
        return _native_functions_source()

    def _shim_schema_provenance(qualname, overload=""):
        """Which of the four sources answered, without reading the artefact.

        The layering is invisible from the outside -- every route returns a
        `_Schema` and they mostly agree -- so "the file answered this" is not
        checkable by looking at the text. It has to be askable, or a reordering
        that quietly stops consulting the file passes every test that compares
        text.
        """
        return _schema_route(qualname, overload)[0]

    def _shim_placeholder_schemas():
        return sorted(placeholders)

    def _shim_unanswered_predicates():
        return sorted(_UNANSWERED_PREDICATES)

    module._shim_schema_source = _shim_schema_source
    module._shim_schema_provenance = _shim_schema_provenance
    module._shim_placeholder_schemas = _shim_placeholder_schemas
    module._shim_unanswered_predicates = _shim_unanswered_predicates

    # A plain function, not `_Schema.parse` itself: `torch/__init__.py:1091`
    # walks every public name in `dir(_C)` and assigns `__module__` on each
    # callable, and a bound classmethod has no writable `__module__`.
    def parse_schema(text):
        return _Schema.parse(text)

    module.parse_schema = parse_schema
    module.FunctionSchema = _Schema
    module.Argument = _Argument
    module.AliasInfo = _AliasInfo

    # -- names discovered by running the tree ------------------------------
    #
    # Everything below is here because `import torch` demanded it, at the file
    # and line noted. None of it is reachable by reading the vendored source
    # for `torch._C.<name>`: these are the calls whose *return value* is used,
    # as opposed to the vast majority which are only looked up.
    for name, value in _DISCOVERED_RETURNS.items():
        setattr(module, name, _constant_function(f"torch._C.{name}", value))
    for name, value in _BUILD_FLAGS.items():
        setattr(module, name, value)
    for dotted, value in _DISCOVERED_TYPE_RETURNS.items():
        type_name, member = dotted.split(".", 1)
        owner = getattr(module, type_name, None)
        if owner is not None:
            setattr(owner, member, staticmethod(
                _constant_function(f"torch._C.{dotted}", value)))

    module._jit_get_operation = _jit_get_operation
    module._get_operation_overload = _get_operation_overload
    module._get_schema = _get_schema

    _install_autocast(module)
    _install_default_generator(module)
    _install_backend_flag_toggles(module)
    _install_functionality_to_backend_keys(module)
    _install_dispatch_key_set(module)



# Upstream's own device vocabulary for the autocast entry points, transcribed
# from the refusal it raises (`torch._C.is_autocast_enabled("nosuch")` on
# 2.13.0). It is *not* the same list `torch.device` accepts -- `opengl`,
# `ideep`, `ve` and `fpga` are autocast-only -- so it cannot be borrowed from
# `PyDevice::resolve` and is written out here instead.
_AUTOCAST_DEVICE_TYPES = (
    "cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip",
    "ve", "fpga", "maia", "xla", "lazy", "vulkan", "mps", "meta", "hpu",
    "mtia", "privateuseone",
)


def _refuse_unrepresentable_memory_format(op, kwargs):
    """Pop `memory_format`, and **refuse the two this build cannot produce.**

    `contiguous_format` and `preserve_format` are accepted and dropped, which is
    honest: every tensor this shim can build is contiguous, so both of them ask
    for what the result already is.

    `channels_last` and `channels_last_3d` were being dropped in the same
    breath, and that was a silent wrong answer on the public surface --
    `x.to(memory_format=torch.channels_last)` returned a contiguous tensor and
    said nothing, so a caller held something it believed was channels-last and
    every later `is_contiguous(memory_format=channels_last)` disagreed with what
    it had asked for.

    It was found by `test_export5.py::test_channels_last_is_false_as_a_fact_
    because_the_build_cannot_make_one`, which is a test written to check the
    *premise* of an answer rather than the answer -- the premise being "no
    tensor in this build can be in that layout" (docs/graph/EXPORT5.md §3). The
    premise was false by way of this door, and nothing else in the suite could
    have noticed, because dropping an argument raises nothing.

    candle carries a `Layout` and no memory-format tag, and no kernel here reads
    one, so there is no representation to return. Refusing is the only answer
    that is not a claim.
    """
    fmt = kwargs.pop("memory_format", None)
    if fmt is None:
        return
    name = getattr(fmt, "_shim_name", None) or str(fmt)
    name = name.rsplit(".", 1)[-1]
    if name in ("contiguous_format", "preserve_format"):
        return
    raise NotImplementedError(
        f"not implemented in torch._C shim: {op}(memory_format=torch.{name}) -- "
        f"this build has no channels-last representation at all (candle carries "
        f"a Layout and no memory-format tag, and no kernel here reads one), so "
        f"the request cannot be honoured. It is refused rather than dropped: "
        f"dropping it returned a contiguous tensor while the caller believed it "
        f"had asked for another layout (bootstrap.py, docs/graph/EXPORT5.md §3)"
    )


def _install_backend_flag_toggles(module) -> None:
    """The `_get_*` / `_set_*` pairs behind `torch.backends.<name>.flags()`.

    `torch.export` enters several of these context managers before it traces
    anything, so each was a raising stub that stopped export outright
    (docs/graph/EXPORT5.md §4).  They are *not* new capabilities: every one of them
    reports on a backend `_BUILD_FLAGS` above already answers `False` for, and
    the point of implementing them is that a flag pair is the one shape where
    a stub and a lie are easy to confuse.

    **Every getter here is derived from `_BUILD_FLAGS`, not from upstream's
    answer**, and on two of them those differ -- which is the whole reason this
    docstring exists rather than a table of transcribed constants:

      * `_get_mkldnn_enabled()` is `True` on upstream 2.13.0 *on this machine*,
        where `torch.backends.mkldnn.is_available()` is `False`.  Upstream's
        flag is a user **preference** that survives the backend being absent.
        Here it is `False`, because `_has_mkldnn` is `False` and there is no
        MKLDNN to prefer; reporting `True` would say a backend exists.
      * `_get_cudnn_allow_tf32()` is `True` upstream for the same reason, and
        `False` here for the same reason -- `_has_cudnn` is `False`.

    `_get_onednn_allow_tf32()` answers `None`, and that is upstream's own
    answer measured on a build without oneDNN, not an invention.

    **Every setter accepts the value the getter already reports and refuses
    every other value by name.**  That asymmetry is the load-bearing part.  A
    setter that quietly accepted `True` would let
    `torch.backends.mkldnn.flags(_enabled=True)` return having changed nothing,
    which is `docs/graph/COMPILE.md` §5's silent-fallback shape and exactly what
    `_len_torch_dispatch_stack`'s constant `0` did (docs/graph/EXPORT.md §2.2): a
    block that entered, reported itself absent, and changed nothing.  Refusing
    means a caller that really wants MKLDNN is told it is not here.

    The accept-the-current-value half is not a loophole, it is what makes the
    save/restore pattern work: `torch/backends/mkldnn/__init__.py::set_flags`
    reads the four getters on entry and writes them back on exit, so the exit
    write is always a no-op and must not raise.
    """

    def _toggle(name, current, *, kind="flag"):
        """A getter/setter pair reporting `current` and refusing to change it."""

        def getter():
            return current

        def setter(value):
            if bool(value) == bool(current):
                return
            raise NotImplementedError(
                f"torch._C._set_{name}({value!r}): this build has no such "
                f"backend -- torch._C._has_{name.split('_')[0]} and the "
                f"_BUILD_FLAGS table above both answer that it is absent, so "
                f"the {kind} cannot be turned on. Refused rather than accepted "
                f"and ignored, which would let torch.backends flags() return "
                f"having changed nothing (bootstrap.py, docs/graph/EXPORT5.md §4)"
            )

        return getter, setter

    for flag, current in (
        ("mkldnn_enabled", False),
        ("mkldnn_deterministic", False),
        ("nnpack_enabled", False),
        ("cudnn_benchmark", False),
        ("cudnn_allow_tf32", False),
    ):
        getter, setter = _toggle(flag, current)
        setattr(module, f"_get_{flag}", getter)
        setattr(module, f"_set_{flag}", setter)

    # oneDNN's tf32 flag answers `None` rather than `False`, and that is
    # upstream's own answer on a build without oneDNN (measured on 2.13.0,
    # darwin/arm64: `torch._C._get_onednn_allow_tf32()` is `None`). `None` here
    # means "not applicable", which is a different claim from "off", and this
    # build is in the same position that answer describes.
    def _get_onednn_allow_tf32():
        return None

    def _set_onednn_allow_tf32(value):
        if not value:
            return
        raise NotImplementedError(
            "torch._C._set_onednn_allow_tf32(True): there is no oneDNN in this "
            "build, so there is no tf32 path to allow. The getter answers None "
            "-- not applicable -- which is upstream's own answer on a build "
            "without oneDNN (bootstrap.py, docs/graph/EXPORT5.md §4)"
        )

    module._get_onednn_allow_tf32 = _get_onednn_allow_tf32
    module._set_onednn_allow_tf32 = _set_onednn_allow_tf32

    # cuDNN's depthwise-convolution kernel selector is a *string*, not a bool
    # ("auto" on 2.13.0), so it does not fit `_toggle` and gets the same
    # accept-what-you-report / refuse-the-rest treatment spelled out.
    def _get_cudnn_depthwise_kernel():
        return "auto"

    def _set_cudnn_depthwise_kernel(value):
        if value == "auto":
            return
        raise NotImplementedError(
            f"torch._C._set_cudnn_depthwise_kernel({value!r}): there is no "
            f"cuDNN in this build, so there is no depthwise kernel to select. "
            f"Only 'auto' -- the value the getter reports, so that "
            f"torch.backends' save/restore round trip does not raise -- is "
            f"accepted (bootstrap.py, docs/graph/EXPORT5.md §4)"
        )

    module._get_cudnn_depthwise_kernel = _get_cudnn_depthwise_kernel
    module._set_cudnn_depthwise_kernel = _set_cudnn_depthwise_kernel

    # `_get_fp32_precision_getter(backend, op)` / `_set_fp32_precision_setter(
    # backend, op, precision)`. Upstream answers "none" for every (backend, op)
    # pair on this machine, measured; "none" means full fp32 with no reduced
    # -precision substitution, which is exactly what every kernel in this shim
    # does -- there is no tf32, no bf16 accumulation and no reduced-precision
    # GEMM path anywhere in aten.rs. So "none" is a fact about this build and
    # not a copy of upstream's default.
    def _get_fp32_precision_getter(backend, op):
        return "none"

    def _set_fp32_precision_setter(backend, op, precision):
        if precision == "none":
            return
        raise NotImplementedError(
            f"torch._C._set_fp32_precision_setter({backend!r}, {op!r}, "
            f"{precision!r}): every kernel in this shim computes fp32 in fp32 "
            f"-- there is no tf32, bf16-accumulation or other reduced-precision "
            f"path to select -- so only 'none' can be honoured. Refused rather "
            f"than accepted and ignored, which would silently promise a "
            f"precision change that never happens (bootstrap.py, "
            f"docs/graph/EXPORT5.md §4)"
        )

    module._get_fp32_precision_getter = _get_fp32_precision_getter
    module._set_fp32_precision_setter = _set_fp32_precision_setter


def _install_dispatch_key_set(module) -> None:
    """`torch._C._dispatch_key_set(tensor)` -- **a string, and device-only.**

    `fake_tensor.py:2083` is the only reader, and it does exactly one thing with
    the result:

        synth_key_set = torch._C._dispatch_key_set(synth_output)
        key_set = torch._C._dispatch_key_set(output)
        if synth_key_set != key_set:
            raise _BypassDispatchCache("dispatch_key_set mismatch")

    So the contract is an **equality token**: two tensors that dispatch the same
    way must compare equal and two that do not must not.  It returns a `str`
    upstream, not a `DispatchKeySet` -- measured on 2.13.0, and worth stating
    because the name says otherwise and a `DispatchKeySet` would have been the
    obvious wrong guess.

    **The rule was measured, not reasoned.**  Ten tensors varying dtype
    (float32/float64/int64/bool/complex64), rank, `requires_grad` and
    view-ness across CPU and meta produced exactly two distinct answers, and
    the only input that changed the answer was the **device**:

        CPU   ->  DispatchKeySet(CPU, ADInplaceOrView, AutogradCPU, AutocastCPU)
        meta  ->  DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)

    Note what is *not* in the rule.  `requires_grad=True` does not change it --
    the Autograd key is on the tensor whether or not it is set -- and neither
    does dtype.  A version keyed on `requires_grad` would have looked more
    thorough and been wrong, and it would have failed only as a *cache miss*,
    which is a silent slowdown rather than an error.  And meta carries **no**
    Autocast key while CPU does, which is the asymmetry a hand-written table
    would most likely have smoothed over.

    The strings are upstream's own spelling.  Equality is all the caller needs,
    so a different spelling would work; matching means a human reading a debug
    log sees the same text on both sides.

    Only the two devices this shim has are answered, and **every other device
    refuses by name** rather than being given a plausible key set -- the same
    reasoning as `_dispatch_has_computed_kernel_for_dispatch_key`
    (docs/graph/EXPORT4.md §6.3): a wrong key set here is a silently wrong cache
    entry, and this shim has no kernels behind those devices to describe.
    """

    def _dispatch_key_set(tensor):
        device = getattr(getattr(tensor, "device", None), "type", None)
        if device == "cpu":
            return "DispatchKeySet(CPU, ADInplaceOrView, AutogradCPU, AutocastCPU)"
        if device == "meta":
            return "DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)"
        raise NotImplementedError(
            f"torch._C._dispatch_key_set: this shim can describe the dispatch "
            f"keys of a cpu or meta tensor and this one is on {device!r}. The "
            f"answer is used to decide whether two tensors dispatch alike, so a "
            f"guess here would be a silently wrong fake-tensor cache entry "
            f"rather than an error (bootstrap.py, docs/graph/EXPORT5.md §4)"
        )

    module._dispatch_key_set = _dispatch_key_set


def _install_functionality_to_backend_keys(module) -> None:
    """`torch._C._functionality_to_backend_keys(k)` -- derived from the enum.

    `fake_tensor.py` reaches it while building its dispatch-key set.  Upstream
    returns, for a *functionality* key, the 16 backend-specific keys that
    functionality has -- `Dense` -> `[CPU, CUDA, HIP, XLA, MPS, IPU, XPU, HPU,
    VE, Lazy, MTIA, MAIA, PrivateUse1..3, Meta]`, `Sparse` -> the `Sparse*`
    spellings of the same 16, `AutogradFunctionality` -> the `Autograd*` ones.

    **It is computed from `_install_dispatch_keys`' own enum rather than
    transcribed**, which is the same choice `_should_allow_numbers_as_tensors`
    made in docs/graph/EXPORT4.md §6.2 and for the same reason: a transcribed table
    goes stale silently, and this one is a pure function of a list that already
    exists three thousand lines above.  The backend order is upstream's, read
    off the enum's own numeric order rather than re-listed here.

    **A key that is not a functionality key answers `[key]` -- itself -- and not
    the empty list.** That was written the other way round first, on the
    reasoning that "no backend spellings" meant "nothing to return"; measuring
    upstream showed `CPU -> [CPU]`, `Meta -> [Meta]`, `ADInplaceOrView ->
    [ADInplaceOrView]`, `Undefined -> [Undefined]`, without exception.

    It is not a detail. `torch/utils/_python_dispatch.py::_push_mode` calls this
    with `DispatchKey.PreDispatch` -- which is a plain key, not a functionality
    one -- and uses the result to uncache per-dispatch-key handlers. The empty
    list makes that a cache invalidation that invalidates nothing, which is the
    "entered and changed nothing" shape docs/graph/EXPORT.md §2.2 is about, on the
    export path. It was found by
    `test_export5.py::test_functionality_to_backend_keys_matches_upstream_key_for_key`,
    which exists because nullifying this whole function went **uncaught**
    (docs/graph/EXPORT5.md §11).
    """
    DispatchKey = module.DispatchKey

    #: The 16 backend suffixes, in upstream's own enum order.
    _BACKENDS = (
        "CPU", "CUDA", "HIP", "XLA", "MPS", "IPU", "XPU", "HPU", "VE",
        "Lazy", "MTIA", "MAIA", "PrivateUse1", "PrivateUse2", "PrivateUse3",
        "Meta",
    )

    #: functionality key name -> how a backend spelling of it is written.
    _SPELLINGS = {
        "Dense": lambda b: b,
        "Quantized": lambda b: f"Quantized{b}",
        "Sparse": lambda b: f"Sparse{b}",
        "SparseCsr": lambda b: f"SparseCsr{b}",
        "NestedTensor": lambda b: f"NestedTensor{b}",
        "AutogradFunctionality": lambda b: f"Autograd{b}",
    }

    def _functionality_to_backend_keys(key):
        spell = _SPELLINGS.get(getattr(key, "name", None))
        if spell is None:
            return [key]
        out = []
        for backend in _BACKENDS:
            member = getattr(DispatchKey, spell(backend), None)
            if member is not None:
                out.append(member)
        return out

    module._functionality_to_backend_keys = _functionality_to_backend_keys


# ---------------------------------------------------------------------------
# The `torch.export` census names -- docs/graph/EXPORT.md §8's hand-off, paid.
# ---------------------------------------------------------------------------
#
# These 32 `torch._C` names lived in
# `torchnative/src/main/torchnative/export/upstream.py`, which installed them
# by monkey-patching `torch._C` *after* `import torch`. docs/graph/EXPORT.md §8 said
# that was the wrong home and gave the patch; docs/graph/EXPORT4.md §10 listed paying
# it as the next mechanical task. This is it.
#
# **The reason it had to move is not tidiness.** Patching after `import torch`
# forces a `rebind()` pass over roughly forty `from torch._C import ...`
# bindings that other torch modules have already made -- including aliases like
# `torch/utils/_mode_utils.py:15`'s `no_dispatch = torch._C._DisableTorchDispatch`,
# which is why that pass had to match by object identity rather than by name.
# Every one of those bindings is made *after* this file runs, so all of it
# disappears here. `rebind`, `install`, `InstallReport`, `installed_names`,
# `_is_ours`, `_mark_ours` and `_MARK` existed only to describe and undo a
# runtime patch and have no meaning in a bootstrap; they are dropped rather
# than carried.
#
# **What was measured before this moved, and is the reason it matters:**
# `rust/torch_c/pytests/export_sweep.py` run against the shim stopped at census
# name #0 (`torch._C._unset_dispatch_mode`) on all 25 architectures, because
# the sweep's subprocess never called `upstream.install()`. Every export
# measurement in docs/graph/EXPORT.md and docs/graph/EXPORT4.md was taken under that
# monkey-patch, and docs/graph/EXPORT4.md §1.1 said so. After this, the names are
# simply there.
#
# The functions keep their `(C, put)` signature and are copied **verbatim**
# rather than rewritten to take `module` alone. docs/graph/EXPORT.md §8 proposed the
# rewrite, and the verbatim copy is preferred here for one reason: this is a
# move of nine hundred lines of working, tested behaviour, and a move that
# changes no line cannot change behaviour. `_put` below is the four-line
# adapter that buys that. Renaming and re-signaturing them is a separate,
# reviewable change and should not ride along inside a bisect boundary that is
# supposed to mean "the same code, in its final home".
#
# `set_eval_frame` is not touched -- docs/graph/EXPORT.md §8 item 4, and
# docs/graph/COMPILE.md §5.1 is why its refusal must outlive any symbol-filling on
# this path. Nothing below goes near it.


def _put(owner, name, value, qual=None, only_if_stub=False):
    """`upstream.py`'s `put`, with the bookkeeping removed.

    That module installed every name unconditionally and reported which ones
    had been placeholders, because it was describing a runtime patch that could
    be inspected and undone. Here there is no report to keep and no patch to
    undo: this is where the names are defined.

    `only_if_stub` survives because one caller needs it --
    `_functorch.is_functorch_wrapped_tensor` is already implemented elsewhere in
    this file and must not be displaced (docs/graph/EXPORT.md §2.2, row 29).
    """
    if only_if_stub and not _is_stub(getattr(owner, name, None)):
        return
    setattr(owner, name, value)


class _Tls(threading.local):
    """Per-thread dispatcher bookkeeping.

    Upstream keeps these in C++ TLS (`c10::impl::LocalDispatchKeySet`, the
    `TorchDispatchModeTLS` stack).  `threading.local` is the same lifetime with
    the same visibility rules, which matters: `torch/utils/_python_dispatch.py`
    enters and exits these in `with` blocks on whatever thread is running, and
    a module-level global would leak one thread's mode stack into another's.
    """

    def __init__(self) -> None:
        self.included = None      # DispatchKeySet, filled on first use
        self.excluded = None      # DispatchKeySet
        self.mode_stack = []      # user dispatch modes, innermost last
        self.infra_modes = {}     # _TorchDispatchModeKey -> mode
        self.only_lift_cpu_tensors = False
        self.reapply_views = False
        self.meta_in_tls_dispatch_include = False
        self.inference_mode = False


_TLS = _Tls()


def _keyset(module):
    return module.DispatchKeySet


def _included(module):
    if _TLS.included is None:
        _TLS.included = _keyset(module)()
    return _TLS.included


def _excluded(module):
    if _TLS.excluded is None:
        _TLS.excluded = _keyset(module)()
    return _TLS.excluded


def _is_stub(obj) -> bool:
    """Is this name a placeholder rather than an implementation?

    Three shapes count, and the bootstrap makes all three.  `_Unimplemented` is
    what it leaves when the stubs say nothing about a name.  `_make_function`
    leaves an ordinary Python function whose body raises `NotImplementedError`.
    `_make_property` leaves a `property` whose getter does the same.  None of
    them may be *called* to find out -- calling raises, and `__bool__` on an
    `_Unimplemented` raises too -- so this reads the code object's constants
    instead of probing behaviour.
    """
    if obj is None:
        return True
    # `upstream.py`'s version had an `_is_ours` branch here, for objects that
    # module had itself installed. There is no such thing in the bootstrap:
    # this runs once, before anything could have been installed, so the
    # question cannot arise and the branch is dropped rather than carried.
    if type(obj).__name__ == "_Unimplemented":
        return True
    if isinstance(obj, property):
        return _is_stub(obj.fget)
    try:
        code = getattr(obj, "__code__", None)
    except Exception:
        # `torch/_classes.py` synthesises attributes on access and raises for
        # anything unregistered.  A module that answers every name is not a
        # module whose bindings need repointing.
        return False
    if code is None:
        return False
    return any(
        isinstance(c, str) and "not implemented in torch._C shim" in c
        for c in code.co_consts
    )


def _install_tls(C, put) -> None:
    """`_dispatch_tls_*`, the four names round 19 died on.

    Round 18's `_dispatch_tls_local_exclude_set` returning `None` is the whole
    of COMPILE.md's stopping point: `meta_utils.py:1061` does
    ``...local_exclude_set().has(DispatchKey.ADInplaceOrView)``.  A real
    `DispatchKeySet` -- which `_install_dispatch_keys` already builds -- answers
    it, and the answer is `False`, which is the truth for a shim that has
    entered no guard.
    """
    KeySet = C.DispatchKeySet

    def _dispatch_tls_local_include_set():
        return _included(C)

    def _dispatch_tls_local_exclude_set():
        return _excluded(C)

    def _dispatch_tls_is_dispatch_key_included(key):
        return _included(C).has(key)

    def _dispatch_tls_is_dispatch_key_excluded(key):
        return _excluded(C).has(key)

    def _dispatch_tls_set_dispatch_key_included(key, included):
        """The write half of `_dispatch_tls_is_dispatch_key_included`.

        `DispatchKeySet` is immutable here (bootstrap.py builds it on a
        `frozenset`, and `add`/`remove` return new sets), so this rebinds
        `_TLS.included` rather than mutating it -- which is also what makes the
        key-state guards' save/restore a matter of holding a reference.

        It is a real write and not a counter: with it inert,
        `_dispatch_tls_is_dispatch_key_included` would keep answering `False`
        for a key something had just included, which is the read/write pair
        disagreeing in the same way `_set_conj` would have.
        """
        _TLS.included = (
            _included(C).add(key) if included else _included(C).remove(key)
        )

    def _functionalization_reapply_views_tls():
        # Upstream: whether the functionalization pass should re-apply view ops
        # rather than materialise copies.  A flag read by
        # `torch/_subclasses/functional_tensor.py`; nothing here sets it, so
        # `False` is the state, not a stand-in.
        return _TLS.reapply_views

    def _meta_in_tls_dispatch_include():
        return _TLS.meta_in_tls_dispatch_include

    def _set_meta_in_tls_dispatch_include(value):
        _TLS.meta_in_tls_dispatch_include = bool(value)

    put(C, "_dispatch_tls_local_include_set", _dispatch_tls_local_include_set)
    put(C, "_dispatch_tls_local_exclude_set", _dispatch_tls_local_exclude_set)
    put(C, "_dispatch_tls_is_dispatch_key_included",
        _dispatch_tls_is_dispatch_key_included)
    put(C, "_dispatch_tls_is_dispatch_key_excluded",
        _dispatch_tls_is_dispatch_key_excluded)
    put(C, "_dispatch_tls_set_dispatch_key_included",
        _dispatch_tls_set_dispatch_key_included)
    put(C, "_functionalization_reapply_views_tls",
        _functionalization_reapply_views_tls)
    put(C, "_meta_in_tls_dispatch_include", _meta_in_tls_dispatch_include)
    put(C, "_set_meta_in_tls_dispatch_include", _set_meta_in_tls_dispatch_include)

    class _ForceDispatchKeyGuard:
        """`with _ForceDispatchKeyGuard(include, exclude):` -- set both, restore both."""

        __module__ = "torch._C"

        def __init__(self, include=None, exclude=None):
            self._include = include
            self._exclude = exclude
            self._saved = None

        def __enter__(self):
            self._saved = (_included(C), _excluded(C))
            if self._include is not None:
                _TLS.included = KeySet(self._include)
            if self._exclude is not None:
                _TLS.excluded = KeySet(self._exclude)
            return self

        def __exit__(self, *exc):
            _TLS.included, _TLS.excluded = self._saved
            return False

    class _ExcludeDispatchKeyGuard:
        __module__ = "torch._C"

        def __init__(self, keyset):
            self._keyset = keyset
            self._saved = None

        def __enter__(self):
            self._saved = _excluded(C)
            _TLS.excluded = self._saved | KeySet(self._keyset)
            return self

        def __exit__(self, *exc):
            _TLS.excluded = self._saved
            return False

    class _IncludeDispatchKeyGuard:
        __module__ = "torch._C"

        def __init__(self, key):
            self._key = key
            self._saved = None

        def __enter__(self):
            self._saved = _included(C)
            _TLS.included = self._saved | KeySet(self._key)
            return self

        def __exit__(self, *exc):
            _TLS.included = self._saved
            return False

    put(C, "_ForceDispatchKeyGuard", _ForceDispatchKeyGuard)
    put(C, "_ExcludeDispatchKeyGuard", _ExcludeDispatchKeyGuard)
    put(C, "_IncludeDispatchKeyGuard", _IncludeDispatchKeyGuard)


def _install_mode_stack(C, put) -> None:
    """`_push_on_torch_dispatch_stack` and friends, as a real stack.

    `_len_torch_dispatch_stack` **used to be** the constant `0` in this file's
    `_DISCOVERED_RETURNS` table, with the comment "nothing pushes onto it here".
    That row was deleted when this function moved here (docs/graph/EXPORT.md §8 item
    3); the sentence below is why.  Under `torch.export` something does push: `FakeTensorMode` and
    `ProxyTorchDispatchMode` both push, and `torch/utils/_python_dispatch.py`
    reads the length back to decide whether a mode is active.  A stack that
    always answered zero would make `with FakeTensorMode():` a block that
    entered and changed nothing -- the same shape of silent no-op
    `docs/graph/COMPILE.md` §5 refuses for `torch.compile`.

    Upstream splits the stack in two: *infra* modes (FAKE, PROXY, FUNCTIONAL)
    live in slots keyed by `_TorchDispatchModeKey` and are not part of the
    ordinary stack, while user modes are appended.  `mode._mode_key` is what
    tells the two apart, and this reproduces that split rather than flattening
    it, because `_get_dispatch_mode(key)` has no meaning otherwise.
    """

    def _push_on_torch_dispatch_stack(mode):
        key = getattr(mode, "_mode_key", None)
        if key is not None:
            if _TLS.infra_modes.get(key) is not None:
                raise RuntimeError(
                    f"torch dispatch mode for {key} is already set"
                )
            _TLS.infra_modes[key] = mode
        else:
            _TLS.mode_stack.append(mode)

    def _pop_torch_dispatch_stack(mode_key=None):
        if mode_key is not None:
            popped = _TLS.infra_modes.pop(mode_key, None)
            if popped is None:
                raise AssertionError(
                    f"no torch dispatch mode set for {mode_key}"
                )
            return popped
        if not _TLS.mode_stack:
            raise AssertionError("torch dispatch mode stack is empty")
        return _TLS.mode_stack.pop()

    def _len_torch_dispatch_stack():
        return len(_TLS.mode_stack)

    def _get_dispatch_stack_at(idx):
        return _TLS.mode_stack[idx]

    def _set_dispatch_mode(mode):
        key = getattr(mode, "_mode_key", None)
        if key is None:
            raise AssertionError(
                "_set_dispatch_mode is for infra modes; this one has no _mode_key"
            )
        if _TLS.infra_modes.get(key) is not None:
            raise RuntimeError(f"torch dispatch mode for {key} is already set")
        _TLS.infra_modes[key] = mode

    def _get_dispatch_mode(mode_key):
        return _TLS.infra_modes.get(mode_key)

    def _unset_dispatch_mode(mode_key):
        return _TLS.infra_modes.pop(mode_key, None)

    def _only_lift_cpu_tensors():
        return _TLS.only_lift_cpu_tensors

    def _set_only_lift_cpu_tensors(value):
        _TLS.only_lift_cpu_tensors = bool(value)

    def _ensureCUDADeviceGuardSet():
        # Upstream primes a CUDA device guard so later lifts land on the right
        # device.  There is no CUDA here (`torch._C._has_cuda` is `False`, and
        # the bootstrap's build-flag table says why that is a fact rather than a
        # stand-in), so there is nothing to prime.  Doing nothing is the
        # implementation, not a no-op standing in for one.
        return None

    put(C, "_push_on_torch_dispatch_stack", _push_on_torch_dispatch_stack)
    put(C, "_pop_torch_dispatch_stack", _pop_torch_dispatch_stack)
    put(C, "_len_torch_dispatch_stack", _len_torch_dispatch_stack)
    put(C, "_get_dispatch_stack_at", _get_dispatch_stack_at)
    put(C, "_set_dispatch_mode", _set_dispatch_mode)
    put(C, "_get_dispatch_mode", _get_dispatch_mode)
    put(C, "_unset_dispatch_mode", _unset_dispatch_mode)
    put(C, "_only_lift_cpu_tensors", _only_lift_cpu_tensors)
    put(C, "_set_only_lift_cpu_tensors", _set_only_lift_cpu_tensors)
    put(C, "_ensureCUDADeviceGuardSet", _ensureCUDADeviceGuardSet)


def _is_definitely_a_view(t) -> bool:
    """A *sound positive* view detector, built from the storage model this shim has.

    Measured against upstream on the same three tensors (`docs/graph/EXPORT.md` §3.2):
    `storage_offset`, `stride`, `numel` and `untyped_storage().nbytes()` agree
    exactly between this shim and upstream for `x`, `x[1:, 1:]` and `x.t()`.
    So three signals each *prove* a view:

    * a non-zero `storage_offset` -- the tensor starts inside someone else's
      buffer;
    * a footprint smaller than the storage -- it covers part of a buffer;
    * non-contiguous strides -- the layout was rearranged over a buffer that
      was laid out for something else.

    **What it misses, said plainly:** a view that covers the whole storage
    contiguously -- `x.view(12)`, `x[:]`, `x.reshape(3, 4)` on a contiguous `x`
    -- is bit-for-bit indistinguishable from the base under every signal this
    shim exposes.  Upstream answers `True` there because `TensorImpl` carries a
    base pointer; nothing in `PyTensorBase` does (`rust/torch_c/src/tensor.rs`
    has storage identity via `storage.rs::origin`, but no base *tensor*).  This
    returns `False` for that case, and that is the one wrong answer in the pair.

    Making it right is a Rust change -- a base slot on `PyTensorBase` -- and it
    is outside this document's territory.  `docs/graph/EXPORT.md` §4.2 records it as
    the gap rather than papering it.
    """
    try:
        if t.storage_offset() != 0:
            return True
        if not t.is_contiguous():
            return True
        nbytes = t.untyped_storage().nbytes()
    except Exception:
        # A meta, quantised or vulkan tensor has no storage to ask about
        # (`tensor.rs::no_dense_storage`, `no_host_storage`).  It also has no
        # base, so "not a detected view" is the right answer, not a dodge.
        return False
    return t.numel() * t.element_size() != nbytes


# The `torch_module` parameter `upstream.py`'s copy carried is dropped: it was
# never read in the body (only `C.TensorBase` is), and it existed so that
# `install(torch_module=...)` could pass the module it had been handed. There
# is no such caller here.
def _install_tensor_predicates(C, put) -> None:
    """`_is_view`, `_base`, `is_mkldnn`, `is_inference`, `is_conj`.

    Three of the five are `False` *as a fact about this build*, not as a
    placeholder:

    * `is_mkldnn` -- the bootstrap's build-flag table already answers
      `_has_mkldnn` `False`; a tensor cannot be in a layout the build does not
      have.
    * `is_inference` -- inference mode is an autograd TLS state
      (`InferenceMode`), and this shim has no autograd TLS to be in.
    * `is_conj` -- the conjugate bit is a dispatch-key bit on `TensorImpl`;
      candle has no such bit and no `aten::conj` view op reaches it.

    `_is_view` and `_base` are the pair that is **not** a constant, and they are
    the one place on this path where the shim has to say something it cannot
    fully know.  Views here are real -- `docs/kernels/VIEWS.md` §6 made in-place ops
    write through a layout into shared storage -- so `False` is not free.  The
    arrangement:

    * `_is_view()` answers `True` when `_is_definitely_a_view` proves it, and
      `False` otherwise, missing exactly the full-coverage contiguous case.
    * `_base` **refuses by name** for a tensor that `_is_view()` called `True`,
      because there is no base object to return and `None` there would be read
      as "not a view" by the very caller that just asked.  For everything else
      it is `None`, which is the fact.

    So a module exported with a sliced input fails loudly at the tensor that
    caused it, rather than producing a graph whose inputs quietly lost their
    aliasing.  `docs/graph/COMPILE.md` §5 refuses the same shape of silence for
    `torch.compile`.
    """
    TensorBase = C.TensorBase

    def _is_view(self):
        return _is_definitely_a_view(self)

    def _base_getter(self):
        if _is_definitely_a_view(self):
            raise NotImplementedError(
                "not implemented in torch._C shim: TensorBase._base. This "
                "tensor is a view (non-zero storage offset, non-contiguous "
                "strides, or a footprint smaller than its storage), and the "
                "shim's PyTensorBase carries no base tensor to return. "
                "Returning None here would tell the caller it is not a view, "
                "one line after _is_view() told it that it is."
            )
        return None

    def is_inference(self):
        return False

    def is_conj(self):
        return False

    put(TensorBase, "_is_view", _is_view, "TensorBase._is_view")
    put(TensorBase, "_base", property(_base_getter), "TensorBase._base")
    put(TensorBase, "is_inference", is_inference, "TensorBase.is_inference")
    put(TensorBase, "is_conj", is_conj, "TensorBase.is_conj")
    put(TensorBase, "is_mkldnn", property(lambda self: False),
        "TensorBase.is_mkldnn")

    # `_set_conj` / `_set_neg`, the write halves of two of the bits above.
    #
    # `meta_utils.py:2173` calls `torch._C._set_conj(r, t.is_conj)` on every
    # meta tensor it builds, and `_set_neg` one line later, so `torch.export`
    # reaches both once per fake tensor.
    #
    # **`False` is accepted as a no-op and `True` refuses by name**, and the
    # asymmetry is the whole implementation.  `is_conj` above answers `False`
    # as a *fact* -- there is no conjugate bit on `PyTensorBase` and candle has
    # nothing to carry one -- so `_set_conj(t, False)` is asking for the state
    # the tensor is already in and has nothing to do.  `_set_conj(t, True)`
    # asks for a state this shim cannot represent, and accepting it would leave
    # `is_conj` answering `False` immediately afterwards: a setter whose effect
    # is invisible to its own getter.  That is the shape docs/graph/EXPORT.md §2.2
    # names, so it raises instead.
    #
    # Note this is reachable in practice and not a hypothetical: it is exactly
    # how a conjugated or negated input tensor would arrive at export, and it
    # now stops there by name rather than being traced as if it were neither.
    def _make_bit_setter(bit):
        def setter(tensor, value):
            if not value:
                return
            raise NotImplementedError(
                f"torch._C._set_{bit}(tensor, True): this shim has no {bit} bit "
                f"-- TensorBase.is_{bit} answers False as a fact, not as a stub "
                f"(docs/graph/EXPORT.md §2.3), because candle carries no such flag and "
                f"PyTensorBase has nowhere to put one. Accepting this would make "
                f"the setter invisible to its own getter, so it is refused "
                f"(torchnative/export/upstream.py, docs/graph/EXPORT5.md §5)"
            )
        return setter

    put(C, "_set_conj", _make_bit_setter("conj"))
    put(C, "_set_neg", _make_bit_setter("neg"))


def _install_functorch(C, put) -> None:
    """`is_batchedtensor`, `is_legacy_batchedtensor`, `is_gradtrackingtensor`.

    All three ask "is this tensor wrapped by a functorch transform?".  There is
    no functorch interpreter stack in this shim -- `vmap`, `grad` and `jvp` are
    not implemented -- so no tensor can be one of these wrappers and `False` is
    the fact.  If a functorch layer is ever added, these three are where it
    announces itself, and they are named here so that addition is a change to a
    body rather than the discovery of a hole.
    """
    F = C._functorch

    put(F, "is_batchedtensor", lambda t: False, "_functorch.is_batchedtensor")
    put(F, "is_legacy_batchedtensor", lambda t: False,
        "_functorch.is_legacy_batchedtensor")
    put(F, "is_gradtrackingtensor", lambda t: False,
        "_functorch.is_gradtrackingtensor")
    put(F, "is_functorch_wrapped_tensor", lambda t: False,
        "_functorch.is_functorch_wrapped_tensor", only_if_stub=True)


class _GatheredFrames:
    """The opaque handle `torch._C._profiler.gather_traceback` returns.

    Opaque is the contract: `torch/utils/_traceback.py` never looks inside one.
    It stores the handle on a `CapturedTraceback` and later hands a *list* of
    handles to `symbolize_tracebacks`, which is where the frames become
    readable.  Splitting it that way is upstream's amortisation -- symbolising
    C++ frames is expensive and worth batching -- and reproducing the split,
    rather than returning formatted strings from `gather_traceback`, is what
    keeps `format_all`'s batch path working.
    """

    __slots__ = ("frames",)

    def __init__(self, frames):
        #: innermost first, which is the order `_extract_symbolized_tb`
        #: assumes: it reverses, and it applies `skip` from the *front* to
        #: elide `CapturedTraceback.extract`'s own frame.
        self.frames = frames

    def __repr__(self):
        return f"<CapturedTraceback {len(self.frames)} frames>"


def _install_profiler(C, put) -> None:
    """`gather_traceback` and `symbolize_tracebacks`, to the shape of their caller.

    COMPILE.md's census reached `gather_traceback` at round 14 and no-opped it
    to `None`; the no-op survived because nothing symbolised it in that run.
    With the later rounds real, `torch/_logging/_internal.py:1510` does symbolise
    it, and that is what turned this from "return anything" into a pair with a
    contract:

        gather_traceback(python, script, cpp)  -> opaque handle
        symbolize_tracebacks([handle, ...])    -> [[{filename, line, name}, ...], ...]

    read out of `torch/utils/_traceback.py:180` and `:259`.  `line` is a line
    *number* -- it is passed as `FrameSummary`'s second positional argument --
    and getting that wrong is a `TypeError` several frames away from here,
    which is how it was found.

    `script` and `cpp` are accepted and ignored.  There is no TorchScript
    interpreter and no C++ stack worth naming in this shim, so there are no
    frames of those kinds to omit; Python frames are the whole traceback here
    rather than a subset of one.
    """

    def gather_traceback(python=True, script=False, cpp=False):
        if not python:
            return _GatheredFrames([])
        # `[:-1]` drops this function's own frame: upstream's is C++ and does
        # not appear in the result, and `CapturedTraceback.extract` passes
        # `skip=skip+1` counted from a stack that does not contain it.
        outermost_first = _traceback.extract_stack()[:-1]
        return _GatheredFrames([
            {"filename": f.filename, "line": f.lineno, "name": f.name}
            for f in reversed(outermost_first)
        ])

    def symbolize_tracebacks(to_symbolize):
        return [
            list(t.frames) if isinstance(t, _GatheredFrames) else []
            for t in to_symbolize
        ]

    put(C._profiler, "gather_traceback", gather_traceback,
        "_profiler.gather_traceback")
    put(C._profiler, "symbolize_tracebacks", symbolize_tracebacks,
        "_profiler.symbolize_tracebacks")


def _install_dynamo_bool(C, put) -> None:
    """`set_is_in_mode_without_ignore_compile_internals`.

    `docs/graph/COMPILE.md` §1.2 identified this as a two-line bool setter at
    `dynamo/guards.cpp:134` that touches none of the frame-hook machinery.  It
    is on the export path because `torch/_dynamo/utils.py` toggles it around
    mode entry; nothing in this shim reads it back, so it is a cell.

    It lives under `torch._C._dynamo.guards`, which is *not* a reason to think
    this opens `torch.compile`.  `set_eval_frame`'s refusal is untouched and
    stays untouched -- see `docs/graph/COMPILE.md` §5.1 for why that refusal must
    outlive any symbol-filling on this path.
    """
    # **`C._dynamo.guards` is not the module this name has to land on**, and
    # that is the whole difficulty of moving this function into the bootstrap.
    #
    # In `upstream.py` this line was fine: that module ran after `import
    # torch`, by which time `torch/utils/_python_dispatch.py:22` had done
    # `import torch._C._dynamo.guards` and `_SubmoduleFinder` had created the
    # real module and put it in `sys.modules`. Here nothing has imported it
    # yet, so the attribute access falls through `_attach_module_catchall`'s
    # PEP 562 `__getattr__`, which -- because "guards" starts with a lowercase
    # letter -- synthesises an `_Unimplemented` and caches it. Installing onto
    # that object succeeded, changed nothing anybody would ever see, and left
    # `torch.export` stopping on this exact name with no error to explain it.
    #
    # So the module is built and registered here the same way `eval_frame` is
    # a few hundred lines above, which is the mechanism `_SubmoduleFinder`
    # would otherwise use later. Registering it in `sys.modules` is what makes
    # the later `import torch._C._dynamo.guards` find this one rather than
    # build a second.
    prefix = C.__name__
    guards = sys.modules.get(f"{prefix}._dynamo.guards")
    if guards is None:
        guards = types.ModuleType(f"{prefix}._dynamo.guards")
        guards.__path__ = []
        _attach_module_catchall(guards)
        sys.modules[f"{prefix}._dynamo.guards"] = guards
    setattr(C._dynamo, "guards", guards)
    state = {"value": False}

    def set_is_in_mode_without_ignore_compile_internals(value):
        state["value"] = bool(value)

    def is_in_mode_without_ignore_compile_internals():
        return state["value"]

    put(guards, "set_is_in_mode_without_ignore_compile_internals",
        set_is_in_mode_without_ignore_compile_internals,
        "_dynamo.guards.set_is_in_mode_without_ignore_compile_internals")
    put(guards, "is_in_mode_without_ignore_compile_internals",
        is_in_mode_without_ignore_compile_internals,
        "_dynamo.guards.is_in_mode_without_ignore_compile_internals")


def _install_inference_mode(C, put) -> None:
    """`torch._C._InferenceMode`, as a context manager that actually enters.

    Not in COMPILE.md's list of 18, and it could not have been: the census
    stopped at `meta_utils.py:1061`, and this is reached at `:1510`.  The
    bootstrap leaves it as a `_ShimMeta` synthesised class, which is
    constructible and has no `__enter__`, so `with torch.inference_mode(...)`
    fails with `AttributeError` rather than by name.  That is worth fixing
    independently of export: an `AttributeError` on a dunder points at
    `grad_mode.py` and says nothing about the shim.

    What it does here is track a flag and nothing else.  Upstream's
    `InferenceMode` switches a dispatch-key bit that makes new tensors skip
    autograd bookkeeping and become invalid outside the block; this shim has no
    autograd bookkeeping to skip (`docs/training/AUTOGRAD.md`) and no version counter to
    invalidate, so entering and leaving is the entire behaviour.  It is
    recorded rather than discarded so `_is_inference_mode_enabled()` can answer
    from the same place instead of guessing.

    `TensorBase.is_inference` stays `False` even inside the block, and that is
    deliberate, not an oversight: upstream's per-tensor flag records that a
    tensor *was created* in inference mode and is therefore unsafe to use
    outside it.  No tensor here is unsafe in that way, so `True` would be a
    claim about lifetime that nothing enforces.
    """

    class _InferenceMode:
        __module__ = "torch._C"
        __slots__ = ("mode", "_saved")

        def __init__(self, mode=True):
            self.mode = bool(mode)
            self._saved = None

        def __enter__(self):
            self._saved = _read()
            _write(self.mode)
            return self

        def __exit__(self, *exc):
            _write(self._saved)
            return False

    # One source of truth, and it is the bootstrap's.  `bootstrap.py`'s
    # `_install_grad_mode` now owns the flag, because `torch.is_inference_mode_enabled`
    # is harvested off `_VariableFunctions` before this module can run and
    # `fake_tensor.py:1801` calls it on every cached dispatch.  Writing through
    # to it -- rather than keeping a second flag on `_TLS` -- is what stops the
    # guard and the predicate from answering differently, which is
    # docs/graph/EXPORT.md §2.2's failure with the operands swapped.
    #
    # The `_TLS` fallback is not dead code: it is what runs against a bootstrap
    # that predates the setter, and it keeps this module importable there.
    def _read():
        fn = getattr(C, "is_inference_mode_enabled", None)
        if fn is not None:
            return bool(fn())
        return getattr(_TLS, "inference_mode", False)

    def _write(value):
        _TLS.inference_mode = value
        setter = getattr(C, "_set_inference_mode_enabled", None)
        if setter is not None:
            setter(bool(value))

    put(C, "_InferenceMode", _InferenceMode)
    put(C, "_is_inference_mode_enabled", _read)


_RAII_GUARDS = {
    "_DisableTorchDispatch":
        "suppresses the torch-dispatch mode stack for the block",
    "_DisableFuncTorch":
        "pops the functorch interpreter stack; there is none here",
    "_DisableAutocast":
        "turns off autocast; this build has no autocast dispatch key",
    "_AutoDispatchBelowAutograd":
        "excludes the autograd keys so a call lands on the backend directly",
    "_RestorePythonTLSSnapshot":
        "restores a saved dispatcher TLS snapshot",
    "_DisablePythonDispatcher": "turns off the Python dispatcher",
    "_EnablePythonDispatcher": "turns on the Python dispatcher",
    "_EnablePreDispatch": "routes through the PreDispatch key",
    "_PreserveDispatchKeyGuard": "saves and restores the whole TLS key state",
    "_SetExcludeDispatchKeyGuard": "sets one key's excluded bit for the block",
}


_KEY_STATE_GUARDS = frozenset({
    "_PreserveDispatchKeyGuard",
    "_ForceDispatchKeyGuard",
    "_ExcludeDispatchKeyGuard",
    "_IncludeDispatchKeyGuard",
    "_SetExcludeDispatchKeyGuard",
})


def _is_guard_active(name) -> bool:
    """Is a named RAII guard currently entered on this thread?"""
    return getattr(_TLS, "guards", {}).get(name, 0) > 0


def _install_raii_guards(C, put) -> None:
    """Give the guard family a real `__enter__`/`__exit__`.

    Read the honesty limit here carefully, because it is the one that matters
    on this path and `docs/graph/EXPORT.md` §4.1 is about it.  `_DisableTorchDispatch`
    is the guard `torch/_subclasses/fake_tensor.py:502` uses to build a meta
    tensor *without re-entering the fake mode*.  Entering and leaving a counter
    is a correct implementation **only because this shim never consults the mode
    stack in the first place** -- `_aten_dispatch` records ops after the fact
    (`aten.rs`, the capture hook) and never asks a Python mode to handle one.
    There is therefore nothing for `no_dispatch()` to suppress.

    That is not a happy accident, it is the wall: see `docs/graph/EXPORT.md` §4.  When
    `_aten_dispatch` learns to consult the stack, this guard stops being a
    counter and starts being load-bearing, and the counter is here so that
    change is a body to fill rather than a hole to find.
    """

    def _make(name, why):
        class _Guard:
            __module__ = "torch._C"
            __slots__ = ()
            __doc__ = f"torch._C.{name} -- upstream {why}."

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                counts = getattr(_TLS, "guards", None)
                if counts is None:
                    counts = _TLS.guards = {}
                counts[name] = counts.get(name, 0) + 1
                # **The key-state family saves the TLS key state here and
                # restores it on exit, and that is not bookkeeping.**
                #
                # `_PreserveDispatchKeyGuard`'s one-line description in
                # `_RAII_GUARDS` above is "saves and restores the whole TLS key
                # state", and until docs/graph/EXPORT5.md it saved and restored
                # nothing.  That is not a cosmetic gap, because upstream
                # *delegates* a restore to it:
                # `fake_tensor.py::in_kernel_invocation_manager` sets
                # `_set_meta_in_tls_dispatch_include(True)` inside this guard and
                # leaves the matching reset **commented out**, with the guard
                # named as the thing that undoes it.  With the guard inert the
                # flag latched `True` after the first kernel invocation and the
                # *next* entry died on that function's own
                # `assert meta_in_tls == prev_in_kernel` -- `AssertionError:
                # True, False`, several frames from anything that mentions a
                # guard.  Same shape as docs/graph/EXPORT4.md §5: no name was missing,
                # no stub was raising, and every existing test passed.
                #
                # The snapshot is by reference because `DispatchKeySet` is
                # immutable (bootstrap.py: `frozenset`, and `add`/`remove`
                # return new sets), so holding the object *is* holding the value.
                if name in _KEY_STATE_GUARDS:
                    saved = getattr(_TLS, "key_state", None)
                    if saved is None:
                        saved = _TLS.key_state = []
                    saved.append((
                        _TLS.included,
                        _TLS.excluded,
                        _TLS.meta_in_tls_dispatch_include,
                    ))
                # `_DisableTorchDispatch` is `no_dispatch()`, and it is the ONE
                # guard in this family that the dispatcher door has to see.  The
                # counter above is this module's own bookkeeping; the write
                # below is the one that actually suppresses, and it lives in
                # `torch._C` because `aten.rs` reads it there.  See
                # `bootstrap.py::_install_dispatch_suppression` and
                # docs/graph/EXPORT4.md §5.
                #
                # The others stay counters on purpose: `_DisableFuncTorch` and
                # friends name subsystems this shim does not have, and making
                # them suppress dispatch would be a guess about what they mean.
                if name == "_DisableTorchDispatch":
                    push = getattr(C, "_shim_push_dispatch_suppression", None)
                    if push is not None:
                        push()
                return self

            def __exit__(self, *exc):
                _TLS.guards[name] -= 1
                if name in _KEY_STATE_GUARDS:
                    (
                        _TLS.included,
                        _TLS.excluded,
                        _TLS.meta_in_tls_dispatch_include,
                    ) = _TLS.key_state.pop()
                if name == "_DisableTorchDispatch":
                    pop = getattr(C, "_shim_pop_dispatch_suppression", None)
                    if pop is not None:
                        pop()
                return False

        _Guard.__name__ = name
        _Guard.__qualname__ = name
        return _Guard

    for name, why in _RAII_GUARDS.items():
        put(C, name, _make(name, why))


def _install_autocast(module) -> None:
    """`torch._C.is_autocast_enabled` and the switch beside it.

    docs/models/E2E_REAL.md. `transformers/utils/generic.py:250` opens
    `maybe_autocast` with `if torch.is_autocast_enabled(device_type) or
    enabled:`, and `modeling_llama.py:121` wraps the rotary embedding in it --
    so this name is the first thing a real `LlamaForCausalLM` forward pass
    asks for, and docs/distributed/DISTRIBUTED.md §7 named it as the wall that round
    stopped at.

    **The read is not a constant; the write is a refusal, and that is the
    argument for the read.** Autocast is a dispatch key: when it is on,
    upstream intercepts each op and casts its inputs to a lower precision
    before the kernel sees them. There is no such key here and no kernel casts
    anything, so a `True` from this predicate would mean every op in the block
    ran in `float32` while the caller believed it ran in `bfloat16` --
    right-shaped numbers, wrong ones. So:

      * `is_autocast_enabled(device_type)` reads a real per-device flag,
      * `set_autocast_enabled(device_type, True)` refuses by name,
      * `set_autocast_enabled(device_type, False)` is accepted, because it is
        already true and a caller restoring the previous state (which is what
        `torch.autocast.__exit__` does) must not fail.

    The flag can therefore never be raised, which is what makes the read
    derived rather than asserted -- the same shape as the functorch dynamic
    layer stack in `_install_repr_surface`, and the test that pins it is
    `test_autocast_is_off_and_cannot_be_turned_on`.

    `get_autocast_dtype` answers upstream's per-device *default*, which is a
    real value even when autocast is off: `torch._C.get_autocast_dtype("cpu")`
    is `torch.bfloat16` on 2.13.0 with no autocast block anywhere, measured.
    Reporting it enables nothing -- nothing consults it while the flag is down
    -- and answering `float32` instead would have been a made-up number that
    happens to describe what the kernels do.
    """
    enabled = {}

    def _check_device(device_type):
        if device_type is None:
            # Upstream's no-argument spelling answers for the default device
            # type; measured, `torch._C.is_autocast_enabled()` is `False`.
            return "cpu"
        if not isinstance(device_type, str) or device_type not in _AUTOCAST_DEVICE_TYPES:
            raise RuntimeError(
                "Expected one of " + ", ".join(_AUTOCAST_DEVICE_TYPES)
                + f" device type at start of device string: {device_type}"
            )
        return device_type

    def is_autocast_enabled(device_type=None):
        return enabled.get(_check_device(device_type), False)

    def set_autocast_enabled(device_type, value=None):
        # Upstream has both `set_autocast_enabled(bool)` (deprecated,
        # CUDA-only) and `set_autocast_enabled(str, bool)`. Only the second
        # spelling is reachable from the vendored tree.
        if value is None:
            device_type, value = "cuda", device_type
        name = _check_device(device_type)
        if value:
            raise NotImplementedError(
                "not implemented in torch._C shim: enabling autocast for "
                f"{name!r}. Autocast is a dispatch key that casts each op's "
                "inputs to a lower precision; this shim has no such key and "
                "no kernel casts anything, so a raised flag would report a "
                "reduced-precision run that did not happen. Disabling it "
                "(the False case) is accepted, because it is already true."
            )
        enabled[name] = False

    # Upstream's per-device *defaults*, measured on 2.13.0. Only the two this
    # build could be asked about were measured; anything else refuses on
    # first read rather than guessing, because the value decides what a cast
    # would make -- until `set_autocast_dtype` (below) gives it one, which is
    # the same round trip `autocast.__enter__`/`__exit__` already does for
    # `self.fast_dtype` (read the current value, install a new one for the
    # region, restore the old one on exit). A dict rather than the previous
    # fixed pair, because `set_autocast_dtype` needs somewhere to write.
    _autocast_dtype = {"cpu": module.bfloat16, "cuda": module.float16}

    def get_autocast_dtype(device_type=None):
        name = _check_device(device_type)
        try:
            return _autocast_dtype[name]
        except KeyError:
            raise NotImplementedError(
                "not implemented in torch._C shim: "
                f"torch._C.get_autocast_dtype({name!r}) -- upstream's default "
                "autocast dtype for this device type was not measured"
            ) from None

    def set_autocast_dtype(device_type, dtype):
        # `_check_device` rather than `_autocast_dtype`'s current keys: a
        # caller may set a dtype for a device type this shim never measured
        # a default for, exactly as upstream would, and that write is what
        # lets a later `get_autocast_dtype` on the same device type succeed.
        name = _check_device(device_type)
        _autocast_dtype[name] = dtype

    def _is_tracing():
        """`torch/jit/_trace.py:1269`.

        Derived from `_get_tracing_state` rather than standing beside it as a
        second constant: upstream's `_is_tracing()` is "is there a tracing
        state?", and there is exactly one answer to that in this process. Two
        independent constants could drift apart; this cannot.
        """
        return module._get_tracing_state() is not None

    # Upstream's build-time registration of which device types have an
    # Autocast dispatch key at all -- measured on 2.13.0
    # (`torch._C._is_autocast_available(<name>)` for every name
    # `_AUTOCAST_DEVICE_TYPES` accepts), not inferred from what this CPU-only
    # build could plausibly support. It does not move when CUDA hardware is
    # absent: `_is_autocast_available("cuda")` is `True` on a CPU-only host
    # too, because the predicate answers "was a key registered for this
    # backend" rather than "is the device physically present" -- confirmed by
    # running the check on this machine (no CUDA) and getting `True`.
    #
    # docs/models/COMPAT.md has the full argument for why this is `True` rather than
    # `False` for `"cpu"`. In short: a `False` here does not make
    # `torch.autocast(..., enabled=False)` a no-op the way a caller might
    # expect -- `autocast.__init__` (`torch/amp/autocast_mode.py`) calls this
    # predicate *unconditionally*, before it even looks at `enabled`, and
    # raises `RuntimeError("User specified an unsupported autocast
    # device_type ...")` if it comes back `False`. That was checked by
    # monkeypatching upstream's own `_is_autocast_available` to `False` and
    # entering `torch.autocast(device_type="cpu", enabled=False)`: it still
    # raises. So `False` would not "let the enabled=False path through" --
    # it would break it, on every device type, unconditionally. `True` for
    # the registered set is what makes `enabled=False` the no-op it already
    # was designed to be (`is_autocast_enabled`/`set_autocast_enabled` above
    # already pin the flag itself to permanently `False`).
    _AUTOCAST_AVAILABLE_DEVICE_TYPES = frozenset((
        "cpu", "cuda", "xpu", "hpu", "mtia", "maia", "ipu", "xla", "mps",
        "privateuseone",
    ))

    def _is_autocast_available(device_type):
        # Upstream's signature takes a required positional `str`; `None` is a
        # `TypeError` there (measured), not the "default to cpu" convenience
        # `is_autocast_enabled()` and friends give -- so this does not reuse
        # `_check_device`, which special-cases `None`.
        if not isinstance(device_type, str):
            raise TypeError(
                "_is_autocast_available(): argument 'device_type' (position "
                f"1) must be str, not {type(device_type).__name__}"
            )
        if device_type not in _AUTOCAST_DEVICE_TYPES:
            raise RuntimeError(
                "Expected one of " + ", ".join(_AUTOCAST_DEVICE_TYPES)
                + f" device type at start of device string: {device_type}"
            )
        return device_type in _AUTOCAST_AVAILABLE_DEVICE_TYPES

    # The rest of `autocast.__enter__`/`__exit__` (`torch/amp/autocast_mode.py`
    # 314-352) reads and restores three more names before and after the
    # region, none of them gated by `enabled` -- so all three are reached
    # even by `torch.autocast(..., enabled=False)`, the no-op path
    # `_is_autocast_available` above exists to unblock. Each is bookkeeping
    # around a cache this shim never populates (no dispatch key ever casts
    # anything here, per `is_autocast_enabled` above), so there is no
    # precision question the way there was for the enabled-flag: a cache
    # tracking nothing is safe to report as enabled, and a nesting counter is
    # safe to count honestly, because neither one changes what a kernel
    # computes.
    _autocast_cache_enabled = [True]  # upstream's default, measured
    _autocast_nesting = [0]

    def is_autocast_cache_enabled():
        return _autocast_cache_enabled[0]

    def set_autocast_cache_enabled(enabled):
        _autocast_cache_enabled[0] = bool(enabled)

    def autocast_increment_nesting():
        _autocast_nesting[0] += 1
        return _autocast_nesting[0]

    def autocast_decrement_nesting():
        _autocast_nesting[0] -= 1
        return _autocast_nesting[0]

    def clear_autocast_cache():
        # Upstream drops cached casts here; this shim never populated any,
        # so there is nothing to drop. Measured to return `None`.
        return None

    for fn, name in (
        (is_autocast_enabled, "is_autocast_enabled"),
        (set_autocast_enabled, "set_autocast_enabled"),
        (get_autocast_dtype, "get_autocast_dtype"),
        (set_autocast_dtype, "set_autocast_dtype"),
        (_is_tracing, "_is_tracing"),
        (_is_autocast_available, "_is_autocast_available"),
        (is_autocast_cache_enabled, "is_autocast_cache_enabled"),
        (set_autocast_cache_enabled, "set_autocast_cache_enabled"),
        (autocast_increment_nesting, "autocast_increment_nesting"),
        (autocast_decrement_nesting, "autocast_decrement_nesting"),
        (clear_autocast_cache, "clear_autocast_cache"),
    ):
        fn.__name__ = fn.__qualname__ = name
        fn.__module__ = "torch._C"
        setattr(module, name, fn)


def _install_device(module, varfns, tensorbase) -> None:
    """The device layer -- the `_C` names that make `.to(device)` mean something.

    docs/devices/DEVICE_ABS.md is the measurement this was built from; §2 is the table
    of what `torch.device` could and could not do before it. The short version
    is that the *label* worked and everything that consumed a label did not:
    `nn.Module.to("cpu")`, `Module.cpu()`, `Module.float()` and every
    `Module._apply` were dead on `torch._C._nn._parse_to` and
    `torch._has_compatible_shallow_copy_type`, which is the whole road a
    checkpoint travels.

    Nothing here reaches the aten dispatcher, and that is measured rather than
    assumed. Run inside a `TorchDispatchMode` on torch 2.13.0, `x.cpu()`,
    `x.to("cpu")`, `x.get_device()`, `x.is_floating_point()`, `x.is_complex()`,
    `m.to("cpu")`, `m.cpu()` and `Generator.device` each record **zero**
    dispatcher calls -- they read metadata off the TensorImpl. Only a `.to()`
    that actually changes something records one, and that is
    `aten._to_copy.default`, which already has a kernel and golden cases. So
    these are not a second door into the dispatcher (DESIGN.md §6); they are
    the same nine-names-that-never-dispatch family `tools/golden/cases.py`
    already documents, and they need no case builders for the same reason
    `device`, `dim` and `dtype` need none.
    """

    # -- "what device is this process defaulting to?" ----------------------
    #
    # Two names, two questions, two different answers on the same build, and
    # the difference is not cosmetic -- one of the callers does not check for
    # `None` and the other does.
    #
    #   `_get_default_device`  is what `torch.get_default_device()` reads and
    #       upstream returns the *string* `'cpu'` from it (measured, torch
    #       2.13.0). Not a `device` -- the Python wrapper builds that.
    #
    #   `_get_accelerator`     is read by `torch.get_device_module()` at
    #       `torch/__init__.py:2978` as `torch._C._get_accelerator().type`,
    #       with no `None` guard, and that function's own docstring says "If no
    #       accelerator is available, it automatically returns CPU device". So
    #       the no-accelerator answer is `device('cpu')`, not `None`.
    #
    #   `_accelerator_getAccelerator` is read by
    #       `torch/accelerator/__init__.py:128` as
    #       `if (acc := torch._C._accelerator_getAccelerator()) is not None:`
    #       -- an explicit `None` check, and `current_accelerator()` returns
    #       `None` when it fires. So *this* one answers `None` here.
    #
    # The last two were read out of the vendored source rather than measured,
    # because this host has MPS and therefore cannot exhibit the
    # no-accelerator branch of either.
    module._get_default_device = _constant_function(
        "torch._C._get_default_device", "cpu"
    )
    module._get_accelerator = _constant_function(
        "torch._C._get_accelerator", module.device("cpu")
    )
    module._accelerator_getAccelerator = _constant_function(
        "torch._C._accelerator_getAccelerator", None
    )
    # `torch.backends.mps.is_available()`. `False` is the honest answer and it
    # is a *different* claim from `_has_mps` (the build flag): candle's `metal`
    # feature is off in Cargo.toml, so there is no Metal backend linked in,
    # which is why `PyDevice::resolve` refuses an `mps` label. DESIGN.md §11.1
    # records that this is a reversible decision, not a capability gap.
    module._mps_is_available = _constant_function("torch._C._mps_is_available", False)

    # -- `torch._has_compatible_shallow_copy_type` -------------------------
    #
    # `nn.Module._apply` (`torch/nn/modules/module.py:938`) is the only caller
    # on this road, and what it decides with the answer is whether to move a
    # converted parameter in place (`param.data = param_applied`) or to build a
    # fresh `Parameter`. It is therefore load-bearing for every `.to()`,
    # `.float()` and `.cpu()` on a module.
    #
    # Upstream's C++ asks whether `input`'s TensorImpl accepts `from_`'s
    # `DispatchKeySet` for a shallow copy. There are no dispatch keys here, so
    # the question has no local meaning and the answer is measured instead:
    # on torch 2.13.0 it is `True` for two plain dense tensors of *different*
    # dtype, of different device (cpu vs mps), and for a `Parameter` against a
    # `Tensor`. The one thing it is not `True` for is a tensor subclass with its
    # own impl -- `FakeTensor`, which `_apply` filters separately anyway, and
    # which this shim cannot produce.
    #
    # So: both dense tensors of this shim => True, and anything else is refused
    # rather than guessed, because a wrong `True` here silently aliases a
    # parameter that should have been replaced.
    def _has_compatible_shallow_copy_type(input, from_):
        if isinstance(input, tensorbase) and isinstance(from_, tensorbase):
            return True
        raise NotImplementedError(
            "not implemented in torch._C shim: "
            "torch._has_compatible_shallow_copy_type on something that is not a "
            f"TensorBase ({type(input).__name__}, {type(from_).__name__}) -- upstream "
            "answers from the DispatchKeySet, which this shim does not have"
        )

    _has_compatible_shallow_copy_type.__name__ = "_has_compatible_shallow_copy_type"
    _has_compatible_shallow_copy_type.__qualname__ = "_has_compatible_shallow_copy_type"
    _has_compatible_shallow_copy_type.__module__ = "torch._C"
    setattr(varfns, "_has_compatible_shallow_copy_type", _has_compatible_shallow_copy_type)

    # -- `torch._C._nn._parse_to` ------------------------------------------
    #
    # The single entrance for `nn.Module.to(...)`: `module.py:1340` unpacks its
    # four-tuple and builds the `convert(t)` closure from it. Everything a
    # module can be moved or cast by goes through here, so with it refusing,
    # `.to()`, `.cpu()`, `.float()`, `.half()` and `.double()` were all dead.
    #
    # It is *not* an overload-table entry, for docs/bindings/OVERLOAD.md §9 item 7's
    # reason -- there is no `aten::_parse_to`; it is a hand-written argument
    # parser in `python_nn_functions.cpp` with no schema at all. The vendored
    # tree carries its own reimplementation at
    # `torch/_dynamo/polyfills/torch_c_nn.py:14`, which is the reference this
    # follows, corrected in three places where the polyfill and the real parser
    # disagree and the real one was measured:
    #
    #   * the polyfill takes at most one positional; the real parser takes up
    #     to three (`_parse_to('cpu', torch.float32, False)` binds fine), and
    #     `Module.to("cpu", torch.float32)` is a documented spelling.
    #   * the polyfill accepts a `memory_format` positionally; the real one is
    #     keyword-only there (measured: passing it fourth is a TypeError).
    #   * `copy` is in the `.pyi` overloads and the real parser rejects it at
    #     runtime -- `RuntimeError: .to() does not accept copy argument`.
    #     Reproduced, because silently accepting it would make
    #     `Module.to(copy=True)` a no-op instead of an error.
    def _parse_to(*args, **kwargs):
        device = None
        dtype = None
        non_blocking = False
        memory_format = None
        bools_seen = 0

        def _reject_copy():
            raise RuntimeError(".to() does not accept copy argument")

        # Four positional slots, and the fourth is `copy` -- which is parsed and
        # then refused, not rejected as arity. Measured:
        # `_parse_to('cpu', torch.float32, False, True)` is
        # `RuntimeError: .to() does not accept copy argument`, while a fifth
        # argument is `TypeError: to() takes from 0 to 4 positional arguments
        # but 5 were given`. Two different mistakes, two different errors.
        if len(args) > 4:
            raise TypeError(
                f"to() takes from 0 to 4 positional arguments but {len(args)} were given"
            )
        for value in args:
            if value is None:
                # `_parse_to(None)` is `(None, None, False, None)` upstream --
                # an explicit "no device", not an error.
                continue
            if isinstance(value, bool):
                # First bool is `non_blocking`; a second one is the `copy`
                # argument the parser refuses.
                bools_seen += 1
                if bools_seen > 1:
                    _reject_copy()
                non_blocking = value
            elif isinstance(value, module.dtype):
                dtype = value
            elif isinstance(value, tensorbase):
                device, dtype = value.device, value.dtype
            elif isinstance(value, (str, module.device)):
                device = module.device(value)
            else:
                raise TypeError(
                    "to() received an invalid combination of arguments - got "
                    f"({type(value).__name__}) in torch._C shim"
                )

        for name, value in kwargs.items():
            if name == "device":
                device = None if value is None else module.device(value)
            elif name == "dtype":
                dtype = value
            elif name == "non_blocking":
                non_blocking = bool(value)
            elif name == "memory_format":
                memory_format = value
            elif name == "copy":
                _reject_copy()
            else:
                raise TypeError(
                    f"to() got an unexpected keyword argument '{name}' in torch._C shim"
                )
        return (device, dtype, non_blocking, memory_format)

    _parse_to.__name__ = "_parse_to"
    _parse_to.__qualname__ = "_parse_to"
    _parse_to.__module__ = "torch._C._nn"
    setattr(module._nn, "_parse_to", _parse_to)
    module._shim_nn_implemented = sorted(
        set(module._shim_nn_implemented) | {"_parse_to"}
    )

    # -- `Generator.device` -------------------------------------------------
    #
    # Overwriting the *class* attribute, which an earlier note in
    # `_install_default_generator` argued against on the grounds that a device
    # is a per-instance value. That argument is right about upstream and wrong
    # about here: `PyDevice::resolve` refuses every label but `cpu`, so every
    # `Generator` this shim can make is a CPU generator, and there is no second
    # value for an instance attribute to hold. It becomes wrong again the day a
    # second backend lands -- at which point the generator has to carry its own
    # device, the same way a tensor will (docs/devices/DEVICE_ABS.md §3.2).
    #
    # It has to be a property rather than a plain attribute because
    # `_make_property` put one there and the callers reach for the descriptor.
    _cpu = module.device("cpu")
    module.Generator.device = property(lambda self: _cpu)


def _install_repr_surface(module, varfns, tensorbase) -> None:
    """What `torch/_tensor_str.py` asks that is neither a kernel nor a device.

    docs/models/E2E_REAL.md is the measurement. `print(tensor)` was the one thing
    docs/platform/WHEEL.md §5 recorded the built wheel could not do, and walking the
    refusals one at a time produced a list of eleven names -- six kernels
    (`aten.rs`), three device predicates and five representation predicates
    (`tensor.rs`), and the three below, which need something Rust cannot
    reach: a `torch.layout` instance, and a stack that lives in Python.

    None of the three is a constant. Each reads something that could answer
    differently, and the thing it reads is the argument for the answer.
    """

    # -- `tensor.layout` ----------------------------------------------------
    #
    # The fact is in Rust (`_layout_name`, an exhaustive match over `Repr`);
    # the object is here, because `torch.strided` is synthesised by
    # `_install_namespace_types` and does not exist on the Rust side.
    #
    # The lookup refuses by name for an unrecognised string rather than
    # returning `None`. That matters more than it looks: `_tensor_str.py`
    # branches on `self.layout != torch.strided` in four places, and a `None`
    # would take every one of those branches -- printing a dense tensor
    # through the sparse formatter -- instead of stopping.
    _LAYOUTS = {}
    for _name in ("strided", "sparse_coo", "sparse_csr", "sparse_csc",
                  "sparse_bsr", "sparse_bsc", "_mkldnn", "jagged"):
        _value = getattr(module, _name, None)
        if _value is not None and not isinstance(_value, _Unimplemented):
            _LAYOUTS[_name] = _value

    def _layout(self):
        name = self._layout_name()
        try:
            return _LAYOUTS[name]
        except KeyError:
            raise NotImplementedError(
                f"torch._C shim: TensorBase._layout_name() said {name!r}, which "
                "is not a torch.layout this build publishes. A representation "
                "was added in tensor.rs without a layout to name it."
            ) from None

    tensorbase.layout = property(_layout)

    # -- the functorch dynamic layer stack ----------------------------------
    #
    # `_tensor_str.py:409` is the very first line of `_str_intern`:
    #
    #     if torch._C._functorch.is_functorch_wrapped_tensor(inp):
    #
    # Upstream's answer is `maybe_get_level(tensor) != -1`, and `maybe_get_level`
    # reads the *dynamic layer stack* -- the one `vmap`/`grad`/`jvp` push an
    # interpreter onto for the duration of a transform. Measured on 2.13.0:
    # `maybe_get_level(plain)` is `-1` outside `vmap` and `1` inside it, and
    # `is_functorch_wrapped_tensor` follows it exactly.
    #
    # So this is not "answer False and move on". The stack is a real list, it
    # is empty, and the predicate reads it -- because everything that pushes
    # onto it (`_wrap_for_grad`, `_add_batch_dim`, `_vmap_increment_nesting`
    # and its `_grad`/`_jvp`/`_func` siblings) is a raising stub, which
    # `pytests/test_shim.py` asserts. A tensor cannot acquire a level here
    # without one of those first, so `-1` is derived from the stack rather
    # than written down. If functorch ever lands, the pushers change and these
    # three follow without being touched.
    _dynamic_layer_stack = []

    def get_dynamic_layer_stack_depth():
        return len(_dynamic_layer_stack)

    def peek_interpreter_stack():
        # Upstream returns the top interpreter or `None`.
        # `_tensor_str.py:668` reads it to decide whether to print a wrapper.
        return _dynamic_layer_stack[-1] if _dynamic_layer_stack else None

    def maybe_get_level(tensor):
        # Upstream returns the level the tensor is wrapped at, or -1. A
        # tensor can only carry a level if a transform put it there, and no
        # transform can start while the stack cannot be pushed to.
        if not _dynamic_layer_stack:
            return -1
        if all(isinstance(entry, _VmapBroadcastInterpreter)
               for entry in _dynamic_layer_stack):
            # The vmap level below is a *broadcast* representation: its
            # batched values are plain tensors with one dimension per level,
            # not wrappers carrying a level. So `-1` is still the derived
            # answer -- `is_functorch_wrapped_tensor` is asking whether this
            # tensor is a wrapper, and here nothing ever is.
            return -1
        raise NotImplementedError(
            "not implemented in torch._C shim: "
            "torch._C._functorch.maybe_get_level with a non-empty dynamic "
            "layer stack -- something pushed an interpreter and this shim has "
            "no wrapper tensors to report a level for"
        )

    def maybe_current_level():
        return len(_dynamic_layer_stack) if _dynamic_layer_stack else None

    def is_functorch_wrapped_tensor(tensor):
        return maybe_get_level(tensor) != -1

    def unwrap_if_dead(tensor):
        """`torch/autograd/function.py:632`, on the ordinary `apply` path.

        `bloom` is the measured caller (docs/architectures/ARCH20.md §6): its `GeLUFunction`
        is a `torch.autograd.Function`, so every MLP in every layer runs
        `Function.apply`, and `apply` maps this over its arguments before it
        calls `forward`.

        Upstream unwraps a `TensorWrapper` whose functorch level has expired
        and returns anything else unchanged -- measured,
        `unwrap_if_dead(torch.ones(2))` is that tensor. Derived from the same
        empty stack the four predicates above read rather than written down as
        "return the argument": a tensor can only *be* a wrapper if a transform
        put it there, and every pusher is a raising stub. If functorch ever
        lands, this refuses instead of silently handing back a live wrapper.
        """
        if _dynamic_layer_stack and not all(
                isinstance(entry, _VmapBroadcastInterpreter)
                for entry in _dynamic_layer_stack):
            raise NotImplementedError(
                "not implemented in torch._C shim: "
                "torch._C._functorch.unwrap_if_dead with a non-empty dynamic "
                "layer stack -- something pushed an interpreter and this shim "
                "has no wrapper tensors to unwrap"
            )
        return tensor

    for _fn, _name in (
        (get_dynamic_layer_stack_depth, "get_dynamic_layer_stack_depth"),
        (peek_interpreter_stack, "peek_interpreter_stack"),
        (maybe_get_level, "maybe_get_level"),
        (maybe_current_level, "maybe_current_level"),
        (is_functorch_wrapped_tensor, "is_functorch_wrapped_tensor"),
        (unwrap_if_dead, "unwrap_if_dead"),
    ):
        _fn.__name__ = _fn.__qualname__ = _name
        _fn.__module__ = "torch._C._functorch"
        setattr(module._functorch, _name, _fn)

    # -- `vmap`: a broadcast batching level for scalar index closures -------
    #
    # `docs/kernels/VMAP.md` is the round that built this; read §2 before widening it.
    #
    # The demand is `transformers/masking_utils.py:348`, reached by four
    # architectures (`nemotron3_5_asr`, `nemotron_asr_streaming`,
    # `nemotron_asr_streaming_encoder`, `t5gemma2`) because they pass an
    # `and_mask_function` and so flip `use_vmap=True` at `masking_utils.py:962`.
    # What they vmap is `_vmap_expansion_sdpa`: four nested `torch.vmap`s of a
    # **scalar** closure `f(b_idx, h_idx, q_idx, kv_idx) -> 0-d bool`, each with
    # `in_dims` one-hot (a single `0`, the rest `None`) over a 1-D `arange`, and
    # `out_dims=0` throughout.
    #
    # For that shape -- and *only* that shape -- vmap is broadcasting. Level `L`
    # owns dimension `L-1`, every batched value is a plain tensor of rank
    # `_VMAP_MAX_RANK` whose non-owned dimensions are 1, and the elementwise ops
    # in the closure line the levels up for free. `_remove_batch_dim` with
    # `out_dims=0` is then bookkeeping, because the dimension is already where
    # out_dim 0 wants it. transformers ships the same identity as
    # `_non_vmap_expansion_sdpa` (`masking_utils.py:352`) and uses it as the
    # *default* path, so this is not a novel claim -- and §3 of VMAP.md measures
    # both paths against upstream torch and finds them bit-identical.
    #
    # Three things keep this from being the "no-op counter" `docs/kernels/COMPLEX.md` §7
    # warned about, which would have produced a wrong attention mask rather than
    # an error:
    #
    #   1. **The shape is checked, not assumed.** Every `_add_batch_dim` must be
    #      a 1-D tensor with `in_dim == 0` (logical rank 0), and every
    #      `_remove_batch_dim` must hand back a tensor whose dimensions are
    #      exactly the level sizes or 1. A reduction, a reshape, a `cat` or a
    #      matmul inside the closure moves that shape and is refused.
    #   2. **The answer is checked against the definition of vmap.** A shape
    #      that survives (1) can still be wrong if an op read *across* a level
    #      -- `cumsum`, `flip`, `sort`. So at the innermost `_remove_batch_dim`
    #      the closure is re-run on eight sampled index points with plain 0-d
    #      scalars and no levels on the stack, which is what `vmap` *means*, and
    #      any disagreement raises. This is why the frame of `_flat_vmap` is
    #      required below: it is where the closure and its arguments live.
    #   3. **Anything else still refuses.** `_vmap_increment_nesting` called
    #      from anywhere but `torch/_functorch/vmap.py:_flat_vmap`, a
    #      `randomness` other than `"error"`, `out_dims != 0`, more than one
    #      output, a nesting deeper than the rank budget, or a self-check that
    #      cannot be run all raise `NotImplementedError`. `grad`/`jvp`/`functionalize`
    #      are untouched: `_wrap_for_grad` is still a raising stub.
    #
    # What this is *not* is a batching-rule system. There is no rule for `sum`,
    # no rule for `matmul`, no rule for anything -- there is one representation
    # that happens to be correct for pointwise-over-indices closures, plus two
    # gates that refuse when it is not. VMAP.md §5 sizes the real thing.
    import operator as _operator
    import random as _random

    _VMAP_MAX_RANK = 8
    _vmap_stack = []
    _vmap_logical = []
    _vmap_unpaired = [0]

    class _VmapBroadcastInterpreter:
        """One entry of the functorch dynamic layer stack, for this vmap."""

        __slots__ = ("lvl", "batch_size", "randomness", "func", "frame",
                     "wrapped", "keep", "removed")

        def __init__(self, lvl, batch_size, randomness):
            self.lvl = lvl
            self.batch_size = batch_size
            self.randomness = randomness
            self.func = None
            self.frame = None
            self.wrapped = {}
            self.keep = []
            self.removed = 0

        def key(self):
            return "Vmap"

        def level(self):
            return self.lvl

        def __repr__(self):
            return "<vmap interpreter level=%d batch_size=%d>" % (
                self.lvl, self.batch_size)

    def _vmap_refuse(what):
        return NotImplementedError(
            "not implemented in torch._C shim: " + what + ". This build's "
            "`vmap` is a broadcast batching level for scalar index closures "
            "only (docs/kernels/VMAP.md); a general vmap needs a batching rule per "
            "operator, which does not exist here. Refusing rather than "
            "returning a plausible wrong answer."
        )

    def _vmap_flat_vmap_frame():
        """The `_flat_vmap` frame that drives this transform, or `None`.

        `torch/_functorch/vmap.py:_flat_vmap` is the only caller of
        `_vmap_increment_nesting` in the vendored tree, and it is where the
        closure (`func`) and the batched arguments (`batched_inputs`, bound
        after the increment) live. The self-check in `_remove_batch_dim`
        cannot run without them, so a call from anywhere else is refused
        instead of silently skipping the check.
        """
        f = sys._getframe(1)
        for _ in range(8):
            if f is None:
                return None
            code = f.f_code
            if (code.co_name == "_flat_vmap"
                    and code.co_filename.replace("\\", "/").endswith(
                        "_functorch/vmap.py")):
                return f
            f = f.f_back
        return None

    def _vmap_lookup(obj):
        """`(level, original_1d_tensor)` if `obj` came out of `_add_batch_dim`."""
        if not isinstance(obj, tensorbase):
            return None
        for interp in _vmap_stack:
            hit = interp.wrapped.get(id(obj))
            if hit is not None and hit[0] is obj:
                return interp.lvl, hit[1]
        return None

    def _vmap_increment_nesting_checked(batch_size, randomness):
        if randomness != "error":
            raise _vmap_refuse(
                "torch._C._functorch._vmap_increment_nesting with "
                "randomness=%r -- only 'error' is handled, because a "
                "randomness mode is a statement about how random ops behave "
                "under the transform and there are no batching rules here to "
                "make that statement true" % (randomness,))
        frame = _vmap_flat_vmap_frame()
        if frame is None:
            raise _vmap_refuse(
                "torch._C._functorch._vmap_increment_nesting called from "
                "outside torch/_functorch/vmap.py:_flat_vmap -- the batched "
                "representation here is checked against the closure it "
                "batches, and that closure is only reachable from that frame")
        try:
            batch_size = _operator.index(batch_size)
        except TypeError:
            raise _vmap_refuse(
                "torch._C._functorch._vmap_increment_nesting with a "
                "non-integer batch_size %r" % (batch_size,)) from None
        if batch_size < 1:
            raise _vmap_refuse(
                "torch._C._functorch._vmap_increment_nesting with "
                "batch_size=%d" % (batch_size,))
        if len(_vmap_stack) >= _VMAP_MAX_RANK:
            raise _vmap_refuse(
                "vmap nested %d deep -- this representation gives each level "
                "one of %d tensor dimensions"
                % (len(_vmap_stack) + 1, _VMAP_MAX_RANK))
        if not _vmap_stack and _vmap_logical:
            del _vmap_logical[:]
        if _vmap_logical:
            raise _vmap_refuse(
                "vmap pushed a level after an output was unwrapped at the "
                "level below it -- this representation cannot hold a batched "
                "value and a partially unwrapped one at once")
        if "func" not in frame.f_locals:
            raise _vmap_refuse(
                "torch/_functorch/vmap.py:_flat_vmap does not bind `func` in "
                "this torch build, so the self-check cannot be run")
        interp = _VmapBroadcastInterpreter(
            len(_vmap_stack) + 1, batch_size, randomness)
        interp.func = frame.f_locals["func"]
        interp.frame = frame
        _vmap_stack.append(interp)
        _dynamic_layer_stack.append(interp)
        return interp.lvl

    def _vmap_increment_nesting(batch_size, randomness):
        """The refusal has to survive `vmap_increment_nesting`'s `finally`.

        `torch/_functorch/vmap.py:487` is a context manager: it increments,
        yields, and decrements in a `finally` that runs even when the
        increment itself raised. So a refusal here is immediately followed by
        an unpaired decrement, and if that decrement raises too it *replaces*
        the refusal -- the caller is told "decrement with no level" and never
        learns what was actually unsupported. Measured: `randomness=
        "different"` reported the decrement error, not the randomness one.

        So a refused increment leaves a debt, and the decrement pays it
        instead of complaining. A hand-called decrement with no debt still
        raises, which is what `test_vmap.py` pins.
        """
        try:
            return _vmap_increment_nesting_checked(batch_size, randomness)
        except NotImplementedError:
            # Only when a `finally` is actually going to call the decrement.
            # A hand call from outside `_flat_vmap` has no context manager
            # behind it, so recording a debt there would make the *next*
            # genuine decrement pay it and leak a live level -- measured, the
            # dynamic layer stack ended one deep.
            if _vmap_flat_vmap_frame() is not None:
                _vmap_unpaired[0] += 1
            raise

    def _vmap_decrement_nesting():
        # Debt first, and before looking at the stack: with nesting, a refused
        # *inner* increment is followed by a decrement while the outer levels
        # are still pushed, and popping one of those would unwind a level that
        # is still live.
        if _vmap_unpaired[0]:
            _vmap_unpaired[0] -= 1
            return -1
        if not _vmap_stack:
            raise RuntimeError(
                "torch._C._functorch._vmap_decrement_nesting with no vmap "
                "level on the dynamic layer stack")
        interp = _vmap_stack.pop()
        _dynamic_layer_stack.pop()
        interp.frame = None
        interp.func = None
        interp.wrapped.clear()
        del interp.keep[:]
        if not _vmap_stack:
            del _vmap_logical[:]
        return interp.lvl

    def _add_batch_dim(tensor, in_dim, level):
        if not _vmap_stack or level != len(_vmap_stack):
            raise _vmap_refuse(
                "torch._C._functorch._add_batch_dim at level %r with %d vmap "
                "levels on the stack -- a batched tensor can only be made "
                "inside the transform that batches it"
                % (level, len(_vmap_stack)))
        interp = _vmap_stack[-1]
        if not isinstance(tensor, tensorbase):
            raise _vmap_refuse(
                "torch._C._functorch._add_batch_dim on a %s"
                % (type(tensor).__name__,))
        if _operator.index(in_dim) != 0 or tensor.dim() != 1:
            raise _vmap_refuse(
                "torch._C._functorch._add_batch_dim of a %d-D tensor at "
                "in_dim=%r -- this representation only holds a batched "
                "*scalar* (a 1-D input mapped at dim 0), because it gives "
                "level L tensor dimension L-1 and leaves nothing for a "
                "logical shape to sit in"
                % (tensor.dim(), in_dim))
        if tensor.shape[0] != interp.batch_size:
            raise _vmap_refuse(
                "torch._C._functorch._add_batch_dim of a size-%d dimension at "
                "a level whose batch size is %d"
                % (tensor.shape[0], interp.batch_size))
        shape = ((1,) * (level - 1) + (interp.batch_size,)
                 + (1,) * (_VMAP_MAX_RANK - level))
        out = tensor.reshape(shape)
        interp.wrapped[id(out)] = (out, tensor)
        interp.keep.append(out)
        return out

    def _vmap_self_check(interp, chk):
        """Re-run the closure on sampled points, which is what vmap *means*.

        `chk` is the broadcast answer, of shape `(b_1, ..., b_D)`. The
        definition says `chk[i_1, ..., i_D] == f(a_1[i_1], ..., a_D[i_D])`
        with plain scalars and no transform running, so that is what this
        computes -- on the two corners and up to six deterministic interior
        points. An op that read across a level (`cumsum`, `flip`, `sort`)
        keeps the shape and so survives the shape gate; it does not survive
        this one.
        """
        frame = interp.frame
        loc = frame.f_locals if frame is not None else {}
        if "batched_inputs" not in loc:
            raise _vmap_refuse(
                "the vmap self-check could not find `batched_inputs` in the "
                "_flat_vmap frame, so the batched answer cannot be compared "
                "against the closure it claims to batch")
        from torch.utils import _pytree as _vmap_pytree

        args = loc["batched_inputs"]
        kwargs = loc.get("kwargs") or {}
        func = interp.func
        flat, spec = _vmap_pytree.tree_flatten(args)
        origins = [_vmap_lookup(leaf) for leaf in flat]
        sizes = [s.batch_size for s in _vmap_stack]
        depth = len(sizes)
        if not any(o is not None for o in origins):
            raise _vmap_refuse(
                "vmap's closure was called with no batched argument, so its "
                "answer cannot be checked against the definition")

        rng = _random.Random(0)
        points = [tuple(0 for _ in sizes), tuple(n - 1 for n in sizes)]
        while len(points) < 8:
            p = tuple(rng.randrange(n) for n in sizes)
            if p not in points:
                points.append(p)
            elif len(points) >= min(8, 1 + math.prod(sizes)):
                break

        saved_stack = list(_vmap_stack)
        saved_layers = list(_dynamic_layer_stack)
        saved_logical = list(_vmap_logical)
        del _vmap_stack[:]
        del _dynamic_layer_stack[:]
        del _vmap_logical[:]
        try:
            for point in points:
                leaves = []
                for leaf, origin in zip(flat, origins):
                    if origin is None:
                        leaves.append(leaf)
                    else:
                        lvl, source = origin
                        leaves.append(source[point[lvl - 1]])
                try:
                    ref = func(*_vmap_pytree.tree_unflatten(leaves, spec),
                               **kwargs)
                except NotImplementedError:
                    raise
                except Exception as exc:
                    raise _vmap_refuse(
                        "vmap's closure raised %s when re-run on plain "
                        "scalars, so the batched answer cannot be checked "
                        "against the definition of vmap: %s"
                        % (type(exc).__name__, exc)) from None
                if not isinstance(ref, tensorbase):
                    if not isinstance(ref, (bool, int, float)):
                        raise _vmap_refuse(
                            "vmap's closure returned a %s for scalar inputs; "
                            "this representation unwraps a single batched "
                            "tensor" % (type(ref).__name__,))
                    ref = sys.modules["torch"].tensor(ref)
                if ref.dim() != 0:
                    raise _vmap_refuse(
                        "vmap's closure returned a %d-D result for scalar "
                        "inputs; this representation only holds a batched "
                        "scalar" % (ref.dim(),))
                got = chk[point] if depth else chk
                if ref.dtype != got.dtype:
                    raise _vmap_refuse(
                        "vmap's batched answer has dtype %s but the closure "
                        "returns %s on the same point"
                        % (got.dtype, ref.dtype))
                if not bool((ref == got).all()):
                    raise _vmap_refuse(
                        "vmap's batched answer disagrees with the closure at "
                        "index %r: broadcasting gave %r, the closure gives "
                        "%r. The closure reads across a batched dimension, "
                        "which needs a real batching rule"
                        % (point, got, ref))
        finally:
            _vmap_stack[:] = saved_stack
            _dynamic_layer_stack[:] = saved_layers
            _vmap_logical[:] = saved_logical

    def _remove_batch_dim(tensor, level, batch_size, out_dim):
        if not _vmap_stack or level != len(_vmap_stack):
            raise _vmap_refuse(
                "torch._C._functorch._remove_batch_dim at level %r with %d "
                "vmap levels on the stack" % (level, len(_vmap_stack)))
        interp = _vmap_stack[-1]
        if _operator.index(out_dim) != 0:
            raise _vmap_refuse(
                "torch._C._functorch._remove_batch_dim with out_dim=%r -- "
                "only out_dims=0 is handled, which is where this "
                "representation already puts the mapped dimension"
                % (out_dim,))
        if _operator.index(batch_size) != interp.batch_size:
            raise _vmap_refuse(
                "torch._C._functorch._remove_batch_dim with batch_size=%r at "
                "a level whose batch size is %d"
                % (batch_size, interp.batch_size))
        if interp.removed:
            raise _vmap_refuse(
                "vmap's closure returned more than one output; this "
                "representation unwraps one")
        if not isinstance(tensor, tensorbase):
            raise _vmap_refuse(
                "torch._C._functorch._remove_batch_dim on a %s"
                % (type(tensor).__name__,))
        t = tensor
        if t.dim() > _VMAP_MAX_RANK:
            raise _vmap_refuse(
                "vmap's closure returned a rank-%d result; this "
                "representation budgets %d dimensions"
                % (t.dim(), _VMAP_MAX_RANK))
        if t.dim() < _VMAP_MAX_RANK:
            # Right-alignment, which is what broadcasting would have done.
            t = t.reshape((1,) * (_VMAP_MAX_RANK - t.dim()) + tuple(t.shape))
        shape = tuple(t.shape)
        for i, other in enumerate(_vmap_stack):
            if shape[i] not in (1, other.batch_size):
                raise _vmap_refuse(
                    "vmap's closure returned shape %r; dimension %d belongs "
                    "to the level with batch size %d and must be that or 1. "
                    "An op inside the closure changed the shape, which needs "
                    "a real batching rule"
                    % (shape, i, other.batch_size))
        tail = shape[level:]
        want = tuple(_vmap_logical)
        if tail[:len(want)] != want or any(d != 1 for d in tail[len(want):]):
            raise _vmap_refuse(
                "vmap's closure returned shape %r, whose dimensions after the "
                "level slots are %r rather than the already-unwrapped %r"
                % (shape, tail, want))
        expanded = shape[:level - 1] + (interp.batch_size,) + shape[level:]
        out = t.expand(expanded)
        if not _vmap_logical:
            sizes = tuple(s.batch_size for s in _vmap_stack)
            chk = out.expand(sizes + (1,) * (_VMAP_MAX_RANK - len(sizes)))
            chk = chk.contiguous().reshape(sizes)
            _vmap_self_check(interp, chk)
        interp.removed += 1
        _vmap_logical.insert(0, interp.batch_size)
        if level == 1:
            return out.contiguous().reshape(tuple(_vmap_logical))
        return out

    for _fn, _name in (
        (_vmap_increment_nesting, "_vmap_increment_nesting"),
        (_vmap_decrement_nesting, "_vmap_decrement_nesting"),
        (_add_batch_dim, "_add_batch_dim"),
        (_remove_batch_dim, "_remove_batch_dim"),
    ):
        _fn.__name__ = _fn.__qualname__ = _name
        _fn.__module__ = "torch._C._functorch"
        setattr(module._functorch, _name, _fn)

    # -- `torch._is_functional_tensor` --------------------------------------
    #
    # `_tensor_str.py:597`, choosing the `_to_functional_tensor(` prefix.
    # Functionalisation wraps a tensor so that mutations are recorded rather
    # than performed; the wrapper is made by `torch._to_functional_tensor`,
    # which is a raising stub here (asserted alongside the others).
    #
    # Not in `_DISCOVERED_RETURNS`: that table wraps its values in
    # `_constant_function`, which ignores its arguments, and a predicate that
    # does not look at what it was asked about is the shape of answer this
    # file's own docstring warns against. This one looks -- and refuses a
    # non-tensor the way upstream does, rather than answering `False` about
    # an object it was never asked to classify.
    def _is_functional_tensor(t):
        if not isinstance(t, tensorbase):
            raise TypeError(
                "_is_functional_tensor(): argument 'tensor' (position 1) must "
                f"be Tensor, not {type(t).__name__}"
            )
        # There is exactly one wrapper maker and it refuses, so no tensor in
        # this process can be a functional wrapper. Same argument, and the
        # same guard, as the representation predicates in tensor.rs.
        return False

    _is_functional_tensor.__name__ = "_is_functional_tensor"
    _is_functional_tensor.__qualname__ = "_is_functional_tensor"
    _is_functional_tensor.__module__ = "torch._C"
    module._is_functional_tensor = _is_functional_tensor
    setattr(varfns, "_is_functional_tensor", _is_functional_tensor)


def _install_distributed_c10d(module, spec) -> None:
    """`torch._C._distributed_c10d`, at world_size 1.

    docs/distributed/DISTRIBUTED.md. This subsystem was **off**: `_c10d_init` was one of the
    names in the "Deliberate omissions" set at the top of this file, because
    `torch/distributed/__init__.py:28` is `hasattr(torch._C, "_c10d_init")` and
    absence is the switch upstream provides. Turning it on is not a one-line
    change, and docs/design/SURFACE_HONESTY.md §2.4 is the measurement that says why:
    an instrument that answered *every* attribute still could not finish
    `import torch`, because `distributed_c10d.py:547` walks
    `ReduceOp.RedOpType.__members__` and `distributed_c10d.py:321` reads
    `ProcessGroup.BackendType.UNDEFINED`. What the tree wants here is structure,
    not names -- real enums, nested types, subclassable bases -- so the generic
    submodule builder above cannot produce it and this replaces its output.

    **What is real and what is refused.** At world_size 1 a great deal of this
    is not a stand-in for anything:

      * The **store** is a genuine key/value store. "Distributed store" and
        "local dict" are the same object when this process is the only writer,
        so `Store`/`HashStore`/`PrefixStore` here are implementations, not
        placeholders. `Store.wait` is the one that has to be careful: at
        world_size 1 an absent key is not "not yet", it is "never", so it
        raises rather than blocking or returning.
      * **Work** is always complete, because every collective a single rank can
        perform finishes before the call returns. There is nothing to wait for
        and saying so is accurate.
      * `TCPStore` **refuses by name**. There is no peer at the other end of
        that socket, and a store that silently behaved like a local dict while
        claiming to be a rendezvous point is exactly the failure docs/models/CKPT.md
        recorded -- it looks like it worked.
      * `ProcessGroupGloo`/`Nccl`/`Ucc`/`Xccl`/`Mpi` are **absent**, not stubbed.
        `distributed_c10d.py:204-242` imports each in its own
        `try/except ImportError` and sets an availability flag, so absence is
        the answer the tree is written to receive. Providing an empty class
        would set the flag to True and route real collectives into it.

    `_c10d_init` is the tree's own initialisation hook, and this module uses it
    for the one thing that cannot happen at `import torch._C` time -- see
    `_c10d_init` below.
    """
    name = f"{module.__name__}._distributed_c10d"
    mod = types.ModuleType(name)
    mod.__path__ = []

    def refuse(what, why):
        raise NotImplementedError(f"torch._C._distributed_c10d.{what}: {why}")

    # -- ReduceOp ----------------------------------------------------------
    #
    # `RedOpType` must be an `enum.Enum` and not merely enum-shaped:
    # `distributed_c10d.py:547` iterates `__members__.items()`, and
    # `ReduceOp.SUM` is a default argument value in seven `def`s, so both are
    # read while the module body runs.
    class _RedOpType(enum.Enum):
        SUM = 0
        AVG = 1
        PRODUCT = 2
        MIN = 3
        MAX = 4
        BAND = 5
        BOR = 6
        BXOR = 7
        PREMUL_SUM = 8
        UNUSED = 9

        def __call__(self, factor):
            # Upstream: only PREMUL_SUM is callable, and it carries a factor.
            if self is not _RedOpType.PREMUL_SUM:
                raise TypeError(
                    "torch._C._distributed_c10d: only ReduceOp.PREMUL_SUM takes "
                    f"a factor, not {self.name}"
                )
            op = _ReduceOp(_RedOpType.PREMUL_SUM)
            op._factor = factor
            return op

    class _ReduceOp:
        RedOpType = _RedOpType

        def __init__(self, op=_RedOpType.SUM):
            if isinstance(op, _ReduceOp):
                op = op.op
            self.op = op
            self._factor = None

        @property
        def factor(self):
            return self._factor

        def __eq__(self, other):
            if isinstance(other, _ReduceOp):
                return self.op == other.op
            if isinstance(other, _RedOpType):
                return self.op == other
            return NotImplemented

        def __hash__(self):
            return hash(self.op)

        def __repr__(self):
            return f"<torch.distributed.distributed_c10d.ReduceOp.{self.op.name}: {self.op.value}>"

    for member in _RedOpType:
        setattr(_ReduceOp, member.name, member)

    _ReduceOp.__name__ = _ReduceOp.__qualname__ = "ReduceOp"
    _RedOpType.__qualname__ = "ReduceOp.RedOpType"

    # -- the store ---------------------------------------------------------
    class Store:
        """A key/value store with one participant.

        Not a stand-in. Upstream's `HashStore` is an in-process dict too; the
        difference between it and `TCPStore` is who else can reach it, and at
        world_size 1 the answer is nobody either way.
        """

        def __init__(self, *args, **kwargs):
            self._entries = {}
            self._queues = {}
            self._timeout = datetime.timedelta(seconds=300)

        @staticmethod
        def _as_bytes(value):
            if isinstance(value, str):
                return value.encode()
            return bytes(value)

        def set(self, key, value):
            self._entries[key] = self._as_bytes(value)

        def get(self, key):
            if key not in self._entries:
                raise RuntimeError(f"Key {key} not found in store")
            return self._entries[key]

        def add(self, key, value):
            total = int(self._entries.get(key, b"0")) + value
            self._entries[key] = str(total).encode()
            return total

        def check(self, keys):
            return all(k in self._entries for k in keys)

        def compare_set(self, key, expected_value, desired_value):
            expected = self._as_bytes(expected_value)
            desired = self._as_bytes(desired_value)
            current = self._entries.get(key)
            if current is None:
                # Upstream treats an empty expected value as "set if absent".
                if expected == b"":
                    self._entries[key] = desired
                    return desired
                return expected
            if current == expected:
                self._entries[key] = desired
                return desired
            return current

        def delete_key(self, key):
            return self._entries.pop(key, None) is not None

        def multi_get(self, keys):
            return [self.get(k) for k in keys]

        def multi_set(self, keys, values):
            for key, value in zip(keys, values):
                self.set(key, value)

        def num_keys(self):
            return len(self._entries)

        def list_keys(self):
            return list(self._entries)

        def set_timeout(self, timeout):
            self._timeout = timeout

        @property
        def timeout(self):
            return self._timeout

        def wait(self, keys, timeout=None):
            """Refuse rather than block.

            Upstream blocks until another rank writes the key. There is no
            other rank, so the key is either already here or it is never
            coming -- and a `wait` that returned quietly on an absent key would
            let the caller proceed as though a peer had answered.
            """
            missing = [k for k in keys if k not in self._entries]
            if missing:
                raise RuntimeError(
                    f"torch._C._distributed_c10d.Store.wait: {missing} absent. "
                    "At world_size 1 this process is the only writer, so no "
                    "other rank can ever set these keys"
                )

        def queue_push(self, key, value):
            self._queues.setdefault(key, []).append(self._as_bytes(value))

        def queue_pop(self, key, block=True):
            queue = self._queues.get(key) or []
            if not queue:
                raise RuntimeError(
                    f"torch._C._distributed_c10d.Store.queue_pop: queue {key!r} "
                    "is empty and, at world_size 1, nothing else can fill it"
                )
            return queue.pop(0)

        def queue_len(self, key):
            return len(self._queues.get(key) or [])

    class HashStore(Store):
        pass

    class FileStore(Store):
        """Backed by the dict, not by the file.

        The path is remembered and reported so the caller can see what it
        asked for, but nothing is written: a file store exists so that two
        processes can find each other, and there is only one.
        """

        def __init__(self, path, numWorkers=-1):
            super().__init__()
            self.path = path

    class TCPStore(Store):
        # `host_name`, spelled the way upstream spells it. `rendezvous.py`
        # builds this store with **keyword** arguments
        # (`TCPStore(host_name=..., port=..., ...)`), so a parameter named
        # `host` is unreachable through `init_process_group(init_method=
        # "tcp://...")` -- which is the only door a user goes through.
        # A test that constructs the store itself cannot see that.
        def __init__(self, host_name, port, world_size, is_master, timeout=None, **kwargs):
            host = host_name
            if world_size == 1:
                refuse(
                    "TCPStore",
                    "no transport is built into this shim, and at world_size 1 "
                    "there is no peer to reach. Pass a HashStore, or pass "
                    "store=... to init_process_group",
                )
            super().__init__()
            import socket, threading, json, time
            self.host = host
            self.port = port
            self.is_master = is_master
            self.store = {}
            self.cond = threading.Condition()
            
            if is_master:
                self.server_thread = threading.Thread(target=self._run_server, daemon=True)
                self.server_thread.start()
                
            for _ in range(100):
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect((self.host, self.port))
                    s.close()
                    break
                except ConnectionRefusedError:
                    time.sleep(0.01)

        def _run_server(self):
            import socket, threading, json
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
                
        def _handle_client(self, conn):
            import json
            with conn:
                while True:
                    length_bytes = conn.recv(4)
                    if not length_bytes: break
                    length = int.from_bytes(length_bytes, 'big')
                    data = b""
                    while len(data) < length:
                        packet = conn.recv(length - len(data))
                        if not packet: break
                        data += packet
                    if not data: break
                    req = json.loads(data.decode('utf-8'))
                    
                    with self.cond:
                        cmd = req['cmd']
                        if cmd == 'set':
                            self.store[req['key']] = req['value']
                            self.cond.notify_all()
                            resp = {'status': 'ok'}
                        elif cmd == 'get':
                            key = req['key']
                            while key not in self.store:
                                self.cond.wait()
                            resp = {'status': 'ok', 'value': self.store[key]}
                        elif cmd == 'wait':
                            keys = req['keys']
                            for key in keys:
                                while key not in self.store:
                                    self.cond.wait()
                            resp = {'status': 'ok'}
                        else:
                            resp = {'status': 'error', 'msg': 'unknown cmd'}
                    
                    resp_bytes = json.dumps(resp).encode('utf-8')
                    conn.sendall(len(resp_bytes).to_bytes(4, 'big') + resp_bytes)
                    
        def _send_req(self, req):
            import socket, json
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.host, self.port))
            req_bytes = json.dumps(req).encode('utf-8')
            s.sendall(len(req_bytes).to_bytes(4, 'big') + req_bytes)
            
            length_bytes = s.recv(4)
            length = int.from_bytes(length_bytes, 'big')
            data = b""
            while len(data) < length:
                data += s.recv(length - len(data))
            s.close()
            return json.loads(data.decode('utf-8'))
            
        def set(self, key, value):
            if isinstance(value, bytes):
                value = value.decode('latin1')
            self._send_req({'cmd': 'set', 'key': key, 'value': value})
            
        def get(self, key):
            resp = self._send_req({'cmd': 'get', 'key': key})
            return resp['value'].encode('latin1')
            
        def wait(self, keys, timeout=None):
            self._send_req({'cmd': 'wait', 'keys': keys})

    class PrefixStore(Store):
        """A real view onto another store, not a copy.

        `_new_process_group_helper` nests these two deep
        (`{group}/` then `{device}/`), and `distributed_c10d` relies on two
        groups sharing one underlying store without colliding.
        """

        def __init__(self, prefix, store):
            super().__init__()
            self.prefix = prefix
            self.underlying_store = store

        def _key(self, key):
            return f"{self.prefix}{key}" if self.prefix.endswith("/") else f"{self.prefix}/{key}"

        def set(self, key, value):
            self.underlying_store.set(self._key(key), value)

        def get(self, key):
            return self.underlying_store.get(self._key(key))

        def add(self, key, value):
            return self.underlying_store.add(self._key(key), value)

        def check(self, keys):
            return self.underlying_store.check([self._key(k) for k in keys])

        def compare_set(self, key, expected_value, desired_value):
            return self.underlying_store.compare_set(
                self._key(key), expected_value, desired_value)

        def delete_key(self, key):
            return self.underlying_store.delete_key(self._key(key))

        def multi_get(self, keys):
            return self.underlying_store.multi_get([self._key(k) for k in keys])

        def multi_set(self, keys, values):
            self.underlying_store.multi_set([self._key(k) for k in keys], values)

        def num_keys(self):
            return self.underlying_store.num_keys()

        def wait(self, keys, timeout=None):
            return self.underlying_store.wait([self._key(k) for k in keys])

    # -- Work --------------------------------------------------------------
    class Work:
        """Always complete.

        `distributed_c10d.py:2809` subclasses this at module scope, so it has
        to be a genuine type. Every collective a single rank can perform is
        done by the time the call returns -- there is no peer to wait for --
        so `is_completed()` is True rather than optimistic.
        """

        def __init__(self, result=None):
            self._result = [] if result is None else list(result)

        def is_completed(self):
            return True

        def is_success(self):
            return True

        def exception(self):
            return None

        def wait(self, timeout=None):
            return True

        def block_current_stream(self):
            return None

        def synchronize(self):
            return None

        def source_rank(self):
            return 0

        def _source_rank(self):
            return 0

        def result(self):
            return self._result

        def get_future(self):
            refuse("Work.get_future",
                   "there is no torch.futures.Future in this shim; the work is "
                   "already complete, so wait() is the whole of it")

        def boxed(self):
            refuse("Work.boxed", "no TorchScript ScriptObject in this shim")

        @staticmethod
        def unbox(obj):
            refuse("Work.unbox", "no TorchScript ScriptObject in this shim")

    class FakeWork(Work):
        pass

    def _tree_clone(obj):
        """Clone an output argument of any shape a collective uses.

        The collectives here take their destinations as a tensor, a list of
        tensors, or a list of lists of tensors. Nothing else appears, so
        nothing else is handled -- a shape this does not know would silently
        become a shared reference, which is the exact bug this whole file is
        about, so it raises instead.
        """
        if obj is None:
            return None
        if isinstance(obj, (list, tuple)):
            return [_tree_clone(x) for x in obj]
        if not hasattr(obj, "clone"):
            raise TypeError(
                "torch._C._distributed_c10d: cannot take ownership of a %r as "
                "a collective output" % (type(obj).__name__,))
        return obj.clone()

    def _tree_copy(dest, src):
        if dest is None:
            return
        if isinstance(dest, (list, tuple)):
            for d, s in zip(dest, src):
                _tree_copy(d, s)
            return
        dest.copy_(src)

    def _tree_flatten(obj, out=None):
        out = [] if out is None else out
        if obj is None:
            return out
        if isinstance(obj, (list, tuple)):
            for x in obj:
                _tree_flatten(x, out)
            return out
        out.append(obj)
        return out

    class AsyncWork(Work):
        """A collective that is still running, and that owns its output until `wait()`.

        The threading half of asynchrony is the easy half. The half with teeth
        is **ownership**, and it is what this class is for: while a collective
        is in flight, the caller's output tensor must hold something the
        implementation deliberately chose to give them, never a buffer being
        written underneath them.

        So the exchange never touches the caller's tensor at all. It runs
        against a private staging clone that this handle owns, and the only
        thing that copies staging onto the caller's buffer is **publication**,
        which happens at a synchronisation point and nowhere else. Two
        consequences, both deliberate, both tested in
        `rust/torch_c/pytests/test_asyncwork.py`:

        * Reading the output buffer before `wait()` yields the
          **pre-collective contents**, deterministically, on every run and at
          every speed. It is not a race that a fast worker thread happens to
          win: there is no code path on which that buffer changes without the
          caller asking for it. A test of this cannot pass by luck, which is
          the whole reason the design is this way round -- see
          `docs/distributed/ASYNCWORK.md` §3.
        * `is_completed()` is a synchronisation point too. It reports whether
          the exchange has finished, and if it has it publishes *before*
          returning True. So a caller who polls to True may then read the
          buffer and find the answer there, which is upstream gloo's contract;
          and a caller who polls to False still holds their own bytes.

        `docs/distributed/ASYNCWORK.md` §5 records the one place this is deliberately
        stricter than gloo: gloo's buffer becomes valid when the work does,
        whether or not anybody asked. Here it becomes valid when somebody asks.
        """

        def __init__(self, dest, staged):
            super().__init__()
            import threading
            self._dest = dest
            self._staged = staged
            self._finished = threading.Event()
            self._exception = None
            self._published = False
            self._publish_lock = threading.Lock()

        # -- worker side ---------------------------------------------------
        def _run(self, body):
            """Runs on the group's worker thread, and never raises out of it.

            A collective that fails takes its exception to whoever calls
            `wait()`, because that is the thread with a caller to tell. Letting
            it escape here would kill the worker and hang every later
            collective on a queue nobody drains.
            """
            try:
                body()
            except BaseException as exc:
                self._exception = exc
            finally:
                self._finished.set()

        # -- caller side ---------------------------------------------------
        def _publish(self):
            with self._publish_lock:
                if self._published:
                    return
                if self._exception is None:
                    _tree_copy(self._dest, self._staged)
                self._published = True

        def is_completed(self):
            if not self._finished.is_set():
                return False
            self._publish()
            return True

        def is_success(self):
            return self._finished.is_set() and self._exception is None

        def exception(self):
            return self._exception

        def wait(self, timeout=None):
            """Block until the collective is done, publish it, return True.

            Idempotent by construction: `_publish` is guarded, and a second
            `wait()` on a handle that has already completed finds the event set
            and the buffer published and does nothing but return True. That is
            upstream's contract for a completed handle -- a no-op, not an
            error -- and it is tested rather than asserted here.
            """
            seconds = None
            if timeout is not None:
                seconds = (timeout.total_seconds()
                           if hasattr(timeout, "total_seconds") else timeout)
                if seconds is not None and seconds <= 0:
                    seconds = None
            if not self._finished.wait(seconds):
                raise RuntimeError(
                    "torch._C._distributed_c10d.Work.wait: the collective did "
                    "not complete within %r" % (timeout,))
            self._publish()
            if self._exception is not None:
                raise self._exception
            return True

        def synchronize(self):
            self.wait()
            return None

        def result(self):
            self.wait()
            return _tree_flatten(self._dest)

    # -- the options structs ----------------------------------------------
    #
    # Plain records. `distributed_c10d` fills the fields in and hands them to a
    # backend; nothing here reads them, and the backend that would is the one
    # that does not exist yet.
    def _options_type(type_name, fields, bases=()):
        def __init__(self, **kwargs):
            for field, default in fields.items():
                setattr(self, field, default() if callable(default) else default)
            for field, value in kwargs.items():
                setattr(self, field, value)

        def __repr__(self):
            shown = ", ".join(f"{f}={getattr(self, f)!r}" for f in fields)
            return f"{type_name}({shown})"

        return type(type_name, bases or (), {
            "__module__": name, "__init__": __init__, "__repr__": __repr__})

    def _default_timeout():
        return datetime.timedelta(minutes=30)

    def _default_reduce_op():
        return _ReduceOp(_RedOpType.SUM)

    AllreduceOptions = _options_type("AllreduceOptions", {
        "reduceOp": _default_reduce_op, "timeout": _default_timeout,
        "asyncOp": True, "sparseIndices": None})
    option_types = {
        "BroadcastOptions": _options_type("BroadcastOptions", {
            "rootRank": 0, "rootTensor": 0, "timeout": _default_timeout,
            "asyncOp": True}),
        "AllreduceOptions": AllreduceOptions,
        "AllreduceCoalescedOptions": _options_type("AllreduceCoalescedOptions", {
            "reduceOp": _default_reduce_op, "timeout": _default_timeout,
            "asyncOp": True}, bases=(AllreduceOptions,)),
        "ReduceOptions": _options_type("ReduceOptions", {
            "reduceOp": _default_reduce_op, "rootRank": 0, "rootTensor": 0,
            "timeout": _default_timeout, "asyncOp": True}),
        "AllgatherOptions": _options_type("AllgatherOptions", {
            "timeout": _default_timeout, "asyncOp": True}),
        "GatherOptions": _options_type("GatherOptions", {
            "rootRank": 0, "timeout": _default_timeout, "asyncOp": True}),
        "ScatterOptions": _options_type("ScatterOptions", {
            "rootRank": 0, "timeout": _default_timeout, "asyncOp": True}),
        "ReduceScatterOptions": _options_type("ReduceScatterOptions", {
            "reduceOp": _default_reduce_op, "timeout": _default_timeout,
            "asyncOp": True}),
        "BarrierOptions": _options_type("BarrierOptions", {
            "device_ids": list, "device": None, "timeout": _default_timeout,
            "asyncOp": True}),
        "AllToAllOptions": _options_type("AllToAllOptions", {
            "timeout": _default_timeout, "asyncOp": True}),
    }

    # -- the flat enums ----------------------------------------------------
    class DebugLevel(enum.Enum):
        OFF = 0
        INFO = 1
        DETAIL = 2

    class ErrorType(enum.Enum):
        SUCCESS = 0
        TIMEOUT = 1
        COMM_ERROR = 2
        REMOTE_ERROR = 3

    class BuiltinCommHookType(enum.Enum):
        ALLREDUCE = 0
        FP16_COMPRESS = 1

    debug_level = [DebugLevel.OFF]

    def get_debug_level():
        return debug_level[0]

    def set_debug_level(level):
        debug_level[0] = level

    def set_debug_level_from_env():
        # `torch/distributed/__init__.py:170` calls this at import. Upstream
        # reads TORCH_DISTRIBUTED_DEBUG; the level only selects how much C++
        # logging happens, and there is none here.
        return None

    # -- Backend -----------------------------------------------------------
    class _BackendOptions:
        """`Backend.Options`.

        `device_mesh.py:72` evaluates `C10dBackend.Options | None` at module
        scope, which is a `TypeError` unless this is a real type -- so it is a
        nested class rather than an attribute holding something type-shaped.
        """

        def __init__(self, backend="undefined", timeout=None):
            self._backend = backend
            self._timeout = timeout if timeout is not None else _default_timeout()
            self.global_ranks_in_group = []
            self.group_name = ""
            self.use_pg_for_symm_mem_rendezvous = False

        @property
        def backend(self):
            return self._backend

    class Backend:
        """The base every concrete backend derives from.

        No concrete backend is registered here. `ProcessGroupGloo` and the rest
        stay *absent* rather than empty, because `distributed_c10d.py:204-242`
        reads their absence as `_GLOO_AVAILABLE = False` and routes around
        them; an empty class would set that flag True and send real collectives
        into a body that does nothing.
        """

        Options = _BackendOptions

        def __init__(self, rank=0, size=1):
            self._rank = rank
            self._size = size
            self._options = _BackendOptions()
            self._timeout = _default_timeout()

        def rank(self):
            return self._rank

        def size(self):
            return self._size

        def name(self):
            return type(self).__name__

        @property
        def supports_splitting(self):
            return False

        @property
        def supports_coalescing(self):
            return False

        @property
        def supports_time_estimate(self):
            return False

        @property
        def options(self):
            return self._options

        def set_timeout(self, timeout):
            self._timeout = timeout

        def _set_default_timeout(self, timeout):
            self._timeout = timeout

        def _set_sequence_number_for_group(self):
            return None

        def abort(self):
            return None

        def shutdown(self):
            return None

        def eager_connect_single_device(self, device=None):
            return None

        def get_error(self):
            return ErrorType.SUCCESS

        def supports_tensor_alloc(self, device):
            return False

        def allocate_tensor(self, size, *, dtype=None, device=None):
            refuse("Backend.allocate_tensor",
                   "no backend owns memory here; tensors come from the aten "
                   "dispatcher")

        @property
        def mem_allocator(self):
            return None

    class FakeProcessGroup(Backend):
        """`torch/testing/_internal/distributed/fake_pg.py` registers this as
        the `fake` backend at import time, and that module is on the road to
        `transformers` (`fsdp/_flat_param.py:31`). It exists so the
        registration succeeds.

        Upstream's own docstring says a fake process group "would produce wrong
        results for every collective", so nothing here tries to make it useful.
        """

        @staticmethod
        def _create_internal(rank, size, opts=None):
            return FakeProcessGroup(rank, size)

    # -- ProcessGroup ------------------------------------------------------
    class _ProcessGroupMeta(type):
        """A heap-type metaclass, so the metaclass can be replaced later.

        `device_mesh._register_distributed_opaque_types` demands that
        `ProcessGroup` carry `torch._opaque_base.OpaqueBaseMeta`, which
        upstream's pybind11 class does (measured on 2.13.0 -- `Work`, `Store`
        and `Backend` do not). `_C` cannot import the tree at its own import
        time, so that binding is late, and `__class__` assignment refuses when
        the *old* metaclass is `type`, because `type` is a static type. This
        exists to give it a heap type to start from. See `_c10d_init`.
        """

    class ProcessGroup(metaclass=_ProcessGroupMeta):
        BackendType = None  # replaced below; the enum needs the class to exist

        def __init__(self, store=None, rank=0, size=1, options=None):
            self._store = store
            self._rank = rank
            self._size = size
            self._backends = {}
            self._default_backend_type = ProcessGroup.BackendType.UNDEFINED
            self._group_name = ""
            self._group_desc = ""
            self.bound_device_id = None

        def rank(self):
            return self._rank

        def size(self):
            return self._size

        def name(self):
            return self._group_name

        def _get_backend_name(self):
            return self._default_backend_type.name.lower()

        def get_group_store(self):
            return self._store

        def _set_default_backend(self, backend_type):
            self._default_backend_type = backend_type

        def _register_backend(self, device, backend_type, backend):
            self._backends[torch_device_key(device)] = (backend_type, backend)
            self._default_backend_type = backend_type

        def _get_backend(self, device):
            entry = self._backends.get(torch_device_key(device))
            if entry is None:
                raise RuntimeError(
                    f"torch._C._distributed_c10d.ProcessGroup: no backend "
                    f"registered for device {device}"
                )
            return entry[1]

        def _has_hooks(self):
            return False

        def _set_group_name(self, group_name):
            self._group_name = group_name

        def _set_group_desc(self, group_desc):
            self._group_desc = group_desc

        @property
        def group_name(self):
            return self._group_name

        @property
        def group_desc(self):
            return self._group_desc

        def _set_sequence_number_for_group(self):
            return None

        def _set_default_timeout(self, timeout):
            return None

        def set_timeout(self, timeout):
            return None

        def abort(self):
            return None

        def shutdown(self):
            return None

        def _end_coalescing(self, *args, **kwargs):
            return Work()

        def _start_coalescing(self, *args, **kwargs):
            return None

        def __repr__(self):
            return (f"<torch.distributed.ProcessGroup rank={self._rank} "
                    f"size={self._size} name={self._group_name!r}>")

    class _BackendType(enum.Enum):
        UNDEFINED = 0
        GLOO = 1
        NCCL = 2
        UCC = 3
        MPI = 4
        XCCL = 5
        CUSTOM = 6

    _BackendType.__qualname__ = "ProcessGroup.BackendType"
    ProcessGroup.BackendType = _BackendType

    def torch_device_key(device):
        # `_register_backend` is keyed on `torch.device`, which is not
        # hashable-by-value in every spelling; its `type` is what selects a
        # backend, so that is the key.
        return getattr(device, "type", device)

    # -- the one backend this build has ------------------------------------
    #
    # DESIGN.md §11.1 puts a backend layer between `torch.distributed` and the
    # device abstraction, registered through `Backend.register_backend`. This
    # is that layer's world_size-1 member, and it is registered from
    # `torchnative`, not from here -- `register_backend` is a
    # `distributed_c10d` API and this module is imported before that file
    # exists.
    #
    # **Every method here is either exact or a refusal.** At world_size 1 a
    # collective has one contribution, so:
    #
    #   all_reduce      identity, for SUM/PRODUCT/MIN/MAX/BAND/BOR/BXOR --
    #                   a reduction over one element is that element -- and
    #                   for AVG, because the divisor is the world size.
    #                   **PREMUL_SUM is refused**: it computes
    #                   `sum(factor * x_i)`, which is `factor * x` and not `x`,
    #                   so identity would be a wrong answer rather than a
    #                   missing one.
    #   broadcast       no-op. The root can only be rank 0, which is this
    #                   process, so the tensor is already what it will be.
    #   barrier         no-op, and genuinely so: a barrier with one
    #                   participant is satisfied the moment it is reached.
    #   gather/scatter/
    #   all_gather/
    #   reduce_scatter/
    #   all_to_all      a copy. The buffers have one slot each and the data has
    #                   to move into them, so these are the ones that do work.
    #   send/recv       **refused by name.** There is no second rank, so there
    #                   is no correct behaviour to fall back on. A silent no-op
    #                   here would be the `filled` guard from docs/models/CKPT.md
    #                   again: the caller would read an unwritten buffer and
    #                   never learn why.
    #
    # A root rank other than 0 is refused rather than clamped -- asking rank 3
    # to be the root of a one-rank group is a bug in the caller, and answering
    # it would hide that.
    # Rendezvous waits for every peer to arrive; the data sockets carry one
    # collective's payload over loopback. They are different waits, so they are
    # different numbers. The data timeout is kept at the 30 s
    # docs/distributed/FEDERATED3.md §4 measured the dropout against; the rendezvous one is
    # generous on purpose, because it is waiting for N interpreters to import
    # torch on a loaded machine and not for bytes that are already in flight.
    _PG_RENDEZVOUS_TIMEOUT = 300.0
    _PG_SOCKET_TIMEOUT = 30.0

    class ProcessGroupLocal(Backend):
        """The collectives of a group of `size` processes joined by a star of sockets.

        Named `Local` rather than after a transport because there was none when
        it was written: the peer set was `{self}`. It now spans
        `world_size >= 1` over loopback TCP, and the name is kept because what
        it is local to is the machine -- docs/distributed/TRANSPORT.md, docs/distributed/FEDERATED4.md.

        **The topology is a star, and rank 0 is the hub.** Every other rank
        connects to it and talks to nobody else. A ring would halve the wire
        volume of an allreduce and is what a real backend does; it is not what
        this is, because a star makes the reduction order a property of one
        process rather than an emergent property of N. See `_star_exchange`.

        **The ordering contract.** The hub folds contributions in **ascending
        rank order**, `((x_0 + x_1) + x_2) + ...`, and sends *one* result back
        to every rank -- nobody adds anything locally. Float addition is not
        associative, so an implementation that summed in arrival order would
        give a different answer on a different day, and a test that allowed
        that difference would also allow a lost rank. This is the contract a
        test can hold: the same expression written centrally, in rank order,
        must be equal bit for bit.
        """

        def __init__(self, rank=0, size=1, store=None):
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError(
                    "torch._C._distributed_c10d.ProcessGroupLocal: world_size "
                    f"{size!r} is not a positive integer"
                )
            if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < size:
                raise ValueError(
                    "torch._C._distributed_c10d.ProcessGroupLocal: rank "
                    f"{rank!r} is not in a world of size {size}"
                )
            super().__init__(rank, size)
            self._store = store
            #: hub only -- {peer rank: socket}. Empty on every other rank.
            self._peers = {}
            #: leaf only -- the socket to rank 0. `None` on the hub.
            self._hub = None
            if size > 1:
                if store is None:
                    refuse(
                        f"ProcessGroupLocal at world_size {size} with no store",
                        "the ranks find each other through a Store -- rank 0 "
                        "publishes the port it bound and the others wait on "
                        "that key -- and none was passed",
                    )
                self._rendezvous(rank, size, store)

        def _rendezvous(self, rank, size, store):
            """Build the star. One store key, `size - 1` connections.

            Rank 0 binds an ephemeral port, publishes it, and accepts until it
            has heard from every other rank. Each leaf sends its own rank as
            the first four bytes it writes, because `accept` returns them in
            whatever order the kernel completed the handshakes and **the
            reduction order is by rank, not by arrival**. Without that greeting
            the hub would have to guess, and the guess would be right most of
            the time -- which is the shape of defect this whole layer refuses.
            """
            import socket, struct

            if rank == 0:
                listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listen.bind(("127.0.0.1", 0))
                listen.listen(size)
                listen.settimeout(_PG_RENDEZVOUS_TIMEOUT)
                store.set("pg_local_port",
                          str(listen.getsockname()[1]).encode("utf-8"))
                try:
                    while len(self._peers) < size - 1:
                        conn, _ = listen.accept()
                        conn.settimeout(_PG_SOCKET_TIMEOUT)
                        peer = struct.unpack("!I", self._recv_exact(conn, 4))[0]
                        if not 1 <= peer < size or peer in self._peers:
                            raise RuntimeError(
                                "torch._C._distributed_c10d.ProcessGroupLocal: "
                                f"a peer announced itself as rank {peer} in a "
                                f"world of size {size}; "
                                f"{sorted(self._peers)} had already arrived"
                            )
                        self._peers[peer] = conn
                finally:
                    listen.close()
            else:
                store.wait(["pg_local_port"])
                port = int(store.get("pg_local_port").decode("utf-8"))
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(_PG_SOCKET_TIMEOUT)
                sock.connect(("127.0.0.1", port))
                sock.sendall(struct.pack("!I", rank))
                self._hub = sock

        def name(self):
            return "local"

        # -- the wire ------------------------------------------------------
        #
        # Length-prefixed JSON, unchanged from docs/distributed/TRANSPORT.md §2. What
        # changed is who talks to whom.
        @staticmethod
        def _recv_exact(sock, n):
            data = b""
            while len(data) < n:
                packet = sock.recv(n - len(data))
                if not packet:
                    raise RuntimeError("connection closed")
                data += packet
            return data

        @staticmethod
        def _frame(payload):
            import json, struct
            data = json.dumps(payload).encode("utf-8")
            return struct.pack("!I", len(data)) + data

        def _recv_json(self, sock):
            import json, struct
            (length,) = struct.unpack("!I", self._recv_exact(sock, 4))
            return json.loads(self._recv_exact(sock, length).decode("utf-8"))

        #: The three faces of a peer that is gone. `_collective` in
        #: `torchnative.nn.federated` translates them into `RankDropped`; here
        #: they are only caught, never interpreted.
        _GONE = (OSError, EOFError, RuntimeError)

        @classmethod
        def _is_gone(cls, exc):
            if isinstance(exc, RuntimeError) and not isinstance(exc, OSError):
                return "connection closed" in str(exc)
            return isinstance(exc, cls._GONE)

        def _drain_async(self):
            """Only one collective may be on the star at a time.

            The star is one socket per peer carrying one length-prefixed frame
            per collective. Two collectives in flight at once would interleave
            their frames on that socket and each would read the other's bytes
            -- so asynchrony here buys *caller* overlap, not wire parallelism,
            and the queue is served by exactly one worker thread to keep the
            wire order equal to the issue order that c10d already requires of
            every rank.

            This is the other half of that invariant: a *synchronous*
            collective issued from the calling thread while asynchronous ones
            are still queued would jump the line. It waits for the queue to
            drain first. Called from the worker itself it is a no-op, or it
            would wait for the item it is currently running.
            """
            import threading
            queue_ = getattr(self, "_work_queue", None)
            if queue_ is None:
                return
            if threading.current_thread() is getattr(self, "_work_thread", None):
                return
            queue_.join()

        def _async_queue(self):
            """The group's single worker thread, started on first async use.

            Lazy because most groups never issue an `async_op=True` collective
            and a thread per group that nobody uses is a thread per group that
            can still deadlock a shutdown. Daemon, so a caller who drops a
            handle on the floor cannot wedge interpreter exit.
            """
            existing = getattr(self, "_work_queue", None)
            if existing is not None:
                return existing
            import queue as _queue, threading
            pending = _queue.Queue()
            self._work_queue = pending

            def _pump():
                while True:
                    item = pending.get()
                    if item is None:
                        pending.task_done()
                        return
                    work, body = item
                    try:
                        work._run(body)
                    finally:
                        pending.task_done()

            thread = threading.Thread(
                target=_pump, daemon=True,
                name="ProcessGroupLocal-async-rank%d" % (self._rank,))
            self._work_thread = thread
            thread.start()
            return pending

        def _async_shutdown(self):
            """Retire the worker thread. Idempotent, and safe if none was started."""
            pending = getattr(self, "_work_queue", None)
            if pending is None:
                return None
            pending.put(None)
            thread = getattr(self, "_work_thread", None)
            if thread is not None:
                thread.join(timeout=60)
            self._work_queue = None
            self._work_thread = None
            return None

        def shutdown(self):
            return self._async_shutdown()

        def abort(self):
            return self._async_shutdown()

        def _star_exchange(self, payload, fold, tolerate=False):
            """One collective. Returns ``(result, missing)``.

            Every leaf sends its payload to the hub and then reads one answer.
            The hub reads from **every** leaf before it writes to any of them,
            which is what makes this deadlock-free at any size: `sendall`
            returns when the kernel took the bytes, not when the peer read
            them (docs/distributed/FEDERATED.md's `_WIRE_*` note), so a leaf blocked
            mid-send is only ever waiting for a hub that is already draining.

            `fold(payloads, ranks)` runs **on the hub only**, over the ranks
            that arrived, in ascending rank order. Its result is what every
            survivor receives -- byte for byte the same JSON -- so no rank
            computes a sum of its own and no two ranks can round differently.

            `missing` is the hub's verdict and travels in the envelope, so the
            survivors agree about who is gone rather than each timing out
            separately. With `tolerate=False` a non-empty verdict is still
            returned rather than raised; the caller decides.
            """
            self._drain_async()
            if self._size == 1:
                return fold([payload], [0]), ()
            if self._rank == 0:
                contributions = {0: payload}
                missing = []
                for peer in sorted(self._peers):
                    try:
                        contributions[peer] = self._recv_json(self._peers[peer])
                    except Exception as exc:
                        if not self._is_gone(exc):
                            raise
                        missing.append(peer)
                present = [r for r in range(self._size) if r not in missing]
                result = fold([contributions[r] for r in present], present)
                frame = self._frame({"missing": missing, "result": result})
                late = []
                for peer in present:
                    if peer == 0:
                        continue
                    try:
                        self._peers[peer].sendall(frame)
                    except Exception as exc:
                        if not self._is_gone(exc):
                            raise
                        late.append(peer)
                if late:
                    # A rank that answered and then died before it could be
                    # told the answer. The ranks it was folded with have
                    # already been sent a result that counts it, so there is no
                    # verdict this can report that every survivor also holds.
                    # Refused rather than patched over -- docs/distributed/FEDERATED4.md.
                    raise RuntimeError(
                        "connection closed: rank(s) %s contributed to this "
                        "collective and then went away before the result "
                        "reached them, so the survivors do not agree about "
                        "who was in it" % (late,)
                    )
                return result, tuple(missing)
            try:
                self._hub.sendall(self._frame(payload))
                envelope = self._recv_json(self._hub)
            except Exception as exc:
                if not self._is_gone(exc):
                    raise
                # The hub is what every rank reaches the others through, so
                # losing it is losing all of them at once.
                return None, tuple(r for r in range(self._size)
                                   if r != self._rank)
            return envelope["result"], tuple(envelope["missing"])

        # -- reductions ----------------------------------------------------
        @staticmethod
        def _check_reduce_op(opts, what):
            op = getattr(opts, "reduceOp", None)
            kind = getattr(op, "op", op)
            if kind is _RedOpType.PREMUL_SUM:
                refuse(
                    f"ProcessGroupLocal.{what} with ReduceOp.PREMUL_SUM",
                    "PREMUL_SUM is `sum(factor * x_i)`, which at world_size 1 "
                    "is `factor * x` and not `x` -- so this is not the identity "
                    "the other reductions are, and it is not implemented",
                )
            if kind is _RedOpType.UNUSED:
                refuse(f"ProcessGroupLocal.{what} with ReduceOp.UNUSED",
                       "UNUSED names no reduction")

        @staticmethod
        def _check_root(opts, what):
            root = getattr(opts, "rootRank", 0)
            if root not in (0, -1):
                raise ValueError(
                    f"torch._C._distributed_c10d.ProcessGroupLocal.{what}: "
                    f"rootRank {root} is not in a world of size 1"
                )

        #: The reductions this backend folds above `world_size` 1. The value is
        #: the in-place `Tensor` method that folds one more contribution into
        #: the accumulator, applied in **ascending rank order** -- see the
        #: class docstring. `AVG` is `SUM` followed by one division, which is
        #: why it is not in this table.
        #: `SUM` and `PRODUCT` accumulate *in the tensor's own dtype*, so the
        #: bracketing is observable and the class docstring's ascending-rank
        #: order is the contract. `MIN`/`MAX` are not here: they are folded
        #: elementwise in `_extremum` instead, because neither rounds -- see
        #: `_reduce_fold`.
        _FOLD_STEP = {"SUM": "add_", "PRODUCT": "mul_"}

        #: The reductions folded above `world_size` 1, `AVG` included.
        _REDUCTIONS = ("SUM", "PRODUCT", "MIN", "MAX", "AVG")

        @classmethod
        def _extremum(cls, a, b, pick):
            """`pick` applied elementwise down two payloads of equal shape."""
            if isinstance(a, list):
                return [cls._extremum(x, y, pick) for x, y in zip(a, b)]
            return pick(a, b)

        def _reduce_kind(self, opts, what):
            """The `_RedOpType` this collective will fold with, or a refusal.

            `_check_reduce_op` has already turned away `PREMUL_SUM` and
            `UNUSED`, which name no fold at any size. What is left is the
            arithmetic four -- folded -- and the bitwise three, which are not.
            """
            op = getattr(opts, "reduceOp", None)
            kind = getattr(op, "op", op)
            if kind is None:
                kind = _RedOpType.SUM
            name = getattr(kind, "name", str(kind))
            if name in self._REDUCTIONS:
                return kind
            refuse(
                f"ProcessGroupLocal.{what} with ReduceOp.{name} at "
                f"world_size {self._size}",
                "the bitwise reductions are not implemented. SUM, PRODUCT, "
                "MIN, MAX and AVG are (docs/distributed/COLLECT2.md); BAND, BOR and BXOR "
                "are defined only on integral dtypes -- upstream gloo raises "
                "`Cannot use ReduceOp.BAND with non-integral dtype` -- and the "
                "float tables this layer's federated callers reduce are "
                "exactly the ones upstream would refuse, so there was no "
                "caller to build it for",
            )

        @classmethod
        def _reduce_fold(cls, template, kind=None):
            """Fold `payloads` with `kind`, in the order they are given.

            The order is the contract, not an implementation detail: float
            addition is not associative and neither is float multiplication,
            so `((x_0 op x_1) op x_2)` is a different number from any other
            bracketing and the class docstring pins which one this is.

            `AVG` divides by **the number of contributions folded**, which is
            the world size when nobody is missing. Under `tolerate=True` that
            is the survivor count rather than the world -- the same verdict the
            hub reports in the envelope, so every survivor divides by the same
            number. `docs/distributed/FEDERATED4.md` §6 is the argument for why that has to
            be one process's decision.
            """
            name = getattr(kind, "name", "SUM") if kind is not None else "SUM"

            if name in ("MIN", "MAX"):
                # Folded on the JSON payloads rather than through a tensor, and
                # this is exact rather than a shortcut: an extremum *selects* a
                # contribution, it never combines two, so no rounding happens at
                # any width and the float64 that `tolist` produced for a float32
                # value converts back to that same float32 bit for bit. It is
                # also the one place where the ascending-rank contract is
                # vacuous -- min and max are associative and commutative, so
                # every bracketing agrees. (`aten.minimum` is not implemented in
                # this shim; `aten.maximum` is. Doing both here rather than one
                # each way keeps the two symmetric.)
                pick = min if name == "MIN" else max

                def fold(payloads, ranks):
                    acc = payloads[0]
                    for payload in payloads[1:]:
                        acc = cls._extremum(acc, payload, pick)
                    return acc
                return fold

            step = cls._FOLD_STEP.get(name, "add_")

            def fold(payloads, ranks):
                import torch
                acc = torch.tensor(payloads[0], dtype=template.dtype,
                                   device=template.device)
                for payload in payloads[1:]:
                    getattr(acc, step)(torch.tensor(
                        payload, dtype=template.dtype, device=template.device))
                if name == "AVG":
                    acc = (acc / len(payloads)).to(template.dtype)
                return acc.tolist()
            return fold

        @classmethod
        def _sum_fold(cls, template):
            """`_reduce_fold` at `SUM`. Kept as its own name because
            `allreduce_partial`'s callers in `torchnative.nn.federated` are
            about summing deltas and nothing else."""
            return cls._reduce_fold(template, None)

        @staticmethod
        def _pick_fold(root, what):
            """Fold that keeps the root rank's contribution and drops the rest.

            This is what `broadcast` and `scatter` are on a star: every rank
            still sends, because the hub reads from everybody before it writes
            to anybody and that is what makes the exchange deadlock-free at any
            size (`_star_exchange`). The leaves' payloads are then thrown away.
            It costs `size - 1` payloads of wire that a real backend would not
            send; on loopback that is a copy, and the alternative is a second
            exchange shape whose deadlock argument would have to be made
            again from scratch.
            """
            def fold(payloads, ranks):
                for rank, payload in zip(ranks, payloads):
                    if rank == root:
                        return payload
                raise RuntimeError(
                    "connection closed: rank %d is the root of this %s and did "
                    "not arrive, so there is nothing to send" % (root, what))
            return fold

        def _check_root_n(self, opts, what):
            """`rootRank`, validated against *this* world rather than against 1."""
            root = getattr(opts, "rootRank", 0)
            if isinstance(root, bool) or not isinstance(root, int) \
                    or not 0 <= root < self._size:
                raise ValueError(
                    f"torch._C._distributed_c10d.ProcessGroupLocal.{what}: "
                    f"rootRank {root!r} is not in a world of size {self._size}"
                )
            return root

        def _allreduce(self, tensors, opts, tolerate):
            self._check_reduce_op(opts, "allreduce")
            if self._size == 1:
                return Work(tensors), ()
            kind = self._reduce_kind(opts, "allreduce")
            import torch
            missing = ()
            for t in tensors:
                result, gone = self._star_exchange(
                    t.tolist(), self._reduce_fold(t, kind), tolerate)
                if gone and not tolerate:
                    raise RuntimeError(
                        "connection closed: rank(s) %s did not contribute to "
                        "this allreduce" % (list(gone),))
                missing = gone
                if result is not None:
                    # Reconstructed from the hub's JSON even on the hub, so
                    # every rank ends holding bytes that went through the same
                    # encoding. `repr` of a float round-trips exactly, so this
                    # is a copy and not a second rounding.
                    t.copy_(torch.tensor(result, dtype=t.dtype,
                                         device=t.device))
            return Work(tensors), missing

        def allreduce(self, tensors, opts=None):
            work, _ = self._allreduce(tensors, opts, tolerate=False)
            return work

        def allreduce_partial(self, tensors, opts=None):
            """`allreduce` that completes over whoever arrived. Shim-only.

            Returns ``(Work, missing_ranks)``. There is no upstream spelling of
            this because upstream's allreduce either completes over the world
            or is an error, and so is `allreduce` above. This exists so that
            `Engine(on_missing='average_arrived')` can name a divisor: the
            survivor set is the **hub's** verdict, carried to every survivor in
            the same envelope as the data, so the ranks that go on divide by
            the same number rather than by whatever each of them timed out on.
            """
            if self._size == 1:
                refuse("ProcessGroupLocal.allreduce_partial at world_size 1",
                       "a world of one has no survivor set to report")
            return self._allreduce(tensors, opts, tolerate=True)
        def allreduce_coalesced(self, tensors, opts=None):
            if self._size != 1:
                refuse("ProcessGroupLocal.allreduce_coalesced", "Only world_size 1 is implemented")
            self._check_reduce_op(opts, "allreduce_coalesced")
            return Work(tensors)

        def reduce(self, tensors, opts=None):
            """`allreduce` whose answer is kept only by the root.

            **Only the root's tensor is specified.** Upstream gloo leaves a
            non-root's buffer holding whatever partial its place in the tree
            produced -- measured on this host at three ranks: rank 1 came back
            with 5.0 and rank 2 with 3.0 where the total was 6.0. Those are not
            errors, they are the absence of a promise. This leaves the non-root
            buffers *untouched*, which is inside the same absence, and
            `docs/distributed/COLLECT2.md` says so rather than letting a caller discover
            which of the two it got.
            """
            self._check_reduce_op(opts, "reduce")
            if self._size == 1:
                self._check_root(opts, "reduce")
                return Work(tensors)
            root = self._check_root_n(opts, "reduce")
            kind = self._reduce_kind(opts, "reduce")
            import torch
            for t in tensors:
                result, gone = self._star_exchange(
                    t.tolist(), self._reduce_fold(t, kind), False)
                if gone:
                    raise RuntimeError(
                        "connection closed: rank(s) %s did not contribute to "
                        "this reduce" % (list(gone),))
                if self._rank == root and result is not None:
                    t.copy_(torch.tensor(result, dtype=t.dtype,
                                         device=t.device))
            return Work(tensors)

        # -- movement ------------------------------------------------------
        def broadcast(self, tensors, opts=None):
            """The root rank's tensor, on every rank.

            Every rank sends and the hub keeps the root's -- `_pick_fold` says
            why the leaves' payloads are sent only to be discarded. The root
            may be any rank, not only the hub: `_check_root_n` validates
            against this world, and rank 0's answer travels back through the
            same envelope every other rank reads, so there is no path on which
            two ranks decode different bytes.
            """
            if self._size == 1:
                self._check_root(opts, "broadcast")
                return Work(tensors)
            root = self._check_root_n(opts, "broadcast")
            import torch
            for t in tensors:
                result, gone = self._star_exchange(
                    t.tolist(), self._pick_fold(root, "broadcast"), False)
                if gone:
                    raise RuntimeError(
                        "connection closed: rank(s) %s never reached this "
                        "broadcast" % (list(gone),))
                if result is not None:
                    t.copy_(torch.tensor(result, dtype=t.dtype,
                                         device=t.device))
            return Work(tensors)

        @staticmethod
        def _gather_fold(size):
            """Fold into ``{str(rank): payload}`` -- who said what, not a sum.

            Keyed by rank rather than positional, so that a caller reading it
            cannot mistake "the third thing that arrived" for "rank 2".
            """
            def fold(payloads, ranks):
                return {str(r): p for r, p in zip(ranks, payloads)}
            return fold

        def _allgather(self, source, tolerate):
            """``([payload per rank], missing)``, ordered by rank."""
            result, gone = self._star_exchange(
                source.tolist(), self._gather_fold(self._size), tolerate)
            if gone and not tolerate:
                raise RuntimeError(
                    "connection closed: rank(s) %s did not contribute to this "
                    "allgather" % (list(gone),))
            if result is None:
                return None, gone
            return [result.get(str(r)) for r in range(self._size)], gone

        def allgather(self, output_tensors, input_tensors, opts=None):
            """Every rank's tensor, on every rank, in rank order.

            This is what makes `federated.agree` exact above two ranks. A
            sum-and-compare (`total == value * world`) is an equality test only
            at two; at three it accepts `(h-1, h, h+1)`, so the check that two
            ranks disagreed would pass on three that did -- docs/distributed/FEDERATED4.md.
            """
            import torch
            for outputs, source in zip(output_tensors, input_tensors):
                if len(outputs) != self._size:
                    raise ValueError(
                        "torch._C._distributed_c10d.ProcessGroupLocal.allgather"
                        f": output list has {len(outputs)} slots for a world "
                        f"of size {self._size}"
                    )
                if self._size == 1:
                    outputs[0].copy_(source)
                    continue
                payloads, _ = self._allgather(source, tolerate=False)
                for rank, payload in enumerate(payloads):
                    outputs[rank].copy_(torch.tensor(
                        payload, dtype=outputs[rank].dtype,
                        device=outputs[rank].device))
            return Work([t for group in output_tensors for t in group])

        def allgather_partial(self, output_tensors, input_tensors, opts=None):
            """`allgather` that completes over whoever arrived. Shim-only.

            Returns ``(Work, missing_ranks)``. The slots belonging to a rank
            that did not report are **left as the caller passed them** rather
            than zeroed: a zero is a value, and a caller that ignored the
            returned survivor set would average it in. See `allreduce_partial`.
            """
            import torch
            if self._size == 1:
                refuse("ProcessGroupLocal.allgather_partial at world_size 1",
                       "a world of one has no survivor set to report")
            missing = ()
            for outputs, source in zip(output_tensors, input_tensors):
                if len(outputs) != self._size:
                    raise ValueError(
                        "torch._C._distributed_c10d.ProcessGroupLocal."
                        f"allgather_partial: output list has {len(outputs)} "
                        f"slots for a world of size {self._size}"
                    )
                payloads, missing = self._allgather(source, tolerate=True)
                if payloads is None:
                    continue
                for rank, payload in enumerate(payloads):
                    if payload is None:
                        continue
                    outputs[rank].copy_(torch.tensor(
                        payload, dtype=outputs[rank].dtype,
                        device=outputs[rank].device))
            return Work([t for group in output_tensors for t in group]), missing

        def _allgather_base(self, output, input, opts=None):
            """The flat spelling: `output` is `world_size` copies of `input`."""
            import torch
            if self._size == 1:
                output.copy_(input)
                return Work([output])
            payloads, _ = self._allgather(input, tolerate=False)
            flat = torch.cat([
                torch.tensor(p, dtype=output.dtype,
                             device=output.device).reshape(-1)
                for p in payloads])
            output.copy_(flat.reshape(output.shape))
            return Work([output])

        # 2.13's spelling of `_allgather_base`;
        # `distributed_c10d.py:4387` calls this one and `all_gather_into_tensor`
        # is the deprecated alias for it.
        def all_gather_single(self, output, input, opts=None):
            return self._allgather_base(output, input, opts)

        def all_gather_single_coalesced(self, outputs, inputs, opts=None):
            if self._size != 1:
                refuse("ProcessGroupLocal.all_gather_single_coalesced", "Only world_size 1 is implemented")
            for output, source in zip(outputs, inputs):
                output.copy_(source)
            return Work(list(outputs))

        def allgather_into_tensor_coalesced(self, outputs, inputs, opts=None):
            if self._size != 1:
                refuse("ProcessGroupLocal.allgather_into_tensor_coalesced", "Only world_size 1 is implemented")
            for output, source in zip(outputs, inputs):
                output.copy_(source)
            return Work(list(outputs))

        def allgather_coalesced(self, output_lists, input_list, opts=None):
            if self._size != 1:
                refuse("ProcessGroupLocal.allgather_coalesced", "Only world_size 1 is implemented")
            for outputs, source in zip(output_lists, input_list):
                outputs[0].copy_(source)
            return Work(input_list)

        def gather(self, output_tensors, input_tensors, opts=None):
            """`allgather` whose result lands only on the root.

            The wire is the same exchange -- the hub has every contribution
            either way -- so what distinguishes this from `allgather` is only
            which ranks copy out. A non-root passes no output list, and this
            does not invent one for it.
            """
            if self._size == 1:
                self._check_root(opts, "gather")
                for outputs, source in zip(output_tensors, input_tensors):
                    outputs[0].copy_(source)
                return Work([t for group in output_tensors for t in group])
            root = self._check_root_n(opts, "gather")
            import torch
            for index, source in enumerate(input_tensors):
                payloads, _ = self._allgather(source, tolerate=False)
                if self._rank != root:
                    continue
                outputs = output_tensors[index] if index < len(output_tensors) else []
                if len(outputs) != self._size:
                    raise ValueError(
                        "torch._C._distributed_c10d.ProcessGroupLocal.gather: "
                        f"the root's output list has {len(outputs)} slots for "
                        f"a world of size {self._size}"
                    )
                for rank, payload in enumerate(payloads):
                    outputs[rank].copy_(torch.tensor(
                        payload, dtype=outputs[rank].dtype,
                        device=outputs[rank].device))
            return Work([t for group in output_tensors for t in group])

        def scatter(self, output_tensors, input_tensors, opts=None):
            """Slot `i` of the root's list, to rank `i`.

            **This was silently wrong above one rank**: it copied `sources[0]`
            unconditionally, so rank 0 got the right chunk by coincidence and
            every other rank got its own untouched buffer -- no refusal, no
            error, a plausible tensor. `docs/distributed/COLLECT2.md` §1 measured it. It is
            the shape of defect this layer exists to refuse, and it survived
            because `world_size >= 3` was described as refusing here when in
            fact nothing checked the size at all.

            The star sends *one* frame to every rank, so the root cannot hand
            each leaf a different chunk in a single exchange. It broadcasts the
            whole list instead and each rank selects its own index. That is
            `size` times the wire a real backend would send; on loopback it is
            a memcpy, and it reuses an exchange whose deadlock-freedom is
            already argued rather than introducing a second shape.
            """
            if self._size == 1:
                self._check_root(opts, "scatter")
                for output, sources in zip(output_tensors, input_tensors):
                    output.copy_(sources[0])
                return Work(list(output_tensors))
            root = self._check_root_n(opts, "scatter")
            import torch
            for index, output in enumerate(output_tensors):
                if self._rank == root:
                    sources = input_tensors[index] if index < len(input_tensors) else []
                    if len(sources) != self._size:
                        raise ValueError(
                            "torch._C._distributed_c10d.ProcessGroupLocal."
                            f"scatter: the root's input list has {len(sources)} "
                            f"chunks for a world of size {self._size}"
                        )
                    payload = [t.tolist() for t in sources]
                else:
                    payload = None
                result, gone = self._star_exchange(
                    payload, self._pick_fold(root, "scatter"), False)
                if gone:
                    raise RuntimeError(
                        "connection closed: rank(s) %s never reached this "
                        "scatter" % (list(gone),))
                if result is not None:
                    output.copy_(torch.tensor(
                        result[self._rank], dtype=output.dtype,
                        device=output.device))
            return Work(list(output_tensors))

        def _reduce_scatter_fold(self, template, kind):
            """Fold chunk `j` across ranks, for every `j`, in rank order.

            Each payload is one rank's *list* of `size` chunks. The result is a
            list of `size` reduced chunks, of which each rank keeps its own.
            The per-chunk fold is `_reduce_fold`'s, so the rank ordering and
            the accumulation dtype are the same contract as `allreduce`'s and
            not a second one that could drift from it.
            """
            def fold(payloads, ranks):
                folded = []
                for j in range(self._size):
                    per_rank = [payload[j] for payload in payloads]
                    folded.append(
                        self._reduce_fold(template, kind)(per_rank, ranks))
                return folded
            return fold

        def reduce_scatter(self, output_tensors, input_tensors, opts=None):
            """Chunk `i` reduced across all ranks, delivered to rank `i`.

            **This was silently wrong above one rank**, in the same way
            `scatter` was and for the same reason: it copied `sources[0]` with
            no size check, so at three ranks it returned each rank's own first
            chunk, unreduced, and said nothing. Measured against upstream gloo
            on this host with the same inputs: gloo returned
            `[30,33] [36,39] [42,45]`, this returned `[0,1] [10,11] [20,21]`
            (`docs/distributed/COLLECT2.md` §1).

            Every rank sends all `size` of its chunks; the hub folds each chunk
            index across the ranks and sends the whole folded list back, and
            each rank keeps index `self._rank`. The hub therefore does `size`
            reductions where a real backend would do one per rank in parallel
            -- the cost of a star, paid on loopback.
            """
            self._check_reduce_op(opts, "reduce_scatter")
            if self._size == 1:
                # One rank, so the scatter picks slot 0 and the reduction over
                # a single element is that element.
                for output, sources in zip(output_tensors, input_tensors):
                    output.copy_(sources[0])
                return Work(list(output_tensors))
            kind = self._reduce_kind(opts, "reduce_scatter")
            import torch
            for output, sources in zip(output_tensors, input_tensors):
                if len(sources) != self._size:
                    raise ValueError(
                        "torch._C._distributed_c10d.ProcessGroupLocal."
                        f"reduce_scatter: the input list has {len(sources)} "
                        f"chunks for a world of size {self._size}"
                    )
                result, gone = self._star_exchange(
                    [t.tolist() for t in sources],
                    self._reduce_scatter_fold(output, kind), False)
                if gone:
                    raise RuntimeError(
                        "connection closed: rank(s) %s did not contribute to "
                        "this reduce_scatter" % (list(gone),))
                if result is not None:
                    output.copy_(torch.tensor(
                        result[self._rank], dtype=output.dtype,
                        device=output.device))
            return Work(list(output_tensors))

        def _reduce_scatter_base(self, output, input, opts=None):
            """The flat spelling: `input` is `world_size` chunks end to end.

            Above one rank this copied `input` into `output` wholesale, which
            at three ranks did not even have a shape to land in -- it raised
            `cannot broadcast [6] to [2]` from inside `copy_`. That is a
            refusal by accident, from the wrong layer, with a message about
            broadcasting; it is now the collective, and the size mismatch it
            *should* refuse is refused by name below.
            """
            self._check_reduce_op(opts, "_reduce_scatter_base")
            if self._size == 1:
                output.copy_(input)
                return Work([output])
            flat = input.reshape(-1)
            total = flat.shape[0]
            width = output.reshape(-1).shape[0]
            if total != width * self._size:
                raise ValueError(
                    "torch._C._distributed_c10d.ProcessGroupLocal."
                    f"reduce_scatter_tensor: the input holds {total} elements, "
                    f"which is not {self._size} chunks of the output's {width}"
                )
            chunks = [flat[j * width:(j + 1) * width] for j in range(self._size)]
            [out] = [output]
            self.reduce_scatter([out], [chunks], opts)
            return Work([output])

        def reduce_scatter_single(self, output, input, opts=None):
            return self._reduce_scatter_base(output, input, opts)

        def reduce_scatter_single_coalesced(self, outputs, inputs, opts=None):
            # The copy below is the identity that one rank makes true, and
            # nothing more. Above one rank it was returning each rank's own
            # input with no reduction and no refusal -- the same silent defect
            # `reduce_scatter` had (docs/distributed/COLLECT2.md §1). Refused by name here
            # rather than built: coalescing is a batching of the single form,
            # and there is no caller for it in this tree yet.
            if self._size != 1:
                refuse("ProcessGroupLocal.reduce_scatter_single_coalesced at world_size %d" % self._size,
                       "only the uncoalesced `reduce_scatter` and "
                       "`reduce_scatter_tensor` are implemented above one rank "
                       "(docs/distributed/COLLECT2.md)")
            self._check_reduce_op(opts, "reduce_scatter_single_coalesced")
            for output, source in zip(outputs, inputs):
                output.copy_(source)
            return Work(list(outputs))

        def reduce_scatter_tensor_coalesced(self, outputs, inputs, opts=None):
            # The copy below is the identity that one rank makes true, and
            # nothing more. Above one rank it was returning each rank's own
            # input with no reduction and no refusal -- the same silent defect
            # `reduce_scatter` had (docs/distributed/COLLECT2.md §1). Refused by name here
            # rather than built: coalescing is a batching of the single form,
            # and there is no caller for it in this tree yet.
            if self._size != 1:
                refuse("ProcessGroupLocal.reduce_scatter_tensor_coalesced at world_size %d" % self._size,
                       "only the uncoalesced `reduce_scatter` and "
                       "`reduce_scatter_tensor` are implemented above one rank "
                       "(docs/distributed/COLLECT2.md)")
            self._check_reduce_op(opts, "reduce_scatter_tensor_coalesced")
            for output, source in zip(outputs, inputs):
                output.copy_(source)
            return Work(list(outputs))

        def alltoall(self, output_tensors, input_tensors, opts=None):
            """Rank `i`'s chunk `j` becomes rank `j`'s chunk `i` -- a transpose.

            **This was silently wrong above one rank.** It copied each input to
            the output beside it, so every rank got its own data back
            unchanged: at three ranks it returned `[[0],[1],[2]]` where upstream
            gloo returns `[[0],[10],[20]]`. Nothing checked the world size and
            nothing refused (`docs/distributed/COLLECT2.md` §1).

            The hub gathers the full `size x size` matrix and sends all of it to
            everybody; each rank reads its own column. Every rank therefore
            receives every other rank's payload, which a real all-to-all does
            not do -- it is the star again, and it is the reason `send`/`recv`
            stay refused rather than being built on top of this.
            """
            if self._size == 1:
                for output, source in zip(output_tensors, input_tensors):
                    output.copy_(source)
                return Work(list(output_tensors))
            if len(input_tensors) != self._size or len(output_tensors) != self._size:
                raise ValueError(
                    "torch._C._distributed_c10d.ProcessGroupLocal.alltoall: "
                    f"{len(input_tensors)} inputs and {len(output_tensors)} "
                    f"outputs in a world of size {self._size}"
                )
            import torch
            matrix, gone = self._star_exchange(
                [t.tolist() for t in input_tensors],
                lambda payloads, ranks: {str(r): p
                                         for r, p in zip(ranks, payloads)},
                False)
            if gone:
                raise RuntimeError(
                    "connection closed: rank(s) %s never reached this alltoall"
                    % (list(gone),))
            if matrix is not None:
                for src in range(self._size):
                    out = output_tensors[src]
                    out.copy_(torch.tensor(
                        matrix[str(src)][self._rank],
                        dtype=out.dtype, device=out.device))
            return Work(list(output_tensors))

        def alltoall_base(self, output, input, output_split_sizes=None,
                          input_split_sizes=None, opts=None):
            """The flat spelling of `alltoall`, equal splits only.

            Uneven splits are refused by name rather than approximated. They are
            not a harder transpose -- they are a different collective, because
            each rank has to know every other rank's split vector before it can
            say where its own chunk lands, and that is an exchange this has not
            done. Silently treating them as equal is what the old body did to
            the whole operation.
            """
            if self._size == 1:
                output.copy_(input)
                return Work([output])
            for name, splits in (("output_split_sizes", output_split_sizes),
                                 ("input_split_sizes", input_split_sizes)):
                if splits is None or len(splits) == 0:
                    continue
                sizes = list(splits)
                if len(set(sizes)) != 1:
                    refuse(
                        "ProcessGroupLocal.all_to_all_single with uneven "
                        f"{name} at world_size {self._size}",
                        f"{name}={sizes} is not one width repeated. An uneven "
                        "all-to-all needs every rank to know every other "
                        "rank's split vector before it can place its own "
                        "chunk, which is an exchange this backend does not do. "
                        "Equal splits are implemented (docs/distributed/COLLECT2.md)",
                    )
            flat_in = input.reshape(-1)
            total = flat_in.shape[0]
            if total % self._size:
                raise ValueError(
                    "torch._C._distributed_c10d.ProcessGroupLocal."
                    f"all_to_all_single: {total} input elements do not divide "
                    f"into {self._size} equal chunks"
                )
            width = total // self._size
            out_total = output.reshape(-1).shape[0]
            if out_total != width * self._size:
                raise ValueError(
                    "torch._C._distributed_c10d.ProcessGroupLocal."
                    f"all_to_all_single: the output holds {out_total} elements "
                    f"where {width * self._size} are being scattered into it"
                )
            import torch
            ins = [flat_in[j * width:(j + 1) * width] for j in range(self._size)]
            outs = [torch.zeros(width, dtype=output.dtype, device=output.device)
                    for _ in range(self._size)]
            self.alltoall(outs, ins, opts)
            output.copy_(torch.cat(outs).reshape(output.shape))
            return Work([output])

        # `distributed_c10d.py:5131`'s spelling. With one rank the split sizes
        # describe a single chunk that goes to itself, so this is the copy.
        def all_to_all_single(self, output, input, output_split_sizes=None,
                              input_split_sizes=None, opts=None):
            return self.alltoall_base(
                output, input, output_split_sizes, input_split_sizes, opts)

        def barrier(self, opts=None):
            """A real barrier above world_size 1, not a no-op that reports one.

            At one rank it is genuinely satisfied on arrival. Above one it is
            one trip through the star: nobody leaves until the hub has heard
            from everybody, which is what a barrier is. Returning `Work()`
            here would be the `filled` guard from docs/models/CKPT.md again -- a
            caller that used a barrier to order two collectives would be
            told it happened.
            """
            if self._size == 1:
                return Work()
            _, gone = self._star_exchange(0, lambda payloads, ranks: 0,
                                          tolerate=False)
            if gone:
                raise RuntimeError(
                    "connection closed: rank(s) %s never reached this barrier"
                    % (list(gone),))
            return Work()

        # -- point to point ------------------------------------------------
        #
        # Named refusals. These are the operations that cannot be made true by
        # any amount of local work.
        def send(self, tensors, dst_rank, tag=0):
            refuse("ProcessGroupLocal.send",
                   f"there is no route from rank {self._rank} to rank "
                   f"{dst_rank}. The topology is a star through rank 0 "
                   "(docs/distributed/FEDERATED4.md), so a message between two leaves "
                   "would have to be relayed by the hub, and a relay that "
                   "the hub can read is not the primitive secure aggregation "
                   "needs. Refused rather than faked")

        def recv(self, tensors, src_rank, tag=0):
            refuse("ProcessGroupLocal.recv",
                   f"there is no route from rank {src_rank} to rank "
                   f"{self._rank}; the topology is a star through rank 0")

        def recv_anysource(self, tensors, tag=0):
            refuse("ProcessGroupLocal.recv_anysource",
                   "the star gives no way to name the sender of the next "
                   "frame that arrives")

        def _start_coalescing(self, *args, **kwargs):
            return None

        def _end_coalescing(self, *args, **kwargs):
            return Work()

    # -- genuine async_op --------------------------------------------------
    #
    # Every collective body above is synchronous, and stays exactly as it is.
    # Asynchrony is added *around* them rather than inside them, for a reason
    # that is about evidence rather than tidiness: the synchronous bodies are
    # what `test_collect2.py` proved equal to upstream gloo at world 3 and 4,
    # and rewriting twelve of them to be re-entrant would have put every one of
    # those results back in question to buy a property none of them is about.
    #
    # So each public collective is renamed `_sync_<name>` and replaced by a
    # wrapper that either calls it inline -- today's behaviour, unchanged, for
    # `asyncOp=False` and for world_size 1 -- or hands it to the worker thread
    # with its **output argument swapped for a staging clone**. See
    # `AsyncWork` for why the swap is the whole design.
    #
    # The value keyed here is the positional index of the output argument.
    # `barrier` has none. `allreduce_partial` and `allgather_partial` are
    # deliberately absent: they return `(Work, missing_ranks)`, and the
    # survivor verdict is not a thing a caller can be handed later, so they
    # stay synchronous (docs/distributed/ASYNCWORK.md §7).
    _ASYNC_COLLECTIVES = {
        "allreduce": 0,
        "reduce": 0,
        "broadcast": 0,
        "allgather": 0,
        "_allgather_base": 0,
        "gather": 0,
        "scatter": 0,
        "reduce_scatter": 0,
        "_reduce_scatter_base": 0,
        "alltoall": 0,
        "alltoall_base": 0,
        "barrier": None,
    }

    def _async_opts(args, kwargs):
        """The options struct among the arguments, wherever it sits.

        `alltoall_base` takes it fifth and `allreduce` second, so it is found
        by the one attribute that matters rather than by position. Nothing else
        a collective is passed carries an `asyncOp`.
        """
        for value in list(args) + list(kwargs.values()):
            if hasattr(value, "asyncOp"):
                return value
        return None

    def _make_async_collective(method_name, out_index, body):
        def wrapper(self, *args, **kwargs):
            opts = _async_opts(args, kwargs)
            wants_async = bool(getattr(opts, "asyncOp", False))
            if out_index is not None and len(args) <= out_index:
                # Called with the output as a keyword. Nothing in this tree
                # does, and guessing which keyword it was would be the kind of
                # cleverness that makes ownership unprovable, so it runs
                # synchronously and says nothing false about itself.
                wants_async = False
            if self._size == 1 or not wants_async:
                # No drain here. `_star_exchange` is the single point every
                # collective body reaches, and the drain lives there instead.
                # It was in both places for a while, and the cost of that was
                # not performance: **neither copy could be nullified on its
                # own**, because the other one silently covered it, so the
                # suite could see the property and neither of its two
                # implementations. One choke point is one thing a test can
                # delete -- docs/distributed/ASYNCWORK.md §8.
                return body(self, *args, **kwargs)
            if out_index is None:
                dest = staged = None
                call_args = args
            else:
                dest = args[out_index]
                staged = _tree_clone(dest)
                call_args = list(args)
                call_args[out_index] = staged
            work = AsyncWork(dest, staged)
            self._async_queue().put(
                (work, lambda: body(self, *call_args, **kwargs)))
            return work

        wrapper.__name__ = method_name
        wrapper.__qualname__ = "ProcessGroupLocal." + method_name
        wrapper.__doc__ = body.__doc__
        return wrapper

    for _name, _out in _ASYNC_COLLECTIVES.items():
        _body = ProcessGroupLocal.__dict__[_name]
        setattr(ProcessGroupLocal, "_sync_" + _name, _body)
        setattr(ProcessGroupLocal, _name,
                _make_async_collective(_name, _out, _body))

    # The collectives `distributed_c10d` calls on the *group* object, forwarded
    # to whichever backend the group registered. Upstream's ProcessGroup does
    # this dispatch in C++ and picks the backend by the tensors' device; here
    # there is at most one backend, so the lookup is the registration itself.
    def _forwarded(method_name):
        def forward(self, *args, **kwargs):
            backend = self._sole_backend(method_name)
            return getattr(backend, method_name)(*args, **kwargs)

        forward.__name__ = method_name
        forward.__qualname__ = f"ProcessGroup.{method_name}"
        return forward

    def _sole_backend(self, method_name):
        if not self._backends:
            raise RuntimeError(
                f"torch._C._distributed_c10d.ProcessGroup.{method_name}: this "
                "group has no backend registered. init_process_group has to "
                "choose one, and this build only has 'local' -- which "
                "torchnative registers"
            )
        return next(iter(self._backends.values()))[1]

    ProcessGroup._sole_backend = _sole_backend
    for _method in (
        "allreduce", "allreduce_coalesced", "reduce", "broadcast",
        "allgather", "_allgather_base", "allgather_coalesced",
        "allgather_into_tensor_coalesced",
        "all_gather_single", "all_gather_single_coalesced",
        "gather", "scatter",
        "reduce_scatter", "_reduce_scatter_base", "reduce_scatter_single",
        "reduce_scatter_single_coalesced", "reduce_scatter_tensor_coalesced",
        "alltoall", "alltoall_base", "all_to_all_single", "barrier",
        "send", "recv", "recv_anysource",
        # Shim-only, and not upstream spellings: they return
        # `(Work, missing_ranks)` rather than `Work`. Forwarded so that
        # `torchnative.nn.federated` can reach them through the group object
        # it already holds instead of reaching into `_backends`.
        "allreduce_partial", "allgather_partial",
    ):
        setattr(ProcessGroup, _method, _forwarded(_method))

    # -- module contents ---------------------------------------------------
    mod.ReduceOp = _ReduceOp
    mod.Store = Store
    mod.HashStore = HashStore
    mod.FileStore = FileStore
    mod.TCPStore = TCPStore
    mod.PrefixStore = PrefixStore
    mod.Work = Work
    mod.FakeWork = FakeWork
    mod.DebugLevel = DebugLevel
    mod.ErrorType = ErrorType
    mod.BuiltinCommHookType = BuiltinCommHookType
    mod.get_debug_level = get_debug_level
    mod.set_debug_level = set_debug_level
    mod.set_debug_level_from_env = set_debug_level_from_env
    mod.Backend = Backend
    mod.FakeProcessGroup = FakeProcessGroup
    mod.ProcessGroupLocal = ProcessGroupLocal
    mod.ProcessGroup = ProcessGroup
    for type_name, option_type in option_types.items():
        setattr(mod, type_name, option_type)

    # A class body does not see the enclosing function's locals, so
    # `__module__` is assigned here rather than written inside each `class`.
    for value in list(vars(mod).values()):
        if isinstance(value, type) or isinstance(value, enum.EnumMeta):
            value.__module__ = name
    _BackendOptions.__module__ = name
    _BackendOptions.__qualname__ = "Backend.Options"
    _RedOpType.__module__ = name
    _BackendType.__module__ = name

    # `distributed_c10d.py` reads these as values, not as calls.
    #
    # `constants.py:12` is `default_pg_timeout = _DEFAULT_PG_TIMEOUT`, which is
    # then a default argument in a dozen signatures, so a placeholder here
    # would be carried into all of them.
    mod._DEFAULT_FIRST_BUCKET_BYTES = 1024 * 1024
    mod._DEFAULT_NO_TIMEOUT = datetime.timedelta(milliseconds=-1)
    mod._DEFAULT_PG_TIMEOUT = datetime.timedelta(minutes=30)
    mod._DEFAULT_PG_NCCL_TIMEOUT = datetime.timedelta(minutes=10)

    # A per-process registry, which is what upstream's is -- the group objects
    # live in C++ there and in `_world` here, and `_resolve_process_group` is
    # how the functional collectives get from a name back to a group.
    registered_groups = {}

    def _register_process_group(group_name, process_group):
        registered_groups[group_name] = process_group

    def _resolve_process_group(group_name):
        group = registered_groups.get(group_name)
        if group is None:
            raise ValueError(
                f"torch._C._distributed_c10d._resolve_process_group: no group "
                f"named {group_name!r} has been registered"
            )
        return group

    def _unregister_process_group(group_name):
        registered_groups.pop(group_name, None)

    def _unregister_all_process_groups():
        registered_groups.clear()

    # Process-global bookkeeping. Upstream keeps these in C++ statics; nothing
    # about them needs a peer, so they are implemented rather than refused.
    # `_update_default_pg` (distributed_c10d.py:1448) calls `_set_global_rank`
    # on every `init_process_group`, so a refusal here stops initialisation.
    global_state = {"rank": -1, "pg": None, "inflight_as_graph_input": False}

    def _set_global_rank(rank):
        global_state["rank"] = rank

    def _get_global_rank():
        return global_state["rank"]

    def _set_process_group(pg):
        global_state["pg"] = pg

    def _current_process_group():
        pg = global_state["pg"]
        if pg is None:
            raise RuntimeError(
                "torch._C._distributed_c10d._current_process_group: none is "
                "current; call init_process_group first"
            )
        return pg

    def _set_allow_inflight_collective_as_graph_input(value):
        global_state["inflight_as_graph_input"] = bool(value)

    def _allow_inflight_collective_as_graph_input():
        return global_state["inflight_as_graph_input"]

    def _get_work_registry_size():
        # Nothing is ever in flight: every Work this module hands back is
        # already complete, so the registry of pending work is empty by
        # construction rather than by not being kept.
        return 0

    mod._set_global_rank = _set_global_rank
    mod._get_global_rank = _get_global_rank
    mod._set_process_group = _set_process_group
    mod._current_process_group = _current_process_group
    mod._set_allow_inflight_collective_as_graph_input = (
        _set_allow_inflight_collective_as_graph_input)
    mod._allow_inflight_collective_as_graph_input = (
        _allow_inflight_collective_as_graph_input)
    mod._get_work_registry_size = _get_work_registry_size
    mod._shim_global_state = global_state

    mod._register_process_group = _register_process_group
    mod._resolve_process_group = _resolve_process_group
    mod._unregister_process_group = _unregister_process_group
    mod._unregister_all_process_groups = _unregister_all_process_groups
    mod._shim_registered_process_groups = registered_groups

    # Everything not named above. `torch/distributed/__init__.py:49-75` imports
    # a dozen more names (`Reducer`, `Logger`, `GradBucket`,
    # `_broadcast_coalesced`, ...) and only binds them; they belong to DDP's
    # gradient-bucketing machinery, which needs a second rank to mean anything.
    # They exist and refuse, which is what DESIGN.md §6 asks for -- the message
    # names the missing thing rather than the caller discovering a no-op later.
    class _MissingC10d:
        __slots__ = ("_shim_name",)

        def __init__(self, missing_name):
            self._shim_name = missing_name

        def __call__(self, *args, **kwargs):
            refuse(self._shim_name,
                   "not implemented; world_size 1 has no peer, so this has no "
                   "meaning here")

        def __bool__(self):
            # Rule 2 from `_Unimplemented`: a placeholder must not answer a
            # truth test, because a truthy stub silently selects a branch.
            raise NotImplementedError(
                f"torch._C._distributed_c10d.{self._shim_name} was asked for a "
                "truth value, and a placeholder has no honest answer"
            )

        def __repr__(self):
            return f"<unimplemented torch._C._distributed_c10d.{self._shim_name}>"

    # Names that must stay *absent*, not become placeholders.
    #
    # `distributed_c10d.py:204-242` imports each of these in its own
    # `try/except ImportError` and records the result as `_GLOO_AVAILABLE` and
    # friends. `from ... import X` turns an `AttributeError` from a module
    # `__getattr__` into an `ImportError`, so absence is how the tree is told
    # "this backend is not built" -- and that is true here.
    #
    # The catch-all below would otherwise synthesise them, because they are
    # capitalised and it treats capitalised names as types. It did, and the
    # result was `_GLOO_AVAILABLE = True` followed by
    # `init_process_group(backend="gloo")` getting a hollow object instead of
    # the tree's own "Distributed package doesn't have GLOO built in". A
    # placeholder here does not refuse -- it *changes the answer to a question
    # about what exists*, which is the distinction the "Deliberate omissions"
    # block at the top of this file draws.
    absent_backends = frozenset({
        "ProcessGroupGloo", "ProcessGroupNCCL", "ProcessGroupUCC",
        "ProcessGroupXCCL", "ProcessGroupMPI", "_ProcessGroupWrapper",
    })
    mod._shim_absent_backends = sorted(absent_backends)

    def _module_getattr(missing_name):
        if missing_name.startswith("__") and missing_name.endswith("__"):
            raise AttributeError(missing_name)
        if missing_name in absent_backends:
            raise AttributeError(
                f"torch._C._distributed_c10d has no {missing_name}: this build "
                "has no transport, so no wire backend is compiled in. The "
                "absence is the answer distributed_c10d.py's try/except is "
                "written to read"
            )
        if missing_name[:1].isupper() or missing_name.lstrip("_")[:1].isupper():
            # Capitalised means a type, and `distributed_c10d.py` puts several
            # of them in annotations, which are evaluated at def time -- `|` on
            # a non-type is a `TypeError`. Same reasoning as the `_ShimMeta`
            # branch in `install`.
            value = _ShimMeta(missing_name, (),
                              {"__module__": name, "__init__": _permissive_init})
        else:
            value = _MissingC10d(missing_name)
        setattr(mod, missing_name, value)
        return value

    # The names the stubs declare and this module does not implement.
    #
    # Built the same way the generic submodule loop above builds every other
    # `_C` submodule -- `_build_type` gives each declared type its declared
    # members as `_Unimplemented`, so `Reducer().prepare_for_backward(...)`
    # raises `NotImplementedError` naming the member. Reaching them through the
    # catch-all instead produced a bare type whose methods were an
    # `AttributeError`, which says "no such thing" where the truth is "not
    # built" -- a different answer to a different question.
    #
    # Most of what lands here is DDP's gradient-bucketing machinery (`Reducer`,
    # `Logger`, `GradBucket`, `_broadcast_coalesced`,
    # `_compute_bucket_assignment_by_size`), which needs a second rank before
    # any of it means anything.
    declared_types = {}
    for type_name, type_spec in _order_types(spec.get("types", {})):
        if hasattr(mod, type_name) or type_name in absent_backends:
            continue
        declared_types[type_name] = _build_type(
            type_name, type_spec, name, declared_types)
        setattr(mod, type_name, declared_types[type_name])
    for fn_name in spec.get("functions", ()):
        if not hasattr(mod, fn_name):
            setattr(mod, fn_name, _MissingC10d(fn_name))
    for value_name in spec.get("values", ()):
        if not hasattr(mod, value_name):
            setattr(mod, value_name, _MissingC10d(value_name))

    mod.__getattr__ = _module_getattr

    setattr(module, "_distributed_c10d", mod)
    sys.modules[name] = mod

    # -- the switch --------------------------------------------------------
    def _c10d_init():
        """`torch/distributed/__init__.py:38`, the tree's own init hook.

        Two things happen here that cannot happen when `_C` is imported.

        The first is the reason this function has a body at all: upstream's
        `ProcessGroup` carries `torch._opaque_base.OpaqueBaseMeta` as its
        metaclass, and `device_mesh._register_distributed_opaque_types`
        refuses anything that does not. `_C` cannot reach into the Python tree
        while `torch/__init__.py` is still running -- and must not, when it is
        loaded standalone as `_C` by the golden harness, where `import torch`
        would pull in a *different* torch. By the time this is called,
        `torch.distributed` is being imported, so the tree is certainly there.

        The shape is `_set_generator_metaclass`'s (VENDOR.md wall 19) with the
        direction reversed: there the tree pushes the metaclass in, here `_C`
        pulls it, because nothing in the tree offers to push this one.
        """
        if module.__name__ == "torch._C":
            from torch._opaque_base import OpaqueBaseMeta

            if type(ProcessGroup) is not OpaqueBaseMeta:
                ProcessGroup.__class__ = OpaqueBaseMeta
        return True

    module._c10d_init = _c10d_init


def _install_default_generator(module) -> None:
    """`torch.default_generator` -- an object with state, not a placeholder.

    `torch/random.py:24` is `from torch._C import default_generator` and every
    function in that module is a method call on it, so the name has to exist
    before `import torch` finishes. It has existed all along as an
    `_Unimplemented`; what changes here is that the CPU RNG behind it is real
    (`rng.rs`), so `manual_seed` can mean what it means upstream -- the same
    seed gives the same numbers as torch, bit for bit for `uniform_`.

    It is an *instance of the synthesised `Generator` class*, not a new type,
    for two reasons that both bite:

      * `_TypeChecker._base` answers the schema's `Generator?` parameter with
        `isinstance(value, module.Generator)`, so anything else would fail to
        bind `tensor.uniform_(a, b, generator=torch.default_generator)`.
      * `Generator` has to stay a heap type -- `_set_generator_metaclass`
        reassigns its `__class__` (VENDOR.md wall 19), which a PyO3 static type
        cannot do.

    Three methods are real and the rest keep refusing. `get_state`/`set_state`
    are the notable absence: upstream's is a 5056-byte legacy blob whose layout
    is the MT19937 struct itself (docs/numerics/RNG.md §1.1), and exchanging that blob
    with real torch is a separate piece of work from reproducing the stream.
    Left refusing by name rather than given a format of our own, because a
    round-trippable-but-incompatible blob is the kind of thing that looks like
    interop until someone tries it.
    """
    generator = module.Generator()

    # Read by `generator_arg` in aten.rs: it is how a kernel tells "the default
    # generator was named explicitly" from "some other generator was", and the
    # second is refused rather than quietly served from this stream.
    generator._shim_is_default_generator = True

    def manual_seed(seed):
        module._shim_manual_seed(int(seed))
        return generator

    def seed():
        return module._shim_reseed()

    def initial_seed():
        return module._shim_initial_seed()

    generator.manual_seed = manual_seed
    generator.seed = seed
    generator.initial_seed = initial_seed
    # `.device` is left alone on purpose: the synthesised `Generator` carries
    # it as a *property* (it is one on upstream's type too), and a property has
    # no setter, so an instance attribute cannot shadow it. Overwriting the
    # class attribute would change it for every `Generator`, which is the wrong
    # shape for a per-instance value -- and nothing reads it on this path.

    module.default_generator = generator
