"""Serialise a lowered trace into the model blob NNAPI consumes.

## The question that decides the shape of this module

Upstream ships a real NNAPI serialiser in the vendored tree,
`torch/backends/_nnapi/serializer.py`. docs/DECOMP.md §12 read its `ADDER_MAP`
for the *operator set*. The next question is whether the same file can be
*driven*, or only read. The answer is the first line of docs/NPU.md, and it is
neither yes nor no:

    `_NnapiSerializer.serialize_model(model, inputs)` needs a TorchScript IR
    graph -- `model.graph`. This shim has no TorchScript compiler. `torch.jit
    .trace` returns the module unchanged and `torch._C.Graph` is a placeholder
    class with a docstring that says so. So the *entry point* is not drivable.

    But the entry point is the only part that is. `serialize_model` reaches its
    argument through a surface of **thirteen methods** -- `Graph.{inputs,
    nodes, return_node}`, `Node.{kind, inputsSize, outputsSize, inputsAt,
    outputsAt, inputs, s}`, `Value.{type, toIValue}`, `Type.{kind,
    getElementType}` -- and every one of them is duck-typed. Values are used as
    plain dict keys, hashed by identity. Nothing downcasts, nothing calls into
    C++.

So this module does **not** write a serialiser. It writes the *graph façade*:
enough of a TorchScript IR to satisfy those thirteen methods, built from our
recorded trace. Every operand, every immediate, every byte of the blob layout
and the whole `ADDER_MAP` is upstream's code running unmodified. What is ours
is `to_jit_module` and `_SIGNATURES`.

## Why `_SIGNATURES` has to exist, and what it is not

docs/DECOMP.md §12.2 flagged this and left it as a caveat; here it becomes a
concrete obstacle. `ADDER_MAP` is keyed by TorchScript *node kind*
(`aten::add`), which carries no overload, and every adder then asserts an exact
`inputsSize()` -- `aten::add` wants three inputs, `aten::_convolution` wants
thirteen. Capture records post-dispatch ATen with overloads and with defaulted
arguments *dropped*: `aten.add.Tensor` arrives with two args and no `alpha`,
`aten.convolution.default` with nine.

`_SIGNATURES` is the map between those two callings. It is a *calling
convention* table -- which recorded argument goes in which TorchScript
position, and what constant fills a position capture did not record. It is not
a semantics table: no entry changes what an op computes, and an entry that
cannot be written faithfully is absent rather than approximated.

The distinction matters because getting it wrong is silent. Passing
`aten.convolution`'s `output_padding` where `_convolution` expects `groups`
serialises a blob that NNAPI will happily accept and that computes something
else. So `verify_shapes` re-derives, from the finished operand table, the shape
of every operand the serialiser assigned to a node output, and compares it with
the shape *capture recorded for that same node*. Those two numbers come from
different places -- one from upstream's shape propagation over NNAPI operands,
one from a real CPU execution -- and a mis-wired argument moves one and not the
other.

## What is executed here and what is not

There is no NNAPI runtime on this machine, so nothing in this module claims a
blob was run. What it claims:

* `serialize()` runs upstream's serialiser to completion and returns its blob.
* `parse_model()` decodes that blob back through the layout `serialize_model`
  wrote it in, and is a real decoder: it walks the operand, value, operation
  and argument tables, checks every table length against the header, checks
  every operand reference is in range, and refuses on the first inconsistency.
  A blob that survives it is structurally a well-formed NNAPI model.
* `verify_shapes()` is the semantic check described above.

None of the three is "this ran on an NPU". docs/NPU.md says so in the same
words, because merging the two claims is the specific failure this project has
paid for before.
"""

from __future__ import annotations

import array
import struct
from typing import Any

from .decompose import DecomposedTrace

__all__ = [
    "JitFacadeRefused",
    "fold_constants",
    "NnapiModel",
    "jit_graph_is_available",
    "parse_model",
    "serialize",
    "supported_ops",
    "to_jit_module",
    "verify_shapes",
]


class JitFacadeRefused(RuntimeError):
    """The trace cannot be presented to upstream's serialiser, and why."""


# ---------------------------------------------------------------------------
# Why the façade is needed at all
# ---------------------------------------------------------------------------


def jit_graph_is_available() -> bool:
    """Whether a real TorchScript graph can be obtained from this build.

    False on the shim, and that is the finding this module is built around
    rather than a defect to route past. Kept as a function so the claim is
    re-measured on whatever build runs, instead of being asserted in prose that
    goes stale the day a `jit` frontend lands.
    """
    import torch

    module = torch.nn.Linear(2, 2)
    try:
        traced = torch.jit.trace(module, torch.zeros(1, 2))
    except Exception:
        return False
    return hasattr(traced, "graph") and traced is not module


# ---------------------------------------------------------------------------
# The thirteen methods
# ---------------------------------------------------------------------------


class _Type:
    """A TorchScript type, to the extent the serialiser inspects one.

    Four members are reached: `kind()`, `getElementType()`, `name()` and
    `str()`. `str()` is load-bearing in exactly one place -- `add_getattr`
    checks the module type's string starts with `__torch__.` -- which is why
    the text is carried rather than derived from the kind.
    """

    __slots__ = ("_kind", "_element", "_name", "_text")

    def __init__(self, kind: str, element: "_Type | None" = None,
                 name: str | None = None, text: str | None = None):
        self._kind = kind
        self._element = element
        self._name = name
        self._text = text or kind

    def kind(self) -> str:
        return self._kind

    def getElementType(self) -> "_Type":
        if self._element is None:
            raise JitFacadeRefused(f"{self._kind} has no element type")
        return self._element

    def name(self) -> str | None:
        return self._name

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"<Type {self._text}>"


TENSOR_TYPE = _Type("TensorType")
INT_TYPE = _Type("IntType")
FLOAT_TYPE = _Type("FloatType")
BOOL_TYPE = _Type("BoolType")
NONE_TYPE = _Type("NoneType")
INT_LIST_TYPE = _Type("ListType", element=INT_TYPE, text="int[]")
TENSOR_LIST_TYPE = _Type("ListType", element=TENSOR_TYPE, text="Tensor[]")
TUPLE_TYPE = _Type("TupleType")
SELF_TYPE = _Type("ClassType", name="NnapiGraph",
                  text="__torch__.torchnative.export.nnapi.NnapiGraph")


class _Val:
    """A jit `Value`.

    Hashed and compared **by identity**, which is what upstream relies on: the
    serialiser keys `jitval_operand_map`, `constants` and `tensor_sequences`
    off these objects directly. Two structurally identical constants must
    therefore be two distinct `_Val`s, and `to_jit_module` never reuses one.
    """

    __slots__ = ("_type", "_ivalue", "debug", "__weakref__")

    def __init__(self, type_: _Type, ivalue: Any = None, debug: str = ""):
        self._type = type_
        self._ivalue = ivalue
        self.debug = debug

    def type(self) -> _Type:
        return self._type

    def toIValue(self) -> Any:
        return self._ivalue

    def __repr__(self) -> str:
        return f"%{self.debug}: {self._type}"


class _Node:
    __slots__ = ("_kind", "_inputs", "_outputs", "_attrs")

    def __init__(self, kind: str, inputs: list, outputs: list,
                 attrs: dict | None = None):
        self._kind = kind
        self._inputs = list(inputs)
        self._outputs = list(outputs)
        self._attrs = dict(attrs or {})

    def kind(self) -> str:
        return self._kind

    def inputsSize(self) -> int:
        return len(self._inputs)

    def outputsSize(self) -> int:
        return len(self._outputs)

    def inputsAt(self, index: int) -> _Val:
        return self._inputs[index]

    def outputsAt(self, index: int) -> _Val:
        return self._outputs[index]

    def inputs(self):
        return list(self._inputs)

    def outputs(self):
        return list(self._outputs)

    def s(self, name: str) -> str:
        return self._attrs[name]

    def __repr__(self) -> str:
        return f"<{self._kind} {self._inputs!r} -> {self._outputs!r}>"


class _Graph:
    __slots__ = ("_inputs", "_nodes", "_return")

    def __init__(self, inputs: list, nodes: list, return_node: _Node):
        self._inputs = list(inputs)
        self._nodes = list(nodes)
        self._return = return_node

    def inputs(self):
        return iter(self._inputs)

    def nodes(self):
        return iter(self._nodes)

    def return_node(self) -> _Node:
        return self._return


class _Module:
    """Stands where a `ScriptModule` would.

    `serialize_model` registers it as the constant behind `graph.inputs()[0]`
    and it is only ever reached again through `prim::GetAttr`, which this
    façade does not emit -- our weights arrive as `prim::Constant` instead,
    since capture already holds the tensors and there is no attribute
    namespace to walk.
    """

    __slots__ = ("graph",)

    def __init__(self, graph: _Graph):
        self.graph = graph


# ---------------------------------------------------------------------------
# Calling convention
# ---------------------------------------------------------------------------


class _Arg:
    """Where one TorchScript input position gets its value."""

    __slots__ = ("source", "index", "type", "value")

    def __init__(self, source: str, index: int = -1, type_: _Type = NONE_TYPE,
                 value: Any = None):
        self.source = source          # "arg" | "const" | "arglist"
        self.index = index
        self.type = type_
        self.value = value


def _arg(index: int) -> _Arg:
    return _Arg("arg", index)


def _arglist(index: int) -> _Arg:
    """A recorded argument that is a *list of tensors*, e.g. `cat`'s first."""
    return _Arg("arglist", index)


def _opt(index: int, type_: _Type, value: Any) -> _Arg:
    """Recorded argument `index` if capture kept it, else this constant.

    Capture drops defaulted arguments, and upstream's adders assert an exact
    `inputsSize()`. Every use of this is a place where the two callings differ
    only by a default, and the default is the one in `native_functions.yaml`.
    """
    return _Arg("opt", index, type_, value)


def _const(type_: _Type, value: Any) -> _Arg:
    return _Arg("const", -1, type_, value)


#: `captured op -> (TorchScript node kind, input plan)`.
#:
#: Absence is a refusal, not a gap to be filled by guessing. Two ops NNAPI
#: nominally has are deliberately *not* here:
#:
#: * `aten.max_pool2d_with_indices.default` -- capture records the two-output
#:   ATen op, `aten::max_pool2d` has one output, and dropping the indices is a
#:   graph rewrite rather than a calling convention. It belongs in a pass.
#: * `aten.size.int` -- `add_size` emits nothing and only participates in
#:   flexible-shape bookkeeping, which this façade does not produce.
_SIGNATURES: dict[str, tuple[str, list[_Arg]]] = {
    # -- pointwise unary
    "aten.relu.default": ("aten::relu", [_arg(0)]),
    "aten.sigmoid.default": ("aten::sigmoid", [_arg(0)]),
    "aten.detach.default": ("aten::detach", [_arg(0)]),
    # -- pointwise binary. `alpha` is position 2 and capture drops it at 1.
    "aten.add.Tensor": ("aten::add", [_arg(0), _arg(1), _opt(2, INT_TYPE, 1)]),
    "aten.sub.Tensor": ("aten::sub", [_arg(0), _arg(1), _opt(2, INT_TYPE, 1)]),
    "aten.mul.Tensor": ("aten::mul", [_arg(0), _arg(1)]),
    "aten.div.Tensor": ("aten::div", [_arg(0), _arg(1)]),
    # -- clamping. NNAPI accepts only (-1, 1) and (0, 6) and says so itself.
    "aten.hardtanh.default": (
        "aten::hardtanh",
        [_arg(0), _opt(1, FLOAT_TYPE, -1.0), _opt(2, FLOAT_TYPE, 1.0)],
    ),
    # -- shape
    "aten.unsqueeze.default": ("aten::unsqueeze", [_arg(0), _arg(1)]),
    "aten.view.default": ("aten::reshape", [_arg(0), _arg(1)]),
    "aten.reshape.default": ("aten::reshape", [_arg(0), _arg(1)]),
    "aten.flatten.using_ints": (
        "aten::flatten",
        [_arg(0), _opt(1, INT_TYPE, 0), _opt(2, INT_TYPE, -1)],
    ),
    "aten.slice.Tensor": (
        "aten::slice",
        [
            _arg(0),
            _opt(1, INT_TYPE, 0),
            _opt(2, INT_TYPE, None),
            _opt(3, INT_TYPE, None),
            _opt(4, INT_TYPE, 1),
        ],
    ),
    "aten.cat.default": ("aten::cat", [_arglist(0), _opt(1, INT_TYPE, 0)]),
    # -- reductions and normalisations
    "aten.mean.dim": (
        "aten::mean",
        [_arg(0), _arg(1), _opt(2, BOOL_TYPE, False), _opt(3, NONE_TYPE, None)],
    ),
    "aten._softmax.default": (
        "aten::softmax", [_arg(0), _arg(1), _const(NONE_TYPE, None)]
    ),
    "aten._log_softmax.default": (
        "aten::log_softmax", [_arg(0), _arg(1), _const(NONE_TYPE, None)]
    ),
    # -- linear algebra
    "aten.linear.default": (
        "aten::linear", [_arg(0), _arg(1), _opt(2, NONE_TYPE, None)]
    ),
    "aten.addmm.default": (
        "aten::addmm",
        [_arg(0), _arg(1), _arg(2), _opt(3, INT_TYPE, 1), _opt(4, INT_TYPE, 1)],
    ),
    # -- convolution. Capture records the nine-argument `aten.convolution`;
    #    `aten::_convolution` takes thirteen, the last four being cuDNN
    #    switches that reach nothing in `add_conv_underscore` (it unpacks them
    #    into `_`). They are constants here for that reason.
    "aten.convolution.default": (
        "aten::_convolution",
        [
            _arg(0), _arg(1), _arg(2), _arg(3), _arg(4), _arg(5),
            _arg(6), _arg(7), _arg(8),
            _const(BOOL_TYPE, False),   # benchmark
            _const(BOOL_TYPE, False),   # deterministic
            _const(BOOL_TYPE, True),    # cudnn_enabled
            _const(BOOL_TYPE, True),    # allow_tf32
        ],
    ),
    # -- pooling
    "aten.avg_pool2d.default": (
        "aten::avg_pool2d",
        [
            _arg(0), _arg(1),
            _opt(2, INT_LIST_TYPE, []),
            _opt(3, INT_LIST_TYPE, [0, 0]),
            _opt(4, BOOL_TYPE, False),
            _opt(5, BOOL_TYPE, True),
            _opt(6, NONE_TYPE, None),
        ],
    ),
    "aten.adaptive_avg_pool2d.default": (
        "aten::adaptive_avg_pool2d", [_arg(0), _arg(1)]
    ),
    "aten._adaptive_avg_pool2d.default": (
        "aten::adaptive_avg_pool2d", [_arg(0), _arg(1)]
    ),
    "aten.upsample_nearest2d.default": (
        "aten::upsample_nearest2d",
        [_arg(0), _arg(1), _opt(2, NONE_TYPE, None)],
    ),
    # -- quantisation boundary
    "aten.prelu.default": ("aten::prelu", [_arg(0), _arg(1)]),
}


def _schema_names(op: str) -> list[str]:
    """The positional parameter names of `op`, read from the shim's schemas.

    Capture records post-dispatch, and the dispatcher is free to pass an
    argument by keyword: `aten.relu.default` arrives with zero positional args
    and `self` in `kwargs`, while `aten.convolution.default` arrives with nine
    positional and none. `_SIGNATURES` indexes by *schema position*, so the two
    callings have to be flattened onto one before the plan can be applied, and
    the only non-guessing way to do that is to ask for the schema.

    `torch._C._get_schema` is the same registry `verify_schemas.py` checks
    against upstream, so a name mismatch here is a schema defect and not a
    silent mis-slotting.
    """
    import torch

    parts = op.split(".")
    if len(parts) != 3 or parts[0] != "aten":
        raise JitFacadeRefused(f"torchnative nnapi: cannot parse op name {op!r}")
    _, name, overload = parts
    try:
        schema = torch._C._get_schema(
            "aten::" + name, "" if overload == "default" else overload
        )
    except Exception as error:
        raise JitFacadeRefused(
            f"torchnative nnapi: no schema for {op}, so its recorded keyword "
            f"arguments cannot be placed positionally: {error}"
        ) from None
    return [argument.name for argument in schema.arguments]


def _positional(op: str, args, kwargs) -> list:
    """Flatten a recorded call onto its schema's positional order.

    Trailing defaults stay *absent* rather than being filled in here: which
    default an omitted argument has is the schema's business, and `_opt` in
    `_SIGNATURES` already carries the one TorchScript needs at that position.
    Filling twice, in two places, from two sources is how they drift apart.
    """
    if not kwargs:
        return list(args)
    names = _schema_names(op)
    unknown = [k for k in kwargs if k not in names]
    if unknown:
        raise JitFacadeRefused(
            f"torchnative nnapi: {op} was recorded with keyword argument(s) "
            f"{unknown}, which its schema {names} does not name"
        )
    slots: dict[int, object] = {i: value for i, value in enumerate(args)}
    for key, value in kwargs.items():
        index = names.index(key)
        if index in slots:
            raise JitFacadeRefused(
                f"torchnative nnapi: {op} passed {key!r} both positionally and "
                f"by keyword"
            )
        slots[index] = value
    if not slots:
        return []
    highest = max(slots)
    missing = [i for i in range(highest + 1) if i not in slots]
    if missing:
        raise JitFacadeRefused(
            f"torchnative nnapi: {op} left schema position(s) {missing} unset "
            f"while setting {highest}; a defaulted argument in the middle "
            f"cannot be reconstructed from the record"
        )
    return [slots[i] for i in range(highest + 1)]


def supported_ops() -> frozenset[str]:
    """Captured op names this façade knows how to present. Read, not claimed.

    Strictly smaller than `target.nnapi_ops()`, and the difference is the point
    of §12.2's warning made concrete: `nnapi_ops()` counts base names and is an
    upper bound, this counts overloads that have an actual calling convention.
    """
    return frozenset(_SIGNATURES)


# ---------------------------------------------------------------------------
# Trace -> jit façade
# ---------------------------------------------------------------------------


def _dtype_of(text: str):
    import torch

    return getattr(torch, text.replace("torch.", ""))


def _literal_type(value: Any) -> _Type:
    import torch

    if value is None:
        return NONE_TYPE
    if isinstance(value, bool):
        return BOOL_TYPE
    if isinstance(value, int):
        return INT_TYPE
    if isinstance(value, float):
        return FLOAT_TYPE
    if isinstance(value, torch.Tensor):
        return TENSOR_TYPE
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, bool) or isinstance(v, int) for v in value):
            return INT_LIST_TYPE
        raise JitFacadeRefused(
            f"torchnative nnapi: no TorchScript type for the list {value!r}; "
            f"the serialiser inspects list element kinds and a wrong one is "
            f"not detectable downstream"
        )
    raise JitFacadeRefused(
        f"torchnative nnapi: no TorchScript type for {type(value).__name__}"
    )


class _Facade:
    def __init__(self, trace, constant_values):
        self.trace = trace
        self.constant_values = list(constant_values)
        self.inputs: list[_Val] = []
        self.nodes: list[_Node] = []
        #: `(kind, index, output) -> _Val` for the trace's own references.
        self.env: dict[tuple, _Val] = {}
        #: node position -> the shapes capture recorded for its outputs.
        self.recorded_shapes: dict[_Val, tuple] = {}

    # -- constants ------------------------------------------------------

    def literal(self, value: Any, type_: _Type | None = None) -> _Val:
        """A fresh `prim::Constant`. Never shared -- see `_Val`."""
        if type_ is None or (type_ is NONE_TYPE and value is not None):
            type_ = _literal_type(value)
        out = _Val(type_, value, debug=f"c{len(self.nodes)}")
        self.nodes.append(_Node("prim::Constant", [], [out]))
        return out

    def tensor_list(self, values: list) -> _Val:
        out = _Val(TENSOR_LIST_TYPE, None, debug=f"l{len(self.nodes)}")
        self.nodes.append(_Node("prim::ListConstruct", list(values), [out]))
        return out

    # -- references -----------------------------------------------------

    def resolve(self, ref) -> _Val:
        import torch

        if isinstance(ref, torch._C.CaptureValue):
            key = (ref.kind, ref.index, ref.output)
            if key in self.env:
                return self.env[key]
            if ref.kind == "const":
                value = self.constant_values[ref.index]
                held = self.literal(value, TENSOR_TYPE)
                self.env[key] = held
                return held
            raise JitFacadeRefused(
                f"torchnative nnapi: reference {key} is used before it is "
                f"produced; the trace is not in topological order"
            )
        raise JitFacadeRefused(f"torchnative nnapi: not a reference: {ref!r}")

    def materialise(self, arg, plan: _Arg) -> _Val:
        import torch

        if plan.source == "arglist":
            if not isinstance(arg, (list, tuple)):
                raise JitFacadeRefused(
                    f"torchnative nnapi: expected a list of tensors, got "
                    f"{type(arg).__name__}"
                )
            return self.tensor_list([self.resolve(a) for a in arg])
        if isinstance(arg, torch._C.CaptureValue):
            return self.resolve(arg)
        return self.literal(arg)


def to_jit_module(trace, *, constant_values=None):
    """Present `trace` as something `_NnapiSerializer.serialize_model` accepts.

    Returns `(module, input_tensors, facade)`. `input_tensors` are built from
    the trace's own guards -- shape, dtype and device are exactly what capture
    pinned, so the operand table the serialiser derives from them describes the
    graph that was recorded and not a shape we chose here.
    """
    import torch

    if constant_values is None:
        constant_values = trace.constant_values

    facade = _Facade(trace, constant_values)

    self_val = _Val(SELF_TYPE, None, debug="self")
    graph_inputs = [self_val]
    input_tensors = []
    for index, guard in enumerate(trace.guards):
        val = _Val(TENSOR_TYPE, None, debug=f"in{index}")
        graph_inputs.append(val)
        facade.env[("input", index, 0)] = val
        input_tensors.append(
            torch.zeros(
                tuple(guard["shape"]),
                dtype=_dtype_of(guard["dtype"]),
                device=guard["device"],
            )
        )

    for position, node in enumerate(trace.nodes):
        op = node["op"]
        if op not in _SIGNATURES:
            raise JitFacadeRefused(
                f"torchnative nnapi: no calling convention for {op}. "
                f"torch/backends/_nnapi/serializer.py keys ADDER_MAP by "
                f"TorchScript node kind and asserts an exact input count, so "
                f"an overload has to be mapped position by position. Absent "
                f"means unmapped, not unsupported -- add an entry to "
                f"_SIGNATURES only if the mapping is exact."
            )
        kind, plan = _SIGNATURES[op]
        args = _positional(op, node["args"], node["kwargs"])

        inputs: list[_Val] = []
        for item in plan:
            if item.source == "const":
                inputs.append(facade.literal(item.value, item.type))
            elif item.source == "opt":
                if item.index < len(args):
                    inputs.append(facade.materialise(args[item.index], _arg(item.index)))
                else:
                    inputs.append(facade.literal(item.value, item.type))
            else:
                if item.index >= len(args):
                    raise JitFacadeRefused(
                        f"torchnative nnapi: {op} needs argument "
                        f"{item.index} and the trace recorded {len(args)}"
                    )
                inputs.append(facade.materialise(args[item.index], item))

        outputs = []
        for slot, meta in enumerate(node["outputs"]):
            val = _Val(TENSOR_TYPE, None, debug=f"n{position}_{slot}")
            facade.env[("node", position, slot)] = val
            facade.recorded_shapes[val] = tuple(meta["shape"])
            outputs.append(val)
        facade.nodes.append(_Node(kind, inputs, outputs))

    returned = [facade.resolve(ref) for ref in trace.outputs]
    if len(returned) == 1:
        retn_input = returned[0]
    else:
        retn_input = _Val(TUPLE_TYPE, None, debug="ret")
        facade.nodes.append(_Node("prim::TupleConstruct", returned, [retn_input]))
    return_node = _Node("prim::Return", [retn_input], [])

    module = _Module(_Graph(graph_inputs, facade.nodes, return_node))
    return module, input_tensors, facade


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------


def fold_constants(trace):
    """Evaluate every node whose inputs are all constants, into a constant.

    This is not an optimisation, it is a prerequisite, and the smallest real
    graph shows why. `nn.Linear` captures as

        %0 = aten.t.default(%weight)
        %1 = aten.addmm.default(%bias, %x, %0)

    and `aten::t` is not in `ADDER_MAP` -- NNAPI has no transpose. But NNAPI
    does not *need* one: `add_addmm` requires `mat2` to be a constant weight
    and transposes it itself (`weight_tensor.t().contiguous()`), because
    FULLY_CONNECTED wants `[out, in]`. So the graph as recorded is
    unserialisable while the graph it denotes is entirely serialisable, and the
    difference is one node over a value that is known before the model runs.

    Folding is done by *executing* the node through `torch._C._aten_dispatch`,
    the same door capture recorded at and the same one `DecomposedTrace.replay`
    goes back through, for the reason docs/CAPTURE.md §3 gives: what comes out
    is then upstream's own answer rather than a second implementation of the
    op that happens to agree.

    A node is foldable only if every tensor argument is a `const` reference.
    Anything reachable from an input is left alone -- there is no partial
    evaluation here and no branching on a value.
    """
    import torch

    constants = list(trace.constants)
    constant_values = list(trace.constant_values)
    nodes: list[dict] = []
    outputs = list(trace.outputs)
    #: old ("node", position, slot) -> new reference, for folded and kept alike
    mapping: dict[tuple, Any] = {}
    folded: list[str] = []

    def is_const(ref) -> bool:
        # `ref` has already been through `remap`, so it is in the *new* trace's
        # numbering. Resolving it again would read the mapping with a key that
        # means something else there -- which it did, and the symptom was
        # `_softmax` folding away because its input's new node index collided
        # with a folded node's old one.
        if isinstance(ref, torch._C.CaptureValue):
            return ref.kind == "const"
        if isinstance(ref, (list, tuple)):
            return all(is_const(item) for item in ref)
        return True

    def remap(ref):
        return mapping.get((ref.kind, ref.index, ref.output), ref)

    def value_of(ref):
        if isinstance(ref, torch._C.CaptureValue):
            if ref.kind != "const":
                raise JitFacadeRefused(
                    f"torchnative nnapi: asked for the compile-time value of "
                    f"a {ref.kind} reference"
                )
            return constant_values[ref.index]
        if isinstance(ref, list):
            return [value_of(item) for item in ref]
        if isinstance(ref, tuple):
            return tuple(value_of(item) for item in ref)
        return ref

    for position, node in enumerate(trace.nodes):
        args = [_walk_ref(a, remap) for a in node["args"]]
        kwargs = {k: _walk_ref(v, remap) for k, v in node["kwargs"].items()}

        all_const = all(is_const(a) for a in args) and all(
            is_const(v) for v in kwargs.values()
        )
        has_tensor = any(
            isinstance(a, torch._C.CaptureValue) for a in list(args) + list(kwargs.values())
        )
        if all_const and has_tensor and not node["sequence"]:
            produced = torch._C._aten_dispatch(
                node["op"],
                *[value_of(a) for a in args],
                **{k: value_of(v) for k, v in kwargs.items()},
            )
            if isinstance(produced, torch.Tensor):
                index = len(constants)
                constants.append(
                    {
                        "index": index,
                        "shape": list(produced.shape),
                        "dtype": str(produced.dtype),
                        "device": str(produced.device),
                    }
                )
                constant_values.append(produced)
                mapping[("node", position, 0)] = _value("const", index)
                folded.append(node["op"])
                continue

        index = len(nodes)
        nodes.append(
            {
                "op": node["op"],
                "args": args,
                "kwargs": kwargs,
                "outputs": list(node["outputs"]),
                "sequence": node["sequence"],
            }
        )
        for slot in range(len(node["outputs"])):
            mapping[("node", position, slot)] = _value("node", index, slot)

    folded_trace = DecomposedTrace(
        list(trace.guards), constants, constant_values, nodes,
        [_walk_ref(ref, remap) for ref in outputs],
    )
    return folded_trace, folded


def _walk_ref(arg, fn):
    import torch

    if isinstance(arg, torch._C.CaptureValue):
        return fn(arg)
    if isinstance(arg, list):
        return [_walk_ref(a, fn) for a in arg]
    if isinstance(arg, tuple):
        return tuple(_walk_ref(a, fn) for a in arg)
    return arg


def _value(kind, index, output=0):
    import torch

    return torch._C._capture_value(kind, index, output)


# ---------------------------------------------------------------------------
# Driving upstream's serialiser
# ---------------------------------------------------------------------------


class NnapiModel:
    """What upstream's serialiser produced, plus what it was produced from."""

    __slots__ = ("blob", "weights", "inp_dim_orders", "out_dim_orders",
                 "shape_lines", "retval_count", "serializer", "facade")

    def __init__(self, blob, weights, inp_dim_orders, out_dim_orders,
                 shape_lines, retval_count, serializer, facade):
        self.blob = blob
        self.weights = weights
        self.inp_dim_orders = inp_dim_orders
        self.out_dim_orders = out_dim_orders
        self.shape_lines = shape_lines
        self.retval_count = retval_count
        self.serializer = serializer
        self.facade = facade

    def as_bytes(self) -> bytes:
        return self.blob.tobytes()

    def __repr__(self) -> str:
        return (
            f"<NnapiModel {len(self.blob) * 4} bytes, "
            f"{len(self.serializer.operands)} operands, "
            f"{len(self.serializer.operations)} operations>"
        )


def serialize(trace, *, constant_values=None, fold=True) -> NnapiModel:
    """Run upstream's serialiser over `trace` and return its output.

    Not a reimplementation: every operand, immediate and byte of layout comes
    from `torch/backends/_nnapi/serializer.py` unchanged. The contribution is
    the argument -- see the module docstring.
    """
    from torch.backends._nnapi.serializer import _NnapiSerializer

    if fold and constant_values is None:
        trace, _ = fold_constants(trace)
    module, inputs, facade = to_jit_module(trace, constant_values=constant_values)
    serializer = _NnapiSerializer(config=None)
    (blob, weights, inp_orders, out_orders, shape_lines, retval_count) = (
        serializer.serialize_model(module, inputs)
    )
    return NnapiModel(blob, weights, inp_orders, out_orders, shape_lines,
                      retval_count, serializer, facade)


# ---------------------------------------------------------------------------
# Structural validation of the blob
# ---------------------------------------------------------------------------


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, count: int) -> bytes:
        end = self.pos + count
        if end > len(self.data):
            raise ValueError(
                f"nnapi blob truncated: wanted {count} bytes at offset "
                f"{self.pos}, only {len(self.data) - self.pos} remain"
            )
        chunk = self.data[self.pos:end]
        self.pos = end
        return chunk

    def ints(self, count: int) -> list[int]:
        return list(array.array("i", self.take(count * 4)))


def parse_model(model) -> dict:
    """Decode an NNAPI blob back through the layout `serialize_model` wrote.

    A real decoder, not a length check. It reads the header, then the operand,
    value and operation tables, then the per-operand dimension lists, the value
    payloads, the flat operation-argument array and the model input/output
    lists -- in the exact order and packing upstream emits them, and it refuses
    on the first thing that does not line up. Cross-checks that go beyond
    "parses":

    * every table's length matches the count the header declared;
    * the bytes are consumed exactly, with nothing left over;
    * every operand index named by a value, an operation argument, or the
      input/output lists is in range;
    * every operation's argument count matches its declared arity, summed
      across the flat argument array.

    That is what "structurally validated" means everywhere this project uses
    the phrase for NNAPI, and it is not "executed": no NNAPI runtime exists on
    a Mac. docs/NPU.md keeps the two claims apart.
    """
    data = model.as_bytes() if isinstance(model, NnapiModel) else bytes(model)
    reader = _Reader(data)

    version, n_operands, n_values, n_operations, n_inputs, n_outputs = struct.unpack(
        "iiiiii", reader.take(24)
    )
    if version != 1:
        raise ValueError(f"nnapi blob: unknown version {version}")
    for name, count in (("operands", n_operands), ("values", n_values),
                        ("operations", n_operations), ("inputs", n_inputs),
                        ("outputs", n_outputs)):
        if count < 0:
            raise ValueError(f"nnapi blob: negative {name} count {count}")

    operands = []
    for _ in range(n_operands):
        op_type, n_dims, scale, zero_point = struct.unpack("iifi", reader.take(16))
        if n_dims < 0:
            raise ValueError(f"nnapi blob: operand with {n_dims} dimensions")
        operands.append(
            {"op_type": op_type, "n_dims": n_dims, "scale": scale,
             "zero_point": zero_point}
        )

    values = []
    for _ in range(n_values):
        op_index, source_type, source_length = struct.unpack("iii", reader.take(12))
        if not 0 <= op_index < n_operands:
            raise ValueError(
                f"nnapi blob: value names operand {op_index}, and the model "
                f"declares {n_operands}"
            )
        if source_length < 0:
            raise ValueError(f"nnapi blob: value with length {source_length}")
        values.append(
            {"operand": op_index, "source_type": source_type,
             "length": source_length}
        )

    operations = []
    for _ in range(n_operations):
        opcode, n_in, n_out = struct.unpack("iii", reader.take(12))
        if n_in < 0 or n_out < 0:
            raise ValueError(f"nnapi blob: operation with arity ({n_in}, {n_out})")
        operations.append({"opcode": opcode, "n_inputs": n_in, "n_outputs": n_out})

    for operand in operands:
        operand["shape"] = tuple(reader.ints(operand["n_dims"]))

    for value in values:
        padded = ((value["length"] - 1) | 0x3) + 1 if value["length"] else 0
        value["data"] = reader.take(padded)[: value["length"]]

    total_args = sum(op["n_inputs"] + op["n_outputs"] for op in operations)
    flat_args = reader.ints(total_args)
    cursor = 0
    for op in operations:
        op["inputs"] = flat_args[cursor:cursor + op["n_inputs"]]
        cursor += op["n_inputs"]
        op["outputs"] = flat_args[cursor:cursor + op["n_outputs"]]
        cursor += op["n_outputs"]
        for ref in op["inputs"] + op["outputs"]:
            if not 0 <= ref < n_operands:
                raise ValueError(
                    f"nnapi blob: operation {op['opcode']} names operand "
                    f"{ref}, and the model declares {n_operands}"
                )
    if cursor != total_args:  # pragma: no cover -- arithmetic guard
        raise ValueError("nnapi blob: operation argument array did not consume")

    model_inputs = reader.ints(n_inputs)
    model_outputs = reader.ints(n_outputs)
    for name, refs in (("input", model_inputs), ("output", model_outputs)):
        for ref in refs:
            if not 0 <= ref < n_operands:
                raise ValueError(
                    f"nnapi blob: model {name} names operand {ref}, and the "
                    f"model declares {n_operands}"
                )

    if reader.pos != len(data):
        raise ValueError(
            f"nnapi blob: {len(data) - reader.pos} trailing byte(s); the "
            f"layout was consumed but the blob is longer than it describes"
        )

    return {
        "version": version,
        "operands": operands,
        "values": values,
        "operations": operations,
        "inputs": model_inputs,
        "outputs": model_outputs,
    }


def verify_shapes(model: NnapiModel) -> dict:
    """Compare the serialiser's operand shapes with what capture recorded.

    Two independent derivations of the same number: upstream propagates shapes
    over NNAPI operands from the input operand table, and capture observed them
    on a real CPU execution. A `_SIGNATURES` entry that puts a recorded
    argument in the wrong TorchScript position moves the first and not the
    second, which is the only cheap way to catch a mis-wiring that would
    otherwise serialise cleanly and compute something else.

    NCHW/NHWC is accounted for: an operand the serialiser marked CHANNELS_LAST
    still carries the PyTorch shape in `Operand.shape` (upstream's own comment
    says so), so the two are directly comparable.
    """
    checked, mismatches = 0, []
    for val, recorded in model.facade.recorded_shapes.items():
        operand_id = model.serializer.jitval_operand_map.get(val)
        if operand_id is None:
            continue  # an op the serialiser folded away, e.g. aten::detach
        got = tuple(model.serializer.operands[operand_id].shape)
        checked += 1
        if got != recorded:
            mismatches.append({"value": repr(val), "serialiser": got,
                               "capture": recorded})
    return {"checked": checked, "mismatches": mismatches}
