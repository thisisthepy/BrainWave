"""Lower a captured trace from ATen to Core ATen.

docs/graph/DECOMP.md is the measurement this module exists to make; docs/graph/CAPTURE.md
§5 is the gap it closes. The short version of both:

    `_capture_end` records **ATen** -- whatever the dispatcher was asked for.
    ExecuTorch's Edge dialect is defined over **Core ATen**, a named subset.
    The smallest example in docs/graph/CAPTURE.md records `aten.t.default`, which is
    outside that subset. So a pass has to stand between capture and any
    delegate, and this is it.

Two things are deliberately *not* written here, because writing them is the
failure this module is most likely to commit:

**The decomposition rules.** They are upstream's, in
`torch/_decomp/decompositions.py` and `torch/_refs`, and the vendored tree
carries them. This module looks them up and *runs* them; it does not restate
them. A hand-copied decomposition table would be a reimplementation of the
thing this project exists not to reimplement, and it would drift the first time
upstream fixed a rule.

**The list of Core ATen ops.** That list is a `tags: core` annotation on each
entry of `torchgen/packaged/ATen/native/native_functions.yaml`, which the
vendored tree also carries -- it is the same file upstream generates
`torch.Tag.core` from. `core_ops()` reads it. Transcribing 193 op names into
this file would be the same mistake one size down.

## How a node is lowered

The rules are Python functions over tensors, so applying one means *running*
it -- and the way to find out what ops a run issues is the recorder that is
already there. Lowering `aten.t.default(%c0)` is:

1. make a placeholder tensor for every tensor operand, shaped and typed by
   what the trace recorded for that operand;
2. `_capture_begin(placeholders)`, call upstream's decomposition, `_capture_end`;
3. splice the resulting sub-trace into the parent in place of the node.

Which means the capture layer's refusals (docs/graph/CAPTURE.md §4) guard the pass
for free. A decomposition that branches on a tensor *value* reaches
`aten._local_scalar_dense.default` and poisons its own sub-recording, and the
pass then refuses that op by name instead of emitting a graph that is only
correct for the placeholder values. A decomposition that branches on *shape* or
*dtype* is fine, because the guards pin both.

Repeat to a fixpoint: a decomposition may itself emit non-core ops.

## What it refuses

Anything left non-core after the fixpoint, **by name**. Passing it through
silently is the failure mode that costs the most later -- ExecuTorch would
reject the program with no indication of which op or which pass was
responsible.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable, Iterable


__all__ = [
    "DecomposedTrace",
    "DecompositionRefused",
    "core_ops",
    "decompose",
    "decomposition_table",
    "decomposition_table_source",
    "is_core",
    "non_core_ops",
]


class DecompositionRefused(NotImplementedError):
    """A node could not be lowered to Core ATen, and this says which.

    `NotImplementedError` rather than a bare exception for the same reason
    `_capture_end` raises one: the caller's sane response is to fall back to
    running the region eagerly, and that is a decision about a missing
    capability rather than about bad input.
    """


# ---------------------------------------------------------------------------
# What Core ATen is, read from the vendored tree
# ---------------------------------------------------------------------------

_NATIVE_FUNCTIONS = os.path.join(
    "torchgen", "packaged", "ATen", "native", "native_functions.yaml"
)


def _native_functions_path() -> str:
    """Where the vendored `native_functions.yaml` is.

    Located relative to `torch.__file__` rather than to this file, because the
    two are siblings in an installed wheel *and* in the source tree, and it is
    the vendored torch whose tags we want -- if two trees are on the path, the
    answer has to come from the one whose ops we are lowering.
    """
    import torch

    root = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    return os.path.join(root, _NATIVE_FUNCTIONS)


def _scan_core_tags(text: str) -> frozenset[str]:
    """Every `aten.<op>.<overload>` whose entry carries `tags: core`.

    A line scanner rather than a YAML parse, for one reason that is not
    aesthetic: the vendored tree's own YAML dependency is not a declared
    dependency of this distribution (see pyproject.toml -- it lists upstream
    torch's pure-Python requirements, and `pyyaml` is not among them), so
    importing `yaml` here would make a runtime import of this module fail on a
    correctly-installed wheel.

    The format this relies on is narrow and checked: entries begin at column 0
    with `- func:`, and `tags:` is a flow scalar or a flow list on one line --
    measured over all 2584 entries of torch 2.13.0's file, with zero
    block-sequence `tags:` among them. `test_core_op_scan_agrees_with_yaml`
    diffs this scan against a real YAML parse so that a format change is a
    test failure rather than a quietly shorter list.
    """
    core: set[str] = set()
    signature: str | None = None
    for line in text.splitlines():
        if line.startswith("- func:"):
            signature = line[len("- func:") :].strip().split("(", 1)[0]
        elif signature is not None and line.startswith("  tags:"):
            tags = line[len("  tags:") :].split("#", 1)[0].strip()
            tags = tags.strip("[]")
            if "core" in {t.strip() for t in tags.split(",")}:
                if "." in signature:
                    name, overload = signature.split(".", 1)
                else:
                    name, overload = signature, "default"
                core.add(f"aten.{name}.{overload}")
        elif line.startswith("#"):
            # A comment at column 0 does not end an entry; YAML never sees it.
            # No `core` tag sits behind one today (the yaml diff below would
            # say so), but three `pointwise` tags do, and treating a comment as
            # a terminator lost them from the shim's `OpOverload.tags`.
            continue
        elif line and not line.startswith(" "):
            signature = None
    return frozenset(core)


@functools.lru_cache(maxsize=1)
def core_ops() -> frozenset[str]:
    """The Core ATen operator set, as `aten.<op>.<overload>` names.

    193 entries on torch 2.13.0. Upstream's own `torch.Tag.core` reports 189 of
    these on an installed build; the four extra
    (`adaptive_avg_pool1d.default`, `avg_pool1d.default`, `resize_.default`,
    `sym_is_contiguous.default`) are in the file and do not surface as
    `OpOverload.tags` there. Including them is the safe direction -- it can
    only make this pass accept a node, never make it refuse one, and each is an
    op upstream's own file calls core.

    Reading the tags out of the vendored file rather than off
    `torch.ops.aten.<op>.<ov>.tags` began as a necessity: `_get_operation_
    overload` in the `_C` shim answered `[]` for every op, so
    `torch.Tag.core in op.tags` was False for all 120 implemented ops and a
    classifier built on it would call nothing core and refuse whole programs
    (docs/graph/DECOMP.md §2). The shim reads the file's real tags now, and the two
    agree on every implemented op -- `test_core_ops_and_op_tags_agree` diffs
    them, so this is a measured agreement and not a second copy of one answer.

    The file stays the source here anyway, for the four ops above: an
    `OpOverload.tags` classifier is the narrower set even upstream.
    """
    with open(_native_functions_path(), encoding="utf-8") as handle:
        return _scan_core_tags(handle.read())


def is_core(op: str) -> bool:
    """Whether `aten.<op>.<overload>` is in the Core ATen operator set."""
    return op in core_ops()


def non_core_ops(ops: Iterable[str]) -> list[str]:
    """The subset of `ops` outside Core ATen, sorted and deduplicated."""
    core = core_ops()
    return sorted({op for op in ops if op not in core})


# ---------------------------------------------------------------------------
# The rules, read from the vendored tree
# ---------------------------------------------------------------------------

#: Why `core_aten_decompositions()` was not the table used, or None if it was.
#: Read `decomposition_table_source()` rather than this.
_TABLE_FALLBACK_REASON: str | None = None


@functools.lru_cache(maxsize=1)
def _load_table() -> dict[str, Callable[..., Any]]:
    global _TABLE_FALLBACK_REASON
    import torch._decomp as decomp

    # Ask for the real thing first, every time, rather than hardcoding the
    # fallback: the day the `_C` shim answers
    # `_dispatch_get_registrations_for_dispatch_key`, this picks the fuller
    # table up with no edit here. Catching only NotImplementedError so that a
    # genuine breakage upstream still surfaces as itself.
    try:
        table = decomp.core_aten_decompositions()
        _TABLE_FALLBACK_REASON = None
    except NotImplementedError as error:
        _TABLE_FALLBACK_REASON = str(error)
        table = decomp._core_aten_decompositions_post_autograd()
    return {str(key): value for key, value in table.items()}


def decomposition_table() -> dict[str, Callable[..., Any]]:
    """Upstream's Core ATen decomposition rules, keyed by `aten.<op>.<ov>`."""
    return _load_table()


def decomposition_table_source() -> tuple[str, str | None]:
    """Which upstream table is in use, and why it is not the full one.

    Returns `(name, reason)`. `reason` is None when
    `core_aten_decompositions()` itself answered.

    This is a getter rather than a comment because the fallback is a real
    reduction in coverage and a silent fallback would hide it. On this build
    the answer is `("_core_aten_decompositions_post_autograd", "<the shim
    function that is missing>")`: the full table is a `CustomDecompTable`,
    whose constructor enumerates every CompositeImplicitAutograd registration
    through `torch._C._dispatch_get_registrations_for_dispatch_key`, and this
    `_C` is a Python shim with no C++ dispatcher to enumerate. docs/graph/DECOMP.md §3
    lists which ops that costs.
    """
    _load_table()
    return (
        "core_aten_decompositions"
        if _TABLE_FALLBACK_REASON is None
        else "_core_aten_decompositions_post_autograd",
        _TABLE_FALLBACK_REASON,
    )


# ---------------------------------------------------------------------------
# The lowered trace
# ---------------------------------------------------------------------------


def _dtype_of(meta: dict) -> Any:
    import torch

    return getattr(torch, meta["dtype"].removeprefix("torch."))


def _placeholder(meta: dict) -> Any:
    """A tensor shaped like `meta`, to run a decomposition on.

    Zeros rather than random values or `empty`: a decomposition must not be a
    function of its input *values* -- one that is reaches
    `aten._local_scalar_dense.default` and poisons its own sub-recording, which
    is how this pass finds out. Given that, a deterministic filler makes a
    failure reproducible, and uninitialised memory would not.
    """
    import torch

    return torch.zeros(
        list(meta["shape"]), dtype=_dtype_of(meta), device=meta["device"]
    )


def _meta_matches(want: dict, got: dict) -> bool:
    return (
        list(want["shape"]) == list(got["shape"])
        and want["dtype"] == got["dtype"]
        and want["device"] == got["device"]
    )


class DecomposedTrace:
    """A trace in the Core ATen dialect, and the conditions it is valid under.

    Structurally the same record `torch._C.CaptureTrace` produces -- same
    `guards`, `constants`, `nodes`, `outputs`, same `CaptureValue` references,
    same `graph()` keys -- so anything that reads one reads the other. It is a
    separate type because `CaptureTrace` is what the *recorder* emits and
    cannot be constructed from Python, and because the two make different
    claims: this one's ops are all in `core_ops()`, and that is checked before
    the object exists.

    `replay` goes back through `_aten_dispatch`, the same door capture recorded
    at, for the reason docs/graph/CAPTURE.md §3 gives: agreement with eager is then
    evidence about the *rewrite*, not about two implementations happening to
    match.
    """

    __slots__ = ("_guards", "_constants", "_constant_values", "_nodes", "_outputs")

    def __init__(self, guards, constants, constant_values, nodes, outputs):
        self._guards = guards
        self._constants = constants
        self._constant_values = constant_values
        self._nodes = nodes
        self._outputs = outputs

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return (
            f"<DecomposedTrace {len(self._nodes)} nodes, {len(self._guards)} inputs, "
            f"{len(self._constants)} constants, {len(self._outputs)} outputs>"
        )

    @property
    def guards(self):
        return list(self._guards)

    @property
    def constants(self):
        return list(self._constants)

    @property
    def constant_values(self):
        return list(self._constant_values)

    @property
    def nodes(self):
        return list(self._nodes)

    @property
    def outputs(self):
        return list(self._outputs)

    @property
    def ops(self) -> list[str]:
        return [node["op"] for node in self._nodes]

    def graph(self) -> dict:
        return {
            "placeholders": self.guards,
            "constants": self.constants,
            "nodes": self.nodes,
            "outputs": self.outputs,
        }

    def replay(self, inputs):
        import torch

        dispatch = torch._C._aten_dispatch
        given = list(inputs)
        if len(given) != len(self._guards):
            raise RuntimeError(
                f"torchnative decompose: this trace was recorded with "
                f"{len(self._guards)} input(s), and replay was given {len(given)}"
            )
        for index, (value, guard) in enumerate(zip(given, self._guards)):
            self._check_guard(index, value, guard)

        env = {
            ("input", index, 0): value for index, value in enumerate(given)
        }
        for index, value in enumerate(self._constant_values):
            env[("const", index, 0)] = value

        for position, node in enumerate(self._nodes):
            args = [self._materialise(a, env) for a in node["args"]]
            kwargs = {k: self._materialise(v, env) for k, v in node["kwargs"].items()}
            produced = dispatch(node["op"], *args, **kwargs)
            slots = list(produced) if node["sequence"] else [produced]
            if len(slots) != len(node["outputs"]):
                raise RuntimeError(
                    f"torchnative decompose: replaying {node['op']} returned "
                    f"{len(slots)} results and the record had {len(node['outputs'])}"
                )
            for slot, produced_value in enumerate(slots):
                env[("node", position, slot)] = produced_value

        return tuple(self._materialise(ref, env) for ref in self._outputs)

    # -- internals ---------------------------------------------------------

    def _check_guard(self, index, value, guard):
        import torch

        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"torchnative decompose: input {index} of replay is a "
                f"{type(value).__name__}, and the trace recorded a tensor there"
            )
        if list(value.shape) != list(guard["shape"]):
            raise RuntimeError(
                f"torchnative decompose: input {index} has shape "
                f"{list(value.shape)} and this trace is only valid for shape "
                f"{list(guard['shape'])}"
            )
        if str(value.dtype) != guard["dtype"]:
            raise RuntimeError(
                f"torchnative decompose: input {index} has dtype {value.dtype} "
                f"and this trace is only valid for dtype {guard['dtype']}"
            )
        if str(value.device) != guard["device"]:
            raise RuntimeError(
                f"torchnative decompose: input {index} is on device "
                f"{value.device} and this trace is only valid for device "
                f"{guard['device']}"
            )

    @staticmethod
    def _materialise(arg, env):
        import torch

        if isinstance(arg, torch._C.CaptureValue):
            key = (arg.kind, arg.index, arg.output)
            if key not in env:
                raise RuntimeError(
                    "torchnative decompose: replay reached a value that had "
                    "not been produced yet"
                )
            return env[key]
        if isinstance(arg, list):
            return [DecomposedTrace._materialise(a, env) for a in arg]
        if isinstance(arg, tuple):
            return tuple(DecomposedTrace._materialise(a, env) for a in arg)
        return arg


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def _walk(arg, fn):
    """Rebuild `arg`, applying `fn` to every `CaptureValue` inside it."""
    import torch

    if isinstance(arg, torch._C.CaptureValue):
        return fn(arg)
    if isinstance(arg, list):
        return [_walk(a, fn) for a in arg]
    if isinstance(arg, tuple):
        return tuple(_walk(a, fn) for a in arg)
    return arg


def _value(kind, index, output=0):
    import torch

    return torch._C._capture_value(kind, index, output)


class _Builder:
    """The trace being written, in the same shape one is read in."""

    def __init__(self, guards, constants, constant_values):
        self.guards = list(guards)
        self.constants = list(constants)
        self.constant_values = list(constant_values)
        self.nodes: list[dict] = []

    def add_constant(self, meta, value):
        index = len(self.constants)
        self.constants.append(
            {
                "index": index,
                "shape": list(meta["shape"]),
                "dtype": meta["dtype"],
                "device": meta["device"],
            }
        )
        self.constant_values.append(value)
        return _value("const", index)

    def add_node(self, op, args, kwargs, outputs, sequence):
        index = len(self.nodes)
        self.nodes.append(
            {
                "op": op,
                "args": list(args),
                "kwargs": dict(kwargs),
                "outputs": list(outputs),
                "sequence": sequence,
            }
        )
        return index


def _resolve_meta(ref, builder, node_metas):
    """What the trace knows about the tensor a reference points at."""
    if ref.kind == "input":
        return builder.guards[ref.index]
    if ref.kind == "const":
        return builder.constants[ref.index]
    meta = node_metas[ref.index][ref.output]
    if meta is None:
        raise DecompositionRefused(
            f"torchnative decompose: a value in this trace is the non-tensor "
            f"result of node {ref.index}, and a decomposition cannot be given "
            f"one"
        )
    return meta


def _schema_of(op: str):
    """The `FunctionSchema` for an `aten.<name>.<overload>` node, or None.

    None when the shim has no text for it -- a placeholder schema answers
    `arguments` with an empty list, which is indistinguishable from a nullary
    op, so it has to be told apart here rather than silently used
    (docs/bindings/SCHEMA.md §6).
    """
    import torch

    parts = op.split(".")
    if len(parts) != 3:
        return None
    namespace, name, overload = parts
    schema = torch._C._get_schema(f"{namespace}::{name}", overload)
    if schema is None or getattr(schema, "is_placeholder", False):
        return None
    return schema


def _as_the_dispatcher_would_call_it(op, args, kwargs):
    """Recorded `(args, kwargs)`, re-spelled in the schema's positional order.

    **A decomposition is called positionally, and the recording is not.** The
    shim's `torch.<fn>` resolver binds a call against the aten schema and
    forwards it by *schema* name, so `torch.transpose(x, 0, 1)` is recorded as
    `aten.transpose.int(self=%0, dim0=0, dim1=1)` -- and upstream's rule for
    that op is `torch._refs.transpose(a, dim0, dim1)`, whose first parameter is
    `a`. Calling it with the recording's kwargs raises `TypeError: transpose()
    got an unexpected keyword argument 'self'`, which is what
    `aten.t.default` -- the op docs/graph/CAPTURE.md §5 named -- used to die of on
    its *second* round, after its own rule had already succeeded.

    Upstream never has this problem because the C++ dispatcher hands a kernel
    its arguments positionally; the parameter names of a decomposition are its
    own business. So this puts the call back into that shape: every argument
    the schema declares before its `*` is passed by position, and only
    genuinely kwarg-only ones stay named.

    A kwarg the schema does not declare, or an op with no schema text, is left
    exactly as recorded -- the pass then fails with the rule's own error rather
    than with a rewrite this function invented.
    """
    if not kwargs:
        return list(args), dict(kwargs)
    schema = _schema_of(op)
    if schema is None:
        return list(args), dict(kwargs)
    positional = [a.name for a in schema.arguments if not a.kwarg_only]
    if not set(kwargs) <= {a.name for a in schema.arguments}:
        return list(args), dict(kwargs)
    remaining = dict(kwargs)
    ordered = list(args)
    for name in positional[len(args):]:
        if name not in remaining:
            break  # a defaulted argument was not passed; the rest must stay named
        ordered.append(remaining.pop(name))
    return ordered, remaining


def _sub_capture(op, fn, args, kwargs, placeholders, sequence):
    """Run one upstream decomposition and record what it issued."""
    import torch

    torch._C._capture_begin(placeholders)
    try:
        result = fn(*args, **kwargs)
    except BaseException as error:  # noqa: BLE001 -- re-raised, named, below
        torch._C._capture_abandon()
        raise DecompositionRefused(
            f"torchnative decompose: cannot lower {op} -- running upstream's "
            f"decomposition for it raised {type(error).__name__}: {error}"
        ) from error
    reason = torch._C._capture_reason()
    if reason is not None:
        torch._C._capture_abandon()
        raise DecompositionRefused(
            f"torchnative decompose: cannot lower {op} -- upstream's "
            f"decomposition for it is not capturable: {reason}"
        )
    if sequence and not isinstance(result, (list, tuple)):
        torch._C._capture_abandon()
        raise DecompositionRefused(
            f"torchnative decompose: cannot lower {op} -- the record has it "
            f"returning a sequence and its decomposition returned a "
            f"{type(result).__name__}"
        )
    return torch._C._capture_end(result)


def _lower_node(node, builder, node_metas, table):
    """Replace one non-core node with its decomposition, spliced in place.

    Returns the list of values the node's result slots now refer to.
    """
    op = node["op"]
    fn = table.get(op)
    if fn is None:
        raise DecompositionRefused(
            f"torchnative decompose: {op} is not in Core ATen and upstream's "
            f"decomposition table ({decomposition_table_source()[0]}) has no "
            f"rule for it"
        )

    # One placeholder per distinct operand: two arguments that were the same
    # value in the recording must be the same object here, or the sub-trace
    # would record a graph in which they are independent.
    order: list[Any] = []
    made: dict[Any, Any] = {}

    def placeholder_for(ref):
        if ref not in made:
            made[ref] = _placeholder(_resolve_meta(ref, builder, node_metas))
            order.append(ref)
        return made[ref]

    args, kwargs = _as_the_dispatcher_would_call_it(op, node["args"], node["kwargs"])
    args = [_walk(a, placeholder_for) for a in args]
    kwargs = {k: _walk(v, placeholder_for) for k, v in kwargs.items()}
    sub = _sub_capture(
        op, fn, args, kwargs, [made[ref] for ref in order], node["sequence"]
    )

    # Splice. The sub-trace's own value space maps into the parent's: its
    # inputs are the parent references they were built from, its constants
    # become parent constants, and its nodes are appended.
    mapping: dict[Any, Any] = {}
    for slot, ref in enumerate(order):
        mapping[_value("input", slot)] = ref
    sub_constant_values = sub.constant_values
    for slot, meta in enumerate(sub.constants):
        mapping[_value("const", slot)] = builder.add_constant(
            meta, sub_constant_values[slot]
        )
    for position, sub_node in enumerate(sub.nodes):
        index = builder.add_node(
            sub_node["op"],
            [_walk(a, mapping.__getitem__) for a in sub_node["args"]],
            {k: _walk(v, mapping.__getitem__) for k, v in sub_node["kwargs"].items()},
            sub_node["outputs"],
            sub_node["sequence"],
        )
        node_metas.append(list(sub_node["outputs"]))
        for slot in range(len(sub_node["outputs"])):
            mapping[_value("node", position, slot)] = _value("node", index, slot)

    produced = [mapping[ref] for ref in sub.outputs]
    if len(produced) != len(node["outputs"]):
        raise DecompositionRefused(
            f"torchnative decompose: cannot lower {op} -- the record has "
            f"{len(node['outputs'])} result(s) and its decomposition produced "
            f"{len(produced)}"
        )
    # A decomposition is supposed to be the same function. If the result it
    # produced does not have the shape and dtype the recording saw, something
    # is different, and passing it on would put a wrong graph downstream where
    # nothing else checks.
    for slot, (ref, want) in enumerate(zip(produced, node["outputs"])):
        if want is None:
            raise DecompositionRefused(
                f"torchnative decompose: cannot lower {op} -- result {slot} of "
                f"the recording is not a tensor"
            )
        got = _resolve_meta(ref, builder, node_metas)
        if not _meta_matches(want, got):
            raise DecompositionRefused(
                f"torchnative decompose: cannot lower {op} -- its decomposition "
                f"produced result {slot} with shape {list(got['shape'])} "
                f"{got['dtype']} on {got['device']}, and the recording has "
                f"shape {list(want['shape'])} {want['dtype']} on "
                f"{want['device']}"
            )
    return produced


def _round(guards, constants, constant_values, nodes, outputs, table):
    """One pass over every node: keep core ones, lower the rest."""
    builder = _Builder(guards, constants, constant_values)
    node_metas: list[list] = []
    mapping: dict[Any, Any] = {}

    def remap(ref):
        return mapping.get(ref, ref)

    changed = False
    for position, node in enumerate(nodes):
        args = [_walk(a, remap) for a in node["args"]]
        kwargs = {k: _walk(v, remap) for k, v in node["kwargs"].items()}
        if is_core(node["op"]):
            index = builder.add_node(
                node["op"], args, kwargs, node["outputs"], node["sequence"]
            )
            node_metas.append(list(node["outputs"]))
            for slot in range(len(node["outputs"])):
                mapping[_value("node", position, slot)] = _value("node", index, slot)
            continue
        changed = True
        produced = _lower_node(
            {
                "op": node["op"],
                "args": args,
                "kwargs": kwargs,
                "outputs": node["outputs"],
                "sequence": node["sequence"],
            },
            builder,
            node_metas,
            table,
        )
        for slot, ref in enumerate(produced):
            mapping[_value("node", position, slot)] = ref

    return (
        builder.guards,
        builder.constants,
        builder.constant_values,
        builder.nodes,
        [remap(ref) for ref in outputs],
        changed,
    )


def decompose(trace, *, max_rounds: int = 8) -> DecomposedTrace:
    """Lower a captured trace to Core ATen, or refuse and say which op.

    `trace` is a `torch._C.CaptureTrace` or a `DecomposedTrace` -- anything
    with `guards`, `constants`, `constant_values`, `nodes` and `outputs`.

    Iterates because a decomposition may itself emit non-core ops: upstream's
    rule for `aten.stack.default`, for instance, emits `cat` and `view`, both
    core, but nothing guarantees that in general. `max_rounds` bounds it; a
    trace that has not converged by then is refused with the ops that are still
    outside Core ATen, rather than looped on.
    """
    table = decomposition_table()
    guards = trace.guards
    constants = trace.constants
    constant_values = trace.constant_values
    nodes = trace.nodes
    outputs = trace.outputs

    for _ in range(max_rounds):
        remaining = non_core_ops(node["op"] for node in nodes)
        if not remaining:
            return DecomposedTrace(guards, constants, constant_values, nodes, outputs)
        guards, constants, constant_values, nodes, outputs, changed = _round(
            guards, constants, constant_values, nodes, outputs, table
        )
        if not changed:  # pragma: no cover -- `remaining` non-empty implies work
            break

    remaining = non_core_ops(node["op"] for node in nodes)
    raise DecompositionRefused(
        f"torchnative decompose: gave up after {max_rounds} rounds with these "
        f"ops still outside Core ATen: {', '.join(remaining)}"
    )
