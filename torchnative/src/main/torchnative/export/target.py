"""Lower a captured trace toward an arbitrary *target* operator set.

`decompose.py` lowers to one destination -- Core ATen -- because that is what
ExecuTorch's Edge dialect is defined over. The device column of the README
lists NNAPI and CoreML unsupported for a different reason: each takes a graph
of its *own* small fixed operator set, and neither set is Core ATen.

Measured, not assumed (docs/graph/DECOMP.md §12). Of the 29 `aten::` ops NNAPI's
in-tree serializer can actually emit:

    14 are also Core ATen          add addmm avg_pool2d cat div hardtanh mean
                                   mul relu sigmoid slice sub unsqueeze
                                   upsample_nearest2d
    15 are NOT                     _convolution adaptive_avg_pool2d conv2d
                                   dequantize detach flatten linear log_softmax
                                   max_pool2d prelu quantize_per_tensor reshape
                                   size softmax to

The second row is the reason this module exists rather than being a wrapper
around `decompose()`. Those 15 are *composite* ops, and Core ATen lowering
takes them apart -- `linear` becomes `addmm`/`permute`, `softmax` becomes
`_softmax`. Running `decompose()` and then handing the result to NNAPI would
destroy fifteen of the twenty-nine ops NNAPI natively accepts. Lowering is
target-relative: there is no single "lowered" form, and a pass with a hardcoded
destination cannot serve a second device.

So the destination becomes a parameter. Everything else is reused from
`decompose.py` unchanged -- in particular `_lower_node`, which is already
target-agnostic: it looks a rule up in upstream's table and *runs* it, and
never consults `is_core`. As in `decompose.py`, no decomposition rule is
written here. They are upstream's.

## The target sets are read, not transcribed

`nnapi_ops()` parses `ADDER_MAP` out of
`torch/backends/_nnapi/serializer.py` in the vendored tree. That table is
authoritative in the strongest available sense: it is not documentation of what
NNAPI supports, it is the code that would do the serialising, so an op absent
from it cannot be lowered to NNAPI by this PyTorch no matter what the hardware
supports. Transcribing the 29 names into this file would be the same mistake
`decompose.py` refuses to make with the Core ATen list, one size down.

**There is no equivalent authority for CoreML in this tree**, and this module
does not invent one -- see `coreml_ops()`.
"""

from __future__ import annotations

import functools
import os
import re
from typing import Any, Iterable

from .decompose import (
    DecompositionRefused,
    DecomposedTrace,
    _Builder,
    _lower_node,
    _value,
    _walk,
    core_ops,
    decomposition_table,
)

__all__ = [
    "TargetSet",
    "full_decomposition_table",
    "CORE_ATEN",
    "NNAPI",
    "coreml_ops",
    "lower_to",
    "nnapi_ops",
    "survey",
]


def full_decomposition_table() -> dict[str, Any]:
    """Upstream's *full* post-autograd decomposition table, not the Core ATen one.

    This distinction is the single largest lever on "what blocks NNAPI", and it
    is easy to miss because `decompose.py` is right to use the core table and
    this module is right not to.

    `core_aten_decompositions()` is a *filtered* view: it keeps the rules whose
    job is to get you down to Core ATen, and deliberately drops the rules for
    ops that already are Core ATen -- there is nothing to do for them when Core
    ATen is the destination. `aten.gelu.default` is core, so the core table has
    no rule for it.

    But NNAPI's set is not Core ATen (see the module docstring: 15 of its 29 ops
    are outside it), and `gelu` is one of the ops NNAPI lacks. Lowering it needs
    exactly the rule the core table dropped -- which upstream still has, in
    `global_decomposition_table["post_autograd"]`, the unfiltered registry that
    `torch/_decomp/decompositions.py` populates.

    Measured on this build: core table 386 entries, post-autograd table 1008.
    Of the ops real captured graphs hit that NNAPI lacks, the core table has a
    rule for 5 and the post-autograd table for 22. docs/graph/DECOMP.md §12 carries
    the per-op table.

    No rule is written here either. This is a different *view of upstream's own
    registry*, selected because the destination is different.
    """
    import torch._decomp

    table = torch._decomp.global_decomposition_table["post_autograd"]
    return {str(key): value for key, value in table.items()}


def _serializer_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))  # .../src/main
    return os.path.join(root, "torch", "backends", "_nnapi", "serializer.py")


@functools.lru_cache(maxsize=1)
def nnapi_ops() -> frozenset[str]:
    """The `aten` base names NNAPI's in-tree serializer can emit.

    Base names, not `aten.<op>.<overload>`: `ADDER_MAP` is keyed by TorchScript
    node kind (`aten::add`), which carries no overload. Matching a captured
    `aten.add.Tensor` by its base name is therefore *generous* to NNAPI -- the
    serializer may still refuse a particular overload or dtype. Every count
    this module produces is an upper bound on NNAPI coverage, and the report
    says so rather than presenting it as exact.
    """
    with open(_serializer_path(), encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r"ADDER_MAP = \{(.*?)\n    \}", text, re.S)
    if match is None:  # pragma: no cover -- vendored tree changed shape
        raise RuntimeError(
            f"torchnative target: no ADDER_MAP in {_serializer_path()}; the "
            f"vendored serializer changed shape and this parser must be "
            f"re-read rather than guessed at"
        )
    keys = re.findall(r'"([^"]+)":', match.group(1))
    names = frozenset(k.split("::", 1)[1] for k in keys if k.startswith("aten::"))
    if not names:  # pragma: no cover
        raise RuntimeError("torchnative target: ADDER_MAP parsed to no aten ops")
    return names


def coreml_ops() -> frozenset[str]:
    """Refuses. There is no CoreML operator set in this tree to read.

    `torch/backends/_coreml/preprocess.py` is a *packaging* wrapper: it calls
    `coremltools.convert` and stores the blob. The op set lives in
    coremltools' own torch frontend (`coremltools.converters.mil.frontend
    .torch.ops`, a `@register_torch_op`-decorated registry), and coremltools is
    not installed in this environment.

    This function raises rather than returning a plausible list, because a
    hand-written CoreML set would decide the answer to "how many ops need
    decomposing for CoreML" by the act of writing it, and that answer would
    then be reported as a measurement. docs/graph/DECOMP.md §12 records the CoreML
    number as *not measured* for exactly this reason.
    """
    raise NotImplementedError(
        "torchnative target: no CoreML operator set is readable from this "
        "tree. torch/backends/_coreml is a packaging wrapper around "
        "coremltools.convert and carries no op list; the registry is "
        "coremltools.converters.mil.frontend.torch.ops, and coremltools is not "
        "installed. Install coremltools and read that registry -- do not "
        "transcribe a list here."
    )


class TargetSet:
    """A device's operator set, as a predicate over captured op names."""

    def __init__(self, name: str, base_names: Iterable[str], *, exact: bool = False):
        self.name = name
        self.base_names = frozenset(base_names)
        #: When True, `base_names` are full `aten.<op>.<overload>` strings and
        #: matching is exact. Core ATen is like this; NNAPI's is not.
        self.exact = exact

    def accepts(self, op: str) -> bool:
        if self.exact:
            return op in self.base_names
        return _base(op) in self.base_names

    def outside(self, ops: Iterable[str]) -> list[str]:
        seen, out = set(), []
        for op in ops:
            if not self.accepts(op) and op not in seen:
                seen.add(op)
                out.append(op)
        return out

    def __repr__(self) -> str:
        return f"<TargetSet {self.name}: {len(self.base_names)} ops>"


def _base(op: str) -> str:
    """`aten.add.Tensor` -> `add`; `aten.relu.default` -> `relu`."""
    parts = op.split(".")
    return parts[1] if len(parts) > 2 else parts[-1]


CORE_ATEN = TargetSet("core-aten", core_ops(), exact=True)
NNAPI = TargetSet("nnapi", nnapi_ops())


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


def _round(guards, constants, constant_values, nodes, outputs, table, target,
           *, best_effort, verdicts):
    """One pass: keep what the target accepts, lower the rest.

    Mirrors `decompose._round` with two differences -- the stop predicate is
    `target.accepts` rather than the module-level `is_core`, and `best_effort`
    turns a refusal into a recorded verdict plus a kept node instead of an
    exception. The second is what makes sizing possible: `decompose()` aborts
    the whole trace on the first op it cannot lower, so it can answer "does
    this graph lower" but never "how many of its ops do".
    """
    builder = _Builder(guards, constants, constant_values)
    node_metas: list[list] = []
    mapping: dict[Any, Any] = {}

    def remap(ref):
        return mapping.get(ref, ref)

    changed = False
    for position, node in enumerate(nodes):
        args = [_walk(a, remap) for a in node["args"]]
        kwargs = {k: _walk(v, remap) for k, v in node["kwargs"].items()}

        def keep():
            index = builder.add_node(
                node["op"], args, kwargs, node["outputs"], node["sequence"]
            )
            node_metas.append(list(node["outputs"]))
            for slot in range(len(node["outputs"])):
                mapping[_value("node", position, slot)] = _value("node", index, slot)

        if target.accepts(node["op"]):
            verdicts.setdefault(node["op"], "IN_TARGET")
            keep()
            continue

        marker = len(builder.nodes)
        try:
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
        except DecompositionRefused as error:
            if not best_effort:
                raise
            # Roll the partial splice back so the kept node sees a clean tail.
            del builder.nodes[marker:]
            del node_metas[marker:]
            verdicts[node["op"]] = f"REFUSED: {error}"
            keep()
            continue
        verdicts[node["op"]] = "LOWERED"
        changed = True
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


def _drive(trace, target, max_rounds, *, best_effort, table=None):
    if table is None:
        table = (
            decomposition_table()
            if target.exact
            else full_decomposition_table()
        )
    guards = trace.guards
    constants = trace.constants
    constant_values = trace.constant_values
    nodes = trace.nodes
    outputs = trace.outputs
    verdicts: dict[str, str] = {}

    for _ in range(max_rounds):
        if not target.outside(node["op"] for node in nodes):
            break
        guards, constants, constant_values, nodes, outputs, changed = _round(
            guards, constants, constant_values, nodes, outputs, table, target,
            best_effort=best_effort, verdicts=verdicts,
        )
        if not changed:
            break

    remaining = target.outside(node["op"] for node in nodes)
    lowered = DecomposedTrace(guards, constants, constant_values, nodes, outputs)
    return lowered, remaining, verdicts


def lower_to(trace, target: TargetSet, *, max_rounds: int = 8, table=None) -> DecomposedTrace:
    """Lower `trace` until every op is one `target` accepts, or refuse.

    The refusal names the ops still outside, as `decompose()` does -- a pass
    that returned a partly-lowered graph silently would hand a delegate a graph
    it cannot compile.
    """
    lowered, remaining, _ = _drive(
        trace, target, max_rounds, best_effort=False, table=table
    )
    if remaining:
        raise DecompositionRefused(
            f"torchnative lower_to({target.name}): gave up with these ops still "
            f"outside the target set: {', '.join(remaining)}"
        )
    return lowered


def survey(trace, target: TargetSet, *, max_rounds: int = 8, table=None) -> dict:
    """Size the gap without aborting: a verdict per op, and the best-effort graph.

    Verdicts are `IN_TARGET`, `LOWERED`, or `REFUSED: <why>`. This is the
    measurement `decompose()` cannot make, because it stops at the first
    refusal -- and "how many ops does NNAPI already have" is a question about
    all of them.
    """
    lowered, remaining, verdicts = _drive(
        trace, target, max_rounds, best_effort=True, table=table
    )
    before = []
    for node in trace.nodes:
        if node["op"] not in before:
            before.append(node["op"])
    after = []
    for node in lowered.nodes:
        if node["op"] not in after:
            after.append(node["op"])
    return {
        "target": target.name,
        "verdicts": verdicts,
        "ops_before": before,
        "ops_after": after,
        "in_target_before": [o for o in before if target.accepts(o)],
        "outside_before": [o for o in before if not target.accepts(o)],
        "outside_after": remaining,
        "nodes_before": len(trace.nodes),
        "nodes_after": len(lowered.nodes),
        "trace": lowered,
    }
