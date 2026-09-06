"""Fold `prims.*` nodes back to the `aten` spellings a target's serialiser reads.

## Why this pass exists at all

docs/DECOMP.md §12.4 counted what blocked NNAPI and found thirteen missing
`prims.*` kernels. Those landed (docs/PRIMS.md), upstream's decomposition table
now runs to completion against this build, and **the number of ops outside
NNAPI did not fall**. For `vit` it rose, 10 to 11.

The reason is structural rather than a bug in anything. With the kernels
present, `_refs.erf` and `_refs.t` no longer stop half-way, so the lowered
graph *ends* on `prims.erf` and `prims.transpose` instead of stopping at
`aten.erf` and `aten.permute`. And a `prims.*` node is strictly worse to be
left holding than the `aten` node it replaced:

* `torch/backends/_nnapi/serializer.py`'s `ADDER_MAP` is keyed by TorchScript
  node kind -- `aten::add`, `aten::mul`. **No `prims.*` node can ever reach
  that serialiser**, whatever it computes.
* prims are reference *primitives*, so no upstream table decomposes them
  further. Lowering has reached its fixed point and the fixed point is outside
  every target set that is spelled in `aten`.

So the refusal changed from "add this kernel" to "there is no rule for it", and
the treatment is not another kernel. It is to run the last step in the other
direction: after lowering has finished, rewrite the prims nodes back to the
`aten` spellings that mean the same thing.

## Direction matters, and it is not symmetric

`prims.transpose` is **stricter** than `aten.permute`. docs/PRIMS.md §1
measured it against upstream 2.13.0: `prims.transpose(t, [-1, 0])` raises
`ValueError: Received an invalid permutation, [-1, 0]!` while
`aten.permute(t, [-1, 0])` computes. `prims.split_dim` rejects a negative
`dim` that every `aten` op in this shim accepts.

That asymmetry is what makes this pass safe in one direction only:

    prims -> aten   total     every argument prims accepted, aten accepts,
                              and computes the same value for it
    aten -> prims   partial   aten accepts arguments prims refuses, so the
                              rewrite would turn a working graph into a raise

This module only ever goes prims -> aten. A pass that went the other way --
"canonicalise everything to prims" -- would look tidier and would be wrong for
`[-1, 0]`, which is a permutation real graphs produce.

## Some prims have no `aten` spelling, and those are refused by name

`prims.broadcast_in_dim(a, shape, broadcast_dimensions)` is XLA's broadcast:
the caller says which axis of the *result* each input axis becomes, so an axis
can be inserted in the middle and a size-3 axis can be broadcast against a
size-2 one. `aten.expand` is right-aligned and cannot express
`broadcast_in_dim(ones(3), [3, 2], [0])` at all -- docs/PRIMS.md §1 keeps that
as the control case.

There is a *composite* that computes it (`view` to insert the size-1 axes, then
`expand`), and this module deliberately does not emit it. Refolding is a
re-spelling: one node in, one node out, same value, nothing to prove
numerically beyond the identity. Emitting a two-node composite is a
decomposition, it changes the node count, and it belongs with the rules in
`decompose.py` -- which are upstream's and not written here. So
`prims.broadcast_in_dim` is **absent from the table and refused by name**,
which is the same treatment `nnapi._SIGNATURES` gives an op with no calling
convention.

## This pass is terminal, and that is deliberate

`refold` runs *after* lowering and its output is not fed back in. It cannot be,
because the two passes are inverses on the ops they share: the union table
lowers `aten.tanh` to `prims.tanh`, this pass folds `prims.tanh` back to
`aten.tanh`, and a loop over both does not terminate. `lower_and_refold` is the
composition in the only order that has a fixed point.

## What it is worth, measured

docs/REFOLD.md leads with the before/after table per model. The short version
is that it recovers the regression docs/PRIMS.md §3 reported and does not
invent progress beyond it: `vit` goes back from 11 ops outside NNAPI to 10,
`smollm2_llama` and `mobilenet_v2` are unchanged at 15 and 6. Two of those
three numbers not moving is the result, not a bug in the pass -- the ops still
outside are outside for reasons that have nothing to do with prims.
"""

from __future__ import annotations

from typing import Any, Iterable

from .decompose import (
    DecomposedTrace,
    _as_the_dispatcher_would_call_it,
    _value,
    _walk,
)

__all__ = [
    "RefoldRefused",
    "REFOLDABLE",
    "UNREFOLDABLE_PRIMS",
    "refold",
    "refoldable_ops",
    "lower_and_refold",
]


class RefoldRefused(NotImplementedError):
    """A `prims.*` node has no `aten` spelling, and this says which."""


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
#
# Each entry maps one recorded `prims.*` op onto one `aten` op, and says how
# the recorded arguments move. A rule returns `(op, args, kwargs)` for the
# single replacement node, or raises `RefoldRefused`.
#
# `node` is the whole recorded node, so a rule that needs the *result* shape
# (only `split_dim` does) can read the meta capture recorded for it rather than
# recomputing a shape from the arguments.


def _same_name(aten_name: str):
    """`prims.f(a) -> aten.f(a)`: nine of the thirteen are this.

    docs/PRIMS.md §1: nine of the prims kernels are the aten kernel under
    another key -- upstream's own `impl_aten` for `prims.cos` *is* `torch.cos`.
    So this is not an approximation, it is the same function reached by its
    other name.
    """

    def rule(node, args, kwargs):
        if len(args) != 1 or kwargs:
            raise RefoldRefused(
                f"torchnative refold: {node['op']} was recorded with "
                f"{len(args)} positional and {sorted(kwargs)} keyword "
                f"argument(s); the refold to aten.{aten_name}.default is only "
                f"defined for the one-tensor call"
            )
        return f"aten.{aten_name}.default", list(args), {}

    return rule


def _clone(node, args, kwargs):
    """`prims.clone(a, *, memory_format=None)` -> `aten.clone.default`.

    Same signature shape, same kernel upstream (`torch.clone`). The
    `memory_format` keyword is carried through rather than dropped: dropping it
    would be a silent change of what the node computes.
    """
    if len(args) != 1 or set(kwargs) - {"memory_format"}:
        raise RefoldRefused(
            f"torchnative refold: {node['op']} was recorded as "
            f"{len(args)} positional + {sorted(kwargs)}, and the refold knows "
            f"only (a, *, memory_format)"
        )
    return "aten.clone.default", list(args), dict(kwargs)


def _view_of(node, args, kwargs):
    """`prims.view_of(a)` -> `aten.alias.default(a)`.

    docs/PRIMS.md §1 states the identity directly: `prims.view_of` *is*
    `aten.alias`. The separate kernel exists only because the parameter is
    named `a` rather than `self`.
    """
    if len(args) != 1 or kwargs:
        raise RefoldRefused(
            f"torchnative refold: {node['op']} was recorded with "
            f"{len(args)} positional and {sorted(kwargs)} keyword argument(s)"
        )
    return "aten.alias.default", list(args), {}


def _transpose(node, args, kwargs):
    """`prims.transpose(a, permutation)` -> `aten.permute.default(a, dims)`.

    Not `aten.transpose.int`, which swaps two axes; `prims.transpose` takes a
    *full* permutation, which is `aten.permute`'s argument exactly.

    This is the direction that is safe. `aten.permute` accepts every
    permutation `prims.transpose` accepts and computes the same result for it;
    the converse fails on negative entries -- `prims.transpose(t, [-1, 0])`
    raises where `aten.permute(t, [-1, 0])` computes (docs/PRIMS.md §1). See
    the module docstring.
    """
    if len(args) != 2 or kwargs:
        raise RefoldRefused(
            f"torchnative refold: {node['op']} was recorded with "
            f"{len(args)} positional and {sorted(kwargs)} keyword argument(s); "
            f"the refold knows only (a, permutation)"
        )
    tensor, permutation = args
    if not isinstance(permutation, (list, tuple)):
        raise RefoldRefused(
            f"torchnative refold: {node['op']} was recorded with a "
            f"{type(permutation).__name__} permutation, and aten.permute needs "
            f"a concrete int list"
        )
    return "aten.permute.default", [tensor, [int(d) for d in permutation]], {}


def _split_dim(node, args, kwargs):
    """`prims.split_dim(a, dim, outer_length)` -> `aten.view.default(a, shape)`.

    `split_dim` replaces axis `dim` of length `n` with two adjacent axes
    `outer_length` and `n // outer_length`. That is a reshape which never
    reorders elements and never crosses a discontiguity, because the two new
    axes come out of one old one -- so `aten.view` is always legal for it, and
    is the same view `prims.split_dim` itself returns.

    The result shape is read from the meta capture recorded for this node's
    output rather than recomputed from `a`'s shape and `outer_length`. Both
    derivations would agree, and the recorded one is the one the rest of the
    trace is already typed by, so using it cannot introduce a disagreement
    between the node and its own meta.
    """
    if kwargs or len(args) != 3:
        raise RefoldRefused(
            f"torchnative refold: {node['op']} was recorded with "
            f"{len(args)} positional and {sorted(kwargs)} keyword argument(s); "
            f"the refold knows only (a, dim, outer_length)"
        )
    metas = node["outputs"]
    if len(metas) != 1 or metas[0] is None or "shape" not in metas[0]:
        raise RefoldRefused(
            f"torchnative refold: {node['op']} has no recorded output shape, "
            f"and the aten.view spelling needs one -- refusing rather than "
            f"recomputing a shape this trace does not carry"
        )
    return "aten.view.default", [args[0], [int(d) for d in metas[0]["shape"]]], {}


#: `prims.<op>.<overload>` -> rule. Everything else is refused by name.
REFOLDABLE: dict[str, Any] = {
    # -- nine that are the aten kernel under another key (docs/PRIMS.md §1)
    "prims.cos.default": _same_name("cos"),
    "prims.sin.default": _same_name("sin"),
    "prims.erf.default": _same_name("erf"),
    "prims.tanh.default": _same_name("tanh"),
    "prims.sqrt.default": _same_name("sqrt"),
    "prims.rsqrt.default": _same_name("rsqrt"),
    "prims.reciprocal.default": _same_name("reciprocal"),
    "prims.neg.default": _same_name("neg"),
    "prims.clone.default": _clone,
    # -- four with their own kernels; three of these mean something different
    #    from the aten op of the same name, so none of them is a rename
    "prims.view_of.default": _view_of,
    "prims.transpose.default": _transpose,
    "prims.split_dim.default": _split_dim,
    # "prims.broadcast_in_dim.default" is deliberately absent -- see below.
}

#: Prims this pass knows about and *will not* fold, with the reason. Kept as
#: data rather than as a comment so a test can require the refusal to name it.
UNREFOLDABLE_PRIMS: dict[str, str] = {
    "prims.broadcast_in_dim.default": (
        "prims.broadcast_in_dim is XLA's broadcast: broadcast_dimensions says "
        "which axis of the result each input axis becomes, so it can insert an "
        "axis in the middle and broadcast a size-3 axis against a size-2 one. "
        "aten.expand is right-aligned and has no spelling for "
        "broadcast_in_dim(ones(3), [3, 2], [0]) at all (docs/PRIMS.md §1). A "
        "view+expand composite computes it, but that is a decomposition rather "
        "than a re-spelling and this pass does not write decompositions"
    ),
}


def refoldable_ops() -> frozenset[str]:
    """The `prims.*` keys this pass can re-spell. Read from the table."""
    return frozenset(REFOLDABLE)


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def refold(trace, *, best_effort: bool = False):
    """Rewrite every foldable `prims.*` node of `trace` to its `aten` spelling.

    Returns `(trace, folded)` where `folded` lists the `prims.*` keys that were
    rewritten, in the order they were met.

    With `best_effort=False` (the default) a `prims.*` node with no rule raises
    `RefoldRefused` naming it, the way `lower_to` refuses rather than handing a
    delegate a graph it cannot compile. With `best_effort=True` the node is
    kept as recorded, so the pass can be *measured* on a graph it cannot fully
    fold -- the same split `target.survey` makes against `target.lower_to`.

    Non-`prims` nodes are copied through untouched. This pass has no opinion
    about them; the target set does.
    """
    nodes: list[dict] = []
    mapping: dict[tuple, Any] = {}
    folded: list[str] = []

    def remap(ref):
        return mapping.get((ref.kind, ref.index, ref.output), ref)

    for position, node in enumerate(trace.nodes):
        op = node["op"]
        args = [_walk(a, remap) for a in node["args"]]
        kwargs = {k: _walk(v, remap) for k, v in node["kwargs"].items()}

        rule = REFOLDABLE.get(op)
        if rule is not None:
            # Capture records post-dispatch, so an argument may have arrived by
            # keyword. Put the call back into schema order first, using the
            # same helper `decompose` uses, rather than each rule guessing.
            flat_args, flat_kwargs = _as_the_dispatcher_would_call_it(
                op, args, kwargs
            )
            if node["sequence"] or len(node["outputs"]) != 1:
                raise RefoldRefused(
                    f"torchnative refold: {op} was recorded with "
                    f"{len(node['outputs'])} outputs; every entry in the "
                    f"refold table is one-in-one-out"
                )
            new_op, new_args, new_kwargs = rule(node, flat_args, flat_kwargs)
            folded.append(op)
            index = len(nodes)
            nodes.append(
                {
                    "op": new_op,
                    "args": list(new_args),
                    "kwargs": dict(new_kwargs),
                    "outputs": list(node["outputs"]),
                    "sequence": node["sequence"],
                }
            )
            mapping[("node", position, 0)] = _value("node", index, 0)
            continue

        if op.startswith("prims.") and not best_effort:
            why = UNREFOLDABLE_PRIMS.get(op)
            raise RefoldRefused(
                f"torchnative refold: no aten spelling for {op}. "
                + (
                    why
                    if why is not None
                    else "It is not in the refold table, and this pass names "
                    "what it cannot do rather than approximating it"
                )
            )

        index = len(nodes)
        nodes.append(
            {
                "op": op,
                "args": args,
                "kwargs": kwargs,
                "outputs": list(node["outputs"]),
                "sequence": node["sequence"],
            }
        )
        for slot in range(len(node["outputs"])):
            mapping[("node", position, slot)] = _value("node", index, slot)

    refolded = DecomposedTrace(
        list(trace.guards),
        list(trace.constants),
        list(trace.constant_values),
        nodes,
        [_walk(ref, remap) for ref in trace.outputs],
    )
    return refolded, folded


def lower_and_refold(trace, target, *, max_rounds: int = 8, table=None):
    """Lower toward `target`, then fold the prims residue back to aten.

    The composition in the only order that terminates -- see the module
    docstring. Returns the same record `target.survey` does, plus `refolded`
    (the prims keys rewritten) and with `outside_after` recomputed against the
    refolded graph, because that is the graph a serialiser would be handed.

    Best-effort throughout, like `survey`: this is a sizing tool, and a pass
    that aborted on the first unfoldable prim could not report how many of them
    there were.
    """
    from . import target as _target_module

    survey = _target_module.survey(
        trace, target, max_rounds=max_rounds, table=table
    )
    refolded, folded = refold(survey["trace"], best_effort=True)
    survey = dict(survey)
    survey["trace"] = refolded
    survey["refolded"] = folded
    survey["ops_after"] = _distinct(node["op"] for node in refolded.nodes)
    survey["outside_after"] = target.outside(node["op"] for node in refolded.nodes)
    survey["nodes_after"] = len(refolded.nodes)
    return survey


def _distinct(ops: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for op in ops:
        if op not in seen:
            seen.add(op)
            out.append(op)
    return out
