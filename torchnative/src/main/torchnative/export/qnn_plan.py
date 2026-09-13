"""What `_compile_model`'s QNN analogue would lower, and what it would not --
without a device, without the Qualcomm SDK, and without ExecuTorch's AOT
export actually running.

`torchnative.export.intelnpu.plan_lowering` exists because a granularity
defect shipped once: a whole Qwen3-4B was refused because one oversized
`lm_head` raised out of the walk. This module gives QNN the same kind of
answer, but the failure mode it defends against is the opposite shape.

**Intel NPU raises when a leaf will not fit** -- `linear_ir` throws
`IntelNPUUnsupported` and the walk catches it. **QNN's partitioner declines
nodes silently**: a node the HTP delegate does not accept is left as a CPU
fallback op inside the exported graph, the graph still runs, and the numbers
are still correct -- ExecuTorch's partitioning is `to_edge_transform_and_lower`
picking nodes it recognises, not a compiler that errors on what it does not
(docs/devices/QNN.md sections referenced from `torchnative.export.qnn`
`delegation_report` describe the artefact-side view of exactly this: some ops
delegate, some do not, and `.pte` metadata is the only place that shows up
after the fact). A CPU-fallback build that runs fine is not evidence that
lowering happened, and a plan that assumed "leaf module type looks liftable,
so it must lower" would be reporting exactly what `docs/graph/NPU2.md`
documents as the failure: a report that says more than anything checked.

So this file's one rule is stricter than intelnpu's: **a leaf is `eligible`
only if something -- the op table this round is given -- affirmatively said
it would be taken.** No leaf is eligible by default, by type-name allowlist,
or by absence of a reason to exclude it.

## The op-table interface

A sibling round owns `torchnative.export.qnn_ops` (`supported_ops()`,
`constraints()`, `check_leaf(name, module)`); it does not exist yet. This
module never assumes it does -- `_default_ops()` imports it lazily, only when
a caller does not supply one, and raises `QnnPlanUnavailable` by name rather
than an `ImportError` a caller has to recognise.

`check_leaf(name, module)` is the contract this module drives, and the one it
was told to program against:

    check_leaf(name: str, module: nn.Module) -> verdict

where `verdict` is either

* a `(taken: bool, reason: str) -> tuple`, or
* an object exposing `.taken` (or `.lowered`) and `.reason`,

so that whichever shape the sibling file lands with, `_read_verdict` below
reads it without this file needing to change. `reason` is required in both
shapes and is always attached to the leaf's report entry, taken or not --
"why" is the point of this module, not merely "yes or no".

Until `qnn_ops` exists, every test in `test_qnn_plan.py` injects its own
`ops` object -- a small stand-in exposing exactly `check_leaf` (and
sometimes `supported_ops` / `constraints`, exercised only to prove
`plan_lowering` does not require them) -- so this module's logic is provable
without the sibling file present.
"""

from __future__ import annotations


__all__ = [
    "QnnPlanUnavailable",
    "plan_lowering",
]


class QnnPlanUnavailable(RuntimeError):
    """`plan_lowering` was called with no `ops=` and `torchnative.export.qnn_ops`
    is not importable.

    Not an `ImportError` a caller has to recognise as "the op table" versus
    some unrelated missing dependency -- named, and pointing at the fix.
    """


def _torch():
    import torch

    return torch


def _default_ops():
    try:
        from torchnative.export import qnn_ops
    except ImportError as exc:
        raise QnnPlanUnavailable(
            "torchnative qnn_plan: plan_lowering() was called with no ops= "
            "argument and torchnative.export.qnn_ops could not be imported "
            f"({exc}). That module carries the QNN op table (supported_ops, "
            "constraints, check_leaf) this plan is built on; pass ops= "
            "explicitly (a stand-in with a check_leaf(name, module) method is "
            "enough for testing) or install/build the module that ships it."
        ) from exc
    return qnn_ops


def _read_verdict(name, module, verdict):
    """Normalise a `check_leaf` return into `(taken: bool, reason: str)`.

    Accepts the `(bool, str)` tuple shape and an object exposing `.taken` (or
    `.lowered`) and `.reason`, per the contract documented at module level.
    Anything else is a programming error in the ops table, not a "declined"
    verdict -- silently treating a malformed return as "declined" would hide
    exactly the kind of bug this module exists to surface, so it raises
    instead of guessing.
    """
    # The mapping shape is what `qnn_ops.check_leaf` actually returns:
    # `{"accepted": bool, "reason": str}`. It is listed first because it is the
    # only shape the SHIPPED table uses -- the tuple and attribute forms below
    # exist so this module is testable without `qnn_ops`, and that injectable
    # fake is precisely why the mismatch went unseen: the two were written in
    # parallel worktrees, each green against its own idea of the contract, and
    # only meeting on develop raised TypeError on every real leaf.
    if isinstance(verdict, dict) or hasattr(verdict, "keys"):
        for key in ("accepted", "taken", "lowered"):
            if key in verdict:
                return bool(verdict[key]), str(verdict.get("reason", ""))
        raise TypeError(
            f"torchnative qnn_plan: check_leaf({name!r}, ...) returned the "
            f"mapping {verdict!r}, which carries none of 'accepted', 'taken' "
            "or 'lowered'. plan_lowering will not guess a verdict it was not "
            "given."
        )

    if isinstance(verdict, tuple):
        if len(verdict) != 2:
            raise TypeError(
                f"torchnative qnn_plan: check_leaf({name!r}, ...) returned a "
                f"{len(verdict)}-tuple; expected (taken: bool, reason: str)."
            )
        taken, reason = verdict
        return bool(taken), str(reason)

    taken = getattr(verdict, "taken", None)
    if taken is None:
        taken = getattr(verdict, "lowered", None)
    if taken is None:
        raise TypeError(
            f"torchnative qnn_plan: check_leaf({name!r}, ...) returned "
            f"{verdict!r}, which has neither a .taken nor a .lowered "
            "attribute and is not a (bool, str) tuple. plan_lowering will "
            "not guess a verdict it was not given."
        )
    reason = getattr(verdict, "reason", "")
    return bool(taken), str(reason)


#: The ATen operator each `nn.Module` leaf decomposes to, which is what the op
#: table is keyed on. `qnn_ops.check_leaf` takes an ATen op name -- it is built
#: from the `target = [...]` lists of ExecuTorch's QNN node-visitor builders,
#: and those name operators, not module classes.
#:
#: This map is the reason the first version of this module reported ZERO
#: eligible leaves on a model of nothing but `nn.Linear`. It passed the module
#: PATH ("fc1") where an op name belonged, the table correctly answered "no
#: node visitor is registered for fc1", and every test was green because every
#: test injected a stand-in that also keyed on the path. Two rounds, each
#: internally consistent, composing into a plan that declined everything.
#:
#: Absent from this map means unmapped, not unsupported: an unmapped leaf is
#: declined with a reason saying which is which, so a missing entry reads as
#: work to do rather than as a hardware limit.
_MODULE_TO_ATEN = {
    "Linear": "aten.linear.default",
    "Conv1d": "aten.convolution.default",
    "Conv2d": "aten.convolution.default",
    "Embedding": "aten.embedding.default",
    "LayerNorm": "aten.native_layer_norm.default",
    "GroupNorm": "aten.native_group_norm.default",
    "RMSNorm": "aten.rms_norm.default",
    "BatchNorm2d": "aten._native_batch_norm_legit_no_training.default",
    "ReLU": "aten.relu.default",
    "GELU": "aten.gelu.default",
    "SiLU": "aten.mul.Tensor",
    "SiLUActivation": "aten.mul.Tensor",
    "Sigmoid": "aten.sigmoid.default",
    "Tanh": "aten.tanh.default",
    "Softmax": "aten._softmax.default",
    "Hardswish": "aten.hardswish.default",
    "Hardsigmoid": "aten.hardsigmoid.default",
    "Hardtanh": "aten.hardtanh.default",
    "ELU": "aten.elu.default",
    "PReLU": "aten.prelu.default",
    "MaxPool2d": "aten.max_pool2d_with_indices.default",
    "AvgPool2d": "aten.avg_pool2d.default",
    "AdaptiveAvgPool2d": "aten.adaptive_avg_pool2d.default",
    "Dropout": "aten._to_copy.default",
    "Identity": "aten._to_copy.default",
    "Flatten": "aten.view_copy.default",
}


def aten_op_for(module) -> "str | None":
    """The ATen operator name `module` decomposes to, or None if unmapped."""
    return _MODULE_TO_ATEN.get(type(module).__name__)


def _check_leaf(ops, name, module):
    if not hasattr(ops, "check_leaf"):
        raise QnnPlanUnavailable(
            f"torchnative qnn_plan: the ops object passed in ({ops!r}) has no "
            "check_leaf(name, module) -- plan_lowering has nothing that could "
            "affirmatively say a leaf would be taken, and it will not assume "
            "one by default."
        )
    # A table keyed on ATen op names (`qnn_ops.check_leaf(op_name, ...)`) gets
    # the op name; a stand-in keyed on (name, module) gets those. Which one
    # this is, is decided by the callable's own signature rather than by
    # catching TypeError, because a TypeError raised INSIDE a correct table
    # would then be misread as "wrong arity" and the leaf silently retried.
    verdict = None
    if getattr(ops, "__name__", "") == "torchnative.export.qnn_ops" or (
            _takes_op_name(ops.check_leaf)):
        op = aten_op_for(module)
        if op is None:
            return False, (
                f"torchnative qnn_plan: {type(module).__name__} has no entry in "
                f"qnn_plan's module-to-ATen map, so no op name could be put to "
                f"the QNN table. This is UNMAPPED, not unsupported -- add it to "
                f"_MODULE_TO_ATEN once its decomposition is known."
            )
        verdict = ops.check_leaf(op)
    else:
        verdict = ops.check_leaf(name, module)
    return _read_verdict(name, module, verdict)


def _takes_op_name(fn) -> bool:
    """True when `fn` looks like `check_leaf(op_name, shapes=None, dtype=None)`.

    Decided from the signature, not from a failed call: see `_check_leaf`.
    """
    import inspect

    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    required = [p for p in params
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(required) == 1


def plan_lowering(model, predicate=None, ops=None):
    """What ExecuTorch's QNN partitioner would take, and what it would leave
    on the CPU, per leaf -- without exporting, without the Qualcomm SDK, and
    without a device.

    Mirrors `torchnative.export.intelnpu.plan_lowering`'s shape and its
    `predicate(name, module) -> bool` signature (same as
    `torchnative.quant.quantize_`), but not its walk: Intel NPU's plan only
    ever asks about `torch.nn.Linear`, because only `Linear` has a lowering
    path (`_NPULinear`) at all. QNN can take a much wider set of ops, and this
    round has no static list of which leaf *types* are liftable -- only the
    op table (`ops.check_leaf`) can say. So every leaf module (one with no
    children) is asked, not just `Linear`s, and none is answered without
    asking.

    A leaf is `eligible` **only if `ops.check_leaf(name, module)` said it
    would be taken.** There is no allowlist-by-type fallback: an op the table
    does not recognise is left on the CPU and named, exactly like an oversized
    Linear is left on the CPU and named by `intelnpu.plan_lowering` -- except
    here it is the ordinary case for anything QNN's HTP delegate declines,
    not an edge case at a size limit.

    `predicate(name, module) -> bool` narrows the selection before `ops` is
    ever consulted, same spelling and same order as intelnpu's plan.

    `ops`, if given, must expose `check_leaf(name, module) -> verdict` (see
    module docstring for the accepted verdict shapes). If omitted,
    `torchnative.export.qnn_ops` is imported lazily; if that module does not
    exist yet, `QnnPlanUnavailable` is raised by name rather than a bare
    `ImportError`.

    Returns a dict with `eligible` (list of leaf names), `skipped`
    (`(name, reason)` pairs -- predicate exclusions and declined ops alike),
    `left_on_cpu` (leaf type name -> count, for every skipped leaf),
    `fully_offloaded`, `parameters_moved`, `parameters_total` and
    `fraction_moved` -- the same field names intelnpu's plan uses, so a
    caller that already reads one can read the other.
    """
    torch = _torch()
    if ops is None:
        ops = _default_ops()

    eligible = []
    skipped = []
    left_on_cpu = {}
    moved = 0

    def _record_skip(path, module, reason):
        skipped.append((path, reason))
        type_name = type(module).__name__
        left_on_cpu[type_name] = left_on_cpu.get(type_name, 0) + 1

    def walk(parent, prefix):
        nonlocal moved
        for name, child in list(parent.named_children()):
            path = f"{prefix}{name}"
            grandchildren = list(child.named_children())
            if grandchildren:
                walk(child, f"{path}.")
                continue

            # A true leaf -- the only thing plan_lowering ever asks about,
            # because the op table's decision is per-node, not per-container.
            if predicate is not None and not predicate(path, child):
                _record_skip(path, child, "excluded by predicate")
                continue

            taken, reason = _check_leaf(ops, path, child)
            if taken:
                eligible.append(path)
                moved += sum(p.numel() for p in child.parameters(recurse=False))
            else:
                if not reason:
                    reason = (
                        f"{type(child).__name__} declined by the QNN op table "
                        "(no reason given)"
                    )
                _record_skip(path, child, reason)

    walk(model, "")
    total = sum(p.numel() for p in model.parameters())
    return {
        "eligible": eligible,
        "skipped": skipped,
        "left_on_cpu": dict(sorted(left_on_cpu.items())),
        "fully_offloaded": not skipped,
        "parameters_moved": moved,
        "parameters_total": total,
        "fraction_moved": (moved / total) if total else 0.0,
    }
