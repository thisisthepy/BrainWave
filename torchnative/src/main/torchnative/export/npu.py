"""The front end: a real `transformers` model with submodules swapped for a delegate.

This module is deliberately small and deliberately says nothing about any
vendor. It holds the one shape docs/QNN.md §2 and docs/INTELNPU.md both need:

    front (FIXED)      real transformers, from_pretrained, generate
    back  (SWAPPABLE)  Apple -> CoreML ; Android -> ExecuTorch/QNN ; Windows -> Intel

The load-bearing decision is that `from_pretrained` **returns the real
`transformers` model object**, with some of its submodules replaced in place.
It does not return a wrapper. That is not a convenience: `generate()` is
`GenerationMixin.generate`, several thousand lines that read `self.config`,
`self.device`, `self.can_generate()`, the cache classes and
`_prepare_generation_config`. Anything that wraps the model has to forward all
of that, and every forward is a place the wrapper can be wrong. Returning the
model means there is nothing to forward and nothing to keep in sync.

`docs/QUANT2.md` §3 already framed torchnative's own `quantize_` this way, and
gave the precedent from the other side -- the archived
`intel_npu_acceleration_library.compile(model, dtype=torch.int8)` replaces
leaves and hands the model back too. The user never holds the compiled graph.

What a back end has to supply is a `DelegateModule` subclass. That is the whole
interface between this file and a vendor:

    class MyDelegate(DelegateModule):
        backend_name = "MyBackend"
        def forward(self, *args, **kwargs): ...

`delegate_` does the swapping and `refuse` is how a subclass says no. Neither
knows what a `.pte`, an `.mlpackage` or an OpenVINO blob is.
"""

from __future__ import annotations

import torch.nn as nn


__all__ = [
    "DelegateRefused",
    "DelegateModule",
    "resolve_submodule",
    "replace_submodule",
    "delegate_",
    "delegated_paths",
    "NpuModelForCausalLM",
]


class DelegateRefused(RuntimeError):
    """A delegate declined, by name, to stand in for a submodule.

    Every refusal in this layer carries the backend's name and the reason,
    because the alternative -- falling back to the eager submodule silently --
    produces a model that is correct, slower than it looks, and indistinguishable
    from a working delegate by its output. docs/NPU2.md is the record of that
    exact failure costing a round twice.
    """


class DelegateModule(nn.Module):
    """An `nn.Module` whose forward is somebody else's compiled artefact.

    Subclasses set `backend_name` and implement `forward`. `refuse` is provided
    so that every refusal in this layer reads the same and always names the
    backend; a bare `RuntimeError` from inside a delegate is indistinguishable
    from a bug in the model.

    It is an `nn.Module` rather than a plain callable because the thing it
    replaces is an `nn.Module` and everything upstream assumes that: `.eval()`,
    `.to()`, `.named_modules()`, `state_dict()` and the `_modules` walk that
    `from_pretrained` uses to tie weights all go through the module protocol.
    """

    #: The vendor name this delegate speaks for. Subclasses must set it.
    backend_name = None

    def __init__(self):
        super().__init__()
        if not self.backend_name:
            raise DelegateRefused(
                f"torchnative npu: {type(self).__name__} did not set "
                "`backend_name`. A delegate that cannot say which back end it "
                "is cannot name itself in a refusal, which is the only thing "
                "this layer requires of it."
            )

    def refuse(self, reason):
        raise DelegateRefused(f"torchnative {self.backend_name}: {reason}")

    def forward(self, *args, **kwargs):
        self.refuse(
            f"{type(self).__name__} has no forward. A DelegateModule subclass "
            "must implement one; inheriting this method means the artefact was "
            "never wired to anything."
        )

    def extra_repr(self):
        return f"backend={self.backend_name}"


def resolve_submodule(model, path):
    """The submodule at a dotted `path`, or a refusal naming what was found.

    `model.get_submodule` exists upstream and does nearly this, but its
    `AttributeError` says only which component was missing. The paths handed to
    this function come from a caller who is naming a layer inside somebody
    else's architecture (`model.layers.0.mlp`), and the useful thing to say
    when that is wrong is what *is* there at the point it stopped.
    """
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise DelegateRefused(
            "torchnative npu: empty submodule path. There is no sensible "
            "reading of '' -- replacing the model itself is not a subgraph "
            "delegate, it is a different design."
        )
    obj = model
    walked = []
    for part in parts:
        children = dict(obj.named_children()) if isinstance(obj, nn.Module) else {}
        if part in children:
            obj = children[part]
        elif part.isdigit() and isinstance(obj, (nn.Sequential, nn.ModuleList)):
            index = int(part)
            if index >= len(obj):
                raise DelegateRefused(
                    f"torchnative npu: {'.'.join(walked) or type(model).__name__}"
                    f" has {len(obj)} entries, so index {index} in path "
                    f"{path!r} does not exist."
                )
            obj = obj[index]
        else:
            raise DelegateRefused(
                f"torchnative npu: no submodule {part!r} on "
                f"{'.'.join(walked) or type(model).__name__} while resolving "
                f"{path!r}. Children there: {sorted(children) or '(none)'}."
            )
        walked.append(part)
    return obj


def replace_submodule(model, path, new):
    """Put `new` at `path` and return what was there. In place; returns the old.

    The old module is returned rather than dropped because a caller that wants
    to undo this has no other handle on it -- the model no longer references it
    and nothing else in this file remembers it.
    """
    if not isinstance(new, nn.Module):
        raise DelegateRefused(
            f"torchnative npu: cannot put a {type(new).__name__} at {path!r}. "
            "The parent holds its children in `_modules` and every walk over "
            "them (state_dict, .to(), .eval()) assumes nn.Module."
        )
    parts = [p for p in path.split(".") if p]
    parent = resolve_submodule(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    leaf = parts[-1]
    old = resolve_submodule(model, path)
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = new
    else:
        setattr(parent, leaf, new)
    return old


def delegate_(model, plan):
    """Replace each submodule named in `plan` with its delegate. Returns `model`.

    `plan` maps a dotted path to a `DelegateModule`. The trailing underscore is
    upstream's spelling for "in place" (`Tensor.add_`, `torchao.quantize_`) and
    it is accurate here: the same object comes back, so a caller who wrote
    `model = AutoModelForCausalLM.from_pretrained(...)` still holds a model
    whose `generate` is `GenerationMixin.generate` and whose `config` is the one
    the checkpoint declared.

    Nothing is replaced if any entry is bad. A half-applied plan leaves a model
    that runs and is neither the eager model nor the delegated one, and no
    output tells the two apart -- the same shape of failure this module's
    docstring is about.
    """
    if not isinstance(plan, dict):
        raise DelegateRefused(
            f"torchnative npu: plan must be a dict of path -> DelegateModule, "
            f"got {type(plan).__name__}."
        )
    if not plan:
        raise DelegateRefused(
            "torchnative npu: empty plan. Replacing nothing and reporting "
            "success would make `delegate_` indistinguishable from a no-op, "
            "which is exactly what a silent fallback looks like."
        )
    for path, new in plan.items():
        if not isinstance(new, DelegateModule):
            raise DelegateRefused(
                f"torchnative npu: {path!r} maps to a "
                f"{type(new).__name__}, not a DelegateModule. Only a "
                "DelegateModule can name a back end when it refuses, and that "
                "naming is the entire contract of this layer."
            )
        resolve_submodule(model, path)  # refuses by name before anything moves

    for path, new in plan.items():
        replace_submodule(model, path, new)
    return model


def delegated_paths(model):
    """Every path in `model` currently held by a `DelegateModule`, sorted.

    This is the answer to "did the swap actually happen", read off the model
    rather than off the plan that was submitted. A plan is a request; this is
    the state.
    """
    return sorted(
        name
        for name, mod in model.named_modules()
        if isinstance(mod, DelegateModule)
    )


class NpuModelForCausalLM:
    """`from_pretrained` that hands back a real model with delegated submodules.

    The whole class is one static method and that is the point. The archived
    `intel_npu_acceleration_library` exposed `NPUModelForCausalLM` with the
    same shape, and docs/QUANT2.md §3 records why this repository already
    agreed with it: the model is a real `transformers` instance, the source is
    not edited, the *instance* is.

    ::

        model = NpuModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            plan={"model.layers.0.mlp": QnnModule("layer0_mlp.pte")},
        )
        model.generate(**tokenizer("hello", return_tensors="pt"))

    `model` there **is** a `LlamaForCausalLM`. `isinstance` says so, `.config`
    is the checkpoint's, and `.generate` is `GenerationMixin.generate` with
    nothing in front of it. Only `model.model.layers[0].mlp` is different.

    `plan` may be a callable taking the freshly-built model and returning the
    dict, for the common case where the caller wants to name every layer of a
    stack it has not seen yet.

    This class deliberately does **not** know how to produce an artefact. That
    is offline work on another host (docs/QNN.md §2), and a `from_pretrained`
    that quietly compiled something would be doing minutes of work behind a
    call that reads like a download.
    """

    @staticmethod
    def from_pretrained(model_id, *, plan, **kwargs):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        resolved = plan(model) if callable(plan) else plan
        return delegate_(model, resolved)
