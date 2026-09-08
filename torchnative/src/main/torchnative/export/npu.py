"""Internal submodule-swapping plumbing. **Its user-facing API was withdrawn.**

This module is deliberately small and deliberately says nothing about any
vendor. It holds the mechanism docs/devices/QNN.md §2 and
docs/devices/INTELNPU.md both need:

    front (FIXED)      real transformers, from_pretrained, generate
    back  (SWAPPABLE)  Apple -> CoreML ; Android -> ExecuTorch/QNN ; Windows -> Intel

**What was withdrawn, and why.** `NpuModelForCausalLM`, `delegate_`,
`delegated_paths`, `DelegateModule`, `replace_submodule` and `resolve_submodule`
were exported as the way to run a model. Three separate faults:

* **Naming.** `NpuModelForCausalLM` copied the archived
  `intel_npu_acceleration_library`. The ecosystem convention names the *project*
  -- `OVModelForCausalLM` (optimum-intel), `ORTModelForCausalLM`,
  `IPEXModelForCausalLM` -- with no `Auto` prefix. "NPU" names neither this
  project nor a single vendor.
* **Shape.** `delegate_(model, plan)` was a second in-place module-replacement
  entry point beside `torchnative.quant.quantize_(model, format=...)`, which
  already had torchao's spelling for the same move. One codebase, two shapes,
  one operation.
* **`.to(device)`.** Because `from_pretrained` here returned a real
  `nn.Module`, there was nothing for `.to("npu")` to mean: `nn.Module.to()`
  moves parameters, PyTorch has no `npu` device on this shim
  (`torch.device("npu")` raises -- `torch._C._rename_privateuse1_backend` is a
  stub), and a graph-compiling accelerator cannot dispatch eager ops one at a
  time. optimum can honour `model.to("npu")` only because `OVModel` is a
  *wrapper* owning a compiled graph and `.to()` recompiles. There was no
  wrapper here.

The replacement is `torchnative.transformers.AutoModelForCausalLM`, and it now
**exists** (docs/api/TRANSFORMERS.md). It does not take optimum's shape exactly,
and the difference is the point: `from_pretrained` returns the **real model**,
so there is no wrapper and `.to()` is `nn.Module.to()` on one object rather than
a recompiling method on a wrapper. The device is `torchnative.device.npu`, not
`torch.device("npu")`.

**Recompiling for the accelerator is still not implemented**:
`model.to(torchnative.device.npu)` resolves the NPU, names the unit, and refuses
at the compile step. So the capability this file's withdrawn names reached for
is still unavailable -- the wall moved and acquired a name, and nothing in this
file should be read as saying the compile step exists.

**What survives, and why.** The plumbing itself is sound and
`torchnative.export.qnn` builds on it, so it is kept private:
`_DelegateModule`, `_delegate_`, `_delegated_paths`, `_replace_submodule`,
`_resolve_submodule`. Every refusal they carry is unchanged. The reasoning that
made returning a real `transformers` model attractive is also unchanged and is
recorded here rather than deleted: `generate()` is `GenerationMixin.generate`,
several thousand lines reading `self.config`, `self.device`,
`self.can_generate()`, the cache classes and `_prepare_generation_config`, and
anything that wraps the model has to forward all of it. That is a real cost of
the wrapper shape, and the wrapper shape was chosen anyway, because it is the
only shape in which `.to(device)` is honest.

A back end supplies a `_DelegateModule` subclass. That is the whole interface
between this file and a vendor:

    class MyDelegate(_DelegateModule):
        backend_name = "MyBackend"
        def forward(self, *args, **kwargs): ...

`_delegate_` does the swapping and `refuse` is how a subclass says no. Neither
knows what a `.pte`, an `.mlpackage` or an OpenVINO blob is.
"""
from __future__ import annotations

import torch.nn as nn


__all__ = [
    "DelegateRefused",
    "DelegateWithdrawn",
]

#: Every user-facing name this module used to export, and the one-line reason
#: each was withdrawn. `__getattr__` below turns each of them into a refusal
#: that names itself, so that reaching for one fails loudly at *access* rather
#: than at call: `npu.NpuModelForCausalLM.from_pretrained(...)` has to refuse on
#: the attribute, since there is no class left to hold the method.
_WITHDRAWN = {
    "NpuModelForCausalLM": (
        "its name copies the archived intel_npu_acceleration_library's "
        "`NPUModelForCausalLM`, and `NPU` names neither this project nor a "
        "single vendor. The HuggingFace convention for a backend is "
        "`<Project>ModelForCausalLM` -- `OVModelForCausalLM` (optimum-intel), "
        "`ORTModelForCausalLM`, `IPEXModelForCausalLM` -- with no `Auto` prefix"
    ),
    "delegate_": (
        "it was a second in-place module-replacement entry point beside "
        "`torchnative.quant.quantize_`, with a different shape, doing the same "
        "kind of thing (replacing leaves) in the same codebase"
    ),
    "delegated_paths": (
        "it only has a meaning for callers of `delegate_`, which is withdrawn"
    ),
    "DelegateModule": (
        "it is the extension point of the withdrawn `delegate_` front end. The "
        "class itself survives privately as `_DelegateModule`, because "
        "`torchnative.export.qnn` still builds on it internally"
    ),
    "replace_submodule": (
        "it is plumbing for the withdrawn `delegate_`, not something a user "
        "was ever meant to call"
    ),
    "resolve_submodule": (
        "it is plumbing for the withdrawn `delegate_`, not something a user "
        "was ever meant to call"
    ),
}


class DelegateRefused(RuntimeError):
    """A delegate declined, by name, to stand in for a submodule.

    Every refusal in this layer carries the backend's name and the reason,
    because the alternative -- falling back to the eager submodule silently --
    produces a model that is correct, slower than it looks, and indistinguishable
    from a working delegate by its output. docs/graph/NPU2.md is the record of that
    exact failure costing a round twice.
    """


class DelegateWithdrawn(DelegateRefused):
    """A user-facing name that this module used to export and no longer does.

    A subclass of `DelegateRefused` on purpose: everything that already caught
    a refusal from this layer keeps catching it, and the withdrawal reads as
    what it is -- a refusal that names itself and its reason, per CLAUDE.md §6.
    """


class _DelegateModule(nn.Module):
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
            f"{type(self).__name__} has no forward. A _DelegateModule subclass "
            "must implement one; inheriting this method means the artefact was "
            "never wired to anything."
        )

    def extra_repr(self):
        return f"backend={self.backend_name}"


def _resolve_submodule(model, path):
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


def _replace_submodule(model, path, new):
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
    parent = _resolve_submodule(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    leaf = parts[-1]
    old = _resolve_submodule(model, path)
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = new
    else:
        setattr(parent, leaf, new)
    return old


def _delegate_(model, plan):
    """Replace each submodule named in `plan` with its delegate. Returns `model`.

    `plan` maps a dotted path to a `_DelegateModule`. The trailing underscore is
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
            f"torchnative npu: plan must be a dict of path -> _DelegateModule, "
            f"got {type(plan).__name__}."
        )
    if not plan:
        raise DelegateRefused(
            "torchnative npu: empty plan. Replacing nothing and reporting "
            "success would make `_delegate_` indistinguishable from a no-op, "
            "which is exactly what a silent fallback looks like."
        )
    for path, new in plan.items():
        if not isinstance(new, _DelegateModule):
            raise DelegateRefused(
                f"torchnative npu: {path!r} maps to a "
                f"{type(new).__name__}, not a _DelegateModule. Only a "
                "_DelegateModule can name a back end when it refuses, and that "
                "naming is the entire contract of this layer."
            )
        _resolve_submodule(model, path)  # refuses by name before anything moves

    for path, new in plan.items():
        _replace_submodule(model, path, new)
    return model


def _delegated_paths(model):
    """Every path in `model` currently held by a `_DelegateModule`, sorted.

    This is the answer to "did the swap actually happen", read off the model
    rather than off the plan that was submitted. A plan is a request; this is
    the state.
    """
    return sorted(
        name
        for name, mod in model.named_modules()
        if isinstance(mod, _DelegateModule)
    )


# --------------------------------------------------------------------------
# Withdrawn. See `_WITHDRAWN` above for the per-name reason.
#
# The three faults were: a name copied from an archived vendor library rather
# than naming this project; a second module-replacement shape beside this
# repository's own `torchnative.quant.quantize_`; and a `from_pretrained` that
# returns a bare `nn.Module`, which leaves nothing for a `.to(device)` to mean
# (`nn.Module.to` moves parameters, and a graph-compiling accelerator cannot
# dispatch eager ops one at a time).
#
# The replacement is `torchnative.transformers.AutoModelForCausalLM`, and it
# now exists. What does NOT exist is the recompile step behind
# `model.to(torchnative.device.npu)`, which refuses by name after resolving --
# this file must not be read as saying that step exists.
# --------------------------------------------------------------------------

#: Named here rather than spelled out at each refusal, so the one place that
#: has to change when it lands is this line.
REPLACEMENT = "torchnative.transformers.AutoModelForCausalLM"


def _withdrawal_message(name):
    """The refusal text for a withdrawn `name`: what went, why, and what instead."""
    return (
        f"torchnative npu: {name} was withdrawn and is not available. It was "
        f"withdrawn because {_WITHDRAWN[name]}.\n"
        f"The replacement is {REPLACEMENT}, and it now EXISTS:\n"
        f"\n"
        f"    import torchnative\n"
        f"    from torchnative.transformers import AutoModelForCausalLM\n"
        f"    model = AutoModelForCausalLM.from_pretrained(model_id)\n"
        f"    model.to(torchnative.device.npu)\n"
        f"\n"
        f"What that gives you today, and what it does not:\n"
        f"\n"
        f"* `from_pretrained` returns the **real model** -- a genuine "
        f"nn.Module on which loss.backward() works. Not a wrapper. That is the "
        f"correction to this refusal's earlier wording, which said the "
        f"replacement would be `OVModel.to()`-shaped, a method on a wrapper "
        f"that recompiles. It is not: it is `nn.Module.to()` on one object, "
        f"which is what lets the model that trains and the model that runs on "
        f"the accelerator stay the same model.\n"
        f"* The device is `torchnative.device.npu`, a namespace this project "
        f"owns, and it RESOLVES per host -- Neural Engine on macOS, Intel NPU "
        f"on Windows, Hexagon on Android -- and says which. It is NOT "
        f"`torch.device(\"npu\")`: that spelling still raises on this shim "
        f"because `torch._C._rename_privateuse1_backend` is a stub, and it is "
        f"now a stub **on purpose**. PyTorch has no `npu` device type, and "
        f"making it appear to have one would be a claim about PyTorch that is "
        f"not true. See docs/devices/DEVICE_NS.md section 1.\n"
        f"* **Recompiling the model for the accelerator is still not "
        f"implemented.** `model.to(torchnative.device.npu)` resolves the NPU, "
        f"tells you which unit this host has, and then refuses by name at the "
        f"compile step rather than returning the model unchanged. So the "
        f"capability {name} was reaching for is still not available here -- "
        f"what changed is that the road to it is now real and named, and the "
        f"wall is at a different place. Saying that precisely is the point of "
        f"this refusal.\n"
        f"* `export=` and `load_in_4bit=` refuse by name on that "
        f"`from_pretrained`; they are not silently ignored."
    )


def __getattr__(name):
    """Refuse a withdrawn name by name, and leave every other miss alone.

    Refusing at *attribute access* rather than at call is deliberate: the
    withdrawn `NpuModelForCausalLM` was reached as
    `NpuModelForCausalLM.from_pretrained(...)`, so a callable stub would have
    to grow a fake `from_pretrained` to be reached at all. There is no class
    left; the attribute is where the truth is.
    """
    if name in _WITHDRAWN:
        raise DelegateWithdrawn(_withdrawal_message(name))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
