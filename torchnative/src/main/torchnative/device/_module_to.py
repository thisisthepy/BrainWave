"""Teach `nn.Module.to` what a torchnative device means.

**Where the interception has to be, and why it cannot be anywhere else.**
Upstream's `nn.Module.to` (`torch/nn/modules/module.py`, the `to` at line
1254) begins:

    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(...)
    ...
    def convert(t): ... return t.to(device, ...)
    return self._apply(convert)

Two consequences, both measured rather than assumed:

* `_parse_to` is the **first statement**, and it only knows `torch.device`
  spellings. A torchnative device reaching it is a `TypeError` about argument
  combinations, which is a refusal but not an interception.
* `_apply(convert)` descends to **tensors** --- `convert(t)` calls
  `t.to(device, ...)`. A compiled target is not a tensor destination at all,
  so there is nothing for `convert` to do with one.

So the interception is a wrapper *around* `Module.to`, before `_parse_to`, and
it is the only place it can be.

**The rule that keeps upstream intact.** For any call that contains no
torchnative device, this wrapper's entire body is

    return _original(self, *args, **kwargs)

--- the same object, the same arguments, unpacked and repacked by nothing. It
cannot change `to("cpu")`, `to(torch.float16)`, `to(device, dtype)`,
`to(other_tensor)`, `non_blocking=` or `memory_format=`, because it does not
look at them. `test_devicens.py` proves it against upstream's own semantics
for every ordinary argument form rather than trusting that sentence.

**Why here and not `bootstrap.py`.** `bootstrap.py` is baked into `torch._C`
and runs while `torch/__init__.py` is on its first lines --- `torch.nn` does
not exist yet, so there is nothing to patch. This module is imported by
`torchnative.device`, and a torchnative device object cannot exist without
that import, so there is no call site that can reach `to(a torchnative
device)` with the patch not installed. That is a stronger guarantee than an
import hook, not a weaker one, and it needs no `sys.meta_path` entry.

**No wrapping, ever.** `to()` returns `self`, the same `nn.Module`, exactly as
upstream does. `optimum` returns an inference object and that is why it cannot
backprop; this project ships its own `torch`, so it does not have to. If this
file ever returns something that is not `self`, the thing that distinguishes
this project from `optimum` is gone.

That holds for the compiled targets too. `to(torchnative.device.npu)` on an
Intel NPU lowers the model's `torch.nn.Linear` leaves **in place** --- the
object that comes back is the object that went in, and `generate()`,
`state_dict()` and `backward()` all still work because nothing was wrapped.
The lowering's partial-offload report is carried out on
`model.torchnative_offload`, and as a `UserWarning` when anything stayed on the
CPU; `_to_compiled` explains why both. See section 5.5 of
`docs/devices/DEVICE_NS.md`.
"""

import functools

_INSTALLED = False


def _scan(args, kwargs):
    """Find torchnative devices among the arguments.

    Returns `(device, position)` where `position` is an `int` index into
    `args` or a `str` key of `kwargs`, or `(None, None)`. Raises if more than
    one is present --- two accelerators in one `to()` has no meaning and
    guessing which wins is how an argument gets silently dropped.
    """
    from . import Device

    found = []
    for i, a in enumerate(args):
        if isinstance(a, Device):
            found.append((a, i))
    for k, v in kwargs.items():
        if isinstance(v, Device):
            found.append((v, k))
    if not found:
        return None, None
    if len(found) > 1:
        names = ", ".join(f"torchnative.device.{d.type}" for d, _ in found)
        raise TypeError(
            f"nn.Module.to: more than one torchnative device given ({names}). "
            f"A module has one device; refusing rather than picking one."
        )
    return found[0]


def _to_eager(original, self, device, position, args, kwargs):
    """An eager torchnative device is upstream's `to`, with the label swapped.

    `device.torch_device` is a real `torch.device`, so everything after this
    point is upstream's own code path --- parameters move, `_apply` descends
    to tensors, and the return value is `self`. The only thing this function
    does is spell the device in the vocabulary `_parse_to` speaks.
    """
    device.require()
    if isinstance(position, int):
        args = args[:position] + (device.torch_device,) + args[position + 1 :]
    else:
        kwargs = dict(kwargs)
        kwargs[position] = device.torch_device
    return original(self, *args, **kwargs)


#: The compile step's own options, accepted by `to(device.npu, ...)` and
#: passed through to `_compile_model`. Named rather than forwarded blindly:
#: a typo'd kwarg must still be refused, not silently ignored.
#:
#: `precision` is **CoreML's and only CoreML's**, which is why the openvino arm
#: refuses it by name rather than ignoring it. On that backend the IR is f16
#: and there is nothing for the word to select; accepting it there would be an
#: argument taken and dropped, which is how a mode goes silent.
_COMPILE_OPTIONS = ("progress", "eager", "precision")


def _to_compiled(self, device, args, kwargs):
    """A compiled target: resolve, then either lower for it or refuse by name.

    Resolution happens **first, always**, whichever branch follows. The caller
    learns which NPU this host actually has --- the question
    `docs/graph/NPU2.md` says a device must be able to answer --- rather than a
    flat verdict that tells them nothing about their machine. The dispatch key
    below is `resolution.backend`, **not** `host()`: the host offers an ordered
    list of candidate backends (`NPU_CANDIDATES`) and a probe picks one of them,
    and the backend is what has or has not been wired --- so keying on the host
    would be reading the wrong fact one step early, and now also the wrong
    *number* of facts, since Windows offers two (docs/devices/NPUVENDOR.md).

    **`openvino` (Intel NPU) is wired.** It goes to
    `torchnative.export.intelnpu._compile_model`, which walks `named_children()`
    and swaps each eligible `torch.nn.Linear` for a leaf whose forward runs on
    the OpenVINO device. That call is **in place** and returns the same object,
    so what comes back here is `self`: still an `nn.Module`, still with its
    `parameters()`, `state_dict()` and `named_children()`, so `generate()`
    keeps working and does not learn anything about the NPU. This module's
    docstring says "no wrapping, ever", and lowering does not break that
    promise --- it is the reason the mechanism was chosen over an inference
    object in the first place.

    No library path is passed. The OpenVINO runtime is discovered from the pip
    package (`pip install torchnative[npu]`), and a path argument here would put
    a filename back into the user's path for no gain.

    **Where the partial-offload report goes, and why there.** `_compile_model`
    returns `(model, report)`; the report names every leaf left behind, because
    "the model is on the NPU" is false for any model with a `LayerNorm` in it.
    That report must not stop here. It is delivered **twice**, deliberately:

    * as `model.torchnative_offload`, a plain dict, for a caller who asks. An
      attribute rather than a return value because the return value is fixed:
      `to()` returns `self` and upstream's contract is not negotiable. An
      attribute rather than a log line alone because a caller who wants to
      assert on the offload needs a value --- `fraction_moved` is a number for
      exactly that reason.
    * as a `UserWarning` when `fully_offloaded` is False, for a caller who does
      **not** ask. This is the load-bearing half. docs/graph/NPU2.md is a whole
      document about a partial offload that went unnoticed *because the answers
      were right*; an attribute nobody reads reproduces it exactly. Silence is
      the defect, so the only silent case is the complete one --- which also
      keeps the warning meaningful when it does fire.

    **Zero leaves lowered is a refusal, not a success.** That refusal is
    `_compile_model`'s own `IntelNPUUnsupported` and it is allowed to propagate
    unchanged: returning an untouched model with a success message is the
    silent CPU fallback this whole path exists to prevent. Nothing is attached
    to the model in that case either --- a report on a model that was never
    lowered is the same lie with a receipt.

    **`coreml` (Apple Neural Engine) is wired too, and it takes a
    `precision`.** It goes to `torchnative.export.coreml._compile_model`, which
    does the same `named_children()` walk and swaps each `torch.nn.Linear` for
    a `_CoreMLLinear`. Same mechanism, same in-place contract, same report.

    What is **not** the same is that CoreML's precision is a real choice with a
    measured cost on each side, so this backend takes two spellings:

        model.to(device.npu)                        # float16
        model.to(device.npu, precision="float32")   # float32

    float16 is what reaches the Neural Engine -- docs/graph/NPU2.md §1.1
    measured that for a float32 program the unit is not in CoreML's *supported*
    column at all, so no `compute_units` setting reaches it -- and it agrees
    with `DecomposedTrace.replay` to about 1e-03. float32 agrees at 2e-05, the
    bar the word *agrees* means here, and runs on the CPU or GPU. They are two
    products and therefore two spellings; a single spelling that silently chose
    would be the defect docs/graph/NPU2.md is about, one layer up.

    Which unit **actually** ran is not inferred from either: `MLComputePlan` is
    read at every compile and the per-operation rows are on the report. The
    case that cannot reach the named unit warns at `to()`; the case that could
    and did not warns at the forward that found out.

    **`qnn` still refuses.** It is not stubbed into a fake success. Its refusal
    is the same `NotImplementedError`, after the same real resolution, at the
    same quality of message it had before.
    """
    # `progress` and `eager` are the compile step's own options, and they reach
    # it ONLY through here. `_compile_model` grew both, and nothing plumbed
    # them, so `to(device.npu, progress=...)` raised TypeError while the
    # feature sat there unreachable -- the gate stayed green because the tests
    # called `_compile_model` directly. A public path that cannot reach a
    # feature is the same as not having it.
    #
    # A dtype or memory format is still refused, because a compiled target is
    # not a conversion. What is accepted is exactly the compile step's own
    # arguments, named.
    options = {}
    for name in _COMPILE_OPTIONS:
        if name in kwargs:
            options[name] = kwargs.pop(name)

    extra = [a for a in args if a is not device]
    if extra or kwargs:
        raise TypeError(
            f"nn.Module.to(torchnative.device.{device.type}) takes no other "
            f"arguments: a compiled target is not a dtype or memory-format "
            f"conversion. It does take the compile step's own options "
            f"({', '.join(_COMPILE_OPTIONS)}). Got extra {extra!r} {kwargs!r}."
        )

    resolution = device.resolve()  # raises NpuUnresolved, by name, if it cannot

    if resolution.backend == "openvino":
        if "precision" in options:
            raise TypeError(
                f"nn.Module.to(torchnative.device.{device.type}, precision=...): "
                f"`precision` is the CoreML arm's argument and the openvino "
                f"backend has no use for it -- its IR is f16 and there is "
                f"nothing for the word to select. Refusing rather than "
                f"accepting and dropping it."
            )
        return _lower_for_openvino(self, device, resolution, **options)

    if resolution.backend == "coreml":
        return _lower_for_coreml(self, device, resolution, **options)

    raise NotImplementedError(
        f"nn.Module.to(torchnative.device.{device.type}): this host's "
        f"{device.type} resolved to the {resolution.unit} via the "
        f"{resolution.backend} backend (probe: {resolution.source}), and "
        f"recompiling an nn.Module for it is NOT implemented in this build.\n"
        f"\n"
        f"This refuses rather than returning the model unchanged. Returning "
        f"`self` here would be an argument accepted and dropped: the caller "
        f"would hold a model they believe is on the {resolution.unit} and "
        f"which is in fact running on the CPU -- docs/graph/NPU2.md section 1 "
        f"is that exact failure, found only by reading MLComputePlan.\n"
        f"\n"
        f"The Intel NPU path (the openvino backend) IS wired: it lowers "
        f"torch.nn.Linear leaves through "
        f"torchnative.export.intelnpu. What the {resolution.backend} backend "
        f"still lacks is the equivalent leaf. What does exist for it today: "
        f"the capture layer (torchnative.export.decompose / refold) and the "
        f"per-vendor execution-device evidence (probe, "
        f"assert_execution_device, verdict_execution_devices). See "
        f"docs/devices/DEVICE_NS.md section 5."
    )


def _lower_for_openvino(model, device, resolution, **options):
    """Lower `model` for the Intel NPU and hand back the same `nn.Module`.

    Separated from `_to_compiled` so that the dispatch --- which backend, and
    what happens to everything that is not it --- reads as five lines, and so
    that a test can point at the branch by name. See `_to_compiled` for why the
    report is delivered both as an attribute and as a warning.
    """
    import warnings

    from ..export import intelnpu

    model, report = intelnpu._compile_model(model, device="NPU", **options)
    # Attached only on success. `_compile_model` raises for zero leaves, so
    # this line is unreachable for a model that was not actually lowered.
    model.torchnative_offload = report

    if not report["fully_offloaded"]:
        left = ", ".join(
            f"{name} x{count}" for name, count in report["left_on_cpu"].items()
        ) or "none"
        skipped = "; ".join(f"{name}: {why}" for name, why in report["skipped"][:4])
        warnings.warn(
            f"nn.Module.to(torchnative.device.{device.type}): a PARTIAL offload. "
            f"{len(report['swapped'])} Linear(s) now run on the {resolution.unit}, "
            f"which is fraction_moved="
            f"{report['fraction_moved']:.4f} "
            f"({report['parameters_moved']} of {report['parameters_total']} "
            f"parameters). Left on the CPU -- leaf module types: {left}."
            + (f" Skipped Linear(s): {skipped}." if skipped else "")
            + f" The full report is on the model as `.torchnative_offload`. "
            f"This warning exists because docs/graph/NPU2.md is about a partial "
            f"offload that went unnoticed while every answer it produced was "
            f"right.",
            UserWarning,
            stacklevel=4,
        )
    return model


def _lower_for_coreml(model, device, resolution, *, precision="float16",
                      **options):
    """Lower `model` for CoreML and hand back the same `nn.Module`.

    The Intel arm's shape, with one addition it does not need: `precision`.
    See `_to_compiled` for why that is a spelling and not a mode, and
    `torchnative.export.coreml`'s lowering section for the measurement.

    The partial-offload warning is the Intel arm's, word for word in intent: a
    complete offload is silent, so the warning stays worth reading. The two
    *unit* warnings -- "this precision cannot reach the Neural Engine" and "it
    could and CoreML preferred something else" -- are raised inside
    `export.coreml`, because only that layer has read the compute plan.
    """
    import warnings

    from ..export import coreml

    model, report = coreml._compile_model(model, precision=precision, **options)
    # Attached only on success. `_compile_model` raises for zero leaves, so
    # this line is unreachable for a model that was not actually lowered.
    model.torchnative_offload = report

    if not report["fully_offloaded"]:
        left = ", ".join(
            f"{name} x{count}" for name, count in report["left_on_cpu"].items()
        ) or "none"
        skipped = "; ".join(f"{name}: {why}" for name, why in report["skipped"][:4])
        warnings.warn(
            f"nn.Module.to(torchnative.device.{device.type}): a PARTIAL offload. "
            f"{len(report['swapped'])} Linear(s) now go through CoreML at "
            f"precision={report['precision']!r}, which is fraction_moved="
            f"{report['fraction_moved']:.4f} "
            f"({report['parameters_moved']} of {report['parameters_total']} "
            f"parameters). Left on the CPU -- leaf module types: {left}."
            + (f" Skipped Linear(s): {skipped}." if skipped else "")
            + f" The full report, including the per-operation MLComputePlan "
            f"rows, is on the model as `.torchnative_offload`. "
            f"This warning exists because docs/graph/NPU2.md is about a partial "
            f"offload that went unnoticed while every answer it produced was "
            f"right.",
            UserWarning,
            stacklevel=4,
        )
    return model


def make(original):
    """Build the wrapper around `original`.

    Factored out of `install` so a test can wrap a **spy** and assert that an
    ordinary call reaches upstream with byte-identical arguments. That is the
    invariant this file claims, and comparing observable module state cannot
    check it: `non_blocking=` has no effect on a CPU-to-CPU copy, so a wrapper
    that silently dropped it produced identical parameters and passed the
    differential test. It was found by nullification (N5) and this is the fix.
    """

    @functools.wraps(original)
    def to(self, *args, **kwargs):
        device, position = _scan(args, kwargs)
        if device is None:
            # The whole of upstream's behaviour, untouched and unexamined.
            return original(self, *args, **kwargs)
        if device.eager:
            return _to_eager(original, self, device, position, args, kwargs)
        return _to_compiled(self, device, args, kwargs)

    to._torchnative_patched = True
    to._torchnative_original = original
    return to


def install():
    """Install the wrapper. Idempotent; returns True if it did the work."""
    global _INSTALLED
    import torch.nn as nn

    if getattr(nn.Module.to, "_torchnative_patched", False):
        _INSTALLED = True
        return False

    original = nn.Module.to
    to = make(original)
    nn.Module.to = to
    _INSTALLED = True
    return True


def installed():
    return _INSTALLED


def original():
    """Upstream's `to`, for a test that wants to compare against it."""
    import torch.nn as nn

    return getattr(nn.Module.to, "_torchnative_original", None)
