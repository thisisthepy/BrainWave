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


def _to_compiled(self, device, args, kwargs):
    """A compiled target: resolve first, then say exactly what is missing.

    Resolution happens **before** the refusal on purpose. The caller learns
    which NPU this host actually has --- which is the question
    `docs/graph/NPU2.md` says a device must be able to answer --- rather than
    a flat "not implemented" that tells them nothing about their machine.
    """
    extra = [a for a in args if a is not device]
    if extra or kwargs:
        raise TypeError(
            f"nn.Module.to(torchnative.device.{device.type}) takes no other "
            f"arguments: a compiled target is not a dtype or memory-format "
            f"conversion. Got extra {extra!r} {kwargs!r}."
        )

    resolution = device.resolve()  # raises NpuUnresolved, by name, if it cannot

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
        f"What does exist today: the capture layer "
        f"(torchnative.export.decompose / refold) and the per-vendor "
        f"execution-device evidence (torchnative.export.intelnpu.probe, "
        f"assert_execution_device, verdict_execution_devices). What is "
        f"missing is the step that turns a captured graph into a leaf this "
        f"module can carry. See docs/devices/DEVICE_NS.md section 5."
    )


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
