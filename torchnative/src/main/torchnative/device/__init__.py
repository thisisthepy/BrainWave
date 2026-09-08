"""`torchnative.device` --- a device namespace this project owns.

    import torchnative
    torchnative.device.cpu        eager
    torchnative.device.mps        eager
    torchnative.device.vulkan     eager
    torchnative.device.cuda       eager
    torchnative.device.npu        compiled target, resolved per host

**Why not `torch.device("npu")`.** PyTorch has no `npu` device type. Making it
appear to have one --- `_rename_privateuse1_backend`, a PrivateUse1
registration --- would be a claim *about PyTorch* that is not true, and the
label would then be accepted by every spelling that takes a device without any
of them being able to honour it. `docs/devices/DEVICE_ABS.md` section 7.3 records
`_rename_privateuse1_backend` as unimplemented and says the demand for it was
never measured; this module is the reason it stays that way. The namespace is
ours, so it can carry meaning ours can honour.

**Two kinds of device, and the type says which.**

`cpu`, `mps`, `vulkan` and `cuda` are `EagerDevice`: they dispatch operator by
operator, they are tensor destinations, and `.torch_device` hands back the
`torch.device` that names them. `npu` is a `CompiledDevice`: an NPU takes a
whole subgraph ahead of time and cannot be handed single operators
(CLAUDE.md section 8), so it is not a tensor destination and `.torch_device`
refuses by name. That is not a convention --- it is the absence of an
attribute, so the wrong use cannot be spelled.

**Availability is measured, and every answer says what measured it.**
`Availability.source` names the probe and `Availability.kind` is `"measured"`
or `"declared"`. Nothing here invents a probe: `_vulkan_probe`, `_cuda_probe`,
`intelnpu.npu_available`, `qnn_device.device_report` and CoreML's own
`MLComputeDevice.get_all_compute_devices` are the existing ones and this module
calls them.

**The one place that disagrees with its own probe, on purpose.**
`torch._C._mps_is_available()` is not a probe --- `bootstrap.py` installs it as
`_constant_function(..., False)`, justified by a comment saying candle's
`metal` feature is off in `Cargo.toml`. That comment is stale: `Cargo.toml`
enables `metal` for Apple targets, and on this host `torch.empty(2, 2,
device="mps")` succeeds and `a + a` on an `mps` tensor returns the right
numbers. So the constant is a **false negative**, and reporting it as
availability would say "no Metal" on a machine that is computing on Metal.

`mps.availability()` therefore reports `kind="measured"` from an actual
allocation, and carries the constant alongside as `detail["declared"]` with
`detail["declared_disagrees"]`. The constant is reused, as required; it is just
not allowed to be the answer. See `docs/devices/DEVICE_NS.md` section 3.
"""

import os
import platform
import sys

__all__ = [
    "Availability",
    "CompiledDevice",
    "Device",
    "DeviceUnavailable",
    "EagerDevice",
    "EagerUseRefused",
    "NpuResolution",
    "cpu",
    "cuda",
    "members",
    "mps",
    "npu",
    "report",
    "vulkan",
]


class DeviceUnavailable(RuntimeError):
    """This device is not usable in this process, and the message says why."""


class EagerUseRefused(TypeError):
    """A compiled target was used where a tensor destination was required."""


class NpuUnresolved(DeviceUnavailable):
    """`npu` does not resolve to any accelerator on this host."""


# --------------------------------------------------------------------------
# Availability records
# --------------------------------------------------------------------------


class Availability:
    """One availability answer, together with what produced it.

    `available` is the answer. `source` names the probe that gave it, as a
    string a reader can go and call themselves. `kind` is `"measured"` when
    the answer came from doing the thing, and `"declared"` when it came from a
    build-time constant --- a distinction this repository has been bitten by
    (CLAUDE.md section 4: "built" and "reached" are different claims).

    `reason` is `None` when available, and a **name** otherwise --- never
    prose alone. `detail` carries the probe's own payload unedited.
    """

    __slots__ = ("device", "available", "reason", "source", "kind", "detail")

    def __init__(self, device, available, source, kind, reason=None, detail=None):
        if kind not in ("measured", "declared"):
            raise ValueError(f"kind must be 'measured' or 'declared', got {kind!r}")
        if available and reason is not None:
            raise ValueError("an available device must not carry a refusal reason")
        if not available and not reason:
            raise ValueError(f"{device}: unavailable without a named reason")
        self.device = device
        self.available = bool(available)
        self.reason = reason
        self.source = source
        self.kind = kind
        self.detail = dict(detail or {})

    def as_dict(self):
        return {
            "device": self.device,
            "available": self.available,
            "reason": self.reason,
            "source": self.source,
            "kind": self.kind,
            "detail": self.detail,
        }

    def __repr__(self):
        state = "available" if self.available else f"unavailable ({self.reason})"
        return f"<Availability {self.device}: {state} via {self.source} [{self.kind}]>"


# --------------------------------------------------------------------------
# Host resolution
# --------------------------------------------------------------------------


def host():
    """Which host family this is, as one of a closed set of names.

    Separated out and overridable through `TORCHNATIVE_DEVICE_HOST` so the
    per-host resolution table can be exercised for hosts this machine is not.
    A resolution that returns the same thing regardless of host is a defect
    this module is specifically meant not to have, and a test cannot show that
    without being able to move the host.
    """
    forced = os.environ.get("TORCHNATIVE_DEVICE_HOST")
    if forced:
        return forced
    if hasattr(sys, "getandroidapilevel"):
        return "android"
    system = platform.system()
    return {
        "Darwin": "darwin",
        "Windows": "windows",
        "Linux": "linux",
    }.get(system, system.lower() or "unknown")


class NpuResolution:
    """Which accelerator `npu` turned out to mean here, and who says so.

    `docs/graph/NPU2.md` is why this type exists rather than a bare boolean.
    That round found three CoreML graphs recorded as "executed" which had run
    on the **CPU** --- every word of the original claim was true and none of it
    was the sentence "ran on the NPU". A device that cannot say which unit it
    resolved to reproduces that, so this object refuses to be constructed
    without a `unit` and a `source`.
    """

    __slots__ = ("host", "backend", "unit", "source", "detail")

    def __init__(self, host, backend, unit, source, detail=None):
        if not (backend and unit and source):
            raise ValueError("an npu resolution must name a backend, a unit and a source")
        self.host = host
        self.backend = backend
        self.unit = unit
        self.source = source
        self.detail = dict(detail or {})

    def as_dict(self):
        return {
            "host": self.host,
            "backend": self.backend,
            "unit": self.unit,
            "source": self.source,
            "detail": self.detail,
        }

    def __repr__(self):
        return (
            f"<NpuResolution {self.host}: {self.backend} -> {self.unit} "
            f"(via {self.source})>"
        )


# The per-host table. Kept as data, and read by `NpuDevice.resolve`, so that a
# host with no entry is a `KeyError` turned into a named refusal rather than a
# silent fallthrough to something that happens to be there.
NPU_BACKENDS = {
    "darwin": ("coreml", "Apple Neural Engine"),
    "windows": ("openvino", "Intel NPU"),
    "android": ("qnn", "Qualcomm Hexagon NPU"),
}


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------


class Device:
    """Base: a name, an availability answer, and nothing that guesses."""

    __slots__ = ("type",)

    #: `True` for devices that dispatch operator by operator.
    eager = None

    def __init__(self, type):
        self.type = type

    def availability(self):
        raise NotImplementedError

    @property
    def available(self):
        return self.availability().available

    def require(self):
        """Return self, or raise `DeviceUnavailable` naming the reason."""
        a = self.availability()
        if not a.available:
            raise DeviceUnavailable(
                f"torchnative.device.{self.type} is not available on this host "
                f"-- reason: {a.reason} (probe: {a.source})"
            )
        return self

    def report(self):
        return {"type": self.type, "eager": self.eager, **self.availability().as_dict()}

    def __repr__(self):
        kind = "eager" if self.eager else "compiled target"
        return f"torchnative.device.{self.type} ({kind})"


class EagerDevice(Device):
    """A tensor destination: it has a `torch.device` and takes single ops."""

    __slots__ = ()
    eager = True

    @property
    def torch_device(self):
        """The `torch.device` this names.

        Not a `torch.device` *subclass*: `torch._C.device` is not an
        acceptable base type (measured), so these objects cannot themselves be
        passed to `torch.empty(device=...)`. `.torch_device` is the conversion,
        and `nn.Module.to` takes the device object directly because this
        module teaches it to (see `_module_to.py`).
        """
        import torch

        return torch.device(self.type)


class CompiledDevice(Device):
    """A compiled target: it takes a whole graph and refuses single ops."""

    __slots__ = ()
    eager = False

    @property
    def torch_device(self):
        raise EagerUseRefused(
            f"torchnative.device.{self.type} is a compiled target, not a tensor "
            f"destination. An NPU is handed a whole subgraph ahead of time and "
            f"cannot dispatch a single operator, so there is no torch.device for "
            f"it and `torch.empty(..., device=torchnative.device.{self.type})` "
            f"refuses rather than half-working. Use "
            f"`model.to(torchnative.device.{self.type})`."
        )


class CpuDevice(EagerDevice):
    __slots__ = ()

    def availability(self):
        return Availability(
            "cpu",
            True,
            source="unconditional -- cpu is the device this interpreter is running on",
            kind="measured",
            detail={"host": host()},
        )


class MpsDevice(EagerDevice):
    """Metal. Measured by allocation, because the declared constant is wrong.

    See this module's docstring: `torch._C._mps_is_available()` is a
    build-time `False` whose justifying comment is stale.
    """

    __slots__ = ()

    def availability(self):
        import torch

        declared = None
        try:
            declared = bool(torch._C._mps_is_available())
        except Exception as exc:  # noqa: BLE001
            declared = None
            declared_error = f"{type(exc).__name__}: {exc}"
        else:
            declared_error = None

        detail = {
            "declared": declared,
            "declared_source": "torch._C._mps_is_available (a build-time constant)",
            "declared_error": declared_error,
            "host": host(),
        }

        if host() != "darwin":
            detail["declared_disagrees"] = False
            return Availability(
                "mps",
                False,
                source="platform.system() -- Metal is an Apple API",
                kind="measured",
                reason="not_apple",
                detail=detail,
            )
        try:
            t = torch.empty(2, 2, device="mps")
            measured = t.device.type == "mps"
            error = None
        except Exception as exc:  # noqa: BLE001
            measured = False
            error = f"{type(exc).__name__}: {exc}"
        detail["error"] = error
        detail["declared_disagrees"] = declared is not None and declared != measured
        return Availability(
            "mps",
            measured,
            source='torch.empty(2, 2, device="mps") -- an allocation, not a constant',
            kind="measured",
            reason=None if measured else "no_metal_backend",
            detail=detail,
        )


class VulkanDevice(EagerDevice):
    __slots__ = ()

    def availability(self):
        import torch

        probe = dict(torch._C._vulkan_probe())
        ok = bool(probe.get("available"))
        detail = dict(probe)
        detail["ops"] = len(torch._C._vulkan_ops())
        detail["host"] = host()
        return Availability(
            "vulkan",
            ok,
            source="torch._C._vulkan_probe",
            kind="measured",
            reason=None if ok else "no_loader",
            detail=detail,
        )


class CudaDevice(EagerDevice):
    """CUDA, keeping the five named reasons `_cuda_probe` already produces.

    This class adds no reason of its own and collapses none of them: the
    `reason` field is passed through exactly as `classify_cuda_refusal`
    produced it, so `not_built` stays distinguishable from `no_driver`.
    """

    __slots__ = ()

    def availability(self):
        import torch

        probe = dict(torch._C._cuda_probe())
        ok = bool(probe.get("available"))
        detail = dict(probe)
        detail["host"] = host()
        return Availability(
            "cuda",
            ok,
            source="torch._C._cuda_probe",
            kind="measured",
            reason=None if ok else (probe.get("reason") or "unclassified"),
            detail=detail,
        )


class NpuDevice(CompiledDevice):
    """The compiled target, resolved per host and never silently to the CPU."""

    __slots__ = ()

    def resolve(self):
        """Which accelerator `npu` means here. Raises `NpuUnresolved` by name.

        Never returns a CPU resolution. `docs/graph/NPU2.md`'s partial offload
        is what that rule is for: a device that answers "yes" while the work
        runs somewhere else is the failure this whole namespace exists to make
        unspellable.
        """
        h = host()
        if h not in NPU_BACKENDS:
            raise NpuUnresolved(
                f"torchnative.device.npu does not resolve on host {h!r}: this "
                f"project knows an NPU path for "
                f"{', '.join(sorted(NPU_BACKENDS))} and no other. It will not "
                f"fall back to the CPU -- an npu that silently means cpu is "
                f"docs/graph/NPU2.md's partial offload again."
            )
        backend, unit = NPU_BACKENDS[h]
        return getattr(self, f"_resolve_{backend}")(h, backend, unit)

    # -- per-host resolvers, each ending in a real probe -------------------

    def _resolve_coreml(self, h, backend, unit):
        source = (
            "coremltools.models.compute_device.MLComputeDevice"
            ".get_all_compute_devices"
        )
        try:
            from coremltools.models.compute_device import MLComputeDevice
        except Exception as exc:  # noqa: BLE001
            raise NpuUnresolved(
                f"torchnative.device.npu resolves to the {unit} on {h}, but "
                f"coremltools is not importable, so nothing here can say "
                f"whether that unit is present: {type(exc).__name__}: {exc}"
            ) from None
        names = [type(d).__name__ for d in MLComputeDevice.get_all_compute_devices()]
        present = [n for n in names if "NeuralEngine" in n]
        if not present:
            raise NpuUnresolved(
                f"torchnative.device.npu resolves to the {unit} on {h}, and "
                f"{source} does not list one -- it lists {names}. Refusing by "
                f"name rather than falling back to the CPU."
            )
        return NpuResolution(
            h, backend, unit, source, {"compute_devices": names, "matched": present}
        )

    def _resolve_openvino(self, h, backend, unit):
        source = "torchnative.export.intelnpu.npu_available"
        try:
            from torchnative.export.intelnpu import available_devices, npu_available
        except Exception as exc:  # noqa: BLE001
            raise NpuUnresolved(
                f"torchnative.device.npu resolves to the {unit} on {h}, but "
                f"torchnative.export.intelnpu is not importable: "
                f"{type(exc).__name__}: {exc}"
            ) from None
        try:
            devices = list(available_devices())
            ok = npu_available()
        except Exception as exc:  # noqa: BLE001
            raise NpuUnresolved(
                f"torchnative.device.npu resolves to the {unit} on {h}, and the "
                f"OpenVINO runtime did not answer: {type(exc).__name__}: {exc}. "
                f"The Intel path refuses without the OpenVINO runtime and this "
                f"does not weaken that."
            ) from None
        if not ok:
            raise NpuUnresolved(
                f"torchnative.device.npu resolves to the {unit} on {h}, and "
                f"OpenVINO lists {devices} with no NPU among them."
            )
        return NpuResolution(h, backend, unit, source, {"devices": devices})

    def _resolve_qnn(self, h, backend, unit):
        source = "torchnative.export.qnn_device.device_report"
        try:
            from torchnative.export.qnn_device import device_report
        except Exception as exc:  # noqa: BLE001
            raise NpuUnresolved(
                f"torchnative.device.npu resolves to the {unit} on {h}, but "
                f"torchnative.export.qnn_device is not importable: "
                f"{type(exc).__name__}: {exc}"
            ) from None
        report = dict(device_report())
        if not report.get("reachable"):
            raise NpuUnresolved(
                f"torchnative.device.npu resolves to the {unit} on {h}, and the "
                f"device is not reachable: {report.get('reason')}"
            )
        if not report.get("htp_arch"):
            raise NpuUnresolved(
                f"torchnative.device.npu resolves to the {unit} on {h}; the "
                f"device is reachable but reports no HTP architecture, so there "
                f"is no Hexagon NPU to target on it."
            )
        # `htp_arch` alone is not enough, and this gate is here because for a
        # while it was the only gate. It comes from `ro.soc.model` mapped
        # through ExecuTorch's chipset table -- it is a *name*. Measured on a
        # Galaxy Tab S9 Ultra: `ro.soc.model=SM8550` yields `htp_arch=73` and
        # this function returned a confident "Qualcomm Hexagon NPU", on a device
        # whose `/sys/class/fastrpc` registers no compute-DSP endpoint at all
        # and on which not one QNN runtime library is present. That is the
        # `_mps_is_available` shape inverted: a hardcoded yes instead of a
        # hardcoded no, and it would have said the same thing about any device
        # that merely *calls itself* an SM8550.
        if not report.get("htp_reachable"):
            raise NpuUnresolved(
                f"torchnative.device.npu resolves to the {unit} on {h}, and the "
                f"device names an SoC whose datasheet has a V{report['htp_arch']} "
                f"HTP -- but that is the part number, not a probe. "
                f"{report.get('htp_unreachable_reason')} Refusing by name rather "
                f"than reporting a Hexagon NPU that nothing here has reached."
            )
        return NpuResolution(h, backend, unit, source, report)

    # -- availability is resolution, caught ---------------------------------

    def availability(self):
        h = host()
        try:
            res = self.resolve()
        except NpuUnresolved as exc:
            backend = NPU_BACKENDS.get(h, (None, None))[0]
            return Availability(
                "npu",
                False,
                source=f"torchnative.device.npu.resolve ({backend or 'no backend for host'})",
                kind="measured",
                reason="unresolved",
                detail={"host": h, "error": str(exc), "backend": backend},
            )
        return Availability(
            "npu",
            True,
            source=res.source,
            kind="measured",
            detail={"host": h, "resolution": res.as_dict()},
        )

    def report(self):
        out = super().report()
        try:
            out["resolution"] = self.resolve().as_dict()
        except NpuUnresolved as exc:
            out["resolution"] = None
            out["resolution_error"] = str(exc)
        return out


# --------------------------------------------------------------------------
# The namespace itself
# --------------------------------------------------------------------------

cpu = CpuDevice("cpu")
mps = MpsDevice("mps")
vulkan = VulkanDevice("vulkan")
cuda = CudaDevice("cuda")
npu = NpuDevice("npu")


def members():
    """Every device in this namespace, in a stable order."""
    return (cpu, mps, vulkan, cuda, npu)


def is_device(obj):
    return isinstance(obj, Device)


def report():
    """Every device's answer, for a bug report or a docstring."""
    return {d.type: d.report() for d in members()}


# Teaching `nn.Module.to` about these objects is what makes them mean
# anything, and it must happen before anyone can call `to()` with one. Since
# a torchnative device cannot exist without importing this module, installing
# it here is sufficient: there is no path that reaches `to(a torchnative
# device)` without first running this line.
from ._module_to import install as _install_module_to  # noqa: E402

_install_module_to()
