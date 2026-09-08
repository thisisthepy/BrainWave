"""`torchnative.device`: the namespace, its probes, and the `nn.Module.to` patch.

See docs/devices/DEVICE_NS.md.

**What is checkable here and what is not.** This is an arm64 Mac. It has a
Metal GPU, a Neural Engine, a Vulkan loader when the emulator's one is supplied,
and no CUDA and no Intel NPU. So:

* The **per-host npu resolution table** is exercised for every host, because
  `torchnative.device.host()` reads `TORCHNATIVE_DEVICE_HOST` and the tests
  move it. Without that a resolution that ignored the host entirely would pass
  --- it would answer "Apple Neural Engine" here and never be asked anything
  else. `test_npu_resolves_differently_per_host` is the one that fails if
  resolution is hardcoded.
* The **availability probes** are checked against an independent measurement
  rather than against themselves. `test_mps_availability_is_measured_not_declared`
  is the load-bearing one: it allocates on Metal itself and requires the
  namespace's answer to match *that*, not to match
  `torch._C._mps_is_available()` --- which is a build-time constant and, on
  this host, a false negative (see section 3 of the document).
* The **`to()` patch** is checked differentially: every ordinary argument form
  is run through the patched `nn.Module.to` and through the unpatched original
  captured at install time, and the resulting parameter devices and dtypes must
  agree. A tautology here would be checking the patch against itself.

Nullifications this file is meant to catch, each verified by making the break
and watching it go red:

* availability answered from a constant  -> `test_mps_availability_is_measured_not_declared`
* resolution ignoring the host           -> `test_npu_resolves_differently_per_host`
* npu falling back to the cpu            -> `test_npu_never_resolves_to_the_cpu`
* `to()` changing upstream behaviour     -> `test_to_is_unchanged_for_every_ordinary_argument_form`
* an eager op accepted by a compiled target -> `test_npu_refuses_to_be_a_tensor_destination`
* cuda's five reasons collapsed          -> `test_cuda_keeps_its_five_named_reasons`
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
sys.path.insert(0, _VENDOR_DIR)

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import torchnative.device as D  # noqa: E402
from torchnative.device import _module_to  # noqa: E402


def _shim():
    """CLAUDE.md section 3: is this the shim or did we silently get upstream?"""
    assert hasattr(torch._C, "_aten_implemented"), (
        "not the torchnative shim -- torch._C has no _aten_implemented, so the "
        "vendored tree is missing and every probe below would measure upstream"
    )


# --------------------------------------------------------------------------
# The namespace
# --------------------------------------------------------------------------


def test_the_namespace_has_the_decided_members():
    _shim()
    names = [d.type for d in D.members()]
    assert names == ["cpu", "mps", "vulkan", "cuda", "npu"], names
    for name in names:
        assert isinstance(getattr(D, name), D.Device)
    print(f"ok   devicens: torchnative.device exposes {names}")


def test_eager_and_compiled_are_distinguished_by_type_not_by_convention():
    _shim()
    for d in (D.cpu, D.mps, D.vulkan, D.cuda):
        assert isinstance(d, D.EagerDevice), d
        assert d.eager is True, d
    assert isinstance(D.npu, D.CompiledDevice)
    assert D.npu.eager is False
    assert not isinstance(D.npu, D.EagerDevice), (
        "npu must not be an EagerDevice -- the distinction is the type, so that "
        "the wrong use cannot be spelled rather than merely being discouraged"
    )
    print("ok   devicens: eager/compiled is the class, not a flag on one class")


def test_npu_refuses_to_be_a_tensor_destination():
    """An NPU takes a subgraph, not single ops, so it is not a destination."""
    _shim()
    for d in (D.cpu, D.mps, D.vulkan, D.cuda):
        assert isinstance(d.torch_device, torch.device), d

    try:
        D.npu.torch_device
    except D.EagerUseRefused as exc:
        assert "compiled target" in str(exc), exc
        assert "npu" in str(exc), exc
    else:
        raise AssertionError("npu.torch_device must refuse by name")

    # And the eager spelling the README says has to refuse, does.
    try:
        torch.empty(2, 2, device=D.npu)
    except TypeError as exc:
        assert "NpuDevice" in str(exc) or "device" in str(exc), exc
    else:
        raise AssertionError("torch.empty(device=npu) must refuse, not half-work")
    print(
        "ok   devicens: npu has no torch_device and torch.empty(device=npu) refuses; "
        "the four eager devices each have one"
    )


# --------------------------------------------------------------------------
# Availability is measured
# --------------------------------------------------------------------------


def test_every_availability_answer_names_the_probe_that_produced_it():
    _shim()
    for d in D.members():
        a = d.availability()
        assert a.source, f"{d.type}: availability with no named source"
        assert a.kind in ("measured", "declared"), a
        if not a.available:
            assert a.reason, f"{d.type}: unavailable with no named reason"
    print(
        "ok   devicens: all "
        f"{len(D.members())} devices name their probe and their refusal reason"
    )


def test_mps_availability_is_measured_not_declared():
    """The one that fails if availability is read off the build-time constant.

    `torch._C._mps_is_available()` is `_constant_function(..., False)` in
    bootstrap.py. On this host Metal computes. So an implementation that
    returned the constant would answer `False` here while the independent
    allocation below succeeds, and this test would go red -- which is the
    point. It compares the namespace against a *measurement*, never against
    the constant.
    """
    _shim()
    a = D.mps.availability()
    assert a.kind == "measured", a

    # The independent measurement, done here and not by the module under test.
    try:
        t = torch.empty(2, 2, device="mps")
        really = t.device.type == "mps"
    except Exception:  # noqa: BLE001
        really = False

    assert a.available == really, (
        f"torchnative.device.mps says available={a.available} but an allocation "
        f"on mps says {really}. Availability must come from the measurement."
    )
    declared = a.detail.get("declared")
    if declared is not None and declared != really:
        assert a.detail["declared_disagrees"] is True, (
            "the declared constant disagrees with the measurement and the "
            "namespace did not say so"
        )
    print(
        f"ok   devicens: mps available={a.available} by allocation; "
        f"torch._C._mps_is_available()={declared} "
        f"(disagrees={a.detail.get('declared_disagrees')})"
    )


def test_mps_availability_would_notice_metal_disappearing():
    """Availability must track the machine, not a literal.

    Moving the host away from darwin must make mps unavailable. A constant
    `True` passes every other test in this file and fails this one.
    """
    _shim()
    old = os.environ.get("TORCHNATIVE_DEVICE_HOST")
    os.environ["TORCHNATIVE_DEVICE_HOST"] = "linux"
    try:
        a = D.mps.availability()
        assert a.available is False, "mps cannot be available on a non-Apple host"
        assert a.reason == "not_apple", a
    finally:
        if old is None:
            os.environ.pop("TORCHNATIVE_DEVICE_HOST", None)
        else:
            os.environ["TORCHNATIVE_DEVICE_HOST"] = old
    print("ok   devicens: mps reports unavailable ('not_apple') off an Apple host")


def test_vulkan_answer_comes_from_the_existing_probe():
    _shim()
    a = D.vulkan.availability()
    probe = dict(torch._C._vulkan_probe())
    assert a.source == "torch._C._vulkan_probe", a
    assert a.available == bool(probe.get("available")), (a, probe)
    if not a.available:
        assert a.reason == "no_loader", a
        assert a.detail.get("error"), "vulkan refused without saying why"
    print(
        f"ok   devicens: vulkan available={a.available} straight from "
        f"_vulkan_probe (ops={a.detail.get('ops')})"
    )


def test_cuda_keeps_its_five_named_reasons():
    """Do not weaken an existing refusal: the five names must survive."""
    _shim()
    reasons = list(torch._C._shim_cuda_refusal_reasons())
    assert set(reasons) == {
        "not_built",
        "no_driver",
        "no_device",
        "wrong_arch",
        "unclassified",
    }, reasons
    a = D.cuda.availability()
    assert a.source == "torch._C._cuda_probe", a
    if not a.available:
        assert a.reason in reasons, (
            f"cuda refused with {a.reason!r}, which is not one of the five names "
            f"_cuda_probe produces -- the namespace has collapsed them"
        )
        assert a.reason == dict(torch._C._cuda_probe()).get("reason"), (
            "the namespace's cuda reason must be _cuda_probe's own, unedited"
        )
    print(f"ok   devicens: cuda reason={a.reason!r}, one of the five {reasons}")


# --------------------------------------------------------------------------
# npu resolution is per host and never silently the cpu
# --------------------------------------------------------------------------


def _with_host(value, fn):
    old = os.environ.get("TORCHNATIVE_DEVICE_HOST")
    os.environ["TORCHNATIVE_DEVICE_HOST"] = value
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("TORCHNATIVE_DEVICE_HOST", None)
        else:
            os.environ["TORCHNATIVE_DEVICE_HOST"] = old


def test_npu_resolves_differently_per_host():
    """The test that fails if `npu` means the same thing everywhere.

    Resolution is allowed to *refuse* on a host whose hardware is not here ---
    that is the honest answer for Windows and Android from a Mac. What it is
    not allowed to do is give the same backend for all three. So this checks
    the backend named in the resolution or in the refusal, which is available
    either way.
    """
    _shim()
    seen = {}
    for host, expected_backend, expected_unit in (
        ("darwin", "coreml", "Apple Neural Engine"),
        ("windows", "openvino", "Intel NPU"),
        ("android", "qnn", "Qualcomm Hexagon NPU"),
    ):
        def probe(host=host):
            try:
                res = D.npu.resolve()
                return res.backend, res.unit, None
            except D.NpuUnresolved as exc:
                return None, None, str(exc)

        backend, unit, err = _with_host(host, probe)
        if backend is None:
            # A refusal must still name the unit it was looking for.
            assert expected_unit in err, (
                f"host {host}: refused without naming the {expected_unit}: {err}"
            )
            seen[host] = f"refused({expected_unit})"
        else:
            assert backend == expected_backend, (host, backend)
            assert unit == expected_unit, (host, unit)
            seen[host] = f"{backend}->{unit}"

    assert len(set(seen.values())) == 3, (
        f"npu resolved to the same thing on every host: {seen}. A resolution "
        f"that ignores the host is exactly the defect this namespace exists to "
        f"not have."
    )
    print(f"ok   devicens: npu resolves per host -- {seen}")


def test_the_windows_and_android_success_branches_resolve_correctly():
    """The success paths this Mac cannot reach, exercised against a fake probe.

    Found by nullification: N13 and N14 changed the unit `_resolve_openvino`
    and `_resolve_qnn` return on success, and nothing went red -- those
    branches never run here, because neither an Intel NPU nor a reachable
    Hexagon device exists on this host, so only their refusal paths were
    covered. That is a real hole and this closes it.

    The probes themselves are NOT reimplemented: the real
    `intelnpu.npu_available` / `available_devices` and
    `qnn_device.device_report` are temporarily replaced with functions that
    answer as a machine with the hardware would, so the code under test is the
    resolver, unchanged. It is not evidence that an NPU was reached -- §8 of
    docs/devices/DEVICE_NS.md says so -- it is evidence that the resolver names
    the right unit when the probe says yes.
    """
    _shim()
    from torchnative.export import intelnpu, qnn_device

    saved = (intelnpu.npu_available, intelnpu.available_devices,
             qnn_device.device_report)
    try:
        intelnpu.npu_available = lambda *a, **k: True
        intelnpu.available_devices = lambda *a, **k: ("CPU", "NPU")
        res = _with_host("windows", D.npu.resolve)
        assert res.backend == "openvino", res
        assert res.unit == "Intel NPU", res
        assert res.host == "windows", res
        assert "NPU" in res.detail["devices"], res.detail

        # `htp_reachable` is what the resolver gates on now, and it is a
        # separate fact from `htp_arch`: the arch is a part number looked up
        # from an SoC name string, which a device with an unreachable compute
        # DSP produces just as readily. This fake must supply BOTH, or it is
        # asserting the success branch against an input the resolver is right
        # to refuse. See test_qnnprobe.py.
        qnn_device.device_report = lambda *a, **k: {
            "reachable": True, "htp_arch": "v75", "soc_model": "SM8550",
            "htp_reachable": True, "cdsp_fastrpc": ["/dev/cdsprpc-smd"],
        }
        res2 = _with_host("android", D.npu.resolve)
        assert res2.backend == "qnn", res2
        assert res2.unit == "Qualcomm Hexagon NPU", res2
        assert res2.detail["htp_arch"] == "v75", res2.detail
    finally:
        (intelnpu.npu_available, intelnpu.available_devices,
         qnn_device.device_report) = saved

    # And the refusal path still refuses once the real probes are back.
    try:
        _with_host("windows", D.npu.resolve)
    except D.NpuUnresolved:
        pass
    else:
        raise AssertionError("windows resolved after the fake probe was removed")
    print(
        "ok   devicens: the windows and android success branches name the Intel "
        "NPU and the Hexagon NPU, and refuse again once the real probes return"
    )


def test_npu_refuses_by_name_on_a_host_with_no_npu_path():
    _shim()

    def probe():
        try:
            D.npu.resolve()
        except D.NpuUnresolved as exc:
            return str(exc)
        return None

    err = _with_host("freebsd", probe)
    assert err is not None, "an unknown host must not resolve"
    assert "freebsd" in err, err
    assert "cpu" in err.lower(), "the refusal must say it will not fall back to cpu"
    print("ok   devicens: an unknown host refuses by name and says it will not use the cpu")


def test_npu_never_resolves_to_the_cpu():
    """docs/graph/NPU2.md section 1: 'executed' was true and meant the CPU."""
    _shim()
    for host in ("darwin", "windows", "android", "freebsd"):
        def probe():
            try:
                return D.npu.resolve()
            except D.NpuUnresolved:
                return None

        res = _with_host(host, probe)
        if res is None:
            continue
        assert "cpu" not in res.unit.lower(), res
        assert "cpu" not in res.backend.lower(), res
    print("ok   devicens: no host resolves npu to a cpu unit")


def test_npu_resolution_says_which_unit_and_who_measured_it():
    _shim()
    try:
        res = D.npu.resolve()
    except D.NpuUnresolved as exc:
        print(f"ok   devicens: npu does not resolve here, by name -- {str(exc)[:90]}")
        return
    assert res.unit and res.backend and res.source
    assert res.host == D.host()
    print(
        f"ok   devicens: npu resolved to {res.unit!r} via {res.backend} "
        f"(probe: {res.source})"
    )


def test_an_npu_resolution_cannot_be_built_without_naming_its_unit():
    _shim()
    for bad in (("h", "", "u", "s"), ("h", "b", "", "s"), ("h", "b", "u", "")):
        try:
            D.NpuResolution(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"NpuResolution accepted {bad}")
    print("ok   devicens: an NpuResolution must name a backend, a unit and a source")


def test_an_availability_cannot_be_unavailable_without_a_reason():
    _shim()
    try:
        D.Availability("x", False, source="s", kind="measured")
    except ValueError:
        pass
    else:
        raise AssertionError("Availability accepted an unavailable with no reason")
    try:
        D.Availability("x", True, source="s", kind="measured", reason="oops")
    except ValueError:
        pass
    else:
        raise AssertionError("Availability accepted an available carrying a reason")
    print("ok   devicens: Availability refuses to be built without a named reason")


# --------------------------------------------------------------------------
# `nn.Module.to`
# --------------------------------------------------------------------------


def test_the_to_patch_is_installed_and_idempotent():
    _shim()
    assert getattr(nn.Module.to, "_torchnative_patched", False)
    assert _module_to.original() is not None
    assert _module_to.install() is False, "install() must be idempotent"
    assert getattr(nn.Module.to, "_torchnative_patched", False)
    print("ok   devicens: nn.Module.to is patched once and re-installing is a no-op")


def _fresh():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


def _fingerprint(m):
    return [(str(p.device), str(p.dtype), tuple(p.shape)) for p in m.parameters()] + [
        (str(b.device), str(b.dtype)) for b in m.buffers()
    ]


def test_to_is_unchanged_for_every_ordinary_argument_form():
    """The differential test. Patched `to` vs the original, same arguments.

    Patching a core method is exactly what breaks silently, so this does not
    assert that the wrapper "delegates" -- it runs both and compares the
    observable result. Any form that the wrapper mishandled would diverge here
    even though every other test in this file stayed green.
    """
    _shim()
    original = _module_to.original()
    assert original is not None

    other = torch.zeros(1, dtype=torch.float64)

    forms = [
        ("to('cpu')", (("cpu",), {})),
        ("to(torch.device('cpu'))", ((torch.device("cpu"),), {})),
        ("to(torch.float16)", ((torch.float16,), {})),
        ("to(torch.float64)", ((torch.float64,), {})),
        ("to(device, dtype)", (("cpu", torch.float16), {})),
        ("to(device=..., dtype=...)", ((), {"device": "cpu", "dtype": torch.float16})),
        ("to(other_tensor)", ((other,), {})),
        ("to('cpu', non_blocking=True)", (("cpu",), {"non_blocking": True})),
        ("to(dtype=..., non_blocking=...)", ((), {"dtype": torch.float16, "non_blocking": True})),
        (
            "to(memory_format=...)",
            ((), {"device": "cpu", "memory_format": torch.preserve_format}),
        ),
    ]

    checked = []
    for label, (args, kwargs) in forms:
        a, b = _fresh(), _fresh()
        ra = nn.Module.to(a, *args, **kwargs)
        rb = original(b, *args, **kwargs)
        assert ra is a, f"{label}: patched to() did not return self"
        assert rb is b, f"{label}: upstream to() did not return self"
        assert _fingerprint(a) == _fingerprint(b), (
            f"{label}: the patch changed upstream behaviour\n"
            f"  patched:  {_fingerprint(a)}\n"
            f"  upstream: {_fingerprint(b)}"
        )
        checked.append(label)

    if D.mps.available:
        for label, args in (("to('mps')", ("mps",)),):
            a, b = _fresh(), _fresh()
            nn.Module.to(a, *args)
            original(b, *args)
            assert _fingerprint(a) == _fingerprint(b), label
            checked.append(label)

    print(f"ok   devicens: to() agrees with upstream on {len(checked)} argument forms")


def test_ordinary_calls_reach_upstream_with_identical_arguments():
    """Byte-identical passthrough, checked against a spy rather than an effect.

    Found by nullification: a wrapper that did `kwargs.pop("non_blocking")`
    before delegating passed `test_to_is_unchanged_for_every_ordinary_argument_form`,
    because `non_blocking` has no observable effect on a CPU-to-CPU copy. The
    differential test compares *results*, so an argument that does not change
    the result is invisible to it. This one compares the **arguments upstream
    received**, which is the invariant `_module_to` actually claims.
    """
    _shim()
    seen = []

    def spy(self, *args, **kwargs):
        seen.append((args, dict(kwargs)))
        return self

    wrapped = _module_to.make(spy)
    other = torch.zeros(1, dtype=torch.float64)
    forms = [
        (("cpu",), {}),
        ((torch.device("cpu"),), {}),
        ((torch.float16,), {}),
        (("cpu", torch.float16), {}),
        ((), {"device": "cpu", "dtype": torch.float16}),
        ((other,), {}),
        (("cpu",), {"non_blocking": True}),
        (("cpu",), {"non_blocking": False}),
        ((), {"dtype": torch.float16, "non_blocking": True}),
        ((), {"device": "cpu", "memory_format": torch.preserve_format}),
        ((), {}),
    ]
    m = _fresh()
    for args, kwargs in forms:
        seen.clear()
        wrapped(m, *args, **kwargs)
        assert len(seen) == 1, (args, kwargs, seen)
        got_args, got_kwargs = seen[0]
        assert got_args == args, (
            f"upstream received positional {got_args!r}, not {args!r}"
        )
        assert got_kwargs == kwargs, (
            f"upstream received keyword {got_kwargs!r}, not {kwargs!r} -- "
            f"the wrapper altered an argument on the way through"
        )
    print(
        f"ok   devicens: {len(forms)} ordinary argument forms reach upstream "
        f"byte-identically (spy, not effect)"
    )


def test_to_raises_the_same_errors_upstream_does():
    """Error behaviour is behaviour: an integer dtype must still be refused."""
    _shim()
    original = _module_to.original()
    for args, kwargs in ((("cpu", torch.int32), {}), ((torch.int64,), {})):
        pa = pb = None
        try:
            nn.Module.to(_fresh(), *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            pa = type(exc).__name__
        try:
            original(_fresh(), *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            pb = type(exc).__name__
        assert pa == pb, (args, kwargs, pa, pb)
        assert pa is not None, (args, kwargs, "expected a refusal from both")
    print("ok   devicens: to() raises what upstream raises for refused dtypes")


def test_to_an_eager_torchnative_device_moves_the_parameters():
    _shim()
    m = _fresh()
    out = m.to(D.cpu)
    assert out is m, "to() must return the same nn.Module, never a wrapper"
    assert all(p.device.type == "cpu" for p in m.parameters())

    if D.mps.available:
        m2 = _fresh()
        out2 = m2.to(D.mps)
        assert out2 is m2
        assert all(p.device.type == "mps" for p in m2.parameters()), (
            "to(torchnative.device.mps) must actually move the parameters"
        )
        print("ok   devicens: to(device.cpu) and to(device.mps) move parameters, in place")
    else:
        print("ok   devicens: to(device.cpu) moves parameters in place (mps unavailable)")


def test_to_an_unavailable_eager_device_refuses_by_name():
    _shim()
    if D.cuda.available:
        print("ok   devicens: cuda is available here, refusal path not exercised")
        return
    try:
        _fresh().to(D.cuda)
    except D.DeviceUnavailable as exc:
        assert "cuda" in str(exc)
        assert D.cuda.availability().reason in str(exc)
    else:
        raise AssertionError("to(cuda) must refuse when cuda is unavailable")
    print("ok   devicens: to(device.cuda) refuses by name with _cuda_probe's reason")


def test_to_the_compiled_target_refuses_and_names_what_it_resolved_to():
    """It must not return `self` unchanged: that is the dropped argument."""
    _shim()
    m = _fresh()
    try:
        result = m.to(D.npu)
    except NotImplementedError as exc:
        text = str(exc)
        assert "npu" in text
        res = D.npu.resolve()
        assert res.unit in text, "the refusal must say which NPU this host has"
        assert "NOT implemented" in text
        print(
            f"ok   devicens: to(device.npu) refuses after resolving to {res.unit!r} "
            f"rather than returning the model unchanged"
        )
        return
    except D.NpuUnresolved as exc:
        assert "npu" in str(exc)
        print(f"ok   devicens: to(device.npu) refuses -- unresolved here: {str(exc)[:70]}")
        return
    raise AssertionError(
        f"to(npu) returned {result!r} instead of refusing -- an accepted and "
        f"dropped argument is the worst outcome available"
    )


def test_to_refuses_two_torchnative_devices():
    _shim()
    try:
        _fresh().to(D.cpu, D.mps)
    except TypeError as exc:
        assert "more than one" in str(exc)
    else:
        raise AssertionError("to() must refuse two torchnative devices")
    print("ok   devicens: to() refuses more than one torchnative device")


def test_to_the_compiled_target_refuses_extra_arguments():
    _shim()
    try:
        _fresh().to(D.npu, torch.float16)
    except TypeError as exc:
        assert "no other" in str(exc), exc
    else:
        raise AssertionError("to(npu, dtype) must refuse")
    print("ok   devicens: to(npu, dtype) refuses -- a compiled target is not a conversion")


def test_the_namespace_report_is_serialisable_and_complete():
    _shim()
    import json

    text = json.dumps(D.report(), default=str)
    for name in ("cpu", "mps", "vulkan", "cuda", "npu"):
        assert f'"{name}"' in text
    print(f"ok   devicens: report() covers all five devices ({len(text)} bytes of JSON)")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
