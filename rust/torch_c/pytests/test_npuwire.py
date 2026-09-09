"""`model.to(torchnative.device.npu)` on an Intel NPU host: the DISPATCH, not the hardware.

This file exists for one sentence the user wrote:

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
    model.to(device.npu)
    out = model.generate(...)

Until this round `to(device.npu)` resolved the NPU, named it, and then raised
`NotImplementedError`. The mechanism it said was missing was in fact already
present, one import away, as `torchnative.export.intelnpu._compile_model` --
the `named_children()` walk that swaps `torch.nn.Linear` for `_NPULinear`.
This round wires the two together and these tests are the evidence.

**What this file is evidence ABOUT, stated once here and again in every test
name and docstring below.** It is evidence about DISPATCH: that
`_to_compiled` sends an `openvino` resolution to the lowering, hands back a
real `nn.Module`, carries the partial-offload report out to the caller, and
still refuses for `coreml` and `qnn`. It is **not** evidence that anything ran
on an Intel NPU. This host is an arm64 Mac: there is no Intel NPU on it, no
OpenVINO runtime for it, and `library_candidates` refuses on darwin by design.

**How the hardware is stood in for, and where the line is drawn.** Two fakes,
each replacing exactly one boundary and nothing above it:

* the **probe** -- `TORCHNATIVE_DEVICE_HOST=windows` plus
  `intelnpu.npu_available` / `available_devices` answering as a machine with
  the hardware would. The resolver itself is untouched. This is the shape
  `test_devicens.py::test_the_windows_and_android_success_branches_resolve_correctly`
  established, and it is used here for the same reason: those branches cannot
  run on this machine, so their only alternative is no coverage at all.
* the **OpenVINO runtime** -- `intelnpu.OpenVINO` is replaced by a stand-in
  with the same four methods `_NPULinear._compile_for` and `forward` call
  (`devices`, `compile_ir`, `execution_devices`, `infer`). Everything above
  that line is the real code: the real `_compile_model` walk, the real
  `linear_ir` eligibility check with the real `MAX_DIM`, the real
  `verdict_execution_devices` assertion, the real report, and the real
  `_module_to` wrapper. The stand-in computes the linear from the weight blob
  the module actually packed, so a wiring error that lost the weights shows up
  as a wrong number rather than as a green test.

**Why this file does not call `test_shim._shim()`.** Everything under test
here is pure Python module-tree manipulation: `_module_to`'s wrapper and
`intelnpu`'s `named_children()` walk. Neither reads a single ATen kernel, so
the answers do not change between the vendored shim and upstream torch, and
requiring the shim would have made this file silent in a worktree that lacks
the vendored build product -- which is exactly the worktree it was written in.
`test_devicens.py` keeps the `_shim()` guard because its probes (mps
allocation, `_mps_is_available`) genuinely do measure the shim.
"""

import os
import re
import sys
import warnings

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
sys.path.insert(0, _VENDOR_DIR)

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import torchnative.device as D  # noqa: E402
from torchnative.device import _module_to  # noqa: E402
from torchnative.export import intelnpu  # noqa: E402


# --------------------------------------------------------------------------
# The two fakes. Each replaces one boundary; nothing above them is stubbed.
# --------------------------------------------------------------------------


class _FakeOpenVINO:
    """Stands in for `intelnpu.OpenVINO`, and for nothing above it.

    Four methods, because those are the four `_NPULinear._compile_for` and
    `_NPULinear.forward` call. It reports `NPU` among its devices and `NPU` as
    the execution device, so the real `verdict_execution_devices` assertion
    runs for real and passes for the same reason it would on the hardware --
    the reading says NPU.

    `infer` is a real linear over the **weight blob the module packed**, read
    back through `intelnpu.unpack_f16`, with the shapes read out of the IR
    `intelnpu.linear_ir` emitted. So it is not a constant: a wiring mistake
    that handed the layer the wrong weights, or transposed them, produces a
    wrong number here. That is deliberately more than a stub needs to do --
    the cheap version would return zeros, and zeros cannot fail.
    """

    #: Every instance built during a test, so a test can assert one was used.
    instances = []

    def __init__(self, path=None):
        assert path is None, (
            f"the user's path must not carry an OpenVINO library path: got {path!r}. "
            f"The runtime is discovered from the pip package now."
        )
        self.compiled = []
        _FakeOpenVINO.instances.append(self)

    def devices(self):
        return ("CPU", "NPU")

    def compile_ir(self, xml, device, weights):
        dims = [int(d) for d in re.findall(r"<dim>(\d+)</dim>", xml)]
        batch, in_f = dims[0], dims[1]
        out_f = dims[-1]
        flat = intelnpu.unpack_f16(weights)
        n = out_f * in_f
        w = flat[:n]
        bias = flat[n:n + out_f] if len(flat) >= n + out_f else None
        handle = {"device": device, "batch": batch, "in": in_f, "out": out_f,
                  "w": w, "bias": bias}
        self.compiled.append(handle)
        return handle

    def execution_devices(self, compiled):
        return (compiled["device"],)

    def infer(self, compiled, blob):
        x = intelnpu.unpack_f16(blob)
        in_f, out_f, batch = compiled["in"], compiled["out"], compiled["batch"]
        w, bias = compiled["w"], compiled["bias"]
        out = []
        for b in range(batch):
            row = x[b * in_f:(b + 1) * in_f]
            for o in range(out_f):
                acc = sum(row[i] * w[o * in_f + i] for i in range(in_f))
                if bias is not None:
                    acc += bias[o]
                out.append(acc)
        return intelnpu.pack_f16(out)


class _on_a_faked_intel_npu_host:
    """`TORCHNATIVE_DEVICE_HOST=windows` + the probe and runtime fakes.

    The resolver, `_compile_model`, `_NPULinear` and `_module_to` are all the
    real ones inside this block. Only `npu_available`, `available_devices` and
    `OpenVINO` are replaced -- the three things that read the machine.
    """

    def __enter__(self):
        self._host = os.environ.get("TORCHNATIVE_DEVICE_HOST")
        os.environ["TORCHNATIVE_DEVICE_HOST"] = "windows"
        self._saved = (intelnpu.npu_available, intelnpu.available_devices,
                       intelnpu.OpenVINO)
        intelnpu.npu_available = lambda *a, **k: True
        intelnpu.available_devices = lambda *a, **k: ("CPU", "NPU")
        intelnpu.OpenVINO = _FakeOpenVINO
        _FakeOpenVINO.instances = []
        return self

    def __exit__(self, *exc):
        (intelnpu.npu_available, intelnpu.available_devices,
         intelnpu.OpenVINO) = self._saved
        if self._host is None:
            os.environ.pop("TORCHNATIVE_DEVICE_HOST", None)
        else:
            os.environ["TORCHNATIVE_DEVICE_HOST"] = self._host


class _on_host:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self._host = os.environ.get("TORCHNATIVE_DEVICE_HOST")
        os.environ["TORCHNATIVE_DEVICE_HOST"] = self.name
        return self

    def __exit__(self, *exc):
        if self._host is None:
            os.environ.pop("TORCHNATIVE_DEVICE_HOST", None)
        else:
            os.environ["TORCHNATIVE_DEVICE_HOST"] = self._host


class _TinyBlock(nn.Module):
    """A LayerNorm next to two Linears -- the shape the report has to describe.

    Not a bare `Sequential` of Linears: a model made only of lowerable leaves
    would report `fully_offloaded=True` and could never distinguish a report
    that tells the truth from one that always says yes.
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.fc1 = nn.Linear(8, 6)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(6, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))


def _fresh_block():
    torch.manual_seed(0)
    return _TinyBlock()


# --------------------------------------------------------------------------
# The wiring
# --------------------------------------------------------------------------


def test_to_device_npu_returns_a_real_module_on_a_faked_intel_npu_host():
    """DISPATCH evidence, not hardware: `to(npu)` hands back `self`, still an `nn.Module`.

    The one thing `generate()` needs is that `to()` did not swap the model for
    an inference wrapper. `optimum` returns such a wrapper and that is why it
    cannot backprop; `_module_to`'s docstring says this file must never do it.
    So this asserts identity (`result is model`), `isinstance(nn.Module)`, and
    that the module still answers `parameters()`, `named_children()` and
    `state_dict()` -- the three things `generate()` and a checkpoint reach for.

    The Intel NPU here is a fake probe and a fake OpenVINO runtime. Nothing in
    this test says a number was computed on an NPU.
    """
    model = _fresh_block()
    with _on_a_faked_intel_npu_host():
        result = model.to(D.npu)

    assert result is model, (
        f"to(device.npu) returned {type(result).__name__}, not the module itself. "
        f"A wrapper here is the thing _module_to.py's docstring forbids."
    )
    assert isinstance(result, nn.Module)
    assert list(result.parameters()), "the returned module has no parameters"
    assert dict(result.named_children()), "the returned module has no children"
    assert result.state_dict(), "the returned module has no state_dict"
    assert type(result.fc1).__name__ == "_NPULinear", (
        f"fc1 is still {type(result.fc1).__name__} -- nothing was lowered, so the "
        f"dispatch did not reach _compile_model"
    )
    assert _FakeOpenVINO.instances, (
        "no OpenVINO was constructed -- the lowering never compiled anything, so "
        "this test would have passed against a to() that merely returned self"
    )
    print("ok   npuwire: to(device.npu) returns the same nn.Module, with its "
          "Linears lowered (dispatch evidence; the NPU is faked)")


def test_a_generate_style_loop_still_runs_after_to_device_npu():
    """DISPATCH evidence, not hardware: repeated forward works and agrees with the CPU.

    `generate()` is a loop of forwards over a model it never inspects. This is
    that loop in miniature -- three forwards through the lowered module -- and
    the results are compared against the same module before lowering. The
    comparison is loose (`atol=2e-2`): the lowering casts the weights to f16
    and the stand-in runtime accumulates in Python floats, so the numbers are
    f16-accurate and no more. It is still a comparison against an independent
    answer, which a stub returning zeros would fail.
    """
    reference = _fresh_block()
    x = torch.randn(1, 8)
    expected = reference(x)

    model = _fresh_block()
    with _on_a_faked_intel_npu_host():
        model.to(D.npu)
        outs = [model(x) for _ in range(3)]

    for i, got in enumerate(outs):
        assert got.shape == expected.shape, (i, got.shape, expected.shape)
        # Not `torch.allclose`: this shim has no table entry for it
        # (rust/torch_c/src/overloads.json), so calling it raises
        # NotImplementedError. That went unnoticed while this file lived in a
        # worktree with no vendored tree, where `import torch` fell through to
        # an upstream install that does have it. The subtraction below is the
        # same claim in operators the shim does implement.
        worst = float((got - expected).abs().max())
        assert worst <= 2e-2, (
            f"forward {i} after to(npu) disagrees with the unlowered module "
            f"by {worst} (tolerance 2e-2):\n"
            f"  got      {got}\n  expected {expected}"
        )
    # `torch.equal` is absent from the shim's table for the same reason as
    # `torch.allclose` above. Bit-identity is still the claim.
    assert float((outs[0] - outs[1]).abs().max()) == 0.0, (
        "the compiled leaf is not deterministic: two forwards on the same "
        "input differ"
    )
    print(f"ok   npuwire: {len(outs)} forwards run after to(device.npu) and agree "
          f"with the unlowered module to f16 (dispatch evidence; the NPU is faked)")


def test_the_partial_offload_report_reaches_the_caller_and_names_what_stayed():
    """DISPATCH evidence, not hardware: the caller cannot end up believing a
    partial offload was a whole one.

    docs/graph/NPU2.md is an entire document about a partial offload that went
    unnoticed BECAUSE THE ANSWERS WERE RIGHT. `_compile_model` already produces
    a report naming every leaf left behind; the defect this guards is the
    report stopping at `_to_compiled` and never reaching the caller.

    Two carriers, and both are checked, because they fail differently:

    * `model.torchnative_offload` -- the report, for a caller who asks.
    * a `UserWarning` -- for a caller who does not. `fully_offloaded=False`
      with nobody told is silence, and silence is the defect.

    The model here has a `LayerNorm`, so `fully_offloaded` MUST be False. A
    report that always said True would pass a test written on an all-Linear
    model, which is why `_TinyBlock` is not one.
    """
    model = _fresh_block()
    with _on_a_faked_intel_npu_host():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.to(D.npu)

    report = getattr(model, "torchnative_offload", None)
    assert report is not None, (
        "to(device.npu) left no offload report on the model -- the caller has no "
        "way to learn that a LayerNorm stayed on the CPU"
    )
    assert report["fully_offloaded"] is False, report
    assert "LayerNorm" in report["left_on_cpu"], report["left_on_cpu"]
    assert report["swapped"] == ["fc1", "fc2"], report["swapped"]
    assert 0.0 < report["fraction_moved"] < 1.0, report["fraction_moved"]
    assert report["execution_devices"] == ["NPU"], report["execution_devices"]

    texts = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert texts, (
        "a partially offloaded model was returned without a word -- a caller who "
        "never reads .torchnative_offload is in docs/graph/NPU2.md section 1"
    )
    joined = "\n".join(texts)
    assert "LayerNorm" in joined, joined
    assert "fraction" in joined or "%" in joined, joined
    print("ok   npuwire: a partial offload is reported on the model AND warned "
          "about, naming LayerNorm (dispatch evidence; the NPU is faked)")


def test_a_full_offload_does_not_warn_so_the_warning_means_something():
    """The other half of the warning: it must not fire when nothing stayed behind.

    A warning that always fires is not information. This model is all
    `Linear`, so `fully_offloaded` is True and the warning must be absent --
    which is what makes the warning in the previous test a signal.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4))
    with _on_a_faked_intel_npu_host():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.to(D.npu)

    report = model.torchnative_offload
    assert report["fully_offloaded"] is True, report
    assert report["fraction_moved"] == 1.0, report
    assert not [w for w in caught if issubclass(w.category, UserWarning)], (
        [str(w.message) for w in caught]
    )
    print("ok   npuwire: a full offload is silent, so the partial-offload warning "
          "carries information (dispatch evidence; the NPU is faked)")


def test_the_report_agrees_with_plan_lowering_taken_beforehand():
    """`plan_lowering` is the public verdict; the report is what happened.

    They call the same `linear_ir`, so they cannot drift -- but only if the
    thing `to()` carries out is genuinely the plan's subject. This takes the
    plan before lowering and compares it against the report afterwards.
    """
    model = _fresh_block()
    plan = intelnpu.plan_lowering(model)
    with _on_a_faked_intel_npu_host():
        model.to(D.npu)
    report = model.torchnative_offload

    assert plan["eligible"] == report["swapped"], (plan["eligible"], report["swapped"])
    assert plan["left_on_cpu"] == report["left_on_cpu"], (plan, report)
    assert plan["fully_offloaded"] == report["fully_offloaded"]
    assert abs(plan["fraction_moved"] - report["fraction_moved"]) < 1e-9
    print("ok   npuwire: plan_lowering taken before to(device.npu) matches the "
          "report it produced (dispatch evidence; the NPU is faked)")


def test_zero_leaves_lowered_is_a_refusal_and_not_a_success():
    """Nothing lowered is a refusal. Returning the model would be the silent fallback.

    A model with no `Linear` at all has nothing to run on the NPU. `to()` must
    raise rather than hand back a CPU model the caller believes is offloaded
    -- and it must not leave a report behind either, because a report on an
    untouched model is the same lie with a receipt.
    """
    model = nn.Sequential(nn.LayerNorm(8), nn.ReLU())
    with _on_a_faked_intel_npu_host():
        try:
            model.to(D.npu)
        except intelnpu.IntelNPUUnsupported as exc:
            assert "nothing was lowered" in str(exc), exc
        else:
            raise AssertionError(
                "to(device.npu) accepted a model with no lowerable leaf -- an "
                "argument accepted and dropped is the worst outcome available"
            )
    assert not hasattr(model, "torchnative_offload"), (
        "a refused lowering still left an offload report on the model"
    )
    print("ok   npuwire: zero leaves lowered refuses by name and leaves no report "
          "(dispatch evidence; the NPU is faked)")


def test_coreml_and_qnn_still_refuse_at_the_quality_they_refused_before():
    """Only the openvino branch was wired. The other two must not become fake successes.

    Each is driven to its success-shaped resolution with the same probe-level
    fake, so this is not merely observing that the probes fail here: the
    resolution is real and names the real unit, and the refusal still happens
    *after* it.
    """
    checked = []

    # CoreML / Apple Neural Engine.
    model = _fresh_block()
    with _on_host("darwin"):
        try:
            res = D.npu.resolve()
        except D.NpuUnresolved:
            res = None
        if res is not None:
            try:
                model.to(D.npu)
            except NotImplementedError as exc:
                text = str(exc)
                assert res.unit in text, text
                assert "NOT implemented" in text, text
                assert "npu" in text, text
                checked.append(f"coreml -> {res.unit}")
            else:
                raise AssertionError("to(npu) on darwin must still refuse")

    # QNN / Hexagon.
    from torchnative.export import qnn_device
    saved = qnn_device.device_report
    try:
        qnn_device.device_report = lambda *a, **k: {
            "reachable": True, "htp_arch": "v75", "soc_model": "SM8550",
            "htp_reachable": True, "cdsp_fastrpc": ["/dev/cdsprpc-smd"],
        }
        with _on_host("android"):
            res = D.npu.resolve()
            assert res.backend == "qnn", res
            try:
                _fresh_block().to(D.npu)
            except NotImplementedError as exc:
                text = str(exc)
                assert res.unit in text, text
                assert "NOT implemented" in text, text
                checked.append(f"qnn -> {res.unit}")
            else:
                raise AssertionError("to(npu) on android must still refuse")
    finally:
        qnn_device.device_report = saved

    assert "qnn -> Qualcomm Hexagon NPU" in checked, checked
    print(f"ok   npuwire: the unwired backends still refuse after resolving "
          f"({', '.join(checked)})")


def test_the_openvino_branch_is_chosen_by_the_resolution_and_not_by_the_platform():
    """The dispatch key is `resolution.backend`, and a different backend must not compile.

    Nullification target: if `_to_compiled` lowered for any resolution that
    succeeded -- rather than for `backend == "openvino"` -- this test is the
    one that goes red, because the android resolution above would then produce
    a lowered model instead of a refusal. Stated separately from the previous
    test because that one could be satisfied by a host check.
    """
    with _on_a_faked_intel_npu_host():
        assert D.npu.resolve().backend == "openvino"
        m = _fresh_block()
        m.to(D.npu)
        assert m.torchnative_offload["device"] == "NPU"
    print("ok   npuwire: the lowering is keyed on resolution.backend == 'openvino'")


def test_to_the_compiled_target_still_refuses_extra_arguments():
    """A compiled target is not a dtype conversion, wired or not."""
    with _on_a_faked_intel_npu_host():
        try:
            _fresh_block().to(D.npu, torch.float16)
        except TypeError as exc:
            assert "no other" in str(exc), exc
        else:
            raise AssertionError("to(npu, dtype) must refuse even now that npu works")
    print("ok   npuwire: to(device.npu, dtype) still refuses")


# --------------------------------------------------------------------------
# The upstream-semantics proofs, re-run here.
#
# These are the two `test_devicens.py` tests the round was told to run:
# the differential test over the ordinary argument forms, and the spy on the
# arguments upstream receives. They are duplicated rather than imported
# because `test_devicens.py` guards every test with `_shim()`, and this
# worktree has no vendored `_C.abi3.so` -- so over there they do not run at
# all. Neither of them measures an ATen kernel, so running them against
# upstream torch answers the same question: did wiring the openvino branch
# change `to()` for anything that is not a torchnative device.
# --------------------------------------------------------------------------


def _fresh():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


def _fingerprint(m):
    return [(str(p.device), str(p.dtype), tuple(p.shape)) for p in m.parameters()] + [
        (str(b.device), str(b.dtype)) for b in m.buffers()
    ]


def test_to_is_still_unchanged_for_every_ordinary_argument_form():
    """The differential test, re-run after the openvino branch was wired."""
    original = _module_to.original()
    assert original is not None, "nn.Module.to is not patched"
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
        ("to(memory_format=...)", ((), {"device": "cpu", "memory_format": torch.preserve_format})),
        ("to()", ((), {})),
    ]
    for label, (args, kwargs) in forms:
        a, b = _fresh(), _fresh()
        ra = nn.Module.to(a, *args, **kwargs)
        rb = original(b, *args, **kwargs)
        assert ra is a, label
        assert rb is b, label
        assert _fingerprint(a) == _fingerprint(b), (
            f"{label}: wiring the openvino branch changed upstream behaviour\n"
            f"  patched:  {_fingerprint(a)}\n  upstream: {_fingerprint(b)}"
        )
    print(f"ok   npuwire: to() still agrees with upstream on {len(forms)} argument forms")


def test_ordinary_calls_still_reach_upstream_with_identical_arguments():
    """The spy, re-run: an ordinary `to()` must not have gained a single touch."""
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
        assert got_args == args, (got_args, args)
        assert got_kwargs == kwargs, (got_kwargs, kwargs)
    print(f"ok   npuwire: {len(forms)} ordinary argument forms still reach upstream "
          f"byte-identically (spy, not effect)")


def test_the_compile_options_are_reachable_through_to_and_a_typo_still_refuses():
    """`progress=` and `eager=` must work through `to()`, the only public spelling.

    They did not. `_compile_model` grew both, `_to_compiled` forwarded neither
    and refused every extra kwarg, so `model.to(device.npu, progress=...)`
    raised TypeError on a real Intel NPU while the feature sat unreachable. The
    gate stayed green because every test that exercised them called
    `_compile_model` DIRECTLY -- including this file's neighbours. A feature the
    public path cannot reach is the same as not having the feature, and a test
    that reaches past the public path cannot notice.

    So this one goes through `to()` and nothing else.

    It also pins the other half: a dtype, a memory format or a misspelled
    option must still be refused. Accepting the compile step's options is not
    the same as accepting anything, and forwarding **kwargs blindly would turn
    `progres=` into silence.

    DISPATCH evidence. The NPU is faked.
    """
    seen = []
    model = _fresh_block()
    with _on_a_faked_intel_npu_host():
        result = model.to(
            D.npu, progress=lambda done, total, name: seen.append((done, total, name)))
    assert result is model
    assert seen, "progress= reached to() but no callback arrived"
    assert seen[-1][0] == seen[-1][1], f"progress never reported completion: {seen[-1]}"
    assert all(isinstance(n, str) and n for _, _, n in seen), seen

    # eager=False takes the other branch and must also survive the trip.
    lazy = _fresh_block()
    with _on_a_faked_intel_npu_host():
        lazy.to(D.npu, eager=False)
    lowered = [m for m in lazy.modules() if type(m).__name__ == "_NPULinear"]
    assert lowered, "eager=False lowered nothing at all"

    # A typo must not be swallowed.
    typo = _fresh_block()
    with _on_a_faked_intel_npu_host():
        try:
            typo.to(D.npu, progres=lambda *a: None)
            raised = False
        except TypeError as exc:
            raised = True
            assert "progres" in str(exc), exc
    assert raised, "a misspelled option was accepted and silently dropped"

    # And a dtype is still not a thing a compiled target accepts.
    dt = _fresh_block()
    with _on_a_faked_intel_npu_host():
        try:
            dt.to(D.npu, torch.float16)
            raised = False
        except TypeError:
            raised = True
    assert raised, "to(device.npu, dtype) stopped refusing"

    print(
        f"ok   npuwire: progress= and eager= reach the compile step through "
        f"to(device.npu) ({len(seen)} progress calls), while a typo and a dtype "
        f"still refuse (dispatch evidence; the NPU is faked)"
    )


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
