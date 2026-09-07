"""Tests for docs/devices/QNN.md -- ExecuTorch's Qualcomm backend as a subgraph delegate.

The round's shape, and therefore this file's:

* the **front end** is a real `transformers` model with one submodule replaced,
  and it is testable on this machine under the shim, with a real checkpoint and
  a real `generate()`. That is the half that runs in the gate.
* the **back end** is an ExecuTorch `.pte`. Producing a *QNN* one needs the
  Qualcomm AI Engine Direct SDK, which ships host libraries for
  `lib/x86_64-linux-clang` only, so it cannot be produced here at all. What
  *can* be produced here is an artefact through the identical
  `torch.export` -> `to_edge_transform_and_lower` -> `.to_executorch()`
  pipeline with the XNNPACK partitioner, and that is used as the control: the
  loading and running machinery is executed, and `QnnModule` is then required
  to **refuse the very same file by name**.
* **nothing here claims anything ran on an NPU.** docs/devices/QNN.md §6.4. The
  evidence that would be needed is written down in §6 and none of it was
  observed, because no Snapdragon device was attached.

Three environments, three fixtures, and each skips **by name** when its
environment is absent -- docs/devices/VULKAN3.md §6.1, a skip with a false reason is
counted as a pass:

| fixture | interpreter | needs |
|---|---|---|
| `_qnn_shim_fixture` | this one, vendored tree on PYTHONPATH | `_C.abi3.so` installed |
| `_qnn_model_fixture` | same | that, plus SmolLM2-135M in the HF cache |
| `_qnn_et_fixture` | `$TORCHNATIVE_QNN_PYTHON` | upstream torch + executorch |

The third is a separate interpreter on purpose. ExecuTorch pulls its own
`torch`, and installing it beside the shim would replace the 2.13.0 upstream
oracle every numerical claim in this repository is measured against.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile

# `torchnative` is not on the path the gate hands this process: `run.sh` puts
# the staged `_C` and `pytests/` there, and the package lives beside the
# vendored `torch` under `torchnative/src/main`. The subprocess fixtures below
# get it from `_VENDOR_DIR`; the tests that import it IN THIS PROCESS have to
# add it themselves, and four of them failed with ModuleNotFoundError until
# they did.
# APPENDED, not inserted at the front. `src/main` holds the vendored `torch`
# beside `torchnative`, and `test_shim.py` imports UPSTREAM torch at module
# scope. Putting this first shadowed upstream with the vendored tree, which
# needs its own initialisation and raised from `_preload_cuda_deps` -- the
# whole file then crashed before running one test, and the suite reported
# 1209 ok / 0 FAIL rather than a failure, because a crashed file has no
# result to report.
_SRC_MAIN = str(pathlib.Path(__file__).resolve().parents[3] / "torchnative" / "src" / "main")
if _SRC_MAIN not in sys.path:
    sys.path.append(_SRC_MAIN)

from test_shim import _CKPT_VENDOR_DIR, _CKPT_VENDOR_SHIM


# ---------------------------------------------------------------------------
# Fixture 1 and 2 -- the shim. One subprocess in the vendored tree.
# ---------------------------------------------------------------------------

_QNN_SHIM_SCRIPT = r"""
import json
import torch
import torch.nn as nn

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}

from torchnative.export import npu, qnn

out["probe"] = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in qnn.probe().items()}
out["aot_refusal"] = qnn.qnn_aot_refusal()
out["aot_available"] = qnn.qnn_aot_available()


class Fake(npu._DelegateModule):
    backend_name = "TestBackend"

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, *a, **k):
        self.calls += 1
        return self.inner(*a, **k)


class Nameless(npu._DelegateModule):
    pass


# -- the swap, on a plain Sequential -------------------------------------
m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
inner = nn.Linear(4, 4)
inner.load_state_dict(m[0].state_dict())
old = npu._replace_submodule(m, "0", Fake(inner))
out["old_type"] = type(old).__name__
out["paths"] = npu._delegated_paths(m)
x = torch.randn(3, 4)
out["forward_shape"] = list(m(x).shape)
out["calls"] = m[0].calls
out["same_object"] = npu._delegate_(m, {"2": Fake(nn.Linear(4, 2))}) is m
out["paths_after"] = npu._delegated_paths(m)

# -- and it computes what the module it replaced computed -----------------
ref = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
ref[0].load_state_dict(inner.state_dict())
ref[2].load_state_dict(m[2].inner.state_dict())
out["max_abs_diff_vs_undelegated"] = float((m(x) - ref(x)).abs().max())

# -- refusals, each by name ----------------------------------------------
def refusal(fn):
    try:
        fn()
    except npu.DelegateRefused as error:
        return str(error)
    except Exception as error:
        return "WRONG-TYPE " + type(error).__name__ + ": " + str(error)
    return None


out["refuse_bad_path"] = refusal(lambda: npu._delegate_(m, {"nope": Fake(nn.Identity())}))
out["refuse_deep_bad_path"] = refusal(lambda: npu._delegate_(m, {"0.inner.zzz": Fake(nn.Identity())}))
out["refuse_index_range"] = refusal(lambda: npu._delegate_(m, {"9": Fake(nn.Identity())}))
out["refuse_empty_path"] = refusal(lambda: npu._delegate_(m, {"": Fake(nn.Identity())}))
out["refuse_not_delegate"] = refusal(lambda: npu._delegate_(m, {"1": nn.Identity()}))
out["refuse_not_module"] = refusal(lambda: npu._replace_submodule(m, "1", 7))
out["refuse_empty_plan"] = refusal(lambda: npu._delegate_(m, {}))
out["refuse_not_dict"] = refusal(lambda: npu._delegate_(m, [("1", Fake(nn.Identity()))]))
out["refuse_nameless"] = refusal(Nameless)
out["refuse_no_forward"] = refusal(
    lambda: type("Bare", (npu._DelegateModule,), {"backend_name": "Bare"})()(x))

# -- a rejected plan must leave the model untouched ----------------------
before = npu._delegated_paths(m)
refusal(lambda: npu._delegate_(m, {"1": Fake(nn.Identity()), "nope": Fake(nn.Identity())}))
out["paths_unchanged_after_refusal"] = npu._delegated_paths(m) == before

# -- the QNN back end, on a host with no executorch ----------------------
def qnn_refusal(fn):
    try:
        fn()
    except qnn.QnnRefused as error:
        return str(error)
    except npu.DelegateRefused as error:
        return str(error)
    except Exception as error:
        return "WRONG-TYPE " + type(error).__name__ + ": " + str(error)
    return None


out["refuse_lower"] = qnn_refusal(
    lambda: qnn.lower(m, (x,), "SM8650", "/tmp/nope.pte"))
out["refuse_read_missing"] = qnn_refusal(
    lambda: qnn.read_artefact("/tmp/definitely-not-here.pte"))
out["refuse_soc_targets"] = qnn_refusal(qnn.soc_targets)
out["refuse_runtime_backends"] = qnn_refusal(qnn.runtime_backends)
out["refuse_qnn_module"] = qnn_refusal(lambda: qnn.QnnModule("/tmp/nope.pte"))
out["refuse_stub_pair"] = qnn_refusal(lambda: qnn.compiler_spec("SM8650"))

# -- qnn_device, with no ANDROID_SERIAL ----------------------------------
import os as _os
import shutil as _shutil
from torchnative.export import qnn_device as QD

saved = _os.environ.pop("ANDROID_SERIAL", None)
try:
    out["device_serial_none"] = QD.serial() is None
    out["device_adb_available"] = QD.adb_available()
    # `adb_available()` is a COMPOUND predicate -- adb on PATH *and* a serial
    # chosen -- and this fixture has just popped ANDROID_SERIAL, so it is
    # always False here whether or not adb exists. Report the two halves
    # separately, or a test branching on it can never take its strong path.
    out["device_adb_on_path"] = _shutil.which("adb") is not None
    try:
        QD.adb("shell", "true")
        out["refuse_no_serial"] = None
    except QD.QnnDeviceRefused as error:
        out["refuse_no_serial"] = str(error)
    out["device_report_unreachable"] = QD.device_report()
finally:
    if saved is not None:
        _os.environ["ANDROID_SERIAL"] = saved

out["device_dir"] = QD.DEVICE_DIR
out["soc_properties"] = list(QD.SOC_PROPERTIES)
try:
    out["stub_for_75"] = list(QD.htp_stub_for(75))
except Exception as error:
    out["stub_for_75"] = "ERROR " + str(error)
out["refuse_stub_for_none"] = None
try:
    QD.htp_stub_for(None)
except QD.QnnDeviceRefused as error:
    out["refuse_stub_for_none"] = str(error)

# -- the three SoC statuses, and the one that used to be a lie --------------
out["soc_status_values"] = [QD.SOC_KNOWN, QD.SOC_NOT_IN_TABLE, QD.SOC_TABLE_UNAVAILABLE]
out["fastrpc_nodes_listed"] = list(QD.FASTRPC_NODES)

_real_getprop = QD.getprop


def _fake_getprop(value):
    def inner(name):
        return value if name == "ro.soc.model" else ""
    return inner


# `SM8550` IS in ExecuTorch's QcomChipset. Under an interpreter with no
# executorch the answer must be TABLE_UNAVAILABLE -- a statement about this
# host -- and never NOT_IN_TABLE, which is a statement about the device.
try:
    QD.getprop = _fake_getprop("SM8550")
    out["soc_known_chipset"] = list(QD.device_soc())
    QD.getprop = _fake_getprop("CQ8750S")   # pytorch/executorch#16465's part
    out["soc_unknown_chipset"] = list(QD.device_soc())
    QD.getprop = _fake_getprop("")
    try:
        QD.device_soc()
        out["refuse_no_soc_property"] = None
    except QD.QnnDeviceRefused as error:
        out["refuse_no_soc_property"] = str(error)
finally:
    QD.getprop = _real_getprop

print(json.dumps(out))
"""


_QNN_MODEL_SCRIPT = r"""
import json
import os
import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
    from transformers.generation.utils import GenerationMixin
except Exception as error:
    out["transformers"] = None
    out["import_error"] = type(error).__name__ + ": " + str(error)
    print(json.dumps(out))
    raise SystemExit(0)

from torchnative.export import npu

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


class Recorder(npu._DelegateModule):
    '''Stands in for a submodule and forwards to it, counting the calls.

    Deliberately *not* an ExecuTorch delegate. This subprocess runs under the
    shim, where there is no ExecuTorch runtime; what it is testing is the
    front-end claim -- that replacing a submodule of a real checkpoint leaves
    `from_pretrained` and `generate` intact -- and that claim is about the
    replacement, not about what the replacement computes.
    '''

    backend_name = "Recorder"

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, *a, **k):
        self.calls += 1
        return self.inner(*a, **k)


try:
    # Was `npu.NpuModelForCausalLM.from_pretrained(MODEL_ID, plan=...)`.
    # That name is withdrawn (test_the_withdrawn_npu_names_refuse_by_name
    # asserts it now refuses), so this spells out what it did: load the real
    # checkpoint, then swap in place. The claim under test is unchanged --
    # a real `generate()` still runs with a submodule delegated.
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model = npu._delegate_(
        model, {"model.layers.0.mlp": Recorder(model.model.layers[0].mlp)}
    )
except Exception as error:
    out["model"] = None
    out["load_error"] = type(error).__name__ + ": " + str(error)
    print(json.dumps(out))
    raise SystemExit(0)

model.eval()
out["model"] = MODEL_ID
out["type"] = type(model).__name__
out["is_llama_for_causal_lm"] = isinstance(model, LlamaForCausalLM)
out["generate_is_upstreams"] = model.generate.__func__ is GenerationMixin.generate
out["has_config"] = hasattr(model, "config")
out["hidden_size"] = int(model.config.hidden_size)
out["intermediate_size"] = int(model.config.intermediate_size)
out["delegated"] = npu._delegated_paths(model)
out["delegate_type"] = type(model.model.layers[0].mlp).__name__
out["inner_type"] = type(model.model.layers[0].mlp.inner).__name__

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad():
    generated = model.generate(ids, max_new_tokens=6, do_sample=False)
out["generated_shape"] = list(generated.shape)
out["generated_text"] = tokenizer.decode(generated[0])
out["delegate_calls"] = model.model.layers[0].mlp.calls
out["state_dict_keeps_the_weights"] = any(
    k.startswith("model.layers.0.mlp.inner.") for k in model.state_dict()
)

print(json.dumps(out))
"""


def _shim_subprocess(script, extra_env=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # docs/platform/VENDOR.md wall 1
    env.update(extra_env or {})
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"qnn shim subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


_SHIM_CACHE = {}


def _qnn_shim_fixture():
    if "shim" not in _SHIM_CACHE:
        _SHIM_CACHE["shim"] = _shim_subprocess(_QNN_SHIM_SCRIPT)
    return _SHIM_CACHE["shim"]


def _hf_home():
    """A Hugging Face cache that already holds the checkpoint, or `None`.

    Never downloads. `HF_HUB_OFFLINE` is set for the subprocess, so a cache
    that turns out not to hold the model produces a load error the test reports
    as a skip rather than a network fetch inside a test suite.
    """
    for root in (
        os.environ.get("HF_HOME"),
        "/Volumes/macMini/caches/hf-home",
        os.path.expanduser("~/.cache/huggingface"),
    ):
        if not root:
            continue
        if os.path.isdir(os.path.join(root, "hub", "models--HuggingFaceTB--SmolLM2-135M")):
            return root
    return None


def _qnn_model_fixture():
    if "model" not in _SHIM_CACHE:
        home = _hf_home()
        if home is None:
            return None
        _SHIM_CACHE["model"] = _shim_subprocess(
            _QNN_MODEL_SCRIPT, {"HF_HOME": home, "HF_HUB_OFFLINE": "1"}
        )
    return _SHIM_CACHE["model"]


# ---------------------------------------------------------------------------
# Fixture 3 -- upstream torch + upstream executorch, in their own interpreter.
# ---------------------------------------------------------------------------

_QNN_ET_SCRIPT = r"""
import json
import os
import tempfile

import torch
import torch.nn as nn

out = {"torch": torch.__version__, "is_shim": hasattr(torch._C, "_aten_implemented")}

from torchnative.export import npu, qnn

out["probe"] = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in qnn.probe().items()}
out["aot_refusal"] = qnn.qnn_aot_refusal()
targets = qnn.soc_targets()
out["soc_targets"] = {k: list(v) for k, v in targets.items()}
member, soc_model, htp_arch = qnn.resolve_soc("sm8650")
out["resolve_sm8650"] = [member.name, soc_model, htp_arch]
try:
    qnn.resolve_soc("NOT-A-CHIPSET")
    out["refuse_unknown_soc"] = None
except qnn.QnnRefused as error:
    out["refuse_unknown_soc"] = str(error)

work = tempfile.mkdtemp(prefix="bw-qnn-")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(8, 16)
        self.down = nn.Linear(16, 8)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


torch.manual_seed(0)
module = MLP().eval()
example = torch.randn(2, 8)

# ---- the control artefact: same pipeline, XNNPACK partitioner ----------
control = os.path.join(work, "control.pte")
_, report = qnn.lower_cpu_reference(module, (example,), control)
out["control_report"] = report
out["control_bytes"] = os.path.getsize(control)
artefact = qnn.read_artefact(control)
out["control_methods"] = list(artefact.methods)
out["control_backends"] = list(artefact.backend_ids)
out["control_is_qnn"] = artefact.is_qnn
try:
    artefact.htp_plan()
    out["refuse_htp_plan_on_control"] = None
except qnn.QnnRefused as error:
    out["refuse_htp_plan_on_control"] = str(error)

# ---- it loads and it runs, and it agrees with eager -------------------
delegate = qnn.ExecuTorchModule(control)
out["control_repr"] = repr(delegate)
with torch.no_grad():
    eager = module(example)
got = delegate(example)
out["control_shape"] = list(got.shape)
out["control_max_abs_diff"] = float((got - eager).abs().max())

# ---- and QnnModule refuses that same file, by name --------------------
try:
    qnn.QnnModule(control)
    out["refuse_qnn_module_on_control"] = None
except npu.DelegateRefused as error:
    out["refuse_qnn_module_on_control"] = str(error)

# ---- a QNN-*shaped* artefact, for the reader's positive path ----------
#
# NOT a QNN artefact. Nothing was compiled by QNN; this takes the control
# program, rewrites its delegate's `id` to `QnnBackend` and attaches a genuine
# `QnnExecuTorchOptions` flatbuffer built by ExecuTorch's own serialiser, then
# writes it back through ExecuTorch's own serialiser. Every byte of schema is
# upstream's; the payload behind the delegate is still XNNPACK's and would be
# rejected by a real QNN backend instantly.
#
# It exists so that `read_artefact`, `htp_plan` and `match_device` are tested
# on the branch that says *yes* as well as the branch that says no. A reader
# that only ever ran against a non-QNN file would pass while returning False
# unconditionally.
from executorch.exir._serialize import _deserialize_pte_binary, _serialize_pte_binary
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.backends.qualcomm.serialization import qc_schema as QS
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    option_to_flatbuffer,
)

option = QS.QnnExecuTorchOptions(
    soc_info=QS._soc_info_table[QS.QcomChipset.SM8650],
    backend_options=QS.QnnExecuTorchBackendOptions(
        backend_type=QS.QnnExecuTorchBackendType.kHtpBackend,
        htp_options=QS.QnnExecuTorchHtpBackendOptions(
            precision=QS.QnnExecuTorchHtpPrecision.kHtpFp16
        ),
    ),
)
spec = [CompileSpec(qnn.COMPILE_SPEC_KEY, bytes(option_to_flatbuffer(option)))]

pte = _deserialize_pte_binary(open(control, "rb").read())
for plan in pte.program.execution_plan:
    for entry in plan.delegates:
        entry.id = qnn.QNN_BACKEND_ID
        entry.compile_specs = list(spec)
shaped = os.path.join(work, "qnn_shaped.pte")
open(shaped, "wb").write(bytes(_serialize_pte_binary(pte)))

shaped_art = qnn.read_artefact(shaped)
out["shaped_backends"] = list(shaped_art.backend_ids)
out["shaped_is_qnn"] = shaped_art.is_qnn
plan = shaped_art.htp_plan()
out["shaped_plan"] = {
    "backend_type": int(plan["backend_type"]),
    "backend_type_name": plan["backend_type_name"],
    "soc_model": plan["soc_model"],
    "htp_arch": plan["htp_arch"],
    "precision": int(plan["precision"]),
    "precision_name": plan["precision_name"],
}
out["match_sm8650"] = list(qnn.match_device(shaped_art, "SM8650"))
out["match_sm8550"] = list(qnn.match_device(shaped_art, "SM8550"))
out["match_sm8750"] = list(qnn.match_device(shaped_art, "SM8750"))
out["match_sa8255_same_arch"] = list(qnn.match_device(shaped_art, "SA8255"))

# a CPU-backend spec must be caught by match_device, not waved through
cpu_option = QS.QnnExecuTorchOptions(
    soc_info=QS._soc_info_table[QS.QcomChipset.SM8650],
    backend_options=QS.QnnExecuTorchBackendOptions(
        backend_type=QS.QnnExecuTorchBackendType.kGpuBackend
    ),
)
cpu_spec = [CompileSpec(qnn.COMPILE_SPEC_KEY, bytes(option_to_flatbuffer(cpu_option)))]
pte2 = _deserialize_pte_binary(open(control, "rb").read())
for plan_ in pte2.program.execution_plan:
    for entry in plan_.delegates:
        entry.id = qnn.QNN_BACKEND_ID
        entry.compile_specs = list(cpu_spec)
gpu_path = os.path.join(work, "qnn_shaped_gpu.pte")
open(gpu_path, "wb").write(bytes(_serialize_pte_binary(pte2)))
out["match_gpu_backend"] = list(qnn.match_device(qnn.read_artefact(gpu_path), "SM8650"))

# a QnnBackend delegate with no compile spec at all
pte3 = _deserialize_pte_binary(open(control, "rb").read())
for plan_ in pte3.program.execution_plan:
    for entry in plan_.delegates:
        entry.id = qnn.QNN_BACKEND_ID
        entry.compile_specs = []
naked = os.path.join(work, "qnn_shaped_naked.pte")
open(naked, "wb").write(bytes(_serialize_pte_binary(pte3)))
try:
    qnn.read_artefact(naked).htp_plan()
    out["refuse_no_compile_spec"] = None
except qnn.QnnRefused as error:
    out["refuse_no_compile_spec"] = str(error)

# the runtime has no QnnBackend, so QnnModule refuses the shaped file too
try:
    qnn.QnnModule(shaped)
    out["refuse_qnn_runtime"] = None
except npu.DelegateRefused as error:
    out["refuse_qnn_runtime"] = str(error)
out["runtime_backends"] = list(qnn.runtime_backends())

inspect = qnn.QnnModule(shaped, require_runtime=False)
out["inspect_repr"] = repr(inspect)
try:
    inspect(example)
    out["refuse_inspect_forward"] = None
except npu.DelegateRefused as error:
    out["refuse_inspect_forward"] = str(error)

# not an ExecuTorch program at all
junk = os.path.join(work, "junk.pte")
open(junk, "wb").write(b"not a flatbuffer, not even close")
try:
    qnn.read_artefact(junk)
    out["refuse_junk"] = None
except qnn.QnnRefused as error:
    out["refuse_junk"] = str(error)

# ---- the same SoC question, on an interpreter that HAS the table ------
from torchnative.export import qnn_device as QD

_real_getprop = QD.getprop


def _fake_getprop(value):
    def inner(name):
        return value if name == "ro.soc.model" else ""
    return inner


try:
    QD.getprop = _fake_getprop("SM8550")
    out["soc_known_chipset"] = list(QD.device_soc())
    QD.getprop = _fake_getprop("sm8650")            # case from the device is not guaranteed
    out["soc_lowercase_chipset"] = list(QD.device_soc())
    QD.getprop = _fake_getprop("CQ8750S")
    out["soc_unknown_chipset"] = list(QD.device_soc())
finally:
    QD.getprop = _real_getprop

# ---- the real model: SmolLM2-135M layer 0 MLP -------------------------
out["real"] = None
if os.environ.get("BW_QNN_HF_HOME"):
    try:
        from transformers import AutoModelForCausalLM

        MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
        model32 = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).eval()
        mlp = model32.model.layers[0].mlp
        hidden = int(model32.config.hidden_size)
        real_pte = os.path.join(work, "smollm2_layer0_mlp.pte")
        torch.manual_seed(0)
        sample = torch.randn(1, 5, hidden)
        _, real_report = qnn.lower_cpu_reference(mlp, (sample,), real_pte)
        real_art = qnn.read_artefact(real_pte)
        real_delegate = qnn.ExecuTorchModule(real_pte)

        model64 = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float64
        ).eval()
        mlp64 = model64.model.layers[0].mlp

        rows = []
        for k in range(5):
            torch.manual_seed(100 + k)
            probe = torch.randn(1, 5, hidden)
            with torch.no_grad():
                up32 = mlp(probe)
                up64 = mlp64(probe.double())
            got = real_delegate(probe)
            scale = float(up32.abs().max())
            oracle = float((up32.double() - up64).abs().max()) / scale
            rel = float((got - up32).abs().max()) / scale
            rows.append({"rel": rel, "oracle": oracle,
                         "ratio": (rel / oracle) if oracle else None})
        out["real"] = {
            "model": MODEL_ID,
            "submodule": "model.layers.0.mlp",
            "submodule_type": type(mlp).__name__,
            "hidden": hidden,
            "intermediate": int(model32.config.intermediate_size),
            "report": real_report,
            "bytes": os.path.getsize(real_pte),
            "backends": list(real_art.backend_ids),
            "rows": rows,
            "eps": float(torch.finfo(torch.float32).eps),
        }
    except Exception as error:  # noqa: BLE001
        out["real_error"] = type(error).__name__ + ": " + str(error)

print(json.dumps(out))
"""


_ET_CACHE = {}


def _qnn_interpreter():
    """The interpreter that has upstream torch + executorch, or `None`.

    `TORCHNATIVE_QNN_PYTHON` only. Never "whichever python is on PATH": the
    interpreter running this suite has the shim staged on its `PYTHONPATH`, and
    reaching for a sibling one by guessing is how a suite ends up measuring a
    build nobody asked for (`run.sh`'s own staleness guard exists for the same
    reason).
    """
    path = os.environ.get("TORCHNATIVE_QNN_PYTHON")
    if path and os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None


def _torchnative_only_path(work):
    """A `PYTHONPATH` entry exposing `torchnative` and **not** the vendored torch.

    `torchnative/src/main` holds both `torchnative/` and the vendored `torch/`
    tree, so putting it on the ExecuTorch interpreter's path would shadow the
    upstream torch ExecuTorch was built against with a shim that has no
    `torch.export` worth the name. A directory with one symlink in it is the
    whole fix.
    """
    root = os.path.join(work, "tn")
    os.makedirs(root, exist_ok=True)
    link = os.path.join(root, "torchnative")
    if not os.path.exists(link):
        os.symlink(os.path.join(_CKPT_VENDOR_DIR, "torchnative"), link)
    return root


def _qnn_et_fixture():
    if "et" in _ET_CACHE:
        return _ET_CACHE["et"]
    interpreter = _qnn_interpreter()
    if interpreter is None:
        return None
    work = tempfile.mkdtemp(prefix="bw-qnn-stage-")
    env = dict(os.environ)
    env["PYTHONPATH"] = _torchnative_only_path(work)
    env.pop("TORCH_USE_RTLD_GLOBAL", None)
    home = _hf_home()
    if home:
        env["HF_HOME"] = home
        env["HF_HUB_OFFLINE"] = "1"
        env["BW_QNN_HF_HOME"] = home
    proc = subprocess.run(
        [interpreter, "-c", _QNN_ET_SCRIPT],
        capture_output=True, text=True, env=env, timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"qnn executorch subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )
    _ET_CACHE["et"] = json.loads(proc.stdout.strip().splitlines()[-1])
    return _ET_CACHE["et"]


def _skip(reason):
    print(f"   (skipped: {reason})")


def _vendor_ready():
    return os.path.isfile(_CKPT_VENDOR_SHIM)


# ---------------------------------------------------------------------------
# Step 1 -- what this host can and cannot do, measured.
# ---------------------------------------------------------------------------


def test_the_qnn_ahead_of_time_half_refuses_here_and_names_the_missing_module():
    """docs/devices/QNN.md §3's headline, measured on whichever host runs this.

    The refusal must name
    `executorch.backends.qualcomm.python.PyQnnManagerAdaptor`, because that is
    the module that is actually missing and the reader's next move depends on
    knowing it -- "macOS is unsupported" sends them to a compatibility table,
    "this pybind extension is built against $QNN_SDK_ROOT and the SDK is
    x86_64-linux only" sends them to a Linux box.

    Written as a **conditional**, not as "this host cannot": run this on a
    correctly-provisioned Ubuntu x86-64 machine and the refusal is `None`, and
    the test then checks that the *other* half of the story holds instead. A
    test that hard-coded the negative would go red on the machine where the
    round finally succeeds, which is the one machine it must not.
    """
    if not _vendor_ready():
        return _skip("vendored tree has no _C.abi3.so")
    result = _qnn_shim_fixture()
    assert result["is_shim"] is True, result["is_shim"]
    refusal = result["aot_refusal"]
    assert result["aot_available"] == (refusal is None)
    if refusal is None:
        assert result["probe"]["qnn_aot_available"] is True
        return
    assert "executorch" in refusal, refusal
    # On a host with no executorch at all the refusal is about the package;
    # with executorch installed it must be about the adaptor specifically.
    if result["probe"]["executorch"]:
        assert "PyQnnManagerAdaptor" in refusal, refusal
        assert "x86_64-linux-clang" in refusal, refusal

    # The interpreter *above* has no executorch, so the branch that names the
    # adaptor is unreachable from it -- and a nullification that deleted the
    # adaptor check entirely was not caught by this test until this block
    # existed. The refusal then fell through to the partitioner import, which
    # also fails, so `aot_available` stayed False and every other assertion
    # here still passed while the sentence docs/devices/QNN.md §3 leads with had
    # stopped being produced. This reads the same function on the interpreter
    # that *does* have executorch, which is the only one where the distinction
    # is observable.
    et = _qnn_et_fixture()
    if et is None:
        return _skip(
            "TORCHNATIVE_QNN_PYTHON unset -- the adaptor-naming half of this "
            "test cannot run without an interpreter that has executorch"
        )
    et_refusal = et["aot_refusal"]
    assert et_refusal, "the executorch interpreter reported no QNN AOT refusal"
    assert "PyQnnManagerAdaptor" in et_refusal, et_refusal
    assert "x86_64-linux-clang" in et_refusal, et_refusal
    assert "is_linux_x86" in et_refusal, et_refusal


def test_lowering_refuses_before_it_touches_the_module():
    """`lower()` must refuse where the AOT half is missing, and say so first.

    The failure this prevents is an `ImportError` surfacing from inside a
    partitioner pass eighteen frames down, after `torch.export` has already
    run. That traceback names a file in somebody else's package and tells the
    reader nothing about what to install.
    """
    if not _vendor_ready():
        return _skip("vendored tree has no _C.abi3.so")
    result = _qnn_shim_fixture()
    refusal = result["refuse_lower"]
    if result["aot_available"]:
        return _skip("this host can lower; there is nothing to refuse")
    assert refusal, "lower() did not refuse on a host with no QNN AOT half"
    assert not refusal.startswith("WRONG-TYPE"), refusal
    assert "cannot lower here" in refusal, refusal


def test_the_qnn_sdk_version_is_read_from_executorch_not_transcribed():
    """`qnn_sdk_version()` must come out of the installed package.

    docs/graph/DECOMP.md §12.6 named the trap for operator lists and this is the same
    trap one size smaller: a version number copied into this repository is
    wrong the first release after somebody bumps it, and nothing here would
    notice. The assertion is on the *shape* of the answer rather than its
    value, because pinning the value would be transcribing it in a test.
    """
    result = _qnn_et_fixture()
    if result is None:
        return _skip("TORCHNATIVE_QNN_PYTHON unset -- no upstream executorch")
    version = result["probe"]["qnn_sdk_version"]
    url = result["probe"]["qnn_sdk_url"]
    assert version, result["probe"]
    parts = version.split(".")
    assert len(parts) >= 3 and all(p.isdigit() for p in parts[:3]), version
    assert url and version in url, (version, url)
    assert "qualcomm.com" in url, url


def test_the_soc_table_is_executorchs_and_every_entry_has_an_htp_arch():
    """`soc_targets()` reads `QcomChipset` + `_soc_info_table`, never a copy.

    The check that it is not a copy is that the two sides agree *and* that the
    HTP architectures are the small closed set ExecuTorch's `HtpArch` declares.
    A hand-written table would drift from either.
    """
    result = _qnn_et_fixture()
    if result is None:
        return _skip("TORCHNATIVE_QNN_PYTHON unset -- no upstream executorch")
    targets = result["soc_targets"]
    assert len(targets) >= 20, len(targets)
    assert "SM8650" in targets and "SM8550" in targets, sorted(targets)
    for name, (soc_model, htp_arch) in targets.items():
        assert soc_model > 0, (name, soc_model)
        assert htp_arch in (68, 69, 73, 75, 79, 81), (name, htp_arch)
    assert result["resolve_sm8650"] == ["SM8650", 57, 75], result["resolve_sm8650"]
    refusal = result["refuse_unknown_soc"]
    assert refusal and "NOT-A-CHIPSET" in refusal, refusal
    assert "SM8650" in refusal, "the refusal should list what it would accept"


# ---------------------------------------------------------------------------
# The front end -- and this half runs under the shim.
# ---------------------------------------------------------------------------


def test_replacing_a_submodule_keeps_the_model_and_computes_the_same_thing():
    """`delegate_` swaps in place, returns the same object, and changes nothing.

    Two claims and the second is the one that can be wrong silently: the
    delegated model must compute what the undelegated one computes when the
    delegate forwards to the module it replaced. If it did not, every later
    numerical comparison would be measuring the harness.
    """
    if not _vendor_ready():
        return _skip("vendored tree has no _C.abi3.so")
    result = _qnn_shim_fixture()
    assert result["old_type"] == "Linear", result["old_type"]
    assert result["paths"] == ["0"], result["paths"]
    assert result["forward_shape"] == [3, 2], result["forward_shape"]
    assert result["calls"] == 1, result["calls"]
    assert result["same_object"] is True
    assert result["paths_after"] == ["0", "2"], result["paths_after"]
    assert result["max_abs_diff_vs_undelegated"] == 0.0, \
        result["max_abs_diff_vs_undelegated"]


def test_every_bad_plan_is_refused_by_name_and_leaves_the_model_alone():
    """Seven refusals, each naming the thing that is wrong.

    The last assertion is the one with teeth. `delegate_` validates the whole
    plan before it moves anything, so a plan with one good entry and one bad
    one leaves the model exactly as it was. A half-applied plan produces a
    model that runs, is neither the eager one nor the delegated one, and whose
    output says nothing about which it is.
    """
    if not _vendor_ready():
        return _skip("vendored tree has no _C.abi3.so")
    result = _qnn_shim_fixture()
    expected = {
        "refuse_bad_path": ["no submodule 'nope'", "Children there"],
        "refuse_deep_bad_path": ["'zzz'"],
        "refuse_index_range": ["9"],
        "refuse_empty_path": ["empty submodule path"],
        "refuse_not_delegate": ["not a _DelegateModule"],
        "refuse_not_module": ["nn.Module"],
        "refuse_empty_plan": ["empty plan"],
        "refuse_not_dict": ["must be a dict"],
        "refuse_nameless": ["backend_name"],
        "refuse_no_forward": ["has no forward"],
    }
    for key, needles in expected.items():
        message = result[key]
        assert message, f"{key} did not refuse"
        assert not message.startswith("WRONG-TYPE"), f"{key}: {message}"
        for needle in needles:
            assert needle in message, f"{key}: {needle!r} not in {message!r}"
    assert result["paths_unchanged_after_refusal"] is True


def test_a_real_checkpoint_still_generates_with_a_submodule_delegated():
    """The front-end claim, on SmolLM2-135M, with `generate()` actually run.

    Everything asserted here is the thing the architecture promises and the
    thing a wrapper class would break:

    * the object is a `LlamaForCausalLM`, so `isinstance` checks elsewhere in
      `transformers` and in user code still hold;
    * `model.generate` is `GenerationMixin.generate` **by identity**, not by
      name -- a wrapper that defined its own `generate` would pass a name check
      and fail this one;
    * `generate()` produces tokens, and the delegate was called once per decode
      step, so it is on the live path rather than bypassed;
    * the replaced weights are still in `state_dict()` under the delegate's
      own prefix, which is what makes the swap survive a save/load.

    Skips by name where SmolLM2-135M is not already in a local HF cache. It is
    never downloaded: a test suite that reaches the network is a test suite
    that fails for reasons that are not about the code.
    """
    if not _vendor_ready():
        return _skip("vendored tree has no _C.abi3.so")
    result = _qnn_model_fixture()
    if result is None:
        return _skip("SmolLM2-135M is not in a local HF cache")
    if result.get("transformers") is None and "import_error" in result:
        return _skip(f"no transformers under the shim: {result['import_error']}")
    if result.get("model") is None:
        return _skip(f"checkpoint would not load: {result.get('load_error')}")

    assert result["is_shim"] is True
    assert result["type"] == "LlamaForCausalLM", result["type"]
    assert result["is_llama_for_causal_lm"] is True
    assert result["generate_is_upstreams"] is True, \
        "generate() is not GenerationMixin.generate -- something is wrapping the model"
    assert result["has_config"] is True
    assert result["delegated"] == ["model.layers.0.mlp"], result["delegated"]
    assert result["delegate_type"] == "Recorder", result["delegate_type"]
    assert result["inner_type"] == "LlamaMLP", result["inner_type"]
    assert result["generated_shape"][0] == 1, result["generated_shape"]
    assert result["generated_shape"][1] > 6, result["generated_shape"]
    assert result["generated_text"].strip(), result["generated_text"]
    assert result["delegate_calls"] == 6, \
        f"delegate ran {result['delegate_calls']} times for 6 new tokens"
    assert result["state_dict_keeps_the_weights"] is True
    assert result["hidden_size"] == 576, result["hidden_size"]
    assert result["intermediate_size"] == 1536, result["intermediate_size"]


# ---------------------------------------------------------------------------
# The artefact -- structure, and the reader's yes as well as its no.
# ---------------------------------------------------------------------------


def test_the_shared_pipeline_produces_a_delegated_program_that_loads_and_runs():
    """The control. Same pipeline as the QNN path, XNNPACK instead of QNN.

    **This is not a QNN claim and the assertions say so**: the artefact's
    backend is `XnnpackBackend` and `is_qnn` is `False`. What it establishes is
    that `torch.export` -> `to_edge_transform_and_lower` -> `.to_executorch()`
    -> `Runtime.load_program` -> `execute` is wired correctly end to end, so a
    failure in the shared machinery is caught on a machine with no Qualcomm
    anything on it.
    """
    result = _qnn_et_fixture()
    if result is None:
        return _skip("TORCHNATIVE_QNN_PYTHON unset -- no upstream executorch")
    assert result["is_shim"] is False, "the executorch interpreter loaded the shim"
    report = result["control_report"]
    assert report["subgraphs"] >= 1, report
    assert report["delegated_nodes"] >= 4, report
    assert report["delegated_fraction"] > 0.5, report
    assert result["control_bytes"] > 0
    assert result["control_methods"] == ["forward"], result["control_methods"]
    assert result["control_backends"] == ["XnnpackBackend"], result["control_backends"]
    assert result["control_is_qnn"] is False
    assert result["control_shape"] == [2, 8], result["control_shape"]
    assert result["control_max_abs_diff"] < 1e-5, result["control_max_abs_diff"]

    refusal = result["refuse_htp_plan_on_control"]
    assert refusal and "no QnnBackend delegate" in refusal, refusal
    assert "XnnpackBackend" in refusal, refusal


def test_the_qnn_module_refuses_the_very_file_the_generic_one_runs():
    """The negative control for the whole narrowing.

    `ExecuTorchModule` loads the control artefact and returns the right
    numbers. If `QnnModule` did too, then `QnnModule` would be a synonym and
    every sentence in docs/devices/QNN.md that distinguishes "delegated" from
    "delegated to QNN" would be unfounded. So the same file must be **refused**,
    and the refusal must name the backends the artefact actually has.
    """
    result = _qnn_et_fixture()
    if result is None:
        return _skip("TORCHNATIVE_QNN_PYTHON unset -- no upstream executorch")
    refusal = result["refuse_qnn_module_on_control"]
    assert refusal, "QnnModule accepted an XNNPACK artefact"
    assert "no QnnBackend delegate" in refusal, refusal
    assert "XnnpackBackend" in refusal, refusal
    assert "docs/devices/QNN.md" in refusal, refusal


def test_the_artefact_reader_decodes_the_htp_plan_out_of_the_compile_spec():
    """`htp_plan()` reads SoC, HTP architecture, backend type and precision.

    Run against a **QNN-shaped** program: ExecuTorch's own program serialiser,
    ExecuTorch's own `QnnExecuTorchOptions` flatbuffer, and a delegate id of
    `QnnBackend` -- with an XNNPACK payload behind it. It is a fixture for the
    reader, not a QNN artefact, and it could not execute anywhere.

    It exists because a reader tested only against non-QNN files passes while
    answering False unconditionally. The values are asserted by number
    (`soc_model` 57, `htp_arch` 75) because those are the two fields a wrong
    decode moves without changing anything else.
    """
    result = _qnn_et_fixture()
    if result is None:
        return _skip("TORCHNATIVE_QNN_PYTHON unset -- no upstream executorch")
    assert result["shaped_backends"] == ["QnnBackend"], result["shaped_backends"]
    assert result["shaped_is_qnn"] is True
    plan = result["shaped_plan"]
    assert plan["backend_type"] == 2, plan          # kHtpBackend
    assert plan["backend_type_name"] == "kHtpBackend", plan
    assert plan["soc_model"] == 57, plan            # SM8650
    assert plan["htp_arch"] == 75, plan             # v75
    assert plan["precision"] == 1, plan             # kHtpFp16
    assert plan["precision_name"] == "kHtpFp16", plan


def test_an_artefact_built_for_the_wrong_silicon_is_refused_before_it_is_pushed():
    """`match_device` compares HTP architecture, and it refuses three ways.

    * **wrong architecture** -- SM8550 is v73 and the artefact is v75. QNN
      catches this at backend init and aborts the method load; catching it here
      costs an `adb getprop`.
    * **same architecture, different part** -- SA8255 is also v73... and
      SM8650's v75 artefact is refused for it too, for the same reason. But
      `SM8650` and any other v75 part are accepted, which is why the comparison
      is on the architecture rather than on the part number: refusing a
      combination that works teaches the reader to stop believing refusals.
    * **not the HTP at all** -- a `kGpuBackend` compile spec still produces a
      `QnnBackend` delegate that runs and returns the right answer, off the
      GPU. That is the fallback that looks most like success, so it is the one
      that must not slip through on a substring match.
    """
    result = _qnn_et_fixture()
    if result is None:
        return _skip("TORCHNATIVE_QNN_PYTHON unset -- no upstream executorch")
    ok, reason = result["match_sm8650"]
    assert ok is True, reason
    assert "v75" in reason, reason

    ok, reason = result["match_sm8550"]
    assert ok is False, reason
    assert "v73" in reason and "v75" in reason, reason
    assert "does not fall back" in reason, reason

    ok, reason = result["match_sm8750"]
    assert ok is False, reason          # v79

    ok, reason = result["match_sa8255_same_arch"]
    assert ok is False, reason          # v73

    ok, reason = result["match_gpu_backend"]
    assert ok is False, reason
    assert "not HTP" in reason, reason
    assert "kGpuBackend" in reason, reason


def test_a_malformed_or_foreign_artefact_is_refused_by_name():
    """Three faults shaped like real ones, each refused with the reason.

    A `QnnBackend` delegate with no compile spec is not a hypothetical: it is
    what a partial or hand-edited artefact looks like, and it decodes perfectly
    as a program. The refusal has to come from the QNN layer, since nothing in
    ExecuTorch's own schema objects to it.
    """
    result = _qnn_et_fixture()
    if result is None:
        return _skip("TORCHNATIVE_QNN_PYTHON unset -- no upstream executorch")
    refusal = result["refuse_no_compile_spec"]
    assert refusal and "qnn_compile_spec" in refusal, refusal
    assert "not written by the QNN partitioner" in refusal, refusal

    refusal = result["refuse_junk"]
    assert refusal and "not an ExecuTorch program" in refusal, refusal

    refusal = result["refuse_qnn_runtime"]
    assert refusal and "not registered in this ExecuTorch runtime" in refusal, refusal
    assert "QnnBackend" not in result["runtime_backends"], result["runtime_backends"]

    refusal = result["refuse_inspect_forward"]
    assert refusal and "require_runtime=False" in refusal, refusal
    assert "inspection handle" in refusal, refusal


# ---------------------------------------------------------------------------
# The numerical claim, at a tolerance nobody chose.
# ---------------------------------------------------------------------------


def test_the_delegated_submodule_agrees_with_upstream_at_a_derived_tolerance():
    """SmolLM2-135M's layer-0 MLP, lowered and executed, against upstream eager.

    The tolerance is docs/numerics/AGREE.md §2's, not a number chosen here. Upstream
    runs the *same* submodule in float64 and the float32 answer is scored
    against it; the tolerance is the p90 of that error floored at 8 ulp, and
    its defence is that anything tighter would have to call upstream wrong on a
    tenth of its own samples.

    The second rule from that section is applied too, and it is the informative
    one: a difference is not a defect if it is within 4x upstream's own
    distance from the float64 truth for the same output. Here the ratio comes
    out well **below** one -- the artefact's answer is nearer the float64 truth
    than upstream's own float32 path is -- which is what two independent
    float32 truncation orders look like, and is the same shape docs/numerics/AGREE.md
    §3 measured across 285 architectures.

    **What this cannot mean.** This is the *XNNPACK* artefact, on this Mac's
    CPU, in float32. It says the export/lower/serialise/load/execute pipeline
    preserves the function. It says **nothing** about QNN, whose HTP path is
    `kHtpFp16` or `kHtpQuantized` -- docs/devices/QNN.md §7 is why a float32 agreement
    and an HTP execution are two different claims that cannot be made on one
    run, exactly as docs/graph/NPU2.md §1.1 found for the Neural Engine.
    """
    result = _qnn_et_fixture()
    if result is None:
        return _skip("TORCHNATIVE_QNN_PYTHON unset -- no upstream executorch")
    if result.get("real") is None:
        return _skip(
            "SmolLM2-135M unavailable to the executorch interpreter"
            + (f": {result['real_error']}" if result.get("real_error") else "")
        )
    real = result["real"]
    assert real["submodule_type"] == "LlamaMLP", real["submodule_type"]
    assert real["hidden"] == 576 and real["intermediate"] == 1536, real
    assert real["backends"] == ["XnnpackBackend"], real["backends"]
    assert real["report"]["delegated_nodes"] >= 6, real["report"]
    assert real["report"]["subgraphs"] >= 1, real["report"]

    rows = real["rows"]
    assert len(rows) == 5, rows
    oracles = sorted(row["oracle"] for row in rows)
    # p90 of upstream's own float32-vs-float64 error, floored at 8 ulp.
    index = min(len(oracles) - 1, int(round(0.9 * (len(oracles) - 1))))
    tolerance = max(oracles[index], 8 * real["eps"])
    assert tolerance > 0, oracles

    worst = max(row["rel"] for row in rows)
    assert worst <= tolerance, (
        f"worst relative error {worst:.3e} exceeds the derived tolerance "
        f"{tolerance:.3e} (p90 of upstream's own f32-vs-f64 error "
        f"{oracles[index]:.3e}, floor 8 ulp {8 * real['eps']:.3e})"
    )
    worst_ratio = max(row["ratio"] for row in rows if row["ratio"] is not None)
    assert worst_ratio <= 4.0, (
        f"worst oracle ratio {worst_ratio:.2f} exceeds docs/numerics/AGREE.md §2's 4x "
        "rule -- the artefact is further from the float64 truth than upstream "
        "is by more than that factor"
    )
    # And the check must be able to fail: upstream's own error has to be a real
    # measurement, not a zero that makes the tolerance vacuous.
    assert min(oracles) > 0.0, (
        "upstream's float32-vs-float64 error came out exactly 0 on some "
        "sample, which would make the derived tolerance the 8-ulp floor and "
        "the derivation decorative"
    )


# ---------------------------------------------------------------------------
# The device half -- none of which ran, and the module says so rather than
# guessing.
# ---------------------------------------------------------------------------


def test_the_device_module_refuses_to_guess_which_device_to_use():
    """`ANDROID_SERIAL` or nothing. The devices on this machine are shared.

    docs/graph/NPU2.md's `nnapi_device` earned this rule and this module inherits it
    verbatim. The refusal has to name the variable, because "no device" reads
    like "unplug and replug" and the actual fix is one `export`.
    """
    if not _vendor_ready():
        return _skip("vendored tree has no _C.abi3.so")
    result = _qnn_shim_fixture()
    assert result["device_serial_none"] is True

    # This test used to assert `device_adb_available is False`, which made it
    # pass only on a machine WITHOUT the Android SDK on PATH -- the case where
    # the rule cannot be exercised at all. The interesting case is the
    # opposite one: adb present, ANDROID_SERIAL absent, a shared device
    # sitting there for `adb` to pick.
    #
    # The branch below reads `device_adb_on_path`, NOT `device_adb_available`.
    # The latter is a compound -- adb on PATH *and* a serial chosen -- and the
    # fixture pops ANDROID_SERIAL before asking, so it is False on every
    # machine and branching on it would make the strong path unreachable.
    refusal = result["refuse_no_serial"]
    assert refusal, "adb ran with no ANDROID_SERIAL -- it would have picked one"
    report = result["device_report_unreachable"]
    assert report["reachable"] is False
    assert report["serial"] is None

    if not result["device_adb_on_path"]:
        assert "adb" in refusal, refusal
        return _skip(
            "no adb on PATH, so the ANDROID_SERIAL half of this rule was not "
            "exercised -- it refused for the other reason, by name"
        )

    assert "ANDROID_SERIAL" in refusal, refusal
    assert "shared" in refusal, refusal
    assert "ANDROID_SERIAL" in report["reason"], report


def test_the_device_module_writes_only_under_its_own_directory():
    """One path, named, and it is not the one `nnapi_device` owns.

    Two modules writing to one directory on a shared device is how one round's
    cleanup deletes another round's staged libraries.
    """
    if not _vendor_ready():
        return _skip("vendored tree has no _C.abi3.so")
    result = _qnn_shim_fixture()
    assert result["device_dir"] == "/data/local/tmp/bw_qnn", result["device_dir"]
    assert result["device_dir"] != "/data/local/tmp/bw_device"


def test_the_htp_stub_names_are_derived_from_the_architecture_not_listed():
    """`htp_stub_for` builds the pair from the arch number and refuses a guess.

    The stub/skel pair is per HTP generation, and a stale list here would
    silently omit the newest silicon -- the same reason `soc_targets()` reads
    ExecuTorch's table. Refusing `None` matters because `None` is exactly what
    `device_soc()` returns for a part QNN does not recognise, and pushing a
    default pair for an unknown device stages files nothing will open.
    """
    if not _vendor_ready():
        return _skip("vendored tree has no _C.abi3.so")
    result = _qnn_shim_fixture()
    assert result["stub_for_75"] == ["libQnnHtpV75Stub.so", "libQnnHtpV75Skel.so"], \
        result["stub_for_75"]
    refusal = result["refuse_stub_for_none"]
    assert refusal and "not an HTP architecture number" in refusal, refusal
    assert "invent" in refusal, refusal
    assert result["soc_properties"][0] == "ro.soc.model", result["soc_properties"]


def test_an_absent_soc_table_is_not_reported_as_an_unrecognised_chipset():
    """The defect real hardware found. docs/devices/QNN.md §5.1.

    The first version of `device_soc` returned `chipset=None` for both "the
    device reported something ExecuTorch's table does not have" and "this
    interpreter cannot read the table at all". Run against a Galaxy Tab S9
    Ultra whose `ro.soc.model` is **`SM8550`** -- a chipset ExecuTorch knows,
    HTP v73 -- from an interpreter without executorch, it reported the SoC as
    unrecognised. That is a refusal about the *host* wearing a verdict about
    the *device*, and the two lead a reader to opposite actions: one says
    `pip install executorch`, the other says the silicon will never work
    (pytorch/executorch#16465, where QNN's own detection fails at runtime with
    "No Snapdragon SOC detected").

    Both interpreters are asked the same question about the same string, and
    the answers must differ. Neither half can catch this alone -- which is why
    the first version passed everything.
    """
    if not _vendor_ready():
        return _skip("vendored tree has no _C.abi3.so")
    shim = _qnn_shim_fixture()
    known, not_in_table, unavailable = shim["soc_status_values"]
    assert len({known, not_in_table, unavailable}) == 3, shim["soc_status_values"]

    # No executorch here, so a chipset that IS in the table still cannot be
    # resolved -- and the status has to say which of the two reasons it is.
    prop, raw, status, chipset, soc_model, htp_arch = shim["soc_known_chipset"]
    assert raw == "SM8550", shim["soc_known_chipset"]
    assert status == unavailable, (
        "an interpreter with no executorch reported SM8550 as "
        f"{status!r} -- that is a statement about the device, and the true "
        "statement is about this host"
    )
    assert (chipset, soc_model, htp_arch) == (None, None, None)

    refusal = shim["refuse_no_soc_property"]
    assert refusal and "ro.soc.model" in refusal, refusal
    assert "guessing a chipset" in refusal, refusal

    et = _qnn_et_fixture()
    if et is None:
        return _skip(
            "TORCHNATIVE_QNN_PYTHON unset -- the half of this test that shows "
            "the same string resolving cannot run"
        )
    prop, raw, status, chipset, soc_model, htp_arch = et["soc_known_chipset"]
    assert (raw, status, chipset) == ("SM8550", known, "SM8550"), et["soc_known_chipset"]
    assert (soc_model, htp_arch) == (43, 73), et["soc_known_chipset"]

    # Case is not guaranteed: `ro.soc.model` is OEM-populated.
    _, raw, status, chipset, _, htp_arch = et["soc_lowercase_chipset"]
    assert (raw, status, chipset, htp_arch) == ("sm8650", known, "SM8650", 75), \
        et["soc_lowercase_chipset"]

    # And the *other* status, on an interpreter that really did consult the
    # table and really did not find the part.
    _, raw, status, chipset, _, _ = et["soc_unknown_chipset"]
    assert (raw, status, chipset) == ("CQ8750S", not_in_table, None), \
        et["soc_unknown_chipset"]


def test_the_fastrpc_check_stats_named_paths_rather_than_listing_dev():
    """`ls /dev` returns nothing to the `shell` user; a stat by name works.

    Measured on Android 16 / API 36: `ls /dev/ | grep rpc` produced an empty
    listing on a device where `ls -l /dev/adsprpc-smd` succeeds. A check built
    on the directory listing would conclude "no DSP" on a device that has one,
    which is a false negative about the only piece of hardware this whole
    round is about.
    """
    if not _vendor_ready():
        return _skip("vendored tree has no _C.abi3.so")
    result = _qnn_shim_fixture()
    nodes = result["fastrpc_nodes_listed"]
    assert "/dev/adsprpc-smd" in nodes, nodes
    assert all(n.startswith("/dev/") for n in nodes), nodes


def test_no_claim_is_made_that_anything_ran_on_an_npu():
    """The claim this round does **not** make, pinned so it cannot drift.

    docs/devices/QNN.md §6.4. No QNN artefact was produced (the host cannot), no
    Snapdragon device was attached, and `QnnBackend` is not registered in any
    ExecuTorch runtime reachable from here. So there is no run to attribute,
    and this test asserts the *absence* rather than leaving it to prose --
    docs/graph/NPU2.md is what a round costs when "executed" and "executed on the
    NPU" are allowed to blur.

    If a future round lands the device half, this goes red and the sentence it
    guards gets rewritten deliberately instead of inherited.
    """
    result = _qnn_et_fixture()
    if result is None:
        if not _vendor_ready():
            return _skip("vendored tree has no _C.abi3.so")
        shim = _qnn_shim_fixture()
        assert shim["aot_available"] is False, \
            "this host can lower to QNN -- docs/devices/QNN.md §6.4 needs rewriting"
        return
    assert "QnnBackend" not in result["runtime_backends"], result["runtime_backends"]
    assert result["control_backends"] == ["XnnpackBackend"], result["control_backends"]
    assert result["probe"]["qnn_backend_registered"] is False
    assert result["probe"]["qnn_aot_available"] is False, (
        "the QNN ahead-of-time half became available on this host -- "
        "docs/devices/QNN.md §3 and §6.4 both need re-measuring"
    )


# --------------------------------------------------------- the withdrawal

#: What `torchnative.export.npu` used to export as the way to run a model.
#: Written out here rather than read off `npu._WITHDRAWN`, so that quietly
#: restoring one of them makes this test fail rather than agree (CLAUDE.md §5.5).
WITHDRAWN_NPU = (
    "NpuModelForCausalLM",
    "delegate_",
    "delegated_paths",
    "DelegateModule",
    "replace_submodule",
    "resolve_submodule",
)


def test_every_withdrawn_npu_name_refuses_by_name_and_names_the_replacement():
    """Reaching for a withdrawn name says so, why, and what to use instead.

    Refusal at *attribute access*, not at call: `NpuModelForCausalLM` was
    reached as `NpuModelForCausalLM.from_pretrained(...)`, so anything that let
    the attribute resolve would have to grow a fake `from_pretrained` to refuse
    at all.
    """
    from torchnative.export import npu

    for name in WITHDRAWN_NPU:
        try:
            getattr(npu, name)
        except npu.DelegateWithdrawn as exc:
            # Bound to a second name deliberately: Python DELETES the `as`
            # target when the except block ends, so reading `exc` below would
            # raise UnboundLocalError -- which is what it did, and the test
            # then failed for its own bug rather than for the refusal.
            raised, text = exc, str(exc)
        else:
            raise AssertionError(f"npu.{name} did not refuse")
        assert text.startswith(f"torchnative npu: {name} was withdrawn"), text
        assert "AutoModelForCausalLM" in text, (name, text)
        assert "not implemented yet" in text, (name, text)
        assert isinstance(raised, npu.DelegateRefused), name
    print(
        f"ok   npu: {len(WITHDRAWN_NPU)} withdrawn names refuse by name, naming "
        f"AutoModelForCausalLM and saying it does not exist yet"
    )


def test_the_withdrawn_npu_message_does_not_imply_a_working_device_string():
    from torchnative.export import npu

    text = npu._withdrawal_message("delegate_")
    assert 'model.to("npu")' in text, text
    assert "torch.device" in text and "raises" in text, text
    assert "_rename_privateuse1_backend" in text, text
    print("ok   npu: the withdrawal says `npu` is not a device on this shim either")


def test_the_swapping_plumbing_is_kept_private_and_qnn_still_builds_on_it():
    """Withdrawing the API did not delete the mechanism -- `qnn` needs it.

    If this fails with an AttributeError the withdrawal went too far; if it
    fails because a public name resolved, it did not go far enough.
    """
    from torchnative.export import npu

    for name in ("_DelegateModule", "_delegate_", "_delegated_paths",
                 "_replace_submodule", "_resolve_submodule"):
        assert getattr(npu, name) is not None, name
    assert npu.__all__ == ["DelegateRefused", "DelegateWithdrawn"], npu.__all__
    print("ok   npu: the plumbing survives privately; only the API surface went")


def test_an_unknown_npu_name_is_still_a_plain_attribute_error():
    from torchnative.export import npu

    try:
        npu.no_such_name_at_all
    except npu.DelegateWithdrawn as exc:  # noqa: BLE001
        raise AssertionError(f"a typo was reported as a withdrawal: {exc}")
    except AttributeError:
        pass
    else:
        raise AssertionError("a missing attribute did not raise")
    print("ok   npu: an unknown name is an AttributeError, not a withdrawal")



def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
