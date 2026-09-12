"""The first real checkpoint the CoreML arm met, and what it cost to take it.

docs/graph/NPU2.md §7 landed `to(torchnative.device.npu)` for two leaf types
and measured the Neural Engine actually running them. Every module it was
measured on was hand-built here, in float32. The obvious next thing --

    m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M",
                                             dtype="auto")
    m.to(torchnative.device.npu)

-- raised `CoreMLRefused: no numpy dtype mapped for torch.bfloat16`. Modern
Hugging Face checkpoints are overwhelmingly bfloat16 and `dtype="auto"` is the
spelling the documentation teaches, so the path worked on everything it was
built against and refused the first real model it was pointed at. `float16`
was missing from the same map, for no reason at all.

**The conversion is a widening and it is exact.** numpy has no bfloat16 and
`ml_dtypes` is not a dependency here, so a bf16 numpy array cannot be produced
at all; and `torch._C._shim_tensor_bytes` refuses *both* half-width floats by
name, so even float16 -- which numpy does have -- cannot cross as its own
bytes. bf16 and f16 are both strict subsets of float32, so widening to float32
moves no value, keeps the byte route that removed the shutdown segfault
(docs/graph/NPU2.md §7.3), and leaves the float16 narrowing where it already
was: in `ct.convert(compute_precision=FLOAT16)`. Narrowing here instead would
round twice, and would be wrong outright for `precision="float32"`.

What that narrowing costs on this checkpoint was measured rather than reasoned
about. bf16 and f16 are the same width split differently -- 8 exponent bits
and 7 of mantissa against 5 and 10 -- so f16 *gains* mantissa and loses range,
and the only error is at the two ends. Over all 134,515,008 SmolLM2-135M
weights: zero overflow to inf, 60 elements flush to zero, worst absolute
error 2.98e-08. That is under the float32 tolerance, so neither grade moves.

**And then the model lowers, runs, and a decode step still does not reach the
unit.** All 211 `Linear`s swap, the whole model forwards through CoreML and
picks the same next token as its own eager forward -- and `MLComputePlan`
prefers the **CPU** for every one of those 211 leaves at batch 1. Four of the
model's five distinct Linear shapes reach the Neural Engine at a
prefill-shaped batch of 128; `lm_head` (576->49152) does not reach it at 128
either. That is the finding, not a defect, and it is asserted here so that
nobody reads "SmolLM2 lowered" as "SmolLM2 ran on the Neural Engine".

**One defect was found and is not fixed here.** `_CoreMLLinear` compiles
lazily, on first use, keyed by the shape that arrived; a real forward is the
first thing that supplies a real shape. Calling `coremltools.convert` from
*inside* the model's forward pass **segfaults the interpreter**, reproducibly,
at the seventh leaf -- with about 1 GB resident, so not memory, and the same
seven leaves compiled and run outside a forward are fine. The `to()`-time
eager probe does not cover it: it compiles at batch 1 and a transformer's
forward is never batch 1. The fixture below therefore compiles at the real
shape before forwarding, and says so where it does it.

Skips say by name what is missing, for docs/devices/VULKAN3.md §6.1's reason.
"""

import glob
import json
import os
import struct

from test_shim import _CKPT_VENDOR_SHIM, _npu_fixture


#: The motivating checkpoint, read from the Hugging Face cache and never
#: downloaded: a gate that reaches the network is a gate that fails for a
#: reason that is not about this repository.
_SMOL = "HuggingFaceTB/SmolLM2-135M"


_BF16_SCRIPT = r"""
import json
import os
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}

try:
    import coremltools as ct
    out["coremltools"] = ct.__version__
except Exception as error:
    out["coremltools"] = None
    out["import_error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(out))
    raise SystemExit(0)

import numpy as np

import torchnative
from torchnative import device as tn_device
from torchnative.export import coreml as C
from torchnative.export.decompose import DecomposedTrace


def capture(fn, *inputs):
    torch._C._capture_begin(list(inputs))
    with torch.no_grad():
        produced = fn(*inputs)
    produced = produced if isinstance(produced, (list, tuple)) else [produced]
    trace = torch._C._capture_end(list(produced))
    return DecomposedTrace(
        trace.guards, trace.constants, trace.constant_values,
        trace.nodes, trace.outputs,
    )


_OURS = ("torchnative ", "nn.Module.to(")


def ours(caught):
    return [str(w.message) for w in caught if str(w.message).startswith(_OURS)]


resolution = tn_device.npu.resolve()
out["resolution"] = {"backend": resolution.backend,
                     "unit": resolution.unit, "source": resolution.source}

try:
    # -- 1. the marshalling question, on its own ---------------------------
    #
    # Deliberately values that make the widening's exactness checkable by
    # hand: each is representable in bfloat16, so the float32 array must
    # carry them with no error whatsoever.
    exact = [1.5, -2.25, 3.0, 0.125, -64.0]
    out["marshal"] = {}
    for label, dt in (("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        entry = {}
        t = torch.tensor(exact).to(dt)
        try:
            arr = C._np(t)
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        else:
            entry["dtype"] = str(arr.dtype)
            entry["values"] = [float(v) for v in arr]
            # The identity that makes "the grade cannot move" a fact rather
            # than a hope: the array from the half-width tensor is the array
            # from its float32 widening, bit for bit.
            widened = C._np(t.to(torch.float32))
            entry["identical_to_widened"] = bool(np.array_equal(arr, widened))
        out["marshal"][label] = entry

    # The two values that tell a widening from a narrowing. Both are
    # representable in bfloat16 and neither is representable in float16 --
    # 1e20 is above f16's largest finite value and 1e-10 is below its
    # smallest subnormal -- so a `_np` that routed bfloat16 through float16
    # would hand back `inf` and `0.0` here while still agreeing on every
    # ordinary number.
    out["range"] = {}
    try:
        t = torch.tensor([1e20, -1e20, 1e-10]).to(torch.bfloat16)
        arr = C._np(t)
        out["range"] = {
            "dtype": str(arr.dtype),
            "finite": bool(np.isfinite(arr).all()),
            "nonzero": bool((arr != 0).all()),
            "values": [float(v) for v in arr],
        }
    except Exception as error:
        out["range"] = {"error": f"{type(error).__name__}: {error}"}

    # -- 2. a bfloat16 Linear at SmolLM2's shape, lowered ------------------
    #
    # 576 -> 1536 is SmolLM2-135M's MLP up/gate projection. The two batches
    # are the two regimes a transformer actually runs in: 1 is a decode step,
    # 128 is a prefill chunk.
    w = torch.randn(1536, 576).to(torch.bfloat16)
    b = torch.randn(1536).to(torch.bfloat16)
    out["bf16_linear"] = {}
    for batch in (1, 128):
        leaf = C._CoreMLLinear(w, b, precision="float16",
                               compute_units=ct.ComputeUnit.ALL)
        leaf._compile_for(batch, probe=True)
        rows = C.computes(leaf._report["plans"][-1]["rows"])
        out["bf16_linear"][str(batch)] = {
            "preferred": sorted({r["preferred"] for r in rows}),
            "supported": sorted({d for r in rows for d in r["supported"]}),
            "ops": [r["op"] for r in rows],
        }
    # It runs, from a bfloat16 input, and hands a bfloat16 tensor back.
    leaf = C._CoreMLLinear(w, b, precision="float16",
                           compute_units=ct.ComputeUnit.ALL)
    x = torch.randn(1, 576).to(torch.bfloat16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y = leaf(x)
    out["bf16_forward"] = {
        "dtype": str(y.dtype),
        "shape": list(y.shape),
        "finite": bool(np.isfinite(C._np(y)).all()),
        "warnings": ours(caught),
    }

    # -- 3. the agreement grades, through coreml.verify and no other ------
    #
    # `verify` grades a *trace*, and `compile_model`'s input specs are
    # float32 by contract, so what is graded here is the float32 widening --
    # which is the point: section 1 has already established that the widening
    # is the identity on the values, so this is the bf16 checkpoint's grade.
    lin = torch.nn.Linear(576, 1536).eval()
    xf = torch.randn(128, 576)
    trace = capture(lin, xf)
    out["verify"] = {}
    for label, float32 in (("float16", False), ("float32", True)):
        out["verify"][label] = C.verify(
            trace, [xf], float32=float32, tolerance=1.0,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

except Exception as error:  # noqa: BLE001
    import traceback
    out["bf16_error"] = traceback.format_exc()

print(json.dumps(out))
"""


#: The real model, in a **subprocess of its own**, and that is not tidiness.
#:
#: Running the marshalling checks, a compiled leaf and two `verify` calls in
#: the same process as `from_pretrained` reproducibly **segfaults inside
#: `coremltools.converters.convert`** -- three stages that each pass alone.
#: docs/graph/NPU2.md §7.3 recorded the same shape of failure for a different
#: cause, and the lesson there applies here: a fixture that crashes after
#: printing its JSON makes every claim in it, and one that crashes before it
#: reports nothing about the thing it was actually measuring. So the model
#: gets a clean interpreter.
_SMOL_SCRIPT = r"""
import json
import os
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}

try:
    import coremltools as ct
    out["coremltools"] = ct.__version__
except Exception as error:
    out["coremltools"] = None
    out["import_error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(out))
    raise SystemExit(0)

import numpy as np

import torchnative
from torchnative import device as tn_device
from torchnative.export import coreml as C

_OURS = ("torchnative ", "nn.Module.to(")


def ours(caught):
    return [str(w.message) for w in caught if str(w.message).startswith(_OURS)]


resolution = tn_device.npu.resolve()
out["resolution"] = {"backend": resolution.backend,
                     "unit": resolution.unit, "source": resolution.source}

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", dtype="auto").eval()
    out["checkpoint_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})

    ids = tokeniser("The capital of France is", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        reference = model(ids).logits
    out["reference"] = {
        "shape": list(reference.shape),
        "argmax": int(reference[0, -1].argmax()),
        "token": tokeniser.decode([int(reference[0, -1].argmax())]),
        "scale": float(np.abs(C._np(reference.to(torch.float32))).max()),
    }

    # `eager=False`: the probe compiles every lowered leaf at batch 1, and
    # this model has 211 of them at a batch its forward will never use. The
    # probe's own guarantee is measured on models where it is cheap
    # (test_anepath, test_coremlops); what is measured here is the shape the
    # forward below actually runs at.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.to(tn_device.npu, eager=False)
    report = model.torchnative_offload
    out["smol"] = {
        "warnings": ours(caught),
        "n_swapped": len(report["swapped"]),
        "left_on_cpu": report["left_on_cpu"],
        "skipped": report["skipped"][:4],
        "n_skipped": len(report["skipped"]),
        "deferred": report["deferred"],
        "fully_offloaded": report["fully_offloaded"],
        "fraction_moved": report["fraction_moved"],
        "parameters_moved": report["parameters_moved"],
        "parameters_total": report["parameters_total"],
        "precision": report["precision"],
        "leaf_types": sorted({type(m).__name__
                              for m in model.modules()
                              if not list(m.children())}),
    }

    leaves = [leaf for _, leaf in model.named_modules()
              if type(leaf).__name__ == "_CoreMLLinear"]

    # -- MLComputePlan per distinct shape, decode batch against prefill ----
    seen = {}
    for name, leaf in model.named_modules():
        if type(leaf).__name__ != "_CoreMLLinear":
            continue
        seen.setdefault((leaf.in_features, leaf.out_features), (name, leaf))
    out["smol_plans"] = {}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        for (in_f, out_f), (name, leaf) in sorted(seen.items()):
            entry = {"example": name}
            for batch in (1, 128):
                leaf._compile_for(batch, probe=True)
                rows = C.computes(leaf._report["plans"][-1]["rows"])
                entry[str(batch)] = sorted({r["preferred"] for r in rows})
            out["smol_plans"][f"{in_f}->{out_f}"] = entry

    # -- and then it runs, all 211 leaves, on the real prompt --------------
    #
    # Compiled here, **outside** the forward, and that is load-bearing rather
    # than an optimisation: calling `ct.convert` from inside the model's
    # forward pass segfaults this interpreter at the seventh leaf, reproducibly
    # (see this file's docstring). `_CoreMLLinear` compiles lazily on first
    # use, so a forward at a shape no leaf has seen does exactly that. Doing
    # it in advance is the only way this fixture can report the executed
    # result at all, and the defect is recorded rather than hidden.
    batch = int(ids.shape[0]) * int(ids.shape[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for leaf in leaves:
            leaf._compile_for(batch, probe=True)
    with torch.no_grad():
        produced = model(ids).logits
    difference = float(np.abs(
        C._np(produced.to(torch.float32)) - C._np(reference.to(torch.float32))
    ).max())
    out["executed"] = {
        "batch": batch,
        "n_leaves": len(leaves),
        "shape": list(produced.shape),
        "dtype": str(produced.dtype),
        "argmax": int(produced[0, -1].argmax()),
        "token": tokeniser.decode([int(produced[0, -1].argmax())]),
        "max_abs_logit_diff": difference,
        "finite": bool(np.isfinite(C._np(produced.to(torch.float32))).all()),
    }
except Exception as error:  # noqa: BLE001
    import traceback
    out["smol_error"] = traceback.format_exc()

print(json.dumps(out))
"""


_CACHE = {}


def _fixture_or_skip():
    """The fixture, or `None` with the reason printed by name."""
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return None
    if not _checkpoint_path():
        print(f"   (skipped: {_SMOL} is not in the Hugging Face cache)")
        return None
    if "f" not in _CACHE:
        _CACHE["f"] = _npu_fixture(_BF16_SCRIPT)
    result = _CACHE["f"]
    if result["coremltools"] is None:
        print("   (skipped: coremltools not installed for this interpreter)")
        return None
    if result.get("resolution", {}).get("backend") != "coreml":
        print("   (skipped: this host's npu does not resolve to the coreml "
              "backend)")
        return None
    if "bf16_error" in result:
        raise AssertionError(
            "the bfloat16 fixture raised; this is a failure and not a "
            "skip:\n" + result["bf16_error"]
        )
    return result


def _smol_or_skip():
    """The real-model fixture, in its own interpreter, or `None` by name."""
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return None
    if not _checkpoint_path():
        print(f"   (skipped: {_SMOL} is not in the Hugging Face cache)")
        return None
    if "s" not in _CACHE:
        _CACHE["s"] = _npu_fixture(_SMOL_SCRIPT)
    result = _CACHE["s"]
    if result["coremltools"] is None:
        print("   (skipped: coremltools not installed for this interpreter)")
        return None
    if result.get("resolution", {}).get("backend") != "coreml":
        print("   (skipped: this host's npu does not resolve to the coreml "
              "backend)")
        return None
    if "smol_error" in result:
        raise AssertionError(
            "the SmolLM2 fixture raised; this is a failure and not a "
            "skip:\n" + result["smol_error"]
        )
    return result


def _checkpoint_path():
    """The cached `model.safetensors` for `_SMOL`, or `None`."""
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    pattern = os.path.join(
        home, "hub", "models--" + _SMOL.replace("/", "--"),
        "snapshots", "*", "model.safetensors")
    found = sorted(glob.glob(pattern))
    return found[0] if found else None


def _checkpoint_tensors():
    """Every tensor in the checkpoint as `(name, dtype, exact float32 array)`.

    Read with `numpy` and `struct` alone, deliberately. The claim being made
    is about the *checkpoint's* numbers, and reading them through the library
    under test would let a marshalling defect agree with itself. bfloat16's
    bit pattern is the top half of the float32 one, so `u16 << 16` is an exact
    widening and needs nothing that knows what bfloat16 is.
    """
    import numpy as np

    path = _checkpoint_path()
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
        body = handle.read()
    for name, entry in sorted(header.items()):
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        raw = np.frombuffer(body[start:end], dtype=np.uint16)
        yield name, entry["dtype"], (raw.astype(np.uint32) << 16).view(np.float32)


def test_the_motivating_checkpoint_really_is_bfloat16_end_to_end():
    """The premise, checked rather than assumed.

    If SmolLM2 were stored in float32 this whole file would be about a
    refusal nobody hits.
    """
    path = _checkpoint_path()
    if path is None:
        print(f"   (skipped: {_SMOL} is not in the Hugging Face cache)")
        return
    dtypes = {dtype for _, dtype, _ in _checkpoint_tensors()}
    assert dtypes == {"BF16"}, dtypes


def test_no_smollm2_weight_leaves_float16s_range_so_the_cast_loses_no_value():
    """The measured half of the conversion decision.

    bfloat16 and float16 are the same width split differently: 8 exponent
    bits and 7 of mantissa against 5 and 10. So float16 **gains** mantissa
    and **loses range**, which means a bf16 value inside f16's normal range
    converts with no error at all and everything that can go wrong is at the
    two ends: above 65504 to `inf`, below 5.96e-08 to zero.

    Measured, not assumed, because "assume either way" is exactly what this
    round was told not to do. The numbers are asserted as bounds rather than
    printed, so a future checkpoint that does leave the range goes red here
    rather than silently producing infinities inside `ct.convert`.
    """
    import numpy as np

    path = _checkpoint_path()
    if path is None:
        print(f"   (skipped: {_SMOL} is not in the Hugging Face cache)")
        return

    F16_MAX = 65504.0
    F16_MIN_SUBNORMAL = 5.960464477539063e-08
    total = overflow = flushed = 0
    biggest = 0.0
    worst_abs = 0.0
    for _, _, values in _checkpoint_tensors():
        magnitude = np.abs(values)
        total += magnitude.size
        biggest = max(biggest, float(magnitude.max()))
        overflow += int((magnitude > F16_MAX).sum())
        nonzero = magnitude[magnitude > 0]
        flushed += int((nonzero < F16_MIN_SUBNORMAL).sum())
        roundtrip = values.astype(np.float16).astype(np.float32)
        worst_abs = max(worst_abs, float(np.abs(roundtrip - values).max()))

    assert total == 134515008, total
    assert overflow == 0, f"{overflow} weights would become inf in float16"
    assert biggest < F16_MAX, biggest
    assert biggest > 1.0, biggest          # not a degenerate all-tiny checkpoint
    assert flushed < total // 1000000, flushed
    # The whole cost of the narrowing, and it is below `verify`'s float32 bar.
    assert worst_abs < 2e-5, worst_abs
    assert worst_abs <= F16_MIN_SUBNORMAL, worst_abs


def test_a_bfloat16_tensor_reaches_numpy_as_an_exact_float32_array():
    """`_np` widens rather than refusing, and the widening moves no value."""
    r = _fixture_or_skip()
    if r is None:
        return
    entry = r["marshal"]["bfloat16"]
    assert "error" not in entry, entry
    assert entry["dtype"] == "float32", entry
    assert entry["values"] == [1.5, -2.25, 3.0, 0.125, -64.0], entry


def test_float16_is_in_the_map_too_and_is_not_a_second_refusal():
    """float16 was missing for no reason: numpy has it, and the same
    `_shim_tensor_bytes` refusal applies to it, so the same widening answers
    it. A user with an f16 checkpoint should not meet the bf16 wall."""
    r = _fixture_or_skip()
    if r is None:
        return
    entry = r["marshal"]["float16"]
    assert "error" not in entry, entry
    assert entry["dtype"] == "float32", entry
    assert entry["values"] == [1.5, -2.25, 3.0, 0.125, -64.0], entry


def test_the_widening_is_the_identity_so_no_agreement_grade_can_move():
    """The array from a half-width tensor **is** the array from its float32
    widening, bit for bit. That is what makes "adding a bfloat16 source does
    not move the grade" a fact about the program handed to CoreML rather than
    a tolerance argument."""
    r = _fixture_or_skip()
    if r is None:
        return
    for label in ("bfloat16", "float16"):
        assert r["marshal"][label]["identical_to_widened"] is True, r["marshal"]


def test_the_widening_keeps_bfloat16s_range_which_a_float16_route_would_lose():
    """Why the conversion is to float32 and not to float16.

    float16 is the precision the Neural Engine runs, so converting straight
    to it looks free. It is not: bfloat16 and float16 are the same width split
    differently, and float16's exponent is three bits shorter. 1e20 and 1e-10
    are ordinary bfloat16 numbers and are `inf` and `0.0` in float16.

    Widening to float32 keeps them and leaves the narrowing where it already
    was -- in `ct.convert(compute_precision=FLOAT16)`, which is the same cast
    a float32 checkpoint has always gone through. It also makes
    `precision="float32"` mean what it says: there, nothing is narrowed at
    all, and a `_np` that had rounded to float16 would have thrown the range
    away before the spelling was consulted.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    got = r["range"]
    assert "error" not in got, got
    assert got["dtype"] == "float32", got
    assert got["finite"] is True, got
    assert got["nonzero"] is True, got
    assert got["values"][0] > 1e19, got
    assert got["values"][1] < -1e19, got
    assert 0.0 < got["values"][2] < 1e-9, got


def test_the_two_agreement_grades_are_where_npu2_left_them():
    """Graded through `coreml.verify`, at its own tolerance, not a new one.

    float32 must still meet verify's own 2e-05 and float16 must still not,
    at SmolLM2's 576->1536 projection. Adding a bfloat16 source is only
    honest if it leaves this split exactly as docs/graph/NPU2.md §1.1 set it.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    f32 = r["verify"]["float32"]
    f16 = r["verify"]["float16"]
    assert f32["executed"] is True and f16["executed"] is True, r["verify"]
    assert f32["max_abs_diff"] <= 2e-5, f32
    assert f16["max_abs_diff"] > 2e-5, f16
    assert f16["max_abs_diff"] < 5e-2, f16


def test_a_bfloat16_linear_lowers_and_runs_and_gives_bfloat16_back():
    """End to end on one leaf: bfloat16 in, CoreML in the middle, bfloat16
    out. `state_dict()` and `generate()` must not learn anything about the
    hardware, which means the dtype the caller had is the dtype it gets."""
    r = _fixture_or_skip()
    if r is None:
        return
    forward = r["bf16_forward"]
    assert forward["dtype"] == "torch.bfloat16", forward
    assert forward["shape"] == [1, 1536], forward
    assert forward["finite"] is True, forward


def test_a_bfloat16_linear_reaches_the_unit_at_the_size_that_reaches_it():
    """The lowering itself is unaffected by where the weights came from.

    Same leaf, same shape, two batches: the unit is supported at both and
    preferred only at the larger one. If widening bfloat16 had changed the
    program CoreML sees, this is where it would show.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    small, large = r["bf16_linear"]["1"], r["bf16_linear"]["128"]
    assert small["ops"] == ["ios16.linear"], small
    assert "NeuralEngine" in small["supported"], small
    assert small["preferred"] == ["CPU"], small
    assert large["preferred"] == ["NeuralEngine"], large


def test_a_real_smollm2_checkpoint_lowers_and_names_everything_it_did_not():
    """The goal of the round: `from_pretrained(dtype="auto").to(device.npu)`.

    All 211 `Linear`s move. Everything else is **named** -- the embedding,
    the norms, the activation -- and the offload is partial, so it warns.
    Zero leaves would still be a refusal; this asserts the other side of that
    contract, that a partial one is never silent.
    """
    r = _smol_or_skip()
    if r is None:
        return
    smol = r["smol"]
    assert r["checkpoint_dtypes"] == ["torch.bfloat16"], r["checkpoint_dtypes"]
    assert smol["n_swapped"] == 211, smol
    assert smol["n_skipped"] == 0, smol["skipped"]
    assert smol["deferred"] == [], smol
    assert smol["precision"] == "float16", smol
    assert smol["fully_offloaded"] is False, smol
    # 0.826 and not 0.9997, and the gap is the tied `lm_head`: before the
    # swap `parameters()` deduplicates it against the embedding, and after it
    # the lowered leaf holds its own copy, so the denominator grows. Asserted
    # at the measured value rather than rounded up to a nicer one.
    assert 0.82 < smol["fraction_moved"] < 0.83, smol
    assert smol["parameters_moved"] == 134479872, smol
    # Named by type and count, not merely absent from `swapped`.
    assert smol["left_on_cpu"] == {
        "Embedding": 1, "LlamaRMSNorm": 61, "LlamaRotaryEmbedding": 1,
        "SiLUActivation": 30,
    }, smol["left_on_cpu"]
    assert any("PARTIAL offload" in w for w in smol["warnings"]), smol["warnings"]
    for kind in ("Embedding", "LlamaRMSNorm", "SiLUActivation"):
        assert any(kind in w for w in smol["warnings"]), (kind, smol["warnings"])
    # The swap really happened in the tree, not just in the report.
    assert "_CoreMLLinear" in smol["leaf_types"], smol["leaf_types"]
    assert "Linear" not in smol["leaf_types"], smol["leaf_types"]


def test_a_decode_step_reaches_the_neural_engine_on_none_of_its_linears():
    """The result that "SmolLM2 lowered" must not be read as.

    `MLComputePlan` at batch 1 -- the shape a decode step actually has --
    prefers the **CPU** for every distinct Linear shape in this model, while
    the unit is in the supported column for all of them. docs/graph/NPU2.md
    §2.1 measured the same crossover and the same cause: CoreML weighs
    dispatch cost against work, and a 576-wide projection of one token is not
    enough work. A prefill-shaped batch is a different answer, which is why
    both are read here on the same leaves.

    This is asserted rather than reported so that a change which made batch 1
    reach the unit would go red and be looked at, instead of quietly
    improving a claim nobody re-measured.
    """
    r = _smol_or_skip()
    if r is None:
        return
    plans = r["smol_plans"]
    assert set(plans) == {"576->1536", "576->576", "576->192",
                          "1536->576", "576->49152"}, sorted(plans)
    for shape, entry in sorted(plans.items()):
        assert entry["1"] == ["CPU"], (shape, entry)
    # And the same leaves at a prefill-shaped batch are not all CPU, so the
    # assertion above is about the size and not about the lowering.
    reached = [s for s, e in plans.items() if e["128"] == ["NeuralEngine"]]
    assert len(reached) >= 3, plans


def test_the_whole_model_runs_through_coreml_and_picks_the_same_next_token():
    """"Lowered" is a claim about a swap; this is the executed one.

    All 211 leaves, the real prompt, and the logits compared against the same
    model's own eager forward before the swap. The token chosen must be the
    same token -- that is the property a user notices, and it is not implied
    by the max-abs number on logits this large.
    """
    r = _smol_or_skip()
    if r is None:
        return
    ran, want = r["executed"], r["reference"]
    assert ran["n_leaves"] == 211, ran
    assert ran["shape"] == want["shape"] == [1, 5, 49152], (ran, want)
    assert ran["finite"] is True, ran
    assert ran["dtype"] == "torch.bfloat16", ran
    assert ran["argmax"] == want["argmax"], (ran, want)
    assert ran["token"] == want["token"] == " the", (ran, want)


def test_the_executed_logits_agree_at_float16_and_are_not_asserted_at_float32():
    """The honest grade for what just ran.

    These logits are a 30-layer float16 accumulation, not a single op, and
    they are not compared through `coreml.verify` because `verify` grades a
    captured trace and this is a whole transformer. So the claim made is the
    weak one and it is named weak: the difference is small *relative to the
    logits' own scale*, and nothing here pretends it meets the 2e-05 that the
    word "agrees" means elsewhere in this project.
    """
    r = _smol_or_skip()
    if r is None:
        return
    ran, want = r["executed"], r["reference"]
    assert want["scale"] > 1.0, want
    relative = ran["max_abs_logit_diff"] / want["scale"]
    assert relative < 0.05, (ran, want, relative)
    # Not zero either: a zero here would mean the swap did not take effect.
    assert ran["max_abs_logit_diff"] > 0.0, ran


def test_the_shape_that_never_reaches_the_unit_is_the_output_projection():
    """576->49152 is CPU-preferred at batch 128 too -- the one shape in this
    model for which prefill does not help either. Named because "most of it
    reaches the unit at prefill" is true and "all of it does" is not."""
    r = _smol_or_skip()
    if r is None:
        return
    plans = r["smol_plans"]
    assert plans["576->49152"]["128"] == ["CPU"], plans["576->49152"]
    assert plans["576->1536"]["128"] == ["NeuralEngine"], plans["576->1536"]


def test_a_cpu_preferred_decode_shape_warns_instead_of_looking_offloaded():
    """The leaf says so at the first real forward.

    A caller who wrote `to(device.npu)`, got no error, and ran a decode step
    would otherwise believe the Neural Engine ran it. docs/graph/NPU2.md §1 is
    that exact failure at model scale.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    warns = r["bf16_forward"]["warnings"]
    assert any("is NOT what it preferred" in w for w in warns), warns
    assert any("MLComputePlan says ['CPU']" in w for w in warns), warns


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
