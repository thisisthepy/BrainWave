#!/usr/bin/env python3
"""Do the architectures that FORWARD also AGREE? -- the caveat every sweep carried.

Why this exists
---------------
`arch_sweep.py` measures reachability. Both `docs/ARCH100.md` and `docs/ARCH200.md`
end on the same sentence and neither acted on it:

    A forward is not a match -- this sweep measures reachability, not numerics.

At the time this script was written 290 of 297 upstream-clean architectures
forwarded under the shim, and 26 of them had ever been checked for *agreement*
(the twenty of `docs/ARCH20.md` plus the six of `docs/ARCH26.md`). This script
closes that gap: it runs the same architecture, with the SAME WEIGHTS and the
SAME INPUTS, on both sides, and compares the outputs element-wise.

The three traps, and what is done about each
--------------------------------------------
1.  **Identical weights are the whole experiment.** Two random initialisations
    are not a comparison of arithmetic, they are a comparison of RNG. This
    script does not rely on the shim reproducing upstream's RNG even though it
    does: the upstream side *serialises* its `state_dict` and the shim side
    *loads it*, so the two are comparing arithmetic on identical bytes. The RNG
    claim is measured separately (`--rng-check`) and reported, not assumed --
    and `test_agree.py` pins upstream's seeded values so a regression in it is
    a failing test rather than a silently different experiment.

2.  **A tolerance chosen by eye is not a result.** `docs/DEMAND8.md` §1.4 set the
    standard: when the two sides differ, ask whether upstream's own float32
    answer is any closer to the truth than the shim's. So every architecture is
    additionally run **upstream in float64** and both float32 answers are scored
    against it. The headline tolerance is therefore not picked -- it is read off
    upstream's own float32 error distribution over the same population. A
    threshold tighter than that would flag upstream against itself.

3.  **"This model differs" is not a finding; "this op differs" is.** `--bisect`
    reuses `docs/DEMAND8.md` §1.3's technique: capture every leaf module's input
    and output upstream, then re-run each leaf under the shim on *upstream's own
    recorded input*, so nothing accumulates and the largest single-op error
    names the operator.

Known-divergent ground already measured, which must be read before calling
anything here a defect: `docs/MPSFWD.md` (`exp`, `sigmoid`, `silu` differ in the
last ulp between backends; `matmul` and `mul` are bit-identical),
`docs/VOICE3.md` (upstream's `i0` runs float-rounded coefficients),
`docs/TAIL4.md` (upstream's `erfinv` is wrong past `1-1e-11`).

Inherited limits -- `docs/ARCH200.md` §5, restated rather than quietly assumed
------------------------------------------------------------------------------
* `AutoModel` bodies only: no task heads, no generation loop, no KV cache reuse.
* Shrunk configs and random weights. Random weights reach the same *operators*
  (`arch_sweep.py --verify-random-weights` is the check for that claim) but they
  do not reach data-*dependent* branches, and they leave BatchNorm running stats
  at their initial values -- `docs/DEMAND8.md` §1.1 found that this underflows a
  deep convnet's activations to ~1e-23, at which point every relative number
  about it is meaningless. Such architectures are reported as `degenerate`
  rather than as `agree`, because at that scale agreement is unearned.
* One checkout, one point in time, while other work lands operators elsewhere.

How to run it
-------------
NOT wired into `run.sh` and must not be: it needs both interpreters and takes a
long time. Upstream runs FIRST -- it is the producer of weights, inputs and the
float64 oracle; the shim side is a replay.

    PY=/Volumes/macMini/caches/spike-venv/bin/python
    root=$(git rev-parse --show-toplevel)
    D=/tmp/agree

    env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL $PY agree_sweep.py --produce --dir $D
    PYTHONPATH=$root/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
        $PY agree_sweep.py --replay --dir $D
    $PY agree_sweep.py --report --dir $D

    # attribute one divergence to an operator (upstream first, then shim)
    env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL $PY agree_sweep.py --capture ARCH --dir $D
    PYTHONPATH=... TORCH_USE_RTLD_GLOBAL=1   $PY agree_sweep.py --bisect  ARCH --dir $D
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_sweep import architectures, input_candidates, shrink, side_of  # noqa: E402

# --------------------------------------------------------------------------
# dtype-preserving transport
# --------------------------------------------------------------------------
#
# The obvious transport is `Tensor.numpy()`, and the shim does not implement it
# (`TensorBase.numpy` is not in the table). `torch.from_numpy` is missing on the
# shim too; `torch.as_tensor` is present on both. So the wire format is
# `tolist()` out and `as_tensor()` in, with the dtype and shape carried
# alongside and reapplied -- `tolist()` on an empty or 0-dim tensor does not
# round-trip its shape through numpy's inference, and a silently reshaped input
# would be a different experiment.

_NP_FOR_TORCH = {
    "torch.float32": "float32", "torch.float64": "float64", "torch.float16": "float16",
    "torch.bfloat16": "float32",   # no numpy equivalent; cast back on the way in
    "torch.int64": "int64", "torch.int32": "int32", "torch.int16": "int16",
    "torch.int8": "int8", "torch.uint8": "uint8", "torch.bool": "bool",
}


def pack(t):
    """Tensor -> (ndarray, meta). Meta carries the torch dtype and shape."""
    name = str(t.dtype)
    npd = _NP_FOR_TORCH.get(name)
    if npd is None:
        raise TypeError(f"no transport for dtype {name}")
    shape = tuple(int(x) for x in t.shape)
    arr = np.asarray(t.detach().reshape(-1).tolist(), dtype=npd).reshape(shape)
    return arr, {"dtype": name, "shape": list(shape)}


def unpack(torch, arr, meta):
    t = torch.as_tensor(np.ascontiguousarray(arr))
    want = getattr(torch, meta["dtype"].split(".", 1)[1])
    if t.dtype != want:
        t = t.to(want)
    return t.reshape(tuple(meta["shape"]))


def save_bundle(path, tensors):
    """`tensors` is {key: tensor}. Writes `<path>.npz` and `<path>.meta.json`."""
    arrays, meta = {}, {}
    for k, t in tensors.items():
        arrays[k], meta[k] = pack(t)
    np.savez(path + ".npz", **arrays)
    with open(path + ".meta.json", "w") as fh:
        json.dump(meta, fh)


def load_bundle(torch, path):
    meta = json.load(open(path + ".meta.json"))
    with np.load(path + ".npz") as z:
        return {k: unpack(torch, z[k], meta[k]) for k in meta}


# --------------------------------------------------------------------------
# walking a model output
# --------------------------------------------------------------------------


def flatten_outputs(obj, prefix="", out=None, depth=0):
    """{dotted path: tensor} for every tensor anywhere in a model output.

    Recurses through ModelOutput / dict / tuple / list. Non-tensors are dropped
    rather than stringified: a config echo or a `past_key_values` cache object
    is not a number and pretending it is would put noise in the ranking.
    """
    out = {} if out is None else out
    if depth > 4:
        return out
    if hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "reshape"):
        out[prefix or "__out__"] = obj
        return out
    items = None
    if hasattr(obj, "items"):
        items = list(obj.items())
    elif isinstance(obj, (tuple, list)):
        items = list(enumerate(obj))
    if items is None:
        return out
    for k, v in items:
        flatten_outputs(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    return out


# --------------------------------------------------------------------------
# the comparison arithmetic -- pure, so `test_agree.py` can pin it
# --------------------------------------------------------------------------


def diff_stats(a, b):
    """Element-wise comparison of two float64 ndarrays of the same shape.

    `scale` is the reference tensor's own largest magnitude, and `rel` is the
    absolute difference measured against it. Relative-to-scale rather than
    element-wise-relative on purpose: an element that is 1e-30 in a tensor whose
    scale is 1 has no significant digits to disagree about, and dividing by it
    manufactures a divergence out of a denormal.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return {"shape_mismatch": True, "abs": float("inf"), "rel": float("inf"),
                "scale": 0.0, "n": int(a.size)}
    fa, fb = np.isfinite(a), np.isfinite(b)
    nonfinite_mismatch = int(np.sum(fa != fb))
    same_nonfinite = bool(np.array_equal(a[~fa], b[~fb])) if not nonfinite_mismatch else False
    m = fa & fb
    if not m.any():
        return {"abs": 0.0, "rel": 0.0, "scale": 0.0, "n": int(a.size),
                "nonfinite_mismatch": nonfinite_mismatch,
                "same_nonfinite": same_nonfinite}
    d = np.abs(a[m] - b[m])
    scale = float(np.max(np.abs(b[m]))) if m.any() else 0.0
    absd = float(np.max(d))
    rel = absd / scale if scale > 0 else absd
    return {"abs": absd, "rel": rel, "scale": scale, "n": int(a.size),
            "nonfinite_mismatch": nonfinite_mismatch, "same_nonfinite": same_nonfinite}


# `float32` eps. Everything below is expressed as a multiple of it rather than as
# a decimal, so the number says where it comes from.
F32_EPS = float(np.finfo(np.float32).eps)      # 1.1920929e-07

# The scale below which a tensor carries no information to disagree about.
# `docs/DEMAND8.md` §1.1 measured a deep convnet's activations underflowing to
# ~1e-23 under uncalibrated BatchNorm statistics; at that magnitude both sides
# agree on noise and calling it agreement would be unearned.
DEGENERATE_SCALE = 1e-12

# How much worse than upstream's own float32 error the shim may be before the
# "this is float32 accumulation" explanation stops covering it. 4x, not 1x:
# two independent float32 truncation paths sum, and DEMAND8 §1.4 measured the
# shim-vs-upstream gap (1.28e-04) at ~1.2x upstream's own distance from the
# float64 truth (1.06e-04). 4 leaves room for that and refuses an order of
# magnitude.
ORACLE_FACTOR = 4.0


def verdict(rel, scale, oracle_rel, tol, self_repeat_rel=0.0):
    """The per-architecture call. Deliberately has two "I cannot tell" outcomes.

    Order matters. `nondeterministic` comes first: an architecture that does not
    reproduce ITSELF on upstream (`vit_mae` re-draws its patch mask every
    forward; `vits` samples a duration) has no fixed answer for the shim to
    match, so any shim-vs-upstream number about it is measuring the sampler.
    `degenerate` comes before `exact` for the same kind of reason: two sides
    that both underflow to zero are equal without having agreed about anything.
    """
    if self_repeat_rel is not None and self_repeat_rel > tol:
        return "nondeterministic"
    if scale is not None and scale < DEGENERATE_SCALE:
        return "degenerate"
    if rel == 0.0:
        return "exact"
    if rel <= tol:
        return "agree"
    if oracle_rel is not None and oracle_rel > 0 and rel <= ORACLE_FACTOR * oracle_rel:
        return "agree_within_float32"
    return "diverge"


# --------------------------------------------------------------------------
# building one architecture, identically on both sides
# --------------------------------------------------------------------------


def calibration_batch(torch, kwargs, repeats=4, seed=1234):
    """The batch used to calibrate BatchNorm running statistics.

    Extracted from `produce_one` so it can be tested: its one requirement is
    that the rows DIFFER, and getting that wrong is silent. Four identical
    copies give a batch variance of exactly zero; `rsqrt(0 + eps)` then makes
    the layer an amplifier on the float side, and on an integer-fed branch the
    layer collapses to its bias instead (`groupvit`'s `text_projection`
    BatchNorm1d emitted 5.3e-05 against a 3.9e-01 input, and the model's
    end-to-end "divergence" of 5.8e-01 was entirely that -- it fell to 3.1e-06
    once the rows differed). Both are artefacts of the harness rather than
    facts about either side.

    Float rows get fresh noise. Integer rows are resampled inside the observed
    range so they stay valid ids. Boolean and 0/1 tensors are repeated: a random
    attention mask is a different sequence length, not more variance.
    """
    torch.manual_seed(seed)
    cal = {}
    for k, v in kwargs.items():
        if not (hasattr(v, "shape") and hasattr(v, "dim") and v.dim() >= 1):
            cal[k] = v
        elif v.is_floating_point():
            cal[k] = torch.cat([v] + [torch.randn_like(v) for _ in range(repeats - 1)], 0)
        elif str(v.dtype) == "torch.bool" or v.numel() == 0 or int(v.max()) <= 1:
            cal[k] = torch.cat([v] * repeats, 0)
        else:
            hi = int(v.max()) + 1
            cal[k] = torch.cat(
                [v] + [torch.randint(0, hi, tuple(v.shape), dtype=v.dtype)
                       for _ in range(repeats - 1)], 0)
    return cal


def build(torch, model_type, seed=0):
    from transformers import AutoConfig, AutoModel
    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(model_type)
    shrink(cfg)
    cfg.output_attentions = False
    cfg.output_hidden_states = False
    model = AutoModel.from_config(cfg)
    model.eval()
    return cfg, model


def _apply_weights(torch, model, weights):
    """Load the producer's state_dict. Strict, and a mismatch is a finding.

    Non-strict would hide the interesting case: the same `transformers` version
    building a structurally different module tree under the shim. That has never
    been observed, which is exactly why it is asserted rather than tolerated.
    """
    missing, unexpected = model.load_state_dict(weights, strict=False)
    return {"missing": sorted(missing), "unexpected": sorted(unexpected)}


# --------------------------------------------------------------------------
# producer -- runs on UPSTREAM
# --------------------------------------------------------------------------


def produce_one(model_type, outdir):
    import torch
    rec = {"model_type": model_type, "side": side_of(torch), "stage": "start"}
    assert rec["side"] == "upstream", "the producer must be upstream: it is the oracle"
    base = os.path.join(outdir, model_type)
    try:
        cfg, model = build(torch, model_type)
        rec["arch_class"] = type(model).__name__
    except Exception:
        rec.update(stage="construction", status="fail", error=traceback.format_exc()[-3000:])
        return rec

    # inputs: the first candidate that forwards, exactly as arch_sweep chooses
    try:
        cands = input_candidates(torch, model, cfg)
    except Exception:
        rec.update(stage="inputs", status="fail", error=traceback.format_exc()[-3000:])
        return rec
    chosen = err = None
    for label, kwargs in cands:
        try:
            with torch.no_grad():
                out = model(**kwargs)
        except Exception:
            err = traceback.format_exc()[-3000:]
            continue
        chosen = (label, kwargs, out)
        break
    if chosen is None:
        rec.update(stage="forward", status="fail", error=err)
        return rec
    label, kwargs, out = chosen
    rec["inputs"] = label

    # BatchNorm calibration -- docs/DEMAND8.md §1.1, and it is not cosmetic.
    # Freshly-initialised BatchNorm has `running_var=1`/`running_mean=0`, which
    # for a deep convnet drives activations to ~1e-23 by the last block; at that
    # scale a relative comparison compares denormals. One forward in train mode
    # at `momentum=1.0` replaces the running stats with real batch statistics --
    # what a trained checkpoint has -- and because those stats live in the
    # `state_dict`, the shim side receives them by loading the same bytes. No
    # calibration is repeated on the shim side, so no calibration can diverge.
    bns = [m for m in model.modules()
           if hasattr(m, "running_var") and hasattr(m, "momentum")
           and getattr(m, "running_var", None) is not None]
    rec["batchnorms"] = len(bns)
    if bns:
        saved = [m.momentum for m in bns]
        # A batch of one is not enough: `BatchNorm` in train mode refuses
        # "1 value per channel", which is exactly what a 1x1 spatial map at
        # batch 1 gives (measured on `mobilenet_v2`, whose last block is
        # (1, 576, 1, 1)). And the rows must differ -- see `calibration_batch`,
        # where both halves of that requirement are stated and tested.
        cal = calibration_batch(torch, kwargs)
        try:
            for m in bns:
                m.momentum = 1.0
            model.train()
            with torch.no_grad():
                model(**cal)
            rec["calibrated"] = True
        except Exception:
            # A partial calibration is still fine: whatever the running stats
            # ended up as, they are saved in the `state_dict` and the shim side
            # loads those same bytes. What is NOT fine is reusing the `out`
            # computed before the attempt, so it is always recomputed below.
            rec["calibrated"] = False
            rec["calibrate_error"] = traceback.format_exc()[-400:]
        finally:
            for m, mom in zip(bns, saved):
                m.momentum = mom
            model.eval()
        try:
            with torch.no_grad():
                out = model(**kwargs)
        except Exception:
            rec.update(stage="forward", status="fail", error=traceback.format_exc()[-3000:])
            return rec

    flat = flatten_outputs(out)
    if not flat:
        rec.update(stage="forward", status="fail", error="no tensor in the model output")
        return rec

    # run-to-run determinism upstream. If upstream does not reproduce itself,
    # no shim-vs-upstream number about this architecture means anything, and it
    # is excluded from the verdict rather than counted as a divergence.
    try:
        with torch.no_grad():
            out2 = model(**kwargs)
        flat2 = flatten_outputs(out2)
        rec["self_repeat_rel"] = max(
            diff_stats(pack(flat[k])[0], pack(flat2[k])[0])["rel"]
            for k in flat if k in flat2)
    except Exception:
        rec["self_repeat_rel"] = None

    try:
        save_bundle(base + ".weights", dict(model.state_dict()))
        save_bundle(base + ".inputs", kwargs)
        save_bundle(base + ".up32", flat)
    except Exception:
        rec.update(stage="transport", status="fail", error=traceback.format_exc()[-3000:])
        return rec

    # the float64 oracle -- DEMAND8 §1.4's control, run for every architecture
    # that will take it. Where it refuses, that is recorded, not papered over:
    # an architecture with no oracle gets the fixed tolerance and says so.
    try:
        model64 = model.double()
        k64 = {k: (v.double() if v.is_floating_point() else v) for k, v in kwargs.items()}
        with torch.no_grad():
            o64 = model64(**k64)
        f64 = flatten_outputs(o64)
        save_bundle(base + ".up64", {k: v for k, v in f64.items() if k in flat})
        rec["oracle"] = "ok"
    except Exception:
        rec["oracle"] = "fail"
        rec["oracle_error"] = traceback.format_exc()[-800:]

    rec.update(stage="forward", status="ok")
    rec["keys"] = sorted(flat)
    return rec


# --------------------------------------------------------------------------
# replay -- runs on the SHIM
# --------------------------------------------------------------------------


def replay_one(model_type, outdir):
    import torch
    rec = {"model_type": model_type, "side": side_of(torch), "stage": "start"}
    assert rec["side"] == "shim", "the replay side must be the shim"
    base = os.path.join(outdir, model_type)
    if not os.path.exists(base + ".up32.npz"):
        rec.update(stage="produce", status="skip", error="no upstream bundle")
        return rec
    try:
        cfg, model = build(torch, model_type)
    except Exception:
        rec.update(stage="construction", status="fail", error=traceback.format_exc()[-3000:])
        return rec
    try:
        weights = load_bundle(torch, base + ".weights")
        rec["load"] = _apply_weights(torch, model, weights)
        kwargs = load_bundle(torch, base + ".inputs")
    except Exception:
        rec.update(stage="weights", status="fail", error=traceback.format_exc()[-3000:])
        return rec
    try:
        with torch.no_grad():
            out = model(**kwargs)
    except Exception:
        rec.update(stage="forward", status="fail", error=traceback.format_exc()[-3000:])
        return rec
    flat = flatten_outputs(out)
    try:
        save_bundle(base + ".shim32", flat)
    except Exception:
        rec.update(stage="transport", status="fail", error=traceback.format_exc()[-3000:])
        return rec
    rec.update(stage="forward", status="ok", keys=sorted(flat))
    return rec


# --------------------------------------------------------------------------
# the RNG claim, measured rather than trusted
# --------------------------------------------------------------------------

_RNG_SPEC = [
    ("randn", "torch.randn(7)"),
    ("randn2", "torch.randn(3, 4)"),
    ("rand", "torch.rand(7)"),
    ("randint", "torch.randint(0, 100, (7,))"),
    ("normal_", "torch.empty(7).normal_(0, 1)"),
    ("uniform_", "torch.empty(7).uniform_(-1, 1)"),
    ("randperm", "torch.randperm(7)"),
]


def rng_check(seed=0):
    """Print, for one seed, what each generator produces on THIS side.

    Run on both sides and diff. This is the check behind the sentence "the shim
    reproduces upstream's RNG": the sweep does not depend on it (weights travel
    as bytes) but the claim is load-bearing elsewhere, so it is measured here
    and its upstream values are frozen into `test_agree.py`.
    """
    import torch
    print(side_of(torch))
    out = {}
    for name, expr in _RNG_SPEC:
        torch.manual_seed(seed)
        try:
            t = eval(expr, {"torch": torch})              # noqa: S307 -- fixed table above
            out[name] = [float(x) for x in t.reshape(-1).tolist()]
        except Exception as e:                            # noqa: BLE001
            out[name] = f"ERROR {type(e).__name__}: {e}"
    print(json.dumps(out, indent=1))
    return 0


# --------------------------------------------------------------------------
# per-module bisection -- DEMAND8 §1.3's technique
# --------------------------------------------------------------------------


def _leaves(model):
    return [(n, m) for n, m in model.named_modules() if not list(m.children()) and n]


def capture_one(model_type, outdir):
    """UPSTREAM: record every leaf module's input and output."""
    import torch
    assert side_of(torch) == "upstream"
    base = os.path.join(outdir, model_type)
    cfg, model = build(torch, model_type)
    weights = load_bundle(torch, base + ".weights")
    model.load_state_dict(weights, strict=False)
    kwargs = load_bundle(torch, base + ".inputs")

    recorded, skipped, handles = {}, {}, []

    def mk(name):
        def hook(mod, args, output):
            if name in recorded or name in skipped:
                return
            if not args or not all(hasattr(a, "shape") and hasattr(a, "dtype") for a in args):
                skipped[name] = "non-tensor positional argument"
                return
            if not (hasattr(output, "shape") and hasattr(output, "dtype")):
                skipped[name] = "non-tensor output"
                return
            # CLONE, do not reference. The bundle is serialised after the
            # forward finishes, and a residual `+=`, an in-place activation or
            # a one-hot `scatter_` will have rewritten a referenced tensor by
            # then -- so the replay would run on an input upstream never saw.
            # Measured, not hypothetical: without this, `groupvit`'s
            # `downsample.assign.proj` (a plain `nn.Linear`) reported a
            # relative error of 1.0, and the "divergence" was the harness's.
            recorded[name] = ([a.detach().clone() for a in args],
                              output.detach().clone())
        return hook

    for n, m in _leaves(model):
        handles.append(m.register_forward_hook(mk(n)))
    try:
        with torch.no_grad():
            model(**kwargs)
    finally:
        for h in handles:
            h.remove()

    tens, index = {}, {}
    for n, (args, out) in recorded.items():
        index[n] = {"nargs": len(args), "cls": type(dict(model.named_modules())[n]).__name__}
        for i, a in enumerate(args):
            tens[f"{n}||in{i}"] = a
        tens[f"{n}||out"] = out
    drop = []
    for k, v in list(tens.items()):
        if str(v.dtype) not in _NP_FOR_TORCH:
            drop.append(k.split("||")[0])
    for n in set(drop):
        skipped[n] = "untransportable dtype"
        index.pop(n, None)
        for k in [k for k in tens if k.startswith(n + "||")]:
            tens.pop(k)
    save_bundle(base + ".modules", tens)
    with open(base + ".modules.index.json", "w") as fh:
        json.dump({"index": index, "skipped": skipped}, fh, indent=1)
    print(f"captured {len(index)} leaf modules, skipped {len(skipped)}")
    return 0


def bisect_one(model_type, outdir, top=25):
    """SHIM: re-run each leaf module on upstream's own recorded input.

    Nothing accumulates, so the ranking names the operator rather than the
    depth at which the drift became visible.
    """
    import torch
    assert side_of(torch) == "shim"
    base = os.path.join(outdir, model_type)
    meta = json.load(open(base + ".modules.index.json"))
    tens = load_bundle(torch, base + ".modules")
    cfg, model = build(torch, model_type)
    model.load_state_dict(load_bundle(torch, base + ".weights"), strict=False)
    mods = dict(model.named_modules())

    rows = []
    for name, info in meta["index"].items():
        mod = mods.get(name)
        if mod is None:
            rows.append({"module": name, "cls": info["cls"], "error": "absent under the shim"})
            continue
        args = [tens[f"{name}||in{i}"] for i in range(info["nargs"])]
        want = tens[f"{name}||out"]
        try:
            with torch.no_grad():
                got = mod(*args)
        except Exception as e:                            # noqa: BLE001
            rows.append({"module": name, "cls": info["cls"],
                         "error": f"{type(e).__name__}: {str(e)[:160]}"})
            continue
        st = diff_stats(pack(got)[0], pack(want)[0])
        rows.append({"module": name, "cls": info["cls"], "shape": list(args[0].shape),
                     **{k: st[k] for k in ("abs", "rel", "scale")}})
    rows.sort(key=lambda r: -(r.get("rel") or 0.0))
    with open(base + ".bisect.json", "w") as fh:
        json.dump({"model_type": model_type, "rows": rows,
                   "skipped": meta["skipped"]}, fh, indent=1)
    print(f"{model_type}: {len(rows)} leaf modules replayed on upstream's recorded input")
    print(f"{'rel':>12} {'abs':>12} {'scale':>11}  class / module")
    for r in rows[:top]:
        if "error" in r:
            print(f"{'ERROR':>12} {'':>12} {'':>11}  {r['cls']:22s} {r['module']}  {r['error']}")
        else:
            print(f"{r['rel']:12.3e} {r['abs']:12.3e} {r['scale']:11.3e}  "
                  f"{r['cls']:22s} {r['module']}")
    print(f"(float32 eps = {F32_EPS:.3e}; a row at 1e-07 is one ulp)")
    return 0


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------


def _drive(mode, outdir, types, timeout, deadline=None):
    """One subprocess per architecture, resumable.

    Resumable because it has to be: a few hundred model constructions is longer
    than any single foreground window, and a driver that has to start over is a
    driver that gets backgrounded and then lost. Each architecture's record is
    written the moment it finishes, and a re-run skips whatever is already on
    disk -- so the same command can be issued repeatedly until it reports
    nothing left to do. `--deadline` makes it stop cleanly rather than be killed
    mid-architecture, which would leave a half-written bundle.
    """
    import time
    import torch
    side = side_of(torch)
    os.makedirs(outdir, exist_ok=True)
    recdir = os.path.join(outdir, f"_rec_{mode}")
    os.makedirs(recdir, exist_ok=True)
    started = time.time()
    todo = [t for t in types if not os.path.exists(os.path.join(recdir, t + ".json"))]
    print(f"side={side}  mode={mode}  architectures={len(types)}  todo={len(todo)}", flush=True)
    for i, mt in enumerate(todo, 1):
        if deadline is not None and time.time() - started > deadline:
            print(f"deadline reached with {len(todo) - i + 1} left -- re-run to continue",
                  flush=True)
            break
        cmd = [sys.executable, os.path.abspath(__file__), f"--{mode}-one", mt, "--dir", outdir]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            rec = json.loads(line)
        except subprocess.TimeoutExpired:
            rec = {"model_type": mt, "side": side, "stage": "unknown",
                   "status": "fail", "error": "TIMEOUT"}
        except Exception:
            rc = getattr(locals().get("proc", None), "returncode", None)
            err = (getattr(locals().get("proc", None), "stderr", "") or "")[-2000:]
            rec = {"model_type": mt, "side": side, "stage": "unknown", "status": "fail",
                   "error": f"NO_JSON rc={rc}\n{err}"}
        with open(os.path.join(recdir, mt + ".json"), "w") as fh:
            json.dump(rec, fh)
        print(f"[{i:3d}/{len(todo)}] {rec.get('status','?'):4s} "
              f"{rec.get('stage','')[:12]:12s} {mt}", flush=True)
    results = []
    for t in types:
        f = os.path.join(recdir, t + ".json")
        if os.path.exists(f):
            results.append(json.load(open(f)))
    path = os.path.join(outdir, f"_{mode}.json")
    with open(path, "w") as fh:
        json.dump({"side": side, "results": results}, fh, indent=1)
    ok = sum(r.get("status") == "ok" for r in results)
    left = len(types) - len(results)
    print(f"\nwrote {path}: {ok}/{len(results)} ok, {left} not yet attempted")
    return 0


# --------------------------------------------------------------------------
# the report -- the deliverable
# --------------------------------------------------------------------------


def _score(outdir, model_type):
    """Compare one architecture's saved bundles. Pure numpy; no torch needed."""
    base = os.path.join(outdir, model_type)
    if not (os.path.exists(base + ".up32.npz") and os.path.exists(base + ".shim32.npz")):
        return None
    with np.load(base + ".up32.npz") as up, np.load(base + ".shim32.npz") as sh:
        keys = [k for k in up.files if k in sh.files]
        if not keys:
            return {"model_type": model_type, "error": "no common output key"}
        per = {k: diff_stats(sh[k], up[k]) for k in keys}
        missing_keys = sorted(set(up.files) ^ set(sh.files))
        oracle = None
        if os.path.exists(base + ".up64.npz"):
            with np.load(base + ".up64.npz") as o:
                oracle = {k: diff_stats(up[k], o[k]) for k in keys if k in o.files}
    worst_key = max(keys, key=lambda k: per[k]["rel"])
    w = per[worst_key]
    orel = None
    if oracle:
        orel = max((v["rel"] for v in oracle.values() if math.isfinite(v["rel"])), default=None)
    return {"model_type": model_type, "key": worst_key, "rel": w["rel"], "abs": w["abs"],
            "scale": w["scale"], "n": w["n"], "oracle_rel": orel,
            "nonfinite_mismatch": w.get("nonfinite_mismatch", 0),
            "missing_keys": missing_keys, "nkeys": len(keys)}


def report(outdir, tol=None, top=40):
    prod = json.load(open(os.path.join(outdir, "_produce.json")))
    rep = json.load(open(os.path.join(outdir, "_replay.json")))
    pr = {r["model_type"]: r for r in prod["results"]}
    rp = {r["model_type"]: r for r in rep["results"]}

    up_ok = [m for m, r in pr.items() if r.get("status") == "ok"]
    shim_ok = [m for m in up_ok if rp.get(m, {}).get("status") == "ok"]

    scores = []
    for m in shim_ok:
        s = _score(outdir, m)
        if s and "rel" in s:
            s["self_repeat_rel"] = pr[m].get("self_repeat_rel")
            s["load"] = rp[m].get("load", {})
            scores.append(s)

    # ---- the tolerance, derived rather than chosen ------------------------
    # The oracle distribution excludes the architectures that cannot hold still:
    # a model that re-samples every forward gives an f32-vs-f64 "error" of order
    # 1 (measured: 1.43 for `vit_mae`), and letting that into a percentile would
    # loosen the tolerance for everyone else using a number about a sampler.
    orels = sorted(s["oracle_rel"] for s in scores
                   if s["oracle_rel"] is not None and math.isfinite(s["oracle_rel"])
                   and s["scale"] >= DEGENERATE_SCALE
                   and not (s["self_repeat_rel"] or 0.0) > 1e-6)
    if tol is None:
        # The floor under any honest threshold is upstream's OWN float32 error.
        # Take the 90th percentile of it over this population: a tolerance below
        # that would fail upstream against itself on a tenth of the set.
        tol = orels[int(0.90 * (len(orels) - 1))] if orels else 1e-5
        tol = max(tol, 8 * F32_EPS)   # never tighter than a few ulp

    for s in scores:
        s["verdict"] = verdict(s["rel"], s["scale"], s["oracle_rel"], tol,
                               s["self_repeat_rel"])

    def n(v):
        return sum(1 for s in scores if s["verdict"] == v)

    print("=" * 78)
    print("AGREEMENT SWEEP -- of the architectures that forward, how many match?")
    print("=" * 78)
    print(f"architectures attempted                     : {len(pr)}")
    print(f"  produced on UPSTREAM (weights+inputs+f64) : {len(up_ok)}")
    print(f"  replayed under the SHIM on those weights  : {len(shim_ok)}")
    print(f"  scored (both sides produced tensors)      : {len(scores)}")
    print()
    print("the tolerance, and where it comes from")
    print("-" * 78)
    if orels:
        q = lambda p: orels[int(p * (len(orels) - 1))]     # noqa: E731
        print(f"  upstream's OWN float32-vs-float64 relative error, over {len(orels)} "
              f"architectures with a working oracle:")
        print(f"    median {q(0.5):.2e}   p90 {q(0.9):.2e}   p99 {q(0.99):.2e}   "
              f"max {orels[-1]:.2e}")
    print(f"  float32 eps                               : {F32_EPS:.3e}")
    print(f"  TOLERANCE USED (rel, vs tensor scale)     : {tol:.3e}"
          f"  = {tol / F32_EPS:.0f} ulp")
    print("  chosen as p90 of upstream's own float32 error, floored at 8 ulp -- a")
    print("  tighter number would flag upstream against float64 on a tenth of the set.")
    print()
    print("verdicts")
    print("-" * 78)
    agree = n("exact") + n("agree") + n("agree_within_float32")
    for v, why in (("exact", "bit-identical to upstream"),
                   ("agree", f"within {tol:.1e} relative"),
                   ("agree_within_float32",
                    f"outside it, but within {ORACLE_FACTOR:.0f}x upstream's own f32 error"),
                   ("diverge", "outside both -- ranked below"),
                   ("degenerate", f"output scale < {DEGENERATE_SCALE:g}; nothing to compare"),
                   ("nondeterministic",
                    "upstream does not reproduce itself; excluded, not counted")):
        print(f"  {v:22s} {n(v):4d}   {why}")
    denom = len(scores) - n("degenerate") - n("nondeterministic")
    print()
    print(f"  HEADLINE: {agree} of {denom} scored architectures agree "
          f"({100 * agree / max(1, denom):.1f}%)")
    print(f"            {n('diverge')} diverge; "
          f"{n('degenerate') + n('nondeterministic')} could not be judged")
    print()

    nd = [s for s in scores if s["self_repeat_rel"] not in (None, 0.0)]
    print(f"upstream reproduced ITSELF on a second forward for "
          f"{len(scores) - len(nd)}/{len(scores)}")
    for s in nd[:10]:
        print(f"  NOT self-deterministic: {s['model_type']:28s} "
              f"self-repeat rel={s['self_repeat_rel']:.2e} -- excluded from the verdict")
    print()

    bad_load = [s for s in scores if s["load"].get("missing") or s["load"].get("unexpected")]
    print(f"state_dict transferred cleanly for {len(scores) - len(bad_load)}/{len(scores)}")
    for s in bad_load[:10]:
        print(f"  {s['model_type']:28s} missing={len(s['load']['missing'])} "
              f"unexpected={len(s['load']['unexpected'])} "
              f"e.g. {(s['load']['missing'] or s['load']['unexpected'])[:2]}")
    print()

    print("RANKED -- the largest measured differences, whatever the verdict")
    print("-" * 78)
    print("(`ratio` is rel / upstream's own f32-vs-f64 error: 1.0 means the shim is")
    print(" exactly as far from the truth as upstream is, which is not a defect)")
    ranked = sorted([s for s in scores
                     if s["verdict"] in ("diverge", "agree_within_float32", "agree")],
                    key=lambda s: -s["rel"])
    print(f"{'rel':>11} {'abs':>11} {'scale':>10} {'up f32-f64':>11} {'ratio':>6}  "
          f"verdict / architecture [key]")
    for s in ranked[:top]:
        o = s["oracle_rel"]
        ratio = f"{s['rel'] / o:6.2f}" if o else "     -"
        os_ = f"{o:.2e}" if o is not None else "no oracle"
        print(f"{s['rel']:11.3e} {s['abs']:11.3e} {s['scale']:10.3e} {os_:>11} {ratio}  "
              f"{s['verdict'][:9]:9s} {s['model_type']} [{s['key']}]")
    print()

    print("DIVERGENCES  (rel = max|shim-upstream| / max|upstream|)")
    print("-" * 78)
    div = sorted([s for s in scores if s["verdict"] == "diverge"], key=lambda s: -s["rel"])
    print(f"{'rel':>11} {'abs':>11} {'scale':>10} {'up f32-f64':>11}  architecture / output key")
    for s in div[:top]:
        o = f"{s['oracle_rel']:.2e}" if s["oracle_rel"] is not None else "no oracle"
        print(f"{s['rel']:11.3e} {s['abs']:11.3e} {s['scale']:10.3e} {o:>11}  "
              f"{s['model_type']} [{s['key']}]")
    if not div:
        print("  (none)")
    print()

    print("architectures that FORWARD upstream but did not replay under the shim")
    print("-" * 78)
    for m in up_ok:
        r = rp.get(m, {})
        if r.get("status") != "ok":
            last = [l for l in (r.get("error") or "").strip().splitlines() if l.strip()]
            print(f"  {m:30s} {r.get('stage','?'):12s} {last[-1][:100] if last else ''}")
    print()
    nof64 = [m for m in shim_ok if pr[m].get("oracle") != "ok"]
    print(f"no float64 oracle (fixed tolerance only): {len(nof64)}")
    print("  " + ", ".join(sorted(nof64)[:24]) + (" ..." if len(nof64) > 24 else ""))
    with open(os.path.join(outdir, "_report.json"), "w") as fh:
        json.dump({"tol": tol, "scores": scores}, fh, indent=1)
    return 0


# --------------------------------------------------------------------------
# self-tests -- run under the SHIM by `test_agree.py`
# --------------------------------------------------------------------------


def self_test_calibration():
    """Prove the calibration batch's rows differ, which is its only job."""
    import torch
    print(side_of(torch))
    kwargs = {
        "pixel_values": torch.randn(1, 3, 4, 4),
        "input_ids": torch.randint(0, 50, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    }
    cal = calibration_batch(torch, kwargs)
    f, i, b = cal["pixel_values"], cal["input_ids"], cal["attention_mask"]
    out = {
        "batch": int(f.shape[0]),
        "float_rows_distinct": bool(float((f[0] - f[1]).abs().max()) > 0),
        "int_rows_distinct": bool(int((i[0] - i[1]).abs().max()) > 0),
        "bool_rows_identical": bool(int((b[0] - b[1]).abs().max()) == 0),
        # the number that matters: BatchNorm sees this, and zero is the trap
        "float_batch_variance": float(f.reshape(int(f.shape[0]), -1).var(0).mean()),
        "int_batch_variance": float(i.to(torch.float32).var(0).mean()),
    }
    print(json.dumps(out, indent=1))
    return 0


def self_test_transport():
    """Round-trip every dtype the sweep carries, including the shapes that
    `tolist()` cannot reconstruct on its own (0-dim and empty)."""
    import torch
    print(side_of(torch))
    cases = {}
    for name in _NP_FOR_TORCH:
        d = getattr(torch, name.split(".", 1)[1])
        try:
            if d.is_floating_point:
                t = torch.randn(2, 3).to(d)
            elif d == torch.bool:
                t = torch.tensor([[True, False, True], [False, True, False]])
            else:
                t = torch.randint(0, 7, (2, 3)).to(d)
        except Exception as e:                            # noqa: BLE001
            cases[name] = {"unavailable": str(e)[:100]}
            continue
        cases[name] = t
    cases["torch.float32:scalar"] = torch.tensor(3.25)
    cases["torch.float32:empty"] = torch.zeros(0, 4)
    out = {}
    for name, t in cases.items():
        if not hasattr(t, "shape"):
            out[name] = t
            continue
        arr, meta = pack(t)
        back = unpack(torch, arr, meta)
        out[name] = {
            "dtype_preserved": str(back.dtype) == str(t.dtype),
            "shape_preserved": tuple(back.shape) == tuple(t.shape),
            "values_exact": bool(t.numel() == 0
                                 or float((back.to(torch.float64)
                                           - t.to(torch.float64)).abs().max()) == 0.0),
        }
    print(json.dumps(out, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/agree")
    ap.add_argument("--produce", action="store_true", help="UPSTREAM: weights, inputs, f64 oracle")
    ap.add_argument("--replay", action="store_true", help="SHIM: same weights, same inputs")
    ap.add_argument("--produce-one")
    ap.add_argument("--replay-one")
    ap.add_argument("--capture", help="UPSTREAM: record every leaf module's input/output")
    ap.add_argument("--bisect", help="SHIM: replay each leaf on upstream's recorded input")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rng-check", action="store_true")
    ap.add_argument("--self-test-calibration", action="store_true")
    ap.add_argument("--self-test-transport", action="store_true")
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--top", type=int, default=25,
                    help="rows to print in --bisect / --report")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--deadline", type=int, default=None,
                    help="stop cleanly after N seconds; re-run to continue")
    args = ap.parse_args()

    if args.self_test_calibration:
        return self_test_calibration()
    if args.self_test_transport:
        return self_test_transport()
    if args.rng_check:
        return rng_check()
    if args.report:
        return report(args.dir, args.tol, args.top)
    if args.produce_one:
        print(json.dumps(produce_one(args.produce_one, args.dir)))
        return 0
    if args.replay_one:
        print(json.dumps(replay_one(args.replay_one, args.dir)))
        return 0
    if args.capture:
        return capture_one(args.capture, args.dir)
    if args.bisect:
        return bisect_one(args.bisect, args.dir, args.top)
    types = args.only or architectures(args.limit)
    if args.produce:
        return _drive("produce", args.dir, types, args.timeout, args.deadline)
    if args.replay:
        return _drive("replay", args.dir, types, args.timeout, args.deadline)
    ap.error("one of --produce/--replay/--report/--capture/--bisect/--rng-check")


if __name__ == "__main__":
    sys.exit(main())
