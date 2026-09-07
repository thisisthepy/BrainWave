#!/usr/bin/env python3
"""Run `torch.export.export` on every `transformers` architecture and record where it stops.

Why this exists
---------------
`arch_sweep.py` asks "does a forward pass run". This asks the *next* question, and
it is a strictly harder one:

    does `torch.export.export()` return an `ExportedProgram`, and does the graph
    it holds REPLAY to the same numbers as the eager module it was traced from?

Those are three separate verdicts and this script keeps them separate, because
docs/graph/EXPORT4.md §3 is about a failure mode where they get merged:

    exported   -- `torch.export.export()` returned an `ExportedProgram`
    replayed   -- `ep.module()(*inputs)` ran without raising
    agreed     -- the replayed outputs match eager element-wise

A graph that BUILDS and a graph that COMPUTES are different claims. An
`ExportedProgram` that prints, serialises, and contains no operators would pass
"exported" and fail "agreed", and docs/graph/EXPORT.md §4.2 is the argument that such a
graph would not look wrong. So `exported` alone is never reported as a success
here: the headline number is `agreed`.

How to run it
-------------
Not wired into `run.sh`, for the same reasons `arch_sweep.py` is not: it takes
minutes, constructs hundreds of models, and produces a measurement rather than an
invariant. Run it by hand, both sides, then compare:

    PY=/Volumes/macMini/caches/spike-venv/bin/python
    root=$(git rev-parse --show-toplevel)
    cd $root/rust/torch_c/pytests

    PYTHONPATH=$root/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
        $PY export_sweep.py --out /tmp/exp-shim.json
    env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL \
        $PY export_sweep.py --out /tmp/exp-up.json

    $PY export_sweep.py --compare /tmp/exp-shim.json /tmp/exp-up.json

The upstream run is not optional, and here it matters MORE than it does for
`arch_sweep.py`. `torch.export` refuses plenty of real architectures on upstream
torch as well -- data-dependent control flow, `.item()` on a traced tensor, mutated
module state. Counting those as this shim's gaps would invent a deficit that is
actually upstream's documented behaviour. The comparison mode attributes nothing
that upstream also refuses.

Method, and its limits
----------------------
* Model construction, config shrinking, input synthesis and failure
  classification are all `arch_sweep.py`'s, imported rather than copied. The two
  sweeps must build the SAME model from the SAME config, or the export result
  cannot be lined up against the forward result.
* Every architecture runs in its own subprocess, for `arch_sweep.py`'s reason: a
  gap can be a segfault as easily as an exception.
* Agreement is element-wise with a tolerance, over every floating tensor in the
  output pytree, and the WORST deviation is recorded rather than a boolean. A
  boolean cannot distinguish "bit-identical" from "just inside tolerance", and
  the difference is the whole question when a decomposition has been applied.
* `agreed` is judged shim-vs-its-own-eager, and separately the comparison mode
  lines the shim's replay up against UPSTREAM's replay. The first catches a graph
  that lost operators; only the second catches a kernel that is wrong in the same
  way on both paths.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arch_sweep import (  # noqa: E402
    classify,
    input_candidates,
    shrink,
    side_of,
)

TOL = 1e-5


def architectures(limit=None):
    from arch_sweep import architectures as _a
    return _a(limit)


def _flat_floats(torch, obj, out=None):
    """Every floating tensor in an arbitrary output pytree, in a stable order.

    `transformers` returns `ModelOutput` dataclasses whose fields are tensors,
    tuples of tensors, or `None`. Walking it by hand rather than with
    `torch.utils._pytree` keeps this working on the shim, where the pytree
    registration for those dataclasses is not guaranteed to be the same.
    """
    if out is None:
        out = []
    if obj is None:
        return out
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            out.append(obj)
        return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _flat_floats(torch, v, out)
        return out
    if isinstance(obj, dict):
        for k in sorted(obj):
            _flat_floats(torch, obj[k], out)
        return out
    if hasattr(obj, "to_tuple"):
        try:
            return _flat_floats(torch, obj.to_tuple(), out)
        except Exception:
            return out
    return out


def _worst_deviation(torch, a, b):
    """Worst element-wise absolute deviation between two output pytrees.

    Returns `(worst, n_compared)` or raises if the two trees do not line up --
    a shape or arity mismatch is a disagreement of a more serious kind than a
    numeric one and must not be averaged into a small number.
    """
    fa, fb = _flat_floats(torch, a), _flat_floats(torch, b)
    if len(fa) != len(fb):
        raise AssertionError(f"output arity differs: {len(fa)} vs {len(fb)}")
    worst, n = 0.0, 0
    for x, y in zip(fa, fb):
        if tuple(x.shape) != tuple(y.shape):
            raise AssertionError(f"output shape differs: {tuple(x.shape)} vs {tuple(y.shape)}")
        d = (x.detach().float() - y.detach().float()).abs().max().item()
        worst = max(worst, d)
        n += x.numel()
    return worst, n


def _summary(torch, obj):
    """A side-independent digest of an output pytree, for cross-side comparison.

    Values rather than a hash: a hash would say "differs" without saying by how
    much, and the amount is what decides whether a disagreement is a wrong kernel
    or a different decomposition.
    """
    out = []
    for t in _flat_floats(torch, obj):
        f = t.detach().float().flatten()
        out.append({
            "shape": list(t.shape),
            "head": [round(v, 6) for v in f[:8].tolist()],
            "sum": round(f.sum().item(), 5),
        })
    return out


def run_one(model_type):
    """Build one architecture, export it, replay it, and judge agreement."""
    import torch
    from transformers import AutoConfig, AutoModel

    out = {"model_type": model_type, "side": side_of(torch)}
    torch.manual_seed(0)

    try:
        cfg = AutoConfig.for_model(model_type)
        shrink(cfg)
        cfg.output_attentions = False
        cfg.output_hidden_states = False
        model = AutoModel.from_config(cfg)
        model.eval()
    except Exception:
        out.update(stage="construction", status="fail", error=traceback.format_exc())
        return out
    out["arch_class"] = type(model).__name__

    # --- eager, first: an architecture that cannot run cannot be exported ----
    try:
        cands = input_candidates(torch, model, cfg)
    except Exception:
        out.update(stage="inputs", status="fail", error=traceback.format_exc())
        return out

    eager, kwargs, errors = None, None, []
    for label, kw in cands:
        try:
            with torch.no_grad():
                eager = model(**kw)
        except Exception:
            errors.append(traceback.format_exc())
            continue
        kwargs, out["inputs"] = kw, label
        break
    if eager is None:
        out.update(stage="forward", status="fail",
                   error=next((e for e in errors if classify(e)[0] != "unclassified"), errors[0]))
        return out

    # --- export -------------------------------------------------------------
    try:
        with torch.no_grad():
            ep = torch.export.export(model, (), dict(kwargs), strict=False)
    except Exception:
        out.update(stage="export", status="fail", error=traceback.format_exc())
        return out
    try:
        out["n_call_function"] = sum(1 for n in ep.graph.nodes if n.op == "call_function")
        out["n_placeholder"] = sum(1 for n in ep.graph.nodes if n.op == "placeholder")
    except Exception:
        out["n_call_function"] = -1

    # An `ExportedProgram` with no operators is the §4.2 failure. It is recorded
    # as its own stage rather than being allowed to pass into replay, where it
    # would happily return the placeholder and "agree" on an identity module.
    if out.get("n_call_function") == 0:
        out.update(stage="export_empty", status="fail",
                   error="ExportedProgram holds zero call_function nodes")
        return out

    # --- replay -------------------------------------------------------------
    try:
        with torch.no_grad():
            replayed = ep.module()(**kwargs)
    except Exception:
        out.update(stage="replay", status="fail", error=traceback.format_exc())
        return out

    # --- agreement ----------------------------------------------------------
    try:
        worst, n = _worst_deviation(torch, eager, replayed)
    except Exception:
        out.update(stage="agree", status="fail", error=traceback.format_exc())
        return out
    out["worst"] = worst
    out["n_elements"] = n
    out["digest"] = _summary(torch, replayed)
    if worst > TOL:
        out.update(stage="agree", status="fail",
                   error=f"replay disagrees with eager: worst |delta| = {worst:.3e} over {n} elements")
        return out
    out.update(stage="agree", status="ok")
    return out


def sweep(out_path, limit, timeout, only):
    import torch
    side = side_of(torch)
    types = only or architectures(limit)
    print(f"side={side}  architectures={len(types)}", flush=True)
    results = []
    for i, mt in enumerate(types, 1):
        cmd = [sys.executable, os.path.abspath(__file__), "--one", mt]
        proc = None
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            rec = json.loads(line)
        except subprocess.TimeoutExpired:
            rec = {"model_type": mt, "side": side, "stage": "unknown",
                   "status": "fail", "error": "TIMEOUT"}
        except Exception:
            rc = getattr(proc, "returncode", None)
            err = getattr(proc, "stderr", "") or ""
            rec = {"model_type": mt, "side": side, "stage": "unknown", "status": "fail",
                   "error": f"NO_JSON rc={rc}\n{err[-4000:]}"}
        rec["kind"], rec["operator"] = (("", None) if rec["status"] == "ok"
                                        else classify(rec.get("error", "")))
        results.append(rec)
        mark = "ok " if rec["status"] == "ok" else f"{rec['stage'][:6]:6s}"
        print(f"[{i:3d}/{len(types)}] {mark} {mt:32s} {rec.get('operator') or ''}", flush=True)
    payload = {"side": side, "count": len(results), "results": results}
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    ok = sum(r["status"] == "ok" for r in results)
    print(f"\nwrote {out_path}: {ok}/{len(results)} exported+replayed+agreed")
    return 0


def compare(shim_path, up_path):
    shim = json.load(open(shim_path))
    up = json.load(open(up_path))
    assert shim["side"] == "shim", f"{shim_path} is a {shim['side']} run"
    assert up["side"] == "upstream", f"{up_path} is a {up['side']} run"
    ups = {r["model_type"]: r for r in up["results"]}

    total = len(shim["results"])
    print(f"architectures swept                   : {total}")

    # Only architectures upstream itself exports are judgeable. Everything else
    # is `torch.export` refusing a model, not this shim refusing one.
    up_ok = [r["model_type"] for r in up["results"] if r["status"] == "ok"]
    print(f"upstream exported+replayed+agreed     : {len(up_ok)}")

    stages = {}
    agree_ok, cross = 0, []
    for r in shim["results"]:
        mt = r["model_type"]
        if mt not in set(up_ok):
            continue
        if r["status"] == "ok":
            agree_ok += 1
            u = ups[mt]
            if r.get("digest") and u.get("digest"):
                d = _cross(r["digest"], u["digest"])
                if d is not None:
                    cross.append((mt, d))
        else:
            stages.setdefault(r["stage"], []).append((mt, r.get("operator")))

    print(f"of those {len(up_ok)}, this shim exported+replayed+agreed : {agree_ok}")
    if up_ok:
        print(f"                                        {100.0 * agree_ok / len(up_ok):.1f}%")

    if cross:
        cross.sort(key=lambda t: -t[1])
        agreeing = sum(1 for _, d in cross if d <= TOL)
        print(f"\ncross-side: shim replay vs upstream replay, {len(cross)} comparable")
        print(f"            agreeing within {TOL:g}: {agreeing}/{len(cross)}")
        print("            worst 10:")
        for mt, d in cross[:10]:
            print(f"              {mt:30s} {d:.3e}")

    if stages:
        print("\nwhere this shim stops, on architectures upstream exports:")
        for st in sorted(stages, key=lambda s: -len(stages[s])):
            rows = stages[st]
            print(f"  {st:14s} {len(rows):3d}")
            ops = {}
            for mt, op in rows:
                ops.setdefault(op or "(unclassified)", []).append(mt)
            for op in sorted(ops, key=lambda o: -len(ops[o]))[:12]:
                print(f"      {len(ops[op]):3d}  {op}")
    return 0


def _cross(a, b):
    """Worst deviation between two side digests, or None if they do not line up."""
    if len(a) != len(b):
        return None
    worst = 0.0
    for x, y in zip(a, b):
        if x["shape"] != y["shape"] or len(x["head"]) != len(y["head"]):
            return None
        for p, q in zip(x["head"], y["head"]):
            worst = max(worst, abs(p - q))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--compare", nargs=2, metavar=("SHIM", "UPSTREAM"))
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if args.one:
        import torch  # noqa: F401
        print(side_of(sys.modules["torch"]), file=sys.stderr)
        print(json.dumps(run_one(args.one)))
        return 0
    if args.out:
        return sweep(args.out, args.limit, args.timeout, args.only)
    ap.error("one of --one/--out/--compare")


if __name__ == "__main__":
    sys.exit(main())
