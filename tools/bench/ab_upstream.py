"""A/B this shim against upstream torch, in the repository, reproducibly.

`docs/perf/PERF.md` was measured with a script at `/tmp/bench.py`, outside the
tree, so nobody could re-run it. This is that script, kept.

Two things make the comparison fair, both inherited from PERF.md §0:

  * The two torches **cannot share a process** (same package name), so each side
    runs in its own subprocess and the driver interleaves them
    (upstream, shim, upstream, shim, ...) so machine drift cannot land on one
    side only.
  * Every op is called through ``torch.ops.aten.*`` on both sides, so a
    spelling difference in the Python surface cannot leak into the numbers.

Usage (one command, from the repo root)::

    PYTHON=/Volumes/macMini/caches/spike-venv/bin/python \
        python tools/bench/ab_upstream.py

    # or pick a suite / more rounds
    ... tools/bench/ab_upstream.py --suite ops --rounds 5
    ... tools/bench/ab_upstream.py --suite model

Suites:

  ``ops``    the microbenchmark of PERF.md §1 -- mm, _softmax, add, sum, tanh
             and a two-layer block, plus bfloat16 mm.
  ``model``  SmolLM2-135M (from the local HF cache): prefill in f32 and
             bfloat16, and greedy decode tok/s.  Reproduces the claims in
             docs/perf/DTYPE_PERF.md and docs/perf/DISPATCH2.md.
  ``all``    both.

Reporting: median of the pooled timed repetitions, with min and max, the
repetition count, and the load average before and after every worker run.
A run whose load average rises materially is reported as such -- see PERF.md §0
on why absolute numbers taken under load are not regression evidence.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIM_PATH = os.path.join(REPO, "torchnative", "src", "main")

WARMUP = 5
REPS = 15


def loadavg():
    return os.getloadavg()


# --------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------


def _time(fn, warmup=WARMUP, reps=REPS):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return {"t": out, "warmup": warmup}


def suite_ops(torch):
    aten = torch.ops.aten
    res = {}

    a256 = torch.randn(256, 256)
    b256 = torch.randn(256, 256)
    res["mm 256x256"] = _time(lambda: aten.mm.default(a256, b256))

    a512 = torch.randn(512, 512)
    b512 = torch.randn(512, 512)
    res["mm 512x512"] = _time(lambda: aten.mm.default(a512, b512))

    s = torch.randn(64, 1024)
    res["_softmax 64x1024"] = _time(lambda: aten._softmax.default(s, -1, False))
    res["add 512x512"] = _time(lambda: aten.add.Tensor(a512, b512))
    res["sum 512x512"] = _time(lambda: aten.sum.default(a512))
    res["tanh 64x1024"] = _time(lambda: aten.tanh.default(s))

    # Two-layer block: the shape a transformer FFN actually makes.
    x = torch.randn(64, 1024)
    w1 = torch.randn(1024, 1024)
    b1 = torch.randn(1024)
    w2 = torch.randn(1024, 1024)
    b2 = torch.randn(1024)

    def block():
        h = aten.addmm.default(b1, x, w1)
        h = aten.tanh.default(h)
        h = aten.addmm.default(b2, h, w2)
        return aten._softmax.default(h, -1, False)

    res["2-layer block"] = _time(block)

    # bfloat16 matmul: PERF.md §6 lists dtype as unmeasured.
    a512b = a512.to(torch.bfloat16)
    b512b = b512.to(torch.bfloat16)
    res["mm 512x512 bf16"] = _time(lambda: aten.mm.default(a512b, b512b))

    # The non-contiguous weight layout that DTYPE_PERF.md §4 found is the one
    # every nn.Linear actually produces (`linear` passes `t(weight)`).
    wt = torch.randn(512, 512).t()
    res["mm 512x512 (t) f32"] = _time(lambda: aten.mm.default(a512, wt))
    wtb = wt.to(torch.bfloat16)
    res["mm 512x512 (t) bf16"] = _time(lambda: aten.mm.default(a512b, wtb))
    return res


MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
PREFILL_LENS = [16, 64, 128, 512]
NEW_TOKENS = 24


def suite_model(torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    res = {}
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    for name, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
        model.eval()
        for S in PREFILL_LENS:
            ids = torch.arange(1, S + 1, dtype=torch.int64).reshape(1, S) % 4096
            with torch.no_grad():
                res["prefill S=%-3d %s" % (S, name)] = _time(
                    lambda ids=ids: model(ids), warmup=2, reps=5
                )
        del model

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    prompt = tok("The capital of France is", return_tensors="pt")

    def gen():
        with torch.no_grad():
            model.generate(
                **prompt,
                max_new_tokens=NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

    res["decode %d tokens f32" % NEW_TOKENS] = _time(gen, warmup=1, reps=5)
    res["__decode_tokens__"] = {"t": [float(NEW_TOKENS)], "warmup": 0}
    return res


def worker(suite):
    os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
    import torch

    out = {"torch": torch.__version__, "file": torch.__file__}
    try:
        out["ops"] = len(torch._C._aten_implemented())
    except Exception:
        out["ops"] = None
    data = {}
    if suite in ("ops", "all"):
        data.update(suite_ops(torch))
    if suite in ("model", "all"):
        data.update(suite_model(torch))
    out["data"] = data
    sys.stdout.write("@@JSON@@" + json.dumps(out) + "\n")


# --------------------------------------------------------------------------
# driver side
# --------------------------------------------------------------------------


def run_side(python, side, suite):
    env = dict(os.environ)
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env["PYTHONPATH"] = SHIM_PATH if side == "shim" else ""
    if side == "upstream":
        env.pop("PYTHONPATH")
    before = loadavg()
    p = subprocess.run(
        [python, os.path.abspath(__file__), "--worker", "--suite", suite],
        env=env,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    after = loadavg()
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:] + "\n" + p.stderr[-4000:] + "\n")
        raise SystemExit("%s worker failed (exit %d)" % (side, p.returncode))
    line = [l for l in p.stdout.splitlines() if l.startswith("@@JSON@@")]
    if not line:
        sys.stderr.write(p.stdout[-4000:] + "\n")
        raise SystemExit("%s worker produced no result" % side)
    res = json.loads(line[-1][len("@@JSON@@"):])
    res["load_before"] = before
    res["load_after"] = after
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--suite", default="ops", choices=["ops", "model", "all"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    args = ap.parse_args()

    if args.worker:
        worker(args.suite)
        return

    pooled = {"upstream": {}, "shim": {}}
    meta = {}
    loads = []
    warmups = {}
    for r in range(args.rounds):
        for side in ("upstream", "shim"):
            res = run_side(args.python, side, args.suite)
            meta[side] = {k: res[k] for k in ("torch", "file", "ops")}
            loads.append((r + 1, side, res["load_before"][0], res["load_after"][0]))
            for k, v in res["data"].items():
                pooled[side].setdefault(k, []).extend(v["t"])
                warmups[k] = warmups.get(k, 0) + v["warmup"]

    print("repo:        %s" % REPO)
    print("python:      %s" % args.python)
    print("suite:       %s   rounds: %d" % (args.suite, args.rounds))
    for side in ("upstream", "shim"):
        m = meta[side]
        print("%-9s torch %s  ops=%s  %s" % (side, m["torch"], m["ops"], m["file"]))
    print()
    print("load average (1 min) around each worker run:")
    for r, side, b, a in loads:
        print("  round %d %-9s before %.2f  after %.2f" % (r, side, b, a))
    lo = min(min(b, a) for _, _, b, a in loads)
    hi = max(max(b, a) for _, _, b, a in loads)
    print("  range %.2f -- %.2f%s" % (lo, hi, "   *** load moved, treat as void" if hi - lo > 2.0 else ""))
    print()

    keys = [k for k in pooled["upstream"] if not k.startswith("__")]
    print("%-24s %26s %26s %8s %6s" % ("", "upstream  med [min-max] ms", "shim  med [min-max] ms", "shim/up", "warm"))
    print("-" * 90)
    for k in keys:
        u = sorted(pooled["upstream"][k])
        s = sorted(pooled["shim"].get(k, []))
        if not s:
            continue
        um, sm = statistics.median(u), statistics.median(s)
        print("%-24s %10.3f [%7.3f-%8.3f] %10.3f [%7.3f-%8.3f] %7.3fx %6d"
              % (k, um, u[0], u[-1], sm, s[0], s[-1], sm / um if um else float("nan"), warmups[k] // 2))
    print("\nn = %d timed repetitions per side per item, pooled over %d interleaved rounds;"
          % (len(pooled["upstream"][keys[0]]), args.rounds))
    print("'warm' is the total number of warmup iterations discarded per side before timing.")

    for k in keys:
        if k.startswith("decode "):
            n = int(pooled["upstream"]["__decode_tokens__"][0])
            um = statistics.median(pooled["upstream"][k])
            sm = statistics.median(pooled["shim"][k])
            print("decode throughput: upstream %.1f tok/s   shim %.1f tok/s"
                  % (n / (um / 1000.0), n / (sm / 1000.0)))


if __name__ == "__main__":
    main()
