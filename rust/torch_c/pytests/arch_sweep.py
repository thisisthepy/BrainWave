#!/usr/bin/env python3
"""Run a forward pass on every `transformers` architecture and record where it stops.

Why this exists
---------------
docs/DEMAND*.md is a *deep* method: pick a handful of models, chase each wall until
it falls, measure the numbers. This script is the same method run *wide*. It asks
one question the deep rounds structurally cannot answer:

    across a large set of real architectures, how many DISTINCT operators does
    this shim still lack, and how many architectures does each one block?

That ranking is the deliverable. "300 ops missing" is not actionable; "these 20
unblock 60 architectures" is, and demand-driven work needs the second form.

How to run it
-------------
It is NOT wired into `run.sh` and must not be: it needs no network but it takes
minutes, constructs hundreds of models, and its result is a measurement rather
than an invariant. Run it by hand, both sides, then compare:

    PY=/Volumes/macMini/caches/spike-venv/bin/python
    root=$(git rev-parse --show-toplevel)

    PYTHONPATH=$root/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
        $PY arch_sweep.py --out /tmp/shim.json
    env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL \
        $PY arch_sweep.py --out /tmp/upstream.json

    $PY arch_sweep.py --compare /tmp/shim.json /tmp/upstream.json

The upstream run is not optional. An architecture that fails on upstream torch too
is not our gap -- it is a `transformers` config that needs a real vocab file, or a
missing `torchaudio`, or an architecture that simply does not build from a config.
Counting those would inflate the one number this script exists to produce. The
comparison mode refuses to attribute anything that upstream also refuses.

Method, and its limits
----------------------
* Random weights, no checkpoints. `AutoModel.from_config` on a shrunk config
  reaches the same *operators* as a real checkpoint -- the graph is decided by the
  config, not by the values in it. That claim is checked rather than assumed:
  `--verify-random-weights <model_type>` runs one architecture from a real
  downloaded checkpoint and from a random-weight config of the same shape, and
  compares the sets of aten keys each dispatches.
* Every architecture runs in its OWN SUBPROCESS. A shim gap can be a segfault as
  easily as an exception, and one crash must not take the sweep with it.
* Failures are classified from the refusal text, and anything that does not match
  a known shape goes to `unclassified` rather than being guessed at. A wrong
  classification here becomes a wrong roadmap.
* Construction failures and forward failures are counted separately. `rwkv`
  (docs/kernels/PRIMS.md §6) reached its forward only after two walls fell and its
  remaining wall is `torch.linalg.qr` inside `_init_weights` -- a construction
  problem. If half the failures are construction the headline means something
  different, so the split is always reported.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import traceback

# --------------------------------------------------------------------------
# side detection -- every script in this project says which torch it got
# --------------------------------------------------------------------------


def side_of(torch_mod) -> str:
    return "shim" if hasattr(torch_mod._C, "_aten_implemented") else "upstream"


# --------------------------------------------------------------------------
# config shrinking
# --------------------------------------------------------------------------

# A tiny config reaches the same operators as a full one: depth and width do not
# change which kernels run. Two layers rather than one, because a single layer can
# hide an op that only fires between blocks.
_SHRINK = {
    "hidden_size": 32,
    "d_model": 32,
    "n_embd": 32,
    "embed_dim": 32,
    "num_hidden_layers": 2,
    "num_layers": 2,
    "n_layer": 2,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "num_attention_heads": 2,
    "n_head": 2,
    "num_heads": 2,
    "encoder_attention_heads": 2,
    "decoder_attention_heads": 2,
    "num_key_value_heads": 2,
    "attention_hidden_size": 32,
    "projection_dim": 32,
    "decoder_hidden_size": 32,
    "encoder_hidden_size": 32,
    "cross_attention_hidden_size": 32,
    "d_ff": 37,
    "d_kv": 16,
    "n_positions": 128,
    "intermediate_size": 37,
    "ffn_dim": 37,
    "encoder_ffn_dim": 37,
    "decoder_ffn_dim": 37,
    "n_inner": 37,
    "vocab_size": 99,
    "max_position_embeddings": 128,
    "image_size": 32,
    "patch_size": 4,
    "num_channels": 3,
    "num_mel_bins": 32,
    "num_labels": 3,
    "num_queries": 4,
    "num_experts": 2,
    "num_local_experts": 2,
    "num_codebooks": 2,
    "hidden_sizes": None,   # list-valued; handled below
    "depths": None,
}
# list-valued fields get a short list of the scalar value
_SHRINK_LISTS = {"hidden_sizes": 32, "depths": 1, "num_attention_heads_list": 2}


def shrink(cfg, seen=None):
    """Make a config small in place, recursing into nested configs.

    Only attributes the config already has are touched -- inventing attributes
    makes models fail for reasons that are ours rather than theirs.
    """
    seen = seen if seen is not None else set()
    if id(cfg) in seen:
        return cfg
    seen.add(id(cfg))
    for key, val in _SHRINK.items():
        if val is None or not hasattr(cfg, key):
            continue
        cur = getattr(cfg, key)
        if isinstance(cur, (list, tuple)):
            continue
        if not isinstance(cur, int) or isinstance(cur, bool):
            continue
        # never grow: a config that already asks for less stays as it is
        setattr(cfg, key, min(cur, val) if cur > 0 else val)
    for key, val in _SHRINK_LISTS.items():
        cur = getattr(cfg, key, None)
        if isinstance(cur, (list, tuple)) and cur and all(isinstance(x, int) for x in cur):
            setattr(cfg, key, type(cur)([val] * min(len(cur), 3)))
    # a mixture-of-experts config routes to `num_experts_per_tok` of its experts,
    # and `topk` on a shrunk expert list raises "selected index k out of range" --
    # our shrinking's failure, not the architecture's
    for k in ("num_experts_per_tok", "top_k", "num_selected_experts", "moe_topk",
              "num_experts_per_token"):
        v = getattr(cfg, k, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            for e in ("num_experts", "num_local_experts", "moe_num_experts", "n_routed_experts"):
                if isinstance(getattr(cfg, e, None), int):
                    setattr(cfg, e, max(getattr(cfg, e), v))
    # `vocab_size` cannot go below the largest special token id the config names
    if isinstance(getattr(cfg, "vocab_size", None), int):
        need = 99
        for name, val in list(vars(cfg).items()):
            if name.endswith("_token_id") and isinstance(val, int) \
                    and not isinstance(val, bool) and 0 <= val < 10 ** 6:
                need = max(need, val + 1)
        cfg.vocab_size = max(cfg.vocab_size, need)
    # heads must divide width
    for hsz, nh in (("hidden_size", "num_attention_heads"), ("d_model", "encoder_attention_heads"),
                    ("d_model", "decoder_attention_heads"), ("n_embd", "n_head")):
        h, n = getattr(cfg, hsz, None), getattr(cfg, nh, None)
        if isinstance(h, int) and isinstance(n, int) and n > 0 and h % n:
            setattr(cfg, hsz, n * max(1, h // n))
    # nested configs (vision_config, text_config, encoder, ...)
    for name, val in list(vars(cfg).items()):
        if hasattr(val, "to_dict") and hasattr(val, "model_type"):
            shrink(val, seen)
        elif isinstance(val, (list, tuple)):
            for item in val:
                if hasattr(item, "to_dict") and hasattr(item, "model_type"):
                    shrink(item, seen)
    return cfg


# --------------------------------------------------------------------------
# input construction
# --------------------------------------------------------------------------


def _vocab(cfg):
    """A vocabulary big enough to hold every special token id the config names.

    Shrinking `vocab_size` to 99 is what makes these models tiny, but a config
    that also carries `decoder_start_token_id=50257` then fails in
    `nn.Embedding` with "Padding_idx must be within num_embeddings" -- a
    construction failure that is entirely our shrinking and not a gap. So the
    floor is raised to clear the largest id the config mentions.
    """
    need = 99
    for name, val in list(vars(cfg).items()):
        if name.endswith(("_token_id", "_token_ids", "_index")) and isinstance(val, int) \
                and not isinstance(val, bool) and 0 <= val < 10 ** 6:
            need = max(need, val + 1)
        elif name.endswith(("_token_id", "_token_ids")) and isinstance(val, (list, tuple)):
            for v in val:
                if isinstance(v, int) and 0 <= v < 10 ** 6:
                    need = max(need, v + 1)
    return min(need, int(_get(cfg, "vocab_size", default=need) or need))


def _get(cfg, *names, default=None):
    for n in names:
        v = getattr(cfg, n, None)
        if v is not None:
            return v
        for sub in ("text_config", "vision_config", "audio_config", "encoder", "decoder"):
            s = getattr(cfg, sub, None)
            if s is not None:
                v = getattr(s, n, None)
                if v is not None:
                    return v
    return default


def _modality(torch, name, cfg):
    kwargs = {}
    if name == "input_ids":
        vocab = _vocab(cfg)
        kwargs["input_ids"] = torch.randint(0, max(2, vocab), (1, 8))
        kwargs["attention_mask"] = torch.ones(1, 8, dtype=torch.long)
    elif name == "pixel_values":
        ch = int(_get(cfg, "num_channels", default=3) or 3)
        sz = _get(cfg, "image_size", default=32) or 32
        sz = int(sz[0]) if isinstance(sz, (list, tuple)) else int(sz)
        kwargs["pixel_values"] = torch.randn(1, ch, sz, sz)
    elif name == "input_features":
        mel = int(_get(cfg, "num_mel_bins", "feature_size", "input_feat_per_channel",
                       default=32) or 32)
        # Whisper and its relatives hard-check this length against the positional
        # embedding table, so it is derived rather than fixed.
        msp = _get(cfg, "max_source_positions")
        length = 2 * int(msp) if isinstance(msp, int) and msp else 32
        kwargs["input_features"] = torch.randn(1, mel, length)
    elif name == "input_values":
        kwargs["input_values"] = torch.randn(1, 4000)
    else:
        kwargs[name] = torch.randn(1, 8, int(_get(cfg, "hidden_size", "d_model", default=32) or 32))
    return kwargs


def input_candidates(torch, model, cfg):
    """Candidate input dicts, tried in order until one gets past argument errors.

    Two candidates rather than one, because multimodal models (CLIP and its many
    descendants) declare a single `main_input_name` but refuse a forward that
    carries only that modality. Feeding every modality unconditionally is worse:
    single-modality models accept the extra keyword and then take a *different*
    path through the graph. So: the declared modality first, everything the
    signature accepts second, and the record says which one ran.
    """
    import inspect
    main = getattr(model, "main_input_name", "input_ids")
    base = _modality(torch, main, cfg)
    if getattr(cfg, "is_encoder_decoder", False) or _get(cfg, "is_encoder_decoder"):
        vocab = _vocab(cfg)
        base["decoder_input_ids"] = torch.randint(0, max(2, vocab), (1, 4))
    try:
        params = set(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        return [("main_only", base)]
    wide = dict(base)
    for other in ("input_ids", "pixel_values", "input_values", "input_features"):
        if other != main and other in params:
            wide.update(_modality(torch, other, cfg))
    cands = [("main_only", base)]
    if wide != base:
        cands.append(("all_modalities", wide))
    return cands


# --------------------------------------------------------------------------
# the classifier
# --------------------------------------------------------------------------

# Each pattern is a shape this shim actually raises, read off bootstrap.py and
# aten.rs rather than invented. Order matters: the more specific first.
_RULES = [
    # `aten op not implemented in torch._C shim: aten.foo.default`
    ("missing_aten_op", re.compile(r"aten op not implemented in torch\._C shim:\s*(\S+)")),
    # `torch.foo(...) -- overload resolution has no table entry for this op`
    ("missing_aten_op", re.compile(
        r"not implemented in torch\._C shim:\s*torch\.(\w+)\(\.\.\.\)\s*--\s*overload")),
    ("missing_aten_op", re.compile(r"overload resolution has no table entry for this op[^\n]*?"
                                   r"\b(aten\.[\w.]+)")),
    # `not implemented in torch._C shim: TensorBase.foo` / `torch._C._nn.foo` / ...
    ("missing_shim_name", re.compile(r"not implemented in torch\._C shim:\s*([\w.]+)")),
    # a name the shim's torch module simply does not carry
    ("missing_torch_spelling", re.compile(
        r"module '(torch(?:\.[\w.]+)?)' has no attribute '(\w+)'")),
    ("missing_nn_module", re.compile(
        r"module 'torch\.nn(?:\.[\w.]+)?' has no attribute '(\w+)'")),
    ("missing_tensor_method", re.compile(
        r"'(?:Tensor|TensorBase)' object has no attribute '(\w+)'")),
    ("missing_dependency", re.compile(r"No module named '([\w.]+)'")),
    # `torch.ones(): no matching overload in torch._C shim for (tuple, dtype=type)`
    # The operator EXISTS; the argument form does not. Counting these as missing
    # operators would overstate the gap, and they are much cheaper to close.
    ("unsupported_arg_form", re.compile(
        r"((?:torch|Tensor)\.\w+)\(\): no matching overload in torch\._C shim for (\([^)]*\))")),
    # `aten.convolution.default: an asymmetric padding [32, 0] is not implemented`
    ("unsupported_arg_form", re.compile(
        r"(aten\.[\w.]+): [^\n]*not implemented in torch\._C shim")),
    # candle refusing a layout the kernel could in principle accept
    ("backend_limitation", re.compile(r"(aten\.[\w.]+): candle: (\w+)")),
]


def classify(text):
    """(kind, operator) or ('unclassified', None). Never guesses."""
    if not text:
        return "unclassified", None
    # a missing third-party package is never an operator gap
    m = re.search(r"No module named '([\w.]+)'", text)
    if m and not m.group(1).startswith("torch"):
        return "missing_dependency", m.group(1)
    for kind, pat in _RULES:
        m = pat.search(text)
        if m:
            return kind, ".".join(g for g in m.groups() if g)
    return "unclassified", None


# --------------------------------------------------------------------------
# worker: one architecture, in its own process
# --------------------------------------------------------------------------


def run_one(model_type):
    import torch
    from transformers import AutoConfig, AutoModel

    out = {"model_type": model_type, "side": side_of(torch)}
    torch.manual_seed(0)

    # --- construction -------------------------------------------------
    try:
        cfg = AutoConfig.for_model(model_type)
        shrink(cfg)
        cfg.output_attentions = False
        cfg.output_hidden_states = False
        model = AutoModel.from_config(cfg)
        model.eval()
    except Exception:
        out["stage"] = "construction"
        out["status"] = "fail"
        out["error"] = traceback.format_exc()
        return out
    out["arch_class"] = type(model).__name__

    # --- forward ------------------------------------------------------
    try:
        cands = input_candidates(torch, model, cfg)
    except Exception:
        out["stage"] = "inputs"
        out["status"] = "fail"
        out["error"] = traceback.format_exc()
        return out
    errors = []
    for label, kwargs in cands:
        try:
            with torch.no_grad():
                model(**kwargs)
        except Exception:
            errors.append(traceback.format_exc())
            continue
        out["stage"] = "forward"
        out["status"] = "ok"
        out["inputs"] = label
        return out
    out["stage"] = "forward"
    out["status"] = "fail"
    # Report the first candidate whose refusal is CLASSIFIABLE, not simply the
    # last one tried. The fallback candidate feeds modalities the architecture
    # may not want, and its failure is then about the inputs rather than about
    # us -- which buried real refusals from the first candidate under
    # "'NoneType' object has no attribute 'device'" in an earlier run.
    out["error"] = next((e for e in errors if classify(e)[0] != "unclassified"), errors[0])
    out["errors_all"] = errors
    return out


# --------------------------------------------------------------------------
# the claim that random weights are enough, checked rather than asserted
# --------------------------------------------------------------------------


def _trace(torch, model, cfg):
    """Run one forward and return the set of aten keys it dispatched.

    Uses the shim's own `torch._C._aten_dispatch`, so what is recorded is exactly
    the set of keys the shim must supply -- not a proxy for it.
    """
    keys = set()
    orig = torch._C._aten_dispatch

    def traced(key, *a, **k):
        keys.add(str(key))
        return orig(key, *a, **k)

    torch._C._aten_dispatch = traced
    try:
        with torch.no_grad():
            model(**input_candidates(torch, model, cfg)[0][1])
    finally:
        torch._C._aten_dispatch = orig
    return keys


def verify_random_weights(model_type, checkpoint):
    """Does a random-weight instance reach the same operators as a real checkpoint?

    The whole sweep rests on this claim, so it is measured rather than asserted.
    Three instances of the SAME architecture are traced:

      1. the real checkpoint, weights loaded from the hub
      2. random weights at the checkpoint's exact shape
      3. random weights at this script's shrunk config

    (1) vs (2) isolates *weights*; (2) vs (3) isolates *shape*. If both sets match,
    neither the values nor the size decide which kernels run -- the config does,
    which is what makes a hundred-architecture sweep possible without a hundred
    downloads.

    Requires the network. Set HF_HOME somewhere disposable before running.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    side = side_of(torch)
    print(f"side={side}")
    if side != "shim":
        print("this check traces `torch._C._aten_dispatch` and needs the shim")
        return 2
    torch.manual_seed(0)

    ck_cfg = AutoConfig.from_pretrained(checkpoint)
    ck = AutoModel.from_pretrained(checkpoint).eval()
    a = _trace(torch, ck, ck_cfg)

    torch.manual_seed(0)
    rand = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint)).eval()
    b = _trace(torch, rand, ck_cfg)

    tiny_cfg = shrink(AutoConfig.for_model(model_type))
    tiny = AutoModel.from_config(tiny_cfg).eval()
    c = _trace(torch, tiny, tiny_cfg)

    print(f"checkpoint {checkpoint}")
    print(f"  1 real weights, real shape : {len(a):3d} aten keys")
    print(f"  2 random weights, real shape: {len(b):3d}   equal to (1): {a == b}")
    print(f"  3 random weights, tiny shape: {len(c):3d}   equal to (1): {a == c}")
    print(f"  in (1) not (3): {sorted(a - c)}")
    print(f"  in (3) not (1): {sorted(c - a)}")
    return 0


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def architectures(limit=None):
    """Every model type `AutoModel` can build, in alphabetical order.

    Alphabetical and exhaustive, deliberately: the point of this sweep is the
    TAIL. Choosing by popularity would answer a different and easier question --
    a hundred decoder-only LLMs share one operator set, and the operators that
    are missing are in the vision, audio, encoder-decoder and multimodal corners
    that no popularity ranking reaches.
    """
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
    names = sorted(MODEL_MAPPING_NAMES)
    return names[:limit] if limit else names


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def sweep(out_path, limit, timeout, only):
    import torch
    side = side_of(torch)
    types = only or architectures(limit)
    print(f"side={side}  architectures={len(types)}", flush=True)
    results = []
    for i, mt in enumerate(types, 1):
        cmd = [sys.executable, os.path.abspath(__file__), "--one", mt]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            rec = json.loads(line)
        except subprocess.TimeoutExpired:
            rec = {"model_type": mt, "side": side, "stage": "unknown",
                   "status": "fail", "error": "TIMEOUT"}
        except Exception:
            # a crash (segfault) leaves no JSON: that is itself a result, and the
            # signal is in the return code
            rc = getattr(locals().get("proc", None), "returncode", None)
            err = getattr(locals().get("proc", None), "stderr", "") or ""
            rec = {"model_type": mt, "side": side, "stage": "unknown", "status": "fail",
                   "error": f"NO_JSON rc={rc}\n{err[-4000:]}"}
        rec["kind"], rec["operator"] = (("", None) if rec["status"] == "ok"
                                        else classify(rec.get("error", "")))
        results.append(rec)
        mark = "ok " if rec["status"] == "ok" else f"{rec['stage'][:5]:5s}"
        print(f"[{i:3d}/{len(types)}] {mark} {mt:32s} {rec.get('operator') or ''}", flush=True)
    payload = {"side": side, "count": len(results), "results": results}
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {out_path}: {sum(r['status']=='ok' for r in results)}/{len(results)} forward")
    return 0


# --------------------------------------------------------------------------
# comparison -- the deliverable
# --------------------------------------------------------------------------


def compare(shim_path, up_path):
    shim = json.load(open(shim_path))
    up = json.load(open(up_path))
    assert shim["side"] == "shim", f"{shim_path} is a {shim['side']} run"
    assert up["side"] == "upstream", f"{up_path} is a {up['side']} run"
    ups = {r["model_type"]: r for r in up["results"]}

    total = len(shim["results"])
    up_ok = [r for r in shim["results"] if ups.get(r["model_type"], {}).get("status") == "ok"]
    print(f"architectures swept                   : {total}")
    print(f"  forward on UPSTREAM torch (baseline): {len(up_ok)}")
    print(f"  fail on upstream too (not our gap)  : {total - len(up_ok)}")
    print()

    ok = [r for r in up_ok if r["status"] == "ok"]
    bad = [r for r in up_ok if r["status"] != "ok"]
    print(f"of the {len(up_ok)} upstream-clean architectures:")
    print(f"  forward under the SHIM today        : {len(ok)}  "
          f"({100*len(ok)/max(1,len(up_ok)):.0f}%)")
    print(f"  blocked by the shim                 : {len(bad)}")
    print()

    ctor = [r for r in bad if r["stage"] == "construction"]
    fwd = [r for r in bad if r["stage"] == "forward"]
    other = [r for r in bad if r["stage"] not in ("construction", "forward")]
    print("construction vs forward split (blocked only):")
    print(f"  blocked at CONSTRUCTION             : {len(ctor)}")
    print(f"  blocked in the FORWARD              : {len(fwd)}")
    print(f"  blocked elsewhere / no JSON         : {len(other)}")
    print()

    bykind = collections.Counter(r["kind"] for r in bad)
    print("failure kinds:")
    for k, n in bykind.most_common():
        print(f"  {k or '(none)':24s} {n}")
    print()

    ops = collections.defaultdict(list)
    for r in bad:
        if r["kind"] in ("missing_aten_op", "missing_shim_name", "missing_torch_spelling",
                         "missing_nn_module", "missing_tensor_method") and r["operator"]:
            ops[(r["kind"], r["operator"])].append(r["model_type"])
    print(f"DISTINCT missing operators            : {len(ops)}")
    print()
    print("ranked: each missing operator and how many architectures it blocks")
    print("(a first wall only -- closing the top entry reveals whatever is behind it)")
    rank = sorted(ops.items(), key=lambda kv: (-len(kv[1]), kv[0][1]))
    cum = 0
    for (kind, op), models in rank:
        cum += len(models)
        print(f"  {len(models):3d}  {op:44s} {kind:22s} cum={cum}")
        print(f"       e.g. {', '.join(sorted(models)[:6])}")
    print()
    for kind, title in (("unsupported_arg_form",
                         "argument forms the shim refuses (the OP exists -- cheaper than a kernel)"),
                        ("backend_limitation",
                         "backend refusals (candle layout, not a missing op)")):
        rows = collections.defaultdict(list)
        for r in bad:
            if r["kind"] == kind and r["operator"]:
                rows[r["operator"]].append(r["model_type"])
        print(f"{title}: {len(rows)} distinct, {sum(len(v) for v in rows.values())} architectures")
        for op, models in sorted(rows.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(models):3d}  {op}")
            print(f"       e.g. {', '.join(sorted(models)[:6])}")
        print()
    unc = [r for r in bad if r["kind"] == "unclassified"]
    print(f"unclassified (NOT guessed at)         : {len(unc)}")
    for r in unc[:40]:
        last = [l for l in r.get("error", "").strip().splitlines() if l.strip()]
        print(f"  {r['model_type']:28s} {r['stage']:12s} {last[-1][:100] if last else ''}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", help="run a single model type and print one JSON line")
    ap.add_argument("--out", help="sweep and write JSON here")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--compare", nargs=2, metavar=("SHIM", "UPSTREAM"))
    ap.add_argument("--verify-random-weights", nargs=2, metavar=("MODEL_TYPE", "CHECKPOINT"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if args.list:
        for m in architectures(args.limit):
            print(m)
        return 0
    if args.verify_random_weights:
        return verify_random_weights(*args.verify_random_weights)
    if args.one:
        import torch  # noqa: F401  -- side line, every script in this project prints it
        print(side_of(sys.modules["torch"]), file=sys.stderr)
        print(json.dumps(run_one(args.one)))
        return 0
    if args.out:
        return sweep(args.out, args.limit, args.timeout, args.only)
    ap.error("one of --one/--out/--compare/--list/--verify-random-weights")


if __name__ == "__main__":
    sys.exit(main())
