"""Size the NNAPI gap against graphs captured from the models this project runs.

This is where docs/graph/DECOMP.md §12's tables come from. Like `decomp_sweep.py` it
is a measurement script and not a test: it prints a verdict per op and exits 0
whatever they are, because "how many ops NNAPI already has" is a number that
moves and a test that pinned it would go red on progress. `test_shim.py` pins
the individual claims that matter (`test_gelu_lowers_to_erf_primitives...` and
the five beside it).

Run it inside the vendored tree, which is where `torchnative.export`,
`torch._decomp` and the shim `_C` all live:

    PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
        python rust/torch_c/pytests/nnapi_sizing.py

`--json` dumps the whole record including every refusal's full text, which is
what makes the "why is this one still shut" column derivable rather than
recalled.

## Why NNAPI and not CoreML

`torch/backends/_nnapi/serializer.py` in the vendored tree carries `ADDER_MAP`,
the table that would do the serialising -- an op absent from it cannot reach
NNAPI through this PyTorch whatever the hardware supports. That is an
authority, and `export/target.py` reads it rather than transcribing it.

There is no counterpart for CoreML in this tree. `torch/backends/_coreml` is a
packaging wrapper around `coremltools.convert`; the op registry lives inside
coremltools, which is not installed here. So this script sizes NNAPI and
reports CoreML as *not measured*. Writing a plausible CoreML list would decide
the answer by the act of writing it -- see `target.coreml_ops`, which refuses
for the same reason.

## What the population is

The ops that appear in graphs captured from real model forwards, not the ops
the shim implements. Those are different populations and the second one
flatters: an op nothing calls is not a gap. The models are the ones
docs/architectures/DEMAND7.md lists as forwarding and matching upstream, at toy config sizes
-- the op *set* a model reaches does not depend on its width or depth, only on
its architecture.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import sys


def _models(torch):
    """`name -> (module, inputs)`. Toy configs; the op set is what matters."""
    import transformers as tf

    torch.manual_seed(0)
    built = {}

    # SmolLM2 is a Llama-architecture decoder-only model.
    cfg = tf.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
    )
    built["smollm2_llama"] = (
        tf.LlamaModel(cfg).eval(), (torch.zeros(1, 8, dtype=torch.long),)
    )

    cfg = tf.ViTConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, image_size=32, patch_size=16,
    )
    built["vit"] = (tf.ViTModel(cfg).eval(), (torch.ones(1, 3, 32, 32),))

    # ResNet is expected to refuse: its residual add is in place, and
    # docs/graph/CAPTURE.md §4 refuses mutation by name. Kept in the list so the
    # refusal is reported rather than quietly shrinking the denominator.
    cfg = tf.ResNetConfig(
        embedding_size=8, hidden_sizes=[8, 16], depths=[1, 1], layer_type="basic",
    )
    built["resnet"] = (tf.ResNetModel(cfg).eval(), (torch.ones(1, 3, 32, 32),))

    cfg = tf.MobileNetV2Config(image_size=32, depth_multiplier=0.25)
    built["mobilenet_v2"] = (
        tf.MobileNetV2Model(cfg).eval(), (torch.ones(1, 3, 32, 32),)
    )
    return built


def _capture(torch, model, inputs):
    C = torch._C
    C._capture_begin(list(inputs))
    try:
        with torch.no_grad():
            out = model(*inputs)
    except Exception as error:  # noqa: BLE001 -- reported, not swallowed
        C._capture_abandon()
        return None, f"forward raised: {type(error).__name__}: {error}"
    if not C._capture_active():
        reason = C._capture_reason()
        C._capture_abandon()
        return None, f"capture refused: {reason}"
    tensors = [
        v for v in (out.values() if hasattr(out, "values") else out)
        if isinstance(v, torch.Tensor)
    ]
    try:
        return C._capture_end(tensors[:1]), None
    except Exception as error:  # noqa: BLE001
        return None, f"capture refused: {type(error).__name__}: {error}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="dump the full record")
    args = parser.parse_args(argv)

    import torch

    print("shim" if hasattr(torch._C, "_aten_implemented") else "upstream")
    T = importlib.import_module("torchnative.export.target")
    D = importlib.import_module("torchnative.export.decompose")

    core_table = D.decomposition_table()
    full_table = T.full_decomposition_table()
    # Neither table dominates: the core table has `matmul` and the
    # post-autograd table does not; the post-autograd table has `gelu` and the
    # core table does not. The union is what a non-core target wants.
    union_table = dict(full_table)
    union_table.update(core_table)
    print(
        f"tables: core={len(core_table)} post_autograd={len(full_table)} "
        f"union={len(union_table)}"
    )
    print(f"NNAPI aten ops (from ADDER_MAP): {len(T.nnapi_ops())}")

    core_base = {op.split(".")[1] for op in D.core_ops()}
    inside = sorted(o for o in T.nnapi_ops() if o in core_base)
    outside = sorted(o for o in T.nnapi_ops() if o not in core_base)
    print(f"  also Core ATen ({len(inside)}): {', '.join(inside)}")
    print(f"  NOT Core ATen ({len(outside)}): {', '.join(outside)}")

    record = {
        "tables": {
            "core": len(core_table),
            "post_autograd": len(full_table),
            "union": len(union_table),
        },
        "nnapi_ops": sorted(T.nnapi_ops()),
        "nnapi_in_core": inside,
        "nnapi_outside_core": outside,
        "models": {},
    }

    traces, captured_ops = {}, set()
    for name, (model, inputs) in _models(torch).items():
        trace, error = _capture(torch, model, inputs)
        if trace is None:
            record["models"][name] = {"error": error}
            print(f"\n{name}: NOT CAPTURED -- {error[:150]}")
            continue
        traces[name] = trace
        ops = collections.Counter(node["op"] for node in trace.nodes)
        captured_ops |= set(ops)
        record["models"][name] = {"nodes": len(trace.nodes), "ops": dict(ops)}
        print(f"\n{name}: {len(trace.nodes)} nodes, {len(ops)} distinct ops")

    have = sorted(o for o in captured_ops if T.NNAPI.accepts(o))
    need = sorted(o for o in captured_ops if not T.NNAPI.accepts(o))
    print(f"\n=== union over captured graphs: {len(captured_ops)} distinct ops ===")
    print(f"  already in NNAPI's set: {len(have)}")
    print(f"  need lowering:          {len(need)}")
    record["captured_union"] = {"in_nnapi": have, "needs_lowering": need}

    for label, table in (("core", core_table), ("post_autograd", full_table),
                         ("union", union_table)):
        verdicts = {}
        per = {}
        for name, trace in traces.items():
            survey = T.survey(trace, T.NNAPI, table=table)
            per[name] = {
                "nodes": [survey["nodes_before"], survey["nodes_after"]],
                "outside": [
                    len(survey["outside_before"]), len(survey["outside_after"])
                ],
                "outside_after": survey["outside_after"],
            }
            for op, verdict in survey["verdicts"].items():
                if op not in verdicts or verdicts[op].startswith("REFUSED"):
                    verdicts[op] = verdict
        counts = collections.Counter(v.split(":")[0] for v in verdicts.values())
        record.setdefault("by_table", {})[label] = {
            "per_model": per, "verdicts": verdicts, "counts": dict(counts)
        }
        print(f"\n--- table={label} ---")
        print(
            f"  IN_TARGET {counts.get('IN_TARGET', 0)}  "
            f"LOWERED {counts.get('LOWERED', 0)}  "
            f"REFUSED {counts.get('REFUSED', 0)}"
        )
        for name, p in per.items():
            print(
                f"  {name}: nodes {p['nodes'][0]}->{p['nodes'][1]}, "
                f"ops outside NNAPI {p['outside'][0]}->{p['outside'][1]}"
            )

    # The classification that matters: a refusal because upstream has no rule
    # is a decomposition to write; a refusal because running upstream's rule hit
    # a missing `prims.*` op is a kernel to add, and there are far fewer of
    # those than there are refusals.
    import re

    refused = {
        op: why for op, why in record["by_table"]["union"]["verdicts"].items()
        if why.startswith("REFUSED")
    }
    buckets = collections.defaultdict(list)
    prims = set()
    for op, why in refused.items():
        hit = re.search(r"not implemented in torch\._C shim: (prims\.[\w.]+)", why)
        if hit:
            prims.add(hit.group(1))
            buckets["missing prims.* op"].append(op)
        elif "has no rule for it" in why:
            buckets["no rule in any upstream table"].append(op)
        elif "not implemented in torch._C shim" in why:
            buckets["other shim gap (argument or overload)"].append(op)
        else:
            buckets["other"].append(op)
    print(f"\n=== why the {len(refused)} refusals refuse ===")
    for label, ops in sorted(buckets.items()):
        print(f"  {label}: {len(ops)}")
        for op in sorted(ops):
            print(f"      {op}")
    print(f"\n  distinct missing prims ops ({len(prims)}): {', '.join(sorted(prims))}")
    record["refusal_buckets"] = {k: sorted(v) for k, v in buckets.items()}
    record["missing_prims"] = sorted(prims)

    print("\n  CoreML: NOT MEASURED -- no operator set is readable from this tree.")
    try:
        T.coreml_ops()
    except NotImplementedError as error:
        record["coreml"] = str(error)
        print(f"    {error}")

    if args.json:
        json.dump(record, sys.stdout, indent=1, default=str)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
