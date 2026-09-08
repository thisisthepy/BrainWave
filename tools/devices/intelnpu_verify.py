"""Run this on the Intel NPU laptop. It answers two SEPARATE questions.

    python tools/devices/intelnpu_verify.py
    python tools/devices/intelnpu_verify.py --model Qwen/Qwen3-4B

Stage A -- SELECTION. Which leaves of a real model would lower, which would
not, and why. Pure Python: no OpenVINO, no NPU, no network beyond the
checkpoint config. This is what the `lm_head` fix changed, and it is
reproducible on any machine, which is the point.

Stage B -- EXECUTION. Whether an operation actually ran on the NPU. This needs
an OpenVINO runtime and real hardware. It is the ONLY stage that is evidence
about hardware; Stage A is not, and says so.

Keeping them apart is deliberate. A green Stage A on a machine with no NPU is
still a true result about selection -- reporting the two together is how a
selection result gets read as an execution claim.
"""

import argparse
import json
import sys



def stage_a(model_id: str) -> int:
    from torchnative.export import intelnpu

    print(f"\n=== Stage A: selection (no NPU involved) -- {model_id} ===")
    print(f"MAX_DIM = {intelnpu.MAX_DIM}")

    from transformers import AutoConfig, AutoModelForCausalLM
    import torch

    cfg = AutoConfig.from_pretrained(model_id)
    # Build on meta so a 4B checkpoint costs no download and no RAM: selection
    # reads shapes and parameter counts, never values.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)

    plan = intelnpu.plan_lowering(model)

    print(f"eligible leaves   : {len(plan['eligible'])}")
    print(f"skipped leaves    : {len(plan['skipped'])}")
    for name, reason in plan["skipped"]:
        print(f"  REFUSED {name}\n          {reason}")
    print(f"left on cpu       : {plan['left_on_cpu']}")
    print(f"fully_offloaded   : {plan['fully_offloaded']}")
    print(f"fraction_moved    : {plan['fraction_moved']:.4f}"
          f"  ({plan['parameters_moved']} / {plan['parameters_total']})")

    # Two different reasons a model is not fully offloaded, and conflating them
    # reports a refusal that did not happen. `skipped` is a Linear DECLINED on
    # its shape; `left_on_cpu` is a leaf that was never a lowering candidate
    # (norms, embeddings, activations). Only the first is the lm_head story.
    if plan["skipped"]:
        print(f"\nNote: {len(plan['skipped'])} Linear(s) were DECLINED on their "
              "shape. The refusal above names each one, its shape and the limit "
              "it exceeded. Before the per-leaf fix, one such leaf refused the "
              "whole model.")
    elif plan["left_on_cpu"]:
        print("\nNote: no Linear was declined. fully_offloaded is False only "
              "because non-Linear leaves (norms, embeddings, activations) were "
              "never lowering candidates.")
    else:
        print("\nNote: every leaf is eligible.")
    print("Either way this is a statement about SELECTION. Nothing has executed; "
          "Stage B is the only stage that is evidence about hardware.")
    return 0


def stage_b(device: str, library: str | None) -> int:
    from torchnative.export import intelnpu

    print(f"\n=== Stage B: execution (needs an OpenVINO runtime and real hardware) ===")
    try:
        report = intelnpu.probe(library, device)
    except intelnpu.IntelNPUUnavailable as exc:
        print(f"UNAVAILABLE: {exc}")
        print("\nThis is Stage B failing to start, not a wrong answer. Stage A above "
              "is unaffected -- it never touches OpenVINO.")
        return 2
    except intelnpu.IntelNPUExecutionError as exc:
        print(f"EXECUTION ERROR: {exc}")
        return 3

    print(json.dumps(report, indent=2, default=str))
    verdict = report.get("verdict")
    print(f"\nverdict: {verdict}")
    print("Read this against docs/devices/INTELNPU.md section 4, field by field.")
    return 0 if verdict not in ("no-npu", "no-device") else 4


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="checkpoint whose SHAPES are read (weights are never downloaded)")
    p.add_argument("--device", default="NPU")
    p.add_argument("--library", default=None,
                   help="path to the OpenVINO runtime; probed if omitted")
    p.add_argument("--skip-a", action="store_true")
    p.add_argument("--skip-b", action="store_true")
    args = p.parse_args()

    # A skipped stage reports SKIPPED, never 0. Printing 0 for a stage that
    # never ran is the same lie the rest of this project keeps finding: green
    # that was never earned.
    rc_a = stage_a(args.model) if not args.skip_a else None
    rc_b = stage_b(args.device, args.library) if not args.skip_b else None
    print(f"\nSTAGE_A_EXIT={'SKIPPED' if rc_a is None else rc_a}")
    print(f"STAGE_B_EXIT={'SKIPPED' if rc_b is None else rc_b}")
    return (rc_a or 0) or (rc_b or 0)


if __name__ == "__main__":
    raise SystemExit(main())
