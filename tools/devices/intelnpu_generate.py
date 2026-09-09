"""Status, then one forward, then generate -- on a model already on the NPU.

    python tools/devices/intelnpu_generate.py --model Qwen/Qwen3-4B

Run AFTER `model.to(device.npu)` has succeeded. Each stage is separate and
reports its own time, because the interesting failures here are not "wrong
answer" but "took so long it looked hung", and a single timing for the whole
thing cannot tell those apart.

WHY THIS IS STAGED, AND WHY IT MAY BE SLOW
------------------------------------------
`_NPULinear` compiles a static-shape IR per BATCH, where batch is the product
of every input dimension except the last (`intelnpu.py`, `_NPULinear.forward`),
and caches it in `self._compiled[batch]`.

`generate()` uses two shapes: the prompt (batch = prompt length) and then, with
a KV cache, one token at a time (batch = 1). So every lowered Linear compiles
**twice**, through the driver-resident compiler. On a 36-layer Qwen3 that is
252 leaves x 2 = 504 compiles before the second token appears.

Everything after that is per-token inference, and that path currently marshals
each activation through a Python list (`.tolist()` in, `unpack_f16` out). It is
correct and it is not fast; measuring it is the point of stage 3.
"""

import argparse
import time


def _fmt(seconds: float) -> str:
    return f"{seconds:7.2f}s"


def stage_0_status(model) -> int:
    print("\n=== Stage 0: what actually lowered ===")
    report = getattr(model, "torchnative_offload", None)
    if report is None:
        print("no .torchnative_offload on this model -- was to(device.npu) called?")
        return 1
    print(f"fully_offloaded : {report['fully_offloaded']}")
    print(f"fraction_moved  : {report['fraction_moved']:.4f} "
          f"({report['parameters_moved']} / {report['parameters_total']})")
    print(f"eligible leaves : {len(report['eligible'])}")
    print(f"left on cpu     : {report['left_on_cpu']}")
    for name, reason in report["skipped"]:
        print(f"  SKIPPED {name}: {reason.splitlines()[0]}")
    print(f"execution_devices (the eager first compile): "
          f"{report.get('execution_devices')}")

    # How many leaves have actually been compiled so far, and for which shapes.
    from torchnative.export import intelnpu

    lowered = [m for m in model.modules()
               if type(m).__name__ == "_NPULinear"]
    compiled = [m for m in lowered if getattr(m, "_compiled", None)]
    shapes = sorted({b for m in lowered for b in getattr(m, "_compiled", {})})
    print(f"lowered leaves  : {len(lowered)}")
    print(f"  compiled so far: {len(compiled)}  for batch shapes {shapes}")
    print(f"  (the rest compile on their first forward -- that is stage 1)")
    return 0


def stage_1_forward(model, tokenizer, prompt: str) -> int:
    import torch

    print("\n=== Stage 1: one forward (compiles every leaf for this shape) ===")
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"prompt tokens   : {tuple(ids.shape)}")
    print("this is where the driver compiler runs once per lowered leaf; on a "
          "36-layer model expect minutes, not seconds")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(ids)
    dt = time.perf_counter() - t0
    logits = out.logits if hasattr(out, "logits") else out
    print(f"forward         : {_fmt(dt)}  logits {tuple(logits.shape)}")

    lowered = [m for m in model.modules() if type(m).__name__ == "_NPULinear"]
    compiled = [m for m in lowered if getattr(m, "_compiled", None)]
    on_npu = [m for m in lowered if m.execution_devices]
    bad = [m for m in on_npu if list(m.execution_devices) != ["NPU"]]
    print(f"compiled leaves : {len(compiled)} / {len(lowered)}")
    print(f"reporting NPU   : {len(on_npu)}")
    if bad:
        print(f"  NOT on the NPU: {len(bad)} -- {[m.execution_devices for m in bad][:3]}")
        return 2

    # A second forward of the SAME shape must reuse the cache, so it isolates
    # inference cost from compile cost. If it is not much faster, the cache is
    # not being hit and that is the finding.
    t0 = time.perf_counter()
    with torch.no_grad():
        model(ids)
    dt2 = time.perf_counter() - t0
    print(f"forward again   : {_fmt(dt2)}  (cached compile; this is inference cost)")
    if dt2 > dt * 0.9:
        print("  NOTE: the second forward was not meaningfully faster, so the "
              "per-batch compile cache is not doing what it claims.")
    return 0


def stage_2_generate(model, tokenizer, prompt: str, new_tokens: int) -> int:
    import torch

    print(f"\n=== Stage 2: generate({new_tokens} tokens) ===")
    print("the first token recompiles every leaf for batch=1; the rest reuse it")
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=new_tokens, do_sample=False)
    dt = time.perf_counter() - t0
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    made = int(out.shape[-1] - ids.shape[-1])
    print(f"generate        : {_fmt(dt)} for {made} tokens "
          f"({dt / made:.2f}s/token including the batch=1 compiles)")
    print(f"\n--- output ---\n{text}\n--------------")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--new-tokens", type=int, default=8)
    p.add_argument("--skip-generate", action="store_true")
    args = p.parse_args()

    from torchnative.transformers import AutoModelForCausalLM
    from torchnative import device
    from transformers import AutoTokenizer

    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", disable_mmap=True)
    model.eval()

    t0 = time.perf_counter()
    model.to(device.npu)
    print(f"to(device.npu)  : {_fmt(time.perf_counter() - t0)}")

    rc = stage_0_status(model)
    if rc:
        return rc
    rc = stage_1_forward(model, tok, args.prompt)
    if rc:
        return rc
    if args.skip_generate:
        print("\nSTAGE_2=SKIPPED")
        return 0
    return stage_2_generate(model, tok, args.prompt, args.new_tokens)


if __name__ == "__main__":
    raise SystemExit(main())
