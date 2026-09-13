"""Find the real per-dimension ceiling for a `Linear` on an Intel NPU.

    python tools/devices/intelnpu_dimsweep.py                 # both stages
    python tools/devices/intelnpu_dimsweep.py --skip-b        # selection only
    python tools/devices/intelnpu_dimsweep.py --dims 8192,8193,151936

`docs/devices/NPUDIM.md` is the investigation this tool exists to close. Its
result: `torchnative.export.intelnpu.MAX_DIM = 2**17` is **unsourced** --
`131072` appears nowhere in the OpenVINO NPU plugin source, the NPU compiler
source, the Level Zero graph-extension header, or the shipped NPU binaries of
the 2025.4.1 and 2026.3.1 wheels. What that does NOT establish is the real
ceiling, and no property reports one (NPUDIM.md section 2.4), so the only
discovery available is trial compilation. That is this file.

Two SEPARATE stages, the split `tools/devices/intelnpu_verify.py` established
and for the same reason.

Stage A -- SELECTION. Which dimensions `linear_ir` will emit today, i.e. where
`MAX_DIM` currently draws the line. Pure Python: no OpenVINO, no NPU, no
network. A green Stage A is a statement about this repository's constant and
about **nothing in silicon**. Running it on a Mac is meaningful; reading it as
a hardware result is the exact confusion the two-stage split prevents.

Stage B -- EXECUTION. Compiles a real MatMul at each swept dimension for device
`NPU` and reads `EXECUTION_DEVICES` back off the compiled model. This needs an
OpenVINO runtime and a real Intel NPU. **It is the only stage that is evidence
about hardware.**

Stage B deliberately bypasses `MAX_DIM`. It sets `intelnpu.MAX_DIM` to
`sys.maxsize` for the duration and says so on stdout, because a sweep that
stopped at the constant under investigation would only measure the constant.
Nothing is written back; the library's own refusal is untouched.

What the sweep brackets, and why each value is in it (NPUDIM.md section 5):

    4096     a dimension every LLM already has; the control
    8192     VPU_DIMENSION_LIMIT, the one per-dimension limit the NPU compiler
             names (npu_compiler nce_invariant.hpp:40)
    8193     the first value that requires the compiler's tiling pass
             (ensure_nce_ops_size_requirements.cpp:255-267)
    16384    one tiling step beyond
    32768    MAX_SAFE_GATHER_DMA_INDICES, the DMA-family limit
    65536    the 65535 hardware single-task limit that one is derived from
    131072   MAX_DIM itself
    131073   the first value MAX_DIM refuses
    151936   Qwen3-4B's lm_head, the shape that motivated the whole question

`in_features` is held at 64 throughout, on purpose: the question is about a
single dimension's extent, and a 151936 x 2560 f16 weight is 778 MB of blob to
push through the driver per compile. 151936 x 64 is 19 MB. If the sweep's
answer turns out to depend on the *product* rather than a dimension, that is
itself a finding, and `--in-features` is there to test it.

How to read the three outcomes (NPUDIM.md section 5):

  * every dimension compiles and executes on NPU -> `MAX_DIM` is a pure safety
    margin and can be removed;
  * the sweep stops at exactly 131072 -> the number is real after all, sourced
    to the driver-resident compiler (NPUDIM.md section 4), and the answer is
    (b) driver/compiler;
  * the sweep stops somewhere else -> that value, not 2**17, is the constant,
    and the printed error names which layer imposed it.

Running it under both compilers -- `NPU_COMPILER_TYPE=PLUGIN` and `DRIVER` --
would settle NPUDIM.md section 4 item 2, since a disagreement would put the
limit in the driver-resident compiler. `--compiler` is present for that and
**refuses by name**: NPU_COMPILER_TYPE is an `ov::Property` handed to
`compile_model`, not an environment variable, and `OpenVINO.compile_ir` takes
no property map yet. A tool that set an env var and reported the result as
DRIVER would be inventing the finding.
"""

import argparse
import sys

#: Every candidate boundary NPUDIM.md found or ruled out, in order.
SWEEP = (4096, 8192, 8193, 16384, 32768, 65536, 131072, 131073, 151936)


def stage_a(sweep, in_features: int) -> int:
    from torchnative.export import intelnpu

    print("\n=== Stage A: selection (no OpenVINO, no NPU involved) ===")
    print(f"MAX_DIM = {intelnpu.MAX_DIM}   (docs/devices/NPUDIM.md: unsourced)")
    print(f"in_features held at {in_features}")
    for out in sweep:
        try:
            intelnpu.linear_ir(in_features, out)
        except intelnpu.IntelNPUUnsupported as exc:
            print(f"  {out:>8}  REFUSED BY NAME  {exc}")
        else:
            print(f"  {out:>8}  emitted")
    print(
        "\nThis is a statement about SELECTION -- about this repository's "
        "constant. Nothing has compiled and nothing has executed. Stage B is "
        "the only stage that is evidence about hardware."
    )
    return 0


def stage_b(sweep, in_features: int, device: str, library) -> int:
    from torchnative.export import intelnpu

    print("\n=== Stage B: execution (needs an OpenVINO runtime and real hardware) ===")
    try:
        ov = intelnpu.OpenVINO(library)
    except intelnpu.IntelNPUUnavailable as exc:
        print(f"UNAVAILABLE: {exc}")
        print(
            "\nThis is Stage B failing to start, not a wrong answer. Stage A "
            "above is unaffected -- it never touches OpenVINO."
        )
        return 2

    devices = ov.devices()
    print(f"available devices: {devices}")
    if device not in devices:
        print(
            f"UNAVAILABLE: OpenVINO does not list a {device!r} device, so nothing "
            f"below would be a hardware result. Refusing to compile for a device "
            f"that is not there rather than reporting a CPU number as an NPU one."
        )
        return 2

    # Bypass the constant under investigation, loudly. A sweep that stopped at
    # MAX_DIM would measure MAX_DIM.
    original = intelnpu.MAX_DIM
    intelnpu.MAX_DIM = sys.maxsize
    print(f"MAX_DIM temporarily raised {original} -> sys.maxsize for this stage "
          f"only; the library's refusal is not modified.")
    results = []
    try:
        for out in sweep:
            xml = intelnpu.linear_ir(in_features, out)
            weights = bytes(out * in_features * 2 + out * 2)
            try:
                compiled = ov.compile_ir(xml, device, weights)
            except Exception as exc:  # noqa: BLE001
                outcome = f"COMPILE FAILED  {type(exc).__name__}: {exc}"
            else:
                where = ov.execution_devices(compiled)
                ran_on_npu = any(str(d).upper().startswith("NPU") for d in where)
                outcome = ("compiled+NPU" if ran_on_npu
                           else f"compiled+OTHER-DEVICE {where}")
            results.append((out, outcome))
            print(f"  {out:>8}  {outcome}")
    finally:
        intelnpu.MAX_DIM = original

    ok = [d for d, o in results if o == "compiled+NPU"]
    bad = [(d, o) for d, o in results if o != "compiled+NPU"]
    print(f"\nlargest out_features on NPU : {max(ok) if ok else 'none'}")
    print(f"first failure               : {bad[0][0] if bad else 'none'}")
    print(f"MAX_DIM                     : {original}")
    if not bad:
        print("VERDICT: every swept dimension ran on the NPU. MAX_DIM is a pure "
              "safety margin (NPUDIM.md section 5, outcome 1).")
    elif bad[0][0] == original + 1:
        print(f"VERDICT: the sweep stops at {original}, exactly MAX_DIM. The "
              f"number is real; find which layer imposed it in the message above "
              f"(NPUDIM.md section 5, outcome 2).")
    else:
        print(f"VERDICT: the sweep stops at {bad[0][0]}, which is NOT MAX_DIM "
              f"({original}). That value is the constant, and the message above "
              f"names the layer (NPUDIM.md section 5, outcome 3).")
    return 0 if ok else 3


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-features", type=int, default=64)
    p.add_argument("--device", default="NPU")
    p.add_argument("--library", default=None,
                   help="path to the OpenVINO runtime; probed if omitted")
    p.add_argument("--compiler", default=None, choices=("PLUGIN", "DRIVER"),
                   help="NOT YET WIRED -- refuses by name. See the note in main().")
    p.add_argument("--dims", default=None,
                   help="comma-separated out_features to sweep instead of the default")
    p.add_argument("--skip-a", action="store_true")
    p.add_argument("--skip-b", action="store_true")
    args = p.parse_args()

    sweep = SWEEP if args.dims is None else tuple(
        int(x) for x in args.dims.split(",") if x.strip())

    if args.compiler:
        # Named and refused rather than silently ignored. `NPU_COMPILER_TYPE` is
        # a *compile-time property* (openvino/runtime/intel_npu/properties.hpp),
        # not an environment variable -- the NPU plugin source has no getenv for
        # it -- and `intelnpu.OpenVINO.compile_ir` takes no property map today.
        # Setting an env var here would report a PLUGIN run as a DRIVER one,
        # which is the whole failure mode this project keeps finding.
        print(f"REFUSED: --compiler {args.compiler} cannot be honoured. "
              f"NPU_COMPILER_TYPE is an ov::Property passed to compile_model, "
              f"not an env var, and intelnpu.OpenVINO.compile_ir accepts no "
              f"properties yet. Settling NPUDIM.md section 4 item 2 needs that "
              f"argument added first; running the sweep now would measure "
              f"whichever compiler is the default and label it {args.compiler}.")
        return 2

    # A skipped stage reports SKIPPED, never 0. Printing 0 for a stage that
    # never ran is green that was never earned.
    rc_a = stage_a(sweep, args.in_features) if not args.skip_a else None
    rc_b = (stage_b(sweep, args.in_features, args.device, args.library)
            if not args.skip_b else None)
    print(f"\nSTAGE_A_EXIT={'SKIPPED' if rc_a is None else rc_a}")
    print(f"STAGE_B_EXIT={'SKIPPED' if rc_b is None else rc_b}")
    return (rc_a or 0) or (rc_b or 0)


if __name__ == "__main__":
    raise SystemExit(main())
