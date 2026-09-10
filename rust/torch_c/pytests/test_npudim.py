"""`MAX_DIM = 2**17`: what `docs/devices/NPUDIM.md` settled, and what it did not.

The question this file guards is not "is 131072 right?". It is **"does this
repository still say where 131072 came from?"** -- because the failure mode the
investigation was run against is a constant with no provenance, copied from an
archived library, that quietly leaves `lm_head` (151936 x 2560 on Qwen3-4B, the
single largest weight in the model) on the CPU.

`docs/devices/NPUDIM.md`'s verdict is **(c)**: `131072` / `2**17` occurs nowhere
in the OpenVINO NPU plugin source, the NPU compiler source, the Level Zero
graph-extension header, or the shipped NPU binaries of the 2025.4.1 and 2026.3.1
wheels. The only per-dimension limit the NPU compiler names is
`VPU_DIMENSION_LIMIT = 8192`, and the compiler *tiles* past it rather than
refusing.

The delicate part, and the reason three of these tests are about what the
repository must **not** say: showing a number is unsourced is not showing that a
larger one works. Nothing above 8192 has been compiled for `NPU` anywhere in
this project. So `MAX_DIM` is unchanged, and these tests fail if a later round
raises it without a Stage B result to point at.

What this file is evidence ABOUT: the constant, its provenance note, the
refusal message, and the shape of the experiment tool. It is evidence about
**nothing in silicon** -- this host is an arm64 Mac with no Intel NPU, no
OpenVINO for it, and `library_candidates` refuses on darwin by design. The one
thing here that runs real code is Stage A of the sweep tool, which is pure
Python and says so in its own output.
"""

import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
sys.path.insert(0, _VENDOR_DIR)

_DOC = os.path.join(_ROOT, "docs", "devices", "NPUDIM.md")
_TOOL = os.path.join(_ROOT, "tools", "devices", "intelnpu_dimsweep.py")
_INTELNPU_SRC = os.path.join(
    _VENDOR_DIR, "torchnative", "export", "intelnpu.py")


def _doc() -> str:
    with open(_DOC, encoding="utf-8") as fh:
        return fh.read()


def test_the_investigation_document_exists_and_carries_its_sources():
    """NPUDIM.md names the verdict, the evidence and each source file.

    The point of the round was to stop `2**17` being an unexplained number.
    A document that recorded the conclusion without the files it was read out
    of would leave the next reader exactly where this one started -- so the
    sources, not the verdict, are what is asserted here.
    """
    text = _doc()

    # Every source the verdict rests on, by the name a reader can go and open.
    for source in (
        "nce_invariant.hpp",                        # VPU_DIMENSION_LIMIT = 8192
        "ensure_nce_ops_size_requirements.cpp",     # tiling, not refusal
        "nce_cluster_task.cpp",                     # where the limit is enforced
        "ze_graph_ext.h",                           # uint32_t dims, no 17-bit field
        "metadata.hpp",                             # the 131072 false positive
        "nn/linear.py:66",                          # the line that was copied
        "openvinotoolkit/npu_compiler",
        "openvinotoolkit/openvino",
        "level-zero-npu-extensions",
    ):
        assert source in text, f"NPUDIM.md no longer cites {source}"

    # The two numbers that carry the argument, and the wheel versions examined.
    for fact in ("8192", "131072", "2025.4.1", "2026.3.1", "151936"):
        assert fact in text, f"NPUDIM.md no longer states {fact}"

    print("ok   npudim: NPUDIM.md states the verdict and cites all nine sources "
          "it rests on, plus both wheel versions examined")


def test_the_document_keeps_an_explicit_unverified_section():
    """The house rule from docs/devices/QNNOPS.md section 5, enforced.

    An investigation that reports only what it settled reads as if it settled
    everything. The one that matters most here is named explicitly: the real
    ceiling was NOT measured, because no dimension above 8192 was compiled for
    NPU on any machine in this round.
    """
    text = _doc()
    assert "## 4. UNVERIFIED" in text, "NPUDIM.md lost its UNVERIFIED section"

    unverified = text.split("## 4. UNVERIFIED", 1)[1].split("\n## ", 1)[0]
    # The single most important open item, and the closed-binary one that is
    # the only place a limit invisible to the document could still live.
    assert "real ceiling" in unverified.lower(), unverified[:400]
    assert "DRIVER" in unverified, unverified[:400]
    # Every item says what would settle it -- that is what makes it publishable
    # rather than a shrug.
    assert unverified.count("Settled by") >= 4, unverified

    print("ok   npudim: NPUDIM.md keeps an explicit UNVERIFIED section naming "
          f"the unmeasured real ceiling, the closed driver compiler, and "
          f"{unverified.count('Settled by')} things that would settle them")


def test_the_document_records_the_131072_false_positive():
    """The one piece of evidence that would have produced a confident wrong answer.

    `strings | grep 131072` on the shipped NPU plugin DOES return hits --
    `intel_npu::Metadata<131072u>` through `<131075u>`. They are blob metadata
    versions (`make_version(2, 0)` is `2 << 16` = 131072), not a dimension
    limit, and they are consecutive integers, which is the tell. If NPUDIM.md
    ever loses this paragraph, the next investigator repeats the trap.
    """
    text = _doc()
    assert "Metadata" in text and "METADATA_VERSION_2_0" in text, text[:200]
    assert "131073" in text, "the consecutive-versions tell is gone"
    assert "make_version" in text

    print("ok   npudim: NPUDIM.md records that Metadata<131072..131075> in the "
          "shipped plugin are metadata versions, not a dimension limit")


def test_max_dim_is_unchanged_and_says_it_is_unsourced():
    """The constant did NOT move, and the module says why not.

    Two ways this round could have gone wrong, and this test is against both:
    raising `MAX_DIM` because OpenVINO does not contain the number (inferring a
    permissive fact from the absence of a restrictive one), or lowering it to
    `VPU_DIMENSION_LIMIT`'s 8192, which the compiler tiles past anyway.
    """
    from torchnative.export import intelnpu

    assert intelnpu.MAX_DIM == 2 ** 17, (
        f"MAX_DIM moved to {intelnpu.MAX_DIM}. NPUDIM.md section 0 says this "
        f"needs a Stage B result from tools/devices/intelnpu_dimsweep.py, not "
        f"an argument from the absence of the number in OpenVINO.")

    with open(_INTELNPU_SRC, encoding="utf-8") as fh:
        src = fh.read()
    head, _, _ = src.partition("MAX_DIM = 2 ** 17")
    note = head[-2400:]
    assert "docs/devices/NPUDIM.md" in note, "the provenance note is gone"
    assert "not sourced to OpenVINO" in note, note[-600:]
    assert "VPU_DIMENSION_LIMIT" in note, note[-600:]
    assert "intelnpu_dimsweep.py" in note, note[-600:]

    print("ok   npudim: MAX_DIM is still 2**17 and its comment names it "
          "unsourced, cites NPUDIM.md and VPU_DIMENSION_LIMIT=8192, and points "
          "at the sweep that would settle it")


def test_an_oversized_leaf_is_still_refused_by_name_with_shape_limit_and_provenance():
    """The one behaviour this round was not allowed to put at risk.

    `docs/graph/NPU2.md` exists because a layer that quietly did not move is
    indistinguishable from one that did. The refusal must name the dimension,
    its value, and the limit -- and now also where the limit came from, so a
    user who hits it can read the investigation instead of guessing like the
    archived library's users had to.
    """
    from torchnative.export.intelnpu import MAX_DIM, IntelNPUUnsupported, linear_ir

    try:
        linear_ir(64, MAX_DIM + 1)
    except IntelNPUUnsupported as exc:
        reason = str(exc)
    else:
        raise AssertionError("an oversized Linear was emitted instead of refused")

    assert "out_features" in reason, reason           # which dimension
    assert str(MAX_DIM + 1) in reason, reason         # its shape
    assert f"MAX_DIM={MAX_DIM}" in reason, reason     # the limit it exceeded
    assert "NPUDIM.md" in reason, reason              # where the limit came from
    assert "intelnpu_dimsweep.py" in reason, reason   # what would settle it

    print(f"ok   npudim: an out_features={MAX_DIM + 1} leaf is refused by name "
          f"with its shape, MAX_DIM, and now the provenance of MAX_DIM")


def test_the_sweep_tool_brackets_every_candidate_boundary():
    """The sweep must contain the values that distinguish the three outcomes.

    A sweep missing 8193 cannot tell "the compiler tiles" from "the compiler
    refuses". One missing 131073 cannot tell "stops at MAX_DIM" from "stops
    somewhere else" -- and those are two different verdicts, (b) and (c).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_dimsweep", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for boundary in (8192, 8193, 32768, 65536, 131072, 131073, 151936):
        assert boundary in mod.SWEEP, f"{boundary} left the sweep"
    assert list(mod.SWEEP) == sorted(mod.SWEEP), mod.SWEEP

    print(f"ok   npudim: the sweep brackets all seven candidate boundaries "
          f"({len(mod.SWEEP)} dimensions, 4096..151936) in ascending order")


def test_the_sweep_tool_keeps_selection_apart_from_execution():
    """Stage A runs here, on a machine with no NPU, and says it proves nothing.

    This is a real run of the real tool, not an inspection of its source: the
    subprocess reaches `linear_ir` and the real `MAX_DIM`. What it must not do
    is let a green selection result read as a hardware claim -- so a skipped
    Stage B reports SKIPPED, never 0.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, _TOOL, "--skip-b"],
        capture_output=True, text=True, env=env, cwd=_ROOT)
    out = proc.stdout
    assert proc.returncode == 0, proc.stderr[-2000:]

    assert "STAGE_B_EXIT=SKIPPED" in out, out[-800:]
    assert "STAGE_B_EXIT=0" not in out, "a stage that never ran reported 0"
    assert "only stage that is evidence about hardware" in out, out[-800:]
    # Selection, on this host, is exactly MAX_DIM's line and nothing else.
    assert "131072  emitted" in out, out
    assert "131073  REFUSED BY NAME" in out, out
    assert "151936  REFUSED BY NAME" in out, out

    print("ok   npudim: the sweep's Stage A runs with no OpenVINO, refuses "
          "131073 and 151936 by name, and reports Stage B as SKIPPED rather "
          "than 0")


def test_the_compiler_switch_refuses_instead_of_setting_an_env_var():
    """`--compiler DRIVER` must not pretend.

    `NPU_COMPILER_TYPE` is an `ov::Property` handed to `compile_model`
    (`openvino/runtime/intel_npu/properties.hpp`); the NPU plugin source has no
    `getenv` for it, and `intelnpu.OpenVINO.compile_ir` accepts no property map.
    A tool that set an environment variable and labelled the run DRIVER would
    manufacture the finding that settles NPUDIM.md section 4 item 2 -- the
    worst available outcome, since that item is the one place a limit invisible
    to the document could still live.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, _TOOL, "--compiler", "DRIVER", "--skip-b"],
        capture_output=True, text=True, env=env, cwd=_ROOT)
    assert proc.returncode == 2, (proc.returncode, proc.stdout[-800:])
    assert "REFUSED" in proc.stdout, proc.stdout[-800:]
    assert "not an env var" in proc.stdout, proc.stdout[-800:]

    with open(_TOOL, encoding="utf-8") as fh:
        tool_src = fh.read()
    assert "OV_NPU_COMPILER_TYPE" not in tool_src, (
        "the tool sets an env var for NPU_COMPILER_TYPE again; the plugin does "
        "not read one, so the run would be labelled with a compiler it did not "
        "use")

    print("ok   npudim: --compiler DRIVER refuses by name (exit 2) and the tool "
          "sets no NPU_COMPILER_TYPE environment variable")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
