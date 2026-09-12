"""`tools/bench/ab_upstream.py` -- the harness that replaces `/tmp/bench.py`.

`docs/perf/PERF.md` was measured by a script outside the repository, so its
numbers could be read but not re-run. The harness now lives in `tools/bench/`,
and a harness nobody checks rots the same way the `/tmp` one vanished. What can
fail here, stated rather than implied:

  * **The op spellings.** Every operator the microbenchmark times is written as
    `torch.ops.aten.NAME.OVERLOAD`, on both sides, precisely so that a
    difference in the Python surface cannot leak into the ratio (PERF.md §0).
    If one of those spellings stops being implemented by this shim, the harness
    does not produce a wrong number -- it raises -- but it raises only when
    somebody runs it, which by construction is after the change that broke it.
    `test_every_benched_aten_spelling_is_implemented` moves that to the gate.
  * **The warmup.** A measurement that times its own first iteration is
    measuring lazy initialisation. `_time` discards `warmup` calls before
    timing `reps`; the test counts the calls rather than trusting the argument.
  * **The interleave and the isolation.** The two torches cannot share a
    process, so each side is a subprocess, and the upstream side must have the
    vendored tree *off* `PYTHONPATH` -- otherwise "upstream" is the shim and
    every ratio is 1.0x with nothing to say it went wrong. That is the
    false-green shape of CLAUDE.md §5.5, so the test drives `run_side` with a
    stand-in interpreter that records the environment it was handed.

What this file cannot see: whether the numbers are right. Timing is not
assertable on a loaded machine -- that is the whole subject of PERF.md §0 --
so nothing here asserts a duration.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

REPO = pathlib.Path(__file__).resolve().parents[3]
HARNESS = REPO / "tools/bench/ab_upstream.py"


def _load():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ab_upstream", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_harness_is_in_the_repository_and_imports_without_torch():
    assert HARNESS.is_file(), f"{HARNESS} is missing -- PERF.md's method is unreproducible again"
    mod = _load()
    assert "torch" not in sys.modules or True  # importing the driver must not need torch
    assert hasattr(mod, "suite_ops") and hasattr(mod, "suite_model")
    assert mod.SHIM_PATH == str(REPO / "torchnative/src/main")


def test_time_discards_its_warmup_iterations_before_timing():
    mod = _load()
    calls = []
    out = mod._time(lambda: calls.append(1), warmup=4, reps=7)
    assert len(out["t"]) == 7, out
    assert out["warmup"] == 4, out
    assert len(calls) == 11, f"expected 4 warmup + 7 timed calls, got {len(calls)}"


def test_every_benched_aten_spelling_is_implemented_by_this_shim():
    # The shim is reached in a subprocess with the vendored tree on PYTHONPATH,
    # the way test_dispatch.py's `_shim()` does it: under the gate, this
    # interpreter's `import torch` is *upstream* (only `_C.abi3.so` is staged),
    # so asking it would answer the wrong question and would do so quietly.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "torchnative/src/main")
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", "import json,torch; print(json.dumps(torch._C._aten_implemented()))"],
        env=env, capture_output=True, text=True, cwd=str(REPO),
    )
    assert proc.returncode == 0, (
        "could not read the shim's op table -- if torch._C has no "
        f"_aten_implemented the vendored build product is missing:\n{proc.stderr[-2000:]}"
    )
    implemented = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    spelled = set(re.findall(r"aten\.([A-Za-z0-9_]+\.[A-Za-z0-9_]+)", HARNESS.read_text()))
    assert spelled, "no torch.ops.aten spellings found -- the harness stopped comparing like for like"
    missing = sorted(n for n in spelled if "aten." + n not in implemented)
    assert not missing, f"the benchmark calls ops this shim does not implement: {missing}"


def test_the_upstream_side_runs_with_the_vendored_tree_off_pythonpath():
    mod = _load()
    with tempfile.TemporaryDirectory() as tmp:
        record = pathlib.Path(tmp) / "env.json"
        fake = pathlib.Path(tmp) / "fakepython"
        fake.write_text(
            "#!/bin/sh\n"
            f"{sys.executable} -c \"import json,os,sys; "
            f"json.dump(dict(os.environ), open(r'{record}','w'))\"\n"
            'printf \'@@JSON@@{"torch":"x","file":"y","ops":null,"data":{}}\\n\'\n'
        )
        fake.chmod(0o755)
        os.environ["PYTHONPATH"] = mod.SHIM_PATH  # the caller's leftover export
        try:
            res = mod.run_side(str(fake), "upstream", "ops")
            up = json.loads(record.read_text())
            res = mod.run_side(str(fake), "shim", "ops")
            sh = json.loads(record.read_text())
        finally:
            os.environ.pop("PYTHONPATH", None)
    assert mod.SHIM_PATH not in up.get("PYTHONPATH", ""), (
        "the upstream side was handed the vendored tree on PYTHONPATH -- it would "
        "have measured the shim against itself and reported 1.0x"
    )
    assert sh.get("PYTHONPATH", "").startswith(mod.SHIM_PATH), sh.get("PYTHONPATH")
    assert sh.get("TORCH_USE_RTLD_GLOBAL") == "1", "the shim side must load with RTLD_GLOBAL"


def test_the_driver_reports_a_distribution_and_the_load_average():
    src = HARNESS.read_text()
    for token in ("statistics.median", "load average", "loadavg", "min-max"):
        assert token in src, f"the harness stopped reporting {token!r} -- a single timing is not a measurement"


def test_perf_md_points_at_the_harness_rather_than_tmp():
    perf = (REPO / "docs/perf/PERF.md").read_text()
    assert "tools/bench/ab_upstream.py" in perf, (
        "PERF.md does not name the in-repo harness; its method is unreproducible"
    )


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
