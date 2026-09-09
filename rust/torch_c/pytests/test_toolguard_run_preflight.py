"""`run.sh` must refuse before building anything when its interpreter is wrong.

Two tooling traps share one shape: a script silently used the wrong
interpreter and the result *looked like* a verdict.

  * `run.sh` launches every suite with `${PYTHON:-python3}`. On this machine
    the bare `python3` on PATH has neither `numpy` nor `typing_extensions` nor
    `safetensors` nor `transformers`, so a run without `PYTHON` set produced
    hundreds of FAILs that were about the interpreter, not about anything
    this repository built.
  * The identical trap in the other direction is silent: a partial
    environment fails only *some* suites, and those read as real
    regressions.

`run.sh` now runs a preflight before `cargo build` (and therefore before any
suite): it tries to import the third-party packages the suites need under
`${PYTHON:-python3}`, and refuses with a named list of what is missing if any
import fails.

`torch` is deliberately NOT in that list, and its absence here is itself the
fix for a second, sharper trap the first version of this preflight fell into:
`torch` is not a third-party package this repository *consumes* -- through
`torchnative/src/main/torch` it is a build *product*, laid down by
`vendor/vendor_torch.sh` + `vendor/install_shim.sh`, gitignored, and absent by
default in a fresh worktree. Requiring `import torch` to succeed before the
gate runs makes the preflight refuse the one interpreter the gate exists to
use, in exactly the situation ("nothing has been vendored yet") the gate is
supposed to run in. The line this file pins is: packages the environment must
already have (checked) vs. packages this repository's own build produces
(never checked, no matter how tempting the `import torch` at the top of half
the suite files looks).

This file does not run `run.sh` end to end -- that needs a real `cargo build`
and the full gate, which belongs to the gate itself, not to a unit test. It
proves the *ordering* claim cheaply instead: a fake `cargo` on PATH records
whether it was ever invoked, and the two cases below assert that a bad
interpreter never reaches it while a good one does. A third case pins that
the verdict does not depend on the caller's own environment leaking in.
"""

import os
import stat
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_SH = HERE / "run.sh"

GOOD_PYTHON = "/Volumes/macMini/caches/spike-venv/bin/python"


def _fake_cargo_dir(tmp: Path, sentinel: Path) -> Path:
    """A directory holding only a `cargo` that touches `sentinel` and exits 1.

    Exiting 1 rather than 0 matters: if the preflight ever let a run reach
    this stub, `run.sh`'s own `set -eu` would otherwise make the whole test
    depend on getting a full fake build environment right just to observe
    "cargo was called". Exiting 1 stops `run.sh` immediately after recording
    that the call happened, which is the only thing being asserted here.
    """
    bindir = tmp / "fakebin"
    bindir.mkdir()
    cargo = bindir / "cargo"
    cargo.write_text(
        "#!/bin/sh\n"
        f"touch {sentinel}\n"
        "exit 1\n"
    )
    cargo.chmod(cargo.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bindir


def _run(python_value: str | None) -> tuple[int, str, bool]:
    """Run `run.sh` with a fake `cargo` on PATH and PYTHON set to `python_value`.

    `python_value is None` means PYTHON is genuinely unset for the child (not
    empty-string), matching the actual trap: nobody sets it at all.

    Returns `(returncode, combined output, cargo_was_called)`. The sentinel is
    resolved to a bool *inside* the temp-directory's lifetime -- the directory
    (and the sentinel file in it) is deleted on the way out of the `with`
    block, so a caller checking `sentinel.exists()` afterward would always see
    `False` regardless of what happened.
    """
    with tempfile.TemporaryDirectory(prefix="toolguard-preflight-") as tmpdir:
        tmp = Path(tmpdir)
        sentinel = tmp / "cargo-was-called"
        fakebin = _fake_cargo_dir(tmp, sentinel)
        env = {k: v for k, v in os.environ.items() if k != "PYTHON"}
        if python_value is not None:
            env["PYTHON"] = python_value
        env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"
        # An isolated stage dir so this never touches (or is confused by) a
        # real worktree's already-staged artefact.
        env["TORCH_C_STAGE"] = str(tmp / "stage")
        proc = subprocess.run(
            ["sh", str(RUN_SH)],
            cwd=str(HERE),
            env=env,
            capture_output=True,
            text=True,
        )
        cargo_was_called = sentinel.exists()
        return proc.returncode, proc.stdout + proc.stderr, cargo_was_called


def test_a_python3_missing_the_third_party_packages_is_refused_before_cargo_runs():
    """The exact trap: `PYTHON` unset, bare `python3` on PATH.

    On this machine that interpreter has none of numpy/typing_extensions/
    safetensors/transformers, which is precisely the condition that produced
    the FAIL pile this preflight exists to turn into one loud, named refusal
    instead. `torch` is deliberately absent from both the checked list and
    this assertion -- see the module docstring; checking it here would pin
    the very trap this file's third case exists to catch.
    """
    rc, out, cargo_was_called = _run(None)
    assert rc != 0, "run.sh must exit non-zero when the interpreter is wrong"
    assert "refusing to start" in out, out
    assert "numpy" in out, "the refusal must name what could not be imported"
    assert not cargo_was_called, (
        "cargo must never run once the preflight has refused -- the whole "
        "point is refusing BEFORE building anything, not after"
    )


def test_the_known_good_interpreter_is_not_rejected():
    """Critical recursion-trap check: the guard must not reject the very
    interpreter it tells callers to use.

    `spike-venv/bin/python` has numpy/safetensors/transformers/
    typing_extensions, so the preflight must pass and execution must reach
    (the fake) `cargo` -- regardless of whether its own, separately
    pip-installed `torch` happens to import cleanly, because the preflight no
    longer asks.
    """
    if not Path(GOOD_PYTHON).exists():
        raise AssertionError(f"known-good interpreter missing: {GOOD_PYTHON}")
    rc, out, cargo_was_called = _run(GOOD_PYTHON)
    assert "refusing to start" not in out, out
    assert cargo_was_called, (
        "the known-good interpreter must pass the preflight and reach cargo "
        f"-- got:\n{out}"
    )


def test_the_verdict_does_not_depend_on_the_callers_pythonpath():
    """A guard whose answer changes with who calls it, and how, is not a
    guard -- it is a coin flip that happens to land the same way most of the
    time. `run.sh`'s preflight now runs its own import check under
    `env -u PYTHONPATH`, so a caller's PYTHONPATH (a leftover export, a test
    harness isolating itself, a wrapper script) cannot shadow a package the
    real interpreter has, nor add a stand-in for one it does not.

    Reproduced directly: a `numpy.py` on a PYTHONPATH entry that raises
    ImportError the instant it is imported -- a poisoned stand-in, not a
    missing package -- must NOT make the known-good interpreter fail the
    preflight when that PYTHONPATH is only in the *caller's* environment.
    """
    if not Path(GOOD_PYTHON).exists():
        raise AssertionError(f"known-good interpreter missing: {GOOD_PYTHON}")
    with tempfile.TemporaryDirectory(prefix="toolguard-poison-pythonpath-") as tmp:
        poison_dir = Path(tmp) / "poison"
        poison_dir.mkdir()
        (poison_dir / "numpy.py").write_text(
            'raise ImportError("poisoned stand-in for numpy")\n'
        )
        sentinel = Path(tmp) / "cargo-was-called"
        fakebin = _fake_cargo_dir(Path(tmp), sentinel)
        env = {k: v for k, v in os.environ.items() if k != "PYTHON"}
        env["PYTHON"] = GOOD_PYTHON
        env["PATH"] = f"{fakebin}:{env.get('PATH', '')}"
        env["TORCH_C_STAGE"] = str(Path(tmp) / "stage")
        # The poison: present in the CALLER's environment, exactly as a
        # leftover export or an isolating test harness would set it.
        env["PYTHONPATH"] = str(poison_dir)
        proc = subprocess.run(
            ["sh", str(RUN_SH)], cwd=str(HERE), env=env,
            capture_output=True, text=True,
        )
        out = proc.stdout + proc.stderr
        assert "refusing to start" not in out, (
            "a poisoned PYTHONPATH in the CALLER's environment made the "
            f"preflight refuse the known-good interpreter:\n{out}"
        )
        assert sentinel.exists(), (
            "the known-good interpreter did not reach cargo despite the "
            f"poison being confined to the caller's own PYTHONPATH:\n{out}"
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
