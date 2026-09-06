"""Proof that `run.sh` runs more than one suite file, and that they can share.

This exists because the split itself needs a test. `run.sh` used to name
`test_shim.py` directly; if a later edit put that back, every `test_*.py`
written since would stop running **silently and green** -- which is the exact
failure shape this repository keeps meeting. So one assertion here reads
`run.sh` and requires the glob.

The import is the other half: helpers live in `test_shim` and the files that
split off it have to be able to reach them, which is what `pytests/` on
PYTHONPATH is for.
"""

import pathlib

from test_shim import _C


def test_run_sh_runs_every_suite_file_and_not_just_test_shim():
    run_sh = pathlib.Path(__file__).with_name("run.sh").read_text()
    assert "pytests/test_*.py" in run_sh, (
        "run.sh no longer globs the suite files. Every test_*.py added since "
        "the split would stop running, silently and green."
    )
    assert '"$crate_dir/pytests/test_shim.py"' not in run_sh, (
        "run.sh names test_shim.py directly again -- the split is undone"
    )


def test_the_shared_helpers_import_across_suite_files():
    assert hasattr(_C, "_aten_implemented"), "test_shim's _C did not import"


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
