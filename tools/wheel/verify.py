#!/usr/bin/env python3
"""Install a built wheel into a throwaway venv and prove the torch inside works.

    python tools/wheel/verify.py dist/torchnative-0.0.1a0-cp313-abi3-macosx_11_0_arm64.whl

Building is not the proof. The claim being tested is "somebody who has never
seen this repository can `pip install` this and `import torch`", and the only
thing that tests it is a interpreter that has never seen this repository.

The load-bearing assertion is `torch.__file__`. If it points into the source
tree, the check proved nothing: the process imported the development copy and
the wheel was never involved. That is easy to do by accident -- running from the
repo root is enough, because `''` leads `sys.path` and `torchnative/src/main` is
reachable from a stray `PYTHONPATH`. So this runs with `-I` (no `PYTHONPATH`, no
user site-packages), with the working directory set outside the repository, and
then checks the path anyway rather than trusting either precaution.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Runs inside the throwaway venv. Prints one JSON object; anything it raises
# surfaces as a non-zero exit, which is the judgement.
PROBE = r"""
import json, sys, importlib.metadata as md

import torch

x = torch.ones(3, 4)
y = torch.ones(4, 2)
mm = torch.ops.aten.mm.default(x, y)

out = {
    "torch_file": torch.__file__,
    "torch_version": torch.__version__,
    "mm": mm.tolist(),
    "mm_dtype": str(mm.dtype),
    "add": (x + x).tolist()[0],
    "C_file": sys.modules["torch._C"].__file__,
    "C_names": len(dir(sys.modules["torch._C"])),
    "aten_ops": len([n for n in dir(torch.ops.aten)]),
    "prefix": sys.prefix,
    "py": sys.version.split()[0],
}
try:
    out["metadata_version_torch"] = md.version("torch")
except Exception as exc:
    out["metadata_version_torch"] = f"<{type(exc).__name__}: {exc}>"
try:
    import torchnative
    out["torchnative"] = torchnative.__file__
except Exception as exc:
    out["torchnative"] = f"<{type(exc).__name__}: {exc}>"

print(json.dumps(out))
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--base-python", default=sys.executable,
                    help="interpreter the throwaway venv is created from")
    ap.add_argument("--venv", type=Path, default=None,
                    help="where to put it (default: a temp dir, removed after)")
    args = ap.parse_args()

    if not args.wheel.exists():
        sys.exit(f"no such wheel: {args.wheel}")

    keep = args.venv is not None
    venv = args.venv or Path(tempfile.mkdtemp(prefix="wheeltest-"))
    if venv.exists() and not keep:
        shutil.rmtree(venv)
    try:
        _run(args.base_python, venv, args.wheel)
    finally:
        if not keep:
            shutil.rmtree(venv, ignore_errors=True)


def _run(base_python: str, venv: Path, wheel: Path) -> None:
    print(f"+ {base_python} -m venv {venv}", flush=True)
    subprocess.run([base_python, "-m", "venv", str(venv)], check=True)
    py = venv / "bin" / "python"

    # Deliberately *with* dependencies. The vendored tree brings upstream
    # torch's pure-Python requirements with it -- `torch/__init__.py:35` imports
    # `typing_extensions` -- so `--no-deps` here would test a state no user is
    # ever in, and the first wheel built by this repo passed a `--no-deps`
    # install and then failed on `import torch`. This reaches the network.
    print(f"+ pip install {wheel.name}", flush=True)
    subprocess.run([str(py), "-m", "pip", "install", "-q", str(wheel)], check=True)
    # `pip list --format=json` rather than `pip freeze`: a wheel installed from
    # a local path freezes as `torchnative @ file:///...#sha256=...`, which
    # splits into three useless tokens on whitespace.
    listed = subprocess.run([str(py), "-m", "pip", "list", "--format=json"],
                            capture_output=True, text=True, check=True)
    print("  pulled in: " + ", ".join(
        f"{d['name']} {d['version']}"
        for d in sorted(json.loads(listed.stdout), key=lambda d: d["name"].lower())
        if d["name"] not in {"torchnative", "pip"}))

    # `-I` drops PYTHONPATH and user site-packages; cwd is moved out of the
    # repository so that `''` on sys.path cannot reach the source tree either.
    print("+ import torch", flush=True)
    proc = subprocess.run([str(py), "-I", "-c", PROBE],
                          cwd=venv, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"probe failed with exit {proc.returncode}")

    out = json.loads(proc.stdout.strip().splitlines()[-1])

    print()
    for key in ("py", "prefix", "torch_version", "torch_file", "C_file",
                "C_names", "aten_ops", "metadata_version_torch", "torchnative"):
        print(f"  {key:24} {out[key]}")
    print(f"  {'aten.mm.default':24} {out['mm']}  ({out['mm_dtype']})")
    print(f"  {'x + x':24} {out['add']}")
    print()

    problems = []
    # `realpath` on both sides: on macOS /tmp is a symlink to /private/tmp, so a
    # plain prefix test on a venv under /tmp fails against a `__file__` the
    # interpreter has already resolved -- reporting "this proves nothing" about
    # a run that proved everything. Comparing unresolved paths is the same class
    # of mistake as the one this check exists to catch.
    root = os.path.realpath(venv)

    def inside(path: str) -> bool:
        return os.path.realpath(path).startswith(root + os.sep)

    # The whole point of the exercise: the torch that answered has to be the one
    # pip put in the venv, not the development tree.
    if not inside(out["torch_file"]):
        problems.append(
            f"torch.__file__ is {out['torch_file']}, outside {root} -- the probe "
            "imported a torch that did not come from the wheel, so this run "
            "proves nothing about the wheel"
        )
    if not inside(out["C_file"]):
        problems.append(f"torch._C came from {out['C_file']}, outside {root}")
    if not out["C_file"].endswith("_C.abi3.so"):
        problems.append(f"torch._C is {out['C_file']}, not our abi3 extension")
    if out["mm"] != [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]:
        problems.append(f"aten.mm.default gave {out['mm']}, expected 3x2 of 4.0")

    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"PASS -- {wheel.name} installs into a clean venv and its torch computes")


if __name__ == "__main__":
    main()
