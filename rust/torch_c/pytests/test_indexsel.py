"""docs/kernels/INDEXSEL.md: index_select, argsort, where.Scalar, new_full,
reshape_as, unflatten, chunk (free-function), diff, multiply, logical_and.

Most of `tools/golden/cases.py`'s `CASE_BUILDERS` entries (added alongside
this file) already prove every op here except `reshape_as` element-wise
against upstream, through `run.sh`'s own `compare.py` step. `reshape_as` is
the one op in this round with **no genuine `torch.ops.aten` entry** --
`TensorBase.reshape_as` lowers to `aten.view.default` upstream (measured with
a `TorchDispatchMode` logger) rather than naming an op of its own, so
`tools/golden/loader.py`'s `resolve_torch_overload` structurally cannot
reach it and it is never offered to `compare.py`'s per-op loop. This file is
where that op is proven instead: a real Python-level call
(`x.reshape_as(y)`), through the full vendored `torch` package (so
`methods.json`'s resolver and this shim's invented schema for the entry are
exercised too, not just the raw `_aten_dispatch` key), computed once under
the shim and once under upstream **in separate processes**, and compared
element-wise.

The other nine ops get a second, cheap check here on top of golden: that the
*Python spelling* a real caller uses (`x.index_select(...)`,
`torch.argsort(...)`, `torch.chunk(...)`, ...) actually reaches the kernel
through the full vendored tree's `methods.json`/`overloads.json` resolver --
golden's `_aten_dispatch(op, ...)` calls skip that resolver entirely and
would not catch a table entry that resolves to the wrong key or not at all.
"""

import json
import os
import subprocess
import sys

import _C

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")


# ---------------------------------------------------------------------------
# Part 1: every new op is advertised where it should be.
# ---------------------------------------------------------------------------

_GOLDEN_COVERED = [
    "aten.index_select.default",
    "aten.argsort.default",
    "aten.argsort.stable",
    "aten.where.Scalar",
    "aten.new_full.default",
    "aten.unflatten.int",
    "aten.chunk.default",
    "aten.diff.default",
    "aten.multiply.Tensor",
    "aten.multiply.Scalar",
    "aten.logical_and.default",
]


def test_the_ten_real_kernels_are_advertised_as_golden_covered():
    implemented = set(_C._aten_implemented())
    missing = [op for op in _GOLDEN_COVERED if op not in implemented]
    assert not missing, (
        f"{missing} have a dispatch arm but are not in _aten_implemented() -- "
        "either the IMPLEMENTED table entry is missing or it was misspelled"
    )


def test_reshape_as_is_parked_in_awaiting_golden_not_implemented():
    # It has a real kernel (`reshape_as_default`) and a dispatch arm, but no
    # genuine `torch.ops.aten.reshape_as` for compare.py to resolve against
    # -- see the module docstring. `_aten_implemented()` promises "golden
    # compares this", which would be a false promise for this key.
    assert "aten.reshape_as.default" not in _C._aten_implemented()
    all_ops = _C._aten_all_implemented()
    assert "aten.reshape_as.default" in all_ops


# ---------------------------------------------------------------------------
# Part 2: the real Python spellings, through the full vendored tree, each
# side in its own process (docs/kernels/INDEXSEL.md's bar).
# ---------------------------------------------------------------------------

_SCRIPT = r"""
import json, sys
import torch

out = {}

# index_select -- m2m_100's own call shape: dim=0, a flattened position index.
w = torch.arange(24, dtype=torch.float32).reshape(4, 6)
idx = torch.tensor([2, 0, 3])
out["index_select"] = w.index_select(0, idx).tolist()

# argsort -- aria's spelling: no dim, no descending, ties present.
x = torch.tensor([3, 1, 3, 2, 1, 3, 0, 1])
out["argsort"] = torch.argsort(x).tolist()

# where.Scalar -- both branches Python scalars.
cond = torch.tensor([True, False, True, False])
out["where_scalar"] = torch.where(cond, 1, 2.5).tolist()

# new_full -- led's spelling: dtype override to int64.
base = torch.zeros(2, 2)
out["new_full"] = base.new_full((2, 3), 7, dtype=torch.long).tolist()

# reshape_as -- roformer's spelling.
sin = torch.tensor([1.0, 2.0])
stacked = torch.stack([sin, sin], dim=-1)
target = torch.zeros(4)
out["reshape_as"] = stacked.reshape_as(target).tolist()

# unflatten -- siglip's spelling, through F.multi_head_attention_forward's
# own call shape (dim=-1, an inferred size).
kv = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
out["unflatten"] = kv.unflatten(-1, (2, -1)).tolist()

# chunk (free function) -- diffllama's spelling.
v = torch.arange(10, dtype=torch.float32)
out["chunk"] = [piece.tolist() for piece in torch.chunk(v, 2, dim=0)]

# diff -- voxtral_realtime_encoder's spelling: prepend, dim=-1.
pos = torch.tensor([0, 1, 2, 0, 1, 2, 3])
dummy = torch.tensor([-1])
out["diff"] = torch.diff(pos, prepend=dummy, dim=-1).tolist()

# multiply -- convbert's spelling.
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
out["multiply"] = torch.multiply(a, b).tolist()

# logical_and -- longt5's spelling.
m1 = torch.tensor([True, False, True])
m2 = torch.tensor([True, True, False])
out["logical_and"] = torch.logical_and(m1, m2).tolist()

json.dump(out, sys.stdout)
"""


def _run_side(shim: bool) -> dict:
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{'shim' if shim else 'upstream'} side failed "
            f"(rc={proc.returncode}):\n{proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout)


def test_every_real_python_spelling_matches_upstream_element_wise():
    shim = _run_side(shim=True)
    upstream = _run_side(shim=False)
    assert set(shim) == set(upstream), (shim.keys(), upstream.keys())
    for key in upstream:
        assert shim[key] == upstream[key], (
            f"{key}: shim={shim[key]!r} vs upstream={upstream[key]!r}"
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
