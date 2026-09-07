"""Spike -- NOT part of the crate.  Follow-on to export_depth.py.

Two questions the first pass left:
  1. Past the one shallow data gap (`_C._export.pt2_archive_constants`, which is
     27 string constants and no behaviour), does `torch.export` hit the same
     PEP 523 wall as `torch.compile`, or a different one?
  2. `torch.jit.trace` returned something.  Is it a real trace or a no-op?
"""
import sys
import traceback
import types

import torch
from torch import nn

print("shim" if hasattr(torch._C, "_aten_implemented") else "upstream")
print("PYTORCH_JIT env:", repr(__import__("os").environ.get("PYTORCH_JIT")))


class _Consts(types.ModuleType):
    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return f"__spike__/{item.lower()}"


torch._C._export.pt2_archive_constants = _Consts("pt2_archive_constants")
sys.modules["torch._C._export.pt2_archive_constants"] = \
    torch._C._export.pt2_archive_constants
print("stubbed pt2_archive_constants (scaffold only, not in the crate)")


class M(nn.Module):
    def forward(self, x):
        return x * 2 + 1


m = M()
x = torch.ones(3)

# --- 2. is jit.trace real? -------------------------------------------------
seen = {"n": 0}


class Counted(nn.Module):
    def forward(self, y):
        seen["n"] += 1
        return y * 2 + 1


c = Counted()
try:
    tm = torch.jit.trace(c, (x,))
    print("jit.trace returned:", type(tm).__name__, "| is it the module itself?",
          tm is c)
    before = seen["n"]
    out = tm(x)
    print("  calls to python forward during traced call:", seen["n"] - before,
          "(0 would mean a real trace ran instead)")
    print("  has .graph?", hasattr(tm, "graph"))
    print("  result:", out)
except BaseException as e:  # noqa: BLE001
    print("jit.trace raised:", type(e).__name__, str(e)[:150])

# --- 1. torch.export past the data gap ------------------------------------
for label, fn in [
    ("torch.export.export", lambda: torch.export.export(m, (x,))),
    ("torch.compile(backend=eager)", lambda: torch.compile(m, backend="eager")(x)),
]:
    try:
        r = fn()
        print(f"{label}: OK -> {type(r).__name__}")
    except BaseException as e:  # noqa: BLE001
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        where = f"{tb.filename.split('/main/')[-1]}:{tb.lineno}"
        print(f"{label}: {type(e).__name__}: {str(e)[:200]}  @ {where}")
