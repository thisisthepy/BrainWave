"""Spike -- NOT part of the crate.  docs/DYNAMO.md §17 item 2, left unmeasured.

`torch.export` is the other front door to a graph.  Does it need the PEP 523
eval-frame hook (abi3-impossible) or does it get there another way?
"""
import sys
import traceback

import torch
from torch import nn

print("shim" if hasattr(torch._C, "_aten_implemented") else "upstream")


class M(nn.Module):
    def forward(self, x):
        return x * 2 + 1


m = M()
x = torch.ones(3)
print("eager:", m(x))

for label, fn in [
    ("torch.export.export", lambda: torch.export.export(m, (x,))),
    ("torch.fx.symbolic_trace", lambda: torch.fx.symbolic_trace(m)),
    ("torch.jit.trace", lambda: torch.jit.trace(m, (x,))),
]:
    try:
        r = fn()
        print(f"{label}: OK -> {type(r).__name__}")
    except BaseException as e:  # noqa: BLE001
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        where = f"{tb.filename.split('/main/')[-1]}:{tb.lineno}"
        print(f"{label}: {type(e).__name__}: {str(e)[:160]}  @ {where}")
