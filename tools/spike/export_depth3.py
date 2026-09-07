"""Spike -- NOT part of the crate.  Sizes the `torch.export` gap.

`torch.export` does not go through PEP 523 (export_depth2.py).  So its blockers
are ordinary missing `torch._C` symbols, which are *not* abi3-impossible.  This
walks the chain, no-opping one blocker at a time, to count them and to see
whether the chain ever reaches an eval-frame symbol (which would kill the idea).

A no-op is not an implementation.  The output here is a *census of blockers*,
not a claim that export works.
"""
import re
import sys
import traceback
import types

import torch
from torch import nn

print("shim" if hasattr(torch._C, "_aten_implemented") else "upstream")


class _Consts(types.ModuleType):
    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return f"__spike__/{item.lower()}"


torch._C._export.pt2_archive_constants = _Consts("pt2_archive_constants")
sys.modules["torch._C._export.pt2_archive_constants"] = torch._C._export.pt2_archive_constants


class M(nn.Module):
    def forward(self, x):
        return x * 2 + 1


m, x = M(), torch.ones(3)
blockers = []
EVAL_FRAME_HIT = False

for rnd in range(80):
    try:
        ep = torch.export.export(m, (x,))
        print(f"ROUND {rnd}: export returned {type(ep).__name__}")
        break
    except BaseException as e:  # noqa: BLE001
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        where = f"{tb.filename.split('/main/')[-1]}:{tb.lineno}"
        msg = str(e)[:130]
        mm = re.match(r"not implemented in torch\._C shim: (.+)", msg)
        if not mm:
            print(f"ROUND {rnd}: STOP -- {type(e).__name__}: {msg}  @ {where}")
            blockers.append(("STOP", f"{type(e).__name__}: {msg} @ {where}"))
            break
        qual = mm.group(1)
        if "eval_frame" in qual:
            EVAL_FRAME_HIT = True
            print(f"ROUND {rnd}: reached an EVAL-FRAME symbol: {qual} -- stopping.")
            blockers.append(("EVAL_FRAME", qual))
            break
        blockers.append(("missing", qual))
        parts = qual.replace("torch._C.", "").split(".")
        owner = torch._C
        try:
            for p_ in parts[:-1]:
                owner = getattr(owner, p_)
            leaf = parts[-1]
        except Exception:
            print(f"ROUND {rnd}: cannot locate {qual} -- stopping.")
            break

        def _noop(*a, **k):
            return None

        _noop.__name__ = leaf
        setattr(owner, leaf, _noop)
        for modname, mod in list(sys.modules.items()):
            if mod is None or not str(modname).startswith(("torch", "functorch")):
                continue
            try:
                cur = getattr(mod, leaf, None)
            except Exception:
                continue
            if cur is not None and getattr(cur, "__module__", "") == "_torch_c_bootstrap":
                try:
                    setattr(mod, leaf, _noop)
                except Exception:
                    pass
        print(f"ROUND {rnd}: missing {qual}  @ {where}  -- no-opped, continuing")

print("\n--- census ---")
print("blockers encountered:", len(blockers))
for kind, name in blockers:
    print(f"  {kind}: {name}")
print("reached PEP 523 eval-frame:", EVAL_FRAME_HIT)
