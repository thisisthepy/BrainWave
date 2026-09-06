"""Spike -- NOT part of the crate.  The one that decides the honesty fix.

`bootstrap.py` already ships `set_eval_frame` as a state cell that remembers a
callback and installs no PEP 523 hook (rust/torch_c/src/bootstrap.py, the
`if "_dynamo" in roots:` block).  docs/DYNAMO.md §14 showed that once the
*other* blockers are filled, that cell turns `torch.compile` into a silent eager
fallback -- no exception, no compilation, no graph.

Today the user is saved from that only by an unrelated accident: a missing data
module, `torch._C._export.pt2_archive_constants`, raises first.  This script
fills exactly that accident (plus the handful of dispatcher no-ops behind it)
and then asks the only question that matters: does the function still run in
Python, and do dynamo's counters stay at zero?

If they do, closing the `pt2_archive_constants` gap -- 27 string constants, an
obviously-correct fix somebody will make -- silently converts a loud failure
into a lie.
"""
import sys
import types

import torch

print("shim" if hasattr(torch._C, "_aten_implemented") else "upstream")


class _Consts(types.ModuleType):
    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return f"__spike__/{item.lower()}"


torch._C._export.pt2_archive_constants = _Consts("pt2_archive_constants")
sys.modules["torch._C._export.pt2_archive_constants"] = \
    torch._C._export.pt2_archive_constants


class _Guard:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_KS = torch._C.DispatchKeySet


def _stub(name, fn):
    setattr(torch._C, name, fn)
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        cur = getattr(mod, name, None)
        if cur is not None and getattr(cur, "__module__", "") == "_torch_c_bootstrap":
            try:
                setattr(mod, name, fn)
            except Exception:
                pass


_stub("_dispatch_tls_local_include_set", lambda: _KS(torch._C.DispatchKey.Undefined))
_stub("_dispatch_tls_local_exclude_set", lambda: _KS(torch._C.DispatchKey.Undefined))
torch._C._ForceDispatchKeyGuard = _Guard
torch._C._dynamo.eval_frame.set_skip_guard_eval_unsafe = lambda v: False
torch._C._functorch.pop_dynamic_layer_stack_and_undo_to_depth = lambda d: None

calls = {"n": 0}


def f(x):
    calls["n"] += 1
    return x * 2 + 1


x = torch.ones(3)
try:
    cf = torch.compile(f, backend="eager")
    outs = [cf(x) for _ in range(3)]
    print("torch.compile(f)(x) returned:", outs[-1])
    print("python-level calls to f:", calls["n"], "(3 == never compiled)")
    from torch._dynamo import utils as du
    print("dynamo frames counter:", dict(du.counters["frames"]))
    print("dynamo stats counter :", dict(du.counters["stats"]))
    print()
    print("VERDICT: SILENT EAGER FALLBACK" if calls["n"] == 3 and
          not du.counters["frames"] else "VERDICT: something compiled")
except BaseException as e:  # noqa: BLE001
    import traceback
    tb = traceback.extract_tb(sys.exc_info()[2])[-1]
    print("still raises:", type(e).__name__, str(e)[:160],
          "@", tb.filename.split("/main/")[-1], tb.lineno)
