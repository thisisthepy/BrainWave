"""Spike -- NOT part of the crate, imported by nothing that ships.

Question: when `torch.compile` fails in this build, is the wall a pile of small
data gaps in `torch._C`, or one structural thing (CPython frame evaluation)?

Method: call `torch.compile(f, backend=...)(x)`.  On failure, record the blocker,
then patch *only the object named in the failing line* with the cheapest
placeholder that could possibly satisfy it, and go again.  When the loop stops
finding cheap patches, the blocker it stopped on is the real wall.

Run:
  PYTHONPATH=<repo>/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 python spike/compile_depth.py
"""
import re
import sys
import traceback
import types

import torch

print("shim" if hasattr(torch._C, "_aten_implemented") else "upstream")

_probe = getattr(getattr(torch._C, "_export", None), "pt2_archive_constants", None)
UNIMPL = type(_probe) if type(_probe).__name__ == "_Unimplemented" else None
print("placeholder class:", UNIMPL)


class _Auto(types.ModuleType):
    """Answers any attribute with a string named after itself."""

    def __init__(self, name):
        super().__init__(name)

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return f"__spike__/{item.lower()}"


def _walk_C(depth=3):
    """Yield (owner, attrname, value) over torch._C and its module attributes."""
    seen = set()
    stack = [("torch._C", torch._C, 0)]
    while stack:
        path, obj, d = stack.pop()
        if id(obj) in seen or d > depth:
            continue
        seen.add(id(obj))
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                v = getattr(obj, name)
            except Exception:
                continue
            yield path, obj, name, v
            if isinstance(v, types.ModuleType):
                stack.append((f"{path}.{name}", v, d + 1))


def patch_named(base_name, log):
    """Replace every placeholder reachable under torch._C bound to `base_name`."""
    hits = 0
    for path, owner, name, v in _walk_C():
        if name == base_name and UNIMPL is not None and isinstance(v, UNIMPL):
            setattr(owner, name, _Auto(f"{path}.{name}"))
            log.append(f"patched {path}.{name} (was a placeholder) -> _Auto")
            hits += 1
    return hits


NOOPPED = []


def _make_noop(qualname):
    def _noop(*a, **k):
        CALLS.append((qualname, len(a)))
        return None
    _noop.__name__ = qualname.rsplit(".", 1)[-1]
    return _noop


CALLS = []


def f(x):
    return x * 2 + 1


x = torch.ones(3)
BACKEND = sys.argv[1] if len(sys.argv) > 1 else "eager"
log = []
verdict = "?"

for rnd in range(60):
    try:
        g = torch.compile(f, backend=BACKEND)
        r = g(x)
        ok = bool((r == f(x)).all())
        print(f"ROUND {rnd}: COMPILED AND RAN -> {r}  matches eager: {ok}")
        verdict = "reached execution"
        break
    except BaseException as e:  # noqa: BLE001
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        where = f"{tb.filename.split('/main/')[-1]}:{tb.lineno}"
        print(f"ROUND {rnd}: {type(e).__name__}: {str(e)[:130]}  @ {where}")
        log.append(f"[{rnd}] {type(e).__name__}: {str(e)[:130]}  @ {where}")

        m = re.match(r"'_Unimplemented' object has no attribute '(\w+)'", str(e))
        base = None
        if m and tb.line:
            # `foo.BAR` -- the placeholder is whatever `foo` resolves to
            mm = re.search(r"(\w+)\." + re.escape(m.group(1)), tb.line)
            if mm:
                base = mm.group(1)
        m2 = re.match(r"not implemented in torch\._C shim: torch\._C\.(.+)", str(e))
        if m2:
            parts = m2.group(1).split(".")
            owner = torch._C
            for p_ in parts[:-1]:
                owner = getattr(owner, p_)
            leaf = parts[-1]
            noop = _make_noop(m2.group(1))
            setattr(owner, leaf, noop)
            # the name is usually also bound by `from ... import X` elsewhere
            rebound = 0
            for modname, mod in list(sys.modules.items()):
                if mod is None or not modname.startswith(("torch", "functorch")):
                    continue
                try:
                    cur = getattr(mod, leaf, None)
                except Exception:
                    continue
                if cur is not None and getattr(cur, "__module__", "") == "_torch_c_bootstrap":
                    try:
                        setattr(mod, leaf, noop)
                        rebound += 1
                    except Exception:
                        pass
            log.append(f"no-opped torch._C.{m2.group(1)} (+{rebound} rebindings)")
            NOOPPED.append(m2.group(1))
            continue

        if base and patch_named(base, log):
            for mod in [k for k in sys.modules if k.startswith("torch.export.pt2_archive")]:
                sys.modules.pop(mod, None)
            continue
        print(f"ROUND {rnd}: no cheap patch for this -- this is the wall.")
        verdict = f"{type(e).__name__} @ {where}"
        traceback.print_exc()
        break

print("\n--- backend:", BACKEND, "---")
print("VERDICT:", verdict)
for line in log:
    print(" ", line)
print("\nno-opped (%d):" % len(NOOPPED))
for n in NOOPPED:
    print("  ", n)
print("\ncalls through no-ops:", CALLS)
