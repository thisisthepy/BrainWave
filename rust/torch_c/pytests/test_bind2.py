"""Four items four earlier rounds each stopped one binding short of, because
`bootstrap.py` and `tools/golden/reach_allow.json` belonged to somebody else
each time. docs/BIND2.md is the write-up; this is the proof.

Per docs/BINDINGS.md's own warning (`mish` was a binding onto a kernel that
had been removed), every kernel this file exercises was independently
confirmed present in `_C._aten_implemented()` first -- see
`test_kernels_this_file_binds_are_actually_implemented` below, which fails
loudly if a future change removes one of them out from under this file.

Golden (`tools/golden/compare.py`) dispatches by key and cannot see whether
any Python spelling reaches an arm -- docs/REACH.md's point, and
`test_pad.py`'s `test_torch_rms_norm_reaches_its_kernel_in_the_vendored_tree`
is the pattern this file follows: run the vendored tree in a **separate
process**, in the spelling a user actually writes (`F.avg_pool2d(...)`,
`torch.einsum(...)`, `F.pad(..., mode=...)`), and diff element-wise against
upstream running in its own separate process with no shim on `PYTHONPATH`.
"""

import json
import os
import subprocess
import sys

from test_shim import _C, _CKPT_VENDOR_DIR, _CKPT_VENDOR_SHIM

_REPO_ROOT = os.path.abspath(os.path.join(_CKPT_VENDOR_DIR, "..", "..", ".."))


def test_kernels_this_file_binds_are_actually_implemented():
    """docs/BINDINGS.md's check: confirm the kernel before trusting the
    binding. `mish`'s binding was two lines onto a kernel that had been
    removed; this is what would have caught it.
    """
    implemented = set(_C._aten_implemented())
    for op in (
        "aten.avg_pool2d.default",
        "aten.reflection_pad1d.default",
        "aten.reflection_pad2d.default",
        "aten.reflection_pad3d.default",
        "aten.replication_pad1d.default",
        "aten.replication_pad2d.default",
        "aten.replication_pad3d.default",
    ):
        assert op in implemented, f"{op} missing from _aten_implemented(); the binding this file exercises has nothing behind it"
    # einsum's ellipsis branch is a bootstrap.py composite, not a leaf op --
    # what it needs is the ops its existing (non-ellipsis) decomposition
    # already uses, which the rest of the einsum suite already exercises.
    for op in ("aten.bmm.default", "aten.permute.default", "aten.reshape.default",
               "aten.sum.dim_IntList", "aten.unsqueeze.default"):
        assert op in implemented, f"{op} missing; einsum's composite decomposition needs it"


def test_torch_std_has_no_kernel_here_yet():
    """docs/BIND2.md item 4: `torch.std` dispatches straight to
    `aten::std.correction` upstream (measured with a `TorchDispatchMode`
    logger, recorded in docs/BIND2.md) -- it is a leaf, not a composite over
    `var`, so there is nothing in this file to bind it to. This pins that
    absence so the day a kernel lands elsewhere, this test goes red as a
    reminder to write the binding rather than silently staying green forever.
    """
    implemented = set(_C._aten_implemented())
    assert "aten.std.correction" not in implemented, (
        "aten.std.correction now exists -- torch.std's binding is a sized, "
        "written-up work item in docs/BIND2.md §4; go land it"
    )
    assert "aten.var.correction" not in implemented, (
        "aten.var.correction now exists -- see docs/BIND2.md §4"
    )


def _flatten(x):
    if isinstance(x, list):
        out = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [x]


def _run_vendored(probe, use_shim):
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return None
    env = dict(os.environ)
    if use_shim:
        env["PYTHONPATH"] = _CKPT_VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, env=env, timeout=180, cwd=_REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"probe exited {proc.returncode} (shim={use_shim})\n"
                           f"{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout)


def _max_abs_diff(a, b):
    fa, fb = _flatten(a), _flatten(b)
    assert len(fa) == len(fb), (len(fa), len(fb))
    return max(abs(x - y) for x, y in zip(fa, fb)), len(fa)


_PROBE = """
import json, torch
import torch.nn.functional as F

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}
torch.manual_seed(0)

x = torch.randn(1, 3, 8, 8)
out["avg_pool2d"] = F.avg_pool2d(
    x, 3, 2, 1, ceil_mode=True, count_include_pad=False
).tolist()

a = torch.randn(2, 3, 4, 8)   # batch, q, h, d
b = torch.randn(2, 5, 4, 8)   # batch, k, h, d
out["einsum_ellipsis"] = torch.einsum('...qhd,...khd->...hqk', a, b).tolist()

out["pad_reflect2d"] = F.pad(x, [1, 1, 1, 1], mode="reflect").tolist()
out["pad_replicate2d"] = F.pad(x, [1, 1, 1, 1], mode="replicate").tolist()

x1 = torch.randn(1, 3, 8)
out["pad_reflect1d"] = F.pad(x1, [1, 1], mode="reflect").tolist()
out["pad_replicate1d"] = F.pad(x1, [1, 1], mode="replicate").tolist()

x3 = torch.randn(1, 3, 6, 6, 6)
out["pad_reflect3d"] = F.pad(x3, [1, 1, 1, 1, 1, 1], mode="reflect").tolist()
out["pad_replicate3d"] = F.pad(x3, [1, 1, 1, 1, 1, 1], mode="replicate").tolist()

print(json.dumps(out))
"""


def test_avg_pool2d_matches_upstream_through_F_avg_pool2d():
    """docs/BIND2.md item 1 -- `F.avg_pool2d`, the exact spelling
    `nn.AvgPool2d.forward` uses, and the arguments that exercise
    `ceil_mode`/`count_include_pad` together, which `avg_pool2d_default`'s
    own docstring calls out as the pair a naive port gets backwards.
    """
    shim = _run_vendored(_PROBE, use_shim=True)
    if shim is None:
        return
    assert shim["is_shim"] is True
    upstream = _run_vendored(_PROBE, use_shim=False)
    assert upstream["is_shim"] is False, "the oracle process loaded the shim"
    diff, n = _max_abs_diff(shim["avg_pool2d"], upstream["avg_pool2d"])
    assert diff == 0.0, (diff, n)


def test_einsum_ellipsis_matches_upstream_through_torch_einsum():
    """docs/BIND2.md item 2 -- `longt5`'s exact equation form,
    `'...qhd,...khd->...hqk'`, through `torch.einsum` itself."""
    shim = _run_vendored(_PROBE, use_shim=True)
    if shim is None:
        return
    upstream = _run_vendored(_PROBE, use_shim=False)
    diff, n = _max_abs_diff(shim["einsum_ellipsis"], upstream["einsum_ellipsis"])
    # bmm-based decomposition vs. whatever upstream's own composite picks:
    # float32 rounding between two valid reduction orders, not a divergence.
    assert diff < 1e-5, (diff, n)


def test_pad_six_modes_match_upstream_through_F_pad():
    """docs/BIND2.md item 3 -- all six kernels, through `F.pad(..., mode=...)`
    at every rank they support, not through `_aten_dispatch` directly (that is
    `test_pad.py`'s job and was already true before this round -- what this
    round adds is that `F.pad` reaches them at all).
    """
    shim = _run_vendored(_PROBE, use_shim=True)
    if shim is None:
        return
    upstream = _run_vendored(_PROBE, use_shim=False)
    for key in ("pad_reflect2d", "pad_replicate2d", "pad_reflect1d",
                "pad_replicate1d", "pad_reflect3d", "pad_replicate3d"):
        diff, n = _max_abs_diff(shim[key], upstream[key])
        assert diff == 0.0, (key, diff, n)


def test_pad_circular_still_refuses_by_name():
    """The mode this round did NOT wire (docs/PAD.md §3): a wrong padding
    should fail loudly, not silently approximate with reflect/replicate."""
    probe = """
import torch
import torch.nn.functional as F
x = torch.randn(1, 3, 8, 8)
try:
    F.pad(x, [1, 1, 1, 1], mode="circular")
    print("NOT_RAISED")
except NotImplementedError as e:
    print("RAISED:" + str(e))
"""
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, env=env, timeout=180, cwd=_REPO_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"probe exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    assert proc.stdout.strip().startswith("RAISED:"), proc.stdout
    assert "circular" in proc.stdout


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
    sys.exit(_main())
