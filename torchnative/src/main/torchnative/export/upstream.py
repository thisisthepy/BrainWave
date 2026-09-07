"""The `torch.export` census names — **moved, and this is what is left.**

`docs/EXPORT.md` §8 described this module as "a staging area, not the final
home" and wrote the patch that would move it.  `docs/EXPORT4.md` §10 listed
paying that debt as the next mechanical task.  `docs/EXPORT5.md` §7 paid it:
every implementation that used to live here is now in
`rust/torch_c/src/bootstrap.py`, installed at the end of its `install()`,
before any `from torch._C import ...` in the vendored tree has run.

**Nothing in this file implements anything any more.**  What is left is a
forwarder, kept for one reason and one reason only: several existing tests
(`test_export.py`, `test_export4.py`, `test_dispatch.py`) reach for
`upstream.install()`, `upstream._RAII_GUARDS` and friends, and pointing those
at the bootstrap's copies is what makes them start testing the **final home**
rather than a staging area that no longer exists.  Deleting the file outright
would have deleted those tests with it.

Why the monkey-patch had to go, restated because it is the point of the move:
installing after `import torch` meant a `rebind()` pass over roughly forty
`from torch._C import ...` bindings that other torch modules had already made
— including aliases like `torch/utils/_mode_utils.py:15`'s
``no_dispatch = torch._C._DisableTorchDispatch``, which is why that pass had to
match by object identity rather than by name.  All of it is gone.

And the measurement that made it urgent rather than tidy: before the move,
`rust/torch_c/pytests/export_sweep.py` run against the shim stopped at census
name #0 (`torch._C._unset_dispatch_mode`) on **every** architecture, because
the sweep's subprocess never called `install()`.  Every export number in
`docs/EXPORT.md` and `docs/EXPORT4.md` was taken under the patch, and
`docs/EXPORT4.md` §1.1 said so.  After the move the names are simply there.
"""

from __future__ import annotations

import sys


__all__ = ["install", "installed_names", "InstallReport"]


def _bootstrap():
    """The module the implementations now live in.

    It is `_torch_c_bootstrap` in `sys.modules` — `rust/torch_c/src/bootstrap.py`
    is `include_str!`'d into the extension at Rust build time and executed under
    that name, which is why this is a `sys.modules` lookup and not an import.
    """
    module = sys.modules.get("_torch_c_bootstrap")
    if module is None:
        raise RuntimeError(
            "torchnative.export.upstream: the torch._C shim's bootstrap module is "
            "not loaded, so the census names it now owns cannot be reached. This "
            "file stopped implementing them in docs/EXPORT5.md §7; import torch "
            "from the vendored tree first (see docs/EXPORT5.md)"
        )
    return module


class InstallReport:
    """What `install()` used to return, with every count now zero.

    Kept so that callers which print it keep working.  The numbers are zero
    because there is nothing left to install: `replaced` counted placeholders
    this module overwrote at runtime, and after the hand-off there are none —
    the bootstrap never creates them in the first place.
    """

    def __init__(self):
        self.replaced = []
        self.overridden = []
        self.left_alone = []
        self.displaced = []
        self.new_objects = {}
        self.rebound = 0

    def __repr__(self):
        return (
            f"<InstallReport moved-to-bootstrap "
            f"replaced={len(self.replaced)} overridden={len(self.overridden)}>"
        )


#: The `torch._C` names the bootstrap now supplies.  Unchanged from the tuple
#: this module carried while it installed them, so the census in
#: `docs/EXPORT.md` §2 and the tests that read it cannot drift apart silently.
_NAMES = (
    "_unset_dispatch_mode",
    "_set_dispatch_mode",
    "_get_dispatch_mode",
    "_push_on_torch_dispatch_stack",
    "_pop_torch_dispatch_stack",
    "_len_torch_dispatch_stack",
    "_get_dispatch_stack_at",
    "_only_lift_cpu_tensors",
    "_set_only_lift_cpu_tensors",
    "_ensureCUDADeviceGuardSet",
    "_dispatch_tls_local_include_set",
    "_dispatch_tls_local_exclude_set",
    "_dispatch_tls_is_dispatch_key_included",
    "_dispatch_tls_is_dispatch_key_excluded",
    "_dispatch_tls_set_dispatch_key_included",
    "_functionalization_reapply_views_tls",
    "_meta_in_tls_dispatch_include",
    "_set_meta_in_tls_dispatch_include",
    "_set_conj",
    "_set_neg",
    "_InferenceMode",
    "_DisableTorchDispatch",
    "_DisableFuncTorch",
    "_DisableAutocast",
    "_AutoDispatchBelowAutograd",
    "_RestorePythonTLSSnapshot",
    "_DisablePythonDispatcher",
    "_EnablePythonDispatcher",
    "_EnablePreDispatch",
    "_PreserveDispatchKeyGuard",
    "_SetExcludeDispatchKeyGuard",
    "_is_inference_mode_enabled",
    "_ForceDispatchKeyGuard",
    "_ExcludeDispatchKeyGuard",
    "_IncludeDispatchKeyGuard",
)


def installed_names():
    """The `torch._C` module-level names the bootstrap supplies."""
    return _NAMES


def install(torch_module=None):
    """**A no-op that reports nothing was installed, because nothing needs to be.**

    It is deliberately not an error to call this.  Probes and tests written
    against the staging arrangement keep running, and what they measure
    afterwards is the bootstrap's behaviour — which is the point.  It does not
    silently *look* like it installed something: every count in the report is
    zero and the repr says `moved-to-bootstrap`.
    """
    _bootstrap()  # refuse by name if the shim is not the torch that is loaded
    return InstallReport()


def rebind(report=None):
    """The pass that no longer has anything to repoint.

    It existed because this module patched `torch._C` *after* the vendored tree
    had already bound names out of it.  The bootstrap runs first, so every one
    of those bindings is made against the real implementation.  Zero.
    """
    return 0


def __getattr__(name):
    """Forward everything else to the bootstrap, so tests read the final home.

    `_RAII_GUARDS`, `_KEY_STATE_GUARDS`, `_is_guard_active`, `_is_stub`,
    `_is_definitely_a_view` and the `_install_*` functions are all reached this
    way.  Forwarding rather than re-exporting at import time matters: this
    module is importable before `torch` is, and the bootstrap only exists once
    the shim has loaded.
    """
    module = sys.modules.get("_torch_c_bootstrap")
    if module is not None and hasattr(module, name):
        return getattr(module, name)
    raise AttributeError(
        f"torchnative.export.upstream has no attribute {name!r}. The "
        f"implementations moved to rust/torch_c/src/bootstrap.py in "
        f"docs/EXPORT5.md §7; this module forwards to it and the bootstrap does "
        f"not have that name either."
    )
