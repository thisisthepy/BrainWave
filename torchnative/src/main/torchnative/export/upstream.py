"""Real implementations for the `torch._C` symbols `torch.export.export()` wants.

`docs/COMPILE.md` §3 censused the blockers on the `torch.export` path with
*crude no-ops* -- every missing name replaced by a function returning `None` --
and stopped at round 19 when `torch/_subclasses/meta_utils.py:1061` called
`.has()` on one of those `None`s.  That stop was an artefact of the stand-in,
and COMPILE.md said so.  This module is what replaces the stand-ins: each name
below gets the behaviour upstream's C++ gives it, so the census can be re-derived
against real values instead of `None`.

**This module is a staging area, not the final home.**  Every function here
belongs in `rust/torch_c/src/bootstrap.py`, beside `_install_dispatch_keys`,
which already builds the `DispatchKey` enum and the `DispatchKeySet` these
depend on.  `docs/EXPORT.md` §5 carries the patch in the form the bootstrap
author applies.  Until it lands, `install()` monkey-patches the same objects at
runtime so the result is measurable and testable.

What is *not* here: anything that makes the dispatcher route a call.  The TLS
sets and the mode stack are **bookkeeping that upstream Python code reads and
branches on**; `_aten_dispatch` remains the one door, exactly as
`_install_dispatch_keys` says.  A shim that stored a key in the TLS include set
and then dispatched differently because of it would be claiming a dispatcher it
does not have.
"""

from __future__ import annotations

import threading
import traceback as _traceback


__all__ = ["install", "installed_names", "InstallReport"]


# --------------------------------------------------------------------------
# Thread-local dispatcher state
# --------------------------------------------------------------------------

class _Tls(threading.local):
    """Per-thread dispatcher bookkeeping.

    Upstream keeps these in C++ TLS (`c10::impl::LocalDispatchKeySet`, the
    `TorchDispatchModeTLS` stack).  `threading.local` is the same lifetime with
    the same visibility rules, which matters: `torch/utils/_python_dispatch.py`
    enters and exits these in `with` blocks on whatever thread is running, and
    a module-level global would leak one thread's mode stack into another's.
    """

    def __init__(self) -> None:
        self.included = None      # DispatchKeySet, filled on first use
        self.excluded = None      # DispatchKeySet
        self.mode_stack = []      # user dispatch modes, innermost last
        self.infra_modes = {}     # _TorchDispatchModeKey -> mode
        self.only_lift_cpu_tensors = False
        self.reapply_views = False
        self.meta_in_tls_dispatch_include = False
        self.inference_mode = False


_TLS = _Tls()


def _keyset(module):
    return module.DispatchKeySet


def _included(module):
    if _TLS.included is None:
        _TLS.included = _keyset(module)()
    return _TLS.included


def _excluded(module):
    if _TLS.excluded is None:
        _TLS.excluded = _keyset(module)()
    return _TLS.excluded


# --------------------------------------------------------------------------
# The installer
# --------------------------------------------------------------------------

class InstallReport:
    """What `install()` did, so a test can assert on it rather than on prose."""

    def __init__(self) -> None:
        self.replaced = []      # names that were `_Unimplemented`/raising stubs
        self.overridden = []    # names that had a non-stub answer, now displaced
        self.left_alone = []    # names already implemented, deliberately kept
        self.new_objects = {}   # qualname -> (leaf name, object), for rebind()
        self.displaced = []     # (old object, new object) pairs, for rebind()

    def __repr__(self) -> str:
        return (
            f"<InstallReport replaced={len(self.replaced)} "
            f"overridden={len(self.overridden)}>"
        )


#: Every `torch._C` name this module supplies, in the order COMPILE.md §3
#: encountered it plus the ones that turned out to be needed once the crude
#: no-ops were replaced by real values.  `installed_names()` returns it so the
#: test file and the doc census cannot drift apart silently.
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
    "_functionalization_reapply_views_tls",
    "_meta_in_tls_dispatch_include",
    "_set_meta_in_tls_dispatch_include",
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
    """The `torch._C` module-level names `install()` supplies."""
    return _NAMES


_MARK = "__torchnative_export_impl__"


def _mark_ours(value):
    """Tag an installed object so `_is_stub` can recognise it later."""
    target = value.fget if isinstance(value, property) else value
    try:
        setattr(target, _MARK, True)
    except (AttributeError, TypeError):
        pass
    return value


def _is_ours(obj) -> bool:
    target = obj.fget if isinstance(obj, property) else obj
    try:
        return getattr(target, _MARK, False) is True
    except Exception:
        return False


def _is_stub(obj) -> bool:
    """Is this name a placeholder rather than an implementation?

    Three shapes count, and the bootstrap makes all three.  `_Unimplemented` is
    what it leaves when the stubs say nothing about a name.  `_make_function`
    leaves an ordinary Python function whose body raises `NotImplementedError`.
    `_make_property` leaves a `property` whose getter does the same.  None of
    them may be *called* to find out -- calling raises, and `__bool__` on an
    `_Unimplemented` raises too -- so this reads the code object's constants
    instead of probing behaviour.
    """
    if obj is None:
        return True
    if _is_ours(obj):
        # An implementation this module installed.  Asked because the check
        # below reads code constants for the bootstrap's refusal message, and
        # `_base`'s implementation *quotes* that message inside its own, richer
        # refusal -- so a text test misreads it as the placeholder it replaced.
        # The test that caught it is
        # `test_install_is_idempotent_in_the_only_sense_that_matters`.
        return False
    if type(obj).__name__ == "_Unimplemented":
        return True
    if isinstance(obj, property):
        return _is_stub(obj.fget)
    try:
        code = getattr(obj, "__code__", None)
    except Exception:
        # `torch/_classes.py` synthesises attributes on access and raises for
        # anything unregistered.  A module that answers every name is not a
        # module whose bindings need repointing.
        return False
    if code is None:
        return False
    return any(
        isinstance(c, str) and "not implemented in torch._C shim" in c
        for c in code.co_consts
    )


def install(torch_module=None) -> InstallReport:
    """Put real behaviour on the `torch.export` blockers.  Idempotent."""
    if torch_module is None:
        import torch as torch_module
    C = torch_module._C
    report = InstallReport()

    def put(owner, name, value, qual=None, only_if_stub=False):
        """Install `value`, recording what it displaced.

        Every name here is installed unconditionally, and the report says which
        of them were placeholders.  The alternative -- skip anything that looks
        implemented -- is the trap `docs/COMPILE.md` §5 names: some of these
        *are* implemented, as constants that answer the wrong thing.
        `_len_torch_dispatch_stack` is the bootstrap's `_DISCOVERED_RETURNS`
        entry `0`, a perfectly ordinary function that is not a stub and is
        nonetheless wrong the moment a mode is pushed.  Skipping it would leave
        `with FakeTensorMode():` entering and changing nothing, silently.
        """
        existing = getattr(owner, name, None)
        if _is_stub(existing):
            report.replaced.append(qual or name)
        elif only_if_stub:
            report.left_alone.append(qual or name)
            return
        else:
            report.overridden.append(qual or name)
        _mark_ours(value)
        setattr(owner, name, value)
        report.new_objects[qual or name] = (name, value)
        if existing is not None:
            report.displaced.append((existing, value))

    _install_tls(C, put)
    _install_mode_stack(C, put)
    _install_tensor_predicates(torch_module, C, put)
    _install_functorch(C, put)
    _install_profiler(C, put)
    _install_dynamo_bool(C, put)
    _install_inference_mode(C, put)
    _install_raii_guards(C, put)
    rebind(report)
    return report


def rebind(report) -> int:
    """Re-point `from torch._C import X` bindings at the new objects.

    Only needed because this module runs *after* `import torch`.  Roughly forty
    modules in the vendored tree bind these names at import time --
    `torch/utils/_python_dispatch.py:16` does
    ``from torch._C import _len_torch_dispatch_stack, ...`` and
    ``from torch._C._dynamo.guards import
    set_is_in_mode_without_ignore_compile_internals`` -- so setting the
    attribute on `torch._C` alone leaves every one of those importers holding
    the old stub.  That is not a subtlety to remember: it is the difference
    between the census stopping at round 4 and continuing.

    **When this lands in `bootstrap.py` this function disappears**, because the
    bootstrap runs before any of those imports.  Its presence here is the mark
    of the staging arrangement, not of a design.
    """
    import sys

    # **By identity, not by name.**  Name matching misses the aliases, and the
    # aliases are not rare: `torch/utils/_mode_utils.py:15` is
    # ``no_dispatch = torch._C._DisableTorchDispatch``, so the binding that
    # `fake_tensor.py` actually enters is spelled `no_dispatch` and a
    # by-name pass walks straight past it.  Identity catches every rebinding of
    # the displaced object whatever it was renamed to, and cannot touch a
    # module's own unrelated `no_dispatch`.
    by_id = {id(old): new for old, new in report.displaced}
    n = 0
    for modname, mod in list(sys.modules.items()):
        if mod is None or not str(modname).startswith(("torch", "functorch")):
            continue
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for leaf, existing in list(d.items()):
            replacement = by_id.get(id(existing))
            if replacement is None or replacement is existing:
                continue
            try:
                d[leaf] = replacement
                n += 1
            except Exception:
                pass
    report.rebound = n
    return n


# --------------------------------------------------------------------------
# 1. Dispatcher TLS -- COMPILE.md rounds 15, 16, 17, 18
# --------------------------------------------------------------------------

def _install_tls(C, put) -> None:
    """`_dispatch_tls_*`, the four names round 19 died on.

    Round 18's `_dispatch_tls_local_exclude_set` returning `None` is the whole
    of COMPILE.md's stopping point: `meta_utils.py:1061` does
    ``...local_exclude_set().has(DispatchKey.ADInplaceOrView)``.  A real
    `DispatchKeySet` -- which `_install_dispatch_keys` already builds -- answers
    it, and the answer is `False`, which is the truth for a shim that has
    entered no guard.
    """
    KeySet = C.DispatchKeySet

    def _dispatch_tls_local_include_set():
        return _included(C)

    def _dispatch_tls_local_exclude_set():
        return _excluded(C)

    def _dispatch_tls_is_dispatch_key_included(key):
        return _included(C).has(key)

    def _dispatch_tls_is_dispatch_key_excluded(key):
        return _excluded(C).has(key)

    def _functionalization_reapply_views_tls():
        # Upstream: whether the functionalization pass should re-apply view ops
        # rather than materialise copies.  A flag read by
        # `torch/_subclasses/functional_tensor.py`; nothing here sets it, so
        # `False` is the state, not a stand-in.
        return _TLS.reapply_views

    def _meta_in_tls_dispatch_include():
        return _TLS.meta_in_tls_dispatch_include

    def _set_meta_in_tls_dispatch_include(value):
        _TLS.meta_in_tls_dispatch_include = bool(value)

    put(C, "_dispatch_tls_local_include_set", _dispatch_tls_local_include_set)
    put(C, "_dispatch_tls_local_exclude_set", _dispatch_tls_local_exclude_set)
    put(C, "_dispatch_tls_is_dispatch_key_included",
        _dispatch_tls_is_dispatch_key_included)
    put(C, "_dispatch_tls_is_dispatch_key_excluded",
        _dispatch_tls_is_dispatch_key_excluded)
    put(C, "_functionalization_reapply_views_tls",
        _functionalization_reapply_views_tls)
    put(C, "_meta_in_tls_dispatch_include", _meta_in_tls_dispatch_include)
    put(C, "_set_meta_in_tls_dispatch_include", _set_meta_in_tls_dispatch_include)

    class _ForceDispatchKeyGuard:
        """`with _ForceDispatchKeyGuard(include, exclude):` -- set both, restore both."""

        __module__ = "torch._C"

        def __init__(self, include=None, exclude=None):
            self._include = include
            self._exclude = exclude
            self._saved = None

        def __enter__(self):
            self._saved = (_included(C), _excluded(C))
            if self._include is not None:
                _TLS.included = KeySet(self._include)
            if self._exclude is not None:
                _TLS.excluded = KeySet(self._exclude)
            return self

        def __exit__(self, *exc):
            _TLS.included, _TLS.excluded = self._saved
            return False

    class _ExcludeDispatchKeyGuard:
        __module__ = "torch._C"

        def __init__(self, keyset):
            self._keyset = keyset
            self._saved = None

        def __enter__(self):
            self._saved = _excluded(C)
            _TLS.excluded = self._saved | KeySet(self._keyset)
            return self

        def __exit__(self, *exc):
            _TLS.excluded = self._saved
            return False

    class _IncludeDispatchKeyGuard:
        __module__ = "torch._C"

        def __init__(self, key):
            self._key = key
            self._saved = None

        def __enter__(self):
            self._saved = _included(C)
            _TLS.included = self._saved | KeySet(self._key)
            return self

        def __exit__(self, *exc):
            _TLS.included = self._saved
            return False

    put(C, "_ForceDispatchKeyGuard", _ForceDispatchKeyGuard)
    put(C, "_ExcludeDispatchKeyGuard", _ExcludeDispatchKeyGuard)
    put(C, "_IncludeDispatchKeyGuard", _IncludeDispatchKeyGuard)


# --------------------------------------------------------------------------
# 2. The torch-dispatch mode stack -- COMPILE.md rounds 0, 5, 6
# --------------------------------------------------------------------------

def _install_mode_stack(C, put) -> None:
    """`_push_on_torch_dispatch_stack` and friends, as a real stack.

    `_len_torch_dispatch_stack` is presently the constant `0` in the bootstrap's
    `_DISCOVERED_RETURNS` table, with the comment "nothing pushes onto it here".
    Under `torch.export` something does: `FakeTensorMode` and
    `ProxyTorchDispatchMode` both push, and `torch/utils/_python_dispatch.py`
    reads the length back to decide whether a mode is active.  A stack that
    always answered zero would make `with FakeTensorMode():` a block that
    entered and changed nothing -- the same shape of silent no-op
    `docs/COMPILE.md` §5 refuses for `torch.compile`.

    Upstream splits the stack in two: *infra* modes (FAKE, PROXY, FUNCTIONAL)
    live in slots keyed by `_TorchDispatchModeKey` and are not part of the
    ordinary stack, while user modes are appended.  `mode._mode_key` is what
    tells the two apart, and this reproduces that split rather than flattening
    it, because `_get_dispatch_mode(key)` has no meaning otherwise.
    """

    def _push_on_torch_dispatch_stack(mode):
        key = getattr(mode, "_mode_key", None)
        if key is not None:
            if _TLS.infra_modes.get(key) is not None:
                raise RuntimeError(
                    f"torch dispatch mode for {key} is already set"
                )
            _TLS.infra_modes[key] = mode
        else:
            _TLS.mode_stack.append(mode)

    def _pop_torch_dispatch_stack(mode_key=None):
        if mode_key is not None:
            popped = _TLS.infra_modes.pop(mode_key, None)
            if popped is None:
                raise AssertionError(
                    f"no torch dispatch mode set for {mode_key}"
                )
            return popped
        if not _TLS.mode_stack:
            raise AssertionError("torch dispatch mode stack is empty")
        return _TLS.mode_stack.pop()

    def _len_torch_dispatch_stack():
        return len(_TLS.mode_stack)

    def _get_dispatch_stack_at(idx):
        return _TLS.mode_stack[idx]

    def _set_dispatch_mode(mode):
        key = getattr(mode, "_mode_key", None)
        if key is None:
            raise AssertionError(
                "_set_dispatch_mode is for infra modes; this one has no _mode_key"
            )
        if _TLS.infra_modes.get(key) is not None:
            raise RuntimeError(f"torch dispatch mode for {key} is already set")
        _TLS.infra_modes[key] = mode

    def _get_dispatch_mode(mode_key):
        return _TLS.infra_modes.get(mode_key)

    def _unset_dispatch_mode(mode_key):
        return _TLS.infra_modes.pop(mode_key, None)

    def _only_lift_cpu_tensors():
        return _TLS.only_lift_cpu_tensors

    def _set_only_lift_cpu_tensors(value):
        _TLS.only_lift_cpu_tensors = bool(value)

    def _ensureCUDADeviceGuardSet():
        # Upstream primes a CUDA device guard so later lifts land on the right
        # device.  There is no CUDA here (`torch._C._has_cuda` is `False`, and
        # the bootstrap's build-flag table says why that is a fact rather than a
        # stand-in), so there is nothing to prime.  Doing nothing is the
        # implementation, not a no-op standing in for one.
        return None

    put(C, "_push_on_torch_dispatch_stack", _push_on_torch_dispatch_stack)
    put(C, "_pop_torch_dispatch_stack", _pop_torch_dispatch_stack)
    put(C, "_len_torch_dispatch_stack", _len_torch_dispatch_stack)
    put(C, "_get_dispatch_stack_at", _get_dispatch_stack_at)
    put(C, "_set_dispatch_mode", _set_dispatch_mode)
    put(C, "_get_dispatch_mode", _get_dispatch_mode)
    put(C, "_unset_dispatch_mode", _unset_dispatch_mode)
    put(C, "_only_lift_cpu_tensors", _only_lift_cpu_tensors)
    put(C, "_set_only_lift_cpu_tensors", _set_only_lift_cpu_tensors)
    put(C, "_ensureCUDADeviceGuardSet", _ensureCUDADeviceGuardSet)


# --------------------------------------------------------------------------
# 3. TensorBase predicates -- COMPILE.md rounds 7, 8, 12, 13
# --------------------------------------------------------------------------

def _is_definitely_a_view(t) -> bool:
    """A *sound positive* view detector, built from the storage model this shim has.

    Measured against upstream on the same three tensors (`docs/EXPORT.md` §3.2):
    `storage_offset`, `stride`, `numel` and `untyped_storage().nbytes()` agree
    exactly between this shim and upstream for `x`, `x[1:, 1:]` and `x.t()`.
    So three signals each *prove* a view:

    * a non-zero `storage_offset` -- the tensor starts inside someone else's
      buffer;
    * a footprint smaller than the storage -- it covers part of a buffer;
    * non-contiguous strides -- the layout was rearranged over a buffer that
      was laid out for something else.

    **What it misses, said plainly:** a view that covers the whole storage
    contiguously -- `x.view(12)`, `x[:]`, `x.reshape(3, 4)` on a contiguous `x`
    -- is bit-for-bit indistinguishable from the base under every signal this
    shim exposes.  Upstream answers `True` there because `TensorImpl` carries a
    base pointer; nothing in `PyTensorBase` does (`rust/torch_c/src/tensor.rs`
    has storage identity via `storage.rs::origin`, but no base *tensor*).  This
    returns `False` for that case, and that is the one wrong answer in the pair.

    Making it right is a Rust change -- a base slot on `PyTensorBase` -- and it
    is outside this document's territory.  `docs/EXPORT.md` §4.2 records it as
    the gap rather than papering it.
    """
    try:
        if t.storage_offset() != 0:
            return True
        if not t.is_contiguous():
            return True
        nbytes = t.untyped_storage().nbytes()
    except Exception:
        # A meta, quantised or vulkan tensor has no storage to ask about
        # (`tensor.rs::no_dense_storage`, `no_host_storage`).  It also has no
        # base, so "not a detected view" is the right answer, not a dodge.
        return False
    return t.numel() * t.element_size() != nbytes


def _install_tensor_predicates(torch_module, C, put) -> None:
    """`_is_view`, `_base`, `is_mkldnn`, `is_inference`, `is_conj`.

    Three of the five are `False` *as a fact about this build*, not as a
    placeholder:

    * `is_mkldnn` -- the bootstrap's build-flag table already answers
      `_has_mkldnn` `False`; a tensor cannot be in a layout the build does not
      have.
    * `is_inference` -- inference mode is an autograd TLS state
      (`InferenceMode`), and this shim has no autograd TLS to be in.
    * `is_conj` -- the conjugate bit is a dispatch-key bit on `TensorImpl`;
      candle has no such bit and no `aten::conj` view op reaches it.

    `_is_view` and `_base` are the pair that is **not** a constant, and they are
    the one place on this path where the shim has to say something it cannot
    fully know.  Views here are real -- `docs/VIEWS.md` §6 made in-place ops
    write through a layout into shared storage -- so `False` is not free.  The
    arrangement:

    * `_is_view()` answers `True` when `_is_definitely_a_view` proves it, and
      `False` otherwise, missing exactly the full-coverage contiguous case.
    * `_base` **refuses by name** for a tensor that `_is_view()` called `True`,
      because there is no base object to return and `None` there would be read
      as "not a view" by the very caller that just asked.  For everything else
      it is `None`, which is the fact.

    So a module exported with a sliced input fails loudly at the tensor that
    caused it, rather than producing a graph whose inputs quietly lost their
    aliasing.  `docs/COMPILE.md` §5 refuses the same shape of silence for
    `torch.compile`.
    """
    TensorBase = C.TensorBase

    def _is_view(self):
        return _is_definitely_a_view(self)

    def _base_getter(self):
        if _is_definitely_a_view(self):
            raise NotImplementedError(
                "not implemented in torch._C shim: TensorBase._base. This "
                "tensor is a view (non-zero storage offset, non-contiguous "
                "strides, or a footprint smaller than its storage), and the "
                "shim's PyTensorBase carries no base tensor to return. "
                "Returning None here would tell the caller it is not a view, "
                "one line after _is_view() told it that it is."
            )
        return None

    def is_inference(self):
        return False

    def is_conj(self):
        return False

    put(TensorBase, "_is_view", _is_view, "TensorBase._is_view")
    put(TensorBase, "_base", property(_base_getter), "TensorBase._base")
    put(TensorBase, "is_inference", is_inference, "TensorBase.is_inference")
    put(TensorBase, "is_conj", is_conj, "TensorBase.is_conj")
    put(TensorBase, "is_mkldnn", property(lambda self: False),
        "TensorBase.is_mkldnn")


# --------------------------------------------------------------------------
# 4. functorch predicates -- COMPILE.md rounds 9, 10, 11
# --------------------------------------------------------------------------

def _install_functorch(C, put) -> None:
    """`is_batchedtensor`, `is_legacy_batchedtensor`, `is_gradtrackingtensor`.

    All three ask "is this tensor wrapped by a functorch transform?".  There is
    no functorch interpreter stack in this shim -- `vmap`, `grad` and `jvp` are
    not implemented -- so no tensor can be one of these wrappers and `False` is
    the fact.  If a functorch layer is ever added, these three are where it
    announces itself, and they are named here so that addition is a change to a
    body rather than the discovery of a hole.
    """
    F = C._functorch

    put(F, "is_batchedtensor", lambda t: False, "_functorch.is_batchedtensor")
    put(F, "is_legacy_batchedtensor", lambda t: False,
        "_functorch.is_legacy_batchedtensor")
    put(F, "is_gradtrackingtensor", lambda t: False,
        "_functorch.is_gradtrackingtensor")
    put(F, "is_functorch_wrapped_tensor", lambda t: False,
        "_functorch.is_functorch_wrapped_tensor", only_if_stub=True)


# --------------------------------------------------------------------------
# 5. profiler -- COMPILE.md round 14
# --------------------------------------------------------------------------

class _GatheredFrames:
    """The opaque handle `torch._C._profiler.gather_traceback` returns.

    Opaque is the contract: `torch/utils/_traceback.py` never looks inside one.
    It stores the handle on a `CapturedTraceback` and later hands a *list* of
    handles to `symbolize_tracebacks`, which is where the frames become
    readable.  Splitting it that way is upstream's amortisation -- symbolising
    C++ frames is expensive and worth batching -- and reproducing the split,
    rather than returning formatted strings from `gather_traceback`, is what
    keeps `format_all`'s batch path working.
    """

    __slots__ = ("frames",)

    def __init__(self, frames):
        #: innermost first, which is the order `_extract_symbolized_tb`
        #: assumes: it reverses, and it applies `skip` from the *front* to
        #: elide `CapturedTraceback.extract`'s own frame.
        self.frames = frames

    def __repr__(self):
        return f"<CapturedTraceback {len(self.frames)} frames>"


def _install_profiler(C, put) -> None:
    """`gather_traceback` and `symbolize_tracebacks`, to the shape of their caller.

    COMPILE.md's census reached `gather_traceback` at round 14 and no-opped it
    to `None`; the no-op survived because nothing symbolised it in that run.
    With the later rounds real, `torch/_logging/_internal.py:1510` does symbolise
    it, and that is what turned this from "return anything" into a pair with a
    contract:

        gather_traceback(python, script, cpp)  -> opaque handle
        symbolize_tracebacks([handle, ...])    -> [[{filename, line, name}, ...], ...]

    read out of `torch/utils/_traceback.py:180` and `:259`.  `line` is a line
    *number* -- it is passed as `FrameSummary`'s second positional argument --
    and getting that wrong is a `TypeError` several frames away from here,
    which is how it was found.

    `script` and `cpp` are accepted and ignored.  There is no TorchScript
    interpreter and no C++ stack worth naming in this shim, so there are no
    frames of those kinds to omit; Python frames are the whole traceback here
    rather than a subset of one.
    """

    def gather_traceback(python=True, script=False, cpp=False):
        if not python:
            return _GatheredFrames([])
        # `[:-1]` drops this function's own frame: upstream's is C++ and does
        # not appear in the result, and `CapturedTraceback.extract` passes
        # `skip=skip+1` counted from a stack that does not contain it.
        outermost_first = _traceback.extract_stack()[:-1]
        return _GatheredFrames([
            {"filename": f.filename, "line": f.lineno, "name": f.name}
            for f in reversed(outermost_first)
        ])

    def symbolize_tracebacks(to_symbolize):
        return [
            list(t.frames) if isinstance(t, _GatheredFrames) else []
            for t in to_symbolize
        ]

    put(C._profiler, "gather_traceback", gather_traceback,
        "_profiler.gather_traceback")
    put(C._profiler, "symbolize_tracebacks", symbolize_tracebacks,
        "_profiler.symbolize_tracebacks")


# --------------------------------------------------------------------------
# 6. dynamo's bool setter -- COMPILE.md round 4
# --------------------------------------------------------------------------

def _install_dynamo_bool(C, put) -> None:
    """`set_is_in_mode_without_ignore_compile_internals`.

    `docs/COMPILE.md` §1.2 identified this as a two-line bool setter at
    `dynamo/guards.cpp:134` that touches none of the frame-hook machinery.  It
    is on the export path because `torch/_dynamo/utils.py` toggles it around
    mode entry; nothing in this shim reads it back, so it is a cell.

    It lives under `torch._C._dynamo.guards`, which is *not* a reason to think
    this opens `torch.compile`.  `set_eval_frame`'s refusal is untouched and
    stays untouched -- see `docs/COMPILE.md` §5.1 for why that refusal must
    outlive any symbol-filling on this path.
    """
    guards = C._dynamo.guards
    state = {"value": False}

    def set_is_in_mode_without_ignore_compile_internals(value):
        state["value"] = bool(value)

    def is_in_mode_without_ignore_compile_internals():
        return state["value"]

    put(guards, "set_is_in_mode_without_ignore_compile_internals",
        set_is_in_mode_without_ignore_compile_internals,
        "_dynamo.guards.set_is_in_mode_without_ignore_compile_internals")
    put(guards, "is_in_mode_without_ignore_compile_internals",
        is_in_mode_without_ignore_compile_internals,
        "_dynamo.guards.is_in_mode_without_ignore_compile_internals")


# --------------------------------------------------------------------------
# 7. `_InferenceMode` -- past round 19, not in COMPILE.md's census
# --------------------------------------------------------------------------

def _install_inference_mode(C, put) -> None:
    """`torch._C._InferenceMode`, as a context manager that actually enters.

    Not in COMPILE.md's list of 18, and it could not have been: the census
    stopped at `meta_utils.py:1061`, and this is reached at `:1510`.  The
    bootstrap leaves it as a `_ShimMeta` synthesised class, which is
    constructible and has no `__enter__`, so `with torch.inference_mode(...)`
    fails with `AttributeError` rather than by name.  That is worth fixing
    independently of export: an `AttributeError` on a dunder points at
    `grad_mode.py` and says nothing about the shim.

    What it does here is track a flag and nothing else.  Upstream's
    `InferenceMode` switches a dispatch-key bit that makes new tensors skip
    autograd bookkeeping and become invalid outside the block; this shim has no
    autograd bookkeeping to skip (`docs/AUTOGRAD.md`) and no version counter to
    invalidate, so entering and leaving is the entire behaviour.  It is
    recorded rather than discarded so `_is_inference_mode_enabled()` can answer
    from the same place instead of guessing.

    `TensorBase.is_inference` stays `False` even inside the block, and that is
    deliberate, not an oversight: upstream's per-tensor flag records that a
    tensor *was created* in inference mode and is therefore unsafe to use
    outside it.  No tensor here is unsafe in that way, so `True` would be a
    claim about lifetime that nothing enforces.
    """

    class _InferenceMode:
        __module__ = "torch._C"
        __slots__ = ("mode", "_saved")

        def __init__(self, mode=True):
            self.mode = bool(mode)
            self._saved = None

        def __enter__(self):
            self._saved = _read()
            _write(self.mode)
            return self

        def __exit__(self, *exc):
            _write(self._saved)
            return False

    # One source of truth, and it is the bootstrap's.  `bootstrap.py`'s
    # `_install_grad_mode` now owns the flag, because `torch.is_inference_mode_enabled`
    # is harvested off `_VariableFunctions` before this module can run and
    # `fake_tensor.py:1801` calls it on every cached dispatch.  Writing through
    # to it -- rather than keeping a second flag on `_TLS` -- is what stops the
    # guard and the predicate from answering differently, which is
    # docs/EXPORT.md §2.2's failure with the operands swapped.
    #
    # The `_TLS` fallback is not dead code: it is what runs against a bootstrap
    # that predates the setter, and it keeps this module importable there.
    def _read():
        fn = getattr(C, "is_inference_mode_enabled", None)
        if fn is not None:
            return bool(fn())
        return getattr(_TLS, "inference_mode", False)

    def _write(value):
        _TLS.inference_mode = value
        setter = getattr(C, "_set_inference_mode_enabled", None)
        if setter is not None:
            setter(bool(value))

    put(C, "_InferenceMode", _InferenceMode)
    put(C, "_is_inference_mode_enabled", _read)


# --------------------------------------------------------------------------
# 8. The RAII guard family -- also past round 19
# --------------------------------------------------------------------------

#: `name -> what upstream's guard switches`.  Every one of these is
#: constructible today (the bootstrap synthesises the *type*) and none of them
#: has `__enter__`, so each fails as
#: ``TypeError: '_X' object does not support the context manager protocol`` --
#: a message that names the class but not the shim, several frames from the
#: `with` that wanted it.  They are one family and they are fixed as one.
#:
#: **None of them changes what this shim dispatches**, and that is the point of
#: recording the flag rather than discarding it: `_is_guard_active(name)` lets a
#: caller -- or a test -- ask what was entered, so the day `_aten_dispatch`
#: learns to consult one of these, the state is already there to consult.
_RAII_GUARDS = {
    "_DisableTorchDispatch":
        "suppresses the torch-dispatch mode stack for the block",
    "_DisableFuncTorch":
        "pops the functorch interpreter stack; there is none here",
    "_DisableAutocast":
        "turns off autocast; this build has no autocast dispatch key",
    "_AutoDispatchBelowAutograd":
        "excludes the autograd keys so a call lands on the backend directly",
    "_RestorePythonTLSSnapshot":
        "restores a saved dispatcher TLS snapshot",
    "_DisablePythonDispatcher": "turns off the Python dispatcher",
    "_EnablePythonDispatcher": "turns on the Python dispatcher",
    "_EnablePreDispatch": "routes through the PreDispatch key",
    "_PreserveDispatchKeyGuard": "saves and restores the whole TLS key state",
    "_SetExcludeDispatchKeyGuard": "sets one key's excluded bit for the block",
}


def _is_guard_active(name) -> bool:
    """Is a named RAII guard currently entered on this thread?"""
    return getattr(_TLS, "guards", {}).get(name, 0) > 0


def _install_raii_guards(C, put) -> None:
    """Give the guard family a real `__enter__`/`__exit__`.

    Read the honesty limit here carefully, because it is the one that matters
    on this path and `docs/EXPORT.md` §4.1 is about it.  `_DisableTorchDispatch`
    is the guard `torch/_subclasses/fake_tensor.py:502` uses to build a meta
    tensor *without re-entering the fake mode*.  Entering and leaving a counter
    is a correct implementation **only because this shim never consults the mode
    stack in the first place** -- `_aten_dispatch` records ops after the fact
    (`aten.rs`, the capture hook) and never asks a Python mode to handle one.
    There is therefore nothing for `no_dispatch()` to suppress.

    That is not a happy accident, it is the wall: see `docs/EXPORT.md` §4.  When
    `_aten_dispatch` learns to consult the stack, this guard stops being a
    counter and starts being load-bearing, and the counter is here so that
    change is a body to fill rather than a hole to find.
    """

    def _make(name, why):
        class _Guard:
            __module__ = "torch._C"
            __slots__ = ()
            __doc__ = f"torch._C.{name} -- upstream {why}."

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                counts = getattr(_TLS, "guards", None)
                if counts is None:
                    counts = _TLS.guards = {}
                counts[name] = counts.get(name, 0) + 1
                # `_DisableTorchDispatch` is `no_dispatch()`, and it is the ONE
                # guard in this family that the dispatcher door has to see.  The
                # counter above is this module's own bookkeeping; the write
                # below is the one that actually suppresses, and it lives in
                # `torch._C` because `aten.rs` reads it there.  See
                # `bootstrap.py::_install_dispatch_suppression` and
                # docs/EXPORT4.md §5.
                #
                # The others stay counters on purpose: `_DisableFuncTorch` and
                # friends name subsystems this shim does not have, and making
                # them suppress dispatch would be a guess about what they mean.
                if name == "_DisableTorchDispatch":
                    push = getattr(C, "_shim_push_dispatch_suppression", None)
                    if push is not None:
                        push()
                return self

            def __exit__(self, *exc):
                _TLS.guards[name] -= 1
                if name == "_DisableTorchDispatch":
                    pop = getattr(C, "_shim_pop_dispatch_suppression", None)
                    if pop is not None:
                        pop()
                return False

        _Guard.__name__ = name
        _Guard.__qualname__ = name
        return _Guard

    for name, why in _RAII_GUARDS.items():
        put(C, name, _make(name, why))
