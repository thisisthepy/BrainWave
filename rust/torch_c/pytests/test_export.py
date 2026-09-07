"""`torch.export` -- what the front end needs, and the wall past the census.

`docs/graph/COMPILE.md` §3 censused the `torch.export` blockers with crude no-ops and
recommended spending the `torch.compile` effort here, because none of the 18
names it found is abi3-impossible.  `docs/graph/EXPORT.md` re-derived that census with
**real implementations** (`torchnative.export.upstream`) instead of no-ops and
found two things the no-op census could not have seen:

1. the 18 are real, they are ordinary binding surface, and past them lie 11 more
   of the same kind plus two that are not binding surface at all;
2. **`torch.fx.Graph()` cannot be constructed in this shim**, and **no
   `TorchDispatchMode` ever sees an operator**.  Every graph front end upstream
   has -- export, `make_fx`, Dynamo -- is built on those two, so the census was
   never the road.  `docs/graph/EXPORT.md` §4.

The tests here hold both halves down.  Most of them check that
`torchnative.export.upstream` does what it says: real `DispatchKeySet`s, a mode
stack that counts, guards that enter and leave.  Two of them are different in
kind and are the reason this file exists --
`test_a_graph_front_end_is_not_offered_while_modes_are_not_consulted` and
`test_capture_is_the_only_working_front_end_and_records_the_module_it_ran`.
They are written so that **closing the gap makes them demand more, not less**:
the moment `torch.fx.Graph()` starts working, the first stops being satisfied by
a refusal and starts requiring that a traced graph agree with `capture.rs`.

Everything runs in one subprocess with the vendored tree on `PYTHONPATH`, the
same shape `test_shim.py`'s decompose-road tests use, because
`torchnative.export.upstream` imports `torch` and this suite's `torch._C` is
standalone.
"""

import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


_SCRIPT = r"""
import json, sys, traceback
import torch
from torch import nn

out = {}
out["is_shim"] = hasattr(torch._C, "_aten_implemented")

from torchnative.export import upstream

# --- the census names, BEFORE anything is installed -------------------------
#
# docs/graph/EXPORT5.md §7 moved every one of these into `bootstrap.py`, so the
# question this probe asks changed direction: it used to be "did install()
# replace the placeholders", and it is now "**are they already implemented
# before anyone calls anything**". That is the stronger claim and it is the one
# the hand-off is for, so it is measured first, before `install()` is touched.
def _stubbed_now(owner, name):
    return upstream._is_stub(getattr(owner, name, None))

out["stubbed_before_any_install"] = sorted(
    n for n in upstream.installed_names() if _stubbed_now(torch._C, n)
)
out["census_stubbed_before_any_install"] = sorted(
    n for n in (
        "_unset_dispatch_mode", "_only_lift_cpu_tensors",
        "_set_only_lift_cpu_tensors", "_ensureCUDADeviceGuardSet",
        "_push_on_torch_dispatch_stack", "_pop_torch_dispatch_stack",
        "_dispatch_tls_is_dispatch_key_included",
        "_functionalization_reapply_views_tls",
        "_dispatch_tls_local_exclude_set", "_dispatch_tls_local_include_set",
    ) if _stubbed_now(torch._C, n)
)
_owners = {
    "TensorBase._is_view": (torch._C.TensorBase, "_is_view"),
    "TensorBase.is_mkldnn": (torch._C.TensorBase, "is_mkldnn"),
    "TensorBase.is_inference": (torch._C.TensorBase, "is_inference"),
    "TensorBase.is_conj": (torch._C.TensorBase, "is_conj"),
    "_functorch.is_batchedtensor": (torch._C._functorch, "is_batchedtensor"),
    "_functorch.is_legacy_batchedtensor": (torch._C._functorch, "is_legacy_batchedtensor"),
    "_functorch.is_gradtrackingtensor": (torch._C._functorch, "is_gradtrackingtensor"),
    "_profiler.gather_traceback": (torch._C._profiler, "gather_traceback"),
    "_dynamo.guards.set_is_in_mode_without_ignore_compile_internals": (
        torch._C._dynamo.guards, "set_is_in_mode_without_ignore_compile_internals"),
}
out["census_stubbed_before_any_install"] += sorted(
    q for q, (owner, leaf) in _owners.items() if _stubbed_now(owner, leaf)
)
# Every name the probe actually interrogated, so a test can assert that its
# census is covered rather than assuming it was.
out["names_checked"] = sorted(
    set(upstream.installed_names())
    | {"_unset_dispatch_mode", "_only_lift_cpu_tensors",
       "_set_only_lift_cpu_tensors", "_ensureCUDADeviceGuardSet",
       "_push_on_torch_dispatch_stack", "_pop_torch_dispatch_stack",
       "_dispatch_tls_is_dispatch_key_included",
       "_functionalization_reapply_views_tls",
       "_dispatch_tls_local_exclude_set", "_dispatch_tls_local_include_set"}
    | set(_owners)
)

# --- what install() reports -------------------------------------------------
report = upstream.install()
out["replaced"] = sorted(report.replaced)
out["overridden"] = sorted(report.overridden)
out["rebound"] = report.rebound

# Idempotence: a second install() must replace nothing, because the first one
# already did.  (`overridden` is expected to be non-empty the second time --
# it is displacing its own objects -- so the claim is about `replaced`.)
again = upstream.install()
out["second_replaced"] = sorted(again.replaced)

# --- the names are no longer stubs -----------------------------------------
def stubbed(owner, name):
    o = getattr(owner, name, None)
    return upstream._is_stub(o)

out["still_stubbed"] = sorted(
    n for n in upstream.installed_names() if stubbed(torch._C, n)
)

# --- dispatcher TLS ---------------------------------------------------------
KeySet = torch._C.DispatchKeySet
Key = torch._C.DispatchKey
inc, exc = torch._C._dispatch_tls_local_include_set(), torch._C._dispatch_tls_local_exclude_set()
out["tls_types_are_keysets"] = isinstance(inc, KeySet) and isinstance(exc, KeySet)
# The precise line COMPILE.md round 19 died on: meta_utils.py:1061 calls
# `.has(...)` on this.
out["exclude_set_answers_has"] = exc.has(Key.ADInplaceOrView)
out["tls_starts_empty"] = (len(inc) == 0 and len(exc) == 0)

with torch._C._ExcludeDispatchKeyGuard(KeySet(Key.ADInplaceOrView)):
    out["exclude_guard_inside"] = torch._C._dispatch_tls_is_dispatch_key_excluded(
        Key.ADInplaceOrView
    )
out["exclude_guard_after"] = torch._C._dispatch_tls_is_dispatch_key_excluded(
    Key.ADInplaceOrView
)

with torch._C._ForceDispatchKeyGuard(KeySet(Key.CPU), KeySet(Key.AutogradCPU)):
    out["force_guard_included"] = torch._C._dispatch_tls_is_dispatch_key_included(Key.CPU)
    out["force_guard_excluded"] = torch._C._dispatch_tls_is_dispatch_key_excluded(
        Key.AutogradCPU
    )
out["force_guard_restored"] = (
    len(torch._C._dispatch_tls_local_include_set()) == 0
    and len(torch._C._dispatch_tls_local_exclude_set()) == 0
)

# --- the mode stack counts --------------------------------------------------
from torch.utils._python_dispatch import TorchDispatchMode

class _Plain(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))

out["stack_len_before"] = torch._C._len_torch_dispatch_stack()
with _Plain():
    out["stack_len_inside"] = torch._C._len_torch_dispatch_stack()
    with _Plain():
        out["stack_len_nested"] = torch._C._len_torch_dispatch_stack()
out["stack_len_after"] = torch._C._len_torch_dispatch_stack()

# Infra modes go to a keyed slot, not onto the ordinary stack.
class _Infra(TorchDispatchMode):
    _mode_key = torch._C._TorchDispatchModeKey.PROXY
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))

infra = _Infra()
torch._C._push_on_torch_dispatch_stack(infra)
out["infra_not_on_ordinary_stack"] = torch._C._len_torch_dispatch_stack() == 0
out["infra_in_slot"] = (
    torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is infra
)
out["infra_unset_returns_it"] = (
    torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is infra
)
out["infra_slot_empty_after"] = (
    torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is None
)

# --- context-manager family -------------------------------------------------
entered = []
for name in sorted(upstream._RAII_GUARDS):
    guard = getattr(torch._C, name)
    try:
        with guard(True):
            entered.append(name if upstream._is_guard_active(name) else f"{name}:not-active")
    except Exception as e:
        entered.append(f"{name}:{type(e).__name__}")
out["guards_entered"] = entered
out["guards_all_left"] = [
    n for n in sorted(upstream._RAII_GUARDS) if upstream._is_guard_active(n)
]

with torch.inference_mode(True):
    out["inference_mode_entered"] = torch._C._is_inference_mode_enabled()
out["inference_mode_left"] = torch._C._is_inference_mode_enabled()

# --- the view detector ------------------------------------------------------
base = torch.arange(12.).reshape(3, 4)
sliced = base[1:, 1:]
transposed = base.t()
out["base_is_not_a_detected_view"] = base._is_view()
out["slice_is_a_detected_view"] = sliced._is_view()
out["transpose_is_a_detected_view"] = transposed._is_view()
out["base_base_is_none"] = base._base is None
try:
    sliced._base
except NotImplementedError as e:
    out["view_base_refuses"] = "TensorBase._base" in str(e)
else:
    out["view_base_refuses"] = False

out["is_conj"] = base.is_conj()
out["is_inference"] = base.is_inference()
out["is_mkldnn"] = base.is_mkldnn

# --- profiler traceback -----------------------------------------------------
from torch.utils._traceback import CapturedTraceback
tb = CapturedTraceback.extract(skip=0)
frames = tb.summary()
out["traceback_nonempty"] = len(frames) > 0
out["traceback_innermost_is_this_file"] = frames[-1].name in ("<module>",)
out["traceback_formats"] = all(isinstance(s, str) for s in tb.format())

# --- THE WALL ---------------------------------------------------------------
# Does any TorchDispatchMode see an operator?
seen = []
class _Log(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        seen.append(str(func))
        return func(*args, **(kwargs or {}))

x = torch.ones(3)
with _Log():
    y = (x * 2 + 1).relu()
out["ops_seen_by_mode"] = seen

# Can an fx graph be built at all?
import torch.fx
try:
    g = torch.fx.Graph()
    a = g.placeholder("x")
    n = g.call_function(torch.ops.aten.mul.Tensor, (a, 2))
    g.output(n)
    out["fx_graph_builds"] = True
    out["fx_graph_refusal"] = None
except BaseException as e:
    out["fx_graph_builds"] = False
    out["fx_graph_refusal"] = f"{type(e).__name__}: {str(e)[:200]}"

# Does torch.export produce anything?
class M(nn.Module):
    def forward(self, t):
        return (t * 2 + 1).relu()

try:
    ep = torch.export.export(M(), (torch.ones(3),))
    out["export_returns"] = True
    out["export_ops"] = sorted(
        str(nd.target) for nd in ep.graph.nodes if nd.op == "call_function"
    )
    out["export_refusal"] = None
except BaseException as e:
    out["export_returns"] = False
    out["export_ops"] = []
    out["export_refusal"] = f"{type(e).__name__}: {str(e)[:200]}"

# --- the oracle: what capture.rs records for the same module ----------------
m = M()
xi = torch.ones(3)
torch._C._capture_begin([xi])
res = m(xi)
trace = torch._C._capture_end(res)
out["capture_ops"] = [nd["op"] for nd in trace.nodes]

json.dump(out, sys.stdout)
"""


def _fixture(_cache={}):
    if "v" in _cache:
        return _cache["v"]
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"export subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    _cache["v"] = json.loads(proc.stdout)
    return _cache["v"]


def _available():
    """Same silent skip as the decompose-road tests, for the same reason.

    These need `torchnative/src/main/torch/_C.abi3.so`, which
    `vendor/install_shim.sh` places and `pytests/run.sh` deliberately does not.
    """
    return os.path.isfile(_VENDOR_SHIM)


# ---------------------------------------------------------------------------
# 1. `torchnative.export.upstream` installs implementations, not placeholders
# ---------------------------------------------------------------------------

def test_install_leaves_no_stub_among_the_names_it_claims():
    if not _available():
        return
    r = _fixture()
    assert r["is_shim"], "subprocess did not get the shim-backed torch"
    assert r["still_stubbed"] == [], r["still_stubbed"]


def test_install_is_idempotent_in_the_only_sense_that_matters():
    """A second `install()` must find nothing left to replace.

    Since docs/graph/EXPORT5.md §7 this holds for a stronger reason than it used to:
    the **first** call finds nothing to replace either, because the names are
    installed by `bootstrap.py` before anything imports them. The assertion is
    unchanged; what changed underneath it is that it can no longer pass by
    `install()` having done the work.
    """
    if not _available():
        return
    r = _fixture()
    assert r["second_replaced"] == [], r["second_replaced"]
    assert r["replaced"] == [], (
        "install() replaced something, so a census name was still a placeholder "
        "at import time -- docs/graph/EXPORT5.md §7's hand-off is incomplete: "
        f"{r['replaced']}"
    )


def test_the_census_names_are_implemented_by_the_bootstrap_with_no_install_call():
    """The re-derived census, as a test rather than as prose -- **inverted.**

    This used to assert that every name in `docs/graph/EXPORT.md` §2's "was a raising
    stub" column appeared in `install()`'s `replaced` list, which was the right
    claim while `torchnative/export/upstream.py` installed them at runtime.

    `docs/graph/EXPORT5.md` §7 moved them into `bootstrap.py`, so `replaced` is now
    empty and that assertion would pass vacuously if it were merely relaxed.
    The claim is therefore turned around into the stronger one the hand-off is
    actually for: **each census name is already an implementation before
    anything calls `install()`.** It fails if any of them regresses to a
    placeholder, which is the same protection pointed the other way.

    Why this is not a weakening: the old test could be satisfied by a name that
    was a stub and then got patched; this one cannot be satisfied by a stub at
    all.
    """
    if not _available():
        return
    r = _fixture()
    assert r["census_stubbed_before_any_install"] == [], (
        "these census names were still placeholders at import time, before any "
        "install() call -- the bootstrap hand-off (docs/graph/EXPORT5.md §7) does not "
        f"cover them: {r['census_stubbed_before_any_install']}"
    )
    census = {
        "_unset_dispatch_mode",
        "_only_lift_cpu_tensors",
        "_set_only_lift_cpu_tensors",
        "_ensureCUDADeviceGuardSet",
        "_push_on_torch_dispatch_stack",
        "_pop_torch_dispatch_stack",
        "_dispatch_tls_is_dispatch_key_included",
        "_functionalization_reapply_views_tls",
        "_dispatch_tls_local_exclude_set",
        "_dispatch_tls_local_include_set",
        "TensorBase._is_view",
        "TensorBase.is_mkldnn",
        "TensorBase.is_inference",
        "TensorBase.is_conj",
        "_functorch.is_batchedtensor",
        "_functorch.is_legacy_batchedtensor",
        "_functorch.is_gradtrackingtensor",
        "_profiler.gather_traceback",
        "_dynamo.guards.set_is_in_mode_without_ignore_compile_internals",
    }
    # `census` is kept as the explicit list of the twenty names docs/graph/EXPORT.md
    # §1 enumerated, so that this test still names *what* it is checking rather
    # than deferring entirely to `installed_names()`. Every one of them must be
    # covered by the probe above, and none of them may be a placeholder.
    uncovered = sorted(census - set(upstream_names_checked(r)))
    assert not uncovered, (
        "the probe did not check these census names at all, so their status is "
        f"unknown rather than good: {uncovered}"
    )


def upstream_names_checked(r):
    """Every name the probe asked `_is_stub` about, stubbed or not."""
    return set(r["names_checked"])


# ---------------------------------------------------------------------------
# 2. dispatcher TLS -- what round 19 actually needed
# ---------------------------------------------------------------------------

def test_the_tls_sets_are_real_keysets_and_answer_has():
    """COMPILE.md's round 19 was `'NoneType' has no attribute 'has'`.

    That is the whole of it: `_dispatch_tls_local_exclude_set()` has to return a
    `DispatchKeySet`, and `_install_dispatch_keys` in `bootstrap.py` already
    builds one.  This pins the exact call `meta_utils.py:1061` makes.
    """
    if not _available():
        return
    r = _fixture()
    assert r["tls_types_are_keysets"]
    assert r["tls_starts_empty"]
    assert r["exclude_set_answers_has"] is False


def test_the_dispatch_key_guards_set_and_restore():
    if not _available():
        return
    r = _fixture()
    assert r["exclude_guard_inside"] is True
    assert r["exclude_guard_after"] is False
    assert r["force_guard_included"] is True
    assert r["force_guard_excluded"] is True
    assert r["force_guard_restored"] is True


# ---------------------------------------------------------------------------
# 3. the mode stack
# ---------------------------------------------------------------------------

def test_the_dispatch_mode_stack_counts_instead_of_answering_zero():
    """`_len_torch_dispatch_stack` was the constant `0` in `_DISCOVERED_RETURNS`.

    A constant zero is not a stub -- it is an ordinary function with a comment
    saying nothing pushes onto the stack -- and it is nonetheless the shape of
    silence `docs/graph/COMPILE.md` §5 refuses: `with SomeMode():` would enter, be
    reported as absent, and change nothing.
    """
    if not _available():
        return
    r = _fixture()
    assert r["stack_len_before"] == 0
    assert r["stack_len_inside"] == 1
    assert r["stack_len_nested"] == 2
    assert r["stack_len_after"] == 0


def test_infra_modes_go_to_a_keyed_slot_not_onto_the_stack():
    """Upstream splits the two, and flattening them would make
    `_get_dispatch_mode(PROXY)` meaningless -- which matters because
    `_detect_infra_mode` in `torch/utils/_python_dispatch.py` asserts that at
    most one of the pre-dispatch and post-dispatch slots is occupied."""
    if not _available():
        return
    r = _fixture()
    assert r["infra_not_on_ordinary_stack"]
    assert r["infra_in_slot"]
    assert r["infra_unset_returns_it"]
    assert r["infra_slot_empty_after"]


# ---------------------------------------------------------------------------
# 4. the RAII guard family and inference mode
# ---------------------------------------------------------------------------

def test_every_raii_guard_enters_and_leaves():
    """The family failed as `TypeError: '_X' object does not support the
    context manager protocol` -- a message naming the class and not the shim,
    raised at the `with` and not at the thing that is missing."""
    if not _available():
        return
    r = _fixture()
    bad = [e for e in r["guards_entered"] if ":" in e]
    assert not bad, bad
    assert r["guards_all_left"] == [], r["guards_all_left"]


def test_inference_mode_is_a_context_manager():
    if not _available():
        return
    r = _fixture()
    assert r["inference_mode_entered"] is True
    assert r["inference_mode_left"] is False


# ---------------------------------------------------------------------------
# 5. the view detector, and the refusal that keeps it honest
# ---------------------------------------------------------------------------

def test_the_view_detector_finds_the_views_this_storage_model_can_prove():
    """`docs/graph/EXPORT.md` §3.2.  `x[1:, 1:]` is proven by a non-zero storage
    offset, `x.t()` by non-contiguous strides.  Both agree with upstream on the
    same tensors, measured side by side."""
    if not _available():
        return
    r = _fixture()
    assert r["base_is_not_a_detected_view"] is False
    assert r["slice_is_a_detected_view"] is True
    assert r["transpose_is_a_detected_view"] is True


def test_base_refuses_for_a_detected_view_rather_than_answering_none():
    """The one that would have been a silent wrong answer.

    `None` from `_base` means "not a view" to every caller in the vendored tree
    -- `meta_utils.py:2246` reads it exactly that way, one line after
    `_is_view()` told it the opposite.  There is no base tensor to return
    (`PyTensorBase` carries none), so the only two options are a refusal and a
    lie.
    """
    if not _available():
        return
    r = _fixture()
    assert r["base_base_is_none"] is True
    assert r["view_base_refuses"] is True


def test_the_three_constant_predicates_are_false_for_a_build_reason():
    if not _available():
        return
    r = _fixture()
    assert r["is_conj"] is False
    assert r["is_inference"] is False
    assert r["is_mkldnn"] is False


# ---------------------------------------------------------------------------
# 6. the profiler pair, to the shape its caller needs
# ---------------------------------------------------------------------------

def test_gather_traceback_and_symbolize_round_trip_to_a_stack_summary():
    """`torch/utils/_traceback.py:259` builds `FrameSummary(f["filename"],
    f["line"], f["name"])`, so `line` is a line *number* and the frames come
    back innermost-first.  Getting either wrong is a `TypeError` or a reversed
    stack several frames from here, which is how both were found."""
    if not _available():
        return
    r = _fixture()
    assert r["traceback_nonempty"]
    assert r["traceback_formats"]


# ---------------------------------------------------------------------------
# 7. the wall -- and the shape that keeps this from going green by absence
# ---------------------------------------------------------------------------

def test_a_graph_front_end_is_not_offered_while_modes_are_not_consulted():
    """**The load-bearing test in this file.**

    `torch.export`, `make_fx` and Dynamo all build their graph from
    `__torch_dispatch__` callbacks on a mode.  `_aten_dispatch` never consults
    the mode stack (`aten.rs`'s capture hook runs *after* the kernel and cannot
    replace a result), so a mode sees nothing.  A front end that ran anyway
    would return a graph with the right placeholders, the right output, and
    **no operators** -- and that graph would look like a graph.

    So the invariant is not "export works" and not "export fails".  It is:

        if no mode sees an operator, no graph front end may return a graph.

    Today both halves hold by refusal -- `torch.fx.Graph()` itself cannot be
    constructed, which is upstream of every front end.  The moment either half
    changes this test demands the other: make modes work and `export` may
    return; make `export` return without making modes work and this fails,
    naming the empty graph.
    """
    if not _available():
        return
    r = _fixture()
    modes_work = len(r["ops_seen_by_mode"]) > 0
    if modes_work:
        # The gap closed.  Then a returned graph must contain operators.
        if r["export_returns"]:
            assert r["export_ops"], (
                "torch.export returned a graph with no call_function nodes "
                "while a mode did see operators -- that is the empty-graph "
                "failure this test exists for"
            )
        return
    assert not r["export_returns"], (
        "torch.export returned a graph, but no TorchDispatchMode saw a single "
        f"operator, so it cannot have recorded one. ops={r['export_ops']}"
    )
    assert not r["fx_graph_builds"], (
        "torch.fx.Graph() now builds while modes are still not consulted; the "
        "empty-graph risk in docs/graph/EXPORT.md §4 is live and this test needs the "
        "graph itself checked, not its absence"
    )
    assert r["fx_graph_refusal"] is not None

    # The hole `docs/bindings/BIND3.md` §7 found in this test, named rather than closed.
    #
    # Every assertion above reads False in *both* columns of the case that
    # matters. A round that installs EXPORT.md §3's census names makes
    # `TorchDispatchMode` **enter** -- reporting a dispatch-stack depth of 1
    # and silently returning eager results -- without making `export` return
    # or `fx.Graph` build. That is COMPILE.md §5's silent fallback one level
    # below the empty graph, and it passes everything above unchanged.
    #
    # It is recorded here and not asserted, and the reason is worth keeping:
    # the depth is **correct bookkeeping**. `_push_on_torch_dispatch_stack` is
    # a state cell and a `with` block did push one. Demanding zero would force
    # that cell to lie. What is missing is on the other side -- `_aten_dispatch`
    # never consults the stack, and `aten.rs`'s capture hook runs *after* the
    # kernel, so it could not replace a result even if it did.
    #
    # So the guard against this is EXPORT.md §6's ordering, not an assertion
    # here: the dispatcher entrance lands first, and only then the names. A
    # round that installs the names while `ops_seen_by_mode` is empty is doing
    # the thing this comment exists to warn it about.
    if r["stack_len_inside"] and not r["ops_seen_by_mode"]:
        print(
            "    NOTE: a TorchDispatchMode enters and reports depth "
            f"{r['stack_len_inside']} while seeing no operators -- "
            "docs/graph/EXPORT.md §6, the dispatcher entrance is not landed"
        )


def test_capture_is_the_only_working_front_end_and_records_the_module_it_ran():
    """`capture.rs` is the oracle `docs/graph/EXPORT.md` §5 compares against.

    Three ops for `(x * 2 + 1).relu()`, in order.  Upstream's `make_fx` on the
    same module records `aten.mul.Tensor`, `aten.add.Tensor`,
    `aten.relu.default` -- the same three operators, but **two of the three
    under a different overload**, because `capture.rs` records the `.Scalar`
    overload for a Python-number operand where upstream's dispatcher records
    `.Tensor`.  That disagreement is pinned here rather than described, because
    Core ATen and the Edge dialect are defined per *overload* and a lowering
    that assumed one spelling would miss the other.
    """
    if not _available():
        return
    r = _fixture()
    assert r["capture_ops"] == [
        "aten.mul.Scalar",
        "aten.add.Scalar",
        "aten.relu.default",
    ], r["capture_ops"]


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
