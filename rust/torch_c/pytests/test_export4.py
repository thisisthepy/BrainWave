"""`torch.export` -- six walls closed, and the seventh named.

`docs/EXPORT.md` §6 gave an ordering and predicted that the wall past the census
would be `aten.empty_strided`.  Steps 1 and 2 of that ordering landed (the
dispatcher consults the mode stack; `_NodeBase` builds), and this round
re-measured rather than assuming.  The prediction held -- `empty_strided` was
exactly where export stopped -- and closing it exposed five more, of which one
was a **defect `docs/EXPORT.md` had itself predicted in prose**:
`no_dispatch()` had stopped suppressing anything the moment the door learned to
consult the mode stack.  `docs/EXPORT4.md` is the measurement.

What these tests are for, and what they are careful not to be
--------------------------------------------------------------
`docs/EXPORT.md` §4.2 warned that an `ExportedProgram` containing **no
operators** would print, serialise, and look right.  This round moved export a
long way and did **not** reach a working `torch.export.export()`, so the risk
now is the mirror image of that one: a test suite that grows while the thing it
is about does not.  So:

* every test here is about a **named behaviour that changed**, not about the
  count of walls closed;
* `test_export_still_stops_and_it_stops_at_the_storage_handle` pins the CURRENT
  wall by name, and it is written to fail **both** if export regresses to an
  earlier wall and if it silently starts working -- the second because a wall
  that falls should be noticed and documented, not absorbed;
* the two predicate tables (`_should_allow_numbers_as_tensors`,
  `_dispatch_has_computed_kernel_for_dispatch_key`) are checked against **real
  upstream torch**, not against a transcription, so a torch upgrade that changes
  them fails here rather than drifting.

`DOCWATCH` markers in `docs/EXPORT4.md` use `ge`, never `eq` on a shared global
count.

Everything runs in subprocesses with the vendored tree on `PYTHONPATH`, the same
shape `test_export.py` uses and for the same reason, and skips silently when
`vendor/install_shim.sh` has not run.
"""

import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


# ---------------------------------------------------------------------------
# the shim-side probe
# ---------------------------------------------------------------------------

_SHIM_SCRIPT = r"""
import json, traceback
import torch
from torch import nn

out = {}
C = torch._C
out["is_shim"] = hasattr(C, "_aten_implemented")


def attempt(fn):
    "Return ('ok', value) or ('raise', type, message)."
    try:
        return {"status": "ok", "value": fn()}
    except BaseException as e:
        return {"status": "raise", "type": type(e).__name__, "message": str(e)}


# --- 1. aten.empty_strided --------------------------------------------------
out["empty_strided_in_implemented"] = "aten.empty_strided.default" in C._aten_implemented()
out["empty_strided_cpu"] = attempt(
    lambda: list(torch.empty_strided((2, 3), (3, 1)).shape))
out["empty_strided_meta"] = attempt(
    lambda: list(torch.empty_strided((2, 3), (3, 1), device="meta").shape))
out["empty_strided_zeroed"] = attempt(
    lambda: torch.empty_strided((2, 2), (2, 1)).sum().item())
# a size-1 axis makes its own stride unobservable, so it must NOT decide a refusal
out["empty_strided_size_one_axis"] = attempt(
    lambda: list(torch.empty_strided((1, 3), (99, 1)).shape))
out["empty_strided_noncontiguous"] = attempt(
    lambda: torch.empty_strided((2, 3), (1, 2)))
out["empty_strided_length_mismatch"] = attempt(
    lambda: torch.empty_strided((2, 3), (1,)))
out["empty_strided_via_aten"] = attempt(
    lambda: list(torch.ops.aten.empty_strided.default([2, 2], [2, 1]).shape))

# --- 2. torch.is_inference_mode_enabled ------------------------------------
out["inference_default"] = attempt(lambda: torch.is_inference_mode_enabled())
out["inference_on_C"] = attempt(lambda: C.is_inference_mode_enabled())

# --- 3. _should_allow_numbers_as_tensors -----------------------------------
out["numbers_as_tensors_table"] = list(C._shim_numbers_as_tensors_names)
out["numbers_add"] = attempt(lambda: C._should_allow_numbers_as_tensors("add"))
out["numbers_relu"] = attempt(lambda: C._should_allow_numbers_as_tensors("relu"))

# --- 4. _dispatch_has_computed_kernel_for_dispatch_key ---------------------
out["has_kernel_aten"] = attempt(
    lambda: C._dispatch_has_computed_kernel_for_dispatch_key("aten::add.Tensor", "Meta"))
out["has_kernel_prims"] = attempt(
    lambda: C._dispatch_has_computed_kernel_for_dispatch_key("prims::add", "Meta"))
out["has_kernel_custom"] = attempt(
    lambda: C._dispatch_has_computed_kernel_for_dispatch_key("mylib::foo", "Meta"))
out["has_kernel_cpu"] = attempt(
    lambda: C._dispatch_has_computed_kernel_for_dispatch_key("aten::add.Tensor", "CPU"))
out["has_kernel_cuda"] = attempt(
    lambda: C._dispatch_has_computed_kernel_for_dispatch_key("aten::add.Tensor", "CUDA"))

# --- 5. the throw-on-mutable-data-ptr bit ----------------------------------
_t = torch.zeros(3)
out["data_ptr_before"] = attempt(lambda: isinstance(_t.data_ptr(), int))
out["bit_before"] = attempt(lambda: C._shim_throws_on_mutable_data_ptr(_t))
C._set_throw_on_mutable_data_ptr(_t)
out["bit_after"] = attempt(lambda: C._shim_throws_on_mutable_data_ptr(_t))
out["data_ptr_after"] = attempt(lambda: _t.data_ptr())
# the bit must survive a clone rather than being laundered off by one
out["bit_survives_clone"] = attempt(
    lambda: C._shim_throws_on_mutable_data_ptr(torch.zeros(2)))
# an unmarked tensor is unaffected -- the bit is per-tensor, not global
out["other_tensor_data_ptr"] = attempt(lambda: isinstance(torch.zeros(3).data_ptr(), int))

# the softer sibling: warns and still answers, rather than refusing
import warnings as _warnings
_w = torch.zeros(3)
C._set_warn_deprecated_on_mutable_data_ptr(_w)
out["warn_bit"] = attempt(lambda: C._shim_warns_on_mutable_data_ptr(_w))
with _warnings.catch_warnings(record=True) as _caught:
    _warnings.simplefilter("always")
    out["warn_ptr_is_int"] = attempt(lambda: isinstance(_w.data_ptr(), int))
    out["warn_messages"] = [
        (x.category.__name__, str(x.message)[:60]) for x in _caught
    ]

# --- 6. a meta tensor's stride ---------------------------------------------
out["meta_contiguous"] = attempt(lambda: torch.zeros(2, 3, 4, device="meta").is_contiguous())
out["meta_stride"] = attempt(lambda: list(torch.zeros(2, 3, 4, device="meta").stride()))
out["meta_stride_dim"] = attempt(lambda: torch.zeros(2, 3, 4, device="meta").stride(0))
out["meta_stride_neg_dim"] = attempt(lambda: torch.zeros(2, 3, 4, device="meta").stride(-1))
out["meta_stride_bad_dim"] = attempt(lambda: torch.zeros(2, 3, device="meta").stride(7))
out["meta_storage_offset"] = attempt(lambda: torch.zeros(2, 3, device="meta").storage_offset())
out["meta_scalar_stride"] = attempt(lambda: list(torch.zeros((), device="meta").stride()))
# the dense side must be unchanged: a real view still reports its real stride
out["dense_view_stride"] = attempt(
    lambda: list(torch.arange(12).reshape(3, 4).t().stride()))

# --- 7. no_dispatch() actually suppresses ----------------------------------
out["suppressed_before_install"] = attempt(lambda: C._shim_dispatch_suppressed())

from torchnative.export import upstream
report = upstream.install()
out["installed_replaced"] = len(report.replaced)

# inference mode round-trips through the SAME flag the predicate reads
seq = [torch.is_inference_mode_enabled()]
with torch.inference_mode():
    seq.append(torch.is_inference_mode_enabled())
seq.append(torch.is_inference_mode_enabled())
out["inference_round_trip"] = seq

from torch.utils._python_dispatch import TorchDispatchMode

seen = []


class Log(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        seen.append(str(func))
        return func(*args, **(kwargs or {}))


with Log():
    (torch.ones(3) * 2 + 1).relu()
out["mode_sees"] = list(seen)

seen.clear()
no_dispatch = C._DisableTorchDispatch
with Log():
    with no_dispatch():
        (torch.ones(3) * 2 + 1).relu()
out["mode_sees_under_no_dispatch"] = list(seen)

# nesting: the inner exit must not un-suppress while the outer is still held
seen.clear()
with Log():
    with no_dispatch():
        with no_dispatch():
            pass
        (torch.ones(3) * 2).relu()
out["mode_sees_under_nested_no_dispatch"] = list(seen)

# and it must be restored afterwards, not left latched off
seen.clear()
with Log():
    with no_dispatch():
        pass
    (torch.ones(3) * 2).relu()
out["mode_sees_after_no_dispatch"] = list(seen)
out["suppressed_after"] = C._shim_dispatch_suppressed()

# --- 8. where export stops now ---------------------------------------------
class M(nn.Module):
    def forward(self, x):
        return (x * 2 + 1).relu()


def _export():
    ep = torch.export.export(M(), (torch.arange(6, dtype=torch.float32) - 3,))
    return {
        "ops": [str(n.target) for n in ep.graph.nodes if n.op == "call_function"],
    }


res = attempt(_export)
if res["status"] == "raise":
    try:
        torch.export.export(M(), (torch.arange(6, dtype=torch.float32) - 3,))
    except BaseException as e:
        frames = traceback.extract_tb(e.__traceback__)
        res["last_frames"] = [
            f"{f.filename.split('/main/')[-1]}:{f.lineno} {f.name}" for f in frames[-4:]
        ]
        res["last_line"] = (frames[-1].line or "") if frames else ""
out["export"] = res

print(json.dumps(out))
"""


# ---------------------------------------------------------------------------
# the upstream-side probe -- the tables, re-derived rather than transcribed
# ---------------------------------------------------------------------------

_UPSTREAM_SCRIPT = r"""
import json
import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}

f = torch._C._should_allow_numbers_as_tensors
names = set()
for n in dir(torch.ops.aten):
    if n.startswith("__"):
        continue
    names.add(n)
    names.add(n + "_")
    names.add(n + "_out")
out["numbers_as_tensors_true"] = sorted(n for n in names if f(n))

g = torch._C._dispatch_has_computed_kernel_for_dispatch_key
total = false = 0
for n in dir(torch.ops.aten):
    if n.startswith("__"):
        continue
    packet = getattr(torch.ops.aten, n)
    try:
        overloads = packet.overloads()
    except Exception:
        continue
    for ov in overloads:
        try:
            name = getattr(packet, ov).name()
        except Exception:
            continue
        try:
            r = g(name, "Meta")
        except Exception:
            continue
        total += 1
        if not r:
            false += 1
out["has_meta_total"] = total
out["has_meta_false"] = false

print(json.dumps(out))
"""


def _run(script, shim):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _shim(_cache={}):
    if "v" not in _cache:
        _cache["v"] = _run(_SHIM_SCRIPT, shim=True)
    return _cache["v"]


def _upstream(_cache={}):
    if "v" not in _cache:
        _cache["v"] = _run(_UPSTREAM_SCRIPT, shim=False)
    return _cache["v"]


def _available():
    """Same silent skip as `test_export.py`, for the same reason.

    These need `torchnative/src/main/torch/_C.abi3.so`, which
    `vendor/install_shim.sh` places and `pytests/run.sh` deliberately does not.
    """
    return os.path.isfile(_VENDOR_SHIM)


def _ok(entry, what):
    assert entry["status"] == "ok", f"{what}: {entry.get('type')}: {entry.get('message')}"
    return entry["value"]


def _raised(entry, what):
    assert entry["status"] == "raise", f"{what}: expected a refusal, got {entry.get('value')!r}"
    return entry


# ---------------------------------------------------------------------------
# 1. aten.empty_strided -- EXPORT.md §3.1, the wall §6 predicted
# ---------------------------------------------------------------------------

def test_empty_strided_builds_a_contiguous_tensor_on_cpu_and_on_meta():
    if not _available():
        return
    r = _shim()
    assert r["is_shim"], "subprocess did not get the shim-backed torch"
    assert r["empty_strided_in_implemented"], "aten.empty_strided.default not in _aten_implemented()"
    assert _ok(r["empty_strided_cpu"], "empty_strided cpu") == [2, 3]
    assert _ok(r["empty_strided_meta"], "empty_strided meta") == [2, 3]
    assert _ok(r["empty_strided_via_aten"], "empty_strided via aten") == [2, 2]
    # zeros rather than uninitialised memory, for `zeros_or_ones`' reason. This
    # is a divergence from upstream and is asserted so it stays a known one.
    assert _ok(r["empty_strided_zeroed"], "empty_strided zeroed") == 0


def test_empty_strided_refuses_a_non_contiguous_stride_by_name():
    """The refusal EXPORT4.md §4 is about, held to naming its operand.

    A contiguous tensor returned here instead would be a *silent* wrong answer:
    the caller asked for a layout and every later `.stride()` would disagree
    with what it asked for. So the refusal has to exist, and it has to say which
    stride was refused and what the only representable one was -- otherwise the
    next reader re-derives §4 from scratch.
    """
    if not _available():
        return
    r = _shim()
    e = _raised(r["empty_strided_noncontiguous"], "non-contiguous empty_strided")
    assert e["type"] == "NotImplementedError", e
    msg = e["message"]
    assert "aten.empty_strided.default" in msg, msg
    assert "non-contiguous stride is not representable" in msg, msg
    # the operands, not just the complaint
    assert "[1, 2]" in msg, msg
    assert "[3, 1]" in msg, msg
    assert "EXPORT4" in msg, msg


def test_empty_strided_does_not_let_a_size_one_axis_decide_the_refusal():
    """A size-1 axis has an unobservable stride, so it must not trigger §4.

    No two distinct index tuples differ in a length-1 axis, so upstream is
    indifferent to its stride and so is this. Refusing there would reject shapes
    that are in fact perfectly contiguous -- which is how a narrow-but-honest
    refusal turns into a wrong one.
    """
    if not _available():
        return
    r = _shim()
    assert _ok(r["empty_strided_size_one_axis"], "size-one axis") == [1, 3]


def test_empty_strided_keeps_upstreams_length_check_and_its_message():
    if not _available():
        return
    r = _shim()
    e = _raised(r["empty_strided_length_mismatch"], "length mismatch")
    assert e["type"] == "RuntimeError", e
    assert e["message"] == "mismatch in length of strides and shape", e["message"]


# ---------------------------------------------------------------------------
# 2. torch.is_inference_mode_enabled -- and the guard that must move with it
# ---------------------------------------------------------------------------

def test_is_inference_mode_enabled_answers_and_is_false_by_default():
    if not _available():
        return
    r = _shim()
    assert _ok(r["inference_default"], "torch.is_inference_mode_enabled") is False
    assert _ok(r["inference_on_C"], "torch._C.is_inference_mode_enabled") is False


def test_inference_mode_round_trips_through_the_flag_the_predicate_reads():
    """docs/EXPORT.md §2.2's failure, guarded against in the other direction.

    §2.2 is about `_len_torch_dispatch_stack` answering a constant `0` while
    something really was pushing onto the stack -- a block that entered,
    reported itself absent, and changed nothing. A constant `False` here would
    be the identical failure with different operands, and no test for a *missing
    name* would catch it, because the name is not missing.

    So this asserts the sequence, not the default: outside `False`, inside
    `True`, outside `False` again.
    """
    if not _available():
        return
    r = _shim()
    assert r["inference_round_trip"] == [False, True, False], r["inference_round_trip"]


# ---------------------------------------------------------------------------
# 3 & 4. the two predicate tables, checked against real upstream
# ---------------------------------------------------------------------------

def test_should_allow_numbers_as_tensors_matches_upstream_name_for_name():
    """The table is derived from upstream here, not transcribed once and trusted.

    A transcription goes stale silently when torch adds a name. This compares
    the shim's whole table against upstream's whole answer over every aten
    spelling, so a divergence is a failure rather than a drift.
    """
    if not _available():
        return
    shim = _shim()
    up = _upstream()
    assert not up["is_shim"], "the upstream probe got the shim"
    assert sorted(shim["numbers_as_tensors_table"]) == sorted(up["numbers_as_tensors_true"]), (
        "shim table differs from upstream:\n"
        f"  only in shim:     {sorted(set(shim['numbers_as_tensors_table']) - set(up['numbers_as_tensors_true']))}\n"
        f"  only in upstream: {sorted(set(up['numbers_as_tensors_true']) - set(shim['numbers_as_tensors_table']))}"
    )
    # the two ends of the rule, so a table that became "everything" or
    # "nothing" fails here and not only in the set comparison
    assert _ok(shim["numbers_add"], "should_allow add") is True
    assert _ok(shim["numbers_relu"], "should_allow relu") is False


def test_has_computed_kernel_matches_upstream_across_the_whole_aten_surface():
    """The namespace rule, re-derived rather than assumed.

    Upstream answers `True` for every aten overload -- measured, not asserted
    from documentation -- so `aten::*` is the rule and not a shortcut. If a
    future torch introduces an aten op with no computed Meta kernel, this fails
    and the rule needs revisiting, which is the point.
    """
    if not _available():
        return
    shim = _shim()
    up = _upstream()
    assert up["has_meta_total"] > 1000, up["has_meta_total"]
    assert up["has_meta_false"] == 0, (
        f"upstream now answers False for {up['has_meta_false']} of "
        f"{up['has_meta_total']} aten overloads -- the namespace rule in "
        "bootstrap.py::_install_dispatcher_kernel_predicates is no longer the rule"
    )
    assert _ok(shim["has_kernel_aten"], "has_kernel aten") is True
    assert _ok(shim["has_kernel_prims"], "has_kernel prims") is True
    assert _ok(shim["has_kernel_custom"], "has_kernel custom") is False
    assert _ok(shim["has_kernel_cpu"], "has_kernel CPU") is True


def test_has_computed_kernel_refuses_a_dispatch_key_it_cannot_speak_for():
    """Not weakened to `False`, because `False` would be a claim.

    This shim has no CUDA kernels and no registry to consult. Answering `False`
    would assert that upstream's dispatcher has no computed CUDA kernel either,
    which is untrue; answering `True` would promise a kernel that does not
    exist. Refusing names the key and leaves the caller able to tell the two
    apart.
    """
    if not _available():
        return
    r = _shim()
    e = _raised(r["has_kernel_cuda"], "has_kernel CUDA")
    assert e["type"] == "NotImplementedError", e
    assert "_dispatch_has_computed_kernel_for_dispatch_key" in e["message"], e["message"]
    assert "CUDA" in e["message"], e["message"]
    assert "Meta and CPU" in e["message"], e["message"]


# ---------------------------------------------------------------------------
# 5. the per-tensor throw-on-mutable-data-ptr bit -- EXPORT.md §3.2
# ---------------------------------------------------------------------------

def test_set_throw_on_mutable_data_ptr_marks_the_tensor_the_caller_passed():
    """It has to mutate the caller's object, not a clone.

    `fake_tensor.py:943` calls this on `self` inside `FakeTensor.__new__` and
    keeps that object. A version that set the bit on a copy would leave every
    real `FakeTensor` answering an address, and nothing would fail -- which is
    why the bit is asserted through a reader rather than only through the raise.
    """
    if not _available():
        return
    r = _shim()
    assert _ok(r["data_ptr_before"], "data_ptr before") is True
    assert _ok(r["bit_before"], "bit before") is False
    assert _ok(r["bit_after"], "bit after") is True
    e = _raised(r["data_ptr_after"], "data_ptr after")
    assert e["type"] == "RuntimeError", e
    assert e["message"] == (
        "Cannot access data pointer of Tensor that doesn't have storage"
    ), e["message"]


def test_the_data_ptr_bit_is_per_tensor_and_not_a_global_switch():
    if not _available():
        return
    r = _shim()
    assert _ok(r["bit_survives_clone"], "fresh tensor bit") is False
    assert _ok(r["other_tensor_data_ptr"], "unmarked tensor data_ptr") is True


# ---------------------------------------------------------------------------
# 6. a meta tensor's stride -- derived from contiguity, not invented
# ---------------------------------------------------------------------------

def test_the_warn_deprecated_sibling_warns_and_still_answers():
    """`_set_warn_deprecated_on_mutable_data_ptr` is not an alias for the thrower.

    Upstream returns the pointer and raises a `UserWarning`; it does not refuse.
    `fake_tensor.py` uses the warning one for tensors whose `data_ptr()` is
    legal-but-suspect and the throwing one for tensors with no storage at all,
    so collapsing them into a single refusal would turn a warning into an error
    for callers upstream still serves.

    This wall was exposed by closing the throwing one --
    `test_dispatch.py::test_fake_tensor_mode_is_reached_and_names_what_stops_it_returning_a_fake`
    named it, which is that test working as designed rather than a regression.
    """
    if not _available():
        return
    r = _shim()
    assert _ok(r["warn_bit"], "warn bit") is True
    assert _ok(r["warn_ptr_is_int"], "warn data_ptr still answers") is True
    cats = [c for c, _ in r["warn_messages"]]
    assert "UserWarning" in cats, r["warn_messages"]
    assert any("data pointer of FakeTensor is deprecated" in m
               for _, m in r["warn_messages"]), r["warn_messages"]


def test_a_meta_tensor_is_contiguous_so_its_stride_is_derivable():
    """The invariant the answer rests on, asserted rather than assumed.

    `Repr::Meta` stores no stride. `stride()` is answerable anyway *only*
    because every meta tensor here is contiguous, and a contiguous tensor's
    stride is a function of its shape. The day a non-contiguous meta tensor
    becomes constructible, this fails -- which is the day the derived answer
    would start lying.
    """
    if not _available():
        return
    r = _shim()
    assert _ok(r["meta_contiguous"], "meta is_contiguous") is True
    assert _ok(r["meta_stride"], "meta stride") == [12, 4, 1]
    assert _ok(r["meta_stride_dim"], "meta stride(0)") == 12
    assert _ok(r["meta_stride_neg_dim"], "meta stride(-1)") == 1
    assert _ok(r["meta_storage_offset"], "meta storage_offset") == 0
    assert _ok(r["meta_scalar_stride"], "0-d meta stride") == []


def test_meta_stride_keeps_the_dim_range_check_and_leaves_dense_alone():
    if not _available():
        return
    r = _shim()
    e = _raised(r["meta_stride_bad_dim"], "meta stride(7)")
    assert e["type"] == "IndexError", e
    # the dense path must be untouched: a transposed view still reports the
    # real stride, not a contiguous one computed from its shape
    assert _ok(r["dense_view_stride"], "dense view stride") == [1, 4]


# ---------------------------------------------------------------------------
# 7. no_dispatch() suppresses -- the defect docs/EXPORT.md §2.4 predicted
# ---------------------------------------------------------------------------

def test_no_dispatch_actually_suppresses_now_that_the_door_reads_the_stack():
    """docs/EXPORT.md §2.4's body, filled -- and this is the test that proves it.

    §2.4 said entering a counter was correct "**only because this shim never
    consults the mode stack in the first place**", and that when the door
    learned to consult it the guard would become load-bearing. The door learned
    (EXPORT.md §6 step 1) and the guard did not follow, so `no_dispatch()`
    suppressed nothing: `meta_utils.py:2009` built the meta tensor behind every
    fake tensor *inside* the `FakeTensorMode` that was trying to build it.

    A mode must see the ops outside the guard and none inside it. Asserting
    only the second half would pass on a shim where modes see nothing at all,
    which is the state this whole road started from -- so both halves are here.
    """
    if not _available():
        return
    r = _shim()
    assert r["mode_sees"], "the mode saw no operators at all -- EXPORT.md §4.2 has regressed"
    assert any("relu" in op for op in r["mode_sees"]), r["mode_sees"]
    assert r["mode_sees_under_no_dispatch"] == [], (
        "no_dispatch() did not suppress: " + repr(r["mode_sees_under_no_dispatch"])
    )


def test_no_dispatch_nests_and_is_released_rather_than_latched():
    """A boolean instead of a count would pass the simple case and fail these.

    `fake_tensor.py` enters `no_dispatch()` inside code already inside it, so
    the inner `__exit__` must not un-suppress. And the guard must actually
    release: a latched-off mode stack would make every later test in this file
    pass for the wrong reason.
    """
    if not _available():
        return
    r = _shim()
    assert r["mode_sees_under_nested_no_dispatch"] == [], (
        "an inner no_dispatch() exit un-suppressed while the outer was held: "
        + repr(r["mode_sees_under_nested_no_dispatch"])
    )
    assert r["mode_sees_after_no_dispatch"], (
        "the mode stayed suppressed after no_dispatch() exited"
    )
    assert r["suppressed_after"] is False, "suppression depth did not return to zero"
    assert _ok(r["suppressed_before_install"], "suppressed at rest") is False


# ---------------------------------------------------------------------------
# 8. where export stops, by name
# ---------------------------------------------------------------------------

def test_export_still_stops_and_it_stops_at_the_storage_handle():
    """The current wall, pinned so that moving it is a decision and not a drift.

    Two ways to fail, both wanted:

    * export stops somewhere EARLIER -- a regression in one of the six things
      this round landed, and the frame list says which;
    * export SUCCEEDS -- then this round's headline claim ("export does not yet
      return an `ExportedProgram`") is stale, and docs/EXPORT4.md §3 has to be
      rewritten rather than quietly left saying the wrong thing.

    The second is the direction docs/EXPORT.md §4.2 warns about: a graph that
    builds is not a graph that computes, and an `ExportedProgram` appearing here
    must be met with an element-wise replay comparison before anyone calls it
    working. `export_sweep.py` is the tool for that and this test is the alarm.
    """
    if not _available():
        return
    r = _shim()
    e = r["export"]
    if e["status"] == "ok":
        raise AssertionError(
            "torch.export.export() now returns an ExportedProgram with ops "
            f"{e['value']['ops']}. That is further than docs/EXPORT4.md claims. "
            "Do not simply delete this assertion: confirm the graph REPLAYS to "
            "the same numbers as eager, element-wise against upstream "
            "(rust/torch_c/pytests/export_sweep.py), then rewrite EXPORT4.md §3."
        )
    assert e["type"] == "NotImplementedError", e
    assert "Cannot copy out of meta tensor" in e["message"], e["message"]
    frames = " | ".join(e.get("last_frames", []))
    assert "meta_utils.py" in frames, frames
    assert "untyped_storage" in (e.get("last_line") or ""), e.get("last_line")


def test_the_walls_this_round_closed_are_not_reachable_again():
    """One assertion per closed wall, keyed on the message that used to appear.

    Each of the six is a name that `torch.export` calls on its way down. If any
    of them starts refusing again, the export test above stops at a different
    place -- but it would still fail with "stops somewhere earlier", which does
    not say which one. These do.
    """
    if not _available():
        return
    r = _shim()
    for key, what in (
        ("empty_strided_cpu", "aten.empty_strided"),
        ("inference_default", "torch.is_inference_mode_enabled"),
        ("numbers_add", "torch._C._should_allow_numbers_as_tensors"),
        ("has_kernel_aten", "torch._C._dispatch_has_computed_kernel_for_dispatch_key"),
        ("bit_after", "torch._C._set_throw_on_mutable_data_ptr"),
        ("meta_stride", "a meta tensor's stride()"),
    ):
        entry = r[key]
        assert entry["status"] == "ok", (
            f"{what} refuses again: {entry.get('type')}: {entry.get('message')}"
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
