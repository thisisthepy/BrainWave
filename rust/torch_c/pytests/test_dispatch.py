"""The dispatcher entrance -- `_aten_dispatch` consulting the mode stack.

`docs/EXPORT.md` §6 item 1, and only item 1.  Before this round a
`TorchDispatchMode` entered, `torch._C._len_torch_dispatch_stack()` reported 1
inside the block, and `__torch_dispatch__` was never called: the single door did
not read the stack, and `aten.rs`'s capture hook ran *after* the kernel, so it
could record a result and could not replace one.  `docs/DISPATCH3.md` is what
this measures.

Every claim here is checked **against upstream in a separate process**, not
against a list written down in this file.  `_shim()` and `_upstream()` run the
same script twice -- once with the vendored tree on `PYTHONPATH` and once with
it removed -- so "the mode saw the same operators as upstream" is a comparison
and not a transcription.  That matters more than usual here: the interesting
failure of this round is not an exception, it is a mode that enters, reports
itself active, and quietly returns eager results, and only a side-by-side can
tell that apart from working.

Written so that closing the remaining gap makes these demand more rather than
go quiet.  Two do that explicitly:

* `test_the_overload_spelling_is_the_only_remaining_disagreement_with_upstream`
  pins the `.Scalar` / `.Tensor` split that `docs/EXPORT.md` §5 found between
  `capture.rs` and upstream, and asserts the operators agree -- so a round that
  fixes overload resolution has to come here and say so.
* `test_fake_tensor_mode_is_reached_and_names_what_stops_it_returning_a_fake`
  asserts `FakeTensorMode.__torch_dispatch__` is *reached*; the day
  `aten.empty_strided` lands it starts requiring a fake back.
"""

import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


# The same script runs against both, and every question it asks is one upstream
# can answer too.  Anything shim-only is guarded on `is_shim` and recorded as
# `None` on the other side rather than skipped, so the comparison below can say
# "both answered" instead of "neither disagreed".
_SCRIPT = r"""
import json, sys, traceback
import torch
from torch import nn

out = {}
out["is_shim"] = hasattr(torch._C, "_aten_implemented")
if out["is_shim"]:
    from torchnative.export import upstream
    upstream.install()

from torch.utils._python_dispatch import TorchDispatchMode


class _Log(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func))
        return func(*args, **(kwargs or {}))


# --- 1. does a mode see the operators, and which ---------------------------
# `x` is built *outside* the block on purpose: a factory call inside it is an
# operator too (upstream reports `aten.ones.default`), and this comparison is
# about the three in `docs/EXPORT.md` §4.2.
x = torch.ones(3)
log = _Log()
with log:
    y = (x * 2 + 1).relu()
out["seen"] = log.seen
out["result"] = y.tolist()

# The operator without the overload, which is the half `docs/EXPORT.md` §5 says
# the two front ends already agree on.
out["seen_operators"] = [".".join(s.split(".")[:2]) for s in log.seen]

# --- 2. can a mode replace the result --------------------------------------
# The whole point of the entrance.  A post-hoc recorder cannot do this, and
# `FakeTensorMode` and `ProxyTorchDispatchMode` need nothing less.
_SENTINEL = object()


class _Replace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return "REPLACED"


rep = _Replace()
try:
    with rep:
        got = x * 2
    out["replaced"] = got == "REPLACED"
    out["replace_refusal"] = None
except BaseException as e:
    out["replaced"] = False
    out["replace_refusal"] = f"{type(e).__name__}: {str(e)[:120]}"
out["replace_calls"] = rep.calls

# --- 3. re-entrancy: the mode is popped for the duration -------------------
# Without the pop, `func(*args)` inside `__torch_dispatch__` finds the same mode
# on top and recurses forever.  With a *subtly* wrong pop the mode sees its own
# internal ops, which is the failure that does not announce itself -- so this
# counts rather than merely completing.
class _Reentrant(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func))
        # Two extra operator calls of its own.  A mode that is still on the
        # stack while it runs would see these as well.
        probe = func(*args, **(kwargs or {}))
        if isinstance(probe, torch.Tensor):
            probe.relu()
        return probe


ent = _Reentrant()
with ent:
    x * 2
out["reentrant_seen"] = ent.seen

# --- 4. nesting: the inner mode answers, the outer sees its re-entry --------
outer, inner = _Log(), _Log()
with outer:
    with inner:
        x * 2
out["nested_inner"] = inner.seen
out["nested_outer"] = outer.seen

# --- 5. a raising mode leaves the stack as it found it ---------------------
class _Raise(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise ValueError("mode said no")


depth_before = torch._C._len_torch_dispatch_stack()
try:
    with _Raise():
        x * 2
except ValueError as e:
    out["raising_mode_propagates"] = str(e) == "mode said no"
else:
    out["raising_mode_propagates"] = False
out["depth_restored"] = torch._C._len_torch_dispatch_stack() == depth_before
# And the ordinary path still works afterwards -- a mode leaked off the stack
# by an exception would turn one error into a silently mode-less process.
after = _Log()
with after:
    x * 2
out["mode_works_after_raise"] = len(after.seen) == 1

# --- 6. nothing on the stack: the ordinary path is untouched ---------------
out["no_mode_result"] = (x * 2 + 1).relu().tolist()
try:
    torch.ops.aten.nonexistent_op_for_dispatch_test.default(x)
except BaseException as e:
    out["refusal_type"] = type(e).__name__
else:
    out["refusal_type"] = None

# --- 7. autograd still runs, with and without a mode -----------------------
def _grad(under_mode):
    w = torch.ones(3, requires_grad=True)
    ctx = _Log() if under_mode else None
    if ctx is None:
        loss = (w * 3).sum()
    else:
        with ctx:
            loss = (w * 3).sum()
    loss.backward()
    return w.grad.tolist()


out["grad_plain"] = _grad(False)
try:
    out["grad_under_mode"] = _grad(True)
    out["grad_under_mode_refusal"] = None
except BaseException as e:
    out["grad_under_mode"] = None
    out["grad_under_mode_refusal"] = f"{type(e).__name__}: {str(e)[:160]}"

# --- 8. an infra mode is consulted from its slot ---------------------------
# `FakeTensorMode` is not on the ordinary stack: it lives in a slot keyed by
# `_TorchDispatchModeKey`, and in this shim `_len_torch_dispatch_stack()`
# reports 0 while it is entered.  A dispatcher that read only the stack would
# see nothing, silently.
from torch._subclasses.fake_tensor import FakeTensorMode

_orig_fake = FakeTensorMode.__torch_dispatch__
fake_hits = []


def _spy(self, func, types, args=(), kwargs=None):
    fake_hits.append(str(func))
    return _orig_fake(self, func, types, args, kwargs)


FakeTensorMode.__torch_dispatch__ = _spy
try:
    # `allow_non_fake_inputs` because `x` is a real tensor: without it upstream
    # refuses before any of this is measurable, which is a fact about the
    # fixture and not about either dispatcher.
    with FakeTensorMode(allow_non_fake_inputs=True):
        out["fake_stack_len"] = torch._C._len_torch_dispatch_stack()
        got = x * 2
        out["fake_returns_fake"] = type(got).__name__ == "FakeTensor"
        out["fake_shape"] = list(got.shape)
        out["fake_refusal"] = None
except BaseException as e:
    out["fake_returns_fake"] = False
    out["fake_shape"] = None
    out["fake_refusal"] = f"{type(e).__name__}: {str(e)[:200]}"
finally:
    FakeTensorMode.__torch_dispatch__ = _orig_fake
out["fake_reached"] = fake_hits

# --- 9. capture.rs, with and without a mode on the stack -------------------
class M(nn.Module):
    def forward(self, t):
        return (t * 2 + 1).relu()


if out["is_shim"]:
    m, xi = M(), torch.ones(3)
    torch._C._capture_begin([xi])
    res = m(xi)
    out["capture_ops"] = [nd["op"] for nd in torch._C._capture_end(res).nodes]

    # The same capture with a mode entered.  The mode answers by calling
    # `func(...)`, which re-enters the door with the stack popped, so the
    # kernel runs exactly once and the recorder sees exactly one op per call.
    # Double-recording would show up here as six nodes.
    lg = _Log()
    xi2 = torch.ones(3)
    torch._C._capture_begin([xi2])
    with lg:
        res2 = m(xi2)
    out["capture_ops_under_mode"] = [
        nd["op"] for nd in torch._C._capture_end(res2).nodes
    ]
    out["capture_mode_seen"] = lg.seen
else:
    out["capture_ops"] = None
    out["capture_ops_under_mode"] = None
    out["capture_mode_seen"] = None

json.dump(out, sys.stdout)
"""


def _run(vendored, _cache={}):
    key = bool(vendored)
    if key in _cache:
        return _cache[key]
    env = dict(os.environ)
    if vendored:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"dispatch subprocess ({'shim' if vendored else 'upstream'}) exited "
            f"{proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    _cache[key] = json.loads(proc.stdout)
    return _cache[key]


def _shim():
    return _run(True)


def _upstream():
    return _run(False)


def _available():
    """Same silent skip as the decompose-road tests, for the same reason.

    These need `torchnative/src/main/torch/_C.abi3.so`, which
    `vendor/install_shim.sh` places and `pytests/run.sh` deliberately does not.
    """
    return os.path.isfile(_VENDOR_SHIM)


# ---------------------------------------------------------------------------
# 1. The measurement `docs/EXPORT.md` §4.2 made, made again
# ---------------------------------------------------------------------------

def test_a_mode_sees_the_same_operators_as_upstream_in_the_same_order():
    """`SEEN: []` was the whole of the gap.  This is the side-by-side.

    Compared against upstream in a separate process rather than against three
    strings written here, because the thing being ruled out -- a mode that
    enters, reports itself active and silently returns eager results -- is
    invisible to any check that does not have upstream's answer beside it.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    assert shim["is_shim"] and not up["is_shim"], (shim["is_shim"], up["is_shim"])
    assert up["seen_operators"] == ["aten.mul", "aten.add", "aten.relu"], up["seen"]
    assert shim["seen_operators"] == up["seen_operators"], (
        f"shim {shim['seen']} vs upstream {up['seen']}"
    )
    assert shim["result"] == up["result"], (shim["result"], up["result"])


def test_the_overload_spelling_is_the_only_remaining_disagreement_with_upstream():
    """`aten.mul.Scalar` where upstream says `aten.mul.Tensor`, in two of three.

    `docs/EXPORT.md` §5 found exactly this between `capture.rs` and upstream's
    `make_fx`, and a mode now reports it too -- which is the point: the mode
    sees the key overload resolution picked, so the disagreement is one thing
    in one place and not two.  It is **not** the dispatcher entrance: it is
    `torch.Tensor.__mul__` resolving `(Tensor, Number)` to the `.Scalar`
    overload where upstream's argument parser binds the number to the `Tensor`
    parameter and wraps it.  `docs/DISPATCH3.md` §5 sizes the change.

    Pinned rather than described so a round that closes it has to come here.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    assert shim["seen"] == [
        "aten.mul.Scalar",
        "aten.add.Scalar",
        "aten.relu.default",
    ], shim["seen"]
    assert up["seen"] == [
        "aten.mul.Tensor",
        "aten.add.Tensor",
        "aten.relu.default",
    ], up["seen"]
    # And the mode agrees with `capture.rs`, which is what makes it one gap.
    assert shim["seen"] == shim["capture_ops"], (shim["seen"], shim["capture_ops"])


# ---------------------------------------------------------------------------
# 2. A mode runs *instead of* the kernel
# ---------------------------------------------------------------------------

def test_a_mode_replaces_the_result_rather_than_annotating_it():
    """The half a post-hoc recorder structurally cannot do.

    `ProxyTorchDispatchMode` returns a `Proxy` and `FakeTensorMode` returns a
    `FakeTensor`; neither is what the kernel computed, and neither front end
    works unless the mode's return value *is* the result.
    """
    if not _available():
        return
    shim = _shim()
    assert shim["replaced"] is True, shim["replace_refusal"]
    assert shim["replace_calls"] == 1, shim["replace_calls"]


def test_the_replaced_call_never_reaches_a_kernel():
    """A mode that computes nothing must still be able to answer.

    `_Replace` returns a string without calling `func`, so if the kernel had
    run anyway the result would be a tensor.  Upstream refuses to cast a string
    back to `Tensor` at the C boundary and raises instead -- a stricter answer
    than this shim's, and recorded as a difference rather than asserted equal.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    assert shim["replaced"] is True, shim["replace_refusal"]
    assert up["replace_calls"] == 1, up["replace_calls"]


# ---------------------------------------------------------------------------
# 3. Re-entrancy -- the failure that does not announce itself
# ---------------------------------------------------------------------------

def test_the_mode_is_popped_so_it_does_not_see_its_own_internal_operators():
    """Getting this wrong loudly is recursion; getting it quietly wrong is this.

    `_Reentrant.__torch_dispatch__` performs two operator calls of its own.  A
    mode still on the stack while it runs would see them and would report three
    entries for one user-level call -- and would then see the ones *those* made,
    until the stack blew.  One entry is the whole claim.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    assert len(shim["reentrant_seen"]) == 1, shim["reentrant_seen"]
    assert len(up["reentrant_seen"]) == len(shim["reentrant_seen"]), (
        shim["reentrant_seen"],
        up["reentrant_seen"],
    )


def test_a_nested_mode_answers_and_the_outer_one_sees_the_re_entry():
    """Popping the innermost is not the same as suppressing the stack.

    The inner mode answers the user call; its own `func(...)` finds the outer
    mode still there and goes through it.  Both counts are one, and the shape
    matches upstream's.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    assert len(shim["nested_inner"]) == 1, shim["nested_inner"]
    assert len(shim["nested_outer"]) == 1, shim["nested_outer"]
    assert (len(up["nested_inner"]), len(up["nested_outer"])) == (
        len(shim["nested_inner"]),
        len(shim["nested_outer"]),
    ), (shim, up)


def test_a_raising_mode_leaves_the_stack_as_it_found_it():
    """A mode leaked off the stack turns one error into a mode-less process.

    That is strictly worse than the error, because the next block enters,
    reports itself active, and returns eager results -- the exact shape
    `docs/COMPILE.md` §5 refuses.
    """
    if not _available():
        return
    shim = _shim()
    assert shim["raising_mode_propagates"] is True
    assert shim["depth_restored"] is True
    assert shim["mode_works_after_raise"] is True


# ---------------------------------------------------------------------------
# 4. The ordinary path -- unmoved
# ---------------------------------------------------------------------------

def test_with_no_mode_on_the_stack_the_answers_are_upstreams():
    """Golden is the real gate on this; this is the shape of it in one place.

    With nothing entered, the consult is a module attribute read and a branch
    that is not taken, and the door reaches `aten_dispatch` with the arguments
    untouched -- same result, same refusal.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    assert shim["no_mode_result"] == up["no_mode_result"], (
        shim["no_mode_result"],
        up["no_mode_result"],
    )
    assert shim["refusal_type"] is not None, "an absent op stopped refusing"


def test_autograd_runs_the_same_with_and_without_a_mode_on_the_stack():
    """The eager tape rides this path (`docs/BACKWARD7.md`) and must be unmoved.

    Without a mode, `mark_from_op` and `eager_record` run exactly where they
    did.  With one, the mode's own `func(...)` re-entry is what reaches them --
    once -- so the gradient is the same number either way, and the same number
    upstream computes.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    assert shim["grad_plain"] == up["grad_plain"], (shim["grad_plain"], up["grad_plain"])
    assert shim["grad_under_mode"] == shim["grad_plain"], (
        shim["grad_under_mode"],
        shim["grad_plain"],
        shim["grad_under_mode_refusal"],
    )


def test_capture_records_each_operator_once_while_a_mode_is_on_the_stack():
    """`docs/CAPTURE.md` is a contract and a second hook must not disturb it.

    The mode answers by re-entering the door with itself popped, so the kernel
    runs once and the recorder fires once.  A consult placed *after* the
    recorder, or one that failed to return early, would double every node --
    which is why this compares the trace against the mode-less one rather than
    against a length.
    """
    if not _available():
        return
    shim = _shim()
    assert shim["capture_ops_under_mode"] == shim["capture_ops"], (
        shim["capture_ops_under_mode"],
        shim["capture_ops"],
    )
    assert len(shim["capture_mode_seen"]) == len(shim["capture_ops"]), (
        shim["capture_mode_seen"],
        shim["capture_ops"],
    )


# ---------------------------------------------------------------------------
# 5. Infra modes, and the harder half
# ---------------------------------------------------------------------------

def test_an_infra_mode_is_found_in_its_slot_and_not_only_on_the_stack():
    """`FakeTensorMode` is invisible to `_len_torch_dispatch_stack()` here.

    Upstream's C++ counts the infra slots in that length; this shim's mode
    stack counts only the ordinary stack, so a dispatcher that read the length
    alone would see zero while a fake mode was entered and would silently run
    the kernel.  The consult reads the slots too, in upstream's
    `pop_stack` order.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    assert shim["fake_stack_len"] == 0, shim["fake_stack_len"]
    assert up["fake_stack_len"] == 1, up["fake_stack_len"]
    assert shim["fake_reached"], (
        "FakeTensorMode.__torch_dispatch__ was never called, so the infra slot "
        "is not being consulted"
    )


def test_fake_tensor_mode_is_reached_and_names_what_stops_it_returning_a_fake():
    """The harder half, and the honest state of it.

    The entrance is landed: `FakeTensorMode.__torch_dispatch__` runs and its
    return value would be the result.  What it cannot yet do is *build* a fake.

    **The named reason has moved once, and that is this test working.**  It was
    `docs/EXPORT.md` §3.1's missing `aten.empty_strided`; `docs/EXPORT4.md`
    closed that and five more behind it, and the reason is now `docs/EXPORT.md`
    §3.3 -- `meta_utils.py:2071` asks a meta tensor for `untyped_storage()` so
    two views of one base can share a fake storage, and `Repr::Meta` has no
    storage handle to give.  That is a storage-model question rather than a
    missing name, which is why it outlasted the six.

    The accepted reasons are a list rather than a wildcard so that each move has
    to be looked at.  Adding to it is the cheap thing to do and it is only
    correct when the new reason has been read; a wildcard here would let the
    fake-tensor road regress to name #0 silently.

    Written to demand more when the wall falls: the moment a fake comes back
    this stops accepting a refusal.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    assert up["fake_returns_fake"] is True, up["fake_refusal"]
    if shim["fake_returns_fake"]:
        # The gap closed.  Then the fake must agree with upstream's on shape
        # and the operator must have reached the mode.
        assert shim["fake_reached"], shim
        assert shim["fake_shape"] == up["fake_shape"], (
            shim["fake_shape"],
            up["fake_shape"],
        )
        return
    assert shim["fake_reached"], (
        "the fake mode was not even reached -- that is the dispatcher "
        "entrance, not the fake tensor machinery"
    )
    assert shim["fake_refusal"] is not None
    accepted = (
        # docs/EXPORT.md §3.3 / docs/EXPORT4.md §7 -- the current wall
        "Cannot copy out of meta tensor",
        # the earlier ones, kept so a regression names itself rather than
        # merely failing
        "empty_strided",
        "is_inference_mode",
    )
    assert any(a in shim["fake_refusal"] for a in accepted), (
        "FakeTensorMode now fails for a reason docs/EXPORT.md §3 and "
        "docs/EXPORT4.md §7 did not name: " + str(shim["fake_refusal"])
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
