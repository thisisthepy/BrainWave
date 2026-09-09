"""Tests for `torchnative.export.qnn_plan.plan_lowering` -- the QNN analogue of
`torchnative.export.intelnpu.plan_lowering`.

Why this is a separate file rather than more of `test_qnn.py`: everything in
`test_qnn_plan.py` is **pure** -- no ExecuTorch, no Qualcomm SDK, no device,
not even the torchnative `_C` shim. `plan_lowering` walks an `nn.Module` tree
and asks an *injected* op table what it would do with each leaf; the op table
itself (`torchnative.export.qnn_ops`) is a sibling round's file and does not
exist yet in this tree. So every test here builds a small stand-in exposing
exactly `check_leaf(name, module)` (the interface `qnn_plan` was written
against) rather than importing the real one -- which is also what proves
`plan_lowering`'s logic does not secretly depend on anything else the real
module might supply.

This file asks only for `torch.nn` (`_real_torch()` below) and touches no
shim `_C`. That is NOT the same as needing nothing from the environment, and
an earlier version of this note said it was.

It puts the vendored tree first on `sys.path` so `torchnative.export.qnn_plan`
resolves, which also makes `import torch` resolve to the VENDORED torch --
whose `_load_global_deps()` dlopens `torch/lib/libtorch_global_deps.*`, a file
`tools/wheel/build.py` creates for a wheel and which does not exist in the
source tree. So `TORCH_USE_RTLD_GLOBAL` is required after all, and is set
below the way every other suite here sets it.

The claim survived review because this file was run from a worktree with no
vendored tree, where `import torch` fell through to an upstream install and
the vendored path was never taken. Under `run.sh` it raised OSError eleven
times.

**The trap this file is checking for is silence, not error.** ExecuTorch's QNN
partitioner declines a node it does not accept *silently* -- the exported
program still runs, on the CPU, with a correct answer, and nothing raises.
`docs/devices/QNN.md` (via `torchnative.export.qnn.delegation_report`,
`PteArtefact.htp_plan`) is what shows the after-the-fact split between
delegated and CPU-fallback ops in a real `.pte`; nothing in this file produces
one. So the one property every other test here serves is:
`plan_lowering` must never put a leaf in `eligible` unless the op table it was
given affirmatively said, for that leaf, that it would be taken -- never by
type-name guess, never by absence of a decline reason, never as a default.
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
sys.path.insert(0, _VENDOR_DIR)

# Before anything can reach `torch`. See the module docstring.
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from torchnative.export.qnn_plan import (  # noqa: E402
    QnnPlanUnavailable,
    plan_lowering,
)


def _real_torch():
    """Plain upstream torch. No shim, no `_C` override -- `plan_lowering`
    never touches CPython-extension internals, only `nn.Module` tree shape."""
    import torch

    return torch


class _FakeVerdict:
    """The object-shaped half of the verdict contract, exercised so both
    accepted shapes (tuple, and `.taken`/`.reason`) are proven, not just one."""

    def __init__(self, taken, reason):
        self.taken = taken
        self.reason = reason


class _StubOps:
    """A hand-rolled stand-in for the not-yet-written `qnn_ops` module.

    `rules` maps a leaf path to `(taken, reason)`. Anything not named in
    `rules` declines by default with a distinguishing reason, so a test can
    tell "explicitly declined" from "the stub forgot this leaf" apart in a
    failure message.
    """

    def __init__(self, rules, use_objects=False):
        self.rules = rules
        self.use_objects = use_objects
        self.seen = []

    def check_leaf(self, name, module):
        self.seen.append(name)
        taken, reason = self.rules.get(
            name, (False, f"{name}: no rule in this stub, declined by default")
        )
        if self.use_objects:
            return _FakeVerdict(taken, reason)
        return (taken, reason)


# ---------------------------------------------------------------- the core trap


def test_a_leaf_the_op_table_declines_is_never_reported_eligible():
    """The central guarantee: QNN's partitioner declines silently, so
    `plan_lowering` must not report a leaf lowered unless the op table said
    so -- explicitly, for that leaf, not by type or by omission."""
    torch = _real_torch()
    model = torch.nn.Sequential()
    model.add_module("linear", torch.nn.Linear(8, 8))
    model.add_module("odd_op", torch.nn.Conv2d(3, 3, 3))

    ops = _StubOps({
        "linear": (True, "aten::linear is in supported_ops() and shape fits"),
        "odd_op": (False, "aten::convolution is not in the QNN HTP op table"),
    })

    plan = plan_lowering(model, ops=ops)
    assert plan["eligible"] == ["linear"], plan["eligible"]
    assert plan["skipped"] == [
        ("odd_op", "aten::convolution is not in the QNN HTP op table")
    ], plan["skipped"]
    assert plan["left_on_cpu"] == {"Conv2d": 1}, plan["left_on_cpu"]
    assert plan["fully_offloaded"] is False, plan
    assert ops.seen == ["linear", "odd_op"], ops.seen
    print(
        "ok   qnn_plan: a leaf the op table declines is left on the CPU and "
        "named, never claimed eligible"
    )


def test_nothing_is_eligible_by_default_when_the_op_table_declines_everything():
    """Nullification-shaped on purpose: an op table that always declines must
    produce an empty `eligible` list, not "all good"."""
    torch = _real_torch()
    model = torch.nn.Sequential()
    model.add_module("a", torch.nn.Linear(4, 4))
    model.add_module("b", torch.nn.Linear(4, 4))

    ops = _StubOps({})  # every leaf falls through to the declined default
    plan = plan_lowering(model, ops=ops)
    assert plan["eligible"] == [], plan["eligible"]
    assert len(plan["skipped"]) == 2, plan["skipped"]
    assert plan["fully_offloaded"] is False, plan
    assert plan["parameters_moved"] == 0, plan
    assert plan["fraction_moved"] == 0.0, plan
    print(
        "ok   qnn_plan: an op table that declines every leaf reports zero "
        "eligible, not a silent pass"
    )


# ---------------------------------------------------------------- the report's quality


def test_a_real_width_qwen3_shaped_model_names_exactly_the_one_leaf_that_does_not_lower():
    """The bar this round was asked to reproduce: 14 of 15 Linears lower, and
    the report says exactly which one did not and why -- not "some model
    somewhere", the leaf by dotted name."""
    torch = _real_torch()

    class Block(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.q_proj = torch.nn.Linear(dim, dim, bias=False)
            self.k_proj = torch.nn.Linear(dim, dim, bias=False)
            self.v_proj = torch.nn.Linear(dim, dim, bias=False)
            self.o_proj = torch.nn.Linear(dim, dim, bias=False)
            self.gate_proj = torch.nn.Linear(dim, dim, bias=False)
            self.up_proj = torch.nn.Linear(dim, dim, bias=False)
            self.down_proj = torch.nn.Linear(dim, dim, bias=False)

    class TinyQwenShaped(torch.nn.Module):
        def __init__(self, dim=8, vocab=200003):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(vocab, dim)
            self.layers = torch.nn.ModuleList([Block(dim), Block(dim)])
            self.lm_head = torch.nn.Linear(dim, vocab, bias=False)

    model = TinyQwenShaped()
    linear_paths = [
        name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)
    ]
    assert len(linear_paths) == 15, linear_paths  # 2 * 7 + lm_head

    # Every Linear lowers except lm_head -- QNN's own real constraint is a
    # vocab-sized output the HTP quantization tables cannot represent; the
    # stub encodes "declined, oversized last dim" for exactly that one leaf,
    # by name, the same way intelnpu's real MAX_DIM check does.
    rules = {path: (True, "aten::linear, weight fits the HTP op table") for path in linear_paths}
    rules["lm_head"] = (
        False,
        "aten::linear declined: out_features=200003 exceeds the HTP quantization "
        "table's largest observed dimension",
    )
    # embed_tokens is also a leaf the walk asks about -- it is not a Linear,
    # so it must be given its own rule rather than falling through to the
    # stub's decline-by-default, or this test would be asserting the wrong
    # count for the wrong reason (an unrelated leaf, not lm_head).
    rules["embed_tokens"] = (True, "aten::embedding is in the HTP op table")

    plan = plan_lowering(model, ops=_StubOps(rules))
    # 14 of the 15 Linears, plus embed_tokens (its own, separately-ruled leaf).
    assert len(plan["eligible"]) == 15, plan["eligible"]
    assert "lm_head" not in plan["eligible"], plan["eligible"]
    assert sum(1 for p in plan["eligible"] if p in linear_paths) == 14, plan["eligible"]
    assert len(plan["skipped"]) == 1, plan["skipped"]
    name, reason = plan["skipped"][0]
    assert name == "lm_head", name
    assert "200003" in reason, reason
    assert "exceeds" in reason, reason
    assert plan["fully_offloaded"] is False, plan
    assert 0.0 < plan["fraction_moved"] < 1.0, plan["fraction_moved"]
    print(
        "ok   qnn_plan: a 15-Linear Qwen3-shaped model lowers 14 and names "
        "lm_head, with its shape and the reason, as the one that does not"
    )


def test_fully_offloaded_when_every_leaf_is_taken():
    torch = _real_torch()
    model = torch.nn.Sequential()
    model.add_module("a", torch.nn.Linear(4, 4))
    model.add_module("b", torch.nn.Linear(4, 4))
    ops = _StubOps({
        "a": (True, "ok"),
        "b": (True, "ok"),
    })
    plan = plan_lowering(model, ops=ops)
    assert plan["fully_offloaded"] is True, plan
    assert plan["fraction_moved"] == 1.0, plan
    assert plan["left_on_cpu"] == {}, plan["left_on_cpu"]
    print("ok   qnn_plan: every leaf taken means fully_offloaded and fraction_moved == 1.0")


def test_fraction_moved_is_a_value_not_only_a_leaf_list():
    """Mirrors intelnpu's `test_how_much_moved_is_a_value_and_not_only_prose`:
    a caller who ignores `skipped` must still be unable to conclude a full
    offload -- `fraction_moved` is a number, weighted by parameter count."""
    torch = _real_torch()
    model = torch.nn.Sequential()
    model.add_module("small", torch.nn.Linear(10, 10, bias=False))   # 100 params
    model.add_module("huge", torch.nn.Linear(10, 9000, bias=False))  # 90000 params

    ops = _StubOps({
        "small": (True, "fits"),
        "huge": (False, "huge: declined, oversized"),
    })
    plan = plan_lowering(model, ops=ops)
    assert plan["parameters_total"] == 100 + 90000, plan
    assert plan["parameters_moved"] == 100, plan
    assert plan["fraction_moved"] == 100 / 90100
    assert plan["fraction_moved"] < 0.02, plan["fraction_moved"]
    print(
        f"ok   qnn_plan: fraction_moved is a value -- "
        f"{plan['fraction_moved']:.5f} when the largest leaf is declined"
    )


# ---------------------------------------------------------------- predicate + verdict shapes


def test_the_predicate_matches_quantize_s_signature_and_narrows_selection_before_ops():
    """Same spelling as `torchnative.quant.quantize_` and
    `intelnpu.plan_lowering`: `predicate(name, module) -> bool`. A predicate
    exclusion must not even reach the op table -- checked via `ops.seen`."""
    import inspect

    torch = _real_torch()
    assert "predicate" in inspect.signature(plan_lowering).parameters

    model = torch.nn.Sequential()
    model.add_module("keep", torch.nn.Linear(8, 8))
    model.add_module("drop", torch.nn.Linear(8, 8))

    def predicate(name, module):
        return name != "drop"

    ops = _StubOps({"keep": (True, "fits")})
    plan = plan_lowering(model, predicate=predicate, ops=ops)
    assert plan["eligible"] == ["keep"], plan["eligible"]
    assert plan["skipped"] == [("drop", "excluded by predicate")], plan["skipped"]
    assert ops.seen == ["keep"], (
        "predicate exclusion reached the op table -- it must not: "
        f"{ops.seen}"
    )
    print(
        "ok   qnn_plan: predicate(name, module) narrows the selection before "
        "the op table is ever consulted"
    )


def test_check_leaf_may_return_either_a_tuple_or_an_object_with_taken_and_reason():
    torch = _real_torch()
    model = torch.nn.Sequential()
    model.add_module("a", torch.nn.Linear(4, 4))
    model.add_module("b", torch.nn.Linear(4, 4))

    tuple_ops = _StubOps({"a": (True, "tuple shape"), "b": (False, "tuple decline")})
    object_ops = _StubOps(
        {"a": (True, "object shape"), "b": (False, "object decline")}, use_objects=True
    )

    plan_tuple = plan_lowering(model, ops=tuple_ops)
    plan_object = plan_lowering(model, ops=object_ops)
    assert plan_tuple["eligible"] == plan_object["eligible"] == ["a"]
    assert plan_tuple["skipped"][0][0] == plan_object["skipped"][0][0] == "b"
    print(
        "ok   qnn_plan: check_leaf's verdict may be a (bool, str) tuple or an "
        "object exposing .taken and .reason -- both are read identically"
    )


def test_a_malformed_verdict_raises_rather_than_being_read_as_declined():
    """A verdict that is neither shape is a bug in the op table, not a
    'declined' answer -- silently downgrading it to declined would hide
    exactly the kind of error this module exists to surface."""
    torch = _real_torch()
    model = torch.nn.Sequential()
    model.add_module("a", torch.nn.Linear(4, 4))

    class BrokenOps:
        def check_leaf(self, name, module):
            return "not a verdict"

    try:
        plan_lowering(model, ops=BrokenOps())
        raised = False
    except TypeError:
        raised = True
    assert raised, "a malformed check_leaf return must raise, not be read as declined"
    print(
        "ok   qnn_plan: a check_leaf return that is neither a tuple nor a "
        ".taken/.reason object raises TypeError instead of being read as declined"
    )


def test_an_ops_object_with_no_check_leaf_is_refused_by_name():
    torch = _real_torch()
    model = torch.nn.Sequential()
    model.add_module("a", torch.nn.Linear(4, 4))

    class NoCheckLeaf:
        pass

    try:
        plan_lowering(model, ops=NoCheckLeaf())
        raised = False
    except QnnPlanUnavailable as exc:
        raised = True
        assert "check_leaf" in str(exc), exc
    assert raised, "an ops object without check_leaf must be refused by name"
    print(
        "ok   qnn_plan: an ops object with no check_leaf(name, module) is "
        "refused by name, not silently treated as an empty table"
    )


# ---------------------------------------------------------------- missing sibling file


#: Sentinel for "this key was not in sys.modules", distinct from a stored None.
_ABSENT = object()


def test_missing_qnn_ops_module_is_refused_by_name_not_a_bare_import_error():
    """With no `ops=` and no importable `qnn_ops`, the refusal must name it.

    A caller who gets a bare `ImportError` has to recognise it as this
    particular missing piece; a named `QnnPlanUnavailable` says which piece.

    This used to rely on `qnn_ops` genuinely not existing, and asserted its own
    premise would go stale once the sibling round landed. It has landed, so the
    absence is SIMULATED here -- blocking the import rather than waiting for a
    tree that no longer occurs. A test that retires itself the moment its
    neighbour ships is a test that stops guarding the path it was written for.
    """
    torch = _real_torch()
    if torch is None:
        print("   (skipped: torch.nn not importable)")
        return
    model = torch.nn.Sequential()
    model.add_module("a", torch.nn.Linear(4, 4))

    import importlib

    saved = sys.modules.get("torchnative.export.qnn_ops", _ABSENT)

    class _RefuseQnnOps:
        """Import hook that makes exactly one module unimportable."""

        def find_module(self, name, path=None):
            return self if name == "torchnative.export.qnn_ops" else None

        def find_spec(self, name, path=None, target=None):
            if name == "torchnative.export.qnn_ops":
                raise ImportError("blocked by test_qnn_plan")
            return None

    hook = _RefuseQnnOps()
    sys.modules.pop("torchnative.export.qnn_ops", None)
    sys.meta_path.insert(0, hook)
    try:
        importlib.invalidate_caches()
        try:
            plan_lowering(model)
            raised = False
        except QnnPlanUnavailable as exc:
            raised = True
            assert "qnn_ops" in str(exc), exc
    finally:
        sys.meta_path.remove(hook)
        if saved is _ABSENT:
            sys.modules.pop("torchnative.export.qnn_ops", None)
        else:
            sys.modules["torchnative.export.qnn_ops"] = saved
        importlib.invalidate_caches()

    assert raised, (
        "plan_lowering() with no ops= and no importable qnn_ops must raise "
        "QnnPlanUnavailable, naming it"
    )
    print(
        "ok   qnn_plan: with qnn_ops made unimportable, plan_lowering raises "
        "QnnPlanUnavailable naming it, not a bare ImportError"
    )


def test_walk_descends_into_nested_containers_and_still_asks_every_leaf():
    torch = _real_torch()

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = torch.nn.Linear(4, 4)
            self.inner = Inner()

    model = Outer()
    ops = _StubOps({
        "pre": (True, "fits"),
        "inner.proj": (False, "declined"),
    })
    plan = plan_lowering(model, ops=ops)
    assert plan["eligible"] == ["pre"], plan["eligible"]
    assert plan["skipped"] == [("inner.proj", "declined")], plan["skipped"]
    print("ok   qnn_plan: the walk descends into nested containers with dotted names")


def test_the_real_qnn_ops_table_drives_the_real_plan():
    """The two modules must agree on `check_leaf`'s return, not merely each on a fake.

    Every other test here injects an `ops` object, which is what makes
    `plan_lowering` testable without `qnn_ops` -- and is exactly why a
    mismatch survived: `qnn_ops.check_leaf` returns
    `{"accepted": ..., "reason": ...}`, `qnn_plan` accepted only a tuple or a
    `.taken`/`.reason` object, and each round was green against its own idea
    of the contract. They met on develop and `plan_lowering` raised TypeError
    on the first real leaf.

    So this one passes NO `ops` and lets the real table answer. It asserts the
    handshake, not a coverage number: which ops QNN takes is
    `test_qnn_ops.py`'s subject, and pinning a count here would go stale
    against that table rather than catch a real defect.
    """
    torch = _real_torch()
    if torch is None:
        print("   (skipped: torch.nn not importable)")
        return
    try:
        from torchnative.export import qnn_ops  # noqa: F401
    except ImportError:
        print("   (skipped: torchnative.export.qnn_ops is not present)")
        return

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.LayerNorm(8),
        torch.nn.Linear(8, 4),
    )
    plan = plan_lowering(model)          # no ops= -- the real table

    for key in ("eligible", "skipped", "left_on_cpu", "fully_offloaded",
                "parameters_moved", "parameters_total", "fraction_moved"):
        assert key in plan, f"the real table produced a plan with no {key!r}: {plan}"
    assert isinstance(plan["fraction_moved"], float), plan["fraction_moved"]
    for name, reason in plan["skipped"]:
        assert reason, f"leaf {name} was declined without a reason"

    # The assertion that would have caught the original defect. `nn.Linear`
    # decomposes to `aten.linear.default`, which IS in the table
    # (test_qnn_ops.py pins that), so a plan in which no Linear lowers is not a
    # coverage fact -- it means the two modules are not composing. The first
    # version of this pairing declined all three leaves here, because the
    # module PATH was being put to a table keyed on ATen op names.
    assert "aten.linear.default" in qnn_ops.supported_ops(), (
        "this test's premise is gone: the table no longer lists linear"
    )
    assert plan["eligible"], (
        "the real qnn_ops table declined EVERY leaf of a model that is mostly "
        f"nn.Linear -- the two modules are not composing: {plan['skipped']}"
    )
    assert len(plan["eligible"]) >= 2, (
        f"both Linears should lower; got {plan['eligible']}"
    )
    assert 0.0 < plan["fraction_moved"] <= 1.0, plan["fraction_moved"]
    print(
        f"ok   qnn_plan: the real qnn_ops table drives plan_lowering with no "
        f"ops= injected ({len(plan['eligible'])} eligible, "
        f"{len(plan['skipped'])} declined)"
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
