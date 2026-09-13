""""CoreML told us nothing" is not "CoreML told us CPU", and must not be silent.

`compute_plan` asks `MLComputePlan` which unit runs each operation. Every
caller of it -- `_say_what_ran`, and `_compile_model`'s `to()`-time guard --
used to begin by dropping the operations CoreML returned *no* usage for and
then returning early when nothing was left. So a model whose plan CoreML
declines to produce got **no warning at all**: the same silence that a FULL
offload deliberately gets, which is the exact outcome docs/graph/NPU2.md §1
exists to prevent.

That this is reachable is measured, not hypothetical. docs/graph/NPU2.md §9.1
records a single-`relu` and a single-`gelu` float16 program whose compiled
`.mlmodelc` gets `None` from
`get_compute_device_usage_for_mlprogram_operation` for **every** operation in
it -- including the `ios16.cast`s, which do get a usage in a `linear` program
compiled seconds earlier in the same process. Renaming the operation inside the
compiled bundle, changing nothing else, restores the usage; renaming it back
removes it again. So the artefact's identity, not the program it encodes, is
what decides -- and a library that reads that as "nothing to say" reports a
CPU-bound model as an offloaded one.

The rule this file fixes in place:

    every computing operation is on the report, and one whose unit CoreML
    would not name is on it as `"unknown"` rather than absent;
    unknown warns; unknown is not reported as the precision's fault;
    and a FULL offload is still silent, because a warning nobody can ignore
    is one nobody reads.

Pure: these are `_say_what_ran` and `computes` over rows built here, so they
run on a machine with no CoreML at all. The end-to-end half -- that a real
relu program reaches this path -- is in test_coremlops.py.
"""

import os
import sys
import warnings

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "torchnative", "src", "main"))

from torchnative.export import coreml as C


def _row(op, preferred, supported):
    return {"op": op, "preferred": preferred, "supported": list(supported)}


_KNOWN_NE = _row("ios16.conv", "NeuralEngine", ["CPU", "GPU", "NeuralEngine"])
_KNOWN_CPU = _row("ios16.relu", "CPU", ["CPU", "GPU", "NeuralEngine"])
_UNKNOWN = _row("ios16.relu", "unknown", [])
_CAST = _row("ios16.cast", "CPU", ["CPU", "GPU"])


def _said(rows, precision="float16", report=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        C._say_what_ran(report if report is not None else {},
                        precision, [1, 128, 32, 32], rows)
    return [str(w.message) for w in caught]


def test_a_plan_with_no_rows_at_all_warns_that_what_ran_is_unknown():
    """The failure this file is named for: zero rows used to return silently.

    `_say_what_ran` began `if not compute: return`, so a program CoreML
    produced no plan for was indistinguishable from a program that reached the
    Neural Engine completely. Those are opposite facts.
    """
    said = _said([])
    assert len(said) == 1, said
    assert "no compute plan at all" in said[0], said[0]
    assert "unknown" in said[0], said[0]


def test_an_unknown_operation_warns_even_when_other_operations_are_known():
    """A partially unknown plan is unknown, not a partial offload.

    If one operation's unit is unnamed, "everything reached the unit" cannot
    be said about this program, so the FULL-offload silence is not available
    to it -- even if every operation CoreML *did* name is on the unit.
    """
    said = _said([_CAST, _KNOWN_NE, _UNKNOWN])
    assert len(said) == 1, said
    assert "no usable compute plan" in said[0], said[0]
    assert "ios16.relu" in said[0], said[0]
    assert "unknown" in said[0], said[0]


def test_a_plan_of_nothing_but_boundary_casts_is_unknown_and_not_silence():
    """`computes()` drops the boundary casts, and dropping them all is not none.

    A program whose only operations are the float16 interface casts did no
    work CoreML will speak about, so "which unit ran this" is unanswered --
    the same sentence as an empty plan, reached a different way.
    """
    said = _said([_CAST, _CAST])
    assert len(said) == 1, said
    assert "unknown" in said[0], said[0]


def test_an_unknown_plan_is_not_blamed_on_the_precision():
    """The wrong sentence is worse than none: it sends the caller to float16.

    `_UNREACHABLE_PRECISION` says the Neural Engine is *not in the supported
    column at this precision*, which is a claim about what MLComputePlan
    returned. An unknown plan returned nothing, so that claim has no evidence
    behind it -- and a caller who acted on it would switch precision and get
    the same silence.
    """
    said = _said([_UNKNOWN])
    assert len(said) == 1, said
    assert "supported column" not in said[0], said[0]
    assert "precision=" not in said[0], said[0]


def test_a_full_offload_is_still_silent():
    """The constraint the existing design carries, unbroken.

    Every computing operation known and preferred on the unit is the one case
    with positive evidence that the offload worked, and it says nothing so
    that a warning stays worth reading. Widening "unknown warns" into
    "everything warns" would destroy that, so it is asserted here rather than
    assumed.
    """
    assert _said([_CAST, _KNOWN_NE, _CAST]) == []


def test_a_known_cpu_preferred_plan_still_gets_its_own_sentence():
    """Unknown must not swallow the case that already worked.

    CPU-preferred-but-NE-supported is a measured fact about the model's size
    (docs/graph/NPU2.md §2.1) and has its own wording. If the unknown branch
    caught this too, the distinction the whole file is about would be lost in
    the other direction.
    """
    said = _said([_CAST, _KNOWN_CPU, _CAST])
    assert len(said) == 1, said
    assert "is NOT what it preferred" in said[0], said[0]
    assert "unknown" not in said[0], said[0]


def test_the_unknown_sentence_is_said_once_per_model_and_not_per_shape():
    """Said through the report's own bookkeeping, like the other two.

    A leaf compiles per shape, so an unknown plan would otherwise be said at
    every batch a caller runs, and a warning repeated on every forward is one
    that gets filtered out.
    """
    report = {}
    first = _said([_UNKNOWN], report=report)
    second = _said([_UNKNOWN], report=report)
    assert len(first) == 1, first
    assert second == [], second


def test_unknown_rows_survive_the_boundary_cast_filter():
    """`computes()` drops casts by name, and an unknown row is not a cast.

    The filter is what stands between "this operation's unit is unnamed" and
    an empty list, so it is asserted directly: an unknown `relu` is still
    there afterwards.
    """
    kept = C.computes([_CAST, _UNKNOWN, _CAST])
    assert kept == [_UNKNOWN], kept


def _at_to(plans, precision="float16", report=None):
    report = report if report is not None else {}
    report["plans"] = plans
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        C._say_at_to_time(report, precision)
    return [str(w.message) for w in caught]


def test_the_to_time_guard_sees_a_plan_that_came_back_with_nothing():
    """The second place the silence lived, and the one a test could not reach.

    `_compile_model`'s tail read `probe_rows` -- the *flattened, cast-filtered*
    rows of every plan. A plan CoreML returned nothing for contributes no rows
    to that list, so `if probe_rows and ...` skipped it and `to()` returned a
    model with a compiled leaf and not one word about it. It is checked over
    the plan entries instead, which is why this test hands it a plan whose
    `rows` is empty rather than handing it no plan.
    """
    said = _at_to([{"leaf": "linear", "rows": []}])
    assert len(said) == 1, said
    assert "no compute plan at all" in said[0], said[0]


def test_the_to_time_guard_does_not_blame_the_precision_for_an_unnamed_op():
    """`_UNREACHABLE_PRECISION` is a claim about a column CoreML filled in.

    An unknown row has an empty `supported` for the opposite reason -- CoreML
    filled in nothing -- and reporting it as "the unit is not in the supported
    column at this precision" is a false sentence with an action attached to
    it. The unknown branch is checked first so that sentence is unreachable
    from this state.
    """
    said = _at_to([{"leaf": "linear", "rows": [_CAST, _UNKNOWN]}])
    assert len(said) == 1, said
    assert "supported column" not in said[0], said[0]
    assert "unknown" in said[0], said[0]


def test_the_to_time_guard_still_says_the_precision_sentence_when_it_is_true():
    """Known rows, unit absent from a column CoreML did fill in: unchanged.

    This is the case that already worked -- a float32 program -- and the
    unknown branch must not have swallowed it.
    """
    known_no_ne = _row("ios16.linear", "CPU", ["CPU", "GPU"])
    said = _at_to([{"leaf": "linear", "rows": [known_no_ne]}], "float32")
    assert len(said) == 1, said
    assert "NOT in CoreML's supported column" in said[0], said[0]
    assert "float32" in said[0], said[0]


def test_the_to_time_guard_is_silent_for_a_leaf_that_reached_the_unit():
    """And silent when there is no plan at all, which is a deferred leaf.

    A conv is deferred because its shape is unknown at `to()` time, so it has
    no plan yet and `to()` has nothing to say; the sentence lands at the first
    forward instead. That is different from a plan that came back empty, and
    the two must not be merged -- which is what makes the `report["plans"]`
    test in the guard load-bearing rather than decorative.
    """
    assert _at_to([{"leaf": "linear", "rows": [_CAST, _KNOWN_NE, _CAST]}]) == []
    assert _at_to([]) == []


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
