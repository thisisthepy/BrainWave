"""Ground truth table for the QNN/HTP backend: `torchnative.export.qnn_ops`.

Pure tests only -- no ExecuTorch import, no device, no artefact. Everything
this file checks is checkable from the module's own source and from the
installed `executorch` wheel at `/Volumes/macMini/caches/qnn-venv` that the
table was built from. `docs/devices/QNNOPS.md` is the sourcing record;
`docs/devices/QNN.md` is the (separate, artefact/device) round this table
does not attempt to redo.

The load-bearing tests are the honesty ones: that `check_leaf` never returns
a bare `False` without a `reason`, that every entry in the supported table
really is a `target =` string extracted from the installed package (checked
directly against that package when it is importable, skipping by name
otherwise), and that the explicit refusal lists are disjoint from the
supported set -- a name in both would mean this module contradicts itself.
"""

import importlib.util
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
sys.path.insert(0, _VENDOR_DIR)

import torchnative.export.qnn_ops as qnn_ops  # noqa: E402
from torchnative.export.qnn_ops import (  # noqa: E402
    EXECUTORCH_VERSION,
    SOURCE_PACKAGE,
    check_leaf,
    constraints,
    not_supported_ops,
    supported_ops,
    to_be_implemented_ops,
)

_QNN_BUILDERS_DIR = (
    "/Volumes/macMini/caches/qnn-venv/lib/python3.13/site-packages/"
    "executorch/backends/qualcomm/builders"
)


def _qnn_builders_available():
    return os.path.isdir(_QNN_BUILDERS_DIR)


def test_supported_ops_is_a_nonempty_frozenset_of_strings():
    ops = supported_ops()
    assert isinstance(ops, frozenset), f"expected frozenset, got {type(ops)}"
    assert len(ops) > 0
    for op in ops:
        assert isinstance(op, str) and op, f"non-string or empty op entry: {op!r}"
    print(f"ok   qnn_ops: supported_ops() is a frozenset of {len(ops)} strings")


def test_supported_ops_matches_a_fresh_extraction_from_the_installed_wheel():
    """The table must not drift from the source it claims -- checked directly.

    Re-derives the same `target = [...]` extraction the module's own
    docstring describes, straight from the installed executorch wheel, and
    diffs it against `supported_ops()`. If this ever goes red it means either
    the table is stale against a newer/older wheel, or the module transcribed
    something wrong -- either way the fix is to re-extract, not to edit the
    assertion.
    """
    if not _qnn_builders_available():
        print(
            "ok   qnn_ops: SKIP supported_ops-vs-wheel cross-check "
            f"({_QNN_BUILDERS_DIR} not present on this machine)"
        )
        return
    extracted = {}
    for name in sorted(os.listdir(_QNN_BUILDERS_DIR)):
        if not (name.startswith("op_") and name.endswith(".py")):
            continue
        text = open(os.path.join(_QNN_BUILDERS_DIR, name), encoding="utf-8").read()
        m = re.search(r"target\s*=\s*\[([^\]]*)\]", text)
        if not m:
            continue
        for op in re.findall(r'"([^"]+)"', m.group(1)):
            extracted[op] = name
    table = supported_ops()
    missing_from_table = sorted(set(extracted) - table)
    extra_in_table = sorted(table - set(extracted))
    assert not missing_from_table, (
        f"the installed wheel registers these ops but qnn_ops.supported_ops() "
        f"does not: {missing_from_table}"
    )
    assert not extra_in_table, (
        f"qnn_ops.supported_ops() claims these ops but the installed wheel's "
        f"op_*.py builders do not register them: {extra_in_table}"
    )
    print(
        f"ok   qnn_ops: supported_ops() agrees with a fresh extraction from "
        f"{len(extracted)} op_*.py builders in the installed wheel"
    )


def test_source_package_names_the_exact_installed_version():
    assert EXECUTORCH_VERSION in SOURCE_PACKAGE
    assert os.path.isdir(
        "/Volumes/macMini/caches/qnn-venv/lib/python3.13/site-packages/"
        f"executorch-{EXECUTORCH_VERSION}.dist-info"
    ) or not _qnn_builders_available(), (
        "SOURCE_PACKAGE claims a version whose dist-info is not on this machine"
    )
    print(f"ok   qnn_ops: SOURCE_PACKAGE names executorch=={EXECUTORCH_VERSION}")


def test_the_refusal_lists_are_disjoint_from_the_supported_set():
    """A name cannot be both accepted and refused -- that would be self-contradiction."""
    supported = supported_ops()
    not_supported = set(not_supported_ops())
    tbi = set(to_be_implemented_ops())
    overlap_a = supported & not_supported
    overlap_b = supported & tbi
    overlap_c = not_supported & tbi
    assert not overlap_a, f"in both supported_ops and not_supported_ops: {overlap_a}"
    assert not overlap_b, f"in both supported_ops and to_be_implemented_ops: {overlap_b}"
    assert not overlap_c, f"in both not_supported_ops and to_be_implemented_ops: {overlap_c}"
    print("ok   qnn_ops: supported/not-supported/to-be-implemented are pairwise disjoint")


def test_not_supported_ops_gives_a_reason_for_every_entry():
    for op, reason in not_supported_ops().items():
        assert isinstance(reason, str) and reason.strip(), f"{op} has no reason"
    print(f"ok   qnn_ops: not_supported_ops() names a reason for all {len(not_supported_ops())} entries")


def test_check_leaf_never_returns_a_bare_false_without_a_reason():
    """The spec's central constraint: refusal is always a reason string."""
    probes = [
        "aten.add.Tensor",             # supported
        "aten._embedding_bag.default",  # explicitly not supported
        "aten.median.default",          # to-be-implemented
        "aten.totally_made_up_op.default",  # absent from every table
    ]
    for op in probes:
        verdict = check_leaf(op)
        assert isinstance(verdict, dict), f"{op}: verdict is not a dict: {verdict!r}"
        assert "accepted" in verdict and "reason" in verdict, f"{op}: {verdict!r}"
        assert isinstance(verdict["accepted"], bool), f"{op}: accepted is not bool"
        assert isinstance(verdict["reason"], str) and verdict["reason"].strip(), (
            f"{op}: reason is missing or empty: {verdict!r}"
        )
        if not verdict["accepted"]:
            # never a bare False -- the reason must say WHY.
            assert len(verdict["reason"]) > 10, f"{op}: reason too thin to be real: {verdict!r}"
    print("ok   qnn_ops: check_leaf always returns a non-empty reason, accepted or not")


def test_check_leaf_agrees_with_the_three_tables():
    assert check_leaf("aten.add.Tensor")["accepted"] is True
    assert check_leaf("aten._embedding_bag.default")["accepted"] is False
    assert check_leaf("aten.median.default")["accepted"] is False
    assert check_leaf("aten.no_such_op.default")["accepted"] is False
    print("ok   qnn_ops: check_leaf's verdict matches supported/not-supported/to-be-implemented")


def test_check_leaf_flags_dtypes_qnn_has_no_native_type_for():
    unsupported = check_leaf("aten.add.Tensor", dtype="complex64")
    assert unsupported["accepted"] is False
    assert "complex64" in unsupported["reason"]

    downcast = check_leaf("aten.add.Tensor", dtype="float64")
    assert downcast["accepted"] is True
    assert "float32" in downcast["reason"], "float64 downcast target must be named"

    plain = check_leaf("aten.add.Tensor", dtype="float32")
    assert plain["accepted"] is True
    print("ok   qnn_ops: check_leaf distinguishes unsupported, downcast, and native dtypes")


def test_constraints_names_what_it_did_not_verify():
    c = constraints()
    assert "static_shape" in c
    assert "UNVERIFIED" in c["static_shape"], (
        "constraints()['static_shape'] must say plainly that no numeric "
        "shape limit was sourced, per the task's ban on guessing entries "
        "into the table"
    )
    assert "quantization" in c
    assert "unquantized_dtypes" in c and "quantized_dtypes" in c
    # quantized path is strictly narrower -- no int64, no bool, no float.
    assert "int64" not in c["quantized_dtypes"]
    assert "bool" not in c["quantized_dtypes"]
    print("ok   qnn_ops: constraints() states its own UNVERIFIED gap by name")


def test_this_repository_s_own_qnn_partitioner_refusal_lists_are_not_contradicted():
    """`torchnative/export/qnn.py` names two refusal reasons of its own kind
    (device/host refusal). This table is about a different axis (per-op), so
    it must not accidentally re-derive or contradict host-refusal machinery
    that already exists in this repo -- it only ever talks about op names.
    """
    import torchnative.export.qnn as qnn_module

    assert not hasattr(qnn_module, "supported_ops"), (
        "torchnative.export.qnn already defines supported_ops -- qnn_ops "
        "would be a silently-shadowing duplicate rather than a new capability"
    )
    print("ok   qnn_ops: does not duplicate a supported_ops already exported by torchnative.export.qnn")


if __name__ == "__main__":
    failures = 0
    for _name, _fn in sorted(list(globals().items())):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        try:
            _fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {_name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
