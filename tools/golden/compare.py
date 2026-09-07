#!/usr/bin/env python3
"""Golden comparison harness: does `torch._C` compute the same answer as
upstream torch?

DESIGN.md §5 names the risk this exists to catch: rebuilding dtype
promotion, broadcasting, and view/stride aliasing on top of candle can
silently diverge from torch's answers, and linking + importing proves
nothing about whether the numbers agree. This harness asks `_C` what it has
implemented (`_C._aten_implemented()`) -- never a hardcoded list, so
coverage grows automatically as rust/torch_c grows -- and for every op it
finds, runs a battery of dtype/shape/boundary-value cases against both
`_C._aten_dispatch(...)` and the matching `torch.ops.aten.*` overload,
comparing dtype, shape, and value.

Usage
-----
Needs a Python environment with real upstream torch installed (this repo's
scratch venv has it: /Volumes/macMini/caches/spike-venv, torch 2.13.0), and
a built host artefact for `_C` (default host build location per
docs/TORCH_C.md §7: /Volumes/macMini/caches/cargo-target/release/lib_C.dylib,
or built fresh via `cd rust/torch_c && ./pytests/run.sh` first).

    /Volumes/macMini/caches/spike-venv/bin/python tools/golden/compare.py
    /Volumes/macMini/caches/spike-venv/bin/python tools/golden/compare.py \
        --artefact /path/to/lib_C.dylib -v

Exit code is 0 iff every case matched its expectation; non-zero (1)
otherwise. Never grep stdout for a success marker -- this project's own
traceback text has, twice, printed source lines that accidentally matched a
success grep. Read the exit code.

Self-test
---------
`--inject-fault MODE` deliberately corrupts a real, already-computed `_C`
result before comparison, to prove the comparator actually rejects a wrong
answer rather than rubber-stamping everything. It injects into **one
representative case per comparator**, not one case for the whole run --
see the FAULT INJECTION block below for why that distinction was a hole
big enough to hide 404 of 1781 cases in.

`--self-test` runs the whole matrix (every fault mode against every
comparator) and prints a coverage table. Exit code is 0 iff every
comparator rejected every fault it is supposed to reject. That is the
gate; `--inject-fault MODE` keeps its historical "should exit 1"
semantics because a caught fault fails a case.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dtypes as dt
from cases import CASE_BUILDERS, Case
from loader import ShimLoadError, load_shim, resolve_torch_overload


@dataclass
class Outcome:
    case: Case
    passed: bool
    detail: str
    # Whether --inject-fault actually managed to build a corrupted result for
    # this case. Not every fault mode is constructible against every result
    # shape (there is no "last chunk" to pad in a result that is one tensor),
    # and a mode that was never injected must never be read as a mode that was
    # injected and caught.
    fault_applied: bool = False


def _flatten(x) -> list:
    if isinstance(x, list):
        out: list = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [x]


def _values_close(torch_nested, c_nested, atol: float, rtol: float) -> tuple[bool, str]:
    ta, ca = _flatten(torch_nested), _flatten(c_nested)
    if len(ta) != len(ca):
        return False, f"flattened length differs: torch has {len(ta)}, c has {len(ca)}"
    for i, (x, y) in enumerate(zip(ta, ca)):
        if isinstance(x, bool) or isinstance(y, bool):
            if x != y:
                return False, f"index {i}: torch={x!r} c={y!r}"
            continue
        if isinstance(x, int) and isinstance(y, int):
            if x != y:
                return False, f"index {i}: torch={x!r} c={y!r} (integer, exact match required)"
            continue
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            if x != y:
                return False, f"index {i}: torch={x!r} c={y!r}"
            continue
        if math.isnan(xf) or math.isnan(yf):
            if not (math.isnan(xf) and math.isnan(yf)):
                return False, f"index {i}: torch={x!r} c={y!r} (NaN mismatch)"
            continue
        if math.isinf(xf) or math.isinf(yf):
            if xf != yf:
                return False, f"index {i}: torch={x!r} c={y!r} (inf mismatch)"
            continue
        if not math.isclose(xf, yf, rel_tol=rtol, abs_tol=atol):
            return (
                False,
                f"index {i}: torch={x!r} c={y!r} "
                f"(|diff|={abs(xf - yf):.6g} exceeds atol={atol} rtol={rtol})",
            )
    return True, ""


def _summarize(result) -> str:
    try:
        return f"{_as_list(result)!r} dtype={dt.dtype_name(result.dtype)} shape={tuple(result.shape)}"
    except Exception:
        return repr(result)


def _run_one(case: Case, inject_fault: str | None) -> Outcome:
    """Run one case. `fault_applied` on the returned Outcome says whether the
    requested corruption was actually constructible for this result shape --
    without it, "the run stayed green" is ambiguous between "the comparator
    caught nothing" and "nothing was ever injected"."""
    tag_box: list[str] = [""]
    outcome = _run_one_body(case, inject_fault, tag_box)
    outcome.fault_applied = bool(tag_box[0])
    return outcome


def _as_list(result):
    """`result.tolist()`, except for `float8_e4m3fn`, which is read through a
    lossless widening to `float32` first.

    This shim refuses `tolist` on `float8_e4m3fn` outright (docs/FLOAT8.md):
    candle 0.11.0's `f8e4m3 -> f64` conversion does not terminate, so the
    refusal is the only safe answer and it is not going away in this round.
    Without this helper that refusal would make the dtype permanently
    uncomparable *here* -- the harness would raise while reading a result both
    sides had already produced correctly.

    The widening is exact rather than a tolerance: `float8_e4m3fn` has 4
    exponent and 3 mantissa bits and every finite value of it is representable
    in `float32`, so `.to(float32).tolist()` and a working `.tolist()` would
    return the same numbers. It reads the same value on both sides by the same
    route, so it cannot hide a disagreement between them.
    """
    # `_FakeResult` (the fault injector's stand-in) already holds a plain list
    # and has no `.to`; it is never a real float8 buffer, so it reads directly.
    if dt.dtype_name(result.dtype) == "float8_e4m3fn" and hasattr(result, "to"):
        return result.to(_float32_of(result)).tolist()
    return result.tolist()


# The two modules under comparison, registered by `run`/`self_test` so
# `_float32_of` can find the `float32` constant that belongs to a given result.
#
# Not an import and not `type(result).__module__`: the shim is loaded from a
# file path under the module name `_C` (see loader.py, which is deliberately
# `sys.path`-independent) while pyo3 spells its classes `torch._C.TensorBase`,
# so the class's own `__module__` names a module that is *not* the one holding
# the dtype constants. And the two `float32` objects are deliberately not
# interchangeable -- docs/TORCH_C.md §1 is about `_C` owning its own dtype type.
_DTYPE_OWNERS: list = []


def _float32_of(result):
    """`float32` as spelled by whichever module owns `result` -- `torch` for the
    torch side, `_C` for the shim side. Identified by dtype *type*, so a result
    can never be widened with the other module's constant."""
    for module in _DTYPE_OWNERS:
        float32 = getattr(module, "float32", None)
        if float32 is not None and type(float32) is type(result.dtype):
            return float32
    raise RuntimeError(
        f"no registered module owns {type(result.dtype)!r}; _as_list cannot "
        "widen this float8 result"
    )


def _run_one_body(case: Case, inject_fault: str | None, tag_box: list[str]) -> Outcome:
    t_res = t_exc = c_res = c_exc = None
    try:
        t_res = case.run_torch()
    except Exception as e:  # noqa: BLE001 - deliberately broad, this is a probe
        t_exc = e
    try:
        c_res = case.run_c()
    except Exception as e:  # noqa: BLE001
        c_exc = e

    fault_tag = ""
    if inject_fault and t_res is not None and c_res is not None:
        c_res, fault_tag = _corrupt(c_res, inject_fault)
        tag_box[0] = fault_tag

    t_ok, c_ok = t_exc is None, c_exc is None

    if case.expect == "both_error":
        if t_ok or c_ok:
            return Outcome(
                case,
                False,
                f"expected both sides to refuse; torch_ok={t_ok} c_ok={c_ok} "
                f"(torch={_summarize(t_res) if t_ok else t_exc!r}, "
                f"c={_summarize(c_res) if c_ok else c_exc!r})",
            )
        return Outcome(case, True, f"both refused as expected (torch={t_exc!r}, c={c_exc!r})")

    if case.expect == "c_error":
        if not (t_ok and not c_ok):
            extra = " -- gap appears CLOSED: both sides now succeed, promote this case to expect=match and diff real values" if (t_ok and c_ok) else ""
            return Outcome(
                case,
                False,
                f"expected torch to succeed and c to refuse (known gap: {case.note}); "
                f"got torch_ok={t_ok} c_ok={c_ok}{extra}",
            )
        return Outcome(case, True, f"known gap still present: c refused ({c_exc!r})")

    if case.expect == "torch_error":
        if not (c_ok and not t_ok):
            extra = " -- gap appears CLOSED: both sides now succeed" if (t_ok and c_ok) else ""
            return Outcome(
                case,
                False,
                f"expected c to succeed and torch to refuse (known gap: {case.note}); "
                f"got torch_ok={t_ok} c_ok={c_ok}{extra}",
            )
        return Outcome(case, True, f"known gap still present: torch refused ({t_exc!r})")

    if case.expect == "diverge":
        # Both sides compute, they disagree, and that disagreement is a kernel
        # bug we have measured and not yet fixed.
        #
        # This case arrived as a deliberately red `expect="match"`, which keeps
        # the bug visible at the cost of the gate: a suite that exits 1 by
        # design cannot tell anyone apart the divergence it knows about from the
        # one they just introduced, and a suite read that way stops being read.
        # Same reasoning as KNOWN_GAP for comparators -- record it, surface it
        # every run, and fail if it silently heals.
        if not (t_ok and c_ok):
            return Outcome(
                case,
                False,
                f"expected both sides to compute and disagree (known divergence: "
                f"{case.note}); got torch_ok={t_ok} c_ok={c_ok} "
                f"(torch={t_exc!r}, c={c_exc!r})",
            )
        if _as_list(t_res) == _as_list(c_res):
            return Outcome(
                case,
                False,
                f"divergence appears CLOSED -- both sides now agree. Promote this case "
                f"to expect=match and drop the note. Was: {case.note}",
            )
        return Outcome(
            case,
            True,
            f"known divergence still present: torch={_as_list(t_res)!r} c={_as_list(c_res)!r}",
        )

    # expect == "match"
    if t_ok != c_ok:
        refusing_side = "torch" if not t_ok else "c"
        computing_side = "c" if not t_ok else "torch"
        exc = t_exc if not t_ok else c_exc
        val = c_res if not t_ok else t_res
        return Outcome(
            case,
            False,
            f"SILENT DIVERGENCE: {refusing_side} raised {exc!r} but {computing_side} "
            f"computed a value: {_summarize(val)}",
        )
    if not t_ok and not c_ok:
        return Outcome(
            case, True, f"both refused (unregistered as expected, but consistent): torch={t_exc!r} c={c_exc!r}"
        )

    prefix = f"[{fault_tag}] " if fault_tag else ""

    if case.value_check is not None:
        # Ops whose result isn't a plain value-comparable tensor (a Python
        # bool from is_floating_point, uninitialized memory from empty), and
        # ops that need a *stronger* check than the default pipeline can make
        # (the RNG ops, whose comparator seeds both sides and demands
        # bit-for-bit agreement rather than a tolerance), supply their own
        # checker instead of the default dtype/shape/value pipeline below. See
        # cases.py for what each one actually checks and why.
        ok, detail = case.value_check(t_res, c_res)
        if not ok:
            return Outcome(case, False, f"{prefix}{detail}")
        return Outcome(case, True, f"{prefix}{detail}{_uncaught_suffix(case, fault_tag, inject_fault)}")

    t_dtype = dt.dtype_name(t_res.dtype)
    c_dtype = dt.dtype_name(c_res.dtype)
    if t_dtype != c_dtype:
        return Outcome(
            case,
            False,
            f"{prefix}dtype mismatch: torch={t_dtype} c={c_dtype} "
            f"(torch value={_as_list(t_res)!r}, c value={_as_list(c_res)!r})",
        )

    t_shape = tuple(int(x) for x in t_res.shape)
    c_shape = tuple(int(x) for x in c_res.shape)
    if t_shape != c_shape:
        return Outcome(case, False, f"{prefix}shape mismatch: torch={t_shape} c={c_shape}")

    tol = dt.tolerance_for(t_dtype)
    ok, detail = _values_close(_as_list(t_res), _as_list(c_res), tol.atol, tol.rtol)
    if not ok:
        return Outcome(
            case,
            False,
            f"{prefix}value mismatch ({detail}); torch={_as_list(t_res)!r} c={_as_list(c_res)!r} dtype={t_dtype}",
        )
    return Outcome(
        case, True, f"dtype={t_dtype} shape={t_shape}{_uncaught_suffix(case, fault_tag, inject_fault)}"
    )


# ============================================================================
# FAULT INJECTION
# ============================================================================
#
# What this used to be, and why it was not enough.
#
# The original `--inject-fault` corrupted exactly ONE result in the whole run:
# the first `expect="match"` case with `value_check is None`. The skip of
# `value_check` cases was correct when it was written -- at that point the only
# three custom checkers were `empty` (uninitialized memory: there IS no correct
# value, so a value fault legitimately must not be caught), `is_floating_point`
# (a plain Python bool with no `.tolist()` for `_corrupt` to touch), and
# `randint` (a random draw). Corrupting those would have produced a false
# "COMPARATOR BUG" report, so steering around them was the right call.
#
# It stopped being the right call as the harness grew. `value_check` is now how
# every multi-result op is compared -- (values, indices), (output, logsumexp),
# (out, mean, rstd), a list of chunks -- and those checkers do real numeric
# work. Measured on this tree: 404 of 1781 cases (22.7%), across 9 distinct
# comparators, sat behind that skip and were never self-tested. `--inject-fault`
# exiting 1 proved that ONE comparator on ONE case rejects ONE kind of wrong
# answer. It said nothing about the other nine.
#
# So the injection now works per COMPARATOR, not per run, and the fault modes
# are shaped after what a plausible wrong implementation of each op would
# actually get wrong:
#
#   value / value-last   a wrong number, in the first or the last member of a
#                        multi-result (the `indices` half of a pair, `rstd` of
#                        a layer-norm triple, the last chunk of a split)
#   shape / shape-last   a wrong shape, same two positions
#   dtype / dtype-last   a wrong dtype, same two positions. `-last` is the one
#                        that says whether `logsumexp`'s float32-under-float16
#                        asymmetry is really being checked
#   permute              the FIRST member's elements reordered and nothing
#                        else. In a (values, indices) pair that is not an
#                        ordering fault at all, it is a PAIRING fault: value
#                        i now claims index j. A multiset comparator catches
#                        that and should
#   permute-all          every member reordered in lockstep -- the real
#                        ordering fault, and the one a multiset comparator is
#                        entitled to ignore. Splitting these two apart was not
#                        planned: the first table run flagged `permute` as
#                        caught by `_topk_multiset_check`, which the
#                        blindness table said was impossible, and the mode was
#                        wrong rather than the table
#   constant             every element collapsed to the first -- the shape a
#                        broken RNG takes, and what a pure range check misses
#   chunk-count          a chunk dropped
#   chunk-pad            the last chunk PADDED to full width instead of left
#                        short. docs/GPT2.md names this as the most plausible
#                        misimplementation of `split`, and it is invisible to
#                        any element-by-element comparison
#
# A mode that is not constructible against a given result shape reports itself
# as such (`fault_applied=False`) instead of silently passing.

FAULT_MODES: tuple[str, ...] = (
    "value",
    "value-last",
    "shape",
    "shape-last",
    "dtype",
    "dtype-last",
    "permute",
    "permute-all",
    "constant",
    "chunk-count",
    "chunk-pad",
)

# (comparator, mode) pairs where NOT catching the fault is the documented,
# intended behaviour. These are not gaps -- each one is a promise the
# comparator deliberately declines to make, and the reason is quoted from the
# comparator's own docstring in cases.py. `--self-test` fails if one of these
# is unexpectedly caught, because that means this table has gone stale.
BLIND_BY_DESIGN: dict[tuple[str, str], str] = {
    ("_dtype_shape_only_check", "value"): "aten.empty returns uninitialized memory -- there is no correct value to diff, only dtype and shape are meaningful",
    ("_dtype_shape_only_check", "value-last"): "same: uninitialized memory has no correct value",
    ("_dtype_shape_only_check", "permute"): "same: uninitialized memory has no correct order",
    ("_dtype_shape_only_check", "permute-all"): "same: uninitialized memory has no correct order",
    ("_dtype_shape_only_check", "constant"): "same: uninitialized memory has no correct distribution",
    # `_range_check` held three entries here (permute, permute-all, constant),
    # all saying the same thing: the sequence a random op produces is
    # deliberately unchecked, so reordering it or replacing it with a constant
    # in range went uncaught. That comparator is gone -- cases.py records why
    # -- and `randint`/`randperm` now run on `_rng_stream_check`, which catches
    # all three. These are not entries that rotted; they are entries whose
    # blindness was removed.
    ("_dtype_shape_stride_only_check", "value"): "aten.empty_strided returns uninitialized memory -- there is no correct value to diff. Unlike _dtype_shape_only_check this one is NOT blind to order: it compares the stride, which is the one argument that distinguishes empty_strided from empty",
    ("_dtype_shape_stride_only_check", "value-last"): "same: uninitialized memory has no correct value",
    ("_topk_multiset_check", "permute-all"): "upstream's own order under ties / sorted=False is a partition artefact, not a promise -- pinning it would pin an implementation detail. Note this is `permute-all` (values and indices moved together); plain `permute` breaks the value/index pairing and IS caught",
}

# (comparator, mode) pairs that are NOT caught and are NOT intended: real holes
# this matrix finds. Recorded here so `--self-test` stays usable as a gate while
# each hole stays loudly visible, printed as `GAP` rather than `blind` and
# re-listed at the end of every run. `_verdict_for` fails the run if one of
# these is ever caught, so a fixed gap cannot be left parked here.
#
# Empty because the three it held are fixed: the indices half of a pair now has
# its dtype and its shape compared, so a shim returning int32 or reshaped
# indices no longer passes. The entries did not rot -- the self-test refused to
# go green until they were removed, which is the whole reason for parking them
# here rather than in a comment.
KNOWN_GAP: dict[tuple[str, str], str] = {}


class _FakeDtype:
    def __init__(self, s):
        self._s = s

    def __str__(self):
        return self._s


class _FakeResult:
    def __init__(self, values, dtype, shape):
        self._values = values
        self.dtype = dtype
        self.shape = shape

    def tolist(self):
        return self._values

    def stride(self):
        """The fault injector's stand-in answers the tensor protocol it is
        asked for. `_dtype_shape_stride_only_check` (empty / empty_strided)
        asks for a stride, and a stand-in that raised `AttributeError` would
        make the self-test fail for a reason unrelated to the injected fault.
        Contiguous, derived from `shape`, so an injected SHAPE fault shows up
        in the stride too rather than being masked by it."""
        stride, acc = [], 1
        for extent in reversed([int(x) for x in self.shape]):
            stride.append(acc)
            acc *= extent
        return tuple(reversed(stride))


def comparator_name(case: Case) -> str:
    """Which checker decides this case. `None` means compare.py's own
    dtype/shape/value pipeline; otherwise the function's own name, with the
    `.<locals>.check` tail of the closure-built ones (`_range_check`,
    `_rng_stream_check`) folded back onto the factory that made them."""
    vc = case.value_check
    if vc is None:
        return "<default pipeline>"
    name = getattr(vc, "__qualname__", None) or getattr(vc, "__name__", None) or repr(vc)
    return name.split(".")[0]


def _is_tensorish(x) -> bool:
    return hasattr(x, "tolist") and hasattr(x, "dtype") and hasattr(x, "shape")


def _decompose(result) -> tuple[str, list]:
    """Split a result into its comparable members. `("tuple", [values,
    indices])`, `("list", [chunk, chunk, ...])`, `("tensor", [t])`,
    `("scalar", [x])`, or `("unknown", [])` when nothing can be built."""
    if _is_tensorish(result):
        return "tensor", [result]
    if isinstance(result, (bool, int, float)):
        return "scalar", [result]
    if isinstance(result, (list, tuple)):
        parts = list(result)
        if parts and all(_is_tensorish(p) for p in parts):
            return ("list" if isinstance(result, list) else "tuple"), parts
        return "unknown", []
    try:
        parts = list(result)
    except TypeError:
        return "unknown", []
    if parts and all(_is_tensorish(p) for p in parts):
        return "list", parts
    return "unknown", []


def _recompose(kind: str, members: list):
    if kind in ("tensor", "scalar"):
        return members[0]
    if kind == "list":
        return list(members)
    return tuple(members)


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _reshape(flat: list, shape) -> Any:
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat[: int(shape[0])])
    step = _numel(shape[1:])
    return [_reshape(flat[i * step : (i + 1) * step], shape[1:]) for i in range(int(shape[0]))]


def _wrong(x):
    """A value the correct answer is not. Deliberately far away rather than
    one ULP off: this is proving the comparator is awake at all, and a
    perturbation inside the dtype's own tolerance would prove the opposite of
    what it looks like it proves."""
    if isinstance(x, bool):
        return not x
    if isinstance(x, int):
        return x + 1000
    return float(x) + 1000.0


def _set_leaf(values, which: int, fn):
    """Apply `fn` to the first (which=0) or last (which=-1) scalar leaf of a
    nested list, leaving the rest alone. Returns None if there are no leaves."""
    flat = _flatten(values)
    if not flat:
        return None
    target = len(flat) - 1 if which == -1 else 0
    seen = [0]

    def walk(x):
        if isinstance(x, list):
            return [walk(v) for v in x]
        i = seen[0]
        seen[0] += 1
        return fn(x) if i == target else x

    return walk(values)


def _corrupt_member(target, base_mode: str, last: bool):
    """Corrupt one tensor-like member. Returns None if this mode cannot be
    built for this member (e.g. `permute` on a single element)."""
    values = _as_list(target)
    shape = tuple(int(v) for v in target.shape)
    if base_mode == "value":
        new = _set_leaf(values, -1 if last else 0, _wrong)
        if new is None:
            return None
        return _FakeResult(new, target.dtype, shape)
    if base_mode == "shape":
        return _FakeResult(values, target.dtype, shape + (1,))
    if base_mode == "dtype":
        other = "torch.int16" if "int" not in str(target.dtype) else "torch.float32"
        return _FakeResult(values, _FakeDtype(other), shape)
    if base_mode == "permute":
        flat = _flatten(values)
        if len(flat) < 2 or flat == flat[::-1]:
            return None
        return _FakeResult(_reshape(flat[::-1], shape), target.dtype, shape)
    if base_mode == "constant":
        flat = _flatten(values)
        if len(flat) < 2 or all(v == flat[0] for v in flat):
            return None
        return _FakeResult(_reshape([flat[0]] * len(flat), shape), target.dtype, shape)
    return None


def _corrupt(result, mode: str) -> tuple[Any, str]:
    """Only used by --inject-fault / --self-test, to prove the comparator
    rejects a wrong answer. Returns `(result, "")` unchanged -- never a
    silent pass-through that looks injected -- when the mode does not apply
    to this result's shape."""
    kind, members = _decompose(result)
    if not members:
        return result, ""

    if kind == "scalar":
        # `is_floating_point` -> bool, `_local_scalar_dense` -> int/float.
        # No dtype and no shape to get wrong; the only wrong answer available
        # is a wrong value (and, for the bool, a flipped one).
        if mode in ("value", "value-last"):
            return _wrong(members[0]), f"INJECTED {mode} fault"
        return result, ""

    if mode in ("chunk-count", "chunk-pad"):
        if kind != "list" or len(members) < 2:
            return result, ""
        if mode == "chunk-count":
            return (
                _recompose(kind, members[:-1]),
                "INJECTED chunk-count fault (last chunk dropped)",
            )
        first, last_chunk = members[0], members[-1]
        f_shape = tuple(int(v) for v in first.shape)
        l_flat = _flatten(_as_list(last_chunk))
        if _numel(f_shape) <= len(l_flat):
            # Uniform split: there is nothing to pad, so this mode does not
            # apply to this case. Say so rather than pretend.
            return result, ""
        pad_value = l_flat[-1] if l_flat else 0
        padded_flat = l_flat + [pad_value] * (_numel(f_shape) - len(l_flat))
        padded = _FakeResult(_reshape(padded_flat, f_shape), last_chunk.dtype, f_shape)
        return (
            _recompose(kind, members[:-1] + [padded]),
            "INJECTED chunk-pad fault (last chunk padded to full width, not left short)",
        )

    if mode == "permute-all":
        # Every member reordered the same way: in a (values, indices) pair
        # that keeps each value with its own index and only moves the pair,
        # which is what "wrong order, right elements" actually means.
        new_members = []
        changed = False
        for m in members:
            fake = _corrupt_member(m, "permute", last=False)
            if fake is None:
                new_members.append(m)
            else:
                new_members.append(fake)
                changed = True
        if not changed:
            return result, ""
        return _recompose(kind, new_members), "INJECTED permute-all fault (every member reordered in lockstep)"

    last = mode.endswith("-last")
    base = mode[: -len("-last")] if last else mode
    if base not in ("value", "shape", "dtype", "permute", "constant"):
        return result, ""
    if last and len(members) == 1 and base in ("shape", "dtype"):
        # A one-member result has no distinct "last member": running this
        # would duplicate the non-`-last` mode and inflate the coverage table
        # with a catch that was already counted.
        return result, ""

    idx = -1 if last else 0
    fake = _corrupt_member(members[idx], base, last)
    if fake is None:
        return result, ""
    members = list(members)
    members[idx] = fake
    return _recompose(kind, members), f"INJECTED {mode} fault"


def _uncaught_suffix(case: Case, fault_tag: str, mode: str | None) -> str:
    """What to say when an injected fault sailed through. Only "COMPARATOR
    BUG" if the comparator was supposed to catch it."""
    if not fault_tag or mode is None:
        return ""
    key = (comparator_name(case), mode)
    if key in BLIND_BY_DESIGN:
        return f" [{fault_tag} not caught -- blind BY DESIGN: {BLIND_BY_DESIGN[key]}]"
    if key in KNOWN_GAP:
        return f" [{fault_tag} not caught -- KNOWN GAP: {KNOWN_GAP[key]}]"
    return f" [{fault_tag} did not get caught -- COMPARATOR BUG]"


def run(artefact: str | None, verbose: bool, inject_fault: str | None) -> int:
    try:
        c_module = load_shim(artefact)
    except ShimLoadError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    import torch  # imported lazily so --help works even without torch installed

    _DTYPE_OWNERS[:] = [torch, c_module]

    implemented = list(c_module._aten_implemented())
    print(f"target={c_module._shim_target()} implemented={implemented}")
    if dt.EXCLUDED_DTYPES:
        for name, reason in dt.EXCLUDED_DTYPES.items():
            print(f"NOTE: dtype {name!r} excluded from all cases -- {reason}")

    # The mirror image of `missing_builders` below: a case builder exists in
    # cases.py for an op that `_aten_implemented()` does NOT (yet) advertise.
    # This is deliberate -- see cases.py's module docstring on pre-seeding
    # builders for ops another change is actively implementing -- and must
    # stay silent (not a failure, doesn't touch the pass/fail count) since
    # it is expected to be non-empty until rust/torch_c catches up. It is
    # still printed so "how much coverage is waiting" is visible at a glance.
    pending_builders = sorted(op for op in CASE_BUILDERS if op not in implemented)
    if pending_builders:
        print(
            f"PENDING: {len(pending_builders)} case builder(s) registered for ops not yet "
            f"in _aten_implemented() -- waiting, not failing: {pending_builders}"
        )

    all_outcomes: list[Outcome] = []
    missing_builders: list[str] = []
    # comparator name -> (case, outcome) for the one case this comparator got
    # the fault injected into. One per COMPARATOR rather than one per run: an
    # injection that only ever lands on compare.py's default pipeline says
    # nothing about the nine other checkers cases.py hands out.
    injected: dict[str, tuple[Case, Outcome]] = {}
    seen_comparators: set[str] = set()

    for op_name in implemented:
        builder = CASE_BUILDERS.get(op_name)
        if builder is None:
            missing_builders.append(op_name)
            continue
        try:
            torch_call = resolve_torch_overload(torch, op_name)
        except ShimLoadError as e:
            all_outcomes.append(
                Outcome(
                    Case(name=f"<resolve {op_name}>", op=op_name, run_torch=lambda: None, run_c=lambda: None),
                    False,
                    f"could not resolve torch overload for {op_name!r}: {e}",
                )
            )
            continue

        cases = builder(torch, c_module, torch_call)
        for case in cases:
            cmp_name = comparator_name(case)
            seen_comparators.add(cmp_name)
            fault_for_this_case = None
            # Keep offering the fault to successive cases of a comparator
            # until one of them can actually be corrupted this way: not every
            # mode is constructible against every result (there is no last
            # chunk to pad in a uniform split). `fault_applied` below is what
            # says whether it landed.
            if inject_fault and case.expect == "match" and cmp_name not in injected:
                fault_for_this_case = inject_fault
            outcome = _run_one(case, fault_for_this_case)
            if fault_for_this_case and outcome.fault_applied:
                injected[cmp_name] = (case, outcome)
            all_outcomes.append(outcome)
            if verbose or not outcome.passed:
                status = "PASS" if outcome.passed else "FAIL"
                print(f"{status} {case.op} :: {case.name} -- {outcome.detail}")

    for op_name in missing_builders:
        print(
            f"FAIL {op_name} :: <no case builder registered> -- "
            f"_aten_implemented() advertises this op but tools/golden/cases.py "
            f"has no entry in CASE_BUILDERS for it. Add one; do not silently skip."
        )

    total = len(all_outcomes) + len(missing_builders)
    failed = sum(1 for o in all_outcomes if not o.passed) + len(missing_builders)
    passed = total - failed

    if inject_fault:
        _print_injection_verdict(inject_fault, seen_comparators, injected)

    print()
    # The gaps are listed here and not only under `--self-test`, because a green
    # SUMMARY is what anyone actually reads. A comparator that cannot see a
    # whole class of wrong answer is part of what this run does not prove, and
    # saying so only where a separate flag is passed hides it from every reader
    # who does not pass it.
    if KNOWN_GAP:
        print(
            f"KNOWN GAP: {len(KNOWN_GAP)} comparator blind spot(s) below -- this run does "
            f"not prove what they cover. Run --self-test for the full matrix."
        )
        for (cmp_name, mode), why in sorted(KNOWN_GAP.items()):
            print(f"  {cmp_name} + {mode}: {why}")
        print()

    # Same reason the gaps above are printed here rather than behind a flag: a
    # green SUMMARY is what anyone actually reads, and these are cases that pass
    # *because* the two sides disagree. Saying so is the price of counting them
    # as passing at all.
    diverging = [o.case for o in all_outcomes if o.case.expect == "diverge" and o.passed]
    if diverging:
        print(
            f"KNOWN DIVERGENCE: {len(diverging)} case(s) pass because the shim and torch "
            f"disagree in a way that is recorded, not fixed. Each fails if they start agreeing."
        )
        for case in diverging:
            print(f"  {case.op}: {case.note}")
        print()

    print(
        f"SUMMARY: {passed}/{total} cases passed, {failed} failed, "
        f"ops covered={len(implemented)}, pending case builders={len(pending_builders)}"
    )

    return 1 if failed else 0


def _verdict_for(cmp_name: str, mode: str, outcome: Outcome | None) -> tuple[str, str]:
    """(verdict, note). `outcome is None` means the fault was never
    constructible for anything this comparator judges."""
    key = (cmp_name, mode)
    if outcome is None:
        return "n/a", "fault not constructible for this result shape"
    if not outcome.passed:
        if key in BLIND_BY_DESIGN:
            return "CAUGHT", "UNEXPECTED -- BLIND_BY_DESIGN says this should not be caught; that table is stale"
        if key in KNOWN_GAP:
            return "CAUGHT", "UNEXPECTED -- KNOWN_GAP says this should not be caught; that entry is fixed, remove it"
        return "CAUGHT", ""
    if key in BLIND_BY_DESIGN:
        return "blind", BLIND_BY_DESIGN[key]
    if key in KNOWN_GAP:
        return "GAP", KNOWN_GAP[key]
    return "MISSED", "the comparator accepted a wrong answer"


def _print_injection_verdict(mode: str, seen_comparators: set[str], injected: dict) -> None:
    print()
    print(f"INJECTION VERDICT (--inject-fault {mode}) -- one representative case per comparator:")
    for cmp_name in sorted(seen_comparators):
        entry = injected.get(cmp_name)
        verdict, note = _verdict_for(cmp_name, mode, entry[1] if entry else None)
        where = f"  [{entry[0].op}]" if entry else ""
        tail = f" -- {note}" if note else ""
        print(f"  {verdict:<7} {cmp_name:<24}{where}{tail}")
    print(
        "  (exit code stays 1 whenever any fault was CAUGHT, which is the documented "
        "behaviour of --inject-fault. Use --self-test for the pass/fail gate.)"
    )


def _group_cases_by_comparator(torch, c_module) -> dict[str, list[Case]]:
    groups: dict[str, list[Case]] = {}
    for op_name in c_module._aten_implemented():
        builder = CASE_BUILDERS.get(op_name)
        if builder is None:
            continue
        try:
            torch_call = resolve_torch_overload(torch, op_name)
        except ShimLoadError:
            continue
        for case in builder(torch, c_module, torch_call):
            if case.expect != "match":
                continue
            groups.setdefault(comparator_name(case), []).append(case)
    return groups


def self_test(artefact: str | None, verbose: bool, scan_limit: int) -> int:
    """Every fault mode against every comparator, printed as a table.

    "This comparator is covered" means "at least one deliberately wrong
    answer was rejected by it". Anything else -- a mode that no case could
    express, a mode the comparator declines to check on purpose, a mode it
    should have caught and did not -- is named, not folded into a total.
    """
    try:
        c_module = load_shim(artefact)
    except ShimLoadError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    import torch

    _DTYPE_OWNERS[:] = [torch, c_module]

    print(f"target={c_module._shim_target()}")
    groups = _group_cases_by_comparator(torch, c_module)
    print(f"comparators={len(groups)} over {sum(len(v) for v in groups.values())} expect=match cases")
    print()

    matrix: dict[tuple[str, str], tuple[str, str, str]] = {}
    for cmp_name in sorted(groups):
        group = groups[cmp_name]
        pending = set(FAULT_MODES)
        for case in group[:scan_limit]:
            if not pending:
                break
            for mode in sorted(pending):
                outcome = _run_one(case, mode)
                if not outcome.fault_applied:
                    continue
                verdict, note = _verdict_for(cmp_name, mode, outcome)
                matrix[(cmp_name, mode)] = (verdict, note, f"{case.op} :: {case.name}")
                pending.discard(mode)
        for mode in pending:
            verdict, note = _verdict_for(cmp_name, mode, None)
            matrix[(cmp_name, mode)] = (verdict, note, "")

    width = max(len(c) for c in groups)
    header = f"{'comparator':<{width}} | " + " | ".join(f"{m:<11}" for m in FAULT_MODES)
    print(header)
    print("-" * len(header))
    for cmp_name in sorted(groups):
        cells = []
        for mode in FAULT_MODES:
            cells.append(f"{matrix[(cmp_name, mode)][0]:<11}")
        print(f"{cmp_name:<{width}} | " + " | ".join(cells))
    print()

    problems: list[str] = []
    uncovered: list[str] = []
    for cmp_name in sorted(groups):
        caught = [m for m in FAULT_MODES if matrix[(cmp_name, m)][0] == "CAUGHT"]
        if not caught:
            uncovered.append(cmp_name)
        for mode in FAULT_MODES:
            verdict, note, where = matrix[(cmp_name, mode)]
            if verdict == "MISSED":
                problems.append(f"{cmp_name} + {mode}: {note} ({where})")
            elif verdict == "CAUGHT" and note:
                problems.append(f"{cmp_name} + {mode}: {note}")
            if verbose and verdict in ("blind", "GAP", "n/a"):
                print(f"NOTE {verdict:<5} {cmp_name} + {mode}: {note}")

    for cmp_name in sorted(groups):
        caught = [m for m in FAULT_MODES if matrix[(cmp_name, m)][0] == "CAUGHT"]
        blind = [m for m in FAULT_MODES if matrix[(cmp_name, m)][0] == "blind"]
        gaps = [m for m in FAULT_MODES if matrix[(cmp_name, m)][0] == "GAP"]
        line = f"{cmp_name}: caught {len(caught)}/{len(FAULT_MODES)} ({', '.join(caught) or 'NOTHING'})"
        if blind:
            line += f"; blind by design on {', '.join(blind)}"
        if gaps:
            line += f"; KNOWN GAP on {', '.join(gaps)}"
        print(line)

    print()
    live_gaps = [
        (cmp_name, mode)
        for cmp_name in sorted(groups)
        for mode in FAULT_MODES
        if matrix[(cmp_name, mode)][0] == "GAP"
    ]
    if live_gaps:
        print(
            f"KNOWN GAP: {len(live_gaps)} (comparator, fault) pair(s) let a wrong answer through "
            "on purpose-of-record, not by design. These do not fail the run, but they are real:"
        )
        for cmp_name, mode in live_gaps:
            print(f"  {cmp_name} + {mode}: {matrix[(cmp_name, mode)][1]}")
        print()

    if uncovered:
        print(
            "COMPARATOR NEVER EXERCISED: "
            + ", ".join(uncovered)
            + " -- no fault mode was rejected by it. Either the modes cannot express "
            "a wrong answer for that result shape, or the comparator checks nothing."
        )
    for p in problems:
        print(f"PROBLEM: {p}")

    ok = not problems and not uncovered
    print()
    print(
        f"SELF-TEST: {'PASS' if ok else 'FAIL'} -- {len(groups)} comparators x "
        f"{len(FAULT_MODES)} fault modes, {len(problems)} problem(s), "
        f"{len(uncovered)} comparator(s) never exercised"
    )
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artefact", default=None, help="path to the built lib_C.dylib/.so (default: TORCH_C_ARTEFACT env var, then docs/TORCH_C.md §7 default host path)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every case, not just failures")
    parser.add_argument(
        "--inject-fault",
        choices=list(FAULT_MODES),
        default=None,
        help=(
            "self-test: deliberately corrupt one representative result PER COMPARATOR before "
            "comparison, to prove each comparator catches it. Should always exit non-zero when set "
            "(a caught fault fails its case)."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "run every fault mode against every comparator and print the coverage table. "
            "Exit 0 iff every comparator rejected every fault it is supposed to reject. "
            "This is the gate; --inject-fault is the single-mode probe."
        ),
    )
    parser.add_argument(
        "--self-test-scan",
        type=int,
        default=120,
        help="how many cases per comparator --self-test may try before declaring a mode inexpressible (default 120)",
    )
    args = parser.parse_args()
    if args.self_test:
        if args.inject_fault:
            parser.error("--self-test runs every mode; do not also pass --inject-fault")
        return self_test(args.artefact, args.verbose, args.self_test_scan)
    return run(args.artefact, args.verbose, args.inject_fault)


if __name__ == "__main__":
    raise SystemExit(main())
