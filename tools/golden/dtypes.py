"""Dtype metadata shared by the golden comparison harness.

DESIGN.md §5 names dtype promotion and quiet numeric drift as the primary
risk of building `torch._C` on top of candle. This module is the single
place that says, for a given dtype name, how close two results are allowed
to be before the harness calls it a mismatch.

Every name in ``TOLERANCES`` must exist as an attribute with the same
spelling on *both* ``torch`` and the built ``_C`` module -- see
docs/design/TORCH_C.md §1 ("dtype 은 파이썬 상수가 아니라 _C 가 소유하는 타입이다")
for why the shim's dtype names were chosen to match torch's exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


# Reasoning per dtype: integer dtypes must match exactly (atol=rtol=0 makes
# `values_close` degenerate to `==`). Floating dtypes get a tolerance sized
# to roughly one ULP at magnitude ~1 for that format, so a real rounding
# *direction* disagreement between candle and torch still trips the check.
TOLERANCES: dict[str, Tolerance] = {
    "float64": Tolerance(atol=1e-9, rtol=1e-9),
    "float32": Tolerance(atol=1e-5, rtol=1e-5),
    "float16": Tolerance(atol=5e-3, rtol=5e-3),
    "bfloat16": Tolerance(atol=6e-2, rtol=6e-2),
    "float8_e4m3fn": Tolerance(atol=2e-2, rtol=2e-2),
    "int64": Tolerance(atol=0.0, rtol=0.0),
    "int32": Tolerance(atol=0.0, rtol=0.0),
    "int16": Tolerance(atol=0.0, rtol=0.0),
    "uint8": Tolerance(atol=0.0, rtol=0.0),
    "uint32": Tolerance(atol=0.0, rtol=0.0),
}

# Dtypes the harness exercises by default.
#
# `float8_e4m3fn` was excluded from 2026-08-24 until docs/numerics/FLOAT8B.md, on the
# stated ground that construction hung on both sides independently. **That
# ground was wrong, and docs/numerics/FLOAT8.md had already disproved it**: construction
# works on both sides and always did. The hang was in `to_dtype(F64)` --
# candle 0.11.0's `WithDType for f8e4m3::to_f64` recursing into itself, which
# release-mode LLVM collapses to `.L1: jmp .L1` -- and construction never calls
# it. An exclusion reason nobody could check outlived the fact it named.
#
# It is included now because the two things that made it uncheckable are gone
# (docs/numerics/FLOAT8B.md): 48 ops that computed answers upstream refuses to produce,
# and 37 that hung. Every op now either matches upstream's value or matches
# upstream's refusal, so `expect="match"` is a real question for this dtype --
# a case where one side refuses and the other computes is a SILENT DIVERGENCE
# failure, which is exactly the check that was missing.
DEFAULT_DTYPES: tuple[str, ...] = (
    "float64",
    "float32",
    "float16",
    "bfloat16",
    "int64",
    "int32",
    "int16",
    "uint8",
    "uint32",
    "float8_e4m3fn",
)

# Nothing is excluded. The key is kept -- with its shape and its consumer in
# `compare.py` intact -- because an empty dict is the honest state and a deleted
# mechanism is not: the next dtype that needs excluding should have to write a
# reason here, and reasons here go stale (see the note above `DEFAULT_DTYPES`).
EXCLUDED_DTYPES: dict[str, str] = {}


def dtype_name(dtype_obj) -> str:
    """`torch.float32` and `_C.float32` both repr as ``"torch.float32"``;
    normalize either to the bare name used as a key above."""
    s = str(dtype_obj)
    return s[len("torch.") :] if s.startswith("torch.") else s


def torch_dtype(torch_module, name: str):
    return getattr(torch_module, name)


def c_dtype(c_module, name: str):
    return getattr(c_module, name)


def tolerance_for(name: str) -> Tolerance:
    # Unknown dtype: fail closed (require exact match) rather than silently
    # let something new drift past under a guessed tolerance.
    return TOLERANCES.get(name, Tolerance(atol=0.0, rtol=0.0))
