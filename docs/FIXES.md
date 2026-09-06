# FIXES.md

A backlog of small, precisely located defects other rounds diagnosed but
could not fix because the files belonged to someone else (`rust/torch_c/src/aten.rs`,
`methods.json`, `overloads.json`, `tools/golden/cases.py` this round).
Each item below records what upstream actually does (measured, not
inferred), what changed, and how it was proven against upstream.

## 1. `Tensor.floor_divide` -- missing `methods.json` row

`x.floor_divide(y)` raised `NotImplementedError: not implemented in torch._C
shim: TensorBase.floor_divide` even though `torch.floor_divide(x, y)` already
worked and the kernel (`floor_divide_default`/`floor_divide_scalar` in
`aten.rs`) already existed. The gap was purely `methods.json` having no
`floor_divide` entry at all -- nothing looks in `overloads.json` for a
receiver method.

Fix: added `floor_divide` to `methods.json` with the same two schemas
`overloads.json` already carries (`aten::floor_divide(Tensor, Tensor)`
first, `.Scalar` second -- `div`'s order). No Rust change.

Proven: `x.floor_divide(2)` and `x.floor_divide(torch.tensor([2.0,2.0,2.0]))`
now match `torch.floor_divide`'s own answers exactly, on the shim, for the
same operands upstream computes `[3, -4, 4]` for `[7, -7, 8] // 2`.

## 2. `Tensor.histc` -- missing `methods.json` row

Same shape as #1: `x.histc(...)` raised the same `NotImplementedError`,
`torch.histc(...)` already worked. Added `histc` to `methods.json` with its
one schema. No Rust change.

Proven: `torch.tensor([1,2,2,3]).histc(bins=4, min=0, max=4)` now answers
`[0, 1, 2, 1]`, matching upstream exactly.

## 3. `F.adaptive_avg_pool2d(x, 2)` with a bare int -- NOT a fixable defect here

Measured: upstream's `F.adaptive_avg_pool2d(x, 2)` and `F.adaptive_avg_pool2d(x,
(2, 2))` do produce identical tensors -- the int really does normalise to a
pair. But that normalising happens in `torch.nn.functional.adaptive_avg_pool2d`,
the Python wrapper, **before** it calls `torch.ops.aten.adaptive_avg_pool2d.default`.
Measured directly: `torch.ops.aten.adaptive_avg_pool2d.default(x, 2)` itself
raises `RuntimeError` from pybind11's own arg parser ("Expected a value of
type 'List[int]' ... instead found type 'int'") -- the raw aten op refuses a
bare int too.

An initial attempt widened the check inside `aten.rs`'s
`adaptive_avg_pool2d_default` (the function `aten.adaptive_avg_pool2d.default`
dispatches to) to accept a length-1 `output_size` and repeat it. That made
`F.adaptive_avg_pool2d(x, 2)` work, but it also made the shim's own aten op
more lenient than upstream's aten op -- confirmed by `tools/golden/compare.py`,
which calls `torch.ops.aten.adaptive_avg_pool2d.default` directly and reported
a SILENT DIVERGENCE: upstream refused the direct call, the shim computed a
value. That is the wrong layer to fix this at.

The actual gap is in `rust/torch_c/src/bootstrap.py`'s `adaptive_avg_pool2d`
composite (around line 6974): `def adaptive_avg_pool2d(input, output_size):
return dispatch("aten.adaptive_avg_pool2d.default", input, output_size)` --
it forwards `output_size` unnormalised, with no int-to-pair step, unlike
upstream's own `torch.nn.functional.adaptive_avg_pool2d`. `bootstrap.py` is
outside this round's territory (`aten.rs`, `methods.json`, `overloads.json`,
`tools/golden/cases.py`), so this is left exactly where it stood, the same
shape as item 8 below.

The `aten.rs` change was reverted (kept as the strict length-2 check, matching
upstream's own aten op), with a doc comment recording the measurement and
where the real fix belongs. No golden case was added for the bare-int form,
since one was tried and correctly failed (it was testing the wrong layer).

## 4. `uint8 // -3` and `uint8 < -3` -- scalar not wrapped into the dtype

**This one is a real conversion defect**, confirmed by sweeping every scalar
from -300 to 300 against a `uint8` tensor on upstream 2.13.0. Upstream wraps
a negative or overflowing Python `int` scalar into the tensor's dtype's bit
pattern (ordinary two's-complement truncation, `(v as u8)` widened back) before
either the division or the comparison runs:

    v          wraps to (uint8)
    -1         255
    -256       0          (then // 0 raises ZeroDivisionError, like an explicit // 0)
    -257       255        (same as -1: -257 mod 256 == 255)
    -300       212
    0          0
    255        255
    256        0          (then // 0 raises)
    300        44

The shim instead did the arithmetic in `i64` with the raw signed scalar and
only wrapped the *output* into `uint8` afterward, which computes a completely
different (and completely wrong) answer -- e.g. `uint8([10,20,250]) // -3`
answered `[252, 249, 172]` (from dividing by literal `-3` in `i64`) instead of
upstream's `[0, 0, 0]` (from dividing by the wrapped `253`).

Fix: added `wrap_unsigned_scalar(v, dtype)` in `aten.rs` -- `v.rem_euclid(1 <<
bits)` for `UInt8`/`UInt16`/`UInt32` (an identity for every other dtype,
including `UInt64`, whose modulus does not fit in `i64` and was not
measured). Wired into `floor_divide_impl`'s scalar branch (before the
division, so `-256`'s wrapped `0` divisor raises exactly as an explicit `// 0`
does) and into `compare_scalar` (before building the right-hand comparison
tensor).

Proven: re-ran the full -300..300 sweep against real upstream after the fix;
every value in the sweep matches upstream exactly, including both boundary
cases (`-256` raising, `256`/`300` wrapping to `0`/`44`). Golden cases added
to `floor_divide_cases` and `lt_scalar_cases` at four points in that sweep
(`-1`, `-256`, `-300`, `300`).

## 5. `uint8 ** 300` -- missing overflow refusal

Measured: `uint8_tensor ** 256` and `int8_tensor ** 128` (any non-negative
exponent one past the base dtype's integer range) both raise upstream:
`RuntimeError: value cannot be converted to type <ctype>_t without overflow`
-- the same message and wording `overflow()` already produces for
`fill_`/`full`. A *negative* out-of-range exponent (`int8_tensor ** -129`)
instead raises "Integers to negative integer powers are not allowed",
measured to fire first -- the two checks do not interact for the same input.

The shim narrowed the exponent into `i64` with no range check at all, so it
silently computed instead of refusing.

Fix: in `pow_tensor_scalar`, after computing the result dtype, check a
non-negative integer exponent against `int_range(tag)` and return `overflow(tag)`
if it is out of range, before calling `side_from_scalar`/`pow_from_pairs`.

Proven: `uint8([2,3,255]) ** 255` still computes (`[0, 171, 255]`, matching
upstream exactly) and `** 256`/`** 300` now raise the same message upstream
raises, where they previously computed silently. Golden cases added at the
boundary (`255` computes, `256` refuses).

## 6. `x ** 0.3` at `float32` -- real defect, opposite of the doc's own example

`side_from_scalar`'s doc comment (already in `aten.rs`, from an earlier
round) argued that upstream narrows a float exponent into the base's dtype
before computing `pow`, and that doing the same here just trades one ULP
error for another that is not worth chasing (citing `5.0 ** -1.5` as the
pair where this shim's `f64` road is the one closer to a correctly-rounded
oracle). That argument does not hold for the case this item names.

Measured against a 50-digit `mpmath` oracle for the true value of `b ** 0.3`:
for `b` in `{7, 11, 13, 96}` at `float32`, upstream's answer is the
**correctly-rounded** `float32` result of `pow(b, 0.3)` computed with the
Python `0.3` at **full `f64` precision**. Narrowing `0.3` to its nearest
`float32` value first (`0.30000001192092896`, what `side_from_scalar` was
doing) and then computing in `f64` gives a **different, not
correctly-rounded**, answer on all four -- always 1 ULP high. This is not
double rounding from computing in `f32` either: Rust's `f32::powf` (and the
platform's C `powf`) reproduce the same wrong answer the `f64`-then-narrow
road does, because the actual discrepancy is in *which exponent value* is
used, not which precision the final multiply happens at.

So unlike item 6's namesake precedent (`5.0 ** -1.5`, `docs/DEMAND8.md`'s
"marginally closer, leave it" shape), this one is a genuine, consistently
one-directional defect: 4 of 4 bases tested were off, all in the same
direction, all matching the narrowed-exponent theory exactly.

Fix: in `pow_tensor_scalar`, when the result dtype is `Float32` and the
exponent is a Python float (not an int), skip `side_from_scalar`'s narrowing
and keep the exponent at full `f64` precision; only the final result is
narrowed (unchanged, `pow_from_pairs` already does that via `fast_to(storage)`).
`float64` is unaffected (narrowing to `float64` is a no-op either way);
`float16`/`bfloat16` and integer exponents are untouched -- this item names
`float32` specifically and that is the only case the fix touches.

Proven: all four bases (`7, 11, 13, 96`) plus the four already-passing ones
(`3, 5, 0.5, 1.0`, `2.0`, `1.3896484375`) now match upstream exactly. A
broader sweep of 13 bases x 11 exponents (143 pairs) found exactly one
unrelated pre-existing mismatch (`123.456 ** 3.0`, an integer-valued float
exponent, not narrowing-related and not one of the pairs this item measures)
-- consistent with `pow_tensor_scalar_cases`'s own standing note that
upstream's vectorised `float32` `pow` disagrees with itself across tensor
lengths for some inputs (a SLEEF-vs-libm tail effect), which is why the new
golden cases use single-element tensors.

## 7. `torch.fmod` -- no row in the overload table at all

`overloads.json` had no `fmod` entry whatsoever (not even one for
`methods.json` either), so `torch.fmod(...)` raised `AttributeError` rather
than resolving to a missing kernel -- a step earlier in the failure chain
than every other item here. There was also no kernel: `fmod` shares
`remainder`'s general shape (elementwise, promotes, refuses `bool`, narrows
a scalar into the tensor's dtype first) but follows the sign of the
*dividend* rather than the *divisor*, which is exactly what Rust's `%`
already computes with no correction -- unlike `remainder_f64`/`remainder_i64`,
which add one.

Fix:
  - `overloads.json`: added `fmod` with its two schemas that have kernels
    (`Tensor`, `Scalar`). Deliberately **not** the `.Scalar_out`/`.Tensor_out`
    pair upstream also has -- `remainder`'s own `overloads.json` entry once
    reached that decision the same way (`docs/GROUPED_MM.md`'s
    `reach_allow.json` reasoning, repeated in this file's own comment): a
    dead overload key with no dispatch arm counts against
    `reach_allow.json`'s `shape1_dead_overload_keys_ceiling` ratchet, and it
    does not fall, so a new pair of dead keys would need `reach_allow.json`
    edited to accept them -- a file outside this round's territory. `maximum`
    beside it in the same table made the identical call for the identical
    reason.
  - `aten.rs`: added `fmod_f64`/`fmod_i64` (plain `%`, no sign correction)
    and `fmod_op`, reusing `remainder_op`'s plumbing (dtype promotion, `bool`
    refusal, scalar-narrows-into-dtype-first) verbatim since only the
    elementwise function differs. Registered in `IMPLEMENTED`, the kernel-name
    table (`fmod_cpu`, matching upstream's own message), and the dispatch
    match.
  - `tools/golden/cases.py`: `fmod_scalar_cases`/`fmod_tensor_cases`, built by
    taking `remainder`'s own case set and flipping the expectation on every
    quadrant the two conventions disagree on (the four rows printed in the
    doc comment above the cases). One case's expectation was corrected after
    a first attempt copied `remainder.Scalar`'s bool-scalar refusal note
    verbatim and mislabeled it "upstream refuses too" -- measured, upstream
    actually *computes* for `fmod(bool_tensor, 2)` (`int64`) and
    `fmod(bool_tensor, 2.0)` (`float32`), exactly as it does for
    `remainder.Scalar`'s own documented gap, and the shim's refusal there is
    this shim's, not upstream's -- so the golden case expects `c_error`, not
    `both_error`.
  - `rust/torch_c/pytests/test_shim.py`: two pinned counts moved because
    `fmod.Tensor`/`fmod.Scalar` are newly-reachable, newly-tagged-`core`
    kernels:
      - `test_core_ops_and_op_tags_agree`'s `tag_core_count`: `108 -> 110`
        (`torch.ops.aten.fmod.Tensor.tags` is `[core, pointwise,
        pt2_compliant_tag]`, read off a real torch, same three tags
        `remainder`'s two overloads already carry).
      - `test_schema_text_survives_the_round_trip_through_the_transcribed_tables`'s
        `len(keys)`: `293 -> 295` (`fmod.Tensor`/`fmod.Scalar` are two
        genuinely new `(qualname, overload)` identities; `floor_divide`'s and
        `histc`'s new `methods.json` rows add none, because both schemas were
        already declared identities through `overloads.json` -- a second door
        onto an existing identity, the `detach`/`maximum` shape the file's own
        comment history already uses repeatedly).

Proven: `torch.fmod`/`Tensor.fmod` (well, `torch.fmod` -- see note below) now
resolve and compute, matching upstream on the sign-of-dividend quadrant table,
signed zero, non-finite operands, integral zero-division, and the `uint8`
scalar-narrowing case (`fmod(uint8(200), -3) == 200`, same as `remainder`'s).
Full suite (456 ok), `DOCWATCH: PASS` (442/442), golden 9137/9137, ops=224.

Note: only `overloads.json` was touched for `fmod`, so `torch.fmod` reaches a
kernel now but `Tensor.fmod` (a `methods.json` row) was not part of this
round's measured gap and was left alone -- adding it would be inventing a
check this backlog did not ask for, not recording one.

## 8. `OpOverload.tags` empty for the thirteen `prims` ops -- not reachable here

Confirmed out of territory before attempting anything: `_scan_aten_tags` (or
whatever answers `torch.ops.prims.<op>.<overload>.tags`) lives in
`bootstrap.py`, which this round's territory explicitly excludes
(`rust/torch_c/src/aten.rs`, `methods.json`, `overloads.json`,
`tools/golden/cases.py` only). `docs/PRIMS.md` §5's own text already says the
gap is structural: `_tagged_core`/the tags reader reads `native_functions.yaml`,
which declares `aten::` entries only, and `prims` schemas come from
`Library.define()` in `torch/_prims/__init__.py`, a different source
`bootstrap.py` does not yet scan. Nothing in `aten.rs`/`methods.json`/
`overloads.json`/`cases.py` participates in answering `.tags` for a
`prims.*` key -- it is a property of the `OpOverload` Python object
`bootstrap.py` constructs, not of the dispatch table this round's files
build. Left exactly as `verify_schemas.py` reports it (4685/4698).

---

## Gate results after this round

    Suite:      456 ok, DOCWATCH: PASS -- 442/442 evaluated marker(s) hold, EXIT=0
    Golden:     9137/9137 cases passed, 0 failed, ops covered=224, pending case builders=0

## Summary by kind

  - **Fixed (real defects, computed correctly now):** #1 floor_divide method
    table entry, #2 histc method table entry, #4 uint8 scalar wrap
    (floor_divide + lt), #5 pow integer-exponent overflow refusal, #6
    float32 `pow` exponent precision, #7 fmod overload table + kernel.
  - **Not a defect / not reachable from this round's files:** #3
    adaptive_avg_pool2d bare-int normalising (real gap, but it is in
    `bootstrap.py`'s Python composite, outside territory), #8 prims tags
    (confirmed structural, in `bootstrap.py`, outside territory).
  - **Test/doc updates required by the above, not independent changes:**
    two pinned counts in `test_shim.py` (`tag_core_count`,
    `len(keys)`/schema round-trip) moved because of #7's new kernels, with
    the arithmetic that keeps each a check spelled out in-line.
