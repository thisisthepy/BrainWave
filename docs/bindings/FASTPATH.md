# The positional fast path, and what it must not skip

`_torch_level_function` and `_tensor_method` in `bootstrap.py` wrap every
`torch.*` function and every `TensorBase` method. Both now compile a per-operator
closure, `_fast`, and try it before `entry.resolve`.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _compile_fast_path present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _CAPTURE_ACTIVE present -->

Measured on 2026-09-02, `darwin/arm64`, CPython 3.13, upstream torch 2.13.0.

---

## 1. Where the saving comes from

`entry.resolve` returns `(key, bound)` and the call is then `dispatch(key, **bound)`.
That costs a `**kwargs` dictionary allocation in Python and a `PyDict_GetItem` per
parameter in Rust. The fast path calls `dispatch(key, arg, arg, ...)` positionally
and both disappear.

```
                  before      after     upstream
  view          1542 ns      618 ns       798 ns
  transpose     1280 ns      553 ns       835 ns
```

`resolve` was 6.9% of decode wall time across ~1832 ops per token
(docs/perf/DISPATCH2.md §5.1), which is the size of the prize.

The closure is compiled with `exec`, once per operator, on first call. It executes
**the same type predicates the slow path does** -- `checker.predicate_for`
closures are bound into its globals -- so overload selection cannot drift between
the two. Keyword arguments are not attempted at all: `_fast` is only reached when
`kwargs` is empty, and the caller falls through to `resolve`.

## 2. What it must not skip, and did

A recording made through the fast path carries **no operand names**, because
calling positionally is the whole saving. So before the guard below, the same
call was recorded two different ways depending on how the caller spelled it:

```
  t.transpose(0, 1)             kwargs []
  t.transpose(dim0=0, dim1=1)   kwargs ['dim0', 'dim1', 'self']
```

A trace's contents must not depend on that. The decomposition pass reads operands
by the schema's names -- the third of the three walls in
`test_decompose_lowers_the_op_capture_md_named` -- so the nameless shape arrived
as an empty map rather than as an error. **A wrong answer, not a crash.**

It matters beyond decomposition: capture is what `adapt.wrap`, `trace.backward`
and the decomposition pass are all built on (docs/graph/CAPTURE.md, docs/training/BACKWARD.md,
docs/models/ADAPT.md). An op recorded without its operand names is degraded input to
every one of them.

**The golden harness cannot see this.** It compares op results against upstream,
and the fast path's results are right; what was short was the recording. 7685/7685
passed throughout. This is the fourth distinct class of gap that harness is blind
to (docs/verification/GOLDEN.md), and the first that a *fast path* rather than a missing
feature produced.

### 2.1 The guard

`_fast` is not entered while a capture is recording:

```python
if not kwargs and not _CAPTURE_ACTIVE():
    res = fn._fast(args)
```

`_CAPTURE_ACTIVE` is bound once, at install time, to `module._capture_active`
rather than looked up per call. Inside a capture the slow path runs and the
recording is identical to what it always was.

**It costs nothing measurable.** Against upstream measured in the same session:

```
                      best        median      ratio to upstream
  fast path, guarded   514.6 ms   519.3 ms    0.949
  upstream             543.7 ms   548.8 ms
  upstream (again)     540.9 ms   542.6 ms

  fast path, unguarded 496.7 ms                0.948   (earlier session)
  upstream then        522.5 / 525.2 ms
```

The two ratios agree to a thousandth; the absolute difference between the two
sessions is the machine, not the guard. This is why upstream is measured on both
sides of the shim rather than once -- the raw numbers would have read as a 3.5%
regression from the guard, and there is none.

A capture is not open during decode, which is where the saving was measured, so
the guard is inert exactly where the speed matters.

### 2.2 Evidence the guard is load-bearing

`test_a_capture_records_the_same_call_the_same_way_either_spelling` fails when
the guard is removed, with both sides in the message:

```
AssertionError: ([('aten.transpose.int', [])],
                 [('aten.transpose.int', ['dim0', 'dim1', 'self'])])
```

It asserts the two spellings *agree* rather than asserting either one's contents,
so it survives a change to the recorded names -- and then pins the names as well,
because two empty maps also agree.

## 3. What else the fast path routes around

| Interposition | Reached? | How it is held |
|---|---|---|
| `_MODE_STACK` (torch function modes) | Checked **before** `_fast` in `_torch_level_function` | pre-existing branch |
| capture recording | Guarded, §2.1 | `test_a_capture_records_the_same_call_the_same_way_either_spelling` |
| overload selection | Same predicates as the slow path | `test_the_fast_path_falls_back_rather_than_binding_the_wrong_slot` |
| keyword arguments | Never entered; falls through | same test |
| defaults | Omitted trailing positionals read as absent by Rust's `optional(args, kwargs, i, name)`, the same answer an omitted keyword gives | same test |

`_tensor_method` has no `_MODE_STACK` branch, and did not have one before this
change either.

## 4. Left open

The overload assertion compares the two spellings rather than producing a value
only one of them could produce. The sharper discriminator would be a `float64`
second operand, separating `add.Tensor` (promotes to the wider operand) from
`add.Scalar` (narrows to `self`) by result dtype. The shim refuses that pair:

```
NotImplementedError: aten.add.Tensor: dtype promotion not implemented
in torch._C shim: float32 vs float64
```

So it is untestable that way until mixed-precision promotion lands. Recorded here
rather than worked around.
