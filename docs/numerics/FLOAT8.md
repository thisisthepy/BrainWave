# `float8_e4m3fn` Investigation

## Path-by-path Table

| Operation | Shim | Upstream (2.13.0) |
|---|---|---|
| construction (`torch.tensor([1.0], dtype=float8_e4m3fn)`) | WORK | WORK |
| `.to()` (`t.to(float8_e4m3fn)`) | WORK | WORK |
| arithmetic (`t + t`) | WORK | REFUSE (`"add_stub" not implemented for 'Float8_e4m3fn'`) |
| comparison (`t1 == t2`) | REFUSE | WORK |
| `print`/`repr` | WORK | WORK |
| `.tolist()` | REFUSE | WORK |
| `.item()` | REFUSE | WORK |
| indexing (`t[0]`) | WORK | WORK |
| `cat`/`stack` | WORK | WORK |
| `matmul` | REFUSE | WORK |

## Diagnosis

The shim previously hung on `.tolist()`, `.item()`, and comparisons.
The root cause of the hang was an infinite recursion bug inside `candle-core`'s implementation of `WithDType` for `float8::F8E4M3`.

In `candle-core` version 0.11.0, the floating-point types implement the internal `WithDType` trait using a macro:
```rust
with_dtype!(f8e4m3, F8E4M3, f8e4m3::from_f64, |v: f8e4m3| v.to_f64());
```
This macro generates:
```rust
impl WithDType for f8e4m3 {
    fn to_f64(self) -> f64 {
        (|v: f8e4m3| v.to_f64())(self)
    }
}
```
The intention was to call the inherent method `pub const fn to_f64(&self) -> f64` defined in the `float8` crate. However, because the inherent method takes `&self` and the `WithDType` trait method takes `self` by value, Rust's method resolution favors the exact match by value if both are in scope. The compiler resolved `v.to_f64()` to the newly generated `WithDType::to_f64(self)` trait method.

This resulted in infinite recursion. When compiled in release mode, LLVM's tail-call optimization collapsed this recursion into an infinite loop (`.L1: jmp .L1`), causing a complete CPU hang without overflowing the stack.

These three paths (`.tolist()`, `.item()`, and `comparison`) all required narrowing/widening intermediate tensor elements to `f64` using `tensor.to_dtype(DType::F64)`. When invoked on an `F8E4M3` tensor, `candle-core` triggered this infinite loop via its `unary_map`.

## Resolution

Instead of patching the `candle-core` logic to fix the `float8` conversion to `f64`, we explicitly refuse `Float8_e4m3fn` by name on these paths in the shim (`aten.rs` and `tensor.rs`). This meets the architectural requirement that unsupported operations must fail explicitly rather than hanging.

- `tolist` now raises a refusal.
- `item` (via `aten._local_scalar_dense.default`) now raises a refusal.
- comparisons (via `compare_common`) now raise a refusal.

These explicit refusals correctly substitute the silent hangs and document what is missing.

---

## The second finding: this build computes where upstream refuses

Closing the hangs made a wider divergence visible, in the direction this project
treats as the dangerous one. Measured on the same tensor, `torch.tensor([1.,2.]).to(float8_e4m3fn)`,
every call under an 6-second alarm:

| op | upstream 2.13.0 | here |
|---|---|---|
| `mul` `abs` `clone` `cat` | WORK | WORK |
| **`add` `sub` `div` `neg` `exp` `sum` `mean`** | **`NotImplementedError`** | **computes** |

Upstream ships `mul` and `abs` for this dtype and refuses the rest — `"add_stub"
not implemented for 'Float8_e4m3fn'` and its siblings. Seven ops here answer
where upstream declines, which means a user gets float8 arithmetic that upstream
would never have produced **and that nothing can check**: there is no oracle for
a result upstream refuses to compute.

That is the same shape as the integer-`requires_grad` divergence closed in
docs/training/BACKWARD2.md — the only other place this build was found to be *more*
permissive than upstream — and it should close the same way: refuse by name, in
upstream's own wording, per op.

**Not done here.** This round's brief was to turn hangs into refusals, and seven
new refusals across the arithmetic surface is its own change with its own
per-op wording to transcribe. It is the next round, and it is named rather than
left for a sweep to rediscover.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs float8_e4m3fn present -->
