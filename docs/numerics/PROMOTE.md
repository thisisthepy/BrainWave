# Dtype Promotion Sites and Truth Table

## Refusal Sites

The following operations raise the `dtype promotion not implemented in torch._C shim` error when given mixed floating-point operands (e.g. float32 and float64):

* `aten._scaled_dot_product_flash_attention_for_cpu.default`
* `aten.add.Tensor`
* `aten.bitwise_or.Tensor`
* `aten.cat.default`
* `aten.eq.Tensor`
* `aten.floor_divide.default`
* `aten.ge.Tensor`
* `aten.gt.Tensor`
* `aten.isin.Tensor_Tensor`
* `aten.le.Tensor`
* `aten.lt.Tensor`
* `aten.matmul.default`
* `aten.max.other`
* `aten.min.other`
* `aten.mm.default`
* `aten.ne.Tensor`
* `aten.stack.default`
* `aten.sub.Tensor`


---

## 1. Verifying the refusal-site list

The list above was inherited. It was re-derived two ways.

**Statically**, by grepping the message. It is emitted from seven places in
`rust/torch_c/src/aten.rs`, not eighteen -- most sites share a helper:

| line | emitter | serves |
|---|---|---|
| 4002 | inline loop in `cat_default` | `aten.cat.default` |
| 4087 | inline loop in `stack_default` | `aten.stack.default` |
| 4773 | `promote_operands` | the *answer* path -- only raises where upstream has no promotion at all |
| 10049 | inline in `arith_inplace_tensor` | `add_`/`sub_`/`mul_`/`div_` |
| 11835 | inline in `floor_divide_impl` | `floor_divide.default`, `floor_divide.Scalar` |
| 15849 | `same_dtype` | 16 call sites |

**Dynamically**, by running every op over the 9x9 dtype cross product on the
shim and on upstream (`/tmp/promo_probe.py`, reproduced in §5).

The inherited list is **wrong in both directions**.

### 1.1 Missing from it (refuse, but were not listed)

* `aten.bmm.default` -- `same_dtype` at line 2249
* `aten.where.self` -- `same_dtype` at line 7430, plus the meta path at 871
* `aten.convolution.default` -- `same_dtype` at line 11561

### 1.2 On it, but not reachable

* `aten.max.other`, `aten.min.other` -- these *do* refuse in `extremum_other`,
  but `torch.maximum` / `torch.minimum` never reach them: overload resolution
  has no table entry, so the user-visible error is
  `overload resolution has no table entry`, a different gap. Reached only
  through `torch.ops.aten.max.other`.

### 1.3 On it, but **refusing is correct** -- upstream refuses too

This is the important correction. Five of the sites are not promotion gaps at
all. Upstream's matmul family and its two "structured" kernels require the
operands to have *equal* dtype and raise otherwise -- they do not consult
`promote_types`:

| op | upstream on any mixed pair | upstream's message |
|---|---|---|
| `aten.mm.default` | **raises** | `expected m1 and m2 to have the same dtype, but got: float != double` |
| `aten.matmul.default` | **raises** | same |
| `aten.bmm.default` | **raises** | `expected scalar type Float but found Double` |
| `aten.convolution.default` | **raises** | `expected scalar type Float but found Double` |
| `aten._scaled_dot_product_flash_attention_for_cpu.default` | **raises** | `expected scalar type Float but found Double` |

Measured: the diagonal is the only non-raising cell in all five 9x9 tables.
So the correct count of genuine promotion gaps is **16 ops**, not 18, and
three of the eighteen (`mm`, `matmul`, `sdpa`) were miscategorised while three
others (`bmm`, `where.self`, `convolution`) were missing.

---

## 2. Which question each site is in

`docs/numerics/SCALAR.md` establishes that upstream's **scalar** promotion rule has no
principle. That conclusion does not carry here and it was checked rather than
assumed.

Every op in §3 below was measured against `torch.promote_types` cell by cell
over the 9x9 grid. **They agree in every cell.** Tensor-tensor promotion is
the lattice, and the lattice is already implemented in this shim as
`promote_types` (aten.rs:4694) -- it was written for `mul`/`div`/`bitwise_and`/
`pow` and re-measured here against eleven more ops without a single
disagreement.

So the two questions really are different, and this round is in the
lattice one:

    scalar operand   ->  no rule, op by op         (SCALAR.md)
    tensor operand   ->  promote_types, every op   (this document)

The five ops in §1.3 are in neither -- they have no promotion rule because
they refuse.

---

## 3. Upstream truth table

Measured on torch 2.13.0 at `/Volumes/macMini/caches/spike-venv`, 9x9 over
`{bool, uint8, int16, int32, int64, float16, bfloat16, float32, float64}`.
Cell = result dtype of `a op b`.


### 3.x `aten.add.Tensor`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **uint8** | uint8 | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int16** | int16 | int16 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int32** | int32 | int32 | int32 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int64** | int64 | int64 | int64 | int64 | int64 | float16 | bfloat16 | float32 | float64 |
| **float16** | float16 | float16 | float16 | float16 | float16 | float16 | float32 | float32 | float64 |
| **bfloat16** | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32 | bfloat16 | float32 | float64 |
| **float32** | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float64 |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 |

### 3.x `aten.sub.Tensor`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE |
| **uint8** | RAISE | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int16** | RAISE | int16 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int32** | RAISE | int32 | int32 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int64** | RAISE | int64 | int64 | int64 | int64 | float16 | bfloat16 | float32 | float64 |
| **float16** | RAISE | float16 | float16 | float16 | float16 | float16 | float32 | float32 | float64 |
| **bfloat16** | RAISE | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32 | bfloat16 | float32 | float64 |
| **float32** | RAISE | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float64 |
| **float64** | RAISE | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 |

### 3.x `aten.mul.Tensor`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **uint8** | uint8 | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int16** | int16 | int16 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int32** | int32 | int32 | int32 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int64** | int64 | int64 | int64 | int64 | int64 | float16 | bfloat16 | float32 | float64 |
| **float16** | float16 | float16 | float16 | float16 | float16 | float16 | float32 | float32 | float64 |
| **bfloat16** | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32 | bfloat16 | float32 | float64 |
| **float32** | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float64 |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 |

### 3.x `aten.div.Tensor`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | float32 | float32 | float32 | float32 | float32 | float16 | bfloat16 | float32 | float64 |
| **uint8** | float32 | float32 | float32 | float32 | float32 | float16 | bfloat16 | float32 | float64 |
| **int16** | float32 | float32 | float32 | float32 | float32 | float16 | bfloat16 | float32 | float64 |
| **int32** | float32 | float32 | float32 | float32 | float32 | float16 | bfloat16 | float32 | float64 |
| **int64** | float32 | float32 | float32 | float32 | float32 | float16 | bfloat16 | float32 | float64 |
| **float16** | float16 | float16 | float16 | float16 | float16 | float16 | float32 | float32 | float64 |
| **bfloat16** | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32 | bfloat16 | float32 | float64 |
| **float32** | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float64 |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 |

### 3.x `aten.floor_divide.default`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | RAISE | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **uint8** | RAISE | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int16** | RAISE | int16 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int32** | RAISE | int32 | int32 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int64** | RAISE | int64 | int64 | int64 | int64 | float16 | bfloat16 | float32 | float64 |
| **float16** | float16 | float16 | float16 | float16 | float16 | float16 | float32 | float32 | float64 |
| **bfloat16** | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32 | bfloat16 | float32 | float64 |
| **float32** | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float64 |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 |

### 3.x `aten.eq.Tensor`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | bool | bool | bool | bool | bool | bool | bool | bool | bool |
| **uint8** | bool | bool | bool | bool | bool | bool | bool | bool | bool |
| **int16** | bool | bool | bool | bool | bool | bool | bool | bool | bool |
| **int32** | bool | bool | bool | bool | bool | bool | bool | bool | bool |
| **int64** | bool | bool | bool | bool | bool | bool | bool | bool | bool |
| **float16** | bool | bool | bool | bool | bool | bool | bool | bool | bool |
| **bfloat16** | bool | bool | bool | bool | bool | bool | bool | bool | bool |
| **float32** | bool | bool | bool | bool | bool | bool | bool | bool | bool |
| **float64** | bool | bool | bool | bool | bool | bool | bool | bool | bool |

### 3.x `aten.max.other`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **uint8** | uint8 | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int16** | int16 | int16 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int32** | int32 | int32 | int32 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int64** | int64 | int64 | int64 | int64 | int64 | float16 | bfloat16 | float32 | float64 |
| **float16** | float16 | float16 | float16 | float16 | float16 | float16 | float32 | float32 | float64 |
| **bfloat16** | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32 | bfloat16 | float32 | float64 |
| **float32** | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float64 |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 |

### 3.x `aten.cat.default`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **uint8** | uint8 | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int16** | int16 | int16 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int32** | int32 | int32 | int32 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int64** | int64 | int64 | int64 | int64 | int64 | float16 | bfloat16 | float32 | float64 |
| **float16** | float16 | float16 | float16 | float16 | float16 | float16 | float32 | float32 | float64 |
| **bfloat16** | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32 | bfloat16 | float32 | float64 |
| **float32** | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float64 |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 |

### 3.x `aten.stack.default`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **uint8** | uint8 | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int16** | int16 | int16 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int32** | int32 | int32 | int32 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int64** | int64 | int64 | int64 | int64 | int64 | float16 | bfloat16 | float32 | float64 |
| **float16** | float16 | float16 | float16 | float16 | float16 | float16 | float32 | float32 | float64 |
| **bfloat16** | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32 | bfloat16 | float32 | float64 |
| **float32** | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float64 |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 |

### 3.x `aten.where.self`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **uint8** | uint8 | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int16** | int16 | int16 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int32** | int32 | int32 | int32 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
| **int64** | int64 | int64 | int64 | int64 | int64 | float16 | bfloat16 | float32 | float64 |
| **float16** | float16 | float16 | float16 | float16 | float16 | float16 | float32 | float32 | float64 |
| **bfloat16** | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32 | bfloat16 | float32 | float64 |
| **float32** | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float64 |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 |

### 3.x `aten.bitwise_or.Tensor`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | bool | uint8 | int16 | int32 | int64 | RAISE | RAISE | RAISE | RAISE |
| **uint8** | uint8 | uint8 | int16 | int32 | int64 | RAISE | RAISE | RAISE | RAISE |
| **int16** | int16 | int16 | int16 | int32 | int64 | RAISE | RAISE | RAISE | RAISE |
| **int32** | int32 | int32 | int32 | int32 | int64 | RAISE | RAISE | RAISE | RAISE |
| **int64** | int64 | int64 | int64 | int64 | int64 | RAISE | RAISE | RAISE | RAISE |
| **float16** | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE |
| **bfloat16** | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE |
| **float32** | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE |
| **float64** | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE |

### 3.x `aten.isin.Tensor_Tensor`

|      lhs \ rhs | bool | uint8 | int16 | int32 | int64 | float16 | bfloat16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|---|
| **bool** | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE | RAISE |
| **uint8** | RAISE | bool | bool | bool | bool | bool | bool | bool | bool |
| **int16** | RAISE | bool | bool | bool | bool | bool | bool | bool | bool |
| **int32** | RAISE | bool | bool | bool | bool | bool | bool | bool | bool |
| **int64** | RAISE | bool | bool | bool | bool | bool | bool | bool | bool |
| **float16** | RAISE | bool | bool | bool | bool | bool | bool | bool | bool |
| **bfloat16** | RAISE | bool | bool | bool | bool | bool | bool | bool | bool |
| **float32** | RAISE | bool | bool | bool | bool | bool | bool | bool | bool |
| **float64** | RAISE | bool | bool | bool | bool | bool | bool | bool | bool |

---

## 4. What now promotes

Every op whose upstream answer is `torch.promote_types` (§2), routed through
the `promote_types` lattice that already existed at `aten.rs:4694` for
`mul`/`div`/`bitwise_and`/`pow`. **No new table was written** -- the lattice
was re-measured against eleven more ops and needed no change, which is the
main reason this round was small.

| op | was | helper it uses now |
|---|---|---|
| `aten.add.Tensor` | refused | `promote_operands` |
| `aten.sub.Tensor` | refused | `promote_operands` |
| `aten.eq/ne/lt/le/gt/ge.Tensor` | refused | `promote_operands` |
| `aten.max.other`, `aten.min.other` | refused | `promote_operands` |
| `aten.bitwise_or.Tensor` | refused | `promote_operands` |
| `aten.where.self` | refused | `promote_operands` |
| `aten.cat.default` | refused | `promote_list` |
| `aten.stack.default` | refused | `promote_list` |

That is **14 ops**. The meta-device path (`meta_dispatch`) moved with the
dense one for the comparisons, the arithmetic four and `where.self`, so meta
cannot advertise a pairing the dense kernel declines nor decline one it would
have answered.

Two refusals were kept deliberately and are *not* promotion gaps:

* **`sub` with a `bool` operand.** Upstream raises even though
  `promote_types(bool, float32)` is `float32` -- the refusal is on the
  operand, not on the promoted type. The check therefore runs **before** the
  promotion; promoting first would answer where upstream raises.
* **`add`/`div` with two `bool` operands.** Same-dtype, so no promotion is
  involved at all; these are the pre-existing `docs/numerics/BOOL.md` gaps (bool `+`
  is a logical OR, bool `/` is arithmetic in `float32`) and this round did
  not touch them.

## 5. Why the values had to be checked separately from the dtypes

**Upstream converts each operand to the common dtype and only then reads it
into the accumulator.** Those two conversions are not interchangeable with a
single conversion straight to the accumulator, and the difference is a value
under an identical dtype label:

```text
sub(int64([2049]), float16([1.0]))
    upstream                       float16   2047.0
    cast straight to the f32 acc   float16   2048.0
```

`float16` cannot hold 2049. Upstream narrows the `int64` operand to 2048
first and subtracts 1; casting both operands to the `float32` accumulator
computes 2049 - 1 = 2048 and narrows once at the end. Both answers are
labelled `float16`.

The comparisons are the worse case, because the result is `bool` either way
and carries no trace at all of where the comparison happened:

```text
eq(int64([16777217]), float32([16777216.0]))   ->  True    upstream
```

16777217 has no `float32` form; after the narrowing the two operands are the
same number. This shim's comparison kernel widens to `f64`/`i64` so candle has
one kernel to run, and `f64` holds both operands exactly -- so without the
common-dtype cast first it answers **False**, in `torch.bool`, indistinguishably.

`operand_in` (aten.rs) is that cast, and it is applied in `arith_tensor`,
`add_tensor`, `compare_tensor`, `extremum_other`, `cat_default` and
`stack_default`. `where_self` does not call it because `where_select` already
performs the same cast; `bitwise_binary` does not because it reads both
operands exactly through `i64`, which is lossless for every dtype it accepts.

### 5.1 The check that this is verified rather than asserted

Removing the cast from `compare_tensor` alone -- leaving the promotion, so
every result dtype stays correct -- makes **4 golden cases fail** and nothing
else:

```text
FAIL aten.eq.Tensor :: (int64 16777217 vs float32 16777216) -- value mismatch
     torch=[True, False]  c=[False, False]  dtype=bool
```

The six shared `_PROMOTION_PAIRS` cases keep passing under that tamper, which
is why they are documented as *not* covering the precision rule: their values
are 0/1 so that one list is legal in `bool` and `uint8` too, and 0/1 survives
every narrowing in the table.

## 6. What was deliberately left out

**`aten.floor_divide.default` and `aten.isin.Tensor_Tensor` still refuse
mixed dtypes.** Both follow `promote_types` in the cells they answer, so
neither is hard, but each carries a second rule that this round did not
measure and would have had to guess at:

* `floor_divide` has upstream's reduced-float **scalar fast path**
  (`is_scalar(2)`, kept at `opmath` precision) sitting in the same function,
  and `scalar_at_opmath` interacts with which dtype the divisor is read in.
  Promotion changes that dtype. `bool // bool` also raises upstream
  (`"div_floor_cpu" not implemented for 'Bool'`).
* `isin` refuses **any** `bool` operand upstream
  (`Unsupported input type encountered for isin(): Bool`) -- an entire row and
  column of its grid -- and promotes the rest to a `bool` result.

A partially-promoting op is worse than a refusing one, so both keep their
refusal and the message that names the pair. `isin` is now the only remaining
caller of `same_dtype`.

The five ops in §1.3 are not on this list: they refuse because upstream
refuses, and they were moved to `require_same_dtype`, whose message says
upstream requires equal dtypes rather than claiming a shim gap.

## 7. Gates

```text
suite    328 ok, 0 failed, DOCWATCH: PASS -- 241/241 evaluated marker(s) hold, EXIT=0
golden   7751/7751 cases passed, 0 failed, ops covered=168
```

Golden rose from 7685 by 66, and **all 66 are new cases**. The 12 cases that
changed from `expect="c_error"` to a real value diff did *not* move the count
-- they were already being counted, they were just asserting a refusal rather
than comparing values. Counting those as additions would have inflated the
round. The 66 break down as:

| new cases | what |
|---|---|
| 42 | the six comparisons x (6 promotion pairs + 1 precision case) |
| 12 | `max.other` and `min.other` x 6 promotion pairs |
|  8 | `cat`: 6 promotion pairs, the three-entry fold, the legacy-empty dtype |
|  2 | `add`/`sub` narrowed-before-the-arithmetic (int64 2049 with float16) |
|  2 | `stack`: the reduced-float tie, and the three-entry fold |

`ops covered` stayed at 168: every op touched already had a builder, so this
round deepened existing coverage rather than reaching new ops.

Separately from the harness, the 9x9 grid was diffed cell by cell against
upstream for 17 ops (`/tmp/diff_dump.py`, values compared as IEEE bit
patterns, not reprs):

```text
cells=1539  agree (dtype AND bit-exact value)=1390  both raise=146
dtype mismatches=0   value mismatches=0
shim refuses / upstream answers = 3   (add|bool|bool, add#alpha|bool|bool, div|bool|bool)
shim answers / upstream raises  = 0
```

All three remaining refusals are `bool x bool` -- same-dtype, so no promotion
is involved; they are the docs/numerics/BOOL.md gaps described in §4.

## 8. Closing the deliberate refusals (`floor_divide`, `isin`)

Two ops deliberately refused mixed dtypes in the initial round due to upstream behavior interacting with their native type promotion rules.

### `isin`

`isin` promotes all inputs, but refuses any `bool` operand natively.

Truth table (tested natively):
- **bool / ***: Upstream raises `RuntimeError: "isin_Tensor_Tensor_out" not implemented for 'Bool'`
- **uint8 / int16**: `bool` (result is always boolean mask)
- **float32 / float64**: `bool`
- **int64 / float16**: `bool`
- **float16 / bfloat16**: `bool`
- **int32 / int64**: `bool`

The implementation intercepts `bool` operands directly in `isin_tensor_tensor` and raises the shim's `PyRuntimeError`, and otherwise delegates safely to the general `promote_operands` path.

### `floor_divide`

`floor_divide.default` (tensor-tensor) accepts mixed dtypes normally, but specifically refuses `bool` with itself. `floor_divide.Scalar` uses PyTorch's `result_type(tensor, scalar)` rules, which are different from standard binary op promotion:

Truth table (tested natively):
- **bool // bool**: Upstream raises `RuntimeError: "div_floor_cpu" not implemented for 'Bool'`
- **bool // int**: `int64`
- **bool // float**: `float32`
- **uint8 // int16**: `int16`
- **float32 // float64**: `float64`
- **int64 // float16**: `float16`
- **float16 // bfloat16**: `float32`
- **int32 // int64**: `int64`

The implementation handles the `bool // bool` tensor pair by throwing `PyRuntimeError` with upstream's exact message, and uses `promote_operands` for other tensor pairs. For `.Scalar`, it simulates upstream's scalar promotion rules directly inline.

### The exception type is part of the answer

`floor_divide` on a `bool` pair raises **`NotImplementedError`** and `isin` on a
`bool` operand raises **`RuntimeError`**, because that is what upstream raises in
each place. They are not made consistent with each other: a caller who writes
`except NotImplementedError` around a bool pair would not catch a `RuntimeError`
carrying the same words, so matching upstream matters more than matching
ourselves. The first landed as `RuntimeError` and was corrected on review.

### A shared `CARGO_TARGET_DIR` reported this round as broken when it was not

The first verification of this work read **11 golden failures**, every one of
them the *old* refusal message, which reads exactly like "the cases were added
and the implementation was not". The implementation was there; the artefact
under test was not built from it.

`CARGO_TARGET_DIR` had been set to the shared `/Volumes/macMini/caches/cargo-target`,
which the main checkout and every worktree write to. Rebuilding into a
per-checkout directory and re-running gave **7763/7763** from the same source.

Other rounds' documents already use per-round directories -- `cargo-target-adapt`
(ADAPT.md), `cargo-target-autograd` (AUTOGRAD.md), `cargo-target-bind`
(BIND.md) -- so the convention existed and the brief for this round contradicted
it. Same shape as the `$TMPDIR/torch-c-stage` collision that made two concurrent
worktrees validate each other's artefacts: **shared mutable build state across
checkouts, and the failure presents as a wrong verdict rather than as an error.**

The tell is that a failure message names behaviour the source no longer has.
When that happens, rebuild into a fresh directory before believing the verdict.
