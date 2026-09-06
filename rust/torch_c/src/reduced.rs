//! The `float16`/`bfloat16` conversions, and the fused form of `opmath_in`.
//!
//! `aten::opmath_in` says what torch computes reduced floats in: `float`, for
//! every arithmetic kernel, narrowed back exactly once. docs/BF16.md measured
//! why that rule is not optional -- without it `add.Tensor` truncates, the
//! truncation is *biased*, and a 30-layer residual stream turns 1 ulp into a
//! logit difference of 11.75.
//!
//! This module is about what that rule costs, which docs/QUANT.md §3 measured
//! and nobody had looked at before: on the host, `float16` was **6.2x slower
//! than `float32`** at `mm` 128 and **27x** at the decoding shape, and
//! `bfloat16` was slower everywhere too. The rule was not the reason. Two
//! implementation choices under it were:
//!
//!   * **candle converts one element at a time.** `CpuStorage::to_dtype` is a
//!     `unary_map` with a per-element closure, and for `float16` that closure
//!     is `half`'s `f16::to_f32`, which on aarch64 is a scalar
//!     `asm!("fcvt s, h")`. Inline assembly is opaque to the vectoriser, so
//!     the loop stays scalar: **1.7 Gelem/s, against 8.3 for the same
//!     conversion written with `fcvtl` over 8 elements at a time** (measured,
//!     docs/DTYPE.md §2). `bfloat16` escapes the worst of it only because its
//!     widening is a shift that LLVM does vectorise.
//!   * **the widening was materialised.** `a.to_dtype(F32) + b.to_dtype(F32)`
//!     then `.to_dtype(BF16)` writes three whole `float32` tensors to memory to
//!     produce one reduced-precision one -- 30 bytes of traffic per element
//!     where a fused loop moves 6. Upstream keeps the widening in registers;
//!     that is the entire difference between "accumulate in `float`" and
//!     "promote the tensor".
//!
//! Both are fixed here without touching the rule. Every kernel below computes
//! *exactly* the function `half` computes, and the equality is checked rather
//! than argued: `reduced_kernels_agree_with_half_on_every_f16_bit_pattern`
//! walks all 65536 of them, and the fused arithmetic is compared against
//! widen/compute/narrow element by element.
//!
//! What this does **not** do is make reduced precision beat `float32` at
//! everything. It cannot -- see docs/DTYPE.md §4. Conversion instructions cost
//! more than the arithmetic they feed, so an op only wins when the halved
//! memory traffic pays for them.

use candle_core::{CpuStorage, DType, Layout, Shape, Tensor};
use half::slice::HalfFloatSliceExt;
use half::{bf16, f16};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ---------------------------------------------------------------------------
// Slice kernels.
//
// Each writes through a raw pointer into memory that may be *uninitialised*,
// which is not a micro-optimisation. `vec![0f32; n]` zeroes four megabytes
// before the conversion writes over them, and that costs a third of the
// conversion: `float16 -> float32` over a million elements is 0.104 ms into a
// zeroed buffer and 0.070 ms into raw capacity (measured, docs/DTYPE.md §2.3).
// candle's `unary_map` avoids it by `.map().collect()`-ing, so a fast kernel
// that allocated the obvious way would have handed the saving straight back.
//
// Which body is fastest is *not* uniform, and was measured rather than
// assumed:
//
//   * `bfloat16 -> float32` is fastest as a **plain loop**. It is one shift,
//     LLVM emits `shll` (widen and shift in a single instruction), and the
//     hand-written NEON below needed two (`ushll` then `shl`) and came out
//     slower -- 8.3 Gelem/s against 9.4. So there is no NEON arm for it.
//   * the other three are fastest as **NEON**, by between 1.2x and 5.6x, and
//     the `float16` widening is the extreme: candle's per-element form runs at
//     2.7 Gelem/s because `half::f16::to_f32` is an `asm!("fcvt s, h")` that
//     the vectoriser cannot see through, against 14.9 for `fcvtl` over eight.
//
// Every one of these is *bitwise* the function `half` computes element by
// element. `fcvtl`/`fcvtn` round to nearest even, which is what `f16::from_f32`
// does, and the `bfloat16` narrowing is `bf16::from_f32` with its branch turned
// into a select. The tests check that over every bit pattern rather than
// arguing it.
// ---------------------------------------------------------------------------

/// A `Vec<T>` whose contents are written by `fill` rather than zeroed first.
///
/// `fill` receives a pointer valid for `n` writes of `T` and **must write all
/// of them**; every caller here is a kernel that writes one element per input
/// element, over a `dst` the same length as its `src`.
#[inline]
fn built<T: Copy>(n: usize, fill: impl FnOnce(*mut T)) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    fill(v.as_mut_ptr());
    // SAFETY: `fill` initialised all `n` elements, and `with_capacity(n)`
    // reserved room for exactly that many.
    unsafe { v.set_len(n) };
    v
}

/// `float16 -> float32`. Exact for every input; the conversion widens.
///
/// # Safety
/// `dst` must be valid for `src.len()` writes of `f32`.
unsafe fn widen_f16_into(src: &[f16], dst: *mut f32) {
    #[cfg(target_arch = "aarch64")]
    {
        let bits: &[u16] = src.reinterpret_cast();
        let (n, s) = (bits.len(), bits.as_ptr());
        let mut i = 0;
        while i + 8 <= n {
            let v = vld1q_u16(s.add(i));
            vst1q_f32(dst.add(i), vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(v))));
            vst1q_f32(dst.add(i + 4), vcvt_high_f32_f16(vreinterpretq_f16_u16(v)));
            i += 8;
        }
        while i < n {
            *dst.add(i) = src[i].to_f32();
            i += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // `half`'s slice conversion, which is F16C on x86 and a plain loop
        // elsewhere -- in both cases better than one `to_f32` call per element.
        let d = core::slice::from_raw_parts_mut(dst, src.len());
        src.convert_to_f32_slice(d);
    }
}

/// `bfloat16 -> float32`. A 16-bit left shift; exact for every input.
///
/// # Safety
/// `dst` must be valid for `src.len()` writes of `f32`.
unsafe fn widen_bf16_into(src: &[bf16], dst: *mut f32) {
    for (i, v) in src.iter().enumerate() {
        *dst.add(i) = v.to_f32();
    }
}

/// `float32 -> float16`, round to nearest even.
///
/// # Safety
/// `dst` must be valid for `src.len()` writes of `f16`.
unsafe fn narrow_f16_into(src: &[f32], dst: *mut f16) {
    #[cfg(target_arch = "aarch64")]
    {
        let (n, s) = (src.len(), src.as_ptr());
        let d = dst as *mut u16;
        let mut i = 0;
        while i + 8 <= n {
            let lo = vcvt_f16_f32(vld1q_f32(s.add(i)));
            let full = vcvt_high_f16_f32(lo, vld1q_f32(s.add(i + 4)));
            vst1q_u16(d.add(i), vreinterpretq_u16_f16(full));
            i += 8;
        }
        while i < n {
            *dst.add(i) = f16::from_f32(src[i]);
            i += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let d = core::slice::from_raw_parts_mut(dst, src.len());
        d.convert_from_f32_slice(src);
    }
}

/// `float32 -> bfloat16`, round to nearest even, NaN quieted.
///
/// The rounding is why this is spelled out rather than left to a shift:
/// truncating here is exactly the fault docs/BF16.md §2 found, and it is
/// invisible to any check with a tolerance.
///
/// # Safety
/// `dst` must be valid for `src.len()` writes of `bf16`.
unsafe fn narrow_bf16_into(src: &[f32], dst: *mut bf16) {
    #[cfg(target_arch = "aarch64")]
    {
        let (n, s) = (src.len(), src.as_ptr());
        let d = dst as *mut u16;
        let mut i = 0;
        while i + 8 <= n {
            let lo = narrow_bf16_x4(vld1q_f32(s.add(i)));
            let hi = narrow_bf16_x4(vld1q_f32(s.add(i + 4)));
            vst1q_u16(d.add(i), vcombine_u16(lo, hi));
            i += 8;
        }
        while i < n {
            *dst.add(i) = bf16::from_f32(src[i]);
            i += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let d = core::slice::from_raw_parts_mut(dst, src.len());
        d.convert_from_f32_slice(src);
    }
}

/// Eight `bfloat16` lanes to two `float32` vectors -- a 16-bit shift. Used by
/// the fused kernels, where the widening never reaches memory and so cannot be
/// left to the autovectoriser.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn widen_bf16_x8(v: uint16x8_t) -> (float32x4_t, float32x4_t) {
    (
        vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_low_u16(v)))),
        vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_high_u16(v))),
    )
}

/// Four `float32` lanes to four `bfloat16` lanes.
///
/// This is `half::bf16::from_f32` with its `if value.is_nan()` turned into a
/// `bsl`. The NaN arm is not decoration: the rounding add alone turns the
/// signalling NaN `0x7f80_0001` into `0x7f80`, which is *infinity*.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn narrow_bf16_x4(f: float32x4_t) -> uint16x4_t {
    let x = vreinterpretq_u32_f32(f);
    let round = vaddq_u32(
        vandq_u32(vshrq_n_u32::<16>(x), vdupq_n_u32(1)),
        vdupq_n_u32(0x7fff),
    );
    let normal = vshrn_n_u32::<16>(vaddq_u32(x, round));
    let quiet = vorr_u16(vshrn_n_u32::<16>(x), vdup_n_u16(0x0040));
    let is_nan = vcgtq_u32(
        vandq_u32(x, vdupq_n_u32(0x7fff_ffff)),
        vdupq_n_u32(0x7f80_0000),
    );
    vbsl_u16(vshrn_n_u32::<16>(is_nan), quiet, normal)
}

// Safe wrappers over the kernels above, so that what the tests check and what
// the ops run are the same code rather than two spellings of an intention.
// `cfg(test)` because only the tests need the slice form -- the ops write
// straight into a fresh `Vec`'s capacity, and a wrapper left compiled in the
// shipping build would be dead weight the linker has to reason about.

#[cfg(test)]
pub fn widen_f16(src: &[f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    unsafe { widen_f16_into(src, dst.as_mut_ptr()) }
}

#[cfg(test)]
pub fn widen_bf16(src: &[bf16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    unsafe { widen_bf16_into(src, dst.as_mut_ptr()) }
}

#[cfg(test)]
pub fn narrow_f16(src: &[f32], dst: &mut [f16]) {
    assert_eq!(src.len(), dst.len());
    unsafe { narrow_f16_into(src, dst.as_mut_ptr()) }
}

#[cfg(test)]
pub fn narrow_bf16(src: &[f32], dst: &mut [bf16]) {
    assert_eq!(src.len(), dst.len());
    unsafe { narrow_bf16_into(src, dst.as_mut_ptr()) }
}

// ---------------------------------------------------------------------------
// `to_dtype`, for the four conversions that matter.
// ---------------------------------------------------------------------------

struct Widen;

impl candle_core::CustomOp1 for Widen {
    fn name(&self) -> &'static str {
        "torch_c_widen"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("torch_c_widen: not contiguous".into()))?;
        let n = end - start;
        let out = match storage {
            // SAFETY: each kernel writes `src.len() == n` elements, which is
            // exactly what `built` reserved.
            CpuStorage::F16(src) => built(n, |p| unsafe { widen_f16_into(&src[start..end], p) }),
            CpuStorage::BF16(src) => built(n, |p| unsafe { widen_bf16_into(&src[start..end], p) }),
            _ => return Err(candle_core::Error::Msg("torch_c_widen: not a reduced float".into())),
        };
        Ok((CpuStorage::F32(out), layout.shape().clone()))
    }
}

struct Narrow(DType);

impl candle_core::CustomOp1 for Narrow {
    fn name(&self) -> &'static str {
        "torch_c_narrow"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("torch_c_narrow: not contiguous".into()))?;
        let CpuStorage::F32(src) = storage else {
            return Err(candle_core::Error::Msg("torch_c_narrow: source is not f32".into()));
        };
        let src = &src[start..end];
        let n = src.len();
        let out = match self.0 {
            // SAFETY: each kernel writes `src.len() == n` elements.
            DType::F16 => CpuStorage::F16(built(n, |p| unsafe { narrow_f16_into(src, p) })),
            DType::BF16 => CpuStorage::BF16(built(n, |p| unsafe { narrow_bf16_into(src, p) })),
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "torch_c_narrow: {other:?} is not a reduced float"
                )))
            }
        };
        Ok((out, layout.shape().clone()))
    }
}

/// `Tensor::to_dtype`, taking the fast path for the reduced-float conversions
/// and delegating everything else to candle unchanged.
///
/// The result is bitwise identical to `t.to_dtype(target)` in every case; only
/// the four `{F16,BF16} <-> F32` pairs are handled here, and only when the
/// tensor is a contiguous CPU one. A non-contiguous input falls through rather
/// than being made contiguous first, so this never changes how much memory an
/// op touches -- only how fast it touches it.
pub fn to_dtype(t: &Tensor, target: DType) -> candle_core::Result<Tensor> {
    let src = t.dtype();
    if src == target || !t.device().is_cpu() || !t.layout().is_contiguous() {
        return t.to_dtype(target);
    }
    match (src, target) {
        (DType::F16 | DType::BF16, DType::F32) => t.apply_op1_no_bwd(&Widen),
        (DType::F32, DType::F16 | DType::BF16) => t.apply_op1_no_bwd(&Narrow(target)),
        // **The one route in this crate that is a correctness fix and not a
        // speed one** (docs/FLOAT8C.md §1). candle 0.11.0's converter reaches
        // `F8E4M3 -> F64` through `WithDType::to_f64`, whose macro-generated
        // body calls itself; release-mode LLVM makes that `.L1: jmp .L1`, so
        // `x.to(torch.float64)` on a float8 tensor spins forever instead of
        // answering. Its `F8E4M3 -> F32` arm is written `v.to_f32()`, which
        // has no trait method to collide with and terminates, so the hang is
        // routed around by going through `F32` first.
        //
        // Exact, not approximate: `float8_e4m3fn` is 4 exponent bits and 3
        // mantissa bits with a max magnitude of 448, so all 256 of its bit
        // patterns are representable in `f32`, and `f32 -> f64` is exact for
        // every one of them. NaN stays NaN; the dtype has no infinities.
        (DType::F8E4M3, DType::F64) => t.to_dtype(DType::F32)?.to_dtype(DType::F64),
        _ => t.to_dtype(target),
    }
}

/// `to_dtype` as a method, so the opmath call sites read the way they did.
///
/// The spelling is deliberately not `to_dtype`: an inherent method wins over a
/// trait one, so a trait that reused the name would compile and never be
/// called. `fast_to` cannot be confused for candle's, and grepping it finds
/// every site that opted in.
pub trait FastDType {
    fn fast_to(&self, target: DType) -> candle_core::Result<Tensor>;
}

impl FastDType for Tensor {
    #[inline]
    fn fast_to(&self, target: DType) -> candle_core::Result<Tensor> {
        to_dtype(self, target)
    }
}

// ---------------------------------------------------------------------------
// The fused arithmetic.
// ---------------------------------------------------------------------------

/// Which arithmetic the fused kernel performs. The widening and the narrowing
/// are not part of the choice -- every arm does both.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Fused {
    Add,
    Sub,
    Mul,
    Div,
}

impl Fused {
    #[inline(always)]
    fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            Fused::Add => a + b,
            Fused::Sub => a - b,
            Fused::Mul => a * b,
            Fused::Div => a / b,
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn apply_x4(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        match self {
            Fused::Add => vaddq_f32(a, b),
            Fused::Sub => vsubq_f32(a, b),
            Fused::Mul => vmulq_f32(a, b),
            Fused::Div => vdivq_f32(a, b),
        }
    }
}

/// # Safety
/// `out` must be valid for `a.len()` writes of `f16`, and `b.len() == a.len()`.
unsafe fn fused_f16_into(kind: Fused, a: &[f16], b: &[f16], out: *mut f16) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        let (ab, bb) = (a.reinterpret_cast(), b.reinterpret_cast());
        let n = ab.len();
        let (pa, pb, po) = (ab.as_ptr(), bb.as_ptr(), out as *mut u16);
        let mut i = 0;
        while i + 8 <= n {
            let va = vld1q_u16(pa.add(i));
            let vb = vld1q_u16(pb.add(i));
            let lo = kind.apply_x4(
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(va))),
                vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(vb))),
            );
            let hi = kind.apply_x4(
                vcvt_high_f32_f16(vreinterpretq_f16_u16(va)),
                vcvt_high_f32_f16(vreinterpretq_f16_u16(vb)),
            );
            let packed = vcvt_high_f16_f32(vcvt_f16_f32(lo), hi);
            vst1q_u16(po.add(i), vreinterpretq_u16_f16(packed));
            i += 8;
        }
        while i < n {
            *out.add(i) = f16::from_f32(kind.apply(a[i].to_f32(), b[i].to_f32()));
            i += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..a.len() {
        *out.add(i) = f16::from_f32(kind.apply(a[i].to_f32(), b[i].to_f32()));
    }
}

/// # Safety
/// `out` must be valid for `a.len()` writes of `bf16`, and `b.len() == a.len()`.
unsafe fn fused_bf16_into(kind: Fused, a: &[bf16], b: &[bf16], out: *mut bf16) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        let (ab, bb) = (a.reinterpret_cast(), b.reinterpret_cast());
        let n = ab.len();
        let (pa, pb, po) = (ab.as_ptr(), bb.as_ptr(), out as *mut u16);
        let mut i = 0;
        while i + 8 <= n {
            let (al, ah) = widen_bf16_x8(vld1q_u16(pa.add(i)));
            let (bl, bh) = widen_bf16_x8(vld1q_u16(pb.add(i)));
            let lo = narrow_bf16_x4(kind.apply_x4(al, bl));
            let hi = narrow_bf16_x4(kind.apply_x4(ah, bh));
            vst1q_u16(po.add(i), vcombine_u16(lo, hi));
            i += 8;
        }
        while i < n {
            *out.add(i) = bf16::from_f32(kind.apply(a[i].to_f32(), b[i].to_f32()));
            i += 1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..a.len() {
        *out.add(i) = bf16::from_f32(kind.apply(a[i].to_f32(), b[i].to_f32()));
    }
}

/// Safe wrappers, for the same reason as the conversion ones above.
#[cfg(test)]
pub fn fused_f16(kind: Fused, a: &[f16], b: &[f16], out: &mut [f16]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    unsafe { fused_f16_into(kind, a, b, out.as_mut_ptr()) }
}

#[cfg(test)]
pub fn fused_bf16(kind: Fused, a: &[bf16], b: &[bf16], out: &mut [bf16]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    unsafe { fused_bf16_into(kind, a, b, out.as_mut_ptr()) }
}

struct FusedOp(Fused);

impl candle_core::CustomOp2 for FusedOp {
    fn name(&self) -> &'static str {
        "torch_c_fused_opmath"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let bad = || candle_core::Error::Msg("torch_c_fused_opmath: unexpected operand".into());
        let (a0, a1) = l1.contiguous_offsets().ok_or_else(bad)?;
        let (b0, b1) = l2.contiguous_offsets().ok_or_else(bad)?;
        if a1 - a0 != b1 - b0 || l1.shape() != l2.shape() {
            return Err(bad());
        }
        let out = match (s1, s2) {
            // SAFETY: both kernels write `a1 - a0` elements, which is what
            // `built` reserved, and the two operand slices are that long.
            (CpuStorage::F16(a), CpuStorage::F16(b)) => CpuStorage::F16(built(a1 - a0, |p| unsafe {
                fused_f16_into(self.0, &a[a0..a1], &b[b0..b1], p)
            })),
            (CpuStorage::BF16(a), CpuStorage::BF16(b)) => CpuStorage::BF16(built(a1 - a0, |p| unsafe {
                fused_bf16_into(self.0, &a[a0..a1], &b[b0..b1], p)
            })),
            _ => return Err(bad()),
        };
        Ok((out, l1.shape().clone()))
    }
}

/// `narrow(widen(lhs) OP widen(rhs))` in one pass, for the shapes where that is
/// the same function as doing it in three.
///
/// Returns `None` when the fast path does not apply, and the caller then does
/// the widening explicitly -- there is no arm here that computes something
/// different from what the slow path would. The conditions are that both
/// operands are the same reduced float, that they live on the CPU, and that
/// after broadcasting they have the same shape and are contiguous.
///
/// **Why fusing matters and widening faster does not go far enough.** The three
/// -pass form writes a `float32` tensor per operand and one for the result:
/// 30 bytes of traffic per element against 6, and on `float16` `add` that is
/// the difference between 1.85 ms and 0.07 ms for a million elements
/// (docs/DTYPE.md §3). The `float16` fused kernel is *faster than the same
/// `add` in `float32`* -- which is the only place in this round where lowering
/// the dtype buys speed rather than costing it.
pub fn fused_arith(
    kind: Fused,
    lhs: &Tensor,
    rhs: &Tensor,
    storage: DType,
) -> Option<candle_core::Result<Tensor>> {
    // `storage` is the dtype the op has decided to *produce*. Requiring both
    // operands to already be it is what keeps this path out of the promoting
    // cases -- `mul` between two different dtypes has a rule of its own, and a
    // fused kernel that ignored it would silently compute in the wrong one.
    if !matches!(storage, DType::F16 | DType::BF16)
        || lhs.dtype() != storage
        || rhs.dtype() != storage
        || !lhs.device().is_cpu()
        || !rhs.device().is_cpu()
    {
        return None;
    }
    // Broadcasting is handled by materialising the smaller operand in its own
    // (narrow) dtype, which is strictly less memory than the widening path
    // would have written for the same tensor. Anything that will not broadcast
    // is left to the caller, which raises whatever candle raises.
    let (a, b) = if lhs.shape() == rhs.shape() {
        (lhs.clone(), rhs.clone())
    } else if let Ok(b) = rhs.broadcast_as(lhs.shape()) {
        (lhs.clone(), b)
    } else if let Ok(a) = lhs.broadcast_as(rhs.shape()) {
        (a, rhs.clone())
    } else {
        return None;
    };
    let a = match a.contiguous() {
        Ok(a) => a,
        Err(e) => return Some(Err(e)),
    };
    let b = match b.contiguous() {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    Some(a.apply_op2_no_bwd(&b, &FusedOp(kind)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `float16` bit pattern, and every `bfloat16` one, through the
    /// widening kernels. `half` is the reference because it is what the
    /// numbers in docs/BF16.md were pinned against.
    #[test]
    fn reduced_kernels_agree_with_half_on_every_f16_bit_pattern() {
        let all: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
        let mut got = vec![0f32; all.len()];
        widen_f16(&all, &mut got);
        for (i, h) in all.iter().enumerate() {
            let want = h.to_f32();
            assert!(
                got[i].to_bits() == want.to_bits() || (got[i].is_nan() && want.is_nan()),
                "f16 0x{:04x}: {:?} != {:?}",
                h.to_bits(),
                got[i],
                want
            );
        }

        let all: Vec<bf16> = (0..=u16::MAX).map(bf16::from_bits).collect();
        let mut got = vec![0f32; all.len()];
        widen_bf16(&all, &mut got);
        for (i, h) in all.iter().enumerate() {
            let want = h.to_f32();
            assert!(
                got[i].to_bits() == want.to_bits() || (got[i].is_nan() && want.is_nan()),
                "bf16 0x{:04x}",
                h.to_bits()
            );
        }
    }

    /// The narrowing rounds like `half`, which rounds like torch. A truncating
    /// kernel would pass a tolerance check and fail this one -- docs/BF16.md
    /// §2.3 is about exactly that difference.
    #[test]
    fn reduced_narrowing_rounds_to_nearest_even_including_nan_and_the_tail() {
        let mut probe: Vec<f32> = vec![
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            // A *signalling* NaN: the rounding add alone turns this into
            // infinity, so it is the input that proves the NaN arm runs.
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0xff80_0001),
            // Exact ties, where truncation and round-to-nearest-even differ.
            f32::from_bits(0x3f80_8000),
            f32::from_bits(0x3f81_8000),
        ];
        let mut x = 1u32;
        for _ in 0..200_000 {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            probe.push(f32::from_bits(x));
        }
        // A length that is not a multiple of 8, so the scalar tail runs too.
        probe.push(0.5);
        probe.push(-0.5);
        probe.push(1e-9);

        let mut got = vec![bf16::ZERO; probe.len()];
        narrow_bf16(&probe, &mut got);
        for (i, f) in probe.iter().enumerate() {
            let want = bf16::from_f32(*f);
            assert!(
                got[i].to_bits() == want.to_bits() || (got[i].is_nan() && want.is_nan()),
                "bf16 narrow of {:?} (0x{:08x}): 0x{:04x} != 0x{:04x}",
                f,
                f.to_bits(),
                got[i].to_bits(),
                want.to_bits()
            );
        }

        let mut got = vec![f16::ZERO; probe.len()];
        narrow_f16(&probe, &mut got);
        for (i, f) in probe.iter().enumerate() {
            let want = f16::from_f32(*f);
            assert!(
                got[i].to_bits() == want.to_bits() || (got[i].is_nan() && want.is_nan()),
                "f16 narrow of {:?}: 0x{:04x} != 0x{:04x}",
                f,
                got[i].to_bits(),
                want.to_bits()
            );
        }
    }

    /// The fused kernels compute what widen/compute/narrow computes. If this
    /// ever stops holding, the fast path is a different function from the slow
    /// one and the whole point of `opmath_in` is gone.
    #[test]
    fn fused_arithmetic_equals_widen_compute_narrow_element_by_element() {
        let n = 4099; // not a multiple of 8: the tail participates
        let mut x = 7u32;
        let mut next = || {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            ((x >> 8) as f32 / 8388608.0) - 1.0
        };
        let a32: Vec<f32> = (0..n).map(|_| next()).collect();
        let b32: Vec<f32> = (0..n).map(|_| next()).collect();

        for kind in [Fused::Add, Fused::Sub, Fused::Mul, Fused::Div] {
            let a: Vec<bf16> = a32.iter().map(|v| bf16::from_f32(*v)).collect();
            let b: Vec<bf16> = b32.iter().map(|v| bf16::from_f32(*v)).collect();
            let mut got = vec![bf16::ZERO; n];
            fused_bf16(kind, &a, &b, &mut got);
            for i in 0..n {
                let want = bf16::from_f32(kind.apply(a[i].to_f32(), b[i].to_f32()));
                assert_eq!(got[i].to_bits(), want.to_bits(), "bf16 at {i}");
            }

            let a: Vec<f16> = a32.iter().map(|v| f16::from_f32(*v)).collect();
            let b: Vec<f16> = b32.iter().map(|v| f16::from_f32(*v)).collect();
            let mut got = vec![f16::ZERO; n];
            fused_f16(kind, &a, &b, &mut got);
            for i in 0..n {
                let want = f16::from_f32(kind.apply(a[i].to_f32(), b[i].to_f32()));
                assert_eq!(got[i].to_bits(), want.to_bits(), "f16 at {i}");
            }
        }
    }

    /// `to_dtype` here and `Tensor::to_dtype` in candle are the same function.
    /// The tensor is deliberately larger than one vector and not a multiple of
    /// eight elements.
    #[test]
    fn fast_to_dtype_matches_candle_bit_for_bit() {
        let dev = candle_core::Device::Cpu;
        let n = 1234usize;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 - 600.0) / 7.0).collect();
        let t = Tensor::from_slice(&src, n, &dev).unwrap();
        for narrow in [DType::F16, DType::BF16] {
            let mine = to_dtype(&t, narrow).unwrap();
            let theirs = t.to_dtype(narrow).unwrap();
            assert_eq!(
                mine.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap(),
                theirs.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap()
            );
            let back_mine = to_dtype(&mine, DType::F32).unwrap().to_vec1::<f32>().unwrap();
            let back_theirs = theirs.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(back_mine, back_theirs);
        }
    }

    /// A broadcasting operand takes the fused path and still agrees with the
    /// three-pass form. Rotary embedding multiplies exactly this way -- a
    /// `(1, 1, s, d)` table against a `(b, h, s, d)` activation -- so a fast
    /// path that quietly refused it would leave the model on the slow one.
    #[test]
    fn fused_arithmetic_broadcasts_and_still_matches_the_slow_path() {
        let dev = candle_core::Device::Cpu;
        let a = Tensor::rand(-2f32, 2f32, (2, 3, 40), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let b = Tensor::rand(-2f32, 2f32, (1, 3, 40), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let fast = fused_arith(Fused::Mul, &a, &b, DType::BF16).unwrap().unwrap();
        let slow = a
            .to_dtype(DType::F32)
            .unwrap()
            .broadcast_mul(&b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        assert_eq!(fast.dims(), slow.dims());
        assert_eq!(
            fast.flatten_all().unwrap().to_vec1::<bf16>().unwrap(),
            slow.flatten_all().unwrap().to_vec1::<bf16>().unwrap()
        );
    }
}
