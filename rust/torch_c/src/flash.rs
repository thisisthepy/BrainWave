//! ATen's CPU flash-attention kernel, reproduced.
//!
//! **Opt-in, and off by default.** This kernel is 20x slower at T=512 than the
//! candle formulation it sits beside in `aten.rs`, so nothing reaches it unless
//! `reference_enabled()` says so. What it is *for* is being a reference: the
//! one implementation here that is bit-identical to upstream, to hold the fast
//! path against when a numeric difference has to be localised. docs/kernels/SDPA.md
//! §12 has the measurement and the decision; `apply_env` below has the switch.
//!
//! `aten::_scaled_dot_product_flash_attention_for_cpu` is not the textbook
//! formula. It is a *blocked* kernel with an online softmax, and the order in
//! which it recombines the blocks is observable: writing the same mathematics
//! out flat disagrees with it on **float32** inputs, on most elements, before
//! any reduced-precision narrowing is involved -- 3562 of 4096, 4.17e-07
//! apart, measured against upstream alone (docs/kernels/SDPA.md §3).
//!
//! Everything here is a shape of arithmetic rather than a shape of code, so it
//! is written out longhand instead of being expressed with tensor operations:
//! the answer depends on which additions happen in which order, and a tensor
//! library's reductions are free to choose their own order.
//!
//! **What this buys, measured.** For `bfloat16` and `float16` the result is
//! bit-identical to upstream: 0 of 226136 elements differ across eleven shapes
//! crossed with causal and masked, and 0 of 103680 inside a real SmolLM2-135M
//! forward (docs/kernels/SDPA.md §3, §6). Every arithmetic step is either exact or
//! reproducible in portable code, because the two matrix products upstream
//! reaches for reduced-precision inputs are *its own* portable kernel and not
//! a BLAS.
//!
//! **What it does not buy.** For `float32`/`float64` upstream calls the
//! platform BLAS (Accelerate on this host), whose summation order is not
//! portable: 159706 of 226136 elements differ, by at most 2.4e-07
//! (docs/kernels/SDPA.md §5). And matching this one kernel does **not** make
//! `bfloat16` inference reproducible in general -- perturbing only the GEMM's
//! accumulation order makes upstream disagree with *itself* on one prompt in
//! three, so no independent implementation can promise token identity there
//! (docs/kernels/SDPA.md §1).
//!
//! Two defects were found in this file by checking those claims rather than
//! believing the comment that used to assert them, and neither is visible to a
//! tolerance: eleven of the polynomial constants were decimal literals that
//! re-rounded the hex values in their own comments (docs/kernels/SDPA.md §4.1), and
//! the masked `qk * scale + mask` strode by the accumulator's vector width
//! instead of the mask dtype's (§4.4).

use candle_core::{DType, Device, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};

/// Whether `aten._scaled_dot_product_flash_attention_for_cpu` computes with
/// this kernel instead of candle's tensor ops.
///
/// `false`, because this kernel costs 20x at T=512 and attention is the hot
/// path of every forward pass (docs/kernels/SDPA.md §12 has the measurement and the
/// argument). What the switch buys is a *reference*: an implementation that is
/// bit-identical to upstream for `bfloat16`/`float16`, to compare the fast
/// path against when a numeric difference has to be localised. Two defects
/// were found that way and neither was visible to a tolerance.
static REFERENCE: AtomicBool = AtomicBool::new(false);

/// Reads the switch. Called once per `sdpa` call, on both paths.
///
/// **This is the "cost of the check" the opt-in has to justify, so it is a
/// relaxed load of a `static` and nothing else** -- no lock, no `env::var`, no
/// lazy initialisation to race on. On every target this compiles to a plain
/// load; there is no fence, because there is nothing to order against. The
/// cheapest shape this op is measured at is tens of microseconds, so the load
/// is some seven orders of magnitude below the thing it guards.
///
/// The env var is read *once*, at module init (`lib.rs::_C`), rather than here
/// -- reading it per call would be a heap allocation and a lock on the process
/// environment, which is exactly the measurable cost this avoids.
pub fn reference_enabled() -> bool {
    REFERENCE.load(Ordering::Relaxed)
}

/// Sets the switch and answers what it was.
///
/// Process-global and not scoped, matching `gemm::set_threading_threshold`
/// (`lib.rs`), the other knob of this shape. A caller that wants it scoped
/// restores the returned value; `test_shim.py` and `tools/golden/cases.py`
/// both do, so that turning it on for the sdpa cases cannot leak into the
/// hundreds of cases that follow them.
pub fn set_reference(enabled: bool) -> bool {
    REFERENCE.swap(enabled, Ordering::Relaxed)
}

/// `BW_SDPA_REFERENCE` -- the switch's env spelling, read once at module init.
///
/// **Why not an upstream name.** The instruction this repository works under is
/// to spell things the way upstream already spells them, so upstream was read
/// rather than recalled, and it has no name for this:
///
///   * `torch.backends.cuda.enable_flash_sdp` / `torch.nn.attention.sdpa_kernel`
///     select *which aten op the dispatcher calls*, not which arithmetic
///     implements one op. Upstream's `SDPBackend.MATH` is a different op
///     sequence -- `mul.Scalar`, `expand`, `bmm`, `_safe_softmax` -- which
///     this shim refuses by name (`bootstrap.py`, `scaled_dot_product_attention`)
///     because `_safe_softmax` has no kernel here. Borrowing that name would
///     have `enable_flash_sdp(False)` quietly change how one op adds up where
///     upstream changes the whole op list, and would leave
///     `flash_sdp_enabled()` describing neither branch: upstream's default for
///     it is `True`, and ours has to be the candle path.
///   * `torch.backends.cpu.get_cpu_capability` is read-only, and its env var
///     `ATEN_CPU_CAPABILITY` picks an *ISA tier* for every kernel in the build,
///     not a reference implementation for one.
///   * `torch.use_deterministic_algorithms` is about run-to-run reproducibility
///     on one build. Both branches here already have that. Users set it for its
///     documented purpose, and hanging a 20x cost off it would be a trap.
///
/// So the name is this repository's own, in this repository's own spelling
/// (`BW_GEMM_THREADING_THRESHOLD`, `BRAINWAVE_CHECK_BOOL`), and the in-process
/// form is `torch._C._shim_sdpa_reference` -- the `_shim_` prefix that
/// `_shim_target` established for names that are ours and not upstream's.
pub fn apply_env() {
    if let Ok(raw) = std::env::var("BW_SDPA_REFERENCE") {
        set_reference(!matches!(raw.as_str(), "" | "0"));
    }
}

/// Which storage dtype the probabilities are written back to between the two
/// matrix products. Upstream keeps a separate reduced-precision buffer for
/// `bfloat16`/`float16` and narrows into it; for the wider dtypes there is no
/// second buffer and the accumulator is used directly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Narrowing {
    /// `float32`/`float64`: no narrowing, the probabilities stay in the
    /// accumulate dtype.
    None,
    BFloat16,
    Float16,
}

/// `float(bfloat16(x))`, rounding to nearest with ties to even.
///
/// The truncating version of this is what docs/numerics/BF16.md §2 found in the
/// elementwise `add`; it is spelled out here rather than reached through a
/// tensor round-trip because it runs once per probability.
pub fn narrow_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    if (bits >> 23) & 0xff == 0xff && bits & 0x007f_ffff != 0 {
        // A NaN keeps its sign and is quietened, exactly as c10's convert does.
        return f32::from_bits(((bits >> 16) | 0x40) << 16);
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

/// `float(float16(x))`, rounding to nearest with ties to even, including the
/// subnormal range and the overflow to infinity.
pub fn narrow_f16(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        // Infinity survives; a NaN stays a NaN.
        return x;
    }
    let shifted = exponent - 127 + 15;
    if shifted >= 0x1f {
        return f32::from_bits(sign | 0x7f80_0000);
    }
    if shifted <= 0 {
        // Subnormal in float16, or below it. `shift` is how far the float32
        // significand (with its implicit bit) has to move to land on the
        // float16 subnormal grid, which is a fixed 2^-24.
        if shifted < -10 {
            return f32::from_bits(sign);
        }
        let full = mantissa | 0x0080_0000;
        let shift = (14 - shifted) as u32;
        let half = 1u32 << (shift - 1);
        let remainder = full & ((1u32 << shift) - 1);
        let mut grid = full >> shift;
        if remainder > half || (remainder == half && grid & 1 == 1) {
            grid += 1;
        }
        // grid * 2^-24, built back up as a float32.
        return f32::from_bits(sign) + f32::from_bits(sign | 0x3380_0000) * grid as f32
            - f32::from_bits(sign | 0x3380_0000) * 0.0;
    }
    let mut half_bits = ((shifted as u32) << 10) | (mantissa >> 13);
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && half_bits & 1 == 1) {
        half_bits += 1;
    }
    // The rounding above may carry into the exponent and even reach infinity,
    // which is the right answer; rebuilding from the fields keeps that.
    let exponent16 = (half_bits >> 10) & 0x1f;
    if exponent16 >= 0x1f {
        return f32::from_bits(sign | 0x7f80_0000);
    }
    let mantissa32 = (half_bits & 0x3ff) << 13;
    f32::from_bits(sign | (((exponent16 as i32 - 15 + 127) as u32) << 23) | mantissa32)
}

/// `Vectorized<float>::fexp_u20()` on one lane.
///
/// The softmax fusion kernel picks this exponential -- not the accurate one --
/// whenever the probabilities are on their way to a reduced-precision buffer,
/// which is every `bfloat16` and `float16` call. It is a third-degree
/// polynomial with re-tuned coefficients, so it is *not* within an ulp of
/// `exp`; reproducing the output means reproducing the polynomial.
///
/// Nothing here couples the lanes: both out-of-range cases are per-lane
/// selects, so walking the lanes one at a time gives the vector's answer.
fn fexp_u20(x: f32) -> f32 {
    // Spelled as bit patterns, not decimals. Rust has no hexadecimal float
    // literal, and a decimal one is a *re-rounding* of the constant: the
    // eight-digit decimals that stood here before disagreed with the hex
    // values in the comments -- `LN2` by one ulp, `C3` by 0.0015 -- and the
    // resulting kernel differed from upstream on 11% of `bfloat16` elements.
    // Nine significant digits would round-trip, but nothing checks that a
    // literal has nine; a bit pattern cannot be one digit short.
    const LOWER: f32 = f32::from_bits(0xc2af_5dc1); // -0x1.5ebb82p+6
    const UPPER: f32 = f32::from_bits(0x42b0_c0a5); //  0x1.61814ap+6
    const INV_LN2: f32 = f32::from_bits(0x3fb8_aa3b); //  0x1.715476p+0
    const LN2: f32 = f32::from_bits(0x3f31_7218); //  0x1.62e43p-1
    const C2: f32 = f32::from_bits(0x3e2a_c976); //  0x1.5592ecp-3
    const C3: f32 = f32::from_bits(0x3f00_be9a); //  0x1.017d34p-1
    let below = x < LOWER;
    let above = x > UPPER;
    let n = (x * INV_LN2).round();
    let r = (-n).mul_add(LN2, x);
    let scaled = ((saturating_i32(n) as u32) << 23) as u32;
    let r2 = r * r;
    let q = r.mul_add(C2, C3);
    let s = 1.0f32 + r;
    let p = q.mul_add(r2, s);
    let y = f32::from_bits(p.to_bits().wrapping_add(scaled));
    if below {
        0.0
    } else if above {
        f32::INFINITY
    } else {
        y
    }
}

/// `Vectorized<float>::exp_u20()`, the accurate-to-20-bits polynomial, on one
/// lane -- valid only when the *whole* four-lane group is in range. See
/// [`exp_u20_group_bails`].
fn exp_u20(x: f32) -> f32 {
    // Bit patterns, for the reason given in [`fexp_u20`].
    const INV_LN2: f32 = f32::from_bits(0x3fb8_aa3b); // 0x1.715476p+0
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200); // 0x1.62e4p-1
    const LN2_LO: f32 = f32::from_bits(0x35bf_be8e); // 0x1.7f7d1cp-20
    const C0: f32 = f32::from_bits(0x3c07_2010); // 0x1.0e4020p-7
    const C1: f32 = f32::from_bits(0x3d2b_9f17); // 0x1.573e2ep-5
    const C2: f32 = f32::from_bits(0x3e2a_af33); // 0x1.555e66p-3
    const C3: f32 = f32::from_bits(0x3eff_fedb); // 0x1.fffdb6p-2
    const C4: f32 = f32::from_bits(0x3f7f_fff6); // 0x1.ffffecp-1
    let n = (x * INV_LN2).round();
    let mut r = (-n).mul_add(LN2_HI, x);
    r = (-n).mul_add(LN2_LO, r);
    let e = (saturating_i32(n) as u32) << 23;
    let scale = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let r2 = r * r;
    let mut p = r.mul_add(C0, C1);
    let mut q = r.mul_add(C2, C3);
    q = p.mul_add(r2, q);
    p = C4 * r;
    let poly = q.mul_add(r2, p);
    poly.mul_add(scale, scale)
}

/// The lane coupling in `exp_u20`: if any lane of the four exceeds this, the
/// vector abandons the polynomial and calls Sleef's `expf` for all four.
///
/// This is why a `float32` causal call cannot be reproduced exactly here even
/// with the right BLAS -- the `-inf` a masked column carries trips the bail
/// for its whole group, and the replacement is Sleef's kernel, which this
/// substitutes the platform `expf` for (docs/kernels/SDPA.md §5.2).
const U20_SPECIAL_BOUND: f32 = f32::from_bits(0x42ae_af15); // 0x1.5d5e2ap+6

fn exp_u20_group_bails(group: &[f32]) -> bool {
    group.iter().any(|x| !(x.abs() <= U20_SPECIAL_BOUND))
}

/// `vcvtq_s32_f32`, which saturates rather than wrapping. `n` reaches `-inf`
/// whenever a masked column does, and the saturated result is what makes the
/// polynomial's answer for that lane discardable.
fn saturating_i32(n: f32) -> i32 {
    if n.is_nan() {
        0
    } else if n >= 2147483648.0 {
        i32::MAX
    } else if n <= -2147483648.0 {
        i32::MIN
    } else {
        n as i32
    }
}

/// The float32 and float64 halves of the kernel, which differ in their vector
/// width and therefore in the association order of the one reduction whose
/// order is visible -- the per-row sum of exponentials.
pub trait FlashFloat: Copy + PartialOrd + std::fmt::Debug {
    /// `Vectorized<Self>::size()` on a 128-bit machine.
    const LANES: usize;
    const ZERO: Self;
    const ONE: Self;
    const NEG_INFINITY: Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn recip(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn is_nan(self) -> bool;
    fn read(tensor: &Tensor) -> candle_core::Result<Vec<Self>>;
    fn build(values: Vec<Self>, shape: Vec<usize>, device: &Device)
        -> candle_core::Result<Tensor>;
    /// The `scale` argument arrives as a `double` on the op's schema whatever
    /// the input dtype is; upstream narrows it to the accumulate type before
    /// the multiply, so this does too.
    fn from_f64(value: f64) -> Self;
    /// The exponential the fusion kernel uses when the probabilities stay in
    /// this dtype. `bails` carries the lane coupling of `exp_u20`.
    fn soft_exp(self, bails: bool) -> Self;
    /// `float(narrowed(self))` for the reduced-precision buffer.
    fn narrow(self, to: Narrowing) -> Self;
}

impl FlashFloat for f32 {
    const LANES: usize = 4;
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const NEG_INFINITY: f32 = f32::NEG_INFINITY;
    fn add(self, other: f32) -> f32 {
        self + other
    }
    fn sub(self, other: f32) -> f32 {
        self - other
    }
    fn mul(self, other: f32) -> f32 {
        self * other
    }
    fn mul_add(self, a: f32, b: f32) -> f32 {
        f32::mul_add(self, a, b)
    }
    fn recip(self) -> f32 {
        1.0 / self
    }
    fn exp(self) -> f32 {
        f32::exp(self)
    }
    fn ln(self) -> f32 {
        f32::ln(self)
    }
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    fn read(tensor: &Tensor) -> candle_core::Result<Vec<f32>> {
        tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    }
    fn build(values: Vec<f32>, shape: Vec<usize>, device: &Device) -> candle_core::Result<Tensor> {
        Tensor::from_vec(values, shape, device)
    }
    fn from_f64(value: f64) -> f32 {
        value as f32
    }
    fn soft_exp(self, bails: bool) -> f32 {
        if bails {
            f32::exp(self)
        } else {
            exp_u20(self)
        }
    }
    fn narrow(self, to: Narrowing) -> f32 {
        match to {
            Narrowing::None => self,
            Narrowing::BFloat16 => narrow_bf16(self),
            Narrowing::Float16 => narrow_f16(self),
        }
    }
}

impl FlashFloat for f64 {
    /// Two, not four -- and that changes where the sum of exponentials
    /// associates. `Vectorized<double>` has no aarch64 horizontal-add
    /// specialisation either, so its fold is the plain left-to-right one.
    const LANES: usize = 2;
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const NEG_INFINITY: f64 = f64::NEG_INFINITY;
    fn add(self, other: f64) -> f64 {
        self + other
    }
    fn sub(self, other: f64) -> f64 {
        self - other
    }
    fn mul(self, other: f64) -> f64 {
        self * other
    }
    fn mul_add(self, a: f64, b: f64) -> f64 {
        f64::mul_add(self, a, b)
    }
    fn recip(self) -> f64 {
        1.0 / self
    }
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    fn ln(self) -> f64 {
        f64::ln(self)
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    fn read(tensor: &Tensor) -> candle_core::Result<Vec<f64>> {
        // `widen_f64` rather than `to_dtype`: docs/numerics/FLOAT8C.md §1 -- candle's
        // `F8E4M3 -> F64` arm does not terminate. This op refuses float8 at the
        // door, so the route matters only if that gate is ever narrowed.
        crate::aten::widen_f64(tensor)?.flatten_all()?.to_vec1::<f64>()
    }
    fn build(values: Vec<f64>, shape: Vec<usize>, device: &Device) -> candle_core::Result<Tensor> {
        Tensor::from_vec(values, shape, device)
    }
    fn from_f64(value: f64) -> f64 {
        value
    }
    /// `Vectorized<double>::exp_u20()` is `exp()`, so there is no polynomial
    /// and no lane coupling here.
    fn soft_exp(self, _bails: bool) -> f64 {
        f64::exp(self)
    }
    fn narrow(self, _to: Narrowing) -> f64 {
        self
    }
}

/// `sum()` in ATen's portable gemm: four independent partial sums folded at
/// the end, which is a different answer from a running total and is the answer
/// the reduced-precision matrix products give.
///
/// Both products this kernel makes have `k` running over contiguous memory in
/// one operand and strided in the other, so the strides are parameters rather
/// than assumed.
fn ilp_dot<A: FlashFloat>(a: &[A], a_step: usize, b: &[A], b_step: usize, k: usize) -> A {
    let mut partial = [A::ZERO; 4];
    let mut i = 0;
    while i + 4 <= k {
        for (lane, slot) in partial.iter_mut().enumerate() {
            *slot = a[(i + lane) * a_step].mul_add(b[(i + lane) * b_step], *slot);
        }
        i += 4;
    }
    while i < k {
        partial[0] = a[i * a_step].mul_add(b[i * b_step], partial[0]);
        i += 1;
    }
    partial[0] = partial[0].add(partial[1]);
    partial[0] = partial[0].add(partial[2]);
    partial[0] = partial[0].add(partial[3]);
    partial[0]
}

/// `_exp_reduce_sum_fusion_kernel`: writes `exp(row - max)` into `out` and
/// answers the row's sum.
///
/// The lane layout is the whole point. When the probabilities are narrowed the
/// kernel strides by the *reduced* dtype's vector width (eight lanes at 128
/// bits) with a four-lane tail; otherwise it strides by four (two for
/// `float64`) with no tail. Both leave a scalar remainder, and the remainder
/// uses the platform `exp` rather than either polynomial. The eight partial
/// sums are then folded pairwise, not left to right.
fn exp_reduce_sum<A: FlashFloat>(row: &[A], out: &mut [A], max: A, narrow: Narrowing) -> A {
    let size = row.len();
    let wide = narrow != Narrowing::None;
    let group = if wide { 8 } else { A::LANES };
    let mut accumulator = [A::ZERO; 8];
    let mut tail = [A::ZERO; 4];
    let end_group = group * (size / group);
    let end_lane = A::LANES * (size / A::LANES);

    let mut i = 0;
    while i < end_group {
        let bails = !wide && {
            let shifted: Vec<f32> = (0..group)
                .map(|l| {
                    let value = row[i + l].sub(max);
                    // Only float32 has the coupling, and only there is this
                    // conversion lossless.
                    if A::LANES == 4 {
                        f32::from_bits(0) + as_f32(value)
                    } else {
                        0.0
                    }
                })
                .collect();
            A::LANES == 4 && exp_u20_group_bails(&shifted)
        };
        for lane in 0..group {
            let shifted = row[i + lane].sub(max);
            let value = if wide {
                from_f32::<A>(fexp_u20(as_f32(shifted)))
            } else {
                shifted.soft_exp(bails)
            };
            accumulator[lane] = accumulator[lane].add(value);
            out[i + lane] = value.narrow(narrow);
        }
        i += group;
    }
    while i < end_lane {
        for lane in 0..A::LANES {
            let shifted = row[i + lane].sub(max);
            let value = if wide {
                from_f32::<A>(fexp_u20(as_f32(shifted)))
            } else {
                shifted.soft_exp(false)
            };
            tail[lane] = tail[lane].add(value);
            out[i + lane] = value.narrow(narrow);
        }
        i += A::LANES;
    }

    for lane in 0..4.min(A::LANES.max(if wide { 4 } else { 0 })) {
        accumulator[lane] = accumulator[lane].add(tail[lane]);
    }
    let mut folded = [A::ZERO; 4];
    if wide {
        for lane in 0..4 {
            folded[lane] = accumulator[lane].add(accumulator[lane + 4]);
        }
    } else {
        folded[..A::LANES].copy_from_slice(&accumulator[..A::LANES]);
    }
    let mut sum = if A::LANES == 4 {
        folded[0].add(folded[2]).add(folded[1].add(folded[3]))
    } else {
        folded[0].add(folded[1])
    };

    while i < size {
        let value = row[i].sub(max).exp();
        sum = sum.add(value);
        out[i] = value.narrow(narrow);
        i += 1;
    }
    sum
}

// The two casts above exist only so the `float32` polynomial can be reached
// from generic code; on `float64` the branch that uses them is never taken.
fn as_f32<A: FlashFloat>(value: A) -> f32 {
    let mut probe = [0.0f32; 1];
    // A is either f32 or f64; the f64 branch is dead in every caller.
    if A::LANES == 4 {
        probe[0] = unsafe { *(&value as *const A as *const f32) };
    }
    probe[0]
}

fn from_f32<A: FlashFloat>(value: f32) -> A {
    debug_assert_eq!(A::LANES, 4);
    unsafe { *(&value as *const f32 as *const A) }
}

/// The blocking upstream picks, from the query length alone.
pub fn split_sizes(q_len: usize, kv_len: usize) -> (usize, usize) {
    let q_split = if q_len >= 768 {
        256
    } else if q_len >= 192 {
        64
    } else {
        32
    };
    (q_split.min(q_len).max(1), 512.min(kv_len).max(1))
}

/// One `(batch, head)` slice of the kernel, laid out `[seq, head_dim]`.
pub struct Inputs<'a, A> {
    pub q: &'a [A],
    pub k: &'a [A],
    pub v: &'a [A],
    /// `[q_len, kv_len]`, already broadcast, or empty for no mask.
    pub mask: &'a [A],
}

/// Runs the kernel over one head. `out` is `[q_len, head_dim]` and `lse` is
/// `[q_len]`.
#[allow(clippy::too_many_arguments)]
pub fn attend_head<A: FlashFloat>(
    input: Inputs<'_, A>,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: A,
    is_causal: bool,
    narrow: Narrowing,
    out: &mut [A],
    lse: &mut [A],
) {
    let (q_split, kv_split) = split_sizes(q_len, kv_len);
    let has_mask = !input.mask.is_empty();

    let mut scores = vec![A::ZERO; q_split * kv_split];
    let mut probabilities = vec![A::ZERO; q_split * kv_split];
    let mut accumulated = vec![A::ZERO; q_split * head_dim];
    let mut row_max = vec![A::NEG_INFINITY; q_split];
    let mut row_sum = vec![A::ZERO; q_split];
    let mut column = vec![A::ZERO; kv_split];

    let mut m = 0;
    while m < q_len {
        let q_block = q_split.min(q_len - m);
        for row in 0..q_block {
            row_max[row] = A::NEG_INFINITY;
            row_sum[row] = A::ZERO;
        }
        let num_keys = if is_causal {
            (m + q_block).min(kv_len)
        } else {
            kv_len
        };

        let mut n = 0;
        while n < num_keys {
            let kv_block = kv_split.min(kv_len - n);

            // scale * q @ k.T, laid out [q row][kv column].
            for i in 0..kv_block {
                for j in 0..q_block {
                    scores[j * kv_block + i] = ilp_dot(
                        &input.k[(n + i) * head_dim..],
                        1,
                        &input.q[(m + j) * head_dim..],
                        1,
                        head_dim,
                    );
                }
            }

            // The causal mask is upper-left aligned: row `t` sees keys
            // `0..=t` even when the key sequence is the longer one.
            if is_causal && num_keys - n <= kv_split {
                for row in 0..q_block {
                    let last = m + row;
                    let first = if last + 1 > n { last + 1 - n } else { 0 };
                    for column in first..kv_block {
                        scores[row * kv_block + column] = A::NEG_INFINITY;
                    }
                }
            }

            for row in 0..q_block {
                let slice = &mut scores[row * kv_block..row * kv_block + kv_block];
                let block_max = if has_mask {
                    // `qk * scale + mask`, and the scaling rides along. The
                    // vector body multiplies and adds separately; only the
                    // scalar remainder is one statement and therefore fused
                    // (both measured -- fusing the body disagrees with
                    // upstream on float16, docs/kernels/SDPA.md §4.3).
                    //
                    // The body stops at a multiple of the *mask* dtype's
                    // vector width, not the accumulator's: upstream strides
                    // this loop by `Vectorized<mask_t>::size()`, and the mask
                    // arrives in the input dtype. For a `bfloat16` mask that
                    // is eight, so a 70-column block leaves six columns to the
                    // fused remainder rather than two. Reading it as four cost
                    // one element in 226136 -- one logsumexp, whose output was
                    // unaffected because the shift cancels (docs/kernels/SDPA.md §4.4).
                    let mask_row = &input.mask[(m + row) * kv_len + n..];
                    let lanes = if narrow == Narrowing::None {
                        A::LANES
                    } else {
                        8
                    };
                    let body = lanes * (kv_block / lanes);
                    for column in 0..body {
                        slice[column] = slice[column].mul(scale).add(mask_row[column]);
                    }
                    for column in body..kv_block {
                        slice[column] = slice[column].mul_add(scale, mask_row[column]);
                    }
                    slice.iter().fold(A::NEG_INFINITY, |acc, &x| {
                        if x.is_nan() || acc.is_nan() {
                            A::NEG_INFINITY.mul(A::ZERO)
                        } else if x > acc {
                            x
                        } else {
                            acc
                        }
                    })
                } else {
                    let mut best = A::NEG_INFINITY;
                    for value in slice.iter_mut() {
                        *value = value.mul(scale);
                        if value.is_nan() {
                            best = value.mul(A::ZERO);
                        } else if *value > best && !best.is_nan() {
                            best = *value;
                        }
                    }
                    best
                };

                let block_max = if row_max[row] > block_max {
                    row_max[row]
                } else {
                    block_max
                };
                let probabilities =
                    &mut probabilities[row * kv_block..row * kv_block + kv_block];
                if block_max == A::NEG_INFINITY {
                    // Every column of this row is masked out. Skipping the
                    // exponential is not an optimisation: `exp(-inf - -inf)`
                    // is a NaN, and upstream writes zeros instead.
                    for value in probabilities.iter_mut() {
                        *value = A::ZERO;
                    }
                } else {
                    let sum = exp_reduce_sum(
                        &scores[row * kv_block..row * kv_block + kv_block],
                        probabilities,
                        block_max,
                        narrow,
                    );
                    let rescale = row_max[row].sub(block_max).exp();
                    row_sum[row] = sum.add(rescale.mul(row_sum[row]));
                    row_max[row] = block_max;
                    if n > 0 {
                        for value in accumulated[row * head_dim..(row + 1) * head_dim].iter_mut() {
                            *value = value.mul(rescale);
                        }
                    }
                }
            }

            // dst += probabilities @ v.
            for i in 0..head_dim {
                for (slot, source) in column[..kv_block].iter_mut().enumerate() {
                    *source = input.v[(n + slot) * head_dim + i];
                }
                for j in 0..q_block {
                    let dot = ilp_dot(
                        &column[..kv_block],
                        1,
                        &probabilities[j * kv_block..],
                        1,
                        kv_block,
                    );
                    accumulated[j * head_dim + i] = if n == 0 {
                        dot
                    } else {
                        accumulated[j * head_dim + i].add(dot)
                    };
                }
            }

            n += kv_split;
        }

        for row in 0..q_block {
            // A row with no unmasked column has sum 0; upstream answers zeros
            // for it rather than a NaN, and reports its logsumexp from 0.
            if row_max[row] == A::NEG_INFINITY {
                row_max[row] = A::ZERO;
            }
            if row_sum[row] == A::ZERO {
                row_sum[row] = A::ONE;
            }
            // A reciprocal and a multiply, not a divide -- they differ in the
            // last bit and upstream takes the reciprocal.
            let reciprocal = row_sum[row].recip();
            for dim in 0..head_dim {
                out[(m + row) * head_dim + dim] = accumulated[row * head_dim + dim]
                    .mul(reciprocal)
                    .narrow(narrow);
            }
            lse[m + row] = row_max[row].add(row_sum[row].ln());
        }

        m += q_split;
    }
}

/// The whole `[batch, heads, seq, head_dim]` call, one `(batch, head)` slice at
/// a time.
///
/// `q`, `k` and `v` are expected in the accumulate dtype and contiguous, with
/// the key/value head count already grown to the query's (upstream repeats them
/// before the kernel too -- see `repeat_kv_heads` in `aten.rs`). `mask`, when
/// present, is `[batch, heads, q_len, kv_len]`, already broadcast, in the same
/// dtype.
///
/// Upstream parallelises this loop over `(batch, head, q_block)`; the order of
/// that loop is not observable because the slices do not interact, so walking
/// them serially answers the same bits.
pub fn attend<A: FlashFloat>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    is_causal: bool,
    narrow: Narrowing,
) -> candle_core::Result<(Tensor, Tensor)> {
    let (batch, heads, q_len, head_dim) = q.dims4()?;
    let (_, _, kv_len, _) = k.dims4()?;

    let q_values = A::read(q)?;
    let k_values = A::read(k)?;
    let v_values = A::read(v)?;
    let mask_values = match mask {
        Some(mask) => A::read(mask)?,
        None => Vec::new(),
    };

    let mut out = vec![A::ZERO; batch * heads * q_len * head_dim];
    let mut lse = vec![A::ZERO; batch * heads * q_len];
    let scale = A::from_f64(scale);

    for slice in 0..batch * heads {
        let q_at = slice * q_len * head_dim;
        let kv_at = slice * kv_len * head_dim;
        let mask_at = slice * q_len * kv_len;
        attend_head(
            Inputs {
                q: &q_values[q_at..q_at + q_len * head_dim],
                k: &k_values[kv_at..kv_at + kv_len * head_dim],
                v: &v_values[kv_at..kv_at + kv_len * head_dim],
                mask: if mask_values.is_empty() {
                    &[]
                } else {
                    &mask_values[mask_at..mask_at + q_len * kv_len]
                },
            },
            q_len,
            kv_len,
            head_dim,
            scale,
            is_causal,
            narrow,
            &mut out[q_at..q_at + q_len * head_dim],
            &mut lse[slice * q_len..(slice + 1) * q_len],
        );
    }

    Ok((
        A::build(out, vec![batch, heads, q_len, head_dim], q.device())?,
        A::build(lse, vec![batch, heads, q_len], q.device())?,
    ))
}
