// The comparison harness, shared by both probe binaries.
//
// Both the `ash` route and the `wgpu` route are judged by *this* code, not by
// two lookalike copies. That is the point: if the two routes disagree, the
// disagreement has to be in the GPU path, and it cannot be an artefact of one
// probe grading itself more leniently than the other.

// ---------------------------------------------------------------------------
// inputs
// ---------------------------------------------------------------------------

/// Deterministic, so the CPU reference and the GPU see identical bytes and so
/// two runs on two devices are comparable. Values land in [-1, 1) with an
/// 8-bit mantissa, which keeps every input exactly representable -- a
/// difference in the output is then the arithmetic, never the input.
pub fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (s >> 33) as u32;
            ((bits & 0x1ff) as f32 - 256.0) / 256.0
        })
        .collect()
}

pub fn cpu_vecadd(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Same loop order as the shader: accumulate over k in ascending order. Written
/// so that neither side is free to reassociate -- `mul_add` is *not* used here,
/// because the point is to find out whether the GPU fuses when we did not.
pub fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += a[row * k + t] * b[t * n + col];
            }
            c[row * n + col] = acc;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// comparison
// ---------------------------------------------------------------------------

pub struct Cmp {
    pub n: usize,
    pub bit_identical: usize,
    pub max_abs: f32,
    pub max_ulp: u32,
    pub first_bad: Option<(usize, f32, f32)>,
    pub nonfinite: usize,
}

/// Distance in representable floats. Bit-identical is 0 ULP; one rounding step
/// apart is 1. Reporting this rather than a tolerance is what lets the caller
/// tell "the plumbing is wrong" (huge, or NaN) from "the GPU contracted a
/// multiply-add" (1-2 ULP) -- a tolerance would report both as "pass".
pub fn ulp_diff(x: f32, y: f32) -> u32 {
    if x == y {
        return 0;
    }
    if !x.is_finite() || !y.is_finite() {
        return u32::MAX;
    }
    // Map to a monotonic ordering across the sign boundary.
    let key = |v: f32| -> i64 {
        let b = v.to_bits() as i64;
        if b < 0 { i64::from(i32::MIN) - (b - i64::from(i32::MIN)) - 1 } else { b }
    };
    let d = (key(x) - key(y)).unsigned_abs();
    u32::try_from(d).unwrap_or(u32::MAX)
}

pub fn compare(got: &[f32], want: &[f32]) -> Cmp {
    let mut c = Cmp {
        n: want.len(),
        bit_identical: 0,
        max_abs: 0.0,
        max_ulp: 0,
        first_bad: None,
        nonfinite: 0,
    };
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if !g.is_finite() {
            c.nonfinite += 1;
        }
        if g.to_bits() == w.to_bits() {
            c.bit_identical += 1;
            continue;
        }
        let ulp = ulp_diff(g, w);
        let abs = (g - w).abs();
        if abs > c.max_abs {
            c.max_abs = abs;
        }
        if ulp > c.max_ulp {
            c.max_ulp = ulp;
        }
        if c.first_bad.is_none() {
            c.first_bad = Some((i, g, w));
        }
    }
    c
}

pub fn report(label: &str, c: &Cmp) -> bool {
    let exact = c.bit_identical == c.n;
    println!(
        "  {label:<22} n={} bit-identical={}/{} max_abs={:e} max_ulp={} nonfinite={}",
        c.n, c.bit_identical, c.n, c.max_abs, c.max_ulp, c.nonfinite
    );
    if let Some((i, g, w)) = c.first_bad {
        println!("  {:<22} first differing: [{i}] gpu={g:e} ({:#010x}) cpu={w:e} ({:#010x})", "", g.to_bits(), w.to_bits());
    }
    if exact {
        println!("  {:<22} VERDICT: bit-identical to CPU", "");
    } else if c.nonfinite == 0 && c.max_ulp <= 2 {
        println!("  {:<22} VERDICT: within {} ULP of CPU (not bit-identical)", "", c.max_ulp);
    } else {
        println!("  {:<22} VERDICT: MISMATCH", "");
    }
    exact || (c.nonfinite == 0 && c.max_ulp <= 2)
}

/// Negative control. `VK_PROBE_TAMPER=<n>` perturbs a single element of the CPU
/// reference by n ULP (default 1024), and the pass/fail sense is inverted: the
/// run is correct only if every case now reports MISMATCH. A harness that still
/// says PASS under tamper is one whose verdict cannot fail, and a verdict that
/// cannot fail says nothing about the run that did pass.
///
/// n is settable because the first version of this hardcoded n=1 and reported
/// MISSED -- correctly, since `report` passes anything within 2 ULP. That is a
/// real property of the criterion and not a bug in it: 1-2 ULP is the signature
/// of a contracted multiply-add, which we tolerate. The control has to perturb
/// past the criterion to test the criterion, so the default is well past it.
pub fn tamper_ulp() -> Option<u32> {
    let v = std::env::var("VK_PROBE_TAMPER").ok()?;
    Some(v.parse().unwrap_or(1024))
}

pub fn tamper(want: &mut [f32]) {
    if let Some(n) = tamper_ulp() {
        let i = want.len() / 3;
        let bits = want[i].to_bits();
        // Move away from zero so the step is n representable floats, whatever
        // the sign.
        want[i] = f32::from_bits(if want[i] >= 0.0 { bits.wrapping_add(n) } else { bits.wrapping_sub(n) });
    }
}

