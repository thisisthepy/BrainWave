//! Layer-3 executable probe: does the layer-1 candle code actually *run* on
//! wasm, as opposed to merely linking?
//!
//! docs/platform/WASM.md §1/§2 could only answer "compiles" and "links" because the first
//! pass had no runtime it was willing to use. There is one: the emsdk on this
//! machine bundles Node 24, and `wasm32-unknown-emscripten` produces a `.js`
//! loader plus a `.wasm` that Node can execute directly.
//!
//! It prints *values*, not "ok". An exit code alone would not distinguish
//! "candle ran" from "the runtime started and the code was gc'd away", which is
//! exactly the false positive docs/platform/WASM.md §2b and §7.1 both hit. Every number
//! below is checked against a value computed by hand, so a silently wrong
//! kernel is a visible diff and not a green tick.
//!
//! Not a benchmark. PEP 783 forbids `-pthread` and candle's wasm SIMD does not
//! compile (§1d), so this is the slowest configuration candle has and timing it
//! would only mislead.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};

fn main() {
    println!("== wasm_probe layer-3 runtime probe ==");
    println!("target_arch={} target_os={}", std::env::consts::ARCH, std::env::consts::OS);

    let dev = Device::Cpu;
    let mut failures = 0usize;

    // 1. Tensor construction and dtype conversion round trip.
    //    zeros(2,3) -> +1 -> sum = 6. Hand-checkable.
    match (|| -> candle_core::Result<f32> {
        let t = (Tensor::zeros((2, 3), DType::F32, &dev)? + 1.0)?;
        let s = t.sum_all()?.to_scalar::<f32>()?;
        Ok(s)
    })() {
        Ok(v) => {
            let ok = (v - 6.0).abs() < 1e-6;
            failures += usize::from(!ok);
            println!("tensor  sum(ones(2,3)) = {v}  expect 6  {}", pass(ok));
        }
        Err(e) => {
            failures += 1;
            println!("tensor  ERR {e}");
        }
    }

    // 2. matmul. A is 2x3 of ones, B is 3x4 of twos -> every entry 6, sum 48.
    match (|| -> candle_core::Result<(Vec<usize>, f32)> {
        let a = (Tensor::zeros((2, 3), DType::F32, &dev)? + 1.0)?;
        let b = (Tensor::zeros((3, 4), DType::F32, &dev)? + 2.0)?;
        let c = a.matmul(&b)?;
        Ok((c.dims().to_vec(), c.sum_all()?.to_scalar::<f32>()?))
    })() {
        Ok((dims, v)) => {
            let ok = dims == vec![2, 4] && (v - 48.0).abs() < 1e-6;
            failures += usize::from(!ok);
            println!("matmul  dims={dims:?} sum={v}  expect [2,4] 48  {}", pass(ok));
        }
        Err(e) => {
            failures += 1;
            println!("matmul  ERR {e}");
        }
    }

    // 3. Reduced precision. f16/bf16 round trips are the first thing to break
    //    when a target lacks the intrinsics candle expects.
    match (|| -> candle_core::Result<(f32, f32)> {
        let t = (Tensor::zeros((4, 4), DType::F32, &dev)? + 1.5)?;
        let h = t.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
        let b = t.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
        Ok((
            h.sum_all()?.to_scalar::<f32>()?,
            b.sum_all()?.to_scalar::<f32>()?,
        ))
    })() {
        Ok((h, b)) => {
            // 1.5 is exact in both f16 and bf16, so both sums are exactly 24.
            let ok = (h - 24.0).abs() < 1e-3 && (b - 24.0).abs() < 1e-3;
            failures += usize::from(!ok);
            println!("reduced f16sum={h} bf16sum={b}  expect 24 24  {}", pass(ok));
        }
        Err(e) => {
            failures += 1;
            println!("reduced ERR {e}");
        }
    }

    // 4. The quantised path -- the part docs/platform/WASM.md §1b singled out, and the
    //    part that on other targets dispatches to hand-written SIMD kernels.
    //    Here it must take the scalar fallback (§1d), so this is also a check
    //    that the scalar fallback is *correct*, not just present.
    match (|| -> candle_core::Result<(usize, f32, f32)> {
        let src = (Tensor::zeros((32, 64), DType::F32, &dev)? + 0.25)?;
        let q = QTensor::quantize(&src, GgmlDType::Q4_0)?;
        let nbytes = q.data()?.len();
        let deq = q.dequantize(&dev)?;
        let err = (deq - &src)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let mm = QMatMul::from_arc(std::sync::Arc::new(q))?;
        let x = (Tensor::zeros((1, 64), DType::F32, &dev)? + 1.0)?;
        let y = mm.forward(&x)?;
        Ok((nbytes, err, y.sum_all()?.to_scalar::<f32>()?))
    })() {
        Ok((nbytes, err, y)) => {
            // Q4_0 packs 32 weights into an 18-byte block: 2048 elems = 64
            // blocks = 1152 bytes. Dequantisation of a constant 0.25 is exact.
            // Each output row is 64 * 0.25 = 16, over 32 rows -> 512.
            let ok = nbytes == 1152 && err < 1e-3 && (y - 512.0).abs() < 1e-1;
            failures += usize::from(!ok);
            println!(
                "q4_0    bytes={nbytes} maxerr={err} matmulsum={y}  expect 1152 ~0 512  {}",
                pass(ok)
            );
        }
        Err(e) => {
            failures += 1;
            println!("q4_0    ERR {e}");
        }
    }

    // 5. Whether SIMD is on. §1d says `+simd128` does not compile, so this must
    //    print `false`; if it ever prints `true` the doc needs revisiting.
    println!("simd128 enabled = {}", cfg!(target_feature = "simd128"));

    println!("== failures = {failures} ==");
    if failures != 0 {
        std::process::exit(1);
    }
}

fn pass(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}
