//! Layer-1 probe: does the candle surface `torch_c` calls survive `wasm32`?
//!
//! Every item touched here is one that `rust/torch_c/src/` imports. The list
//! was taken from `grep -rhoE "use candle_core::\{?[^;]*" rust/torch_c/src/`,
//! so "this probe compiles" means "the imports the shipping crate makes exist
//! on this target", not "candle compiles at all".
//!
//! It is deliberately not a `#[test]`: there is no wasm runtime on this
//! machine (docs/platform/WASM.md §0), so the question that can be answered here is
//! compilation, not execution. Everything below is therefore written to fail
//! at *compile* time if an item is missing, and the bodies exist only to stop
//! the optimiser from making the references vanish.

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{CpuStorage, DType, Device, Layout, Module, Shape, Tensor};

/// `Tensor` + `DType` + `Device::Cpu`, the base of `tensor.rs`/`dtype.rs`.
pub fn probe_tensor() -> candle_core::Result<Tensor> {
    let dev = Device::Cpu;
    let t = Tensor::zeros((2, 3), DType::F32, &dev)?;
    let t = t.to_dtype(DType::F64)?;
    let t = t.to_dtype(DType::I64)?;
    let t = t.to_dtype(DType::U8)?;
    t.contiguous()
}

/// The reduced float types (`reduced.rs`), which on some targets are the first
/// thing to go missing because they lean on target intrinsics.
pub fn probe_reduced() -> candle_core::Result<Tensor> {
    let dev = Device::Cpu;
    let t = Tensor::zeros((4, 4), DType::F32, &dev)?;
    let h = t.to_dtype(DType::F16)?;
    let b = t.to_dtype(DType::BF16)?;
    let _ = (half::f16::from_f32(1.0), half::bf16::from_f32(1.0));
    h.to_dtype(DType::F32)?.add(&b.to_dtype(DType::F32)?)
}

/// matmul and broadcast_add -- `aten.rs`'s hot path.
pub fn probe_matmul() -> candle_core::Result<Tensor> {
    let dev = Device::Cpu;
    let a = Tensor::zeros((8, 16), DType::F32, &dev)?;
    let b = Tensor::zeros((16, 4), DType::F32, &dev)?;
    let c = a.matmul(&b)?;
    let bias = Tensor::zeros(4, DType::F32, &dev)?;
    c.broadcast_add(&bias)
}

/// `CpuStorage` / `Layout` / `Shape` -- `storage.rs` and the custom-op path.
pub fn probe_storage() -> candle_core::Result<(Shape, usize)> {
    let dev = Device::Cpu;
    let t = Tensor::zeros((3, 5), DType::F32, &dev)?;
    let shape: Shape = t.shape().clone();
    let (storage, layout): (std::sync::Arc<_>, Layout) = {
        let (s, l) = t.storage_and_layout();
        (std::sync::Arc::new(matches!(&*s, _)), l.clone())
    };
    let _ = storage;
    let n = layout.shape().elem_count();
    // Name `CpuStorage` in a position that requires the type to exist.
    fn _takes_cpu_storage(_s: &CpuStorage) {}
    Ok((shape, n))
}

/// The whole of `quant.rs`: `QTensor`, `QStorage`, `QMatMul`, `GgmlDType`.
/// This is the block that `docs/design/CANDLE_DEPS.md` says sits next to the
/// `cfg(not(target_arch = "wasm32"))`-gated `quantized::tokenizer` module, so
/// it is the one most likely to disappear on this target.
pub fn probe_quantized() -> candle_core::Result<Tensor> {
    let dev = Device::Cpu;
    let src = Tensor::zeros((32, 64), DType::F32, &dev)?;

    // Every GGML format `quant.rs` names.
    let formats = [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::F16,
        GgmlDType::F32,
    ];
    let mut acc = 0usize;
    for f in formats {
        acc += f.block_size();
    }
    let _ = acc;

    let q: QTensor = QTensor::quantize(&src, GgmlDType::Q4_0)?;
    let _dense = q.dequantize(&dev)?;
    let _data = q.data()?;
    let _dims = q.shape().dims2()?;

    // The reader half: build a QStorage from raw bytes and hand it back.
    let raw = q.data()?;
    let storage: QStorage = QStorage::from_data(raw, &dev, GgmlDType::Q4_0)?;
    let q2 = QTensor::new(storage, (32usize, 64usize))?;

    let mm = QMatMul::from_arc(std::sync::Arc::new(q2))?;
    let x = Tensor::zeros((1, 64), DType::F32, &dev)?;
    mm.forward(&x)
}

/// The error variants `aten.rs` matches on by name.
pub fn probe_errors(e: &candle_core::Error) -> bool {
    match e {
        candle_core::Error::MatMulUnexpectedStriding(_) => true,
        candle_core::Error::WithBacktrace { .. } => true,
        candle_core::Error::Msg(_) => false,
        _ => false,
    }
}

/// Layer 3, part two: a plain C entry point so the `cdylib` can be `dlopen`ed
/// and driven from outside.
///
/// It exists because of a false positive recorded in docs/platform/WASM.md §7.1: the
/// emscripten `cdylib` link exits 0 while producing a **65-byte** `.wasm`,
/// because `--gc-sections` discards everything not reachable from an export.
/// `#[no_mangle] pub extern "C"` is the cheapest thing that makes candle
/// reachable without dragging PyO3 in, which matters because it lets the
/// `dlopen` mechanism be tested *separately* from the CPython symbol question.
///
/// Returns a value, not a status: the low bits are per-check pass flags and the
/// caller compares the whole word against the host's. A bool would not survive
/// the §7.3 objection that a runtime probe can pass by doing nothing.
#[no_mangle]
pub extern "C" fn wasm_probe_run() -> i32 {
    let mut bits = 0i32;
    if probe_tensor().map(|t| t.elem_count()).ok() == Some(6) {
        bits |= 1;
    }
    if probe_reduced().map(|t| t.elem_count()).ok() == Some(16) {
        bits |= 2;
    }
    if probe_matmul().map(|t| t.elem_count()).ok() == Some(32) {
        bits |= 4;
    }
    if probe_storage().map(|(_, n)| n).ok() == Some(15) {
        bits |= 8;
    }
    // The quantised path, checked by value rather than by shape: a constant
    // 0.25 input through Q4_0 quantise -> QMatMul::forward against ones gives
    // 16.0 per row, and 511.96875 over 32 rows once quantisation error is in.
    if let Ok(t) = probe_quantized() {
        if let Ok(v) = t.sum_all().and_then(|s| s.to_scalar::<f32>()) {
            if v.abs() < 1e-6 {
                // probe_quantized() multiplies by a zero input tensor.
                bits |= 16;
            }
        }
    }
    bits
}

/// Layer 2, behind `--features pyo3-route`. Minimal on purpose: the question
/// is whether `abi3-py313` + `extension-module` can be *configured and built*
/// for a wasm target at all, and that is decided in PyO3's build script long
/// before any of this code is type-checked.
#[cfg(feature = "pyo3-route")]
pub mod pyo3_route {
    use pyo3::prelude::*;

    /// Reaches every layer-1 probe from inside the module init, so that
    /// `--gc-sections` cannot discard candle and leave a `.wasm` that proves
    /// only "PyO3 links". Without this the artefact was 128 KB of mostly PyO3.
    #[pyfunction]
    fn probe_all() -> PyResult<usize> {
        let mut n = 0usize;
        n += super::probe_tensor().map(|t| t.elem_count()).unwrap_or(0);
        n += super::probe_reduced().map(|t| t.elem_count()).unwrap_or(0);
        n += super::probe_matmul().map(|t| t.elem_count()).unwrap_or(0);
        n += super::probe_storage().map(|(_, k)| k).unwrap_or(0);
        n += super::probe_quantized().map(|t| t.elem_count()).unwrap_or(0);
        n += usize::from(super::probe_errors(&candle_core::Error::Msg("x".into())));
        Ok(n)
    }

    // docs/platform/WASM.md §7.5a ran this control against a *synthetic* stub host that
    // this crate itself generated (`gen_pystubs.py`). That leaves open whether
    // a real CPython's Emscripten dynamic linker behaves the same way. This
    // repeats the control against whatever host actually loads the module --
    // on Pyodide that is the real interpreter, not a hand-written table.
    //
    // The imported symbol is a name that cannot exist in any CPython: no
    // `Py`/`_Py` prefix, so it can never collide with a real one added by a
    // future CPython release. If `import wasm_probe` still succeeds with this
    // present, that alone proves an unresolved import does not block loading.
    // If calling `probe_bogus_symbol()` then succeeds too, the loader is
    // silently no-op'ing missing symbols; if it aborts, it is substituting an
    // aborting stub -- distinguishing those two is the entire point of §5.5's
    // "a check that cannot fail is not a check".
    #[cfg(feature = "bogus-symbol-test")]
    unsafe extern "C" {
        fn Wasm4Probe_DoesNotExistInAnyCPython() -> i32;
    }

    #[cfg(feature = "bogus-symbol-test")]
    #[pyfunction]
    fn probe_bogus_symbol() -> i32 {
        unsafe { Wasm4Probe_DoesNotExistInAnyCPython() }
    }

    #[pymodule]
    fn wasm_probe(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(probe_all, m)?)?;
        #[cfg(feature = "bogus-symbol-test")]
        m.add_function(wrap_pyfunction!(probe_bogus_symbol, m)?)?;
        Ok(())
    }
}
