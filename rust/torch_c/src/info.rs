//! `torch.finfo` / `torch.iinfo`.
//!
//! Pure dtype metadata, so it belongs next to `dtype.rs` rather than in the
//! bootstrap: these are facts about a format, not names.
//!
//! They are also import-blocking. `torch/ao/quantization/observer.py:238` has
//! `eps=torch.finfo(torch.float32).eps` as a *class-body default*, which runs
//! while `import torch` is still going. So the shim cannot defer them.
//!
//! Every number below was read off torch 2.13.0 rather than computed, because
//! two of them are not what the format suggests: `bfloat16.tiny` is
//! `1.1754943508222875e-38` (float32's, not bfloat16's own denormal floor),
//! and `resolution` is a rounded decimal (`0.01` for bfloat16) rather than a
//! representable value. Deriving these would have produced different numbers
//! that looked right.
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::dtype::{PyDtype, TorchDType};
use crate::err::not_implemented;

/// `bits, eps, max, min, tiny, resolution, and the dtype torch reports`.
///
/// Complex dtypes report the metadata of their *component* float, including
/// `dtype` -- `torch.finfo(torch.complex64).dtype` is `float32`. That is
/// torch's behaviour, measured.
fn finfo_row(dtype: TorchDType) -> Option<(u32, f64, f64, f64, f64, f64, TorchDType)> {
    use TorchDType::*;
    Some(match dtype {
        Float64 | Complex128 => (
            64,
            2.220446049250313e-16,
            1.7976931348623157e308,
            -1.7976931348623157e308,
            2.2250738585072014e-308,
            1e-15,
            Float64,
        ),
        Float32 | Complex64 => (
            32,
            1.1920928955078125e-07,
            3.4028234663852886e38,
            -3.4028234663852886e38,
            1.1754943508222875e-38,
            1e-06,
            Float32,
        ),
        Float16 | Complex32 => (
            16,
            0.0009765625,
            65504.0,
            -65504.0,
            6.103515625e-05,
            0.001,
            Float16,
        ),
        BFloat16 => (
            16,
            0.0078125,
            3.3895313892515355e38,
            -3.3895313892515355e38,
            // Not bfloat16's own smallest normal: torch reports float32's.
            1.1754943508222875e-38,
            0.01,
            BFloat16,
        ),
        Float8E4M3FN => (8, 0.125, 448.0, -448.0, 0.015625, 1.0, Float8E4M3FN),
        Float8E4M3FNUZ => (8, 0.125, 240.0, -240.0, 0.0078125, 1.0, Float8E4M3FNUZ),
        Float8E5M2 => (8, 0.25, 57344.0, -57344.0, 6.103515625e-05, 1.0, Float8E5M2),
        Float8E5M2FNUZ => (
            8,
            0.125,
            57344.0,
            -57344.0,
            3.0517578125e-05,
            1.0,
            Float8E5M2FNUZ,
        ),
        // Unsigned exponent-only format: `min` is the smallest *positive*
        // value, not a negative number. torch reports it that way.
        Float8E8M0FNU => (
            8,
            1.0,
            1.7014118346046923e38,
            5.877471754111438e-39,
            5.877471754111438e-39,
            1.0,
            Float8E8M0FNU,
        ),
        // `float4_e2m1fn_x2` is absent on purpose: torch itself refuses it
        // (`"epsilon" not implemented for 'Float4_e2m1fn_x2'`).
        _ => return None,
    })
}

fn iinfo_row(dtype: TorchDType) -> Option<(u32, i128, i128)> {
    use TorchDType::*;
    Some(match dtype {
        UInt8 => (8, 0, u8::MAX as i128),
        UInt16 => (16, 0, u16::MAX as i128),
        UInt32 => (32, 0, u32::MAX as i128),
        UInt64 => (64, 0, u64::MAX as i128),
        Int8 => (8, i8::MIN as i128, i8::MAX as i128),
        Int16 => (16, i16::MIN as i128, i16::MAX as i128),
        Int32 => (32, i32::MIN as i128, i32::MAX as i128),
        Int64 => (64, i64::MIN as i128, i64::MAX as i128),
        // torch refuses `torch.bool` here, in so many words. Matching that
        // keeps `bool` from quietly behaving as an integer type -- the same
        // guardrail BOOL.md §7 counts among the six that aliasing would erase.
        _ => return None,
    })
}

#[pyclass(name = "finfo", module = "torch", frozen)]
pub struct PyFinfo {
    #[pyo3(get)]
    bits: u32,
    #[pyo3(get)]
    eps: f64,
    #[pyo3(get)]
    max: f64,
    #[pyo3(get)]
    min: f64,
    #[pyo3(get)]
    tiny: f64,
    #[pyo3(get)]
    smallest_normal: f64,
    #[pyo3(get)]
    resolution: f64,
    #[pyo3(get)]
    dtype: String,
}

#[pymethods]
impl PyFinfo {
    #[new]
    #[pyo3(signature = (dtype = None))]
    fn new(dtype: Option<PyDtype>) -> PyResult<Self> {
        // torch defaults to the current default dtype. That used to be fixed
        // at float32 here because `set_default_dtype` was not implemented; it
        // is now a process-global, and `torch.finfo()` follows it upstream --
        // measured: under a float64 default, `torch.finfo()` reports
        // `dtype=float64`, `max=1.79769e+308`.
        let tag = dtype.map(|d| d.tag()).unwrap_or_else(crate::dtype::default_float);
        let (bits, eps, max, min, tiny, resolution, reported) =
            finfo_row(tag).ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "torch.{} is not a floating point dtype",
                    tag.name()
                ))
            })?;
        Ok(Self {
            bits,
            eps,
            max,
            min,
            tiny,
            smallest_normal: tiny,
            resolution,
            dtype: reported.name().to_string(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "finfo(resolution={}, min={}, max={}, eps={}, smallest_normal={}, tiny={}, dtype={})",
            self.resolution, self.min, self.max, self.eps, self.smallest_normal, self.tiny,
            self.dtype
        )
    }
}

#[pyclass(name = "iinfo", module = "torch", frozen)]
pub struct PyIinfo {
    #[pyo3(get)]
    bits: u32,
    #[pyo3(get)]
    min: i128,
    #[pyo3(get)]
    max: i128,
    #[pyo3(get)]
    dtype: String,
}

#[pymethods]
impl PyIinfo {
    #[new]
    #[pyo3(signature = (dtype = None))]
    fn new(dtype: Option<PyDtype>) -> PyResult<Self> {
        let tag = dtype.map(|d| d.tag()).unwrap_or(TorchDType::Int64);
        let (bits, min, max) = iinfo_row(tag).ok_or_else(|| {
            not_implemented(format!("torch.{} is not supported by torch.iinfo", tag.name()))
        })?;
        Ok(Self {
            bits,
            min,
            max,
            dtype: tag.name().to_string(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "iinfo(min={}, max={}, dtype={})",
            self.min, self.max, self.dtype
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFinfo>()?;
    m.add_class::<PyIinfo>()?;
    Ok(())
}
