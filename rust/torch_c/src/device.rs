//! `torch._C.device`.
//!
//! Deliberately *not* a wrapper around `candle_core::Device`. In torch,
//! `torch.device("cuda")` is constructible on a CPU-only build -- it is a label,
//! and only using it fails. candle's `Device` is the opposite: the enum variant
//! carries a live handle, so it cannot represent a device this build has no
//! backend for. Storing the label and resolving it on use keeps torch's
//! semantics and makes the failure land where torch puts it.
//!
//! docs/DEVICE_ABS.md §3 is where that decision was re-examined once there was
//! something to measure it against. Two things came out of it and live here:
//!
//! *The label is validated.* A label that accepts any string is not a label,
//! it is a free-form note -- `torch.device("cuad")` used to construct fine and
//! then fail at `resolve()` with a message naming a device type nobody asked
//! for. Upstream rejects at construction against a closed list of device
//! types, and that list is the vocabulary `torch.distributed`'s backend
//! registration keys off, so it has to be the same list.
//!
//! *The label is the authority on the index; candle is not.* `from_candle`
//! below reconstructs a label from a live handle, and that direction is lossy:
//! candle's `Cuda`/`Metal` variants do not surface an ordinal through the API
//! this crate builds against, so the reconstruction hardcodes 0. That is sound
//! only while every device kind this build can produce has exactly one device.
//! It is true today (CPU only) and it stops being true the moment a second
//! accelerator of the same kind is addressable -- at which point `PyTensorBase`
//! has to carry the label the way it already carries `tag` for dtype. See
//! docs/DEVICE_ABS.md §3.2.
use candle_core::Device;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::err::not_implemented;

/// The closed vocabulary of device types, in upstream's own order.
///
/// Transcribed from the `RuntimeError` torch 2.13.0 raises for an unknown
/// device string, character for character, so that the message this shim
/// produces for the same mistake is the message the user would have got:
///
/// ```text
/// >>> torch.device("nosuchdevice")
/// RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl,
/// ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia,
/// privateuseone device type at start of device string: nosuchdevice
/// ```
///
/// Note what is *not* in it: there is no `npu` spelling at all (DESIGN.md
/// §11.1's accelerator table). `vulkan` is already a torch device type, so a
/// future Vulkan backend needs no new spelling here; an NPU would have to
/// arrive as `privateuseone`, which is the slot upstream reserves for exactly
/// that.
pub const DEVICE_TYPES: [&str; 20] = [
    "cpu",
    "cuda",
    "ipu",
    "xpu",
    "mkldnn",
    "opengl",
    "opencl",
    "ideep",
    "hip",
    "ve",
    "fpga",
    "maia",
    "xla",
    "lazy",
    "vulkan",
    "mps",
    "meta",
    "hpu",
    "mtia",
    "privateuseone",
];

fn runtime_err(message: String) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(message)
}

/// Split `"cuda:1"` into `("cuda", Some(1))`, rejecting what upstream rejects.
///
/// Every branch here is a measured upstream refusal, not a guess:
///
/// | input | torch 2.13.0 |
/// |---|---|
/// | `""` | `RuntimeError: Device string must not be empty` |
/// | `" cpu"` | `RuntimeError: Invalid device string: ' cpu'` |
/// | `"CPU"` | `RuntimeError: Expected one of cpu, ... device type ...` |
/// | `"cuda:-1"` | `RuntimeError: Device index must not be negative` |
fn parse_device_string(text: &str) -> PyResult<(String, Option<i64>)> {
    if text.is_empty() {
        return Err(runtime_err("Device string must not be empty".to_string()));
    }
    let (kind, index) = match text.split_once(':') {
        Some((kind, digits)) => {
            // Upstream's parser wants digits -- `"cuda:+1"` and `"cuda:1 "` are
            // `Invalid device string`, not a negative-index error, because the
            // negative check happens after a successful parse.
            let parsed: i64 = digits
                .parse()
                .map_err(|_| runtime_err(format!("Invalid device string: '{text}'")))?;
            (kind, Some(parsed))
        }
        None => (text, None),
    };
    if !DEVICE_TYPES.contains(&kind) {
        return Err(runtime_err(format!(
            "Expected one of {} device type at start of device string: {text}",
            DEVICE_TYPES.join(", ")
        )));
    }
    if index.is_some_and(|i| i < 0) {
        return Err(runtime_err("Device index must not be negative".to_string()));
    }
    Ok((kind.to_string(), index))
}

#[pyclass(name = "device", module = "torch._C", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyDevice {
    #[pyo3(get, name = "type")]
    pub kind: String,
    #[pyo3(get)]
    pub index: Option<i64>,
}

impl PyDevice {
    pub fn cpu() -> Self {
        Self {
            kind: "cpu".to_string(),
            index: None,
        }
    }

    /// The one `meta` device.
    ///
    /// Index-less on purpose and by measurement, not by simplification:
    /// upstream normalises every meta index away, so `torch.zeros(2,
    /// device="meta:7").device` is `device(type='meta')`, exactly as
    /// `device="cpu:3"` reports plain `cpu`. `meta:7` is still a *label* that
    /// constructs (the closed vocabulary accepts it and `==` distinguishes it
    /// from bare `meta`); it is only tensors that forget the index.
    pub fn meta() -> Self {
        Self {
            kind: "meta".to_string(),
            index: None,
        }
    }

    /// Does this label name the meta device?
    ///
    /// The one place a device kind is branched on outside `resolve`, because
    /// meta is the one kind that is *not* a candle backend: it resolves to no
    /// handle at all rather than to a handle this build lacks.
    pub fn is_meta(&self) -> bool {
        self.kind == "meta"
    }

    /// Build from a validated type/index pair. The only constructor Rust code
    /// should use, so that no path can create a label the Python constructor
    /// would have refused.
    pub fn checked(kind: &str, index: Option<i64>) -> PyResult<Self> {
        let (kind, parsed) = parse_device_string(kind)?;
        if parsed.is_some() && index.is_some() {
            return Err(runtime_err(format!(
                "type (string) must not include an index because index was passed \
                 explicitly: {kind}"
            )));
        }
        let index = index.or(parsed);
        if index.is_some_and(|i| i < 0) {
            return Err(runtime_err("Device index must not be negative".to_string()));
        }
        Ok(Self { kind, index })
    }

    /// `torch.device(x)` for any `x` torch accepts there: a string, another
    /// `device`, or an integer index.
    ///
    /// Taking `Bound<PyAny>` rather than `&str` is the point. `torch.device`
    /// is idempotent upstream -- `torch.device(torch.device("cpu"))` is a
    /// no-op copy -- and the vendored tree relies on that: `Module.to`'s
    /// argument, `_parse_to`'s device slot and every `device=` keyword are
    /// normalised by calling `torch.device` on whatever arrived. Refusing a
    /// `device` there (as this used to, with `'device' object is not an
    /// instance of 'str'`) breaks the normalisation everything else assumes.
    pub fn coerce(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(device) = value.extract::<PyDevice>() {
            return Ok(device);
        }
        if let Ok(text) = value.extract::<String>() {
            let (kind, index) = parse_device_string(&text)?;
            return Ok(Self { kind, index });
        }
        // `bool` is a subclass of `int` in Python and upstream rejects
        // `torch.device(True)` at the parser, so it is filtered before the
        // integer branch rather than caught by it.
        if value.is_instance_of::<pyo3::types::PyBool>() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "device() received an invalid combination of arguments - got (bool)",
            ));
        }
        if let Ok(index) = value.extract::<i64>() {
            // Upstream reads a bare integer as "index `n` of the current
            // accelerator" -- measured on this host, `torch.device(0)` is
            // `device(type='mps', index=0)`, i.e. it goes through
            // `torch.accelerator.current_accelerator()`. This build has no
            // accelerator (see `_get_accelerator` in bootstrap.py), so there is
            // no device type to attach the index to, and saying that is more
            // use than the `'int' object is not an instance of 'str'` this
            // used to raise.
            return Err(runtime_err(format!(
                "torch._C shim: torch.device({index}) means index {index} of the current \
                 accelerator, and this build has none -- name the device type instead, \
                 e.g. torch.device(\"cpu\", {index})"
            )));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "device() received an invalid combination of arguments - got ({})",
            value.get_type().name()?,
        )))
    }

    /// The backend this label resolves to. Only CPU exists today; Metal and
    /// CUDA are feature-gated off in `Cargo.toml` on purpose (device builds
    /// must not link them), so asking for one is a loud failure, not a silent
    /// fallback to CPU.
    ///
    /// The index is deliberately *not* checked against a device count: there is
    /// exactly one CPU, and `cpu:3` resolves to it the way upstream's does
    /// (measured: `torch.zeros(2, device="cpu:3").device` is `device(type='cpu')`).
    pub fn resolve(&self) -> PyResult<Device> {
        match self.kind.as_str() {
            "cpu" => Ok(Device::Cpu),
            // The one accelerator this build has, and it is one *arm* rather
            // than a subsystem because candle already owns the backend:
            // `Device::Metal(MetalDevice)` is a variant of the same closed enum
            // that `docs/VULKAN2.md` §5.1 found had nowhere to put a Vulkan
            // handle. So an `mps` tensor is an ordinary `candle_core::Tensor`
            // and needs no `tensor::Repr` arm, no dispatcher arm and no kernel
            // of ours -- every kernel in this crate that already goes through
            // candle runs on the GPU here because candle's own op does.
            // docs/VULKAN3.md §1 is why that asymmetry put `mps` first.
            //
            // Not a silent fallback in either direction. Where the Metal
            // feature is not compiled in -- Android, Linux, wasm, all of which
            // fail `target_vendor = "apple"` in `Cargo.toml` -- this arm does
            // not exist and `mps` falls through to the `other` arm's refusal
            // naming it. Where it is compiled in but no GPU answers,
            // `new_metal` returns its own error and it is raised, not swallowed.
            //
            // The index is passed through rather than pinned to 0, unlike
            // `from_candle`'s hardcoding: here it comes from the label the
            // caller wrote, so `mps:1` asks candle for device 1 and gets
            // candle's own out-of-range error rather than device 0's results
            // under device 1's name.
            #[cfg(target_vendor = "apple")]
            "mps" => Self::metal_device(self.index.unwrap_or(0).max(0) as usize),
            // `meta` is not a backend this build is missing -- it is a device
            // with no backend *by definition*, so it never becomes a candle
            // handle. A caller that reaches here with `meta` has forgotten to
            // branch on `is_meta()` before resolving, and saying so is more
            // use than repeating the "not available" message the other
            // nineteen kinds share.
            "meta" => Err(not_implemented(
                "torch._C shim: the meta device has no backend to resolve to -- a \
                 meta tensor holds shape and dtype and no storage, so this call site \
                 has to branch on PyDevice::is_meta() before resolving (docs/META.md)",
            )),
            other => Err(not_implemented(format!(
                "device not available in torch._C shim: {other}"
            ))),
        }
    }

    /// One `MetalDevice` per index, for the process.
    ///
    /// **Not an optimisation -- without it `mps` is broken, and it was broken
    /// in the obvious-looking first version of the arm above.** `Device::new_metal`
    /// *constructs* a device: it opens its own `MTLCommandQueue` and takes a
    /// fresh id. candle's `Device::same_device` compares those ids, so calling
    /// `resolve()` twice produced two handles that did not consider themselves
    /// equal, and the mixed-device gate in `aten.rs` then rejected two `mps`
    /// tensors against each other with
    ///
    /// ```text
    /// Expected all tensors to be on the same device, but found at least two
    /// devices, mps:0 and mps:0!
    /// ```
    ///
    /// -- a message that names the same device twice, which is what a
    /// per-call constructor looks like from the outside. Caching makes
    /// `resolve()` an accessor, which is what every caller already assumed it
    /// was and what the `Cpu` arm has always been.
    ///
    /// Keyed by index and a `Vec` rather than a `OnceLock<Device>`, because
    /// `mps:1` on a two-GPU Mac must not silently hand back device 0's handle
    /// -- the exact class of mistake `from_candle`'s hardcoded index comment
    /// warns about, from the other direction.
    ///
    /// A poisoned lock is recovered rather than raised on: the only thing this
    /// mutex guards is a lookup table, so a panic elsewhere cannot have left it
    /// meaningfully inconsistent, and refusing every subsequent `mps` call for
    /// the life of the process would be a worse failure than the one that
    /// poisoned it.
    #[cfg(target_vendor = "apple")]
    fn metal_device(index: usize) -> PyResult<Device> {
        use std::sync::{Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<Vec<(usize, Device)>>> = OnceLock::new();
        let mut cached = CACHE
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, device)) = cached.iter().find(|(i, _)| *i == index) {
            return Ok(device.clone());
        }
        let device = Device::new_metal(index).map_err(|e| {
            not_implemented(format!(
                "torch._C shim: the mps device is compiled in but did not open                  (metal device {index}): {e}"
            ))
        })?;
        cached.push((index, device.clone()));
        Ok(device)
    }

    /// A label for a live candle handle.
    ///
    /// **This direction is lossy and the loss is load-bearing.** candle's
    /// `Cuda`/`Metal` variants carry an ordinal that this crate cannot read
    /// (both features are off, so the inner types are opaque here), so the
    /// index is hardcoded to 0. Sound while each kind has at most one device;
    /// wrong the moment it does not. Nothing in this build can reach those two
    /// arms -- `resolve()` refuses every non-CPU label before a handle could be
    /// made -- so the hardcoding is unreachable rather than merely untested,
    /// and it is left visible instead of `unreachable!()` so that turning a
    /// feature on fails a review rather than silently mislabelling a tensor.
    pub fn from_candle(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::cpu(),
            Device::Cuda(_) => Self {
                kind: "cuda".to_string(),
                index: Some(0),
            },
            Device::Metal(_) => Self {
                kind: "mps".to_string(),
                index: Some(0),
            },
        }
    }

    /// Do two labels name the same physical device?
    ///
    /// Not `==`. `torch.device("cpu")` and `torch.device("cpu:0")` are
    /// *unequal* upstream (measured: `False`, and their hashes differ too), but
    /// a tensor made with either reports plain `cpu` and the two interoperate
    /// freely. Equality is a property of the label; this is a property of what
    /// the label points at, and the mixed-device check in `aten.rs` wants the
    /// second one.
    pub fn same_physical_device(&self, other: &Self) -> bool {
        self.kind == other.kind
            && match (self.index, other.index) {
                (Some(a), Some(b)) => a == b,
                // An index-less label means "the current device of this kind",
                // and with one device per kind that is device 0.
                _ => true,
            }
    }
}

#[pymethods]
impl PyDevice {
    /// `torch.device("cpu")`, `torch.device("cuda", 0)`, `torch.device(other)`.
    ///
    /// Hand-parsed rather than declared with `#[pyo3(signature = ...)]`
    /// because the first parameter has two names upstream and PyO3 gives a
    /// parameter one. Both are reachable in the wild and both were measured:
    /// `torch.device(type="cuda", index=1)` and `torch.device(device="cuda:1")`
    /// each produce `device(type='cuda', index=1)` on torch 2.13.0.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut first: Option<Bound<'_, PyAny>> = None;
        let mut index: Option<i64> = None;

        if args.len() > 2 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "device() takes from 0 to 2 positional arguments but {} were given",
                args.len()
            )));
        }
        if !args.is_empty() {
            first = Some(args.get_item(0)?);
        }
        if args.len() == 2 {
            index = Some(args.get_item(1)?.extract()?);
        }
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                let key: String = key.extract()?;
                match key.as_str() {
                    "type" | "device" => {
                        if first.is_some() {
                            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                                "device() got multiple values for argument '{key}'"
                            )));
                        }
                        first = Some(value);
                    }
                    "index" => {
                        if index.is_some() {
                            return Err(pyo3::exceptions::PyTypeError::new_err(
                                "device() got multiple values for argument 'index'",
                            ));
                        }
                        index = Some(value.extract()?);
                    }
                    other => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                            "device() got an unexpected keyword argument '{other}'"
                        )));
                    }
                }
            }
        }

        let Some(first) = first else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "device() missing required argument 'type' (pos 1)",
            ));
        };

        let base = PyDevice::coerce(&first)?;
        match index {
            None => Ok(base),
            Some(index) => {
                if let Some(already) = base.index {
                    return Err(runtime_err(format!(
                        "type (string) must not include an index because index was passed \
                         explicitly: {}:{already}",
                        base.kind,
                    )));
                }
                if index < 0 {
                    return Err(runtime_err("Device index must not be negative".to_string()));
                }
                Ok(Self {
                    kind: base.kind,
                    index: Some(index),
                })
            }
        }
    }

    /// `with torch.device("meta"): ...`
    ///
    /// **A device is not a context manager; it makes one.** Upstream's
    /// `THPDevice_enter` (`torch/csrc/Device.cpp`) imports
    /// `torch.utils._device`, builds a `DeviceContext` -- a `TorchFunctionMode`
    /// -- pushes it onto the torch-function stack and returns *`self`*, not the
    /// mode. That last detail is measured, not guessed:
    /// `with torch.device("meta") as d: repr(d)` is `device(type='meta')`.
    ///
    /// It pushes directly rather than calling `DeviceContext.__enter__`, again
    /// following upstream. The Python `__enter__` does an unstack/restack dance
    /// to force the mode to the *bottom* of the stack, which is what
    /// `torch.set_default_device` wants (a default should not shadow a mode
    /// entered after it) and what a lexically nested `with` block must not have
    /// (the inner device has to win). Measured both ways: nested
    /// `with meta: with cpu:` gives `cpu` inside and `meta` outside.
    ///
    /// The whole thing hangs on a real mode stack existing, and on every
    /// factory consulting it. `bootstrap.py` `_install_torch_function_modes`
    /// and `_torch_level_function` are the other two thirds; without them this
    /// method would make `with torch.device("meta"):` a block that succeeded
    /// and changed nothing, which docs/DEVICE_ABS.md §7.2 argued is worse than
    /// refusing.
    fn __enter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let context = py
            .import("torch.utils._device")?
            .getattr("DeviceContext")?
            .call1((slf,))?;
        py.import("torch._C")?
            .getattr("_push_on_torch_function_stack")?
            .call1((context,))?;
        Ok(slf.clone().into_any())
    }

    /// Pops what `__enter__` pushed. The arguments are the exception triple
    /// the protocol passes and upstream ignores; a `None` return propagates any
    /// exception, which is what a device context should do.
    #[pyo3(signature = (*_exc))]
    fn __exit__(slf: &Bound<'_, Self>, _exc: &Bound<'_, PyTuple>) -> PyResult<()> {
        let py = slf.py();
        py.import("torch._C")?
            .getattr("_pop_torch_function_stack")?
            .call0()?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        match self.index {
            Some(index) => format!("device(type='{}', index={})", self.kind, index),
            None => format!("device(type='{}')", self.kind),
        }
    }

    pub fn __str__(&self) -> String {
        match self.index {
            Some(index) => format!("{}:{}", self.kind, index),
            None => self.kind.clone(),
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyDevice>() {
            Ok(other) => self.kind == other.kind && self.index == other.index,
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hash = 0u64;
        for byte in self.kind.as_bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(*byte as u64);
        }
        hash.wrapping_mul(31)
            .wrapping_add(self.index.unwrap_or(-1) as u64)
    }

    /// Makes a device picklable, which `torch.save`/`torch.load` and every
    /// `copy.deepcopy` of a config object need.
    ///
    /// Upstream's shape, measured: `torch.device('cuda', 1).__reduce__()` is
    /// `(<class 'torch.device'>, ('cuda', 1))` and the index-less form drops
    /// the second element rather than passing `None` -- which matters, because
    /// `torch.device('cpu', None)` is not the same call as `torch.device('cpu')`
    /// once `index` is parsed positionally.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        let kind = this.kind.clone().into_py_any(py)?;
        let args = match this.index {
            Some(index) => PyTuple::new(py, [kind, index.into_py_any(py)?])?,
            None => PyTuple::new(py, [kind])?,
        };
        PyTuple::new(py, [slf.get_type().into_py_any(py)?, args.into_py_any(py)?])
    }
}

/// The rule `check_devices_agree` applies, exposed so it can be tested.
///
/// It is `_shim_`-prefixed for the reason `_shim_overloads` and `_shim_target`
/// are: upstream has no such name, and shim-only introspection should be
/// impossible to mistake for surface.
///
/// The reason it exists at all is that the rule is otherwise unreachable. The
/// mixed-device gate in `aten.rs` cannot fire while `resolve()` refuses every
/// non-CPU label, and this crate has **no runnable Rust unit tests** -- it is
/// `crate-type = ["cdylib"]` with pyo3's `extension-module`, so `cargo test`
/// builds a harness that aborts on the first `_Py*` symbol
/// (`dyld: symbol not found in flat namespace '_PyExc_BaseException'`, which is
/// how it behaved before this file had a `#[cfg(test)]` block too). A
/// `#[test]` here would have looked like coverage and run nowhere.
#[pyfunction]
#[pyo3(name = "_shim_same_device")]
fn shim_same_device(left: PyDevice, right: PyDevice) -> bool {
    left.same_physical_device(&right)
}

// ---------------------------------------------------------------------------
// The `mps` host-readback gate (docs/MPS.md)
// ---------------------------------------------------------------------------

/// The ops whose kernels in this crate read a dispatched tensor's bytes back
/// to host memory to compute their result.
///
/// **Deriving this list mechanically is the whole design, so the derivation is
/// written down here and re-run as a test.** `aten.rs` is scanned for the four
/// candle calls that move device bytes to the host -- `to_vec0`, `to_vec1`,
/// `to_vec2`, `to_vec3`, `to_scalar`, `to_cpu` -- and an op is on this list if
/// its kernel makes one, or calls one of the six helpers that do
/// (`read_flat`, `side_from_tensor`, `mask_to_indices`, `nan_along_dim`,
/// `narrow_through`, `local_scalar_dense`).
/// `test_the_mps_readback_list_is_what_the_kernels_actually_do` re-runs that
/// scan against the source and fails if the answer has moved, which is what
/// stops the list from decaying as `aten.rs` changes -- a kernel that acquires
/// a readback fails the suite until it is named here.
///
/// **Why a list of the unsafe ops rather than a list of the safe ones.** The
/// safe direction would be the conservative choice if the unsafe set were
/// unknowable, and it is not: candle's Metal backend has **no silent CPU
/// fallback**. Every op it does not implement bails
/// (`candle-core-0.11.0/src/metal_backend/mod.rs`), which is why
/// `aten.sort.default` on an mps tensor already raised
/// `Metal contiguous to_dtype F32 F64 not implemented` before this gate
/// existed. So the only way an mps tensor's arithmetic can happen on the CPU
/// is that a kernel *here* read it back, and that is a property of this
/// crate's own source, which the scan above can see completely at the level it
/// looks. An allowlist would have paid for that with over-refusal -- `abs` and
/// `neg` read back only on their integral path and run on the GPU for floats,
/// and an op-level allowlist cannot express that.
///
/// **What the scan cannot see** is stated so nobody reads it as more than it
/// is: it follows one level of helper calls, chosen by name, so a readback
/// added *two* levels down inside a helper this list does not know about would
/// not be caught. That is why the test also asserts that every function in
/// `aten.rs` holding a readback marker is either a kernel on this list, one of
/// the six helpers, or on a named exemption list -- so a new readback anywhere
/// in the file has to be classified by a human before the suite goes green.
///
/// **Four ops left this list by being rewritten, not by being excused**
/// (docs/MPSFWD.md §3): `aten.neg.default` and `prims.neg.default` are now
/// `0 - x` in `int64` on the device instead of a `to_vec1` and `wrapping_neg`
/// in Rust; `aten.pow.Tensor_Scalar` is `widen_f64` + candle's `powf` (the
/// same `f64::powf` the host loop called) with squaring-by-multiplication on
/// the integral path; `aten.cumsum.default` is `n - 1` narrow/add pairs in the
/// loop's own order. Each of those is a SmolLM2 forward's path -- `rotate_half`,
/// RMSNorm's `x.pow(2)` and the attention mask -- and none of them is a
/// widening of this gate: the scan re-derives the list from the kernels and
/// would put them straight back if the readback were still there.
pub const MPS_HOST_READBACK_OPS: [&str; 72] = [
    "aten._fft_c2c.default",
    "aten._fft_c2r.default",
    "aten._fft_r2c.default",
    "aten._grouped_mm.default",
    "aten._log_softmax.default",
    "aten._safe_softmax.default",
    "aten._softmax.default",
    "aten.abs.default",
    "aten.abs_.default",
    "aten.acos.default",
    "aten.adaptive_avg_pool1d.default",
    "aten.adaptive_avg_pool2d.default",
    "aten.argmax.default",
    "aten.avg_pool2d.default",
    "aten.bitwise_and.Scalar",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_not.default",
    "aten.bitwise_or.Scalar",
    "aten.bitwise_or.Tensor",
    "aten.bitwise_xor.Scalar",
    "aten.bitwise_xor.Tensor",
    "aten.bucketize.Scalar",
    "aten.bucketize.Tensor",
    "aten.div.Scalar_mode",
    "aten.div.Tensor_mode",
    "aten.erfinv.default",
    "aten.expm1.default",
    "aten.expm1_.default",
    "aten.fmod.Scalar",
    "aten.fmod.Tensor",
    "aten.gather.default",
    "aten.histc.default",
    "aten.index.Tensor",
    "aten.index_add.default",
    "aten.index_add_.default",
    "aten.index_put_.default",
    "aten.isin.Tensor_Tensor",
    "aten.linalg_qr.default",
    "aten.log2.default",
    "aten.log2_.default",
    "aten.masked_scatter.default",
    "aten.masked_select.default",
    "aten.max.default",
    "aten.max.dim",
    "aten.max.other",
    "aten.max_pool1d.default",
    "aten.max_pool2d.default",
    "aten.maximum.default",
    "aten.min.default",
    "aten.min.dim",
    "aten.min.other",
    "aten.multinomial.default",
    "aten.native_dropout.default",
    "aten.nll_loss_forward.default",
    "aten.nonzero.default",
    "aten.one_hot.default",
    "aten.pow.Scalar",
    "aten.pow.Tensor_Tensor",
    "aten.prod.default",
    "aten.prod.dim_int",
    "aten.remainder.Scalar",
    "aten.remainder.Tensor",
    "aten.scatter.src",
    "aten.scatter.value",
    "aten.scatter_reduce.two",
    "aten.softplus.default",
    "aten.stft.center",
    "aten.stft.default",
    "aten.upsample_bicubic2d.default",
    "aten.upsample_bilinear2d.default",
    "aten.upsample_nearest2d.default",
    "aten.where.default",
];

/// The two ops that read device bytes back and are **not** refused, with the
/// reason each is different in kind from the sixty-seven above.
///
/// The scan finds these too, so leaving them out of `MPS_HOST_READBACK_OPS`
/// without saying why would look like an oversight rather than a decision.
///
/// * `aten._local_scalar_dense.default` is `.item()`. The readback *is* what
///   the caller asked for, exactly as `.cpu()` is; refusing it would refuse
///   the only way to read a value off the device. It is the mps analogue of
///   `aten._to_copy.default`, not of `aten.nonzero.default`.
/// * `aten.uniform_.default` never reads the tensor it writes. Its markers are
///   on host-generated random values and on a dtype round-trip of a
///   *constant* (`narrow_roundtrip_f32`); there is no input datum to compute
///   on the wrong device, so refusing it would cost `nn.init` on mps and buy
///   nothing.
pub const MPS_READBACK_BUT_ALLOWED: [&str; 2] = [
    "aten._local_scalar_dense.default",
    "aten.uniform_.default",
];

/// Is this candle handle a Metal one?
///
/// A free function rather than a `matches!` at the call site so that the one
/// place `aten.rs` has to know about Metal is a call by name. The variant
/// exists in candle's enum on every target (the `metal` feature changes what
/// `MetalDevice` *is*, not whether the arm is there), so no `cfg` is needed
/// and the gate cannot go missing on a target nobody compiled.
#[inline]
pub fn is_metal(device: &Device) -> bool {
    matches!(device, Device::Metal(_))
}

/// Refuse an op that would compute on the CPU under an `mps` label.
///
/// Called from the single door in `aten.rs` for exactly the dispatches whose
/// tensor arguments live on Metal, so a CPU dispatch pays nothing at all and
/// an mps dispatch pays one scan of a 67-entry static table of `&'static str`.
///
/// The refusal is `NotImplementedError` and not `RuntimeError`, matching what
/// this shim raises for an op it does not implement on a device -- because
/// that is what this is. The op is not implemented *for mps*; it is
/// implemented for the CPU, and the message says so and says how to get it.
pub fn mps_host_readback_gate(op: &str) -> PyResult<()> {
    if !MPS_HOST_READBACK_OPS.contains(&op) {
        return Ok(());
    }
    Err(not_implemented(format!(
        "{op}: not implemented for the mps device. This kernel reads the tensor \
         back to host memory and computes there, so it would return a correct \
         value that the GPU did not compute, under an mps label -- the shim \
         refuses that rather than doing it silently. Move the tensor with \
         .cpu() to ask for the CPU on purpose. {} of the ops this build \
         implements are refused on mps for this reason; \
         torch._C._shim_mps_host_readback_ops() lists them (docs/MPS.md).",
        MPS_HOST_READBACK_OPS.len(),
    )))
}

/// The gate's table, readable from Python so a test can check the *artefact*
/// rather than the constant it was compiled from.
///
/// `_shim_`-prefixed for the reason `_shim_same_device` is: upstream has no
/// such name and shim-only introspection must be impossible to mistake for
/// surface.
#[pyfunction]
#[pyo3(name = "_shim_mps_host_readback_ops")]
fn shim_mps_host_readback_ops() -> Vec<&'static str> {
    MPS_HOST_READBACK_OPS.to_vec()
}

/// The companion list, for the same reason: a test that checked only the
/// refusals could not tell a deliberate exemption from a missed op.
#[pyfunction]
#[pyo3(name = "_shim_mps_readback_but_allowed")]
fn shim_mps_readback_but_allowed() -> Vec<&'static str> {
    MPS_READBACK_BUT_ALLOWED.to_vec()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_function(wrap_pyfunction!(shim_same_device, m)?)?;
    m.add_function(wrap_pyfunction!(shim_mps_host_readback_ops, m)?)?;
    m.add_function(wrap_pyfunction!(shim_mps_readback_but_allowed, m)?)?;
    Ok(())
}
