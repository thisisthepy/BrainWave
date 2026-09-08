//! `torch._C.device`.
//!
//! Deliberately *not* a wrapper around `candle_core::Device`. In torch,
//! `torch.device("cuda")` is constructible on a CPU-only build -- it is a label,
//! and only using it fails. candle's `Device` is the opposite: the enum variant
//! carries a live handle, so it cannot represent a device this build has no
//! backend for. Storing the label and resolving it on use keeps torch's
//! semantics and makes the failure land where torch puts it.
//!
//! docs/devices/DEVICE_ABS.md §3 is where that decision was re-examined once there was
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
//! docs/devices/DEVICE_ABS.md §3.2.
use std::sync::atomic::AtomicU64;

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
            // that `docs/devices/VULKAN2.md` §5.1 found had nowhere to put a Vulkan
            // handle. So an `mps` tensor is an ordinary `candle_core::Tensor`
            // and needs no `tensor::Repr` arm, no dispatcher arm and no kernel
            // of ours -- every kernel in this crate that already goes through
            // candle runs on the GPU here because candle's own op does.
            // docs/devices/VULKAN3.md §1 is why that asymmetry put `mps` first.
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
            // The second accelerator, and it is the *same* one arm as `mps` for
            // the same structural reason: `Device::Cuda(CudaDevice)` sits beside
            // `Device::Metal(MetalDevice)` in candle's closed enum, so a `cuda`
            // tensor is an ordinary `candle_core::Tensor` -- no `tensor::Repr`
            // arm, no dispatcher arm, no kernel of ours. docs/devices/CUDA.md §1.
            //
            // **Unlike the `mps` arm this one carries no `#[cfg]`, and that is a
            // property of candle rather than a decision here.** When the `cuda`
            // feature is off, `candle_core::CudaDevice` resolves to
            // `dummy_cuda_backend::CudaDevice` and `Device::new_cuda` returns
            // `Error::NotCompiledWithCudaSupport` -- so the arm compiles on
            // every target this crate builds for, including Android, iOS and
            // wasm, and *runs* there. What it does there is refuse, by name,
            // with `reason: not_built`. That is why the refusal below is a live
            // code path on a machine with no NVIDIA GPU at all, and why this
            // round could test it rather than only wire it.
            //
            // The index comes from the label the caller wrote, so `cuda:1` asks
            // candle for device 1 and gets a refusal naming `no_device` rather
            // than device 0's results under device 1's name.
            "cuda" => Self::cuda_device(self.index.unwrap_or(0).max(0) as usize),
            "meta" => Err(not_implemented(
                "torch._C shim: the meta device has no backend to resolve to -- a \
                 meta tensor holds shape and dtype and no storage, so this call site \
                 has to branch on PyDevice::is_meta() before resolving (docs/devices/META.md)",
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

    /// One `CudaDevice` per index, for the process -- `metal_device`'s twin,
    /// and cached for the same measured reason.
    ///
    /// `Device::new_cuda` is a constructor, not a lookup: it opens a context
    /// and a stream and takes a fresh `DeviceId`, and candle's
    /// `Device::same_device` compares those ids. Two `resolve()` calls without
    /// this cache would produce two handles that do not consider themselves
    /// equal, and `aten.rs`'s mixed-device gate would then reject two `cuda`
    /// tensors against each other with a message naming the same device twice.
    /// That failure was *measured* on `mps` (docs/devices/VULKAN3.md §2); it is
    /// structural rather than Metal-specific, so it is pre-empted here rather
    /// than rediscovered on the first machine with a GPU.
    ///
    /// Keyed by index, for `metal_device`'s reason: a second GPU is the common
    /// case on CUDA hardware, and `cuda:1` silently receiving device 0's handle
    /// is the exact mistake `from_candle`'s hardcoded index warns about.
    ///
    /// **Nothing is cached unless it passed the architecture check below**, so
    /// a device that is refused once is refused every time rather than being
    /// admitted by a second caller.
    fn cuda_device(index: usize) -> PyResult<Device> {
        let mut cached = cuda_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, device)) = cached.iter().find(|(i, _)| *i == index) {
            CUDA_RESOLVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(device.clone());
        }
        let device = Device::new_cuda(index).map_err(|e| cuda_refusal(index, &e.to_string()))?;
        // The one check candle does not make, and the one this crate is in a
        // position to make: a GPU older than the kernels in the artefact.
        // Without it the failure arrives at the first kernel launch as
        // `CUDA_ERROR_NO_BINARY_FOR_GPU`, several frames inside candle and
        // attached to whichever op happened to be first.
        if let Some(problem) = cuda_arch_mismatch(&device) {
            return Err(cuda_refusal(index, &problem));
        }
        cached.push((index, device.clone()));
        CUDA_RESOLVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
            // `cuda` no longer hardcodes 0, because `resolve()` above can now
            // hand out `cuda:1` and this function is what a tensor's `.device`
            // is read back through. `cuda_stream().context().ordinal()` is the
            // ordinal candle itself opened. The `Metal` arm below is untouched
            // and still hardcodes 0 -- that is a real gap, stated rather than
            // widened, and it belongs to whoever makes `mps:1` reachable.
            Device::Cuda(_) => Self {
                kind: "cuda".to_string(),
                index: Some(cuda_ordinal(device)),
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
    /// and changed nothing, which docs/devices/DEVICE_ABS.md §7.2 argued is worse than
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
// The `mps` host-readback gate (docs/devices/MPS.md)
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
/// (docs/devices/MPSFWD.md §3): `aten.neg.default` and `prims.neg.default` are now
/// `0 - x` in `int64` on the device instead of a `to_vec1` and `wrapping_neg`
/// in Rust; `aten.pow.Tensor_Scalar` is `widen_f64` + candle's `powf` (the
/// same `f64::powf` the host loop called) with squaring-by-multiplication on
/// the integral path; `aten.cumsum.default` is `n - 1` narrow/add pairs in the
/// loop's own order. Each of those is a SmolLM2 forward's path -- `rotate_half`,
/// RMSNorm's `x.pow(2)` and the attention mask -- and none of them is a
/// widening of this gate: the scan re-derives the list from the kernels and
/// would put them straight back if the readback were still there.
///
/// **Two more left it the same way** (docs/devices/MPSATTN.md): `aten._softmax.default`
/// and `aten._safe_softmax.default` are now `max_keepdim` / `broadcast_sub` /
/// `exp` / `sum_keepdim` / `broadcast_div` on the device instead of `read_flat`
/// and a scalar loop. That pair is the one `docs/platform/RELEASE_0_1_0b0.md` §5 named
/// as the reason no transformer forwards on `mps`. §5 was **half right**: the
/// SDPA path does not go through `_softmax` (docs/devices/MPSFWD.md measured that on
/// SmolLM2 and it still holds), but an **eager** attention block does, twice a
/// layer, and a BERT with `attn_implementation="eager"` stopped there.
pub const MPS_HOST_READBACK_OPS: [&str; 85] = [
    "aten._fft_c2c.default",
    "aten._fft_c2r.default",
    "aten._fft_r2c.default",
    "aten._grouped_mm.default",
    "aten._log_softmax.default",
    "aten._unique2.default",
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
    "aten.col2im.default",
    "aten.diag.default",
    "aten.div.Scalar_mode",
    "aten.div.Tensor_mode",
    "aten.erfinv.default",
    "aten.expm1.default",
    "aten.expm1_.default",
    "aten.fmod.Scalar",
    "aten.fmod.Tensor",
    "aten.gather.default",
    "aten.histc.default",
    "aten.i0.default",
    "aten.im2col.default",
    "aten.index.Tensor",
    "aten.index_add.default",
    "aten.index_add_.default",
    "aten.index_put_.default",
    "aten.isin.Tensor_Tensor",
    "aten.linalg_qr.default",
    "aten.log2.default",
    "aten.log2_.default",
    "aten.lstm.input",
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
    "aten.repeat_interleave.Tensor",
    "aten.scatter.src",
    "aten.scatter.value",
    "aten.scatter_reduce.two",
    "aten.softplus.default",
    "aten.std.correction",
    "aten.std.default",
    "aten.std.dim",
    "aten.stft.center",
    "aten.stft.default",
    "aten.upsample_bicubic2d.default",
    "aten.upsample_bilinear2d.default",
    "aten.upsample_linear1d.default",
    "aten.upsample_nearest1d.default",
    "aten.upsample_nearest2d.default",
    "aten.var.correction",
    "aten.var.default",
    "aten.var.dim",
    "aten.where.default",
];

/// The two ops that read device bytes back and are **not** refused, with the
/// reason each is different in kind from the eighty-seven above.
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
    host_readback_gate(op, "mps", "an", "_shim_mps_host_readback_ops", "docs/devices/MPS.md")
}

/// The same gate for `cuda`, over the same derived list, and **one body**.
///
/// Two copies of a guard is the shape the round before this one had to fix
/// ("the drain guard existed twice, so neither copy could be tested"), so the
/// wording lives in `host_readback_gate` below and the two public entry points
/// differ only in the four words that name the device.
///
/// The counter is here rather than in the shared body because only the cuda
/// side has one: `_cuda_counters()` reports refusals so a run on a GPU can tell
/// "the model never touched a readback op" from "the gate fired and something
/// upstream swallowed it".
pub fn cuda_host_readback_gate(op: &str) -> PyResult<()> {
    let verdict = host_readback_gate(op, "cuda", "a", "_shim_cuda_host_readback_ops", "docs/devices/CUDA.md");
    if verdict.is_err() {
        CUDA_READBACK_REFUSALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    verdict
}

fn host_readback_gate(
    op: &str,
    device: &str,
    article: &str,
    lister: &str,
    doc: &str,
) -> PyResult<()> {
    if !MPS_HOST_READBACK_OPS.contains(&op) {
        return Ok(());
    }
    Err(not_implemented(format!(
        "{op}: not implemented for the {device} device. This kernel reads the tensor \
         back to host memory and computes there, so it would return a correct \
         value that the GPU did not compute, under {article} {device} label -- the shim \
         refuses that rather than doing it silently. Move the tensor with \
         .cpu() to ask for the CPU on purpose. {} of the ops this build \
         implements are refused on {device} for this reason; \
         torch._C.{lister}() lists them ({doc}).",
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

// ---------------------------------------------------------------------------
// cuda -- the named refusals, the gate, and the runtime evidence
// ---------------------------------------------------------------------------
//
// docs/devices/CUDA.md is the whole argument; this comment says only what a reader of
// this file needs in order not to misread the code below.
//
// **What is here and what is deliberately not.** `cuda` costs one `resolve()`
// arm because `Device::Cuda(CudaDevice)` is already a variant of candle's
// closed enum -- the `mps` asymmetry (docs/devices/VULKAN3.md §1), not the `vulkan`
// one. So there is no `Repr::Cuda`, no dispatcher arm and no kernel. What
// *does* have to exist is everything below: `mps` proved that an accelerator
// which is an ordinary `Repr::Dense` has no structural protection against this
// crate's own kernels reading the tensor back and computing on the host
// (docs/devices/MPS.md §1.1), and `cuda` inherits that exactly.
//
// **And it inherits it in a worse form, which is the one finding here that is
// about CUDA rather than about accelerators in general.** On Metal the
// float readback path was *loud*: `read_flat` widens through `f64` and Metal
// has no `F32 -> F64`, so thirteen of the fourteen probed silent fallbacks
// were on the integer path and the float ones raised (docs/devices/MPS.md §2). CUDA
// implements `f64`. The same kernels that were noisy on Metal will be silent
// on CUDA, so the gate matters more here, not less.

/// Every reason `cuda` can be unavailable, as a closed vocabulary.
///
/// The instruction for this round was that a refusal must **say which** of no
/// driver / no device / wrong arch it was. Five names rather than three,
/// because two more are real:
///
/// * `not_built` -- the artefact has no CUDA in it at all. On this project's
///   own machine, and on every wheel published so far, this is the answer, and
///   it is a live code path rather than a `cfg`'d-out one (see `resolve()`).
/// * `unclassified` -- the driver said something this table does not know. A
///   taxonomy that maps everything onto four names would be lying at exactly
///   the moment it mattered most; this arm hands back the driver's own words
///   and admits it did not recognise them.
pub const CUDA_REFUSAL_REASONS: [&str; 5] = [
    "not_built",
    "no_driver",
    "no_device",
    "wrong_arch",
    "unclassified",
];

/// `(CUresult token, reason)`, in the order they are tried.
///
/// **These tokens are not invented and not guessed from an error seen once.**
/// `cudarc::driver::DriverError`'s `Debug` -- which is also its `Display` --
/// prints `DriverError(CUresult::CUDA_ERROR_*, "<cuGetErrorString text>")`, so
/// the `CUresult` variant name is in every message candle wraps. Each token
/// below is a variant of `cudarc::driver::sys::CUresult` (cudarc 0.19.8,
/// `src/driver/sys/mod.rs`), which is the version `candle-core` 0.11.0
/// resolves. The *classification* is a judgement; the spelling is not.
///
/// Matching on the token rather than on the human sentence is deliberate: the
/// sentence comes from `cuGetErrorString` in whatever driver is installed and
/// is free to be reworded, while the enumerator name is ABI.
const CUDA_REFUSAL_TOKENS: [(&str, &str); 12] = [
    // No usable driver: the library is there (or the process would not have
    // loaded at all -- see docs/devices/CUDA.md §3) but it cannot be used.
    ("CUDA_ERROR_NOT_INITIALIZED", "no_driver"),
    ("CUDA_ERROR_STUB_LIBRARY", "no_driver"),
    ("CUDA_ERROR_SYSTEM_DRIVER_MISMATCH", "no_driver"),
    ("CUDA_ERROR_SYSTEM_NOT_READY", "no_driver"),
    ("CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE", "no_driver"),
    // A driver, but not the device that was asked for.
    ("CUDA_ERROR_NO_DEVICE", "no_device"),
    ("CUDA_ERROR_INVALID_DEVICE", "no_device"),
    ("CUDA_ERROR_DEVICE_UNAVAILABLE", "no_device"),
    ("CUDA_ERROR_DEVICE_NOT_LICENSED", "no_device"),
    // A device, but not one these kernels have code for. All four arrive at
    // module load or kernel launch, which is why `cuda_device()` checks the
    // compute capability at open time as well -- by the time one of these is
    // raised it is attached to whichever op happened to run first.
    ("CUDA_ERROR_NO_BINARY_FOR_GPU", "wrong_arch"),
    ("CUDA_ERROR_INVALID_PTX", "wrong_arch"),
    ("CUDA_ERROR_UNSUPPORTED_PTX_VERSION", "wrong_arch"),
];

/// The compute capability the kernels in *this* artefact were compiled for, or
/// `None` on a build with no CUDA in it.
///
/// Read from `CUDA_COMPUTE_CAP` at compile time. That variable is not this
/// crate's invention: `candle-kernels`'s build script detects the capability
/// with `nvidia-smi --query-gpu=compute_cap` and requires the variable when
/// there is no GPU to ask, so any CUDA build has it set. `build.rs` declares
/// the rerun dependency.
pub const CUDA_BUILT_COMPUTE_CAP: Option<&str> = option_env!("CUDA_COMPUTE_CAP");

/// Parse `"75"`, `"90a"`, `"sm_90a"` to `75` / `90` / `90`.
///
/// The `a` suffix is nvcc's "architecture-specific" marker; it does not change
/// which generation the code is for, so the comparison below drops it.
fn compute_cap_base(text: &str) -> Option<u32> {
    let text = text.trim();
    let text = text.strip_prefix("sm_").unwrap_or(text);
    let digits: String = text.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Which of `CUDA_REFUSAL_REASONS` a candle error text is.
///
/// Public and reachable from Python (`_shim_cuda_classify_refusal`) because
/// **this is the only part of the CUDA refusal that a machine with no GPU can
/// test**, and it is the part that decides what the user is told. Four of the
/// five arms cannot be produced live on the machine this was written on; a
/// classifier that is only exercised by the one arm that can would be three
/// quarters untested.
pub fn classify_cuda_refusal(detail: &str) -> &'static str {
    // candle's own words when the feature is off -- `Error::NotCompiledWith
    // CudaSupport`'s `#[error(...)]` text, verbatim. Checked first because it
    // is the only one that is not a driver answer at all.
    if detail.contains("has not been built with cuda support") {
        return "not_built";
    }
    // This crate's own architecture check, which runs before any kernel does.
    if detail.contains("compute capability") {
        return "wrong_arch";
    }
    for (token, reason) in CUDA_REFUSAL_TOKENS {
        if detail.contains(token) {
            return reason;
        }
    }
    "unclassified"
}

/// The refusal itself. `NotImplementedError`, matching every other "this build
/// cannot do that device" refusal in this crate.
///
/// The reason is in the message **as a token, near the front**, so that a
/// caller can match on it without parsing prose -- `reason: no_driver` -- and
/// the driver's own text is carried through unedited after it. Both halves
/// matter: the token is what a test asserts, the text is what a person needs.
fn cuda_refusal(index: usize, detail: &str) -> PyErr {
    let reason = classify_cuda_refusal(detail);
    CUDA_REFUSALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let advice = match reason {
        "not_built" => {
            "This artefact has no CUDA in it. The CUDA build is a separate wheel, \
             not a flag on this one: candle pins cudarc to `dynamic-linking`, so a \
             CUDA-enabled `_C.so` names libcuda/libcudart/libcublas as load-time \
             dependencies and would fail to import at all on a machine without \
             them -- taking `import torch` with it (docs/devices/CUDA.md §3)."
        }
        "no_driver" => {
            "The CUDA libraries loaded but the driver did not answer. This is \
             usually a host with the toolkit and no kernel module, or a container \
             started without `--gpus`, or a driver older than the runtime \
             (docs/devices/CUDA.md §4)."
        }
        "no_device" => {
            "The driver answered and there is no such device. Check the index \
             against `_C._cuda_probe()['device_count']`, and CUDA_VISIBLE_DEVICES \
             (docs/devices/CUDA.md §4)."
        }
        "wrong_arch" => {
            "There is a GPU and it is older than the kernels in this build. PTX \
             is forward-compatible only: the driver can JIT sm_N code onto sm_M \
             for M > N and cannot go the other way, and this build's statically \
             compiled kernels are exact-arch with no JIT at all (docs/devices/CUDA.md §2). \
             Rebuild with CUDA_COMPUTE_CAP set to this device's capability."
        }
        _ => {
            "This build does not recognise that failure. The driver's own words \
             are above, unedited; docs/devices/CUDA.md §4 has the token table this was \
             matched against and is where a new one should be added."
        }
    };
    not_implemented(format!(
        "cuda:{index} is not available -- reason: {reason}. {advice} \
         The underlying error was: {detail}"
    ))
}

/// The one `CudaDevice` per index, for the process.
///
/// A free function rather than a `static` inside `cuda_device`, because
/// `cuda_memory` below has to be able to *look* without opening.
fn cuda_cache() -> &'static std::sync::Mutex<Vec<(usize, Device)>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Vec<(usize, Device)>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

/// A device this process has already opened, or `None`.
///
/// **Never opens one.** This is the difference between an instrument and a
/// participant: `_cuda_counters()` must be able to read free device memory
/// without allocating a CUDA context as a side effect of being asked whether a
/// CUDA context exists -- which is exactly what routing it through `resolve()`
/// did in the first draft, and which would also have made every call to the
/// counters bump `resolves` and quietly ruin the deltas the counters are for.
#[allow(dead_code)]
fn cuda_cached_device(index: usize) -> Option<Device> {
    cuda_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .find(|(i, _)| *i == index)
        .map(|(_, device)| device.clone())
}

/// Is this GPU older than the kernels in the artefact?
///
/// `None` means "no objection", which on a non-CUDA build is the only answer
/// there is -- and it is never reached there, because `Device::new_cuda` has
/// already refused with `not_built`.
///
/// The comparison is `>=`, not `==`, and the asymmetry is the point: nvcc's
/// `--ptx` output for `compute_N` is JIT-compiled by the driver onto any
/// `sm_M` with `M >= N`, and never downwards.
///
/// **What this check does not cover, stated because it would otherwise read as
/// covered.** `candle-kernels` also builds `moe_*.cu` and `mmq_*.cu` into a
/// static `libmoe.a` with `-gencode arch=compute_N,code=sm_N`, which is cubin
/// and carries no PTX -- those kernels have no JIT path and need `M == N`. The
/// probe reports the mismatch; this function does not refuse on it, because
/// refusing every `M != N` would refuse the ordinary and correct case of
/// running sm_80-built PTX on an sm_89 card.
#[cfg(torch_c_cuda)]
fn cuda_arch_mismatch(device: &Device) -> Option<String> {
    let built = compute_cap_base(CUDA_BUILT_COMPUTE_CAP?)?;
    let handle = device.as_cuda_device().ok()?;
    let (major, minor) = handle.cuda_stream().context().compute_capability().ok()?;
    let found = (major as u32) * 10 + (minor as u32);
    if found >= built {
        return None;
    }
    Some(format!(
        "this device reports compute capability {major}.{minor} (sm_{found}) and \
         the kernels in this build were compiled for sm_{built}"
    ))
}

#[cfg(not(torch_c_cuda))]
fn cuda_arch_mismatch(_device: &Device) -> Option<String> {
    None
}

/// The ordinal candle actually opened, for `from_candle`.
#[cfg(torch_c_cuda)]
fn cuda_ordinal(device: &Device) -> i64 {
    device
        .as_cuda_device()
        .ok()
        .map(|d| d.cuda_stream().context().ordinal() as i64)
        .unwrap_or(0)
}

/// Unreachable on a build with no CUDA -- `resolve()` refuses `cuda` before a
/// handle could exist -- and left as a function rather than `unreachable!()`
/// for the reason the `Metal` arm's hardcoded 0 is left visible: turning the
/// feature on should fail a review, not silently mislabel a tensor.
#[cfg(not(torch_c_cuda))]
fn cuda_ordinal(_device: &Device) -> i64 {
    0
}

/// Is this candle handle a CUDA one?
///
/// `is_metal`'s twin, and a free function for the same reason: the one place
/// `aten.rs` has to know about CUDA is a call by name. The variant exists in
/// candle's enum on every target (the `cuda` feature changes what `CudaDevice`
/// *is*, not whether the arm is there), so no `cfg` is needed and the gate
/// cannot go missing on a target nobody compiled.
#[inline]
pub fn is_cuda(device: &Device) -> bool {
    matches!(device, Device::Cuda(_))
}

/// The ops refused on `cuda`, which are **the same ops, from the same
/// derivation**, as the ones refused on `mps`.
///
/// This is an alias and not a second list, and that is the whole design.
/// `MPS_HOST_READBACK_OPS` is not a statement about Metal: it is the set of
/// kernels in `aten.rs` that pull a tensor's bytes to the host and do the
/// arithmetic in Rust, derived by scanning that file (docs/devices/MPS.md §3.1). What
/// makes an op unsafe under an accelerator label is a property of *this
/// crate's kernel*, not of the accelerator, so the two devices cannot
/// legitimately disagree -- and a hand-written second list is exactly how they
/// would come to.
///
/// `test_the_cuda_refusal_list_is_the_mps_one_and_both_are_derived` asserts
/// both halves against the loaded artefact: that the two tables are equal, and
/// that the table equals the set re-derived from `aten.rs` right then.
///
/// docs/architectures/VOICE3.md §6 is why the derivation is not trusted further than it goes:
/// it follows helper calls **one level, by name**, and `var`/`std` reached the
/// list through nothing until the read was spelled at each dispatch target.
/// The companion classification test is what covers the level below that.
pub const CUDA_HOST_READBACK_OPS: &[&str] = &MPS_HOST_READBACK_OPS;

// ---------------------------------------------------------------------------
// The instrument
// ---------------------------------------------------------------------------

/// Process-wide counters, in the shape `_vulkan_counters()` established.
///
/// **Why counters and not a source-scanning test.** docs/devices/MPSATTN.md §3.1
/// records, against its own round, that moving a `read_flat` one call deeper --
/// into a helper the scan does not know by name -- passes *both* of the `mps`
/// derivation tests while keeping the readback. Every check that greps the
/// source can be defeated by moving the thing it greps for. So the three
/// numbers below are not descriptions of this crate's source:
///
/// * `resolves` / `dispatches` / `readback_refusals` are incremented at the
///   only doors those events can pass through -- `cuda_device()`, the `is_cuda`
///   arm of `aten_dispatch`, and `cuda_host_readback_gate`. There is one door
///   each, and a kernel cannot reach a CUDA tensor without going through the
///   dispatcher's.
/// * `device_free_bytes` / `device_total_bytes` are read from the **driver**,
///   at the moment `_cuda_counters()` is called, through
///   `CudaContext::mem_get_info()`. Nothing in this repository can move that
///   out of the way, because it is not in this repository: it is the GPU
///   reporting its own memory. A tensor that is on the device makes it fall;
///   an answer computed on the host does not.
///
/// **What the pair does and does not establish, so it is not over-read.**
/// `dispatches` says an op ran with CUDA tensors and was not refused by the
/// readback gate. `device_free_bytes` says the bytes are on the GPU. Together
/// with candle's CUDA backend having no silent CPU fallback -- which is a read
/// of candle's source, not a measurement this round performed -- that is the
/// argument. docs/devices/CUDA.md §6 is a procedure for someone with a GPU that closes
/// the remaining gap from outside the process, with `nvidia-smi`.
static CUDA_RESOLVES: AtomicU64 = AtomicU64::new(0);
static CUDA_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static CUDA_READBACK_REFUSALS: AtomicU64 = AtomicU64::new(0);
static CUDA_REFUSALS: AtomicU64 = AtomicU64::new(0);

/// Per-op dispatch counts, so a run on a GPU can say *which* ops ran there
/// rather than only how many.
///
/// A `Mutex` on the dispatch path is defensible only because it is inside the
/// `is_cuda` arm: a CPU or meta or mps dispatch never reaches this function, so
/// the cost is paid by the device that is being investigated and by nothing
/// else. `_vulkan_counters()` did not need this because `_vulkan_ops()` is a
/// closed list of eighteen; here the list is every op the crate implements.
static CUDA_DISPATCHED_OPS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::BTreeMap<String, u64>>,
> = std::sync::OnceLock::new();

/// Called from the one `is_cuda` arm in `aten.rs`, after the readback gate has
/// passed and before the kernel runs.
pub fn note_cuda_dispatch(op: &str) {
    CUDA_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut seen = CUDA_DISPATCHED_OPS
        .get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *seen.entry(op.to_string()).or_insert(0) += 1;
}

/// `(free, total)` device bytes, straight from the driver, or `None`.
///
/// Deliberately takes no argument and reports device 0's context only when one
/// has already been opened by `resolve()`: querying is not a reason to open a
/// context, and opening one here would make `_cuda_counters()` allocate GPU
/// memory as a side effect of being asked whether GPU memory was allocated.
#[cfg(torch_c_cuda)]
fn cuda_memory(index: usize) -> Option<(u64, u64)> {
    let device = cuda_cached_device(index)?;
    let handle = device.as_cuda_device().ok()?;
    let (free, total) = handle.cuda_stream().context().mem_get_info().ok()?;
    Some((free as u64, total as u64))
}

#[cfg(not(torch_c_cuda))]
fn cuda_memory(_index: usize) -> Option<(u64, u64)> {
    None
}

/// `_C._cuda_counters()` -- the runtime answer to "did the GPU do it?".
///
/// `device_free_bytes` and `device_total_bytes` are `None` where there is no
/// CUDA build or no device already open. `None` rather than `0`: a zero would
/// read as "the GPU has no free memory", which is a different and alarming
/// claim.
///
/// **This function opens nothing and changes no counter.** `_cuda_probe()`
/// does -- it resolves, so it increments `resolves` -- and the two are
/// deliberately different that way: a probe is a question about the machine and
/// may open a device to answer it, while the counters are the measuring
/// instrument and must not perturb what they measure. So `device_free_bytes` is
/// `None` until something has actually put a tensor on the GPU, and that is the
/// correct answer rather than a gap.
#[pyfunction]
#[pyo3(name = "_cuda_counters")]
#[pyo3(signature = (index = 0))]
fn cuda_counters(py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
    use std::sync::atomic::Ordering::Relaxed;
    let d = PyDict::new(py);
    d.set_item("resolves", CUDA_RESOLVES.load(Relaxed))?;
    d.set_item("dispatches", CUDA_DISPATCHES.load(Relaxed))?;
    d.set_item("readback_refusals", CUDA_READBACK_REFUSALS.load(Relaxed))?;
    d.set_item("refusals", CUDA_REFUSALS.load(Relaxed))?;
    let ops = PyDict::new(py);
    if let Some(seen) = CUDA_DISPATCHED_OPS.get() {
        let seen = seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        for (op, count) in seen.iter() {
            ops.set_item(op, count)?;
        }
    }
    d.set_item("ops", ops)?;
    match cuda_memory(index) {
        Some((free, total)) => {
            d.set_item("device_free_bytes", free)?;
            d.set_item("device_total_bytes", total)?;
        }
        None => {
            d.set_item("device_free_bytes", py.None())?;
            d.set_item("device_total_bytes", py.None())?;
        }
    }
    Ok(d.into_any().unbind())
}

/// `_C._cuda_probe()` -- `_vulkan_probe()`'s twin: absence is a value, not an
/// exception.
///
/// It never raises. A caller asking "is there a GPU here" should not have to
/// catch anything, and every field that cannot be answered is `None` with
/// `reason` saying which of `CUDA_REFUSAL_REASONS` it was.
#[pyfunction]
#[pyo3(name = "_cuda_probe")]
#[pyo3(signature = (index = 0))]
fn cuda_probe(py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("built", cfg!(torch_c_cuda))?;
    d.set_item("built_compute_cap", CUDA_BUILT_COMPUTE_CAP)?;
    // Parsed, because the raw value may carry nvcc's `a` suffix (`sm_90a`) and
    // the comparison in `cuda_arch_mismatch` is on the base number. Reporting
    // both means a reader can see that the parse agreed with the string.
    d.set_item(
        "built_compute_cap_base",
        CUDA_BUILT_COMPUTE_CAP.and_then(compute_cap_base),
    )?;
    let resolved = PyDevice {
        kind: "cuda".to_string(),
        index: Some(index as i64),
    }
    .resolve();
    match resolved {
        Ok(device) => {
            d.set_item("available", true)?;
            d.set_item("reason", py.None())?;
            d.set_item("error", py.None())?;
            cuda_probe_device(&d, &device)?;
        }
        Err(e) => {
            let detail = e.value(py).to_string();
            d.set_item("available", false)?;
            d.set_item("reason", classify_cuda_refusal(&detail))?;
            d.set_item("error", detail)?;
            d.set_item("device_count", py.None())?;
            d.set_item("name", py.None())?;
            d.set_item("compute_cap", py.None())?;
            d.set_item("free_bytes", py.None())?;
            d.set_item("total_bytes", py.None())?;
            d.set_item("arch_exact", py.None())?;
        }
    }
    Ok(d.into_any().unbind())
}

#[cfg(torch_c_cuda)]
fn cuda_probe_device(d: &Bound<'_, PyDict>, device: &Device) -> PyResult<()> {
    let handle = device
        .as_cuda_device()
        .map_err(|e| crate::err::candle_err("_cuda_probe", e))?;
    let stream = handle.cuda_stream();
    let context = stream.context();
    d.set_item(
        "device_count",
        cudarc::driver::CudaContext::device_count().ok(),
    )?;
    d.set_item("name", context.name().ok())?;
    let cap = context.compute_capability().ok();
    d.set_item(
        "compute_cap",
        cap.map(|(major, minor)| format!("{major}{minor}")),
    )?;
    // Whether the statically compiled (cubin, no PTX, no JIT) half of
    // `candle-kernels` has code for this exact device -- see
    // `cuda_arch_mismatch`'s note. Reported and not refused on.
    d.set_item(
        "arch_exact",
        match (cap, CUDA_BUILT_COMPUTE_CAP.and_then(compute_cap_base)) {
            (Some((major, minor)), Some(built)) => {
                Some((major as u32) * 10 + (minor as u32) == built)
            }
            _ => None,
        },
    )?;
    match context.mem_get_info() {
        Ok((free, total)) => {
            d.set_item("free_bytes", free as u64)?;
            d.set_item("total_bytes", total as u64)?;
        }
        Err(_) => {
            d.set_item("free_bytes", None::<u64>)?;
            d.set_item("total_bytes", None::<u64>)?;
        }
    }
    Ok(())
}

/// Unreachable: on a build with no CUDA, `resolve()` cannot return `Ok`.
#[cfg(not(torch_c_cuda))]
fn cuda_probe_device(_d: &Bound<'_, PyDict>, _device: &Device) -> PyResult<()> {
    Ok(())
}

/// The refusal classifier, callable from Python.
///
/// **This exists so that four refusals that cannot happen on this machine can
/// still be tested by name here**, and the test that uses it says so in its own
/// name. It is not a stand-in for running on a GPU and docs/devices/CUDA.md §8 says
/// exactly what it does and does not establish: that the taxonomy maps the
/// driver's tokens onto the five names, not that any of those states was ever
/// entered.
#[pyfunction]
#[pyo3(name = "_shim_cuda_classify_refusal")]
fn shim_cuda_classify_refusal(detail: &str) -> &'static str {
    classify_cuda_refusal(detail)
}

/// The cuda gate's table, readable from Python for the reason
/// `_shim_mps_host_readback_ops` is: a test must check the *artefact*, not the
/// constant it was compiled from.
#[pyfunction]
#[pyo3(name = "_shim_cuda_host_readback_ops")]
fn shim_cuda_host_readback_ops() -> Vec<&'static str> {
    CUDA_HOST_READBACK_OPS.to_vec()
}

/// The closed vocabulary, so a test can assert that every refusal this build
/// can produce is one of the names the documentation lists -- rather than
/// asserting the five names it happens to know about, which would go stale in
/// the one direction that matters (a sixth reason added and undocumented).
#[pyfunction]
#[pyo3(name = "_shim_cuda_refusal_reasons")]
fn shim_cuda_refusal_reasons() -> Vec<&'static str> {
    CUDA_REFUSAL_REASONS.to_vec()
}

#[cfg(test)]
mod cuda_tests {
    use super::{classify_cuda_refusal, compute_cap_base, CUDA_REFUSAL_REASONS};

    /// nvcc's architecture spellings, all three of them.
    ///
    /// The `a` suffix (`sm_90a`, `sm_100a`) is cudaforge's `auto_suffix`, which
    /// it applies to every capability at or above 90 -- so on any Hopper or
    /// newer build machine `CUDA_COMPUTE_CAP` reaches this crate *with* a
    /// suffix, and a parser that only handled digits would silently answer
    /// `None` and switch the architecture check off. That is a failure that
    /// opens rather than closes, which is the direction docs/devices/MPS.md §3.2 says
    /// not to accept.
    #[test]
    fn compute_cap_parses_every_spelling_nvcc_uses() {
        assert_eq!(compute_cap_base("75"), Some(75));
        assert_eq!(compute_cap_base("sm_80"), Some(80));
        assert_eq!(compute_cap_base("90a"), Some(90));
        assert_eq!(compute_cap_base("sm_100a"), Some(100));
        assert_eq!(compute_cap_base(" 89 "), Some(89));
        assert_eq!(compute_cap_base(""), None);
        assert_eq!(compute_cap_base("hopper"), None);
    }

    /// Each of the five reasons, from a string of the shape the driver
    /// produces, and each one distinct from the others.
    ///
    /// The texts are `cudarc::driver::DriverError`'s own `Debug` shape --
    /// `DriverError(CUresult::CUDA_ERROR_*, "<cuGetErrorString text>")` -- with
    /// the sentence half deliberately *wrong* or missing, to prove the match is
    /// on the enumerator name and not on prose that a driver update may reword.
    #[test]
    fn every_cuda_refusal_reason_is_reachable_and_distinct() {
        let cases = [
            ("the candle crate has not been built with cuda support", "not_built"),
            (
                "DriverError(CUresult::CUDA_ERROR_NOT_INITIALIZED, \"whatever\")",
                "no_driver",
            ),
            (
                "DriverError(CUresult::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, \"\")",
                "no_driver",
            ),
            ("DriverError(CUresult::CUDA_ERROR_NO_DEVICE, \"\")", "no_device"),
            (
                "DriverError(CUresult::CUDA_ERROR_INVALID_DEVICE, \"\")",
                "no_device",
            ),
            (
                "DriverError(CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU, \"\")",
                "wrong_arch",
            ),
            (
                "DriverError(CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION, \"\")",
                "wrong_arch",
            ),
            (
                "this device reports compute capability 6.1 (sm_61) and the kernels \
                 in this build were compiled for sm_80",
                "wrong_arch",
            ),
            ("DriverError(CUresult::CUDA_ERROR_UNKNOWN, \"\")", "unclassified"),
            ("", "unclassified"),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for (detail, expected) in cases {
            let got = classify_cuda_refusal(detail);
            assert_eq!(got, expected, "classifying {detail:?}");
            seen.insert(got);
        }
        // Every name in the published vocabulary was produced by one of the
        // cases above. A sixth reason added to the constant without a case here
        // fails, which is the only way this test can keep meaning what it says.
        let published: std::collections::BTreeSet<&str> =
            CUDA_REFUSAL_REASONS.iter().copied().collect();
        assert_eq!(seen, published, "some reason has no case in this test");
    }

    /// The two lists are one list.
    ///
    /// `CUDA_HOST_READBACK_OPS` is an alias of `MPS_HOST_READBACK_OPS` rather
    /// than a copy, and this asserts that from inside the crate so the
    /// Python-side test is checking a property that is *structural* here and
    /// not merely currently true.
    #[test]
    fn the_cuda_readback_list_is_the_mps_list() {
        assert_eq!(super::CUDA_HOST_READBACK_OPS, &super::MPS_HOST_READBACK_OPS);
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_function(wrap_pyfunction!(shim_same_device, m)?)?;
    m.add_function(wrap_pyfunction!(shim_mps_host_readback_ops, m)?)?;
    m.add_function(wrap_pyfunction!(shim_mps_readback_but_allowed, m)?)?;
    m.add_function(wrap_pyfunction!(shim_cuda_host_readback_ops, m)?)?;
    m.add_function(wrap_pyfunction!(shim_cuda_classify_refusal, m)?)?;
    m.add_function(wrap_pyfunction!(shim_cuda_refusal_reasons, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_probe, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_counters, m)?)?;
    Ok(())
}
